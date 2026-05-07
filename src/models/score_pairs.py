import argparse
import sys
from pathlib import Path

project_root = next(
    p for p in Path(__file__).resolve().parents
    if (p / "src" / "models" / "candidate_sets.py").exists()
)
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch

from src.models.artifacts import load_pair_scoring_artifact
from src.models.two_tower.model import RankingModel, TwoTowerModel
from src.models.two_tower_v2 import TwoTowerV2Model
from src.models.two_tower_yt import YouTubeTwoTowerModel

USER_COL = "user_id"
ITEM_COL = "movie_id"
SCORE_COL = "score"
BATCH_SIZE = 1024


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def require_columns(df: pd.DataFrame, path: str, columns: set[str]):
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def read_pairs(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Pairs file not found: {path}")
    pairs = pd.read_csv(path, dtype={USER_COL: str, ITEM_COL: str})
    require_columns(pairs, path, {USER_COL, ITEM_COL})
    return pairs


def read_interactions(train_path: str, val_path: str) -> pd.DataFrame:
    frames = []
    for path in (train_path, val_path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Interaction file not found: {path}")
        df = pd.read_csv(path, dtype={USER_COL: str, ITEM_COL: str})
        require_columns(df, path, {USER_COL, ITEM_COL, "rating"})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def read_users(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Users file not found: {path}")
    users = pd.read_csv(path, dtype={USER_COL: str})
    require_columns(users, path, {USER_COL, "gender", "age", "occupation", "zip_code"})
    return users


def validate_movies(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"Movies file not found: {path}")
    cols = pd.read_csv(path, nrows=0)
    require_columns(cols, path, {ITEM_COL})


def as_str_key_dict(mapping: dict) -> dict[str, int]:
    return {str(key): int(value) for key, value in mapping.items()}


def refresh_user_features(features: dict, users: pd.DataFrame, gender_dtype) -> dict:
    if not {"age_vocab", "occ_vocab", "zip_vocab"} <= set(features):
        return features

    out = dict(features)
    user_id2idx = as_str_key_dict(features["user_id2idx"])
    age_vocab = {str(key): int(value) for key, value in features["age_vocab"].items()}
    occ_vocab = {str(key): int(value) for key, value in features["occ_vocab"].items()}
    zip_vocab = {str(key): int(value) for key, value in features["zip_vocab"].items()}

    gender = np.asarray(features["gender"], dtype=gender_dtype).copy()
    age = np.asarray(features["age"], dtype=np.int64).copy()
    occ = np.asarray(features["occ"], dtype=np.int64).copy()
    zip_prefix = np.asarray(features["zip"], dtype=np.int64).copy()

    for _, row in users.iterrows():
        user_idx = user_id2idx.get(str(row[USER_COL]))
        if user_idx is None:
            continue
        gender[user_idx] = 1 if str(row["gender"]) == "F" else 0
        age[user_idx] = age_vocab.get(str(row["age"]), 0)
        occ[user_idx] = occ_vocab.get(str(row["occupation"]), 0)
        zip_prefix[user_idx] = zip_vocab.get(str(row["zip_code"])[:3], 0)

    out["user_id2idx"] = user_id2idx
    out["movie_id2idx"] = as_str_key_dict(features["movie_id2idx"])
    out["gender"] = gender
    out["age"] = age
    out["occ"] = occ
    out["zip"] = zip_prefix
    return out


def build_indexed_history(
    interactions: pd.DataFrame,
    features: dict,
    max_history: int,
    sort_by_time: bool,
) -> tuple[np.ndarray, np.ndarray]:
    num_users = len(features["user_ids"])
    pad_item_idx = len(features["movie_ids"])
    hist_items = np.full((num_users, max_history), pad_item_idx, dtype=np.int64)
    hist_weights = np.zeros((num_users, max_history), dtype=np.float32)
    user_id2idx = as_str_key_dict(features["user_id2idx"])
    movie_id2idx = as_str_key_dict(features["movie_id2idx"])

    if sort_by_time and "timestamp" in interactions.columns:
        interactions = interactions.sort_values([USER_COL, "timestamp"]).reset_index(drop=True)

    for user_id, group in interactions.groupby(USER_COL, sort=False):
        user_idx = user_id2idx.get(str(user_id))
        if user_idx is None:
            continue

        item_idxs = [movie_id2idx.get(str(movie_id)) for movie_id in group[ITEM_COL]]
        ratings = group["rating"].astype(np.float32).to_numpy() / 5.0
        valid = [(idx, rating) for idx, rating in zip(item_idxs, ratings) if idx is not None]
        if not valid:
            continue

        h = min(len(valid), max_history)
        hist_items[user_idx, :h] = [idx for idx, _ in valid[-h:]]
        hist_weights[user_idx, :h] = [rating for _, rating in valid[-h:]]

    return hist_items, hist_weights


def build_two_tower_user_features(feature_store: dict, users: pd.DataFrame) -> dict[str, dict]:
    age_vocab = {str(key): int(value) for key, value in feature_store["age_vocab"].items()}
    occ_vocab = {str(key): int(value) for key, value in feature_store["occ_vocab"].items()}
    zip_vocab = {str(key): int(value) for key, value in feature_store["zip_vocab"].items()}
    gender_map = {"M": np.float32(0.0), "F": np.float32(1.0)}

    out = {}
    for _, row in users.iterrows():
        user_id = str(row[USER_COL])
        out[user_id] = {
            "gender": gender_map.get(str(row["gender"]), np.float32(0.0)),
            "age_idx": age_vocab.get(str(row["age"]), 0),
            "occ_idx": occ_vocab.get(str(row["occupation"]), 0),
            "zip_idx": zip_vocab.get(str(row["zip_code"])[:3], 0),
        }
    return out


def build_two_tower_history(interactions: pd.DataFrame, feature_store: dict) -> dict[str, tuple]:
    max_history = int(feature_store["max_history"])
    movie_id2idx = as_str_key_dict(feature_store["movie_id2idx"])
    out = {}

    for user_id, group in interactions.groupby(USER_COL, sort=False):
        item_idxs = [movie_id2idx.get(str(movie_id)) for movie_id in group[ITEM_COL]]
        ratings = ((group["rating"].astype(np.float32).to_numpy() - 1.0) / 4.0).tolist()
        valid = [(idx, rating) for idx, rating in zip(item_idxs, ratings) if idx is not None]
        h = min(len(valid), max_history)

        pad_ids = np.zeros(max_history, dtype=np.int64)
        pad_ratings = np.zeros(max_history, dtype=np.float32)
        if h:
            pad_ids[:h] = [idx for idx, _ in valid[-h:]]
            pad_ratings[:h] = [rating for _, rating in valid[-h:]]
        out[str(user_id)] = (pad_ids, pad_ratings, h)

    return out


class TwoTowerPairScorer:
    def __init__(
        self,
        artifact: dict,
        interactions: pd.DataFrame,
        users: pd.DataFrame,
        device: str,
    ):
        self.device = device
        self.feature_store = artifact["feature_store"]
        self.movie_ids = [str(movie_id) for movie_id in self.feature_store["movie_ids"]]
        self.movie_id2idx = as_str_key_dict(self.feature_store["movie_id2idx"])
        self.user_features = build_two_tower_user_features(self.feature_store, users)
        self.histories = build_two_tower_history(interactions, self.feature_store)
        self.max_history = int(self.feature_store["max_history"])
        self.zero_history = (
            np.zeros(self.max_history, dtype=np.int64),
            np.zeros(self.max_history, dtype=np.float32),
            0,
        )

        self.retrieval_model = TwoTowerModel(**artifact["retrieval_model_kwargs"]).to(device)
        self.ranker = RankingModel(**artifact["ranking_model_kwargs"]).to(device)
        self.retrieval_model.load_state_dict(artifact["retrieval_model_state_dict"])
        self.ranker.load_state_dict(artifact["ranking_model_state_dict"])
        self.retrieval_model.eval()
        self.ranker.eval()

        self.item_genres_t = torch.tensor(
            np.asarray(self.feature_store["item_genres"]),
            dtype=torch.float32,
            device=device,
        )
        self.item_year_t = torch.tensor(
            np.asarray(self.feature_store["item_year"]),
            dtype=torch.float32,
            device=device,
        )

    def score(self, pairs: pd.DataFrame) -> np.ndarray:
        scores = np.full(len(pairs), np.nan, dtype=np.float64)
        valid = []
        user_ids = pairs[USER_COL].astype(str).to_numpy()
        movie_ids = pairs[ITEM_COL].astype(str).to_numpy()

        for row_idx, (user_id, movie_id) in enumerate(zip(user_ids, movie_ids)):
            item_idx = self.movie_id2idx.get(movie_id)
            if user_id not in self.user_features or item_idx is None:
                continue
            valid.append((row_idx, user_id, item_idx))

        with torch.no_grad():
            for start in range(0, len(valid), BATCH_SIZE):
                batch = valid[start:start + BATCH_SIZE]
                batch_users = [row[1] for row in batch]
                item_idx_np = np.asarray([row[2] for row in batch], dtype=np.int64)
                histories = [self.histories.get(user_id, self.zero_history) for user_id in batch_users]

                gender = torch.tensor(
                    [self.user_features[user_id]["gender"] for user_id in batch_users],
                    dtype=torch.float32,
                    device=self.device,
                )
                age = torch.tensor(
                    [self.user_features[user_id]["age_idx"] for user_id in batch_users],
                    dtype=torch.long,
                    device=self.device,
                )
                occ = torch.tensor(
                    [self.user_features[user_id]["occ_idx"] for user_id in batch_users],
                    dtype=torch.long,
                    device=self.device,
                )
                zip_prefix = torch.tensor(
                    [self.user_features[user_id]["zip_idx"] for user_id in batch_users],
                    dtype=torch.long,
                    device=self.device,
                )
                hist_ids = torch.tensor(
                    np.stack([history[0] for history in histories]),
                    dtype=torch.long,
                    device=self.device,
                )
                hist_ratings = torch.tensor(
                    np.stack([history[1] for history in histories]),
                    dtype=torch.float32,
                    device=self.device,
                )
                hist_lens = torch.tensor(
                    [history[2] for history in histories],
                    dtype=torch.long,
                    device=self.device,
                )
                item_idx = torch.tensor(item_idx_np, dtype=torch.long, device=self.device)

                user_emb = self.retrieval_model.encode_user(
                    gender,
                    age,
                    occ,
                    zip_prefix,
                    hist_ids,
                    hist_ratings,
                    hist_lens,
                )
                item_emb = self.retrieval_model.encode_item(
                    item_idx,
                    self.item_genres_t[item_idx],
                    self.item_year_t[item_idx],
                )
                batch_scores = self.ranker(user_emb, item_emb).detach().cpu().numpy()
                for (row_idx, _, _), score in zip(batch, batch_scores):
                    scores[row_idx] = float(score)

        return scores


class IndexedTwoTowerPairScorer:
    def __init__(
        self,
        artifact: dict,
        interactions: pd.DataFrame,
        users: pd.DataFrame,
        device: str,
    ):
        self.device = device
        self.model_name = artifact["model"]
        self.raw_user_ids = set(users[USER_COL].astype(str))

        if self.model_name == "TwoTowerV2Recommender":
            self.features = refresh_user_features(artifact["features"], users, np.int64)
            self.model = TwoTowerV2Model(**artifact["model_kwargs"]).to(device)
            self.sort_history_by_time = False
            self.gender_dtype = torch.long
            self.uses_example_age = False
        else:
            self.features = refresh_user_features(artifact["features"], users, np.float32)
            self.model = YouTubeTwoTowerModel(**artifact["model_kwargs"]).to(device)
            self.sort_history_by_time = True
            self.gender_dtype = torch.float32
            self.uses_example_age = True

        self.model.load_state_dict(artifact["model_state_dict"])
        self.model.eval()
        self.user_id2idx = as_str_key_dict(self.features["user_id2idx"])
        self.movie_id2idx = as_str_key_dict(self.features["movie_id2idx"])
        self.hist_items, self.hist_weights = build_indexed_history(
            interactions,
            self.features,
            int(artifact["config"]["max_history"]),
            self.sort_history_by_time,
        )

        self.gender_t = torch.tensor(
            np.asarray(self.features["gender"]),
            dtype=self.gender_dtype,
            device=device,
        )
        self.age_t = torch.tensor(self.features["age"], dtype=torch.long, device=device)
        self.occ_t = torch.tensor(self.features["occ"], dtype=torch.long, device=device)
        self.zip_t = torch.tensor(self.features["zip"], dtype=torch.long, device=device)
        self.hist_items_t = torch.tensor(self.hist_items, dtype=torch.long, device=device)
        self.hist_weights_t = torch.tensor(self.hist_weights, dtype=torch.float32, device=device)
        self.item_genres_t = torch.tensor(
            self.features["item_genres"],
            dtype=torch.float32,
            device=device,
        )
        self.item_years_t = torch.tensor(
            self.features["item_years"],
            dtype=torch.float32,
            device=device,
        )

    def score(self, pairs: pd.DataFrame) -> np.ndarray:
        scores = np.full(len(pairs), np.nan, dtype=np.float64)
        valid = []
        user_ids = pairs[USER_COL].astype(str).to_numpy()
        movie_ids = pairs[ITEM_COL].astype(str).to_numpy()

        for row_idx, (user_id, movie_id) in enumerate(zip(user_ids, movie_ids)):
            user_idx = self.user_id2idx.get(user_id)
            item_idx = self.movie_id2idx.get(movie_id)
            if user_id not in self.raw_user_ids or user_idx is None or item_idx is None:
                continue
            valid.append((row_idx, user_idx, item_idx))

        with torch.no_grad():
            for start in range(0, len(valid), BATCH_SIZE):
                batch = valid[start:start + BATCH_SIZE]
                user_idx = torch.tensor(
                    [row[1] for row in batch],
                    dtype=torch.long,
                    device=self.device,
                )
                item_idx = torch.tensor(
                    [row[2] for row in batch],
                    dtype=torch.long,
                    device=self.device,
                )

                if self.uses_example_age:
                    user_emb = self.model.encode_user(
                        self.gender_t[user_idx],
                        self.age_t[user_idx],
                        self.occ_t[user_idx],
                        self.zip_t[user_idx],
                        self.hist_items_t[user_idx],
                        self.hist_weights_t[user_idx],
                        torch.zeros(len(batch), dtype=torch.float32, device=self.device),
                    )
                    item_emb = self.model.encode_item(
                        item_idx,
                        self.item_genres_t[item_idx],
                        self.item_years_t[item_idx],
                    )
                    batch_scores = self.model.score_pair(user_emb, item_emb, item_idx)
                else:
                    user_emb = self.model.encode_user(
                        user_idx,
                        self.gender_t[user_idx],
                        self.age_t[user_idx],
                        self.occ_t[user_idx],
                        self.zip_t[user_idx],
                        self.hist_items_t[user_idx],
                        self.hist_weights_t[user_idx],
                    )
                    item_emb = self.model.encode_item(
                        item_idx,
                        self.item_genres_t[item_idx],
                        self.item_years_t[item_idx],
                    )
                    batch_scores = self.model.pair_score(user_emb, item_emb, item_idx)

                for (row_idx, _, _), score in zip(batch, batch_scores.detach().cpu().numpy()):
                    scores[row_idx] = float(score)

        return scores


def build_scorer(
    artifact: dict,
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    device: str,
):
    if artifact["model"] == "TwoTowerRecommender":
        return TwoTowerPairScorer(artifact, interactions, users, device)
    return IndexedTwoTowerPairScorer(artifact, interactions, users, device)


def run(
    model_path: str,
    pairs_path: str,
    output_path: str,
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/val.csv",
    users_path: str = "data/raw/users.csv",
    movies_path: str = "data/raw/movies.csv",
    device: str = "auto",
):
    resolved_device = resolve_device(device)
    pairs = read_pairs(pairs_path)
    interactions = read_interactions(train_path, val_path)
    users = read_users(users_path)
    validate_movies(movies_path)
    artifact = load_pair_scoring_artifact(model_path, map_location=resolved_device)
    scorer = build_scorer(artifact, interactions, users, resolved_device)

    out = pairs.copy()
    out[SCORE_COL] = scorer.score(out)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, na_rep="NaN")

    nan_count = int(out[SCORE_COL].isna().sum())
    scored_count = len(out) - nan_count
    print(f"Scored {scored_count:,} rows; {nan_count:,} rows -> NaN")
    print(f"Saved {len(out):,} rows -> {output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--pairs-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--train-path", default="data/processed/train.csv")
    parser.add_argument("--val-path", default="data/processed/val.csv")
    parser.add_argument("--users-path", default="data/raw/users.csv")
    parser.add_argument("--movies-path", default="data/raw/movies.csv")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(
            model_path=args.model_path,
            pairs_path=args.pairs_path,
            output_path=args.output_path,
            train_path=args.train_path,
            val_path=args.val_path,
            users_path=args.users_path,
            movies_path=args.movies_path,
            device=args.device,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
