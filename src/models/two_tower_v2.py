import re
import sys
import time
from pathlib import Path

project_root = next(
    p for p in Path(__file__).resolve().parents
    if (p / "src" / "models" / "candidate_sets.py").exists()
)
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.artifacts import cpu_state_dict, save_model_artifact
from src.models.candidate_sets import candidate_indices, load_candidate_map


GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]


def _extract_year(title: str) -> float | None:
    match = re.search(r"\((\d{4})\)\s*$", str(title).strip())
    return float(match.group(1)) if match else None


class TwoTowerV2Model(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_ages: int,
        num_occs: int,
        num_zips: int,
        num_genres: int,
        embed_dim: int,
        pad_item_idx: int,
    ):
        super().__init__()
        self.pad_item_idx = pad_item_idx
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items + 1, embed_dim, padding_idx=pad_item_idx)
        self.gender_embed = nn.Embedding(2, 4)
        self.age_embed = nn.Embedding(num_ages, 8)
        self.occ_embed = nn.Embedding(num_occs, 12)
        self.zip_embed = nn.Embedding(num_zips, 12)
        self.item_feature = nn.Sequential(
            nn.Linear(num_genres + 1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        user_in_dim = embed_dim * 2 + 4 + 8 + 12 + 12
        self.user_mlp = nn.Sequential(
            nn.Linear(user_in_dim, embed_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.item_bias = nn.Embedding(num_items, 1)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def encode_user(
        self,
        user_idx: torch.Tensor,
        gender_idx: torch.Tensor,
        age_idx: torch.Tensor,
        occ_idx: torch.Tensor,
        zip_idx: torch.Tensor,
        hist_items: torch.Tensor,
        hist_weights: torch.Tensor,
    ) -> torch.Tensor:
        hist_emb = self.item_embed(hist_items)
        weights = hist_weights.unsqueeze(-1)
        hist_vec = (hist_emb * weights).sum(1) / weights.sum(1).clamp(min=1e-6)
        x = torch.cat(
            [
                self.user_embed(user_idx),
                hist_vec,
                self.gender_embed(gender_idx),
                self.age_embed(age_idx),
                self.occ_embed(occ_idx),
                self.zip_embed(zip_idx),
            ],
            dim=1,
        )
        return F.normalize(self.user_mlp(x), dim=-1)

    def encode_item(
        self,
        item_idx: torch.Tensor,
        item_genres: torch.Tensor,
        item_years: torch.Tensor,
    ) -> torch.Tensor:
        item_features = torch.cat([item_genres, item_years.unsqueeze(1)], dim=1)
        x = torch.cat([self.item_embed(item_idx), self.item_feature(item_features)], dim=1)
        return F.normalize(self.item_mlp(x), dim=-1)

    def pair_score(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=50.0)
        return (user_emb * item_emb).sum(-1) * scale + self.item_bias(item_idx).squeeze(-1)


class TwoTowerV2Recommender:
    def __init__(
        self,
        n_recs: int = 100,
        max_history: int = 100,
        embed_dim: int = 96,
        epochs: int = 8,
        batch_size: int = 4096,
        n_neg: int = 8,
        lr: float = 2e-3,
        weight_decay: float = 1e-5,
        seed: int = 42,
        user_col: str = "user_id",
        item_col: str = "movie_id",
        score_col: str = "score",
    ):
        self.n_recs = n_recs
        self.max_history = max_history
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_neg = n_neg
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col

    def _device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_data(self, train_path, val_path, test_path, users_path, movies_path):
        dtype = {self.user_col: str, self.item_col: str}
        train = pd.read_csv(train_path, dtype=dtype)
        val = pd.read_csv(val_path, dtype=dtype)
        test = pd.read_csv(test_path, dtype=dtype)
        users = pd.read_csv(users_path, dtype={self.user_col: str})
        movies = pd.read_csv(movies_path, dtype={self.item_col: str})
        return train, val, test, users, movies

    def _build_features(self, users: pd.DataFrame, movies: pd.DataFrame):
        user_ids = users[self.user_col].astype(str).tolist()
        movie_ids = movies[self.item_col].astype(str).tolist()
        user_id2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        movie_id2idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

        ages = sorted(users["age"].astype(str).unique())
        occs = sorted(users["occupation"].astype(str).unique())
        zips = sorted(users["zip_code"].astype(str).str[:3].unique())
        age2idx = {age: idx for idx, age in enumerate(ages)}
        occ2idx = {occ: idx for idx, occ in enumerate(occs)}
        zip2idx = {zip_prefix: idx for idx, zip_prefix in enumerate(zips)}

        gender = np.zeros(len(user_ids), dtype=np.int64)
        age = np.zeros(len(user_ids), dtype=np.int64)
        occ = np.zeros(len(user_ids), dtype=np.int64)
        zip_prefix = np.zeros(len(user_ids), dtype=np.int64)
        for _, row in users.iterrows():
            idx = user_id2idx[str(row[self.user_col])]
            gender[idx] = 1 if str(row["gender"]) == "F" else 0
            age[idx] = age2idx[str(row["age"])]
            occ[idx] = occ2idx[str(row["occupation"])]
            zip_prefix[idx] = zip2idx[str(row["zip_code"])[:3]]

        years = movies["title"].apply(_extract_year).dropna()
        year_mean = float(years.mean())
        year_std = float(max(years.std(), 1.0))
        item_genres = np.zeros((len(movie_ids), len(GENRES)), dtype=np.float32)
        item_years = np.zeros(len(movie_ids), dtype=np.float32)
        for _, row in movies.iterrows():
            idx = movie_id2idx[str(row[self.item_col])]
            for genre in str(row["genres"]).split("|"):
                if genre in GENRES:
                    item_genres[idx, GENRES.index(genre)] = 1.0
            year = _extract_year(row["title"])
            if year is not None:
                item_years[idx] = (year - year_mean) / year_std

        return {
            "user_ids": user_ids,
            "movie_ids": movie_ids,
            "user_id2idx": user_id2idx,
            "movie_id2idx": movie_id2idx,
            "age_vocab": age2idx,
            "occ_vocab": occ2idx,
            "zip_vocab": zip2idx,
            "gender": gender,
            "age": age,
            "occ": occ,
            "zip": zip_prefix,
            "num_ages": len(ages),
            "num_occs": len(occs),
            "num_zips": len(zips),
            "item_genres": item_genres,
            "item_years": item_years,
        }

    def _map_interactions(self, interactions: pd.DataFrame, features: dict):
        user_idx = interactions[self.user_col].map(features["user_id2idx"]).to_numpy(np.int64)
        item_idx = interactions[self.item_col].map(features["movie_id2idx"]).to_numpy(np.int64)
        ratings = interactions["rating"].astype(np.float32).to_numpy() / 5.0
        return user_idx, item_idx, ratings

    def _build_history(self, interactions: pd.DataFrame, features: dict):
        num_users = len(features["user_ids"])
        pad_item_idx = len(features["movie_ids"])
        hist_items = np.full((num_users, self.max_history), pad_item_idx, dtype=np.int64)
        hist_weights = np.zeros((num_users, self.max_history), dtype=np.float32)
        watched = np.zeros((num_users, len(features["movie_ids"])), dtype=bool)

        for user_id, group in interactions.groupby(self.user_col, sort=False):
            user_idx = features["user_id2idx"][str(user_id)]
            item_idxs = [
                features["movie_id2idx"][str(movie_id)]
                for movie_id in group[self.item_col]
            ]
            ratings = group["rating"].astype(np.float32).to_numpy() / 5.0
            watched[user_idx, item_idxs] = True
            h = min(len(item_idxs), self.max_history)
            if h:
                hist_items[user_idx, :h] = item_idxs[-h:]
                hist_weights[user_idx, :h] = ratings[-h:]

        return hist_items, hist_weights, watched

    def _sample_negatives(
        self,
        rng: np.random.Generator,
        user_idx: np.ndarray,
        pos_item_idx: np.ndarray,
        watched: np.ndarray,
        probs: np.ndarray,
    ) -> np.ndarray:
        num_items = watched.shape[1]
        neg = rng.choice(num_items, size=(len(user_idx), self.n_neg), p=probs)
        bad = watched[user_idx[:, None], neg] | (neg == pos_item_idx[:, None])
        while bad.any():
            neg[bad] = rng.choice(num_items, size=int(bad.sum()), p=probs)
            bad = watched[user_idx[:, None], neg] | (neg == pos_item_idx[:, None])
        return neg.astype(np.int64)

    def _train(
        self,
        model: TwoTowerV2Model,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        ratings: np.ndarray,
        features: dict,
        hist_items: np.ndarray,
        hist_weights: np.ndarray,
        watched: np.ndarray,
        device: str,
    ) -> TwoTowerV2Model:
        rng = np.random.default_rng(self.seed)
        model = model.to(device)

        item_counts = np.bincount(item_idx, minlength=len(features["movie_ids"])).astype(np.float64)
        neg_probs = np.power(item_counts + 1.0, 0.75)
        neg_probs /= neg_probs.sum()
        with torch.no_grad():
            popularity_bias = np.log((item_counts + 1.0) / (item_counts.sum() + len(item_counts)))
            model.item_bias.weight.copy_(
                torch.tensor(popularity_bias, dtype=torch.float32).unsqueeze(1)
            )

        gender_t = torch.tensor(features["gender"], dtype=torch.long, device=device)
        age_t = torch.tensor(features["age"], dtype=torch.long, device=device)
        occ_t = torch.tensor(features["occ"], dtype=torch.long, device=device)
        zip_t = torch.tensor(features["zip"], dtype=torch.long, device=device)
        hist_items_t = torch.tensor(hist_items, dtype=torch.long, device=device)
        hist_weights_t = torch.tensor(hist_weights, dtype=torch.float32, device=device)
        item_genres_t = torch.tensor(features["item_genres"], dtype=torch.float32, device=device)
        item_years_t = torch.tensor(features["item_years"], dtype=torch.float32, device=device)
        user_idx_t = torch.tensor(user_idx, dtype=torch.long, device=device)
        item_idx_t = torch.tensor(item_idx, dtype=torch.long, device=device)
        ratings_t = torch.tensor(ratings, dtype=torch.float32, device=device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        for epoch in range(self.epochs):
            model.train()
            order = rng.permutation(len(user_idx))
            total_loss = 0.0
            total_count = 0

            for start in range(0, len(order), self.batch_size):
                batch_np = order[start:start + self.batch_size]
                batch = torch.tensor(batch_np, dtype=torch.long, device=device)
                users = user_idx_t[batch]
                pos_items = item_idx_t[batch]
                weights = ratings_t[batch].clamp(min=0.2)

                neg_np = self._sample_negatives(
                    rng,
                    user_idx[batch_np],
                    item_idx[batch_np],
                    watched,
                    neg_probs,
                )
                neg_items = torch.tensor(neg_np, dtype=torch.long, device=device)

                user_emb = model.encode_user(
                    users,
                    gender_t[users],
                    age_t[users],
                    occ_t[users],
                    zip_t[users],
                    hist_items_t[users],
                    hist_weights_t[users],
                )
                pos_emb = model.encode_item(
                    pos_items,
                    item_genres_t[pos_items],
                    item_years_t[pos_items],
                )

                flat_neg = neg_items.reshape(-1)
                neg_emb = model.encode_item(
                    flat_neg,
                    item_genres_t[flat_neg],
                    item_years_t[flat_neg],
                ).view(len(users), self.n_neg, -1)

                pos_scores = model.pair_score(user_emb, pos_emb, pos_items)
                neg_scores = (
                    (user_emb.unsqueeze(1) * neg_emb).sum(-1)
                    * model.logit_scale.exp().clamp(max=50.0)
                    + model.item_bias(neg_items).squeeze(-1)
                )
                loss = (-F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean(1) * weights).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * len(users)
                total_count += len(users)

            print(f"Epoch {epoch + 1}/{self.epochs} loss={total_loss / total_count:.4f}")

        return model

    def _recommend(
        self,
        model: TwoTowerV2Model,
        test: pd.DataFrame,
        features: dict,
        hist_items: np.ndarray,
        hist_weights: np.ndarray,
        watched: np.ndarray,
        candidate_map: dict[str, list[str]] | None,
        device: str,
    ) -> pd.DataFrame:
        model.eval()
        test_users = test[self.user_col].astype(str).unique().tolist()
        test_user_idx = np.array([features["user_id2idx"][user_id] for user_id in test_users])
        candidate_idx_by_user = candidate_indices(candidate_map, features["movie_id2idx"])

        gender_t = torch.tensor(features["gender"], dtype=torch.long, device=device)
        age_t = torch.tensor(features["age"], dtype=torch.long, device=device)
        occ_t = torch.tensor(features["occ"], dtype=torch.long, device=device)
        zip_t = torch.tensor(features["zip"], dtype=torch.long, device=device)
        hist_items_t = torch.tensor(hist_items, dtype=torch.long, device=device)
        hist_weights_t = torch.tensor(hist_weights, dtype=torch.float32, device=device)
        item_idx_t = torch.arange(len(features["movie_ids"]), dtype=torch.long, device=device)
        item_genres_t = torch.tensor(features["item_genres"], dtype=torch.float32, device=device)
        item_years_t = torch.tensor(features["item_years"], dtype=torch.float32, device=device)

        rows = []
        with torch.no_grad():
            item_emb = model.encode_item(item_idx_t, item_genres_t, item_years_t)
            item_emb_np = item_emb.cpu().numpy()
            item_bias = model.item_bias.weight.squeeze(-1).detach().cpu().numpy()
            scale = float(model.logit_scale.exp().clamp(max=50.0).detach().cpu())

            for start in range(0, len(test_user_idx), 512):
                batch_idx = test_user_idx[start:start + 512]
                users = torch.tensor(batch_idx, dtype=torch.long, device=device)
                user_emb = model.encode_user(
                    users,
                    gender_t[users],
                    age_t[users],
                    occ_t[users],
                    zip_t[users],
                    hist_items_t[users],
                    hist_weights_t[users],
                ).cpu().numpy()

                for row_pos, user_idx in enumerate(batch_idx):
                    user_id = features["user_ids"][user_idx]
                    if candidate_idx_by_user is None:
                        item_candidates = np.flatnonzero(~watched[user_idx])
                    else:
                        item_candidates = candidate_idx_by_user.get(
                            user_id,
                            np.asarray([], dtype=np.int64),
                        )
                        item_candidates = item_candidates[~watched[user_idx, item_candidates]]

                    top_n = min(self.n_recs, len(item_candidates))
                    if top_n == 0:
                        continue

                    scores = (
                        user_emb[row_pos] @ item_emb_np[item_candidates].T * scale
                        + item_bias[item_candidates]
                    )
                    top = np.argpartition(scores, -top_n)[-top_n:]
                    top = top[np.argsort(scores[top])[::-1]]
                    for item in top:
                        item_idx = item_candidates[item]
                        rows.append({
                            self.user_col: user_id,
                            self.item_col: features["movie_ids"][item_idx],
                            self.score_col: float(scores[item]),
                        })

        return pd.DataFrame(rows)

    def run(
        self,
        train_path: str = "data/processed/train.csv",
        val_path: str = "data/processed/val.csv",
        test_path: str = "data/processed/test.csv",
        users_path: str = "data/raw/users.csv",
        movies_path: str = "data/raw/movies.csv",
        candidates_path: str = "data/processed/inference_candidates.csv",
        recs_path: str = "data/recs/two_tower_v2.csv",
        metrics_dir: str = "data/metrics",
        model_dir: str = "data/models",
    ):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self._device()
        print(f"Device: {device}")

        train, val, test, users, movies = self._load_data(
            train_path,
            val_path,
            test_path,
            users_path,
            movies_path,
        )
        fit = pd.concat([train, val], ignore_index=True)
        features = self._build_features(users, movies)
        user_idx, item_idx, ratings = self._map_interactions(fit, features)
        hist_items, hist_weights, watched = self._build_history(fit, features)
        test_users = test[self.user_col].astype(str).unique().tolist()
        candidate_map = load_candidate_map(
            candidates_path,
            self.user_col,
            self.item_col,
            test_users,
        )

        model = TwoTowerV2Model(
            num_users=len(features["user_ids"]),
            num_items=len(features["movie_ids"]),
            num_ages=features["num_ages"],
            num_occs=features["num_occs"],
            num_zips=features["num_zips"],
            num_genres=len(GENRES),
            embed_dim=self.embed_dim,
            pad_item_idx=len(features["movie_ids"]),
        )

        t_train_start = time.perf_counter()
        model = self._train(
            model,
            user_idx,
            item_idx,
            ratings,
            features,
            hist_items,
            hist_weights,
            watched,
            device,
        )
        time_training_s = time.perf_counter() - t_train_start
        experiment = Path(recs_path).stem
        save_model_artifact(
            model_dir,
            experiment,
            {
                "artifact_version": 1,
                "model": "TwoTowerV2Recommender",
                "model_state_dict": cpu_state_dict(model),
                "model_kwargs": {
                    "num_users": len(features["user_ids"]),
                    "num_items": len(features["movie_ids"]),
                    "num_ages": features["num_ages"],
                    "num_occs": features["num_occs"],
                    "num_zips": features["num_zips"],
                    "num_genres": len(GENRES),
                    "embed_dim": self.embed_dim,
                    "pad_item_idx": len(features["movie_ids"]),
                },
                "features": features,
                "config": {
                    "n_recs": self.n_recs,
                    "max_history": self.max_history,
                    "embed_dim": self.embed_dim,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "n_neg": self.n_neg,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "seed": self.seed,
                },
                "columns": {
                    "user_col": self.user_col,
                    "item_col": self.item_col,
                    "score_col": self.score_col,
                },
            },
        )

        t_infer_start = time.perf_counter()
        recs = self._recommend(
            model,
            test,
            features,
            hist_items,
            hist_weights,
            watched,
            candidate_map,
            device,
        )
        time_inference_s = time.perf_counter() - t_infer_start

        out = Path(recs_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        recs.to_csv(out, index=False)
        print(f"Saved {len(recs):,} rows -> {out}")

        timing = pd.DataFrame([
            {"metric": "time_training_s", "value": time_training_s},
            {"metric": "time_inference_s", "value": time_inference_s},
        ])
        metrics_out = Path(metrics_dir) / f"{experiment}.csv"
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        timing.to_csv(metrics_out, index=False)
        print(f"Timing -> {metrics_out}")


if __name__ == "__main__":
    TwoTowerV2Recommender().run()
