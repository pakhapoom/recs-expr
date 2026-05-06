import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]


def _extract_year(title: str) -> float | None:
    match = re.search(r"\((\d{4})\)\s*$", str(title).strip())
    return float(match.group(1)) if match else None


class YouTubeTwoTowerModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_ages: int,
        num_occs: int,
        num_zips: int,
        embed_dim: int,
        pad_item_idx: int,
    ):
        super().__init__()
        self.history_embed = nn.Embedding(num_items + 1, embed_dim, padding_idx=pad_item_idx)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.age_embed = nn.Embedding(num_ages, 8)
        self.occ_embed = nn.Embedding(num_occs, 12)
        self.zip_embed = nn.Embedding(num_zips, 12)
        self.item_feature = nn.Sequential(
            nn.Linear(len(GENRES) + 1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.user_tower = nn.Sequential(
            nn.Linear(embed_dim + 8 + 12 + 12 + 2, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.item_bias = nn.Embedding(num_items, 1)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_user(
        self,
        gender: torch.Tensor,
        age_idx: torch.Tensor,
        occ_idx: torch.Tensor,
        zip_idx: torch.Tensor,
        hist_items: torch.Tensor,
        hist_weights: torch.Tensor,
        example_age: torch.Tensor,
    ) -> torch.Tensor:
        hist_emb = self.history_embed(hist_items)
        weights = hist_weights.unsqueeze(-1)
        hist_vec = (hist_emb * weights).sum(1) / weights.sum(1).clamp(min=1e-6)
        x = torch.cat(
            [
                hist_vec,
                self.age_embed(age_idx),
                self.occ_embed(occ_idx),
                self.zip_embed(zip_idx),
                gender.float().unsqueeze(1),
                example_age.float().unsqueeze(1),
            ],
            dim=1,
        )
        return F.normalize(self.user_tower(x), dim=-1)

    def encode_item(
        self,
        item_idx: torch.Tensor,
        item_genres: torch.Tensor,
        item_years: torch.Tensor,
    ) -> torch.Tensor:
        item_features = torch.cat([item_genres, item_years.unsqueeze(1)], dim=1)
        x = torch.cat([self.item_embed(item_idx), self.item_feature(item_features)], dim=1)
        return F.normalize(self.item_tower(x), dim=-1)

    def score_pair(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=50.0)
        return (user_emb * item_emb).sum(-1) * scale + self.item_bias(item_idx).squeeze(-1)


class YouTubeTwoTowerRecommender:
    def __init__(
        self,
        n_recs: int = 100,
        max_history: int = 50,
        max_examples_per_user: int = 50,
        embed_dim: int = 256,
        epochs: int = 10,
        batch_size: int = 1024,
        n_neg: int = 128,
        lr: float = 2e-3,
        weight_decay: float = 1e-5,
        seed: int = 42,
        max_users: int | None = None,
        user_col: str = "user_id",
        item_col: str = "movie_id",
        score_col: str = "score",
    ):
        self.n_recs = n_recs
        self.max_history = max_history
        self.max_examples_per_user = max_examples_per_user
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_neg = n_neg
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.max_users = max_users
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

        if self.max_users is not None:
            keep = test[self.user_col].astype(str).drop_duplicates().head(self.max_users)
            train = train[train[self.user_col].isin(keep)].copy()
            val = val[val[self.user_col].isin(keep)].copy()
            test = test[test[self.user_col].isin(keep)].copy()
            users = users[users[self.user_col].isin(keep)].copy()

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

        gender = np.zeros(len(user_ids), dtype=np.float32)
        age = np.zeros(len(user_ids), dtype=np.int64)
        occ = np.zeros(len(user_ids), dtype=np.int64)
        zip_prefix = np.zeros(len(user_ids), dtype=np.int64)
        for _, row in users.iterrows():
            idx = user_id2idx[str(row[self.user_col])]
            gender[idx] = 1.0 if str(row["gender"]) == "F" else 0.0
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

    def _build_training_examples(self, interactions: pd.DataFrame, features: dict):
        pad_item_idx = len(features["movie_ids"])
        sorted_df = interactions.sort_values([self.user_col, "timestamp"]).reset_index(drop=True)
        max_ts = sorted_df["timestamp"].astype(np.int64).max()
        min_ts = sorted_df["timestamp"].astype(np.int64).min()
        ts_span_days = max((max_ts - min_ts) / 86400.0, 1.0)

        user_rows = []
        item_rows = []
        hist_item_rows = []
        hist_weight_rows = []
        example_age_rows = []
        rating_rows = []

        for user_id, group in sorted_df.groupby(self.user_col, sort=False):
            user_idx = features["user_id2idx"][str(user_id)]
            item_idxs = group[self.item_col].map(features["movie_id2idx"]).to_numpy(np.int64)
            ratings = group["rating"].astype(np.float32).to_numpy() / 5.0
            timestamps = group["timestamp"].astype(np.int64).to_numpy()
            positions = np.arange(1, len(group))

            if (
                self.max_examples_per_user > 0
                and len(positions) > self.max_examples_per_user
            ):
                positions = positions[-self.max_examples_per_user:]

            for pos in positions:
                h = min(pos, self.max_history)
                hist_items = np.full(self.max_history, pad_item_idx, dtype=np.int64)
                hist_weights = np.zeros(self.max_history, dtype=np.float32)
                hist_items[:h] = item_idxs[pos - h:pos]
                hist_weights[:h] = ratings[pos - h:pos]

                user_rows.append(user_idx)
                item_rows.append(item_idxs[pos])
                hist_item_rows.append(hist_items)
                hist_weight_rows.append(hist_weights)
                example_age_rows.append((max_ts - timestamps[pos]) / 86400.0 / ts_span_days)
                rating_rows.append(ratings[pos])

        if not user_rows:
            raise ValueError("No training examples were generated.")

        return {
            "user_idx": np.asarray(user_rows, dtype=np.int64),
            "item_idx": np.asarray(item_rows, dtype=np.int64),
            "hist_items": np.vstack(hist_item_rows),
            "hist_weights": np.vstack(hist_weight_rows),
            "example_age": np.asarray(example_age_rows, dtype=np.float32),
            "rating": np.asarray(rating_rows, dtype=np.float32),
        }

    def _build_inference_history(self, interactions: pd.DataFrame, features: dict):
        num_users = len(features["user_ids"])
        num_items = len(features["movie_ids"])
        pad_item_idx = num_items
        hist_items = np.full((num_users, self.max_history), pad_item_idx, dtype=np.int64)
        hist_weights = np.zeros((num_users, self.max_history), dtype=np.float32)
        watched = np.zeros((num_users, num_items), dtype=bool)
        sorted_df = interactions.sort_values([self.user_col, "timestamp"]).reset_index(drop=True)

        for user_id, group in sorted_df.groupby(self.user_col, sort=False):
            user_idx = features["user_id2idx"][str(user_id)]
            item_idxs = group[self.item_col].map(features["movie_id2idx"]).to_numpy(np.int64)
            ratings = group["rating"].astype(np.float32).to_numpy() / 5.0
            watched[user_idx, item_idxs] = True
            h = min(len(item_idxs), self.max_history)
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
        model: YouTubeTwoTowerModel,
        examples: dict,
        features: dict,
        watched: np.ndarray,
        device: str,
    ) -> YouTubeTwoTowerModel:
        rng = np.random.default_rng(self.seed)
        model = model.to(device)

        item_counts = np.bincount(
            examples["item_idx"],
            minlength=len(features["movie_ids"]),
        ).astype(np.float64)
        neg_probs = np.power(item_counts + 1.0, 0.75)
        neg_probs /= neg_probs.sum()

        with torch.no_grad():
            popularity_bias = np.log((item_counts + 1.0) / (item_counts.sum() + len(item_counts)))
            model.item_bias.weight.copy_(
                torch.tensor(popularity_bias, dtype=torch.float32).unsqueeze(1)
            )

        gender_t = torch.tensor(features["gender"], dtype=torch.float32, device=device)
        age_t = torch.tensor(features["age"], dtype=torch.long, device=device)
        occ_t = torch.tensor(features["occ"], dtype=torch.long, device=device)
        zip_t = torch.tensor(features["zip"], dtype=torch.long, device=device)
        item_genres_t = torch.tensor(features["item_genres"], dtype=torch.float32, device=device)
        item_years_t = torch.tensor(features["item_years"], dtype=torch.float32, device=device)
        neg_probs_t = torch.tensor(neg_probs, dtype=torch.float32, device=device)
        sample_correction_t = torch.log(neg_probs_t * self.n_neg + 1e-12)

        user_idx_t = torch.tensor(examples["user_idx"], dtype=torch.long, device=device)
        item_idx_t = torch.tensor(examples["item_idx"], dtype=torch.long, device=device)
        hist_items_t = torch.tensor(examples["hist_items"], dtype=torch.long, device=device)
        hist_weights_t = torch.tensor(examples["hist_weights"], dtype=torch.float32, device=device)
        example_age_t = torch.tensor(examples["example_age"], dtype=torch.float32, device=device)
        rating_t = torch.tensor(examples["rating"], dtype=torch.float32, device=device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        labels = torch.zeros(self.batch_size, dtype=torch.long, device=device)

        for epoch in range(self.epochs):
            model.train()
            order = rng.permutation(len(examples["user_idx"]))
            total_loss = 0.0
            total_count = 0

            for start in range(0, len(order), self.batch_size):
                batch_np = order[start:start + self.batch_size]
                batch = torch.tensor(batch_np, dtype=torch.long, device=device)
                users = user_idx_t[batch]
                pos_items = item_idx_t[batch]
                weights = rating_t[batch].clamp(min=0.2)

                neg_np = self._sample_negatives(
                    rng,
                    examples["user_idx"][batch_np],
                    examples["item_idx"][batch_np],
                    watched,
                    neg_probs,
                )
                neg_items = torch.tensor(neg_np, dtype=torch.long, device=device)

                user_emb = model.encode_user(
                    gender_t[users],
                    age_t[users],
                    occ_t[users],
                    zip_t[users],
                    hist_items_t[batch],
                    hist_weights_t[batch],
                    example_age_t[batch],
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

                pos_logits = model.score_pair(user_emb, pos_emb, pos_items)
                neg_logits = (
                    (user_emb.unsqueeze(1) * neg_emb).sum(-1)
                    * model.logit_scale.exp().clamp(max=50.0)
                    + model.item_bias(neg_items).squeeze(-1)
                )
                pos_logits = pos_logits - sample_correction_t[pos_items]
                neg_logits = neg_logits - sample_correction_t[neg_items]
                logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)

                batch_labels = labels[:len(users)]
                loss = F.cross_entropy(logits, batch_labels, reduction="none")
                loss = (loss * weights).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * len(users)
                total_count += len(users)

            print(f"Epoch {epoch + 1}/{self.epochs} loss={total_loss / total_count:.4f}")

        return model

    def _recommend(
        self,
        model: YouTubeTwoTowerModel,
        test: pd.DataFrame,
        features: dict,
        hist_items: np.ndarray,
        hist_weights: np.ndarray,
        watched: np.ndarray,
        device: str,
    ) -> pd.DataFrame:
        model.eval()
        test_users = test[self.user_col].astype(str).drop_duplicates().tolist()
        test_user_idx = np.array([features["user_id2idx"][user_id] for user_id in test_users])

        gender_t = torch.tensor(features["gender"], dtype=torch.float32, device=device)
        age_t = torch.tensor(features["age"], dtype=torch.long, device=device)
        occ_t = torch.tensor(features["occ"], dtype=torch.long, device=device)
        zip_t = torch.tensor(features["zip"], dtype=torch.long, device=device)
        hist_items_t = torch.tensor(hist_items, dtype=torch.long, device=device)
        hist_weights_t = torch.tensor(hist_weights, dtype=torch.float32, device=device)
        serving_age_t = torch.zeros(len(features["user_ids"]), dtype=torch.float32, device=device)
        item_idx_t = torch.arange(len(features["movie_ids"]), dtype=torch.long, device=device)
        item_genres_t = torch.tensor(features["item_genres"], dtype=torch.float32, device=device)
        item_years_t = torch.tensor(features["item_years"], dtype=torch.float32, device=device)

        rows = []
        with torch.no_grad():
            item_emb = model.encode_item(item_idx_t, item_genres_t, item_years_t).cpu().numpy()
            item_bias = model.item_bias.weight.squeeze(-1).detach().cpu().numpy()
            scale = float(model.logit_scale.exp().clamp(max=50.0).detach().cpu())

            for start in range(0, len(test_user_idx), 512):
                batch_idx = test_user_idx[start:start + 512]
                users = torch.tensor(batch_idx, dtype=torch.long, device=device)
                user_emb = model.encode_user(
                    gender_t[users],
                    age_t[users],
                    occ_t[users],
                    zip_t[users],
                    hist_items_t[users],
                    hist_weights_t[users],
                    serving_age_t[users],
                ).cpu().numpy()

                scores = user_emb @ item_emb.T * scale + item_bias[None, :]
                scores[watched[batch_idx]] = -np.inf

                for row_pos, user_idx in enumerate(batch_idx):
                    top_n = min(self.n_recs, int(np.isfinite(scores[row_pos]).sum()))
                    if top_n == 0:
                        continue
                    top = np.argpartition(scores[row_pos], -top_n)[-top_n:]
                    top = top[np.argsort(scores[row_pos, top])[::-1]]
                    user_id = features["user_ids"][user_idx]
                    for item in top:
                        rows.append({
                            self.user_col: user_id,
                            self.item_col: features["movie_ids"][item],
                            self.score_col: float(scores[row_pos, item]),
                        })

        return pd.DataFrame(rows)

    def run(
        self,
        train_path: str = "data/processed/train.csv",
        val_path: str = "data/processed/val.csv",
        test_path: str = "data/processed/test.csv",
        users_path: str = "data/raw/users.csv",
        movies_path: str = "data/raw/movies.csv",
        recs_path: str = "data/recs/two_tower_yt.csv",
        metrics_dir: str = "data/metrics",
    ):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

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
        examples = self._build_training_examples(fit, features)
        hist_items, hist_weights, watched = self._build_inference_history(fit, features)

        print(f"Training examples: {len(examples['user_idx']):,}")
        model = YouTubeTwoTowerModel(
            num_items=len(features["movie_ids"]),
            num_ages=features["num_ages"],
            num_occs=features["num_occs"],
            num_zips=features["num_zips"],
            embed_dim=self.embed_dim,
            pad_item_idx=len(features["movie_ids"]),
        )

        t_train_start = time.perf_counter()
        model = self._train(model, examples, features, watched, device)
        time_training_s = time.perf_counter() - t_train_start

        t_infer_start = time.perf_counter()
        recs = self._recommend(model, test, features, hist_items, hist_weights, watched, device)
        time_inference_s = time.perf_counter() - t_infer_start

        out = Path(recs_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        recs.to_csv(out, index=False)
        print(f"Saved {len(recs):,} rows -> {out}")

        experiment = out.stem
        timing = pd.DataFrame([
            {"metric": "time_training_s", "value": time_training_s},
            {"metric": "time_inference_s", "value": time_inference_s},
        ])
        metrics_out = Path(metrics_dir) / f"{experiment}.csv"
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        timing.to_csv(metrics_out, index=False)
        print(f"Timing -> {metrics_out}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/processed/train.csv")
    parser.add_argument("--val-path", default="data/processed/val.csv")
    parser.add_argument("--test-path", default="data/processed/test.csv")
    parser.add_argument("--users-path", default="data/raw/users.csv")
    parser.add_argument("--movies-path", default="data/raw/movies.csv")
    parser.add_argument("--recs-path", default="data/recs/two_tower_yt.csv")
    parser.add_argument("--metrics-dir", default="data/metrics")
    parser.add_argument("--n-recs", type=int, default=100)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--max-examples-per-user", type=int, default=50)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-neg", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    YouTubeTwoTowerRecommender(
        n_recs=args.n_recs,
        max_history=args.max_history,
        max_examples_per_user=args.max_examples_per_user,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_neg=args.n_neg,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_users=args.max_users,
    ).run(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        users_path=args.users_path,
        movies_path=args.movies_path,
        recs_path=args.recs_path,
        metrics_dir=args.metrics_dir,
    )
