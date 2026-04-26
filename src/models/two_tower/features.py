import re

import numpy as np
import pandas as pd

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]


def _extract_year(title: str) -> float | None:
    m = re.search(r'\((\d{4})\)\s*$', str(title).strip())
    return float(m.group(1)) if m else None


class FeatureStore:
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.movie_ids: list[str] = []
        self.movie_id2idx: dict[str, int] = {}
        self.age_vocab: dict[str, int] = {}
        self.occ_vocab: dict[str, int] = {}
        self.zip_vocab: dict[str, int] = {}
        self.year_mean: float = 0.0
        self.year_std: float = 1.0
        self.item_genres: np.ndarray | None = None
        self.item_year: np.ndarray | None = None

    def fit(self, movies_df: pd.DataFrame, users_df: pd.DataFrame) -> "FeatureStore":
        self.movie_ids = movies_df["movie_id"].astype(str).tolist()
        self.movie_id2idx = {mid: i for i, mid in enumerate(self.movie_ids)}

        self.age_vocab = {a: i for i, a in enumerate(sorted(users_df["age"].astype(str).unique()))}
        self.occ_vocab = {o: i for i, o in enumerate(sorted(users_df["occupation"].astype(str).unique()))}
        zip_prefixes = sorted(users_df["zip_code"].astype(str).str[:3].unique())
        self.zip_vocab = {z: i for i, z in enumerate(zip_prefixes)}

        years = movies_df["title"].apply(_extract_year).dropna()
        self.year_mean = float(years.mean())
        self.year_std = float(max(years.std(), 1.0))

        n = len(self.movie_ids)
        self.item_genres = np.zeros((n, len(GENRES)), dtype=np.float32)
        self.item_year = np.zeros(n, dtype=np.float32)

        for _, row in movies_df.iterrows():
            idx = self.movie_id2idx[str(row["movie_id"])]
            for g in str(row["genres"]).split("|"):
                if g in GENRES:
                    self.item_genres[idx, GENRES.index(g)] = 1.0
            yr = _extract_year(row["title"])
            if yr is not None:
                self.item_year[idx] = (yr - self.year_mean) / self.year_std

        return self

    def encode_users(self, users_df: pd.DataFrame) -> dict[str, dict]:
        gender_map = {"M": np.float32(0.0), "F": np.float32(1.0)}
        out = {}
        for _, row in users_df.iterrows():
            uid = str(row["user_id"])
            out[uid] = {
                "gender": gender_map.get(str(row["gender"]), np.float32(0.0)),
                "age_idx": self.age_vocab.get(str(row["age"]), 0),
                "occ_idx": self.occ_vocab.get(str(row["occupation"]), 0),
                "zip_idx": self.zip_vocab.get(str(row["zip_code"])[:3], 0),
            }
        return out

    def encode_histories(self, interactions_df: pd.DataFrame) -> dict[str, tuple]:
        out = {}
        for uid, group in interactions_df.groupby("user_id"):
            mid_list = [self.movie_id2idx.get(str(mid), 0) for mid in group["movie_id"]]
            rat_list = [(float(r) - 1.0) / 4.0 for r in group["rating"]]

            h = min(len(mid_list), self.max_history)
            mid_list = mid_list[-h:]
            rat_list = rat_list[-h:]

            pad_ids = np.zeros(self.max_history, dtype=np.int64)
            pad_rats = np.zeros(self.max_history, dtype=np.float32)
            pad_ids[:h] = mid_list
            pad_rats[:h] = rat_list
            out[str(uid)] = (pad_ids, pad_rats, h)
        return out
