import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


class CandidateSampler:
    def __init__(
        self,
        n_candidates: int = 100,
        seed: int = 42,
        user_col: str = "user_id",
        item_col: str = "movie_id",
    ):
        self.n_candidates = n_candidates
        self.seed = seed
        self.user_col = user_col
        self.item_col = item_col

    def _rng_for_user(self, user_id: str) -> np.random.Generator:
        key = f"{self.seed}:{user_id}".encode()
        digest = hashlib.blake2b(key, digest_size=8).digest()
        user_seed = int.from_bytes(digest, "little") % (2**32)
        return np.random.default_rng(user_seed)

    def _load_data(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        movies_path: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dtype = {self.user_col: str, self.item_col: str}
        train = pd.read_csv(train_path, dtype=dtype)
        val = pd.read_csv(val_path, dtype=dtype)
        test = pd.read_csv(test_path, dtype=dtype)
        movies = pd.read_csv(movies_path, dtype={self.item_col: str})
        return train, val, test, movies

    def _build_history(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
    ) -> dict[str, set[str]]:
        fit = pd.concat([train, val], ignore_index=True)
        return fit.groupby(self.user_col)[self.item_col].apply(set).to_dict()

    def _sample_user(
        self,
        user_id: str,
        true_item: str,
        catalog: set[str],
        watched: set[str],
    ) -> list[str]:
        pool = sorted(catalog - watched - {true_item})
        n = min(self.n_candidates, len(pool))
        sampled = self._rng_for_user(user_id).choice(pool, size=n, replace=False).tolist()
        return sampled + [true_item]

    def run(
        self,
        train_path: str = "data/processed/train.csv",
        val_path: str = "data/processed/val.csv",
        test_path: str = "data/processed/test.csv",
        movies_path: str = "data/raw/movies.csv",
        output_path: str = "data/processed/inference_candidates.csv",
    ) -> pd.DataFrame:
        train, val, test, movies = self._load_data(
            train_path,
            val_path,
            test_path,
            movies_path,
        )
        catalog = set(movies[self.item_col].astype(str))
        history = self._build_history(train, val)

        rows = []
        for _, row in test.iterrows():
            user_id = str(row[self.user_col])
            true_item = str(row[self.item_col])
            candidates = self._sample_user(
                user_id,
                true_item,
                catalog,
                history.get(user_id, set()),
            )
            rows.extend(
                {self.user_col: user_id, self.item_col: movie_id}
                for movie_id in candidates
            )

        out_df = pd.DataFrame(rows)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out, index=False)
        print(f"Saved {len(out_df):,} candidate rows -> {out}")
        return out_df


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/processed/train.csv")
    parser.add_argument("--val-path", default="data/processed/val.csv")
    parser.add_argument("--test-path", default="data/processed/test.csv")
    parser.add_argument("--movies-path", default="data/raw/movies.csv")
    parser.add_argument("--output-path", default="data/processed/inference_candidates.csv")
    parser.add_argument("--n-candidates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    CandidateSampler(
        n_candidates=args.n_candidates,
        seed=args.seed,
    ).run(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        movies_path=args.movies_path,
        output_path=args.output_path,
    )
