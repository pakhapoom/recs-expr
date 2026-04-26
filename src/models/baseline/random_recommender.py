import time
from pathlib import Path

import numpy as np
import pandas as pd


class RandomRecommender:
    def __init__(
        self,
        n_recs: int = 100,
        seed: int = 42,
        user_col: str = "user_id",
        item_col: str = "movie_id",
        score_col: str = "score",
    ):
        self.n_recs = n_recs
        self.seed = seed
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col

    def _load_data(
        self, train_path: str, val_path: str, test_path: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dtype = {self.user_col: str, self.item_col: str}
        train = pd.read_csv(train_path, dtype=dtype)
        val = pd.read_csv(val_path, dtype=dtype)
        test = pd.read_csv(test_path, dtype=dtype)
        return train, val, test

    def _get_item_pool(self, train_df: pd.DataFrame) -> list[str]:
        return sorted(train_df[self.item_col].unique())

    def _get_user_history(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> dict[str, set[str]]:
        combined = pd.concat([train_df, val_df], ignore_index=True)
        return (
            combined.groupby(self.user_col)[self.item_col]
            .apply(set)
            .to_dict()
        )

    def _recommend_user(
        self, user_id: str, item_pool: list[str], watched: set[str]
    ) -> pd.DataFrame:
        candidates = [i for i in item_pool if i not in watched]
        rng = np.random.default_rng(hash(f"{self.seed}:{user_id}") % (2**32))
        n = min(self.n_recs, len(candidates))
        sampled = rng.choice(candidates, size=n, replace=False)
        scores = rng.uniform(size=n)
        return pd.DataFrame(
            {self.user_col: user_id, self.item_col: sampled, self.score_col: scores}
        )

    def run(
        self,
        train_path: str = "data/processed/train.csv",
        val_path: str = "data/processed/val.csv",
        test_path: str = "data/processed/test.csv",
        recs_path: str = "data/recs/random.csv",
        metrics_dir: str = "data/metrics",
    ):
        train, val, test = self._load_data(train_path, val_path, test_path)
        item_pool = self._get_item_pool(train)
        user_history = self._get_user_history(train, val)
        test_users = test[self.user_col].unique()

        t0 = time.perf_counter()
        recs = pd.concat(
            [
                self._recommend_user(user, item_pool, user_history.get(user, set()))
                for user in test_users
            ],
            ignore_index=True,
        )
        time_inference_s = time.perf_counter() - t0

        out = Path(recs_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        recs.to_csv(out, index=False)
        print(f"Saved {len(recs):,} rows → {out}")

        experiment = out.stem
        timing = pd.DataFrame([
            {"metric": "time_training_s", "value": 0.0},
            {"metric": "time_inference_s", "value": time_inference_s},
        ])
        metrics_out = Path(metrics_dir) / f"{experiment}.csv"
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        timing.to_csv(metrics_out, index=False)
        print(f"Timing → {metrics_out}")


if __name__ == "__main__":
    RandomRecommender().run()
