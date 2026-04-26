from pathlib import Path

import numpy as np
import pandas as pd


class LOOEvaluator:
    def __init__(
        self,
        k_values: list[int] = [5, 10, 50, 100],
        user_col: str = "user_id",
        item_col: str = "movie_id",
        score_col: str = "score",
    ):
        self.k_values = sorted(k_values)
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col

    def _load_ground_truth(self, test_path: Path) -> dict[str, str]:
        df = pd.read_csv(test_path, dtype=str)
        # LOO: one held-out item per user
        return dict(zip(df[self.user_col], df[self.item_col]))

    def _rank_recommendations(self, recs: pd.DataFrame) -> pd.DataFrame:
        recs = recs.sort_values([self.user_col, self.score_col], ascending=[True, False])
        recs["rank"] = recs.groupby(self.user_col).cumcount() + 1
        return recs

    def _hit_at_k(self, rank: int | None, k: int) -> float:
        return 1.0 if rank is not None and rank <= k else 0.0

    def _ndcg_at_k(self, rank: int | None, k: int) -> float:
        if rank is None or rank > k:
            return 0.0
        return 1.0 / np.log2(rank + 1)

    def _compute_user_metrics(
        self, ground_truth: dict[str, str], recs: pd.DataFrame
    ) -> pd.DataFrame:
        # build a lookup: user -> {item -> rank}
        user_item_rank = (
            recs.set_index([self.user_col, self.item_col])["rank"]
            .to_dict()
        )

        rows = []
        for user, true_item in ground_truth.items():
            rank = user_item_rank.get((user, true_item))
            row = {self.user_col: user}
            for k in self.k_values:
                row[f"hit@{k}"] = self._hit_at_k(rank, k)
                row[f"ndcg@{k}"] = self._ndcg_at_k(rank, k)
            rows.append(row)

        return pd.DataFrame(rows)

    def evaluate(
        self,
        recs: pd.DataFrame,
        test_path: str = "data/processed/test.csv",
    ) -> pd.DataFrame:
        ground_truth = self._load_ground_truth(Path(test_path))
        recs = self._rank_recommendations(recs.copy())
        user_metrics = self._compute_user_metrics(ground_truth, recs)

        metric_cols = [c for c in user_metrics.columns if c != self.user_col]
        summary = user_metrics[metric_cols].mean().to_frame("value").reset_index()
        summary.columns = ["metric", "value"]
        return summary

    def run(
        self,
        recs_path: str,
        experiment: str = None,
        test_path: str = "data/processed/test.csv",
        metrics_dir: str = "data/metrics",
    ):
        if experiment is None:
            experiment = Path(recs_path).stem
        recs = pd.read_csv(recs_path, dtype={self.user_col: str, self.item_col: str})
        summary = self.evaluate(recs, test_path)

        print(summary.to_string(index=False))

        out = Path(metrics_dir) / f"{experiment}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"\nSaved → {out}")

        return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--recs_path", required=True)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--test_path", default="data/processed/test.csv")
    parser.add_argument("--metrics_dir", default="data/metrics")
    args = parser.parse_args()

    LOOEvaluator().run(
        recs_path=args.recs_path,
        experiment=args.experiment,
        test_path=args.test_path,
        metrics_dir=args.metrics_dir,
    )
