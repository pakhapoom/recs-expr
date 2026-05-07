import sys
import time
from pathlib import Path

import pandas as pd

project_root = next(
    p for p in Path(__file__).resolve().parents
    if (p / "src" / "models" / "candidate_sets.py").exists()
)
sys.path.insert(0, str(project_root))

from src.models.candidate_sets import load_candidate_map


class PopularityRecommender:
    def __init__(
        self,
        n_recs: int = 100,
        segment_cols: list[str] = ["gender", "age", "occupation", "zip_region"],
        min_segment_size: int = 10,
        user_col: str = "user_id",
        item_col: str = "movie_id",
        score_col: str = "score",
    ):
        self.n_recs = n_recs
        self.segment_cols = segment_cols
        self.min_segment_size = min_segment_size
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col

    def _load_data(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        users_path: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dtype = {self.user_col: str, self.item_col: str}
        train = pd.read_csv(train_path, dtype=dtype)
        val = pd.read_csv(val_path, dtype=dtype)
        test = pd.read_csv(test_path, dtype=dtype)
        users = pd.read_csv(users_path, dtype={self.user_col: str})
        users["zip_region"] = users["zip_code"].astype(str).str[0]
        users = users.drop(columns=["zip_code"])
        return train, val, test, users

    def _compute_popularity(
        self, interactions: pd.DataFrame, users: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        enriched = interactions.merge(users, on=self.user_col, how="left")

        seg_counts = (
            enriched.groupby(self.segment_cols + [self.item_col])
            .size()
            .rename("count")
            .reset_index()
        )
        seg_totals = seg_counts.groupby(self.segment_cols)["count"].transform("sum")
        seg_counts[self.score_col] = seg_counts["count"] / seg_totals
        seg_pop = seg_counts.drop(columns=["count"])

        global_counts = (
            interactions.groupby(self.item_col).size().rename("count").reset_index()
        )
        global_counts[self.score_col] = global_counts["count"] / global_counts["count"].sum()
        global_pop = global_counts.drop(columns=["count"]).sort_values(
            self.score_col, ascending=False
        )

        return seg_pop, global_pop

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
        self,
        user_id: str,
        seg_pop: pd.DataFrame,
        global_pop: pd.DataFrame,
        watched: set[str],
        user_segment: dict,
    ) -> pd.DataFrame:
        mask = pd.Series([True] * len(seg_pop))
        for col, val in user_segment.items():
            mask &= seg_pop[col] == val

        candidates = seg_pop[mask][[self.item_col, self.score_col]]
        candidates = candidates[~candidates[self.item_col].isin(watched)]
        if len(candidates) < self.min_segment_size:
            candidates = global_pop[~global_pop[self.item_col].isin(watched)][
                [self.item_col, self.score_col]
            ]

        candidates = candidates.sort_values(self.score_col, ascending=False).head(self.n_recs)
        candidates = candidates.copy()
        candidates[self.user_col] = user_id
        return candidates[[self.user_col, self.item_col, self.score_col]]

    def _recommend_candidate_user(
        self,
        user_id: str,
        candidate_ids: list[str],
        watched: set[str],
        user_segment: dict,
        seg_lookup: dict,
        global_lookup: dict,
    ) -> pd.DataFrame:
        seg_prefix = tuple(user_segment.get(col) for col in self.segment_cols)
        rows = []
        scored = 0

        for movie_id in dict.fromkeys(candidate_ids):
            if movie_id in watched:
                continue
            score = seg_lookup.get((*seg_prefix, movie_id))
            scored += score is not None
            rows.append({self.item_col: movie_id, self.score_col: score})

        if scored < self.min_segment_size:
            for row in rows:
                row[self.score_col] = global_lookup.get(row[self.item_col], 0.0)
        else:
            for row in rows:
                if row[self.score_col] is None:
                    row[self.score_col] = 0.0

        candidates = pd.DataFrame(rows)
        candidates = candidates.sort_values(self.score_col, ascending=False).head(self.n_recs)
        candidates = candidates.copy()
        candidates[self.user_col] = user_id
        return candidates[[self.user_col, self.item_col, self.score_col]]

    def run(
        self,
        train_path: str = "data/processed/train.csv",
        val_path: str = "data/processed/val.csv",
        test_path: str = "data/processed/test.csv",
        users_path: str = "data/raw/users.csv",
        candidates_path: str = "data/processed/inference_candidates.csv",
        recs_path: str = "data/recs/popularity.csv",
        metrics_dir: str = "data/metrics",
    ):
        train, val, test, users = self._load_data(train_path, val_path, test_path, users_path)
        interactions = pd.concat([train, val], ignore_index=True)

        t0 = time.perf_counter()
        seg_pop, global_pop = self._compute_popularity(interactions, users)
        time_training_s = time.perf_counter() - t0

        user_history = self._get_user_history(train, val)
        user_segments = users.set_index(self.user_col)[self.segment_cols].to_dict("index")
        test_users = test[self.user_col].astype(str).unique().tolist()
        candidate_map = load_candidate_map(
            candidates_path,
            self.user_col,
            self.item_col,
            test_users,
        )

        t0 = time.perf_counter()
        if candidate_map is None:
            recs = pd.concat(
                [
                    self._recommend_user(
                        user,
                        seg_pop,
                        global_pop,
                        user_history.get(user, set()),
                        user_segments.get(user, {}),
                    )
                    for user in test_users
                ],
                ignore_index=True,
            )
        else:
            seg_lookup = seg_pop.set_index(
                self.segment_cols + [self.item_col]
            )[self.score_col].to_dict()
            global_lookup = global_pop.set_index(self.item_col)[self.score_col].to_dict()
            recs = pd.concat(
                [
                    self._recommend_candidate_user(
                        user,
                        candidate_map[user],
                        user_history.get(user, set()),
                        user_segments.get(user, {}),
                        seg_lookup,
                        global_lookup,
                    )
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
            {"metric": "time_training_s", "value": time_training_s},
            {"metric": "time_inference_s", "value": time_inference_s},
        ])
        metrics_out = Path(metrics_dir) / f"{experiment}.csv"
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        timing.to_csv(metrics_out, index=False)
        print(f"Timing → {metrics_out}")


if __name__ == "__main__":
    PopularityRecommender().run()
