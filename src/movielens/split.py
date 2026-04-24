from pathlib import Path

import pandas as pd


class LOOSplitter:
    def __init__(
        self,
        min_interactions: int = 3,
        user_col: str = "user_id",
        item_col: str = "movie_id",
        timestamp_col: str = "timestamp",
    ):
        self.min_interactions = min_interactions
        self.user_col = user_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col

    def _filter_users(self, df: pd.DataFrame) -> pd.DataFrame:
        counts = df.groupby(self.user_col)[self.item_col].transform("count")
        return df[counts >= self.min_interactions].copy()

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values([self.user_col, self.timestamp_col]).reset_index(drop=True)

    def _assign_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        rank = df.groupby(self.user_col).cumcount(ascending=False)
        df["split"] = "train"
        df.loc[rank == 1, "split"] = "val"
        df.loc[rank == 0, "split"] = "test"
        return df

    def run(self, input_path: str = "data/raw", output_path: str = "data/processed"):
        input_dir = Path(input_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(input_dir / "ratings.csv", dtype=str)
        df[self.timestamp_col] = df[self.timestamp_col].astype(int)

        df = self._filter_users(df)
        df = self._sort(df)
        df = self._assign_splits(df)

        for split in ("train", "val", "test"):
            subset = df[df["split"] == split].drop(columns="split")
            path = output_dir / f"{split}.csv"
            subset.to_csv(path, index=False)
            print(f"{split}: {len(subset):,} rows → {path}")


if __name__ == "__main__":
    LOOSplitter().run()
