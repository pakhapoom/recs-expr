import io
import zipfile
from pathlib import Path

import requests
import pandas as pd

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


class MovieLensDownloader:
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_zip(self) -> zipfile.ZipFile:
        print("Downloading MovieLens 1M...")
        response = requests.get(DATASET_URL, stream=True)
        response.raise_for_status()
        return zipfile.ZipFile(io.BytesIO(response.content))

    def _read_dat(self, zf: zipfile.ZipFile, filename: str) -> list[str]:
        with zf.open(f"ml-1m/{filename}") as f:
            return f.read().decode("latin-1").splitlines()

    def _parse_users(self, zf: zipfile.ZipFile) -> pd.DataFrame:
        rows = [line.split("::") for line in self._read_dat(zf, "users.dat")]
        return pd.DataFrame(rows, columns=["user_id", "gender", "age", "occupation", "zip_code"])

    def _parse_movies(self, zf: zipfile.ZipFile) -> pd.DataFrame:
        rows = [line.split("::") for line in self._read_dat(zf, "movies.dat")]
        return pd.DataFrame(rows, columns=["movie_id", "title", "genres"])

    def _parse_ratings(self, zf: zipfile.ZipFile) -> pd.DataFrame:
        rows = [line.split("::") for line in self._read_dat(zf, "ratings.dat")]
        return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])

    def run(self):
        zf = self._fetch_zip()

        datasets = {
            "users": self._parse_users(zf),
            "movies": self._parse_movies(zf),
            "ratings": self._parse_ratings(zf),
        }

        for name, df in datasets.items():
            path = self.output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            print(f"Saved {len(df):,} rows → {path}")


if __name__ == "__main__":
    MovieLensDownloader().run()
