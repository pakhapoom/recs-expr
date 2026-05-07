from pathlib import Path

import numpy as np
import pandas as pd


def load_candidate_map(
    candidates_path: str | None,
    user_col: str = "user_id",
    item_col: str = "movie_id",
    required_users: list[str] | None = None,
) -> dict[str, list[str]] | None:
    if candidates_path is None:
        return None

    path = Path(candidates_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {path}. "
            "Create it with `uv run src/movielens/sample_candidates.py`."
        )

    candidates = pd.read_csv(path, dtype={user_col: str, item_col: str})
    missing_cols = {user_col, item_col} - set(candidates.columns)
    if missing_cols:
        raise ValueError(f"Candidate file is missing columns: {sorted(missing_cols)}")

    candidate_map = (
        candidates.drop_duplicates([user_col, item_col])
        .groupby(user_col, sort=False)[item_col]
        .apply(list)
        .to_dict()
    )

    if required_users is not None:
        missing_users = sorted(set(required_users) - set(candidate_map))
        if missing_users:
            preview = ", ".join(missing_users[:5])
            raise ValueError(
                f"Candidate file is missing {len(missing_users)} users, e.g. {preview}"
            )

    return candidate_map


def candidate_indices(
    candidate_map: dict[str, list[str]] | None,
    movie_id2idx: dict[str, int],
) -> dict[str, np.ndarray] | None:
    if candidate_map is None:
        return None

    out = {}
    for user_id, movie_ids in candidate_map.items():
        idxs = []
        seen = set()
        for movie_id in movie_ids:
            idx = movie_id2idx.get(str(movie_id))
            if idx is not None and idx not in seen:
                idxs.append(idx)
                seen.add(idx)
        out[str(user_id)] = np.asarray(idxs, dtype=np.int64)
    return out
