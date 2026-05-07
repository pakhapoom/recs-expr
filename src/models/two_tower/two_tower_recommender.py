import sys
import time
from pathlib import Path

project_root = next(
    p for p in Path(__file__).resolve().parents
    if (p / "src" / "models" / "candidate_sets.py").exists()
)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

from src.models.candidate_sets import candidate_indices, load_candidate_map

from features import FeatureStore
from model import TwoTowerModel, RankingModel
from train import RetrievalDataset, RankingDataset, train_retrieval, train_ranking


class TwoTowerRecommender:
    def __init__(
        self,
        n_recs: int = 100,
        n_candidates: int = 500,
        max_history: int = 50,
        item_emb_dim: int = 32,
        embed_dim: int = 64,
        retrieval_epochs: int = 10,
        ranking_epochs: int = 5,
        batch_size: int = 256,
        lr: float = 1e-3,
        n_neg: int = 4,
        seed: int = 42,
        user_col: str = "user_id",
        item_col: str = "movie_id",
        score_col: str = "score",
    ):
        self.n_recs = n_recs
        self.n_candidates = n_candidates
        self.max_history = max_history
        self.item_emb_dim = item_emb_dim
        self.embed_dim = embed_dim
        self.retrieval_epochs = retrieval_epochs
        self.ranking_epochs = ranking_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_neg = n_neg
        self.seed = seed
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col

    def _load_data(self, train_path, val_path, test_path, users_path, movies_path):
        dtype = {self.user_col: str, self.item_col: str}
        train = pd.read_csv(train_path, dtype=dtype)
        val = pd.read_csv(val_path, dtype=dtype)
        test = pd.read_csv(test_path, dtype=dtype)
        users = pd.read_csv(users_path, dtype={self.user_col: str})
        movies = pd.read_csv(movies_path, dtype={self.item_col: str})
        return train, val, test, users, movies

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _encode_user_single(
        self,
        uid: str,
        user_feats: dict,
        histories: dict,
        model: TwoTowerModel,
        device: str,
    ) -> torch.Tensor:
        uf = user_feats[uid]
        _zero_ids = np.zeros(self.max_history, dtype=np.int64)
        _zero_rats = np.zeros(self.max_history, dtype=np.float32)
        pad_ids, pad_rats, hist_len = histories.get(uid, (_zero_ids, _zero_rats, 0))
        return model.encode_user(
            torch.tensor([uf["gender"]], device=device),
            torch.tensor([uf["age_idx"]], dtype=torch.long, device=device),
            torch.tensor([uf["occ_idx"]], dtype=torch.long, device=device),
            torch.tensor([uf["zip_idx"]], dtype=torch.long, device=device),
            torch.from_numpy(pad_ids).unsqueeze(0).to(device),
            torch.from_numpy(pad_rats).unsqueeze(0).to(device),
            torch.tensor([hist_len], dtype=torch.long, device=device),
        )  # (1, D)

    def run(
        self,
        train_path: str = "data/processed/train.csv",
        val_path: str = "data/processed/val.csv",
        test_path: str = "data/processed/test.csv",
        users_path: str = "data/raw/users.csv",
        movies_path: str = "data/raw/movies.csv",
        candidates_path: str = "data/processed/inference_candidates.csv",
        recs_path: str = "data/recs/two_tower.csv",
        metrics_dir: str = "data/metrics",
    ):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self._get_device()
        print(f"Device: {device}")

        # --- Load ---
        train, val, test, users, movies = self._load_data(
            train_path, val_path, test_path, users_path, movies_path
        )

        # --- Features ---
        store = FeatureStore(max_history=self.max_history).fit(movies, users)
        user_feats = store.encode_users(users)
        train_histories = store.encode_histories(train)

        t_train_start = time.perf_counter()

        # --- Stage 1: retrieval ---
        print("Stage 1: Training retrieval model...")
        retrieval_model = TwoTowerModel(
            num_movies=len(store.movie_ids),
            num_ages=len(store.age_vocab),
            num_occs=len(store.occ_vocab),
            num_zips=len(store.zip_vocab),
            item_emb_dim=self.item_emb_dim,
            embed_dim=self.embed_dim,
        )
        retrieval_dataset = RetrievalDataset(train, user_feats, train_histories, store)
        retrieval_model = train_retrieval(
            retrieval_model, retrieval_dataset,
            epochs=self.retrieval_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device=device,
        )

        # --- Build history sets (train + val) for exclusion and user encoding at inference ---
        train_val = pd.concat([train, val], ignore_index=True)
        train_val_histories = store.encode_histories(train_val)
        history_sets: dict[str, set[int]] = {}
        for uid, group in train_val.groupby(self.user_col):
            history_sets[str(uid)] = {
                store.movie_id2idx.get(str(mid), 0)
                for mid in group[self.item_col]
            }
        test_users = test[self.user_col].astype(str).unique().tolist()
        candidate_map = load_candidate_map(
            candidates_path,
            self.user_col,
            self.item_col,
            test_users,
        )
        candidate_idx_by_user = candidate_indices(candidate_map, store.movie_id2idx)

        # --- Stage 2: ranking ---
        print("Stage 2: Training ranking model...")
        ranker = RankingModel(embed_dim=self.embed_dim)
        ranking_dataset = RankingDataset(
            train, user_feats, train_histories, store,
            user_history_sets=history_sets,
            n_neg=self.n_neg,
            seed=self.seed,
        )
        ranker = train_ranking(
            retrieval_model, ranker, ranking_dataset,
            epochs=self.ranking_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device=device,
        )

        time_training_s = time.perf_counter() - t_train_start

        t_infer_start = time.perf_counter()

        # Pre-encode all items
        print("Encoding all items...")
        retrieval_model.eval()
        all_item_idx = torch.arange(len(store.movie_ids), dtype=torch.long, device=device)
        all_genres = torch.tensor(store.item_genres, device=device)
        all_year = torch.tensor(store.item_year, device=device)
        with torch.no_grad():
            all_item_emb = retrieval_model.encode_item(all_item_idx, all_genres, all_year)
            all_item_emb_np = all_item_emb.cpu().numpy()  # (M, D)

        # --- Recommend for test users ---
        print("Generating recommendations...")
        retrieval_model.eval()
        ranker.eval()

        recs_list = []
        with torch.no_grad():
            for uid in test_users:
                if uid not in user_feats:
                    continue

                user_emb = self._encode_user_single(
                    uid, user_feats, train_val_histories, retrieval_model, device
                )  # (1, D)

                # Retrieval: dot-product over all items, mask watched
                watched_idxs = history_sets.get(uid, set())
                if candidate_idx_by_user is None:
                    retrieval_scores = (user_emb.cpu().numpy() @ all_item_emb_np.T)[0]
                    for w in watched_idxs:
                        retrieval_scores[w] = -np.inf

                    n_candidates = min(self.n_candidates, int((retrieval_scores > -np.inf).sum()))
                    top_k_idxs = np.argpartition(retrieval_scores, -n_candidates)[-n_candidates:]
                    top_k_idxs = top_k_idxs[np.argsort(retrieval_scores[top_k_idxs])[::-1]]
                else:
                    item_candidates = candidate_idx_by_user.get(
                        uid,
                        np.asarray([], dtype=np.int64),
                    )
                    item_candidates = np.asarray(
                        [idx for idx in item_candidates if idx not in watched_idxs],
                        dtype=np.int64,
                    )
                    if len(item_candidates) == 0:
                        continue

                    retrieval_scores = (
                        user_emb.cpu().numpy() @ all_item_emb_np[item_candidates].T
                    )[0]
                    n_candidates = min(self.n_candidates, len(item_candidates))
                    top = np.argpartition(retrieval_scores, -n_candidates)[-n_candidates:]
                    top = top[np.argsort(retrieval_scores[top])[::-1]]
                    top_k_idxs = item_candidates[top]

                # Ranking: re-score top-K candidates
                cand_idx_t = torch.tensor(top_k_idxs, dtype=torch.long, device=device)
                cand_emb = retrieval_model.encode_item(
                    cand_idx_t,
                    all_genres[top_k_idxs],
                    all_year[top_k_idxs],
                )  # (K, D)
                user_emb_exp = user_emb.expand(len(top_k_idxs), -1)
                ranking_scores = ranker(user_emb_exp, cand_emb).cpu().numpy()  # (K,)

                top_n = min(self.n_recs, len(top_k_idxs))
                best = np.argpartition(ranking_scores, -top_n)[-top_n:]
                best = best[np.argsort(ranking_scores[best])[::-1]]

                for rank_pos in best:
                    recs_list.append({
                        self.user_col: uid,
                        self.item_col: store.movie_ids[top_k_idxs[rank_pos]],
                        self.score_col: float(ranking_scores[rank_pos]),
                    })

        time_inference_s = time.perf_counter() - t_infer_start

        recs = pd.DataFrame(recs_list)
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
    TwoTowerRecommender().run()
