import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from features import FeatureStore
from model import TwoTowerModel, RankingModel


class RetrievalDataset(Dataset):
    def __init__(
        self,
        interactions_df,
        user_feats: dict,
        user_histories: dict,
        store: FeatureStore,
    ):
        self.user_ids = interactions_df["user_id"].astype(str).tolist()
        self.item_idxs = [
            store.movie_id2idx.get(str(mid), 0)
            for mid in interactions_df["movie_id"]
        ]
        self.user_feats = user_feats
        self.user_histories = user_histories
        self.item_genres = store.item_genres
        self.item_year = store.item_year
        self.max_history = store.max_history
        _zeros_ids = np.zeros(store.max_history, dtype=np.int64)
        _zeros_rats = np.zeros(store.max_history, dtype=np.float32)
        self._empty_hist = (_zeros_ids, _zeros_rats, 0)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        item_idx = self.item_idxs[idx]
        uf = self.user_feats[uid]
        pad_ids, pad_rats, hist_len = self.user_histories.get(uid, self._empty_hist)
        return {
            "gender": torch.tensor(uf["gender"]),
            "age_idx": torch.tensor(uf["age_idx"], dtype=torch.long),
            "occ_idx": torch.tensor(uf["occ_idx"], dtype=torch.long),
            "zip_idx": torch.tensor(uf["zip_idx"], dtype=torch.long),
            "hist_ids": torch.tensor(pad_ids, dtype=torch.long),
            "hist_ratings": torch.tensor(pad_rats),
            "hist_lens": torch.tensor(hist_len, dtype=torch.long),
            "item_idx": torch.tensor(item_idx, dtype=torch.long),
            "genres": torch.tensor(self.item_genres[item_idx]),
            "year": torch.tensor(self.item_year[item_idx]),
        }


class RankingDataset(Dataset):
    def __init__(
        self,
        interactions_df,
        user_feats: dict,
        user_histories: dict,
        store: FeatureStore,
        user_history_sets: dict[str, set[int]],
        n_neg: int = 4,
        seed: int = 42,
    ):
        self.user_ids = interactions_df["user_id"].astype(str).tolist()
        self.item_idxs = [
            store.movie_id2idx.get(str(mid), 0)
            for mid in interactions_df["movie_id"]
        ]
        self.user_feats = user_feats
        self.user_histories = user_histories
        self.item_genres = store.item_genres
        self.item_year = store.item_year
        self.max_history = store.max_history
        self.user_history_sets = user_history_sets
        self.n_neg = n_neg
        self.num_movies = len(store.movie_ids)
        self.rng = np.random.default_rng(seed)
        _zeros_ids = np.zeros(store.max_history, dtype=np.int64)
        _zeros_rats = np.zeros(store.max_history, dtype=np.float32)
        self._empty_hist = (_zeros_ids, _zeros_rats, 0)

    def __len__(self):
        return len(self.user_ids)

    def _sample_negatives(self, uid: str, pos_idx: int) -> list[int]:
        watched = self.user_history_sets.get(uid, set())
        neg_idxs = []
        while len(neg_idxs) < self.n_neg:
            cand = int(self.rng.integers(0, self.num_movies))
            if cand != pos_idx and cand not in watched:
                neg_idxs.append(cand)
        return neg_idxs

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        pos_idx = self.item_idxs[idx]
        uf = self.user_feats[uid]
        pad_ids, pad_rats, hist_len = self.user_histories.get(uid, self._empty_hist)
        neg_idxs = self._sample_negatives(uid, pos_idx)

        user_data = {
            "gender": torch.tensor(uf["gender"]),
            "age_idx": torch.tensor(uf["age_idx"], dtype=torch.long),
            "occ_idx": torch.tensor(uf["occ_idx"], dtype=torch.long),
            "zip_idx": torch.tensor(uf["zip_idx"], dtype=torch.long),
            "hist_ids": torch.tensor(pad_ids, dtype=torch.long),
            "hist_ratings": torch.tensor(pad_rats),
            "hist_lens": torch.tensor(hist_len, dtype=torch.long),
        }
        pos_data = {
            "item_idx": torch.tensor(pos_idx, dtype=torch.long),
            "genres": torch.tensor(self.item_genres[pos_idx]),
            "year": torch.tensor(self.item_year[pos_idx]),
        }
        neg_data = {
            "item_idx": torch.tensor(neg_idxs, dtype=torch.long),
            "genres": torch.tensor(self.item_genres[neg_idxs]),
            "year": torch.tensor(self.item_year[neg_idxs]),
        }
        return user_data, pos_data, neg_data


def train_retrieval(
    model: TwoTowerModel,
    dataset: RetrievalDataset,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TwoTowerModel:
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            user_emb = model.encode_user(
                batch["gender"], batch["age_idx"], batch["occ_idx"], batch["zip_idx"],
                batch["hist_ids"], batch["hist_ratings"], batch["hist_lens"],
            )
            item_emb = model.encode_item(batch["item_idx"], batch["genres"], batch["year"])

            # InfoNCE with in-batch negatives
            temp = torch.exp(model.log_temp).clamp(min=0.01, max=1.0)
            logits = torch.matmul(user_emb, item_emb.T) / temp  # (B, B)
            labels = torch.arange(len(user_emb), device=device)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Retrieval epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")

    return model


def train_ranking(
    two_tower: TwoTowerModel,
    ranker: RankingModel,
    dataset: RankingDataset,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> RankingModel:
    two_tower = two_tower.to(device).eval()
    ranker = ranker.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(ranker.parameters(), lr=lr)

    for epoch in range(epochs):
        ranker.train()
        total_loss = 0.0
        for user_data, pos_data, neg_data in loader:
            user_data = {k: v.to(device) for k, v in user_data.items()}
            pos_data = {k: v.to(device) for k, v in pos_data.items()}
            neg_data = {k: v.to(device) for k, v in neg_data.items()}

            with torch.no_grad():
                user_emb = two_tower.encode_user(**user_data)  # (B, D)
                pos_emb = two_tower.encode_item(**pos_data)    # (B, D)

                B, n_neg = neg_data["item_idx"].shape
                neg_emb = two_tower.encode_item(
                    neg_data["item_idx"].view(-1),
                    neg_data["genres"].view(B * n_neg, -1),
                    neg_data["year"].view(-1),
                ).view(B, n_neg, -1)  # (B, n_neg, D)

            pos_score = ranker(user_emb, pos_emb)  # (B,)

            user_emb_exp = user_emb.unsqueeze(1).expand(B, n_neg, -1).reshape(B * n_neg, -1)
            neg_score = ranker(user_emb_exp, neg_emb.view(B * n_neg, -1)).view(B, n_neg)

            # BPR loss
            loss = -F.logsigmoid(pos_score.unsqueeze(1) - neg_score).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Ranking epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")

    return ranker
