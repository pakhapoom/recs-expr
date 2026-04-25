import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(
        self,
        num_ages: int,
        num_occs: int,
        num_zips: int,
        item_emb_dim: int = 32,
        age_emb_dim: int = 8,
        occ_emb_dim: int = 16,
        zip_emb_dim: int = 16,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.item_embed: nn.Embedding | None = None  # shared with ItemTower, injected by TwoTowerModel
        self.age_embed = nn.Embedding(num_ages, age_emb_dim)
        self.occ_embed = nn.Embedding(num_occs, occ_emb_dim)
        self.zip_embed = nn.Embedding(num_zips, zip_emb_dim)

        in_dim = 1 + age_emb_dim + occ_emb_dim + zip_emb_dim + item_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(
        self,
        gender: torch.Tensor,
        age_idx: torch.Tensor,
        occ_idx: torch.Tensor,
        zip_idx: torch.Tensor,
        hist_ids: torch.Tensor,
        hist_ratings: torch.Tensor,
        hist_lens: torch.Tensor,
    ) -> torch.Tensor:
        item_embs = self.item_embed(hist_ids)  # (B, max_hist, item_emb_dim)
        seq_len = hist_ids.size(1)
        mask = (
            torch.arange(seq_len, device=hist_ids.device).unsqueeze(0) < hist_lens.unsqueeze(1)
        ).unsqueeze(-1).float()  # (B, max_hist, 1)

        weighted = item_embs * hist_ratings.unsqueeze(-1) * mask
        hist_vec = weighted.sum(1) / mask.sum(1).clamp(min=1)  # (B, item_emb_dim)

        x = torch.cat(
            [gender.unsqueeze(1), self.age_embed(age_idx), self.occ_embed(occ_idx),
             self.zip_embed(zip_idx), hist_vec],
            dim=1,
        )
        return F.normalize(self.mlp(x), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, item_emb_dim: int = 32, num_genres: int = 18, embed_dim: int = 64):
        super().__init__()
        self.item_embed: nn.Embedding | None = None  # shared, injected by TwoTowerModel

        in_dim = item_emb_dim + num_genres + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(
        self,
        item_idx: torch.Tensor,
        genres: torch.Tensor,
        year: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([self.item_embed(item_idx), genres, year.unsqueeze(1)], dim=1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_movies: int,
        num_ages: int,
        num_occs: int,
        num_zips: int,
        item_emb_dim: int = 32,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.shared_item_embed = nn.Embedding(num_movies, item_emb_dim)

        self.user_tower = UserTower(
            num_ages, num_occs, num_zips,
            item_emb_dim=item_emb_dim,
            embed_dim=embed_dim,
        )
        self.item_tower = ItemTower(item_emb_dim=item_emb_dim, embed_dim=embed_dim)

        self.user_tower.item_embed = self.shared_item_embed
        self.item_tower.item_embed = self.shared_item_embed

        self.log_temp = nn.Parameter(torch.zeros(1))

    def encode_user(self, gender, age_idx, occ_idx, zip_idx, hist_ids, hist_ratings, hist_lens):
        return self.user_tower(gender, age_idx, occ_idx, zip_idx, hist_ids, hist_ratings, hist_lens)

    def encode_item(self, item_idx, genres, year):
        return self.item_tower(item_idx, genres, year)


class RankingModel(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        in_dim = embed_dim * 3  # user ‖ item ‖ user⊙item
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([user_emb, item_emb, user_emb * item_emb], dim=-1)
        return self.mlp(x).squeeze(-1)
