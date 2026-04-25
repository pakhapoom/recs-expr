# Two-Tower Recommender

A two-stage neural recommender (retrieval → ranking) trained on MovieLens 1M.

## Data Requirements

| File | Role |
|---|---|
| `data/processed/train.csv` | Model training + user watch histories |
| `data/processed/val.csv` | Merged with train for inference-time user histories |
| `data/processed/test.csv` | Provides the set of users to recommend for |
| `data/raw/users.csv` | User demographic features |
| `data/raw/movies.csv` | Item features (genres, release year) |

## Architecture

### Stage 1 — Retrieval (`TwoTowerModel`)

Two encoders produce L2-normalized 64-dim embeddings, then trained with **InfoNCE** (in-batch negatives, learned temperature).

- **User tower:** gender + age/occupation/zip embeddings + rating-weighted mean of watch history → MLP
- **Item tower:** item embedding + 18-dim genre vector + z-scored year → MLP
- Both towers share a single 32-dim movie embedding

### Stage 2 — Ranking (`RankingModel`)

Two-tower weights are frozen. A shallow MLP scores `[u ‖ i ‖ u⊙i]` trained with **BPR loss** (4 uniform random negatives per positive, sampled outside each user's full history).

## Inference

1. Pre-encode all items once.
2. Per test user: dot-product retrieval over all items → mask watched → top 500 candidates.
3. `RankingModel` re-scores candidates → top 100 recommendations.
4. Output: `data/recs/two_tower.csv` (`user_id`, `movie_id`, `score`).

## Usage

```bash
uv run src/models/two_tower/two_tower_recommender.py
```
