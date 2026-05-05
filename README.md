# Recommendations experiment

## Getting Started

### 1. Download MovieLens 1M

Downloads the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/) and saves `users.csv`, `movies.csv`, and `ratings.csv` to `data/raw/`.

```bash
uv run src/movielens/download.py
```

### 2. Split into train / val / test

Applies leave-one-out splitting and writes `train.csv`, `val.csv`, and `test.csv` to `data/processed/`.

```bash
uv run src/movielens/split.py
```

### 3. Train a model

Before implementing a model, read [`src/models/recs_schema.md`](src/models/recs_schema.md) — it defines the output schema your model must produce (columns: `user_id`, `movie_id`, `score`).

Place your model under `src/models/` and write its recommendations to a CSV conforming to that schema, e.g. `data/recs/<experiment>.csv`.

### 4. Evaluate

Run the LOO evaluator against your recommendations CSV. Pass the experiment name used when saving the file — metrics are written to `data/metrics/<experiment>.csv`.

```bash
uv run src/evaluate.py --recs_path data/recs/my_model.csv
```

Example output:

```
   metric     value
   hit@5  0.043200
  hit@10  0.072100
  hit@50  0.198300
 hit@100  0.301400
  ndcg@5  0.025100
 ndcg@10  0.033800
 ndcg@50  0.052600
ndcg@100  0.059200

Saved → data/metrics/my_model.csv
```

### 5. Plot NDCG@k

Plots the `ndcg@k` rows from the default experiment metrics files in `data/metrics/` and saves a comparison chart to `data/viz/`.

```bash
uv run src/viz/plot_ndcg.py
```
