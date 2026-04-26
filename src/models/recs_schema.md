# Recommendations Result Schema

The `recs` DataFrame passed to `LOOEvaluator.evaluate()` (or loaded from `recs_path` in `run()`) must conform to this schema.

## Columns

| Column | Type | Description |
|---|---|---|
| `user_id` | `str` | User identifier. Must match user IDs in `test.csv`. |
| `movie_id` | `str` | Recommended item identifier. |
| `score` | `float` | Relevance score. Higher values rank higher. |

> Column names are configurable via `LOOEvaluator(user_col=, item_col=, score_col=)`. The defaults above match the constructor defaults.

## Constraints

- Each `(user_id, movie_id)` pair must be unique — duplicate pairs produce incorrect ranks.
- All users present in `test.csv` should appear in `recs`; missing users count as misses for every metric.
- `score` must be a numeric type comparable by `sort_values`; `NaN` values are not handled and will produce undefined ranking behavior.
- No minimum number of rows per user is enforced, but Hit@K and NDCG@K are 0 for any user with fewer than K recommendations.

## Example

```csv
user_id,movie_id,score
1,296,4.82
1,318,4.75
1,527,4.61
2,296,3.90
2,858,3.85
```

## Output Schema (`evaluate()` return value)

| Column | Type | Description |
|---|---|---|
| `metric` | `str` | Metric name, e.g. `hit@10`, `ndcg@50`. |
| `value` | `float` | Mean value across all users in `test.csv`. |

Default metrics computed for each K in `[5, 10, 50, 100]`:

- `hit@K` — 1.0 if the held-out item appears in the top-K recommendations, else 0.0.
- `ndcg@K` — Normalized Discounted Cumulative Gain at K (ideal DCG = 1, single relevant item).

## Metrics File (`data/metrics/{experiment}.csv`)

Each experiment produces a single CSV combining timing and ranking metrics. The same `metric,value` schema is used throughout.

**Write order:** the model's `run()` writes timing rows first (overwriting any existing file), then `LOOEvaluator.run()` appends ranking rows.

### Timing metrics (written by the model)

| `metric` | `value` | Description |
|---|---|---|
| `time_training_s` | `float` | Wall-clock seconds spent fitting the model. `0.0` for stateless models (e.g. random). |
| `time_inference_s` | `float` | Wall-clock seconds spent generating recommendations for all test users. |

**What counts as training vs inference per model type:**

| Model | `time_training_s` | `time_inference_s` |
|---|---|---|
| `RandomRecommender` | `0.0` (stateless) | rec generation loop |
| `PopularityRecommender` | `_compute_popularity()` | rec generation loop |
| `TwoTowerRecommender` | stage 1 retrieval training + stage 2 ranking training | item embedding pre-computation + per-user retrieval and ranking loop |

### Full file example

```csv
metric,value
time_training_s,42.3
time_inference_s,5.1
hit@5,0.21
hit@10,0.34
hit@50,0.58
hit@100,0.71
ndcg@5,0.18
ndcg@10,0.24
ndcg@50,0.31
ndcg@100,0.34
```

> When implementing a new model, call `time.perf_counter()` around the training and inference phases inside `run()`, then write a `data/metrics/{experiment}.csv` with the two timing rows before returning. `LOOEvaluator` will append the ranking rows.
