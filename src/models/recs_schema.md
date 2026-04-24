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
