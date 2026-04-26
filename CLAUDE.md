# Claude Instructions

## Package Manager
Use `uv` for all Python dependency and execution tasks — never use `pip`, `python`, or `python3` directly.

| Task | Command |
|---|---|
| Sync dependencies / prepare venv | `uv sync` |
| Install a package | `uv add <package>` (space-separated for multiple) |
| Run a Python file | `uv run <path/to/file.py>` |
| Run a module command (e.g. pytest) | `uv run pytest <path/to/tests>` |

## Code Style
- Python 3.12+
- Prefer simple, direct code over abstractions
- No unnecessary comments, docstrings, or type annotations on unchanged code
- Comments use `# <comment>` style and American English

## General Preferences
- Short, concise responses — lead with the answer
- No trailing summaries after completing a task
- Ask before taking irreversible actions (force push, deleting files, etc.)

## Input Data Schema (MovieLens 1M)

Raw data lives in `data/raw/` as CSV files (originally `::` delimited `.dat` files, converted on ingest).

### `data/raw/ratings.csv` — columns: `user_id, movie_id, rating, timestamp`
| Column | Type | Notes |
|---|---|---|
| user_id | int | 1–6040 |
| movie_id | int | 1–3952 |
| rating | int | 1–5 (whole stars only) |
| timestamp | int | Unix epoch seconds |

### `data/raw/users.csv` — columns: `user_id, gender, age, occupation, zip_code`
| Column | Type | Notes |
|---|---|---|
| user_id | int | 1–6040 |
| gender | str | `M` or `F` |
| age | int | Age bracket code (see below) |
| occupation | int | Occupation code (see below) |
| zip_code | str | US zip code |

**Age brackets:** `1`=Under 18, `18`=18–24, `25`=25–34, `35`=35–44, `45`=45–49, `50`=50–55, `56`=56+

**Occupation codes:** `0`=other, `1`=academic/educator, `2`=artist, `3`=clerical/admin, `4`=college/grad student, `5`=customer service, `6`=doctor/health care, `7`=executive/managerial, `8`=farmer, `9`=homemaker, `10`=K-12 student, `11`=lawyer, `12`=programmer, `13`=retired, `14`=sales/marketing, `15`=scientist, `16`=self-employed, `17`=technician/engineer, `18`=tradesman/craftsman, `19`=unemployed, `20`=writer

### `data/raw/movies.csv` — columns: `movie_id, title, genres`
| Column | Type | Notes |
|---|---|---|
| movie_id | int | 1–3952 |
| title | str | Includes release year, e.g. `Toy Story (1995)` |
| genres | str | Pipe-separated list, e.g. `Animation\|Children's\|Comedy` |

**Valid genres:** Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western

## Data Splits

Data is split using a **leave-one-out (LOO)** strategy: the most recent interaction per user is held out as the test item. Processed splits are saved to `data/processed/`:

| File | Description |
|---|---|
| `data/processed/train.csv` | All interactions except each user's last two |
| `data/processed/val.csv` | Each user's second-to-last interaction |
| `data/processed/test.csv` | Each user's last interaction (held-out item) |

## Implementing a Model

Before implementing a new recommender, read `src/models/recs_schema.md`. It defines:
- The required output format for recommendations (`data/recs/{experiment}.csv`)
- The timing and ranking metrics format (`data/metrics/{experiment}.csv`)
- How to wire timing instrumentation into `run()`
