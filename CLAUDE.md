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

Delimiter: `::` between fields.

### ratings.dat — `UserID::MovieID::Rating::Timestamp`
| Field | Type | Notes |
|---|---|---|
| UserID | int | 1–6040 |
| MovieID | int | 1–3952 |
| Rating | int | 1–5 (whole stars only) |
| Timestamp | int | Unix epoch seconds |

### users.dat — `UserID::Gender::Age::Occupation::Zip-code`
| Field | Type | Notes |
|---|---|---|
| UserID | int | 1–6040 |
| Gender | str | `M` or `F` |
| Age | int | Age bracket code (see below) |
| Occupation | int | Occupation code (see below) |
| Zip-code | str | US zip code |

**Age brackets:** `1`=Under 18, `18`=18–24, `25`=25–34, `35`=35–44, `45`=45–49, `50`=50–55, `56`=56+

**Occupation codes:** `0`=other, `1`=academic/educator, `2`=artist, `3`=clerical/admin, `4`=college/grad student, `5`=customer service, `6`=doctor/health care, `7`=executive/managerial, `8`=farmer, `9`=homemaker, `10`=K-12 student, `11`=lawyer, `12`=programmer, `13`=retired, `14`=sales/marketing, `15`=scientist, `16`=self-employed, `17`=technician/engineer, `18`=tradesman/craftsman, `19`=unemployed, `20`=writer

### movies.dat — `MovieID::Title::Genres`
| Field | Type | Notes |
|---|---|---|
| MovieID | int | 1–3952 |
| Title | str | Includes release year, e.g. `Toy Story (1995)` |
| Genres | str | Pipe-separated list, e.g. `Animation\|Children's\|Comedy` |

**Valid genres:** Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western
