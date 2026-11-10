# AGENTS.md

## What this project does
Predicts MLB contract length and average annual value (AAV) for free agents and extensions
(offseasons 2014-2025) from performance history, using a two-output neural network compared
with linear and boosted-tree baselines. Course project (ANN and Deep Learning); final output
is a blog post plus this repo.

## Folders
- `src/` pipeline code (scraping, cleaning, features, models)
- `tests/` pytest tests
- `data/raw/` source data. Never edit by hand.
- `data/processed/` generated datasets and drop/review logs
- `blog_figures/` every figure used in the blog post, with the code that makes it
- `DATA_SPEC.md` the data rules. Read before touching the data code.

## Run and test
```
pip install pandas requests pybaseball pytest
python src/build_ids.py            # Chadwick IDs
python src/prepare_contracts.py    # raw Spotrac scrape -> rule-compliant rows
python src/clean_contracts.py      # validated data/processed/contracts.csv
python -m pytest -q tests
```

## Style
Python 3.11+, pandas for data, PyTorch for models. Small functions, type hints on public
functions, no notebooks-only logic (notebooks may call `src/`).

## Rules
- Never edit `data/raw/`. Re-run the scripts instead.
- Split by offseason (train 2014-2021, validate 2022-2023, test 2024-2025). The test classes are
  used once, after every choice is fixed.
- Features may only use seasons completed before the signing date (no leakage).
- Every dropped row must be logged with a reason.
- No API keys or credentials in the repo.
- Follow `.github/` templates: one issue, one PR, under about 300 changed lines, disclose AI use.
