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
- `results/` validation metrics per model, tuning tables, and `model_history.csv`
- `blog_figures/` every figure used in the blog post, with the code that makes it
- `DATA_SPEC.md` the data rules. Read before touching the data code.

## Run and test
```
pip install -r requirements.txt
python -m pytest -q tests

# data pipeline (only needed to rebuild the datasets)
python src/build_ids.py            # Chadwick IDs
python src/prepare_contracts.py    # raw Spotrac scrape -> rule-compliant rows
python src/clean_contracts.py      # validated data/processed/contracts.csv
python src/scrape_bref_stats.py    # Baseball-Reference season tables (needs internet)
python src/build_features.py       # contracts_features_hitters.csv / _pitchers.csv

# models (validation only; never the 2024-2025 test classes until the final run)
python src/linear_model.py                         # -> results/linear_val_metrics.csv
python src/boosted_tree.py                         # -> results/boosted_tree_val_metrics.csv
python src/train_nn.py --group hitters --tuned     # -> results/nn_val_metrics.csv (and --group pitchers)
python src/tune_nn.py                              # network grid search -> results/nn_best_settings.json
```

## Style
Python 3.11+, pandas for data, PyTorch for models. Small functions, type hints on public
functions, no notebooks-only logic (notebooks may call `src/`).

## Rules
- Read `DECISIONS.md` before starting any task. Follow every entry marked Decided, never reverse
  one without the user's approval, and add an entry for any new data or modelling choice.
- Never edit scraped files in `data/raw/`. The `manual_*.csv` files there are hand-kept, source-cited
  corrections; add a row with its evidence and a `DECISIONS.md` entry instead of changing code.
- Split by offseason (train 2014-2021, validate 2022-2023, test 2024-2025). The test classes are
  used once, after every choice is fixed.
- Features may only use seasons completed before the signing date (no leakage).
- Every dropped row must be logged with a reason.
- No API keys or credentials in the repo.
- Follow `.github/` templates: one issue, one PR, under about 300 changed lines, disclose AI use.
- Every model is built twice: once trained only on the hitter data
  (`data/processed/contracts_features_hitters.csv`) and once only on the pitcher data
  (`contracts_features_pitchers.csv`), never on both together (D-09). Report results per group.
- Model versions are added, never replaced, unless the team says otherwise. A change that alters a
  model's results is a new version: keep the old behaviour reachable (a flag, a settings file, or a
  new file), and append one row per group to `results/model_history.csv` with the new version number,
  PR, decision IDs, a one-line description and the D-19 metrics. Never edit or delete existing rows.
