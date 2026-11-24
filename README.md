# MLB Free-Agent Contract Predictor

Can a neural network predict the **length** and **average annual value (AAV)** of an MLB contract
from a player's performance history? This project builds a cleaned dataset of 1,904 post-rookie
MLB contracts (free agents and extensions, offseasons 2014-2025), then compares a two-output
neural network against linear and boosted-tree baselines under a strict time-based split.

Final project for *Artificial Neural Networks and Deep Learning* by Kyle Vogel, Alex Elias and Liam Scanlon.

## Status

| Stage | State |
| --- | --- |
| Contract data collected, cleaned, validated | Done |
| Performance features from Baseball-Reference (3-season averages of WAR plus OPS+/BA/R for hitters, ERA+/FIP/WHIP for pitchers, and award flags) | Done: 862 hitter and 1,031 pitcher rows (11 contracts with no MLB season in the window are dropped and logged) |
| Train / validation / test split and evaluation metrics | Done (`src/splits.py`, `src/metrics.py`) |
| Linear baseline (ridge) | Built (`src/linear_model.py`); D-19 update in progress (#38) |
| Boosted-tree baseline | Built and updated to D-19 (`src/boosted_tree.py`) |
| Two-output neural network | Built, updated to D-19 and tuned on validation (`src/train_nn.py`, `src/tune_nn.py`) |
| Final test-set run (2024-2025, once) | After all choices are fixed |

## The dataset

`data/processed/contracts.csv`: one row per guaranteed major-league contract signed after a player's
rookie deal.

| | |
| --- | --- |
| Rows | 1,904 (863 hitters, 1,041 pitchers) |
| Contract types | 1,586 free agent, 146 arbitration-years extension, 111 veteran extension, 61 pre-arbitration extension |
| Window | Offseasons 2014 to 2025 (a class runs Nov 1 to Oct 31 of the following year) |
| Targets | `years_new` (new years from the signing date) and `aav_nominal` ($M) |
| Also stored | `aav_adjusted` (deferrals discounted), previous-contract features, contract flags |
| Split | Train 2014-2021 (1,192), validation 2022-2023 (346), test 2024-2025 (366, used once) |

## Results so far (validation, 2022-2023)

All three models use the same inputs (D-18), the same split and the same metrics. AAV is fit as a share of
that season's luxury-tax threshold, so salary growth over time doesn't bias the model, and scored on
nominal AAV in $M (D-19). Contract length is predicted as one of five classes: 1, 2, 3, 4-5 or 6+ years.
Balanced accuracy weights each class equally, so a model cannot score well by always guessing 1 year.

| Model (version) | Group | R² (log AAV) | MAE ($M) | Length accuracy | Length balanced accuracy |
| --- | --- | --- | --- | --- | --- |
| Linear (v1, D-18) | Hitters | 0.519 | 4.13 | 0.497 | n/a |
| Boosted trees (v3, D-19) | Hitters | **0.574** | **3.27** | 0.610 | **0.443** |
| Neural network (v4, tuned, 5 seeds) | Hitters | 0.565 | 3.36 | **0.620** | 0.405 |
| Linear (v1, D-18) | Pitchers | 0.455 | 4.20 | 0.535 | n/a |
| Boosted trees (v3, D-19) | Pitchers | 0.498 | 3.57 | 0.562 | **0.377** |
| Neural network (v4, tuned, 5 seeds) | Pitchers | **0.545** | **3.55** | **0.580** | 0.365 |

These are validation numbers, used to make choices, so they are optimistic. The 2024-2025 test classes
have not been touched yet. Every earlier version of every model is kept in
[`results/model_history.csv`](results/model_history.csv), so the improvement from one version to the
next can be traced (for example, pitcher R² for the network went from 0.428 to 0.545 over four versions).

Every rule (what counts as a contract, what is excluded and why) is in [`DATA_SPEC.md`](DATA_SPEC.md).
Every judgment call, who made it and the evidence is in [`DECISIONS.md`](DECISIONS.md), including the
mistakes the AI assistant made and how we caught them. Every dropped row is logged with a reason in
`data/processed/prepare_dropped.csv`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests          # data validation and unit tests
python src/linear_model.py         # ridge baseline       -> results/linear_val_metrics.csv
python src/boosted_tree.py         # boosted trees        -> results/boosted_tree_val_metrics.csv
python src/train_nn.py --group hitters --tuned    # network, 5 seeds -> results/nn_val_metrics.csv
python src/train_nn.py --group pitchers --tuned
```

`python src/tune_nn.py` reruns the network's validation grid search (about 12 minutes on 2 cores).

To rebuild the performance features:

```bash
python src/scrape_bref_stats.py    # downloads Baseball-Reference season tables to data/raw/ (needs internet)
python src/build_features.py       # writes data/processed/contracts_features_hitters.csv and _pitchers.csv
```

The Baseball-Reference season tables are not committed, so the feature files cannot be rebuilt without the scrape step.
`src/fetch_stats.py` is the earlier FanGraphs approach, replaced by the Baseball-Reference features (D-15) and no longer used.
To rebuild the contract dataset from the raw Spotrac scrape, see the pipeline order in [`AGENTS.md`](AGENTS.md).

## Repo layout

```
src/        pipeline (scrape, clean, features), splits, metrics, salary scale, and the three models
tests/      pytest tests, including regression tests for data-quality fixes
data/raw/   hand-kept, source-cited correction files (raw Spotrac scrapes are git-ignored)
data/processed/   cleaned dataset, drop/review logs, feature datasets (hitters, pitchers)
results/          validation metrics per model (shared D-19 format), tuning tables, model_history.csv
blog_figures/     figures for the write-up
```

## How we work

Issue-driven development with enforced review: every change starts as a GitHub issue (feature,
experiment or chore template), lands through one small pull request that links it, and needs a
teammate's approval before merging. Templates are in `.github/`; agent rules are in [`AGENTS.md`](AGENTS.md).

## AI use

Planning, parts of the data pipeline and parts of the model code were done with Claude (Anthropic). AI-assisted commits carry a
`Co-Authored-By` line and each PR states how AI was used. AI mistakes and fixes are logged in `DECISIONS.md`.

## Data sources and credits

Contract terms were collected from [Spotrac](https://www.spotrac.com), checked against MLB.com, MLB Trade
Rumors, Baseball-Reference, FanGraphs and news reports, and joined to player IDs from the
[Chadwick Bureau register](https://github.com/chadwickbureau/register). Performance statistics and awards come from
[Baseball-Reference](https://www.baseball-reference.com) season tables. The raw Spotrac pages are not in this
repo (personal-use terms); the cleaned dataset is published with the permission of our course instructor.
If a data owner objects, we will remove the file.
