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
| Performance features (3 completed seasons of WAR, PA/IP, wRC+/FIP) | Code ready; needs a run on a machine with FanGraphs access (`src/fetch_stats.py`) |
| Train / validation / test split and evaluation metrics | Done (`src/splits.py`, `src/metrics.py`) |
| Linear and boosted-tree baselines | Next |
| Neural network | After baselines |

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

Every rule (what counts as a contract, what is excluded and why) is in [`DATA_SPEC.md`](DATA_SPEC.md).
Every judgment call, who made it and the evidence is in [`DECISIONS.md`](DECISIONS.md), including the
mistakes the AI assistant made and how we caught them. Every dropped row is logged with a reason in
`data/processed/prepare_dropped.csv`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests          # data validation and unit tests
python src/fetch_stats.py          # needs internet access to FanGraphs; writes data/processed/features_*.csv
```

To rebuild the dataset from the raw Spotrac scrape, see the pipeline order in [`AGENTS.md`](AGENTS.md).

## Repo layout

```
src/        pipeline (scrape, clean, features), splits and metrics; models will live here
tests/      pytest tests, including regression tests for data-quality fixes
data/raw/   hand-kept, source-cited correction files (raw Spotrac scrapes are git-ignored)
data/processed/   cleaned dataset, drop/review logs, features
blog_figures/     figures for the write-up
```

## How we work

Issue-driven development with enforced review: every change starts as a GitHub issue (feature,
experiment or chore template), lands through one small pull request that links it, and needs a
teammate's approval before merging. Templates are in `.github/`; agent rules are in [`AGENTS.md`](AGENTS.md).

## AI use

Planning and parts of the data pipeline were done with Claude (Anthropic). AI-assisted commits carry a
`Co-Authored-By` line and each PR states how AI was used. AI mistakes and fixes are logged in `DECISIONS.md`.

## Data sources and credits

Contract terms were collected from [Spotrac](https://www.spotrac.com), checked against MLB.com, MLB Trade
Rumors, Baseball-Reference, FanGraphs and news reports, and joined to player IDs from the
[Chadwick Bureau register](https://github.com/chadwickbureau/register). Performance statistics come from
FanGraphs via [pybaseball](https://github.com/jldbc/pybaseball). The raw Spotrac pages are not in this
repo (personal-use terms); the cleaned dataset is published with the permission of our course instructor.
If a data owner objects, we will remove the file.
