"""Tune the network on validation only (issue #32).

Grid over hidden sizes, dropout, weight decay, the length-loss weight and class weighting. Each
setting is trained on 3 seeds per group; the 3 best by mean validation log-AAV MAE are rerun on 5
seeds. Among those, every setting within TIE_TOL of the best mean log-AAV MAE counts as tied (the
seed-to-seed std of that MAE is about 0.007-0.013), and the tied setting with the best balanced length
accuracy is saved per group.
The test classes are never loaded.

Writes results/nn_tuning.csv (every setting, mean over seeds) and results/nn_best_settings.json.
Usage: python src/tune_nn.py [--group hitters|pitchers] [--grid-seeds 3] [--final-seeds 5]
"""
import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
import torch

from nn_data import load_group
from train_nn import run_seeds

ROOT = Path(__file__).parents[1]
GRID = {"hidden": [(32,), (64, 32), (128, 64)], "dropout": [0.1, 0.3], "weight_decay": [1e-4, 1e-3],
        "years_weight": [0.5, 1.0], "weighted": [True, False]}
TIE_TOL = 0.005
KEEP = ["mae_usd_m", "mae_log", "r2_log", "years_bucket_accuracy", "years_balanced_accuracy", "years_mae"]


def score(data, settings: dict, n_seeds: int) -> dict:
    per_seed, _ = run_seeds(data, list(range(n_seeds)), **settings)
    return {**{k: (list(v) if k == "hidden" else v) for k, v in settings.items()}, "n_seeds": n_seeds,
            **{m: float(per_seed[m].mean()) for m in KEEP}, "r2_log_std": float(per_seed["r2_log"].std(ddof=0))}


def tune_group(group: str, grid_seeds: int, final_seeds: int) -> tuple[pd.DataFrame, dict]:
    data = load_group(group)
    rows = [score(data, dict(zip(GRID, values)), grid_seeds) for values in itertools.product(*GRID.values())]
    grid = pd.DataFrame(rows).assign(group=group, stage="grid").sort_values("mae_log")
    finals = [score(data, {k: (tuple(r[k]) if k == "hidden" else r[k]) for k in GRID}, final_seeds)
              for _, r in grid.head(3).iterrows()]
    final = pd.DataFrame(finals).assign(group=group, stage="final")
    return pd.concat([grid, final], ignore_index=True), pick_best(final)


def pick_best(final: pd.DataFrame, tol: float = TIE_TOL) -> dict:
    """Best balanced length accuracy among settings within `tol` of the lowest mean log-AAV MAE."""
    tied = final[final["mae_log"] <= final["mae_log"].min() + tol]
    best = tied.sort_values(["years_balanced_accuracy", "mae_log"], ascending=[False, True]).iloc[0]
    out = {}
    for k in GRID:
        v = best[k]
        out[k] = [int(h) for h in (json.loads(v) if isinstance(v, str) else v)] if k == "hidden" else \
            (bool(v) if k == "weighted" else float(v))
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--group", choices=["hitters", "pitchers"], action="append")
    p.add_argument("--grid-seeds", type=int, default=3)
    p.add_argument("--final-seeds", type=int, default=5)
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "nn_best_settings.json"
    best_all = json.loads(best_path.read_text()) if best_path.exists() else {}
    for group in args.group or ["hitters", "pitchers"]:
        table, best = tune_group(group, args.grid_seeds, args.final_seeds)
        table.to_csv(out / f"nn_tuning_{group}.csv", index=False)
        best_all[group] = best
        best_path.write_text(json.dumps(best_all, indent=2))
        print(group, best)


if __name__ == "__main__":
    main()
