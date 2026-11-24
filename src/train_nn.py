"""Train the two-output MLP for one group and score it on validation (issues #22, #31).

The AAV head fits log(AAV share of the luxury-tax threshold) (D-19). Predicted shares are capped at
the largest training share, then converted back to nominal $M for scoring. The length head uses
cross-entropy weighted by inverse class frequency on train, so the rare 3-year and 6+ classes count.
Each seed early-stops on validation loss and keeps its best epoch's weights.

Writes
  results/nn_<group>_val_predictions.csv   one block of rows per seed
  results/nn_val_metrics.csv               one row per group, D-19 column order; metrics are the
                                           mean over seeds, with _std columns after them
The test classes are never loaded: nn_data.load_group only returns train and validation.

Usage: python src/train_nn.py --group hitters [--seeds 5] [--epochs 500] [--patience 30] [--no-class-weights]
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from metrics import report, years_class_report
from nn_data import YEAR_BUCKET_MIDPOINTS, GroupData, load_group
from nn_model import ContractMLP, combined_loss
from salary_scale import from_share

ROOT = Path(__file__).parents[1]
SHARED_COLUMNS = ["model", "group", "n_train", "n_val", "within_5pct", "within_10pct", "mae_usd_m", "mae_log",
                  "r2_log", "years_bucket_accuracy", "years_balanced_accuracy", "years_mae"]  # DECISIONS.md D-19
STD_COLUMNS = ["mae_usd_m", "mae_log", "r2_log", "years_bucket_accuracy", "years_balanced_accuracy", "years_mae"]


def _tensors(X, y_aav, y_bucket):
    return (torch.tensor(X, dtype=torch.float32), torch.tensor(y_aav, dtype=torch.float32),
            torch.tensor(y_bucket, dtype=torch.long))


def class_weights(buckets: np.ndarray, n_classes: int = 5) -> torch.Tensor:
    """Inverse class frequency on train, normalised to mean 1 over the classes present."""
    counts = np.bincount(buckets, minlength=n_classes).astype(float)
    w = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    return torch.tensor(w / w[counts > 0].mean(), dtype=torch.float32)


def train(data: GroupData, seed: int = 0, epochs: int = 500, patience: int = 30, batch_size: int = 64,
          lr: float = 1e-3, weight_decay: float = 1e-4, hidden=(64, 32), dropout: float = 0.2,
          years_weight: float = 1.0, weighted: bool = True) -> tuple[ContractMLP, list[dict]]:
    """Fit on train, early-stop on validation loss, return the best model and per-epoch history."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    Xt, at, bt = _tensors(data.X_train, data.y_log_share_train, data.y_bucket_train)
    Xv, av, bv = _tensors(data.X_val, data.y_log_share_val, data.y_bucket_val)
    cw = class_weights(data.y_bucket_train) if weighted else None
    model = ContractMLP(Xt.shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss, best_state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xt), generator=gen).split(batch_size):
            opt.zero_grad()
            loss, _, _ = combined_loss(*model(Xt[idx]), at[idx], bt[idx], years_weight, cw)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss, val_mse, val_ce = combined_loss(*model(Xv), av, bv, years_weight, cw)
        history.append({"epoch": epoch, "val_loss": val_loss.item(), "val_mse": val_mse.item(), "val_ce": val_ce.item()})
        if val_loss.item() < best_loss:
            best_loss, best_state, best_epoch = val_loss.item(), copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, history


def predict(model: ContractMLP, X: np.ndarray, offseason, max_log_share: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (AAV in $M, length class index). The share is capped at the training maximum (D-19)."""
    model.eval()
    with torch.no_grad():
        log_share, logits = model(torch.tensor(X, dtype=torch.float32))
    share = np.exp(np.minimum(log_share.numpy(), max_log_share))
    return from_share(share, offseason), logits.argmax(dim=1).numpy()


def predict_val(model: ContractMLP, data: GroupData) -> tuple[np.ndarray, np.ndarray]:
    return predict(model, data.X_val, data.meta_val["offseason"], float(data.y_log_share_train.max()))


def evaluate(data: GroupData, aav_pred: np.ndarray, bucket_pred: np.ndarray) -> dict:
    """metrics.report on AAV ($M), length accuracy and balanced accuracy, and MAE in years (class midpoints)."""
    years = data.meta_val["years_new"].to_numpy()
    return {**report(data.meta_val["aav_nominal"].to_numpy(), aav_pred), **years_class_report(years, bucket_pred),
            "years_mae": float(np.abs(YEAR_BUCKET_MIDPOINTS[bucket_pred] - years).mean())}


def run_seeds(data: GroupData, seeds: list[int], **train_kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train once per seed. Returns (one metrics row per seed, validation predictions with a seed column)."""
    rows, preds = [], []
    for seed in seeds:
        model, history = train(data, seed=seed, **train_kwargs)
        aav, bucket = predict_val(model, data)
        rows.append({**evaluate(data, aav, bucket), "seed": seed,
                     "best_epoch": int(np.argmin([h["val_loss"] for h in history])), "epochs_run": len(history)})
        preds.append(data.meta_val.assign(seed=seed, aav_pred=aav, years_bucket=data.y_bucket_val,
                                          years_bucket_pred=bucket))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarise(group: str, data: GroupData, per_seed: pd.DataFrame) -> dict:
    """One D-19 row: mean over seeds for every metric, then _std columns and run details."""
    row = {"model": "nn", "group": group, "n_train": len(data.X_train), "n_val": len(data.X_val)}
    row.update({c: float(per_seed[c].mean()) for c in SHARED_COLUMNS[4:]})
    row.update({f"{c}_std": float(per_seed[c].std(ddof=0)) for c in STD_COLUMNS})
    row.update(n_seeds=len(per_seed), mean_best_epoch=float(per_seed["best_epoch"].mean()))
    return row


def write_metrics_row(path: Path, row: dict) -> None:
    """Add or replace this group's row in the shared-format metrics CSV."""
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(table) and set(SHARED_COLUMNS) <= set(table.columns):
        table = table[table["group"] != row["group"]]
    else:
        table = pd.DataFrame()  # older format: start over
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True).sort_values("group")
    table.to_csv(path, index=False)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--group", choices=["hitters", "pitchers"], required=True)
    p.add_argument("--seeds", type=int, default=5, help="seeds 0..N-1; metrics are the mean over them")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    args = p.parse_args(argv)

    data = load_group(args.group)
    per_seed, preds = run_seeds(data, list(range(args.seeds)), epochs=args.epochs, patience=args.patience,
                                weighted=not args.no_class_weights)
    row = summarise(args.group, data, per_seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / f"nn_{args.group}_val_predictions.csv", index=False)
    write_metrics_row(out / "nn_val_metrics.csv", row)
    print(json.dumps(row, indent=2))
    return row


if __name__ == "__main__":
    main()
