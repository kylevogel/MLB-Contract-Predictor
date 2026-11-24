"""Train the two-output MLP for one group and score it on validation (issue #22).

Early-stops on validation loss and keeps the best epoch's weights. Writes
  results/nn_<group>_val_predictions.csv
  results/nn_<group>_val_metrics.json   (metrics.report on AAV in $M, plus length accuracy and MAE)
The test classes are never loaded: nn_data.load_group only returns train and validation.

Usage: python src/train_nn.py --group hitters [--seed 0] [--epochs 500] [--patience 30]
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from metrics import report
from nn_data import YEAR_BUCKET_MIDPOINTS, GroupData, load_group
from nn_model import ContractMLP, combined_loss

ROOT = Path(__file__).parents[1]


def _tensors(X, y_aav, y_bucket):
    return (torch.tensor(X, dtype=torch.float32), torch.tensor(y_aav, dtype=torch.float32),
            torch.tensor(y_bucket, dtype=torch.long))


def train(data: GroupData, seed: int = 0, epochs: int = 500, patience: int = 30, batch_size: int = 64,
          lr: float = 1e-3, weight_decay: float = 1e-4, hidden=(64, 32), dropout: float = 0.2,
          years_weight: float = 1.0) -> tuple[ContractMLP, list[dict]]:
    """Fit on train, early-stop on validation loss, return the best model and per-epoch history."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    Xt, at, bt = _tensors(data.X_train, data.y_log_aav_train, data.y_bucket_train)
    Xv, av, bv = _tensors(data.X_val, data.y_log_aav_val, data.y_bucket_val)
    model = ContractMLP(Xt.shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss, best_state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xt), generator=gen).split(batch_size):
            opt.zero_grad()
            loss, _, _ = combined_loss(*model(Xt[idx]), at[idx], bt[idx], years_weight)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss, val_mse, val_ce = combined_loss(*model(Xv), av, bv, years_weight)
        history.append({"epoch": epoch, "val_loss": val_loss.item(), "val_mse": val_mse.item(), "val_ce": val_ce.item()})
        if val_loss.item() < best_loss:
            best_loss, best_state, best_epoch = val_loss.item(), copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, history


def predict(model: ContractMLP, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (AAV in $M, length class index) for each row."""
    model.eval()
    with torch.no_grad():
        log_aav, logits = model(torch.tensor(X, dtype=torch.float32))
    return np.exp(log_aav.numpy()), logits.argmax(dim=1).numpy()


def evaluate(data: GroupData, aav_pred: np.ndarray, bucket_pred: np.ndarray) -> dict:
    """metrics.report on AAV ($M) plus length-class accuracy and MAE in years (class midpoints)."""
    out = report(data.meta_val["aav_nominal"].to_numpy(), aav_pred)
    out["years_bucket_accuracy"] = float((bucket_pred == data.y_bucket_val).mean())
    out["years_mae"] = float(np.abs(YEAR_BUCKET_MIDPOINTS[bucket_pred] - data.meta_val["years_new"].to_numpy()).mean())
    return out


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--group", choices=["hitters", "pitchers"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    args = p.parse_args(argv)

    data = load_group(args.group)
    model, history = train(data, seed=args.seed, epochs=args.epochs, patience=args.patience)
    aav_pred, bucket_pred = predict(model, data.X_val)
    metrics = evaluate(data, aav_pred, bucket_pred)
    metrics.update(group=args.group, seed=args.seed, best_epoch=int(np.argmin([h["val_loss"] for h in history])),
                   epochs_run=len(history), n_train=len(data.X_train), n_val=len(data.X_val))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds = data.meta_val.assign(aav_pred=aav_pred, years_bucket=data.y_bucket_val, years_bucket_pred=bucket_pred)
    preds.to_csv(out / f"nn_{args.group}_val_predictions.csv", index=False)
    (out / f"nn_{args.group}_val_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
