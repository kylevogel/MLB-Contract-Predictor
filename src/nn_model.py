"""Two-output MLP for contract prediction (issue #22).

A shared trunk learns one representation of the player; two heads read it:
  - aav head:   1 output, log nominal AAV (regression, MSE loss)
  - years head: 5 logits, contract length class 1 / 2 / 3 / 4-5 / 6+ (cross-entropy loss)
"""
import torch
from torch import nn


class ContractMLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 32),
                 dropout: float = 0.2, n_buckets: int = 5):
        super().__init__()
        layers, width = [], n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        self.trunk = nn.Sequential(*layers)
        self.aav_head = nn.Linear(width, 1)
        self.years_head = nn.Linear(width, n_buckets)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (log AAV of shape [batch], length-class logits of shape [batch, n_buckets])."""
        z = self.trunk(x)
        return self.aav_head(z).squeeze(-1), self.years_head(z)


def combined_loss(log_aav_pred, logits, log_aav_true, bucket_true, years_weight: float = 1.0):
    """MSE on log AAV plus years_weight x cross-entropy on the length class."""
    mse = nn.functional.mse_loss(log_aav_pred, log_aav_true)
    ce = nn.functional.cross_entropy(logits, bucket_true)
    return mse + years_weight * ce, mse, ce
