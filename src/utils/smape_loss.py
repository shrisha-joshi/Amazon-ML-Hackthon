# MIT License
# src/utils/smape_loss.py
"""
SMAPE metric and PyTorch loss wrapper.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

def smape_numpy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, eps, denom)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

class SMAPELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # both pred and target are in original price space (not log)
        denom = (pred.abs() + target.abs()) / 2.0
        denom = torch.clamp(denom, min=self.eps)
        loss = torch.abs(pred - target) / denom
        return loss.mean() * 100.0