"""Post-hoc calibration methods: Temperature Scaling, Platt Scaling, Isotonic Regression.

Must measure Brier scores and ECE. Use on depression binary classification primarily.
"""
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
import numpy as np


class TemperatureScaling(nn.Module):
    """Learn a single temperature parameter T for logits → calibrated probabilities."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


class PlattScaling(nn.Module):
    """Learn affine parameters a, b: calibrated_prob = sigmoid(a * logits + b)."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.a * logits + self.b)


class IsotonicCalibrator:
    """Non-parametric isotonic regression calibration using sklearn."""

    def __init__(self):
        self.ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        self.ir.fit(logits, labels)

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        return self.ir.predict(logits)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if in_bin.sum() > 0:
            bin_acc = labels[in_bin].mean()
            bin_conf = probs[in_bin].mean()
            ece += (in_bin.sum() / len(labels)) * abs(bin_acc - bin_conf)
    return ece


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score for binary predictions."""
    return np.mean((probs - labels) ** 2)