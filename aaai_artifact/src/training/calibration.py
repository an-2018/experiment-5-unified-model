"""Post-hoc calibration methods: Temperature Scaling, Platt Scaling, Isotonic Regression.

Must measure Brier scores and ECE. Use on depression binary classification primarily.
Includes training loops for optimizing calibration parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def fit_calibrate(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit and return calibrated probabilities in one call."""
        self.fit(logits, labels)
        return self.calibrate(logits)


def train_temperature_scaling(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.01,
    verbose: bool = False,
) -> TemperatureScaling:
    """Train temperature scaling via NLL optimization on a validation set.

    Args:
        logits: [n] tensor of raw logits from the model
        labels: [n] tensor of binary labels (0/1)
        max_iter: maximum optimization iterations
        lr: learning rate for SGD
        verbose: print loss during training

    Returns:
        trained TemperatureScaling module
    """
    ts = TemperatureScaling()
    optimizer = torch.optim.LBFGS(
        ts.parameters(), lr=lr, max_iter=max_iter,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        calibrated = ts(logits)
        loss = F.binary_cross_entropy_with_logits(calibrated, labels.float())
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        final_loss = F.binary_cross_entropy_with_logits(
            ts(logits), labels.float()
        ).item()

    if verbose:
        print(f"  Temperature scaling: T={ts.temperature.item():.4f}, "
              f"NLL={final_loss:.4f}")

    return ts


def train_platt_scaling(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.1,
    verbose: bool = False,
) -> PlattScaling:
    """Train Platt scaling (logistic calibration) via NLL optimization.

    Args:
        logits: [n] tensor of raw logits
        labels: [n] tensor of binary labels (0/1)
        max_iter: maximum optimization iterations
        lr: learning rate
        verbose: print loss during training

    Returns:
        trained PlattScaling module
    """
    ps = PlattScaling()
    optimizer = torch.optim.LBFGS(
        ps.parameters(), lr=lr, max_iter=max_iter,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        probs = ps(logits)
        loss = F.binary_cross_entropy(probs, labels.float())
        loss.backward()
        return loss

    optimizer.step(closure)

    if verbose:
        print(f"  Platt scaling: a={ps.a.item():.4f}, b={ps.b.item():.4f}")

    return ps


def calibrate_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    method: str = "temperature",
    val_logits: np.ndarray = None,
    val_labels: np.ndarray = None,
) -> tuple[np.ndarray, dict]:
    """Calibrate logits using specified method and return calibrated probabilities.

    Args:
        logits: raw logits to calibrate
        labels: ground truth labels (for fitting)
        method: "temperature", "platt", "isotonic", or "none"
        val_logits: optional held-out logits for fitting (separate from test)
        val_labels: optional held-out labels for fitting

    Returns:
        (calibrated_probs, calibration_info_dict)
    """
    fit_logits = val_logits if val_logits is not None else logits
    fit_labels = val_labels if val_labels is not None else labels

    info = {"method": method}

    if method == "none":
        probs = 1.0 / (1.0 + np.exp(-logits))
        info["params"] = {}
        return probs, info

    if method == "temperature":
        ts = train_temperature_scaling(
            torch.from_numpy(fit_logits).float(),
            torch.from_numpy(fit_labels).float(),
            verbose=False,
        )
        with torch.no_grad():
            calibrated_logits = ts(torch.from_numpy(logits).float()).numpy()
        probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
        info["params"] = {"temperature": float(ts.temperature.item())}

    elif method == "platt":
        ps = train_platt_scaling(
            torch.from_numpy(fit_logits).float(),
            torch.from_numpy(fit_labels).float(),
            verbose=False,
        )
        with torch.no_grad():
            probs = ps(torch.from_numpy(logits).float()).numpy()
        info["params"] = {"a": float(ps.a.item()), "b": float(ps.b.item())}

    elif method == "isotonic":
        cal = IsotonicCalibrator()
        cal.fit(fit_logits, fit_labels)
        probs = cal.calibrate(logits)
        info["params"] = {}

    else:
        raise ValueError(f"Unknown calibration method: {method}")

    # Compute calibration metrics on the calibrated outputs
    # (these functions are defined above in this module)
    info["brier"] = float(compute_brier_score(probs, labels))
    info["ece"] = float(compute_ece(probs, labels))

    return probs, info


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
    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score for binary predictions."""
    return float(np.mean((probs - labels) ** 2))
