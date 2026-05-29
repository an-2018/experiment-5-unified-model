"""Metrics: AUROC, AUPRC, F1, MAE, CCC, sensitivity, specificity.

Use DeLong's test for AUROC comparisons. Use bootstrap for F1/CCC/MAE.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, mean_absolute_error,
    roc_curve, precision_recall_curve,
)
from scipy.stats import pearsonr, spearmanr
import torch


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return roc_auc_score(y_true, y_score)


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return average_precision_score(y_true, y_score)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    y_binary = (y_pred >= threshold).astype(int)
    return f1_score(y_true, y_binary)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)


def compute_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance Correlation Coefficient."""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)


def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return pearsonr(y_true, y_pred)[0]


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return spearmanr(y_true, y_pred)[0]


def compute_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    y_binary = (y_pred >= threshold).astype(int)
    tp = ((y_binary == 1) & (y_true == 1)).sum()
    tn = ((y_binary == 0) & (y_true == 0)).sum()
    fp = ((y_binary == 1) & (y_true == 0)).sum()
    fn = ((y_binary == 0) & (y_true == 1)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens, spec


def compute_all_depression_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    probs = 1 / (1 + np.exp(-logits))
    return {
        "auroc": compute_auroc(labels, probs),
        "auprc": compute_auprc(labels, probs),
        "f1": compute_f1(labels, probs),
        "mae": compute_mae(labels, probs),
        "sensitivity": compute_sensitivity_specificity(labels, probs)[0],
        "specificity": compute_sensitivity_specificity(labels, probs)[1],
        "brier": compute_brier_score(probs, labels),
        "ece": compute_ece(probs, labels),
    }