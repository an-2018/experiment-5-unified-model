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
from scipy import stats
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


def delong_test(y_true, preds_a, preds_b):
    """DeLong test for paired AUROC comparison.
    Returns dict with z_stat, p_value, significant (p<0.05).
    """
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)

    auc_a = roc_auc_score(y_true, preds_a)
    auc_b = roc_auc_score(y_true, preds_b)

    mask_pos = y_true == 1
    mask_neg = y_true == 0

    if sum(mask_pos) == 0 or sum(mask_neg) == 0:
        return {'z_stat': 0, 'p_value': 1.0, 'significant': False,
                'auc_a': auc_a, 'auc_b': auc_b}

    X_pos = preds_a[mask_pos, np.newaxis]
    X_neg = preds_a[mask_neg, np.newaxis]
    Y_pos = preds_b[mask_pos, np.newaxis]
    Y_neg = preds_b[mask_neg, np.newaxis]

    n_pos = len(X_pos)
    n_neg = len(X_neg)

    V10_a = np.var(np.mean(X_pos > X_neg.T, axis=1))
    V01_a = np.var(np.mean(X_pos > X_neg.T, axis=0))
    var_a = V10_a / n_pos + V01_a / n_neg

    V10_b = np.var(np.mean(Y_pos > Y_neg.T, axis=1))
    V01_b = np.var(np.mean(Y_pos > Y_neg.T, axis=0))
    var_b = V10_b / n_pos + V01_b / n_neg

    diff = (X_pos > X_neg.T).astype(float) - (Y_pos > Y_neg.T).astype(float)
    V10_ab = np.var(np.mean(diff, axis=1))
    V01_ab = np.var(np.mean(diff, axis=0))

    var_diff = var_a + var_b - 2 * (V10_ab / n_pos + V01_ab / n_neg)

    z_stat = (auc_a - auc_b) / np.sqrt(var_diff + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        'z_stat': float(z_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'auc_a': float(auc_a),
        'auc_b': float(auc_b)
    }


def paired_bootstrap_ci(y_true, preds_a, preds_b, metric_fn, n_iterations=2000, alpha=0.05):
    """Paired bootstrap confidence interval for difference between two models."""
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    n = len(y_true)

    diffs = []
    for _ in range(n_iterations):
        indices = np.random.choice(n, n, replace=True)
        try:
            score_a = metric_fn(y_true[indices], preds_a[indices])
            score_b = metric_fn(y_true[indices], preds_b[indices])
            diffs.append(score_b - score_a)
        except:
            continue

    if not diffs:
        return {'ci_lower': float('nan'), 'ci_upper': float('nan'), 'mean_diff': 0.0}

    diffs = np.array(diffs)
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return {'ci_lower': float(lower), 'ci_upper': float(upper), 'mean_diff': float(np.mean(diffs))}


def cohens_d(y_true, preds_a, preds_b):
    """Cohen's d effect size for paired comparison."""
    diff = np.asarray(preds_a) - np.asarray(preds_b)
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))


def compute_all_depression_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    probs = 1 / (1 + np.exp(-logits))
    try:
        from training.calibration import compute_brier_score, compute_ece
        brier = float(compute_brier_score(probs, labels))
        ece_val = float(compute_ece(probs, labels))
    except ImportError:
        brier = float(np.mean((probs - labels) ** 2))
        ece_val = 0.0  # Fallback
    return {
        "auroc": float(compute_auroc(labels, probs)),
        "auprc": float(compute_auprc(labels, probs)),
        "f1": float(compute_f1(labels, probs)),
        "mae": float(compute_mae(labels, probs)),
        "sensitivity": float(compute_sensitivity_specificity(labels, probs)[0]),
        "specificity": float(compute_sensitivity_specificity(labels, probs)[1]),
        "brier": brier,
        "ece": ece_val,
    }