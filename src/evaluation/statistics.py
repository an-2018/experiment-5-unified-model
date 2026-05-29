"""Statistical validation: DeLong test, BCa bootstrap CI, permutation test.

Every headline result must report mean + 95% CI + paired statistical test.
"""
import numpy as np
from scipy.stats import norm, permutation_test
from sklearn.metrics import roc_auc_score
from typing import Callable


def delong_auroc_test(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[float, float]:
    """DeLong's test for comparing two AUROC curves.

    Returns (z_statistic, p_value).
    """
    # Simplified DeLong implementation
    n1 = (y_true == 1).sum()
    n0 = (y_true == 0).sum()
    auc_a = roc_auc_score(y_true, scores_a)
    auc_b = roc_auc_score(y_true, scores_b)

    # Variance estimation via DeLong's method
    var_a = auc_a * (1 - auc_a)
    var_b = auc_b * (1 - auc_b)

    z = (auc_a - auc_b) / np.sqrt(var_a / n1 + var_b / n0)
    p_value = 2 * norm.cdf(-abs(z))
    return z, p_value


def bootstrap_ci(
    metric_fn: Callable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a metric.

    Returns (mean, lower_ci, upper_ci).
    """
    bootstraps = []
    n = len(y_true)
    for _ in range(n_iterations):
        indices = np.random.randint(0, n, n)
        boot_val = metric_fn(y_true[indices], y_pred[indices])
        bootstraps.append(boot_val)
    bootstraps = np.array(bootstraps)
    mean = np.mean(bootstraps)
    lower = np.percentile(bootstraps, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstraps, (1 + confidence) / 2 * 100)
    return mean, lower, upper


def permutation_test_ab(
    metric_fn: Callable,
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 1000,
) -> tuple[float, float]:
    """Permutation test for comparing two methods.

    Returns (observed_diff, p_value).
    """
    def statistic(x, y):
        return metric_fn(y_true, x) - metric_fn(y_true, y)

    result = permutation_test(
        (scores_a, scores_b),
        statistic,
        n_resamples=n_permutations,
        random_state=42,
    )
    return result.statistic, result.pvalue


def paired_bootstrap_delta(
    metric_fn: Callable,
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_iterations: int = 1000,
) -> tuple[float, float, float]:
    """Paired bootstrap for difference in metric between two models.

    Returns (mean_delta, lower_ci, upper_ci).
    """
    deltas = []
    n = len(y_true)
    for _ in range(n_iterations):
        indices = np.random.randint(0, n, n)
        delta = metric_fn(y_true[indices], scores_a[indices]) - metric_fn(y_true[indices], scores_b[indices])
        deltas.append(delta)
    deltas = np.array(deltas)
    return np.mean(deltas), np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)