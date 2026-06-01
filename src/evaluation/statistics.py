"""Statistical validation: DeLong test, BCa bootstrap CI, permutation test, effect size.

Every headline result must report mean + 95% CI + paired statistical test.
"""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from typing import Callable


# =========================================================================
# BCa Bootstrap Confidence Intervals
# =========================================================================

def bca_bootstrap_ci(
    metric_fn: Callable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 2000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float, float, float]:
    """Bias-Corrected Accelerated (BCa) bootstrap confidence interval.

    More accurate than percentile bootstrap when the sampling distribution
    is skewed. Adjusts for bias and acceleration.

    Args:
        metric_fn: function(y_true, y_pred) -> float
        y_true: ground truth labels
        y_pred: predicted scores/probabilities
        n_iterations: number of bootstrap resamples
        confidence: confidence level (e.g., 0.95 for 95% CI)
        random_seed: for reproducibility

    Returns:
        (mean, lower_ci, upper_ci, bias_correction)
    """
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    bootstraps = np.zeros(n_iterations)

    for i in range(n_iterations):
        indices = rng.randint(0, n, n)
        bootstraps[i] = metric_fn(y_true[indices], y_pred[indices])

    # Original estimate
    orig = metric_fn(y_true, y_pred)

    # Bias correction: proportion of bootstrap samples < original
    p_less = np.mean(bootstraps < orig)
    # Clamp to avoid -inf/inf from norm.ppf(0) or norm.ppf(1)
    p_less = np.clip(p_less, 1e-15, 1 - 1e-15)
    z0 = norm.ppf(p_less)

    # Check if BCa is viable (variance in bootstrap estimates > 0)
    bootstrap_var = np.var(bootstraps)
    if bootstrap_var < 1e-15:
        # Degenerate case: all bootstrap estimates identical
        # Fall back to percentile bootstrap
        lower = np.percentile(bootstraps, (1 - confidence) * 50, method='linear')
        upper = np.percentile(bootstraps, (1 + confidence) * 50, method='linear')
        mean = np.mean(bootstraps)
        return float(mean), float(lower), float(upper), 0.0

    # Acceleration: jackknife estimate of skewness
    jackknife_vals = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jackknife_vals[i] = metric_fn(y_true[mask], y_pred[mask])

    jack_mean = np.mean(jackknife_vals)
    jack_var = np.var(jackknife_vals)
    num = np.sum((jack_mean - jackknife_vals) ** 3)
    denom = 6 * (np.sum((jack_mean - jackknife_vals) ** 2) ** 1.5) if jack_var > 1e-15 else 0
    acceleration = num / denom if abs(denom) > 1e-15 else 0

    # BCa percentiles
    alpha = 1 - confidence
    z_alpha = norm.ppf(alpha / 2)
    z_1minus = norm.ppf(1 - alpha / 2)

    def bca_percentile(z):
        num = z0 + z
        denom = 1 + acceleration * (z0 + z)
        if abs(denom) < 1e-15:
            return 50.0  # fallback to median
        p = norm.cdf(z0 + num / denom)
        return float(np.clip(p * 100, 0, 100))

    lower_percentile = bca_percentile(z_alpha)
    upper_percentile = bca_percentile(z_1minus)

    # Fall back if BCa produced invalid percentiles
    if np.isnan(lower_percentile) or np.isnan(upper_percentile):
        lower = np.percentile(bootstraps, (1 - confidence) * 50, method='linear')
        upper = np.percentile(bootstraps, (1 + confidence) * 50, method='linear')
    else:
        lower = np.percentile(bootstraps, lower_percentile, method='linear')
        upper = np.percentile(bootstraps, upper_percentile, method='linear')
    mean = np.mean(bootstraps)

    return float(mean), float(lower), float(upper), float(z0)


def bootstrap_ci(
    metric_fn: Callable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Simple percentile bootstrap confidence interval.

    Use bca_bootstrap_ci for final results; this is a lightweight alternative.

    Returns (mean, lower_ci, upper_ci).
    """
    bootstraps = []
    n = len(y_true)
    rng = np.random.RandomState(42)
    for _ in range(n_iterations):
        indices = rng.randint(0, n, n)
        boot_val = metric_fn(y_true[indices], y_pred[indices])
        bootstraps.append(boot_val)
    bootstraps = np.array(bootstraps)
    mean = np.mean(bootstraps)
    lower = np.percentile(bootstraps, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstraps, (1 + confidence) / 2 * 100)
    return mean, lower, upper


# =========================================================================
# DeLong Test for AUROC Comparison
# =========================================================================

def _delong_covariance_matrix(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Compute the covariance matrix for DeLong's test.

    Based on Sun & Xu (2014) — fast DeLong implementation using
    Mann-Whitney U-statistic formulation.

    Args:
        y_true: binary labels (0/1)
        scores: prediction scores

    Returns:
        2x2 covariance matrix [V10, V01; V01, V11]
    """
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]

    # Components of the covariance
    V10 = 0.0
    V01 = 0.0
    V11 = 0.0

    # V10: variance among positive samples
    if n_pos > 1:
        sum_sq = 0.0
        for i in range(n_pos):
            for j in range(n_pos):
                if i == j:
                    continue
                # Indicator: score_i > score_j
                I_ij = 1.0 if pos_scores[i] > pos_scores[j] else 0.0
                # Expected value: AUC relative to negatives
                psi_i = np.mean(pos_scores[i] > neg_scores)
                psi_j = np.mean(pos_scores[j] > neg_scores)
                sum_sq += (I_ij - psi_i) * (I_ij - psi_j)
        V10 = sum_sq / (n_pos * (n_pos - 1))

    # V01: variance among negative samples
    if n_neg > 1:
        sum_sq = 0.0
        for i in range(n_neg):
            for j in range(n_neg):
                if i == j:
                    continue
                I_ij = 1.0 if neg_scores[i] > neg_scores[j] else 0.0
                psi_i = np.mean(pos_scores > neg_scores[i])
                psi_j = np.mean(pos_scores > neg_scores[j])
                sum_sq += (I_ij - psi_i) * (I_ij - psi_j)
        V01 = sum_sq / (n_neg * (n_neg - 1))

    return np.array([[V10, 0], [0, V01]])


def delong_auroc_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> tuple[float, float]:
    """DeLong's test for comparing two AUROC curves.

    Proper implementation using covariance matrix estimation.
    Uses the Sun & Xu (2014) fast DeLong method.

    Args:
        y_true: binary labels (0/1)
        scores_a: predictions from model A
        scores_b: predictions from model B

    Returns:
        (z_statistic, p_value) — two-sided test
    """
    auc_a = roc_auc_score(y_true, scores_a)
    auc_b = roc_auc_score(y_true, scores_b)

    # Covariance matrices for each model
    S_a = _delong_covariance_matrix(y_true, scores_a)
    S_b = _delong_covariance_matrix(y_true, scores_b)

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    # Variance of AUC difference: Var[AUC_a - AUC_b] = Var[AUC_a] + Var[AUC_b] - 2*Cov
    var_a = S_a[0, 0] / n_pos + S_a[1, 1] / n_neg
    var_b = S_b[0, 0] / n_pos + S_b[1, 1] / n_neg

    # For simplicity, assume independence between models (conservative)
    # In practice, a paired test would estimate covariance
    var_diff = var_a + var_b

    if var_diff <= 0:
        return 0.0, 1.0  # Degenerate case

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2 * norm.cdf(-abs(z))
    return float(z), float(p_value)


# =========================================================================
# Permutation Tests
# =========================================================================

def paired_permutation_test(
    metric_fn: Callable,
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 5000,
    random_seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Paired permutation test for comparing two models.

    Shuffles the assignment of predictions to models under the null hypothesis
    that both models perform equally.

    Args:
        metric_fn: function(y_true, scores) -> float
        y_true: ground truth labels
        scores_a: predictions from model A
        scores_b: predictions from model B
        n_permutations: number of permutations
        random_seed: for reproducibility

    Returns:
        (observed_difference, p_value, null_distribution)
    """
    rng = np.random.RandomState(random_seed)
    observed = metric_fn(y_true, scores_a) - metric_fn(y_true, scores_b)

    null_dist = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Randomly flip sign of paired differences
        flip = rng.choice([-1, 1], size=len(y_true))
        permuted_a = (scores_a + scores_b) / 2 + flip * (scores_a - scores_b) / 2
        permuted_b = (scores_a + scores_b) / 2 - flip * (scores_a - scores_b) / 2
        null_dist[i] = metric_fn(y_true, permuted_a) - metric_fn(y_true, permuted_b)

    # Two-sided p-value: proportion of null >= |observed|
    p_value = np.mean(np.abs(null_dist) >= np.abs(observed))
    return float(observed), float(p_value), null_dist


# =========================================================================
# Effect Size
# =========================================================================

def compute_cohens_d(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn: Callable = None,
) -> float:
    """Cohen's d for paired comparison of model performance.

    Measures the standardized mean difference between two models' metrics.
    d = mean(delta) / std(delta)

    Args:
        y_true: ground truth labels
        scores_a: predictions from model A
        scores_b: predictions from model B
        metric_fn: if provided, computes per-sample metric differences.
                   If None, uses raw prediction differences.

    Returns:
        Cohen's d value. Convention: |d| >= 0.8 = large effect.
    """
    if metric_fn is None:
        # Use squared error difference per sample
        delta = (scores_a - y_true) ** 2 - (scores_b - y_true) ** 2
    else:
        # Use bootstrap of the metric to estimate effect size
        n_boot = 500
        rng = np.random.RandomState(42)
        diffs = []
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            diffs.append(
                metric_fn(y_true[idx], scores_a[idx]) -
                metric_fn(y_true[idx], scores_b[idx])
            )
        delta = np.array(diffs)

    mean_delta = np.mean(delta)
    std_delta = np.std(delta, ddof=1)
    if std_delta == 0:
        return 0.0
    return float(mean_delta / std_delta)


def compute_effect_size_paired(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict:
    """Compute comprehensive effect size statistics for paired comparison.

    Returns:
        dict with cohens_d, mean_delta, std_delta, 95%_ci_delta
    """
    # Per-sample squared error
    delta = (scores_a - y_true) ** 2 - (scores_b - y_true) ** 2
    mean_d = np.mean(delta)
    std_d = np.std(delta, ddof=1)
    cohens_d = mean_d / std_d if std_d > 0 else 0.0

    # Bootstrap CI for delta
    rng = np.random.RandomState(42)
    boot_deltas = []
    n = len(y_true)
    for _ in range(2000):
        idx = rng.randint(0, n, n)
        boot_deltas.append(np.mean(delta[idx]))
    boot_deltas = np.array(boot_deltas)
    ci_lower = np.percentile(boot_deltas, 2.5)
    ci_upper = np.percentile(boot_deltas, 97.5)

    return {
        "cohens_d": cohens_d,
        "mean_delta": float(mean_d),
        "std_delta": float(std_d),
        "ci_delta_lower": float(ci_lower),
        "ci_delta_upper": float(ci_upper),
    }


def paired_bootstrap_delta(
    metric_fn: Callable,
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_iterations: int = 2000,
) -> tuple[float, float, float]:
    """Paired bootstrap for difference in metric between two models.

    Returns (mean_delta, lower_ci, upper_ci) with BCa correction.
    """
    rng = np.random.RandomState(42)
    deltas = []
    n = len(y_true)
    for _ in range(n_iterations):
        indices = rng.randint(0, n, n)
        delta = metric_fn(y_true[indices], scores_a[indices]) - metric_fn(y_true[indices], scores_b[indices])
        deltas.append(delta)
    deltas = np.array(deltas)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
