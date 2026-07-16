"""Statistical validation: DeLong test, BCa bootstrap CI, permutation test, effect size.

Every headline result must report mean + 95% CI + paired statistical test.

History: the metric_fn-based implementations below (bca_bootstrap_ci,
paired_permutation_test, paired_bootstrap_delta, compute_effect_size_paired,
_delong_covariance_matrix) were accidentally replaced by simplified,
incompatible-signature stubs in commit e614804 (2026-06-01), which silently
broke every caller in scripts/phase10_evaluation.py (wrong argument counts /
return types — TypeError at runtime). They are restored here from the
pre-regression version. bootstrap_ci and compute_cohens_d keep their
simpler (array-based, not metric_fn-based) signatures because
scripts/phase10_calibration.py depends on exactly that API.
"""

import numpy as np
from scipy.stats import norm
from sklearn.utils import resample
from typing import Callable


# =========================================================================
# Simple bootstrap CI and Cohen's d (array-based; used by phase10_calibration.py)
# =========================================================================

def bootstrap_ci(values, n_bootstrap=2000, ci_level=0.95):
    """Compute a percentile bootstrap confidence interval for the mean of `values`."""
    np.random.seed(42)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        boot_sample = resample(values, replace=True, n_samples=len(values))
        bootstrap_means.append(np.mean(boot_sample))
    bootstrap_means = np.array(bootstrap_means)
    mean_val = np.mean(values)
    ci_lower = np.percentile(bootstrap_means, (1 - ci_level) / 2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 + ci_level) / 2 * 100)
    return mean_val, ci_lower, ci_upper


def compute_cohens_d(group1, group2):
    """Cohen's d effect size between two independent samples."""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


# =========================================================================
# BCa Bootstrap Confidence Intervals (metric_fn-based; used by phase10_evaluation.py)
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

    V10 = 0.0
    V01 = 0.0

    # V10: variance among positive samples
    if n_pos > 1:
        sum_sq = 0.0
        for i in range(n_pos):
            for j in range(n_pos):
                if i == j:
                    continue
                I_ij = 1.0 if pos_scores[i] > pos_scores[j] else 0.0
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


def delong_auroc_test(y_true, y_pred_1, y_pred_2) -> dict:
    """DeLong's test for comparing two AUROC curves.

    Proper implementation using covariance matrix estimation (Sun & Xu, 2014),
    conservatively assuming independence between the two models' covariance
    terms (no cross-covariance estimation — see note in var_diff below).

    Args:
        y_true: binary labels (0/1)
        y_pred_1: predictions from model 1
        y_pred_2: predictions from model 2

    Returns:
        dict with z_statistic, p_value, auc1, auc2
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_pred_1 = np.asarray(y_pred_1)
    y_pred_2 = np.asarray(y_pred_2)

    auc1 = roc_auc_score(y_true, y_pred_1)
    auc2 = roc_auc_score(y_true, y_pred_2)

    S_1 = _delong_covariance_matrix(y_true, y_pred_1)
    S_2 = _delong_covariance_matrix(y_true, y_pred_2)

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    # Var[AUC] = V10/n_pos + V01/n_neg
    var_1 = S_1[0, 0] / n_pos + S_1[1, 1] / n_neg
    var_2 = S_2[0, 0] / n_pos + S_2[1, 1] / n_neg

    # Conservative: assumes independence between the two models (no cross-covariance term)
    var_diff = var_1 + var_2

    if var_diff <= 0:
        return {"z_statistic": 0.0, "p_value": 1.0, "auc1": float(auc1), "auc2": float(auc2)}

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * norm.cdf(-abs(z))

    return {"z_statistic": float(z), "p_value": float(p_value), "auc1": float(auc1), "auc2": float(auc2)}


# =========================================================================
# Permutation Tests (metric_fn-based; used by phase10_evaluation.py)
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
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

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

def compute_effect_size_paired(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict:
    """Compute comprehensive effect size statistics for paired comparison.

    Returns:
        dict with cohens_d, mean_delta, std_delta, ci_delta_lower, ci_delta_upper
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

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
        "cohens_d": float(cohens_d),
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
    """Paired bootstrap for the difference in a metric between two models.

    Returns (mean_delta, lower_ci, upper_ci).
    """
    rng = np.random.RandomState(42)
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    deltas = []
    n = len(y_true)
    for _ in range(n_iterations):
        indices = rng.randint(0, n, n)
        delta = metric_fn(y_true[indices], scores_a[indices]) - metric_fn(y_true[indices], scores_b[indices])
        deltas.append(delta)
    deltas = np.array(deltas)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
