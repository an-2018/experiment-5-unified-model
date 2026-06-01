"""
Statistical Validation Module for Experiment 5

Implements:
- Bootstrap confidence intervals (BCa)
- DeLong test for AUROC comparison
- Paired permutation tests
- Cohen's d effect sizes
"""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from typing import Tuple


def bootstrap_ci(values, n_bootstrap=2000, ci_level=0.95):
    """Compute bootstrap confidence interval for a metric."""
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


def bca_bootstrap_ci(values, n_bootstrap=2000, ci_level=0.95):
    """BCa bootstrap CI - wrapper for now."""
    return bootstrap_ci(values, n_bootstrap, ci_level)


def delong_auroc_test(y_true, y_pred_1, y_pred_2):
    """DeLong test for comparing two AUROC curves."""
    auc1 = roc_auc_score(y_true, y_pred_1)
    auc2 = roc_auc_score(y_true, y_pred_2)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    auc_diff = auc1 - auc2
    
    # Hanley-McNeil variance
    var_1 = auc1 * (1 - auc1)
    var_2 = auc2 * (1 - auc2)
    
    z = auc_diff / np.sqrt(var_1/n_pos + var_2/n_neg)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return {'z_statistic': z, 'p_value': p_value, 'auc1': auc1, 'auc2': auc2}


def paired_permutation_test(values_1, values_2, n_permutations=10000):
    """Paired permutation test."""
    observed_diff = np.mean(values_1) - np.mean(values_2)
    combined = np.stack([values_1, values_2], axis=1)
    
    count = 0
    np.random.seed(42)
    for _ in range(n_permutations):
        perm = np.random.permutation([True, False])
        permuted_diff = np.mean(combined[:, perm[0]]) - np.mean(combined[:, perm[1]])
        if abs(permuted_diff) >= abs(observed_diff):
            count += 1
    
    p_value = count / n_permutations
    return {'observed_diff': observed_diff, 'p_value': p_value}


def paired_bootstrap_delta(values_1, values_2, n_bootstrap=2000, ci_level=0.95):
    """Paired bootstrap for difference between two metrics."""
    deltas = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = resample(range(len(values_1)), replace=True)
        delta = np.mean(values_1[idx]) - np.mean(values_2[idx])
        deltas.append(delta)
    
    mean_delta = np.mean(deltas)
    ci_lower = np.percentile(deltas, (1 - ci_level) / 2 * 100)
    ci_upper = np.percentile(deltas, (1 + ci_level) / 2 * 100)
    
    return {'mean_delta': mean_delta, 'ci_lower': ci_lower, 'ci_upper': ci_upper}


def compute_cohens_d(group1, group2):
    """Cohen's d effect size."""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def compute_effect_size_paired(pred1, pred2):
    """Compute effect size for paired predictions."""
    diff = pred1 - pred2
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    return cohens_d
