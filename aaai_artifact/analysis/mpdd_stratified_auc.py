#!/usr/bin/env python3
"""Stratified (within-track) AUC for the pooled MPDD Young+Elderly arm.

Naively pooling 26 subjects across two tracks into one AUROC lets
between-track differences contribute to the ranking: a between-track
(pos_young, neg_elderly) pair counts toward concordance even though it says
nothing about whether personality discriminates depression *within* a
population. Young 0.667 and Elderly 0.925 average to ~0.796, but the naive
pooled figure was 0.819 — the gap is exactly that spurious between-track
contribution.

Stratified AUC = (sum of within-track concordant pairs) / (sum of
within-track total pairs), computed from ONE model (fit on the pooled train
set, matching the original pooled arm) but scored only against same-track
pairs — removing the between-track contribution while still using a single
joint decoder.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from mpdd_groundtruth_traits_ci import get_traits_and_labels  # noqa: E402
from e1_e7_profile_gate import fit_eval_auroc  # noqa: E402


def stratified_auc(y_true, y_score, strata):
    """AUC computed only over pairs (i, j) with y_true[i]=1, y_true[j]=0,
    and strata[i] == strata[j]. Handles ties with 0.5 credit (standard
    Mann-Whitney AUC convention)."""
    strata = np.asarray(strata)
    total_concordant = 0.0
    total_pairs = 0
    for s in np.unique(strata):
        mask = strata == s
        y_s, score_s = y_true[mask], y_score[mask]
        pos, neg = score_s[y_s == 1], score_s[y_s == 0]
        n_pos, n_neg = len(pos), len(neg)
        if n_pos == 0 or n_neg == 0:
            continue
        # pairwise comparison, small n so no need for a smarter O(n log n) method
        diff = pos[:, None] - neg[None, :]
        concordant = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
        total_concordant += concordant
        total_pairs += n_pos * n_neg
    if total_pairs == 0:
        return float("nan"), 0
    return total_concordant / total_pairs, total_pairs


def bootstrap_stratified_auc_ci(y_true, y_score, strata, n_resamples=2000, seed=42):
    """Bootstrap CI for the stratified AUC, resampling subjects WITHIN each
    stratum (track) separately so every resample preserves the original
    per-track sample sizes."""
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    unique_strata = np.unique(strata)
    strata_indices = {s: np.where(strata == s)[0] for s in unique_strata}

    boot_vals = []
    for _ in range(n_resamples):
        idx = np.concatenate([
            rng.choice(strata_indices[s], size=len(strata_indices[s]), replace=True)
            for s in unique_strata
        ])
        val, n_pairs = stratified_auc(y_true[idx], y_score[idx], strata[idx])
        if n_pairs > 0 and not np.isnan(val):
            boot_vals.append(val)
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    young = get_traits_and_labels("young")
    elderly = get_traits_and_labels("elderly")

    def stack(rows):
        X = np.stack([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        return X, y

    Xtr_y, ytr_y = stack(young["train"])
    Xte_y, yte_y = stack(young["test"])
    Xtr_e, ytr_e = stack(elderly["train"])
    Xte_e, yte_e = stack(elderly["test"])

    Xtr = np.vstack([Xtr_y, Xtr_e])
    ytr = np.concatenate([ytr_y, ytr_e])
    Xte = np.vstack([Xte_y, Xte_e])
    yte = np.concatenate([yte_y, yte_e])
    strata_te = np.array(["young"] * len(yte_y) + ["elderly"] * len(yte_e))

    naive_pooled_auroc, proba = fit_eval_auroc(Xtr, ytr, Xte, yte)
    print(f"Naive pooled AUROC (for reference, NOT the headline): {naive_pooled_auroc:.4f}")

    strat_auc, n_pairs = stratified_auc(yte, proba, strata_te)
    ci_lo, ci_hi = bootstrap_stratified_auc_ci(yte, proba, strata_te)
    print(f"Stratified (within-track) AUC: {strat_auc:.4f}  (95% CI {ci_lo:.4f}-{ci_hi:.4f}), "
          f"n_within_track_pairs={n_pairs}")

    simple_avg = (0.6667 + 0.9250) / 2  # from mpdd_groundtruth_traits_ci.py's per-track numbers
    print(f"\nSimple average of separately-fit per-track AUROCs (Young 0.667, Elderly 0.925): {simple_avg:.4f}")
    print(f"Naive pooled (single model, cross-track pairs included): {naive_pooled_auroc:.4f}")
    print(f"Stratified (single model, within-track pairs only): {strat_auc:.4f}")
    print(f"Between-track contribution removed: {naive_pooled_auroc - strat_auc:.4f}")

    import json
    out = {
        "naive_pooled_auroc": float(naive_pooled_auroc),
        "stratified_auroc": float(strat_auc),
        "stratified_ci_95": [ci_lo, ci_hi],
        "n_within_track_pairs": n_pairs,
        "per_track_simple_average": simple_avg,
        "between_track_contribution": float(naive_pooled_auroc - strat_auc),
    }
    with open(REPO_ROOT / "artifacts" / "stats" / "mpdd_stratified_auc.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: artifacts/stats/mpdd_stratified_auc.json")


if __name__ == "__main__":
    main()
