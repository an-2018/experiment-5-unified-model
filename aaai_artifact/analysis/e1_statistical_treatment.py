#!/usr/bin/env python3
"""Proper inferential treatment for the E1 inversion claim (SPEC-H4-01 style),
separating it from the weaker "profile doesn't decode depression" null claim.

For each of the 5 seeds: DeLong test (profile vs. each of 20 random-projection
draws, on the correlated same-test-set AUROCs) + a paired comparison across
seeds (sign test, paired t-test, 95% CI on the delta) using the
degeneracy-controlled numbers as the primary evidence (tighter, matched
dimensionality) with the naive 12-dim version reported alongside.

DeLong's test implementation: standard O(n log n) algorithm (Sun & Xu 2014;
originally DeLong et al. 1988), covariance of the two correlated AUCs computed
via structural components, no external dependency beyond numpy/scipy.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy import stats

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from e1_e7_profile_gate import (  # noqa: E402
    ARTIFACTS_TABLES, LLM_LEVEL, SEED, N_RANDOM_PROJ_DRAWS,
    build_inference_model, checkpoint_sha256, load_checkpoint_from_path,
    load_profile_schema, extract_daic_profiles, fit_eval_auroc,
)

CHECKPOINTS = {
    "original": ARTIFACTS_TABLES / "mmoe_ex_best_original_seed_unknown.pt",
    "seed17": ARTIFACTS_TABLES / "mmoe_ex_best_seed17.pt",
    "seed1337": ARTIFACTS_TABLES / "mmoe_ex_best_seed1337.pt",
    "seed2024": ARTIFACTS_TABLES / "mmoe_ex_best_seed2024.pt",
    "seed31415": ARTIFACTS_TABLES / "mmoe_ex_best_seed31415.pt",
}
STD_THRESHOLD = 1e-3
rng_global = np.random.default_rng(SEED)


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_list, y_true):
    """Sun & Xu (2014) fast DeLong. preds_list: (k, n) array of k scorers'
    predictions on the same n samples. Returns AUCs (k,) and covariance (k,k)."""
    pos = preds_list[:, y_true == 1]
    neg = preds_list[:, y_true == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds_list.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(np.concatenate([pos[r], neg[r]]))

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:
        sx, sy = sx.reshape(1, 1), sy.reshape(1, 1)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true, proba_a, proba_b):
    """Two-sided p-value for AUC(a) == AUC(b), correlated (paired) DeLong test."""
    preds = np.vstack([proba_a, proba_b])
    aucs, cov = _fast_delong(preds, y_true)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), np.nan
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(p)


def main():
    schema = load_profile_schema()
    dim_names = [d["name"] for d in schema["dimensions"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed = []
    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        ckpt_sha = checkpoint_sha256(ckpt_path)
        model = build_inference_model(LLM_LEVEL, device)
        load_checkpoint_from_path(model, ckpt_path, device)
        model.eval()

        train_data = extract_daic_profiles(model, device, "train", schema, ckpt_sha)
        test_data = extract_daic_profiles(model, device, "test", schema, ckpt_sha)
        y_train, y_test = train_data["y"], test_data["y"]

        X_train_full = train_data["profiles"][dim_names].to_numpy()
        X_test_full = test_data["profiles"][dim_names].to_numpy()

        train_std = X_train_full.std(axis=0)
        keep_idx = [i for i, s in enumerate(train_std) if s >= STD_THRESHOLD]
        X_train_reduced = X_train_full[:, keep_idx]
        X_test_reduced = X_test_full[:, keep_idx]
        matched_dim = len(keep_idx)

        auroc_profile, proba_profile = fit_eval_auroc(X_train_reduced, y_train, X_test_reduced, y_test)

        fused_dim = train_data["fused"].shape[1]
        rp_aurocs, rp_probas, delong_ps = [], [], []
        for i in range(N_RANDOM_PROJ_DRAWS):
            W = rng_global.normal(size=(fused_dim, matched_dim)) / np.sqrt(fused_dim)
            Xtr_rp = train_data["fused"] @ W
            Xte_rp = test_data["fused"] @ W
            auc_rp, proba_rp = fit_eval_auroc(Xtr_rp, y_train, Xte_rp, y_test, seed=SEED + i)
            rp_aurocs.append(auc_rp)
            rp_probas.append(proba_rp)
            _, _, p = delong_test(y_test, proba_profile, proba_rp)
            delong_ps.append(p)

        rp_aurocs = np.array(rp_aurocs)
        delong_ps = np.array(delong_ps)
        n_sig = int(np.sum(delong_ps < 0.05))
        print(f"  matched_dim={matched_dim}  profile_AUROC={auroc_profile:.4f}  "
              f"RP_mean={rp_aurocs.mean():.4f}  delta={auroc_profile - rp_aurocs.mean():.4f}")
        print(f"  DeLong (profile vs each of {N_RANDOM_PROJ_DRAWS} RP draws): "
              f"median p={np.median(delong_ps):.4f}, {n_sig}/{N_RANDOM_PROJ_DRAWS} significant at p<0.05")

        per_seed.append({
            "checkpoint": name, "matched_dim": matched_dim,
            "auroc_profile": auroc_profile, "auroc_rp_mean": float(rp_aurocs.mean()),
            "delta": auroc_profile - rp_aurocs.mean(),
            "delong_median_p": float(np.median(delong_ps)),
            "delong_n_significant": n_sig, "delong_n_draws": N_RANDOM_PROJ_DRAWS,
        })

    deltas = np.array([r["delta"] for r in per_seed])
    n_negative = int(np.sum(deltas < 0))
    n_seeds = len(deltas)

    # Sign test: one-sided binomial, H0: P(delta<0) = 0.5
    sign_p = stats.binomtest(n_negative, n_seeds, p=0.5, alternative="greater").pvalue

    # Paired t-test / CI on the 5 seed-level deltas against 0
    t_res = stats.ttest_1samp(deltas, popmean=0.0)
    ci = t_res.confidence_interval(confidence_level=0.95)

    print(f"\n{'='*70}\nCROSS-SEED SUMMARY ({n_seeds} seeds)\n{'='*70}")
    print(f"Deltas: {deltas.tolist()}")
    print(f"Sign test: {n_negative}/{n_seeds} negative, one-sided binomial p = {sign_p:.4f}")
    print(f"Paired t-test vs 0: t={t_res.statistic:.4f}, p={t_res.pvalue:.4f}")
    print(f"95% CI on mean delta: [{ci.low:.4f}, {ci.high:.4f}], mean={deltas.mean():.4f}")
    print(f"DeLong significance rate across all seeds x draws: "
          f"{sum(r['delong_n_significant'] for r in per_seed)}/{sum(r['delong_n_draws'] for r in per_seed)}")

    import json
    out = {
        "per_seed": per_seed,
        "n_seeds": n_seeds, "n_negative": n_negative,
        "sign_test_p_one_sided": float(sign_p),
        "paired_ttest_t": float(t_res.statistic), "paired_ttest_p": float(t_res.pvalue),
        "delta_mean": float(deltas.mean()), "delta_ci_95": [float(ci.low), float(ci.high)],
    }
    with open(REPO_ROOT / "artifacts" / "stats" / "e1_statistical_treatment.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: artifacts/stats/e1_statistical_treatment.json")


if __name__ == "__main__":
    main()
