#!/usr/bin/env python3
"""Subject-level bootstrap for the E1 inversion — the uncertainty source the
cross-seed paired t-test doesn't cover.

All 5 seeds are evaluated on the SAME 47 DAIC test participants, so the
across-seed CI (artifacts/e1_statistical_treatment_report.md) only answers
"would the delta stay negative under retraining." It says nothing about
whether the result would hold with a different sample of test subjects,
because test-set sampling error is fully shared and invisible across seeds.

This resamples the 47 test participants (2000 bootstrap draws, with
replacement) and, within each resample, computes the delta (degeneracy-
controlled profile AUROC - matched-dim random-projection-ensemble AUROC)
averaged across all 5 seeds using each seed's already-fitted predictions
(no refitting inside the bootstrap loop — only the evaluation set is
resampled, which is the correct way to isolate test-sampling uncertainty
from model-fitting uncertainty).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

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
N_BOOTSTRAP = 2000
rng_global = np.random.default_rng(SEED)


def main():
    schema = load_profile_schema()
    dim_names = [d["name"] for d in schema["dimensions"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed_proba_profile = []
    per_seed_proba_rp_ensemble = []
    y_test_ref = None

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"Extracting fixed predictions for {name}...")
        ckpt_sha = checkpoint_sha256(ckpt_path)
        model = build_inference_model(LLM_LEVEL, device)
        load_checkpoint_from_path(model, ckpt_path, device)
        model.eval()

        train_data = extract_daic_profiles(model, device, "train", schema, ckpt_sha)
        test_data = extract_daic_profiles(model, device, "test", schema, ckpt_sha)
        y_train, y_test = train_data["y"], test_data["y"]
        if y_test_ref is None:
            y_test_ref = y_test
        else:
            assert np.array_equal(y_test, y_test_ref), \
                f"Test labels differ across seeds for {name} — sample order not consistent!"

        X_train_full = train_data["profiles"][dim_names].to_numpy()
        X_test_full = test_data["profiles"][dim_names].to_numpy()
        train_std = X_train_full.std(axis=0)
        keep_idx = [i for i, s in enumerate(train_std) if s >= STD_THRESHOLD]
        X_train_reduced, X_test_reduced = X_train_full[:, keep_idx], X_test_full[:, keep_idx]
        matched_dim = len(keep_idx)

        _, proba_profile = fit_eval_auroc(X_train_reduced, y_train, X_test_reduced, y_test)
        per_seed_proba_profile.append(proba_profile)

        fused_dim = train_data["fused"].shape[1]
        rp_probas = []
        for i in range(N_RANDOM_PROJ_DRAWS):
            W = rng_global.normal(size=(fused_dim, matched_dim)) / np.sqrt(fused_dim)
            Xtr_rp = train_data["fused"] @ W
            Xte_rp = test_data["fused"] @ W
            _, proba_rp = fit_eval_auroc(Xtr_rp, y_train, Xte_rp, y_test, seed=SEED + i)
            rp_probas.append(proba_rp)
        per_seed_proba_rp_ensemble.append(np.mean(rp_probas, axis=0))

    y_test = y_test_ref
    n_test = len(y_test)
    n_seeds = len(CHECKPOINTS)
    print(f"\nFixed predictions ready for {n_seeds} seeds, n_test={n_test}. Running "
          f"{N_BOOTSTRAP}-resample subject-level bootstrap...")

    boot_deltas = []
    rng_boot = np.random.default_rng(SEED)
    n_skipped = 0
    for b in range(N_BOOTSTRAP):
        idx = rng_boot.integers(0, n_test, n_test)
        yb = y_test[idx]
        if len(np.unique(yb)) < 2:
            n_skipped += 1
            continue
        seed_deltas = []
        for s in range(n_seeds):
            auc_profile = roc_auc_score(yb, per_seed_proba_profile[s][idx])
            auc_rp = roc_auc_score(yb, per_seed_proba_rp_ensemble[s][idx])
            seed_deltas.append(auc_profile - auc_rp)
        boot_deltas.append(np.mean(seed_deltas))

    boot_deltas = np.array(boot_deltas)
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    frac_negative = float(np.mean(boot_deltas < 0))

    print(f"\n{'='*70}\nSUBJECT-LEVEL BOOTSTRAP RESULT ({len(boot_deltas)} valid resamples, "
          f"{n_skipped} skipped for single-class)\n{'='*70}")
    print(f"Mean delta (profile - RP ensemble, averaged across {n_seeds} seeds): {boot_deltas.mean():.4f}")
    print(f"95% CI (subject-resampled): [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Fraction of bootstrap resamples with delta < 0: {frac_negative:.4f}")
    print(f"Entirely below zero: {bool(ci_hi < 0)}")

    import json
    out = {
        "n_bootstrap": N_BOOTSTRAP, "n_valid": int(len(boot_deltas)), "n_skipped": n_skipped,
        "mean_delta": float(boot_deltas.mean()),
        "ci_95_subject_resampled": [float(ci_lo), float(ci_hi)],
        "fraction_negative": frac_negative,
        "entirely_below_zero": bool(ci_hi < 0),
    }
    with open(REPO_ROOT / "artifacts" / "stats" / "e1_subject_bootstrap.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: artifacts/stats/e1_subject_bootstrap.json")


if __name__ == "__main__":
    main()
