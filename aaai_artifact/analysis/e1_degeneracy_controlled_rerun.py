#!/usr/bin/env python3
"""Re-run E1 (full-profile decoding vs. random-projection control) after
dropping near-constant profile dimensions, per SPEC-H2-03 no_constant_output.

Motivation: analysis/e1_degeneracy_check.py found that 'happiness' (3/4
checkpoints) and 'fear' (1/4) are near-constant-output, and seed1337 has a
near-duplicate pair (extraversion<->agreeableness, r=0.964). If the original
12-dim "profile beaten by random projection" finding were merely an artifact
of the profile having fewer than 12 real informative dimensions, dropping the
degenerate ones and re-projecting to MATCHED dimensionality should close (or
reverse) the gap. If the gap survives, the degeneracy is not the explanation.

Degenerate dims are identified from TRAIN std only (a standard, label-blind
feature-selection step), never from test.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.e1_e7_profile_gate import (  # noqa: E402
    ARTIFACTS_TABLES,
    bootstrap_ci_auroc,
    build_inference_model,
    checkpoint_sha256,
    extract_daic_profiles,
    fit_eval_auroc,
    load_checkpoint_from_path,
    load_profile_schema,
    LLM_LEVEL,
    SEED,
    N_RANDOM_PROJ_DRAWS,
)

STD_THRESHOLD = 1e-3
CHECKPOINTS = {
    "original": ARTIFACTS_TABLES / "mmoe_ex_best_original_seed_unknown.pt",
    "seed17": ARTIFACTS_TABLES / "mmoe_ex_best_seed17.pt",
    "seed1337": ARTIFACTS_TABLES / "mmoe_ex_best_seed1337.pt",
    "seed2024": ARTIFACTS_TABLES / "mmoe_ex_best_seed2024.pt",
    "seed31415": ARTIFACTS_TABLES / "mmoe_ex_best_seed31415.pt",
}

rng = np.random.default_rng(SEED)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schema = load_profile_schema()
    dim_names = [d["name"] for d in schema["dimensions"]]

    summary_rows = []

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        ckpt_sha = checkpoint_sha256(ckpt_path)
        model = build_inference_model(LLM_LEVEL, device)
        load_checkpoint_from_path(model, ckpt_path, device)
        model.eval()

        train_data = extract_daic_profiles(model, device, "train", schema, ckpt_sha)
        test_data = extract_daic_profiles(model, device, "test", schema, ckpt_sha)

        X_train_full = train_data["profiles"][dim_names].to_numpy()
        X_test_full = test_data["profiles"][dim_names].to_numpy()
        y_train = train_data["y"]
        y_test = test_data["y"]

        # Identify degenerate dims from TRAIN std only (label-blind).
        train_std = X_train_full.std(axis=0)
        degenerate_mask = train_std < STD_THRESHOLD
        degenerate = [d for d, m in zip(dim_names, degenerate_mask) if m]
        keep = [d for d, m in zip(dim_names, degenerate_mask) if not m]
        keep_idx = [dim_names.index(d) for d in keep]
        print(f"  Degenerate dims (train std < {STD_THRESHOLD}): {degenerate or 'none'}")
        print(f"  Non-degenerate dims retained: {len(keep)}/12 -> {keep}")

        X_train_reduced = X_train_full[:, keep_idx]
        X_test_reduced = X_test_full[:, keep_idx]

        # Full 12-dim profile (for reference, matches original run)
        auroc_full12, _ = fit_eval_auroc(X_train_full, y_train, X_test_full, y_test)

        # Degeneracy-controlled profile (only non-degenerate dims)
        auroc_reduced, proba_reduced = fit_eval_auroc(X_train_reduced, y_train, X_test_reduced, y_test)
        ci_lo, ci_hi = bootstrap_ci_auroc(y_test, proba_reduced)

        # Matched-dimensionality random projection of the FUSED embedding
        # (same reduced dimensionality as the non-degenerate profile subset)
        matched_dim = len(keep)
        fused_dim = train_data["fused"].shape[1]
        rp_aurocs = []
        for i in range(N_RANDOM_PROJ_DRAWS):
            W = rng.normal(size=(fused_dim, matched_dim)) / np.sqrt(fused_dim)
            Xtr_rp = train_data["fused"] @ W
            Xte_rp = test_data["fused"] @ W
            try:
                auc_rp, _ = fit_eval_auroc(Xtr_rp, y_train, Xte_rp, y_test, seed=SEED + i)
                rp_aurocs.append(auc_rp)
            except Exception:
                continue
        rp_aurocs = np.array(rp_aurocs)

        print(f"  Full 12-dim profile AUROC (reference):        {auroc_full12:.4f}")
        print(f"  Degeneracy-controlled profile AUROC ({matched_dim}-dim): {auroc_reduced:.4f} "
              f"(95% CI {ci_lo:.4f}-{ci_hi:.4f})")
        print(f"  Matched-dim ({matched_dim}) random projection: mean={rp_aurocs.mean():.4f} "
              f"std={rp_aurocs.std():.4f}")
        print(f"  Delta (controlled profile - matched random projection): "
              f"{auroc_reduced - rp_aurocs.mean():.4f}")

        summary_rows.append({
            "checkpoint": name,
            "n_degenerate": len(degenerate),
            "degenerate_dims": degenerate,
            "n_kept": matched_dim,
            "auroc_full12": auroc_full12,
            "auroc_reduced": auroc_reduced,
            "rp_matched_mean": rp_aurocs.mean(),
            "rp_matched_std": rp_aurocs.std(),
            "delta_reduced_vs_matched_rp": auroc_reduced - rp_aurocs.mean(),
        })

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'checkpoint':10s} {'full12':>8s} {'reduced':>8s} {'matchedRP':>10s} {'delta':>8s}  degenerate_dims")
    for r in summary_rows:
        print(f"{r['checkpoint']:10s} {r['auroc_full12']:8.4f} {r['auroc_reduced']:8.4f} "
              f"{r['rp_matched_mean']:10.4f} {r['delta_reduced_vs_matched_rp']:8.4f}  {r['degenerate_dims']}")

    deltas = np.array([r["delta_reduced_vs_matched_rp"] for r in summary_rows])
    print(f"\nMean delta (controlled profile - matched random projection): "
          f"{deltas.mean():.4f} +- {deltas.std():.4f}  (all negative: {bool(np.all(deltas < 0))})")

    full12 = np.array([r["auroc_full12"] for r in summary_rows])
    reduced = np.array([r["auroc_reduced"] for r in summary_rows])
    rp_matched = np.array([r["rp_matched_mean"] for r in summary_rows])

    import json
    out = {
        "per_seed": summary_rows,
        "aggregate": {
            "n_seeds": len(summary_rows),
            "auroc_full12_mean": float(full12.mean()), "auroc_full12_std": float(full12.std()),
            "auroc_reduced_mean": float(reduced.mean()), "auroc_reduced_std": float(reduced.std()),
            "rp_matched_mean_mean": float(rp_matched.mean()), "rp_matched_mean_std": float(rp_matched.std()),
            "delta_mean": float(deltas.mean()), "delta_std": float(deltas.std()),
            "all_negative": bool(np.all(deltas < 0)),
        },
    }
    with open(REPO_ROOT / "artifacts" / "stats" / "e1_degeneracy_controlled_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: artifacts/stats/e1_degeneracy_controlled_summary.json")


if __name__ == "__main__":
    main()
