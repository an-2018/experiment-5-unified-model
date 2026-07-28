#!/usr/bin/env python3
"""SPEC-H2-03 no_constant_output check, applied post-hoc to the E1/E7 profile
gate. The ablation ladder reported FI CCC=0.000 (a personality-head constant-
output collapse) on a graph-routed variant; this checks whether the SAME
degeneracy is present in the personality dimensions of the profiles extracted
from the non-graph MMoEEx checkpoints used for the Phase 1 gate. If any of
the 12 profile dimensions has near-zero variance, the "random projection
beats the construct profile" finding could be a trivial consequence of the
profile effectively having fewer than 12 informative dimensions, not a real
construct-supervision effect.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = REPO_ROOT / "artifacts" / "profiles"

CHECKPOINTS = {
    "original": "920479c669e1",
    "seed17": "d0c155bccf8b",
    "seed1337": "756012800725",
    "seed2024": "a0934a1f5ad1",
    "seed31415": "42ea65043def",
}

DIM_NAMES = [
    "sentiment",
    "anger", "disgust", "fear", "happiness", "sadness", "surprise",
    "extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness",
]
AXIS = {
    "sentiment": "valence",
    **{d: "state" for d in ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]},
    **{d: "trait" for d in ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]},
}

NEAR_ZERO_STD_THRESHOLD = 1e-3  # matches SPEC-H2-03's "std(predictions) > 1e-6" spirit,
                                 # loosened slightly since these are post-sigmoid/bounded
                                 # outputs, not raw logits


def main():
    for name, sha in CHECKPOINTS.items():
        print(f"\n{'='*70}\n{name} (checkpoint {sha})\n{'='*70}")
        train_df = pd.read_parquet(PROFILES_DIR / f"daic_train_profiles_{sha}.parquet")
        test_df = pd.read_parquet(PROFILES_DIR / f"daic_test_profiles_{sha}.parquet")
        combined = pd.concat([train_df[DIM_NAMES], test_df[DIM_NAMES]], ignore_index=True)

        print(f"{'dimension':18s} {'axis':10s} {'std':>10s} {'min':>8s} {'max':>8s} {'range':>8s}  flag")
        degenerate = []
        for dim in DIM_NAMES:
            vals = combined[dim].to_numpy()
            std = vals.std()
            vmin, vmax = vals.min(), vals.max()
            rng = vmax - vmin
            flag = "DEGENERATE" if std < NEAR_ZERO_STD_THRESHOLD else ""
            if flag:
                degenerate.append(dim)
            print(f"{dim:18s} {AXIS[dim]:10s} {std:10.5f} {vmin:8.4f} {vmax:8.4f} {rng:8.4f}  {flag}")

        if degenerate:
            print(f"  >>> DEGENERATE DIMENSIONS: {degenerate}")
        else:
            print("  No degenerate (near-constant) dimensions detected.")

        # Pairwise correlation matrix (flag near-1.0 off-diagonal = collapsed/duplicate dims)
        corr = combined.corr()
        print("\n  Pairwise |correlation| > 0.95 (off-diagonal, would indicate collapsed/duplicate dims):")
        found_high_corr = False
        for i, d1 in enumerate(DIM_NAMES):
            for j, d2 in enumerate(DIM_NAMES):
                if j <= i:
                    continue
                c = corr.loc[d1, d2]
                if abs(c) > 0.95:
                    print(f"    {d1} <-> {d2}: r={c:.4f}")
                    found_high_corr = True
        if not found_high_corr:
            print("    (none)")

        # Effective rank via PCA explained variance (informative-dimensionality proxy)
        X = combined.to_numpy()
        Xc = X - X.mean(axis=0, keepdims=True)
        Xs = Xc / (Xc.std(axis=0, keepdims=True) + 1e-12)
        cov = np.cov(Xs, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        eigvals = np.clip(eigvals, 0, None)
        explained = eigvals / eigvals.sum()
        cum = np.cumsum(explained)
        n_for_95 = int(np.searchsorted(cum, 0.95) + 1)
        print(f"\n  PCA eigenvalues (standardized): {np.round(eigvals, 3).tolist()}")
        print(f"  Dimensions needed for 95% variance: {n_for_95} / 12")


if __name__ == "__main__":
    main()
