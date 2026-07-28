#!/usr/bin/env python3
"""Add a PCA-12 rung to the ladder: an unsupervised, variance-preserving
projection of the same fused representation, fit on train only. If PCA-12
lands near the random-projection control (~0.61) while the construct-aligned
profile sits lower (~0.52), the effect is specifically about construct
alignment, not dimensionality reduction in general.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from e1_e7_profile_gate import (  # noqa: E402
    ARTIFACTS_TABLES, LLM_LEVEL, build_inference_model, checkpoint_sha256,
    load_checkpoint_from_path, load_profile_schema, extract_daic_profiles,
    fit_eval_auroc, bootstrap_ci_auroc,
)

CHECKPOINTS = {
    "original": ARTIFACTS_TABLES / "mmoe_ex_best_original_seed_unknown.pt",
    "seed17": ARTIFACTS_TABLES / "mmoe_ex_best_seed17.pt",
    "seed1337": ARTIFACTS_TABLES / "mmoe_ex_best_seed1337.pt",
    "seed2024": ARTIFACTS_TABLES / "mmoe_ex_best_seed2024.pt",
    "seed31415": ARTIFACTS_TABLES / "mmoe_ex_best_seed31415.pt",
}


def main():
    schema = load_profile_schema()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for name, ckpt_path in CHECKPOINTS.items():
        ckpt_sha = checkpoint_sha256(ckpt_path)
        model = build_inference_model(LLM_LEVEL, device)
        load_checkpoint_from_path(model, ckpt_path, device)
        model.eval()

        train_data = extract_daic_profiles(model, device, "train", schema, ckpt_sha)
        test_data = extract_daic_profiles(model, device, "test", schema, ckpt_sha)
        y_train, y_test = train_data["y"], test_data["y"]

        pca = PCA(n_components=12, random_state=42).fit(train_data["fused"])
        Xtr_pca = pca.transform(train_data["fused"])
        Xte_pca = pca.transform(test_data["fused"])
        explained = pca.explained_variance_ratio_.sum()

        auroc_pca, proba_pca = fit_eval_auroc(Xtr_pca, y_train, Xte_pca, y_test)
        ci_lo, ci_hi = bootstrap_ci_auroc(y_test, proba_pca)
        print(f"{name:12s}  PCA-12 explained_var={explained:.3f}  AUROC={auroc_pca:.4f} "
              f"(95% CI {ci_lo:.4f}-{ci_hi:.4f})")
        results.append({"checkpoint": name, "explained_variance": float(explained),
                        "auroc": float(auroc_pca), "ci": [float(ci_lo), float(ci_hi)]})

    aurocs = np.array([r["auroc"] for r in results])
    print(f"\nPCA-12 mean AUROC across 5 seeds: {aurocs.mean():.4f} +- {aurocs.std():.4f}")

    import json
    with open(REPO_ROOT / "artifacts" / "stats" / "e1_pca_control.json", "w") as f:
        json.dump({"per_seed": results, "mean": float(aurocs.mean()), "std": float(aurocs.std())}, f, indent=2)
    print("Saved: artifacts/stats/e1_pca_control.json")


if __name__ == "__main__":
    main()
