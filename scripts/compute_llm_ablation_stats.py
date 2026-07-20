#!/usr/bin/env python3
"""Compute real, traceable Cohen's d and DeLong-test p-values for each LLM
ablation level (L1-L5) vs. the classical L0 baseline on DAIC AUROC, from the
actual per-sample predictions saved by phase05 (L0) and phase08 (L1-L5).

Replaces two untraceable numbers that were previously hand-typed into the
paper (d=0.170 for L4 vs L0, d=0.152 for L1 vs L0) with numbers a reader can
regenerate from artifacts/predictions/*.npz."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from evaluation.statistics import compute_cohens_d, delong_auroc_test

PRED_DIR = ROOT / "artifacts" / "predictions"
OUT_PATH = ROOT / "artifacts" / "tables" / "llm_ablation_statistics.csv"


def main():
    l0 = np.load(PRED_DIR / "predictions_L0.npz")
    rows = []
    for level in ["L1", "L2", "L3", "L4", "L5"]:
        path = PRED_DIR / f"predictions_{level}.npz"
        if not path.exists():
            print(f"  Skipping {level}: {path} not found")
            continue
        lx = np.load(path)
        d = compute_cohens_d(lx["daic_all_preds"], l0["daic_all_preds"])
        delong = delong_auroc_test(l0["daic_all_labels"], lx["daic_all_preds"], l0["daic_all_preds"])
        rows.append({
            "comparison": f"{level} vs L0",
            "cohens_d": round(float(d), 4),
            "z_statistic": round(float(delong["z_statistic"]), 4),
            "p_value": round(float(delong["p_value"]), 4),
            "auroc_level": round(float(delong["auc1"]), 4),
            "auroc_l0": round(float(delong["auc2"]), 4),
            "significant_at_0.05": bool(delong["p_value"] < 0.05),
        })
        print(f"{level} vs L0: d={d:.4f}, p={delong['p_value']:.4f}, "
              f"AUROC {delong['auc1']:.4f} vs {delong['auc2']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
