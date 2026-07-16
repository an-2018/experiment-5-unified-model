#!/usr/bin/env python3
"""Merge the real per-variant results from run_ggmoe_ablation_real.sh into a
single, honest artifacts/tables/ggmoe_results.csv (replacing the fabricated
file of the same name that was deleted). Run once all of V0-V4 have produced
artifacts/tables/ggmoe_ablation_real/{V0,V1,V2,V3,V4}_results.csv.
"""
import csv
from pathlib import Path

ROOT = Path("/home/anilson/thesis/thesis-experiment-5-unified-model")
RESULTS_DIR = ROOT / "artifacts/tables/ggmoe_ablation_real"
OUT_PATH = ROOT / "artifacts/tables/ggmoe_results.csv"

VARIANT_DESC = {
    "V0": "inductive-k10",
    "V1": "split-local-k10",
    "V2": "transductive-k10",
    "V3": "inductive-k15",
    "V4": "split-local-k15",
}

rows = []
missing = []
for variant in ["V0", "V1", "V2", "V3", "V4"]:
    csv_path = RESULTS_DIR / f"{variant}_results.csv"
    if not csv_path.exists():
        missing.append(variant)
        continue
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        metrics = {r["metric"]: r["value"] for r in reader}
    rows.append({
        "variant": variant,
        "daic_auroc": metrics["daic_auroc"],
        "mosei_sentiment_ccc": metrics["mosei_sentiment_ccc"],
        "mosei_emotion_auc": metrics["mosei_emotion_auc"],
        "fi_avg_ccc": metrics["fi_avg_ccc"],
    })

if missing:
    print(f"WARNING: missing results for {missing} — not writing until all 5 variants complete.")
    raise SystemExit(1)

with open(OUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["variant", "daic_auroc", "mosei_sentiment_ccc",
                                            "mosei_emotion_auc", "fi_avg_ccc"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote real V0-V4 results to {OUT_PATH}:")
for r in rows:
    print(f"  {r['variant']} ({VARIANT_DESC[r['variant']]}): "
          f"DAIC={r['daic_auroc']} MOSEI_CCC={r['mosei_sentiment_ccc']} "
          f"MOSEI_AUC={r['mosei_emotion_auc']} FI_CCC={r['fi_avg_ccc']}")
