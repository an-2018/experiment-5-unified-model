#!/usr/bin/env python3
"""Proper statistical treatment for the 5-seed routing Table 1 comparison
(SPEC-H4-03 reportability rule): a delta may be described directionally only
if |delta| > max(CI_halfwidth_baseline, 2*seed_std_pooled). Otherwise render
as "indistinguishable at this sample size."
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent

METRICS = ["daic_auroc", "mosei_sentiment_ccc", "mosei_emotion_auc", "fi_avg_ccc"]
VARIANTS = ["V0", "V1", "V2", "V3", "V4"]


def main():
    data = json.load(open(REPO_ROOT / "artifacts" / "stats" / "routing_table1_5seed.json"))
    baseline = data["baseline"]
    variants = data["variants"]

    results = {}
    print(f"{'Variant':6s} {'Metric':22s} {'Delta':>9s} {'PairedT p':>10s} {'Reportable?':>12s}")
    for v in VARIANTS:
        results[v] = {}
        for m in METRICS:
            b_vals = np.array(baseline[m]["values"])
            v_vals = np.array(variants[v][m]["values"])
            delta_vals = v_vals - b_vals  # paired per-seed (same 5 seeds)
            t_res = stats.ttest_1samp(delta_vals, popmean=0.0)

            b_std = baseline[m]["std"]
            v_std = variants[v][m]["std"]
            pooled_std = np.sqrt((b_std**2 + v_std**2) / 2)
            mean_delta = float(delta_vals.mean())

            # CI half-width on baseline (from its own 5-seed std, t-based)
            b_ci = baseline[m]["ci_95"]
            b_ci_halfwidth = (b_ci[1] - b_ci[0]) / 2

            threshold = max(b_ci_halfwidth, 2 * pooled_std)
            reportable = abs(mean_delta) > threshold

            results[v][m] = {
                "mean_delta": mean_delta, "paired_t": float(t_res.statistic),
                "paired_p": float(t_res.pvalue), "threshold": float(threshold),
                "reportable": bool(reportable),
            }
            flag = "YES" if reportable else "no (noise)"
            print(f"{v:6s} {m:22s} {mean_delta:+9.4f} {t_res.pvalue:10.4f} {flag:>12s}")

    with open(REPO_ROOT / "artifacts" / "stats" / "routing_table1_statistics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: artifacts/stats/routing_table1_statistics.json")

    # Summary per metric: how many variants show a reportable, consistent-direction effect
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for m in METRICS:
        reportable_variants = [v for v in VARIANTS if results[v][m]["reportable"]]
        pos = [v for v in reportable_variants if results[v][m]["mean_delta"] > 0]
        neg = [v for v in reportable_variants if results[v][m]["mean_delta"] < 0]
        print(f"{m}: {len(reportable_variants)}/5 variants reportable "
              f"({len(pos)} positive: {pos}, {len(neg)} negative: {neg})")


if __name__ == "__main__":
    main()
