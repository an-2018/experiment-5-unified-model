#!/usr/bin/env python3
"""Aggregate the 5-seed K-sensitivity sweep (K=5, K=20; K=10 and K=15 are
already covered by V0/V3 in routing_table1_5seed.json) and the 5-seed learned
per-task lambda run (V0learned), applying the same SPEC-H4-03 reportability
rule used for Table 1: a delta is described directionally only if
|delta| > max(CI_halfwidth_baseline, 2*seed_std_pooled).

Both K5/K20 and V0learned are compared against the same non-graph MMoEEx
baseline already aggregated in routing_table1_5seed.json, so the comparison
is apples-to-apples with the main Table 1 result.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES = REPO_ROOT / "artifacts" / "tables"
STATS = REPO_ROOT / "artifacts" / "stats"

SEEDS = [17, 42, 1337, 2024, 31415]
METRICS = ["daic_auroc", "mosei_sentiment_ccc", "mosei_emotion_auc", "fi_avg_ccc"]
CONFIGS = {
    "K5": "K5",
    "K20": "K20",
    "V0learned": "V0learned",
}


def read_csv_metrics(path: Path) -> dict:
    vals = {}
    with open(path) as f:
        next(f)
        for line in f:
            k, v = line.strip().split(",")
            try:
                vals[k] = float(v)
            except ValueError:
                pass
    return vals


def bca_ci(values: np.ndarray, n_resamples=2000, seed=42) -> tuple[float, float]:
    if len(values) < 2 or np.std(values) == 0:
        return float(values.mean()), float(values.mean())
    res = stats.bootstrap((values,), np.mean, n_resamples=n_resamples,
                           method="BCa", random_state=seed)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def main():
    missing = []
    config_files = {}
    for label, tag in CONFIGS.items():
        for s in SEEDS:
            p = TABLES / f"phase07_results_{tag}_seed{s}.csv"
            config_files[(label, s)] = p
            if not p.exists():
                missing.append(str(p))

    if missing:
        print("MISSING FILES, cannot aggregate yet:")
        for m in missing:
            print(" ", m)
        return 1

    baseline_data = json.load(open(STATS / "routing_table1_5seed.json"))
    baseline = baseline_data["baseline"]
    v0 = baseline_data["variants"]["V0"]

    config_metrics = {(label, s): read_csv_metrics(p) for (label, s), p in config_files.items()}

    agg = {"seeds": SEEDS, "configs": {}}
    print(f"{'':12s}" + "".join(f"{m:>20s}" for m in METRICS))
    for label in CONFIGS:
        row = {}
        for m in METRICS:
            vals = np.array([config_metrics[(label, s)][m] for s in SEEDS])
            lo, hi = bca_ci(vals)
            row[m] = {"mean": float(vals.mean()), "std": float(vals.std()),
                      "values": vals.tolist(), "ci_95": [lo, hi]}
        agg["configs"][label] = row
        print(f"{label:12s}" +
              "".join(f"{row[m]['mean']:.4f}+-{row[m]['std']:.4f}   "[:20].rjust(20) for m in METRICS))

    with open(STATS / "ksweep_learned_lambda_5seed.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved: {STATS / 'ksweep_learned_lambda_5seed.json'}")

    # Reportability: K5/K20 vs non-graph baseline (matching Table 1's comparison point)
    # V0learned vs non-graph baseline AND vs V0 (fixed lambda=0.5) specifically
    stat_results = {}
    print(f"\n{'Config':12s} {'Comparator':16s} {'Metric':22s} {'Delta':>9s} {'PairedT p':>10s} {'Reportable?':>12s}")
    for label in CONFIGS:
        stat_results[label] = {}
        comparators = {"non_graph_baseline": baseline}
        if label == "V0learned":
            comparators["V0_fixed_lambda"] = v0
        for comp_name, comp in comparators.items():
            stat_results[label][comp_name] = {}
            for m in METRICS:
                c_vals = np.array([config_metrics[(label, s)][m] for s in SEEDS])
                b_vals = np.array(comp[m]["values"])
                delta_vals = c_vals - b_vals
                t_res = stats.ttest_1samp(delta_vals, popmean=0.0)

                b_std = comp[m]["std"]
                c_std = agg["configs"][label][m]["std"]
                pooled_std = np.sqrt((b_std**2 + c_std**2) / 2)
                mean_delta = float(delta_vals.mean())

                b_ci = comp[m]["ci_95"]
                b_ci_halfwidth = (b_ci[1] - b_ci[0]) / 2

                threshold = max(b_ci_halfwidth, 2 * pooled_std)
                reportable = abs(mean_delta) > threshold

                stat_results[label][comp_name][m] = {
                    "mean_delta": mean_delta, "paired_t": float(t_res.statistic),
                    "paired_p": float(t_res.pvalue), "threshold": float(threshold),
                    "reportable": bool(reportable),
                }
                flag = "YES" if reportable else "no (noise)"
                print(f"{label:12s} {comp_name:16s} {m:22s} {mean_delta:+9.4f} {t_res.pvalue:10.4f} {flag:>12s}")

    with open(STATS / "ksweep_learned_lambda_statistics.json", "w") as f:
        json.dump(stat_results, f, indent=2)
    print(f"\nSaved: {STATS / 'ksweep_learned_lambda_statistics.json'}")
    return 0


if __name__ == "__main__":
    exit(main())
