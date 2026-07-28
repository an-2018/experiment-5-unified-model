#!/usr/bin/env python3
"""Aggregate the 5-seed graph-routing rerun (V0-V4) and the matched-seed
non-graph baseline into Table 1's mean +- std / 95% CI, replacing the stale
v2 single-seed numbers (REQ-02: every headline metric needs >=5 seeds with CI).

Seeds: [17, 42, 1337, 2024, 31415] for BOTH baseline and variants (the
baseline previously used an "unknown seed" stand-in for one slot; this
requires the true seed=42 baseline run, analysis/../scripts/phase05_mmoe_ex.py
--seed 42, to exist before this aggregation is valid).
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES = REPO_ROOT / "artifacts" / "tables"

SEEDS = [17, 42, 1337, 2024, 31415]
VARIANTS = ["V0", "V1", "V2", "V3", "V4"]
METRICS = ["daic_auroc", "mosei_sentiment_ccc", "mosei_emotion_auc", "fi_avg_ccc"]


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

    baseline_path = TABLES / "mmoe_ex_results_seed42.csv"
    if not baseline_path.exists():
        missing.append(str(baseline_path))

    baseline_files = {
        17: TABLES / "mmoe_ex_results_seed17.csv",
        42: TABLES / "mmoe_ex_results_seed42.csv",
        1337: TABLES / "mmoe_ex_results_seed1337.csv",
        2024: TABLES / "mmoe_ex_results_seed2024.csv",
        31415: TABLES / "mmoe_ex_results_seed31415.csv",
    }
    for s, p in baseline_files.items():
        if not p.exists():
            missing.append(str(p))

    variant_files = {}
    for v in VARIANTS:
        for s in SEEDS:
            p = TABLES / f"phase07_results_{v}_seed{s}.csv"
            variant_files[(v, s)] = p
            if not p.exists():
                missing.append(str(p))

    if missing:
        print("MISSING FILES, cannot aggregate yet:")
        for m in missing:
            print(" ", m)
        return 1

    baseline_metrics = {s: read_csv_metrics(p) for s, p in baseline_files.items()}
    variant_metrics = {(v, s): read_csv_metrics(p) for (v, s), p in variant_files.items()}

    results = {"seeds": SEEDS, "baseline": {}, "variants": {}}

    print(f"{'':25s}" + "".join(f"{m:>20s}" for m in METRICS))
    baseline_row = {}
    for m in METRICS:
        vals = np.array([baseline_metrics[s][m] for s in SEEDS])
        lo, hi = bca_ci(vals)
        baseline_row[m] = {"mean": float(vals.mean()), "std": float(vals.std()),
                           "values": vals.tolist(), "ci_95": [lo, hi]}
        results["baseline"][m] = baseline_row[m]
    print(f"{'Non-graph MMoEEx':25s}" +
          "".join(f"{baseline_row[m]['mean']:.4f}+-{baseline_row[m]['std']:.4f}   "[:20].rjust(20) for m in METRICS))

    for v in VARIANTS:
        variant_row = {}
        for m in METRICS:
            vals = np.array([variant_metrics[(v, s)][m] for s in SEEDS])
            lo, hi = bca_ci(vals)
            variant_row[m] = {"mean": float(vals.mean()), "std": float(vals.std()),
                              "values": vals.tolist(), "ci_95": [lo, hi]}
        results["variants"][v] = variant_row
        print(f"{v:25s}" +
              "".join(f"{variant_row[m]['mean']:.4f}+-{variant_row[m]['std']:.4f}   "[:20].rjust(20) for m in METRICS))

    # Sign-test style assertion: for each metric, count variants whose mean beats/underperforms baseline
    print("\nDelta vs baseline (mean):")
    for m in METRICS:
        b = baseline_row[m]["mean"]
        for v in VARIANTS:
            d = results["variants"][v][m]["mean"] - b
            print(f"  {m:22s} {v}: delta={d:+.4f}")

    out_path = REPO_ROOT / "artifacts" / "stats" / "routing_table1_5seed.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
