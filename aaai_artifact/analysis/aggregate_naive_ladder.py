#!/usr/bin/env python3
"""Aggregate the naive 12-dim ladder (full profile, random projection, raw
text) across the 5 canonical-seed phase1_gate_e1_e7_*.json files into one
summary JSON, for the claim ledger to point at without needing list-aggregation
logic in the verifier."""
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
STATS_DIR = REPO_ROOT / "artifacts" / "stats"

FILES = {
    "original": "phase1_gate_e1_e7.json",
    "seed17": "phase1_gate_e1_e7_mmoe_ex_best_seed17.json",
    "seed1337": "phase1_gate_e1_e7_mmoe_ex_best_seed1337.json",
    "seed2024": "phase1_gate_e1_e7_mmoe_ex_best_seed2024.json",
    "seed31415": "phase1_gate_e1_e7_mmoe_ex_best_seed31415.json",
}


def main():
    per_seed = {}
    for name, fname in FILES.items():
        d = json.load(open(STATS_DIR / fname))
        per_seed[name] = {
            "full_profile_auroc": d["E1_full_profile_auroc"],
            "random_projection_mean": d["E1_random_projection_mean"],
            "text_only_auroc": d["E1_text_only_auroc"],
            "permutation_p_value": d["E1_permutation_p_value"],
        }

    profile = np.array([v["full_profile_auroc"] for v in per_seed.values()])
    rp = np.array([v["random_projection_mean"] for v in per_seed.values()])
    text = np.array([v["text_only_auroc"] for v in per_seed.values()])
    perm_p = np.array([v["permutation_p_value"] for v in per_seed.values()])
    delta = profile - rp

    out = {
        "per_seed": per_seed,
        "aggregate": {
            "n_seeds": len(per_seed),
            "profile_auroc_mean": float(profile.mean()), "profile_auroc_std": float(profile.std()),
            "rp_auroc_mean": float(rp.mean()), "rp_auroc_std": float(rp.std()),
            "text_only_auroc": float(text.mean()), "text_only_identical_across_seeds": bool(text.std() < 1e-9),
            "delta_mean": float(delta.mean()), "delta_std": float(delta.std()),
            "all_negative": bool(np.all(delta < 0)),
            "permutation_p_max": float(perm_p.max()), "permutation_p_min": float(perm_p.min()),
            "all_p_above_005": bool(np.all(perm_p > 0.05)),
        },
    }
    with open(STATS_DIR / "e1_naive_ladder_aggregate.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["aggregate"], indent=2))
    print(f"\nSaved: artifacts/stats/e1_naive_ladder_aggregate.json")


if __name__ == "__main__":
    main()
