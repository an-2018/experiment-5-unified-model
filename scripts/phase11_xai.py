#!/usr/bin/env python3
"""
Phase 11: XAI Package
=====================
SHAP for modality/feature importance, GNNExplainer for subgraph explanations,
and GraphXAIN for LLM-generated narrative explanations from graph + SHAP data.

Usage:
    uv run python scripts/phase11_xai.py --sample_id daic_test_001 --explain_mode shap
    uv run python scripts/phase11_xai.py --sample_id daic_test_001 --explain_mode gnn
    uv run python scripts/phase11_xai.py --sample_id daic_test_001 --explain_mode graphxain
"""
import argparse
import sys
from pathlib import Path

ROOT = Path("/home/anilson/thesis/thesis-experiment-5-unified-model")

MODES = {
    "shap":     "SHAP — modality and feature-level attribution",
    "gnn":      "GNNExplainer — influential subgraph extraction",
    "graphxain": "GraphXAIN — LLM narrative from GNN subgraph + SHAP values",
}

def main():
    parser = argparse.ArgumentParser(description="Phase 11: XAI Package")
    parser.add_argument("--sample_id", type=str, required=True, help="Sample to explain")
    parser.add_argument("--explain_mode", type=str, choices=list(MODES.keys()), required=True)
    parser.add_argument("--dataset", type=str, default="daic")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="artifacts/figures/phase_11_xai")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    desc = MODES.get(args.explain_mode, "unknown")

    print(f"\n{'='*60}")
    print(f"Phase 11: XAI Package — STUB")
    print(f"  sample_id    : {args.sample_id}")
    print(f"  explain_mode : {args.explain_mode}")
    print(f"  dataset      : {args.dataset}")
    print(f"  desc         : {desc}")
    print(f"{'='*60}")
    print("\n  ⚠  STUB — Phase 11 not yet implemented by @evaluation-xai-engineer.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Phase 11 Stub — {args.explain_mode}\nSample: {args.sample_id}\n(To be implemented by @evaluation-xai-engineer)", fontsize=10)

    titles = [f"{args.explain_mode.upper()} beeswarm", "Reliability", "Counterfactual delta"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_facecolor("#f0f0f0")
        ax.text(0.5, 0.5, f"Phase 11 stub\n{args.explain_mode}\n{args.sample_id}\n{desc[:50]}",
                transform=ax.transAxes, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    stub_path = out_dir / f"phase_11_{args.explain_mode}_{args.sample_id}_stub.png"
    plt.savefig(stub_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Stub figure saved: {stub_path}")
    print("  ✓ Phase 11 stub completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())