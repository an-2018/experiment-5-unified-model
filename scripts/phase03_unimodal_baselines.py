#!/usr/bin/env python3
"""
Phase 3: Unimodal Baselines
============================
Train isolated single-modality models for each dataset:
  Text (RoBERTa), Audio (WavLM), Video (ViT)
as baseline comparisons before fusion.

Usage:
    uv run python scripts/phase03_unimodal_baselines.py --dataset daic --modality text
    uv run python scripts/phase03_unimodal_baselines.py --dataset daic --modality audio
    uv run python scripts/phase03_unimodal_baselines.py --dataset daic --modality video
    uv run python scripts/phase03_unimodal_baselines.py --dataset mosei --modality text
    uv run python scripts/phase03_unimodal_baselines.py --dataset fi --modality video
"""
import argparse
import sys
from pathlib import Path

ROOT = Path("/home/anilson/thesis/thesis-experiment-5-unified-model")

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Unimodal Baselines")
    parser.add_argument("--dataset", type=str, choices=["daic", "mosei", "fi", "all"], required=True)
    parser.add_argument("--modality", type=str, choices=["text", "audio", "video", "all"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="artifacts/figures/phase_03_unimodal_baselines")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase 3: Unimodal Baselines — STUB")
    print(f"  dataset  : {args.dataset}")
    print(f"  modality : {args.modality}")
    print(f"  epochs   : {args.epochs}")
    print(f"  lr       : {args.lr}")
    print(f"{'='*60}")
    print("\n  ⚠  STUB — Phase 3 not yet implemented by @multimodal-architect.")
    print("     This script will be replaced with actual unimodal training code.")
    print()

    # Generate a placeholder visualization so the pipeline remains valid
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Phase 3 Stub — {args.dataset} / {args.modality}\n(To be implemented by @multimodal-architect)", fontsize=12)

    for ax, (name, color) in zip(axes, [
        ("train_loss", "steelblue"),
        ("val_auroc", "forestgreen"),
        ("val_f1", "coral"),
    ]):
        ax.set_title(f"{name} (placeholder)")
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.set_facecolor("#f0f0f0")
        ax.text(0.5, 0.5, f"Phase 3 stub\n{args.dataset}/{args.modality}\n{args.epochs} epochs\nlr={args.lr}",
                transform=ax.transAxes, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    stub_path = out_dir / f"phase_03_{args.dataset}_{args.modality}_stub.png"
    plt.savefig(stub_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Stub figure saved: {stub_path}")
    print("\n  ✓ Phase 3 stub completed (no actual training performed)")
    print("  Implement Phase 3 in src/models/ and scripts/phase03_unimodal_baselines.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())