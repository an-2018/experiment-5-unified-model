#!/usr/bin/env python3
"""
run_full_pipeline.py — Full experiment pipeline runner (Phase 0 → 12)

Usage:
    uv run python scripts/run_full_pipeline.py              # run all phases
    uv run python scripts/run_full_pipeline.py --stop_phase 7  # stop at phase 7 (core architecture)
    uv run python scripts/run_full_pipeline.py --dry_run       # print commands only
    uv run python scripts/run_full_pipeline.py --phase 2      # run only phase 2
    uv run python scripts/run_full_pipeline.py --skip_completed # skip phases with existing output

Hardware:
    - Phase 2: CPU for text/egemaps/openface, GPU (--parallel 1) for WavLM
    - Phase 3-7: GPU recommended (--device cuda)
    - Phase 8 LLM ablations: Requires 4x NVIDIA RTX A6000 (48GB VRAM each)
      Auto-detects GPUs and distributes Mistral/LLaVA via accelerate device_map="auto"
    - Phase 9-12: CPU acceptable for most, GPU speeds up XAI

Phases:
    0  — Repository & Environment Setup (verification only)
    1  — Dataset EDA and Data Contract
    2  — Preprocessing and Feature Extraction (OpenSMILE eGeMAPSv02, WavLM, RoBERTa, ViT, OpenFace)
    3  — Unimodal Baselines
    4  — Fusion Baselines (Gated, LMF, LR-DGN)
    5  — MMoEEx (no graph)
    6  — Graph Construction + GraphSAGE/GAT Router
    7  — Joint Multitask Training
    8  — LLM Modality Ablations (L0–L5 on 4x A6000)
    9  — Domain Adaptation (CORAL, MMD, DANN)
    10 — Calibration + Statistical Validation
    11 — XAI (SHAP, GNNExplainer, GraphXAIN)
    12 — Thesis Chapter
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = ROOT / "scripts"


def run_command(cmd: str, dry_run: bool, env_override: dict = None):
    """Run a single command, optionally just print it."""
    if dry_run:
        print(f"  [DRY] {cmd}")
        return True
    print(f"  ▶ {cmd}")
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
    result = subprocess.run(cmd, shell=True, cwd=ROOT, env=env)
    if result.returncode != 0:
        print(f"  ✗ FAILED (exit code {result.returncode})")
        return False
    print(f"  ✓ done")
    return True


def check_output_exists(pattern: str, dry_run: bool) -> bool:
    """Check if phase output already exists (to skip if --skip_completed)."""
    import glob
    matches = glob.glob(pattern)
    return len(matches) > 0


PHASE_DESCRIPTIONS = {
    0: "Repository & Environment Setup (verification)",
    1: "Dataset EDA and Data Contract",
    2: "Preprocessing and Feature Extraction",
    3: "Unimodal Baselines",
    4: "Fusion Baselines (Gated, LMF, LR-DGN)",
    5: "MMoEEx (no graph)",
    6: "Graph Construction + GraphSAGE/GAT Router",
    7: "Joint Multitask Training",
    8: "LLM Modality Ablations (L0–L5)",
    9: "Domain Adaptation",
    10: "Calibration + Statistical Validation",
    11: "XAI (SHAP, GNNExplainer, GraphXAIN)",
    12: "Thesis Chapter",
}


def get_phase_commands(phase: int, args) -> list:
    """Return the list of commands for a given phase."""
    cmds = []

    if phase == 0:
        cmds = [
            "uv run python scripts/test_verify.py",
        ]

    elif phase == 1:
        # Phase 1 — EDA: no arguments, runs all datasets
        cmds = [
            "uv run python scripts/phase01_eda.py",
        ]

    elif phase == 2:
        # Phase 2 — Feature extraction for all datasets and encoders
        # Text (RoBERTa) — CPU, parallel
        for dataset in ["daic", "mosei", "fi"]:
            cmds.append(
                f"uv run python scripts/phase02_preprocess.py --dataset {dataset} --encoder roberta --parallel 16 --device cpu"
            )
        # Audio eGeMAPS (OpenSMILE eGeMAPSv02) — CPU, parallel
        for dataset in ["daic", "mosei", "fi"]:
            cmds.append(
                f"uv run python scripts/phase02_preprocess.py --dataset {dataset} --encoder egemaps --parallel 16 --device cpu"
            )
        # Audio WavLM — GPU, --parallel 1 (CUDA OOM on >1)
        for dataset in ["daic", "mosei", "fi"]:
            cmds.append(
                f"uv run python scripts/phase02_preprocess.py --dataset {dataset} --encoder wavlm --parallel 1 --device cuda"
            )
        # Video OpenFace — DAIC only (no raw video for MOSEI/FI)
        cmds.append(
            "uv run python scripts/phase02_preprocess.py --dataset daic --encoder openface --parallel 16 --device cpu"
        )
        # Video ViT — CPU, parallel
        for dataset, par in [("mosei", 16), ("fi", 8)]:
            cmds.append(
                f"uv run python scripts/phase02_preprocess.py --dataset {dataset} --encoder vit --parallel {par} --device cpu"
            )
        # Rebuild manifest after all extractions
        cmds.append("uv run python scripts/rebuild_manifest.py")
        # Regenerate visualizations (no re-extraction)
        cmds.append("uv run python scripts/phase02_preprocess.py --only-visualize")

    elif phase == 3:
        # Phase 3 — Unimodal baselines (all 9 dataset×modality combos)
        cmds = [
            "uv run python scripts/phase03_unimodal_baselines.py --dataset all --modality all --device cuda",
            # Regenerate all figures + SoA comparison from cached results (no retraining)
            "uv run python scripts/phase03_unimodal_baselines.py --only-visualize",
        ]

    elif phase == 4:
        # Phase 4 — Fusion baselines
        # Gated + LMF for all datasets
        for dataset in ["daic", "mosei", "fi"]:
            cmds.append(
                f"uv run python scripts/phase04_fusion.py --dataset {dataset} --fusion gated --device cuda"
            )
        cmds.append("uv run python scripts/phase04_fusion.py --dataset mosei --fusion lmf --device cuda")
        # LR-DGN (Low-Rank Dynamic Gating Network) for DAIC — r=16 default, r=8 max regularization
        cmds.append("uv run python scripts/phase04_fusion.py --dataset daic --fusion lrdgn --device cuda")
        cmds.append("uv run python scripts/phase04_fusion.py --dataset daic --fusion lrdgn --lrdgn_rank 8 --device cuda")

    elif phase == 5:
        # Phase 5 — MMoEEx (no graph)
        cmds = [
            "uv run python scripts/phase05_mmoe_ex.py --dataset all --tasks depression,sentiment,emotion,personality --device cuda",
        ]

    elif phase == 6:
        # Phase 6 — Graph construction
        # Primary result: split-local graph (leakage-safe)
        cmds.append(
            "uv run python scripts/phase06_graph.py --graph_type split-local --k 10 --device cuda"
        )
        # Visualization: inductive graph (for comparison)
        cmds.append(
            "uv run python scripts/phase06_graph.py --graph_type inductive --k 10 --device cuda"
        )
        # Ablation matrix (V0-V4): runs none/graphsage/gat router variants
        # V0: router=none (no graph), V1: graphsage, V2: gat
        cmds.append(
            "uv run python scripts/phase06_graph.py --run_ablation --graph_type split-local --k 10 --epochs 150 --device cuda"
        )

    elif phase == 7:
        # Phase 7 — Joint multitask training
        # Quick test (3 epochs, no graph routing — fast iteration)
        cmds.append(
            "uv run python scripts/phase07_joint_training.py --epochs 3 --quick_test --router none --graph_type split-local --temperature 2.0 --device cuda"
        )
        # Primary full training run (GraphSAGE router, 150 epochs)
        cmds.append(
            "uv run python scripts/phase07_joint_training.py --epochs 150 --router graphsage --graph_type split-local "
            "--k 10 --graph_weight 0.5 --freeze_epochs 20 --temperature 2.0 --batch_size 32 --lr 1e-3 --device cuda"
        )
        # Router ablation variants (V0=no graph, V2=GAT)
        # V1=GraphSAGE already done as primary above
        for router in ["none", "gat"]:
            cmds.append(
                f"uv run python scripts/phase07_joint_training.py --epochs 150 --router {router} --graph_type split-local "
                f"--k 10 --graph_weight 0.5 --freeze_epochs 20 --temperature 2.0 --batch_size 32 --lr 1e-3 --device cuda"
            )
        # Resume from checkpoint (if interrupted)
        # cmds.append("uv run python scripts/phase07_joint_training.py --resume artifacts/tables/phase07_best.pt ...")

    elif phase == 8:
        # Phase 8 — LLM modality ablations (L0–L5)
        # Uses run_phase08_all.sh which auto-detects 4x A6000 and orchestrates L0-L5
        # First run: ~14-19 hours (extraction + training). After caching: ~1-3 hours.
        cmds.append(
            "bash scripts/run_phase08_all.sh --execute --epochs 30 --device cuda"
        )
        # To skip LLM extraction (use cache from previous run):
        # cmds.append("bash scripts/run_phase08_all.sh --execute --epochs 30 --device cuda --skip_extraction")
        # To generate report from existing results:
        # cmds.append("bash scripts/run_phase08_all.sh --report")

    elif phase == 9:
        # Phase 9 — Domain adaptation (no CLI args — runs all methods by default)
        cmds = [
            "uv run python scripts/phase09_domain_adaptation.py",
        ]

    elif phase == 10:
        # Phase 10 — Calibration + statistical validation
        for method in ["temperature", "isotonic", "platt"]:
            cmds.append(
                f"uv run python scripts/phase10_calibration.py --dataset daic --method {method} --device cuda"
            )

    elif phase == 11:
        # Phase 11 — XAI (SHAP, GNNExplainer, GraphXAIN)
        # Use a representative test sample — update sample_id as needed
        sample = "daic_test_001"  # or use --list_samples to see available
        for mode in ["shap", "gnn", "graphxain"]:
            cmds.append(
                f"uv run python scripts/phase11_xai.py --sample_id {sample} --explain_mode {mode} --device cuda"
            )

    elif phase == 12:
        # Phase 12 — Thesis chapter
        cmds = [
            "uv run python scripts/phase12_thesis.py --output_dir paper/",
        ]

    return cmds


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Unified Multimodal Graph-Gated MoE experiment pipeline (Phase 0→12)."
    )
    parser.add_argument(
        "--stop_phase", type=int, default=12,
        help="Stop after this phase (default: 12)",
    )
    parser.add_argument(
        "--start_phase", type=int, default=0,
        help="Start from this phase (default: 0)",
    )
    parser.add_argument(
        "--phase", type=int, default=None,
        help="Run only a specific phase number",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--skip_completed", action="store_true",
        help="Skip phases whose output already exists",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show command output",
    )
    args = parser.parse_args()

    # Determine which phases to run
    if args.phase is not None:
        phases_to_run = [args.phase]
    else:
        phases_to_run = list(range(args.start_phase, args.stop_phase + 1))

    print("=" * 70)
    print(f"Full Pipeline Run — phases {phases_to_run[0]}→{phases_to_run[-1]}")
    print(f"Hardware: 4x NVIDIA RTX A6000 (48GB VRAM each)")
    print(f"Visualization output: artifacts/figures/phase_XX_name/")
    if args.dry_run:
        print("[DRY RUN — no actual execution]")
    print("=" * 70)

    all_passed = True
    for phase_num in phases_to_run:
        if phase_num not in PHASE_DESCRIPTIONS:
            print(f"\nPhase {phase_num}: unknown, skipping")
            continue

        desc = PHASE_DESCRIPTIONS[phase_num]
        print(f"\n{'='*70}")
        print(f"Phase {phase_num}: {desc}")
        print(f"{'='*70}")

        cmds = get_phase_commands(phase_num, args)
        if not cmds:
            print(f"  (No commands defined for phase {phase_num})")
            continue

        for cmd in cmds:
            if not run_command(cmd, args.dry_run):
                all_passed = False
                print(f"  ✗ Command failed: {cmd}")
                # Continue with next command
                continue

    print("\n" + "=" * 70)
    if args.dry_run:
        print("DRY RUN complete — no commands were executed.")
        print(f"Would run phases: {phases_to_run}")
    elif all_passed:
        print("✓ All phases completed successfully.")
        print("\nVisualization outputs:")
        print("  artifacts/figures/phase_01_eda/")
        print("  artifacts/figures/phase_02_preprocessing/")
        print("  artifacts/figures/phase_03_unimodal_baselines/")
        print("  artifacts/figures/phase_04_fusion/")
        print("  artifacts/figures/phase_05_mmoe_ex/")
        print("  artifacts/figures/phase_06_graph/")
        print("  artifacts/figures/phase_07_joint_training/")
        print("  artifacts/figures/phase_08_llm_ablations/")
        print("  artifacts/figures/phase_09_domain_adaptation/")
        print("  artifacts/figures/phase_10_evaluation/")
        print("  artifacts/figures/phase_11_xai/")
        print("  artifacts/figures/phase_12_thesis/")
    else:
        print("⚠ Some commands failed — check output above.")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()