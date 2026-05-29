#!/usr/bin/env python3
"""
run_full_pipeline.py — Full experiment pipeline runner (Phase 0 → 12)

Usage:
    uv run python scripts/run_full_pipeline.py              # run all phases
    uv run python scripts/run_full_pipeline.py --stop_phase 7  # stop at phase 7
    uv run python scripts/run_full_pipeline.py --dry_run       # print commands only

Phases:
    0  — Repository & Environment Setup
    1  — Dataset EDA and Data Contract
    2  — Preprocessing and Feature Extraction
    3  — Unimodal Baselines
    4  — Fusion Baselines
    5  — MMoEEx (no graph)
    6  — Graph Construction + GraphSAGE/GAT Router
    7  — Joint Multitask Training
    8  — LLM Modality Ablations
    9  — Domain Adaptation
    10 — Calibration + Statistical Validation
    11 — XAI Package
    12 — Thesis Chapter
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent.resolve()

SCRIPTS = {
    0: {
        "name": "Phase 0 — Repository & Environment Setup",
        "commands": [
            "uv run python scripts/test_verify.py",
        ],
    },
    1: {
        "name": "Phase 1 — Dataset EDA and Data Contract",
        "commands": [
            # phase01_eda.py takes NO arguments — runs all datasets
            "uv run python scripts/phase01_eda.py",
        ],
    },
    2: {
        "name": "Phase 2 — Preprocessing and Feature Extraction",
        "commands": [
            # DAIC: all encoders (roberta, egemaps, wavlm, openface, vit) + parallel=8 workers
            "uv run python scripts/phase02_preprocess.py --dataset daic --encoder all --parallel 8",
            # MOSEI: text only (fast path — 22k utterances; audio/video done separately if needed)
            "uv run python scripts/phase02_preprocess.py --dataset mosei --encoder roberta --parallel 8",
            # FI: text only (FI has no transcript, but we create empty placeholders)
            "uv run python scripts/phase02_preprocess.py --dataset fi --encoder roberta --parallel 8",
            # MOSEI audio (optional — large dataset, run if resources allow)
            # "uv run python scripts/phase02_preprocess.py --dataset mosei --encoder wavlm --parallel 8",
            # MOSEI video (optional)
            # "uv run python scripts/phase02_preprocess.py --dataset mosei --encoder vit --parallel 8",
            # FI video (optional)
            # "uv run python scripts/phase02_preprocess.py --dataset fi --encoder vit --parallel 8",
        ],
    },
    3: {
        "name": "Phase 3 — Unimodal Baselines",
        "commands": [
            "uv run python scripts/phase03_unimodal_baselines.py --dataset daic --modality text",
            "uv run python scripts/phase03_unimodal_baselines.py --dataset daic --modality audio",
            "uv run python scripts/phase03_unimodal_baselines.py --dataset daic --modality video",
            "uv run python scripts/phase03_unimodal_baselines.py --dataset mosei --modality text",
            "uv run python scripts/phase03_unimodal_baselines.py --dataset fi --modality video",
        ],
    },
    4: {
        "name": "Phase 4 — Fusion Baselines",
        "commands": [
            "uv run python scripts/phase04_fusion.py --dataset daic --fusion gated",
            "uv run python scripts/phase04_fusion.py --dataset daic --fusion lmf",
            "uv run python scripts/phase04_fusion.py --dataset mosei --fusion gated",
            "uv run python scripts/phase04_fusion.py --dataset fi --fusion gated",
        ],
    },
    5: {
        "name": "Phase 5 — MMoEEx (no graph)",
        "commands": [
            "uv run python scripts/phase05_mmoeex.py --dataset all --tasks depression,sentiment,emotion,personality",
        ],
    },
    6: {
        "name": "Phase 6 — Graph Construction + GraphSAGE/GAT Router",
        "commands": [
            "uv run python scripts/phase06_graph.py --graph_type split-local --k 10",
        ],
    },
    7: {
        "name": "Phase 7 — Joint Multitask Training",
        "commands": [
            "uv run python scripts/phase07_joint_training.py --epochs 50 --batch_size 32 --temperature 2.0",
        ],
    },
    8: {
        "name": "Phase 8 — LLM Modality Ablations",
        "commands": [
            "uv run python scripts/phase08_llm_ablations.py --ablation L0",
            "uv run python scripts/phase08_llm_ablations.py --ablation L1",
            "uv run python scripts/phase08_llm_ablations.py --ablation L3",
            "uv run python scripts/phase08_llm_ablations.py --ablation L5",
        ],
    },
    9: {
        "name": "Phase 9 — Domain Adaptation",
        "commands": [
            "uv run python scripts/phase09_domain_adaptation.py --method mmd",
            "uv run python scripts/phase09_domain_adaptation.py --method coral",
            "uv run python scripts/phase09_domain_adaptation.py --method dann",
        ],
    },
    10: {
        "name": "Phase 10 — Calibration + Statistical Validation",
        "commands": [
            "uv run python scripts/phase10_calibration.py --dataset daic --method temperature",
            "uv run python scripts/phase10_calibration.py --dataset daic --method isotonic",
            "uv run python scripts/phase10_calibration.py --dataset daic --method platt",
        ],
    },
    11: {
        "name": "Phase 11 — XAI Package",
        "commands": [
            "uv run python scripts/phase11_xai.py --sample_id daic_test_001 --explain_mode shap",
            "uv run python scripts/phase11_xai.py --sample_id daic_test_001 --explain_mode gnn",
            "uv run python scripts/phase11_xai.py --sample_id daic_test_001 --explain_mode graphxain",
        ],
    },
    12: {
        "name": "Phase 12 — Thesis Chapter",
        "commands": [
            "uv run python scripts/phase12_thesis.py --output_dir paper/",
        ],
    },
}

# Phases whose scripts are implemented and runnable
IMPLEMENTED_SCRIPTS = {0, 1, 2}


def run_command(cmd: str, dry_run: bool, verbose: bool):
    """Run a single command, optionally just print it."""
    if dry_run:
        print(f"  [DRY] {cmd}")
        return True
    print(f"  ▶ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"  ✗ FAILED (exit code {result.returncode})")
        return False
    print(f"  ✓ done")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Unified Multimodal Graph-Gated MoE experiment pipeline."
    )
    parser.add_argument(
        "--stop_phase",
        type=int,
        default=12,
        help="Stop after this phase (default: 12)",
    )
    parser.add_argument(
        "--start_phase",
        type=int,
        default=0,
        help="Start from this phase (default: 0)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        help="Run only a specific phase number",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--skip_implemented",
        action="store_true",
        help="Skip phases whose scripts don't exist yet",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
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
    if args.dry_run:
        print("[DRY RUN — no actual execution]")
    print("=" * 70)

    all_passed = True
    for phase_num in phases_to_run:
        if phase_num not in SCRIPTS:
            print(f"\nPhase {phase_num}: unknown, skipping")
            continue

        info = SCRIPTS[phase_num]
        is_implemented = phase_num in IMPLEMENTED_SCRIPTS

        print(f"\n{'='*70}")
        print(f"Phase {phase_num}: {info['name']}")
        if not is_implemented:
            status = "NOT IMPLEMENTED YET (script missing)"
            if args.skip_implemented:
                print(f"  [{status}] — skipping")
                continue
        else:
            status = "implemented"
        print(f"{'='*70}")

        if info.get("optional") and phase_num == 0 and not args.dry_run:
            print("  (Phase 0 is optional — already completed in setup)")
            # Still run the verification as a sanity check
            if not run_command(SCRIPTS[0]["commands"][0], args.dry_run, args.verbose):
                all_passed = False
                print("  ⚠ Phase 0 verification failed — but continuing...")
            continue

        for cmd in info["commands"]:
            if not run_command(cmd, args.dry_run, args.verbose):
                all_passed = False
                print(f"  ✗ Command failed: {cmd}")
                # Continue with next command, don't abort the whole phase
                # unless it's critical

    print("\n" + "=" * 70)
    if args.dry_run:
        print("DRY RUN complete — no commands were executed.")
        print(f"Would run phases: {phases_to_run}")
    elif all_passed:
        print("✓ All phases completed successfully.")
    else:
        print("⚠ Some commands failed — check output above.")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()