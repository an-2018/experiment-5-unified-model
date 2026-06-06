#!/usr/bin/env python3
"""
run_full_pipeline.py — Full experiment pipeline runner (Phase 0 → 12)

Logs are saved to: logs/pipeline_YYYYMMDD_HHMMSS.log
Each phase's output is also logged to: logs/phase_XX_<phase_name>.log

Usage:
    uv run python scripts/run_full_pipeline.py              # run all phases
    uv run python scripts/run_full_pipeline.py --stop_phase 7  # stop at phase 7 (core architecture)
    uv run python scripts/run_full_pipeline.py --dry_run       # print commands only
    uv run python scripts/run_full_pipeline.py --phase 2      # run only phase 2
    uv run python scripts/run_full_pipeline.py --skip_completed # skip phases with existing output
    uv run python scripts/run_full_pipeline.py --log_dir /path/to/logs  # custom log directory

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
from datetime import datetime

# Project root
ROOT = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = ROOT / "scripts"
DEFAULT_LOG_DIR = ROOT / "logs"


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(
    cmd: str,
    dry_run: bool,
    log_file: Path = None,
    env_override: dict = None,
) -> tuple[bool, float]:
    """Run a single command with optional log file.
    Captures stdout/stderr and writes to both console and log file.
    Returns (success: bool, elapsed_seconds: float).
    """
    start = datetime.now()
    start_str = start.strftime("%Y-%m-%d %H:%M:%S")

    if log_file:
        with open(log_file, "a") as lf:
            lf.write(f"\n{'='*70}\n")
            lf.write(f"[{start_str}] START: {cmd}\n")
            lf.write(f"{'='*70}\n")

    if dry_run:
        print(f"  [DRY] {cmd}")
        return True, 0.0

    print(f"  ▶ {cmd}")

    env = os.environ.copy()
    if env_override:
        env.update(env_override)

    result = subprocess.run(
        cmd, shell=True, cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    output = result.stdout.decode("utf-8", errors="replace")

    # Print output to console
    print(output)

    # Write to log file
    if log_file:
        with open(log_file, "a") as lf:
            lf.write(output)
            elapsed = (datetime.now() - start).total_seconds()
            lf.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FINISH: rc={result.returncode}, elapsed={elapsed:.1f}s\n")

    elapsed = (datetime.now() - start).total_seconds()

    if result.returncode != 0:
        print(f"\n  ✗ FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False, elapsed

    print(f"  ✓ done in {elapsed:.1f}s")
    return True, elapsed


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
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="Directory for log files (default: logs/)",
    )
    args = parser.parse_args()

    # Setup log directory
    log_dir = Path(args.log_dir) if args.log_dir else DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped master log file
    ts = get_timestamp()
    master_log = log_dir / f"pipeline_{ts}.log"
    phase_logs = {}  # phase_num -> log file path

    # Determine which phases to run
    if args.phase is not None:
        phases_to_run = [args.phase]
    else:
        phases_to_run = list(range(args.start_phase, args.stop_phase + 1))

    # Print header
    print("=" * 70)
    print(f"Full Pipeline Run — phases {phases_to_run[0]}→{phases_to_run[-1]}")
    print(f"Hardware: 4x NVIDIA RTX A6000 (48GB VRAM each)")
    print(f"Master log: {master_log}")
    print(f"Visualization output: artifacts/figures/phase_XX_name/")
    if args.dry_run:
        print("[DRY RUN — no actual execution]")
    print("=" * 70)

    # Write run header to master log
    with open(master_log, "w") as ml:
        ml.write(f"# {'='*68}\n")
        ml.write(f"# Pipeline Run — {datetime.now().isoformat()}\n")
        ml.write(f"# Phases: {phases_to_run[0]}→{phases_to_run[-1]}\n")
        ml.write(f"# Hardware: 4x NVIDIA RTX A6000 (48GB VRAM each)\n")
        ml.write(f"# {'='*68}\n")
        ml.write(f"# Log directory: {log_dir}\n")
        ml.write(f"# Master log: {master_log}\n")
        ml.write(f"\n")

    all_passed = True
    phase_results = {}  # phase_num -> list of (cmd, success, elapsed)
    total_elapsed = 0.0

    for phase_num in phases_to_run:
        if phase_num not in PHASE_DESCRIPTIONS:
            print(f"\nPhase {phase_num}: unknown, skipping")
            continue

        desc = PHASE_DESCRIPTIONS[phase_num]
        print(f"\n{'='*70}")
        print(f"Phase {phase_num}: {desc}")
        print(f"{'='*70}")

        # Create per-phase log file (sanitize name: replace / and spaces)
        phase_name = desc.split("—")[0].strip().replace(" ", "_").replace("/", "_").replace("+", "_")
        phase_log = log_dir / f"phase_{phase_num:02d}_{phase_name}_{ts}.log"
        phase_logs[phase_num] = phase_log

        # Write phase header to master log
        with open(master_log, "a") as ml:
            ml.write(f"\n## Phase {phase_num}: {desc}\n")
            ml.write(f"## Phase log: {phase_log}\n")

        # Write phase header to phase log
        with open(phase_log, "w") as pl:
            pl.write(f"# Phase {phase_num}: {desc}\n")
            pl.write(f"# Started: {datetime.now().isoformat()}\n")
            pl.write(f"# {'='*60}\n\n")

        cmds = get_phase_commands(phase_num, args)
        if not cmds:
            print(f"  (No commands defined for phase {phase_num})")
            continue

        phase_results[phase_num] = []

        for cmd in cmds:
            success, elapsed = run_command(cmd, args.dry_run, log_file=phase_log)
            phase_results[phase_num].append((cmd, success, elapsed))
            total_elapsed += elapsed

            if not success:
                all_passed = False
                # Continue with next command
                continue

        # Write phase footer to phase log
        with open(phase_log, "a") as pl:
            phase_elapsed = sum(e for _, _, e in phase_results[phase_num])
            n_cmds = len(phase_results[phase_num])
            n_ok = sum(1 for _, ok, _ in phase_results[phase_num] if ok)
            pl.write(f"\n# Phase {phase_num} complete: {n_ok}/{n_cmds} commands succeeded in {phase_elapsed:.1f}s\n")

        # Update master log with phase summary
        with open(master_log, "a") as ml:
            n_cmds = len(phase_results[phase_num])
            n_ok = sum(1 for _, ok, _ in phase_results[phase_num] if ok)
            phase_elapsed = sum(e for _, _, e in phase_results[phase_num])
            ml.write(f"## Phase {phase_num} result: {n_ok}/{n_cmds} OK, {phase_elapsed:.1f}s total\n")

    # Write final summary
    print("\n" + "=" * 70)
    if args.dry_run:
        print("DRY RUN complete — no commands were executed.")
        print(f"Would run phases: {phases_to_run}")
        print(f"Would create logs in: {log_dir}")
    elif all_passed:
        print("✓ All phases completed successfully.")
        print(f"\nTotal elapsed time: {total_elapsed:.1f}s ({total_elapsed/3600:.1f}h)")
    else:
        print("⚠ Some commands failed — check output above.")
        print(f"\nTotal elapsed time: {total_elapsed:.1f}s ({total_elapsed/3600:.1f}h)")

    print(f"\nLog files saved to: {log_dir}/")
    for phase_num, plog in phase_logs.items():
        print(f"  Phase {phase_num:02d}: {plog.name}")
    print(f"  Master log: {master_log.name}")

    print("=" * 70)

    # Write summary to master log
    with open(master_log, "a") as ml:
        ml.write(f"\n# {'='*68}\n")
        ml.write(f"# Pipeline Run Summary — {datetime.now().isoformat()}\n")
        ml.write(f"# Status: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n")
        ml.write(f"# Total elapsed: {total_elapsed:.1f}s ({total_elapsed/3600:.1f}h)\n")
        ml.write(f"# {'='*68}\n")
        ml.write(f"\nPhase Summary:\n")
        for phase_num, results in phase_results.items():
            n_cmds = len(results)
            n_ok = sum(1 for _, ok, _ in results if ok)
            phase_elapsed = sum(e for _, _, e in results)
            desc = PHASE_DESCRIPTIONS[phase_num]
            ml.write(f"  Phase {phase_num:02d} ({desc}): {n_ok}/{n_cmds} OK, {phase_elapsed:.1f}s\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()