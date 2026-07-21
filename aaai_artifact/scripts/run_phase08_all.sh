#!/bin/bash
# ============================================================================
# run_phase08_all.sh — Phase 8 LLM Modality Ablations
#
# Runs L0–L5 sequentially with proper logging, checkpointing, and error
# handling. Generates summary CSV + figures at the end.
#
# Usage:
#   chmod +x scripts/run_phase08_all.sh
#   ./scripts/run_phase08_all.sh                         # dry-run (show what's planned)
#   ./scripts/run_phase08_all.sh --execute                # full run
#   ./scripts/run_phase08_all.sh --execute --epochs 30    # full run, 30 epochs each
#   ./scripts/run_phase08_all.sh --execute --resume       # skip completed levels
#   ./scripts/run_phase08_all.sh --report                 # generate report only
#
# Requirements:
#   - uv (Python package manager)
#   - GPU strongly recommended for L1-L5 (A6000 48GB tested)
#   - transformers, peft, accelerate for LLM feature extraction
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Configuration ──────────────────────────────────────────────────────────
EPOCHS=30
BATCH_SIZE=32
LR=0.0001
EXECUTE=false
RESUME=false
DRY_RUN=true       # stays true unless --execute is passed
DEVICE="auto"      # auto-detect GPU; override with --device cuda or --device cpu
SKIP_EXTRACTION=false

LOG_DIR="logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/phase08_full_run_${TIMESTAMP}.log"
CHECKPOINT_FILE="${LOG_DIR}/phase08_checkpoint.txt"

mkdir -p "$LOG_DIR"

# ── Parse arguments ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --execute)   EXECUTE=true; DRY_RUN=false; shift ;;
        --resume)    RESUME=true; EXECUTE=true; DRY_RUN=false; shift ;;
        --report)    EXECUTE=false; DRY_RUN=false; shift ;;
        --epochs)    EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --lr)        LR="$2"; shift 2 ;;
        --skip_extraction) SKIP_EXTRACTION=true; shift ;;
        --device)    DEVICE="$2"; shift 2 ;;
        --help|-h)   head -30 "$0"; exit 0 ;;
        *)           echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── GPU detection ──────────────────────────────────────────────────────────
# "auto" → detect GPU, "cuda" → force GPU, "cpu" → force CPU
if [ "$DEVICE" = "auto" ]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi

if [ "$DEVICE" = "cuda" ]; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    echo "  🖥️  GPU: $GPU_NAME ($GPU_MEM)"
fi

# ── GPU detection ─────────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
if [ "$NUM_GPUS" -gt 0 ]; then
    GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    echo "🖥️  ${NUM_GPUS}x NVIDIA GPU(s) detected (${GPU_MEM_TOTAL} MiB each)"
fi
echo "   peft $(uv run python -c "import peft; print(peft.__version__)" 2>/dev/null || echo 'N/A') | accelerate $(uv run python -c "import accelerate; print(accelerate.__version__)" 2>/dev/null || echo 'N/A') | opensmile $(uv run python -c "import opensmile; print(opensmile.__version__)" 2>/dev/null || echo 'N/A')"

# ── Ablation definitions ──────────────────────────────────────────────────
declare -A ABLATIONS
ABLATIONS[L0]="Classical encoders (instant — reuses Phase 5)"
ABLATIONS[L1]="Mistral-7B-Instruct frozen — text branch"
ABLATIONS[L2]="Mistral LoRA (r=16, alpha=32) — text branch"
ABLATIONS[L3]="Mistral + CLAP audio — text + audio"
ABLATIONS[L4]="Mistral + LLaVA video — text + video"
ABLATIONS[L5]="Full LLM stack — text + audio + video"

ABLATION_ORDER=("L0" "L1" "L2" "L3" "L4" "L5")

# ── Checkpoint helpers ─────────────────────────────────────────────────────
get_completed_levels() {
    if [ ! -f "$CHECKPOINT_FILE" ]; then
        return
    fi
    cat "$CHECKPOINT_FILE" 2>/dev/null || true
}

mark_completed() {
    local level=$1
    echo "$level" >> "$CHECKPOINT_FILE"
    echo "  📝 Checkpoint: $level completed"
}

is_completed() {
    local level=$1
    if ! $RESUME; then
        return 1
    fi
    if [ -f "$CHECKPOINT_FILE" ] && grep -qxF "$level" "$CHECKPOINT_FILE" 2>/dev/null; then
        return 0
    fi
    return 1
}

# ── Print header ──────────────────────────────────────────────────────────
print_header() {
    local level=$1
    local desc=$2
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Phase 8 — Ablation $level: $desc"
    echo "  $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# ── Run single ablation ────────────────────────────────────────────────────
run_ablation() {
    local level=$1
    local desc=${ABLATIONS[$level]}
    local extra_args=""

    print_header "$level" "$desc"

    if is_completed "$level"; then
        echo "  ⏭️  $level already completed (resume mode). Skipping."
        return 0
    fi

    # Parse GPU hours start — no reliable GPU hours tracking, just pass it
    local cmd="uv run python scripts/phase08_llm_ablations.py \
        --ablation $level \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --device $DEVICE"

    if $SKIP_EXTRACTION; then
        cmd="$cmd --skip_extraction"
    fi

    echo "  Running: $cmd"
    echo ""

    if $DRY_RUN; then
        echo "  [DRY RUN] Would execute: $cmd"
        return 0
    fi

    # Execute and tee
    START_TIME=$(date +%s)
    if ! eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
        echo "  ❌ $level FAILED at $(date)" | tee -a "$LOG_FILE"
        echo "     Check $LOG_FILE for details."
        return 1
    fi
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "  ✅ $level completed in ${DURATION}s ($(date))" | tee -a "$LOG_FILE"
    mark_completed "$level"
    return 0
}

# ── Generate report ────────────────────────────────────────────────────────
generate_report() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Generating Summary Report"
    echo "  $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    if $DRY_RUN; then
        echo "  [DRY RUN] Would generate report with:"
        echo "    uv run python scripts/phase08_llm_ablations.py --generate_report"
        return 0
    fi

    uv run python scripts/phase08_llm_ablations.py --generate_report 2>&1 | tee -a "$LOG_FILE"
    echo ""
    echo "  ✅ Report generated at $(date)" | tee -a "$LOG_FILE"
    return 0
}

# ── Main ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Phase 8 — LLM Modality Ablations (L0–L5)                   ║"
echo "║     $(date)                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Device:      $DEVICE"
echo "  Epochs:      $EPOCHS"
echo "  Batch size:  $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Skip extraction: $SKIP_EXTRACTION"
echo "  Mode:        $([ "$DRY_RUN" = true ] && echo 'DRY RUN' || echo 'EXECUTE')"
echo "  Resume:      $RESUME"
echo "  Log:         $LOG_FILE"
echo "  Checkpoint:  $CHECKPOINT_FILE"
echo ""

# LLM extraction mode
if $SKIP_EXTRACTION; then
    echo "  ⚠️  LLM FEATURES: Using CACHED features (--skip_extraction). Classical fallback if no cache."
elif [ "$DEVICE" = "cpu" ]; then
    echo "  ⚠️  LLM FEATURES: CPU mode — Mistral/CLAP/LLaVA extraction will be EXTREMELY slow."
    echo "                   Recommend --device cuda for real LLM feature extraction."
else
    echo "  ✅ LLM FEATURES: Will extract REAL Mistral/CLAP/LLaVA features on GPU (first run slow)."
    echo "                   Use --skip_extraction after cache is populated."
fi
echo ""

if [ "$DRY_RUN" = true ] && [ "$EXECUTE" = false ]; then
    echo "  ⚡ Dry-run mode. Pass --execute to actually run."
    echo ""
fi

# Handle --report only mode
if [ "$EXECUTE" = false ] && [ "$DRY_RUN" = false ]; then
    echo "  Report-only mode. Generating summary from existing results..."
    generate_report
    echo ""
    echo "  Output files:"
    echo "    ├── artifacts/tables/phase08_llm_ablations.csv"
    echo "    └── artifacts/figures/phase_08_llm_ablations/"
    echo ""
    exit 0
fi

echo "  ┌────────────────────────────────────────────────────────────────┐"
echo "  │  Ablation Order:                                              │"
for level in "${ABLATION_ORDER[@]}"; do
    status="⬜"
    if is_completed "$level"; then
        status="✅"
    fi
    printf "  │  %s %-5s %-55s │\n" "$status" "$level" "${ABLATIONS[$level]}"
done
echo "  └────────────────────────────────────────────────────────────────┘"
echo ""

# Print estimated time (informational only — no prompt)
if $EXECUTE && ! $DRY_RUN; then
    if [ "$DEVICE" = "cuda" ] && ! $SKIP_EXTRACTION; then
        echo "  ⏱️  First run on GPU: ~14-19 hours (extraction + 30 epochs each level)"
        echo "     After caching: ~1-3 hours (use --skip_extraction)"
    elif [ "$DEVICE" = "cpu" ]; then
        echo "  ⏱️  CPU mode: ~3 min total (classical fallback — not real LLM features)"
    fi
    echo ""
fi

# Create checkpoint file if starting fresh
if [ ! -f "$CHECKPOINT_FILE" ] && $EXECUTE; then
    touch "$CHECKPOINT_FILE"
fi

# Start log
if ! $DRY_RUN; then
    {
        echo "========================================"
        echo "Phase 8 Full Run — $(date)"
        echo "Device: $DEVICE, Epochs: $EPOCHS, BS: $BATCH_SIZE"
        echo "========================================"
    } > "$LOG_FILE"
fi

# Run each ablation
FAILED_LEVELS=()
for level in "${ABLATION_ORDER[@]}"; do
    if ! run_ablation "$level"; then
        FAILED_LEVELS+=("$level")
    fi
done

# Generate final report
generate_report

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Phase 8 Summary                                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if $EXECUTE && [ ${#FAILED_LEVELS[@]} -eq 0 ]; then
    echo "  ✅ All levels completed successfully!"
elif $EXECUTE; then
    echo "  ⚠️  Completed with ${#FAILED_LEVELS[@]} failure(s): ${FAILED_LEVELS[*]}"
fi

echo ""
echo "  Results:"
echo "    ├── artifacts/tables/phase08_llm_ablations.csv"
echo "    ├── artifacts/figures/phase_08_llm_ablations/llm_delta_bar.png"
echo "    ├── artifacts/figures/phase_08_llm_ablations/embedding_umap.png"
echo "    └── artifacts/figures/phase_08_llm_ablations/cost_performance.png"
echo ""
echo "  Log: $LOG_FILE"
echo "  Checkpoint: $CHECKPOINT_FILE"
echo ""

# ── Show CSV ─────────────────────────────────────────────────────────────
if [ -f "artifacts/tables/phase08_llm_ablations.csv" ]; then
    echo "  ┌─ Results CSV ──────────────────────────────────────────────┐"
    column -t -s, "artifacts/tables/phase08_llm_ablations.csv" | sed 's/^/  │ /'
    echo "  └────────────────────────────────────────────────────────────┘"
fi
echo ""
