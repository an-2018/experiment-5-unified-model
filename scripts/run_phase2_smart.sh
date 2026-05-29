#!/bin/bash
# Phase 2: Full feature extraction pipeline
# Run all datasets × 5 encoders sequentially
#
# Usage:
#   bash scripts/run_phase2_smart.sh          # full extraction
#   bash scripts/run_phase2_smart.sh --dry-run  # print commands only
#
# Strategy:
#   - CPU encoders (roberta, egemaps, openface): parallel=1 (already fast enough)
#   - WavLM on GPU: must be serial (parallel=1) to avoid CUDA OOM
#   - vit encoder returns zeros for DAIC/FI native data (CLNF, no video frames) → skip
#   - MOSEI vit on CPU is fine (pre-extracted features, just loading pickle)
#   - FI has no text → skips roberta
#   - FI test uses 'interview' column mapped to 'openness' → fixed in Python
#   - Visualization uses dataset prefix in filenames to avoid overwrites
set -e
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then DRY_RUN=true; fi

cd /home/anilson/thesis/thesis-experiment-5-unified-model
LOG="data/phase2_smart.log"
[ "$DRY_RUN" = false ] && echo "STARTED: $(date)" > "$LOG"

run() {
  if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] $*"
  else
    echo "$(date) Running: $*" >> "$LOG"
    uv run python "$@" >> "$LOG" 2>&1
  fi
}

# =============================================================
# BLOCK 1: MOSEI (pre-extracted pickle → cache, CPU only)
# =============================================================
echo "=== BLOCK 1: MOSEI (pre-extracted, CPU) ===" | tee -a "$LOG"
run scripts/phase02_preprocess.py --dataset mosei --split all --encoder roberta --parallel 1 --device cpu --visualize
run scripts/phase02_preprocess.py --dataset mosei --split all --encoder egemaps --parallel 1 --device cpu
run scripts/phase02_preprocess.py --dataset mosei --split all --encoder wavlm --parallel 1 --device cpu
run scripts/phase02_preprocess.py --dataset mosei --split all --encoder openface --parallel 1 --device cpu
run scripts/phase02_preprocess.py --dataset mosei --split all --encoder vit --parallel 1 --device cpu
echo "MOSEI done at $(date)" | tee -a "$LOG"

# =============================================================
# BLOCK 2: DAIC text + eGeMAPS + OpenFace (CPU)
# =============================================================
echo "=== BLOCK 2: DAIC text + eGeMAPS + OpenFace (CPU) ===" | tee -a "$LOG"
for split in train val test; do
    run scripts/phase02_preprocess.py --dataset daic --split "$split" --encoder roberta --parallel 1 --device cpu
    run scripts/phase02_preprocess.py --dataset daic --split "$split" --encoder egemaps --parallel 1 --device cpu
    run scripts/phase02_preprocess.py --dataset daic --split "$split" --encoder openface --parallel 1 --device cpu
done
# One visualization at the end of the block
run scripts/phase02_preprocess.py --dataset daic --split train --encoder roberta --parallel 1 --device cpu --visualize
echo "DAIC non-WavLM done at $(date)" | tee -a "$LOG"

# =============================================================
# BLOCK 3: DAIC WavLM (GPU, serial to avoid OOM)
# =============================================================
echo "=== BLOCK 3: DAIC WavLM (GPU, serial) ===" | tee -a "$LOG"
for split in train val test; do
    run scripts/phase02_preprocess.py --dataset daic --split "$split" --encoder wavlm --parallel 1 --device cuda
done
run scripts/phase02_preprocess.py --dataset daic --split train --encoder wavlm --parallel 1 --device cuda --visualize
echo "DAIC WavLM done at $(date)" | tee -a "$LOG"

# =============================================================
# BLOCK 4: FI eGeMAPS + OpenFace (CPU)
# note: FI has no text; vit returns zeros; skip both
# =============================================================
echo "=== BLOCK 4: FI eGeMAPS + OpenFace (CPU) ===" | tee -a "$LOG"
for split in train val test; do
    run scripts/phase02_preprocess.py --dataset fi --split "$split" --encoder egemaps --parallel 1 --device cpu
done
for split in train val test; do
    run scripts/phase02_preprocess.py --dataset fi --split "$split" --encoder openface --parallel 1 --device cpu
done
run scripts/phase02_preprocess.py --dataset fi --split train --encoder egemaps --parallel 1 --device cpu --visualize
echo "FI non-WavLM done at $(date)" | tee -a "$LOG"

# =============================================================
# BLOCK 5: FI WavLM (GPU, serial)
# =============================================================
echo "=== BLOCK 5: FI WavLM (GPU, serial) ===" | tee -a "$LOG"
for split in train val test; do
    run scripts/phase02_preprocess.py --dataset fi --split "$split" --encoder wavlm --parallel 1 --device cuda
done
run scripts/phase02_preprocess.py --dataset fi --split train --encoder wavlm --parallel 1 --device cuda --visualize
echo "FI WavLM done at $(date)" | tee -a "$LOG"

# =============================================================
# DONE
# =============================================================
echo "COMPLETED: $(date)" >> "$LOG"
echo "ALL DONE" | tee -a "$LOG"
