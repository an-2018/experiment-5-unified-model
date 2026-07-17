#!/bin/bash
# Phase 2: Full feature extraction pipeline
# Run all datasets × 5 encoders sequentially
#
# Usage:
#   bash scripts/run_phase2_smart.sh          # full extraction
#   bash scripts/run_phase2_smart.sh --dry-run  # print commands only
#
# Strategy:
#   - CPU encoders (roberta, egemaps, openface, vit): parallel=1 (fast enough)
#   - WavLM on GPU: must be serial (parallel=1) to avoid CUDA OOM
#   - DAIC ViT: impossible — DAIC-WOZ has no video files, only CLNF tracking data
#   - FI RoBERTa: now enabled — transcriptions exist for train/val (6000+2000 clips)
#     Test split transcription zip is encrypted; skips roberta for test
#   - FI ViT: enabled — 10,000 mp4 files exist; ViT produces non-zero [1536] features
#   - FI test uses 'interview' column mapped to 'openness' → fixed in Python
#   - Visualization uses dataset prefix in filenames to avoid overwrites
#   - Runs sequentially to avoid CPU contention on shared hardware
set -e
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then DRY_RUN=true; fi

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
# BLOCK 4: FI eGeMAPS + OpenFace + RoBERTa + ViT (CPU)
#   RoBERTa: works for train/val (6k+2k transcriptions), test zip encrypted → skip
#   ViT: all 10k videos work, ~0.4s/clip on CPU
# =============================================================
echo "=== BLOCK 4: FI eGeMAPS + OpenFace + RoBERTa + ViT (CPU) ===" | tee -a "$LOG"
for split in train val test; do
    run scripts/phase02_preprocess.py --dataset fi --split "$split" --encoder egemaps --parallel 1 --device cpu
done
for split in train val test; do
    run scripts/phase02_preprocess.py --dataset fi --split "$split" --encoder openface --parallel 1 --device cpu
done
# FI RoBERTa: train+val only (test transcriptions encrypted)
for split in train val; do
    run scripts/phase02_preprocess.py --dataset fi --split "$split" --encoder roberta --parallel 1 --device cpu
done
# FI ViT: all splits
for split in train val test; do
    run scripts/phase02_preprocess.py --dataset fi --split "$split" --encoder vit --parallel 1 --device cpu
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
