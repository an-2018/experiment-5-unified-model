#!/bin/bash
# Phase 2 Full Extraction - runs each encoder separately on CPU (avoids CUDA OOM)
# CPU is fine for text (fast), eGeMAPS (librosa), and video_openface (CLNF parsing)
# MOSEI is fast (pre-extracted)
LOG="/home/anilson/thesis/thesis-experiment-5-unified-model/data/phase2_extraction.log"
cd /home/anilson/thesis/thesis-experiment-5-unified-model

echo "STARTED: $(date)" > "$LOG"

run() {
    local dataset=$1 split=$2 enc=$3
    echo "[$(date)] --- $dataset/$split $enc ---" | tee -a "$LOG"
    uv run python scripts/phase02_preprocess.py \
        --dataset "$dataset" --split "$split" --encoder "$enc" \
        --parallel 1 --device cpu >> "$LOG" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date)] *** $dataset/$split $enc FAILED (rc=$rc) ***" | tee -a "$LOG"
    else
        echo "[$(date)] --- $dataset/$split $enc OK ---" | tee -a "$LOG"
    fi
    return $rc
}

# === MOSEI (fast, pre-extracted) ===
run mosei all roberta
run mosei all egemaps
run mosei all wavlm
run mosei all openface
run mosei all vit

# === DAIC text+video (fast on CPU) ===
for split in train val test; do
    run daic "$split" roberta
    run daic "$split" openface
done

# === DAIC audio (slow - librosa + WavLM on CPU) ===
for split in train val test; do
    run daic "$split" egemaps
    run daic "$split" wavlm
done

# === FI (text skips, video uses mp4, audio from video) ===
for split in train val test; do
    run fi "$split" openface   # CLNF only for DAIC, skip for FI
    run fi "$split" vit
done
for split in train val test; do
    run fi "$split" egemaps
    run fi "$split" wavlm
done

echo "COMPLETED: $(date)" >> "$LOG"
echo "DONE - see $LOG"
