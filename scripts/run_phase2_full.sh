#!/bin/bash
# Full Phase 2 extraction — sequential by dataset to isolate issues
set -e
LOG="/home/anilson/thesis/thesis-experiment-5-unified-model/data/full_extraction.log"

echo "=== FULL PHASE 2 EXTRACTION ===" > "$LOG"
echo "Started: $(date)" >> "$LOG"

cd /home/anilson/thesis/thesis-experiment-5-unified-model

echo "--- 1/6 DAIC train ---" | tee -a "$LOG"
uv run python scripts/phase02_preprocess.py --dataset daic --split train --encoder all --parallel 4 --device cuda >> "$LOG" 2>&1

echo "--- 2/6 DAIC val ---" | tee -a "$LOG"  
uv run python scripts/phase02_preprocess.py --dataset daic --split val --encoder all --parallel 4 --device cuda >> "$LOG" 2>&1

echo "--- 3/6 DAIC test ---" | tee -a "$LOG"
uv run python scripts/phase02_preprocess.py --dataset daic --split test --encoder all --parallel 4 --device cuda >> "$LOG" 2>&1

echo "--- 4/6 MOSEI all splits ---" | tee -a "$LOG"
uv run python scripts/phase02_preprocess.py --dataset mosei --split all --encoder all --parallel 8 --device cuda >> "$LOG" 2>&1

echo "--- 5/6 FI train ---" | tee -a "$LOG"
uv run python scripts/phase02_preprocess.py --dataset fi --split train --encoder all --parallel 4 --device cuda >> "$LOG" 2>&1

echo "--- 6/6 FI val+test ---" | tee -a "$LOG"
uv run python scripts/phase02_preprocess.py --dataset fi --split val --encoder all --parallel 4 --device cuda >> "$LOG" 2>&1
uv run python scripts/phase02_preprocess.py --dataset fi --split test --encoder all --parallel 4 --device cuda >> "$LOG" 2>&1

echo "=== COMPLETED: $(date) ===" | tee -a "$LOG"
