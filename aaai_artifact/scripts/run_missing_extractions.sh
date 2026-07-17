#!/bin/bash
# Sequential extraction runner for missing FI features
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_job() {
    local label="$1"
    local args="$2"
    local log="$ROOT/data/fi_${label}.log"
    echo ""
    echo "=========================================="
    echo "  STARTING: $label"
    echo "  $(date)"
    echo "=========================================="
    uv run python scripts/phase02_preprocess.py $args --parallel 1 --device cpu > "$log" 2>&1
    local status=$?
    echo "  FINISHED: $label (exit code $status) at $(date)"
    return $status
}

# Run jobs sequentially
run_job "roberta_val" "--dataset fi --split val --encoder roberta"
run_job "roberta_train" "--dataset fi --split train --encoder roberta"
run_job "vit_val" "--dataset fi --split val --encoder vit"
run_job "vit_train" "--dataset fi --split train --encoder vit"

echo ""
echo "=========================================="
echo "  ALL EXTRACTIONS COMPLETE"
echo "  $(date)"
echo "=========================================="
