#!/usr/bin/env bash
# Real V0-V4 graph-routing ablation, using the validated phase07_joint_training.py
# (real labels, real evaluate()) instead of the fabricated phase06_graph.py path.
# Variant -> (graph_type, k) mapping matches chapter_8.tex / journal_paper.tex:
#   V0 inductive-k10, V1 split-local-k10, V2 transductive-k10,
#   V3 inductive-k15, V4 split-local-k15  (router=graphsage throughout)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RESULTS_DIR=artifacts/tables/ggmoe_ablation_real
FIG_BASE=artifacts/figures/phase_07_joint_training_ablation
mkdir -p "$RESULTS_DIR" "$FIG_BASE"

declare -A GRAPH_TYPE=( [V0]=inductive [V1]=split-local [V2]=transductive [V3]=inductive [V4]=split-local )
declare -A KVAL=( [V0]=10 [V1]=10 [V2]=10 [V3]=15 [V4]=15 )

for V in V0 V1 V2 V3 V4; do
  gt=${GRAPH_TYPE[$V]}
  k=${KVAL[$V]}
  echo "=================================================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $V: graph_type=$gt k=$k router=graphsage"
  echo "=================================================================="

  uv run python scripts/phase07_joint_training.py \
    --router graphsage --graph_type "$gt" --k "$k" \
    --output_dir "$FIG_BASE/$V" \
    > "logs/ggmoe_real_${V}_${gt}_k${k}.log" 2>&1
  rc=$?

  if [ -f artifacts/tables/phase07_results.csv ]; then
    cp artifacts/tables/phase07_results.csv "$RESULTS_DIR/${V}_results.csv"
  fi
  if [ -f artifacts/tables/phase07_best.pt ]; then
    cp artifacts/tables/phase07_best.pt "$RESULTS_DIR/${V}_best.pt"
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished $V (rc=$rc)"
done

echo "ALL VARIANTS COMPLETE"
