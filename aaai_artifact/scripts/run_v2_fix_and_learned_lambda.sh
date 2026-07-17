#!/usr/bin/env bash
# 1) Rerun V2 (transductive) after fixing the split-boundary bug in construct_graphs.
# 2) Run a learned-per-task-lambda variant on the V0 config (inductive-k10) for a
#    clean A/B comparison against the fixed lambda=0.5 baseline.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RESULTS_DIR=artifacts/tables/ggmoe_ablation_real
FIG_BASE=artifacts/figures/phase_07_joint_training_ablation
mkdir -p "$RESULTS_DIR" "$FIG_BASE"

echo "=================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting V2-fixed: graph_type=transductive k=10 router=graphsage (post-fix)"
echo "=================================================================="
uv run python scripts/phase07_joint_training.py \
  --router graphsage --graph_type transductive --k 10 \
  --output_dir "$FIG_BASE/V2_fixed" \
  > "logs/ggmoe_real_V2_fixed_transductive_k10.log" 2>&1
rc=$?
[ -f artifacts/tables/phase07_results.csv ] && cp artifacts/tables/phase07_results.csv "$RESULTS_DIR/V2_fixed_results.csv"
[ -f artifacts/tables/phase07_best.pt ] && cp artifacts/tables/phase07_best.pt "$RESULTS_DIR/V2_fixed_best.pt"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished V2-fixed (rc=$rc)"

echo "=================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting V0-learned: graph_type=inductive k=10 router=graphsage graph_weight_mode=learned"
echo "=================================================================="
uv run python scripts/phase07_joint_training.py \
  --router graphsage --graph_type inductive --k 10 --graph_weight_mode learned \
  --output_dir "$FIG_BASE/V0_learned" \
  > "logs/ggmoe_real_V0_learned_inductive_k10.log" 2>&1
rc=$?
[ -f artifacts/tables/phase07_results.csv ] && cp artifacts/tables/phase07_results.csv "$RESULTS_DIR/V0_learned_results.csv"
[ -f artifacts/tables/phase07_best.pt ] && cp artifacts/tables/phase07_best.pt "$RESULTS_DIR/V0_learned_best.pt"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished V0-learned (rc=$rc)"

echo "V2 FIX + LEARNED LAMBDA RUNS COMPLETE"
