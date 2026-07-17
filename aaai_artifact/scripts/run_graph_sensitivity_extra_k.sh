#!/usr/bin/env bash
# Runs after run_ggmoe_ablation_real.sh completes: adds the two extra K values
# (K=5, K=20, inductive graph, matching the V0/V3 protocol) needed for the full
# graph-sensitivity sweep (K in {5,10,15,20}) requested in
# context/detailed_implementation_plan.md Task 1.2, then regenerates
# artifacts/tables/graph_sensitivity.csv with real density + real performance.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for main V0-V4 ablation to finish..."
while ! grep -q "ALL VARIANTS COMPLETE" logs/ggmoe_real_orchestrator.log 2>/dev/null; do
  sleep 60
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Main ablation done. Merging real V0-V4 results and regenerating figure..."
uv run python scripts/merge_ggmoe_real_results.py
uv run python -c "
import sys
sys.path.insert(0, 'scripts')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from phase06_graph import plot_ablation_comparison
out = Path('artifacts/figures/phase_06_graph')
out.mkdir(parents=True, exist_ok=True)
plot_ablation_comparison(Path('.'), out)
"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Real ggmoe_results.csv + figure ready. Starting extra K runs."

RESULTS_DIR=artifacts/tables/ggmoe_ablation_real
FIG_BASE=artifacts/figures/phase_07_joint_training_ablation

declare -A KVAL=( [K5]=5 [K20]=20 )

for NAME in K5 K20; do
  k=${KVAL[$NAME]}
  echo "=================================================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $NAME: graph_type=inductive k=$k router=graphsage"
  echo "=================================================================="

  uv run python scripts/phase07_joint_training.py \
    --router graphsage --graph_type inductive --k "$k" \
    --output_dir "$FIG_BASE/$NAME" \
    > "logs/ggmoe_real_${NAME}_inductive_k${k}.log" 2>&1
  rc=$?

  if [ -f artifacts/tables/phase07_results.csv ]; then
    cp artifacts/tables/phase07_results.csv "$RESULTS_DIR/${NAME}_results.csv"
  fi
  if [ -f artifacts/tables/phase07_best.pt ]; then
    cp artifacts/tables/phase07_best.pt "$RESULTS_DIR/${NAME}_best.pt"
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished $NAME (rc=$rc)"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Regenerating graph_sensitivity.csv with real K=5/10/15/20 data..."
uv run python scripts/phase13_graph_sensitivity.py > logs/phase13_graph_sensitivity_real.log 2>&1

echo "EXTRA K RUNS + SENSITIVITY SWEEP COMPLETE"
