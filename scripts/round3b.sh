#!/bin/bash
# Evaluate the step-800 checkpoint of v3 (lowest validation loss region) after round 3 finishes.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false
while pgrep -f round3.sh > /dev/null; do sleep 20; done
python src/run_agent.py --mode agent --adapter adapters/policyguard_v3_800 --name tuned_v3_800_agent > logs/tuned_v3_800.log 2>&1
python src/score.py results/base_*.jsonl results/tuned_agent.jsonl results/tuned_v2_agent*.jsonl results/tuned_v3*.jsonl
echo "ROUND3B DONE"
