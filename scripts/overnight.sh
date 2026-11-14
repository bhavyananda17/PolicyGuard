#!/bin/bash
# Overnight pipeline: dataset -> smoke test -> base baselines -> MLX LoRA training -> tuned eval -> scores.
# Stops at the first failure (set -e). Every stage is resumable, so re-running skips finished work.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(grep ^HF_TOKEN= .env | cut -d= -f2- | tr -d '"')
mkdir -p logs results
stamp() { echo "[$(date '+%H:%M:%S')] $*"; }

stamp "waiting for question generation to finish"
while pgrep -f gen_questions.py > /dev/null; do sleep 30; done

stamp "build dataset"
python src/build_dataset.py
python src/make_mlx_data.py

stamp "smoke test (2 questions, agent mode)"
rm -f results/smoke.jsonl
python src/run_agent.py --mode agent --limit 2 --name smoke
python src/score.py results/smoke.jsonl

for mode in agent agent_force naive_rag no_tools; do
  stamp "baseline: base_$mode"
  python src/run_agent.py --mode "$mode" > "logs/base_$mode.log" 2>&1
done
python src/score.py results/base_*.jsonl

stamp "LoRA training (MLX)"
N=$(wc -l < data/mlx/train.jsonl)
ITERS=$(( N * 2 / 2 ))            # about 2 epochs at batch size 2
[ "$ITERS" -lt 200 ] && ITERS=200
[ "$ITERS" -gt 1200 ] && ITERS=1200
sed "s/^iters: .*/iters: $ITERS/" configs/lora.yaml > configs/lora_run.yaml
stamp "train examples=$N iters=$ITERS"
mlx_lm.lora -c configs/lora_run.yaml > logs/train.log 2>&1

for mode in agent agent_force; do
  stamp "tuned: tuned_$mode"
  python src/run_agent.py --mode "$mode" --adapter adapters/policyguard > "logs/tuned_$mode.log" 2>&1
done

stamp "final scores"
python src/score.py results/base_*.jsonl results/tuned_*.jsonl
stamp "ALL DONE"
