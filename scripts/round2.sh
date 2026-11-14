#!/bin/bash
# Round 2: fewer "not found" trajectories (18%) + 60 "search again" trajectories -> adapters/policyguard_v2.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(grep ^HF_TOKEN= .env | cut -d= -f2- | tr -d '"')
stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
N=$(wc -l < data/mlx/train.jsonl | tr -d ' ')
sed -e "s/^iters: .*/iters: $N/" -e "s#^adapter_path: .*#adapter_path: adapters/policyguard_v2#" configs/lora.yaml > configs/lora_v2.yaml
stamp "train v2 examples=$N iters=$N"
mlx_lm.lora -c configs/lora_v2.yaml > logs/train_v2.log 2>&1
stamp "eval tuned_v2_agent"
python src/run_agent.py --mode agent --adapter adapters/policyguard_v2 --name tuned_v2_agent > logs/tuned_v2_agent.log 2>&1
python src/score.py results/base_*.jsonl results/tuned_agent.jsonl results/tuned_v2_agent.jsonl
stamp "ROUND2 DONE"
