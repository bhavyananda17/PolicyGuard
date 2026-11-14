#!/bin/bash
# Round 3: parser recovers unclosed search calls; training adds 60 "answer is the last result"
# and 25 near-miss "not found" trajectories; 3 epochs -> adapters/policyguard_v3.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(grep ^HF_TOKEN= .env | cut -d= -f2- | tr -d '"')
stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
N=$(wc -l < data/mlx/train.jsonl | tr -d ' ')
ITERS=$(( N * 3 / 2 ))
sed -e "s/^iters: .*/iters: $ITERS/" -e "s#^adapter_path: .*#adapter_path: adapters/policyguard_v3#" \
    -e "s/^save_every: .*/save_every: 400/" configs/lora.yaml > configs/lora_v3.yaml
stamp "train v3 examples=$N iters=$ITERS"
mlx_lm.lora -c configs/lora_v3.yaml > logs/train_v3.log 2>&1
stamp "eval tuned_v3_agent"
python src/run_agent.py --mode agent --adapter adapters/policyguard_v3 --name tuned_v3_agent > logs/tuned_v3_agent.log 2>&1
stamp "re-eval v2 with the parser fix"
python src/run_agent.py --mode agent --adapter adapters/policyguard_v2 --name tuned_v2_agent_fixparse > logs/tuned_v2_fix.log 2>&1
python src/score.py results/base_*.jsonl results/tuned_agent.jsonl results/tuned_v2_agent*.jsonl results/tuned_v3_agent.jsonl
stamp "ROUND3 DONE"
