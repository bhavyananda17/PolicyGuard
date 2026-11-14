#!/bin/bash
# Round 4 (v4): v3's data with self-distilled final answers -> adapters/policyguard_v4, then eval + judge.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(grep ^HF_TOKEN= .env | cut -d= -f2- | tr -d '"')
stamp() { echo "[$(date '+%H:%M:%S')] $*"; }
stamp "waiting for the untrained-model answers on the training questions"
while pgrep -f distill_train_rag > /dev/null; do sleep 20; done
[ "$(wc -l < results/distill_train_rag.jsonl)" -eq "$(wc -l < data/qa/train.jsonl)" ] || { echo "distill run incomplete"; exit 1; }
(cd src && python distill_answers.py)
SFT_FILE=data/sft/train_v4.jsonl MLX_OUT=data/mlx_v4 NC_SHARE=0.2 python src/make_mlx_data.py
N=$(wc -l < data/mlx_v4/train.jsonl | tr -d ' ')
ITERS=$(( N * 3 / 2 ))
sed -e "s/^iters: .*/iters: $ITERS/" -e "s#^adapter_path: .*#adapter_path: adapters/policyguard_v4#" \
    -e "s#^data: .*#data: data/mlx_v4#" -e "s/^save_every: .*/save_every: 400/" configs/lora.yaml > configs/lora_v4.yaml
stamp "train v4 examples=$N iters=$ITERS"
mlx_lm.lora -c configs/lora_v4.yaml > logs/train_v4.log 2>&1
stamp "eval tuned_v4_agent"
python src/run_agent.py --mode agent --adapter adapters/policyguard_v4 --name tuned_v4_agent > logs/tuned_v4_agent.log 2>&1
python src/score.py results/base_*.jsonl results/tuned_agent.jsonl results/tuned_v2_agent*.jsonl results/tuned_v3*.jsonl results/tuned_v4_agent.jsonl
stamp "judge v4"
(cd src && python judge.py ../results/tuned_v4_agent.jsonl)
stamp "ROUND4 DONE"
