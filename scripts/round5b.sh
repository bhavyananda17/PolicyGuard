#!/bin/zsh
# Round 5b (v5): train on 6-result searches, 40% keyword first queries, fuller self-distilled answers.
# Unattended: distill -> data -> train -> evaluate at k=6 and k=4 -> judge -> score.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
step() { echo "=== $(date +%H:%M) $*"; }

step "1/7 naive RAG (k=6) on train questions, for self-distilled answers"
TOPK=6 python src/run_agent.py --mode naive_rag --split train --name distill_train_rag_k6

step "2/7 build v5 trajectories (k=6, 40% keyword first queries)"
(cd src && BUILD_K=6 KEYWORD_SHARE=0.4 SFT_OUT=data/sft/train_v5_base.jsonl python build_dataset.py | tail -5)
(cd src && RAG_FILE=results/distill_train_rag_k6.jsonl SFT_IN=data/sft/train_v5_base.jsonl \
   SFT_OUT=data/sft/train_v5.jsonl MAX_WORDS=150 python distill_answers.py)

step "3/7 MLX training files"
SFT_FILE=data/sft/train_v5.jsonl MLX_OUT=data/mlx_v5 NC_SHARE=0.2 python src/make_mlx_data.py
N=$(wc -l < data/mlx_v5/train.jsonl | tr -d ' ')
ITERS=$(( (3 * N + 1) / 2 ))
sed -e "s#data/mlx_v4#data/mlx_v5#; s#adapters/policyguard_v4#adapters/policyguard_v5#; s#^iters: .*#iters: $ITERS#; s#^max_seq_length: .*#max_seq_length: 4608#" \
    configs/lora_v4.yaml > configs/lora_v5.yaml
echo "examples=$N iters=$ITERS"

step "4/7 train v5"
if ! mlx_lm.lora -c configs/lora_v5.yaml; then
  step "training failed (likely memory); retrying with batch 1 x 2 accumulation"
  sed -i '' -e "s#^batch_size: .*#batch_size: 1#" configs/lora_v5.yaml
  echo "grad_accumulation_steps: 2" >> configs/lora_v5.yaml
  sed -i '' -e "s#^iters: .*#iters: $(( 3 * N ))#" configs/lora_v5.yaml
  mlx_lm.lora -c configs/lora_v5.yaml
fi

step "5/7 evaluate v5 at k=6 and k=4"
TOPK=6 python src/run_agent.py --mode agent --adapter adapters/policyguard_v5 --name tuned_v5_agent_k6
TOPK=4 python src/run_agent.py --mode agent --adapter adapters/policyguard_v5 --name tuned_v5_agent

step "6/7 judge"
(cd src && python judge.py ../results/tuned_v5_agent_k6.jsonl ../results/tuned_v5_agent.jsonl)

step "7/7 score"
python src/score.py results/base_no_tools.jsonl results/base_agent.jsonl results/base_agent_force.jsonl \
  results/base_naive_rag.jsonl results/base_naive_rag_k6.jsonl results/tuned_v2_agent_fixparse.jsonl \
  results/tuned_v3_agent.jsonl results/tuned_v4_agent.jsonl results/tuned_v4_agent_k6.jsonl \
  results/tuned_v5_agent.jsonl results/tuned_v5_agent_k6.jsonl
step ROUND5B_DONE
