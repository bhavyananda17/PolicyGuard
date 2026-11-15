#!/bin/zsh
# Round 5: same models, 6 search results instead of 4 (no retraining).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
export TOPK=6
python src/run_agent.py --mode agent --adapter adapters/policyguard_v4 --name tuned_v4_agent_k6
python src/run_agent.py --mode naive_rag --name base_naive_rag_k6
echo ROUND5_DONE
