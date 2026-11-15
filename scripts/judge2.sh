#!/bin/zsh
# Second judge (gpt-oss-120b) on the three main setups, to check agreement with the qwen judge.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
cd src && JUDGE_MODEL=openai/gpt-oss-120b JUDGE_DIR=results/judge2 \
  python judge.py ../results/base_naive_rag.jsonl ../results/base_agent.jsonl ../results/tuned_v4_agent.jsonl
echo JUDGE2_DONE
