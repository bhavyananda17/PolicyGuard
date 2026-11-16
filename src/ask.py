"""Ask PolicyGuard one question from the command line and print the full agent transcript.

    python src/ask.py "What is the moratorium period for health insurance policies?"
    python src/ask.py --adapter none "..."        # the untrained model, for comparison
"""
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from run_agent import LM, MODEL, run_one
from search import Searcher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--adapter", default="adapters/policyguard_v5", help='LoRA adapter dir, or "none"')
    ap.add_argument("--mode", default="agent", choices=["agent", "agent_force", "naive_rag", "no_tools"])
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    adapter = None if a.adapter == "none" else os.path.join(root, a.adapter)
    lm = LM(MODEL, adapter)
    r = run_one(lm, Searcher(), {"question": a.question}, a.mode)
    for m in r["transcript"][1:]:
        print(f"--- {m['role']}\n{m['content'][:1500]}\n")
    print("=== Answer:", r["answer"])


if __name__ == "__main__":
    main()
