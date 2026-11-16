"""Agentic trajectory metrics per question type: task success, search (tool-call) quality, latency.

    python src/agent_metrics.py base_agent base_naive_rag tuned_v4_agent tuned_v5_agent

- task success: answerable questions graded "correct" by the judge (results/judge/<run>.jsonl);
  unanswerable questions answered with "not found".
- valid calls: share of questions whose transcript parsed cleanly (no invalid Action format).
- search hit: answerable questions where at least one search returned a chunk holding the answer
  (a source chunk, or one containing a key fact; see score.holds_fact).
- latency: mean wall-clock seconds per question (MLX, greedy, M5 Pro), and searches per question.
Writes results/agent_metrics.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import EXCLUDED, holds_fact, says_not_found  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = ["single_hop", "multi_hop", "superseded", "not_covered", "all"]


def load(path):
    with open(os.path.join(ROOT, path)) as f:
        return [json.loads(l) for l in f]


def main(runs):
    items = {x["id"]: x for x in load("data/qa/test.jsonl")}
    with open(os.path.join(ROOT, "data/qa/test_exclusions.json")) as f:
        EXCLUDED.update(json.load(f)["ids"])
    out = []
    for run in runs:
        rows = [r for r in load(f"results/{run}.jsonl") if r["id"] not in EXCLUDED]
        judge = {x["id"]: x["verdict"] for x in load(f"results/judge/{run}.jsonl")}
        for t in TYPES:
            sel = [r for r in rows if t == "all" or r["type"] == t]
            ok, hit, ans = [], [], []
            for r in sel:
                it = items[r["id"]]
                if r["type"] == "not_covered":
                    ok.append(says_not_found(r["answer"]))
                else:
                    ok.append(judge.get(r["id"]) == "correct")
                    ans.append(r)
                    hit.append(any(holds_fact(c, it) for s in r["searches"] for c in s["chunks"]))
            n = len(sel)
            out.append({"run": run, "type": t, "n": n,
                        "task_success": round(sum(ok) / n, 3),
                        "valid_calls": round(1 - sum(int(r.get("invalid_format", 0)) for r in sel) / n, 3),
                        "search_hit": round(sum(hit) / len(hit), 3) if hit else None,
                        "searches_per_q": round(sum(len(r["searches"]) for r in sel) / n, 2),
                        "seconds_per_q": round(sum(float(r["seconds"]) for r in sel) / n, 2)})
    cols = ["run", "type", "n", "task_success", "valid_calls", "search_hit", "searches_per_q", "seconds_per_q"]
    print(" | ".join(cols))
    for s in out:
        print(" | ".join(str(s[c]) for c in cols))
    with open(os.path.join(ROOT, "results/agent_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
