"""LLM judge for answer correctness on answerable test questions.

Word overlap with the reference answer penalises short correct answers and rewards long ones,
so each answer is also graded by an LLM (Groq, free tier) against the reference answer and the
verbatim key facts from the source passage. Refusals ("could not find") are graded "refused"
without a call. Unanswerable (not_covered) items are scored by score.py, not here.

    python src/judge.py results/base_naive_rag.jsonl results/tuned_v3_agent.jsonl
Writes results/judge/<run>.jsonl and prints correct / partial / wrong / refused rates with 95% CIs.
Every call is cached in data/cache/ (tag "judge"), so re-running costs nothing.
"""
import json
import os
import sys

from llm import chat, parse_json, USAGE
from score import boot, says_not_found

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGE = os.environ.get("JUDGE_MODEL", "qwen/qwen3.8-27b")
OUT = os.environ.get("JUDGE_DIR", "results/judge")    # second judge: JUDGE_DIR=results/judge2

PROMPT = """You grade answers about Indian health-insurance rules (IRDAI).

Question: {q}

Reference answer: {ref}
Key facts from the source rule: {facts}

Answer to grade: {ans}

Grade the answer against the reference and key facts only:
- "correct": states the rule correctly; it may be shorter or worded differently, extra detail is fine if not wrong
- "partial": right in part, but misses a key condition, number or date, or is vague
- "wrong": contradicts the reference, gives a different rule or number, or does not answer
Ignore citation markers like [chunk 12].
Reply with JSON only: {{"verdict": "correct" | "partial" | "wrong", "reason": "<at most 12 words>"}}"""


def grade(item, answer):
    if says_not_found(answer):
        return {"verdict": "refused", "reason": "said it could not find the answer"}
    msg = PROMPT.format(q=item["question"], ref=item["answer"], facts=" | ".join(item["facts"])[:900],
                        ans=answer[:1200])
    out = parse_json(chat([{"role": "user", "content": msg}], model=JUDGE, max_tokens=600 if JUDGE.startswith("openai/") else 120,
                          temperature=0.0, json_mode=True, tag="judge")) or {}
    v = out.get("verdict", "wrong")
    return {"verdict": v if v in ("correct", "partial", "wrong") else "wrong", "reason": out.get("reason", "")}


def main():
    with open(os.path.join(ROOT, "data/qa/test.jsonl")) as f:
        items = {json.loads(l)["id"]: json.loads(l) for l in f}
    with open(os.path.join(ROOT, "data/qa/test_exclusions.json")) as f:
        excluded = set(json.load(f)["ids"])
    os.makedirs(os.path.join(ROOT, OUT), exist_ok=True)
    summary = []
    print("run | n | correct | correct_ci | correct+partial | wrong | refused")
    for path in sys.argv[1:]:
        run = os.path.basename(path)[:-6]
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        rows = [r for r in rows if r["id"] not in excluded and items[r["id"]]["type"] != "not_covered"]
        graded = []
        for r in rows:
            g = grade(items[r["id"]], r["answer"])
            graded.append(dict(id=r["id"], type=r["type"], **g))
        with open(os.path.join(ROOT, f"{OUT}/{run}.jsonl"), "w") as f:
            for g in graded:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        rate = lambda *vs: [float(g["verdict"] in vs) for g in graded]
        m = lambda v: round(sum(v) / len(v), 3)
        s = {"run": run, "n": len(graded), "judge": JUDGE, "correct": m(rate("correct")), "correct_ci": boot(rate("correct")),
             "correct_or_partial": m(rate("correct", "partial")), "wrong": m(rate("wrong")), "refused": m(rate("refused")),
             "by_type": {t: m([float(g["verdict"] == "correct") for g in graded if g["type"] == t])
                         for t in sorted({g["type"] for g in graded})}}
        summary.append(s)
        print(" | ".join(str(s[k]) for k in ("run", "n", "correct", "correct_ci", "correct_or_partial", "wrong", "refused")),
              flush=True)
    out = os.path.join(ROOT, OUT, "summary.json")
    old = []
    if os.path.exists(out):
        with open(out) as f:
            old = [s for s in json.load(f) if s["run"] not in {x["run"] for x in summary}]
    with open(out, "w") as f:
        json.dump(old + summary, f, indent=2)
    print("groq usage:", USAGE, file=sys.stderr)


if __name__ == "__main__":
    main()
