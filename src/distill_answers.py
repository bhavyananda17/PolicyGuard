"""Self-distillation for v4: replace teacher-written final answers with the untrained model's own
always-search (naive RAG) answers, where those are good.

Why: v1-v3 learned short answers that gpt-oss wrote from the source passage, not from the search
results the model actually sees, and they scored well below the untrained model in always-search
mode. Here the untrained model answers each training question from the same 4 search results the
trajectory shows, and its answer replaces the teacher's only if:
  - the trajectory's search results are the same chunks the naive-RAG run saw,
  - it does not say "not found",
  - it covers the gold answer (content-word overlap >= 0.6) and at least one key fact,
  - for superseded items, it does not state the replaced rule,
  - it cites only chunks it was shown and is at most 120 words.
Otherwise the teacher answer stays. Trajectories with several searches keep the teacher answer.

    python src/distill_answers.py     # data/sft/train.jsonl + results/distill_train_rag.jsonl -> data/sft/train_v4.jsonl
"""
import json
import os
import re

from agent import CITE
from score import fact_hit, old_rule_used, says_not_found, toks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean(a):
    a = a.replace("**", "").replace("*", "")
    a = re.sub(r"\s+", " ", a).strip()
    return a


RAG_FILE = os.environ.get("RAG_FILE", "results/distill_train_rag.jsonl")
SFT_IN = os.environ.get("SFT_IN", "data/sft/train.jsonl")
SFT_OUT = os.environ.get("SFT_OUT", "data/sft/train_v4.jsonl")
MAX_WORDS = int(os.environ.get("MAX_WORDS", 120))


def main():
    with open(os.path.join(ROOT, "data/qa/train.jsonl")) as f:
        items = {json.loads(l)["question"]: json.loads(l) for l in f}
    with open(os.path.join(ROOT, RAG_FILE)) as f:
        rag = {json.loads(l)["id"]: json.loads(l) for l in f}
    with open(os.path.join(ROOT, SFT_IN)) as f:
        trajs = [json.loads(l) for l in f]

    stats = {"replaced": 0, "kept_teacher": 0, "skip_multi_search": 0, "skip_other_results": 0, "reject_quality": 0}
    out = []
    for t in trajs:
        m = t["messages"]
        it = items.get(m[1]["content"])
        final = m[-1]["content"]
        if t["type"] == "not_covered" or it is None or "Final Answer:" not in final:
            out.append(t)
            continue
        obs = [x["content"] for x in m if x["role"] == "user" and x["content"].startswith("Observation:")]
        if len(obs) != 1:
            stats["skip_multi_search"] += 1
            out.append(t)
            continue
        shown = {int(c) for c in re.findall(r"^\[chunk (\d+)\]", obs[0], re.M)}
        r = rag.get(it["id"])
        if not r or set(r["searches"][0]["chunks"]) != shown:
            stats["skip_other_results"] += 1
            out.append(t)
            continue
        a = clean(r["answer"])
        ref = set(toks(it["answer"]))
        cited = {int(c) for c in CITE.findall(a)}
        ok = (not says_not_found(a) and ref and len(ref & set(toks(a))) / len(ref) >= 0.6
              and any(fact_hit(f, a) for f in it["facts"]) and len(a.split()) <= MAX_WORDS
              and cited and cited <= shown
              and not (it["type"] == "superseded" and old_rule_used(it, a)))
        if not ok:
            stats["reject_quality"] += 1
            out.append(t)
            continue
        thought = final.split("Final Answer:")[0].rstrip()
        m = m[:-1] + [{"role": "assistant", "content": f"{thought}\nFinal Answer: {a}"}]
        out.append(dict(t, messages=m, answer_source="self_distilled"))
        stats["replaced"] += 1
    stats["kept_teacher"] = sum(1 for t in out if t.get("answer_source") != "self_distilled"
                                and t["type"] != "not_covered")
    with open(os.path.join(ROOT, SFT_OUT), "w") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
