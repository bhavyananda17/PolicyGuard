"""Validate the generated questions, split train/test, and build agent trajectories.

1. Validation: drop near-duplicate questions (MiniLM cosine > 0.9) and empty items.
2. Split: items that share a source chunk (or, for superseded items, a topic) stay on the same
   side, so the test set never asks about a passage the model trained on. About 25% of each
   type goes to test; not_covered is split at random.
3. Trajectories (train only): real searches of the index. The teacher-written answer
   (gpt-oss, written from the source passage when the question was generated) becomes the
   final answer, but only if the source passage was actually retrieved. If the first search
   misses it, a second, shorter query is tried; if that misses too, the item is dropped.

    python src/build_dataset.py      # writes data/qa/{train,test}.jsonl and data/sft/train.jsonl
"""
import json
import os
import random
import re

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from agent import SYSTEM
from gen_questions import verbatim
from search import Searcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA = os.path.join(ROOT, "data/qa")
TYPES = ["single_hop", "multi_hop", "superseded", "not_covered"]
TEST_SHARE = 0.25
RETRY_MAX = 60
K = 4
rnd = random.Random(7)

STOP = set("the a an of to in on for and or is are be by with as at from that this which shall any its it under such "
           "will can may must if not no all per has have been their them they other than".split())


def short_query(text, n=8):
    words = [w for w in re.findall(r"[A-Za-z0-9%-]+", text) if w.lower() not in STOP]
    return " ".join(words[:n])


def load_items():
    items = []
    for t in TYPES:
        p = os.path.join(QA, f"raw_{t}.jsonl")
        if os.path.exists(p):
            with open(p) as f:
                items += [json.loads(l) for l in f]
    ex_path = os.path.join(QA, "excluded_topics.json")
    if os.path.exists(ex_path):                       # manual review, reasons in the file
        with open(ex_path) as f:
            excluded = set(json.load(f)) - {"_why"}
        items = [it for it in items if it.get("topic") not in excluded]
    return items


def dedupe(items, model):
    vecs = model.encode([it["question"] for it in items], normalize_embeddings=True, convert_to_numpy=True)
    keep, kept_vecs = [], []
    for it, v in zip(items, vecs):
        if kept_vecs and float(np.max(np.stack(kept_vecs) @ v)) > 0.9:
            continue
        keep.append(it)
        kept_vecs.append(v)
    return keep


def split(items):
    """Union-find over items that share a source chunk or a superseded topic."""
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, it in enumerate(items):
        keys = [f"c{c}" for c in it["source_chunks"]]
        if it["type"] == "superseded":
            keys.append(f"o{it['old_chunk']}")
        for k in keys:
            if k in owner:
                parent[find(i)] = find(owner[k])
            else:
                owner[k] = i
    groups = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)

    test = set()
    for t in TYPES:
        idx = [i for i in range(len(items)) if items[i]["type"] == t]
        target = round(len(idx) * TEST_SHARE)
        gs = sorted({find(i) for i in idx}, key=lambda g: rnd.random())
        for g in gs:
            have = sum(items[i]["type"] == t for i in test)
            if have >= target:
                break
            add = sum(items[i]["type"] == t for i in groups[g])
            if have + add > target + 2:                 # skip groups that would overshoot
                continue
            test.update(groups[g])
    return [it for i, it in enumerate(items) if i not in test], [it for i, it in enumerate(items) if i in test]


def grounded(hits, it, target):
    """-> (retrieved chunk ids to cite, target chunks covered). A target chunk is covered if it was
    retrieved, or an overlapping neighbour that contains one of the item's key facts was."""
    cite_ids, covered = set(), set()
    for h in hits:
        for t in target:
            if h["chunk_id"] == t or (h["status"] == "current" and abs(h["chunk_id"] - t) <= 1
                                      and any(verbatim(f, h["text"]) for f in it["facts"])):
                cite_ids.add(h["chunk_id"])
                covered.add(t)
    return cite_ids, covered


def cite(ids):
    return " ".join(f"[chunk {c}]" for c in ids)


def trajectory(it, s):
    """-> list of chat messages, or None if the source passage cannot be retrieved."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": it["question"]}]
    need = set(it["source_chunks"])
    queries = it.get("sub_queries") or [it["question"]]
    cited, covered, old_seen = set(), set(), False
    first = rnd.choice(["I need to check what the IRDAI documents say about this.",
                        "I should search the IRDAI rules for this.",
                        "Let me look this up in the IRDAI documents."])
    for qi, q in enumerate(queries):
        thought = first if qi == 0 else "I have the first part. Now I need the second rule."
        msgs.append({"role": "assistant", "content": f'Thought: {thought}\nAction: search_docs("{q}")'})
        hits = s.search(q, K)
        msgs.append({"role": "user", "content": "Observation:\n" + s.render(hits)})
        target = need - covered                      # multi-hop: whichever source is still missing
        got, cov = grounded(hits, it, target)
        old_seen |= any(h["status"] == "superseded" for h in hits)
        if target and not got:
            # one retry with a shorter keyword query built from a key fact of the target chunk
            ctext = s.chunks[sorted(target)[0]]["text"]
            fact = next((f for f in it["facts"] if verbatim(f, ctext)), it["facts"][0])
            q2 = short_query(fact)
            msgs.append({"role": "assistant", "content":
                         f'Thought: These results do not answer the question. I will search with more specific terms.\n'
                         f'Action: search_docs("{q2}")'})
            hits = s.search(q2, K)
            msgs.append({"role": "user", "content": "Observation:\n" + s.render(hits)})
            got, cov = grounded(hits, it, target)
            old_seen |= any(h["status"] == "superseded" for h in hits)
        if target and not got:
            return None
        cited |= got
        covered |= cov
    if covered != need:
        return None

    if it["type"] == "not_covered":
        final = ("Thought: None of the results answer this question, so I should not guess.\n"
                 "Final Answer: I could not find this in the IRDAI health-insurance documents I can search, "
                 "so I can't answer it reliably.")
    else:
        if it["type"] == "superseded" and old_seen:
            thought = ("The 2020 standardization circular and the May 2024 master circular give different rules here. "
                       "The 2024 circular replaced the 2020 one, so I follow the 2024 rule.")
        elif it["type"] == "multi_hop":
            thought = "I now have both rules and can combine them."
        else:
            thought = "The results answer the question."
        final = f"Thought: {thought}\nFinal Answer: {it['answer']} {cite(sorted(cited))}"
    msgs.append({"role": "assistant", "content": final})
    return msgs


def main():
    s = Searcher()
    items = [it for it in load_items() if it.get("question") and it.get("answer")]
    n0 = len(items)
    items = dedupe(items, s.model)
    # cap "not covered" at 25% of the final set (design: 20-25%)
    nc = [it for it in items if it["type"] == "not_covered"]
    rest = [it for it in items if it["type"] != "not_covered"]
    cap = round(len(rest) * 0.25 / 0.75)
    if len(nc) > cap:
        nc = rnd.sample(nc, cap)
    items = rest + nc
    train, test = split(items)
    os.makedirs(os.path.join(ROOT, "data/sft"), exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        with open(os.path.join(QA, f"{name}.jsonl"), "w") as f:
            for i, it in enumerate(rows):
                f.write(json.dumps(dict(it, id=f"{name}-{i:04d}"), ensure_ascii=False) + "\n")

    kept, dropped = [], {t: 0 for t in TYPES}
    for it in train:
        msgs = trajectory(it, s)
        if msgs is None:
            dropped[it["type"]] += 1
        else:
            kept.append({"messages": msgs, "type": it["type"]})
    # "search again instead of giving up": a vague first query that misses the source, then a
    # sharper query that finds it. Teaches the model to retry before saying "not found".
    rr = random.Random(11)
    retry = []
    for it in train:
        if it["type"] not in ("single_hop", "superseded") or len(retry) >= RETRY_MAX:
            continue
        need = set(it["source_chunks"])
        words = [w for w in re.findall(r"[A-Za-z]+", it["question"]) if w.lower() not in STOP and len(w) > 3]
        if len(words) < 3:
            continue
        q1 = " ".join(rr.sample(words, 2))
        h1 = s.search(q1, K)
        if grounded(h1, it, need)[0]:
            continue                                   # the vague query already worked: not a retry case
        ctext = s.chunks[sorted(need)[0]]["text"]
        fact = next((f for f in it["facts"] if verbatim(f, ctext)), it["facts"][0])
        q2 = short_query(fact)
        h2 = s.search(q2, K)
        got, cov = grounded(h2, it, need)
        if cov != need:
            continue
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": it["question"]},
                {"role": "assistant", "content": f'Thought: I should search the IRDAI rules for this.\nAction: search_docs("{q1}")'},
                {"role": "user", "content": "Observation:\n" + s.render(h1)},
                {"role": "assistant", "content": "Thought: These results do not answer the question. I will search with "
                                                 f'more specific terms before giving up.\nAction: search_docs("{q2}")'},
                {"role": "user", "content": "Observation:\n" + s.render(h2)},
                {"role": "assistant", "content": f"Thought: The results answer the question.\nFinal Answer: {it['answer']} "
                                                 f"{cite(sorted(got))}"}]
        retry.append({"messages": msgs, "type": it["type"], "aug": "retry"})
    kept += retry

    with open(os.path.join(ROOT, "data/sft/train.jsonl"), "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    count = lambda rows: {t: sum(r["type"] == t for r in rows) for t in TYPES}
    report = {"generated": n0, "after_dedupe": len(items), "train_items": count(train), "test_items": count(test),
              "trajectories": count(kept), "retry_augmented": len(retry), "dropped_not_retrieved": dropped,
              "not_covered_share_train": round(count(kept)["not_covered"] / max(1, len(kept)), 3)}
    with open(os.path.join(QA, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
