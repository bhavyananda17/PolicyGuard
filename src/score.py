"""Score result files with deterministic metrics (no LLM judge) and bootstrap 95% intervals.

  answer_overlap    : share of the gold answer's content words found in the answer (answerable items)
  false_not_found   : share of answerable items where the model says it could not find the answer
  key_fact_recall   : share of an item's key facts present in the answer (content-word overlap >= 0.6)
  old_rule_rate     : superseded items whose answer contains the replaced rule (its distinctive numbers/terms)
  not_found_f1      : F1 for saying "not found" on not_covered items vs answering on answerable ones
  searches_per_q    : mean number of search_docs calls
  cites_source      : answerable items whose answer cites a source chunk of that item

    python src/score.py results/*.jsonl          # prints a table, writes results/summary.json
"""
import json
import os
import random
import re
import sys

from agent import CITE, NOT_FOUND

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP = set("the a an of to in on for and or is are be by with as at from that this which shall any its it under "
           "such will can may must if not no all per has have been their them they other than".split())
NUM_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
             "eight": "8", "nine": "9", "ten": "10", "thirty": "30", "sixty": "60"}


def toks(s):
    out = []
    for w in re.findall(r"[a-z0-9]+", s.lower()):
        w = NUM_WORDS.get(w, w)
        if w not in STOP and len(w) > 1 or w.isdigit():
            out.append(w)
    return out


def fact_hit(fact, answer):
    f, a = set(toks(fact)), set(toks(answer))
    return bool(f) and len(f & a) / len(f) >= 0.6


def nums(s):
    return {t for t in toks(s) if t.isdigit()}


def old_rule_used(item, answer):
    old = item.get("old_fact", "") + " " + item.get("old_rule", "")
    cur = " ".join(item["facts"]) + " " + item["answer"]
    distinct = nums(old) - nums(cur)
    if distinct:
        return bool(distinct & nums(answer))
    words = set(toks(old)) - set(toks(cur))
    return len(words & set(toks(answer))) >= 3


def says_not_found(answer):
    return bool(NOT_FOUND.search(answer or ""))


def boot(vals, n=1000):
    if not vals:
        return None, None
    rnd = random.Random(0)
    means = sorted(sum(rnd.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)


EXCLUDED = set()


def score(path, items):
    with open(path) as f:
        rows = [json.loads(l) for l in f if json.loads(l)["id"] not in EXCLUDED]
    kfr, old, cites, srch, aov, fnf = [], [], [], [], [], []
    tp = fp = fn = 0
    for r in rows:
        it = items[r["id"]]
        a = r["answer"]
        nf = says_not_found(a)
        srch.append(len(r["searches"]))
        if it["type"] == "not_covered":
            tp += nf
            fn += not nf
            continue
        fp += nf
        fnf.append(float(nf))
        ref = set(toks(it["answer"]))
        aov.append(len(ref & set(toks(a))) / len(ref) if ref else 0.0)
        kfr.append(sum(fact_hit(f, a) for f in it["facts"]) / len(it["facts"]))
        cites.append(any(int(c) in it["source_chunks"] for c in CITE.findall(a)))
        if it["type"] == "superseded":
            old.append(old_rule_used(it, a))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    m = lambda v: round(sum(v) / len(v), 3) if v else None
    return {"run": os.path.basename(path)[:-6], "n": len(rows),
            "key_fact_recall": m(kfr), "kfr_ci": boot(kfr),
            "answer_overlap": m(aov), "aov_ci": boot(aov),
            "false_not_found": m(fnf),
            "old_rule_rate": m(old), "old_ci": boot([float(x) for x in old]),
            "not_found_f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
            "not_found_precision": round(prec, 3), "not_found_recall": round(rec, 3),
            "searches_per_q": m(srch), "cites_source": m([float(x) for x in cites]),
            "invalid_format": sum(r.get("invalid_format", 0) for r in rows)}


def main():
    items = {}
    for split in ("test", "train"):
        p = os.path.join(ROOT, f"data/qa/{split}.jsonl")
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    it = json.loads(line)
                    items[it["id"]] = it
    ex_path = os.path.join(ROOT, "data/qa/test_exclusions.json")
    if os.path.exists(ex_path):                       # broken reference answers, reasons in the file
        with open(ex_path) as f:
            EXCLUDED.update(json.load(f)["ids"])
    res = [score(p, items) for p in sys.argv[1:]]
    cols = ["run", "n", "key_fact_recall", "answer_overlap", "false_not_found", "old_rule_rate", "not_found_f1", "searches_per_q", "cites_source", "invalid_format"]
    print(" | ".join(cols))
    for r in res:
        print(" | ".join(str(r[c]) for c in cols))
    with open(os.path.join(ROOT, "results/summary.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
