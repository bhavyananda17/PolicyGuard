"""Generate the PolicyGuard question set from the IRDAI index (four item types).

  single_hop   : answered by one CURRENT chunk
  multi_hop    : needs two CURRENT chunks (with one sub-query per chunk)
  superseded   : a current rule that differs from the replaced 2020 rule on the same topic
  not_covered  : plausible policyholder question the documents do not answer

Every answer comes only from current (2024) documents. Key facts must be verbatim spans
of their source chunk, otherwise the item is dropped. Output: data/qa/raw_<type>.jsonl

    python src/gen_questions.py --types single_hop,multi_hop,superseded,not_covered [--limit N]
"""
import argparse
import json
import os
import random
import re

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np

from llm import USAGE, chat, parse_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data/qa")
# Groq free tier: 200K tokens/day per model, so the work is split across two models.
MODEL = {"single_hop": "openai/gpt-oss-20b", "multi_hop": "openai/gpt-oss-20b",
         "superseded": "openai/gpt-oss-120b", "not_covered": "qwen/qwen3.8-27b"}
random.seed(13)

SYSTEM = ("You write evaluation questions for an assistant that answers questions about Indian "
          "health-insurance rules issued by the regulator IRDAI. Questions must sound like a real "
          "policyholder, agent or claims officer asking. Never mention 'the passage', 'the circular "
          "above' or page numbers in a question. Answers are 1-3 plain sentences. Reply with JSON only.")


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%.,/ ]", " ", s.lower())).strip()


def verbatim(fact, text):
    f = norm(fact)
    return len(f.split()) >= 3 and f in norm(text)


def body(c):
    return c["text"].split("\n", 1)[1] if "\n" in c["text"] else c["text"]


def load():
    with open(os.path.join(ROOT, "data/index/chunks.jsonl")) as f:
        chunks = [json.loads(l) for l in f]
    index = faiss.read_index(os.path.join(ROOT, "data/index/faiss.index"))
    vecs = np.vstack([index.reconstruct(i) for i in range(index.ntotal)])
    return chunks, vecs


def usable(c):
    b = body(c)
    letters = sum(ch.isalpha() for ch in b)
    return c["words"] >= 50 and letters / max(1, len(b)) > 0.6


def write(name, items):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"raw_{name}.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{name}: {len(items)} items  | usage {USAGE}")


def single_hop(chunks, cur, limit):
    items = []
    for c in cur[:limit]:
        prompt = (f"Passage (from {c['text'].splitlines()[0]}):\n\"\"\"\n{body(c)}\n\"\"\"\n\n"
                  "Write up to 2 different questions that this passage alone fully answers. Prefer concrete "
                  "rules: time limits, amounts, conditions, who must do what. If the passage is only "
                  "boilerplate, headings or a form, return {\"items\": []}.\n"
                  "JSON: {\"items\": [{\"question\": str, \"answer\": str, \"facts\": [2-4 short spans copied "
                  "word-for-word from the passage that the answer needs]}]}")
        js = parse_json(chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                             model=MODEL["single_hop"], json_mode=True, tag="sh")) or {}
        for it in js.get("items", [])[:2]:
            facts = [f for f in it.get("facts", []) if verbatim(f, body(c))]
            if it.get("question") and it.get("answer") and facts:
                items.append({"type": "single_hop", "question": it["question"], "answer": it["answer"],
                              "facts": facts, "source_chunks": [c["chunk_id"]]})
    return items


def multi_hop(chunks, cur, vecs, limit):
    ids = [c["chunk_id"] for c in cur]
    pairs, seen = [], set()
    for c in cur:
        sims = vecs[ids] @ vecs[c["chunk_id"]]
        for j in np.argsort(-sims):
            d = cur[j]
            if d["chunk_id"] == c["chunk_id"] or abs(d["chunk_in_doc"] - c["chunk_in_doc"]) <= 1 and d["doc"] == c["doc"]:
                continue
            key = tuple(sorted((c["chunk_id"], d["chunk_id"])))
            if key not in seen and 0.35 < sims[j] < 0.85:
                seen.add(key)
                pairs.append((c, d))
            break
    random.shuffle(pairs)
    items = []
    for a, b in pairs[:limit]:
        prompt = (f"Passage A:\n\"\"\"\n{body(a)}\n\"\"\"\n\nPassage B:\n\"\"\"\n{body(b)}\n\"\"\"\n\n"
                  "Write ONE question whose full answer needs a fact from Passage A AND a fact from Passage B "
                  "(e.g. combine two rules, or apply one rule to the situation in the other). If no natural "
                  "question needs both, return {\"items\": []}.\n"
                  "JSON: {\"items\": [{\"question\": str, \"answer\": str, \"facts_a\": [1-3 word-for-word spans "
                  "from A], \"facts_b\": [1-3 word-for-word spans from B], \"sub_queries\": [a short search "
                  "query for A's fact, a short search query for B's fact]}]}")
        js = parse_json(chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                             model=MODEL["multi_hop"], json_mode=True, tag="mh")) or {}
        for it in js.get("items", [])[:1]:
            fa = [f for f in it.get("facts_a", []) if verbatim(f, body(a))]
            fb = [f for f in it.get("facts_b", []) if verbatim(f, body(b))]
            sq = [q for q in it.get("sub_queries", []) if isinstance(q, str)][:2]
            if it.get("question") and it.get("answer") and fa and fb and len(sq) == 2:
                items.append({"type": "multi_hop", "question": it["question"], "answer": it["answer"],
                              "facts": fa + fb, "sub_queries": sq,
                              "source_chunks": [a["chunk_id"], b["chunk_id"]]})
    return items


def superseded(chunks, cur, vecs, limit):
    # MiniLM similarity is uniformly high inside one legal domain (0.7-0.85), so the old
    # passages are found with TF-IDF, where rare terms like "moratorium" carry the weight.
    from sklearn.feature_extraction.text import TfidfVectorizer
    old = [c for c in chunks if c["status"] == "superseded" and usable(c)]
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    m = tfidf.fit_transform([body(c) for c in old] + [body(c) for c in cur])
    m_old, m_cur = m[:len(old)], m[len(old):]
    has_rule = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|thirty|sixty)\b", re.I)
    items = []
    for ci, c in enumerate(cur[:limit]):
        if not has_rule.search(body(c)):
            continue
        sims = (m_old @ m_cur[ci].T).toarray().ravel()
        top = [old[j] for j in np.argsort(-sims)[:3] if sims[j] > 0.08]
        if not top:
            continue
        olds = "\n\n".join(f"OLD-{o['chunk_id']}:\n\"\"\"\n{body(o)}\n\"\"\"" for o in top)
        prompt = (f"CURRENT rule (IRDAI Master Circular, May 2024, in force):\n\"\"\"\n{body(c)}\n\"\"\"\n\n"
                  f"REPLACED rules (IRDAI 2020 circular, no longer in force):\n{olds}\n\n"
                  "Does any REPLACED passage state a DIFFERENT concrete rule on the SAME topic as the CURRENT "
                  "passage (a different number of months/years/hours/days, amount, or condition)? Only real "
                  "contradictions count, not extra detail. If none, return {\"conflict\": false}.\n"
                  "If yes, JSON: {\"conflict\": true, \"old_chunk\": <id number of the OLD passage>, "
                  "\"topic\": str, \"current_rule\": str, \"old_rule\": str, \"current_facts\": [1-3 word-for-word "
                  "spans from CURRENT], \"old_fact\": a word-for-word span from the OLD passage with the old rule, "
                  "\"questions\": [2 differently worded questions a policyholder would ask about this rule today]}. "
                  "The answer to the questions must be the CURRENT rule.")
        js = parse_json(chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                             model=MODEL["superseded"], json_mode=True, max_tokens=1500, tag="ss")) or {}
        if not js.get("conflict"):
            continue
        oc = re.sub(r"\D", "", str(js.get("old_chunk")))      # model often writes "OLD-333"
        o = next((x for x in top if str(x["chunk_id"]) == oc), None)
        facts = [f for f in js.get("current_facts", []) if verbatim(f, body(c))]
        if not o or not facts:
            continue
        # old fact: verbatim, or every number in the stated old rule appears in the old passage
        old_nums = set(re.findall(r"\d+", js.get("old_rule", "")))
        if not (verbatim(js.get("old_fact", ""), body(o)) or
                (old_nums and old_nums <= set(re.findall(r"\d+", body(o))))):
            continue
        qs = list(js.get("questions", []))[:2]
        more = parse_json(chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content":
            f"Rule today (2024): {js['current_rule']}\nReplaced rule (2020): {js['old_rule']}\n"
            f"Existing questions: {qs}\nWrite 2 NEW, differently worded questions (e.g. a concrete scenario "
            "with a policyholder) whose correct answer today is the 2024 rule. JSON: {\"questions\": [str, str]}"}],
            model=MODEL["superseded"], json_mode=True, max_tokens=600, tag="ss2")) or {}
        qs += [q for q in more.get("questions", []) if isinstance(q, str)][:2]
        for q in qs:
            items.append({"type": "superseded", "question": q, "answer": js["current_rule"],
                          "facts": facts, "topic": js.get("topic"), "old_rule": js["old_rule"],
                          "old_fact": js["old_fact"], "old_chunk": o["chunk_id"],
                          "source_chunks": [c["chunk_id"]]})
    return items


def not_covered(chunks, vecs, n_calls):
    """Candidates from the LLM, then a retrieval + LLM check that no chunk answers them."""
    from search import Searcher
    s = Searcher()
    themes = ["tax deductions and premiums", "motor, travel and life insurance products",
              "specific insurers, hospitals, apps or plan prices", "medical treatment advice and drug costs",
              "procedures in other countries or other regulators", "very specific numbers about claims or policies"]
    cands = []
    for i in range(n_calls):
        prompt = (f"Write 20 questions a person in India might ask a health-insurance help assistant, on the theme "
                  f"'{themes[i % len(themes)]}', that are NOT answered by IRDAI's health-insurance circulars on "
                  "product standards, waiting periods, moratorium, portability, cashless claims, grievances and "
                  "standard definitions/exclusions. They should sound related to insurance so the assistant is "
                  "tempted to answer. JSON: {\"questions\": [str]}")
        js = parse_json(chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                             model=MODEL["not_covered"], json_mode=True, temperature=0.9, tag="nc")) or {}
        if isinstance(js, list):                          # some models return a bare list
            js = {"questions": js}
        cands += [q for q in js.get("questions", []) if isinstance(q, str)]
    items = []
    for k in range(0, len(cands), 8):
        batch = cands[k:k + 8]
        blocks = []
        for qi, q in enumerate(batch):
            hits = s.search(q, 4)
            ctx = "\n".join(f"- {h['text'][:450]}" for h in hits)
            blocks.append(f"Q{qi}: {q}\nRetrieved passages:\n{ctx}")
        prompt = ("For each question, decide if ANY of its retrieved passages answers it (fully or mostly). "
                  "Be strict: general related text is not an answer.\n\n" + "\n\n".join(blocks) +
                  "\n\nJSON: {\"verdicts\": [{\"q\": index, \"answered\": true|false}]}")
        js = parse_json(chat([{"role": "user", "content": prompt}], model=MODEL["not_covered"], json_mode=True,
                             max_tokens=800, tag="ncv")) or {}
        if isinstance(js, list):
            js = {"verdicts": js}
        for v in js.get("verdicts", []):
            qi = v.get("q")
            if isinstance(qi, int) and 0 <= qi < len(batch) and v.get("answered") is False:
                items.append({"type": "not_covered", "question": batch[qi],
                              "answer": "This is not covered in the IRDAI health-insurance documents I can search.",
                              "facts": [], "source_chunks": []})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="single_hop,multi_hop,superseded,not_covered")
    ap.add_argument("--limit", type=int, default=10_000, help="max chunks/pairs per type (pilot: small)")
    ap.add_argument("--nc_calls", type=int, default=12)
    a = ap.parse_args()
    chunks, vecs = load()
    cur = [c for c in chunks if c["status"] == "current" and usable(c)]
    print(f"{len(cur)} usable current chunks")
    t = a.types.split(",")
    if "single_hop" in t:
        write("single_hop", single_hop(chunks, cur, a.limit))
    if "multi_hop" in t:
        write("multi_hop", multi_hop(chunks, cur, vecs, a.limit))
    if "superseded" in t:
        write("superseded", superseded(chunks, cur, vecs, a.limit))
    if "not_covered" in t:
        write("not_covered", not_covered(chunks, vecs, min(a.nc_calls, a.limit)))


if __name__ == "__main__":
    main()
