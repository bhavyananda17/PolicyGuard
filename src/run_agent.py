"""Run Tiny Aya (optionally with the PolicyGuard LoRA) on the test set with MLX, in one of four modes.

  no_tools      : question only, no retrieval
  naive_rag     : always retrieve top-k for the question, answer in one turn
  agent         : ReAct loop, the model decides whether and what to search (max 3 searches)
  agent_force   : ReAct loop, but the harness rejects a Final Answer before the first search

    python src/run_agent.py --mode agent [--adapter adapters/policyguard] [--limit 20]
Writes results/<run_name>.jsonl (one row per question, with the full transcript).
"""
import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from agent import SYSTEM, observation, parse
from search import Searcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "CohereLabs/tiny-aya-global"
K = int(os.environ.get("TOPK", 4))         # search results per call (TOPK=6 for the k=6 run)
MAX_SEARCHES = 3

PLAIN_SYSTEM = ("You answer questions about Indian health-insurance rules issued by IRDAI. "
                "If you are not sure, say you could not find the answer. Be brief.")
RAG_SYSTEM = ("You answer questions about Indian health-insurance rules issued by IRDAI, using only the documents "
              "below. Newer circulars replace older ones. Cite chunks like [chunk 12]. If the documents do not "
              "answer the question, say you could not find it. Be brief.")


class LM:
    def __init__(self, model, adapter=None):
        from mlx_lm import load
        self.model, self.tok = load(model, adapter_path=adapter)

    def __call__(self, messages, max_tokens=400):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        out = generate(self.model, self.tok, prompt=prompt, max_tokens=max_tokens,
                       sampler=make_sampler(temp=0.0), verbose=False)
        # stop at a hallucinated observation: the model must not write tool results itself
        out = out.split("<|END_RESPONSE|>")[0].split("<|END_OF_TURN_TOKEN|>")[0]
        return out.split("Observation:")[0].strip()


def run_one(lm, s, item, mode):
    q = item["question"]
    if mode == "no_tools":
        msgs = [{"role": "system", "content": PLAIN_SYSTEM}, {"role": "user", "content": q}]
        return {"answer": lm(msgs), "searches": [], "transcript": msgs}
    if mode == "naive_rag":
        hits = s.search(q, K)
        msgs = [{"role": "system", "content": RAG_SYSTEM + "\n\n" + s.render(hits)}, {"role": "user", "content": q}]
        return {"answer": lm(msgs), "searches": [{"query": q, "chunks": [h["chunk_id"] for h in hits]}],
                "transcript": msgs}

    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    searches, answer, invalid = [], None, 0
    for _ in range(MAX_SEARCHES + 2):
        reply = lm(msgs)
        kind, val = parse(reply)
        msgs.append({"role": "assistant", "content": reply})
        if kind == "final":
            if mode == "agent_force" and not searches:
                msgs.append({"role": "user", "content": "Observation: You must search the documents before answering."})
                continue
            answer = val
            break
        if kind == "search" and len(searches) < MAX_SEARCHES:
            hits = s.search(val, K)
            searches.append({"query": val, "chunks": [h["chunk_id"] for h in hits]})
            msgs.append({"role": "user", "content": observation(s, hits)})
            continue
        if kind == "search":
            msgs.append({"role": "user", "content": "Observation: Search limit reached. Give your Final Answer now."})
            continue
        invalid += 1
        answer = val          # no protocol: treat the whole reply as the answer
        break
    return {"answer": answer or "", "searches": searches, "invalid_format": invalid, "transcript": msgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["no_tools", "naive_rag", "agent", "agent_force"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"{'tuned' if a.adapter else 'base'}_{a.mode}"
    with open(os.path.join(ROOT, f"data/qa/{a.split}.jsonl")) as f:
        items = [json.loads(l) for l in f]
    if a.limit:
        items = items[:a.limit]
    out_path = os.path.join(ROOT, f"results/{name}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):                      # resumable
        with open(out_path) as f:
            done = {json.loads(l)["id"] for l in f}
    lm, s = LM(a.model, a.adapter), Searcher()
    with open(out_path, "a") as f:
        for i, it in enumerate(items):
            if it["id"] in done:
                continue
            t0 = time.time()
            r = run_one(lm, s, it, a.mode)
            r.update(id=it["id"], type=it["type"], seconds=round(time.time() - t0, 2))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(items)}] {it['type']:<12} {r['seconds']:>5}s searches={len(r['searches'])}", flush=True)


if __name__ == "__main__":
    main()
