"""Agreement between the main judge (results/judge) and a second judge (results/judge2).

    python src/judge_agreement.py base_naive_rag base_agent tuned_v4_agent
Prints, per run: correct rate under each judge, exact verdict agreement, Cohen's kappa,
and agreement on the binary "correct vs not" call. Writes results/judge2/agreement.json.
"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(d, run):
    with open(os.path.join(ROOT, d, f"{run}.jsonl")) as f:
        return {r["id"]: r["verdict"] for r in map(json.loads, f)}


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def main():
    out = []
    pooled_a, pooled_b = [], []
    print("run | n | correct (judge 1) | correct (judge 2) | exact agree | kappa | correct-vs-not agree")
    for run in sys.argv[1:]:
        j1, j2 = load("results/judge", run), load("results/judge2", run)
        ids = sorted(set(j1) & set(j2))
        a, b = [j1[i] for i in ids], [j2[i] for i in ids]
        pooled_a += a; pooled_b += b
        bc = [x == "correct" for x in a], [y == "correct" for y in b]
        s = {"run": run, "n": len(ids),
             "correct_judge1": round(sum(bc[0]) / len(ids), 3), "correct_judge2": round(sum(bc[1]) / len(ids), 3),
             "exact_agreement": round(sum(x == y for x, y in zip(a, b)) / len(ids), 3), "kappa": kappa(a, b),
             "binary_agreement": round(sum(x == y for x, y in zip(*bc)) / len(ids), 3)}
        out.append(s)
        print(" | ".join(str(v) for v in s.values()))
    pooled = {"run": "pooled", "n": len(pooled_a),
              "exact_agreement": round(sum(x == y for x, y in zip(pooled_a, pooled_b)) / len(pooled_a), 3),
              "kappa": kappa(pooled_a, pooled_b),
              "confusion": {f"{x}->{y}": c for (x, y), c in sorted(Counter(zip(pooled_a, pooled_b)).items())}}
    out.append(pooled)
    print(json.dumps(pooled, indent=1))
    with open(os.path.join(ROOT, "results/judge2/agreement.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
