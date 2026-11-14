"""Turn agent trajectories into MLX LoRA training files.

Each trajectory with N assistant turns becomes N examples: the conversation up to that turn,
with that assistant turn last. Training with --mask-prompt then puts the loss on every
assistant turn (search decisions, queries and the final answer) and never on the system
prompt, the question or the retrieved observations.

    python src/make_mlx_data.py     # writes data/mlx/{train,valid}.jsonl
"""
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    with open(os.path.join(ROOT, "data/sft/train.jsonl")) as f:
        trajs = [json.loads(l) for l in f]
    rnd = random.Random(3)
    rnd.shuffle(trajs)
    n_val = max(8, len(trajs) // 20)
    train = trajs[n_val:]
    # cap the share of "not found" trajectories in training (v1 used 28% and over-refused)
    share = float(os.environ.get("NC_SHARE", "1"))
    nc = sorted((t for t in train if t["type"] == "not_covered"), key=lambda t: t.get("aug") != "nearmiss")
    rest = [t for t in train if t["type"] != "not_covered"]
    cap = round(len(rest) * share / (1 - share)) if share < 1 else len(nc)
    train = rest + nc[:cap]
    print("not_covered in train:", min(cap, len(nc)), "of", len(train))
    out = {"valid": trajs[:n_val], "train": train}
    os.makedirs(os.path.join(ROOT, "data/mlx"), exist_ok=True)
    for name, rows in out.items():
        ex = []
        for t in rows:
            m = t["messages"]
            for i, msg in enumerate(m):
                if msg["role"] == "assistant":
                    ex.append({"messages": m[:i + 1]})
        rnd.shuffle(ex)
        with open(os.path.join(ROOT, f"data/mlx/{name}.jsonl"), "w") as f:
            for e in ex:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(name, len(rows), "trajectories ->", len(ex), "examples")


if __name__ == "__main__":
    main()
