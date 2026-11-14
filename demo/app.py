"""PolicyGuard demo: compare pre-run answers on the test set, or ask your own question live."""
import json
import os
import threading

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 2))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr

from agent import SYSTEM, observation, parse

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "CohereLabs/tiny-aya-global"
ADAPTER = "bhavyananda17/policyguard-tiny-aya-global-lora"
MAX_SEARCHES = 3
REPO = "https://github.com/bhavyananda17/lexsi_labs_project"
COLLECTION = "https://huggingface.co/collections/bhavyananda17/policyguard-lexsi-hackathon-6af8a31e037f3d7eaf41b4b4"

with open(os.path.join(HERE, "examples.json"), encoding="utf-8") as f:
    EXAMPLES = json.load(f)
BY_LABEL = {}
TYPE_NAME = {"single_hop": "one rule", "multi_hop": "two rules", "superseded": "replaced rule",
             "not_covered": "not in the documents"}
for e in EXAMPLES:
    BY_LABEL[f"[{TYPE_NAME[e['type']]}] {e['question']}"] = e
LABELS = list(BY_LABEL)
SETUPS = [("untrained_agent", "Untrained Tiny Aya (agent loop)"),
          ("fixed_pipeline", "Untrained Tiny Aya (always search once)"),
          ("policyguard", "PolicyGuard (fine-tuned, agent loop)")]
BADGE = {"correct": "✅ correct", "partial": "🟡 partly correct", "wrong": "❌ wrong",
         "refused": "⛔ said not found (but the answer exists)", "correct refusal": "✅ correctly said not found",
         "answered anyway": "❌ answered a question the documents don't cover"}


def steps_md(run):
    if not run["steps"]:
        q = run["queries"][0] if run["queries"] else ""
        return f"Searched once for the question: *{q}*\n\nRetrieved chunks: {', '.join(map(str, run['chunks']))}"
    parts = []
    for m in run["steps"]:
        if m["role"] == "assistant":
            parts.append("**Model:**\n```\n" + m["content"].strip() + "\n```")
        else:
            body = m["content"].replace("Observation:", "").strip()
            parts.append("**Search results:**\n> " + body[:1200].replace("\n", "\n> ") + (" …" if len(body) > 1200 else ""))
    return "\n\n".join(parts)


def show(label):
    e = BY_LABEL[label]
    outs = [f"**Reference answer:** {e['reference']}"]
    for key, _ in SETUPS:
        r = e["runs"][key]
        outs += [f"{BADGE.get(r['verdict'], r['verdict'])}\n\n{r['answer'] or '(empty answer)'}", steps_md(r)]
    return outs


def filter_labels(kind):
    ls = [l for l in LABELS if kind == "all" or BY_LABEL[l]["type"] == kind]
    return gr.update(choices=ls, value=ls[0])


# ---------------- live mode (lazy load: the Space starts fast, the first question loads the model) ----------------
_lock = threading.Lock()
_state = {}


def load():
    with _lock:
        if "model" in _state:
            return _state
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from search import Searcher
        token = os.environ.get("HF_TOKEN")
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(BASE, token=token)
        base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, token=token).to(device)
        model = PeftModel.from_pretrained(base, os.environ.get("ADAPTER_PATH", ADAPTER)).eval()
        _state.update(model=model, tok=tok, device=device, searcher=Searcher(), torch=torch)
        return _state


def generate(st, messages, use_adapter):
    torch, tok, model = st["torch"], st["tok"], st["model"]
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(st["device"])
    kw = dict(max_new_tokens=220, do_sample=False, stop_strings=["Observation:"], tokenizer=tok)
    with torch.no_grad():
        if use_adapter:
            out = model.generate(**enc, **kw)
        else:
            with model.disable_adapter():
                out = model.generate(**enc, **kw)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.split("Observation:")[0].strip()


def ask(question, which):
    question = (question or "").strip()
    if not question:
        yield "Type a question first.", ""
        return
    if not os.environ.get("HF_TOKEN") and not os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global")):
        yield "Live mode needs access to the gated base model (an HF_TOKEN secret). Use the comparison tab instead.", ""
        return
    log = []
    yield "Loading the model (first question only, about a minute)…", ""
    st = load()
    use_adapter = which.startswith("PolicyGuard")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    searches = 0
    for _ in range(MAX_SEARCHES + 2):
        yield "Thinking…", "\n\n".join(log)
        reply = generate(st, msgs, use_adapter)
        kind, val = parse(reply)
        msgs.append({"role": "assistant", "content": reply})
        log.append("**Model:**\n```\n" + reply + "\n```")
        if kind == "final":
            yield val, "\n\n".join(log)
            return
        if kind == "search" and searches < MAX_SEARCHES:
            searches += 1
            hits = st["searcher"].search(val, 4)
            obs = observation(st["searcher"], hits)
            msgs.append({"role": "user", "content": obs})
            log.append("**Search results:**\n> " + obs.replace("Observation:", "").strip()[:1500].replace("\n", "\n> ") + " …")
            continue
        if kind == "search":
            msgs.append({"role": "user", "content": "Observation: Search limit reached. Give your Final Answer now."})
            continue
        yield reply, "\n\n".join(log)          # no protocol: the reply is the answer
        return
    yield "(no final answer within the step limit)", "\n\n".join(log)


ABOUT = f"""
**PolicyGuard** is [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B) fine-tuned with LoRA to act as a search
agent for Indian health-insurance rules. It searches two IRDAI circulars (May 2024 current, July 2020 replaced),
cites the passages it used as `[chunk N]`, and says it could not find an answer instead of guessing.

| On 74 answerable test questions (LLM judge) | Correct | Correct or partly correct | Cites the right passage |
|---|---|---|---|
| Untrained Tiny Aya, agent loop | 4% | 24% | 3% |
| **PolicyGuard, agent loop** | **26%** | **49%** | **42%** |
| Untrained Tiny Aya, always search once | 32% | 54% | 30% |

Fine-tuning turns the untrained model, which rarely searches and invents citations, into a working search agent.
It does not beat a simple always-search pipeline on accuracy (the confidence intervals overlap); it cites better and rarely
refuses answerable questions. Details, limitations and code: [GitHub]({REPO}) · [Model, dataset and collection]({COLLECTION}).

*Not legal or financial advice. Built for the Lexsi AI Safety & Alignment Hackathon, Track 3.*
"""

with gr.Blocks(title="PolicyGuard") as demo:
    gr.Markdown("# PolicyGuard\nA small fine-tuned search agent for IRDAI health-insurance rules.")
    with gr.Tab("Compare on test questions"):
        gr.Markdown("Pre-run answers on the held-out test set (none of these questions or their source passages were "
                    "used in training). Grades: LLM judge against the reference answer.")
        with gr.Row():
            kind = gr.Radio(["all", "single_hop", "multi_hop", "superseded", "not_covered"], value="all",
                            label="Question type", scale=1)
        pick = gr.Dropdown(LABELS, value=LABELS[0], label="Question", scale=3)
        ref = gr.Markdown()
        cols = []
        with gr.Row(equal_height=False):
            for _, title in SETUPS:
                with gr.Column():
                    gr.Markdown(f"### {title}")
                    ans = gr.Markdown()
                    with gr.Accordion("Steps", open=False):
                        st = gr.Markdown()
                    cols += [ans, st]
        kind.change(filter_labels, kind, pick)
        pick.change(show, pick, [ref] + cols)
        demo.load(show, pick, [ref] + cols)
    with gr.Tab("Ask your own question"):
        gr.Markdown("Runs the model live. On the free CPU hardware this takes **1–3 minutes** per question, "
                    "and the first question also loads the model.")
        q = gr.Textbox(label="Question", placeholder="e.g. What is the moratorium period for a health insurance policy?")
        which = gr.Radio(["PolicyGuard (fine-tuned)", "Untrained Tiny Aya"], value="PolicyGuard (fine-tuned)",
                         label="Model")
        go = gr.Button("Ask", variant="primary")
        answer = gr.Markdown(label="Answer")
        with gr.Accordion("Steps", open=True):
            trace = gr.Markdown()
        gr.Examples(["What is the moratorium period for a health insurance policy?",
                     "How many months is the waiting period for pre-existing diseases?",
                     "Can I claim the GST paid on my premium as a tax deduction?"], q)
        go.click(ask, [q, which], [answer, trace], concurrency_limit=1)
    with gr.Tab("About"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
