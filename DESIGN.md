# Final plan: PolicyGuard

*Lexsi hackathon, Track 3 (Agentic Fine-tuning). Final scoped plan, 10 Nov 2026.
Submission deadline: **15 Nov, 11:59 PM AoE** (about 5:30 PM IST on 16 Nov). Target: submit on 15 Nov IST.*

## 1. What we are building

**PolicyGuard: a small 3B agent that answers health-insurance questions only from the current rules, ignores rules that have been replaced, and says "not covered / not found" instead of guessing.**

- **Model:** `CohereLabs/tiny-aya-global` (3.35B, Cohere2 architecture, gated: accept the licence once on HF).
- **Main problem statement:** **4. Fine-tuning for RAG** (when and what to retrieve, grounding answers, ignoring distractors).
- **Covered along the way:** 5 (synthetic multi-hop data from a corpus) and 3 (distilling a teacher into a small student). These are listed in the submission but are not separate goals.
- **Cut from scope:** GRPO, problem statement 9, a multi-judge panel, a polished UI.

### Why this niche

- Insurance rules change, and old circulars and policy wordings stay online. A bot that quotes a **replaced** rule gives a customer a wrong answer with real cost.
- Indian health insurance has public regulatory documents with **real versions**: IRDAI's 2024 master circulars replaced many older circulars, and standard products such as Arogya Sanjeevani have published wordings that were revised over time. Real superseded text is a stronger distractor than planted fake errors.
- BFSI is one of the domains the organizers name.
- **Check before relying on them:** download terms for each document. Prefer IRDAI regulatory documents and standard product wordings over individual insurers' branded material. The demo must not use any insurer's name or branding.

## 2. The pipeline (5 steps)

```
 1. Corpus            2. Questions          3. Teacher runs        4. Train             5. Evaluate + demo
 IRDAI circulars  ──▶ ~1,000 Q&A items  ──▶ gpt-oss-120b on   ──▶ LoRA SFT on      ──▶ 4 metrics, base vs
 + standard policy    AgentTune synthesis   Groq answers as a      Tiny Aya Global      tuned, on held-out
 wordings, current    + my validate_items   ReAct agent with       (AgentTune distill   test set (MLX on Mac)
 AND superseded       (fact tracing, dedup, search_docs; keep      trainer, Kaggle T4)  → Gradio demo on a
 versions, chunked    leakage checks)       only correct, cited                         free HF CPU Space
 + FAISS index                              runs
```

| Step | Reused from my earlier work | New |
|---|---|---|
| 1. Corpus + index | Chunking + FAISS (`build_index.py`, `rag.py`), with a `superseded` flag on each chunk instead of `distractor` | PDF text extraction (`pypdf`), version tagging of each document |
| 2. Questions | `validate_items.py` checks | AgentTune `agenttune.rag.synthesis` for generation; four item types (below) |
| 3. Teacher runs | — | AgentTune `Project` + `DictToolHarness` + `ReActStrategy`, one tool `search_docs(query, k)` |
| 4. Training | Hyper-parameter ideas from `train_lora.py` | AgentTune `create_distill_trainer` (LoRA + TRL) |
| 5. Evaluation | Key-fact recall scorer | AuditKIT `evaluate` / `compare_models` (bootstrap tests), Gradio app |

### Question types

| Type | Example | The right behaviour |
|---|---|---|
| Single-hop | "What is the waiting period for pre-existing diseases under Arogya Sanjeevani?" | Search once, answer with a citation |
| Multi-hop | "If a claim is rejected, how many days does the insurer have to respond to a grievance, and where can the customer escalate after that?" | Search two or more times, combine |
| **Superseded** | A question whose answer changed between the old and new rule | Answer with the **current** rule, ignore the old chunk |
| **Not covered** | A question the documents do not answer | Answer "not found in the policy documents" |

**Sizes:** about 700 training runs kept after filtering, 100 dev, 200 test (held out, leakage-checked).

**Mix:** **20–25% "not covered" items** in training (about 150–175 of 700), with the same share in dev and test. Refusal only sticks when training has enough refusal examples (noted by the author of [`ferscm44/tiny-aya-agent`](https://huggingface.co/ferscm44/tiny-aya-agent)); Trust-Align (ICLR 2025) also shows that refusal training lifts grounded answers. The rest is split roughly evenly between single-hop, multi-hop and superseded items. If teacher runs fall short, cut the other types before cutting "not covered" items.

### Format detail (checked)

Tiny Aya's chat template only allows **alternating user and assistant turns**; there is no `tool` role. Tool results must therefore go back to the model inside the next **user** turn, as `Observation: ...`, which is the ReAct text format. No native function-calling is needed.

## 3. Metrics (all deterministic, no LLM judge)

| Metric | What it shows |
|---|---|
| **Key-fact recall** | Is the answer correct and complete? |
| **Superseded-rule rate** | How often it quotes the old rule (lower is better) |
| **"Not found" F1** | Abstains when it should and answers when it should |
| **Searches per question** | Efficiency and the "when to retrieve" part |

**Compared models (same engine for all, MLX on the Mac):**
1. Tiny Aya with no tools
2. Tiny Aya with naive RAG (always retrieve top-k)
3. Tiny Aya with the agent prompt, untrained
4. **Tiny Aya + PolicyGuard LoRA, model decides when to search**
5. **Tiny Aya + PolicyGuard LoRA, always search**: the harness forces at least one `search_docs` call before any answer

**Always search vs. model decides (rows 4 and 5):** "Search, Do not Guess" (arXiv 2604.04651) found that small models choosing when to search often do worse than ones forced to always search. Row 5 needs no extra training, only a harness switch, so it costs one more evaluation run. Either outcome is a result:
- If **model decides** matches or beats always search, it saves searches with no loss in accuracy.
- If **always search** wins, we report that, ship always search as the default, and confirm the paper's finding on a new domain.

Searches per question and key-fact recall are reported for both rows.

A cross-family LLM judge for groundedness is optional, reported as a side check only.

**ROI paragraph:** cost per 1,000 questions on gpt-oss-120b via an API vs. self-hosting the 3B model, using the measured token counts.

## 4. Can this be done for free on an M5 Pro MacBook (24 GB)?

**Yes.** The Mac does nearly everything, and one free cloud GPU does the training.

| Step | Where | Free? | Notes |
|---|---|---|---|
| Corpus, PDFs, index, question checks | Mac | ✅ | CPU work |
| Question generation + teacher runs | Mac → Groq API | ✅ | Groq free tier, no card. Rate limits are the main risk: start on day 1, cache every call |
| **Training (LoRA SFT)** | **Kaggle 2× T4** | ✅ | About 30 free GPU hours a week, needs phone verification only. Expected 2–4 h. AgentTune needs **Python 3.12**: check Kaggle's version first, install with `uv` if older |
| Training fallback | Mac, `mlx-lm lora` | ✅ | **Checked:** MLX supports the `cohere2` architecture, and `mlx-community` already has Tiny Aya conversions. Several hours; fine overnight. Weakens the "used AgentTune for training" point, so it is a fallback only |
| Evaluation (4 models × 200 questions) | Mac, MLX | ✅ | A 4-bit 3B model is about 2 GB and fast on Apple silicon. vLLM is not available on macOS |
| Hosting models, adapters, datasets | Hugging Face free account | ✅ | Must be public and ungated (the base model is gated, our adapter can be public) |
| Demo | Free HF CPU Space (2 vCPU, 16 GB) | ✅ | 4-bit GGUF via llama.cpp, a few tokens per second, plus pre-run examples. ZeroGPU hosting needs a paid PRO account, so it is not used |
| Write-up | HF Article | ✅ | — |

**Why not train on the Mac with AgentTune directly:** AgentTune trains through PyTorch/TRL. On a Mac that runs on the MPS backend, which is much slower than CUDA, and 4-bit loading isn't reliable there, so a 3B model in 16-bit would use about 12–16 GB and take well over 10 hours.

## 5. Hackathon benefits we can use

| Benefit | How to use it | Catch |
|---|---|---|
| **$20 HF credits** (first 800 teams that join Discord) | Spare GPU time or a faster Space if Kaggle fails | Team lead's HF username must match the registration form. HF **may ask for a payment method** to redeem, so the plan does not depend on it. Expires 30 days after redemption |
| **Lexsi open-source stack** | AgentTune (synthesis, harness, distillation), AuditKIT (evaluation, cost tracking, significance tests) | Some known silent failures; sanity-check outputs on 5 items before long runs |
| **AgentTune notebooks** (9 mirrored on Colab) | Start from #27 (RAG synthesis), #02 (distillation), #15 (financial RAG agent) | — |
| **Kickoff slides** (`/kickoff-slides/` on the site) | Hints about what judges want | — |
| **Discord: #hackathon-lexsi, #report-issues** | Ask the organizers the 3 open questions (section 7), validate the idea | — |
| **Discord: #find-teammates** | One teammate could take the demo and write-up, halving the risk | Teams of up to 3 |
| **Finalist perk** (top 20) | $200 HF credits to polish between 16 and 22 Nov | Only if shortlisted |
| **Winners' prizes** | $1,162–$3,662 + Lexsi/Cohere credits + 6 months of HF Pro | — |
| **Interviews** | DevRel/research internships, research scientist/engineer roles | For standout candidates |
| **Participation certificate** | For everyone who takes part | — |
| **Referral programme** | $5–$90 in milestone rewards for teams that register through your link and submit | Registration closed on 8 Nov, so probably no longer useful |

## 6. Day-by-day plan

| Day | Date | Work | Done when |
|---|---|---|---|
| 1 | Mon 10 Nov | Accept the Tiny Aya licence. Join Discord, ask the 3 questions. Collect 15–20 PDFs (current + superseded). Build the index. Test Python 3.12 + AgentTune on Kaggle | Index built; AgentTune quickstart runs on Kaggle |
| 2 | Tue 11 Nov | Generate and validate questions. Start teacher runs. Score the 3 baselines on the Mac | Train/dev/test frozen; baselines scored |
| 3 | Wed 12 Nov | Finish teacher runs. Train the LoRA on Kaggle. Evaluate | First before/after table |
| 4 | Thu 13 Nov | Gradio demo (base vs tuned side by side, searches, retrieved clauses, citations). Push to a free Space | Demo works from a public link |
| 5 | Fri 14 Nov | HF model + dataset collections public. HF Article with metrics, method, limits, ROI. GitHub README with reproduction steps | All links public |
| — | Sat 15 Nov | Check every link logged out. **Submit.** | Submission confirmed |

**If behind schedule:** shrink the training set to 400 runs, drop multi-hop items, and keep the four metrics. Never cut the before/after table or the public links: they are required.

## 7. Questions for Discord (day 1)

1. May one submission list several problem statements (4 primary, 5 and 3 supporting)?
2. Is reusing my own pre-hackathon chunking and validation code acceptable if declared?
3. Is gpt-oss-120b acceptable as the teacher (supplementary model)?

## 8. Submission checklist

- [ ] Registered email
- [ ] Abstract (< 500 words)
- [ ] Public HF **model** collection: PolicyGuard LoRA adapter (+ merged GGUF for the demo)
- [ ] Public HF **dataset** collection: corpus chunks with version tags, Q&A set, teacher trajectories, evaluation outputs
- [ ] HF Article write-up
- [ ] GitHub repo with README and reproduction notes
- [ ] HF Space demo link
- [ ] Repos used (AgentTune, AuditKIT), model (Tiny Aya Global), Track 3, problem statements 4 (+5, 3)
