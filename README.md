# PolicyGuard

**A small search agent for Indian health-insurance rules, fine-tuned from Tiny Aya Global (3.35B) on a laptop at zero cost.**

PolicyGuard answers questions about IRDAI health-insurance rules by searching the official circulars, citing the passages it used, and saying "I could not find this" instead of guessing. It was built for the [Lexsi AI Safety & Alignment Hackathon](https://hackathon.lexsi.ai/), Track 3 (Agentic Fine-tuning), problem statement 4: *fine-tuning for retrieval-augmented generation*. It also touches problem statements 5 (synthetic data) and 3 (distillation).

- Base model: [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global)
- **Demo:** [huggingface.co/spaces/bhavyananda17/policyguard-demo](https://huggingface.co/spaces/bhavyananda17/policyguard-demo): side-by-side answers on all 95 test questions, with grades and every search step
- Adapter: [`bhavyananda17/policyguard-tiny-aya-global-lora`](https://huggingface.co/bhavyananda17/policyguard-tiny-aya-global-lora) (PEFT and MLX formats)
- Dataset: [`bhavyananda17/policyguard-irdai-qa`](https://huggingface.co/datasets/bhavyananda17/policyguard-irdai-qa) (questions, trajectories, corpus)
- Write-up: [docs/hf_article.md](docs/hf_article.md) · Abstract: [docs/abstract.md](docs/abstract.md)
- Collection: [PolicyGuard (Lexsi hackathon)](https://huggingface.co/collections/bhavyananda17/policyguard-lexsi-hackathon-6af8a31e037f3d7eaf41b4b4)
- Hardware: one MacBook Pro (M5 Pro, 24 GB). Training: about 50 minutes with MLX. Cost: zero (free Groq tier for data generation).

## Why this problem

Indian health insurance rules are set by IRDAI circulars, and newer circulars replace older ones. The May 2024 Master Circular replaced the 2020 standardization circular, and some numbers changed: the moratorium period went from 8 years to 60 months, the pre-existing-disease wait from 48 to 36 months, and the cancellation notice from 15 to 7 days. A model that reads the old circular gives a confident, wrong answer. A model that does not search at all makes up rules and citations.

The untrained Tiny Aya shows both problems. Asked how often the Master Circular is reviewed, it answers without searching and invents a citation:

> **Untrained:** The Master Circular is reviewed biennially or as required to ensure alignment with IRDAI's evolving regulations. [chunk 12]
>
> **PolicyGuard:** *(searches "How often will this Master Circular be reviewed?")* The Master Circular is reviewed every year unless review or repeal is warranted earlier. [chunk 1]

## Results

Test set: 95 questions that share no source passage with training (61 single-rule, 8 two-rule, 8 replaced-rule, 21 that the documents do not answer; 3 broken items excluded, see [below](#evaluation)). All setups use the same model, the same search index and greedy decoding.

**Answer correctness (LLM judge, 74 answerable questions)**

| Setup | Correct | 95% CI | Correct or partly correct | Says "not found" wrongly |
|---|---|---|---|---|
| Untrained, agent loop (model decides when to search) | 4% | 0–8% | 24% | 14% |
| **PolicyGuard v4, agent loop** | **26%** | 16–35% | **49%** | **3%** |
| Untrained, fixed pipeline (always search once, then answer) | 32% | 23–43% | 54% | 11% |

**Behaviour (deterministic metrics, 95 questions)**

| Setup | Searches per question | Cites the right passage | "Not found" precision / recall / F1 | Gold-answer word overlap |
|---|---|---|---|---|
| Untrained, no search | 0 | 0% | 0.50 / 0.19 / 0.28 | 0.46 |
| Untrained, agent loop | 0.10 | 3% | 0.29 / 0.19 / 0.23 | 0.37 |
| **PolicyGuard v4, agent loop** | **1.07** | **42%** | **0.78** / 0.33 / 0.47 | 0.52 |
| Untrained, fixed pipeline | 1.00 (forced) | 30% | 0.56 / 0.48 / 0.51 | 0.66 |

What this shows, plainly:

1. **Fine-tuning turns Tiny Aya into a working search agent.** In the same agent loop, correct answers go from 4% to 26%, correct-or-partial from 24% to 49%, and citing the right passage from 3% to 42%. The untrained model searches on 10% of questions and invents citations; the tuned model searches every time and retries with sharper terms when the first search misses.
2. **It does not beat a fixed always-search pipeline on accuracy.** The fixed pipeline is ahead (32% vs 26% correct); the confidence intervals overlap, so with 74 questions we cannot say which is better. PolicyGuard cites the right passage more often (42% vs 30%), almost never refuses an answerable question (3% vs 11%), and is more precise when it does say "not found" (0.78 vs 0.56), but catches fewer unanswerable questions (recall 0.33 vs 0.48).
3. **The replaced-rule result is inconclusive.** With only 8 replaced-rule test questions from about 2 topics, every setup lands between 25% and 50% old-rule use, and we make no claim either way.
4. **Absolute accuracy is low for every setup.** Search finds the source passage for 55 of 77 answerable questions (71%) at 4 results, and a 3B model reading dense regulatory text often drops a condition or a date.

### How we got here (four training rounds)

| Version | Change | Correct (judge) | Says "not found" wrongly | "Not found" F1 |
|---|---|---|---|---|
| v1 | Teacher answers, 28% "not found" examples | not judged | 26% | 0.57 |
| v2 | 18% "not found", 60 "search again" examples | 24% | 18% | 0.55 |
| v3 | + answer passage moved to last result, near-miss "not found" examples, 3 epochs | 26% | 0% | 0.50 |
| v4 | + 79 final answers self-distilled from the untrained model's fixed-pipeline answers | 26% | 3% | 0.47 |

The main lesson: the share and kind of "not found" examples sets a trade-off between answering and refusing. v1 learned to refuse 26% of answerable questions; v3 never refuses but misses more unanswerable ones. A single "not found" sentence repeated in training made refusal an easy exit; varied wording and near-miss examples (on-topic results that do not answer) moved the model back toward answering.

## How it works

```
IRDAI PDFs (2024 current + 2020 replaced)
   │  build_index.py: page-wise cleaning, ~140-word chunks, hybrid search
   ▼                  (MiniLM dense + TF-IDF, reciprocal rank fusion)
search index (495 chunks)
   │  gen_questions.py: Groq free tier writes questions + answers from passages
   ▼                    (gpt-oss-120b, gpt-oss-20b, qwen3.8-27b); key facts must be verbatim
377 questions → 335 after dedupe
   │  build_dataset.py: train/test split with no shared passages; agent trajectories
   ▼                    built from REAL searches of the index (items whose source is not retrieved are dropped)
373 trajectories (+ retry, buried-answer and near-miss variants)
   │  distill_answers.py: swap in the untrained model's own fixed-pipeline answers where they pass filters
   ▼
MLX LoRA (rank 16, attention, top 16 layers, 3.4M params = 0.10%, 3 epochs, ~50 min on M5 Pro)
   │
   ▼
run_agent.py → score.py (deterministic metrics) + judge.py (LLM judge)
```

**Agent format.** Tiny Aya's chat template has no tool role, so the agent uses a ReAct text protocol: the model writes `Thought: …` then either `Action: search_docs("…")` or `Final Answer: … [chunk N]`, and search results come back as a user turn starting with `Observation:`. The model sees each result's document title and date but not whether it is current or replaced; it has to work that out from the dates. At most 3 searches per question.

**Training data.** The final answers were written by teacher models (gpt-oss on Groq) from the source passage, or, in v4, by the untrained Tiny Aya itself from the same search results. The search steps are real calls to the index, built by code, not written by a teacher. Loss is on assistant turns only (search decisions, queries, final answer), never on the question or the retrieved text.

| | Train | Test |
|---|---|---|
| Single-rule | 132 | 61 |
| Two-rule | 19 | 8 |
| Replaced-rule | 23 | 8 |
| Not answered by the documents | 63 | 21 (target 20–25%) |

The 237 training questions give 373 trajectories (239 single-rule, 16 two-rule, 30 replaced-rule, 88 "not found"), including 60 "search again", 60 "answer is the last result" and 25 near-miss "not found" variants. After holding out 18 for validation and capping "not found" at 20%, training uses 339 trajectories split into 781 per-turn examples.

## Evaluation

- **Setups.** `no_tools` (question only), `naive_rag` (fixed pipeline: always search the question once, answer from the top 4), `agent` (model decides), `agent_force` (must search before answering). For the tuned model, `agent` and `agent_force` give identical results because it always searches.
- **Deterministic metrics** ([score.py](src/score.py)): gold-answer word overlap, key-fact recall, wrong "not found" rate, "not found" precision/recall/F1 on the 21 unanswerable questions, replaced-rule use (numbers that appear only in the old rule), searches per question, cites the source passage. 95% bootstrap CIs in `results/summary.json`.
- **LLM judge** ([judge.py](src/judge.py)): Groq `qwen/qwen3.8-27b` grades each answer as correct / partial / wrong against the reference answer and the verbatim key facts. Refusals are graded without a call. Every call is cached in `data/cache/`, so the numbers are reproducible offline.
- **Exclusions.** 3 test questions came from blank IRDAI policy templates that literally say "XXX" where a value should be; they have no checkable answer and are excluded from every score ([test_exclusions.json](data/qa/test_exclusions.json)).
- **Measurement fixes we made along the way.** (a) Word overlap rewards long answers that copy the passage, so we added the LLM judge. (b) Our first "not found" detector missed phrasings like "not explicitly stated", which undercounted the fixed pipeline's refusals; widening it raised that pipeline's "not found" F1 from 0.22 to 0.51 and removed what had looked like a large PolicyGuard advantage. All tables above use the widened detector.

## Reproduce

Requirements: Apple Silicon Mac (MLX), Python 3.12, a Hugging Face token with access to `CohereLabs/tiny-aya-global` (gated: accept the license on its model page), and a free Groq API key only if you want to regenerate questions or re-run the judge without the cache.

```bash
git clone https://github.com/bhavyananda17/lexsi_labs_project && cd lexsi_labs_project
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env        # fill in HF_TOKEN (and GROQ_API_KEY if needed)
export $(grep -v '^#' .env | xargs)
```

Download the three circulars from irdai.gov.in into `data/raw/` with these names:

| File | Document |
|---|---|
| `current_mc_health_2024.pdf` | Master Circular on Health Insurance Business, 29 May 2024 ([source](https://irdai.gov.in/document-detail?documentId=4942918)) |
| `current_mc_health_2024_annexure.pdf` | Annexure to the same circular (same page) |
| `superseded_mc_standardization_2020.pdf` | Master Circular on Standardization of Health Insurance Products, 22 July 2020 ([source](https://irdai.gov.in/document-detail?documentId=395035)) |

Then:

```bash
python src/build_index.py                       # search index (data/index/, also committed)
python src/gen_questions.py                     # optional: questions are committed in data/qa/raw_*.jsonl; Groq replies are cached
python src/build_dataset.py                     # train/test split + trajectories -> data/sft/train.jsonl
python src/run_agent.py --mode naive_rag --split train --name distill_train_rag
(cd src && python distill_answers.py)           # -> data/sft/train_v4.jsonl
SFT_FILE=data/sft/train_v4.jsonl MLX_OUT=data/mlx_v4 NC_SHARE=0.2 python src/make_mlx_data.py
mlx_lm.lora -c configs/lora_v4.yaml             # -> adapters/policyguard_v4 (~50 min on M5 Pro)
for m in no_tools naive_rag agent agent_force; do python src/run_agent.py --mode $m; done
python src/run_agent.py --mode agent --adapter adapters/policyguard_v4 --name tuned_v4_agent
python src/score.py results/base_*.jsonl results/tuned_v4_agent.jsonl
(cd src && python judge.py ../results/base_naive_rag.jsonl ../results/base_agent.jsonl ../results/tuned_v4_agent.jsonl)
```

Ask a single question (prints the full search transcript):

```bash
python src/ask.py "What is the moratorium period for a health insurance policy?"
python src/ask.py --adapter none "..."          # untrained model, for comparison
```

Run the live demo locally (Gradio; compare tab plus live questions, about a minute per question on an M-series Mac):

```bash
python demo/app.py      # http://localhost:7860
```

The hosted demo is a static page (`demo_static/`), because free Gradio Spaces now need a paid plan; it shows the pre-run results only.

Earlier rounds are in `scripts/overnight.sh` (data, baselines, v1), `scripts/round2.sh`, `scripts/round3.sh` and `scripts/round4.sh`; their logs are in `logs/` and every run's answers and transcripts are in `results/`.

## Repository layout

```
src/
  build_index.py      PDF -> chunks -> FAISS + TF-IDF hybrid index
  search.py           the search_docs tool
  llm.py              Groq client with disk cache and free-tier rate-limit handling
  gen_questions.py    synthetic questions (single-rule, two-rule, replaced-rule, not answered)
  build_dataset.py    validation, leak-free split, agent trajectories and variants
  distill_answers.py  self-distilled final answers (v4)
  make_mlx_data.py    per-turn MLX training files, "not found" cap
  agent.py            system prompt and ReAct parser
  run_agent.py        evaluation harness (4 setups)
  score.py, judge.py  deterministic metrics, LLM judge
  ask.py              ask one question
configs/              MLX LoRA configs per round
data/qa/              questions, split, exclusions, generation report
data/sft/, data/mlx*  trajectories and training files
results/              every run's answers + transcripts, summary.json, judge/
```

## Limitations

- **Small test set.** 74 answerable and 21 unanswerable questions; differences of a few points are inside the confidence intervals. Only 8 replaced-rule questions, so the project's central safety question (does the model follow the newer rule?) is not answered by this evaluation.
- **Synthetic questions.** Questions were written by LLMs from the passages, not by policyholders, and lean toward the wording of the circulars. "Not answered" questions were checked by retrieval plus an LLM, not by a human.
- **Teacher and judge models.** Training answers come from gpt-oss (and, for 79 items, the untrained Tiny Aya); the judge is qwen3.8-27b. Judge verdicts were spot-checked by hand on a handful of items, not validated at scale.
- **Near-miss "not found" examples** are built by removing passages that contain a key fact verbatim; a remaining passage could still state the answer in other words.
- **Search is the ceiling.** The source passage is in the top 4 results for 71% of answerable questions (81% at top 6). 52 of 495 chunks exceed MiniLM's 256-token window.
- **Not legal or financial advice.** Two IRDAI documents only; real policies and later circulars may differ.

## AI assistance and data provenance

- Code and documentation were written with help from Claude (Anthropic) as a coding assistant; the author reviewed the design, experiments and results.
- Questions, reference answers and key facts were generated with `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and `qwen/qwen3.8-27b` on Groq's free tier; every request and response is in `data/cache/`.
- Source documents are public IRDAI circulars from irdai.gov.in; IRDAI is not affiliated with this project.

## License

Code: MIT. The LoRA adapter is a derivative of Tiny Aya Global and follows its CC-BY-NC-4.0 license (non-commercial). Dataset: CC-BY-NC-4.0.
