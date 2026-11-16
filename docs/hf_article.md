---
title: "PolicyGuard: teaching a 3B model to search insurance rules instead of guessing"
thumbnail: https://raw.githubusercontent.com/bhavyananda17/lexsi_labs_project/main/docs/figures/correct_ci.png
authors:
- user: bhavyananda17
tags:
- agents
- rag
- lora
- fine-tuning
- evaluation
---

# PolicyGuard: teaching a 3B model to search insurance rules instead of guessing

*Lexsi AI Safety & Alignment Hackathon, Track 3 (Agentic Fine-tuning), problem statement 4: fine-tuning for retrieval-augmented generation. It also touches problem statements 5 (synthetic data) and 3 (distillation).*

**Links:** [demo](https://huggingface.co/spaces/bhavyananda17/policyguard-demo) · [adapter](https://huggingface.co/bhavyananda17/policyguard-tiny-aya-global-lora) · [dataset](https://huggingface.co/datasets/bhavyananda17/policyguard-irdai-qa) · [code](https://github.com/bhavyananda17/lexsi_labs_project) · [collection](https://huggingface.co/collections/bhavyananda17/policyguard-lexsi-hackathon-6af8a31e037f3d7eaf41b4b4)

## TL;DR

- I fine-tuned [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B) with LoRA to act as a search agent over Indian health-insurance circulars. It searches, cites passages as `[chunk N]`, and says "not found" instead of guessing.
- **What fine-tuning bought.** In the same agent loop, judged-correct answers go from **4% to 26%** (v4) and then **38%** (v5, the released adapter). Citing the right passage goes from **3% to 45%**, and wrongly refusing an answerable question drops from 14% to 1%.
- **What it didn't buy.** It does **not** clearly beat a fixed "always search once, then answer" pipeline built on the same untrained model. With 4 results that pipeline scores 32% and the intervals overlap. With 6 results it scores 47%, the best setup I measured.
- **A fifth round closed part of the gap.** v5, trained on 6-result observations with fuller self-distilled answers, is 12 points above v4 (paired 95% CI +3 to +23). It still trails the fixed 6-result pipeline, though that gap is no longer statistically clear. The cost: it catches fewer unanswerable questions.
- **The grading holds up.** A second judge model agrees on 80% of verdicts (κ = 0.70) and ranks the setups the same way.
- **Cost.** Everything ran on one laptop (MacBook Pro M5 Pro, 24 GB) and the free Groq tier, at zero cost. Training took about an hour with MLX and touched 0.10% of the parameters.

![Correct answers with 95% confidence intervals for each setup](https://raw.githubusercontent.com/bhavyananda17/lexsi_labs_project/main/docs/figures/correct_ci.png)

## The problem: rules that change, and models that make them up

Indian health insurance is regulated by IRDAI circulars, and newer circulars replace older ones. The May 2024 Master Circular on Health Insurance Business replaced the July 2020 standardization circular, and several numbers changed:
- **Moratorium period:** from 8 years to 60 months.
- **Pre-existing-disease waiting period:** from 48 to 36 months.
- **Cancellation notice:** from 15 to 7 days.

A model can fail here in two ways:

1. **It doesn't look.** It answers from memory and invents a rule and a citation.
2. **It looks at the wrong document.** It reads the replaced 2020 rule and gives a confident, outdated answer.

The untrained Tiny Aya shows the first failure clearly. Asked how often the Master Circular is reviewed, it doesn't search at all:

> **Untrained:** The Master Circular is reviewed biennially or as required to ensure alignment with IRDAI's evolving regulations. [chunk 12]
>
> **PolicyGuard:** *(searches "How often will this Master Circular be reviewed?")* The Master Circular is reviewed every year unless review or repeal is warranted earlier. [chunk 1]

Small models are the ones that would actually run on a phone or in a call centre at low cost. So the question was whether a small, cheap fine-tune can make a 3B model *behave* like a careful search agent.

## Setup

### Corpus and search

- **Documents:** three public IRDAI PDFs: the 2024 Master Circular, its annexure, and the replaced 2020 circular.
- **Chunks:** cleaned page by page and split into 495 chunks of about 140 words.
- **Search:** hybrid. MiniLM dense embeddings and TF-IDF are merged with reciprocal rank fusion, and the agent gets the top 4 results.
- **What the agent sees:** each result's document title, date and page, but **not** whether the document is current or replaced. It has to work that out from the dates.

### Agent format

Tiny Aya's chat template has no tool role, so I used a plain-text ReAct protocol. This is PolicyGuard's real transcript for the review question above, with the search results shortened:

```
Thought: I need to check what the IRDAI documents say about this.
Action: search_docs("How often will this Master Circular be reviewed?")

Observation:
[chunk 0] IRDAI Master Circular on Health Insurance Business (29 May 2024) | page 1
... B. The Master Circular shall be reviewed every year unless review or repeal is warranted earlier. ...
[chunk 1] IRDAI Master Circular on Health Insurance Business (29 May 2024) | page 2 ...

Thought: The results answer the question.
Final Answer: The Master Circular is reviewed every year unless review or repeal is warranted earlier. [chunk 1]
```

The answer is right, but the sentence is in chunk 0 and the model cites chunk 1. Picking the right number among adjacent chunks is a typical small-model slip, and it's why "cites the right passage" is measured separately from correctness.

Search results go back to the model as a user turn starting with `Observation:`. The agent can search at most 3 times per question.

### Synthetic data (problem statement 5)

**Questions.** Groq's free tier wrote 377 questions from the passages, of four kinds:

| Kind | Meaning | Written by |
|---|---|---|
| Single-rule | answered by one current passage | gpt-oss-20b |
| Two-rule | needs two passages | gpt-oss-20b |
| Replaced-rule | the 2024 rule differs from the 2020 rule | gpt-oss-120b |
| Not answered | the documents don't answer it | qwen3.8-27b, then checked by retrieval plus an LLM |

**Filtering.** Every question's key facts had to appear **verbatim** in its source passage. Near-duplicates (MiniLM cosine above 0.9) were dropped. 335 questions survived.

**Train/test split.** Train and test **share no source passage**. Items that share a passage, or a replaced 2020 passage, stay on the same side. The result is 237 training questions and 98 test questions; 3 broken test items are excluded, leaving 95.

**Trajectories.** The search steps in the training trajectories are **real calls to the index, built by code**. They are not imagined by a teacher model, so the model learns from the same messy results it will see at test time. Training items whose source passage the search couldn't find were dropped.

**Variants added to teach specific behaviour:**
- **Search again (60):** a vague first search misses, and a sharper second search finds the passage.
- **Answer in the last result (60):** the answering passage is moved to the 4th result, so the model learns to read all of them.
- **Near-miss "not found" (25):** the results are on topic, but the passages that answer the question are removed, and the answer is a refusal. Three different refusal phrasings are used.

### Self-distilled final answers (problem statement 3)

In the final round, 79 teacher answers were swapped for the **untrained Tiny Aya's own** answers from the fixed always-search pipeline on the same results. A swap was made only when strict filters passed:
- the same chunks were retrieved;
- the answer overlaps the reference well and hits a key fact;
- it is 120 words or fewer;
- it cites only chunks that were shown;
- for replaced-rule items, it doesn't use the old rule.

The idea is that answers written in the model's own voice are easier to learn than a much larger teacher's style.

### Training

| Setting | Value |
|---|---|
| Method | MLX LoRA, rank 16, alpha 32 |
| Where | q/k/v/o attention projections of the top 16 layers (20–35) |
| Trainable parameters | 3.4M (0.10%) |
| Data | 339 trajectories split into 781 per-turn examples; about 20% "not found" |
| Loss | assistant turns only: search decisions, queries and answers, never the retrieved text |
| Schedule | 3 epochs, learning rate 1e-4, batch 2, about 50 minutes (v4) to an hour (v5) on an M5 Pro |
| Validation loss | 2.79 → 0.43 |

For people without a Mac, the adapter is also published in PEFT format. The conversion produces the same greedy outputs as MLX on the questions I checked.

## Evaluation

### Test set

95 held-out questions:

| Kind | Questions |
|---|---|
| Single-rule | 61 |
| Two-rule | 8 |
| Replaced-rule | 8 |
| Not answered by the documents | 21 |

Three more test items were excluded. They came from blank policy templates whose "answer" is literally `XXX`.

### Setups compared

All setups use the same base model, the same search index and greedy decoding:
- **No search:** the question only.
- **Fixed pipeline:** always search the question once, then answer from the top 4 results.
- **Untrained agent loop:** the untrained model decides when to search.
- **PolicyGuard agent loop:** the fine-tuned model decides when to search.

### Metrics

- **LLM judge.** qwen3.8-27b grades each answer on the 74 answerable questions as correct, partly correct or wrong, given the reference answer and the verbatim key facts. Every call is cached, so the numbers reproduce offline.
- **Deterministic metrics:**
  - citing the right passage;
  - "not found" precision, recall and F1 on the 21 unanswerable questions;
  - wrongly saying "not found";
  - searches per question;
  - use of the replaced rule;
  - word overlap with the reference.
- **Confidence intervals:** 95% bootstrap.

## Results

| Setup | Correct (95% CI) | Correct or partly correct | Wrongly says "not found" | Cites the right passage | "Not found" P / R / F1 |
|---|---|---|---|---|---|
| Untrained, agent loop | 4% (0–8) | 24% | 14% | 3% | 0.29 / 0.19 / 0.23 |
| PolicyGuard v4, agent loop | 26% (16–35) | 49% | 3% | 42% | 0.78 / 0.33 / 0.47 |
| **PolicyGuard v5, agent loop (released, see [round 5](#what-changed-across-five-training-rounds))** | **38% (27–50)** | **53%** | **1%** | **45%** | **0.80** / 0.19 / 0.31 |
| Untrained, fixed pipeline | 32% (23–43) | 54% | 11% | 30% | 0.56 / 0.48 / 0.51 |

**1. Fine-tuning makes a working agent.**
- The untrained model searches on only 10% of questions and invents citations.
- PolicyGuard searches on essentially every question (1.07 searches per question), searches again with sharper terms when the first search misses, and cites a passage it actually saw.

**2. Through v4, the fixed pipeline was as good or better on accuracy.** (v5 changes part of this; see round 5 below.)
- On correctness, the fixed pipeline is 6 points ahead of v4, inside overlapping confidence intervals.
- PolicyGuard is better on some behaviour:
  - It cites the right passage more often (42% vs 30%).
  - It rarely refuses answerable questions (3% vs 11%).
  - When it does say "not found", it is usually right (precision 0.78 vs 0.56).
- But it misses more of the truly unanswerable questions (recall 0.33 vs 0.48).

I think this is the honest headline. On a corpus this small, with a search that finds the source passage 71% of the time at top 4, "always search" is a strong baseline. An agent loop has to *earn* its extra flexibility. A 3B model given 339 trajectories gets close to that baseline but doesn't pass it.

**3. Giving every setup more search results widens the gap.** I re-ran both setups with the top 6 results instead of 4, without retraining. The source passage is then in the results 81% of the time instead of 71%.

| Setup | Correct, 4 results → 6 results | Cites the right passage |
|---|---|---|
| PolicyGuard v4 | 26% → 30% (CI 20–41%) | 42% → 35% |
| Untrained, fixed pipeline | 32% → **47%** (CI 35–58%) | 30% → 43% |

The fixed pipeline gains 15 points; PolicyGuard gains 4 and cites less precisely. PolicyGuard only ever saw 4-result observations in training, and it doesn't use the extra context well. So the best setup I measured is the *untrained* model in a fixed 6-result pipeline. That's the bar a future version has to clear, and it suggests training with 6-result observations, which is what v5 does (below).

**4. A second judge agrees.** I re-graded the three main setups with gpt-oss-120b.
- The two judges agree on 80% of 222 verdicts, with Cohen's κ = 0.70.
- gpt-oss-120b is more lenient: it scores 34% correct for PolicyGuard, 43% for the fixed pipeline and 12% for the untrained agent. Most disagreements are "partial" upgraded to "correct".
- It ranks the setups the same way, so the conclusions above don't depend on the choice of judge.

**5. The replaced-rule question is not answered.** With 8 replaced-rule test items from about 2 topics, every setup used the old rule on 25–50% of them. I make no claim either way.

## What changed across five training rounds

| Version | Change | Correct | Wrongly says "not found" | "Not found" F1 |
|---|---|---|---|---|
| v1 | Teacher answers, 28% "not found" examples | not judged | 26% | 0.57 |
| v2 | 18% "not found", plus 60 "search again" examples | 24% | 18% | 0.55 |
| v3 | Plus answer in the last result, near-miss "not found", 3 epochs | 26% | 0% | 0.50 |
| v4 | Plus 79 self-distilled final answers | 26% | 3% | 0.47 |
| v5 | 6-result observations, 40% keyword first queries, fuller distilled answers | **38%** | 1% | 0.31 |

The clearest lesson is about **refusals**. v1 saw one fixed "not found" sentence many times and learned it as an easy exit: it refused 26% of questions it could have answered. Two changes moved it back toward answering:
- cutting the share of "not found" examples;
- varying the refusal wording and adding *near-miss* examples, where the results are on topic but the answer isn't there.

By v3 it never wrongly refused, and in exchange it caught fewer unanswerable questions. **The mix of "not found" data sets the balance between answering and refusing.** The mix of training data mattered more than any hyperparameter I changed.

Self-distillation (v4) didn't move correctness. It slightly raised the share of answers judged correct or partly correct (45% to 49%) and the word overlap with the reference (0.46 to 0.52). Both changes are within noise.

**v5: training with the context it gets.** After the 6-result experiment, I changed three things in the data and kept everything else fixed:
- observations show 6 results instead of 4;
- about 40% of trajectories start with a short keyword query instead of the question copied word for word;
- the self-distilled answers come from the untrained model's 6-result pipeline and may run to 150 words.

| Setup | Correct (95% CI) | Correct or partly | Wrongly says "not found" | "Not found" P / R / F1 |
|---|---|---|---|---|
| PolicyGuard v4, 4 results | 26% (16–35) | 49% | 3% | 0.78 / 0.33 / 0.47 |
| **PolicyGuard v5, 4 results** | **38% (27–50)** | 53% | 1% | 0.80 / 0.19 / 0.31 |
| PolicyGuard v5, 6 results | 32% (22–43) | 46% | 1% | 0.83 / 0.24 / 0.37 |
| Untrained, fixed pipeline, 6 results | 47% (35–58) | 58% | 7% | 0.62 / 0.38 / 0.47 |

On the same questions, v5 beats v4 by 12 points (it gains 12 questions and loses 3; paired 95% CI +3 to +23). It is the first change across five rounds that clearly moved correctness. It still trails the fixed 6-result pipeline by 9.5 points (CI −3 to +20). Three things I didn't expect:
- **It is worse with 6 results than with 4**, although it was trained on 6. The gap (7 questions vs 3) is within noise.
- **It catches fewer unanswerable questions.** Recall fell to 4 of 21, though its refusals stay precise.
- **It mostly still copies the question** as its first query (82 of 98, down from 97), so the keyword-query training only partly took.

v5 is now the released adapter; v4 stays in the Hugging Face repo's commit history.

## Two measurement mistakes I caught

I'm including these because they changed the conclusion.

**1. Word overlap rewards copying.** My first headline metric was word overlap with the reference answer. It favours long answers that paste the passage, which made the fixed pipeline look much stronger than it is. That's why I added the LLM judge.

**2. The "not found" detector was too narrow.**
- My first detector missed phrasings like "not explicitly stated", which the untrained model often uses.
- That undercounted the fixed pipeline's refusals and made PolicyGuard look about twice as good at spotting unanswerable questions.
- Widening the detector raised the fixed pipeline's "not found" F1 from 0.22 to 0.51, and the apparent PolicyGuard advantage disappeared.
- Every number in this article uses the corrected detector.

## Limitations

- **Small test set.** 74 answerable and 21 unanswerable questions. Differences of a few points are noise.
- **Few replaced-rule items.** Only 8, so the project's central safety question isn't settled: does the model prefer the current rule?
- **Synthetic questions.** LLMs wrote them from the passages, so they lean on the circulars' own wording. "Not answered" items were checked automatically, not by a person.
- **LLM judges only.** Two judge models agree well (κ = 0.70), but no human labels were collected at scale.
- **Search is the ceiling.** The source passage is in the top 4 results for 71% of answerable questions, and 52 of the 495 chunks are longer than MiniLM's 256-token window.
- **Not legal or financial advice.** Only two IRDAI documents are covered.

## What I'd try next

- **Close the last gap to the fixed pipeline.** v5 shows that matching the training context helps (26% → 38%). Next would be more keyword-query trajectories, so the agent actually searches differently from the fixed pipeline, and a reranker, which would help every setup.
- **Win back the unanswerable questions.** v5 traded "not found" recall for correctness; near-miss examples written at 6 results may restore it.
- **More replaced-rule items.** Hand-checked questions across more topics, so the current-versus-old-rule question can actually be measured.
- **Training on the agent's own trajectories.** For example, rejection sampling over the agent's own runs, rather than only teacher-shaped ones.
- **A real comparison against the always-search pipeline.** It should be the baseline to beat, not the untrained agent.

## Try it

- **[Demo](https://huggingface.co/spaces/bhavyananda17/policyguard-demo):** all 95 test questions side by side, with grades and every search step. It's a static page, because free Gradio Spaces now need a paid plan.
- **Live questions:** `python demo/app.py` from the repo (an Apple Silicon Mac answers in about a minute), or load the PEFT adapter with `transformers`. You need to accept the base model's licence first.
- **Everything else:** code, every run's transcripts and the cached judge calls are in the [GitHub repo](https://github.com/bhavyananda17/lexsi_labs_project).

## Disclosure

- **Coding assistant:** code and write-up were produced with help from Claude (Anthropic). I reviewed the design, experiments and results.
- **Generated data:** questions, reference answers and most training answers were generated by gpt-oss-120b, gpt-oss-20b and qwen3.8-27b on Groq's free tier. The judge is qwen3.8-27b.
- **Sources:** public IRDAI circulars. IRDAI isn't affiliated with this project.
- **Licences:** the adapter and dataset are CC-BY-NC-4.0, following the base model; the code is MIT.
