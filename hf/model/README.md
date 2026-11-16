---
base_model: CohereLabs/tiny-aya-global
library_name: peft
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-generation
tags:
- lora
- peft
- mlx
- agent
- rag
- react
- insurance
- irdai
- lexsi-hackathon
datasets:
- bhavyananda17/policyguard-irdai-qa
---

# PolicyGuard: Tiny Aya Global LoRA for IRDAI health-insurance search

A LoRA adapter that turns [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B) into a small search agent for Indian health-insurance rules. The agent searches IRDAI circulars, cites the passages it used, and says it could not find an answer instead of guessing.

Built for the Lexsi AI Safety & Alignment Hackathon, Track 3 (Agentic Fine-tuning), problem statement 4: fine-tuning for retrieval-augmented generation. Code, data pipeline and full evaluation: [github.com/bhavyananda17/lexsi_labs_project](https://github.com/bhavyananda17/lexsi_labs_project).

**Not legal or financial advice.** Trained and tested on two IRDAI documents only.

## Results

95 test questions that share no source passage with training (74 answerable, 21 not answered by the documents). Correctness is graded by an LLM judge (qwen3.8-27b) against reference answers. "Cites the answer" counts a cited passage that contains the answer.

This is **v5**, the fifth training round. The setups below use the same base model, search index and greedy decoding.

| Setup | Correct (95% CI) | Correct or partly correct | Says "not found" wrongly | Cites the answer | "Not found" P / R / F1 |
|---|---|---|---|---|---|
| Tiny Aya, agent loop, untrained | 4% (0–8) | 24% | 14% | 3% | 0.29 / 0.19 / 0.23 |
| Tiny Aya + v4 adapter, agent loop | 26% (16–35) | 49% | 3% | 61% | 0.78 / 0.33 / 0.47 |
| **Tiny Aya + this adapter (v5), agent loop** | **38% (27–50)** | **53%** | **1%** | **61%** | 0.80 / 0.19 / 0.31 |
| Tiny Aya, fixed pipeline (always search once), untrained | 32% (23–43) | 54% | 11% | 41% | 0.56 / 0.48 / 0.51 |
| Same fixed pipeline with 6 results instead of 4 | 47% (35–58) | 58% | 7% | 50% | 0.62 / 0.38 / 0.47 |

All rows use 4 search results unless stated.
- **v5 vs v4.** On the same questions v5 gets 12 right that v4 missed and misses 3 that v4 got, a net gain of 12 points (paired 95% CI +3 to +23).
- **v5 vs the fixed pipeline.** It is level with the 4-result pipeline (+5 points, CI −7 to +18). It is still behind the 6-result pipeline (−9.5 points, CI −20 to +3), which is the best setup measured.
- **Given 6 results, v5 scores 32%**, lower than with 4, although it was trained on 6. The gap (7 questions vs 3) is within noise.
- **The cost.** v5 catches only 4 of 21 unanswerable questions (v4 caught 7). When it does say "not found", it is right 80% of the time.

**Second judge.** gpt-oss-120b re-graded the untrained agent, v4 and the fixed pipeline. It agrees with qwen3.8-27b on 80% of 222 verdicts (Cohen's κ = 0.70), is more lenient, and ranks the setups the same way. v5 was graded by qwen3.8-27b only.

## Files

- `adapter_model.safetensors`, `adapter_config.json`: PEFT format, for `transformers` + `peft`.
- `mlx/adapters.safetensors`, `mlx/adapter_config.json`: the original MLX adapter, for `mlx-lm` on Apple Silicon.

The PEFT version was converted from the MLX adapter (`src/export_peft.py` in the repo) and reproduces the MLX greedy outputs on 8 of 10 checked agent turns. The other 2 diverge mid-answer where the top two tokens are within 0.12 logits, which is a numeric-precision difference.

## Use

The base model is gated: accept its license on the [model page](https://huggingface.co/CohereLabs/tiny-aya-global) and log in with a token first.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "CohereLabs/tiny-aya-global"
tok = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "bhavyananda17/policyguard-tiny-aya-global-lora")
```

With MLX:

```bash
huggingface-cli download bhavyananda17/policyguard-tiny-aya-global-lora --include "mlx/*" --local-dir policyguard
python -m mlx_lm generate --model CohereLabs/tiny-aya-global --adapter-path policyguard/mlx --prompt "..."
```

The adapter expects the PolicyGuard agent format: the system prompt from `src/agent.py`, then turns of `Thought: …` / `Action: search_docs("…")`, with search results returned as a user turn starting with `Observation:`, until `Final Answer: … [chunk N]`. Tiny Aya's chat template has no tool role, so tool results go in user turns. The full loop, with the search index, is `src/run_agent.py` and `src/ask.py` in the repo.

## Training

- **Data:** the `train_v5` trajectories in [policyguard-irdai-qa](https://huggingface.co/datasets/bhavyananda17/policyguard-irdai-qa): 341 used for training (778 per-turn examples) and 18 for validation.
  - **Questions and reference answers:** written by gpt-oss-120b, gpt-oss-20b and qwen3.8-27b on Groq.
  - **Search steps:** real calls to a hybrid index (MiniLM + TF-IDF) over the circulars. Each observation shows the top 6 results.
  - **First queries:** about 40% of trajectories start with a short keyword query instead of the question copied word for word.
  - **Final answers:** 49 come from the untrained model's own 6-result always-search answers (up to 150 words), kept where they passed quality filters. The rest are teacher answers.
  - **Composition:** 20% "not found" examples, plus variants for searching again, answers in the last result, and near-miss "not found" cases.
- **Method:** LoRA rank 16, scale 2.0 (alpha 32), dropout 0.05, on q/k/v/o attention projections of layers 20–35. That is 3.4M trainable parameters (0.10%).
- **Optimisation:** loss on assistant turns only. Adam, learning rate 1e-4, batch 2, 1,167 steps (3 epochs), max length 4,608.
- **Hardware:** MacBook Pro M5 Pro 24 GB with `mlx-lm` 0.32.2, about 60 minutes.
- **Earlier versions:** v1–v4 are described in the GitHub README. v4 (26% correct) was the previous release; it is in this repo's commit history.

## Limitations

- **Small test set:** 74 answerable questions, so differences of a few points sit inside the confidence intervals.
- **Replaced rules:** only 8 test questions are about rules that were replaced, so whether the model follows the newer rule is not established.
- **Synthetic questions:** they were written by LLMs from the passages, not by policyholders.
- **Search is a ceiling:** the source passage is in the top 4 results for 71% of answerable questions (81% in the top 6).
- **Fewer refusals of unanswerable questions:** v5 says "not found" on only 4 of 21 such questions; it more often answers from a loosely related passage.
- **LLM judges only:** two judge models agree well (κ = 0.70), but there are no human labels at scale.
- **Missed details:** a 3B model reading dense regulatory text often drops a condition or a date.

## License

CC-BY-NC-4.0, following the base model. Non-commercial use only.
