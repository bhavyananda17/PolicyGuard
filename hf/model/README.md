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

95 test questions that share no source passage with training (74 answerable, 21 not answered by the documents). Correctness is graded by an LLM judge (qwen3.8-27b) against reference answers.

| Setup | Correct | Correct or partly correct | Says "not found" wrongly | Cites the right passage | "Not found" F1 |
|---|---|---|---|---|---|
| Tiny Aya, agent loop, untrained | 4% | 24% | 14% | 3% | 0.23 |
| **Tiny Aya + this adapter, agent loop** | **26%** | **49%** | **3%** | **42%** | 0.47 |
| Tiny Aya, fixed pipeline (always search once), untrained | 32% | 54% | 11% | 30% | 0.51 |

In the same agent loop, the adapter takes correct answers from 4% to 26% and right-passage citations from 3% to 42%. It does not beat a fixed always-search pipeline on accuracy (the confidence intervals overlap: 16–35% vs 23–43%), but it cites better and rarely refuses an answerable question. Full tables, confidence intervals and training history are in the GitHub README.

## Files

- `adapter_model.safetensors`, `adapter_config.json`: PEFT format, for `transformers` + `peft`.
- `mlx/adapters.safetensors`, `mlx/adapter_config.json`: the original MLX adapter, for `mlx-lm` on Apple Silicon.

The PEFT version was converted from the MLX adapter (`src/export_peft.py` in the repo) and produces the same greedy outputs on the checked test questions.

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

- **Data:** 339 agent trajectories (781 per-turn examples) from [policyguard-irdai-qa](https://huggingface.co/datasets/bhavyananda17/policyguard-irdai-qa).
  - **Questions and reference answers:** written by gpt-oss-120b, gpt-oss-20b and qwen3.8-27b on Groq.
  - **Search steps:** real calls to a hybrid index (MiniLM + TF-IDF) over the circulars.
  - **Final answers:** 79 come from the untrained model's own always-search answers, kept where they passed quality filters.
  - **Composition:** about 20% "not found" examples, plus variants for searching again, answers in the last result, and near-miss "not found" cases.
- **Method:** LoRA rank 16, scale 2.0 (alpha 32), dropout 0.05, on q/k/v/o attention projections of layers 20–35. That is 3.4M trainable parameters (0.10%).
- **Optimisation:** loss on assistant turns only. Adam, learning rate 1e-4, batch 2, 1,171 steps (3 epochs), max length 3,072.
- **Hardware:** MacBook Pro M5 Pro 24 GB with `mlx-lm` 0.32.2, about 50 minutes.

## Limitations

- **Small test set:** 74 answerable questions, so differences of a few points sit inside the confidence intervals.
- **Replaced rules:** only 8 test questions are about rules that were replaced, so whether the model follows the newer rule is not established.
- **Synthetic questions:** they were written by LLMs from the passages, not by policyholders.
- **Search is a ceiling:** the source passage is in the top 4 results for 71% of answerable questions.
- **Missed details:** a 3B model reading dense regulatory text often drops a condition or a date.

## License

CC-BY-NC-4.0, following the base model. Non-commercial use only.
