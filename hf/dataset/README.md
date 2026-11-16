---
license: cc-by-nc-4.0
language:
- en
task_categories:
- question-answering
- text-generation
tags:
- rag
- agent
- insurance
- irdai
- synthetic
- lexsi-hackathon
size_categories:
- n<1K
configs:
- config_name: questions
  data_files:
  - split: train
    path: questions_train.jsonl
  - split: test
    path: questions_test.jsonl
- config_name: trajectories
  data_files:
  - split: train
    path: trajectories_train.jsonl
  - split: train_v5
    path: trajectories_train_v5.jsonl
- config_name: corpus
  data_files:
  - split: train
    path: corpus_chunks.jsonl
---

# PolicyGuard IRDAI QA

Questions, agent trajectories and the search corpus used to train and evaluate [PolicyGuard](https://huggingface.co/bhavyananda17/policyguard-tiny-aya-global-lora), a Tiny Aya Global search agent for Indian health-insurance rules. Built for the Lexsi AI Safety & Alignment Hackathon (Track 3). Code: [github.com/bhavyananda17/lexsi_labs_project](https://github.com/bhavyananda17/lexsi_labs_project).

## Source documents

Public IRDAI circulars from irdai.gov.in (IRDAI is not affiliated with this dataset):

| Document | Ref | Date | Status |
|---|---|---|---|
| Master Circular on Health Insurance Business | IRDAI/HLT/CIR/PRO/84/5/2024 | 29 May 2024 | current |
| Annexure to the same circular | same | 29 May 2024 | current |
| Master Circular on Standardization of Health Insurance Products | IRDAI/HLT/REG/CIR/193/07/2020 | 22 July 2020 | replaced by the 2024 circular |

## Configs

### `questions` (237 train, 98 test)

| Field | Meaning |
|---|---|
| `id` | `train-XXXX` / `test-XXXX` |
| `type` | `single_hop` (one current passage), `multi_hop` (two passages), `superseded` (the 2024 rule differs from the replaced 2020 rule), `not_covered` (the documents do not answer it) |
| `question`, `answer` | question and reference answer (from current documents only) |
| `facts` | key facts, verbatim spans of the source passage |
| `source_chunks` | ids of the passages in `corpus` that answer it |
| `sub_queries` | two-rule items: one search query per passage |
| `old_chunk`, `old_fact`, `old_rule`, `topic` | replaced-rule items: the 2020 passage and rule that no longer apply |

| | Train | Test |
|---|---|---|
| single_hop | 132 | 61 |
| multi_hop | 19 | 8 |
| superseded | 23 | 8 |
| not_covered | 63 | 21 |

Train and test share no source passage: items that share a passage, or a replaced-rule passage, are kept on the same side. 3 test items (`test-0022`, `test-0062`, `test-0065`) come from blank policy templates whose reference value is literally "XXX"; they are excluded from scoring (`test_exclusions.json`).

### `trajectories` (373 in `train`, 375 in `train_v5`)

ReAct conversations used for supervised fine-tuning. `train` built the v4 adapter. `train_v5` built the current v5 adapter: it shows 6 search results per observation instead of 4, starts about 40% of trajectories with a short keyword query, and has 49 self-distilled answers taken from 6-result answers. Both have the fields `messages` (system / user / assistant turns), `type`, and optional `aug` and `answer_source`.

- **Format:**
  - Assistant turns are `Thought: …` followed by `Action: search_docs("…")` or `Final Answer: … [chunk N]`.
  - Search results come back as user turns starting with `Observation:`.
- **Search steps:** real calls to the index, built by code.
- **Final answers:** from the teacher, or `answer_source: "self_distilled"` for answers taken from the untrained model's own always-search answers (79 in `train`, 49 in `train_v5`).
- **Variants (`aug`):**
  - `retry`: a vague first search misses, a sharper second search finds the passage.
  - `buried`: the answering passage is the last result.
  - `nearmiss`: on-topic results with the answering passages removed, ending in "not found".

### `corpus` (495 passages)

About 140-word chunks of the three circulars, with `chunk_id`, `doc`, `page`, `status` (current / superseded), `ref`, `date` and `text`. The agent sees the title, date and page in each result, but not `status`.

## How it was made

1. **Questions:** generated from the passages on Groq's free tier.
   - `single_hop` and `multi_hop`: openai/gpt-oss-20b.
   - `superseded`: openai/gpt-oss-120b.
   - `not_covered`: qwen/qwen3.8-27b, then checked by retrieval plus an LLM that the documents do not answer it.
2. **Validation:** key facts must appear verbatim in the source passage. Near-duplicate questions (MiniLM cosine above 0.9) were dropped, and 4 weak replaced-rule topics were excluded by hand. 377 generated, 335 kept.
3. **Trajectories:** built with real searches. Items whose source passage could not be retrieved were dropped (9).

## Also included

- `metrics_summary.json`: deterministic metrics for every evaluated setup.
- `judge_summary.json`: LLM-judge correctness for every evaluated setup.

## Limitations

- **Synthetic questions:** written by LLMs from the passages; they follow the circulars' wording more closely than real policyholder questions would.
- **Not human-verified:** "not_covered" items were checked automatically.
- **Small replaced-rule set:** 31 items in total, 8 in test, from a few topics.
- **Not legal or financial advice.**

## License

CC-BY-NC-4.0.
