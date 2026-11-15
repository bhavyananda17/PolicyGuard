# PolicyGuard: fine-tuning a 3B model into a citing, refusing search agent for insurance regulation

**Track 3, problem statement 4 (fine-tuning for RAG); also touches 5 (synthetic data) and 3 (distillation).**

Indian health-insurance rules come from IRDAI circulars, and newer circulars replace older ones. The 2024 Master Circular, for example, cut the moratorium period from 8 years to 60 months. Small, cheap models fail here in two ways: they invent rules and citations, or confidently state the outdated rule. PolicyGuard asks whether a zero-cost LoRA fine-tune can make Tiny Aya Global (3.35B) a careful search agent: one that searches first, cites its passage and says "not found" instead of guessing.

**Method.**
- **Search index:** 495 chunks of three public IRDAI circulars (2024 current, 2020 replaced) with hybrid MiniLM and TF-IDF retrieval. Each result shows its date but not whether it is current.
- **Questions:** Groq's free tier generated single-rule, two-rule, replaced-rule and unanswerable questions. Key facts had to appear verbatim in the source; 335 were kept, split so train and test share no source passage.
- **Trajectories:** plain-text ReAct (`Thought` / `Action: search_docs` / `Observation` / `Final Answer [chunk N]`). Search steps are real calls to the index, with variants for searching again, answers in the last result and near-miss refusals. In the final round, 79 answers were self-distilled from the untrained model's own filtered outputs.
- **Training:** MLX LoRA at rank 16 on 0.10% of the parameters, about 50 minutes on a laptop.

**Evaluation.** 95 held-out questions, with the same model, index and greedy decoding for every setup:
- an LLM judge (qwen3.8-27b) on the 74 answerable questions;
- deterministic citation, refusal and old-rule metrics;
- bootstrap confidence intervals.

**Results.**
- **Fine-tuning produces a working agent.** In the same agent loop, judged-correct answers rise from 4% to 26% and right-passage citations from 3% to 42%. Wrong refusals fall from 14% to 3%, and "not found" precision is 0.78.
- **It does not beat a fixed always-search-once pipeline on the untrained model.** That pipeline scores 32% correct (intervals overlap). With 6 results instead of 4 it reaches 47%, while PolicyGuard, trained on 4, reaches only 30%.
- **The training mix sets the answer/refuse balance.** One repeated refusal phrase taught the model to refuse 26% of answerable questions. Varied wording and near-miss examples reduced this to 0–3%, at the cost of catching fewer unanswerable questions.

**Limitations.**
- The test set is small. With 8 replaced-rule items, current-rule preference is not established.
- Questions are synthetic. A second judge (gpt-oss-120b) agrees on 80% of verdicts (κ = 0.70), but there are no human labels.
- Search finds the source passage for only 71% of questions, which caps every setup.
- I found and fixed two of my own measurement errors (a length-biased overlap metric and a narrow refusal detector). Fixing the detector removed an apparent PolicyGuard advantage.

**Artifacts.** [Adapter](https://huggingface.co/bhavyananda17/policyguard-tiny-aya-global-lora) (PEFT and MLX), [dataset](https://huggingface.co/datasets/bhavyananda17/policyguard-irdai-qa), [demo](https://huggingface.co/spaces/bhavyananda17/policyguard-demo) with all 95 test transcripts, and [code](https://github.com/bhavyananda17/lexsi_labs_project) with cached LLM calls.
