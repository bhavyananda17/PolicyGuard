---
title: PolicyGuard
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Fine-tuned Tiny Aya search agent for IRDAI insurance rules
models:
- CohereLabs/tiny-aya-global
- bhavyananda17/policyguard-tiny-aya-global-lora
datasets:
- bhavyananda17/policyguard-irdai-qa
---

# PolicyGuard demo

Tiny Aya Global (3.35B) fine-tuned with LoRA to search IRDAI health-insurance circulars, cite what it used, and say "not found" instead of guessing.

- **Compare on test questions:** pre-run answers from the untrained model (agent loop and always-search) and PolicyGuard on the held-out test set, with LLM-judge grades and full search steps.
- **Ask your own question:** runs the model live on CPU (1–3 minutes per question).

Code and full evaluation: https://github.com/bhavyananda17/lexsi_labs_project

Live mode loads the gated base model, so the Space needs an `HF_TOKEN` secret from an account that has accepted the Tiny Aya license. Not legal or financial advice.
