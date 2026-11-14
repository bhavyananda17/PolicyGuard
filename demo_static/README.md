---
title: PolicyGuard
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
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

Side-by-side answers from the untrained Tiny Aya (agent loop and always-search) and PolicyGuard (fine-tuned) on all 95 held-out test questions, with LLM-judge grades and every search step. Code: https://github.com/bhavyananda17/lexsi_labs_project
