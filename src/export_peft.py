"""Convert the MLX LoRA adapter to Hugging Face PEFT format (usable with transformers + peft).

MLX LoRALinear computes  y = W x + scale * (x @ lora_a) @ lora_b  with lora_a: (in, r), lora_b: (r, out).
PEFT computes            y = W x + (alpha / r) * B(A(x))          with A.weight: (r, in), B.weight: (out, r).
So A = lora_a.T, B = lora_b.T and alpha = scale * r.

    python src/export_peft.py adapters/policyguard_v4 hf/model
"""
import json
import os
import re
import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def main(src, dst):
    with open(os.path.join(src, "adapter_config.json")) as f:
        cfg = json.load(f)
    lp = cfg["lora_parameters"]
    r, scale = lp["rank"], lp["scale"]
    out, layers, modules = {}, set(), set()
    with safe_open(os.path.join(src, "adapters.safetensors"), "np") as f:
        for k in f.keys():
            m = re.match(r"model\.layers\.(\d+)\.(self_attn\.\w+)\.lora_([ab])$", k)
            if not m:
                raise ValueError(f"unexpected key {k}")
            layer, mod, ab = m.groups()
            layers.add(int(layer))
            modules.add(mod.split(".")[-1])
            t = np.ascontiguousarray(f.get_tensor(k).T).astype(np.float32)
            out[f"base_model.model.model.layers.{layer}.{mod}.lora_{ab.upper()}.weight"] = t
    os.makedirs(dst, exist_ok=True)
    save_file(out, os.path.join(dst, "adapter_model.safetensors"))
    peft_cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": cfg["model"],
        "r": r, "lora_alpha": scale * r, "lora_dropout": lp["dropout"], "bias": "none",
        "target_modules": sorted(modules), "layers_to_transform": sorted(layers),
        "fan_in_fan_out": False, "inference_mode": True, "use_rslora": False, "use_dora": False,
    }
    with open(os.path.join(dst, "adapter_config.json"), "w") as f:
        json.dump(peft_cfg, f, indent=2)
    print(f"{len(out)} tensors, layers {min(layers)}-{max(layers)}, modules {sorted(modules)} -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
