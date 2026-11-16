"""Create the public HF model + dataset repos and a collection, and upload hf/model and hf/dataset.

    python hf/upload.py      # needs HF_TOKEN with write access in .env
"""
import os
from huggingface_hub import HfApi, add_collection_item, create_collection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = "bhavyananda17"
MODEL = f"{USER}/policyguard-tiny-aya-global-lora"
DATA = f"{USER}/policyguard-irdai-qa"


def token():
    with open(os.path.join(ROOT, ".env")) as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"')


def main():
    t = token()
    api = HfApi(token=t)
    api.create_repo(MODEL, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=MODEL, repo_type="model", folder_path=os.path.join(ROOT, "hf/model"),
                      commit_message="PolicyGuard LoRA adapter (PEFT + MLX)")
    api.create_repo(DATA, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(repo_id=DATA, repo_type="dataset", folder_path=os.path.join(ROOT, "hf/dataset"),
                      commit_message="PolicyGuard IRDAI QA: questions, trajectories, corpus, metrics")
    col = create_collection("PolicyGuard (Lexsi hackathon)", namespace=USER, private=False, exists_ok=True, token=t,
                            description="Tiny Aya Global fine-tuned as a search agent for IRDAI health-insurance rules.")
    for rid, kind in ((MODEL, "model"), (DATA, "dataset")):
        add_collection_item(col.slug, item_id=rid, item_type=kind, exists_ok=True, token=t)
    print("model:      https://huggingface.co/" + MODEL)
    print("dataset:    https://huggingface.co/datasets/" + DATA)
    print("collection: https://huggingface.co/collections/" + col.slug)


if __name__ == "__main__":
    main()
