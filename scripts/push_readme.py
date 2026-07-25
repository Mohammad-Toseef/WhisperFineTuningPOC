"""
Push the model card (local MODEL_CARD.md) to the HF repo as README.md.

Run:
    modal run scripts/push_readme.py
"""
import modal

app = modal.App("whisper-push-readme")

REPO_ID = "mohammad-toseef059/whisper-large-v3-urdu"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["huggingface_hub>=0.23,<1.0"])
    .add_local_file("MODEL_CARD.md", "/root/README.md", copy=True)
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 10,
)
def push():
    import os
    from huggingface_hub import HfApi

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not token:
        raise RuntimeError("No HF token in the Modal secret (expected HF_TOKEN).")

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj="/root/README.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add model card",
    )
    print(f"✅ Pushed README.md to https://huggingface.co/{REPO_ID}")


@app.local_entrypoint()
def main():
    push.remote()