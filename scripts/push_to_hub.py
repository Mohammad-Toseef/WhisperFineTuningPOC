"""
Push the fine-tuned merged model from the Modal volume to HuggingFace Hub (PRIVATE).

Uploads the raw files in /model/whisper-urdu-final (safetensors shards + config +
processor) directly with huggingface_hub.upload_folder — no torch load / re-serialize.

Auth: reads the HF write token from the Modal secret "huggingface-secret" (injected
as an env var). Tries HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HUGGINGFACE_TOKEN.

Run:
    modal run scripts/push_to_hub.py
"""
import modal

app = modal.App("whisper-push-hub")

volume = modal.Volume.from_name("whisper-training-vol", create_if_missing=False)
VOLUME_PATH = "/data"
FINAL_MODEL_PATH = f"{VOLUME_PATH}/model/whisper-urdu-final"

REPO_ID = "mohammad-toseef059/whisper-large-v3-urdu"
PRIVATE = True

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["huggingface_hub>=0.23,<1.0"])
)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60,
)
def push():
    import os
    from pathlib import Path
    from huggingface_hub import HfApi

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not token:
        # Don't print values — just the KEY NAMES present, to debug the secret.
        keys = [k for k in os.environ if "TOKEN" in k.upper() or "HF" in k.upper()
                or "HUGG" in k.upper()]
        raise RuntimeError(
            "No HF token found in the Modal secret. Expected env var HF_TOKEN "
            f"(or HUGGING_FACE_HUB_TOKEN). Token-ish keys present: {keys or 'none'}. "
            "Set the secret's key name to HF_TOKEN."
        )

    src = Path(FINAL_MODEL_PATH)
    if not src.exists():
        raise FileNotFoundError(f"Model not found on volume: {src}")
    files = sorted(p.name for p in src.iterdir() if p.is_file())
    print(f"Uploading {len(files)} files from {src} -> {REPO_ID} (private={PRIVATE})")
    print("  files:", files)

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)
    api.upload_folder(
        folder_path=str(src),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add fine-tuned Whisper large-v3 Urdu (LoRA merged, full 49-ep run)",
    )
    print(f"✅ Pushed to https://huggingface.co/{REPO_ID} (private)")


@app.local_entrypoint()
def main():
    push.remote()