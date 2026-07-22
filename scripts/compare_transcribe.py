"""
Standalone one-off test: transcribe the 2 sample episodes in full_audio_samples/
with BOTH the base whisper-large-v3 model and our fine-tuned LoRA model on Modal,
producing plain-text transcripts for a qualitative side-by-side comparison. No
WER/reference transcripts involved — this is purely "read both outputs and
eyeball the diff".

Run:
    modal run scripts/compare_transcribe.py
"""
import modal
from pathlib import Path

app = modal.App("whisper-compare-transcribe")

# Reuse the same trained-model volume from modal_app.py (must already exist).
volume = modal.Volume.from_name("whisper-training-vol", create_if_missing=False)
VOLUME_PATH = "/data"
FINAL_MODEL_PATH = f"{VOLUME_PATH}/model/whisper-urdu-final"
BASE_MODEL_NAME = "openai/whisper-large-v3"
LANGUAGE = "ur"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install([
        "transformers>=4.40.0,<4.46",
        "torch>=2.2.0,<2.5",
        "torchaudio>=2.2.0,<2.5",
        "accelerate>=0.28.0,<1.0",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "numpy<2.0",
    ])
)


def _chunks_to_text(chunks) -> str:
    return "\n".join(chunk["text"].strip() for chunk in chunks)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    volumes={VOLUME_PATH: volume},
)
def transcribe_with_model(audio_bytes: bytes, suffix: str, model_path: str) -> str:
    """Long-form chunked transcription (28s window, 4s/2s stride) -> plain text."""
    import os
    import tempfile
    import torch
    from transformers import pipeline

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        chunk_length_s=28,
        stride_length_s=(4, 2),
        device=0,
        torch_dtype=torch.float16,
        generate_kwargs={"language": LANGUAGE, "task": "transcribe"},
    )
    # return_timestamps=True is still required to enable long-form chunking
    # in the HF pipeline; we just discard the timestamps in the output.
    result = pipe(tmp_path, return_timestamps=True)
    os.unlink(tmp_path)
    return _chunks_to_text(result["chunks"])


@app.local_entrypoint()
def main():
    audio_dir = Path("full_audio_samples")
    out_dir = audio_dir / "compare_transcripts"
    out_dir.mkdir(exist_ok=True)

    files = ["EP19_xNjY-mZlyEU.mp3", "EP2_q1Q6B2JrY58.mp3"]

    for fname in files:
        path = audio_dir / fname
        if not path.exists():
            print(f"⚠️  Skipping missing file: {path}")
            continue
        audio_bytes = path.read_bytes()
        stem = path.stem
        suffix = path.suffix

        print(f"🔎 [{fname}] transcribing with BASE ({BASE_MODEL_NAME})...")
        base_text = transcribe_with_model.remote(audio_bytes, suffix, BASE_MODEL_NAME)
        base_out = out_dir / f"{stem}_base.txt"
        base_out.write_text(base_text, encoding="utf-8")
        print(f"   -> {base_out}")

        print(f"🔎 [{fname}] transcribing with FINE-TUNED ({FINAL_MODEL_PATH})...")
        ft_text = transcribe_with_model.remote(audio_bytes, suffix, FINAL_MODEL_PATH)
        ft_out = out_dir / f"{stem}_finetuned.txt"
        ft_out.write_text(ft_text, encoding="utf-8")
        print(f"   -> {ft_out}")

    print(f"\n✅ Done. Compare transcript pairs in {out_dir}/")
