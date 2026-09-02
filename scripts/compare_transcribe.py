"""
Transcribe audio with BOTH the base whisper-large-v3 model and our fine-tuned
LoRA model on Modal, writing the *_base.txt / *_finetuned.txt pair that
scripts/compare_transcripts.py reads for a qualitative side-by-side
comparison. No WER/reference transcripts involved here — this is purely
"read both outputs and eyeball the diff"; compare_transcripts.py is what
scores against a reference, when one exists.

Third step of the pipeline, after scripts/compare_transcripts.py --fetch has
put the audio in full_audio_samples/ (or run directly via
scripts/compare_transcripts.py --transcribe, which just shells out to this):

    modal run scripts/compare_transcribe.py --audio EP42_abc123XYZ.mp3

A bare filename resolves against full_audio_samples/; a path that already
exists as given is used as-is. Comma-separate for more than one file:

    modal run scripts/compare_transcribe.py --audio "EP42_abc.mp3,EP43_def.mp3"

Group the output under a named subfolder of compare_transcripts/ (useful for
a batch, e.g. a new training round) instead of the default flat layout:

    modal run scripts/compare_transcribe.py --audio EP42_abc123XYZ.mp3 --out-subdir "Round 3"
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
def main(audio: str, out_subdir: str = "", model_path: str = ""):
    """model_path  which fine-tuned model to put opposite the base one.

    Defaults to round 1's /data/model/whisper-urdu-final, which is where this
    script pointed when only one round existed. Later rounds live at run_tag'd
    paths this default would never reach, so a demo of "our fine-tuned model"
    would silently show round 1 — a real model, a plausible transcript, and the
    wrong answer. Pass it explicitly:

        modal run scripts/compare_transcribe.py --audio clip.mp3 \
            --model-path /data/model/whisper-urdu-r3-final --out-subdir "Round 3"
    """
    ft_path = model_path or FINAL_MODEL_PATH
    audio_dir = Path("full_audio_samples")
    out_dir = audio_dir / "compare_transcripts" / out_subdir if out_subdir \
        else audio_dir / "compare_transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [f.strip() for f in audio.split(",") if f.strip()]

    for fname in files:
        path = Path(fname)
        if not path.exists():
            path = audio_dir / fname
        if not path.exists():
            print(f"⚠️  Skipping missing file: {fname}  (looked in . and {audio_dir}/)")
            continue
        audio_bytes = path.read_bytes()
        stem = path.stem
        suffix = path.suffix

        print(f"🔎 [{path.name}] transcribing with BASE ({BASE_MODEL_NAME})...")
        base_text = transcribe_with_model.remote(audio_bytes, suffix, BASE_MODEL_NAME)
        base_out = out_dir / f"{stem}_base.txt"
        base_out.write_text(base_text, encoding="utf-8")
        print(f"   -> {base_out}")

        print(f"🔎 [{path.name}] transcribing with FINE-TUNED ({ft_path})...")
        ft_text = transcribe_with_model.remote(audio_bytes, suffix, ft_path)
        # The model name goes in the FILENAME, not only in _meta.json. A bare
        # *_finetuned.txt is indistinguishable between rounds once it is sitting
        # in a folder next to three others, and these files get shared.
        model_tag = Path(ft_path.rstrip("/")).name
        ft_out = out_dir / f"{stem}_finetuned_{model_tag}.txt"
        ft_out.write_text(ft_text, encoding="utf-8")
        print(f"   -> {ft_out}")

    # Self-describing output. A *_finetuned.txt is indistinguishable between
    # rounds by inspection, and this pair is going in front of leadership — the
    # claim "this is round 3" needs to be checkable afterwards, not remembered.
    import json
    from datetime import datetime, timezone
    meta_path = out_dir / "_meta.json"
    # MERGE, don't overwrite. Videos are transcribed one run at a time into a
    # shared folder, so a plain write leaves the meta describing only whichever
    # ran last — while the folder, and the report built from it, hold both.
    prior = {}
    if meta_path.exists():
        try:
            prior = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior = {}
    runs = prior.get("runs", [])
    runs.append({
        "base_model": BASE_MODEL_NAME,
        "finetuned_model": ft_path,
        "audio": files,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    meta_path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")

    print(f"\n✅ Done. Compare transcript pairs in {out_dir}/")
    print(f"   base      : {BASE_MODEL_NAME}")
    print(f"   fine-tuned: {ft_path}")
