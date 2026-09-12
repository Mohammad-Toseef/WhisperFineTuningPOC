"""
Transcribe audio with ONE named model on Modal. Nothing else.

scripts/compare_transcribe.py always runs the base model AND a fine-tuned one,
because its whole purpose is the side-by-side pair. When you already know which
model you want -- a spot check, a demo clip, a slice of a live stream -- that
second GPU pass is pure waste. This script is that job, and only that job.

    modal run scripts/transcribe.py \
        --audio ALRA_TV_LIVE_STREAMS/2026-09-03_05-30-00_05-49-00.mp3 \
        --model-path /data/model/whisper-urdu-r3-final

--model-path is REQUIRED and has no default. compare_transcribe.py defaulted to
round 1's path, which silently produced a real transcript from the wrong model
once rounds 2 and 3 existed; there is no safe default here, so there isn't one.
A /data/... path is a directory on the whisper-training-vol volume; anything
else is passed to transformers as-is, so a HuggingFace id works too:

    --model-path openai/whisper-large-v3
    --model-path mohammad-toseef059/whisper-large-v3-urdu-r3

Output goes next to the audio unless --out-dir says otherwise, named
<stem>_<model tag>.txt -- the model is in the FILENAME on purpose. These files
get shared and compared, and a bare *_transcript.txt is indistinguishable
between rounds once it is sitting in a folder next to three others.
"""
import modal
from pathlib import Path

app = modal.App("whisper-transcribe")

# Same volume the trained models live on (must already exist).
volume = modal.Volume.from_name("whisper-training-vol", create_if_missing=False)
VOLUME_PATH = "/data"

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


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={VOLUME_PATH: volume},
)
def transcribe_audio(audio_bytes: bytes, suffix: str, model_path: str,
                     language: str = "ur") -> str:
    """Long-form chunked transcription (28s window, 4s/2s stride) -> plain text.

    Window and stride match modal_align.py and compare_transcribe.py so output
    from this script is comparable with everything else we have produced.
    """
    import os
    import tempfile
    import torch
    from transformers import pipeline

    # Say it out loud, in the container, every run. A transcript carries no trace
    # of which weights made it, and we have already shipped one demo that claimed
    # a round it did not use. This line in the logs is the proof.
    print(f"🔎 transcribing with model: {model_path}")
    if model_path.startswith(VOLUME_PATH) and not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"No model directory at {model_path!r} on the volume.\n"
            "   If that path grew a 'C:/Program Files/Git' prefix, Git Bash rewrote it —\n"
            "   re-run from PowerShell or with MSYS_NO_PATHCONV=1.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(audio_bytes)
        tmp_path = handle.name

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        chunk_length_s=28,
        stride_length_s=(4, 2),
        device=0,
        torch_dtype=torch.float16,
        generate_kwargs={"language": language, "task": "transcribe"},
    )
    # return_timestamps=True is what enables long-form chunking in the HF
    # pipeline; the timestamps themselves are discarded here.
    result = pipe(tmp_path, return_timestamps=True)
    os.unlink(tmp_path)
    return "\n".join(chunk["text"].strip() for chunk in result["chunks"])


@app.local_entrypoint()
def main(audio: str, model_path: str, out_dir: str = "", language: str = "ur"):
    """audio       one path, or several comma-separated.
    model_path  /data/... on the volume, or a HuggingFace model id. Required.
    out_dir     defaults to the folder each audio file already sits in.
    language    forced decoder language token (default ur).
    """
    files = [f.strip() for f in audio.split(",") if f.strip()]
    if not files:
        raise SystemExit("--audio is empty")

    # Resolve everything BEFORE spending a GPU second on the first file: a typo in
    # the third of four paths should not surface twenty minutes into the run.
    resolved = []
    for name in files:
        path = Path(name)
        if not path.exists():
            path = Path("full_audio_samples") / name
        if not path.exists():
            raise SystemExit(f"audio not found: {name}  (looked in . and full_audio_samples/)")
        resolved.append(path)

    model_tag = Path(model_path.rstrip("/")).name
    print(f"model: {model_path}   ({len(resolved)} file(s))")

    for path in resolved:
        target = Path(out_dir) if out_dir else path.parent
        target.mkdir(parents=True, exist_ok=True)
        out_path = target / f"{path.stem}_{model_tag}.txt"

        size_mb = path.stat().st_size / 1e6
        print(f"\n🔎 {path.name}  ({size_mb:.1f} MB) ...")
        text = transcribe_audio.remote(
            path.read_bytes(), path.suffix, model_path, language)
        out_path.write_text(text, encoding="utf-8")
        words = len(text.split())
        print(f"   -> {out_path}   ({words:,} words, {len(text.splitlines())} segments)")

    print(f"\n✅ Done — {len(resolved)} transcript(s) from {model_path}")
