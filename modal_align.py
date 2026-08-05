"""
Run align_to_srt's forced-alignment pipeline on Modal (GPU), to avoid the flaky local
network/CPU constraints. Usage:

    modal run modal_align.py --audio-path EP3.mp3 --text-path EP3.txt --out-path EP3.srt
"""

import sys
from pathlib import Path

import modal

# align_to_srt lives in src/srt_pipeline/, so make it importable both here (for
# add_local_python_source, which resolves the module locally) and in the container.
sys.path.insert(0, str(Path(__file__).parent / "src" / "srt_pipeline"))

app = modal.App("srt-forced-alignment")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("whisperx")
    .add_local_python_source("align_to_srt")
)


@app.function(image=image, gpu="T4", timeout=3600)
def run_alignment(audio_bytes: bytes, gt_text: str, asr_model: str, asr_language: str, align_lang: str, skip_start: float = 0.0) -> str:
    import tempfile

    import whisperx
    from whisperx.audio import SAMPLE_RATE

    import align_to_srt as core

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    audio = whisperx.load_audio(audio_path)
    if skip_start > 0:
        audio = audio[int(skip_start * SAMPLE_RATE):]
    gt_words = gt_text.split()
    return core.run_pipeline(audio, gt_words, asr_model, asr_language, align_lang, device="cuda", time_offset=skip_start, log=print)


@app.local_entrypoint()
def main(audio_path: str, text_path: str, out_path: str, asr_model: str = "large-v3", asr_language: str = "ur", align_lang: str = "ur", skip_start: float = 0.0):
    audio_bytes = Path(audio_path).read_bytes()
    gt_text = Path(text_path).read_text(encoding="utf-8")
    srt_text = run_alignment.remote(audio_bytes, gt_text, asr_model, asr_language, align_lang, skip_start)
    Path(out_path).write_text(srt_text, encoding="utf-8")
    print(f"Wrote SRT to {out_path}")


@app.local_entrypoint()
def batch(
    start: int = 1,
    end: int = 10,
    audio_dir: str = "music_cleaned_audio",
    text_dir: str = "raw_transcripts",
    out_dir: str = "timestamped_srts",
    asr_model: str = "large-v3",
    asr_language: str = "ur",
    align_lang: str = "ur",
):
    """Forced-align EP<start>..EP<end> from audio_dir/text_dir, fanned out across
    Modal GPU containers in parallel, writing one SRT per episode into out_dir.
    Audio in audio_dir is assumed pre-trimmed (intro/outro music already cut), so
    skip_start is always 0.

    Usage: modal run modal_align.py::batch --start 1 --end 10
    """
    audio_path_dir = Path(audio_dir)
    text_path_dir = Path(text_dir)
    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    episodes, call_args = [], []
    for n in range(start, end + 1):
        matches = sorted(audio_path_dir.glob(f"EP{n}_*.mp3"))
        if not matches:
            print(f"EP{n}: no audio file found in {audio_dir}, skipping")
            continue
        audio_path = matches[0]
        text_path = text_path_dir / (audio_path.stem + ".txt")
        if not text_path.exists():
            print(f"EP{n}: no transcript found ({text_path}), skipping")
            continue

        episodes.append(audio_path.stem)
        call_args.append((
            audio_path.read_bytes(),
            text_path.read_text(encoding="utf-8"),
            asr_model,
            asr_language,
            align_lang,
            0.0,
        ))

    print(f"Aligning {len(episodes)} episodes in parallel: {episodes}")
    for stem, srt_text in zip(episodes, run_alignment.starmap(call_args)):
        out_path = out_path_dir / f"{stem}.srt"
        out_path.write_text(srt_text, encoding="utf-8")
        print(f"{stem}: wrote {out_path}")
