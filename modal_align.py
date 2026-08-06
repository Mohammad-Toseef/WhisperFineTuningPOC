"""
Forced alignment on Modal GPUs. Two entry points, for two different situations:

  run_alignment       -- audio + a HUMAN-VERIFIED transcript. Runs a throwaway Whisper
                         pass for segment boundaries, fuzzy-matches the verified text
                         into those windows, then force-aligns. The round-1 path.

  transcribe_and_align -- audio only. Transcribes with OUR FINE-TUNED model, then
                         force-aligns its output. The Batch 3 path, where no human
                         transcript exists yet (review happens later, on the manifest).

Stages 3 and 4 of the Batch 3 pipeline are fused into transcribe_and_align on purpose:
the audio is already in the container and both stages need it, so splitting them would
re-upload every file and pay a second cold start. The fine-tuned Whisper runs ONCE;
the aligner is a separate, much smaller wav2vec2 CTC model that locates known words
rather than transcribing.

Usage:
    # round-1 style, verified transcript
    modal run modal_align.py --audio-path EP3.mp3 --text-path EP3.txt --out-path EP3.srt

    # Batch 3: transcribe + align trimmed audio into SRTs
    modal run --detach modal_align.py::transcribe_align --batch batch3
    modal run modal_align.py::transcribe_align --only B3001,B3002
    # a later batch, with a later model
    modal run --detach modal_align.py::transcribe_align --batch batch4 \
        --model-path /data/model/whisper-urdu-round2
"""

import sys
from pathlib import Path

import modal

# align_to_srt lives in src/srt_pipeline/, so make it importable both here (for
# add_local_python_source, which resolves the module locally) and in the container.
sys.path.insert(0, str(Path(__file__).parent / "src" / "srt_pipeline"))

app = modal.App("srt-forced-alignment")

# Reuse the trained-model volume (must already exist -- created by modal_app.py).
volume = modal.Volume.from_name("whisper-training-vol", create_if_missing=False)
VOLUME_PATH = "/data"
FINAL_MODEL_PATH = f"{VOLUME_PATH}/model/whisper-urdu-final"

# Upper bounds are load-bearing: unbounded installs pulled transformers 5.x / torch 2.12
# / numpy 2.4 during the round-1 training run and broke the generate path. whisperx is
# installed alongside because transcribe_and_align needs both the HF pipeline (stage 3)
# and whisperx.align (stage 4) in one container.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install([
        "whisperx",
        "transformers>=4.40.0,<4.46",
        "torch>=2.2.0,<2.5",
        "torchaudio>=2.2.0,<2.5",
        "accelerate>=0.28.0,<1.0",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "numpy<2.0",
    ])
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


@app.function(
    image=image,
    gpu="A10G",           # large-v3 in fp16 + a wav2vec2 aligner; T4's 16GB is tight
    timeout=60 * 60 * 2,  # a 90-min episode transcribes + aligns well inside this
    volumes={VOLUME_PATH: volume},
)
def transcribe_and_align(
    audio_bytes: bytes,
    suffix: str,
    language: str = "ur",
    align_lang: str = "ur",
    model_path: str = FINAL_MODEL_PATH,
) -> dict:
    """Stage 3+4 fused: fine-tuned Whisper transcribes, wav2vec2 CTC re-times.

    Returns {"srt": str, "segments": int, "words": int, "text": str, "error": str|None}
    so a failure on one episode is reported rather than killing the whole batch.
    """
    import os
    import tempfile

    import torch
    import whisperx
    from transformers import pipeline

    import align_to_srt as core

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(audio_bytes)
        audio_path = handle.name

    try:
        # ── Stage 3: transcribe (the ONLY fine-tuned Whisper pass) ──────────────
        # chunk_length_s + return_timestamps are what enable long-form decoding; a bare
        # processor() call would silently truncate to Whisper's 30s window. Same
        # 28s/(4,2) settings as scripts/compare_transcribe.py, which is known-good here.
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            chunk_length_s=28,
            stride_length_s=(4, 2),
            device=0,
            torch_dtype=torch.float16,
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        result = pipe(audio_path, return_timestamps=True)
        del pipe
        torch.cuda.empty_cache()  # free large-v3 before loading the aligner

        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / whisperx.audio.SAMPLE_RATE

        # HF chunk timestamps are (start, end); the final chunk's end can be None, and a
        # None start would break align(). Fill both from what we know.
        segments, previous_end = [], 0.0
        for chunk in result["chunks"]:
            start, end = chunk.get("timestamp", (None, None))
            start = previous_end if start is None else float(start)
            end = duration if end is None else float(end)
            if end <= start:
                end = min(start + 0.1, duration)
            segments.append({"start": start, "end": end, "text": chunk["text"].strip()})
            previous_end = end

        # The pipeline's chunk timestamps are approximate and sometimes degenerate, which
        # makes alignment fail outright. Repair before aligning, not after.
        segments = core.repair_segment_windows(segments, duration, log=print)

        # ── Stage 4: forced-align (separate, much smaller wav2vec2 model) ───────
        srt_text = core.align_segments_to_srt(
            audio, segments, align_lang, device="cuda", log=print,
        )
        return {
            "srt": srt_text,
            "segments": len(segments),
            # Count blocks, not "\n\n" separators -- N cues yield N-1 of those.
            "cues": len([b for b in srt_text.strip().split("\n\n") if b.strip()]),
            "text": "\n".join(s["text"] for s in segments),
            "error": None,
        }
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return {"srt": "", "segments": 0, "cues": 0, "text": "",
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        os.unlink(audio_path)


@app.local_entrypoint()
def transcribe_align(
    batch: str = "batch3",
    csv_path: str = "",
    audio_dir: str = "",
    out_dir: str = "",
    asr_text_dir: str = "",
    only: str = "",
    limit: int = 0,
    language: str = "ur",
    align_lang: str = "ur",
    model_path: str = FINAL_MODEL_PATH,
):
    """Transcribe + align every trimmed episode of a batch, fanned out across GPUs.

    All paths derive from --batch (data/<batch>/...); pass an explicit path only for a
    one-off layout. --model-path selects which fine-tuned model transcribes, so a later
    batch can use a later model without editing this file.

    Idempotent: an existing .srt means done, so a re-run only does what is left.
    Prefer `modal run --detach` -- a local network drop killed a long round-1 run.
    """
    import csv as csv_module

    from batch_paths import BatchPaths

    paths = BatchPaths(batch)
    csv_path = csv_path or str(paths.validated_csv)
    audio_dir = audio_dir or str(paths.audio_trimmed)
    out_dir = out_dir or str(paths.srt_dir)
    asr_text_dir = asr_text_dir or str(paths.transcript_dir)

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise SystemExit(f"Validated CSV not found: {csv_file}\n"
                         f"Run: python src/srt_pipeline/validate_sheet.py --batch {batch}")
    rows = list(csv_module.DictReader(csv_file.open(encoding="utf-8")))
    if only:
        wanted = {token.strip() for token in only.split(",") if token.strip()}
        unknown = wanted - {row["label"] for row in rows}
        if unknown:
            raise SystemExit(f"--only referenced unknown labels: {sorted(unknown)}")
        rows = [row for row in rows if row["label"] in wanted]
    if limit:
        rows = rows[:limit]

    audio_path_dir, out_path_dir, text_path_dir = Path(audio_dir), Path(out_dir), Path(asr_text_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)
    text_path_dir.mkdir(parents=True, exist_ok=True)

    pending, missing, done = [], [], []
    for row in rows:
        audio_file = audio_path_dir / row["audio_filename"]
        srt_file = out_path_dir / f"{Path(row['audio_filename']).stem}.srt"
        if srt_file.exists():
            done.append(row["label"])
        elif not audio_file.exists():
            missing.append(row["label"])
        else:
            pending.append((row, audio_file, srt_file))

    print(f"Selected {len(rows)}  |  to process {len(pending)}  |  already done {len(done)}  "
          f"|  not trimmed yet {len(missing)}")
    if missing:
        print(f"   not trimmed: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")
    if not pending:
        print("Nothing to do.")
        return

    call_args = [
        (audio_file.read_bytes(), audio_file.suffix, language, align_lang, model_path)
        for _, audio_file, _ in pending
    ]
    print(f"\nDispatching {len(call_args)} episode(s) to Modal using {model_path}...")

    failures = []
    for (row, audio_file, srt_file), result in zip(pending, transcribe_and_align.starmap(call_args)):
        label = row["label"]
        if result["error"]:
            print(f"  {label}: FAILED — {result['error']}")
            failures.append(label)
            continue
        if not result["srt"].strip():
            print(f"  {label}: FAILED — produced an empty SRT")
            failures.append(label)
            continue
        srt_file.write_text(result["srt"], encoding="utf-8")
        # Also keep the model's pre-alignment text. It costs nothing -- a string join over
        # the same single transcription pass -- and it is the ONLY reference against which
        # alignment word-loss is measurable: an early bug silently dropped 5% of the words
        # while every structural check on the SRT still passed.
        # UNREVIEWED machine output: never feed it anywhere that expects verified text.
        (text_path_dir / f"{srt_file.stem}.txt").write_text(result["text"], encoding="utf-8")
        print(f"  {label}: {result['segments']} segments -> {result['cues']} cues -> {srt_file.name}")

    print(f"\nDone. Wrote {len(pending) - len(failures)} SRT(s) to {out_path_dir}/")
    if failures:
        raise SystemExit(f"{len(failures)} episode(s) failed: {', '.join(failures)}")


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

    NOTE: text_dir here really is round 1's HUMAN-VERIFIED raw_transcripts/ -- this
    entrypoint force-aligns text a person checked. Do NOT point it at a batch's
    asr_transcripts/, which holds unreviewed model output (see BatchPaths.transcript_dir).

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
