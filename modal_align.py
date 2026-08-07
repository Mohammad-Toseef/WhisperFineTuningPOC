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

Results are durable independent of the client: transcribe_and_align commits each .srt and
.txt to the volume under srt_results/<batch>/ BEFORE returning, and transcribe_align checks
the volume before dispatching. So a run interrupted by a network drop is recovered for free
on the next run instead of paying for the same GPU minutes twice. Use --detach so the
container survives the disconnect long enough to commit.

Usage:
    # round-1 style, verified transcript
    modal run modal_align.py --audio-path EP3.mp3 --text-path EP3.txt --out-path EP3.srt

    # Batch 3: transcribe + align trimmed audio into SRTs
    modal run --detach modal_align.py::transcribe_align --batch batch3
    modal run modal_align.py::transcribe_align --only B3001,B3002
    # a later batch, with a later model
    modal run --detach modal_align.py::transcribe_align --batch batch4 \
        --model-path /data/model/whisper-urdu-round2

    # pull results off the volume without starting a run (recovery / inspection)
    modal run modal_align.py::fetch_results --batch batch3
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

# Where the GPU stage parks its results, volume-relative (so both the container path and
# the client-side read path derive from one constant).
#
# Why results go to the volume at all: transcribe_and_align RETURNS the srt/text to the
# caller, and the local entrypoint is what writes them to disk. That makes the local
# client a single point of failure for a 20+ minute GPU job -- drop the connection and
# the work completes (or is killed) with nothing to show for it, and the next run
# recomputes from scratch because idempotency keys off the local .srt existing.
# Committing to the volume first makes the output durable server-side, independent of
# the client, so a re-run FETCHES instead of paying for the GPU twice.
RESULTS_SUBDIR = "srt_results"


def results_dir_for(batch: str) -> tuple[str, str]:
    """(container absolute path, volume-relative path) for a batch's results."""
    relative = f"{RESULTS_SUBDIR}/{batch}"
    return f"{VOLUME_PATH}/{relative}", relative

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
    out_stem: str = "",
    results_dir: str = "",
) -> dict:
    """Stage 3+4 fused: fine-tuned Whisper transcribes, wav2vec2 CTC re-times.

    Returns {"srt": str, "segments": int, "words": int, "text": str, "error": str|None}
    so a failure on one episode is reported rather than killing the whole batch.

    When `out_stem` and `results_dir` are given, the srt and text are ALSO committed to the
    volume before returning. The return value stays authoritative for a normal run; the
    volume copy is what survives a client disconnect (see RESULTS_SUBDIR).
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
        transcript_text = "\n".join(s["text"] for s in segments)

        # Persist to the volume BEFORE returning, so the result outlives the client.
        # Only on success: an empty SRT must not create a file, or the fetch-first check
        # would treat a failed episode as done and never retry it.
        if out_stem and results_dir and srt_text.strip():
            from pathlib import Path as _Path

            target = _Path(results_dir)
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{out_stem}.srt").write_text(srt_text, encoding="utf-8")
            (target / f"{out_stem}.txt").write_text(transcript_text, encoding="utf-8")
            volume.commit()  # without this the writes stay container-local and are lost
            print(f"committed {out_stem}.srt + .txt to {results_dir}")

        return {
            "srt": srt_text,
            "segments": len(segments),
            # Count blocks, not "\n\n" separators -- N cues yield N-1 of those.
            "cues": len([b for b in srt_text.strip().split("\n\n") if b.strip()]),
            "text": transcript_text,
            "error": None,
        }
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return {"srt": "", "segments": 0, "cues": 0, "text": "",
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        os.unlink(audio_path)


def volume_result_stems(relative_dir: str) -> set[str]:
    """Stems having a committed .srt on the volume. Empty set if nothing is there yet.

    No volume.reload() here: that is container-only (it raises "can only be called from
    within a running function" on the client), and it is not needed -- a client-side
    listdir queries the volume's current state directly. Only NotFoundError is caught,
    and only because an absent directory is the normal first-run state for a batch.
    Catching broadly here hid exactly this bug: reload()'s RuntimeError was swallowed and
    the function returned an empty set unconditionally, silently disabling fetch-first.
    """
    try:
        return {
            Path(entry.path).stem
            for entry in volume.listdir(relative_dir)
            if entry.path.endswith(".srt")
        }
    except modal.exception.NotFoundError:
        return set()


def read_volume_text(relative_path: str) -> str:
    return b"".join(volume.read_file(relative_path)).decode("utf-8")


def save_result(srt_text: str, transcript_text: str, srt_file: Path, text_dir: Path) -> None:
    srt_file.write_text(srt_text, encoding="utf-8")
    # Keep the model's pre-alignment text. It costs nothing -- a string join over the same
    # single transcription pass -- and it is the ONLY reference against which alignment
    # word-loss is measurable: an early bug silently dropped 5% of the words while every
    # structural check on the SRT still passed.
    # UNREVIEWED machine output: never feed it anywhere that expects verified text.
    (text_dir / f"{srt_file.stem}.txt").write_text(transcript_text, encoding="utf-8")


@app.local_entrypoint()
def fetch_results(batch: str = "batch3", out_dir: str = "", asr_text_dir: str = ""):
    """Pull GPU results off the volume into the local batch directories.

    The recovery path after a disconnected run: the container already committed its work,
    so this costs no GPU. transcribe_align does this automatically before dispatching, so
    reach for this only to inspect or restore results without starting a run.
    """
    from batch_paths import BatchPaths

    paths = BatchPaths(batch)
    out_path_dir = Path(out_dir) if out_dir else paths.srt_dir
    text_path_dir = Path(asr_text_dir) if asr_text_dir else paths.transcript_dir
    out_path_dir.mkdir(parents=True, exist_ok=True)
    text_path_dir.mkdir(parents=True, exist_ok=True)

    _, relative = results_dir_for(batch)
    stems = volume_result_stems(relative)
    if not stems:
        print(f"No results on the volume at {relative}/")
        return

    fetched, present = 0, 0
    for stem in sorted(stems):
        srt_file = out_path_dir / f"{stem}.srt"
        if srt_file.exists():
            present += 1
            continue
        save_result(read_volume_text(f"{relative}/{stem}.srt"),
                    read_volume_text(f"{relative}/{stem}.txt"),
                    srt_file, text_path_dir)
        print(f"  fetched {stem}")
        fetched += 1
    print(f"\nFetched {fetched} | already local {present} | on volume {len(stems)}")


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

    # Fetch-first: a previous run may have finished on the GPU and committed its results
    # while the client was gone (network drop, Ctrl-C, the cp1252 crash). Recovering those
    # is free; recomputing them is not.
    container_results, relative_results = results_dir_for(batch)
    if pending:
        on_volume = volume_result_stems(relative_results)
        recovered = []
        for entry in list(pending):
            row, _, srt_file = entry
            if srt_file.stem not in on_volume:
                continue
            save_result(read_volume_text(f"{relative_results}/{srt_file.stem}.srt"),
                        read_volume_text(f"{relative_results}/{srt_file.stem}.txt"),
                        srt_file, text_path_dir)
            recovered.append(row["label"])
            pending.remove(entry)
        if recovered:
            print(f"\nRecovered {len(recovered)} result(s) from the volume — no GPU needed: "
                  f"{', '.join(recovered)}")

    if not pending:
        print("Nothing to do.")
        return

    call_args = [
        (audio_file.read_bytes(), audio_file.suffix, language, align_lang, model_path,
         srt_file.stem, container_results)
        for _, audio_file, srt_file in pending
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
        save_result(result["srt"], result["text"], srt_file, text_path_dir)
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
