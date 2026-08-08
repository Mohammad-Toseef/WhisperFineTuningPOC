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

import json
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


def results_dir_for(batch: str, windows: str = "vad") -> tuple[str, str]:
    """(container absolute path, volume-relative path) for a batch's results.

    The window mode is part of the path on purpose. Cached results are keyed by filename
    stem, so without this a "vad" run would find the "chunks" result already sitting on the
    volume and RECOVER it instead of running -- silently returning the very output the run
    was meant to replace, and making the two paths look identical.
    """
    relative = f"{RESULTS_SUBDIR}/{batch}/{windows}"
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


# How the segment windows handed to forced alignment get drawn.
#
#   "chunks" -- HF's long-form pipeline on a fixed 28s grid with a 4s/2s stride. Its
#              timestamps are quantized to whole seconds and occasionally degenerate; on
#              B3003 that stranded 52 words on a 0.62s cue (83.9 w/s vs a 2.68 median) and
#              left 351s of speech uncovered. The stride also duplicates boundary text.
#   "vad"    -- voice-activity detection, so windows START AND END WHERE SPEECH DOES.
#              Round 1 built its windows this way. Measured on B3003: 95% of the audio the
#              chunk path discards falls inside a window VAD would have built, and VAD
#              still correctly rejects genuine non-speech (the 3.9s gap at 40:17).
#
# 30s matches Whisper's own input window, so a window needs no further chunking -- which is
# what removes the degenerate-timestamp failure mode rather than repairing it.
VAD_CHUNK_SIZE = 30
VAD_ONSET, VAD_OFFSET = 0.5, 0.363


def _segments_from_fixed_chunks(audio_path: str, duration: float, model_path: str,
                                language: str) -> list[dict]:
    """Original path: HF long-form decoding on a fixed grid, timestamps from the pipeline."""
    import torch
    from transformers import pipeline

    # chunk_length_s + return_timestamps are what enable long-form decoding; a bare
    # processor() call would silently truncate to Whisper's 30s window. Same 28s/(4,2)
    # settings as scripts/compare_transcribe.py, which is known-good here.
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
    torch.cuda.empty_cache()

    # HF chunk timestamps are (start, end); the final chunk's end can be None, and a None
    # start would break align(). Fill both from what we know.
    segments, previous_end = [], 0.0
    for chunk in result["chunks"]:
        start, end = chunk.get("timestamp", (None, None))
        start = previous_end if start is None else float(start)
        end = duration if end is None else float(end)
        if end <= start:
            end = min(start + 0.1, duration)
        segments.append({"start": start, "end": end, "text": chunk["text"].strip()})
        previous_end = end
    return segments


def _speech_turns(scores) -> list[dict]:
    """Binarize the VAD score curve into raw speech turns [{start, end}, ...].

    Pure CPU post-process on a tensor _segments_from_vad has already computed, so this adds
    no model load and no GPU pass -- the turns were simply being discarded.

    NOT the same thing as the merged windows: merge_chunks pads to VAD_CHUNK_SIZE and
    bridges short pauses, so on all three Batch 3 episodes 100% of the SRT's uncovered time
    fell inside some merged window. The turns are what actually distinguishes speech from
    silence, which is the whole point for the QA gate.

    max_duration is deliberately huge: a turn split at 30s would report one continuous
    stretch of speech as several, which changes nothing for coverage but makes the counts
    misleading to read.
    """
    import traceback

    from whisperx.vad import Binarize

    # Broad catch, deliberately: the turns are a diagnostic by-product of a two-hour GPU
    # job. Losing them must never cost the transcription. This is NOT the swallowed-error
    # pattern that hid the volume.reload() bug -- the traceback is printed here, and their
    # absence is reported loudly by the QA gate rather than silently degrading a metric.
    try:
        binarized = Binarize(max_duration=100000, onset=VAD_ONSET, offset=VAD_OFFSET)(scores)
        return [{"start": round(float(s.start), 3), "end": round(float(s.end), 3)}
                for s in binarized.get_timeline()]
    except Exception:
        print("WARNING: could not binarize VAD scores into speech turns; the QA gate will "
              "fall back to charging all uncovered time. Transcription continues.")
        traceback.print_exc()
        return []


def _segments_from_vad(audio, duration: float, model_path: str, language: str,
                       batch_size: int) -> tuple[list[dict], list[dict]]:
    """VAD path: cut windows at detected silence, then transcribe each window.

    The window boundaries come from the AUDIO, never from the decoder, so a segment's
    (start, end) is correct by construction and needs no repair. Each window is at most
    Whisper's own 30s input, so it is transcribed whole -- no internal chunking, hence no
    chunk timestamps to be wrong, and no stride overlap to duplicate boundary text.

    Returns (segments, speech_turns). The turns are a free by-product -- see _speech_turns.
    """
    import torch
    import whisperx
    from transformers import pipeline
    from whisperx.vad import load_vad_model, merge_chunks

    sample_rate = whisperx.audio.SAMPLE_RATE
    vad_model = load_vad_model(device="cuda", vad_onset=VAD_ONSET, vad_offset=VAD_OFFSET)
    scores = vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0),
                        "sample_rate": sample_rate})
    windows = merge_chunks(scores, VAD_CHUNK_SIZE, onset=VAD_ONSET, offset=VAD_OFFSET)
    turns = _speech_turns(scores)
    del vad_model
    torch.cuda.empty_cache()

    spans = [(max(0.0, float(w["start"])), min(duration, float(w["end"]))) for w in windows]
    spans = [(s, e) for s, e in spans if e - s > 0.1]
    covered = sum(e - s for s, e in spans)
    speech = sum(t["end"] - t["start"] for t in turns)
    print(f"VAD: {len(spans)} windows covering {covered:.0f}s of {duration:.0f}s "
          f"({covered / duration * 100:.1f}%) | {len(turns)} speech turns "
          f"totalling {speech:.0f}s ({speech / duration * 100:.1f}%)")

    # No chunk_length_s here on purpose: every window already fits Whisper's input window,
    # so the pipeline sees each as one utterance and returns text only -- there are no
    # per-chunk timestamps to go wrong, because we are not asking for any.
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=0,
        torch_dtype=torch.float16,
        generate_kwargs={"language": language, "task": "transcribe"},
    )
    inputs = [{"raw": audio[int(s * sample_rate):int(e * sample_rate)],
               "sampling_rate": sample_rate} for s, e in spans]
    outputs = pipe(inputs, batch_size=batch_size)
    del pipe
    torch.cuda.empty_cache()

    segments = []
    for (start, end), output in zip(spans, outputs):
        text = str(output["text"]).strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})
    return segments, turns


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
    windows: str = "vad",
    batch_size: int = 8,
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

    import align_to_srt as core

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(audio_bytes)
        audio_path = handle.name

    try:
        if windows not in ("chunks", "vad"):
            raise ValueError(f"windows must be 'chunks' or 'vad', got {windows!r}")

        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / whisperx.audio.SAMPLE_RATE

        # ── Stage 3: transcribe (the ONLY fine-tuned Whisper pass, either path) ──
        if windows == "vad":
            segments, speech_turns = _segments_from_vad(
                audio, duration, model_path, language, batch_size)
        else:
            segments = _segments_from_fixed_chunks(audio_path, duration, model_path, language)
            speech_turns = []  # the chunks path never runs VAD; nothing to report
        torch.cuda.empty_cache()  # free the transcriber before loading the aligner

        # Repair runs on BOTH paths, but means different things. On "chunks" it is load
        # bearing -- degenerate timestamps make alignment fail outright. On "vad" the
        # windows come from the audio, so it should be a no-op; if it logs anything there,
        # that is a finding worth reading, not routine maintenance.
        segments = core.repair_segment_windows(segments, duration, log=print)

        # ── Stage 4: forced-align (separate, much smaller wav2vec2 model) ───────
        srt_text = core.align_segments_to_srt(
            audio, segments, align_lang, device="cuda", log=print,
        )
        transcript_text = "\n".join(s["text"] for s in segments)

        # Persist to the volume BEFORE returning, so the result outlives the client.
        # Only on success: an empty SRT must not create a file, or the fetch-first check
        # would treat a failed episode as done and never retry it.
        vad_payload = json.dumps({"duration": duration, "turns": speech_turns}) if speech_turns else ""

        if out_stem and results_dir and srt_text.strip():
            from pathlib import Path as _Path

            target = _Path(results_dir)
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{out_stem}.srt").write_text(srt_text, encoding="utf-8")
            (target / f"{out_stem}.txt").write_text(transcript_text, encoding="utf-8")
            if vad_payload:
                (target / f"{out_stem}.vad.json").write_text(vad_payload, encoding="utf-8")
            volume.commit()  # without this the writes stay container-local and are lost
            print(f"committed {out_stem}.srt + .txt"
                  f"{' + .vad.json' if vad_payload else ''} to {results_dir}")

        return {
            "srt": srt_text,
            "segments": len(segments),
            # Count blocks, not "\n\n" separators -- N cues yield N-1 of those.
            "cues": len([b for b in srt_text.strip().split("\n\n") if b.strip()]),
            "text": transcript_text,
            "vad": vad_payload,
            "error": None,
        }
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return {"srt": "", "segments": 0, "cues": 0, "text": "", "vad": "",
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


def read_volume_text_optional(relative_path: str) -> str:
    """read_volume_text, but "" when the file is not there.

    Only for genuinely optional sidecars: .vad.json is absent for every result committed
    before it existed, and for every --windows chunks run. Missing must degrade the QA gate,
    not fail the fetch.
    """
    try:
        return read_volume_text(relative_path)
    except (modal.exception.NotFoundError, FileNotFoundError):
        return ""


def save_result(srt_text: str, transcript_text: str, srt_file: Path, text_dir: Path,
                vad_payload: str = "", vad_dir: Path | None = None) -> None:
    srt_file.write_text(srt_text, encoding="utf-8")
    # Keep the model's pre-alignment text. It costs nothing -- a string join over the same
    # single transcription pass -- and it is the ONLY reference against which alignment
    # word-loss is measurable: an early bug silently dropped 5% of the words while every
    # structural check on the SRT still passed.
    # UNREVIEWED machine output: never feed it anywhere that expects verified text.
    (text_dir / f"{srt_file.stem}.txt").write_text(transcript_text, encoding="utf-8")
    # Detected speech turns, when the run produced them (vad windows only). The QA gate
    # uses these to separate dropped speech from a pause the speaker took; without them it
    # falls back to charging all uncovered time, which is ~2x too harsh.
    if vad_payload and vad_dir is not None:
        vad_dir.mkdir(parents=True, exist_ok=True)
        (vad_dir / f"{srt_file.stem}.vad.json").write_text(vad_payload, encoding="utf-8")


@app.local_entrypoint()
def fetch_results(batch: str = "batch3", out_dir: str = "", asr_text_dir: str = "",
                  vad_dir: str = "", windows: str = "vad"):
    """Pull GPU results off the volume into the local batch directories.

    The recovery path after a disconnected run: the container already committed its work,
    so this costs no GPU. transcribe_align does this automatically before dispatching, so
    reach for this only to inspect or restore results without starting a run.

    --windows must match the run that produced them; results are stored per window mode.
    """
    from batch_paths import BatchPaths

    paths = BatchPaths(batch)
    out_path_dir = Path(out_dir) if out_dir else paths.srt_dir
    text_path_dir = Path(asr_text_dir) if asr_text_dir else paths.transcript_dir
    vad_path_dir = Path(vad_dir) if vad_dir else paths.vad_dir
    out_path_dir.mkdir(parents=True, exist_ok=True)
    text_path_dir.mkdir(parents=True, exist_ok=True)

    _, relative = results_dir_for(batch, windows)
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
                    srt_file, text_path_dir,
                    read_volume_text_optional(f"{relative}/{stem}.vad.json"), vad_path_dir)
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
    vad_dir: str = "",
    only: str = "",
    limit: int = 0,
    language: str = "ur",
    align_lang: str = "ur",
    model_path: str = FINAL_MODEL_PATH,
    windows: str = "vad",
    batch_size: int = 8,
    no_fetch: bool = False,
):
    """Transcribe + align every trimmed episode of a batch, fanned out across GPUs.

    --no-fetch forces a recompute instead of reusing a committed result. The volume cache
    is keyed by (batch, window mode) and knows NOTHING about the alignment code, so after
    changing align_to_srt.py a plain re-run would hand back the stale output and the fix
    would look like it did nothing. Use it whenever the reason for re-running is a code
    change rather than a crash.

    --windows vad cuts segment windows at detected silence instead of on a fixed 28s grid.
    See the VAD_CHUNK_SIZE comment for why that matters. Default stays "chunks" until the
    two paths have been compared on the same episode.

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
    vad_path_dir = Path(vad_dir) if vad_dir else paths.vad_dir
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
    container_results, relative_results = results_dir_for(batch, windows)
    if no_fetch:
        print("--no-fetch: ignoring committed results, recomputing from audio")
    if pending and not no_fetch:
        on_volume = volume_result_stems(relative_results)
        recovered = []
        for entry in list(pending):
            row, _, srt_file = entry
            if srt_file.stem not in on_volume:
                continue
            save_result(read_volume_text(f"{relative_results}/{srt_file.stem}.srt"),
                        read_volume_text(f"{relative_results}/{srt_file.stem}.txt"),
                        srt_file, text_path_dir,
                        read_volume_text_optional(f"{relative_results}/{srt_file.stem}.vad.json"),
                        vad_path_dir)
            recovered.append(row["label"])
            pending.remove(entry)
        if recovered:
            print(f"\nRecovered {len(recovered)} result(s) from the volume — no GPU needed: "
                  f"{', '.join(recovered)}")

    if not pending:
        print("Nothing to do.")
        return

    if windows not in ("chunks", "vad"):
        raise SystemExit(f"--windows must be 'chunks' or 'vad', got {windows!r}")

    call_args = [
        (audio_file.read_bytes(), audio_file.suffix, language, align_lang, model_path,
         srt_file.stem, container_results, windows, batch_size)
        for _, audio_file, srt_file in pending
    ]
    print(f"\nDispatching {len(call_args)} episode(s) to Modal using {model_path} "
          f"[windows={windows}]...")

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
        save_result(result["srt"], result["text"], srt_file, text_path_dir,
                    result.get("vad", ""), vad_path_dir)
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
