#!/usr/bin/env python3
"""Daily driver: YouTube live-stream URL + a time range -> Urdu transcript.

Three stages, each of which is also a standalone script you can run on its own:

    download   scripts/download_stream_audio.py   full audio track, cached by video id
    slice      scripts/slice_audio.py             local ffmpeg cut, verified with ffprobe
    transcribe scripts/transcribe.py              one model on Modal (A10G)

    python run_stream_pipeline.py \
        --url "https://www.youtube.com/live/iDwoXZqL56g" \
        --start 5:30:00 --end 5:49:00 \
        --model-path /data/model/whisper-urdu-r3-final

Every stage is idempotent: the source track, the slice and the transcript are each
skipped if they already exist, so re-running after a failure resumes rather than
repeats. Cutting a second segment out of the same day's stream therefore costs one
ffmpeg call -- the 400 MB download is already on disk.

--model-path is REQUIRED. There is no default because the wrong one is not detectable
by looking at the output: an old round produces a real, fluent, plausible Urdu
transcript, and we have already shipped one demo that way.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from scripts.download_stream_audio import download          # noqa: E402
from scripts.slice_audio import slice_audio                 # noqa: E402

SOURCE_DIR = REPO / "ALRA_TV_LIVE_STREAMS" / "_source"
SLICE_DIR = REPO / "ALRA_TV_LIVE_STREAMS"


def modal_executable() -> str:
    """The venv's modal, not whatever is first on PATH.

    Modal auth lives in ~/.modal.toml but the CLI has to come from the environment that
    has our dependencies; picking up a different one is how a run dies at import time.
    """
    for name in ("modal.exe", "modal"):
        candidate = Path(sys.executable).parent / name
        if candidate.exists():
            return str(candidate)
    return "modal"


def check_model_path(model_path: str) -> None:
    """Catch Git Bash's path rewriting before it reaches the GPU.

    MSYS turns a leading /data/... into C:/Program Files/Git/data/..., which then fails
    inside the container as a missing model — twenty minutes and one A10G later. Cheaper
    to say so here.
    """
    if "Program Files/Git" in model_path or "Program Files\\Git" in model_path:
        raise SystemExit(
            f"--model-path is {model_path!r} — Git Bash rewrote a /data/... path.\n"
            "Re-run from PowerShell, or prefix the command with MSYS_NO_PATHCONV=1.")


def transcribe(audio: Path, model_path: str, out_dir: Path, language: str) -> Path:
    model_tag = Path(model_path.rstrip("/")).name
    out_path = out_dir / f"{audio.stem}_{model_tag}.txt"
    if out_path.exists():
        print(f"   already transcribed: {out_path} — skipping")
        return out_path

    command = [
        modal_executable(), "run", "scripts/transcribe.py",
        "--audio", str(audio),
        "--model-path", model_path,
        "--out-dir", str(out_dir),
        "--language", language,
    ]
    print(f"\n$ {' '.join(command)}")
    code = subprocess.run(command, cwd=str(REPO)).returncode
    if code != 0:
        raise SystemExit(f"transcription failed (exit {code})")
    if not out_path.exists():
        raise SystemExit(f"transcription reported success but {out_path} is not there")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", required=True, help="YouTube live-stream or video URL")
    parser.add_argument("--start", required=True, help="H:MM:SS, MM:SS or seconds")
    parser.add_argument("--end", required=True, help="H:MM:SS, MM:SS or seconds")
    parser.add_argument("--model-path", required=True,
                        help="/data/... on the Modal volume, or a HuggingFace model id")
    parser.add_argument("--out-dir", default=str(SLICE_DIR),
                        help=f"slice + transcript go here (default: {SLICE_DIR.name}/)")
    parser.add_argument("--source-dir", default=str(SOURCE_DIR),
                        help="where full downloaded tracks are cached")
    parser.add_argument("--date", default="",
                        help="stream date for the filename; defaults to the stream's own")
    parser.add_argument("--language", default="ur", help="forced decoder language (default: ur)")
    parser.add_argument("-N", "--concurrent", type=int, default=8,
                        help="concurrent fragment downloads (default: 8)")
    parser.add_argument("--keep-source", action="store_true",
                        help="(default) keep the full track for further slices")
    parser.add_argument("--drop-source", action="store_true",
                        help="delete the full track once the slice is made")
    args = parser.parse_args()

    check_model_path(args.model_path)
    out_dir = Path(args.out_dir)

    print("=" * 78)
    print("STAGE 1  download")
    print("=" * 78)
    source, _ = download(args.url, Path(args.source_dir), args.concurrent)

    print("\n" + "=" * 78)
    print("STAGE 2  slice")
    print("=" * 78)
    clip = slice_audio(source, args.start, args.end, out_dir, args.date)

    print("\n" + "=" * 78)
    print(f"STAGE 3  transcribe  ({args.model_path})")
    print("=" * 78)
    transcript = transcribe(clip, args.model_path, out_dir, args.language)

    if args.drop_source:
        source.unlink()
        print(f"\n   removed source track {source.name}")

    text = transcript.read_text(encoding="utf-8")
    print("\n" + "=" * 78)
    print(f"✅ {transcript}")
    print(f"   {len(text.split()):,} words in {len(text.splitlines())} segments")
    print(f"   model: {args.model_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
