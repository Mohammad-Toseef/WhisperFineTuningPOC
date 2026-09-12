#!/usr/bin/env python3
"""Cut [start, end) out of a local audio file with ffmpeg.

Local files ARE seekable, which is the whole reason this step exists: yt-dlp cannot
slice an ended live stream over HTTP (see download_stream_audio.py), but once the track
is on disk the cut is exact and takes seconds.

Output is named <stream date>_<start>_<end>.mp3 with clock parts hyphenated, e.g.
2026-09-03_05-30-00_05-49-00.mp3 — the agreed convention for these daily segments.

    python scripts/slice_audio.py ALRA_TV_LIVE_STREAMS/_source/iDwoXZqL56g.m4a \
        --start 5:30:00 --end 5:49:00 --date 2026-09-03
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DIR = Path("ALRA_TV_LIVE_STREAMS")
INDEX_NAME = "slices.jsonl"

# The cut is verified against ffprobe afterwards. mp3 frame boundaries and the encoder's
# own padding move the duration by a few tens of ms, so the tolerance is not zero -- but
# it is small enough that a silently-empty or wrongly-seeked file cannot slip through.
DURATION_TOLERANCE_S = 1.0


def parse_clock(value: str) -> float:
    """'5:30:00' -> 19800.0, '12:30' -> 750.0, '90' -> 90.0.

    Colon count decides the meaning, nothing else: three parts are H:MM:SS, two are
    MM:SS, one is seconds. Stated explicitly because guessing from magnitude is how the
    Batch 4 spreadsheet ended up with speech_end values 60x too large.
    """
    text = str(value).strip()
    if not text:
        raise ValueError("empty time")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"{value!r} is not H:MM:SS, MM:SS or seconds")
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"{value!r} is not a time") from None
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def clock_label(seconds: float) -> str:
    """19800 -> '05-30-00'. Hyphens because ':' is illegal in a Windows filename."""
    total = int(round(seconds))
    return f"{total // 3600:02d}-{(total % 3600) // 60:02d}-{total % 60:02d}"


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    return float(result.stdout.strip())


def source_meta(source: Path) -> dict:
    info = source.with_suffix(".info.json")
    if info.exists():
        try:
            return json.loads(info.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


def slice_audio(source: Path, start: str, end: str, out_dir: Path = DEFAULT_DIR,
                date: str = "", force: bool = False) -> Path:
    if not source.exists():
        raise SystemExit(f"source audio not found: {source}")

    start_s, end_s = parse_clock(start), parse_clock(end)
    if end_s <= start_s:
        raise SystemExit(f"end ({end}) must be after start ({start})")
    duration = end_s - start_s

    meta = source_meta(source)
    stamp = date or meta.get("stream_date") or "unknown-date"

    total = probe_duration(source)
    if end_s > total:
        raise SystemExit(
            f"end {end} ({end_s:.0f}s) is past the end of {source.name} "
            f"({total:.0f}s / {int(total // 3600)}:{int(total % 3600 // 60):02d}).")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp}_{clock_label(start_s)}_{clock_label(end_s)}.mp3"
    if out_path.exists() and not force:
        print(f"   already sliced: {out_path} — skipping")
        return out_path

    # -ss BEFORE -i seeks first and decodes only what is needed; on a local file that is
    # both fast and frame-accurate. -t (duration) rather than -to (absolute) because -to
    # is measured against the input timeline and silently produced an empty file when the
    # timestamps did not start at zero.
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_s:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
        "-vn", "-c:a", "libmp3lame", "-q:a", "2", "-ar", "16000", "-ac", "1",
        str(out_path),
    ]
    print(f"   cutting {start} → {end}  ({duration / 60:.1f} min) from {source.name}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")

    # Verify, do not assume. The whole reason this script exists is that the obvious way
    # to do this produced a well-formed file containing no audio, and reported success.
    actual = probe_duration(out_path)
    if abs(actual - duration) > DURATION_TOLERANCE_S:
        raise RuntimeError(
            f"{out_path.name} is {actual:.1f}s but {duration:.1f}s was requested. "
            "The cut did not land where it was asked to — not using this file.")

    print(f"   ✓ {out_path}  ({actual / 60:.1f} min, {out_path.stat().st_size / 1e6:.1f} MB)")

    # The filename is date_start_end by convention and carries no video id, so the link
    # back to the source would be lost the moment the file is copied anywhere. Append-only
    # index, one line per slice.
    if meta:
        record = {
            "slice": out_path.name,
            "video_id": meta.get("video_id", ""),
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "stream_date": stamp,
            "start": start, "end": end,
            "sliced_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with (out_dir / INDEX_NAME).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", help="local audio file to cut")
    parser.add_argument("--start", required=True, help="H:MM:SS, MM:SS or seconds")
    parser.add_argument("--end", required=True, help="H:MM:SS, MM:SS or seconds")
    parser.add_argument("-o", "--out-dir", default=str(DEFAULT_DIR),
                        help=f"where the slice goes (default: {DEFAULT_DIR})")
    parser.add_argument("--date", default="",
                        help="stream date for the filename; defaults to the source .info.json")
    parser.add_argument("--force", action="store_true", help="overwrite an existing slice")
    args = parser.parse_args()

    slice_audio(Path(args.source), args.start, args.end,
                Path(args.out_dir), args.date, args.force)


if __name__ == "__main__":
    main()
