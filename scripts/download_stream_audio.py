#!/usr/bin/env python3
"""Download the full audio track of one YouTube video (or ended live stream).

Separate from download_playlist_audio.py on purpose: that one walks a playlist and
converts each entry to mp3. This one fetches ONE url, keeps the original m4a, and is
built for the thing that kept biting us -- a multi-hour ENDED LIVE STREAM.

Two properties of a post-live VOD that a normal upload does not have:

1. It is served as thousands of small DASH fragments. Fetched one at a time (yt-dlp's
   default) a 7-hour stream came down at 165 KiB/s; with concurrent_fragment_downloads=8
   the same file ran at 1.1 MiB/s -- 7x, on the same connection, same minute. The
   bottleneck is per-fragment round-trips, not bandwidth. Hence --concurrent, default 8.

2. It is NOT seekable over HTTP, so yt-dlp's --download-sections cannot slice it. Every
   attempt returns a ~1 KB file containing a container header and no audio: ffmpeg seeks,
   the server sends from the top anyway, ffmpeg subtracts the seek offset, every packet
   lands before zero and is discarded. Measured on iDwoXZqL56g: asking for 5:30:00 gave
   `time=-05:29:55` (= -19795s, the 19800s seek minus ~5s of real packets). This is why
   the pipeline downloads the WHOLE track and slices it locally -- see slice_audio.py.

The download is cached by video id, so slicing a second segment out of today's stream
costs nothing. Writes a .info.json beside the audio: the slice filenames are
date_start_end by convention and carry no video id, so this file is where provenance
lives.

    python scripts/download_stream_audio.py "https://www.youtube.com/live/iDwoXZqL56g"
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    sys.exit("yt-dlp is not installed. Run: pip install yt-dlp")

DEFAULT_DIR = Path("ALRA_TV_LIVE_STREAMS/_source")

# 140 is YouTube's 128 kbps AAC audio-only stream: ~50 MB/hour, and the same format
# every one of these livestreams offers. Falling back to bestaudio keeps this working
# on a video that happens not to have it.
AUDIO_FORMAT = "140/bestaudio[ext=m4a]/bestaudio"

# A container header with no audio is ~1 KB (see the docstring). Even a one-minute clip
# clears 100 KB, so anything under this is a failed download wearing a plausible name.
MIN_PLAUSIBLE_BYTES = 100_000


def stream_date(info: dict) -> str:
    """YYYY-MM-DD for the day the stream went out.

    release_date is the broadcast day for a live stream; upload_date can differ (and for
    a stream that ran past midnight, does). Prefer release_date, fall back to upload.
    """
    raw = info.get("release_date") or info.get("upload_date") or ""
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return "unknown-date"


def probe(url: str) -> dict:
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        return ydl.extract_info(url, download=False)


def download(url: str, out_dir: Path = DEFAULT_DIR, concurrent: int = 8,
             force: bool = False) -> tuple[Path, dict]:
    """Fetch the full audio track. Returns (audio path, info dict)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe(url)
    video_id = info["id"]
    target = out_dir / f"{video_id}.m4a"

    meta = {
        "video_id": video_id,
        "url": url,
        "title": info.get("title", ""),
        "stream_date": stream_date(info),
        "duration_s": info.get("duration"),
        "duration_str": info.get("duration_string", ""),
        "was_live": bool(info.get("was_live")),
    }
    (out_dir / f"{video_id}.info.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{meta['title']}")
    print(f"   {video_id}   {meta['duration_str']}   streamed {meta['stream_date']}"
          f"   {'(was live)' if meta['was_live'] else ''}")

    if target.exists() and not force:
        size_mb = target.stat().st_size / 1e6
        print(f"   already downloaded: {target}  ({size_mb:.0f} MB) — skipping")
        return target, meta

    options = {
        "format": AUDIO_FORMAT,
        "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
        "concurrent_fragment_downloads": concurrent,
        "continuedl": True,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
    }
    print(f"   downloading with {concurrent} concurrent fragments ...")
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])

    # Never trust the exit status alone. yt-dlp returns 0 after producing an empty file
    # in exactly the failure mode described in the docstring, and the caller then reports
    # success on a file that has no audio in it.
    if not target.exists():
        raise RuntimeError(f"download produced no file at {target}")
    size = target.stat().st_size
    if size < MIN_PLAUSIBLE_BYTES:
        raise RuntimeError(
            f"{target} is only {size} bytes — that is a container header, not audio. "
            "The stream was not fetched.")

    print(f"   ✓ {target}  ({size / 1e6:.0f} MB)")
    return target, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("url", help="YouTube video or live-stream URL")
    parser.add_argument("-o", "--out-dir", default=str(DEFAULT_DIR),
                        help=f"where to keep source audio (default: {DEFAULT_DIR})")
    parser.add_argument("-N", "--concurrent", type=int, default=8,
                        help="concurrent fragment downloads (default: 8; 1 is ~7x slower)")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file is already there")
    args = parser.parse_args()

    download(args.url, Path(args.out_dir), args.concurrent, args.force)


if __name__ == "__main__":
    main()
