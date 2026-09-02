#!/usr/bin/env python3
"""Download audio for every video in a YouTube playlist using yt-dlp."""

import argparse
import re
import sys
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    sys.exit("yt-dlp is not installed. Run: pip install yt-dlp")

EP_PATTERN = re.compile(r"EP\d+", re.IGNORECASE)
INVALID_CHARS = re.compile(r'[<>:"/\\|?*]')


def build_filename(title: str, video_id: str) -> str:
    match = EP_PATTERN.search(title)
    label = match.group(0).upper() if match else INVALID_CHARS.sub("", title)[:50].strip()
    return f"{label}_{video_id}"


def download_playlist_audio(playlist_url: str, output_dir: str, audio_format: str, audio_quality: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with yt_dlp.YoutubeDL({"extract_flat": True, "quiet": True}) as ydl:
        playlist_info = ydl.extract_info(playlist_url, download=False)

    # A playlist URL yields "entries"; a single-video URL yields the video info
    # itself, so fall back to treating the top-level info as one entry.
    entries = [entry for entry in playlist_info.get("entries", []) if entry]
    if not entries:
        entries = [playlist_info]

    failed: list = []
    for entry in entries:
        video_id = entry["id"]
        title = entry.get("title", video_id)
        filename = build_filename(title, video_id)

        if (output_path / f"{filename}.{audio_format}").exists():
            print(f"Skipping (already downloaded): {filename}")
            continue

        video_url = entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(output_path / f"{filename}.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": audio_quality,
            }],
            "ignoreerrors": True,
        }

        print(f"Downloading: {filename}  ({title})")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # ignoreerrors=True is deliberate — one dead video should not abort a
        # 50-video playlist — but it also means ydl.download() returns quietly
        # after a 403, and the caller then printed "✅ Saved to ...". Check that
        # the file is actually on disk, or the failure is invisible until
        # something downstream cannot find the audio.
        if not (output_path / f"{filename}.{audio_format}").exists():
            failed.append((video_id, title))
            print(f"  ❌ FAILED: {filename} produced no {audio_format} file")

    if failed:
        print(f"\n❌ {len(failed)} of {len(entries)} download(s) failed:")
        for vid, title in failed:
            print(f"     {vid}  {title}")
        if len(failed) == len(entries):
            raise RuntimeError(
                f"No audio was downloaded ({len(failed)}/{len(entries)} failed). "
                "A 403 here usually means yt-dlp is behind YouTube's player "
                "changes — try: python -m pip install -U yt-dlp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download audio from a YouTube playlist or single video.")
    parser.add_argument("playlist_url", help="YouTube playlist or video URL")
    parser.add_argument("-o", "--output-dir", default="audio", help="Directory to save audio files (default: audio)")
    parser.add_argument("-f", "--format", default="mp3", choices=["mp3", "m4a", "wav", "opus", "flac"], help="Audio format (default: mp3)")
    parser.add_argument("-q", "--quality", default="192", help="Audio bitrate in kbps (default: 192)")
    args = parser.parse_args()

    download_playlist_audio(args.playlist_url, args.output_dir, args.format, args.quality)


if __name__ == "__main__":
    main()
