"""
Trim the intro speech + music off the front of an episode's audio, given a
human-verified timestamp (in seconds) where the real speech begins.

Run:
  python clean_intro_music.py --audio audio/EP1_hBK8bkFgus8.mp3 --skip-start 82
"""
import argparse
import subprocess
from pathlib import Path


def probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def trim_audio(audio_path: str, skip_start: float, out_dir: str, skip_end: float = 0.0) -> str:
    out_path = str(Path(out_dir) / Path(audio_path).name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(skip_start), "-i", audio_path]
    if skip_end > 0:
        duration = probe_duration(audio_path) - skip_start - skip_end
        cmd += ["-t", str(duration)]
    cmd += ["-c", "copy", out_path]
    subprocess.run(cmd, check=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Trim intro speech+music off an episode audio file at a human-verified timestamp."
    )
    parser.add_argument("--audio", required=True)
    parser.add_argument("--skip-start", type=float, required=True, help="Seconds where real speech begins.")
    parser.add_argument("--skip-end", type=float, default=0.0, help="Seconds of outro music to cut off the end.")
    parser.add_argument("--out-dir", default="music_cleaned_audio")
    args = parser.parse_args()

    original_duration = probe_duration(args.audio)
    out_path = trim_audio(args.audio, args.skip_start, args.out_dir, skip_end=args.skip_end)
    trimmed_duration = probe_duration(out_path)

    print(f"Input:  {args.audio} ({original_duration:.2f}s)")
    print(f"Output: {out_path} ({trimmed_duration:.2f}s)")
    print(f"Trimmed {original_duration - trimmed_duration:.2f}s off the front (requested {args.skip_start:.2f}s)")


if __name__ == "__main__":
    main()
