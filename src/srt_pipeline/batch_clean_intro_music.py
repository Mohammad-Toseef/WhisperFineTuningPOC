"""
Batch-trim intro speech+music (and outro music, where noted) off episode
audio using the timestamps in an episode_skip_start*.xlsx file. Only
processes episodes that already have a transcript in raw_transcripts/ --
the rest aren't ready for alignment yet.

Columns expected:
  audio_filename        -- matches a file in audio/
  skip_start_seconds     -- seconds where real speech begins
  speech_end_timestamp   -- optional, "MM:SS"/"H:MM:SS", where real speech
                             ends (same timeline as skip_start_seconds);
                             blank means no outro to cut

Run:
  python batch_clean_intro_music.py --xlsx episode_skip_start_v2.xlsx
"""
import argparse
from pathlib import Path

import pandas as pd

from clean_intro_music import probe_duration, trim_audio

AUDIO_DIR = Path("audio")
TRANSCRIPT_DIR = Path("raw_transcripts")
OUT_DIR = "music_cleaned_audio"


def parse_timestamp(value) -> float:
    seconds = 0.0
    for part in str(value).strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", default="episode_skip_start_v2.xlsx")
    args = parser.parse_args()

    df = pd.read_excel(args.xlsx)

    processed, skipped = [], []
    for _, row in df.iterrows():
        filename = row["audio_filename"]
        audio_path = AUDIO_DIR / filename
        transcript_path = TRANSCRIPT_DIR / (Path(filename).stem + ".txt")

        if not transcript_path.exists():
            continue  # no transcript yet -- not ready for alignment

        skip_start = row["skip_start_seconds"]
        if pd.isna(skip_start):
            print(f"  ! no skip_start_seconds, skipping: {filename}")
            skipped.append(filename)
            continue
        if not audio_path.exists():
            print(f"  ! audio file missing, skipping: {filename}")
            skipped.append(filename)
            continue

        original_duration = probe_duration(str(audio_path))

        skip_end = 0.0
        end_ts = row.get("speech_end_timestamp")
        if pd.notna(end_ts):
            skip_end = original_duration - parse_timestamp(end_ts)
            if skip_end <= 0:
                print(f"  ! speech_end_timestamp ({end_ts}) is at/after the end of {filename} ({original_duration:.1f}s), ignoring outro cut")
                skip_end = 0.0

        out_path = trim_audio(str(audio_path), float(skip_start), OUT_DIR, skip_end=skip_end)
        trimmed_duration = probe_duration(out_path)
        print(f"  {filename}: {original_duration:.1f}s -> {trimmed_duration:.1f}s (skip_start {skip_start:.0f}s, outro cut {skip_end:.1f}s)")
        processed.append(filename)

    print(f"\nProcessed {len(processed)} episodes into {OUT_DIR}/")
    if skipped:
        print(f"Skipped {len(skipped)}: {skipped}")


if __name__ == "__main__":
    main()
