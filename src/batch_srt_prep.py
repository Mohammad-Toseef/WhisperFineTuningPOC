"""
Batch-process a folder of audio + .srt pairs through srt_audio_prep.py's
per-video pipeline, then merge every video's chunks into one combined
manifest.json.

Pairing: audio and .srt files are matched by the 11-char YouTube ID embedded
in both filenames (see srt_audio_prep.find_youtube_id) -- not by shared
filename stem, since downloaded audio/subtitle filenames often differ
entirely except for that ID.

Run locally:
  python src/batch_srt_prep.py \
    --input_dir ./raw_batch \
    --output_dir ./data/processed/batch1
"""
import re
import sys
import json
import argparse
from pathlib import Path
from dataclasses import asdict

from srt_audio_prep import prepare_from_srt, find_youtube_id, make_video_id

sys.stdout.reconfigure(encoding="utf-8")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac"}


def natural_sort_key(path: str) -> list:
    """Split into digit/non-digit chunks so "EP2" sorts before "EP10",
    unlike plain string or YouTube-ID sorting (both arbitrary w.r.t. episode
    order)."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", Path(path).name)]


def find_pairs(input_dir: str) -> list[tuple[str, str]]:
    """Match audio files to .srt files in input_dir by shared YouTube ID."""
    audio_by_id = {}
    srt_by_id = {}
    for path in Path(input_dir).iterdir():
        if not path.is_file():
            continue
        youtube_id = find_youtube_id(str(path))
        if youtube_id is None:
            continue
        if path.suffix.lower() == ".srt":
            srt_by_id[youtube_id] = str(path)
        elif path.suffix.lower() in AUDIO_EXTENSIONS:
            audio_by_id[youtube_id] = str(path)

    pairs = []
    for youtube_id, audio_path in sorted(audio_by_id.items(), key=lambda kv: natural_sort_key(kv[1])):
        srt_path = srt_by_id.get(youtube_id)
        if srt_path is None:
            print(f"  ⚠ no matching .srt for YouTube ID {youtube_id} ({Path(audio_path).name}), skipping")
            continue
        pairs.append((audio_path, srt_path))
    for youtube_id in srt_by_id.keys() - audio_by_id.keys():
        print(f"  ⚠ no matching audio for YouTube ID {youtube_id} ({Path(srt_by_id[youtube_id]).name}), skipping")

    return pairs


def load_existing_manifest(output_dir: str) -> list[dict]:
    """Load this output_dir's manifest.json if one already exists, so a new
    batch run can merge in rather than overwrite previously processed videos."""
    manifest_path = Path(output_dir) / "manifest.json"
    if not manifest_path.exists():
        return []
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def batch_prepare(input_dir: str, output_dir: str, max_duration: float = 28.0):
    pairs = find_pairs(input_dir)
    print(f"Found {len(pairs)} audio+srt pairs in {input_dir}")

    existing_samples = load_existing_manifest(output_dir)
    # video_id is the audio chunk's parent folder name -- see make_video_id().
    already_done = {Path(s["audio_path"]).parent.name for s in existing_samples}
    if already_done:
        print(f"Existing manifest has {len(existing_samples)} chunks across {len(already_done)} video(s) -- merging new videos in")

    all_samples = list(existing_samples)
    new_video_count = 0
    for video_index, (audio_path, srt_path) in enumerate(pairs, start=1):
        video_id = make_video_id(audio_path, video_index)
        if video_id in already_done:
            print(f"\n--- [{video_index}/{len(pairs)}] {Path(audio_path).name}: already in manifest ({video_id}), skipping ---")
            continue

        print(f"\n--- [{video_index}/{len(pairs)}] {Path(audio_path).name} ---")
        try:
            samples = prepare_from_srt(
                audio_path, srt_path, output_dir, max_duration,
                video_index=video_index, write_manifest=False,
            )
        except Exception as e:
            print(f"  ✗ failed: {e}, skipping this video")
            continue
        all_samples.extend(asdict(s) for s in samples)
        new_video_count += 1

    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    total_hours = sum(s["duration"] for s in all_samples) / 3600
    print(f"\n✅ Batch complete: {new_video_count} new video(s) processed, {len(all_samples)} total chunks across {len(already_done) + new_video_count} videos ({total_hours:.2f} hours)")
    print(f"   Combined manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="./data/processed/batch")
    parser.add_argument("--max_duration", type=float, default=28.0)
    args = parser.parse_args()

    batch_prepare(args.input_dir, args.output_dir, args.max_duration)
