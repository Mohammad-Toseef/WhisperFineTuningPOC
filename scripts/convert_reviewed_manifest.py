"""
Convert a reviewed manifest (exported from the review portal) back to the
standard manifest.json format used by the training pipeline.

Reviewed format (flat list):
    {
        "audio_s3_key": "dataset/{batch_name}/{ep}_{ytid}/{index:03d}.wav",
        "transcript": "...",
        "duration": 22.98,
        "language": "ur",
        "batch_name": "...",
        "episode_label": "EP1",
        "youtube_video_id": "hBK8bkFgus8",
        "chunk_index": 0,
        "status": "reviewed"
    }

Original format:
    {
        "audio_path": "data\\processed\\{batch_folder}\\audio\\{ep}_{ytid}\\{ep}_{ytid}_{index:03d}.wav",
        "transcript": "...",
        "duration": 22.98,
        "language": "ur"
    }

The script auto-detects the matching local batch folder by comparing the set of
(episode_label, youtube_video_id, chunk_index) tuples across all local manifests.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def parse_local_manifest(manifest_path: Path) -> dict[tuple, dict]:
    """Return a lookup: (episode_label, youtube_video_id, chunk_index) -> entry."""
    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)

    lookup = {}
    for entry in entries:
        audio_path = Path(entry["audio_path"].replace("\\", "/"))
        folder_name = audio_path.parent.name          # e.g. EP1_hBK8bkFgus8
        filename_stem = audio_path.stem               # e.g. EP1_hBK8bkFgus8_000
        chunk_index = int(filename_stem.rsplit("_", 1)[-1])

        m = re.match(r"^(EP\d+)_(.+)$", folder_name)
        if not m:
            continue
        episode_label, youtube_video_id = m.group(1), m.group(2)
        key = (episode_label, youtube_video_id, chunk_index)
        lookup[key] = entry

    return lookup


def find_matching_batch(
    reviewed_keys: set[tuple],
    processed_dir: Path,
) -> tuple[str, dict[tuple, dict]]:
    """
    Find the local batch folder whose manifest covers exactly the same
    (episode_label, youtube_video_id, chunk_index) tuples as the reviewed file.
    Returns (batch_folder_name, lookup_dict).
    """
    candidates = []
    for manifest_path in sorted(processed_dir.glob("*/manifest.json")):
        batch_folder = manifest_path.parent.name
        lookup = parse_local_manifest(manifest_path)
        local_keys = set(lookup.keys())

        overlap = reviewed_keys & local_keys
        if not overlap:
            continue

        # Jaccard similarity: penalises superset matches (local has extra chunks)
        jaccard = len(overlap) / len(reviewed_keys | local_keys)
        candidates.append((jaccard, batch_folder, lookup))

    if not candidates:
        print("ERROR: No local manifest shares any entries with the reviewed manifest.", file=sys.stderr)
        sys.exit(1)

    candidates.sort(key=lambda c: c[0], reverse=True)
    best_jaccard, best_folder, best_lookup = candidates[0]
    overlap_count = len(reviewed_keys & set(best_lookup.keys()))
    coverage = overlap_count / len(reviewed_keys)

    print(f"Matched local batch folder: '{best_folder}' ({coverage:.0%} coverage, Jaccard={best_jaccard:.2f})")

    if coverage < 1.0:
        missing = len(reviewed_keys) - overlap_count
        print(
            f"WARNING: {missing} reviewed entries have no matching local audio file. "
            "They will use a reconstructed path.",
            file=sys.stderr,
        )

    return best_folder, best_lookup


def build_audio_path(batch_folder: str, episode_label: str, youtube_video_id: str, chunk_index: int) -> str:
    ep_ytid = f"{episode_label}_{youtube_video_id}"
    filename = f"{ep_ytid}_{chunk_index:03d}.wav"
    return f"data\\processed\\{batch_folder}\\audio\\{ep_ytid}\\{filename}"


def convert(reviewed_path: Path, processed_dir: Path, output_path: Path) -> None:
    with open(reviewed_path, encoding="utf-8") as f:
        reviewed = json.load(f)

    reviewed_keys = {
        (entry["episode_label"], entry["youtube_video_id"], entry["chunk_index"])
        for entry in reviewed
    }

    batch_folder, local_lookup = find_matching_batch(reviewed_keys, processed_dir)

    output_entries = []
    unmatched = 0

    for entry in reviewed:
        ep = entry["episode_label"]
        ytid = entry["youtube_video_id"]
        idx = entry["chunk_index"]
        key = (ep, ytid, idx)

        local_entry = local_lookup.get(key)
        if local_entry:
            audio_path = local_entry["audio_path"]
        else:
            # Reconstruct path from identifiers — may not exist on disk
            audio_path = build_audio_path(batch_folder, ep, ytid, idx)
            unmatched += 1

        output_entries.append({
            "audio_path": audio_path,
            "transcript": entry["transcript"],
            "duration": entry["duration"],
            "language": entry["language"],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_entries, f, ensure_ascii=False, indent=2)

    print(f"Written {len(output_entries)} entries -> {output_path}")
    if unmatched:
        print(f"  ({unmatched} paths reconstructed — verify these files exist on disk)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert reviewed manifest back to training manifest format.")
    parser.add_argument(
        "reviewed_manifest",
        nargs="?",
        default="all_batches_reviewed_manifest (1).json",
        help="Path to the reviewed manifest exported from the review portal.",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Root directory containing local batch folders (default: data/processed).",
    )
    parser.add_argument(
        "--batch-folder",
        default=None,
        help=(
            "Force a specific local batch folder (e.g. Batch1_EP23) instead of "
            "auto-detecting by Jaccard. Use when multiple folders contain the same "
            "audio and you want the canonical one."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output manifest path. Defaults to data/processed/{matched_batch}/manifest_reviewed.json. "
            "Use --output-name to change just the filename."
        ),
    )
    parser.add_argument(
        "--output-name",
        default="manifest_reviewed.json",
        help="Output filename within the matched batch folder (default: manifest_reviewed.json).",
    )
    args = parser.parse_args()

    reviewed_path = Path(args.reviewed_manifest)
    if not reviewed_path.exists():
        print(f"ERROR: Reviewed manifest not found: {reviewed_path}", file=sys.stderr)
        sys.exit(1)

    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        print(f"ERROR: Processed directory not found: {processed_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine output path after we know the batch folder
    with open(reviewed_path, encoding="utf-8") as f:
        reviewed = json.load(f)

    reviewed_keys = {
        (entry["episode_label"], entry["youtube_video_id"], entry["chunk_index"])
        for entry in reviewed
    }
    if args.batch_folder:
        forced_manifest = processed_dir / args.batch_folder / "manifest.json"
        if not forced_manifest.exists():
            print(f"ERROR: --batch-folder manifest not found: {forced_manifest}", file=sys.stderr)
            sys.exit(1)
        batch_folder = args.batch_folder
        local_lookup = parse_local_manifest(forced_manifest)
        overlap = reviewed_keys & set(local_lookup.keys())
        coverage = len(overlap) / len(reviewed_keys)
        print(f"Forced local batch folder: '{batch_folder}' ({coverage:.0%} coverage)")
        if coverage < 1.0:
            print(
                f"WARNING: {len(reviewed_keys) - len(overlap)} reviewed entries have no "
                f"matching audio in '{batch_folder}' — reconstructed paths may not exist.",
                file=sys.stderr,
            )
    else:
        batch_folder, local_lookup = find_matching_batch(reviewed_keys, processed_dir)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = processed_dir / batch_folder / args.output_name

    # Re-run conversion using already-resolved batch info (avoid double scan)
    output_entries = []
    unmatched = 0
    for entry in reviewed:
        ep = entry["episode_label"]
        ytid = entry["youtube_video_id"]
        idx = entry["chunk_index"]
        key = (ep, ytid, idx)

        local_entry = local_lookup.get(key)
        if local_entry:
            audio_path = local_entry["audio_path"]
        else:
            audio_path = build_audio_path(batch_folder, ep, ytid, idx)
            unmatched += 1

        output_entries.append({
            "audio_path": audio_path,
            "transcript": entry["transcript"],
            "duration": entry["duration"],
            "language": entry["language"],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_entries, f, ensure_ascii=False, indent=2)

    print(f"Written {len(output_entries)} entries -> {output_path}")
    if unmatched:
        print(f"  ({unmatched} paths reconstructed — verify these files exist on disk)")


if __name__ == "__main__":
    main()
