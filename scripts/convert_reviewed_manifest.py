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


def _audio_exists(audio_path: str) -> bool:
    """True if the (Windows-or-POSIX) audio_path resolves to a file on disk."""
    return Path(audio_path.replace("\\", "/")).exists()


def build_global_lookup(
    processed_dir: Path,
) -> tuple[dict[tuple, dict], dict[tuple, str]]:
    """
    Build a lookup across ALL local batch manifests, keyed by
    (episode_label, youtube_video_id, chunk_index) -> entry.

    A reviewed manifest that spans several batch folders (e.g. 49 episodes across
    Batch1_EP23 = EP1-23 and Batch2_EP24_EP50 = EP24-50) resolves each clip from
    whichever folder actually contains it — no single "best" folder is assumed.

    On duplicate keys (same clip present in multiple folders), prefer an entry
    whose audio file exists on disk; otherwise keep the first (folder-sorted) one.

    Returns (lookup, source_folder) where source_folder maps key -> batch folder.
    """
    lookup: dict[tuple, dict] = {}
    source_folder: dict[tuple, str] = {}

    for manifest_path in sorted(processed_dir.glob("*/manifest.json")):
        batch_folder = manifest_path.parent.name
        for key, entry in parse_local_manifest(manifest_path).items():
            if key in lookup:
                existing_ok = _audio_exists(lookup[key]["audio_path"])
                candidate_ok = _audio_exists(entry["audio_path"])
                # Keep existing unless it's a dead path and the candidate is live.
                if existing_ok or not candidate_ok:
                    continue
            lookup[key] = entry
            source_folder[key] = batch_folder

    if not lookup:
        print(f"ERROR: No local */manifest.json found under {processed_dir}", file=sys.stderr)
        sys.exit(1)

    return lookup, source_folder


def _ep_num(episode_label: str) -> int:
    m = re.match(r"EP(\d+)", episode_label)
    return int(m.group(1)) if m else 0


def report_coverage(
    reviewed: list[dict],
    lookup: dict[tuple, dict],
    source_folder: dict[tuple, str] | None,
) -> int:
    """Print per-episode matched/total coverage. Returns total unmatched count."""
    from collections import defaultdict

    matched_by_ep: dict[str, int] = defaultdict(int)
    total_by_ep: dict[str, int] = defaultdict(int)
    folders_used: set[str] = set()

    for entry in reviewed:
        ep = entry["episode_label"]
        key = (ep, entry["youtube_video_id"], entry["chunk_index"])
        total_by_ep[ep] += 1
        if key in lookup:
            matched_by_ep[ep] += 1
            if source_folder is not None:
                folders_used.add(source_folder[key])

    print("Coverage by episode:")
    n_missing = 0
    for ep in sorted(total_by_ep, key=_ep_num):
        matched, total = matched_by_ep[ep], total_by_ep[ep]
        missing = total - matched
        n_missing += missing
        flag = "" if missing == 0 else f"   <-- {missing} MISSING"
        print(f"  {ep:<6} {matched}/{total}{flag}")
    print(f"  {'TOTAL':<6} {sum(matched_by_ep.values())}/{sum(total_by_ep.values())} "
          f"across {len(total_by_ep)} episodes")
    if source_folder is not None and folders_used:
        print(f"  Resolved from folders: {sorted(folders_used)}")
    return n_missing


def build_audio_path(batch_folder: str, episode_label: str, youtube_video_id: str, chunk_index: int) -> str:
    ep_ytid = f"{episode_label}_{youtube_video_id}"
    filename = f"{ep_ytid}_{chunk_index:03d}.wav"
    return f"data\\processed\\{batch_folder}\\audio\\{ep_ytid}\\{filename}"


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
    parser.add_argument(
        "--allow-reconstruct",
        action="store_true",
        help=(
            "Do not fail on unmatched clips; reconstruct their audio_path from "
            "identifiers instead (paths may not exist on disk). Only meaningful "
            "with --batch-folder. Default: unmatched clips are a hard error."
        ),
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

    # ── Resolve the local audio lookup ──────────────────────────────
    #  --batch-folder   → single forced folder (legacy behaviour)
    #  default          → GLOBAL lookup across every data/processed/*/manifest.json
    #                     so a manifest spanning multiple batch folders (e.g. 49
    #                     episodes across Batch1_EP23 + Batch2_EP24_EP50) resolves.
    if args.batch_folder:
        forced_manifest = processed_dir / args.batch_folder / "manifest.json"
        if not forced_manifest.exists():
            print(f"ERROR: --batch-folder manifest not found: {forced_manifest}", file=sys.stderr)
            sys.exit(1)
        batch_folder = args.batch_folder
        local_lookup = parse_local_manifest(forced_manifest)
        source_folder = None            # single-folder mode
        print(f"Using forced local batch folder: '{batch_folder}'")
    else:
        batch_folder = None
        local_lookup, source_folder = build_global_lookup(processed_dir)
        print(f"Built global lookup: {len(local_lookup)} local clips across all batch folders")

    # ── Coverage report (per episode) ───────────────────────────────
    n_missing = report_coverage(reviewed, local_lookup, source_folder)
    if n_missing and not args.allow_reconstruct:
        print(
            f"\nERROR: {n_missing} reviewed clips have no matching local audio. "
            "Fix the source audio / batch folders, or pass --allow-reconstruct "
            "to write reconstructed paths anyway (requires --batch-folder).",
            file=sys.stderr,
        )
        sys.exit(1)
    if n_missing and args.allow_reconstruct and batch_folder is None:
        print(
            "\nERROR: --allow-reconstruct needs --batch-folder to know which folder "
            "to reconstruct the missing paths under.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Output path ─────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    elif batch_folder is not None:
        output_path = processed_dir / batch_folder / args.output_name
    else:
        output_path = processed_dir / args.output_name

    # ── Build converted manifest ────────────────────────────────────
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
            # Only reachable with --allow-reconstruct + --batch-folder.
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

    print(f"\nWritten {len(output_entries)} entries -> {output_path}")
    if unmatched:
        print(f"  ({unmatched} paths reconstructed — verify these files exist on disk)")


if __name__ == "__main__":
    main()
