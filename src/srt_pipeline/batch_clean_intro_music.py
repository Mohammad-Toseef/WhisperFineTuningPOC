"""
Batch-trim intro speech+music (and outro music, where noted) off episode audio.

Two input modes:

  --csv  (preferred)  a validated CSV from validate_sheet.py, which already has
                      every Excel quirk repaired: skip_start in real seconds and
                      speech_end_seconds numeric. Columns used:
                        audio_filename, skip_start_seconds, speech_end_seconds
                      plus label / excel_row for reporting.

  --xlsx (legacy)     the round-1 sheet shape, read straight from Excel. Columns:
                        audio_filename        -- matches a file in --audio-dir
                        skip_start_seconds    -- seconds where real speech begins
                        speech_end_timestamp  -- optional "MM:SS"/"H:MM:SS", blank
                                                 means no outro to cut
                      Only safe for sheets whose cells are plain ints/strings. The
                      'Batch 3' sheet is NOT: Excel retyped 80 of its skip cells as
                      datetime.time and 8 more use MM.SS notation. Use --csv there.

The outro cut is computed against ffprobe's REAL duration, never a sheet's stated
Duration -- the sheets are hand-maintained and some rows carry a speech_end past
the true end of the file:
    skip_end = real_duration - speech_end
A non-positive skip_end means the marked speech runs to the end, so there is no
outro to cut.

--require-transcript DIR restores the round-1 behaviour of only processing
episodes that already have a transcript (those were the ones ready for forced
alignment). It is OFF by default: in the Batch 3 flow transcripts are generated
*after* trimming, so gating on them would skip every row.

Idempotent: an existing output file means done, so a re-run only does what is
left. The filesystem is the state.

Run:
  # preflight -- probes every file, writes nothing
  python src/srt_pipeline/batch_clean_intro_music.py --dry-run
  python src/srt_pipeline/batch_clean_intro_music.py --limit 2
  # legacy round-1 shape
  python src/srt_pipeline/batch_clean_intro_music.py --xlsx episode_skip_start_v2.xlsx \
      --audio-dir audio --out-dir music_cleaned_audio --require-transcript raw_transcripts
"""
import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

from batch_paths import BatchPaths, add_batch_argument
from clean_intro_music import probe_duration, trim_audio

sys.stdout.reconfigure(encoding="utf-8")

# -c copy cuts on frame boundaries, so the result can differ from the arithmetic by
# a frame or two. Anything past this means the cut did not land where we asked.
TRIM_TOLERANCE_SECONDS = 2.0
# Below this a "trimmed episode" is almost certainly a bad row, not real audio.
MIN_TRIMMED_SECONDS = 60.0


def parse_timestamp(value) -> float:
    """'MM:SS' / 'H:MM:SS' / plain seconds -> seconds (legacy --xlsx path)."""
    if isinstance(value, dt.time):
        # Excel read a typed "MM:SS" as "H:MM". Bail loudly rather than guess here;
        # validate_sheet.py handles this properly and emits a clean CSV.
        raise ValueError(
            f"cell is a datetime.time ({value}) -- Excel retyped it. "
            "Run validate_sheet.py and feed this script --csv instead."
        )
    seconds = 0.0
    for part in str(value).strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def load_from_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [{
        "name": row.get("label") or Path(row["audio_filename"]).stem,
        "filename": row["audio_filename"],
        "skip_start": float(row["skip_start_seconds"]),
        "speech_end": float(row["speech_end_seconds"]),
        "source_ref": f"xlsx row {row['excel_row']}" if row.get("excel_row") else "",
    } for row in rows]


def load_from_xlsx(path: Path) -> list[dict]:
    import pandas as pd

    frame = pd.read_excel(path)
    records = []
    for index, row in frame.iterrows():
        filename = row["audio_filename"]
        if pd.isna(filename) or pd.isna(row["skip_start_seconds"]):
            continue
        end = row.get("speech_end_timestamp")
        records.append({
            "name": Path(str(filename)).stem,
            "filename": str(filename),
            "skip_start": parse_timestamp(row["skip_start_seconds"]),
            # Blank speech_end means "keep everything to the end"; carried as None and
            # resolved against the real duration once probed.
            "speech_end": None if pd.isna(end) else parse_timestamp(end),
            "source_ref": f"sheet row {index + 2}",
        })
    return records


def plan_one(record: dict, audio_dir: Path, out_dir: Path, transcript_dir: Path | None) -> dict:
    """Resolve one record into a concrete trim plan, or a reason it can't be trimmed."""
    raw_path = audio_dir / record["filename"]
    out_path = out_dir / record["filename"]
    plan = {
        **record,
        "raw_path": raw_path,
        "out_path": out_path,
        "status": "ready",
        "real_duration": None,
        "skip_end": 0.0,
        "expected_duration": None,
        "notes": [],
        "error": None,
    }

    if transcript_dir is not None and not (transcript_dir / f"{Path(record['filename']).stem}.txt").exists():
        plan["status"] = "no_transcript"
        return plan
    if out_path.exists():
        plan["status"] = "done"
        return plan
    if not raw_path.exists():
        plan["status"] = "not_downloaded"
        return plan

    real = probe_duration(str(raw_path))
    plan["real_duration"] = real

    if plan["skip_start"] >= real:
        plan["status"] = "error"
        plan["error"] = f"skip_start {plan['skip_start']:.0f}s is at/after end of file ({real:.0f}s)"
        return plan

    speech_end = real if record["speech_end"] is None else record["speech_end"]
    skip_end = real - speech_end
    if skip_end <= 0:
        if record["speech_end"] is not None:
            plan["notes"].append(f"speech_end {speech_end:.0f}s >= real duration {real:.0f}s — no outro cut")
        skip_end = 0.0
    plan["skip_end"] = skip_end

    expected = real - plan["skip_start"] - skip_end
    plan["expected_duration"] = expected
    if expected < MIN_TRIMMED_SECONDS:
        plan["status"] = "error"
        plan["error"] = f"trimmed audio would be only {expected:.0f}s — check the source row"
        return plan

    # A very large outro cut usually means speech_end was typed against a different
    # upload; flag it rather than silently discarding minutes of speech.
    if skip_end > 0.25 * real:
        plan["notes"].append(f"outro cut is {skip_end:.0f}s ({skip_end / real:.0%} of the file) — unusually large")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", help="Validated CSV from validate_sheet.py (default when neither flag is given)")
    source.add_argument("--xlsx", help="Legacy round-1 spreadsheet")
    add_batch_argument(parser)
    parser.add_argument("--audio-dir", help="Source audio (default: data/<batch>/audio_raw)")
    parser.add_argument("--out-dir", help="Trimmed output (default: data/<batch>/audio_trimmed)")
    parser.add_argument("--require-transcript", metavar="DIR",
                        help="Only process episodes that already have DIR/<stem>.txt (round-1 behaviour; off by default)")
    parser.add_argument("--limit", type=int, help="Only process the first N rows")
    parser.add_argument("--only", help="Comma-separated names/labels, e.g. B3001,B3002")
    parser.add_argument("--dry-run", action="store_true", help="Preflight: probe and report; write nothing")
    args = parser.parse_args()

    paths = BatchPaths(args.batch)
    if args.xlsx:
        source_path, loader = Path(args.xlsx), load_from_xlsx
    else:
        source_path, loader = Path(args.csv) if args.csv else paths.validated_csv, load_from_csv
    if not source_path.exists():
        sys.exit(f"Input not found: {source_path}\n"
                 f"Run: python src/srt_pipeline/validate_sheet.py --batch {args.batch}")

    records = loader(source_path)
    if args.only:
        wanted = {token.strip() for token in args.only.split(",") if token.strip()}
        unknown = wanted - {r["name"] for r in records}
        if unknown:
            sys.exit(f"--only referenced names not in {source_path}: {sorted(unknown)}")
        records = [r for r in records if r["name"] in wanted]
    if args.limit is not None:
        records = records[:args.limit]

    audio_dir = Path(args.audio_dir) if args.audio_dir else paths.audio_raw
    out_dir = Path(args.out_dir) if args.out_dir else paths.audio_trimmed
    transcript_dir = Path(args.require_transcript) if args.require_transcript else None
    plans = [plan_one(r, audio_dir, out_dir, transcript_dir) for r in records]

    buckets: dict[str, list[dict]] = {}
    for plan in plans:
        buckets.setdefault(plan["status"], []).append(plan)
    ready = buckets.get("ready", [])

    print("=" * 78)
    print(f"Input:        {source_path}")
    print(f"Audio dir:    {audio_dir}")
    print(f"Output dir:   {out_dir}")
    if transcript_dir:
        print(f"Transcript gate: {transcript_dir}")
    print(f"Selected {len(records)}  |  ready {len(ready)}  "
          f"|  already trimmed {len(buckets.get('done', []))}  "
          f"|  not downloaded {len(buckets.get('not_downloaded', []))}  "
          f"|  no transcript {len(buckets.get('no_transcript', []))}  "
          f"|  errors {len(buckets.get('error', []))}")
    print("=" * 78)

    if ready:
        print(f"\nPREFLIGHT — {len(ready)} file(s) ready to trim")
        print(f"  {'name':12} {'real':>7} {'head':>6} {'tail':>7} {'result':>8}")
        for plan in ready:
            print(f"  {plan['name']:12} {plan['real_duration']:7.0f} {plan['skip_start']:6.0f} "
                  f"{plan['skip_end']:7.0f} {plan['expected_duration']:8.0f}")
            for note in plan["notes"]:
                print(f"               ! {note}")

    for status, header in (
        ("not_downloaded", "NOT DOWNLOADED (run download_batch.py first)"),
        ("no_transcript", "NO TRANSCRIPT — skipped by --require-transcript"),
        ("done", "ALREADY TRIMMED — skipping"),
    ):
        entries = buckets.get(status, [])
        if entries:
            print(f"\n{header} — {len(entries)}")
            for plan in entries:
                print(f"   {plan['name']}  {plan['filename']}")

    errors = buckets.get("error", [])
    if errors:
        print(f"\nERRORS — {len(errors)}")
        for plan in errors:
            ref = f" ({plan['source_ref']})" if plan["source_ref"] else ""
            print(f"   {plan['name']}{ref}: {plan['error']}")

    if args.dry_run:
        total = sum(p["expected_duration"] for p in ready)
        print(f"\nWould produce {len(ready)} file(s), {total / 3600:.2f} hours of trimmed audio.")
        print("Dry-run — nothing written.")
        return

    if errors:
        sys.exit(f"\nRefusing to trim: fix the {len(errors)} erroring row(s) first.")
    if not ready:
        print("\nNothing to do.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    trimmed, failed = [], []

    print()
    for index, plan in enumerate(ready, start=1):
        print(f"[{index}/{len(ready)}] {plan['name']}  {plan['filename']}")
        try:
            trim_audio(str(plan["raw_path"]), plan["skip_start"], str(out_dir), skip_end=plan["skip_end"])
        except Exception as exc:
            print(f"   FAILED: {type(exc).__name__}: {exc}")
            failed.append((plan, str(exc)))
            continue

        actual = probe_duration(str(plan["out_path"]))
        delta = actual - plan["expected_duration"]
        flag = "" if abs(delta) <= TRIM_TOLERANCE_SECONDS else f"  ⚠ off by {delta:+.1f}s"
        print(f"   {plan['real_duration']:.0f}s -> {actual:.0f}s "
              f"(cut {plan['skip_start']:.0f}s head, {plan['skip_end']:.0f}s tail){flag}")
        trimmed.append(plan)

    print("\n" + "=" * 78)
    print(f"Trimmed: {len(trimmed)}    Failed: {len(failed)}")
    if failed:
        for plan, message in failed:
            print(f"   {plan['name']}  {message}")
        sys.exit(1)
    print("=" * 78)


if __name__ == "__main__":
    main()
