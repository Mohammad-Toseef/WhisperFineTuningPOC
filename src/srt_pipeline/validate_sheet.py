"""
Stage 0 of the batch pipeline: turn a hand-maintained episode spreadsheet into a
validated, machine-readable CSV that every downstream stage reads instead of the
.xlsx.

Why this stage exists: the sheet is filled in by hand and Excel silently
re-types things.  In the 'Batch 3' sheet, 80 of 99 skip_start_seconds cells came
back as datetime.time because Excel read the typed "0:09" as H:MM -- so a naive
float(cell) raises TypeError and batch_clean_intro_music.py dies on row 11.
Several duration cells use "MM.SS" decimal notation, one has a ';' for ':', and
three videos are listed twice.  Repairing that inline in each stage would mean
re-deriving the same guesses over and over, invisibly; doing it once here leaves
an auditable record of every value that was changed.

Repairs applied (each one reported, per row, in the `repairs` column):
  1. skip_start_seconds as datetime.time -> seconds.  Excel parsed the humans'
     "MM:SS" as "H:MM", so time(0,9) means 9s and time(1,9) means 69s; recover
     as hour*60 + minute.  (Verified across the sheet: .second is always 0 and
     the hour field never exceeds 1, so this reading is unambiguous.)
  2. "MM.SS" decimal clock -> seconds ("19.17" is 19m17s, not 19.17s).
  3. ';' typed for ':' in a clock value.
  4. Duplicate video IDs -> first occurrence wins, the rest are dropped.

Also assigns each video a STABLE label (B3001, B3002, ...) in sheet order.
Downstream, srt_audio_prep.make_video_id() reuses a filename's own label only
when it is a single token matching ^[A-Za-z]+\\d+$ -- otherwise it falls back to
vid{NNNN} derived from directory iteration order, which would renumber chunks
between runs and break the manifest<->audio correspondence.  Naming the
downloads "<label>_<youtube_id>.mp3" pins the video_id deterministically.

Run:
  python src/srt_pipeline/validate_sheet.py \
    --xlsx Whisper_Second_Round_Training_list.xlsx \
    --sheet "Batch 3" \
    --out data/batch3/batch3_validated.csv
"""
import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

import pandas as pd

from batch_paths import BatchPaths, add_batch_argument, label_prefix_for

sys.stdout.reconfigure(encoding="utf-8")

# Matches the 11-char video ID in every YouTube URL shape present in the sheet:
# youtu.be/<id>?si=..., watch?v=<id>&list=...&index=N, /shorts/<id>, /embed/<id>
VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")
DECIMAL_CLOCK_RE = re.compile(r"^\d+\.\d{1,2}$")

# A skip_start beyond this is almost certainly a misread cell rather than a real
# intro length -- worth surfacing even though we still emit the row.
IMPLAUSIBLE_SKIP_SECONDS = 300.0


def parse_skip_start(value) -> tuple[float | None, str | None, str | None]:
    """skip_start_seconds cell -> (seconds, repair_label, error).

    Handles the plain numeric cells and the datetime.time cells Excel produced by
    reading a typed "MM:SS" as "H:MM" (see module docstring, repair 1).
    """
    if pd.isna(value):
        return None, None, "skip_start_seconds is empty"
    if isinstance(value, dt.time):
        seconds = value.hour * 60 + value.minute + value.second / 60.0
        return seconds, f"skip_start: Excel MM:SS misparse ({value.strftime('%H:%M')} -> {seconds:.0f}s)", None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, None, f"skip_start_seconds unparseable: {value!r}"

    # A NON-INTEGER skip is MM.SS decimal notation, not fractional seconds. Every
    # such cell in 'Batch 3' (0.47, 0.15, 0.23, ...) sits in the block of rows whose
    # speech_end/Duration also use MM.SS, while every integer cell is 6-10 -- literal
    # seconds. Read 0.47 as 0m47s; reading it as 0.47s would leave the intro music in.
    if not number.is_integer():
        text = f"{number:.10g}"
        if DECIMAL_CLOCK_RE.match(text):
            minutes, _, secs = text.partition(".")
            seconds = float(minutes) * 60 + float(secs.ljust(2, "0"))
            return seconds, f"skip_start: MM.SS decimal notation ({text} -> {seconds:.0f}s)", None
        return None, None, (
            f"skip_start_seconds {value!r} is fractional but not MM.SS notation -- "
            "cannot tell seconds from minutes.seconds; fix the cell"
        )

    return number, None, None


def parse_clock(value, field: str) -> tuple[float | None, str | None, str | None]:
    """'H:MM:SS' / 'MM:SS' / 'MM.SS' -> (seconds, repair_label, error)."""
    if pd.isna(value):
        return None, None, f"{field} is empty"
    raw = str(value).strip()
    text, repair = raw, None

    if ";" in text:
        text = text.replace(";", ":")
        repair = f"{field}: ';' typo for ':' ({raw!r})"

    if ":" not in text:
        if DECIMAL_CLOCK_RE.match(text):
            minutes, _, secs = text.partition(".")
            seconds = float(minutes) * 60 + float(secs.ljust(2, "0"))
            return seconds, f"{field}: MM.SS decimal notation ({raw!r} -> {seconds:.0f}s)", None
        return None, None, f"{field} has no recognisable clock format: {raw!r}"

    total = 0.0
    for part in text.split(":"):
        try:
            total = total * 60 + float(part)
        except ValueError:
            return None, None, f"{field} has an unparseable component {part!r} in {raw!r}"
    return total, repair, None


def extract_video_id(link) -> str | None:
    match = VIDEO_ID_RE.search(str(link))
    return match.group(1) if match else None


def clean_title(value) -> str:
    """Collapse the embedded newlines/whitespace 18 of the titles carry."""
    return re.sub(r"\s+", " ", str(value)).strip()


def build_rows(df: pd.DataFrame, label_prefix: str) -> tuple[list[dict], list[dict], list[str]]:
    """Returns (rows, dropped, fatal_errors)."""
    rows: list[dict] = []
    dropped: list[dict] = []
    fatal: list[str] = []
    seen: dict[str, dict] = {}

    for sheet_row, record in df.iterrows():
        excel_row = sheet_row + 2  # +1 for zero-index, +1 for the header line
        link = record["Video Link"]
        repairs: list[str] = []

        video_id = extract_video_id(link)
        if video_id is None:
            fatal.append(f"row {excel_row}: cannot extract a video ID from {link!r}")
            continue

        if video_id in seen:
            dropped.append({
                "excel_row": excel_row,
                "video_id": video_id,
                "title": clean_title(record["Video Title"]),
                "reason": f"duplicate of row {seen[video_id]['excel_row']} ({seen[video_id]['label']})",
            })
            continue

        skip, repair, error = parse_skip_start(record["skip_start_seconds"])
        if error:
            fatal.append(f"row {excel_row} ({video_id}): {error}")
            continue
        if repair:
            repairs.append(repair)

        speech_end, repair, error = parse_clock(record["speech_end_timestamp"], "speech_end")
        if error:
            fatal.append(f"row {excel_row} ({video_id}): {error}")
            continue
        if repair:
            repairs.append(repair)

        # Duration is advisory only: the trim stage measures the real duration with
        # ffprobe, which is authoritative and avoids trusting a stale sheet value.
        duration, repair, error = parse_clock(record["Duration"], "duration")
        if repair:
            repairs.append(repair)
        duration_note = error or ""

        if skip >= speech_end:
            fatal.append(
                f"row {excel_row} ({video_id}): skip_start {skip:.0f}s >= speech_end "
                f"{speech_end:.0f}s -- would yield empty audio"
            )
            continue

        warnings: list[str] = []
        if skip > IMPLAUSIBLE_SKIP_SECONDS:
            warnings.append(f"skip_start {skip:.0f}s is implausibly long -- check the cell")
        if duration is not None and speech_end > duration:
            warnings.append(
                f"speech_end {speech_end:.0f}s exceeds sheet duration {duration:.0f}s "
                f"by {speech_end - duration:.0f}s -- no outro will be cut"
            )
        if duration_note:
            warnings.append(duration_note)

        label = f"{label_prefix}{len(rows) + 1:03d}"
        entry = {
            "label": label,
            "video_id": video_id,
            "audio_filename": f"{label}_{video_id}.mp3",
            "video_link": str(link).strip(),
            "title": clean_title(record["Video Title"]),
            "skip_start_seconds": round(skip, 3),
            "speech_end_seconds": round(speech_end, 3),
            "sheet_duration_seconds": round(duration, 3) if duration is not None else "",
            "speech_seconds": round(speech_end - skip, 3),
            "year": "" if pd.isna(record.get("Year")) else str(int(record["Year"])),
            "assignee": "" if pd.isna(record.get("Assignee")) else str(record["Assignee"]).strip(),
            "excel_row": excel_row,
            "repairs": " | ".join(repairs),
            "warnings": " | ".join(warnings),
        }
        rows.append(entry)
        seen[video_id] = entry

    return rows, dropped, fatal


FIELDNAMES = [
    "label", "video_id", "audio_filename", "video_link", "title",
    "skip_start_seconds", "speech_end_seconds", "sheet_duration_seconds", "speech_seconds",
    "year", "assignee", "excel_row", "repairs", "warnings",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_batch_argument(parser)
    parser.add_argument("--xlsx", default="Whisper_Second_Round_Training_list.xlsx")
    parser.add_argument("--sheet", help="Sheet name; defaults to the batch name titled, e.g. batch3 -> 'Batch 3'")
    parser.add_argument("--out", help="Output CSV (default: data/<batch>/<batch>_validated.csv)")
    parser.add_argument("--label-prefix", help="Stable per-video label prefix; must match ^[A-Za-z]+\\d+$ (default derived from --batch, e.g. batch3 -> B3)")
    parser.add_argument("--dry-run", action="store_true", help="Report only; write nothing")
    args = parser.parse_args()

    paths = BatchPaths(args.batch)
    if args.label_prefix is None:
        try:
            args.label_prefix = label_prefix_for(args.batch)
        except ValueError as exc:
            sys.exit(str(exc))
    if args.out is None:
        args.out = str(paths.validated_csv)
    if args.sheet is None:
        # "batch3" -> "Batch 3", matching how the sheets are actually named.
        digits = "".join(re.findall(r"\d+", args.batch))
        args.sheet = f"{args.batch[:1].upper()}{args.batch[1:len(args.batch) - len(digits)]} {digits}".strip()

    if not re.fullmatch(r"[A-Za-z]+\d*", args.label_prefix):
        sys.exit(
            f"--label-prefix {args.label_prefix!r} would not survive "
            "srt_audio_prep.make_video_id()'s ^[A-Za-z]+\\d+$ check; use letters (optionally "
            "followed by digits), e.g. 'B3'"
        )

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        sys.exit(f"Spreadsheet not found: {xlsx_path}")

    available = pd.ExcelFile(xlsx_path).sheet_names
    if args.sheet not in available:
        sys.exit(
            f"Sheet {args.sheet!r} not found in {xlsx_path.name}.\n"
            f"Available sheets: {available}\n"
            f"(--batch {args.batch} derived the name {args.sheet!r}; pass --sheet to override.)"
        )

    df = pd.read_excel(xlsx_path, sheet_name=args.sheet)
    missing_columns = [c for c in ("Video Link", "Video Title", "skip_start_seconds",
                                   "speech_end_timestamp", "Duration") if c not in df.columns]
    if missing_columns:
        sys.exit(
            f"Sheet {args.sheet!r} is missing required column(s): {missing_columns}\n"
            f"Found: {list(df.columns)}"
        )
    total_sheet_rows = len(df)
    df = df[df["Video Link"].notna()].copy()

    rows, dropped, fatal = build_rows(df, args.label_prefix)

    print("=" * 72)
    print(f"Sheet:          {xlsx_path.name} :: {args.sheet!r}")
    print(f"Rows in sheet:  {total_sheet_rows}  ({len(df)} with a Video Link)")
    print(f"Unique videos:  {len(rows)}")
    print("=" * 72)

    repaired = [r for r in rows if r["repairs"]]
    print(f"\nREPAIRS APPLIED — {len(repaired)} of {len(rows)} rows")
    tally: dict[str, int] = {}
    for row in repaired:
        for repair in row["repairs"].split(" | "):
            kind = repair.split(" (")[0]  # repairs are formatted "<kind> (<detail>)"
            tally[kind] = tally.get(kind, 0) + 1
    for kind, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"   {count:>4}  {kind}")

    if dropped:
        print(f"\nDROPPED — {len(dropped)} duplicate row(s)")
        for entry in dropped:
            print(f"   row {entry['excel_row']:>4}  {entry['video_id']}  {entry['reason']}")
            print(f"             {entry['title'][:70]!r}")

    warned = [r for r in rows if r["warnings"]]
    if warned:
        print(f"\nWARNINGS — {len(warned)} row(s) emitted but worth a look")
        for row in warned:
            print(f"   row {row['excel_row']:>4}  {row['label']}  {row['video_id']}")
            for warning in row["warnings"].split(" | "):
                print(f"             {warning}")

    if fatal:
        print(f"\nFATAL — {len(fatal)} row(s) could not be validated:")
        for message in fatal:
            print(f"   {message}")
        sys.exit(f"\nRefusing to write {args.out}: fix the sheet and re-run.")

    speech_hours = sum(r["speech_seconds"] for r in rows) / 3600
    print(f"\nUsable speech after trim: {speech_hours:.2f} hours across {len(rows)} videos")

    if args.dry_run:
        print("\nDry-run — no file written.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written → {out_path}")


if __name__ == "__main__":
    main()
