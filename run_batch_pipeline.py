#!/usr/bin/env python3
"""
One command to drive the whole batch pipeline, so routine runs need no hand-holding.

    # see what is done and what is pending
    python run_batch_pipeline.py --batch batch3 status

    # process a range
    python run_batch_pipeline.py --batch batch3 run --from B3003 --to B3010

    # or just take the next N pending videos
    python run_batch_pipeline.py --batch batch3 run --next 8

    # print the commands without executing them
    python run_batch_pipeline.py --batch batch3 run --next 8 --dry-run

Stages, in order:
    0  validate_sheet.py            xlsx  -> data/<batch>/<batch>_validated.csv
    1  download_batch.py            links -> data/<batch>/audio_raw/
    2  batch_clean_intro_music.py   trim  -> data/<batch>/audio_trimmed/
    3+4 modal_align.py              GPU   -> data/<batch>/timestamped_srts/ + asr_transcripts/
    5  batch_srt_prep.py            chunk -> data/processed/<Batch>/{audio/,manifest.json}
    6  normalize_manifest.py              -> data/processed/<Batch>/manifest_normalized.json

Every stage is idempotent -- work already done is skipped -- so an interrupted run is
resumed simply by re-running the same command. That is also why a mid-run network drop is
not costly: nothing is lost, just repeat.

Stages 5 and 6 are deliberately NOT restricted to the selected videos. Stage 5 picks up any
audio+SRT pair not yet in the manifest and merges it in, and stage 6 normalises the whole
manifest; both are naturally incremental, and running them over everything keeps the single
combined manifest consistent with whatever is on disk.

After stage 6 the manifest goes to the review team (stage 7), and their export comes back
through scripts/convert_reviewed_manifest.py --batch-folder <Batch> (stage 8). Those two
steps involve people, so they are outside this driver.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src" / "srt_pipeline"))
from batch_paths import BatchPaths, label_prefix_for  # noqa: E402

REPO = Path(__file__).parent
STAGE_NAMES = {
    "0": "validate sheet",
    "1": "download audio",
    "2": "trim intro/outro",
    "34": "transcribe + align (GPU)",
    "5": "chunk + manifest",
    "6": "normalize manifest",
    "qa": "quality gate",
}
# 'qa' runs last and is the only stage that can fail on OUTPUT quality rather than on a
# crash. Every silent defect this pipeline has produced -- dropped words, deleted windows,
# discarded speech, cues at 30x human speaking rate -- passed all structural checks, so
# without this the driver reports success on unusable data.
ALL_STAGES = ["0", "1", "2", "34", "5", "6", "qa"]


def modal_executable() -> str:
    """Prefer the modal CLI beside the running interpreter, so a venv is respected."""
    for name in ("modal.exe", "modal"):
        candidate = Path(sys.executable).parent / name
        if candidate.exists():
            return str(candidate)
    return "modal"


def child_environment() -> dict:
    """Force UTF-8 on child stdio.

    Every stage prints non-ASCII -- Urdu transcript previews, ✓/⚠ markers, and Modal's own
    CLI output. On Windows a child's stdout defaults to the ANSI codepage (cp1252), which
    cannot encode any of it, so the stage dies with 'charmap' codec can't encode character
    '\\u2713'. Observed for real: stage 3+4 crashed on Modal printing a single ✓ *after*
    the GPU work had been dispatched, turning a cosmetic issue into a failed stage.
    """
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    return environment


def run(command: list[str], dry_run: bool, cwd: Path = REPO) -> int:
    printable = " ".join(f'"{c}"' if " " in c else c for c in command)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=str(cwd), env=child_environment()).returncode


# ── status ────────────────────────────────────────────────────────────────────────

def manifest_video_ids(paths: BatchPaths) -> set[str]:
    manifest = paths.processed_dir / "manifest.json"
    if not manifest.exists():
        return set()
    with manifest.open(encoding="utf-8") as handle:
        return {Path(entry["audio_path"]).parent.name for entry in json.load(handle)}


def build_status(paths: BatchPaths) -> list[dict]:
    """Per-video state across the stages, derived from the filesystem + manifest.

    The filesystem is the source of truth rather than a state file, which cannot drift out
    of sync with what actually exists on disk.
    """
    csv_path = paths.validated_csv
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    in_manifest = manifest_video_ids(paths)
    status = []
    for row in rows:
        stem = Path(row["audio_filename"]).stem
        status.append({
            "label": row["label"],
            "video_id": row["video_id"],
            "title": row["title"],
            "downloaded": (paths.audio_raw / row["audio_filename"]).exists(),
            "trimmed": (paths.audio_trimmed / row["audio_filename"]).exists(),
            "srt": (paths.srt_dir / f"{stem}.srt").exists(),
            "chunked": stem in in_manifest,
        })
    return status


STEPS = ("downloaded", "trimmed", "srt", "chunked")


def is_complete(entry: dict) -> bool:
    return all(entry[step] for step in STEPS)


def print_status(status: list[dict], verbose: bool) -> None:
    if not status:
        print("No validated CSV yet -- run stage 0 first (or just `run`, which does it).")
        return
    totals = {step: sum(1 for e in status if e[step]) for step in STEPS}
    complete = [e for e in status if is_complete(e)]
    print(f"{'videos':<22} {len(status)}")
    for step in STEPS:
        print(f"{step:<22} {totals[step]:>3} / {len(status)}")
    print(f"{'fully complete':<22} {len(complete):>3} / {len(status)}")

    partial = [e for e in status if not is_complete(e) and any(e[step] for step in STEPS)]
    if partial:
        print(f"\nIn progress ({len(partial)}):")
        for entry in partial:
            done = " ".join(step if entry[step] else "-" * len(step) for step in STEPS)
            print(f"   {entry['label']}  {done}   {entry['title'][:44]}")
    pending = [e for e in status if not any(e[step] for step in STEPS)]
    print(f"\nNot started: {len(pending)}"
          + (f"  (next: {', '.join(e['label'] for e in pending[:8])}"
             + (" ..." if len(pending) > 8 else "") + ")" if pending else ""))
    if verbose:
        print("\nAll videos:")
        for entry in status:
            done = " ".join(step if entry[step] else "-" * len(step) for step in STEPS)
            print(f"   {entry['label']}  {done}   {entry['title'][:44]}")


# ── selection ─────────────────────────────────────────────────────────────────────

def normalise_label(token: str, prefix: str) -> str:
    """Accept 'B3003', 'b3003' or bare '3' and return the canonical label."""
    token = token.strip()
    if not token:
        raise ValueError("empty label")
    if re.fullmatch(r"\d+", token):
        return f"{prefix}{int(token):03d}"
    if token.upper().startswith(prefix.upper()):
        return f"{prefix}{int(token[len(prefix):]):03d}"
    raise ValueError(f"cannot read {token!r} as a label for prefix {prefix!r}")


def select_labels(status: list[dict], args, prefix: str) -> list[str]:
    if args.only:
        wanted = [normalise_label(t, prefix) for t in args.only.split(",") if t.strip()]
        known = {e["label"] for e in status}
        unknown = [label for label in wanted if label not in known]
        if unknown:
            sys.exit(f"--only refers to labels not in the validated CSV: {unknown}")
        return wanted

    if args.from_label or args.to_label:
        start = normalise_label(args.from_label, prefix) if args.from_label else status[0]["label"]
        end = normalise_label(args.to_label, prefix) if args.to_label else status[-1]["label"]
        if start > end:
            sys.exit(f"--from {start} is after --to {end}")
        selected = [e["label"] for e in status if start <= e["label"] <= end]
        if not selected:
            sys.exit(f"no videos in the range {start}..{end}")
        return selected

    pending = [e["label"] for e in status if not is_complete(e)]
    if args.next:
        return pending[:args.next]
    return pending  # default: everything outstanding


# ── run ───────────────────────────────────────────────────────────────────────────

def do_run(args) -> int:
    paths = BatchPaths(args.batch)
    prefix = label_prefix_for(args.batch)
    stages = args.stages.split(",") if args.stages else ALL_STAGES
    unknown = [s for s in stages if s not in STAGE_NAMES]
    if unknown:
        sys.exit(f"unknown stage(s) {unknown}; valid: {', '.join(ALL_STAGES)}")

    # Stage 0 must happen before anything can be selected, since selection reads the CSV.
    if "0" in stages and (not paths.validated_csv.exists() or args.revalidate):
        print(f"\n{'=' * 78}\nSTAGE 0 — {STAGE_NAMES['0']}\n{'=' * 78}")
        command = [sys.executable, "src/srt_pipeline/validate_sheet.py", "--batch", args.batch]
        if args.sheet:
            command += ["--sheet", args.sheet]
        if args.xlsx:
            command += ["--xlsx", args.xlsx]
        if run(command, args.dry_run) != 0:
            return 1
    elif "0" in stages:
        print(f"\nSTAGE 0 — {STAGE_NAMES['0']}: {paths.validated_csv} exists "
              f"(pass --revalidate to rebuild it)")

    status = build_status(paths)
    if not status:
        if args.dry_run:
            print("\n(dry run: no validated CSV yet, so no videos can be selected)")
            return 0
        sys.exit(f"No validated CSV at {paths.validated_csv} -- stage 0 must run first.")

    labels = select_labels(status, args, prefix)
    if not labels:
        print("\nNothing to do: every video is already complete.")
        return 0
    only = ",".join(labels)
    print(f"\nSelected {len(labels)} video(s): {labels[0]}"
          + (f" .. {labels[-1]}" if len(labels) > 1 else ""))

    commands = {
        "1": [sys.executable, "src/srt_pipeline/download_batch.py",
              "--batch", args.batch, "--only", only, "--sleep", str(args.sleep)]
             + (["--cookies-from-browser", args.cookies_from_browser] if args.cookies_from_browser else [])
             + (["--cookies", args.cookies] if args.cookies else []),
        "2": [sys.executable, "src/srt_pipeline/batch_clean_intro_music.py",
              "--batch", args.batch, "--only", only],
        # modal run passes entrypoint args after the function reference.
        #
        # --detach protects a long GPU stage from a local network drop. It does NOT make
        # the command return early -- the local entrypoint still runs to completion here,
        # so stages 5-6 cannot start before the SRTs exist. What it changes is the
        # disconnect case: without it Modal stops the app and the GPU work is lost;
        # with it the container runs on and commits its result to the volume, and the
        # next run recovers that for free (transcribe_align checks the volume first).
        "34": [modal_executable(), "run", "--detach", "modal_align.py::transcribe_align",
               "--batch", args.batch, "--only", only, "--windows", args.windows]
              + (["--no-fetch"] if args.no_fetch else [])
              + (["--model-path", args.model_path] if args.model_path else []),
        # Stages 5-6 are batch-wide on purpose: both are incremental, and running them over
        # everything keeps the single combined manifest consistent with what is on disk.
        "5": [sys.executable, "src/batch_srt_prep.py", "--batch", args.batch],
        "6": [sys.executable, "src/normalize_manifest.py",
              "--manifest", str(paths.processed_dir / "manifest.json")],
        # Batch-wide like 5-6: the gate is cheap and a regression in an earlier episode
        # matters just as much as one in the episode just processed.
        "qa": [sys.executable, "src/srt_pipeline/qa_gate.py", "--batch", args.batch]
              + (["--warn-only"] if args.qa_warn_only else []),
    }

    for stage in stages:
        if stage == "0":
            continue
        print(f"\n{'=' * 78}\nSTAGE {stage} — {STAGE_NAMES[stage]}\n{'=' * 78}")
        if stage == "6" and not (paths.processed_dir / "manifest.json").exists():
            print("no manifest.json yet -- skipping (stage 5 must succeed first)")
            continue
        code = run(commands[stage], args.dry_run)
        if code != 0:
            print(f"\nSTAGE {stage} ({STAGE_NAMES[stage]}) exited {code} — stopping here.")
            print("Every stage is idempotent: fix the cause and re-run the same command; "
                  "completed work is skipped.")
            return code

    if not args.dry_run:
        print(f"\n{'=' * 78}\nFINAL STATUS\n{'=' * 78}")
        print_status(build_status(paths), verbose=False)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", default="batch3", help="Batch name (default: batch3)")
    sub = parser.add_subparsers(dest="command", required=True)

    status_parser = sub.add_parser("status", help="Show per-video progress and exit")
    status_parser.add_argument("--all", action="store_true", help="List every video, not just in-progress ones")

    run_parser = sub.add_parser("run", help="Run the pipeline over a selection of videos")
    group = run_parser.add_mutually_exclusive_group()
    group.add_argument("--only", help="Explicit labels, e.g. B3003,B3007 (bare numbers accepted)")
    group.add_argument("--next", type=int, metavar="N", help="Take the next N incomplete videos")
    run_parser.add_argument("--from", dest="from_label", metavar="LABEL", help="Range start, e.g. B3003 or 3")
    run_parser.add_argument("--to", dest="to_label", metavar="LABEL", help="Range end, e.g. B3010 or 10")
    run_parser.add_argument("--stages", help=f"Comma-separated subset of {','.join(ALL_STAGES)} (default: all)")
    run_parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    run_parser.add_argument("--revalidate", action="store_true", help="Re-run stage 0 even if the CSV exists")
    run_parser.add_argument("--sheet", help="Override the sheet name passed to stage 0")
    run_parser.add_argument("--xlsx", help="Override the spreadsheet passed to stage 0")
    run_parser.add_argument("--model-path", help="Fine-tuned model for stage 3+4 (default: the volume's whisper-urdu-final)")
    run_parser.add_argument("--windows", choices=["chunks", "vad"], default="vad",
                            help="How stage 3+4 draws segment windows: 'vad' = cut at detected "
                                 "silence (default), 'chunks' = the old fixed 28s grid. See modal_align.py.")
    run_parser.add_argument("--qa-warn-only", action="store_true",
                            help="Let the quality gate report failures without stopping the run")
    run_parser.add_argument("--no-fetch", action="store_true",
                            help="Force stage 3+4 to recompute instead of reusing a result "
                                 "committed to the volume. The cache is keyed by (batch, window "
                                 "mode) and is blind to the alignment code, so pass this after "
                                 "changing align_to_srt.py or the stale output comes back.")
    run_parser.add_argument("--sleep", type=float, default=3.0, help="Seconds between downloads (default: 3)")
    run_parser.add_argument("--cookies-from-browser", help="Pass through to stage 1 if YouTube demands sign-in")
    run_parser.add_argument("--cookies", help="Pass a cookies.txt through to stage 1")

    args = parser.parse_args()
    if args.command == "status":
        print_status(build_status(BatchPaths(args.batch)), verbose=args.all)
        return
    sys.exit(do_run(args))


if __name__ == "__main__":
    main()
