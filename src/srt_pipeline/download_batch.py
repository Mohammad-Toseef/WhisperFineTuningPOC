"""
Stage 1 of the batch pipeline: download audio for every video listed in a
validated batch CSV (the output of validate_sheet.py).

Differs from scripts/download_playlist_audio.py, which walks a playlist and
derives filenames from video titles. Here the filename is dictated by the CSV's
`audio_filename` column ("B3001_<youtube_id>.mp3"), because downstream
srt_audio_prep.make_video_id() only reuses a filename's own label when it is a
single token matching ^[A-Za-z]+\\d+$ -- otherwise chunk IDs fall back to
directory-iteration order and shift between runs.

Idempotent: an existing output file means "done", so re-running after a failure
or an interrupted batch only fetches what is missing. The filesystem is the
state; there is no separate state file to drift out of sync with it.

After each download the real duration is probed and compared with the sheet's
Duration. A large mismatch means we fetched the wrong video or got a truncated
file -- worth catching HERE, because stage 2 trims using the sheet's
speech_end_timestamp and would silently cut the wrong region.

Run:
  # verify naming on a few first
  python src/srt_pipeline/download_batch.py --limit 3
  # then the rest
  python src/srt_pipeline/download_batch.py
  # or target specific rows
  python src/srt_pipeline/download_batch.py --only B3001,B3025
"""
import argparse
import csv
import random
import subprocess
import sys
import time
from pathlib import Path

from batch_paths import BatchPaths, add_batch_argument

try:
    import yt_dlp
except ImportError:
    sys.exit("yt-dlp is not installed. Run: pip install yt-dlp")

sys.stdout.reconfigure(encoding="utf-8")

# Beyond this, a downloaded file's length disagrees with the sheet enough that the
# trim points are suspect (wrong video, re-uploaded cut, or a truncated download).
DURATION_TOLERANCE_SECONDS = 30.0


def probe_duration(path: str) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def load_rows(csv_path: Path, limit: int | None, only: str | None) -> list[dict]:
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if only:
        wanted = {token.strip() for token in only.split(",") if token.strip()}
        unknown = wanted - {row["label"] for row in rows}
        if unknown:
            sys.exit(f"--only referenced labels not in {csv_path}: {sorted(unknown)}")
        rows = [row for row in rows if row["label"] in wanted]
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_auth_opts(cookies_from_browser: str | None, cookies_file: str | None) -> dict:
    """YouTube increasingly answers datacenter/flagged IPs with 'Sign in to confirm
    you're not a bot'. Supplying the cookies of a signed-in session is yt-dlp's
    documented remedy; without it a whole batch can fail on extraction alone."""
    opts: dict = {}
    if cookies_from_browser:
        browser, _, profile = cookies_from_browser.partition(":")
        opts["cookiesfrombrowser"] = (browser, profile or None, None, None)
    if cookies_file:
        opts["cookiefile"] = cookies_file
    return opts


def download_one(row: dict, out_dir: Path, audio_format: str, quality: str, extra_opts: dict) -> tuple[bool, str]:
    """Returns (ok, message). Raises nothing -- a failed video must not kill the batch."""
    stem = Path(row["audio_filename"]).stem
    # Fetch by canonical single-video URL built from the validated video_id, NOT the
    # sheet's raw link: 8 of the Batch 3 links are "watch?v=<id>&list=<playlist>&index=N",
    # and yt-dlp expands those to the WHOLE playlist. Every entry then renders to this
    # same outtmpl, so each download silently overwrites the last and the surviving file
    # is some arbitrary other video. Dropping &list=/&index=/?si= also avoids the
    # tracking params entirely.
    video_url = f"https://www.youtube.com/watch?v={row['video_id']}"

    # Drop any stale partial for this stem before starting. yt-dlp RESUMES from a
    # .part file, so a partial left behind by an earlier bad run (e.g. the playlist
    # expansion above, which wrote other videos' bytes under this same name) would be
    # resumed into a corrupt file that looks complete.
    for partial in out_dir.glob(f"{stem}.*.part"):
        print(f"   discarding stale partial: {partial.name}")
        partial.unlink()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / f"{stem}.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": audio_format,
            "preferredquality": quality,
        }],
        "quiet": True,
        "no_warnings": True,
        # Belt-and-braces with the canonical URL above: refuse playlist expansion even
        # if a list parameter ever reaches yt-dlp.
        "noplaylist": True,
        # Deliberately NOT ignoreerrors: we want the failure surfaced per-row so the
        # summary can list exactly what still needs fetching.
        "retries": 5,
        "fragment_retries": 5,
        # NO js_runtimes key on purpose. YouTube needs a JavaScript runtime to solve its
        # signature challenge, and yt-dlp's default is exactly what we want: {"deno": {}}.
        # Setting this key REPLACES that default rather than adding to it, so
        # {"node": {}} would silently DISABLE deno -- which is how this was first written,
        # against a node too old to be accepted, leaving no usable runtime at all.
        #
        # Two host requirements this depends on, neither enforced by pip:
        #   * deno >= 2.3.0 on PATH (yt-dlp also accepts node >= 22, bun >= 1.2.11,
        #     quickjs >= 2023-12-09; anything older is reported "unsupported")
        #   * the yt-dlp-ejs package, which carries the solver scripts -- a runtime alone
        #     is NOT enough, and without it every provider reads "unavailable"
        # Missing either one is not fatal: extraction falls back to JS-less clients, where
        # "some formats may be missing" and the path is deprecated. That is the state that
        # preceded the intermittent HTTP 403 costing 2 retries on B3003, so treat the
        # "No supported JavaScript runtime could be found" warning as a real finding.
        **extra_opts,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as exc:  # yt-dlp raises a wide variety of network/extractor errors
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"

    target = out_dir / row["audio_filename"]
    if not target.exists():
        return False, f"yt-dlp reported success but {target.name} is missing"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_batch_argument(parser)
    parser.add_argument("--csv", help="Validated CSV (default: data/<batch>/<batch>_validated.csv)")
    parser.add_argument("--out-dir", help="Where to write audio (default: data/<batch>/audio_raw)")
    parser.add_argument("--format", default="mp3", choices=["mp3", "m4a", "wav", "opus", "flac"])
    parser.add_argument("--quality", default="192", help="Audio bitrate in kbps (default: 192)")
    parser.add_argument("--limit", type=int, help="Only process the first N rows -- use to verify naming before the full run")
    parser.add_argument("--only", help="Comma-separated labels to process, e.g. B3001,B3025")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched; download nothing")
    parser.add_argument(
        "--cookies-from-browser",
        help="Read YouTube cookies from a signed-in browser, e.g. 'chrome', 'edge', 'firefox' "
             "(optionally 'chrome:ProfileName'). Needed when YouTube answers with "
             "'Sign in to confirm you're not a bot'.",
    )
    parser.add_argument("--cookies", help="Path to a Netscape-format cookies.txt, as an alternative to --cookies-from-browser")
    # RANDOMISED, not fixed. A constant gap is itself a fingerprint -- requests arriving
    # exactly N seconds apart for 90 videos look like automation no human produces, and a
    # fixed 3s was not enough on its own: B3018 came back HTTP 403 mid-run and only
    # succeeded on a retry with a longer wait. Jitter costs ~10s per video (~16 min across
    # the remaining 76) and is far cheaper than a throttled batch.
    parser.add_argument(
        "--sleep-min", type=float, default=5.0,
        help="Lower bound of the random wait between downloads, in seconds (default: 5).",
    )
    parser.add_argument(
        "--sleep-max", type=float, default=15.0,
        help="Upper bound of the random wait between downloads, in seconds (default: 15). "
             "Pass --sleep-min 0 --sleep-max 0 to disable waiting entirely.",
    )
    args = parser.parse_args()
    if args.sleep_max < args.sleep_min:
        parser.error(f"--sleep-max ({args.sleep_max}) is below --sleep-min ({args.sleep_min})")

    paths = BatchPaths(args.batch)
    csv_path = Path(args.csv) if args.csv else paths.validated_csv
    if not csv_path.exists():
        sys.exit(f"Validated CSV not found: {csv_path}\n"
                 f"Run: python src/srt_pipeline/validate_sheet.py --batch {args.batch}")

    rows = load_rows(csv_path, args.limit, args.only)
    out_dir = Path(args.out_dir) if args.out_dir else paths.audio_raw

    pending, already = [], []
    for row in rows:
        (already if (out_dir / row["audio_filename"]).exists() else pending).append(row)

    print("=" * 72)
    print(f"Source CSV:   {csv_path}")
    print(f"Output dir:   {out_dir}")
    print(f"Selected:     {len(rows)} row(s)   already downloaded: {len(already)}   to fetch: {len(pending)}")
    print("=" * 72)

    if args.dry_run:
        for row in pending:
            print(f"   would fetch  {row['audio_filename']:<32} {row['title'][:52]}")
        for row in already:
            print(f"   skip (exists) {row['audio_filename']}")
        print("\nDry-run — nothing downloaded.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    auth_opts = build_auth_opts(args.cookies_from_browser, args.cookies)
    if not auth_opts:
        print("NOTE: no cookies supplied. If YouTube answers 'Sign in to confirm you're not")
        print("      a bot', re-run with --cookies-from-browser chrome (or edge/firefox).\n")
    ok, failed, mismatched = [], [], []

    for index, row in enumerate(pending, start=1):
        if index > 1 and args.sleep_max > 0:
            delay = random.uniform(args.sleep_min, args.sleep_max)
            # Printed, not silent: a run that stalls should be distinguishable from a run
            # that is merely waiting, and the log is the only record of a 90-video batch.
            print(f"   waiting {delay:.1f}s before the next download")
            time.sleep(delay)
        print(f"\n[{index}/{len(pending)}] {row['label']}  {row['video_id']}")
        print(f"   {row['title'][:66]}")
        success, message = download_one(row, out_dir, args.format, args.quality, auth_opts)
        if not success:
            print(f"   FAILED: {message}")
            failed.append((row, message))
            continue

        target = out_dir / row["audio_filename"]
        actual = probe_duration(str(target))
        size_mb = target.stat().st_size / 1024 / 1024
        expected = row["sheet_duration_seconds"]
        note = ""
        if actual is not None and expected:
            delta = actual - float(expected)
            if abs(delta) > DURATION_TOLERANCE_SECONDS:
                note = f"  ⚠ sheet says {float(expected):.0f}s, file is {actual:.0f}s (off by {delta:+.0f}s)"
                mismatched.append((row, actual, float(expected)))
        print(f"   OK  {size_mb:.1f} MB  {actual:.0f}s{note}" if actual else f"   OK  {size_mb:.1f} MB")
        ok.append(row)

    print("\n" + "=" * 72)
    print(f"Downloaded: {len(ok)}    Failed: {len(failed)}    Already present: {len(already)}")
    if mismatched:
        print(f"\nDURATION MISMATCH — {len(mismatched)} file(s); trim points may be wrong:")
        for row, actual, expected in mismatched:
            print(f"   {row['label']} {row['video_id']}  sheet={expected:.0f}s file={actual:.0f}s  (xlsx row {row['excel_row']})")
    if failed:
        print(f"\nFAILED — {len(failed)}; re-run to retry just these:")
        for row, message in failed:
            print(f"   {row['label']} {row['video_id']}  {message}")
        sys.exit(1)
    print("=" * 72)


if __name__ == "__main__":
    main()
