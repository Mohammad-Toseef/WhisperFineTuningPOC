#!/usr/bin/env python3
"""Quality gate for a batch's SRTs and chunks — fails the run when timings are wrong.

WHY THIS EXISTS
---------------
Every serious defect found in this pipeline so far was SILENT. None raised an error:

  * flatten_words dropped 5% of an episode's words -- the SRT still had contiguous
    indices, monotonic timings, no overlaps, and ended exactly at the audio duration.
  * a 180-word window was deleted by a 500-character cap, as one line in a 200-line log.
  * 351s of confirmed speech (VAD-verified) was discarded from a 72-minute episode.
  * 52 words were stranded on a 0.62s cue -- a perfectly well-formed SRT entry.

Structural validity is not quality. What exposes these is comparing each cue's
WORDS-PER-SECOND against the episode's own median, plus tracking how much audio and how
many words survive each stage. This module makes those checks a gate rather than
something a person has to remember to run.

Usage:
    python src/srt_pipeline/qa_gate.py --batch batch3
    python src/srt_pipeline/qa_gate.py --batch batch3 --only B3001,B3003
    python src/srt_pipeline/qa_gate.py --batch batch3 --warn-only --json qa.json
"""
import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # src/  -> srt_audio_prep
sys.path.insert(0, str(Path(__file__).resolve().parent))              # srt_pipeline/ -> batch_paths

from batch_paths import BatchPaths, add_batch_argument  # noqa: E402
from srt_audio_prep import GAP_THRESHOLD, parse_srt     # noqa: E402
from repetition import find_repetition, load_chant_units  # noqa: E402

# ── thresholds ───────────────────────────────────────────────────────────────────────
# Deliberately set where a HUMAN would call the output unusable, not at the best number yet
# seen. The percentages quoted per limit below are from the PRE-VAD chunks runs of
# 2026-08-07 -- that is what the thresholds had to catch. The current VAD + interpolation
# output sits far inside all of them (0.0% word loss, 97.5-99.2% coverage), so those figures
# are the calibration record, not a description of today's numbers.
DEFAULTS = {
    # Fraction of the trimmed audio covered by manifest chunks. B3001 99.4% / B3003 92.2%.
    "min_coverage": 95.0,
    # Words surviving asr_transcripts/*.txt -> manifest. B3001 1.4% / B3002 6.9%.
    "max_word_loss": 3.0,
    # Seconds of SPEECH sitting in >GAP_THRESHOLD holes, as a share of episode duration.
    # Those holes are DELETED by stage 5, so speech inside one is data loss.
    #
    # Scored against VAD's detected speech turns, NOT against raw uncovered time. Measured
    # 2026-08-08 on all three episodes: of the time the holes span, VAD calls 65% (B3001) /
    # 52% (B3002) / 75% (B3003) SILENCE -- pauses the speaker took. Charging those made the
    # metric ~2x too harsh and was the sole reason B3001 and B3002 failed the gate while
    # their actual loss was 1.06% and 1.35%. Raw uncovered time is still reported as
    # gap_seconds; it is just not what the gate rules on.
    #
    # 2.0% is where a human would object to losing that much of an episode, not the best
    # number seen -- the three measured episodes sit at 0.22-1.35%, so there is headroom,
    # but a real regression trips it.
    "max_speech_loss": 2.0,
    # No cue may exceed this rate. Urdu here runs at a ~2.1-2.8 w/s median; 8 w/s is
    # roughly 3x the median and far past what a person can say.
    "max_words_per_second": 8.0,
    # A rate is only meaningful with enough words over enough time. Without BOTH guards a
    # legitimate one-word cue in a 0.04s sliver reports 25 w/s and fails the batch --
    # observed on B3001 cue 81, which is why this is two conditions and not one.
    "rate_min_words": 3,
    "rate_min_duration": 0.5,
    # Chunks in the manifest containing a repeated-word run. Stage 5 drops these at cue
    # level, so the only tolerable number is ZERO -- anything above means the filter did not
    # run (old manifest), was bypassed, or a run slipped its thresholds. This is the check
    # that would have caught B3014 shipping 127 words of fabricated Urdu while passing every
    # other gate.
    "max_repetition_chunks": 0,
}


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def manifest_by_video(processed_dir: Path) -> dict[str, list[dict]]:
    manifest = processed_dir / "manifest.json"
    if not manifest.exists():
        return {}
    grouped: dict[str, list[dict]] = {}
    for entry in json.loads(manifest.read_text(encoding="utf-8")):
        grouped.setdefault(Path(entry["audio_path"]).parent.name, []).append(entry)
    return grouped


class Layout:
    """Where to read each input from. Defaults derive from --batch; every one is
    overridable so an A/B run (e.g. timestamped_srts_vad vs timestamped_srts) can be
    scored with the identical thresholds instead of a hand-written comparison script."""

    def __init__(self, paths: BatchPaths, srt_dir: str = "", transcript_dir: str = "",
                 audio_dir: str = "", processed_dir: str = "", vad_dir: str = ""):
        self.srt_dir = Path(srt_dir) if srt_dir else paths.srt_dir
        self.transcript_dir = Path(transcript_dir) if transcript_dir else paths.transcript_dir
        self.audio_trimmed = Path(audio_dir) if audio_dir else paths.audio_trimmed
        self.processed_dir = Path(processed_dir) if processed_dir else paths.processed_dir
        self.vad_dir = Path(vad_dir) if vad_dir else paths.vad_dir


def load_speech_turns(stem: str, vad_dir: Path) -> list[tuple[float, float]] | None:
    """Detected speech turns for one episode, or None when the run did not record any.

    Written by modal_align.py's vad window path (see BatchPaths.vad_dir). None is the
    expected state for a --windows chunks run and for anything aligned before this existed;
    the caller must degrade to raw uncovered time and SAY SO rather than score a stricter
    number silently.
    """
    path = vad_dir / f"{stem}.vad.json"
    if not path.exists():
        return None
    turns = json.loads(path.read_text(encoding="utf-8")).get("turns") or []
    return sorted((float(t["start"]), float(t["end"])) for t in turns)


def load_exclusions(processed_dir: Path) -> dict:
    """Stage 5's repetition ledger, keyed by video_id (== the SRT stem). {} when absent."""
    path = processed_dir / "repetition_exclusions.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("episodes", {}) if isinstance(payload, dict) else {}


def subtract_spans(intervals: list[tuple[float, float]],
                   removed: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """`intervals` minus `removed`, both sorted by start.

    Used to take deliberately-excluded repetition out of VAD's speech turns before charging
    anything as loss. Those seconds ARE speech -- 526s of real zikr in B3015 -- so without
    this the gate correctly detects speech in the holes and wrongly calls a policy decision
    data loss, failing an episode for doing exactly what it was told.
    """
    result = []
    for start, end in intervals:
        pieces = [(start, end)]
        for cut_start, cut_end in removed:
            if cut_end <= start or cut_start >= end:
                continue
            next_pieces = []
            for piece_start, piece_end in pieces:
                if cut_start > piece_start:
                    next_pieces.append((piece_start, min(piece_end, cut_start)))
                if cut_end < piece_end:
                    next_pieces.append((max(piece_start, cut_end), piece_end))
            pieces = [(a, b) for a, b in next_pieces if b > a]
        result.extend(pieces)
    return sorted(result)


def overlap_seconds(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    """Seconds of [start, end) covered by `intervals`, which must be sorted by start."""
    total = 0.0
    for interval_start, interval_end in intervals:
        if interval_end <= start:
            continue
        if interval_start >= end:
            break
        total += min(end, interval_end) - max(start, interval_start)
    return total


def measure(stem: str, paths: Layout, chunks: list[dict], limits: dict,
            excluded: dict | None = None, chant_units: set[str] | None = None) -> dict:
    """All metrics for one episode. Missing inputs are reported, never guessed at."""
    srt_path = paths.srt_dir / f"{stem}.srt"
    txt_path = paths.transcript_dir / f"{stem}.txt"
    audio_path = paths.audio_trimmed / f"{stem}.mp3"

    report: dict = {"label": stem.split("_")[0], "stem": stem, "failures": [], "notes": []}
    if not srt_path.exists():
        report["failures"].append("no SRT")
        return report

    cues = parse_srt(str(srt_path))
    if not cues:
        report["failures"].append("SRT parsed to 0 cues")
        return report

    duration = probe_duration(audio_path) if audio_path.exists() else 0.0
    report["duration"] = duration
    report["cues"] = len(cues)

    # ── repetition deliberately excluded by stage 5 ───────────────────────────
    # Everything below measures against a baseline with these REMOVED. They are not loss:
    # they are audio and words we chose not to train on, and charging them would make the
    # coverage, word-loss and speech-loss checks all fail an episode that is correct.
    excluded = excluded or {}
    excluded_spans = sorted((float(a), float(b)) for a, b in excluded.get("spans", []))
    excluded_seconds = float(excluded.get("seconds_excluded", 0.0))
    excluded_words = int(excluded.get("words_excluded", 0))
    if excluded:
        report["excluded_cues"] = excluded.get("cues_excluded", 0)
        report["excluded_seconds"] = excluded_seconds
        report["excluded_words"] = excluded_words
        report["excluded_by_kind"] = excluded.get("by_kind", {})
    expected_duration = max(0.0, duration - excluded_seconds)

    # ── words through the stages ──────────────────────────────────────────────
    srt_words = sum(len(text.split()) for _, _, text in cues)
    report["srt_words"] = srt_words
    if txt_path.exists():
        asr_words = len(txt_path.read_text(encoding="utf-8").split())
        report["asr_words"] = asr_words
        if chunks:
            manifest_words = sum(len(c["transcript"].split()) for c in chunks)
            report["manifest_words"] = manifest_words
            expected_words = max(0, asr_words - excluded_words)
            loss = ((expected_words - manifest_words) / expected_words * 100
                    if expected_words else 0.0)
            report["word_loss_pct"] = loss
            if loss > limits["max_word_loss"]:
                detail = (f"asr {asr_words} - {excluded_words} repetition = {expected_words} "
                          f"-> manifest {manifest_words}" if excluded_words
                          else f"asr {asr_words} -> manifest {manifest_words}")
                report["failures"].append(
                    f"word loss {loss:.1f}% > {limits['max_word_loss']}% ({detail})")
    else:
        report["notes"].append("no asr_transcripts/*.txt — word loss unmeasurable")

    # ── coverage: how much audio actually reaches training ────────────────────
    if chunks and duration:
        covered = sum(c["duration"] for c in chunks)
        report["chunk_seconds"] = covered
        # Denominator is the audio we INTENDED to keep. Scoring B3015's 13% of excluded zikr
        # against the raw duration reports 85% coverage on a correct build.
        report["coverage_pct"] = covered / expected_duration * 100 if expected_duration else 0.0
        if excluded_seconds:
            report["raw_coverage_pct"] = covered / duration * 100
        if report["coverage_pct"] < limits["min_coverage"]:
            basis = (f" (of {expected_duration:.0f}s kept, after {excluded_seconds:.0f}s "
                     f"repetition excluded)" if excluded_seconds else "")
            report["failures"].append(
                f"coverage {report['coverage_pct']:.1f}% < {limits['min_coverage']}%{basis}")
    elif not chunks:
        report["notes"].append("not in manifest — coverage and word loss unmeasurable")

    # ── gaps: audio stage 5 will DELETE ───────────────────────────────────────
    # Stage 5 discards every hole wider than GAP_THRESHOLD. That is only a LOSS where the
    # hole contains speech -- a pause the speaker took is correctly dropped. So the gate
    # rules on speech seconds scored against VAD, and falls back to raw uncovered time only
    # when no turns were recorded (with a note, because that fallback is much harsher).
    holes = [(cues[i][1], cues[i + 1][0]) for i in range(len(cues) - 1)
             if cues[i + 1][0] - cues[i][1] > GAP_THRESHOLD]
    report["gaps_over_threshold"] = len(holes)
    report["gap_seconds"] = sum(end - start for start, end in holes)
    report["max_gap"] = max((cues[i + 1][0] - cues[i][1] for i in range(len(cues) - 1)),
                            default=0.0)

    turns = load_speech_turns(stem, paths.vad_dir)
    if turns is not None:
        # Speech inside an excluded repetition span is speech we chose to drop, so it is not
        # chargeable. Today these spans are covered by SRT cues and therefore rarely fall in
        # a hole -- this keeps the two accounts consistent anyway, so a later change that
        # filters the SRT itself cannot silently start charging them.
        chargeable = subtract_spans(turns, excluded_spans) if excluded_spans else turns
        lost = sum(overlap_seconds(start, end, chargeable) for start, end in holes)
        report["gap_speech_seconds"] = lost
        report["loss_basis"] = "vad"
    else:
        lost = report["gap_seconds"]
        report["loss_basis"] = "uncovered"
        report["notes"].append(
            f"no {stem}.vad.json — every uncovered second is charged as loss, which "
            f"measured ~2x too harsh on VAD-scored episodes. Re-run stage 34 with "
            f"--windows vad to record speech turns.")

    if duration:
        report["gap_share_pct"] = report["gap_seconds"] / duration * 100
        share = lost / duration * 100
        report["speech_loss_pct"] = share
        if share > limits["max_speech_loss"]:
            detail = (f"{lost:.0f}s of SPEECH" if turns is not None
                      else f"{lost:.0f}s of uncovered audio (VAD-unscored)")
            report["failures"].append(
                f"{detail} in {len(holes)} gaps >{GAP_THRESHOLD}s = {share:.1f}% "
                f"of the episode > {limits['max_speech_loss']}% (this audio is discarded)")

    # ── speaking rate: the only signal that catches a misplaced window ────────
    # Repetition cues are skipped: they are not in the manifest, and a chanted اللہ x178
    # legitimately runs at 10 w/s. Scoring them made B3015 fail on 59 "over-rate" cues that
    # describe removed content -- a gate reporting a policy decision as a timing defect,
    # which is how thresholds get quietly loosened until real regressions stop tripping.
    def excluded_cue(start: float, end: float) -> bool:
        span = end - start
        return span > 0 and overlap_seconds(start, end, excluded_spans) / span > 0.5

    rates = [(len(text.split()) / (end - start), start, end, text)
             for start, end, text in cues
             if end - start > limits["rate_min_duration"]
             and len(text.split()) >= limits["rate_min_words"]
             and not (excluded_spans and excluded_cue(start, end))]
    if rates:
        values = [r[0] for r in rates]
        report["median_wps"] = statistics.median(values)
        report["max_wps"] = max(values)
        over = [r for r in rates if r[0] > limits["max_words_per_second"]]
        report["cues_over_rate"] = len(over)
        if over:
            worst = max(over, key=lambda r: r[0])
            report["worst_cue"] = {
                "rate": worst[0], "start": worst[1], "end": worst[2],
                "words": len(worst[3].split()), "text": worst[3][:70],
            }
            report["failures"].append(
                f"{len(over)} cue(s) over {limits['max_words_per_second']} w/s "
                f"(worst {worst[0]:.1f} w/s: {len(worst[3].split())} words in "
                f"{worst[2] - worst[1]:.2f}s at {fmt(worst[1])}) — median is "
                f"{report['median_wps']:.2f}")

    # ── repetition that survived into training data ───────────────────────────
    # Scored on the MANIFEST, not the SRT: the SRT is the faithful record and legitimately
    # still contains the zikr. What must be zero is repetition reaching a training sample.
    # B3014 passed every check above while shipping 127 words of fabricated Urdu; nothing
    # here looked at its text.
    srt_repeats = [find_repetition(text, chant_units) for _, _, text in cues]
    found = [r for r in srt_repeats if r]
    report["srt_repetition_cues"] = len(found)
    report["srt_suspect_loop_cues"] = sum(1 for r in found if r.kind == "suspect_loop")
    if chunks:
        hits = [(c, r) for c in chunks
                if (r := find_repetition(c["transcript"], chant_units))]
        report["manifest_repetition_chunks"] = len(hits)
        report["manifest_repetition_words"] = sum(r.covered for _, r in hits)
        if len(hits) > limits["max_repetition_chunks"]:
            worst_chunk, worst_rep = max(hits, key=lambda h: h[1].covered)
            report["failures"].append(
                f"{len(hits)} manifest chunk(s) contain a repeated-word run "
                f"(worst: {worst_rep.unit!r} x{worst_rep.repeats} = {worst_rep.covered} words, "
                f"{worst_rep.kind}, in {Path(worst_chunk['audio_path']).name}) — this text "
                f"trains the model to loop. Re-run stage 5 so the cue filter removes it.")
    return report


def fmt(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


# Which way is bad, per metric, and how much movement is worth mentioning. Without the
# tolerance a report is a wall of ±0.1 noise and the one number that actually moved is lost
# in it.
REGRESSION_DIRECTIONS = {
    "word_loss_pct":    ("up", 0.2),
    "coverage_pct":     ("down", 0.2),
    "speech_loss_pct":  ("up", 0.1),
    "gap_seconds":      ("up", 2.0),
    "cues_over_rate":   ("up", 0.5),
    "max_wps":          ("up", 0.5),
    "manifest_repetition_chunks": ("up", 0.5),
    # Rising means the DECODER got worse, not the pipeline -- the filter removes these from
    # training either way, so this is the only place a fabricating decoder shows up.
    "srt_suspect_loop_cues":      ("up", 0.5),
}


def latest_report(target: Path) -> Path | None:
    """Newest *.json in `target` if it is a directory, else `target` itself."""
    if target.is_dir():
        reports = sorted(target.glob("*.json"))
        return reports[-1] if reports else None
    return target if target.exists() else None


def compare_reports(current: list[dict], previous_path: Path) -> None:
    """Print how each episode moved since a previous report.

    The gate alone only answers "does this pass NOW". It cannot answer "was this better
    last week" -- and because it re-scores every finished episode on each run, a change that
    quietly worsens an already-passing episode would otherwise go unnoticed. That is exactly
    the failure mode this whole session was about: output that is wrong while every check
    still says pass.
    """
    payload = json.loads(previous_path.read_text(encoding="utf-8"))
    # Accept both the current envelope and the older bare-list format.
    rows = payload.get("episodes", payload) if isinstance(payload, dict) else payload
    before = {r["stem"]: r for r in rows if isinstance(r, dict) and "stem" in r}
    stamp = payload.get("generated_at", "unknown time") if isinstance(payload, dict) else "unknown time"

    print(f"\nCompared against {previous_path.name} ({stamp})")
    regressions, improvements, new = [], [], []
    for report in current:
        prior = before.get(report["stem"])
        if not prior:
            new.append(report["label"])
            continue
        for metric, (bad_direction, tolerance) in REGRESSION_DIRECTIONS.items():
            if metric not in report or metric not in prior:
                continue
            delta = report[metric] - prior[metric]
            if abs(delta) < tolerance:
                continue
            worse = (delta > 0) if bad_direction == "up" else (delta < 0)
            line = (f"{report['label']} {metric} {prior[metric]:.1f} -> "
                    f"{report[metric]:.1f} ({delta:+.1f})")
            (regressions if worse else improvements).append(line)

    for line in regressions:
        print(f"  WORSE      {line}")
    for line in improvements:
        print(f"  better     {line}")
    if new:
        print(f"  new        {', '.join(new)} (not in the previous report)")
    if not (regressions or improvements or new):
        print("  no metric moved beyond its noise tolerance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_batch_argument(parser)
    parser.add_argument("--only", help="Comma-separated labels, e.g. B3001,B3003")
    parser.add_argument("--warn-only", action="store_true",
                        help="Report failures but exit 0 (use to inspect a known-bad batch)")
    parser.add_argument("--json", help="Also write the full per-episode report here")
    parser.add_argument("--compare", default="",
                        help="Report how each metric moved since a previous report. Give a "
                             "report file, or a directory to use its newest one.")
    parser.add_argument("--srt-dir", default="", help="Override data/<batch>/timestamped_srts")
    parser.add_argument("--transcript-dir", default="", help="Override data/<batch>/asr_transcripts")
    parser.add_argument("--audio-dir", default="", help="Override data/<batch>/audio_trimmed")
    parser.add_argument("--processed-dir", default="", help="Override data/processed/<Batch>")
    parser.add_argument("--vad-dir", default="", help="Override data/<batch>/vad_spans")
    for name, value in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    args = parser.parse_args()
    limits = {name: getattr(args, name) for name in DEFAULTS}

    paths = Layout(BatchPaths(args.batch), args.srt_dir, args.transcript_dir,
                   args.audio_dir, args.processed_dir, args.vad_dir)
    grouped = manifest_by_video(paths.processed_dir)
    stems = sorted(p.stem for p in paths.srt_dir.glob("*.srt")) if paths.srt_dir.exists() else []
    if args.only:
        wanted = {t.strip() for t in args.only.split(",") if t.strip()}
        stems = [s for s in stems if s.split("_")[0] in wanted]
        unknown = wanted - {s.split("_")[0] for s in stems}
        if unknown:
            raise SystemExit(f"--only referenced labels with no SRT: {sorted(unknown)}")
    if not stems:
        print(f"No SRTs found in {paths.srt_dir}/ — nothing to check.")
        return

    exclusions = load_exclusions(paths.processed_dir)
    chant_units = load_chant_units()
    reports = [measure(stem, paths, grouped.get(stem, []), limits,
                       exclusions.get(stem), chant_units) for stem in stems]

    # "gap s" is raw uncovered time; "spch s"/"lost%" is the part of it VAD calls speech --
    # the number the gate rules on. Showing both makes the correction visible instead of
    # replacing one opaque figure with another.
    header = (f"{'label':<8}{'min':>6}{'cues':>6}{'loss%':>7}{'cover%':>8}"
              f"{'gaps':>6}{'gap s':>7}{'spch s':>8}{'lost%':>7}"
              f"{'medW/s':>8}{'maxW/s':>8}{'over':>6}{'rep s':>7}{'inMan':>6}  result")
    print(header)
    print("-" * len(header))
    for r in reports:
        mark = "FAIL" if r["failures"] else "pass"
        # An unscored episode has no speech figure; "-" beats printing the raw seconds
        # twice, which would read as "VAD says all of it is speech".
        speech = (f"{r['gap_speech_seconds']:8.0f}" if "gap_speech_seconds" in r
                  else f"{'-':>8}")
        print(f"{r['label']:<8}{r.get('duration', 0) / 60:6.1f}{r.get('cues', 0):6d}"
              f"{r.get('word_loss_pct', float('nan')):7.1f}{r.get('coverage_pct', float('nan')):8.1f}"
              f"{r.get('gaps_over_threshold', 0):6d}{r.get('gap_seconds', 0):7.0f}{speech}"
              f"{r.get('speech_loss_pct', float('nan')):7.1f}{r.get('median_wps', float('nan')):8.2f}"
              f"{r.get('max_wps', float('nan')):8.1f}{r.get('cues_over_rate', 0):6d}"
              f"{r.get('excluded_seconds', 0):7.0f}{r.get('manifest_repetition_chunks', 0):6d}"
              f"  {mark}")

    failed = [r for r in reports if r["failures"]]
    for r in reports:
        for note in r["notes"]:
            print(f"\n  note  {r['label']}: {note}")
    for r in failed:
        print(f"\n  FAIL  {r['label']}")
        for failure in r["failures"]:
            print(f"          - {failure}")

    # Compare BEFORE writing: --compare and --json usually point at the same directory, so
    # writing first would make this run its own baseline and always report "no change".
    if args.compare:
        previous = latest_report(Path(args.compare))
        if previous is None:
            print(f"\nNo previous report in {args.compare} — nothing to compare against.")
        else:
            compare_reports(reports, previous)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        # The thresholds are CLI-overridable, so a report without them cannot be compared
        # against another -- the same numbers can pass one run and fail the next. Record the
        # limits and the layout that produced these figures, not just the figures.
        payload = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "batch": args.batch,
            "limits": limits,
            "gap_threshold": GAP_THRESHOLD,
            "inputs": {"srt_dir": str(paths.srt_dir), "transcript_dir": str(paths.transcript_dir),
                       "audio_dir": str(paths.audio_trimmed), "vad_dir": str(paths.vad_dir),
                       "processed_dir": str(paths.processed_dir)},
            "passed": len(reports) - len(failed),
            "failed": len(failed),
            "episodes": reports,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {out}")

    print(f"\n{len(reports) - len(failed)} passed | {len(failed)} failed")
    if failed and not args.warn_only:
        raise SystemExit(
            f"QA gate FAILED for {len(failed)} episode(s): "
            f"{', '.join(r['label'] for r in failed)}\n"
            f"These timings are not fit for review or training. Re-run the alignment stage "
            f"(consider --windows vad), or pass --warn-only to inspect without blocking.")


if __name__ == "__main__":
    main()