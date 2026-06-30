"""
One-off prep path for samples delivered as audio + matching .srt transcript.

Groups SRT cues into <=28s windows, then snaps each chunk boundary to the
nearest true silence point in the audio (via short-time energy) instead of
trusting the raw SRT timestamp directly -- downloaded/auto-aligned subtitles
are frequently off by several hundred ms to ~2s relative to actual speech,
which otherwise cuts audio mid-word at chunk boundaries while the word's
text stays assigned to the neighboring chunk.

Gaps between cues larger than GAP_THRESHOLD (e.g. background music or an
intro/outro with no captioned speech) are excluded from chunk audio rather
than silence-snapped like a normal pause -- training audio should not
contain stretches with no corresponding transcript.

Writes a manifest.json compatible with dataset_builder.py.

Naming convention: chunks are written to
  audio/{video_id}/{video_id}_{chunk_idx:03d}.wav
where video_id is either:
  - {episode_label}_{youtube_id}, reusing a short label already in the
    source filename (e.g. "EP1_hBK8bkFgus8" from "EP1_hBK8bkFgus8.mp3"), or
  - vid{NNNN}_{youtube_id} (falling back to the sequential --video_index)
    when the source filename has no such clean label -- e.g. long,
    auto-downloaded titles. See make_video_id() for the exact rule.

Run locally:
  python src/srt_audio_prep.py \
    --audio "samples/audio.mp3" \
    --srt "samples/audio.srt" \
    --output_dir ./data/processed/sample_test \
    --video_index 1
"""
import re
import sys
import json
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from dataclasses import asdict

from data_prep import Sample, validate_transcript

sys.stdout.reconfigure(encoding="utf-8")

SRT_TIME_RE = re.compile(r"(\d+):(\d+):(\d+),(\d+)")
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
EPISODE_LABEL_RE = re.compile(r"^[A-Za-z]+\d+$")  # e.g. "EP1", "Episode12" -- short label, no separators
GAP_THRESHOLD = 3.0  # inter-cue gaps beyond this are treated as non-speech filler, not a pause
SENTENCE_END_RE = re.compile(r"[۔؟!]\s*$")  # Urdu/Arabic full stop, question mark, or "!" at cue end


def ends_sentence(text: str) -> bool:
    """True if a cue's text ends a complete sentence (Urdu ۔, ؟, or !)."""
    return bool(SENTENCE_END_RE.search(text.strip()))


def find_youtube_id(path: str) -> str | None:
    """Pull an 11-char YouTube video ID from a filename, if present."""
    for part in Path(path).stem.split("_"):
        if YOUTUBE_ID_RE.match(part):
            return part
    return None


def make_video_id(audio_path: str, video_index: int) -> str:
    """
    Prefer the source filename's own episode label (e.g. "EP1" in
    "EP1_hBK8bkFgus8.mp3") + YouTube ID when the filename is already clean --
    keeps output names consistent with input names for well-named deliveries.

    Falls back to vid{NNNN} (sequential, always present) + YouTube ID when no
    such label is found -- e.g. the long, messy auto-downloaded filenames seen
    in practice ("YTDown_YouTube_<80-char-title>_<id>_009_128k.mp3"), where
    reusing the filename verbatim would reintroduce the unscalable long-name
    problem this convention was built to avoid.
    """
    youtube_id = find_youtube_id(audio_path)
    if youtube_id:
        stem_parts = Path(audio_path).stem.split("_")
        prefix_parts = stem_parts[:stem_parts.index(youtube_id)]
        if len(prefix_parts) == 1 and EPISODE_LABEL_RE.match(prefix_parts[0]):
            return f"{prefix_parts[0]}_{youtube_id}"

    video_id = f"vid{video_index:04d}"
    return f"{video_id}_{youtube_id}" if youtube_id else video_id


def parse_srt_time(ts: str) -> float:
    h, m, s, ms = SRT_TIME_RE.match(ts.strip()).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(srt_path: str) -> list[tuple[float, float, str]]:
    """Parse an SRT file into (start_sec, end_sec, text) cues."""
    blocks = Path(srt_path).read_text(encoding="utf-8-sig").strip().split("\n\n")
    cues = []
    for block in blocks:
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        # lines[0] is the cue index, lines[1] is the timestamp range
        time_line = lines[1] if "-->" in lines[1] else lines[0]
        start_str, end_str = [t.strip() for t in time_line.split("-->")]
        text = " ".join(lines[2:]) if "-->" in lines[1] else " ".join(lines[1:])
        cues.append((parse_srt_time(start_str), parse_srt_time(end_str), text))
    return cues


def group_cues(
    cues: list[tuple[float, float, str]],
    max_duration: float = 28.0,
    gap_threshold: float = GAP_THRESHOLD,
) -> list[tuple[float, float, str]]:
    """
    Greedily group consecutive cues into windows no longer than max_duration,
    preferring to break at a sentence boundary (cue text ending in ۔ ؟ !)
    over a mid-sentence cut. When growing the window would exceed
    max_duration, the window is split right after its last sentence-ending
    cue rather than wherever the duration limit happens to land -- avoids
    training the model that mid-utterance truncation is acceptable output.
    Falls back to the old hard duration cut if no sentence boundary exists
    anywhere in the window (rare run-on speech with no punctuated cue).

    Also forces a new window whenever the gap before a cue exceeds
    gap_threshold, regardless of accumulated duration -- such gaps usually
    mean non-speech filler (music, intro/outro) rather than a normal pause.
    """
    chunks = []
    current = []
    last_sentence_end = -1  # index within `current` of its last sentence-ending cue

    for cue in cues:
        if current:
            gap_before = cue[0] - current[-1][1]
            prospective_span = cue[1] - current[0][0]
            if gap_before > gap_threshold:
                chunks.append(current)
                current, last_sentence_end = [], -1
            elif prospective_span > max_duration:
                if last_sentence_end >= 0:
                    chunks.append(current[: last_sentence_end + 1])
                    current = current[last_sentence_end + 1 :]
                else:
                    # No sentence boundary anywhere in this window -- nothing
                    # better to split on, so keep the old fixed-duration cut.
                    chunks.append(current)
                    current = []
                last_sentence_end = next(
                    (i for i in range(len(current) - 1, -1, -1) if ends_sentence(current[i][2])),
                    -1,
                )
        current.append(cue)
        if ends_sentence(cue[2]):
            last_sentence_end = len(current) - 1
    if current:
        chunks.append(current)

    return [
        (chunk[0][0], chunk[-1][1], " ".join(c[2] for c in chunk))
        for chunk in chunks
    ]


def find_silence_boundary(
    y: np.ndarray,
    sr: int,
    nominal_time: float,
    search_before: float = 0.5,
    search_end_time: float | None = None,
    frame_length: int = 400,
    hop_length: int = 160,
    silence_ratio: float = 0.5,
) -> float:
    """
    Snap nominal_time to the nearest local energy minimum within
    [nominal_time - search_before, search_end_time]. search_end_time must be
    an absolute time and should never exceed the start of the next cue's own
    span, or a single cue's audio could be split across chunks while its
    text stays whole in one (only safe to search within real inter-cue gaps).
    Falls back to nominal_time if no clear silence dip is found, so
    continuous speech with no real pause isn't cut at an arbitrary point.
    """
    if search_end_time is None or search_end_time <= nominal_time:
        return nominal_time
    start_sample = max(0, int((nominal_time - search_before) * sr))
    end_sample = min(len(y), int(search_end_time * sr))
    if end_sample <= start_sample + frame_length:
        return nominal_time

    rms = librosa.feature.rms(
        y=y[start_sample:end_sample], frame_length=frame_length, hop_length=hop_length
    )[0]
    if len(rms) == 0:
        return nominal_time

    min_idx = int(np.argmin(rms))
    median_val = float(np.median(rms))
    if median_val <= 0 or rms[min_idx] > silence_ratio * median_val:
        return nominal_time  # no clear silence dip -- keep the nominal boundary

    return (start_sample + min_idx * hop_length) / sr


def prepare_from_srt(
    audio_path: str,
    srt_path: str,
    output_dir: str,
    max_duration: float = 28.0,
    search_before: float = 0.5,
    video_index: int = 1,
    write_manifest: bool = True,
) -> list[Sample]:
    video_id = make_video_id(audio_path, video_index)
    audio_dir = Path(output_dir) / "audio" / video_id
    audio_dir.mkdir(parents=True, exist_ok=True)

    cues = parse_srt(srt_path)
    # Group with headroom below max_duration: silence-snapping below can push
    # a chunk's boundaries later on both sides, so the *grouped* (pre-snap)
    # window must leave room or the final audio could exceed max_duration.
    # A normal-boundary snap is bounded by the real inter-cue gap, which is
    # capped at GAP_THRESHOLD (larger gaps go through exclusion instead).
    grouping_duration = max(1.0, max_duration - GAP_THRESHOLD - 0.5)
    windows = group_cues(cues, grouping_duration)
    num_chunks = len(windows)
    print(f"Parsed {len(cues)} cues -> grouped into {num_chunks} chunks (<={grouping_duration}s each, pre-snap)")

    print("Loading audio for silence-boundary snapping...")
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    total_duration = len(y) / sr

    # Per-chunk (start, end) spans -- NOT a single shared boundary list,
    # because a real non-speech gap (see GAP_THRESHOLD) must be excluded
    # from both neighboring chunks rather than assigned to one of them.
    chunk_starts = [max(0.0, windows[0][0] - 0.3)]
    chunk_ends = []
    for i in range(num_chunks - 1):
        prev_end_nominal = windows[i][1]
        next_start_nominal = windows[i + 1][0]
        gap = next_start_nominal - prev_end_nominal

        if gap > GAP_THRESHOLD:
            # Real non-speech filler: trim each side independently and drop the gap.
            end_i = find_silence_boundary(y, sr, prev_end_nominal, search_before, prev_end_nominal + min(1.0, gap))
            start_next = max(end_i, next_start_nominal - 0.3)
            print(f"  boundary {i:03d}->{i+1:03d}: {gap:.2f}s gap excluded ({prev_end_nominal:.2f}s -> {next_start_nominal:.2f}s)")
        else:
            # Bounded strictly to [prev_end_nominal, next_start_nominal] -- the
            # real gap between these two specific cues -- so the search can
            # never land inside either cue's own span and split its audio
            # while its (whole-cue-granularity) text stays in just one chunk.
            snapped = find_silence_boundary(y, sr, prev_end_nominal, search_before, next_start_nominal)
            shift = snapped - prev_end_nominal
            if abs(shift) > 0.05:
                print(f"  boundary {i:03d}->{i+1:03d}: snapped {prev_end_nominal:.2f}s -> {snapped:.2f}s ({shift:+.2f}s)")
            end_i = start_next = snapped

        chunk_ends.append(end_i)
        chunk_starts.append(start_next)
    chunk_ends.append(min(total_duration, windows[-1][1] + 0.3))

    # Re-derive each chunk's TEXT from the final (snapped) boundaries rather
    # than the original nominal grouping above. A snap can shift a boundary
    # by up to `search_after` seconds, enough to pull a whole short cue's
    # audio into the previous chunk while group_cues() still left its text
    # in the next chunk's window -- this re-walk keeps text and audio in sync.
    chunk_texts = [[] for _ in range(num_chunks)]
    chunk_idx = 0
    for cue_start, _, cue_text in cues:
        while chunk_idx < num_chunks - 1 and cue_start >= chunk_ends[chunk_idx]:
            chunk_idx += 1
        chunk_texts[chunk_idx].append(cue_text)
    texts = [" ".join(t) for t in chunk_texts]

    samples = []
    for i, text in enumerate(texts):
        if not validate_transcript(text):
            print(f"  ⚠ Chunk {i:03d} failed transcript validation, skipping")
            continue

        start, end = chunk_starts[i], chunk_ends[i]
        if end - start > max_duration:
            # Safety backstop -- should be rare given grouping_duration's headroom.
            # Clips this chunk's tail only; does not touch the next chunk's start.
            print(f"  ⚠ chunk {i:03d} would be {end - start:.1f}s, clamping to {max_duration}s")
            end = start + max_duration
        segment = y[int(start * sr):int(end * sr)]

        out_path = str(audio_dir / f"{video_id}_{i:03d}.wav")
        sf.write(out_path, segment, sr, subtype="PCM_16")

        duration = end - start
        samples.append(Sample(audio_path=out_path, transcript=text, duration=duration))
        print(f"  ✓ chunk {i:03d} ({duration:.1f}s): {text[:50]}...")

    total_hours = sum(s.duration for s in samples) / 3600
    print(f"\n✅ Prepared {len(samples)} chunks ({total_hours:.3f} hours)")

    if write_manifest:
        manifest_path = Path(output_dir) / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in samples], f, ensure_ascii=False, indent=2)
        print(f"   Manifest saved to: {manifest_path}")

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--srt", required=True)
    parser.add_argument("--output_dir", default="./data/processed/sample_test")
    parser.add_argument("--max_duration", type=float, default=28.0)
    parser.add_argument("--video_index", type=int, default=1, help="Sequential video number, e.g. 1 -> vid0001")
    args = parser.parse_args()

    prepare_from_srt(args.audio, args.srt, args.output_dir, args.max_duration, video_index=args.video_index)
