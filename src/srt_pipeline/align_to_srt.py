"""
Forced alignment POC: given (audio, verified plain-text transcript), produce a timestamped SRT.

Approach:
  1. Run a quick Whisper ASR pass over the audio. We throw away its transcribed text and
     only keep its segment timing (Whisper/VAD naturally chunks long audio into short,
     pause-aligned windows -- exactly the scaffolding whisperx.align() needs, since align()
     does one full forward pass per segment and chokes on long-audio segments).
  2. Fuzzy-match the ASR's (possibly wrong) words against our verified ground-truth words to
     find anchor points, then use those anchors to slice the ground-truth text into the same
     segment boundaries Whisper found. This only needs to be roughly right -- the actual
     word-level timing precision comes from step 3, not from this matching step.
  3. Run whisperx.align() per segment using the verified text (not Whisper's) against a
     Urdu wav2vec2 CTC model -- this is the actual forced alignment.
  4. Flatten word timestamps and regroup into SRT-sized cues.
"""

import argparse
import difflib
import sys
from pathlib import Path

import whisperx
from whisperx.audio import SAMPLE_RATE

PUNCT_STRIP = " \t\r\n۔؟،!?.,;:\"'()[]{}«»ـ"
SENTENCE_END_CHARS = "۔؟!?."


def normalize(word: str) -> str:
    return word.strip(PUNCT_STRIP).casefold()


def read_ground_truth(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return text.split()


def transcribe_rough(audio, model_size: str, device: str, language: str) -> list[dict]:
    model = whisperx.load_model(model_size, device=device, language=language)
    result = model.transcribe(audio, language=language, print_progress=True)
    return result["segments"]


def map_asr_to_gt(asr_words: list[str], gt_words: list[str]) -> list[int]:
    """For each ASR word, find its best-guess index into gt_words via anchor matching +
    linear interpolation across the gaps between anchors."""
    a_norm = [normalize(w) for w in asr_words]
    g_norm = [normalize(w) for w in gt_words]
    matcher = difflib.SequenceMatcher(None, a_norm, g_norm, autojunk=False)

    mapping: list[int | None] = [None] * len(asr_words)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            mapping[block.a + k] = block.b + k

    known = [(i, v) for i, v in enumerate(mapping) if v is not None]
    if not known:
        # No anchors at all -- fall back to pure proportional spread.
        n = max(len(asr_words), 1)
        return [round(i * len(gt_words) / n) for i in range(len(asr_words))]

    first_i, first_v = known[0]
    for i in range(first_i):
        mapping[i] = round(first_v * i / max(first_i, 1))

    last_i, last_v = known[-1]
    for i in range(last_i + 1, len(mapping)):
        frac = (i - last_i) / max(len(mapping) - 1 - last_i, 1)
        mapping[i] = round(last_v + frac * (len(gt_words) - 1 - last_v))

    for (i1, v1), (i2, v2) in zip(known, known[1:]):
        if i2 - i1 > 1:
            for i in range(i1 + 1, i2):
                frac = (i - i1) / (i2 - i1)
                mapping[i] = round(v1 + frac * (v2 - v1))

    return mapping  # type: ignore[return-value]


def build_segments_with_gt_text(
    rough_segments: list[dict],
    asr_word_seg: list[int],
    mapping: list[int],
    gt_words: list[str],
) -> list[dict]:
    n = len(rough_segments)
    starts: list[int | None] = [None] * n
    for seg_idx in range(n):
        word_indices = [i for i, s in enumerate(asr_word_seg) if s == seg_idx]
        if word_indices:
            starts[seg_idx] = mapping[word_indices[0]]

    for i in range(n):
        if starts[i] is None:
            starts[i] = starts[i - 1] if i > 0 else 0
    for i in range(1, n):
        starts[i] = max(starts[i], starts[i - 1])  # type: ignore[operator]
    starts.append(len(gt_words))

    segments_out = []
    for i in range(n):
        b0, b1 = starts[i], starts[i + 1]
        text = " ".join(gt_words[b0:b1]).strip()
        if not text:
            continue  # no ground-truth words landed in this window; skip it, nothing is lost
        segments_out.append({"start": rough_segments[i]["start"], "end": rough_segments[i]["end"], "text": text})
    return segments_out


def flatten_words(aligned_segments: list[dict]) -> list[dict]:
    words = []
    for seg in aligned_segments:
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append(w)
    return words


def group_into_cues(words: list[dict], max_words: int = 12, max_duration: float = 7.0) -> list[dict]:
    cues = []
    current: list[dict] = []
    cur_start = None
    for w in words:
        if cur_start is None:
            cur_start = w["start"]
        current.append(w)
        duration = w["end"] - cur_start
        stripped = w["word"].strip()
        ends_sentence = bool(stripped) and stripped[-1] in SENTENCE_END_CHARS
        if ends_sentence or len(current) >= max_words or duration >= max_duration:
            cues.append({"start": cur_start, "end": w["end"], "text": " ".join(x["word"] for x in current)})
            current = []
            cur_start = None
    if current:
        cues.append({"start": cur_start, "end": current[-1]["end"], "text": " ".join(x["word"] for x in current)})
    return cues


def format_timestamp(t: float) -> str:
    t = max(t, 0.0)
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cues_to_srt_text(cues: list[dict]) -> str:
    lines = []
    for i, cue in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(cue['start'])} --> {format_timestamp(cue['end'])}")
        lines.append(cue["text"])
        lines.append("")
    return "\n".join(lines)


def write_srt(cues: list[dict], out_path: str) -> None:
    Path(out_path).write_text(cues_to_srt_text(cues), encoding="utf-8")


def run_pipeline(
    audio,
    gt_words: list[str],
    asr_model: str,
    asr_language: str,
    align_lang: str,
    device: str,
    time_offset: float = 0.0,
    log=lambda msg: print(msg, file=sys.stderr),
) -> str:
    """Runs the full rough-ASR -> anchor-match -> forced-align -> SRT pipeline and returns SRT text.

    `time_offset` is added back to every cue timestamp -- use it when `audio` has already had
    its leading seconds (e.g. intro music) trimmed off before being passed in here.
    """
    log(f"Ground truth: {len(gt_words)} words")

    log(f"Running rough ASR pass ({asr_model}) for segment timing...")
    rough_segments = transcribe_rough(audio, asr_model, device, asr_language)
    log(f"Rough ASR produced {len(rough_segments)} segments")

    asr_words, asr_word_seg = [], []
    for i, seg in enumerate(rough_segments):
        for w in seg["text"].split():
            asr_words.append(w)
            asr_word_seg.append(i)

    mapping = map_asr_to_gt(asr_words, gt_words)
    segments_for_align = build_segments_with_gt_text(rough_segments, asr_word_seg, mapping, gt_words)
    log(f"Built {len(segments_for_align)} ground-truth-text segments for alignment")

    log(f"Loading alignment model for language={align_lang}...")
    align_model, align_meta = whisperx.load_align_model(language_code=align_lang, device=device)
    aligned = whisperx.align(segments_for_align, align_model, align_meta, audio, device, return_char_alignments=False)

    words = flatten_words(aligned["segments"])
    log(f"Aligned {len(words)} words")
    cues = group_into_cues(words)
    if time_offset:
        for cue in cues:
            cue["start"] += time_offset
            cue["end"] += time_offset
    return cues_to_srt_text(cues)


def main():
    parser = argparse.ArgumentParser(description="Forced-align a verified transcript against audio and emit an SRT.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--asr-model", default="small", help="Whisper model size for the rough timing pass.")
    parser.add_argument("--asr-language", default="ur", help="Language hint fed to the rough ASR pass.")
    parser.add_argument("--align-lang", default="ur", help="wav2vec2 alignment model language.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-start", type=float, default=0.0, help="Seconds of leading audio (e.g. intro music) to trim before alignment.")
    args = parser.parse_args()

    print(f"Loading audio: {args.audio}", file=sys.stderr)
    audio = whisperx.load_audio(args.audio)
    if args.skip_start > 0:
        audio = audio[int(args.skip_start * SAMPLE_RATE):]
    gt_words = read_ground_truth(args.text)

    srt_text = run_pipeline(audio, gt_words, args.asr_model, args.asr_language, args.align_lang, args.device, time_offset=args.skip_start)
    Path(args.out).write_text(srt_text, encoding="utf-8")
    print(f"Wrote SRT to {args.out}")


if __name__ == "__main__":
    main()
