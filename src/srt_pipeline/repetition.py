"""Detect runs of repeated words in a cue, and say which kind they are.

WHY
---
Two different things produce repeated text in these episodes, and they look IDENTICAL
to every structural check we have:

    یا اللہ، یا اللہ، یا اللہ، یا اللہ،        <- real zikr, correctly transcribed
    کامیابی سے کامیابی سے کامیابی سے کامیابی سے   <- Whisper decoder loop, fabricated

Both are 2-token units repeated 4+ times. Token count, cue duration, speaking rate and
VAD coverage all fail to separate them -- measured across batch3, an earlier
"single token = chant, multi-word = hallucination" rule misclassified three real
devotional formulae as hallucinations.

NEITHER BELONGS IN TRAINING. A Whisper fine-tune that sees a 178x اللہ sample learns to
emit repetition loops -- which is the exact defect we are trying to remove. So the
manifest excludes both, and this module's CLASSIFICATION does not decide inclusion. It
decides what we KNOW:

  * `chant`        -- the repeated unit is a known devotional formula (config/chant_units.json).
                      The SRT text is CORRECT; only its training value is bad.
  * `suspect_loop` -- the unit is not a known formula. Probably a decoder failure, i.e. the
                      SRT text is WRONG. Not proof: an unlisted-but-genuine chant lands here
                      too, which is why the exclusion ledger records the unit for review
                      rather than asserting hallucination.

Merging the two buckets would hide the only signal we have that decode settings need
fixing: once the manifest looks clean, a fabricating decoder becomes invisible.

THRESHOLDS
----------
4 repeats, covering 40% of the cue. Urdu reduplication is real and common
("جلدی جلدی", "آہستہ آہستہ") but it is a 2x doubling of a lexical item, not a 4x run --
so the repeat floor is what keeps ordinary speech out, not a vocabulary list.
"""
import json
import re
import unicodedata
from pathlib import Path

# A run must repeat this many times, and cover this share of the cue, to count. Both
# conditions: 4 repeats inside a 60-word cue is a stutter, not a loop.
MIN_REPEATS = 4
MIN_SHARE = 0.4
# Below this the share test is meaningless -- a 3-token cue is trivially 100% anything.
MIN_TOKENS = 4
# Longest repeating unit considered. Observed loops are 1-3 tokens; 6 is headroom without
# making the scan quadratic in practice.
MAX_UNIT = 6
# A cue ADJACENT to a detected run is absorbed when this share of its tokens are the run's
# own words. Not 1.0: the aligner truncates a chant's last word into a fragment, which is
# still chant audio and not speech worth training on. See is_continuation().
CONTINUATION_SHARE = 0.8

CHANT_UNITS_PATH = Path(__file__).resolve().parents[2] / "config" / "chant_units.json"

# Urdu/Arabic combining marks + the punctuation Whisper attaches to chanted words. Stripped
# both to allowlist matching AND to run detection -- the cue text itself is never rewritten.
# Whisper emits mixed punctuation INSIDE a single chant (U+3001 IDEOGRAPHIC COMMA turns up
# between Urdu commas on B3015); undetected, that run reads as three distinct tokens and
# escapes detection entirely, so these are folded before comparing tokens.
_WIDE_PUNCTUATION = re.compile("[、。，．？！]")
# Without the sets below,
# "اللہ ہو،" and "اللہ ہو" are different strings and the list would need every variant.
_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
_PUNCTUATION = re.compile(r"[،۔؟!\.,\?\"'’“”\-–—:;]")


class Repetition:
    """One repeated-word run found in a cue."""

    __slots__ = ("unit", "unit_tokens", "repeats", "covered", "total", "kind", "origin")

    def __init__(self, unit: str, unit_tokens: int, repeats: int, covered: int,
                 total: int, kind: str, origin: str = "run"):
        self.unit = unit
        self.unit_tokens = unit_tokens
        self.repeats = repeats
        self.covered = covered
        self.total = total
        self.kind = kind
        # "run" -- this cue crossed the thresholds on its own.
        # "continuation" -- absorbed because it adjoins a run and is made of its words.
        # Recorded so the ledger never implies a 2-word cue independently looked like a loop.
        self.origin = origin

    @property
    def share(self) -> float:
        return self.covered / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {"unit": self.unit, "unit_tokens": self.unit_tokens, "repeats": self.repeats,
                "covered_words": self.covered, "share": round(self.share, 3),
                "kind": self.kind, "origin": self.origin}

    def __repr__(self) -> str:
        return f"Repetition({self.unit!r} x{self.repeats}, {self.share:.0%}, {self.kind})"


def normalize_unit(text: str) -> str:
    """Fold a repeated unit to its allowlist key: NFC, no diacritics, no punctuation."""
    folded = unicodedata.normalize("NFC", text)
    folded = _WIDE_PUNCTUATION.sub("", _PUNCTUATION.sub("", _DIACRITICS.sub("", folded)))
    return " ".join(folded.split()).strip()


def load_chant_units(path: Path | str | None = None) -> set[str]:
    """Normalized devotional formulae that mark a run as `chant` rather than `suspect_loop`.

    A missing file is not an error -- every run then classifies as `suspect_loop`, which is
    the honest default (we cannot vouch for text we have no list for). It is NOT silent:
    callers surface the empty list, because "no chants found" and "no list loaded" would
    otherwise be indistinguishable in the ledger.
    """
    target = Path(path) if path else CHANT_UNITS_PATH
    if not target.exists():
        return set()
    payload = json.loads(target.read_text(encoding="utf-8"))
    return {normalize_unit(u) for u in payload.get("units", []) if normalize_unit(u)}


def longest_repeat_run(tokens: list[str], max_unit: int = MAX_UNIT) -> tuple[int, int, int]:
    """(unit_tokens, repeats, covered_tokens) for the largest ADJACENT repeated run.

    Adjacent is the point: "اللہ" appearing 40 times scattered through a 70-minute episode
    is normal speech. The same 40 back to back is the defect.
    """
    best = (0, 0, 0)
    for size in range(1, max_unit + 1):
        index = 0
        while index + size <= len(tokens):
            unit = tokens[index:index + size]
            repeats = 1
            while tokens[index + repeats * size: index + (repeats + 1) * size] == unit:
                repeats += 1
            covered = repeats * size
            if repeats >= 2:
                if covered > best[2]:
                    best = (size, repeats, covered)
                index += covered      # skip the run; its interior repeats nothing new
            else:
                index += 1
    return best


def find_repetition(text: str, chant_units: set[str] | None = None,
                    min_repeats: int = MIN_REPEATS, min_share: float = MIN_SHARE,
                    min_tokens: int = MIN_TOKENS) -> Repetition | None:
    """The repeated run in `text` that crosses both thresholds, or None."""
    tokens = text.split()
    if len(tokens) < min_tokens:
        return None
    # Compare NORMALIZED tokens so punctuation drift inside one chant does not hide the run,
    # but quote the ORIGINAL words back so the ledger shows what the SRT actually says.
    folded = [normalize_unit(token) for token in tokens]
    size, repeats, covered = longest_repeat_run(folded)
    if repeats < min_repeats or covered / len(tokens) < min_share:
        return None
    start = _run_start(folded, size, repeats)
    unit = " ".join(tokens[start:start + size])
    kind = "chant" if normalize_unit(unit) in (chant_units or set()) else "suspect_loop"
    return Repetition(unit, size, repeats, covered, len(tokens), kind)


def is_continuation(text: str, unit: str, share: float = CONTINUATION_SHARE) -> bool:
    """True if `text` is the tail/head of a neighbouring cue's repeated `unit`.

    A long chant does not sit inside one cue -- the aligner splits it across many, and the
    cues at each end hold only 2-3 repeats, or a truncated fragment ("را ریاض، را ری").
    Judged alone they fall under min_repeats and survive into training as exactly the
    looping text we are removing; measured on B3015, 8 chunks / 31 words / 26s of residue.

    Deliberately NOT a lower global repeat threshold: "کیوں ہوا یہ کوئی نہیں جانتا، کیوں ہوا
    یہ" is a 2x rhetorical repetition in ordinary oratory and must survive. This only absorbs
    a cue that is ADJACENT to an already-detected run and made of that run's own words.
    """
    tokens = [normalize_unit(token) for token in text.split()]
    tokens = [token for token in tokens if token]
    if not tokens:
        return False
    vocabulary = {token for token in normalize_unit(unit).split() if token}
    if not vocabulary:
        return False
    inside = sum(1 for token in tokens if token in vocabulary)
    return inside / len(tokens) >= share


def _run_start(tokens: list[str], size: int, repeats: int) -> int:
    """Index where the winning run begins, so the ledger can quote the actual unit."""
    for index in range(len(tokens) - size * repeats + 1):
        unit = tokens[index:index + size]
        if all(tokens[index + r * size: index + (r + 1) * size] == unit for r in range(repeats)):
            return index
    return 0