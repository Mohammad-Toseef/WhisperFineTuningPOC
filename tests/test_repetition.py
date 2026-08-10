"""Cases here are REAL cue text from batch3, not invented strings.

Every earlier attempt at this rule was defeated by actual data -- "single token = chant"
misfiled یا اللہ، and اللہ ہو، as hallucinations. So the fixtures are the eight distinct
repeated units the batch3 scan found, plus the ordinary-speech cases that must NOT trip.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "srt_pipeline"))

from repetition import (  # noqa: E402
    find_repetition, is_continuation, longest_repeat_run, normalize_unit, load_chant_units,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

UNITS = load_chant_units()


def _find(text):
    return find_repetition(text, UNITS)


# ── real repeated runs found in batch3 ───────────────────────────────────────────────
def test_single_token_zikr_is_a_chant():
    found = _find("اللہ " * 20)
    assert found is not None
    assert found.unit_tokens == 1 and found.repeats == 20
    assert found.kind == "chant"


def test_two_token_zikr_with_punctuation_is_a_chant():
    # B3003. The punctuation is why the allowlist matches on a normalized form.
    found = _find("اور روح جو ہے " + "یا اللہ، " * 6)
    assert found is not None
    assert found.unit_tokens == 2
    assert found.kind == "chant", "یا اللہ، must not be filed as a decoder loop"


def test_allah_hu_is_a_chant():
    found = _find("گا " + "اللہ ہو، " * 5 + "یہ دل")
    assert found is not None and found.kind == "chant"


def test_multi_word_decoder_loop_is_suspect():
    # B3014 -- fabricated. Structurally identical to the two chants above.
    found = _find("اگر صلاة مجھے " + "کامیابی سے " * 10)
    assert found is not None
    assert found.unit_tokens == 2 and found.kind == "suspect_loop"


def test_three_token_loop_is_detected():
    found = _find("کہ، اُنجاہی ہے " * 5)          # B3016
    assert found is not None and found.repeats >= 4


def test_unlisted_unit_is_suspect_not_chant():
    # B3015's را ریاض is deliberately off the allowlist pending an ear-check. The honest
    # label is "we cannot vouch for this", not "hallucination" and not "fine".
    found = _find("را ریاض، " * 8)
    assert found is not None and found.kind == "suspect_loop"


# ── ordinary speech that must survive ────────────────────────────────────────────────
def test_urdu_reduplication_is_not_repetition():
    # Doubling is normal Urdu emphasis. The 4-repeat floor, not a vocabulary list, is what
    # keeps it out.
    assert _find("وہ جلدی جلدی گھر گیا اور آہستہ آہستہ بیٹھ گیا تھا") is None


def test_scattered_repeats_are_not_a_run():
    # The same word many times in a long cue is speech; ADJACENCY is the defect.
    assert _find("اللہ کا ذکر ہے اللہ کی رحمت ہے اللہ کا کرم ہے اللہ کی بات ہے") is None


def test_short_cue_is_not_judged():
    assert _find("اللہ اللہ اللہ") is None       # 3 tokens: below min_tokens


def test_run_below_share_threshold_survives():
    # 4 repeats buried in a long sentence is a stutter, not a loop -- the cue's other
    # words are real speech we should keep.
    text = "یہ بات " + "ہاں ہاں ہاں ہاں " + " ".join(f"لفظ{i}" for i in range(20))
    assert _find(text) is None


# ── mechanics ────────────────────────────────────────────────────────────────────────
def test_longest_run_prefers_wider_coverage():
    # "a b" x3 (6 tokens) must beat "c" x4 (4 tokens).
    assert longest_repeat_run("a b a b a b c c c c".split()) == (2, 3, 6)


def test_normalize_strips_diacritics_and_punctuation():
    assert normalize_unit("اللہ ہو،") == normalize_unit("اللہ ہو")
    assert normalize_unit("اُٹھ") == normalize_unit("اٹھ")


def test_missing_allowlist_degrades_to_suspect_not_crash():
    found = find_repetition("اللہ " * 20, load_chant_units("does/not/exist.json"))
    assert found is not None and found.kind == "suspect_loop"


def test_covered_words_counts_the_whole_run():
    found = _find("کامیابی سے " * 62)
    assert found.covered == 124 and found.repeats == 62


def test_mixed_punctuation_inside_a_chant_is_still_one_run():
    # B3015 chunk 178: Whisper slipped a U+3001 ideographic comma into an Urdu chant. Before
    # folding it, this read as three distinct tokens and reached the manifest.
    found = _find("اللہ، اللہ、 اللہ، اللہ، اللہ،")
    assert found is not None and found.repeats >= 4


# ── continuation absorption (cues at the ends of a long chant) ───────────────────────
def test_short_tail_of_a_chant_is_a_continuation():
    assert is_continuation("اللہ اللہ", "اللہ")
    assert is_continuation("اللہ", "اللہ")


def test_truncated_fragment_is_a_continuation():
    # "را ریاض، را ریاض، را ریاض، را ری" -- 7 of 8 tokens belong to the unit.
    assert is_continuation("را ریاض، را ریاض، را ریاض، را ری", "را ریاض،")


def test_ordinary_speech_is_not_a_continuation():
    # B3015 really contains this 2x rhetorical repetition next to other content. It must
    # survive even when it sits beside a chant.
    assert not is_continuation(
        "کربلا میں کیوں ہوا یہ کوئی نہیں جانتا، کیوں ہوا یہ", "اللہ")


def test_empty_text_is_not_a_continuation():
    assert not is_continuation("", "اللہ")
    assert not is_continuation("اللہ", "")


# ── the stage-5 filter, end to end ───────────────────────────────────────────────────
def test_filter_absorbs_boundary_cues_but_keeps_speech():
    from srt_audio_prep import drop_repetition_cues

    cues = [
        (0.0, 3.0, "یہ بات سمجھنے کی ہے کہ ہم سب کو چلنا ہے"),   # speech, keep
        (3.0, 4.0, "اللہ اللہ"),                                  # 2x -> continuation
        (4.0, 12.0, "اللہ " * 20),                                # the run itself
        (12.0, 13.0, "اللہ"),                                     # 1x -> continuation
        (13.0, 18.0, "کیوں ہوا یہ کوئی نہیں جانتا، کیوں ہوا یہ"),  # 2x rhetoric, keep
    ]
    kept, excluded = drop_repetition_cues(cues, UNITS)
    assert [c[0] for c in kept] == [0.0, 13.0], "adjacent rhetoric must survive"
    assert len(excluded) == 3
    assert [e["origin"] for e in excluded] == ["continuation", "run", "continuation"]
    assert all(e["kind"] == "chant" for e in excluded)
