r"""Compare base-model vs trained-model transcripts, episode by episode.

WHY THIS EXISTS
---------------
`triage_errors.py` answers "which lever to pull" from the eval_predictions
JSON that `modal_app.py::evaluate` produces — hundreds of short eval clips,
always three rounds, always with a reference. After a training run finishes,
the more common question is simpler: "did THIS training help, on THESE
episodes?" — starting from a folder of plain .txt transcripts (what
transcribe_batch actually writes), sometimes with a reference transcript
available, sometimes not.

This script covers that: point it at a folder, it discovers base/trained
(/reference) triples by filename convention, and produces the same kind of
colour-coded, browser-readable diff report — extended with WER/CER, a
per-language breakdown (this speaker code-switches Urdu/Arabic/English), and
sort/search/isolate controls in the report itself, since there is no console
step in between to slice the data.

It deliberately reuses triage_errors.py's classification engine (the
diacritic/script_variant/spelling/near_miss/misheard/dropped/inserted
taxonomy, the Urdu-Arabic phonetic folding, the colours) rather than
reimplementing it — that logic was hard-won by reading samples, and a second
copy would drift from it silently.

FILE NAMING CONVENTION
-----------------------
For an episode named "EP19_xNjY-mZlyEU" in a folder, this script looks for:
    EP19_xNjY-mZlyEU_base.txt         <- base model output      (required)
    EP19_xNjY-mZlyEU_finetuned.txt    <- trained model output    (required)
    EP19_xNjY-mZlyEU.txt              <- reference transcript    (optional)
Matches full_audio_samples/compare_transcripts/ already in this repo.
Suffixes are configurable with --base-suffix / --trained-suffix / --ref-suffix
for other naming schemes.

    python scripts/compare_transcripts.py full_audio_samples/compare_transcripts
    python scripts/compare_transcripts.py path/to/folder --no-ref
    python scripts/compare_transcripts.py path/to/folder --html reports/round3_compare.html

END-TO-END WORKFLOW FOR A NEW CLIP
-----------------------------------
    1. python scripts/compare_transcripts.py --fetch "<youtube_url>"
       -> downloads the audio into full_audio_samples/ (reuses
          download_playlist_audio.py; a single-video URL works, not just a
          playlist — see that script for why)
    2. python scripts/compare_transcripts.py --transcribe EP42_abc123XYZ.mp3
       -> shells out to `modal run scripts/compare_transcribe.py`, which
          transcribes it with base + trained model on Modal's GPU and writes
          the *_base.txt / *_finetuned.txt pair this script reads. Costs
          Modal GPU time — nothing here runs it for you unasked.
    3. python scripts/compare_transcripts.py full_audio_samples/compare_transcripts
       -> the report

⚠️ Same caveat as triage_errors.py: the categories are a heuristic read of
WHERE two transcripts differ, not a verdict on which one is right. Without a
reference transcript there is no "error" at all — only a difference, and
this script labels it that way throughout.
"""
import argparse
import html
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from triage_errors import (
    CATEGORIES, CAT_COLOR, bare, classify, diff_clip, mark, words,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Language tagging ─────────────────────────────────────────────────────────
# Heuristic, built on the SAME codepoint knowledge triage_errors.py already
# encodes in SCRIPT_VARIANT (which member of each letter-class is the Urdu
# form vs a form Urdu writers essentially never use). Reused here instead of
# re-derived because getting this wrong quietly mislabels every stat below it.
#
#   ARABIC-only forms: Urdu orthography does not use these — ك ي ى ه ة ۃ are
#   always the Arabic/Quranic spelling of a letter Urdu writes as ک ی ہ, and
#   أ إ ٱ ؤ are hamza/wasla forms Urdu flattens to plain ا/و. Seeing one of
#   these means the speaker is reciting/quoting, not speaking Urdu.
#   URDU-only forms: ٹ ڈ ڑ (retroflex), ں (noon-ghunna), ھ (aspiration), ے/ۓ
#   (bari ye) have no equivalent in Arabic at all.
_ARABIC_ONLY = set("كيىهةۃأإٱؤ")
_URDU_ONLY = set("ٹڈڑںھےۓ")
_LATIN_WORD = re.compile(r"^[A-Za-z][A-Za-z0-9'-]*$")
_ARABIC_BLOCK = re.compile(r"[؀-ۿݐ-ݿ]")


def word_lang(word: str) -> str:
    """english | arabic | urdu | other, for one already-punctuation-stripped word."""
    w = bare(word)
    if not w:
        return "other"
    if _LATIN_WORD.match(w):
        return "english"
    if not _ARABIC_BLOCK.search(w):
        return "other"  # digits, symbols, mixed junk
    if any(c in _ARABIC_ONLY for c in w):
        return "arabic"
    if any(c in _URDU_ONLY for c in w):
        return "urdu"
    return "urdu"  # shared/ambiguous letters (ا ب ت م ر ل و ک ی ہ ...) default Urdu


LANG_LABEL = {"urdu": "Urdu", "arabic": "Arabic", "english": "English", "other": "Other"}
LANG_ORDER = ["urdu", "arabic", "english", "other"]

BLAME_REF = {
    "diacritic":      "cosmetic — vowel marks only, no letter differs",
    "script_variant": "convention — Arabic vs Urdu form of the SAME letter (الله/اللہ)",
    "spelling":       "DECODER — right sound, wrong letter (کثرت/کسرت)",
    "near_miss":      "ambiguous — morphology / izafat / word form",
    "misheard":       "ENCODER — heard a different word",
    "dropped":        "acoustic or segmentation — word missing",
    "inserted":       "acoustic or segmentation — word invented",
}
# No reference => no ground truth => "error" doesn't apply. Same taxonomy,
# reframed as what changed between the two model outputs.
BLAME_DIFF = {
    "diacritic":      "cosmetic — vowel marks only, no letter differs",
    "script_variant": "convention — Arabic vs Urdu form of the SAME letter",
    "spelling":       "trained model spelled it differently (same sound)",
    "near_miss":      "reworded — morphology / word form changed",
    "misheard":       "trained model chose a completely different word",
    "dropped":        "base had this word, trained model dropped it",
    "inserted":       "trained model added a word base didn't have",
}


# ── Edit-rate (WER/CER), no external metric library required ────────────────
# jiwer/evaluate aren't installed in this venv (they're a Modal-image-only
# dependency — see requirements.txt). Rather than add a dependency for a
# number this script can already derive: an edit-op count run through
# difflib.SequenceMatcher — the same mechanism diff_clip() already uses to
# build the category counts above, so this number and that breakdown are
# always consistent with each other by construction.
def _edit_rate(a: list, b: list) -> float:
    """% of `a` that must change (sub/del/ins) to become `b`. WER if a,b are
    words; CER if a,b are characters."""
    if not a:
        return 0.0
    sm = SequenceMatcher(None, a, b, autojunk=False)
    ops = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        la, lb = i2 - i1, j2 - j1
        ops += max(la, lb)
    return 100 * ops / len(a)


def wer(ref_text: str, hyp_text: str) -> float:
    return _edit_rate(words(ref_text), words(hyp_text))


def cer(ref_text: str, hyp_text: str) -> float:
    # Character-level, diacritics stripped — same reasoning as the
    # "diacritic" category itself: vowel marks are optional notation for
    # these transcripts, not spelling. Comparable across models even if one
    # produces harakat and the other doesn't.
    r = " ".join(bare(w) for w in words(ref_text))
    h = " ".join(bare(w) for w in words(hyp_text))
    return _edit_rate(list(r), list(h))


# ── File discovery ───────────────────────────────────────────────────────────
def discover(folder: Path, base_suffix: str, trained_suffix: str, ref_suffix: str,
             recursive: bool, use_ref: bool) -> list:
    pattern = f"*{base_suffix}.txt"
    paths = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))
    episodes = []
    seen_names = Counter()
    for base_path in paths:
        stem = base_path.name[: -len(base_suffix + ".txt")]
        trained_path = base_path.with_name(f"{stem}{trained_suffix}.txt")
        if not trained_path.exists():
            print(f"⚠️  skipping {stem}: no {trained_path.name} next to it")
            continue
        ref_path = None
        if use_ref:
            candidate = base_path.with_name(f"{stem}{ref_suffix}.txt")
            if candidate.exists() and candidate not in (base_path, trained_path):
                ref_path = candidate
        name = stem
        seen_names[stem] += 1
        if seen_names[stem] > 1 or (recursive and base_path.parent != folder):
            name = f"{base_path.parent.name}/{stem}"
        episodes.append({
            "name": name, "batch": base_path.parent.name,
            "base_path": base_path, "trained_path": trained_path, "ref_path": ref_path,
        })
    # A second pass to disambiguate names that only collided after batching —
    # rglob visits folders in sorted order so this is deterministic.
    return episodes


def read_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return " ".join(line.strip() for line in lines if line.strip())


# ── Hunk view: collapse the long unchanged stretches ─────────────────────────
# A full episode transcript runs to thousands of words. Showing each side as
# one continuous paragraph means a coloured word on the base row and its
# counterpart on the trained row can be paragraphs apart — the "I have to
# scroll every time" problem. Instead, only the region AROUND each difference
# is shown, with reference/base/trained stacked immediately next to each
# other for that region; everything else collapses to a small
# "N words unchanged" marker — the same idea as `git diff`'s context lines.
HUNK_CONTEXT = 6  # words of unchanged context kept on each side of a difference


def _mark_tokens(r_words: list, h_words: list, opcodes: list, near_miss: float) -> tuple:
    """Same per-word colouring as triage_errors.marked_pair, but returned as
    two PARALLEL LISTS (one HTML string per word) rather than a joined
    string — a marked word's HTML contains spaces of its own (`style="..."
    title="..."`), so a joined string can't be split back into per-word
    tokens. Needed so hunks can slice by word-index without re-parsing HTML."""
    ref_out, hyp_out = [], []
    for op, i1, i2, j1, j2 in opcodes:
        if op == "equal":
            ref_out += [html.escape(w) for w in r_words[i1:i2]]
            hyp_out += [html.escape(w) for w in h_words[j1:j2]]
        elif op == "delete":
            ref_out += [mark(w, "dropped") for w in r_words[i1:i2]]
        elif op == "insert":
            hyp_out += [mark(w, "inserted") for w in h_words[j1:j2]]
        else:
            rs, hs = r_words[i1:i2], h_words[j1:j2]
            for a, b in zip(rs, hs):
                cat = classify(a, b, near_miss)
                ref_out.append(mark(a, cat))
                hyp_out.append(mark(b, cat))
            ref_out += [mark(w, "dropped") for w in rs[len(hs):]]
            hyp_out += [mark(w, "inserted") for w in hs[len(rs):]]
    return ref_out, hyp_out


def _diff_ranges(opcodes: list) -> list:
    """Non-'equal' ranges in the FIRST sequence's index space. Zero-width
    ranges (a pure insert, which consumes none of the first sequence) are
    kept as (i1, i1) — still a point that needs to be shown."""
    return [(i1, i2) for op, i1, i2, j1, j2 in opcodes if op != "equal"]


def _merge_windows(ranges: list, seq_len: int, context: int) -> list:
    """Pad each range by `context` on both sides and merge overlaps."""
    if not ranges:
        return []
    padded = sorted((max(0, a - context), min(seq_len, b + context)) for a, b in ranges)
    merged = [list(padded[0])]
    for a, b in padded[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def build_hunks(primary_len: int, opcode_lists: list, context: int = HUNK_CONTEXT) -> list:
    """Union the "interesting" ranges from one or more opcode lists — all
    sharing the same first-sequence index space — into (start, end) windows
    covering every difference plus surrounding context."""
    ranges = [r for ops in opcode_lists for r in _diff_ranges(ops)]
    return _merge_windows(ranges, primary_len, context)


def _project(opcodes: list, lo: int, hi: int) -> tuple:
    """Given a [lo, hi) range in the FIRST sequence's index space, return the
    corresponding SECOND-sequence range.

    'equal' opcodes map 1:1 by definition (same elements on both sides, same
    length) — so a window that only partially overlaps a long equal run is
    projected EXACTLY, taking just the overlapping slice. Rounding an equal
    run outward to its full span was the earlier bug here: one clip's single
    800-word equal block turned a 6-word context window into an 800-word one.
    A non-equal (replace/delete/insert) opcode has no such 1:1 alignment to
    interpolate within, so it's taken in full — it's the difference itself,
    not context, and these blocks are short."""
    j_lo = j_hi = None
    for op, i1, i2, j1, j2 in opcodes:
        overlaps = (i1 < hi and i2 > lo) or (i1 == i2 and lo <= i1 < hi)
        if not overlaps:
            continue
        if op == "equal":
            seg_lo, seg_hi = max(i1, lo), min(i2, hi)
            this_j_lo, this_j_hi = j1 + (seg_lo - i1), j1 + (seg_hi - i1)
        else:
            this_j_lo, this_j_hi = j1, j2
        if j_lo is None:
            j_lo = this_j_lo
        j_hi = this_j_hi
    return (j_lo or 0), (j_hi or 0)


def _build_hunk_rows(primary_len: int, opcode_lists: list, rows_spec: list,
                      context: int = HUNK_CONTEXT) -> dict:
    """rows_spec: list of (label, token_list, opcodes) — opcodes=None marks
    the row that IS the primary sequence (sliced directly); every other row
    is projected into its own index space via `opcodes`."""
    windows = build_hunks(primary_len, opcode_lists, context)
    hunks, prev_end = [], 0
    for start, end in windows:
        rows = []
        for label, tokens, opcodes in rows_spec:
            lo, hi = (start, end) if opcodes is None else _project(opcodes, start, end)
            rows.append((label, " ".join(tokens[lo:hi])))
        hunks.append({"gap_before": start - prev_end, "rows": rows})
        prev_end = end
    return {"hunks": hunks, "tail_gap": primary_len - prev_end}


# ── Per-episode analysis ─────────────────────────────────────────────────────
def analyze_episode(ep: dict, near_miss: float) -> dict:
    base_text = read_text(ep["base_path"])
    trained_text = read_text(ep["trained_path"])
    has_ref = ep["ref_path"] is not None
    ref_text = read_text(ep["ref_path"]) if has_ref else None

    result = {**ep, "has_ref": has_ref}

    if has_ref:
        ref_w, base_w, trained_w = words(ref_text), words(base_text), words(trained_text)
        result["ref_words"] = len(ref_w)
        result["base_words"] = len(base_w)
        result["trained_words"] = len(trained_w)
        result["base_wer"] = wer(ref_text, base_text)
        result["trained_wer"] = wer(ref_text, trained_text)
        result["base_cer"] = cer(ref_text, base_text)
        result["trained_cer"] = cer(ref_text, trained_text)
        result["delta_wer"] = result["trained_wer"] - result["base_wer"]  # negative = improved

        result["base_pairs"] = list(diff_clip(ref_text, base_text, near_miss))
        result["trained_pairs"] = list(diff_clip(ref_text, trained_text, near_miss))
        result["base_cats"] = Counter(c for c, _, _ in result["base_pairs"])
        result["trained_cats"] = Counter(c for c, _, _ in result["trained_pairs"])

        ops_base = SequenceMatcher(None, ref_w, base_w, autojunk=False).get_opcodes()
        ops_trained = SequenceMatcher(None, ref_w, trained_w, autojunk=False).get_opcodes()
        _, base_marks = _mark_tokens(ref_w, base_w, ops_base, near_miss)
        _, trained_marks = _mark_tokens(ref_w, trained_w, ops_trained, near_miss)
        ref_plain = [html.escape(w) for w in ref_w]

        result["hunk_data"] = _build_hunk_rows(
            len(ref_w), [ops_base, ops_trained],
            [("reference", ref_plain, None),
             ("base", base_marks, ops_base),
             ("trained", trained_marks, ops_trained)])
        result["display"] = [
            ("reference", " ".join(ref_plain)),
            ("base", " ".join(base_marks)),
            ("trained", " ".join(trained_marks)),
        ]
        result["ref_text_for_lang"] = ref_text
    else:
        base_w, trained_w = words(base_text), words(trained_text)
        result["base_words"] = len(base_w)
        result["trained_words"] = len(trained_w)
        pairs = list(diff_clip(base_text, trained_text, near_miss))
        result["diff_pairs"] = pairs
        result["diff_cats"] = Counter(c for c, _, _ in pairs)
        result["diff_pct"] = _edit_rate(base_w, trained_w)

        ops = SequenceMatcher(None, base_w, trained_w, autojunk=False).get_opcodes()
        base_marks, trained_marks = _mark_tokens(base_w, trained_w, ops, near_miss)

        result["hunk_data"] = _build_hunk_rows(
            len(base_w), [ops],
            [("base", base_marks, None), ("trained", trained_marks, ops)])
        result["display"] = [("base", " ".join(base_marks)), ("trained", " ".join(trained_marks))]
        result["ref_text_for_lang"] = base_text

    return result


def language_breakdown(episodes: list, has_ref: bool) -> dict:
    """total words and differing words per language, summed across episodes.

    base_diffs and trained_diffs are kept SEPARATE (both measured against the
    same totals, from the reference/base text) rather than pooled — pooling
    two independent diff runs against one word count let the rate exceed
    100%, which is meaningless.
    """
    totals = Counter()
    base_diffs = Counter()
    trained_diffs = Counter()
    for ep in episodes:
        for w in words(ep["ref_text_for_lang"]):
            totals[word_lang(w)] += 1
        if has_ref:
            for cat, a, b in ep["base_pairs"]:
                base_diffs[word_lang(a or b)] += 1
            for cat, a, b in ep["trained_pairs"]:
                trained_diffs[word_lang(a or b)] += 1
        else:
            for cat, a, b in ep["diff_pairs"]:
                trained_diffs[word_lang(a or b)] += 1
    return {"totals": totals, "base_diffs": base_diffs, "trained_diffs": trained_diffs}


# ── HTML report ───────────────────────────────────────────────────────────────
def render_html(episodes: list, has_ref: bool, near_miss: float,
                 title: str, path: Path) -> None:
    blame = BLAME_REF if has_ref else BLAME_DIFF
    lang = language_breakdown(episodes, has_ref)

    # Overall category counts.
    overall_base = Counter()
    overall_trained = Counter()
    for ep in episodes:
        if has_ref:
            overall_base.update(ep["base_cats"])
            overall_trained.update(ep["trained_cats"])
        else:
            overall_trained.update(ep["diff_cats"])  # single count, "trained" col reused

    cat_rows = []
    for cat in CATEGORIES:
        fg, bg = CAT_COLOR[cat]
        b = overall_base[cat]
        t = overall_trained[cat]
        if has_ref:
            d = f"{100 * (t - b) / b:+.1f}%" if b else "—"
            cls = "up" if b and t > b else ("down" if b and t < b else "")
            cat_rows.append(
                f'<tr><th><span class="chip" style="color:{fg};background:{bg}" '
                f'data-cat="{cat}">{cat}</span></th><td>{b}</td><td><b>{t}</b></td>'
                f'<td class="{cls}">{d}</td><td class="why">{html.escape(blame[cat])}</td></tr>')
        else:
            cat_rows.append(
                f'<tr><th><span class="chip" style="color:{fg};background:{bg}" '
                f'data-cat="{cat}">{cat}</span></th><td>{t}</td>'
                f'<td class="why">{html.escape(blame[cat])}</td></tr>')

    # Overall WER/CER.
    if has_ref:
        n = len(episodes)
        avg_base_wer = sum(e["base_wer"] for e in episodes) / n if n else 0
        avg_trained_wer = sum(e["trained_wer"] for e in episodes) / n if n else 0
        avg_base_cer = sum(e["base_cer"] for e in episodes) / n if n else 0
        avg_trained_cer = sum(e["trained_cer"] for e in episodes) / n if n else 0
        wer_d = avg_trained_wer - avg_base_wer
        cer_d = avg_trained_cer - avg_base_cer
        headline = f"""<table class="headline"><thead><tr><th></th><th>base</th><th>trained</th><th>&Delta;</th></tr></thead>
<tbody>
<tr><th>WER</th><td>{avg_base_wer:.1f}%</td><td><b>{avg_trained_wer:.1f}%</b></td>
    <td class="{'down' if wer_d < 0 else 'up' if wer_d > 0 else ''}">{wer_d:+.1f}pt</td></tr>
<tr><th>CER (bare)</th><td>{avg_base_cer:.1f}%</td><td><b>{avg_trained_cer:.1f}%</b></td>
    <td class="{'down' if cer_d < 0 else 'up' if cer_d > 0 else ''}">{cer_d:+.1f}pt</td></tr>
</tbody></table>
<p class="sub">Averaged per-episode, not pooled — each episode counts equally regardless of length.
Lower is better; green = trained improved on base.</p>"""
    else:
        avg_diff = sum(e["diff_pct"] for e in episodes) / len(episodes) if episodes else 0
        headline = f"""<table class="headline"><thead><tr><th></th><th>value</th></tr></thead>
<tbody><tr><th>words differing, base&rarr;trained</th><td><b>{avg_diff:.1f}%</b></td></tr></tbody></table>
<p class="sub">No reference transcript found for these episodes, so this is a DIFFERENCE
rate between the two models' output, not an error rate — it doesn't say which one is right.</p>"""

    # Per-language table.
    lang_rows = []
    for lg in LANG_ORDER:
        tot = lang["totals"][lg]
        if tot == 0:
            continue
        t = lang["trained_diffs"][lg]
        t_rate = 100 * t / tot
        if has_ref:
            b = lang["base_diffs"][lg]
            b_rate = 100 * b / tot
            lang_rows.append(f"<tr><th>{LANG_LABEL[lg]}</th><td>{tot}</td>"
                              f"<td>{b_rate:.1f}%</td><td><b>{t_rate:.1f}%</b></td></tr>")
        else:
            lang_rows.append(f"<tr><th>{LANG_LABEL[lg]}</th><td>{tot}</td><td>{t_rate:.1f}%</td></tr>")
    ref_or_base_label = "reference" if has_ref else "base"

    def _rows_html(rows: list) -> str:
        return "".join(
            f'<div class="lbl" data-role="{lbl}">{lbl}</div>'
            f'<div class="ur" data-role="{lbl}" dir="rtl">{marked}</div>'
            for lbl, marked in rows)

    def _gap_html(n: int) -> str:
        return f'<div class="gap">⋯ {n} word{"s" if n != 1 else ""} unchanged ⋯</div>' if n > 0 else ""

    # Episode sections.
    sections = []
    def _youtube_link(name: str) -> str:
        """A YouTube id is the trailing _XXXXXXXXXXX of the audio filename, which
        is how download_playlist_audio.py names what it fetches. Linking it makes
        the report checkable: anyone reading a disagreement can open the video and
        listen to the audio that produced it, instead of taking the text on trust.
        Ids are 11 chars of [A-Za-z0-9_-]; anything else is not linked."""
        m = re.search(r"[_/]([A-Za-z0-9_-]{11})$", name)
        if not m:
            return ""
        vid = m.group(1)
        return (f'<a class="yt" href="https://www.youtube.com/watch?v={vid}" '
                f'target="_blank" rel="noopener">▶ youtube.com/watch?v={vid}</a>')

    for i, ep in enumerate(episodes):
        if has_ref:
            delta = ep["delta_wer"]
            metric = (f'WER {ep["base_wer"]:.1f}% &rarr; {ep["trained_wer"]:.1f}% '
                      f'<span class="{"down" if delta < 0 else "up" if delta > 0 else ""}">'
                      f'({delta:+.1f}pt)</span>')
            sort_key = delta
        else:
            metric = f'{ep["diff_pct"]:.1f}% of words differ'
            sort_key = ep["diff_pct"]

        hd = ep["hunk_data"]
        if hd["hunks"]:
            compact_parts = [
                _gap_html(h["gap_before"]) + f'<div class="hunk">{_rows_html(h["rows"])}</div>'
                for h in hd["hunks"]]
            compact_parts.append(_gap_html(hd["tail_gap"]))
            compact = "".join(compact_parts)
        else:
            compact = '<div class="gap">✓ identical, no differences to show</div>'
        full = _rows_html(ep["display"])

        sections.append(
            f'<div class="episode" data-order="{i}" data-delta="{sort_key}" '
            f'data-name="{html.escape(ep["name"].lower())}">'
            f'<div class="ep"><span>{html.escape(ep["name"])}'
            f'{_youtube_link(ep["name"])}</span>'
            f'<span class="pair">{metric}'
            f'<button type="button" class="toggle-full">full transcript</button></span></div>'
            f'<div class="hunks">{compact}</div>'
            f'<div class="full" style="display:none">{full}</div>'
            f'</div>')

    ref_note = ("" if has_ref else
                '<p class="sub">⚠️ Categories below describe how the two outputs '
                'differ, not which is correct — there is no reference transcript for these episodes.</p>')

    sort_options = (
        '<option value="order">episode order</option>'
        '<option value="regressed">most regressed first</option>'
        '<option value="improved">most improved first</option>'
        '<option value="changed">most changed first</option>'
        if has_ref else
        '<option value="order">episode order</option>'
        '<option value="changed">most changed first</option>'
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""<!doctype html>
<html lang="ur"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&display=swap">
<style>
 body {{ font-family: system-ui, "Segoe UI", sans-serif; max-width: 1100px;
        margin: 0 auto; padding: 32px 24px 80px; color: #16181D; background: #fff;
        line-height: 1.6; }}
 h1 {{ font-size: 1.6rem; margin: 0 0 4px; }}
 h2 {{ font-size: 1rem; margin: 30px 0 10px; }}
 .sub {{ color: #6B7280; margin: 4px 0 20px; font-size: 13px; }}
 table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; font-size: 14px; }}
 th, td {{ padding: 8px 12px; border-bottom: 1px solid #E5E7EB; text-align: right;
           font-variant-numeric: tabular-nums; }}
 th:first-child, td.why {{ text-align: left; }}
 thead th {{ font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: #6B7280; }}
 table.headline {{ max-width: 420px; }}
 td.down {{ color: #047857; font-weight: 600; }}
 td.up   {{ color: #B45309; font-weight: 600; }}
 span.down {{ color: #047857; font-weight: 600; }}
 span.up   {{ color: #B45309; font-weight: 600; }}
 .why {{ color: #6B7280; font-weight: 400; font-size: 13px; }}
 .chip {{ padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;
          font-family: ui-monospace, monospace; }}
 .legend .chip {{ cursor: pointer; border: 2px solid transparent; }}
 .legend .chip.active {{ border-color: #16181D; }}
 .toolbar {{ position: sticky; top: 0; background: #fff; padding: 12px 0;
            border-bottom: 1px solid #E5E7EB; margin: 24px 0 20px; display: flex;
            gap: 10px; align-items: center; flex-wrap: wrap; z-index: 10; }}
 .toolbar input[type=search] {{ flex: 1 1 200px; padding: 6px 10px; border: 1px solid #D1D5DB;
                                border-radius: 6px; font-size: 13px; }}
 .toolbar select {{ padding: 6px 10px; border: 1px solid #D1D5DB; border-radius: 6px; font-size: 13px; }}
 .legend {{ display: flex; gap: 6px; flex-wrap: wrap; }}
 .count {{ color: #9CA3AF; font-size: 12px; }}
 .yt {{ margin-left: 12px; font-family: ui-monospace, monospace; font-size: 11.5px;
        color: #B4232B; text-decoration: none; border-bottom: 1px solid #F0C7C9; }}
 .yt:hover {{ border-bottom-color: #B4232B; }}
 .episode {{ border: 1px solid #E5E7EB; border-radius: 6px; padding: 14px 18px; margin-bottom: 14px; }}
 .episode.hide {{ display: none; }}
 .ep {{ font-family: ui-monospace, monospace; font-size: 11.5px; color: #6B7280;
        display: flex; justify-content: space-between; gap: 16px; margin-bottom: 10px; }}
 .pair {{ font-family: system-ui, sans-serif; font-size: 12.5px; }}
 .lbl {{ font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
         color: #6B7280; margin-top: 16px; padding-top: 10px; border-top: 1px dashed #E5E7EB; }}
 .lbl:first-of-type {{ margin-top: 4px; padding-top: 0; border-top: none; }}
 .lbl[data-role="reference"] {{ color: #4B5563; }}
 .lbl[data-role="base"]      {{ color: #1D4ED8; }}
 .lbl[data-role="trained"]   {{ color: #047857; }}
 .ur {{ font-family: "Noto Nastaliq Urdu", "Jameel Noori Nastaleeq", "Urdu Typesetting", serif;
        font-size: 19px; line-height: 2.2; text-align: right; margin-top: 2px; }}
 /* margin, not just padding: two highlighted words sitting back-to-back
    (common — a dropped word right next to a misheard one) otherwise touch
    with no visible gap and read as one fused, unreadable blob. */
 .m {{ padding: 1px 4px; margin: 0 2px; border-radius: 3px; display: inline-block; }}
 .m.dim {{ opacity: .12; }}
 /* Compact diff: each hunk's reference/base/trained rows sit right on top of
    each other, so a coloured word and its counterpart on the next row are a
    glance apart, not a scroll apart. Skipped stretches collapse to .gap. */
 .hunk {{ margin: 4px 0 14px; }}
 .hunk .ur {{ line-height: 1.9; }}
 .gap {{ text-align: center; color: #9CA3AF; font-size: 11.5px; font-family: ui-monospace, monospace;
         margin: 10px 0; }}
 .full .ur {{ line-height: 2.6; }}
 .toggle-full {{ background: #F3F4F6; border: 1px solid #D1D5DB; border-radius: 4px;
                 padding: 2px 8px; font-size: 11px; cursor: pointer; color: #374151;
                 margin-left: 10px; font-family: system-ui, sans-serif; }}
 .toggle-full:hover {{ background: #E5E7EB; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="sub">{len(episodes)} episode{"s" if len(episodes) != 1 else ""} compared, base vs trained
{"against a reference transcript" if has_ref else "(no reference transcripts found)"}.
Coloured words show where {"a model differs from " + ref_or_base_label if has_ref else "the two outputs differ"}.</p>

<h2>Overall</h2>
{headline}

<h2>By category</h2>
{ref_note}
<table><thead><tr><th>category</th><th>base</th>{"<th>trained</th><th>&Delta;</th>" if has_ref else ""}<th class="why">points at</th></tr></thead>
<tbody>{"".join(cat_rows)}</tbody></table>

<h2>By language</h2>
<p class="sub">Language is guessed per word from its script (Urdu vs Arabic letter-forms, Latin = English) —
a heuristic, not a language ID model. Rate = share of that language's words which fall into some category above.</p>
<table><thead><tr><th>language</th><th>words in {ref_or_base_label}</th>{"<th>base differs</th><th>trained differs</th>" if has_ref else "<th>differs</th>"}</tr></thead>
<tbody>{"".join(lang_rows)}</tbody></table>

<div class="toolbar">
  <input type="search" id="search" placeholder="Search episode name or text…">
  <select id="sort">{sort_options}</select>
  <div class="legend" id="legend">
    {"".join(f'<span class="chip" style="color:{CAT_COLOR[c][0]};background:{CAT_COLOR[c][1]}" data-cat="{c}" title="click to isolate">{c}</span>' for c in CATEGORIES if overall_base[c] or overall_trained[c])}
  </div>
  <span class="count" id="count"></span>
</div>

<div id="episodes">{"".join(sections)}</div>

<script>
const episodes = Array.from(document.querySelectorAll('.episode'));
const search = document.getElementById('search');
const sortSel = document.getElementById('sort');
const legend = document.getElementById('legend');
const countEl = document.getElementById('count');
let isolated = null;

function updateCount() {{
  const visible = episodes.filter(e => !e.classList.contains('hide')).length;
  countEl.textContent = visible + ' / ' + episodes.length + ' episodes';
}}

function applyFilter() {{
  const q = search.value.trim().toLowerCase();
  for (const ep of episodes) {{
    const hay = ep.dataset.name + ' ' + ep.textContent.toLowerCase();
    ep.classList.toggle('hide', q.length > 0 && !hay.includes(q));
  }}
  updateCount();
}}

function applySort() {{
  const mode = sortSel.value;
  const container = document.getElementById('episodes');
  const sorted = episodes.slice().sort((a, b) => {{
    if (mode === 'order') return (+a.dataset.order) - (+b.dataset.order);
    if (mode === 'regressed') return (+b.dataset.delta) - (+a.dataset.delta);
    if (mode === 'improved') return (+a.dataset.delta) - (+b.dataset.delta);
    if (mode === 'changed') return Math.abs(+b.dataset.delta) - Math.abs(+a.dataset.delta);
    return 0;
  }});
  for (const ep of sorted) container.appendChild(ep);
}}

legend.addEventListener('click', (e) => {{
  const chip = e.target.closest('.chip');
  if (!chip) return;
  const cat = chip.dataset.cat;
  isolated = (isolated === cat) ? null : cat;
  for (const c of legend.querySelectorAll('.chip')) c.classList.toggle('active', c.dataset.cat === isolated);
  for (const m of document.querySelectorAll('.m')) {{
    m.classList.toggle('dim', isolated !== null && m.title !== isolated);
  }}
}});

document.getElementById('episodes').addEventListener('click', (e) => {{
  const btn = e.target.closest('.toggle-full');
  if (!btn) return;
  const ep = btn.closest('.episode');
  const hunks = ep.querySelector('.hunks');
  const full = ep.querySelector('.full');
  const showingFull = full.style.display !== 'none';
  full.style.display = showingFull ? 'none' : 'block';
  hunks.style.display = showingFull ? 'block' : 'none';
  btn.textContent = showingFull ? 'full transcript' : 'compact diff';
}});

search.addEventListener('input', applyFilter);
sortSel.addEventListener('change', applySort);
updateCount();
</script>
</body></html>""", encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, nargs="?",
                     default=Path("full_audio_samples/compare_transcripts"))
    ap.add_argument("--fetch", metavar="YOUTUBE_URL", default=None,
                     help="download this video's audio into --fetch-dir (via "
                          "download_playlist_audio.py) and exit — run "
                          "scripts/compare_transcribe.py on it afterwards to "
                          "get base/trained transcripts to compare")
    ap.add_argument("--fetch-dir", type=Path, default=Path("full_audio_samples"),
                     help="where --fetch saves audio (default: full_audio_samples)")
    ap.add_argument("--fetch-format", default="mp3",
                     choices=["mp3", "m4a", "wav", "opus", "flac"])
    ap.add_argument("--fetch-quality", default="192", help="audio bitrate in kbps")
    ap.add_argument("--transcribe", metavar="AUDIO_FILE", default=None,
                     help="shell out to `modal run scripts/compare_transcribe.py` "
                          "for this file (comma-separated for more than one; a "
                          "bare filename resolves against full_audio_samples/), "
                          "then exit. Runs on Modal's GPU — uses your Modal "
                          "account/credits, nothing here triggers it silently")
    ap.add_argument("--transcribe-out-subdir", default="",
                     help="subfolder of full_audio_samples/compare_transcripts/ "
                          "to write the new transcripts into (default: none)")
    ap.add_argument("--transcribe-model-path", default="",
                     help="which fine-tuned model --transcribe should use, e.g. "
                          "/data/model/whisper-urdu-r3-final. WITHOUT this the "
                          "downstream script falls back to round 1's model, which "
                          "produces a real transcript under a filename that says "
                          "nothing about which round made it")
    ap.add_argument("--base-suffix", default="_base")
    ap.add_argument("--trained-suffix", default="_finetuned")
    ap.add_argument("--ref-suffix", default="",
                     help='suffix for the reference file, default "" means "{stem}.txt"')
    ap.add_argument("--no-ref", action="store_true",
                     help="ignore reference files even if found")
    ap.add_argument("--recursive", action="store_true", default=True)
    ap.add_argument("--no-recursive", dest="recursive", action="store_false")
    ap.add_argument("--near-miss", type=float, default=0.65)
    ap.add_argument("--html", type=Path, default=Path("reports/compare_report.html"))
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.fetch:
        # Lazy import: download_playlist_audio.py exits at import time if
        # yt-dlp isn't installed, and plain comparison runs (the common case)
        # shouldn't need that dependency at all.
        from download_playlist_audio import download_playlist_audio
        print(f"⬇️  Fetching audio for {args.fetch} -> {args.fetch_dir}/")
        download_playlist_audio(args.fetch, str(args.fetch_dir),
                                 args.fetch_format, args.fetch_quality)
        print(f"\n✅ Saved to {args.fetch_dir}/")
        print("   Next: --transcribe <the downloaded filename> to generate "
              "base/trained transcripts for it, then re-run this script "
              "(no flags) to compare them.")
        return 0

    if args.transcribe:
        cmd = ["modal", "run", "scripts/compare_transcribe.py", "--audio", args.transcribe]
        if args.transcribe_out_subdir:
            cmd += ["--out-subdir", args.transcribe_out_subdir]
        if args.transcribe_model_path:
            cmd += ["--model-path", args.transcribe_model_path]
        print("▶️  " + " ".join(cmd))
        print("   Runs on Modal's A10G GPU — this uses your Modal account/credits.")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return result.returncode
        print("\n   Next: re-run this script (no flags) to compare the new transcripts.")
        return 0

    if not args.folder.exists():
        sys.exit(f"folder not found: {args.folder}")

    raw = discover(args.folder, args.base_suffix, args.trained_suffix,
                    args.ref_suffix, args.recursive, use_ref=not args.no_ref)
    if not raw:
        sys.exit(f"no *{args.base_suffix}.txt files found under {args.folder}")

    print("=" * 78)
    print(f"COMPARE TRANSCRIPTS — {len(raw)} episode(s) in {args.folder}")
    print("=" * 78)

    episodes = [analyze_episode(ep, args.near_miss) for ep in raw]
    has_ref = all(e["has_ref"] for e in episodes)
    mixed = any(e["has_ref"] for e in episodes) and not has_ref
    if mixed:
        n_ref = sum(e["has_ref"] for e in episodes)
        print(f"⚠️  {n_ref}/{len(episodes)} episodes have a reference transcript, the rest don't.")
        print("    Splitting: episodes without a reference are scored base-vs-trained only.")

    for group_has_ref, group in ((True, [e for e in episodes if e["has_ref"]]),
                                  (False, [e for e in episodes if not e["has_ref"]])):
        if not group:
            continue
        print()
        if group_has_ref:
            avg_b = sum(e["base_wer"] for e in group) / len(group)
            avg_t = sum(e["trained_wer"] for e in group) / len(group)
            print(f"  {len(group)} episode(s) WITH reference — avg WER base {avg_b:.1f}%  "
                  f"trained {avg_t:.1f}%  ({avg_t - avg_b:+.1f}pt)")
        else:
            avg_d = sum(e["diff_pct"] for e in group) / len(group)
            print(f"  {len(group)} episode(s) WITHOUT reference — avg {avg_d:.1f}% of words differ")
        for e in sorted(group, key=lambda e: e.get("delta_wer", e.get("diff_pct", 0)), reverse=True):
            if group_has_ref:
                print(f"    {e['name']:<40} base {e['base_wer']:6.1f}%  trained {e['trained_wer']:6.1f}%"
                      f"  ({e['delta_wer']:+.1f}pt)")
            else:
                print(f"    {e['name']:<40} {e['diff_pct']:6.1f}% differ")

    # Reports need one consistent mode; render ref-scored and diff-only
    # episodes as two separate reports if the folder was mixed rather than
    # forcing a false comparison into one table.
    title = args.title or f"Transcript comparison — {args.folder.name}"
    if mixed:
        ref_path = args.html.with_stem(args.html.stem + "_scored")
        diff_path = args.html.with_stem(args.html.stem + "_unscored")
        render_html([e for e in episodes if e["has_ref"]], True, args.near_miss,
                     title + " (scored)", ref_path)
        render_html([e for e in episodes if not e["has_ref"]], False, args.near_miss,
                     title + " (no reference)", diff_path)
        print(f"\n📄 {ref_path}\n📄 {diff_path}")
    else:
        render_html(episodes, has_ref, args.near_miss, title, args.html)
        print(f"\n📄 {args.html}")
    print("   Open in a browser — the console cannot shape Urdu/Arabic script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())