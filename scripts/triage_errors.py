r"""Sort transcription errors into SPELLED WRONG vs HEARD WRONG.

WHY THIS EXISTS
---------------
Round 2 improved CER about twice as much as WER, and trained the decoder only
(the encoder's 192 LoRA modules never received a gradient — see
`scripts/inspect_adapter.py`). The open question for round 3 is which lever to
pull: more epochs on the decoder, or finally training the encoder.

That is answerable from data instead of theory. The model has two halves — the
encoder hears, the decoder writes — so errors split the same way:

  * the model HEARD right and WROTE the wrong letter  -> decoder problem
  * the model HEARD something else entirely           -> encoder problem

If round 2 fixed mostly the first kind and barely touched the second, the
remaining headroom is acoustic and the encoder is the lever. If the mis-heard
pile is small, it is not.

WHY NOT JUST CHARACTER SIMILARITY
---------------------------------
Because Urdu breaks it. Several letters sound IDENTICAL but are written
differently — س ص ث are all /s/, ز ذ ض ظ are all /z/, ت and ط are both /t/. Writing
ص for س means the model heard perfectly and spelled wrong: a DECODER error, and
exactly what the reviewers were correcting.

But ٹ vs ت, ڈ vs د, ڑ vs ر are genuinely different sounds (retroflex vs dental).
Confusing those is real mis-hearing: an ENCODER error.

A similarity score calls both "one character off" and tells you nothing. So
same-sound letters are collapsed BEFORE comparing, and the retroflex/dental and
aspirated (ھ) distinctions are deliberately preserved.

    python scripts/triage_errors.py data/eval
    python scripts/triage_errors.py data/eval --source primary --samples 8

⚠️ The categories are a HEURISTIC, not a diagnosis: the mis-heard pile will hold
some genuine language-model failures. Read the printed samples — a human who
reads Urdu is the ground truth here, the counts are only the first cut.
"""
import argparse
import html
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

# Urdu text on a cp1252 Windows console raises UnicodeEncodeError and kills the run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FILES = {
    "base": "eval_predictions-r2-base.json",
    "r1": "eval_predictions-r2-whisper-urdu-final.json",
    "r2": "eval_predictions-r2-whisper-urdu-r2-final.json",
    "r3": "eval_predictions-r3-whisper-urdu-r3-final.json",
}
# The newest model: the one whose remaining errors are being triaged, and the one
# samples are drawn from. The column before it is the control it is judged against.
LATEST, CONTROL = "r3", "r2"

# ── Letters that SOUND THE SAME in Urdu but are written differently ──────────
# Collapsing these turns "wrote the wrong letter for the right sound" into an
# exact match, which is what makes it detectable as a spelling error.
# Two DIFFERENT phenomena, kept apart because they mean different things.
#
# 1. SCRIPT_VARIANT — the SAME letter written in its Arabic vs its Urdu form
#    (ه/ہ, ك/ک, ي/ی, أ إ ٱ/ا). الله vs اللہ is not a mistake, it is a house-style
#    choice: reviewers preserve Arabic orthography inside Quranic quotations and
#    the model Urdu-ises it. Fixing it is a data-consistency decision.
SCRIPT_VARIANT = [
    "اآأإٱ",      # alef forms
    "یيى",        # yeh: Urdu/Arabic/alef-maksura codepoints of one letter.
                  # ے (bari ye) and ئ are NOT folded in: کے / کی / کا are
                  # different words, and folding them scored a grammar error as
                  # a spelling error. Found by reading the samples.
    "کك",         # kaf
    "ہهۃة",       # he: Urdu ہ vs Arabic ه — same letter, different codepoint
    "وؤ",         # waw + hamza-on-waw
]

# 2. TRUE_HOMOPHONE — DIFFERENT letters that sound identical in Urdu. Writing
#    one for the other is a real spelling error: the model heard the sound
#    correctly and chose the wrong letter. کثرت -> کسرت (ث for س) is the
#    canonical case, and exactly what the reviewers were correcting.
TRUE_HOMOPHONE = [
    "سصث",        # /s/  — کثرت / کسرت
    "زذضظ",       # /z/
    "تط",         # /t/  — تائب / طائب. ٹ EXCLUDED: retroflex is a different sound
    "حہ",         # /h/  — applied AFTER the script fold, so ه is already ہ.
                  #        ھ EXCLUDED: it marks aspiration (کھ != ک)
    "نں",         # noon vs noon-ghunna
]

# عا and آ are both long /aa/ — عاجزی / آجزی and عام / آم are homophone pairs, so
# choosing between them is spelling, not hearing. Added after a reviewer who reads
# Urdu supplied عاجزی -> آجزی, which was scoring as an unexplained near-miss.
#
# Deliberately NOT a blanket ع -> ا. That merged علم (ilm) with الم (alam) —
# different words with different vowels — and the regression check caught it. Only
# ع IMMEDIATELY BEFORE alef is folded; a bare ع keeps its identity.
# Applied after the script fold, so آ has already become ا.
_AIN_ALEF = re.compile(r"عا")
# Deliberately left out — enable if you disagree:
#   "قک"  ق and ک merge to /k/ for many speakers, but they are distinct letters
#         and merging them would hide a real substitution class.

_SCRIPT_FOLD, _HOMO_FOLD = {}, {}
for _cls in SCRIPT_VARIANT:
    for _ch in _cls:
        _SCRIPT_FOLD[_ch] = _cls[0]
for _cls in TRUE_HOMOPHONE:
    for _ch in _cls:
        _HOMO_FOLD[_ch] = _cls[0]

# Harakat/diacritics, superscript alef, and the zero-width joiners. Purely
# orthographic: their presence or absence never changes what was heard.
_STRIP = re.compile(r"[ً-ْٰـ‌‍­]")
# Copied verbatim from modal_app.py:678 so this script tokenises exactly the way
# the reported WER/CER did. An approximation here would count the Urdu comma ،
# and full stop ۔ as errors and inflate every category.
_PUNCT = re.compile(r"[۔،؛؟!?.,:;\"'“”‘’()\-—…]")


def words(text: str) -> list:
    """Same punctuation handling as modal_app.py::evaluate's normalizer."""
    text = _PUNCT.sub(" ", unicodedata.normalize("NFC", text or ""))
    return text.split()


def bare(word: str) -> str:
    """Diacritics and joiners removed — still distinguishes ت from ط."""
    return _STRIP.sub("", word)


def script_key(word: str) -> str:
    """Bare form with Arabic/Urdu codepoint variants of the SAME letter unified.

    Two words with the same script_key are the same word spelled the same way —
    only the script convention differs.
    """
    return "".join(_SCRIPT_FOLD.get(c, c) for c in bare(word))


def ortho_key(word: str) -> str:
    """script_key with same-sound but DIFFERENT letters also collapsed.

    Two words that share this key but not script_key were pronounced identically
    and spelled differently — a genuine spelling error.
    """
    folded = _AIN_ALEF.sub("ا", script_key(word))
    return "".join(_HOMO_FOLD.get(c, c) for c in folded)


def classify(ref: str, hyp: str, near_miss: float) -> str:
    # Diacritics FIRST, and as their own category. Reviewers write the vowel
    # marks (اِس, اَنا, سِری) and the model omits them; lumping that in with real
    # letter errors made "spelling" look like the biggest problem when most of it
    # is optional vowel notation. Whether the model should reproduce these at all
    # is a decision, not a defect.
    if bare(ref) == bare(hyp):
        return "diacritic"
    if script_key(ref) == script_key(hyp):
        return "script_variant"
    if ortho_key(ref) == ortho_key(hyp):
        return "spelling"
    if SequenceMatcher(None, bare(ref), bare(hyp)).ratio() >= near_miss:
        return "near_miss"
    return "misheard"


CATEGORIES = ["diacritic", "script_variant", "spelling", "near_miss",
              "misheard", "dropped", "inserted"]

# Diacritics are NOT scored: these transcripts are for subtitles and search,
# where vowel marks carry nothing for a reader (decision, 2026-08-28 — see
# scripts/rescore.py). Still counted and displayed, because "the model
# under-produces harakat" stays worth knowing; just excluded from the totals so
# the remaining-error split reflects what actually gets measured.
UNSCORED = {"diacritic"}
SCORED = [c for c in CATEGORIES if c not in UNSCORED]
BLAME = {
    "diacritic":      "cosmetic — vowel marks only, no letter differs",
    "script_variant": "convention — Arabic vs Urdu form of the SAME letter (الله/اللہ)",
    "spelling":       "DECODER — right sound, wrong letter (کثرت/کسرت)",
    "near_miss":      "ambiguous — morphology / izafat / word form",
    "misheard":       "ENCODER — heard a different word",
    "dropped":        "acoustic or segmentation — word missing",
    "inserted":       "acoustic or segmentation — word invented",
}


def diff_clip(reference: str, hypothesis: str, near_miss: float):
    """Yields (category, ref_word, hyp_word) for every error in one clip."""
    r, h = words(reference), words(hypothesis)
    # Align on the RAW words. Aligning on ortho_key would fold same-sound spellings
    # into equal elements, so every orthographic error would be scored as a match
    # and silently disappear instead of being classified — which is exactly the
    # category this script exists to count.
    sm = SequenceMatcher(None, r, h, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "delete":
            for w in r[i1:i2]:
                yield "dropped", w, ""
        elif op == "insert":
            for w in h[j1:j2]:
                yield "inserted", "", w
        else:  # replace — pair positionally, surplus counts as dropped/inserted
            rs, hs = r[i1:i2], h[j1:j2]
            for a, b in zip(rs, hs):
                yield classify(a, b, near_miss), a, b
            for w in rs[len(hs):]:
                yield "dropped", w, ""
            for w in hs[len(rs):]:
                yield "inserted", "", w


# ── HTML report ─────────────────────────────────────────────────────────────
# The Windows console cannot shape Arabic script: even with UTF-8 output it draws
# isolated letter forms in the wrong visual order, so the samples are unreadable
# exactly where a human needs to check them. A browser does RTL, shaping and
# Nastaliq correctly, so the reviewable output goes to HTML.
CAT_COLOR = {
    "diacritic":      ("#8A8F98", "#EEF0F2"),
    "script_variant": ("#0E7490", "#CFFAFE"),
    "spelling":       ("#1D4ED8", "#DBEAFE"),
    "near_miss":      ("#7C3AED", "#EDE9FE"),
    "misheard":       ("#B91C1C", "#FEE2E2"),
    "dropped":        ("#B45309", "#FEF3C7"),
    "inserted":       ("#047857", "#D1FAE5"),
}


def mark(word: str, cat: str) -> str:
    fg, bg = CAT_COLOR[cat]
    return (f'<span class="m" style="color:{fg};background:{bg}" '
            f'title="{cat}">{html.escape(word)}</span>')


def marked_pair(reference: str, hypothesis: str, near_miss: float):
    """Both sides of one clip, every differing word wrapped and colour-coded."""
    r, h = words(reference), words(hypothesis)
    sm = SequenceMatcher(None, r, h, autojunk=False)
    ref_out, hyp_out = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            ref_out += [html.escape(w) for w in r[i1:i2]]
            hyp_out += [html.escape(w) for w in h[j1:j2]]
        elif op == "delete":
            ref_out += [mark(w, "dropped") for w in r[i1:i2]]
        elif op == "insert":
            hyp_out += [mark(w, "inserted") for w in h[j1:j2]]
        else:
            rs, hs = r[i1:i2], h[j1:j2]
            for a, b in zip(rs, hs):
                cat = classify(a, b, near_miss)
                ref_out.append(mark(a, cat))
                hyp_out.append(mark(b, cat))
            ref_out += [mark(w, "dropped") for w in rs[len(hs):]]
            hyp_out += [mark(w, "inserted") for w in hs[len(rs):]]
    return " ".join(ref_out), " ".join(hyp_out)


def write_html(path: Path, scope: str, counts: dict, totals: dict,
               samples: dict, near_miss: float) -> None:
    rows = []
    for cat in CATEGORIES:
        b, r1 = counts["base"][cat], counts["r1"][cat]
        ctl, new = counts[CONTROL][cat], counts[LATEST][cat]
        d = f"{100 * (new - ctl) / ctl:+.1f}%" if ctl else "—"
        cls = "up" if ctl and new > ctl else ("down" if ctl and new < ctl else "")
        fg, bg = CAT_COLOR[cat]
        rows.append(
            f'<tr><th><span class="chip" style="color:{fg};background:{bg}">{cat}</span></th>'
            f"<td>{b}</td><td>{r1}</td><td>{ctl}</td><td><b>{new}</b></td>"
            f'<td class="{cls}">{d}</td><td class="why">{html.escape(BLAME[cat])}</td></tr>')

    blocks = []
    for cat in CATEGORIES:
        if not samples.get(cat):
            continue
        fg, bg = CAT_COLOR[cat]
        items = []
        for ep, rw, hw, ref, hyp in samples[cat]:
            rh, hh = marked_pair(ref, hyp, near_miss)
            items.append(
                f'<div class="clip"><div class="ep">{html.escape(str(ep))}'
                f'<span class="pair">{html.escape(rw or "—")} '
                f'&rarr; {html.escape(hw or "—")}</span></div>'
                f'<div class="lbl">reviewer</div><div class="ur" dir="rtl">{rh}</div>'
                f'<div class="lbl">model</div><div class="ur" dir="rtl">{hh}</div></div>')
        blocks.append(
            f'<section><h2><span class="chip" style="color:{fg};background:{bg}">{cat}</span>'
            f'<span class="why">{html.escape(BLAME[cat])}</span></h2>{"".join(items)}</section>')

    path.write_text(f"""<!doctype html>
<html lang="ur"><head><meta charset="utf-8">
<title>Error triage{html.escape(scope)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&display=swap">
<style>
 body {{ font-family: system-ui, "Segoe UI", sans-serif; max-width: 1100px;
        margin: 0 auto; padding: 32px 24px 80px; color: #16181D; background: #fff;
        line-height: 1.6; }}
 h1 {{ font-size: 1.6rem; margin: 0 0 4px; }}
 .sub {{ color: #6B7280; margin: 0 0 28px; }}
 table {{ border-collapse: collapse; width: 100%; margin-bottom: 40px; font-size: 14px; }}
 th, td {{ padding: 8px 12px; border-bottom: 1px solid #E5E7EB; text-align: right;
           font-variant-numeric: tabular-nums; }}
 th:first-child, td.why {{ text-align: left; }}
 thead th {{ font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
             color: #6B7280; }}
 td.down {{ color: #047857; font-weight: 600; }}
 td.up   {{ color: #B45309; font-weight: 600; }}
 .why {{ color: #6B7280; font-weight: 400; font-size: 13px; }}
 .chip {{ padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;
          font-family: ui-monospace, monospace; }}
 h2 {{ font-size: 1rem; margin: 36px 0 12px; display: flex; gap: 12px;
       align-items: baseline; }}
 .clip {{ border: 1px solid #E5E7EB; border-radius: 6px; padding: 14px 18px;
          margin-bottom: 14px; }}
 .ep {{ font-family: ui-monospace, monospace; font-size: 11.5px; color: #6B7280;
        display: flex; justify-content: space-between; gap: 16px; margin-bottom: 10px; }}
 .pair {{ font-family: "Noto Nastaliq Urdu", serif; font-size: 15px; color: #16181D;
          direction: rtl; }}
 .lbl {{ font-size: 10.5px; letter-spacing: .07em; text-transform: uppercase;
         color: #9CA3AF; margin-top: 8px; }}
 .ur {{ font-family: "Noto Nastaliq Urdu", "Jameel Noori Nastaleeq",
        "Urdu Typesetting", serif; font-size: 19px; line-height: 2.6;
        text-align: right; }}
 .m {{ padding: 1px 4px; border-radius: 3px; }}
</style></head><body>
<h1>Error triage{html.escape(scope)}</h1>
<p class="sub">Round 3 vs round 2 vs round 1 vs base, same {totals['clips']} clips.
Samples below are <b>round 3's</b> remaining errors. The delta column is
round&nbsp;2&rarr;round&nbsp;3, the control comparison: both resumed round 1, so it
isolates the encoder. Coloured words are where the model differs from the
reviewer. The console cannot render Urdu — read it here.</p>
<table><thead><tr><th>category</th><th>base</th><th>round 1</th><th>round 2</th>
<th>round 3</th><th>r2&rarr;r3</th><th class="why">points at</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
{''.join(blocks)}
</body></html>""", encoding="utf-8")


def load(folder: Path) -> dict:
    out = {}
    for name, filename in FILES.items():
        path = folder / filename
        if not path.exists():
            sys.exit(f"missing {path}\n"
                     f"  modal volume get whisper-training-vol logs/{filename} .\\{path}")
        blob = json.loads(path.read_text(encoding="utf-8"))
        if not blob.get("labels_aligned"):
            print(f"⚠️  {filename}: labels_aligned is false — episode/source/bucket "
                  "labels are absent, so slicing will not work.")
        key = "base" if name == "base" else "finetuned"
        rows = {}
        for c in blob["clips"]:
            if key not in c:
                sys.exit(f"{filename}: clip {c['i']} has no '{key}' prediction")
            rows[c["i"]] = c
        out[name] = (rows, key)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, nargs="?", default=Path("data/eval"))
    ap.add_argument("--source", choices=["primary", "eval_only"], default=None,
                    help="primary = Set B (new batch); eval_only = Set A (round 1)")
    ap.add_argument("--bucket", default=None,
                    help="nastaliq_only | code_switch | spiritual_term")
    ap.add_argument("--near-miss", type=float, default=0.65,
                    help="similarity at or above which a substitution is 'near miss'")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--html", type=Path, default=None, metavar="PATH",
                    help="write a browser-readable report (Urdu renders correctly "
                         "there; a Windows console cannot shape Arabic script)")
    args = ap.parse_args()

    data = load(args.folder)
    idx = sorted(data["r2"][0])

    # The three files must describe the SAME clips, or nothing below compares.
    for name, (rows, _) in data.items():
        if sorted(rows) != idx:
            sys.exit(f"{name}: clip indices differ from r2 — not the same eval split")
        for i in idx:
            if rows[i]["reference"] != data["r2"][0][i]["reference"]:
                sys.exit(f"{name}: reference text differs at clip {i}")

    selected = []
    for i in idx:
        row = data["r2"][0][i]
        if args.source and row.get("source") != args.source:
            continue
        if args.bucket and args.bucket not in (row.get("buckets") or []):
            continue
        selected.append(i)

    scope = []
    if args.source:
        scope.append("Set B (new batch)" if args.source == "primary" else "Set A (round 1)")
    if args.bucket:
        scope.append(args.bucket)
    print("=" * 78)
    print(f"ERROR TRIAGE — {len(selected)} clips" + (f"  [{', '.join(scope)}]" if scope else ""))
    print("=" * 78)
    if not selected:
        print("no clips match that filter")
        return 1

    counts = {m: Counter() for m in data}
    # Collect every example, then spread the printed sample across the whole eval
    # set. Taking the first N walks index order, so all of them came from the
    # first clip of the first episode — which made one clip's quirk look like the
    # whole category.
    pool = defaultdict(list)
    for i in selected:
        for m, (rows, key) in data.items():
            row = rows[i]
            for cat, rw, hw in diff_clip(row["reference"], row[key], args.near_miss):
                counts[m][cat] += 1
                if m == LATEST:
                    pool[cat].append((row.get("episode", "?"), rw, hw,
                                      row["reference"], row[key]))
    samples = {}
    for cat, items in pool.items():
        if args.samples <= 0 or not items:
            samples[cat] = []
            continue
        step = max(1, len(items) // args.samples)
        samples[cat] = items[::step][:args.samples]

    print()
    print(f"  {'category':<15}{'base':>9}{'round 1':>10}{'round 2':>10}{'round 3':>10}"
          f"{'r2→r3':>10}   what it points at")
    print("  " + "-" * 118)
    for cat in CATEGORIES:
        b, r1 = counts["base"][cat], counts["r1"][cat]
        ctl, new = counts[CONTROL][cat], counts[LATEST][cat]
        # The delta that matters is against the CONTROL, not against round 1.
        # Round 2 and round 3 both resumed round 1, so r2→r3 isolates the encoder.
        delta = f"{100 * (new - ctl) / ctl:+.1f}%" if ctl else "—"
        note = "  [NOT SCORED] " if cat in UNSCORED else "   "
        print(f"  {cat:<15}{b:>9}{r1:>10}{ctl:>10}{new:>10}{delta:>10}{note}{BLAME[cat]}")
    tot = {m: sum(counts[m][c] for c in SCORED) for m in counts}
    d = (f"{100 * (tot[LATEST] - tot[CONTROL]) / tot[CONTROL]:+.1f}%"
         if tot[CONTROL] else "—")
    print("  " + "-" * 118)
    print(f"  {'TOTAL':<15}{tot['base']:>9}{tot['r1']:>10}{tot[CONTROL]:>10}"
          f"{tot[LATEST]:>10}{d:>10}")

    # The decision number: of what the newest model still gets wrong, how much is
    # the encoder's fault (misheard + dropped) vs the decoder's (orthographic)?
    enc = counts[LATEST]["misheard"] + counts[LATEST]["dropped"]
    dec = counts[LATEST]["spelling"]
    amb = counts[LATEST]["near_miss"] + counts[LATEST]["inserted"]
    conv = counts[LATEST]["script_variant"]
    if tot[LATEST]:
        print()
        print(f"  REMAINING SCORED ERRORS IN {LATEST.upper()}, "
              "by what would actually fix them")
        print(f"    convention (Arabic/Urdu){conv:>6}  {100*conv/tot[LATEST]:>5.1f}%"
              "   ← a DATA decision, not a model fault")
        print(f"    decoder (real spelling) {dec:>6}  {100*dec/tot[LATEST]:>5.1f}%"
              "   ← more decoder training")
        print(f"    encoder (mis-hearing)   {enc:>6}  {100*enc/tot[LATEST]:>5.1f}%"
              "   ← encoder training")
        print(f"    ambiguous               {amb:>6}  {100*amb/tot[LATEST]:>5.1f}%")
        print(f"    ({counts[LATEST]['diacritic']} diacritic-only differences excluded "
              "— no longer scored)")
        # This round's hypothesis, stated as a number rather than a direction.
        enc_ctl = counts[CONTROL]["misheard"] + counts[CONTROL]["dropped"]
        if enc_ctl:
            print()
            print(f"  ★ MIS-HEARING: {enc_ctl} in {CONTROL} → {enc} in {LATEST}"
                  f"  ({100 * (enc - enc_ctl) / enc_ctl:+.1f}%)"
                  "   ← the question this round exists to answer")

    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        write_html(args.html, f"  [{', '.join(scope)}]" if scope else "",
                   counts, {"clips": len(selected)}, samples, args.near_miss)
        print()
        print(f"  📄 Browser report written to {args.html}")
        print("     Open it to read the Urdu — the console cannot shape Arabic script.")
        return 0

    print()
    print("=" * 78)
    print(f"SAMPLES from {LATEST} — check these by eye; the counts are only a first cut")
    print("=" * 78)
    print("⚠️  A Windows console cannot render Urdu correctly. Use --html for a")
    print("    readable report, or pipe this to a file and open it in an editor.")
    for cat in CATEGORIES:
        if not samples[cat]:
            continue
        print(f"\n── {cat}  ({BLAME[cat]})")
        for ep, rw, hw, ref, hyp in samples[cat]:
            print(f"   [{ep}]  reference: {rw or '—'}   model: {hw or '—'}")
            print(f"      ref: {ref}")
            print(f"      out: {hyp}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
