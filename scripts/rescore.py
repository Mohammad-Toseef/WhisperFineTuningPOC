r"""Recompute WER/CER from saved predictions, under a chosen normalisation.

WHY THIS EXISTS
---------------
The transcripts are for subtitles and search, where Urdu vowel marks (harakat)
carry no value to a reader. But the scoring normaliser only stripped punctuation,
so a reviewer's اُحد against the model's احد — a single invisible U+064F — counted
as a WHOLE substituted word in WER. Measured on the real eval set, that inflated
round 2's Set B WER by 1.80 points, roughly a quarter of its remaining errors.

That is a scoring decision, not a model defect, and it costs no GPU to revisit:
`evaluate()` saved every prediction, so any normaliser can be applied after the
fact.

TWO NORMALISERS ARE REPORTED, DELIBERATELY:

  legacy  punctuation only        — what round 1's published 10.50% and the model
                                    card were measured with. Kept so historical
                                    figures stay checkable and comparable.
  bare    punctuation + diacritics — the subtitle/search-relevant number.

Silently redefining "WER" would have made every future figure incomparable to the
published ones while looking identical. Reporting both costs one column.

    python scripts/rescore.py data/eval
    python scripts/rescore.py data/eval --buckets
"""
import argparse
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Verbatim from modal_app.py:678 — must match, or these are different metrics.
_PUNCT = re.compile(r"[۔،؛؟!?.,:;\"'“”‘’()\-—…]")
# Harakat, superscript alef, tatweel. NOT hamza or the letters themselves.
_DIACRITIC = re.compile(r"[ً-ْٰـ]")

MODELS = {
    "base": ("eval_predictions-r2-base.json", "base"),
    "round 1": ("eval_predictions-r2-whisper-urdu-final.json", "finetuned"),
    "round 2": ("eval_predictions-r2-whisper-urdu-r2-final.json", "finetuned"),
}
BUCKETS = ["nastaliq_only", "code_switch", "spiritual_term"]
SOURCES = {"primary": "Set B (new batch)", "eval_only": "Set A (round 1)"}


def norm(text: str, bare: bool) -> str:
    text = _PUNCT.sub(" ", text or "")
    if bare:
        text = _DIACRITIC.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a, b) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(refs, hyps, bare: bool):
    """WER and CER as edit distance over the reference length — the definition
    `evaluate`'s wer/cer metrics implement. Verified against the reported run:
    reproduces all twelve published figures exactly."""
    we = wn = ce = cn = 0
    for r, h in zip(refs, hyps):
        nr, nh = norm(r, bare), norm(h, bare)
        rw, hw = nr.split(), nh.split()
        we += levenshtein(rw, hw); wn += len(rw)
        ce += levenshtein(list(nr), list(nh)); cn += len(nr)
    return (100 * we / wn if wn else 0.0), (100 * ce / cn if cn else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, nargs="?", default=Path("data/eval"))
    ap.add_argument("--buckets", action="store_true",
                    help="also break down by source x bucket")
    args = ap.parse_args()

    loaded = {}
    for name, (fn, key) in MODELS.items():
        p = args.folder / fn
        if not p.exists():
            sys.exit(f"missing {p}\n  modal volume get whisper-training-vol "
                     f"logs/{fn} .\\{p}")
        blob = json.loads(p.read_text(encoding="utf-8"))
        loaded[name] = [(c["reference"], c[key], c.get("source"),
                         c.get("buckets") or []) for c in blob["clips"]]

    n_clips = len(next(iter(loaded.values())))
    print("=" * 92)
    print(f"RESCORED FROM SAVED PREDICTIONS — {n_clips} clips, no GPU")
    print("  legacy = punctuation stripped (round 1's published basis)")
    print("  bare   = punctuation + diacritics stripped (subtitles / search)")
    print("=" * 92)

    def block(title, keep):
        rows = []
        for name, clips in loaded.items():
            sel = [(r, h) for r, h, s, b in clips if keep(s, b)]
            if not sel:
                return
            refs = [r for r, _ in sel]
            hyps = [h for _, h in sel]
            lw, lc = score(refs, hyps, bare=False)
            bw, bc = score(refs, hyps, bare=True)
            rows.append((name, len(sel), lw, bw, lc, bc))
        print(f"\n  {title}   (n={rows[0][1]})")
        print(f"  {'model':<10}{'WER legacy':>12}{'WER bare':>10}{'Δ':>8}"
              f"{'CER legacy':>13}{'CER bare':>10}{'Δ':>8}")
        print("  " + "-" * 73)
        for name, _, lw, bw, lc, bc in rows:
            print(f"  {name:<10}{lw:>12.2f}{bw:>10.2f}{bw-lw:>8.2f}"
                  f"{lc:>13.2f}{bc:>10.2f}{bc-lc:>8.2f}")

    block("ALL CLIPS", lambda s, b: True)
    for src, label in SOURCES.items():
        block(label, lambda s, b, _s=src: s == _s)
    if args.buckets:
        for src, label in SOURCES.items():
            for bk in BUCKETS:
                block(f"{label} / {bk}",
                      lambda s, b, _s=src, _b=bk: s == _s and _b in b)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
