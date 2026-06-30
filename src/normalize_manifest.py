"""
Normalize transcripts in a manifest.json for Whisper fine-tuning.

Rules applied (in order):
  1. ﷺ  (U+FDFA) → صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ
  2. ؐ  (U+0610) → صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ   (combining form, same expansion)
  3. ؑ  (U+0611) → عَلَیْہِ السَّلَام
  4. ؓ  (U+0613) → رَضِیَ اللَّهُ عَنْہُ
  5. U+200C (ZWNJ) → removed
  6. Collapse any double spaces introduced by expansions
  7. Remove space(s) before Urdu fullstop ۔   e.g. "کہا ۔" → "کہا۔"
  8. Add space after Urdu comma ، when missing  e.g. "کہا،اور" → "کہا، اور"

Everything else — all Arabic/Urdu diacritics (harakat), بھئی, curly quotes,
em dash, Arabic ي/ك inside Quranic text — is left untouched.

Run:
  python src/normalize_manifest.py --manifest data/processed/Batch1_EP10/manifest.json
  python src/normalize_manifest.py --manifest data/processed/Batch1_EP10/manifest.json --inplace
"""
import re
import json
import argparse
import sys
from pathlib import Path
from copy import deepcopy

sys.stdout.reconfigure(encoding="utf-8")

# ── Expansion strings ──────────────────────────────────────────────────────────
SALAWAT     = "صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ"
ALAYHISSALAM = "عَلَیْہِ السَّلَام"
RADIALLAHU  = "رَضِیَ اللَّهُ عَنْہُ"

# Each rule: (label, pattern, replacement)
# Patterns wrap the target in optional surrounding whitespace so the expansion
# slots in naturally and we can collapse any resulting double spaces afterward.
RULES = [
    # ﷺ  single-codepoint ligature — always surrounded by word chars or spaces
    ("U+FDFA ﷺ",  re.compile(r"ﷺ"),  f" {SALAWAT} "),
    # Combining honorific marks — sit on the preceding base letter with no space
    ("U+0610 ؐ",  re.compile(r"ؐ"),  f" {SALAWAT} "),
    ("U+0611 ؑ",  re.compile(r"ؑ"),  f" {ALAYHISSALAM} "),
    ("U+0613 ؓ",  re.compile(r"ؓ"),  f" {RADIALLAHU} "),
    # Zero-width non-joiner
    ("U+200C ZWNJ", re.compile(r"‌"), ""),
    # Issue 5b — space(s) before Urdu fullstop
    ("Issue 5b: space before ۔", re.compile(r" +۔"), "۔"),
    # Issue 5a — missing space after Urdu comma (only when next char is a non-space)
    ("Issue 5a: no space after ،", re.compile(r"،(\S)"), r"، \1"),
]

_MULTI_SPACE = re.compile(r"  +")


def normalize(text: str) -> tuple[str, list[str]]:
    """Return (normalized_text, list_of_rule_labels_that_fired)."""
    fired = []
    for label, pattern, replacement in RULES:
        new_text, n = pattern.subn(replacement, text)
        if n:
            fired.append(f"{label} x{n}")
            text = new_text
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text, fired


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize manifest.json transcripts")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument(
        "--inplace", action="store_true",
        help="Overwrite manifest.json (default: write to manifest_normalized.json beside it)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print changes without writing any file",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f"File not found: {manifest_path}")

    with manifest_path.open(encoding="utf-8") as f:
        data = json.load(f)

    rule_totals: dict[str, int] = {}
    changed_entries = 0
    changed_log: list[dict] = []

    normalized = deepcopy(data)
    for i, entry in enumerate(normalized):
        original = entry["transcript"]
        new_text, fired = normalize(original)
        if fired:
            changed_entries += 1
            entry["transcript"] = new_text
            for rule_hit in fired:
                label = rule_hit.rsplit(" x", 1)[0]
                count = int(rule_hit.rsplit(" x", 1)[1])
                rule_totals[label] = rule_totals.get(label, 0) + count
            changed_log.append({
                "index": i,
                "audio": entry.get("audio_path", ""),
                "rules": fired,
                "before": original,
                "after": new_text,
            })

    # ── Report ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Manifest:  {manifest_path}")
    print(f"Entries:   {len(data)}")
    print(f"Changed:   {changed_entries}")
    print()
    print("Rule hits:")
    for label, total in rule_totals.items():
        print(f"  {label:<35} {total:>4}")
    print()

    if changed_log:
        print("Changed entries (before / after):")
        for rec in changed_log:
            print(f"  [{rec['index']:>3}] {rec['audio'].split(chr(92))[-1]}")
            print(f"        rules : {', '.join(rec['rules'])}")
            print(f"        before: {rec['before'][:120]}")
            print(f"        after : {rec['after'][:120]}")
            print()

    if args.dry_run:
        print("Dry-run — no file written.")
        return

    if args.inplace:
        out_path = manifest_path
    else:
        out_path = manifest_path.with_name("manifest_normalized.json")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)

    print(f"Written → {out_path}")


if __name__ == "__main__":
    main()
