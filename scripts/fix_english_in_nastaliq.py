#!/usr/bin/env python3
"""
Fix English words written in Urdu/Nastaliq script in manifest_normalized.json.

Per the CLAUDE.md transcription convention:
  - English words  → English script  (handled by frozen base Whisper)
  - Urdu words     → Nastaliq script  (learned by LoRA)
  - Arabic phrases → Arabic script   (learned by LoRA)

Creates manifest_normalized_v2.json. Does NOT overwrite the original.
"""

import json
from pathlib import Path

INPUT_PATH  = Path("data/processed/Batch1_EP23/manifest_normalized.json")
OUTPUT_PATH = Path("data/processed/Batch1_EP23/manifest_normalized_v2.json")

# ──────────────────────────────────────────────────────────────────────────────
# Replacement map: Urdu/Nastaliq spelling → correct English spelling
#
# ORDER IS CRITICAL:
#   1. Multi-word phrases come first (longest / most specific first).
#   2. Single words follow.
# This prevents partial constituent-word matches from firing before the
# full phrase is handled.
# ──────────────────────────────────────────────────────────────────────────────
REPLACEMENTS = [

    # ── Multi-word phrases ────────────────────────────────────────────────────

    # Geographical / directional
    ("ساؤت ایسٹ ایشیا", "South East Asia"),
    ("میڈل ایسٹ",       "Middle East"),
    ("فار ایسٹ",        "Far East"),

    # Cricket-specific phrases
    ("لوہر میڈل آرڈر",  "lower middle order"),
    ("لوور میڈل آرڈر",  "lower middle order"),
    ("اوپننگ پیئر",     "opening pair"),
    ("فاسٹ بولر",       "fast bowler"),
    ("بولنگ اٹیک",      "bowling attack"),
    ("بیٹنگ آرڈر",      "batting order"),
    ("میڈل آرڈر",       "middle order"),
    ("ٹیل اینڈرز",      "tail enders"),
    ("ٹیل انڈر",        "tail ender"),
    ("ون ڈاؤن",         "one down"),
    ("ال بی",           "LBW"),

    # Science / biology
    ("فیمیل کرومسومز",  "female chromosomes"),
    ("میل کرومسومز",    "male chromosomes"),
    ("وائی کرومسومز",   "Y chromosomes"),
    ("ایکس کرومسومز",   "X chromosomes"),
    ("بلڈ سیمپل",       "blood sample"),
    ("سائنٹیفک مائنڈ",  "scientific mind"),

    # Time / quantity phrases
    ("نائنٹی نائن پرسنٹ", "ninety-nine percent"),
    ("مکسیمم ٹائم",    "maximum time"),
    ("سیبن ڈیز",        "seven days"),

    # Other multi-word
    ("پارلیمنٹ ہاؤس",  "Parliament House"),
    ("اوٹو ٹریڈر",     "auto trader"),
    ("رولز رویس",       "Rolls Royce"),
    ("فاسٹ ٹریک",      "fast track"),
    ("موبائل فونز",     "mobile phones"),
    ("وال مارٹ",        "Walmart"),
    ("وال پیپر",        "wallpaper"),
    ("ملٹی پلائی",      "multiply"),
    ("اپ گریڈ",         "upgrade"),
    ("ایکسپینڈشن",      "expansion"),
    ("کیس کلوزڈ",       "case closed"),
    ("بورڈنگ پاس",      "boarding pass"),
    ("شورٹ لسٹ",        "shortlist"),
    ("ٹیرسٹریل سولز",   "terrestrial souls"),
    ("ڈیٹو کاپی",       "ditto copy"),
    ("ڈی این اے",       "DNA"),
    ("ایم ایف آئی",     "MFI"),
    ("سیٹ اپ",          "set up"),
    ("ویزا اپلائی",     "visa apply"),

    # ── Single words ──────────────────────────────────────────────────────────

    # Scholars / academia
    ("اسکولرز",         "scholars"),
    ("اسکولر",          "scholar"),
    ("سکولر",           "scholar"),

    # Episodes
    ("اپیسوڈز",         "episodes"),
    ("اپیسوڈ",          "episode"),

    # Grammar / language
    ("اسٹیٹمنٹ",        "statement"),

    # Science / biology (single words)
    ("کرومسومز",        "chromosomes"),
    ("لیکویڈ",          "liquid"),
    ("لوپس",            "loops"),
    ("پولز",            "poles"),
    ("ریشو",            "ratio"),

    # Math / quantities
    ("پرسنٹ",           "percent"),

    # Cricket (single words)
    ("کرکیٹ",           "cricket"),
    ("بیٹسمن",          "batsman"),
    ("اوپنر",           "opener"),
    ("بولنگ",           "bowling"),
    ("بولر",            "bowler"),
    ("ویکٹ",            "wicket"),
    ("وکیٹ",            "wicket"),
    ("ایننگ",           "innings"),
    ("سکور",            "score"),
    ("سکسٹی ٹو",        "sixty-two"),
    ("پوویلین",         "pavilion"),
    ("نمبرنگ",          "numbering"),

    # IT / tech
    ("یوٹیوب",          "YouTube"),
    ("چینل",            "channel"),
    ("لائیو",           "live"),
    ("پروٹوکول",        "protocol"),
    ("پلاننگ",          "planning"),
    ("اسٹیبلش",         "establish"),
    ("ایکسپینڈ",        "expand"),
    ("ڈیٹیکٹ",          "detect"),
    ("اسٹیملیٹ",        "stimulate"),
    ("اپلائی",          "apply"),

    # Finance / economy
    ("ڈسکاؤنٹ",         "discount"),
    ("بیلنس",           "balance"),
    ("پروجیکٹ",         "project"),
    ("ٹیسکو",           "Tesco"),

    # Transport / travel
    ("بورڈنگ",          "boarding"),
    ("ایمیگریشن",       "immigration"),
    ("ایگزٹ",           "exit"),
    ("فلائٹ",           "flight"),
    ("امیریٹس",         "Emirates"),
    ("کرائسلر",         "Chrysler"),
    ("ٹویوٹا",          "Toyota"),

    # Stamps / identity
    ("اسٹیمپ",          "stamp"),
    ("سٹیمپ",           "stamp"),
    ("ویریفائیڈ",       "verified"),
    ("ویریفائ",         "verified"),
    ("کنفرم",           "confirm"),
    ("سیٹیزن",          "citizen"),
    ("پریولیج",         "privilege"),
    ("گورنمنٹ",         "government"),
    ("یویس",            "US"),
    ("یوکے",            "UK"),

    # Medical / body
    ("میڈیکلی",         "medically"),
    ("سائنٹیفیکلی",     "scientifically"),
    ("سٹور",            "store"),
    ("باتھروم",         "bathroom"),
    ("لیٹرین",          "latrine"),

    # General English loanwords
    ("شیڈو",            "shadow"),
    ("پیرامیٹرز",       "parameters"),
    ("انکارنیشن",       "incarnation"),
    ("کمپاؤنڈ",         "compound"),
    ("ریزلٹ",           "result"),
    ("سپیریشن",         "separation"),
    ("سپریشن",          "separation"),
    ("گروتھ",           "growth"),
    ("پارشلی",          "partially"),
    ("ریپریزنٹیشن",     "representation"),
    ("ڈسٹریبیوٹ",       "distributed"),
    ("پوزیشن",          "position"),
    ("ریکاگنائز",       "recognized"),
    ("ڈائرکٹ",          "direct"),
    ("اکسپلین",         "explain"),
    ("کلیر",            "clear"),
    ("کورس",            "course"),
    ("پروٹیکشن",        "protection"),
    ("پارٹیز",          "parties"),
    ("پرکٹیکلی",        "practically"),
    ("ٹیکنیکلٹیز",      "technicalities"),
    ("ڈامیننٹ",         "dominant"),
    ("گفٹ",             "gift"),
]


def apply_replacements(text: str) -> str:
    for urdu, english in REPLACEMENTS:
        text = text.replace(urdu, english)
    return text


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    changed = 0
    for entry in manifest:
        original = entry["transcript"]
        fixed = apply_replacements(original)
        if fixed != original:
            entry["transcript"] = fixed
            changed += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Input:     {INPUT_PATH}  ({len(manifest)} entries)")
    print(f"Output:    {OUTPUT_PATH}")
    print(f"Changed:   {changed} transcripts")
    print(f"Unchanged: {len(manifest) - changed} transcripts")


if __name__ == "__main__":
    main()
