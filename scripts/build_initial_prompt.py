#!/usr/bin/env python3
"""
Build the Whisper `initial_prompt` (CLAUDE.md Layer 2) from config/domain_terms.json.

The prompt biases the decoder toward rare domain terminology at inference time —
each term appears in the script it should be transcribed in (Urdu spiritual terms
in Nastaliq, Arabic phrases in Arabic, English domain terms in English). Keep it
well under Whisper's ~224-token prompt budget; this trims to --max-terms per group.

Usage:
    python scripts/build_initial_prompt.py                 # print prompt
    python scripts/build_initial_prompt.py --out config/initial_prompt.txt
"""
import argparse
import json
from pathlib import Path


def build_prompt(terms_path: str, max_terms: int = 20) -> str:
    with open(terms_path, encoding="utf-8") as f:
        data = json.load(f)

    spiritual = data.get("spiritual_terms", [])[:max_terms]
    arabic = data.get("arabic_phrases", [])[:max_terms]
    english = data.get("english_domain", [])[:max_terms]

    # A short natural Urdu carrier sentence primes the code-switching register,
    # then the domain terms follow, each in its target script.
    parts = ["آج کی گفتگو روحانیت اور تصوف کے بارے میں ہے۔"]
    if spiritual:
        parts.append("، ".join(spiritual) + "۔")
    if arabic:
        parts.append("، ".join(arabic) + "۔")
    if english:
        parts.append(", ".join(english) + ".")
    return " ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description="Build initial_prompt from domain_terms.json.")
    p.add_argument("--terms", default="config/domain_terms.json")
    p.add_argument("--max-terms", type=int, default=20, help="Max terms per group (prompt-budget guard).")
    p.add_argument("--out", default=None, help="Write prompt to this file (else print to stdout).")
    args = p.parse_args()

    prompt = build_prompt(args.terms, args.max_terms)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(prompt, encoding="utf-8")
        print(f"Wrote initial_prompt ({len(prompt)} chars) -> {args.out}")
    else:
        print(prompt)


if __name__ == "__main__":
    main()
