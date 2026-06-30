# Issues Fix Log

Tracks fixes applied to the data pipeline and training setup, keyed to issues
from `Manifest_Analysis_Report.docx`.

---

## Issue 10 — Chunk Boundary Truncation

**Severity:** CRITICAL  
**Status:** FIXED  
**Affected (before fix):** 310 / 535 entries (58%)  
**Fixed in:** `src/srt_audio_prep.py` → `group_cues()`  
**Date:** 2026-06-30

### Root Cause
`group_cues()` split chunks purely when accumulating the next SRT cue would
exceed `max_duration` (24.5s pre-snap), with no awareness of sentence
boundaries. Audio cuts therefore landed mid-utterance, and the transcript ended
mid-sentence — teaching the model that truncated output is acceptable.

### Fix
Introduced a three-tier split priority inside `group_cues()`:

| Priority | Split point | Punctuation |
|---|---|---|
| 1 (best) | Last complete-sentence boundary | ۔ ؟ ! |
| 2 | Last clause boundary | ، |
| 3 (fallback) | Hard duration cut — no punctuation in window | — |

Added two helpers alongside the fix:
- `ends_sentence(text)` — detects ۔ ؟ ! at cue end
- `ends_clause(text)` — detects ، at cue end

When adding the next cue would exceed `max_duration`, the algorithm now
backtracks to the best available split point in the current window rather than
cutting wherever the overflow happens to land.

### Results (EP1–EP10, `data/processed/Batch1_EP10/`)

| Ending type | Count | % |
|---|---|---|
| Complete sentence (۔ ؟ !) | 439 | 71.0% |
| Clause boundary (،) | 33 | 5.3% |
| Hard cut — no punctuation | 146 | 23.6% |
| **Total** | **618** | — |

**Before → After:** 58% truncated → 23.6% hard-cut residual  
The residual 23.6% are genuine run-on passages (long Quranic recitation,
dense monologue) with no punctuated cue anywhere in the window — no further
improvement possible through splitting logic alone.

### Decision on residual hard-cut samples
Kept in training data for the POC. Reasoning:
- Audio cuts land at real silence points (silence-snapping is unaffected).
- Content is Urdu/Arabic-heavy — highest training value per labeling hour.
- LoRA (Path B) partially insulates base model's EOT behavior from truncation signal.
- If post-training inference shows early-cutoff artifacts, exclude and retrain.

### Files changed
- `src/srt_audio_prep.py` — `group_cues()`, `ends_sentence()`, `ends_clause()`,
  `SENTENCE_END_RE`, `CLAUSE_END_RE`

---

## Issues 3–8 — Unicode Normalization

**Severity:** HIGH (presentation-form ligatures Whisper cannot output)  
**Status:** FIXED  
**Affected:** 167 / 618 entries (244 total character replacements)  
**Fixed in:** `src/normalize_manifest.py`  
**Output:** `data/processed/Batch1_EP10/manifest_normalized.json`  
**Date:** 2026-06-30

### Changes Applied

| Rule | Count | Treatment |
|------|-------|-----------|
| ﷺ (U+FDFA) | 221 | → `صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ` |
| ؐ (U+0610) combining salawat | 3 | → `صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ` |
| ؑ (U+0611) combining alayhe assalam | 17 | → `عَلَیْہِ السَّلَام` |
| ؓ (U+0613) combining radi allahu anhu | 2 | → `رَضِیَ اللَّهُ عَنْہُ` |
| U+200C ZWNJ | 1 | Removed |
| Issue 5b: space before ۔ | 71 | Removed — `کہا ۔` → `کہا۔` |
| Issue 5a: no space after ، | 8 | Added — `کہا،اور` → `کہا، اور` |

### Intentionally Left Unchanged
- All Arabic/Urdu diacritics (harakat) — kept, including aِس/اُس disambiguating cases
- بھئی — valid colloquial Urdu, not a transcription error
- Curly quotes `" "`, em dash `—` — negligible count, Whisper tokenizer handles them
- Arabic ي/ك in Quranic text — correct Arabic script, not Urdu typos

### Files changed
- `src/normalize_manifest.py` — new script

---
