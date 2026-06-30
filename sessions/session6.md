# Session 6

## Context loaded
- Resumed from `CLAUDE.MD` + prior sessions (1-5). Last WhisperFineTuningPOC-side
  state (session4): `batch_28july` manifest at 535 chunks / 3.17 hours (EP1-EP10),
  manual spot-check of EP6-EP10 still outstanding.
- `Manifest_Analysis_Report.docx` has been generated analyzing
  `data/processed/batch_28july/manifest.json`.

## Goal / Context
Focus this session on resolving **Issue 10: Chunk boundary truncation** from
the Manifest Analysis Report.

## Key Decisions
- Root cause of Issue 10: `group_cues()` in `src/srt_audio_prep.py` only broke
  chunks on `max_duration` overflow or inter-cue gap, never on sentence-ending
  punctuation (۔ ؟ !), so chunks routinely landed mid-utterance.
- Fix iteration 1: added `ends_sentence()` + reworked `group_cues()` to prefer
  splitting right after the last sentence-ending cue (۔ ؟ !) when the window
  would exceed `max_duration`.
- Fix iteration 2: added `ends_clause()` + ، (Urdu comma) as a second-priority
  fallback tier between sentence-end and hard cut, rescuing 33 more chunks.
- Final three-tier split priority: (1) ۔ ؟ ! → (2) ، → (3) hard duration cut.
- Residual 23.6% hard-cut samples assessed and kept for POC — content value
  (dense Urdu/Arabic) outweighs mild EOT boundary risk under LoRA.
- Re-ran full batch into `data/processed/Batch1_EP10/` (canonical going forward);
  old `batch_28july/` can be archived or deleted.

## Open Items / TODOs
- Carried over: manual listen+read spot-check of EP6-EP10 chunks still not done.
- Carried over: team confirmation on audio+SRT as standard format; LoRA smoke
  test still pending first real GPU run.
- Entry #532-equivalent (now `EP10_0n12YXValEk_062.wav`, index 608 in the new
  manifest, 28.0s audio / 8-char transcript "ہوگا، تو") still present —
  source SRT alignment defect, not a chunking bug. Still needs manual
  exclusion per the report's Action Plan step 1.
- The other automated fixes from the report (diacritics, Unicode symbols,
  بھئی→بھائی, spacing, English-in-Nastaliq) are not yet applied to
  `Batch1_EP10/manifest.json` — still need the normalization script pass.

## Next Steps
- Run the normalization script (diacritics/Unicode/spacing fixes, steps 3-8
  of the report's Action Plan) against `data/processed/Batch1_EP10/manifest.json`.
- Exclude index 608 (`EP10_0n12YXValEk_062.wav`) — catastrophic alignment
  failure (28s audio, 8-char transcript "ہوگا، تو").
- Residual 23.6% hard-cut samples: keep for POC, revisit only if post-training
  inference shows early-cutoff artifacts.
- Archive or delete `batch_28july/` — `Batch1_EP10/` is now canonical.

## Summary
Resolved Issue 10 (chunk boundary truncation) across two fix iterations in
`src/srt_audio_prep.py → group_cues()`:

1. Added sentence-boundary splitting (۔ ؟ !) as the primary split point.
2. Added ، (Urdu comma) as a second-priority clause-boundary fallback.

Final outcome for `data/processed/Batch1_EP10/` (618 chunks, 3.17 hrs):
71.0% complete sentences · 5.3% clause boundaries · 23.6% unavoidable hard cuts.
Down from 58% truncated in the original `batch_28july/` manifest.

Also created `Issues_Fix.md` at the project root to track all pipeline fixes
going forward.
