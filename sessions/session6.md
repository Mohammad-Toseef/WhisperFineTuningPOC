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
- Fix (Option A from the report): added `ends_sentence()` + reworked
  `group_cues()` to prefer splitting right after the last sentence-ending cue
  when growing the window would exceed `max_duration`, falling back to the
  old hard duration cut only when no sentence boundary exists anywhere in the
  window (e.g. long Quranic recitation/monologue with no punctuated cue).
- Re-ran `batch_srt_prep.py` against `samples/28JulyBatch/` into a **new**
  output dir `data/processed/Batch1_EP10/` (left `batch_28july/` untouched, per
  user's choice) — confirms the fix without destroying the prior batch.

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
- Exclude index 608 (`EP10_0n12YXValEk_062.wav`).
- Review the residual ~29% truncated entries manually (run-on sentences with
  no punctuation) — decide whether to merge across chunk boundaries or accept
  as-is.
- Decide whether `Batch1_EP10/` replaces `batch_28july/` as the canonical
  manifest going forward, or both are kept temporarily for comparison.

## Summary
Resolved Issue 10 (chunk boundary truncation) by making `group_cues()` in
`src/srt_audio_prep.py` split at sentence-ending punctuation (۔ ؟ !) instead
of purely at fixed time/duration boundaries, with a fallback to the old hard
cut only when no sentence boundary exists in the window. Verified against all
10 episodes in `samples/28JulyBatch/`: truncation dropped from 58% (310/535)
to 28.8% (176/611), with remaining cases being genuine run-on speech rather
than algorithm misses. Re-ran the full batch into a new directory
(`data/processed/Batch1_EP10/`) to avoid overwriting the existing
`batch_28july/` manifest.
