# Session 2

## Context loaded
- Resumed from `CLAUDE.MD` + `sessions/session1.md`.
- Verified current repo state still matches session1's notes: no LoRA wiring,
  no `evaluate()`, no `src/train.py`/`src/evaluate.py`/`src/transcribe_batch.py`/
  `scripts/` yet — logic still inline in `modal_app.py`. Still only the one
  sample audio+SRT pair processed (`data/processed/sample_test/`, 11 chunks).

## Q&A: do we need audio+SRT for all 4,000+ videos, or is ~100 enough?
- **No** — the 4,000+ videos are the *production inference target* (Phase 6:
  `transcribe_batch.py` + faster-whisper + WhisperX runs on them post-training,
  generating transcripts is the point, no ground truth needed for them).
- **Training data** is a separate, much smaller curated pool — POC target is
  **5–10 hrs** of human-verified audio+transcript pairs (per CLAUDE.MD's Data
  Requirements table), not video count.
- Back-of-envelope from the one processed sample: ~270s (4.5min) source →
  ~0.069hrs (~4.1min) usable after gap/music dropping (~92% survival). At that
  ratio, ~100 similarly-length videos ≈ ~7.5hrs — lands in the POC target band.
- **Caveat — single data point, not a guarantee**: depends heavily on actual
  video length variance, and per CLAUDE.MD's quality rules the 100 should be
  *selected* for dense Urdu/Arabic + code-switching content, not just any 100
  (English-dominant clips are explicitly deprioritized).
- Practical next step (not yet done): once the next batch of pairs lands, run
  `batch_srt_prep.py`, sum actual manifest durations, and check against the
  5–10hr target rather than assuming the count alone is sufficient.

## LoRA (Path B) wiring — implemented
Closed the biggest gap from session1: `modal_app.py::train()` was still doing
full fine-tuning despite CLAUDE.MD committing to LoRA. Used Plan Mode before
touching code since it affects training correctness and model save/load
format used elsewhere.

### Decisions made
- **`target_modules`**: kept `["q_proj", "v_proj"]` (not expanded to
  `k_proj`/`out_proj`/`fc1`/`fc2`) — user explicitly confirmed this via a
  direct question. Rationale: matches the LoRA paper's own recommended
  default and CLAUDE.MD's already-documented config, ~5% trainable params,
  lower overfitting risk on the small (5–10hr) dataset. Establishes a clean
  baseline WER before spending added capacity.
  - **Flagged next lever** if WER undershoots the ~20–25% POC target: expand
    `target_modules` (more capacity for the *script-switching/output*
    distribution shift this task needs — Nastaliq vs Arabic vs English mid-
    sentence — which leans on decoder FFN/output projection more than
    attention alone). Cheaper to try than collecting more data.
- **Save strategy**: save *both* an adapter-only checkpoint and a merged
  full model, rather than picking one.
  - Adapter-only → new path `model/whisper-medium-urdu-lora-adapter/`
    (~60MB, fast, useful for iterating without re-merging).
  - Merged (`model.merge_and_unload()`) → existing path
    `model/whisper-medium-urdu-final/`.
  - Reason: `transcribe_batch()` already loads from `...-final/` via plain
    `WhisperForConditionalGeneration.from_pretrained()`, and CLAUDE.MD's HF
    Hub push + `ct2-transformers-converter` steps both expect a standard
    (non-PEFT) checkpoint there. Saving merged weights at that path means
    **zero changes needed** to `transcribe_batch()`, the Hub push step, or
    the ct2 conversion step.
- **Scope deliberately excluded**: `evaluate()` function and the
  `src/train.py`/`src/evaluate.py` file-structure split from CLAUDE.MD's
  documented layout — separate gaps, left for later (the `src/` split needs
  Modal-specific image-mounting work unrelated to LoRA itself).

### Implementation
- `config/training_config.yaml` — added `lora:` section: `r=32`,
  `lora_alpha=64`, `target_modules=["q_proj","v_proj"]`, `lora_dropout=0.05`,
  `task_type="SEQ_2_SEQ_LM"`.
- `modal_app.py` image — added `peft>=0.10.0` to the container's
  `pip_install([...])` list. (Was already pinned in `requirements.txt` but
  missing from the Modal image — a real gap that would have crashed the
  container at import time.)
- `modal_app.py::train()` — after loading the base model: wraps it with
  `LoraConfig`/`get_peft_model`, then calls `model.enable_input_require_grads()`
  and `model.print_trainable_parameters()`.
  - **The one real gotcha**: `training_config.yaml` has
    `gradient_checkpointing: true`. With base weights frozen (PEFT) *and*
    gradient checkpointing on, gradients can't reach the LoRA adapters
    unless input embeddings are explicitly marked to require grad — without
    `enable_input_require_grads()`, training crashes on the first backward
    pass with `RuntimeError: element 0 of tensors does not require grad and
    does not have a grad_fn`. This is the standard documented fix from HF's
    own PEFT+Whisper guide.
  - `print_trainable_parameters()` is a sanity check against CLAUDE.MD's
    documented ~15M/300M (5%) trainable params figure.
- `modal_app.py::train()` save block — replaced the old
  `trainer.save_model(model_save_path)` (which on a PEFT-wrapped model would
  silently save *adapter-only* weights to the path `transcribe_batch()`
  expects a full model at) with the explicit two-step save described above.

### Status — implemented but NOT runtime-validated
No GPU available locally and no training data landed yet, so this is
untested against a real run. Local IDE shows "Import peft could not be
resolved" — expected and correct: `peft` intentionally lives only in Modal's
remote image, not the local lean venv (same local/remote split established
in session1: heavy ML stack stays in Modal's container only).

**Smoke test to run once data + a GPU pass are available** (from the plan):
run `modal run modal_app.py::train` on the smallest available dataset slice
and confirm:
1. `print_trainable_parameters()` output is roughly ~5%.
2. Training doesn't crash on the first backward pass.
3. Both `whisper-medium-urdu-lora-adapter/` and `whisper-medium-urdu-final/`
   appear on the volume after completion.

## Repo housekeeping
- Added `README.md` (practical entry point: status, setup, layout, workflow —
  points to `CLAUDE.MD` for design rationale and `sessions/*.md` for history
  rather than duplicating either).
- Added root `.gitignore` (`venv/`, Python cache, `.env`, model artifacts,
  logs, IDE/OS cruft, plus `data/` and `samples/` since both are large
  regenerable/raw binary content). Note: this excludes the validated sample
  audio+SRT pairs from version control by default — flagged to user as a
  judgment call, not yet revisited.

## Team update: 50 plain-text transcripts received (no timestamps)
Team delivered 50 human-reviewed-for-correctness transcripts, but plain text
only — no timestamps, so they can't feed `srt_audio_prep.py` as-is.
- Confirmed forced alignment (audio + known-correct text → word-level
  timestamps) is the right tool: cheaper than re-timestamping by hand, and
  the accuracy bar needed (good enough to find clean chunk-cut points) is
  well below subtitle-grade precision.
- Flagged real risk: alignment quality degrades wherever the "reviewed"
  text isn't verbatim to what's actually spoken (cleaned-up filler words
  etc.) — worth confirming with the team which kind of review this was.
- Wrote a self-contained prompt for a separate one-time POC to do this work
  (user's call — "one time work" doesn't justify building inside this repo).

## Second real sample processed: EP3 (aq4uNu-gY0M) — and a forced-alignment deep dive
User dropped a second audio+SRT pair into `samples/` (`EP3_aq4uNu-gY0M.srt` +
matching mp3). Ran `srt_audio_prep.py` → 95 chunks, 0.576 hours,
`data/processed/ep3_test/manifest.json`. Running total toward POC target:
~0.069 (vid0001) + 0.576 (vid0002) ≈ 0.645 hrs.

**Manual review caught 2 real text/audio boundary mismatches** (consistent
with the established practice of listen+read spot-checks catching real
sync bugs automated checks miss):
1. `طور` at the end of chunk 030 actually belongs (audio-wise) to chunk 031.
2. `سیّدنا گوھر شاہی کی ذات والا` at the end of chunk 009 actually belongs to
   the start of chunk 010 — hand-fixed directly in `manifest.json`.

**Root-cause investigation — this was NOT a `srt_audio_prep.py` bug.**
Confirmed EP3's SRT wasn't a downloaded caption like sample 1 — it was
already produced by forced alignment. Discovered an existing sibling project,
`../SRTTimeStampPOC` (`align_to_srt.py` + `modal_align.py`), already built for
exactly this: rough Whisper ASR pass for segment timing → fuzzy-match
ASR words against verified ground truth via `difflib` to find anchors →
interpolate to slice ground-truth text into ~76 segments → real forced
alignment (`whisperx.align()`, Urdu wav2vec2 CTC) per segment → regroup into
SRT cues. Confirmed via `modal_run.log` this ran successfully on Modal GPU
(Whisper-medium rough pass, 4334 ground-truth words, 76 segments).
- Diagnosis: both mismatches are the signature of a word landing on the
  wrong side of one of those ~76 *rough segment* boundaries (decided by
  ASR-anchor interpolation, which the script's own comment admits "only
  needs to be roughly right") — **before** the real wav2vec2 alignment ever
  runs. The CTC alignment itself isn't at fault; it faithfully aligns
  whatever text it's handed within its assigned segment window, and can't
  reach across to a neighboring segment.
- Implication: future fixes for this class of error belong in
  `SRTTimeStampPOC` (anchor-matching/segmentation step), not in
  `srt_audio_prep.py`'s silence-snapping, which is already working as
  designed on its input.

**Tested whether Whisper-large fixes it — mixed result, one regression found.**
Changed `modal_align.py`'s default `asr_model` to `"large-v3"` and re-ran on
Modal (had to fix the same Windows cp1252-vs-Unicode-checkmark crash from
session1, this time inside Modal CLI's own output — fixed via
`PYTHONIOENCODING=utf-8`, same root cause as before).
- `سیّدنا گوھر شاہی` boundary: **improved** — large-v3 moved the cut one word
  later ("والا" now correctly grouped with the next segment), matching the
  manual fix.
- `طور`/`پر` boundary: **unchanged** — identical split, not fixed by model
  size alone.
- **New problem found**: large-v3 placed the first spoken word at 44.67s,
  but the user confirmed by ear that intro music runs until ~1:07 — and the
  original medium run had placed it at 67.571s, matching that almost
  exactly. Root cause: neither run passed `--skip-start`, so unskipped
  intro music risked confusing the rough ASR/anchor-matching step right at
  the start of the file.
- Fixed by re-running with `--skip-start 67` → confirmed all three
  checkpoints now correct in `EP3_large_skip67.srt`: intro timing matches
  (1:07.571), the گوھر شاہی improvement is preserved, طور/پر is still
  unfixed (expected — unrelated to skip-start).
- Conclusion: bigger ASR model is a partial, not complete, fix for the
  segment-boundary error class. If `طور`/`پر`-style errors keep showing up
  at scale, the more durable fix is restructuring `align_to_srt.py`'s
  segment slicing to use overlapping windows + post-alignment reconciliation
  (or switching to a dedicated long-form aligner like `ctc-segmentation` /
  NeMo's forced aligner / `torchaudio.functional.forced_align`, which avoid
  the risky ASR-pass-for-segmentation step entirely) — deferred until the
  current spot-check tells us how often this actually recurs.

Ran `srt_audio_prep.py` again on `EP3_large_skip67_aq4uNu-gY0M.srt` →
95 chunks, 0.576 hours, `data/processed/ep3_test_v2/manifest.json` (kept
separate from `ep3_test/` for comparison). **User is spot-checking this now**
— outcome not yet known.

## Open items / next steps
- **In progress**: user spot-checking `data/processed/ep3_test_v2/manifest.json`
  (large-v3 + skip-start-67 version of EP3). Outcome determines whether the
  `align_to_srt.py` overlap/reconciliation fix is worth doing now.
- Still waiting on team to confirm audio+SRT is the standard format for all
  4,000+ videos (carried over from session1).
- Still waiting on team to confirm whether the 50 plain-text transcripts are
  verbatim or cleaned-up — affects forced-alignment reliability for them.
- Separate `SRTTimeStampPOC` POC (sibling project, not in this repo) needed
  to convert the 50 plain-text transcripts into SRTs before they can feed
  `srt_audio_prep.py`/`batch_srt_prep.py` here.
- Process the next batch of audio+SRT pairs (the 50, working toward ~100)
  through `batch_srt_prep.py` once they land; sum real manifest durations
  against the 5–10hr POC target instead of assuming count alone is enough.
- Run the LoRA smoke test (above) on first real training data.
- `evaluate()` function still not built (WER computation on held-out set).
- `src/train.py`, `src/evaluate.py`, `src/transcribe_batch.py`,
  `scripts/upload_data.py`, `scripts/download_model.py` still don't exist —
  logic still inline in `modal_app.py`.
- If first LoRA run undershoots ~20–25% WER target, next lever is expanding
  `target_modules` (see Decisions above) before reaching for more data.
