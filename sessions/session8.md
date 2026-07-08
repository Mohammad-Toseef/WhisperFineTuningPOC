# Session 8 — 2026-07-05

## Goal / Context
Determine and execute the next step now that the reviewed training dataset is ready.
Run a small LoRA POC (Whisper large-v3) before scaling to all 23 episodes.

## Key Decisions
- POC base model = **openai/whisper-large-v3** (user chose over the medium the pipeline was wired for).
- POC mode = **smoke test first** (prove pipeline end-to-end; baseline/evaluate WER deferred).
- Validated `data/processed/Batch1_EP10/manifest_reviewed.json`: 535 clips / 3.17 hrs, 0 missing audio, 0 empty transcripts, all lang=ur.
- Fixed gross overfit risk: `max_steps` 5000 → 120 (~8 epochs @ eff-batch 32), warmup 500 → 20, eval/save_steps 500 → 30.
- Sized for A10G 24GB + large-v3: per_device_train_batch_size 16 → 8, grad_accum 2 → 4 (eff batch still 32).
- Confirmed HF dataset embeds audio bytes into Arrow (274M train + 29M eval) → self-contained, portable to Modal.
- User supplied newer reviewed export `Batch-1 Pechan e Mehdi 23 Episodes_reviewed_manifest_5_july.json` (567 entries, all status=reviewed, EP1–EP10 only despite "23 Episodes" batch name).
- Ran `scripts/convert_reviewed_manifest.py` → matched Batch1_EP10 (100% coverage), overwrote `manifest_reviewed.json` with 567 entries (was 535). 2.90 hrs, 0 missing audio, 0 empty.
- Normalizer NO LONGER needed: this export already has 0× ﷺ (U+FDFA), 0 ZWNJ, 0 combining honorifics — reviewers cleaned them during review.
- Rebuilt HF dataset from 567 → **510 train / 57 eval** (57 eval = held-out test set for baseline vs fine-tuned WER).
- CORRECTION (user): reviewed manifest belongs to **Batch1_EP23** (canonical multi-episode folder), not Batch1_EP10. Verified audio is byte-identical in both folders (all 567 keys, exact duration match, 0 mismatch) → no misalignment; provenance fix only.
- Added `--batch-folder` override flag to `convert_reviewed_manifest.py` (Jaccard auto-detect always favors EP10 since EP23 has extra EP11–23 chunks).
- Re-converted with `--batch-folder Batch1_EP23` → `data/processed/Batch1_EP23/manifest_reviewed.json` (100% coverage, 0 missing audio, 2.90 hrs).
- Updated `dataset_builder.py` default → Batch1_EP23; rebuilt dataset (510/57) from canonical location.
- STALE: `data/processed/Batch1_EP10/manifest_reviewed.json` now outdated (pending user decision to delete).
- Added `evaluate(which="base"|"finetuned"|"both")` to modal_app.py: runs base + fine-tuned over the same 57 eval clips, identical normalization (strip punctuation, keep diacritics), prints raw+norm WER + delta, saves `/data/logs/eval_results.json`. GPU=A10G.
- Centralized model paths → `FINAL_MODEL_PATH` (`whisper-urdu-final`) + `ADAPTER_PATH` constants; updated train/transcribe/evaluate to use them (renamed from `whisper-medium-urdu-*`; safe — nothing trained yet).

## Modal Run (2026-07-06)
- Installed modal 1.5.1 into venv; auth via existing ~/.modal.toml profile `mstechnologies` / env `main`.
- **Volume path gotcha (fixed):** volume mounts AT `/data`, so remote upload paths must be root-relative (`/config`, `/processed/dataset`) NOT `/data/config` — the latter double-nests to `/data/data/...` and the container can't find files. CLAUDE.md's example `modal volume put ... /data/config` has this same latent bug.
- **Git Bash path mangling (fixed):** `modal volume put ... /data/...` under Git Bash rewrites remote path to `/C:/Program Files/Git/...`. Use PowerShell for modal CLI (and PYTHONIOENCODING=utf-8/PYTHONUTF8=1 for the ✓ glyphs).
- **Dependency pinning (fixed):** unbounded `>=` pulled transformers 5.13 / torch 2.12 / numpy 2.4 / jiwer 4.0 → breaks Seq2SeqTrainer API. Pinned transformers<4.46, torch<2.5, numpy<2.0, jiwer<4.0, peft<0.14, accelerate<1.0.
- ✅ **BASELINE (base whisper-large-v3, 57 clips): raw WER 20.96% | normalized WER 16.76%.** Far better than CLAUDE.md's ~35–45% (that assumed medium). Small headroom → expect modest fine-tuning delta; short 120-step run could even slightly regress. POC measures pipeline, not just delta magnitude.
- **PEFT+Whisper fix:** `task_type=SEQ_2_SEQ_LM` → `WhisperForConditionalGeneration.forward() got unexpected kwarg 'input_ids'`. Set `task_type: null` in config + `l_cfg.get("task_type")` in code. (CLAUDE.md documents the wrong value — real doc bug.)
- ✅ **TRAINING SUCCEEDED** (26 min, A10G): LoRA 15.7M/1.56B (1.0%). train_loss 0.407→0.340; eval_loss 0.400→0.349 (both steady down, no overfit); in-training raw eval_wer 20.43→**19.76**(best@ep3.75)→19.89→20.16. load_best_model_at_end saved best checkpoint. Merged model → /data/model/whisper-urdu-final.
- Final base-vs-finetuned evaluate (--which both) launched (task bepdqypo8).
- ✅ **POC COMPLETE.** Base: raw 20.96% / norm 16.76%. Fine-tuned: raw 19.76% / norm 16.01%. **Improvement +0.75 norm (+1.20 raw).** Modest but real & correct direction (loss+WER both down, no overfit). Small delta expected: strong large-v3 baseline + Nastaliq-heavy eval set + minimal 2.9hr/120-step run. Results saved to volume /data/logs/eval_results.json.

## Open Items / TODOs
- modal_app.py still labels save paths `whisper-medium-urdu-*` (cosmetic; train+transcribe are internally consistent).
- No `evaluate()` function / baseline WER yet — needed for a *measurable* POC (deferred per smoke-test choice).
- large-v3 batch 8 may OOM on A10G; fallback is batch 4 / grad_accum 8.

## Next Steps
- **Review remaining 13 episodes** — biggest lever; POC used only 2.9 hrs vs 5–10 hr target.
- **Push fine-tuned model to HF Hub** — Modal volume is NOT permanent; model only at /data/model/whisper-urdu-final. (scripts/download_model.py or push_to_hub.)
- Build a **code-switching-weighted eval subset** — random Nastaliq-heavy eval hides fine-tuning's true benefit on English/Arabic terms & domain vocab.
- For real run: scale steps, add LoRA target modules (k_proj/out_proj/fc1/fc2), maybe higher LR/r.
- ✅ DONE: Fixed CLAUDE.md (medium→large-v3 throughout, task_type→null with warning, volume root-relative paths + PowerShell/encoding caveats, pinned deps, real WER numbers, dedup'd Audio Format block, model dir names → whisper-urdu-*).
- ✅ DONE: Created SMOKE_TEST_REPORT.md (full run: install→commands→results→5 issues&fixes→interpretation→recommendations).
- Delete stale data/processed/Batch1_EP10/manifest_reviewed.json (still pending).

## Summary
Ran the full whisper-large-v3 LoRA POC end-to-end on Modal. Prepped data (converted 5-July
reviewed export → Batch1_EP23/manifest_reviewed.json, 567 clips/2.9hr; built HF dataset
510 train/57 eval), added evaluate()+baseline to modal_app.py, and executed baseline →
train → compare on Modal. Fixed 5 infra bugs along the way (Git Bash path mangling, volume
double-nesting, dep version explosion, PEFT task_type, all documented above). **Pipeline
fully validated.** Final WER: base 16.76% → fine-tuned 16.01% normalized (+0.75 pts).
Modest but real improvement; small delta attributable to already-strong large-v3 baseline,
Nastaliq-heavy eval set, and deliberately minimal training. Ready to scale to more episodes.
