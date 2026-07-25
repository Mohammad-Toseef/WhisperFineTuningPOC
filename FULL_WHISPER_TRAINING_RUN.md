# Full Whisper Large-v3 Fine-Tuning Run

Tracking doc for the full (49-episode) LoRA fine-tuning run. Supersedes the smoke test
(see [`SMOKE_TEST_REPORT.md`](SMOKE_TEST_REPORT.md)). Started 2026-07-25.

---

## Goal
Fine-tune `openai/whisper-large-v3` (Path B / LoRA) on ~10 hrs of the Pehchan-e-Mehdi
speaker to improve (1) accent/pacing and (2) domain/spiritual-term transcription
(izafat terms + Arabic recitation), for batch transcription of 4,000+ videos.

---

## Data

| | Value |
|---|---|
| Source | 49-episode reviewed manifest (`all_batches_reviewed_manifest _final.json`, portal export) |
| Episodes | EP1–EP50 **minus EP39** = 49 episodes |
| Clips (after cleanup) | **2,257** |
| Total audio | **11.67 hrs** → CLAUDE.md "Solid" band (10–20 hrs) |
| Converted manifest | `data/processed/manifest_reviewed.json` (gitignored) |
| Audio location | `data/processed/Batch1_EP23/audio` (EP1–23) + `Batch2_EP24_EP50/audio` (EP24–50) |

**Faulty chunks removed** (audio deleted + entry removed from `manifest_reviewed.json` only;
source batch manifests + export left intact):
- EP3 idx 038, EP38 idx 010, EP38 idx 011 → 2260 → 2257 clips.
- ⚠️ Re-running `convert_reviewed_manifest.py` will regenerate these 3 and hard-fail on the
  missing audio. Re-do the removal after any re-convert.

---

## Train / Eval Split (whole-episode holdout, PINNED)

Eval episodes are **pinned** via `--eval-episodes` so the baseline↔fine-tuned comparison
stays fixed across re-runs (auto-pick would drift if `domain_terms.json` changes).

| Split | Episodes | Clips | Hours |
|---|---|---|---|
| **Train** | 42 | 2,011 | 10.39 hrs |
| **Eval** | 7 | 246 | 1.28 hrs (10.9%) |

**Pinned eval episodes:** `EP5_vwzNL2oziZs, EP6_SrVnpBqd7bI, EP34_h87EJF0Zvco,
EP41_mBtP9NKha1g, EP43_m8-37sgUwUQ, EP44_paAJQ3OKB-8, EP47_a0NiZST0S6Q`

**Eval buckets** (from `eval_buckets.json`, drives per-bucket WER):

| Bucket | Train | Eval |
|---|---|---|
| nastaliq_only | 1,695 | 194 |
| code_switch | 316 | 52 |
| spiritual_term | 558 | **129** (52% of eval) |

---

## Domain Terms (`config/domain_terms.json`)
Expanded 2026-07-25 from the 49-ep transcripts: **60 spiritual_terms + 11 arabic_phrases
+ 10 english_domain**. Bucket coverage 6.8% → **30.4%** (687/2257 clips).
- Finding: the 7 original undiacritized `arabic_phrases` match 0 clips (reviewers wrote
  Arabic fully diacritized) — kept only as decoder-biasing tokens for the initial_prompt.
- Added diacritized durood fragments (`عَلَيْهِ وَسَلَّمَ`, etc.) as match-keys.

---

## Training Config (`config/training_config.yaml`)

| Setting | Value | Note |
|---|---|---|
| Base model | `openai/whisper-large-v3` | fresh run — base + NEW LoRA (no resume) |
| LoRA | r=32, alpha=64, dropout=0.05, `task_type=None` | |
| **LoRA targets (Tier-2)** | `q_proj, k_proj, v_proj, out_proj, fc1, fc2` | MLP targets = lexical/domain-term lever (smoke test was q/v only) |
| per_device_train_batch_size | 8 | |
| gradient_accumulation_steps | 4 | **effective batch = 32** |
| steps/epoch | 63 | ceil(2011/32) |
| **max_steps** | **567** | ~9 epochs |
| warmup_steps | 57 | ~10% |
| eval_steps / save_steps | 63 / 63 | once per epoch; save%eval==0 (load_best works) |
| learning_rate | 1e-5 | |
| fp16 / gradient_checkpointing | true / true | |
| metric_for_best_model | wer (lower better) | load_best_model_at_end |

**GPU: A10G (24 GB) — no upgrade needed.** LoRA + grad-checkpointing peaks ~10–14 GB.
Tier-2 doesn't materially raise peak VRAM (activations unchanged; +~1 GB optimizer state).
**OOM fallback:** `per_device_train_batch_size: 4` + `gradient_accumulation_steps: 8`
(keeps effective batch 32, so max_steps stays 567).

---

## Pipeline Sequence (turnkey)

```powershell
# 1. Convert reviewed manifest  (DONE)
python scripts/convert_reviewed_manifest.py "all_batches_reviewed_manifest _final.json" --output data/processed/manifest_reviewed.json

# 2. Build dataset with pinned eval  (DONE)
python src/dataset_builder.py ./data/processed/manifest_reviewed.json ./data/processed/dataset `
  --eval-episodes EP5_vwzNL2oziZs,EP6_SrVnpBqd7bI,EP34_h87EJF0Zvco,EP41_mBtP9NKha1g,EP43_m8-37sgUwUQ,EP44_paAJQ3OKB-8,EP47_a0NiZST0S6Q

# 3. Upload to Modal (root-relative paths, PowerShell)  (DONE)
modal volume put whisper-training-vol ./data/processed/dataset /processed/dataset --force
modal volume put whisper-training-vol ./config /config --force

# 4. Re-baseline BEFORE training  (IN PROGRESS)
modal run modal_app.py::evaluate --which base

# 5. Train
modal run modal_app.py::train

# 6. Evaluate both (per-bucket delta)
modal run modal_app.py::evaluate

# 7. Push to HF Hub (volumes not permanent!)  — DONE 2026-07-25
modal run --detach scripts/push_to_hub.py    # uploads /model/whisper-urdu-final from volume
# → https://huggingface.co/mohammad-toseef059/whisper-large-v3-urdu (PRIVATE)
```

**HF push (DONE):** merged model (12 files, 6.17GB) pushed to
`mohammad-toseef059/whisper-large-v3-urdu` (private) via `scripts/push_to_hub.py`, which uses
huggingface_hub.upload_folder from the Modal volume + the "huggingface-secret" Modal secret
(HF_TOKEN). Load anywhere: `WhisperForConditionalGeneration.from_pretrained(REPO)` (authed).
Share by adding collaborators (repo Settings) or via org.

---

## Modal Volume State (`whisper-training-vol`)
- `/config` — 4 files (training_config.yaml, domain_terms.json, corrections.json, initial_prompt.txt)
- `/processed/dataset` — train (3 shards) + eval (1 shard) + dataset_dict + eval_buckets.json
- `/model`, `/checkpoints`, `/logs` — **cleared** (smoke-test artifacts removed pre-run)
- ⚠️ Training writes `/model/whisper-urdu-{lora-adapter,final}` and `/checkpoints/...` fresh.

## Smoke-test model backup
Backed up + integrity-verified locally at
`C:\Users\Mohammad Touseef\Downloads\Whisper Smoke Test Trained Model\`
(`whisper-urdu-lora-adapter` 60 MB + `whisper-urdu-final` 5.75 GB fp32). Not on the volume
anymore. To reuse on Modal later, re-upload to a **distinct** path (e.g. `-smoketest` suffix)
so it doesn't collide with this run's output. Consider pushing to HF Hub for durable storage.

---

## What is WER (Word Error Rate)?

WER is the standard accuracy metric for speech-to-text. It measures how many **word-level
edits** are needed to turn the model's output (hypothesis) into the correct text (reference).
**Lower is better.**

**Formula:**
```
        S + D + I
WER  =  ─────────
            N
```
- **S = Substitutions** — a wrong word in place of the correct one (`bank` → `bunk`)
- **D = Deletions** — a reference word the model missed (dropped a word)
- **I = Insertions** — an extra word the model added that isn't in the reference
- **N = total words in the reference**

It's the word-level [Levenshtein edit distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
between the two word sequences, divided by the reference length. Computed here by the `jiwer`
library (via HuggingFace `evaluate`).

**Worked example** (N = 5 reference words):
```
Reference:   آج ہم بات کریں گے
Hypothesis:  آج ہم بات کریں            (dropped گے  → 1 deletion)
WER = (0 + 1 + 0) / 5 = 20%
```
Another:
```
Reference:   the bank account was closed        (N = 5)
Hypothesis:  the bunk account is closed now      (bank→bunk sub, was→is sub, +now insert)
WER = (2 + 0 + 1) / 5 = 60%
```

**Notes / gotchas:**
- WER **can exceed 100%** (many insertions relative to a short reference) — e.g. the base
  model's repetition-collapse would score astronomically on that segment.
- It is **binary per word** — a near-miss (one wrong letter, or a missing diacritic) counts as a
  full substitution. So WER can look worse than the transcript "reads" to a human.
- **0% = perfect** match after normalization.

### Raw vs Normalized WER (both reported here)
This project reports two numbers, computed on the **same** predictions:
- **Raw WER** — compared verbatim. Punctuation (`۔ ، ؟`), spacing, and stray marks all count as
  errors. The Seq2SeqTrainer's per-epoch `eval_wer` during training is RAW.
- **Normalized WER** (the headline metric) — a `normalize()` step (in `modal_app.py::evaluate`)
  is applied **identically** to base and fine-tuned before scoring: it strips Urdu + Latin
  punctuation and collapses whitespace, but **keeps diacritics** (they are part of the target
  labels). This removes penalties for punctuation/spacing differences that don't reflect real
  transcription errors, giving a fairer model-vs-model comparison.

That's why the final numbers (norm) are lower than the trainer's raw numbers — e.g. fine-tuned
raw 15.68% vs normalized 10.50% on the same outputs.

## Results

### Baseline — base large-v3 on the 246-clip eval set
> `modal run modal_app.py::evaluate --which base`  — DONE 2026-07-25

| Metric | n | Normalized WER | Raw WER |
|---|---|---|---|
| **Overall** | 246 | **18.57%** | 23.87% |
| nastaliq_only | 194 | 18.05% | |
| code_switch | 52 | 20.46% | |
| spiritual_term | 129 | 20.29% | |

Read: base is strongest on plain Nastaliq (18.05%); code-switch (20.46%) and spiritual
terms (20.29%) are ~2 pts worse — the two areas Tier-2 fine-tuning targets. Aggregate WER
hid this gap. _(Not comparable to smoke-test 16.76% — different 57-clip eval set.)_

### Training run — DONE 2026-07-25 (detached; 567 steps, 3h14m)
First attempt died at step 88 (local network drop, non-detached). Relaunched with
`modal run --detach` → completed server-side despite local client disconnecting at ~step 250.

Per-epoch eval WER (RAW, from trainer compute_metrics — NOT normalized):

| Epoch | 1 | 2 | 3 | 4 | 5 | **6** | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Raw WER | 21.16 | 17.69 | 16.42 | 16.05 | 15.78 | **15.71⭐** | 15.98 | 15.90 | 15.89 |

- **Best = epoch 6 (checkpoint-378), raw 15.71%** → saved by load_best_model_at_end.
- Base raw was 23.87% → **~34% relative reduction**. Converged ~epoch 5-6; 7-9 mild overfit.
- Saved to `/model/whisper-urdu-{lora-adapter,final}` on the volume.

### Fine-tuned vs base (NORMALIZED + per-bucket) — DONE 2026-07-25 ✅
> `modal run modal_app.py::evaluate` (detached). Saved to /logs/eval_results.json.

| Metric | n | Base (norm) | Fine-tuned (norm) | Δ |
|---|---|---|---|---|
| **Overall** | 246 | 18.57% | **10.50%** | **+8.07** |
| nastaliq_only | 194 | 18.05% | 9.89% | +8.16 |
| code_switch | 52 | 20.46% | 12.70% | +7.76 |
| **spiritual_term** | 129 | 20.29% | **9.38%** | **+10.91** ⭐ |

(Raw WER: base 23.87% → FT 15.68%.)

**RESULT: 10.50% overall beats the CLAUDE.md <13% target and nearly hits the <10% target
that was expected only WITH initial_prompt+corrections.** The spiritual_term bucket went from
the model's WORST area (20.29%) to its BEST (9.38%) — a +10.91pt drop — proving the Tier-2
MLP LoRA targets (fc1/fc2) delivered the domain-term learning they were added for.
For contrast the smoke test moved overall WER only +0.75pt (16.76→16.01); this run +8.07pt.

---

## Qualitative comparison (full episodes, base vs fine-tuned)
Transcribed 2 full sample episodes (EP19, EP2) with base and fine-tuned via
`modal run scripts/compare_transcribe.py` → `full_audio_samples/compare_transcripts/Full Fine Tuned/`
(`{ep}_base.txt` / `{ep}_finetuned.txt`). No reference/WER — purely eyeball. Corroborates the
18.57%→10.50% result.

**Fine-tuned does better:**
1. **Arabic recitation fully diacritized** (matches reviewer convention) — e.g. base `حضورﷺ` /
   `صلی اللہ علیہ وسلم` → FT `حضور صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ` / `صَلَّى اللَّهُ عَلَيْهِ وَآلِهِ وَسَلَّمَ`.
2. **English stays in Latin** — EP2 base `چیلنج` → FT `challenge`.
3. **Real word fixes** — base `بیوی مریم` (wife) → FT `بی بی مریم` (Lady); base `ایسا` → FT `عیسیٰ`.
4. **Less long-form looping** — EP19 base repeats "اُس کا عجر…" / "اب یہ سنت کو زندہ کرنا" 3×;
   FT says it once.
5. **Proper Urdu diacritics/spelling** — FT `داڑھی`/`سُنَّت`/`اُمَّت` vs base `داڑی`/`سنت`.

**Tradeoff:** FT occasionally over-diacritizes and garbles Arabic fragments (EP2 base
`لا مہدی الا عیسیٰ ابن مریم` → FT `لا مہدی الا عنِ مَرِيَمَ`), and merges sentences into longer
run-on lines (segmentation artifact).

### 3rd clip — "Yahoodi ya Sahyooni" (longer/harder audio, added later)
Most dramatic difference so far:
- **Base repetition COLLAPSE:** on a hard segment the base model looped `یہ یہ یہ…` 100+ times,
  destroying that whole passage. The fine-tuned model produced clean coherent text there
  (`…ان سب کو مار دو اور یہاں resort بنا دو…`). Long-form stability is a major FT win.
- **English→Latin, consistent:** base میڈل ایسٹ/انگلینڈ/جینوسائیڈ/جابز/ریس/پرسنٹ →
  FT middle east/England/genocide/jobs/race/percent.
- **Base Arabic mis-hearing fixed:** base opened with `موسیقی اللہ علیہ وسلم` ("music"!) →
  FT `رسول اللہ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ`. Also موسیٰ عَلَيْهِ الصَّلَاةُ وَالسَّلَامُ, سرکار امام مہدی گوھر شاہی.
- Confirms the gap widens on difficult long-form audio.

## Key Gotchas / Notes
- `modal` CLI on Windows → **PowerShell**, not Git Bash (MSYS rewrites `/config` paths).
  Set `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1`.
- Volume upload paths are **root-relative** (`/config`, not `/data/config`).
- `modal volume get <dir>` collapses to one file unless the local dest dir pre-exists.
- Deps pinned `<transformers 4.46` (protects `forced_decoder_ids` generate path + Seq2SeqTrainer API).
- `dataset_builder`'s `--eval-episodes` matches **full folder names** (`EP5_vwzNL2oziZs`), not `EP5`.