# Round 2 — Whisper Large-v3 LoRA Continuation Run (Batch 3)

Planning + tracking doc for the second fine-tuning round. Supersedes nothing — round 1 is
recorded in [`FULL_WHISPER_TRAINING_RUN.md`](FULL_WHISPER_TRAINING_RUN.md) and stays the
baseline. Planned 2026-08-27 (session 015). **NOT YET EXECUTED.**

---

## Goal
Continue the round-1 LoRA adapter on the 96-episode reviewed Batch 3 corpus, primarily to fix
**sustained English audio producing fabricated Urdu** (session 014 blocking item #16) without
losing round 1's domain-term gains.

Round-1 result to beat, on an eval set that stays identical:

| Metric | n | Base (norm) | Round 1 (norm) |
|---|---|---|---|
| Overall | 246 | 18.57% | **10.50%** |
| nastaliq_only | 194 | 18.05% | 9.89% |
| code_switch | 52 | 20.46% | 12.70% |
| spiritual_term | 129 | 20.29% | 9.38% |

---

## Decisions (user, 2026-08-27)

| Decision | Choice | Note |
|---|---|---|
| Training data | **Batch 3 only** | 42.49 hrs. Round-1 clips are NOT added to train. |
| Adapter init | **Continue from the round-1 adapter** | Not a fresh LoRA. |
| Eval | **Round-1's 7 pinned EP episodes AND a Batch-3 holdout** | Two sets, one combined eval split. |
| LoRA capacity | **r=32 / alpha=64, unchanged** | Matches the adapter being resumed. |

### Why this combination hangs together
- The round-1 adapter is **Tier-2**, verified from the volume's `adapter_config.json`:
  `r=32, lora_alpha=64, lora_dropout=0.05`, targets `q_proj, k_proj, v_proj, out_proj, fc1, fc2`,
  base `openai/whisper-large-v3`. So "keep r=32/alpha=64" needs no reconciliation — resuming is
  config-compatible by construction. (Confirms it is the full run's adapter, not the smoke test's
  q/v-only one.)
- Training on Batch 3 alone while resuming the round-1 adapter is **sequential fine-tuning**. The
  risk it carries is forgetting round 1's strengths — Batch 3's `spiritual_term` density is
  14.3% vs round 1's 30.4%. Keeping round 1's eval set is exactly the instrument that measures
  that risk, so the eval decision guards the data + init decisions.
- The 7 pinned round-1 eval episodes were held out of round-1 **training** and are not in Batch 3.
  They are therefore **never-trained-on for both adapters**, so 10.50% remains directly
  comparable. Verified: all 7 resolve in `manifest_reviewed.json`, 246 clips, buckets
  194 / 52 / 129 — an exact match to `logs/eval_results.json`.

---

## Data

### Source
`data/processed/Batch3/Batch 3_reviewed_manifest.json` — reviewed portal export, 2026-08-27.

| | Value |
|---|---|
| Reviewed clips | **8,597** |
| Audio | **42.49 hrs** |
| Episodes | 96 (B3001–B3096, contiguous) |
| Duration min/med/max | 0.69 / 19.31 / 28.00 s |
| Empty transcripts / >28s / duplicate keys | 0 / 0 / 0 |
| Reviewer edits | **7,200 of 8,597 (84%)** differ from machine output |

Verified against disk: 8,597 reviewed clips ↔ 8,597 entries in `manifest.json`, **0 missing,
0 extra**, and **8,597 / 8,597 wav files present** (4.56 GB). Round-1 audio also intact
(1,335 + 922 = 2,257 files, 1.25 GB).

### What Batch 3 adds that round 1 could not
| longest consecutive Latin run | round 1 | Batch 3 |
|---|---|---|
| clips containing any Latin | 368 (16.3%) | **2,126 (24.7%)** |
| ≥10 words | 0 | **69** |
| maximum | **8** | **56** |

Round 1's longest English utterance was 8 words; item #16's diagnosis noted that ceiling.

Repetition is clean: max immediate-token repeat across all 8,597 clips is **6** (round-1-era
decoder loops ran to 178×). The session-014 filter plus human review both held.

---

## Splits (PINNED — pass explicitly via `--eval-episodes`)

Pinning matters for the same reason as round 1: auto-selection is term-weighted, so it would
drift the moment `domain_terms.json` changes and break baseline↔round-2 comparability.

### Set A — frozen comparison set (round 1's, unchanged)
`EP5_vwzNL2oziZs, EP6_SrVnpBqd7bI, EP34_h87EJF0Zvco, EP41_mBtP9NKha1g, EP43_m8-37sgUwUQ,
EP44_paAJQ3OKB-8, EP47_a0NiZST0S6Q` — **246 clips / 1.28 hrs**, buckets 194 / 52 / 129.

Its only job is to answer "did round 2 keep round 1's gains", read against 10.50%.

### Set B — Batch-3 holdout (Option 3, recommended)
`B3039, B3028, B3012, B3017, B3031, B3033` — **341 clips / 1.73 hrs (4.0% of Batch 3)**,
buckets: nastaliq_only 209 / **code_switch 132 (39%)** / spiritual_term 68.

**B3039 is the decisive episode.** It is the only confirmed, ear-verified fabrication in the
corpus (session 014: chunks 043/044, the model invented `بیوی مرنے میں بھی ہے،` ×19 over English
audio). Reviewers corrected it to the true English:

```
chunk 043  (43-word English run)
  "While the baby is still in the womb of the mother. The baby is actually
   going through the process, process of ..."
chunk 044  (21-word run)
  "drinking anything. Why? Because everything is coming from the mother
   through the cord and getting into the bod..."
```

So Set B carries a **human-verified English reference for the exact failure case**, which is the
only way to measure item #16 directly rather than by proxy.

Set B alternatives considered and their trade-offs:

| Option | eval clips | code_switch | B3039 | sustained-Eng left in train | eval cost/pass |
|---|---|---|---|---|---|
| 1 — adversarial (9 eps) | 407 (4.7%) | 200 (49%) | no | 62 / 69 | 2.7× |
| 2 — representative (8 eps) | 474 (5.5%) | 137 (29%) | yes | 65 / 69 | 2.9× |
| **3 — compact + B3039 (6 eps)** | **341 (4.0%)** | **132 (39%)** | **yes** | **66 / 69** | **2.4×** |

⚠️ **Accepted cost of Option 3:** B3039's 3 sustained-English clips move from "learn from" to
"measure with". Justified because the fix cannot be measured on clips it trained on, and 66 of
the 69 sustained-English clips remain in training.

⚠️ **Set B's overall WER is NOT comparable to Set A's.** At 39% code_switch vs Batch 3's
batch-wide 24.7%, it is deliberately the harder subset. Read Set B **per bucket**, never as a
single headline number, and never against 10.50%.

### Resulting splits
| Split | Episodes | Clips | Hours |
|---|---|---|---|
| **Train** | 90 (Batch 3 only) | **8,256** | **40.75** |
| **Eval** | 6 B3 + 7 EP | **587** (341 + 246) | 3.01 |

Train buckets: nastaliq_only ~6,262 / code_switch ~1,994 / spiritual_term ~1,159.

⚠️ The eval split is **2.4× round 1's 246 clips**, and in-training eval uses
`predict_with_generate`. That is the main new wall-clock cost; it is why the epoch count comes
down rather than up.

---

## Training config (proposed — `config/training_config.yaml`)

| Setting | Round 1 | **Round 2** | Why |
|---|---|---|---|
| Init | fresh LoRA on base | **resume `/model/whisper-urdu-lora-adapter`** | user decision |
| LoRA | r=32, α=64, 6 targets | **unchanged** | must match the resumed adapter |
| learning_rate | 1.0e-5 | **5.0e-6** | warm start from a converged adapter; halved to limit forgetting |
| per_device_train_batch_size | 8 | 8 | |
| gradient_accumulation_steps | 4 | 4 | effective batch **32** |
| steps/epoch | 63 | **258** | ceil(8256/32) |
| max_steps | 567 (~9 ep) | **774 (~3 ep)** | warm start + 4× data ⇒ far fewer epochs |
| warmup_steps | 57 | **77** | ~10% |
| eval_steps / save_steps | 63 / 63 | **258 / 258** | once per epoch; save%eval==0 so `load_best` works |
| fp16 / gradient_checkpointing | true / true | unchanged | |
| metric_for_best_model | wer | wer (blended) | plus `wer_r1` / `wer_b3` reported, see below |
| GPU / timeout | A10G / 8 h | **A10G / 12 h** | ~4.4 h train + 3 eval passes + 8.8k-clip feature extraction |

**LR is a judgement call, not a derived number.** The tripwire is Set A's WER at epoch 1: if it
lands far above 10.50%, the LR is too high and round 2 is overwriting round 1. That is only
visible if the two eval sources are reported separately — hence `wer_r1` / `wer_b3` below.

**Epoch estimate** from round 1's measured throughput (93.3 clip-passes/min on A10G):
2 ep ≈ 2.9 h, **3 ep ≈ 4.4 h**, 4 ep ≈ 5.9 h — excluding eval passes and preprocessing.

---

## Code changes required (none of these exist yet)

| # | File | Change | Severity |
|---|---|---|---|
| 1 | `scripts/convert_reviewed_manifest.py:49` | `^(EP\d+)_(.+)$` → `^([A-Za-z]+\d+)_(.+)$`; same for `_ep_num` | ⛔ **hard blocker** — does not match `B3001_…`, so every clip reports unmatched and the script exits 1 |
| 2 | `src/dataset_builder.py` | accept **multiple** manifest paths (Batch 3 + round-1's 246 eval clips in one eval split); `_ep_num` generic | ⛔ required for the two-set eval |
| 3 | `modal_app.py::train` | resume via `PeftModel.from_pretrained(model, ADAPTER, is_trainable=True)` instead of `get_peft_model` | ⛔ required |
| 4 | `modal_app.py::train` | **assert trainable params > 0** after load | ⛔ see trap below |
| 5 | `modal_app.py` | version output paths so round 1 is never clobbered | ⛔ see volume hygiene |
| 6 | `modal_app.py::train` | `compute_metrics` reports `wer_r1` / `wer_b3` by slicing the eval order via `eval_buckets.json` | ★ the forgetting tripwire |
| 7 | `modal_app.py::evaluate` | `--dataset-path` + `--model-path` overrides, so base / r1 / r2 can be scored on Set A and Set B | ★ needed for the final table |
| 8 | `modal_app.py::train` | timeout 8 h → 12 h | ★ |
| 9 | `config/training_config.yaml` | steps/warmup/eval_steps/LR per the table above; new `lora.resume_from_adapter` key | ★ |

### ⚠️ THE TRAP — `inference_mode: true`
The round-1 `adapter_config.json` on the volume has `"inference_mode": true`. Loading it with
`PeftModel.from_pretrained(...)` **without `is_trainable=True`** yields a frozen adapter: training
runs to completion, reports a loss curve, saves a model — and has changed **nothing**. This is the
same silent-no-op class as session 014's BUG C and BUG F. Change #4 exists specifically to make it
loud: assert trainable params > 0 rather than only printing them.

---

## Volume hygiene — write to NEW paths, rename nothing

Round 1 currently occupies `/model/whisper-urdu-{lora-adapter,final}` and `/processed/dataset`.
`train()` as written would **overwrite both**: the adapter it is supposed to resume *from*, and
the dataset holding Set A. Renaming on the volume is avoidable risk, so instead:

| Purpose | Path | Access |
|---|---|---|
| Round-1 adapter (resume source) | `/model/whisper-urdu-lora-adapter` | **read-only** |
| Round-1 merged model (eval comparison) | `/model/whisper-urdu-final` | **read-only** |
| Round-1 dataset (Set A) | `/processed/dataset` | **read-only** |
| Round-2 dataset (train + Set A + Set B) | `/processed/dataset_r2` | new |
| Round-2 adapter | `/model/whisper-urdu-r2-lora-adapter` | new |
| Round-2 merged model | `/model/whisper-urdu-r2-final` | new |
| Round-2 checkpoints / logs | `/checkpoints/…-r2`, `/logs/r2` | new |

Upload size: the round-2 dataset embeds audio bytes (round 1 was 1.25 GB for 2,257 clips), so
expect **~4.4 GB** for 8,843 clips. Local disk has 123 GB free.

⚠️ Modal volumes are not permanent — push round 2 to HF Hub immediately after training, to a
**separate repo or revision** from round 1's `mohammad-toseef059/whisper-large-v3-urdu`.

---

## Sequence

```powershell
# ── 0. Code changes 1-9 above, then unit-check the resume path locally ──

# 1. Convert the reviewed export  (needs fix #1)
python scripts/convert_reviewed_manifest.py "data/processed/Batch3/Batch 3_reviewed_manifest.json" `
  --batch-folder Batch3 --output data/processed/Batch3/manifest_reviewed.json

# 2. Build the dataset: Batch 3 train + combined (Set A + Set B) eval  (needs fix #2)
python src/dataset_builder.py `
  data/processed/Batch3/manifest_reviewed.json data/processed/manifest_reviewed.json `
  ./data/processed/dataset_r2 `
  --eval-episodes B3039_<ytid>,B3028_<ytid>,B3012_<ytid>,B3017_<ytid>,B3031_<ytid>,B3033_<ytid>,EP5_vwzNL2oziZs,EP6_SrVnpBqd7bI,EP34_h87EJF0Zvco,EP41_mBtP9NKha1g,EP43_m8-37sgUwUQ,EP44_paAJQ3OKB-8,EP47_a0NiZST0S6Q

# 3. Upload (PowerShell, ROOT-RELATIVE remote paths, ~4.4 GB)
modal volume put whisper-training-vol ./data/processed/dataset_r2 /processed/dataset_r2 --force
modal volume put whisper-training-vol ./config /config --force

# 4. Train — DETACHED (round 1's first attempt died at step 88 on a local network drop)
modal run --detach modal_app.py::train

# 5. Evaluate: base / round 1 / round 2 on BOTH sets  (needs fix #7)
modal run modal_app.py::evaluate --dataset-path /data/processed/dataset --model-path <r1|r2>
modal run modal_app.py::evaluate --dataset-path /data/processed/dataset_r2

# 6. Push to HF Hub immediately (volumes are not permanent)
modal run --detach scripts/push_to_hub.py
```

⚠️ `--eval-episodes` matches **full folder names** (`EP5_vwzNL2oziZs`, not `EP5`) — the B3 ytids
must be filled in from the manifest before running step 2.

---

## Success criteria (set BEFORE the run, so the result cannot be rationalised)

| # | Criterion | Measured on |
|---|---|---|
| 1 | **No forgetting**: Set A overall norm WER ≤ ~11% (i.e. not materially worse than 10.50%) | Set A |
| 2 | **Domain retained**: Set A `spiritual_term` stays near 9.38% | Set A |
| 3 | **Code-switch improves**: Set B `code_switch` beats round 1's model on the same clips | Set B, r1 vs r2 |
| 4 | **Item #16 fixed**: B3039 chunks 043/044 transcribe as English, not fabricated Urdu | B3039, qualitative |

Criterion 4 is the one this round exists for, and it is qualitative on purpose — WER on 3 clips
is not a statistic. Read the actual output text.

---

## Open questions for discussion
1. **Epoch count** — 3 (774 steps) is the proposal. 2 is cheaper and less forgetting-prone;
   4 risks overfitting a warm-started adapter. Round 1 converged at epoch 6 of 9 from scratch.
2. **LR 5e-6** — halved from round 1 by judgement, not measurement. Alternative: keep 1e-5 and
   rely on `load_best_model_at_end` + the epoch-1 Set A tripwire to catch forgetting.
3. **Re-mine `domain_terms.json` from Batch 3?** The `spiritual_term` bucket reads 14.3% on
   Batch 3 vs 30.4% on round 1, almost certainly because the 71 terms were mined from round-1
   transcripts only. Affects bucketing/reporting, not training. Cheap to redo, but it would
   change Set A's bucket membership and break comparability with round 1's per-bucket numbers —
   so if done, report BOTH term sets.
4. **Decoding guards for item #16** — session 014 found the fabrication is *decoding
   instability*, not purely a data gap (B3014 handled 40 English words correctly with `<|ur|>`
   forced, while B3039 failed). Data alone may not fully fix it;
   `condition_on_previous_text` / `compression_ratio_threshold` / `repetition_penalty` in
   `modal_align.py` are untouched. Out of scope for training, but it decides whether criterion 4
   failing means "retrain" or "guard at decode".
