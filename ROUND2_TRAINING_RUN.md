# Round 2 — Whisper Large-v3 LoRA Continuation Run (Batch 3)

Planning + tracking doc for the second fine-tuning round. Supersedes nothing — round 1 is
recorded in [`FULL_WHISPER_TRAINING_RUN.md`](FULL_WHISPER_TRAINING_RUN.md) and stays the
baseline. Planned 2026-08-27 (session 015). **NOT YET EXECUTED.**

---

## Goal

Continue the round-1 LoRA adapter on the 96-episode reviewed Batch 3 corpus. Priority order,
**stated by the user 2026-08-27** — this ordering drives the eval design and the success criteria:

1. **PRIMARY — transcribe Urdu speech correctly, with fewer spelling mistakes.**
2. **SECONDARY — code-switching**, including session 014's blocking item #16 (sustained English
   audio producing fabricated Urdu).

…without losing round 1's domain-term gains.

⚠️ This doc originally led with #16 as the primary aim. The reordering changed three things: the
success criteria (which measured only *retention* and *code-switching*, and nothing about Urdu
improving), the metrics (**CER added** — WER cannot see spelling), and Set B's composition
(re-selected to track the corpus distribution instead of over-weighting code-switching). The
training setup itself needed no change; it was already 75.3% Urdu-only.

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

### ★ What "continue from round 1" means concretely (user clarification, 2026-08-27)

**Confirmed intent: round 2 trains the ROUND-1 MODEL, not the base model.** That is what this plan
does. Spelling it out because the mechanics can read the wrong way.

Round 1 left two artifacts on the volume, and they are the *same weights* stored differently:

| Artifact | What it is | Size |
|---|---|---|
| `/model/whisper-urdu-lora-adapter` | the learned delta only | **222 MB** (10 files) |
| `/model/whisper-urdu-final` | base with the adapter already **merged in** | 6.17 GB |

⚠️ **Corrected 2026-08-27:** this adapter is **220.1 MB of weights**, not the "~60 MB" quoted in
round 1's doc and in `CLAUDE.md`. That figure describes the **smoke test's** adapter, which targeted
`q_proj, v_proj` only. Tier-2 adds `k_proj, out_proj, fc1, fc2`, and the MLP matrices are the large
ones — hence 3.7×. Measured from the verified backup, not inferred.

**base large-v3 + round-1 adapter IS the round-1 model.** The base weights were frozen throughout
round 1 — they are the substrate, never the thing trained. So loading `openai/whisper-large-v3` and
attaching round 1's adapter reconstructs exactly the model that scored 10.50%: at step 0 of round 2
its outputs are identical to the round-1 model's. Nothing from round 1 is discarded or diluted.

⚠️ The thing that *would* train the base model is the code as it stands today — `get_peft_model`
builds a **fresh, zero-initialised** adapter (`lora_B` inits to zeros, so its step-0 contribution is
exactly nothing) and **never reads round 1's adapter at all**. In `modal_app.py`, `ADAPTER_PATH` has
three references: one definition and **two writes** (lines 210–211). Nothing loads it. That is why
change #3 is a blocker rather than a refinement.

Two valid readings of "train the round-1 model" were considered:

| | Approach | Verdict |
|---|---|---|
| **(a)** | Keep training round 1's **same** adapter — `PeftModel.from_pretrained(base, adapter, is_trainable=True)` | ✅ **CHOSEN** |
| (b) | Load the **merged** `whisper-urdu-final` as the new base, attach a **fresh** adapter on top | rejected |

Both start from the round-1 model. (a) was chosen because (b) stacks a second adapter's worth of
capacity on top — drifting from the "keep r=32" decision — bakes round-1 knowledge into a frozen
substrate that can no longer be adjusted, loads 6.17 GB instead of 222 MB per cold start, and leaves
a messier artifact lineage in which the final model depends on shipping the merged r1 model too.

---

## ✅ Round-1 adapter BACKED UP before the run (2026-08-27)

**Why it was needed.** `scripts/push_to_hub.py:19` pushes only `/model/whisper-urdu-final`, so:

| artifact | copies before | copies now |
|---|---|---|
| merged model (6.17 GB) | volume + HF Hub ✅ | unchanged |
| **LoRA adapter (222 MB)** | **volume ONLY** ❌ | volume + local ✅ |

`CLAUDE.md` warns Modal volumes are not permanent, and this is the exact artifact round 2 resumes
from. Losing it would keep round 1 usable for **inference** (the merged model is safe on the Hub) but
end the ability to **continue training** from it — LoRA's A/B factors cannot be cleanly recovered from
a merged model. Not a round-2 risk; a pre-existing one that round 2 made relevant.

`Downloads\Whisper Smoke Test Trained Model\` is the **smoke test's** artifacts, not this run's — so
the full run's adapter had never been backed up.

**Backed up to** `Downloads\Whisper Round1 Full Run Adapter\whisper-urdu-lora-adapter`
(10 files, 222.0 MB). Verified: all 10 expected files present, none zero-length,
`adapter_model.safetensors` 220.1 MB, `adapter_config.json` **byte-identical** to the copy pulled
independently from the volume (sha256 `8689f3c8…`), and `r=32 / alpha=64 / 6 Tier-2 targets`.

★ **Positive control run, because a healthy-looking backup of the WRONG adapter is the failure that
would go unnoticed:** compared against the smoke-test copy — `targets=[q_proj, v_proj]`, weights
60.1 MB, sha256 `fefea31c…` vs the backup's `d47b1539…`. Different artifact, confirmed on weights
rather than on metadata alone.

⚠️ Still worth doing: `push_to_hub.py` ignores the adapter entirely, so every future round has this
same single-copy exposure until that script also pushes it.

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

### ★ Why there are TWO eval sets

"Set A" / "Set B" are this plan's shorthand, not a repo convention. They exist because **no single
eval set can answer both questions round 2 raises.** Round 2 changes two things at once — it adds a
new capability (code-switching) while risking an old one (domain terms), so one number cannot
report both. Set A is the **guard rail**; Set B is the **goal**.

| | Set A | Set B |
|---|---|---|
| Corpus | round 1 (49 EP episodes) | Batch 3 (96 episodes) |
| Clips / hours | 246 / 1.28 | 474 / 2.36 |
| nastaliq_only (PRIMARY goal) | 194 (78.9%) | **337 (71.1%)** |
| code_switch | 52 (21.1%) | 137 (28.9%) |
| spiritual_term | 129 (52.4%) | 121 (25.5%) |
| Sustained English (≥10-word run) | **none — round 1 topped out at 8** | present, incl. B3039 |
| Unseen by round 1 / round 2 | yes / yes | yes / yes |
| Answers | *did we keep what we had?* | *did we gain what we wanted?* |

Both are **whole-episode** holdouts, so no near-duplicate clip leaks across the split.

Both are also clean holdouts for **both models at once**, which is what makes r1-vs-r2 a fair
comparison rather than a rigged one: Set A's episodes were held out of round-1 training and are
absent from Batch 3, and Set B is unseen by round 1 because Batch 3 did not exist yet.

### Set A — the round-1 holdout (regression detection)
`EP5_vwzNL2oziZs, EP6_SrVnpBqd7bI, EP34_h87EJF0Zvco, EP41_mBtP9NKha1g, EP43_m8-37sgUwUQ,
EP44_paAJQ3OKB-8, EP47_a0NiZST0S6Q` — **246 clips / 1.28 hrs**, buckets 194 / 52 / 129.

Round 1's own pinned eval set, unchanged. It is the **only set on which 10.50% was ever measured**,
so it is the only place an r1↔r2 comparison means anything — both models must be scored on identical
clips or the delta is noise.

Its job is to catch the characteristic failure of sequential fine-tuning: trading old competence for
new. Batch 3's `spiritual_term` density is 14.3% vs round 1's 30.4%, so the specific worry is round
1's best result (spiritual_term **9.38%**) eroding while overall WER still looks acceptable.

### Set B — the Batch-3 holdout (new-capability measurement)
`B3013, B3017, B3029, B3031, B3039, B3051, B3063, B3076` — **474 clips / 2.36 hrs (5.5% of
Batch 3)**, buckets: **nastaliq_only 337 (71.1%)** / code_switch 137 (28.9%) /
spiritual_term 121 (25.5%).

⚠️ **Re-selected 2026-08-27 after the priority was clarified.** The original pick (option 3 below)
was 38.7% code_switch against Batch 3's real 24.7% — deliberately adversarial for what turned out to
be the *secondary* goal. Option 2 tracks the corpus distribution far more closely (71.1%
nastaliq_only) and nearly doubles `spiritual_term` (121 vs 68), which matters because that bucket is
the **domain-vocabulary spelling** measure. B3039 is retained, and its share of Set B drops from 45%
to 32%, so the concentration noted below eases too.

**Set A physically cannot measure what round 2 is for.** It holds 52 code_switch clips and — because
round-1 data topped out at 8 consecutive English words — **no sustained-English examples at all**.
Asked whether the fabrication bug is fixed, it has nothing to answer with. Set B is built for that
question: 2.5× the code-switch sample, and B3039 as the linchpin.

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

| Option | eval clips | nastaliq_only | code_switch | spiritual | B3039 | eval cost/pass |
|---|---|---|---|---|---|---|
| 1 — adversarial (9 eps) | 407 (4.7%) | 51% | 200 (49%) | 64 | no | 2.7× |
| **2 — representative (8 eps) ← CHOSEN** | **474 (5.5%)** | **337 (71.1%)** | **137 (28.9%)** | **121** | **yes** | **2.9×** |
| 3 — compact + B3039 (6 eps) | 341 (4.0%) | 209 (61.3%) | 132 (38.7%) | 68 | yes | 2.4× |
| *(Batch 3 overall, for reference)* | 8,597 | 75.3% | 24.7% | 14.3% | — | — |

⚠️ **Accepted cost:** B3039's sustained-English clips move from "learn from" to "measure with".
Justified because the fix cannot be measured on clips it trained on, and the great majority of the
69 sustained-English clips remain in training.

⚠️ **Set B's overall WER is still not directly comparable to Set A's** — different corpus, different
reviewers, different era. Option 2 narrows the distribution gap but does not close it. Read Set B
**per bucket**, and never against 10.50%.

### Resulting splits
| Split | Episodes | Clips | Hours |
|---|---|---|---|
| **Train** | 88 (Batch 3 only) | **8,123** | **40.13** |
| **Eval** | 8 B3 + 7 EP | **720** (474 + 246) | 3.64 |

Combined eval buckets: **nastaliq_only 531 (73.8%)** / code_switch 189 (26.2%) /
spiritual_term 250 (34.7%) — close to Batch 3's own distribution, which is what the primary goal
needs.

#### ✅ BUILT AND VERIFIED (2026-08-28) — every predicted number came out exactly

`data/processed/dataset_r2`, **4.7 GB** on disk (over the 4.4 GB estimate because `datasets` embeds
the audio as bytes rather than referencing the wavs). Uploaded to `/processed/dataset_r2`.

| check | result |
|---|---|
| splits | **8,123 train / 720 eval** (8% eval) |
| train episodes | **88** = 96 Batch-3 minus the 8 held out |
| eval buckets | 531 / 189 / 250 — as predicted above |
| held-out episodes leaked into train | **NONE** |
| eval episode set == the 15 pinned | **True** — none missing, none extra |
| round-1 episodes in train | **NONE** — all 2,011 unpinned clips dropped and logged |
| eval by corpus | Set B 474 / 8 eps · Set A 246 / 7 eps |
| sidecar alignment | 720 rows, `source` on every row, 474 primary / 246 eval_only |
| volume config == local config | **identical** (fetched back with `modal volume get` and diffed) |
| round-1 artifacts after upload | `model/whisper-urdu-lora-adapter` + `model/whisper-urdu-final` intact |

★ **Set A's `code_switch` bucket came out at 52 clips — the exact count reproduced from round 1's
pinned eval set (194/52/129).** Independent confirmation that Set A is compositionally identical to
what produced round 1's **10.50%**, so the r1-vs-r2 comparison rests on a measurement rather than an
assumption.

Two operational notes worth keeping:

- **`dataset_builder` must run under `.\venv\Scripts\python.exe`.** The `python` on PATH has `torch`
  but not `datasets`, so the first launch died instantly on `ModuleNotFoundError`. The pre-flight
  suite passes under either interpreter (it needs only `yaml` and `torch`), which is why this did not
  surface until the build.
- **The `modal` CLI on PATH belongs to an unrelated project** —
  `SRTTimeStampPOC\.venv\Scripts\modal.exe`, client 1.5.1. It is what round 1 used, so it was not
  changed mid-run, but it means this project's auth and client version depend on another project's
  venv. Worth pinning `modal` into `WhisperFineTuningPOC\venv`.
- `modal volume get` needs a **Windows-style** destination path; a git-bash-style `/c/...` path fails
  with a bare `No such file or directory`.

⚠️ The eval split is **2.9× round 1's 246 clips**, and in-training eval uses
`predict_with_generate`. That is the main new wall-clock cost; it is why the epoch count comes
down rather than up.

### How the two sets are used together
Set A and Set B go into **ONE combined eval split** (720 clips) in the round-2 dataset, but are
**reported separately**:

- **Combined for checkpoint selection.** The trainer scores all 720 clips each epoch and
  `load_best_model_at_end` therefore picks a checkpoint good at *both*, not one that wins on new
  data by sacrificing old.
- **Separated for reading.** Change #6 reports `wer_r1` / `wer_b3` alongside the blended `wer`.
  Averaging them would hide the exact trade-off the two sets exist to expose.
- **The tripwire:** Set B improving while Set A degrades is the signature of forgetting. Read it at
  **epoch 1** — that is the earliest point at which the learning rate can be judged, and the reason
  the LR proposal (5e-6) is falsifiable rather than a guess.

---

## Training config (proposed — `config/training_config.yaml`)

| Setting | Round 1 | **Round 2** | Why |
|---|---|---|---|
| Init | fresh LoRA on base | **resume `/model/whisper-urdu-lora-adapter`** | user decision |
| LoRA | r=32, α=64, 6 targets | **unchanged** | must match the resumed adapter |
| learning_rate | 1.0e-5 | **5.0e-6** | warm start from a converged adapter; halved to limit forgetting |
| per_device_train_batch_size | 8 | 8 | |
| gradient_accumulation_steps | 4 | 4 | effective batch **32** |
| steps/epoch | 63 | **254** | ceil(8123/32) |
| max_steps | 567 (~9 ep) | **762 (~3 ep)** | warm start + 4× data ⇒ far fewer epochs |
| warmup_steps | 57 | **76** | 9.97% of 762 |
| eval_steps / save_steps | 63 / 63 | **254 / 254** | once per epoch; save%eval==0 so `load_best` works |
| fp16 / gradient_checkpointing | true / true | unchanged | |
| metric_for_best_model | wer | wer (blended) | plus `wer_r1` / `wer_b3` reported, see below |
| GPU / timeout | A10G / 8 h | **A10G / 12 h** | ~4.4 h train + 3 eval passes + 8.8k-clip feature extraction |

**LR is a judgement call, not a derived number.** The tripwire is Set A's WER at epoch 1: if it lands
far above round 1's **raw 15.71%** (⚠️ not 10.50% — see the raw/normalized note in "What to read in
the first minute"), the LR is too high and round 2 is overwriting round 1. That is only visible if the
two eval sources are reported separately — hence `wer_r1` / `wer_b3` below.

✅ **MEASURED, epoch 1:** `eval_wer_r1` **15.80** raw vs round 1's best 15.71 raw — **+0.09, inside
round 1's own epoch-to-epoch spread of 0.27** (15.71→15.98→15.90→15.89). Round 1's competence is
preserved; **5e-6 is vindicated, no intervention needed.**

**Epoch estimate** from round 1's measured throughput (93.3 clip-passes/min on A10G):
2 ep ≈ 2.9 h, **3 ep ≈ 4.4 h**, 4 ep ≈ 5.9 h — excluding eval passes and preprocessing.

---

## Code changes required (none of these exist yet)

| # | File | Change | Severity |
|---|---|---|---|
| 1 | ✅ **DONE** `scripts/convert_reviewed_manifest.py` | `^(EP\d+)_(.+)$` → shared `_FOLDER_RE = ^([A-Za-z]+\d+)_(.+)$`; `_ep_num` → `(prefix, number)` | ⛔ was a **hard blocker** — did not match `B3001_…` |
| 2 | ✅ **DONE** `src/dataset_builder.py` | new **`--eval-only-manifest`** (repeatable) + generic `_ep_num` | ⛔ required for the two-set eval |
| 3 | ✅ **DONE** `modal_app.py::train` | resume via `PeftModel.from_pretrained(model, adapter, is_trainable=True)`, behind `lora.resume_from_adapter` | ⛔ required |
| 4 | ✅ **DONE** `modal_app.py::train` | assert trainable params > 0, **plus a `lora_B` non-zero check** that the resumed weights are trained rather than fresh | ⛔ see trap below |
| 5 | ✅ **DONE** `modal_app.py` | `training.run_tag` versions adapter / merged model / eval_results | ⛔ see volume hygiene |
| 6 | ✅ **DONE** `modal_app.py::train` | `compute_metrics` reports `wer_r1` / `wer_b3` alongside the blended `wer` | ★ the forgetting tripwire |
| 7 | ✅ **DONE** `modal_app.py::evaluate` | `--dataset-path` + `--model-path`, **per-source and source×bucket reporting** | ★ needed for the final table; without it, 6 runs instead of 3 |
| 8 | ✅ **DONE** `modal_app.py::train` | timeout 8 h → 12 h | ★ |
| 9 | ✅ **DONE** `config/training_config.yaml` | `run_tag: r2`, `resume_from_adapter`, `dataset_path`, 762 steps / 3 epochs, LR 5.0e-6 | ★ the switch that turns round 2 on |

### Status of #1–#3 (built 2026-08-27)

**#1** — regexes centralised as `_FOLDER_RE` / `_LABEL_RE`. Run on the real export:
**8,597 / 8,597 across 96 episodes, 0 missing**, written to
`data/processed/Batch3/manifest_reviewed.json` (42.49 hrs). Verified transcripts and durations
survive 1:1 in order, all 8,597 `audio_path`s resolve on disk, no duplicates. `_ep_num` now sorts
`("B3", 1) < ("EP", 5)` numerically within prefix — previously every B3 label returned 0 and so
compared equal. Also made the module docstring raw: `\d` in a non-raw string emitted a
`SyntaxWarning`.

**#2** — `--eval-only-manifest PATH` (repeatable). Clips from it are **eval-eligible only**; any of
its episodes not named in `--eval-episodes` are dropped, counted, and logged.

⚠️ **Why a flag rather than just passing both manifests:** round 1's manifest as a second primary
would put its other **42 episodes into TRAIN**, silently breaking the "Batch 3 only" decision.
Hand-filtering the file instead just relocates the risk into an undocumented artifact that can
drift. The flag makes it structural — an assertion fails the build if an eval-only clip reaches
train. Three guards exit 1: the flag without `--eval-episodes`, an episode present in both corpora,
and an unknown forced episode id (validated against the union, so real eval-only ids still pass).
**6/6 tests pass, including that omitting the flag reproduces the previous output byte-identically.**

**#3** — resume behind `lora.resume_from_adapter`; the fresh-adapter path is preserved, so round 1
stays reproducible. Two guards raise rather than falling through to a fresh adapter: a
`resume_from_adapter` that is not a directory, and one equal to the trainer's own output path.
Also warns when the adapter's `r` / `alpha` / `target_modules` disagree with the yaml, since the
adapter's values are what actually train — the cheapest version of that mistake is resuming the
smoke test's q/v-only adapter while the yaml claims 6 Tier-2 targets.

⚠️ `target_modules` is compared as a **set**, not a list. The volume's order is
`v, out, q, fc1, k, fc2`; the yaml's is `q, k, v, out, fc1, fc2`. A list comparison would have
reported a mismatch on **every** resume.

⚠️ **NOT verified locally:** `torch` / `peft` / `transformers` are not installed outside the Modal
image, so the `PeftModel.from_pretrained` call itself and "trainable params > 0 after load" are
unchecked. That is exactly what change **#4** is for, on the first Modal run. What *was* checked
locally: the drift comparison against the real `adapter_config.json` pulled from the volume (no
drift, as expected), two negative controls proving the detector fires, and the presence of
`is_trainable=True` in the call.

### Status of #4–#5 (built 2026-08-27)

**#5** — one knob, `training.run_tag`. Unset reproduces round 1's layout **byte-identically**
(verified against the three live paths); `run_tag: r2` moves every artifact this run produces:

| artifact | unset (round 1) | `run_tag: r2` |
|---|---|---|
| adapter | `/model/whisper-urdu-lora-adapter` | `/model/whisper-urdu-r2-lora-adapter` |
| merged model | `/model/whisper-urdu-final` | `/model/whisper-urdu-r2-final` |
| eval results | `/logs/eval_results.json` | `/logs/eval_results-r2.json` |

⚠️ **`eval_results.json` was an unnoticed hazard.** `evaluate()` hardcoded that filename, so a
round-2 eval would have **overwritten the file holding round 1's 18.57% / 10.50%** — the baseline
this entire round is measured against, and the one artifact here that cannot be regenerated without
a GPU pass *and* the model that produced it. It was not in the original change list.

Per-path config keys were the alternative and were rejected: forgetting one is silent, and the one
most likely to be forgotten — the dataset — clobbers nothing. It just trains round 2 on round 1's
data and looks like a success. `run_tag` cannot protect a *read* path, so `train()` now warns when a
tagged run points at the untagged `/processed/dataset`, and prints the **full resolved layout**
(dataset, resume source, adapter, merged, checkpoints, logs) before loading anything — "which paths
did that run actually use" is otherwise unanswerable from a finished log.

**#4** — three `raise`s, not warnings, all on the failure modes that otherwise look like success:

1. `trainable == 0` → the adapter loaded **frozen** (the `inference_mode` trap).
2. no `lora_B` parameters at all → the load did not produce a LoRA model.
3. ★ **every `lora_B` tensor is zero** → this is a *freshly-initialised* adapter, so round 1's
   training was never loaded.

Check 3 is the one worth having. A fresh LoRA has `lora_B == 0` in every layer — that is precisely
why a fresh adapter is a no-op at step 0 — so a trained adapter cannot be all-zero. It therefore
distinguishes "resumed round 1" from "silently started from scratch", **the one failure mode a
parameter count and a loss curve look identical under**, and it closes the gap that #3 could not
verify locally. Gated on `resume_from`, since a fresh adapter is legitimately zero.

#### #4b — a fourth check: did the weights actually MOVE? (added 2026-08-28)

All three raises above fire **before step 1**. Together they prove the adapter is loaded and
trainable at step 0 — *not* that it ever changed. One path survives all of them: with `fp16: true`
the gradient scaler **skips** any step whose gradients are `inf`/`NaN`. If every step is skipped, the
run produces a loss curve, three evals, and a saved adapter **byte-identical to round 1's**. Low
probability — round 1 ran fp16 at *twice* this learning rate — but invisible in the artifact
afterwards, which is the whole problem.

So `train()` now snapshots a squared-L2 signature of every trainable `lora_*` tensor before
`trainer.train()` and compares after, raising if **zero** tensors changed.

Two deliberate design choices:

- **The check runs after the adapter save and commit**, for the same reason the commit was moved
  before the merge: a check at the end of a ~6 h run must never be able to destroy training that
  already succeeded. On a true positive nothing is lost — the adapter equals the resume source, which
  is backed up — and raising there **skips the merge**, so no "production" round-2 model is minted
  from untrained weights. If the snapshot is somehow empty the code *warns* rather than raises: an
  inoperative check is not evidence of a bad adapter.
- **Squared L2, in float64** — not `.abs().sum()`, which was the first version and is blind. On a
  tensor with balanced signs a uniform `+ε` cancels term-for-term in an absolute sum (`|x+ε|` shrinks
  where `x<0` by roughly what it grows where `x>0`), and a float32 sum swallows the remainder, so a
  genuinely-changed tensor reads as unchanged. Found by the behavioural test below, not by review.

It correctly stays quiet when `load_best_model_at_end` restores an earlier checkpoint — those weights
still differ from the starting point.

**Live version of the same signal:** `grad_norm`, printed every `logging_steps` (25). A run of `0.0`
or `nan` there *is* this failure happening — kill it rather than waiting for the guard.

### Status of #6–#8 (built 2026-08-27)

**A supporting change first:** `dataset_builder` now writes **`source`** into each
`eval_buckets.json` row — `"primary"` (the corpus being trained on, Set B) or `"eval_only"` (the
retained comparison corpus, Set A). Both #6 and #7 read it. It is *recorded* rather than inferred
from the `EP` label prefix because prefix-sniffing silently mis-splits the moment a batch reuses a
prefix, and mis-splitting is the one thing the whole exercise exists to prevent. A prefix fallback
remains for round 1's existing sidecar, which predates the field — verified to reproduce the
recorded split exactly.

**#6** — `compute_metrics` returns `wer` (blended) **plus** `wer_r1` and `wer_b3`. Checkpoint
selection is deliberately unchanged: `metric_for_best_model` stays on the blend, because a
checkpoint should be good at both. These make the blend's *composition* visible.

⚠️ **Ordering is load-bearing.** The per-source indices are built **before**
`dataset.map(remove_columns=...)`, which strips `sentence`. Built after, there would be nothing left
to align the sidecar against. Asserted in the test.

Why it matters at all: with Set B at 474 of 720 clips and Set A at 246, a 4-point gain on B against
a 4-point loss on A reads as `0.66 × (−4) + 0.34 × (+4) = −1.3` — a *modest improvement*. That is
forgetting, displayed as progress. And it now shows at **epoch 1 (~1.5 h in)** rather than after the
run, which is the difference between wasting one epoch and wasting ~4.4 h of A10G. Zero extra GPU:
the predictions already exist, this slices them.

Since the priority reordering, `compute_metrics` reports **`cer` and `cer_r1` / `cer_b3` too** — the
per-epoch spelling signal, which is what criterion 1 is about. The CER metric is loaded inside a
`try` on purpose: it is a reporting by-product, and an exception at the first eval would kill a ~6 h
run ~90 minutes in. Same reasoning as the `Binarize` catch in `modal_align.py`.

**#7** — `evaluate(which, dataset_path, model_path)`, both falling back to today's behaviour.
Reports overall, per bucket, **per source**, and **source × bucket** — the last being the table the
success criteria are actually read from (`r1/*` = retention, `b3/code_switch` = the gain). The saved
JSON now records `dataset_path`, `which`, and per-source clip counts, so a results file is
interpretable on its own rather than only alongside whatever the config said at the time.

⚠️ Cross-corpus buckets now carry a printed warning. A `code_switch` figure blending round 1's 52
clips with Batch 3's 132 is not a comparison — it is an average of two different questions.

### 🐞 Two bugs caught in my own #7 changes
- **The sidecar was still being read from `cfg["data"]["dataset_path"]`** while the dataset itself
  became overridable. Any `--dataset-path` run would have loaded the *wrong* `eval_buckets.json`,
  failed the length/sentence alignment check, and **silently dropped all bucket and source
  reporting** — printing only overall WER, which is exactly the number the plan says not to trust.
  Now reads from `ds_path`.
- **`run_model(model_path, label)` shadowed the new `model_path` argument.** Correct as written, but
  two different meanings for one name in nested scopes is how the next edit introduces a real bug.
  Renamed to `path`.

**#8** — `train`'s timeout 8 h → 12 h; `evaluate`'s 2 h left alone (three short runs, not one long
one). The rationale is recorded beside it: the overrun is *total*, not partial — the adapter, the
merged model and the only `volume.commit()` all happen after training, and nothing commits during
it. Modal bills time used, not the ceiling, so the margin is free.

### Status of #9 (built 2026-08-27) — ALL 9 CHANGES DONE

`config/training_config.yaml` is the switch: every earlier change added capability that stayed inert
until this file turned it on. **Decisions taken: 3 epochs, LR 5.0e-6.**

| setting | round 1 | **round 2** |
|---|---|---|
| `run_tag` | — | **`r2`** |
| `lora.resume_from_adapter` | — | `/data/model/whisper-urdu-lora-adapter` |
| `data.dataset_path` | `/processed/dataset` | `/processed/dataset_r2` |
| `learning_rate` | 1.0e-5 | **5.0e-6** |
| `max_steps` | 567 (~9 ep) | **762 (3 ep × 254)** |
| `warmup_steps` | 57 | **77** (9.9%) |
| `eval_steps` / `save_steps` | 63 / 63 | **254 / 254** (3 evals) |
| `output_dir` | `…/whisper-large-v3-urdu` | `…/whisper-large-v3-urdu-r2` |
| `logging_dir` | `/data/logs` | `/data/logs/r2` |

**The step arithmetic is verified against the real manifest, not assumed:** Batch 3 minus Set B is
**8,123** train clips, effective batch 8×4 = 32, so `ceil(8123/32) = 254` steps/epoch and 762 is
exactly 3 epochs. `save_steps % eval_steps == 0`, so `load_best_model_at_end` can find the best
checkpoint.

**LR rationale.** Round 1 started from scratch — nothing to preserve, so large updates were free.
Round 2 starts from a model that already knows this domain, and large updates can *overwrite* that.
Crucially this is now **falsifiable at epoch 1**: watch `eval_wer_r1` (change #6). If it has climbed
well above 10.50%, the rate is too high — stop, lower it, restart, having spent ~1.5 h rather than
the run. That is what made a conservative default cheap to choose and cheap to be wrong about.

⚠️ **`5.0e-6`, not `5e-6`.** YAML parses the first as a float and the second as the **string**
`"5e-6"`, which would reach the optimizer as a string. Asserted as a float in the test.

### 🐞 #9 EXPOSED A COLLISION — the three eval runs shared one filename
Setting `run_tag: r2` made all three planned eval runs write `eval_results-r2.json`, so runs 1 and 2
would have been **silently clobbered by run 3** — losing exactly the comparison baselines the round
is judged against, including the r1-on-Set-B number that does not otherwise exist. Fixed by deriving
a label from what was scored:

```
r1 control + missing Set B baseline -> eval_results-r2-whisper-urdu-final.json
secondary anchor (base)             -> eval_results-r2-base.json
the result (round 2)                -> eval_results-r2-whisper-urdu-r2-final.json
```

Derived automatically rather than left to operator discipline, and round 1's `eval_results.json` is
never written again.

### ✅ ORDERING CONSTRAINT RESOLVED — #5 unblocked #9
`ADAPTER_PATH` is `/data/model/whisper-urdu-lora-adapter`, which is *also* where round 1's adapter
lives, so resuming from it while the trainer still saved there would have overwritten the source
adapter partway through the run. #3's collision guard raises on that; **#5 is what clears it** — the
guard now compares against the *run-scoped* output, so `run_tag: r2` makes resume legal.

`lora.resume_from_adapter` and `training.run_tag` are both still **absent** from
`training_config.yaml`, so current behaviour is unchanged and round 1 stays reproducible. #9 sets
them together — and setting `resume_from_adapter` **without** `run_tag` now fails loudly rather than
destroying round 1's adapter.

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

✅ **Implemented as `training.run_tag` (change #5).** The table below is what `run_tag: r2` produces;
leaving it unset reproduces round 1's layout byte-identically.

| Purpose | Path | Access |
|---|---|---|
| Round-1 adapter (resume source) | `/model/whisper-urdu-lora-adapter` | **read-only** |
| Round-1 merged model (eval comparison) | `/model/whisper-urdu-final` | **read-only** |
| Round-1 dataset (Set A) | `/processed/dataset` | **read-only** |
| Round-1 eval results (18.57% / 10.50%) | `/logs/eval_results.json` | **read-only** |
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

# 2. Build the dataset: Batch 3 train + combined (Set A + Set B) eval
#    Round 1's manifest comes in via --eval-only-manifest, so its other 42
#    episodes are DROPPED rather than silently added to train.
python src/dataset_builder.py `
  data/processed/Batch3/manifest_reviewed.json ./data/processed/dataset_r2 `
  --eval-only-manifest data/processed/manifest_reviewed.json `
  --eval-episodes B3013_XqT8W6Aqpug,B3017_FARjVywUYpA,B3029_9fYyFC7TlVg,B3031_rWVxHX5yIRw,B3039_UnaZu-y5G3s,B3051_jnOCvu03zXM,B3063_FfWwQvaXizA,B3076_FS_nv_p3W1g,EP5_vwzNL2oziZs,EP6_SrVnpBqd7bI,EP34_h87EJF0Zvco,EP41_mBtP9NKha1g,EP43_m8-37sgUwUQ,EP44_paAJQ3OKB-8,EP47_a0NiZST0S6Q

# 3. Upload (PowerShell, ROOT-RELATIVE remote paths, ~4.4 GB)
modal volume put whisper-training-vol ./data/processed/dataset_r2 /processed/dataset_r2 --force
modal volume put whisper-training-vol ./config /config --force

# 4. Train — DETACHED (round 1's first attempt died at step 88 on a local network drop)
modal run --detach modal_app.py::train

# 5. Evaluate. THREE runs, not six: dataset_r2's eval split already holds
#    Set A + Set B, so one pass per model yields BOTH, split by source.
#    Run the r1 control FIRST — it validates the harness before any new number
#    is trusted, and supplies the r1-on-Set-B baseline that does not yet exist.
#    --dataset-path is optional now (config points at dataset_r2) but passing it
#    makes each run self-describing in the saved results.
modal run modal_app.py::evaluate --which finetuned `
  --model-path /data/model/whisper-urdu-final        # r1 — Set A must reproduce 10.50%
modal run modal_app.py::evaluate --which base        # base — Set A should reproduce 18.57%
modal run modal_app.py::evaluate --which finetuned `
  --model-path /data/model/whisper-urdu-r2-final     # r2 — the result

# 6. Push to HF Hub immediately (volumes are not permanent)
modal run --detach scripts/push_to_hub.py
```

⚠️ `--eval-episodes` matches **full folder names** (`EP5_vwzNL2oziZs`, not `EP5`) — the B3 ytids
must be filled in from the manifest before running step 2.

---

## ★ Evaluation baselines — what WER is measured against (user clarification, 2026-08-27)

**Terminology first, because it changes what is possible.** WER is always computed against the
**human reference transcript** — never against another model's output. So neither round 1 nor the
base model is a "reference"; both are **comparison baselines**. The reference text is the reviewed
transcript in every case.

**Round 1 is the primary baseline.** The question this round exists to answer is "did we improve on
round 1", so 10.50% on Set A is the number to beat and every success criterion below is framed
against r1, not base.

**Base large-v3 is kept as a secondary anchor**, for two cheap reasons: it is what the project's
targets were set against (`CLAUDE.md`: <13%, <10%), and it distinguishes "worse than round 1" from
the far more serious "worse than no fine-tuning at all".

### What we already have vs what must be run

| Model | Set A (246 clips) | Set B (474 clips) |
|---|---|---|
| Base large-v3 | 18.57% ✅ have | ⚠️ **needs a run** |
| Round 1 | 10.50% ✅ have | ⚠️ **needs a run** |
| Round 2 | needs a run | needs a run |

⚠️ **Set B has no round-1 baseline and cannot have one without a new eval run.** Round 1's model
was never scored on any Batch-3 episode — Batch 3 did not exist then. So the comparison that matters
most, **r1 vs r2 on code-switching**, does not exist in the numbers already on hand. It is cheap
(r1's merged model is on the volume at `/model/whisper-urdu-final`) but it is a separate run, and
without it Set B has nothing to compare against.

### ★ Re-run round 1 on Set A as a POSITIVE CONTROL, before trusting any new number
Even though 10.50% is already stored in `logs/eval_results.json`. Change #7 modifies `evaluate()`,
and reusing a stored figure silently assumes the normalizer and batching are unchanged. If the
modified harness **reproduces 10.50%** on r1, every new number it produces is trustworthy; if it
does not, a harness bug has been found *before* it contaminates the round-2 result.

This project has been repeatedly saved by exactly this move — session 014's `volume_result_stems`
bug returned an empty set unconditionally and was caught only by a positive control against real
volume contents, not by reading the code. An empty or unchanged result proves nothing on its own.

### Only THREE eval runs are needed, not six
Because `dataset_r2`'s eval split **already contains both sets** (720 clips), one pass per model
yields Set A and Set B together — provided `evaluate()` reports per source, which is why change #7
must add the split to `evaluate()` and not only to training's `compute_metrics`. The `episode` field
`dataset_builder` already writes into `eval_buckets.json` is what makes the split derivable
(`EP…` → Set A, `B3…` → Set B), so no new sidecar is needed.

| Run | Model | Yields | Doubles as |
|---|---|---|---|
| **1st** | round 1 | r1 on Set A **and** Set B | **harness control** (Set A must give 10.50%) + the missing code-switch baseline |
| 2nd | base large-v3 | base on both | secondary anchor (Set A should give 18.57%) |
| 3rd | round 2 | r2 on both | the result |

Run the r1 control **first**. It is the cheapest run, it supplies the one baseline that does not yet
exist, and if Set A comes back at anything other than ~10.50% then the harness — not the model — is
what needs fixing, and that is far better learned before the round-2 numbers arrive.

---

## ★ Split shares, and why the eval size is a ONE-WAY decision

### How Batch 3 divides
| | clips | share | hours | episodes |
|---|---|---|---|---|
| **Train** | 8,123 | **94.5%** | 40.13 | 88 |
| **Eval (Set B)** | 474 | **5.5%** | 2.36 | 8 |

The eval split itself is larger than Batch 3's 5.5%, because round 1's episodes join it:
**Set B 474 (65.8%) + Set A 246 (34.2%) = 720 clips.** Round 1's episodes are 34% of what is
evaluated and 0% of what is trained.

Round 1 held out **10.9%** of its corpus; round 2 holds out **5.5%**. That is the same decision
against a bigger corpus, not a loss of rigour — what an eval set needs is absolute size, and Set B
(341) is larger than round 1's entire eval set (246), with `code_switch` at 132 vs round 1's 52.
Round 1 had to spend 10.9% to reach a usable size because its corpus was small.

### ⚠️ The clean holdout is ALREADY EXHAUSTED — 720 clips is all there is
An eval set only means anything on clips the model never trained on. For the **round-2** model:

| data | clips | clean? |
|---|---|---|
| Batch 3 — Set B (8 eps) | 474 | ✅ never trained on |
| Round 1 — Set A (7 pinned eps) | 246 | ✅ never trained on, round 1 *or* 2 |
| Batch 3 — the 88 train episodes | 8,123 | ❌ trained on in round 2 |
| Round 1 — its other 42 episodes | 2,011 | ❌ **trained on in ROUND 1** |

That last row is the easy one to miss: round 2 does not train on round-1 data, but it **inherits
round 1's adapter**, and round 1 trained on those 42 episodes. The learning is in the weights, so
they are contaminated for the round-2 model too.

**Consequence: the eval set cannot be enlarged after training.** There is no uncontaminated labelled
data left. (The 4,000 unlabelled videos support eyeball comparison — as round 1 did with
`compare_transcribe.py` — but not WER, having no references.)

### The asymmetry to decide on
**Holding data out is reversible; training on it is not.**
- Hold out 8% and not need it → nothing lost, that data can train in a future round 3.
- Train on it now → permanently unusable as eval for this model. No later decision undoes it.

**Decision REVISED to 5.5%** (was 4%) when the priority was clarified — see the Set B section. The
extra ~133 eval clips buy a distribution much closer to the corpus (71.1% nastaliq_only vs 61.3%)
and nearly double `spiritual_term`, which is the domain-spelling instrument the primary goal needs.
Cost: 133 training clips and an eval pass at 2.9× rather than 2.4×.

Recorded as a **one-way** choice, and this is exactly why it was worth revisiting *before* the build:
had the dataset already been built and trained on, the 133 clips could not have been reclaimed.

⚠️ **Set B remains somewhat concentrated**: B3039 is 154 of 474 clips = **32%** (down from 45% under
the 6-episode pick). Checked against every criterion and it touches none: retention is measured on
Set A; `code_switch` draws only ~22% of its 137 clips from B3039; and Set B's *overall* WER was never
meant to be read as a headline. It would have been a defect had `code_switch` or `nastaliq_only` been
concentrated too — neither is.

---

## Per-clip predictions are kept (added 2026-08-27)
`evaluate()` computed per-clip predictions and then **discarded** them, so any later question cost a
fresh GPU pass. They are now written to a **separate** sidecar,
`/logs/eval_predictions-<run_tag>-<label>.json`, one row per clip carrying `reference`, `episode`,
`source`, `buckets`, and **each model's output side by side** — so base/r1/r2 diff directly, and
**criterion 4 (does B3039 come out in English?) is readable straight from a file** instead of needing
a separate transcription run.

Separate file rather than inside the scores: the scores file is small enough to read or grep by eye,
and burying ~600 transcripts in it would end that.

✅ **Confirmed not to affect training** (asserted in the test, not assumed): `evaluate` is a distinct
`@app.function` from `train`, loads models under `torch.no_grad()`, never writes weights, writes to
different volume paths, and needed no config change. `eval_predictions_path` /`eval_results_path` are
referenced only inside `evaluate`. Note `train()`'s `pred.predictions` is HuggingFace's
`EvalPrediction` attribute — the same name, an unrelated thing, untouched.

Degrades rather than raising: with a missing or misaligned `eval_buckets.json` the predictions are
still written, minus the per-clip labels. `bmeta` is initialised empty specifically so this cannot
`NameError` at the very end, *after* the GPU time is already spent.

---

## ✅ FINAL RESULT (2026-08-28) — all four criteria met

Plain-language write-up: <https://claude.ai/code/artifact/a5f4f777-b107-4397-beb0-35a5514c4fa9>

Harness validated before anything was read: round 1 re-scored **10.50%** on Set A — its original
figure to the decimal — plus `r1/spiritual_term` **9.38%** and bucket counts **194/52/129**, all
matching round 1's record; base returned **18.57%**. Nothing drifted in the rebuild.

Normalized WER / CER from `modal_app.py::evaluate`:

| subset | n | base | round 1 | **round 2** | ΔWER | ΔCER |
|---|---|---|---|---|---|---|
| **Set B** (new corpus) | 474 | 14.47 / 9.94 | 8.19 / 4.19 | **7.47 / 3.47** | −8.8% | **−17.2%** |
| **Set A** (must not regress) | 246 | 18.57 / 13.91 | 10.50 / 5.44 | **10.30 / 5.06** | −1.9% | −7.0% |
| **b3/nastaliq_only** ★ crit. 1 | 337 | 13.09 / 7.53 | 8.20 / 3.89 | **7.83 / 3.50** | −4.5% | **−10.0%** |
| **b3/code_switch** ★ crit. 4 | 137 | 17.55 / 15.24 | 8.16 / 4.86 | **6.65 / 3.42** | **−18.5%** | **−29.6%** |
| b3/spiritual_term | 121 | 16.40 / 11.54 | 8.65 / 4.47 | **8.15 / 3.83** | −5.8% | −14.3% |
| r1/code_switch | 52 | 20.46 / 17.49 | 12.70 / 7.70 | **11.24 / 6.18** | −11.5% | −19.7% |
| r1/nastaliq_only | 194 | 18.05 / 12.91 | 9.89 / 4.80 | **10.04 / 4.74** | +1.5% | −1.3% |
| r1/spiritual_term | 129 | 20.29 / 16.63 | 9.38 / 4.42 | **9.46 / 4.41** | +0.9% | −0.2% |
| blended | 720 | 15.85 / 11.30 | 8.97 / 4.62 | **8.42 / 4.02** | −6.1% | −13.0% |

★ **CER improves ~2× more than WER in almost every row** — the direct signature of a corpus edited at
median character similarity 0.970 (see line 825). WER is binary per word, so a one-letter correction
scores identically to no correction: **most of what the reviewers taught is invisible to WER.**
Judging this round on WER alone understates it by roughly half, which is the retrospective
justification for adding CER.

Honest scale: round 1 cut Set A 18.57 → 10.50 (~43% rel). Round 2 adds ~11–13% **on top of** round 1's
total gain, with no regression. The two cells that tick up are both <1 point with flat-or-better CER.

Training was still improving at epoch 3 (no overfitting, unlike round 1's epoch-6 peak), so **more
epochs is the cheapest next test** — ahead of open question 0's encoder work.

## Success criteria (set BEFORE the run, so the result cannot be rationalised)

### ★ Stated priority (user, 2026-08-27), and criteria ordered to match it
> **1st — transcribe Urdu speech correctly, with fewer spelling mistakes.
> 2nd — code-switching.**

The criteria were originally two retention checks plus two code-switching checks, i.e. **nothing
measured improvement in the primary goal.** Reordered:

| # | Criterion | Priority | Baseline | Read from |
|---|---|---|---|---|
| **1** | **Urdu improves**: `nastaliq_only` WER **and CER** both fall on fresh Batch-3 Urdu | **PRIMARY** | r1 on Set B — **must be measured, does not exist yet** | `b3/nastaliq_only` |
| 2 | **No forgetting**: overall norm WER ≤ ~11%, not materially worse than round 1 | guard | r1 = 10.50% | Set A |
| 3 | **Domain retained**: `spiritual_term` stays near round 1's best result | guard | r1 = 9.38% | `r1/spiritual_term` |
| 4 | **Code-switch improves**: `code_switch` beats round 1 on identical clips | secondary | r1 on Set B | `b3/code_switch` |
| 5 | **Item #16 fixed**: B3039 chunks 043/044 come out as English, not fabricated Urdu | secondary | the reviewers' corrected English | predictions sidecar, qualitative |

**Criterion 1 needs CER, not just WER.** WER is binary per word — a word missing one diacritic
scores exactly the same as a wholly wrong word — so it cannot tell "spelling improved" from "words
changed". CER counts characters and shows the magnitude. Both are now reported everywhere, at no
extra GPU cost.

Why the training signal should move criterion 1 at all: **7,200 of 8,597 transcripts (83.8%) were
edited by reviewers, at a median character similarity of 0.970** — 82.8% of edits are ≥0.90 similar
and only 5.6% are heavy rewrites. The typical edit changes ~3% of characters, i.e. a few letters in a
~150-character clip. That is 40 hours of dense, surgical spelling supervision, which is exactly what
criterion 1 asks about.

Criterion 5 is qualitative on purpose — WER over 3 clips is not a statistic. Read the actual text,
now available in the predictions sidecar without a separate run.

⚠️ **Criteria 1 and 4 cannot be evaluated until r1 has been run on Set B.** That is why it is the
first eval run: without it, a round-2 number on Set B has nothing to be better *than*, and the
temptation is to read it against Set A — a different corpus, a different distribution, not a valid
comparison.

⚠️ **A gain on 1/4 alongside a regression on 2/3 is the expected failure mode**, not a surprise —
it is what sequential fine-tuning on a differently-distributed corpus does. The lever is the
learning rate (then the epoch count), not the data.

---

## 💰 Expected Modal cost (from ACTUAL round-1 billing, 2026-08-27)

Pulled with `modal billing report --start 2026-07-24 --end … --show-resources --json`.

⚠️ **Round 1's training was billed on 2026-07-24 UTC**, not the 2026-07-25 this project's docs
record — the workspace is billed in UTC and the operator is UTC+5:30, so a run finishing early on
the 25th IST lands on the 24th.

| round-1 training day (2026-07-24) | |
|---|---|
| A10G | **$3.9612** → **3.60 GPU-hours** at $1.10/hr |
| Memory (32 GB requested) | $0.9219 |
| CPU | $0.2489 |
| **day total** | **$5.13** |

**All-in rate $1.43/hr — a 30% non-GPU overhead on the A10G line**, mostly the 32 GB memory
request. A GPU-only estimate understates by that much. Cross-check: 3.60 GPU-h against the
documented 3h14m training run leaves ~0.37 h, which is the baseline eval that ran the same day.

### Round-2 projection

| item | time | cost |
|---|---|---|
| training run | ~3.7–5.9 h | **$5.20–8.50** |
| 3 standalone eval runs (720 clips each) | ~1.3–1.8 h | **$1.90–2.60** |
| HF push (CPU only) | — | **$0.001** (measured) |
| **total** | | **≈ $7–11, most likely ~$9** |

Workload ratios driving it: training clip-passes **1.35×** round 1 (24,369 vs 18,099), in-training
eval **0.98×**, clips to preprocess **3.92×** (8,843 vs 2,257).

⚠️ Weakest part of the estimate: round 1's 194 min was split into phases (train/eval/prep/save) by
inference, not from billed detail. The GPU-hours and the $1.43/hr rate are real; the phase split is
reasoning, hence a ±30% band.

### Project spend to date, for scale
| app | total | of which GPU |
|---|---|---|
| `whisper-urdu-poc` (training + eval) | $6.63 | $5.12 |
| `srt-forced-alignment` (all 96 Batch-3 episodes) | $5.08 | $4.75 |
| `whisper-compare-transcribe` | $0.27 | $0.25 |
| probes | $0.03 | $0.02 |
| **Jul 1 – Sep 1** | **$12.02** | $10.14 |

Transcribing + force-aligning all 96 episodes cost **$5.08**. Round 2 roughly doubles the project's
lifetime Modal spend in one run, and still lands under $12.

★ **A killed run is cheap: ~$2.14** for one epoch. That is change #6's value in money — a wrong
learning rate costs $2 to discover instead of $8.

⚠️ Volume storage did **not** appear as a line item in the report, so it is either below the
reporting threshold or billed elsewhere — not confirmed free. Round 2 roughly doubles the volume
(+4.4 GB dataset, +6.2 GB merged model).

---

## ✅ PRE-FLIGHT CODE REVIEW (2026-08-27) — 0 blocking issues

Full read of the training path before launching. **Two real bugs found, both silent, both in code
written during this session** — see commit `3aed2e2`:

1. **`evaluate()` would MISLABEL every prediction row.** The sidecar is rejected on two conditions
   (wrong length; right length but non-matching sentences), but the predictions sidecar decides
   whether to attach `episode`/`source`/`buckets` labels by comparing **lengths** — so a
   right-length/wrong-content sidecar passed that test and every row got a confident, wrong label.
   Worse than no labels: it would have someone reading another episode's output believing it was
   B3039's. Both branches now discard `bmeta`. The earlier test covered length-mismatch and empty,
   **not** right-length/wrong-content, which is why it passed.
2. **A merge failure would have discarded the adapter too.** `volume.commit()` ran once, after both
   the adapter save and the 6.2 GB merged-model save. The adapter is the only irreplaceable output.
   Now committed before the merge is attempted.

Verified clean, no changes needed:

| check | result |
|---|---|
| config keys the code reads | **21 training + 4 lora, all present in the yaml** |
| `Seq2SeqTrainingArguments` | 21 args, all valid under the `<4.46` pin |
| step schedule vs real manifest | 8,123 clips → 254/epoch → **762 = exactly 3 epochs** |
| `save_steps % eval_steps` | 0, so `load_best_model_at_end` can resolve the best checkpoint |
| `learning_rate` type | float `5e-06` (not the string `"5e-6"`) |
| round-2 vs round-1 paths | **no collision**; resume source ≠ this run's output |
| all 15 eval episodes | resolve → 474 (Set B) + 246 (Set A) = **720** |

⚠️ Two warnings recorded, not fixed — same root cause: **`evaluation_strategy` and
`Trainer(tokenizer=...)` were renamed in transformers 4.46.** Correct under the current pin; they
break the moment the pin moves.

### ✅ The checks now live in the repo, not in a scratchpad
`tests/test_round2_preflight.py` — **17 checks, read-only, no GPU, seconds.**
**Re-run after ANY edit to `training_config.yaml` or `modal_app.py`:**
```
python tests/test_round2_preflight.py      # no pytest needed
```
Ships a `__main__` runner because neither venv has pytest, and skips rather than fails when the
gitignored manifests are absent. Verified with a **mutation-based negative control: 21 deliberate
breakages introduced, 21 caught** — a test that cannot fail is worthless.

Two of the 17 cover #4b, and they cover different things — deliberately:

| check | what it can catch |
|---|---|
| `test_weight_movement_is_checked_after_training` | **structure**: snapshot before `trainer.train()`, comparison between the commit and the merge, and a `raise` rather than a `print` |
| `test_moved_weights_check_answers_correctly` | **behaviour**: runs the real snapshot/compare lines (lifted by AST, not retyped) against a tiny `nn.Module` over 6 scenarios — nothing moved, a real AdamW step at 5e-6, only the frozen base moved, single-tensor nudges, and a balanced-sign tensor |

The structural check alone would have passed the broken `.abs().sum()` signature. The behavioural one
failed it, which is how the cancellation was found — so the last scenario is kept as a regression.

### What to read in the first minute of the run
`train()` prints these before any real GPU time is spent, and they are the confirmation that resume,
`run_tag` and the dataset all point where intended:
1. the **resolved layout** (dataset / resume / adapter / merged / ckpts / logs)
2. `✅ N trainable / M total` — must be non-zero
3. `✅ resume confirmed: N/M lora_B tensors carry trained (non-zero) weights`
4. from **step 25 onward**: `grad_norm` in the logging lines — `0.0` or `nan` means fp16 is skipping
   every step and the run is training nothing. Kill it; don't wait for the end-of-run guard.
5. at **step 254 (epoch 1)**: `eval_wer_r1` — the forgetting tripwire.
   ⚠️ **Compare against 15.71%, NOT 10.50%.** `compute_metrics` reports **RAW** WER; 10.50% is
   `evaluate()`'s **normalized** figure and on identical outputs raw 15.68% == normalized 10.50%
   (`FULL_WHISPER_TRAINING_RUN.md:193`). Round 1's RAW curve on Set A ran
   21.16 / 17.69 / 16.42 / 16.05 / 15.78 / **15.71⭐** / 15.98 / 15.90 / 15.89, base raw 23.87%.
   Read it as: **~15.7–16.0 = round 1 preserved · >17 = concerning · ~20+ = real forgetting** ⇒ stop,
   lower the LR, restart (~$2, not ~$8).

At the **end** of the run, two lines confirm the run was real rather than merely complete:
`✅ adapter committed` then `✅ weights moved: N/M LoRA tensors changed during training`.

---

## Open questions for discussion

0. ★ **ROUND 3: should the encoder train at all?** Round 1 trained **only the decoder** — all 192
   encoder LoRA modules (**23,592,960 params, 40.9% of the adapter**) had `lora_B == 0` and `lora_A`
   still at random init, so they contributed `B @ A = 0` to every forward pass. Round 2 is
   reproducing this, deliberately left unchanged so the r1↔r2 comparison stays attributable.

   Cause: `enable_input_require_grads()` hooks `get_input_embeddings()` = the **decoder's**
   `embed_tokens`. The encoder's input is a mel spectrogram through frozen convs
   (`requires_grad=False`), and the **reentrant** gradient-checkpointing path decides whether to
   build a backward graph from the *inputs* of the checkpointed block — parameters inside are
   invisible to it. Reproduced locally; `backward()` does **not** raise, because the decoder supplies
   a valid graph, which is why round 1 looked healthy.

   Fix: `gradient_checkpointing_kwargs={"use_reentrant": False}` in `Seq2SeqTrainingArguments`.

   For round 3, weigh: the encoder is *acoustic* adaptation (this speaker, the recording chain, Urdu
   phonetics, recitation cadence) whereas spelling — the primary goal — lives in the decoder. Against
   that: +41% learning capacity under an LR tuned for decoder-only, higher forgetting risk, and
   non-reentrant checkpointing's extra saved-tensor bookkeeping may not fit A10G 24 GB at batch 8.
   Needs its own baseline, not a mid-flight change.

   Diagnostic: `python scripts/inspect_adapter.py <adapter> [--compare <other>]`.

1. **Epoch count** — 3 (762 steps) is the proposal. 2 is cheaper and less forgetting-prone;
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
