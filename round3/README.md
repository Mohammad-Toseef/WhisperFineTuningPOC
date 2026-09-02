# Round 3 — train the encoder

**Status:** ✅ complete — trained 2026-08-28 (Modal app `ap-ymw97lY4HQZmytWLTMU5ex`,
`weights moved: 1024/1024`), evaluated, triaged, and **published** to
[`mohammad-toseef059/whisper-large-v3-urdu-r3`](https://huggingface.co/mohammad-toseef059/whisper-large-v3-urdu-r3)
(private). See [`RESULTS.md`](RESULTS.md) for the scorecard and
[`MODEL_CARD.md`](MODEL_CARD.md) for the Hub card.

Everything about this round lives in this folder. Rounds 1 and 2 are documented in
`FULL_WHISPER_TRAINING_RUN.md` and `ROUND2_TRAINING_RUN.md` at the repo root.

| | |
|---|---|
| [`README.md`](README.md) | this file — premise, design, setup, what to watch |
| [`RESULTS.md`](RESULTS.md) | filled in as the run produces numbers |

---

## Why this round exists

**Rounds 1 and 2 trained the decoder only.** All 192 encoder LoRA modules —
**23,592,960 parameters, 41% of the adapter** — never received a single gradient.
Their `lora_B` stayed at zero and their `lora_A` at random init, so the entire
encoder contributed `B @ A = 0 @ A = 0` to every forward pass, in both rounds.

Nothing could see it. A parameter count, a loss curve, and a WER look identical
whether the encoder learns or is inert, because `lora_B` initialises to zeros — an
untrained module contributes exactly what a correctly-initialised one contributes
at step 0.

**Cause**, reproduced rather than inferred: `enable_input_require_grads()` hooks
`get_input_embeddings()`, which for Whisper is the **decoder's** `embed_tokens`.
The encoder's input is a mel spectrogram through frozen convs, so
`requires_grad=False`, and the **reentrant** gradient-checkpointing path decides
whether to build a backward graph from *the inputs of the checkpointed block* —
parameters inside it are invisible to that decision. `backward()` does not raise,
because the decoder half supplies a valid graph. That is why it survived two
rounds.

**Fix:** `gradient_checkpointing_kwargs={"use_reentrant": False}`.

## Why it is worth doing now

Round 2's errors were sorted by what would fix them (`scripts/triage_errors.py`).
Of what round 2 still gets wrong on the new corpus:

| what would fix it | share |
|---|---|
| **encoder — genuine mis-hearing** | **44.2%** |
| ambiguous — morphology, word forms | 47.9% |
| convention — Arabic vs Urdu letter forms | 5.3% |
| decoder — real spelling | **2.9%** |

Real spelling is close to solved. The untrained half is aimed at the largest
remaining category.

⚠️ Caveat carried forward: the *ambiguous* bucket is 48% and under-examined. Some
of it is likely mis-hearing and some is decoder-side word choice. A skim of the
purple `near_miss` samples in a triage report would sharpen this — one such skim
already moved the ع family (`عاجزی` → `آجزی`) out of ambiguity into spelling.

---

## Design: one variable

**Round 3 resumes ROUND 1's adapter — not round 2's.** Round 2 also resumed round
1, so starting from the same point makes round 2 the control:

| | round 2 (control) | **round 3** |
|---|---|---|
| resume from | round 1 adapter | **round 1 adapter** |
| dataset | `dataset_r2` | **`dataset_r2`** |
| steps / epochs | 762 / 3 | **762 / 3** |
| decoder LR | 5e-6 | **5e-6** |
| **encoder** | **frozen** | **trains @ 1e-5** |

Resuming round 2 instead would also give the decoder a *third* pass over the same
data, and any difference would be unattributable between "the encoder helped" and
"more decoder training helped". Nothing is discarded by starting from round 1:
round 3's decoder gets the same training round 2's decoder got.

### Two learning rates

The halves are in opposite states. The decoder has trained on this domain twice
and needs protecting; the encoder has never been updated and needs to move. One
rate serves one of them badly — at 5e-6 a cold encoder may barely shift in three
epochs, and the run would read as *"the encoder does not help"* when it merely
under-trained.

So the encoder gets **1e-5** — round 1's rate, which is what a cold LoRA half was
successfully trained at before — through a second optimizer parameter group. The
scheduler scales both groups proportionally, so warmup and linear decay still
apply to each.

This keeps round 3 a one-*concept* change against round 2: *the encoder now
trains, at a rate suited to a cold start.* The decoder's treatment is untouched,
so a Set A regression remains attributable.

`encoder_attn` is the **decoder's** cross-attention; its parameter names contain
`.encoder_attn.`, not `.encoder.`, so the split predicate leaves it on the decoder
side. Confirmed on the real model: **384 encoder / 640 decoder** LoRA tensors
(192×2 and 320×2 modules).

---

## Pre-launch probe

`modal run modal_app.py::probe` — writes nothing (`/tmp` output dir,
`save_strategy="no"`, no `save_pretrained`, no `volume.commit()`), so it cannot
damage an artifact however it is invoked.

```
resume <- /data/model/whisper-urdu-lora-adapter
gradient_checkpointing=True  use_reentrant=False
two LR groups — decoder 640 tensors @ 5.0e-06 | encoder 384 tensors @ 1.0e-05

encoder   384/384 LoRA tensors moved        ← round 2's real run moved ZERO
decoder   640/640 LoRA tensors moved
peak VRAM 10.8 GB of 23.7 GB (46%)
16.8 s/step
```

Three questions answered before spending anything: **the encoder trains** (on real
Whisper, 32 checkpointed layers behind a frozen conv stack — not a toy repro),
**it fits** with more than half the card spare so batch 8 stands, and **a step
costs ~16.8 s** versus round 2's ~10.3, so the encoder adds ~60%.

### The probe found three bugs in itself first

Recorded because each would have cost real money or a wrong conclusion:

1. It checked `p.grad is not None` and raised **"THE ENCODER RECEIVED NO GRADIENTS
   — do not launch"** on a run where all 384 encoder tensors had moved. The
   Trainer zeroes gradients after each optimizer step, so post-hoc `.grad` is
   `None` for *both* halves. It now checks **movement**, which is also the
   stronger claim: a parameter cannot move unless a real gradient was applied.
2. It built its own Trainer **without** the dual-rate groups, so a 3.6-hour run
   would have depended on an optimizer construction that had never executed. Both
   entry points now call one shared `build_dual_rate_optimizer`, and a test
   asserts both do.
3. `train_block()` in the pre-flight suite sliced text from `def train(` to the
   Evaluation banner. `probe()` landed between them, so three ordering assertions
   were silently reading across two functions. Now extracted by AST.

---

## Pre-flight

`python tests/test_preflight.py` — **19 checks, seconds, no GPU.** Renamed off its
round-2 identity, which was not cosmetic: it asserted `run_tag == "r2"` and
*"resume source should be round 1's adapter"*, so it would have **failed on a
correct round-3 config**. A test that cries wolf on correct input trains people to
ignore it.

It now checks invariants rather than one round's literal values: the tag is
non-empty and not already used, the resume source is a round that exists and is
not this run's own output, and no output path collides with **any** prior round.

Two checks are specific to this round, and both have two halves because the second
half is what fails silently:

- `test_encoder_training_is_actually_enabled` — the flag says non-reentrant **and**
  the trainer is actually given it. A config key nothing reads would look correct
  and change nothing.
- `test_encoder_gets_its_own_optimizer_group` — a distinct encoder rate exists
  **and** both entry points use the shared builder.

Verified with **11 deliberate mutations, 11 caught**, including flipping
`use_reentrant` back to `true` and deleting the kwargs from `modal_app.py`.

---

## What to watch

1. `two LR groups — decoder 640 @ 5.0e-06 | encoder 384 @ 1.0e-05`
2. `✅ resume confirmed: 320/512 lora_B tensors carry trained weights` —
   **320 is correct**, not a warning: round 1's encoder `lora_B` is still zero.
3. `grad_norm` from step 25 — `0.0` or `nan` means fp16 is skipping every step.
4. **Step 254 (epoch 1): `eval_wer_r1`.** ⚠️ Compare against round 1's **raw
   15.71%**, *not* the normalized 10.50% — `compute_metrics` reports RAW WER.
   ~15.7–16.0 = preserved · >17 = concerning · ~20+ = real forgetting.
   Round 2 read 15.80 here.
5. **End of run: `weights moved: 1024/1024`** — versus round 2's 640/1024. This is
   the proof the encoder trained.

**Estimate:** ~3.6 h training + ~1.6 h for three in-training evals ≈ **5.2 h**,
about **$8–9** at the measured $1.43/h.

---

## After the run

1. `python scripts/inspect_adapter.py <r3 adapter> --compare <r1 adapter>` — how
   far each half actually moved. **This is the false-negative guard:** if the
   encoder barely shifted, it under-trained and 1e-5 was too low — a different
   conclusion from "the encoder does not help".
2. Three eval passes, r1 control first (must reproduce **10.50%** normalized on
   Set A — it did, exactly, for round 2).
3. `scripts/rescore.py` for the diacritic-free figures, which are the reported
   basis.
4. `scripts/triage_errors.py --html` — did the mis-hearing category actually
   shrink? That is the question this round exists to answer.
5. Push the adapter: `modal run scripts/push_to_hub.py --run-tag r3`.
