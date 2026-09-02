# Round 3 — results

**Status:** ✅ complete — trained, evaluated, scored and triaged. Targets below
were recorded **before** the run produced a number, so the result cannot be
rationalised after the fact.

**Scorecard:** 1 ★ **met** (primary goal, best yet) · 2 ★ **missed** by 0.10 ·
3 ★ **not cleanly met** · 4 **met**.

**One-line verdict:** round 3 is the best model this project has produced on the
new corpus, but it did **not** confirm the theory it was built to test — the
encoder improved everything roughly evenly instead of preferentially fixing
mis-hearing, and it cost a little on the older Set A.

---

# Glossary — read this first if the project is new to you

**What this project is.** Fine-tuning OpenAI's `whisper-large-v3` speech-to-text
model to transcribe **Urdu** lecture audio accurately. The output is used for
**subtitles and search**, which is why some scoring choices below ignore
decorative detail a reader would never notice. Training happens in **rounds**;
this document covers round 3.

## The two evaluation sets

Every number in this file is measured on **720 held-out clips** — audio the model
has never been trained on. They come in two groups, and *they are never averaged
together*, because they answer different questions.

| name | also written | n | what it is | the question it answers |
|---|---|---|---|---|
| **Set B** | `b3`, `primary` | **474** (8 episodes) | The newest reviewed corpus ("batch 3"), the material recent rounds are trying to get better at. | **Did we improve?** |
| **Set A** | `r1`, `eval_only` | **246** (7 episodes) | Round 1's original held-out set, kept frozen ever since. | **Did we break anything we could already do?** |

Set A exists to catch **catastrophic forgetting** — the tendency of a model
retrained on new material to get *worse* at the old material. A round that
improves Set B while wrecking Set A is not progress, so Set A is a guard rail
rather than a goal.

## Buckets — content tags within those sets

| bucket | what it marks |
|---|---|
| `nastaliq_only` | Clips that are **pure Urdu**, no English mixed in. |
| `code_switch` | Clips where the speaker **mixes English into Urdu** mid-sentence, common in these lectures. |
| `spiritual_term` | Clips containing **domain religious/spiritual vocabulary** — the terms a general-purpose model is least likely to know. |

Two structural facts that matter when reading the tables:

- `nastaliq_only` and `code_switch` are **mutually exclusive and exhaustive** —
  every clip is exactly one of them (531 + 189 = 720).
- `spiritual_term` **overlaps both** (250 of 720). It is a tag, not a third
  category, so bucket counts deliberately sum to more than the clip count.

A row like `b3 / nastaliq_only` means *Set B, pure-Urdu clips only*.

## How accuracy is measured

| term | meaning |
|---|---|
| **WER** — Word Error Rate | Percentage of **words** wrong. A word counts as an error if it differs *at all*, so one wrong letter fails the whole word. Lower is better. |
| **CER** — Character Error Rate | Percentage of **characters** wrong. Finer-grained: a word that is nearly right costs little. Lower is better. |

Both are written **WER / CER** throughout, e.g. `5.19 / 2.30`.

A few more terms that appear in the epoch-by-epoch tables:

| term | meaning |
|---|---|
| **blended** | Set A and Set B scored together as one pool of 720 clips. Only the trainer reports this; it is a convenient single progress number, **not** a result — a blended figure can improve while one of the two sets gets worse. |
| **`compute_metrics`** | The function the trainer calls at each checkpoint. Its numbers are **raw** basis and blended, so they compare only to *other in-training numbers*, never to the tables in this file. |
| **`eval_loss`** | The model's internal training objective. Useful as a sanity check that learning is happening; it is not an accuracy figure and means nothing to an end user. |
| **epoch** | One complete pass over the training data. This run did three. |

Why both: they disagree in informative ways. A **dropped word** costs one word
error but *every character* of that word — so a rise in CER with flat WER points
at omissions rather than mistakes. That exact pattern shows up in round 3's Set A
result below.

### Three scoring bases — the same predictions, scored differently

| basis | what is ignored before scoring | used for |
|---|---|---|
| **raw** | nothing | The numbers the trainer prints *during* training. Always the highest. |
| **legacy** (a.k.a. normalized) | punctuation | Round 1's published **10.50%** was measured this way. Kept so historical figures stay comparable. |
| **bare** | punctuation **and diacritics** | **The reported basis**, and what every headline table here uses. |

**Diacritics** (Urdu *harakat*) are small vowel marks such as the ` ُ ` in اُحد.
They are usually optional in written Urdu and invisible to a subtitle reader or a
search query. Scoring them as errors inflated round 2's Set B WER by 1.80 points —
about a quarter of its remaining errors — for differences no user would notice.
Since 2026-08-28 they are excluded. `scripts/rescore.py` prints legacy and bare
side by side so nothing is silently redefined.

⚠️ **Never compare a raw number to a bare one.** Round 1 is *both* "15.71%" and
"10.50%" and "8.65%" depending on basis — the same model, three bases.

## Rounds, and what "the bar" means

| round | what it trained | resumed from |
|---|---|---|
| **base** | nothing — stock `whisper-large-v3` | — |
| **round 1** | decoder only | base |
| **round 2** | decoder only | round 1 |
| **round 3** | decoder **and encoder** | round 1 |

Round 3 resumes **round 1, not round 2** — deliberately. Round 2 also resumed
round 1, so starting from the same point makes **round 2 the control**: the two
rounds differ in one concept only (*the encoder now trains*), and any difference
between them is attributable to that. This is why round 2's figures are called
**"the bar"**: beating round 1 proves nothing, since round 2 already did.

**Criteria 1–4** are pass/fail targets written down *before* the run produced any
number, so a disappointing result cannot be quietly reinterpreted as a success.

## Model anatomy — encoder, decoder, LoRA

Whisper has two halves, and errors sort neatly between them:

- The **encoder** *hears* — it turns audio into an internal representation.
  Its failures are **mis-hearings**: the model heard a different word.
- The **decoder** *writes* — it turns that representation into Urdu text.
  Its failures are **spellings**: it heard correctly and wrote the wrong letter.

**LoRA** (Low-Rank Adaptation) is how the fine-tuning is done cheaply: the 1.5-billion-parameter
base model stays frozen, and small trainable matrices — **`lora_A`** and
**`lora_B`** — are attached to it. What a module actually adds to the frozen
weight is **ΔW = B @ A**. The collection of these matrices is the **adapter**
(here 512 modules, 57.7M parameters, ~3.6% of the model).

| term | meaning |
|---|---|
| **inert** module | `lora_B` is still exactly zero, so `B @ A = 0` — the module changes nothing. Since `lora_B` *initialises* to zero, an untrained module is indistinguishable from a correctly-initialised one at step 0. **This is why rounds 1 and 2 trained no encoder at all without anyone noticing.** |
| **`weights moved: N/1024`** | End-of-run check: how many of the adapter's 1024 tensors actually changed during training. Round 2 read **640/1024**; round 3 reads **1024/1024**. |
| **merged model** | The adapter arithmetically folded back into the base weights, producing a single standard 6.2 GB model that needs no special loading code. What ships. |

## Error triage categories

`scripts/triage_errors.py` sorts each remaining error by **what would fix it**,
using Urdu-aware rules — letters that *sound identical* but are written
differently (س ص ث are all /s/) are treated as spelling, while genuinely
different sounds (ٹ vs ت, retroflex vs dental) are treated as mis-hearing.

| category | means | blame |
|---|---|---|
| `misheard` | Heard a different word | **encoder** |
| `dropped` | A word is missing | **encoder** / segmentation |
| `inserted` | A word was invented | encoder / segmentation |
| `spelling` | Right sound, wrong letter (کثرت → کسرت) | **decoder** |
| `script_variant` | Arabic vs Urdu form of the *same* letter (الله / اللہ) | a **data convention** decision, not a model fault |
| `near_miss` | Word form, morphology, grammatical endings | **ambiguous** — could be either half |
| `diacritic` | Vowel marks only, no letter differs | **not scored** (see bases above) |

⚠️ These are a **heuristic first cut, not a diagnosis.** `near_miss` in particular
is large and genuinely mixed.

---

## What round 3 has to beat

Error rates in **% WER / % CER**, punctuation *and* diacritics removed from both
sides — the reported basis since 2026-08-28, because these transcripts are for
subtitles and search where vowel marks carry nothing. Produced by
`scripts/rescore.py` from saved per-clip predictions.

| subset | n | base | round 1 | **round 2 (the bar)** |
|---|---|---|---|---|
| **Set B** — new corpus | 474 | 11.70 / 8.15 | 6.49 / 3.24 | **5.67 / 2.52** |
| **Set A** — must not regress | 246 | 15.62 / 10.68 | 8.65 / 4.31 | **8.31 / 3.96** |
| b3 / nastaliq_only ★ primary | 337 | 9.93 / 5.35 | 6.21 / 2.83 | **5.72 / 2.45** |
| b3 / code_switch ★ secondary | 137 | 15.66 / 14.20 | 7.13 / 4.13 | **5.56 / 2.69** |
| b3 / spiritual_term | 121 | 13.24 / 7.86 | 6.81 / 3.24 | **6.13 / 2.59** |
| r1 / nastaliq_only | 194 | 14.92 / 9.61 | 7.89 / 3.60 | **7.85 / 3.57** |
| r1 / code_switch | 52 | 18.16 / 14.48 | 11.40 / 6.80 | **9.98 / 5.36** |
| r1 / spiritual_term | 129 | 17.28 / 12.22 | 7.48 / 3.31 | **7.28 / 3.21** |

Legacy basis (punctuation only) is kept for continuity with round 1's published
**10.50%** on Set A; `rescore.py` prints both.

## The question this round answers

Not "is the WER lower" but **"did the mis-hearing category shrink?"** Round 2's
remaining errors on Set B, by what would fix them:

| category | round 2 | round 3 | target |
|---|---|---|---|
| encoder — mis-hearing | **544 (44.2%)** | _tbd_ | **↓ substantially** |
| ambiguous — word forms | 594 (47.9%) | _tbd_ | ↓ or flat |
| convention — Arabic/Urdu forms | 66 (5.3%) | _tbd_ | flat (a data question) |
| decoder — real spelling | 36 (2.9%) | _tbd_ | flat (nearly solved) |
| *(diacritic, not scored)* | *369* | _tbd_ | — |
| **total scored** | **1,240** | _tbd_ | ↓ |

A WER gain with a *flat* mis-hearing count would mean the encoder helped for some
other reason and the diagnosis was wrong. That is worth knowing either way.

## Success criteria, set in advance

1. ★ **`b3/nastaliq_only` CER ≤ 2.45** — the primary goal, must not go backwards.
2. ★ **Set A CER ≤ 3.96** — no forgetting. This is the one to watch: 41% more of
   the model is moving, at double the decoder's rate.
3. ★ **Mis-hearing count on Set B falls materially below 544** — the round's
   actual hypothesis.
4. `weights moved: 1024/1024` at the end of training, versus round 2's 640/1024.

## Did the encoder actually train *enough*?

Separate from whether it helped. Run after the adapter lands:

```
python scripts/inspect_adapter.py <r3-adapter> --compare <r1-adapter-backup>
```

**This is the false-negative guard.** If the encoder's weights barely moved
relative to the decoder's, then 1e-5 was too low for three epochs and the correct
conclusion is *"under-trained"*, not *"the encoder does not help"*. Those two
readings imply completely different next rounds, and only the weights can tell
them apart.

---

## Live log

| when | what |
|---|---|
| launch | `✅ 57,671,680 trainable (3.60%)` · `resume confirmed: 320/512` — 320 is correct, round 1's encoder `lora_B` is still zero |
| launch | `two LR groups — decoder 640 @ 5.0e-06 \| encoder 384 @ 1.0e-05` |
| step 254 | epoch 1 eval — see below |
| step 508 | epoch 2 eval — see below; the epoch-1 Set A CER rise reversed |
| step 762 | epoch 3 eval — every metric improved every epoch |
| end | ✅ **`weights moved: 1024/1024`** — round 2 read 640/1024 |
| end | adapter committed, merge to production format started |

### Epoch 1 (step 254) — no forgetting, ahead of round 2

In-training **RAW** metrics from `compute_metrics`. Not comparable to the
normalized figures in the table above; compare only against round 2's in-training
numbers, and Set A against round 1's raw curve.

| | r2 @ ep1 | **r3 @ ep1** | Δ |
|---|---|---|---|
| blended WER | 12.271 | **12.081** | −0.19 |
| blended CER | 5.129 | **4.977** | −0.15 |
| Set B WER | 10.469 | **10.252** | −0.22 |
| Set B CER | 4.472 | **4.213** | −0.26 |
| **Set A WER** | 15.799 | **15.664** | −0.14 |
| Set A CER | 6.401 | 6.458 | **+0.06** |
| eval_loss | 0.1978 | **0.1923** | −0.006 |

★ **The forgetting risk did not materialise.** Set A WER **15.664** is below round
1's own best of **15.71** and below round 2's 15.799 at the same point — despite
41% more of the model moving at twice the decoder's rate. That was the obvious way
this round could go wrong.

⚠️ Set A **CER** is the single metric that worsened (+0.06, +0.9% rel). Small;
watch it at epochs 2 and 3 rather than reading it now.

Round 3 at epoch 1 reaches what round 2 reached at **epoch 2** on blended WER.
Suggestive, not conclusive — the round's actual hypothesis is about the
mis-hearing category, which only the post-run triage can answer.

**Not a bug:** round 3's blended WER (12.0810098658006) is identical to 13 decimal
places to round 2's *epoch 2*. WER is an integer error count over a fixed word
count, and the two subsets moved in opposite directions relative to that run —
Set B better, Set A worse — summing to the same total. Every other metric differs.

**Eval cost is up:** 1,926 s vs round 2's 1,875 s per pass, so ~32 min each.

### Epoch 2 (step 508) — the Set A CER concern reversed

Round 3's own epoch-over-epoch trend. RAW metrics, so comparable only to other
raw figures.

| | r3 @ ep1 | **r3 @ ep2** | Δ |
|---|---|---|---|
| blended WER | 12.081 | **11.845** | −0.24 |
| blended CER | 4.977 | **4.736** | −0.24 |
| Set B WER | 10.252 | **9.909** | −0.34 |
| Set B CER | 4.213 | **3.860** | −0.35 |
| Set A WER | 15.664 | **15.636** | −0.03 |
| **Set A CER** | 6.458 | **6.432** | **−0.03** |
| eval_loss | 0.1923 | **0.1888** | −0.004 |

★ **The one worsening metric turned around.** Set A CER rose +0.06 at epoch 1;
at epoch 2 it fell instead of continuing. Reading it at epoch 1 would have been
reading noise, which is why the note there said to wait.

★ **Still no forgetting.** Set A WER **15.636** remains below round 1's own best
of **15.71**, two thirds of the way through a run in which 41% more of the model
is moving at twice the decoder's rate.

Blended WER **11.845** is below the **12.081** round 2 reached at its own epoch 2.

⚠️ These are raw in-training numbers on the blended eval set, useful only as a
trend. The decisive round-2 comparison is the post-run one: `scripts/rescore.py`
on saved per-clip predictions, diacritic-free basis, against the table at the top
of this file. Nothing here substitutes for that, and none of it touches the
round's actual hypothesis — the mis-hearing count, which only the triage answers.

### Epoch 3 (step 762) — run complete

| | ep1 | ep2 | **ep3** | ep1 → ep3 |
|---|---|---|---|---|
| blended WER | 12.081 | 11.845 | **11.716** | −0.36 |
| blended CER | 4.977 | 4.736 | **4.682** | −0.30 |
| Set B WER | 10.252 | 9.909 | **9.747** | −0.50 |
| Set B CER | 4.213 | 3.860 | **3.788** | −0.43 |
| Set A WER | 15.664 | 15.636 | **15.573** | −0.09 |
| Set A CER | 6.458 | 6.432 | **6.413** | −0.05 |
| eval_loss | 0.1923 | 0.1888 | **0.1870** | −0.005 |

**Every metric improved at every epoch.** No trade-off appeared between the two
sets and nothing turned over at the end, so the run was not stopped early by
luck — three epochs was not too many.

★ **Set A improved rather than merely holding.** 15.573 raw is below round 1's own
best of **15.71** and below every reading round 2 produced. The forgetting risk
this round was designed around — 41% more of the model moving, at twice the
decoder's rate — did not appear in any form.

★ The epoch-1 Set A CER rise (+0.06) was noise. It fell at both later epochs.

## ✅ Criterion 4 met: the encoder actually trained

```
✅ weights moved: 1024/1024 LoRA tensors changed during training
```

Round 2's identical line read **640/1024**. The 384 encoder tensors that sat inert
through rounds 1 and 2 all moved this time — measured by comparing each
parameter's sum of squares before and after `trainer.train()`, so it cannot be
satisfied by a module that merely *received* a gradient.

Adapter committed to `/data/model/whisper-urdu-r3-lora-adapter` before the merge,
then `✅ Training complete. Model saved to /data/model/whisper-urdu-r3-final`.

⚠️ **1024/1024 proves the encoder moved, not that it moved *enough*.** That is a
separate question, answered next.

## ✅ The false-negative guard clears — the encoder is not under-trained

`python scripts/inspect_adapter.py <r3> --compare <r1>`

| | encoder median ΔW | decoder median ΔW | inert modules |
|---|---|---|---|
| **round 1** | **0.000e+00** | 8.135e-05 | 192 of 512 (the whole encoder) |
| **round 3** | **5.088e-05** | 9.204e-05 | **0 of 512** |

`ΔW = (alpha/r)·B@A` is what a LoRA module actually adds to the frozen base
weight; its RMS is comparable across differently shaped modules, so the two halves
can be put side by side.

★ **encoder / decoder median ΔW = 0.55x.** The encoder went from *exactly zero* to
55% of the decoder's update magnitude. It did not merely receive gradients — it
moved at a real magnitude, so **1e-5 was not too low**. This is what makes the
upcoming eval interpretable: a flat result would mean *the encoder does not help
on this data*, and could no longer be explained away as under-training.

All **1024 tensors changed** versus round 1. This corroborates the in-run
`weights moved` line from an independent source — the saved artifact, rather than
the training process's own bookkeeping.

The decoder also grew (8.135e-05 → 9.204e-05, +13%), which is expected: round 3
gave it a second pass over this data, exactly as round 2 did. That is why round 2
and not round 1 is the control.

Criteria 1–3 remain open until the post-run evals.

---

# Measured result

Three eval passes on the merged production model, then `scripts/rescore.py`.
**Both controls reproduced exactly** — round 1 at Set A **10.50** legacy and the
frozen base at **18.57** — and all twelve of round 1's and round 2's figures below
match the pre-registered table at the top of this file to the decimal. The
instrument is sound, so the round-3 column is round 3.

Diacritic-free basis, **WER / CER**:

| subset | n | round 1 | round 2 (the bar) | **round 3** | |
|---|---|---|---|---|---|
| **Set B** — new corpus | 474 | 6.49 / 3.24 | 5.67 / 2.52 | **5.19 / 2.30** | ✅ |
| **Set A** — must not regress | 246 | 8.65 / 4.31 | 8.31 / **3.96** | **8.24 / 4.06** | ⚠️ |
| b3 / nastaliq_only ★ primary | 337 | 6.21 / 2.83 | 5.72 / **2.45** | **5.17 / 2.21** | ✅ |
| b3 / code_switch | 137 | 7.13 / 4.13 | 5.56 / 2.69 | **5.23 / 2.49** | ✅ |
| b3 / spiritual_term | 121 | 6.81 / 3.24 | 6.13 / 2.59 | **5.33 / 2.29** | ✅ |
| r1 / nastaliq_only | 194 | 7.89 / 3.60 | 7.85 / 3.57 | **8.06 / 3.95** | ❌ |
| r1 / code_switch | 52 | 11.40 / 6.80 | 9.98 / 5.36 | **8.89 / 4.47** | ✅ |
| r1 / spiritual_term | 129 | 7.48 / 3.31 | 7.28 / 3.21 | **6.95 / 3.20** | ✅ |

## ✅ Criterion 1 met — the primary goal

`b3/nastaliq_only` CER **2.21** against a bar of **2.45**: a **9.8% relative**
improvement, and the best figure this metric has reached. Set B improved on every
bucket, `spiritual_term` most (2.59 → 2.29, −11.6%).

## ❌ Criterion 2 missed — Set A CER 4.06 against a 3.96 bar

Over by **0.10**, +2.5% relative. Small, but the criterion was fixed in advance so
that it could not be renegotiated once the number arrived. It is a miss.

**It is localised to exactly one bucket.** `r1/nastaliq_only` CER went 3.57 → 3.95
(+10.6% relative) while every other Set A bucket improved — `code_switch` sharply
(5.36 → 4.47, −16.6%) and `spiritual_term` flat-to-better. Set A is 246 clips and
`r1/nastaliq_only` is 194 of them, so one bucket moving 0.38 carries the whole set
past the bar.

Set A **WER improved** (8.31 → 8.24) while its CER worsened. Fewer words are wrong,
but the words that are wrong are wrong by more characters — the two metrics
genuinely disagree here, which is itself a lead for the triage.

## ⚠️ Criterion 3 — mis-hearing fell, but not preferentially

`scripts/triage_errors.py`, Set B (474 clips). Reports:
`reports/round3_triage_setB.html`, `reports/round3_triage_setA.html`.

| category | round 2 | **round 3** | Δ | fixed by |
|---|---|---|---|---|
| misheard | 341 | **318** | −6.7% | encoder |
| dropped | 203 | **177** | −12.8% | encoder |
| near_miss | 454 | **400** | −11.9% | ambiguous |
| spelling | 36 | **34** | −5.6% | decoder |
| script_variant | 66 | **63** | −4.5% | data convention |
| inserted | 140 | **142** | +1.4% | acoustic |
| **TOTAL scored** | **1240** | **1134** | **−8.5%** | |
| *(diacritic, unscored)* | *369* | *383* | *+3.8%* | |

★ **Mis-hearing: 544 → 495, −9.0%. Total errors: −8.5%.**

The category the round targeted fell at **the same rate as everything else**. Its
share of what remains is effectively unchanged — **44.2% → 43.7%** — and the
ambiguous `near_miss` bucket improved *faster* (−11.9%).

So the honest reading: **training the encoder made the model broadly better, but
did not do what the diagnosis predicted.** The pre-registered criterion asked for
mis-hearing to fall *materially below* 544 — a proportional decline is not that.
Recorded as **not cleanly met** rather than argued into a pass.

This is close to the failure mode named in advance in `README.md`: *"A WER gain
with a flat mis-hearing count would mean the encoder helped for some other reason
and the diagnosis was wrong."* Mis-hearing did not stay flat, but neither did it
shrink preferentially, so the encoder-headroom theory is **not confirmed**.

## The Set A regression, explained

Set A triage (246 clips) locates criterion 2's miss exactly:

| category | round 2 | **round 3** | Δ |
|---|---|---|---|
| **dropped** | 116 | **158** | **+36.2%** |
| **inserted** | 141 | **112** | **−20.6%** |
| misheard | 245 | **228** | −6.9% |
| near_miss | 328 | **322** | −1.8% |
| spelling | 40 | **40** | 0.0% |
| **TOTAL scored** | **926** | **917** | −1.0% |

**Round 3 trades insertions for deletions on Set A**, ending 42 words further
ahead on dropped and 29 behind on inserted.

That resolves the WER/CER disagreement mechanically, and it is not a coincidence:
**a dropped word costs one word error but every character of that word.** Shifting
errors from insertions into deletions, plus a net increase in deletions, moves CER
up while leaving WER flat or slightly better — which is exactly the pattern
observed (WER 8.31 → 8.24, CER 3.96 → 4.06).

Mis-hearing on Set A rose 6.9% (361 → 386), against a 9.0% fall on Set B. The
encoder helped on the corpus it trained on and mildly hurt on the older one.

### ⚠️ A false alarm worth recording

A first run of this comparison reported **77 inert modules** and an 0.58x ratio.
Both were wrong: the safetensors file was still being written, and `safe_open`
memory-maps, so unflushed regions read as zeros. The file had already reached its
final byte count, so a size check would not have caught it — only waiting for the
download to exit did. Inspect adapters only after the transfer has *exited*, not
merely when the size looks right.
