# Urdu Whisper fine-tuning — all rounds, one page

**What this is.** The combined history of every training round on this project:
what each round was trying to do, what it actually trained, and what it measured.
Written to be readable by someone who has never seen the project before.

**Status:** three rounds complete. **Round 3 is the current best model.**

Per-round working documents, kept for detail this page deliberately omits:
[`FULL_WHISPER_TRAINING_RUN.md`](FULL_WHISPER_TRAINING_RUN.md) (round 1) ·
[`ROUND2_TRAINING_RUN.md`](ROUND2_TRAINING_RUN.md) ·
[`round3/`](round3/) (round 3, incl. a fuller glossary).

---

## The project in one paragraph

We fine-tune OpenAI's `whisper-large-v3` speech-to-text model to transcribe
**Urdu** lecture audio accurately. The transcripts feed **subtitles and search**
across a library of 4,000+ videos. The speaker uses domain religious and spiritual
vocabulary, and frequently mixes English into Urdu mid-sentence — both things a
general-purpose model handles poorly. Training uses **LoRA**, which freezes the
1.5-billion-parameter base model and trains a small adapter alongside it, so each
round costs hours and single-digit dollars rather than days.

---

# Read this first — the terms

## The two evaluation sets

All results are measured on **720 held-out clips** the model never trained on.
They form two groups that are **never averaged together**, because they answer
different questions.

| name | also written | n | what it is | the question it answers |
|---|---|---|---|---|
| **Set B** | `b3`, `primary` | **474** (8 episodes) | The newest reviewed corpus, "Batch 3" | **Did we improve?** |
| **Set A** | `r1`, `eval_only` | **246** (7 episodes) | Round 1's original holdout, frozen ever since | **Did we break what already worked?** |

Set A is a **guard rail, not a goal**. It exists to catch *catastrophic
forgetting* — a model retrained on new material getting worse at the old. A round
that improves Set B while wrecking Set A is not progress.

## Buckets — content tags inside those sets

| bucket | what it marks |
|---|---|
| `nastaliq_only` | **Pure Urdu**, no English mixed in |
| `code_switch` | Speaker **mixes English into Urdu** mid-sentence |
| `spiritual_term` | Contains **domain religious/spiritual vocabulary** |

`nastaliq_only` and `code_switch` are **mutually exclusive and exhaustive** —
every clip is exactly one (531 + 189 = 720). `spiritual_term` **overlaps both**
(250 of 720), so bucket counts intentionally sum to more than the clip count.
A row `b3 / nastaliq_only` means *Set B, pure-Urdu clips only*.

## How accuracy is measured

| term | meaning |
|---|---|
| **WER** — Word Error Rate | % of **words** wrong. A word fails if it differs *at all*, so one wrong letter fails the whole word. Lower is better. |
| **CER** — Character Error Rate | % of **characters** wrong. A nearly-right word costs little. Lower is better. |

Written **WER / CER** throughout, e.g. `5.19 / 2.30`.

Both are reported because they disagree informatively. A **dropped word** costs
one word error but *every character* of that word — so CER rising while WER stays
flat points at omissions rather than mistakes. That exact signature appears in
round 3's Set A result.

### Three scoring bases

The same predictions, scored under different rules:

| basis | ignores | where you see it |
|---|---|---|
| **raw** | nothing | Numbers printed *during* training. Always highest. |
| **legacy** | punctuation | Round 1's published **10.50%**. Kept for historical comparability. |
| **bare** | punctuation **and diacritics** | **The reported basis** — every headline table below. |

**Diacritics** (Urdu *harakat*) are small vowel marks like the ` ُ ` in اُحد.
They are optional in written Urdu and invisible to a subtitle reader or a search
query. Scoring them as errors inflated round 2's Set B WER by 1.80 points for
differences no user would ever notice, so since 2026-08-28 they are excluded.

⚠️ **Never compare a raw number to a bare one.** Round 1 is simultaneously
"15.71%", "10.50%" and "8.65%" — same model, three bases. This is the single
easiest way to misread these documents.

## Model anatomy

Whisper has two halves, and errors sort between them:

- The **encoder** *hears* — audio to internal representation. Its failures are
  **mis-hearings**: it heard a different word.
- The **decoder** *writes* — representation to Urdu text. Its failures are
  **spellings**: it heard right and wrote the wrong letter.

**LoRA** attaches small trainable matrices `lora_A` and `lora_B` to the frozen
base. What a module adds is **ΔW = B @ A**. The full set is the **adapter**:
512 modules, 57.7M parameters, ~3.6% of the model. A module is **inert** when
`lora_B` is still exactly zero, because then `B @ A = 0` and it changes nothing.

---

# The rounds at a glance

| | **Round 1** | **Round 2** | **Round 3** |
|---|---|---|---|
| Date | 2026-07-25 | 2026-08-28 | 2026-08-28/29 |
| **Intent** | Teach the model this speaker and this domain, from scratch | Add a 4× larger corpus; fix spelling and code-switching | Train the encoder, which had never trained at all |
| Started from | base model, fresh adapter | round 1's adapter | **round 1's adapter** |
| Training data | 2,011 clips / 10.4 h | 8,123 clips / ~40 h | same as round 2 |
| **Intended to train** | whole adapter | whole adapter | whole adapter |
| **Actually trained** | ⚠️ **decoder only** | ⚠️ **decoder only** | ✅ **decoder + encoder** |
| Learning rate | 1e-5 | 5e-6 | decoder 5e-6 · **encoder 1e-5** |
| Steps (epochs) | 567 (~9) | 762 (3) | 762 (3) |
| Wall clock | 3 h 14 m | ~4.4 h | ~5.2 h incl. evals |
| `weights moved` | — | 640 / 1024 | **1024 / 1024** |

**Rounds 1 and 2 trained the decoder only, and nobody knew.** That was a silent
bug, not a decision — see [The encoder discovery](#the-encoder-discovery) below.

---

# Results — all four models, same 720 clips, same basis

Diacritic-free (**bare**) basis, **WER / CER**, lower is better. Produced by
`scripts/rescore.py` from saved per-clip predictions, so every column is
recomputed under identical rules rather than copied between documents.

| subset | n | base | round 1 | round 2 | **round 3** | base → r3 |
|---|---|---|---|---|---|---|
| **Set B** — new corpus | 474 | 11.70 / 8.15 | 6.49 / 3.24 | 5.67 / 2.52 | **5.19 / 2.30** | **−56% / −72%** |
| **Set A** — must not regress | 246 | 15.62 / 10.68 | 8.65 / 4.31 | 8.31 / 3.96 | **8.24 / 4.06** | −47% / −62% |
| b3 / nastaliq_only ★ | 337 | 9.93 / 5.35 | 6.21 / 2.83 | 5.72 / 2.45 | **5.17 / 2.21** | −48% / −59% |
| b3 / code_switch | 137 | 15.66 / 14.20 | 7.13 / 4.13 | 5.56 / 2.69 | **5.23 / 2.49** | −67% / −82% |
| b3 / spiritual_term | 121 | 13.24 / 7.86 | 6.81 / 3.24 | 6.13 / 2.59 | **5.33 / 2.29** | −60% / −71% |
| r1 / nastaliq_only | 194 | 14.92 / 9.61 | 7.89 / 3.60 | 7.85 / 3.57 | **8.06 / 3.95** | −46% / −59% |
| r1 / code_switch | 52 | 18.16 / 14.48 | 11.40 / 6.80 | 9.98 / 5.36 | **8.89 / 4.47** | −51% / −69% |
| r1 / spiritual_term | 129 | 17.28 / 12.22 | 7.48 / 3.31 | 7.28 / 3.21 | **6.95 / 3.20** | −60% / −74% |

**Round 3 is the best model on every Set B row and on two of three Set A rows.**
The single exception is `r1/nastaliq_only`, which is explained in round 3 below.

Legacy basis, for continuity with round 1's published figures:

| subset | base | round 1 | round 2 | **round 3** |
|---|---|---|---|---|
| **Set A** | 18.57 / 13.91 | **10.50** / 5.44 | 10.30 / 5.06 | 10.31 / 5.21 |
| **Set B** | 14.47 / 9.94 | 8.19 / 4.19 | 7.47 / 3.47 | **7.06 / 3.21** |

---

# Round 1 — teach it the speaker and the domain

**Intent.** The base model had never heard this speaker and did not know the
domain vocabulary. Round 1 established a baseline capability from scratch: accent
and pacing, plus spiritual terms and Arabic recitation.

**What was trained.** A fresh LoRA adapter (r=32, α=64) on six target modules
(`q_proj, k_proj, v_proj, out_proj, fc1, fc2`) over 2,011 clips / 10.4 hours from
49 reviewed episodes. 567 steps, ~9 epochs, LR 1e-5. Best checkpoint was epoch 6;
epochs 7–9 showed mild overfitting.

**Result (legacy basis, Set A):** **18.57% → 10.50%** WER, a ~43% relative cut.

The standout: `spiritual_term` went from the model's **worst** bucket (20.29%) to
its **best** (9.38%) — a 10.91-point drop. This validated including the MLP
modules `fc1`/`fc2` as LoRA targets, which had been added specifically as the
domain-vocabulary lever.

---

# Round 2 — more data, better spelling

**Intent.** A much larger reviewed corpus had become available — Batch 3, 8,597
clips across 42.5 hours, with 84% of clips carrying reviewer edits. Two aims, in
the user's stated priority order:

1. **Primary — transcribe Urdu correctly, with fewer spelling mistakes.**
2. **Secondary — code-switching**, including a known failure where sustained
   English audio produced fabricated Urdu.

**Two changes to how the round was judged**, both consequences of that priority:

- **CER was added.** WER is binary per word, so correcting one letter in a word
  scores identically to not correcting it. The reviewers' edits had median
  character similarity 0.970 — *most of what they taught is invisible to WER*.
  Judging round 2 on WER alone understates it by roughly half.
- **Set A was introduced** as a separate frozen regression set, so improvement on
  new material and damage to old material could be seen apart.

**What was trained.** Resumed round 1's adapter on 8,123 clips. 762 steps,
3 epochs, LR halved to 5e-6 to limit forgetting. Same LoRA shape (it must match
the adapter being resumed).

**Result.** All four pre-registered criteria met. Set B **5.67 / 2.52** bare, no
regression anywhere. CER improved roughly twice as much as WER in nearly every
row — the direct signature of a corpus edited at the character level.

Training was still improving at epoch 3 with no overfitting, which at the time
made "more epochs" look like the cheapest next test.

---

# The encoder discovery

**Between rounds 2 and 3, an audit found that 41% of the adapter had never
received a single gradient.** All 192 encoder LoRA modules — 23.6M parameters —
still had `lora_B` at exactly zero, in *both* completed rounds.

**Why nothing caught it.** `lora_B` initialises to zeros. An untrained module
therefore contributes `B @ A = 0 @ A = 0`, which is *exactly* what a correctly
initialised module contributes at step 0. Parameter counts, loss curves and WER
all look identical either way. Backward passes never raised, because the decoder
half supplied a valid gradient graph.

**Cause**, reproduced rather than inferred: `enable_input_require_grads()` hooks
`get_input_embeddings()`, which for Whisper is the **decoder's** `embed_tokens`.
The encoder's input is a mel spectrogram through frozen convolutions, so it
carries `requires_grad=False`. With gradient checkpointing on, the **reentrant**
checkpoint path decides whether to build a backward graph from the *inputs* of the
checkpointed block — parameters inside it are invisible to that decision.

**Fix:** `gradient_checkpointing_kwargs={"use_reentrant": False}`.

**What this means for rounds 1 and 2.** Both were, unintentionally, decoder-only
fine-tunes. Their results stand — they are real, measured improvements — but they
were achieved with 59% of the intended adapter.

Two permanent guards now exist:

- **`weights moved: N/1024`** — an end-of-training check comparing every LoRA
  tensor's sum of squares before and after. It fails the run if nothing moved.
- **`scripts/inspect_adapter.py`** — reports inert modules and per-half update
  magnitudes from a saved adapter, independent of the training process.

---

# Round 3 — finally train the encoder

**Intent.** Round 2's errors were sorted by *what would fix them*. Of what round 2
still got wrong on Set B: **44.2% was genuine mis-hearing** (encoder), while real
spelling was down to **2.9%** (decoder). The decoder's job looked close to done;
the untrained half was aimed at the largest remaining category.

**Design — one variable.** Round 3 resumed **round 1's** adapter, not round 2's.
Round 2 also resumed round 1, so starting from the same point makes **round 2 the
control**, and the two rounds differ in one concept: *the encoder now trains*.
Resuming round 2 instead would have given the decoder a third pass over the same
data, making any difference unattributable.

**Two learning rates.** The halves were in opposite states — the decoder had
trained twice and needed protecting; the encoder had never been updated and needed
to move. So the encoder got **1e-5** (round 1's rate, known to work for a cold
LoRA half) via a second optimizer parameter group, while the decoder stayed at
5e-6.

**Result: `weights moved: 1024/1024`** (round 2 read 640/1024), and an independent
check of the saved adapter confirmed the encoder reached **0.55×** the decoder's
median update magnitude — so it genuinely trained, and a flat result could not be
dismissed as under-training.

### Scorecard against criteria set before the run

| criterion | target | result | |
|---|---|---|---|
| 1 ★ `b3/nastaliq_only` CER | ≤ 2.45 | **2.21** | ✅ best yet, −9.8% rel |
| 2 ★ Set A CER (no forgetting) | ≤ 3.96 | **4.06** | ❌ missed by 0.10 |
| 3 ★ Set B mis-hearing count | ≪ 544 | **495** | ⚠️ −9.0%, but see below |
| 4 `weights moved` | 1024/1024 | **1024/1024** | ✅ |

**Criterion 2 — missed, and localised.** Set A CER went over by 0.10 (+2.5% rel).
The cause is one bucket: `r1/nastaliq_only` CER 3.57 → 3.95. Every other Set A
bucket improved, `code_switch` sharply (5.36 → 4.47). The triage explains the
mechanism: round 3 **trades insertions for deletions** on Set A (`dropped`
116 → 158, `inserted` 141 → 112). A dropped word costs one word error but every
character of that word — so Set A WER improved (8.31 → 8.24) while CER rose.

**Criterion 3 — the round's actual hypothesis, not confirmed.** Mis-hearing fell
9.0%, but *total* errors fell 8.5%. Its share of what remains barely moved:
**44.2% → 43.7%**. The ambiguous `near_miss` bucket improved faster (−11.9%).

Set B remaining errors, by what would fix them:

| category | base | round 1 | round 2 | **round 3** |
|---|---|---|---|---|
| misheard (encoder) | 845 | 421 | 341 | **318** |
| dropped (encoder) | 852 | 260 | 203 | **177** |
| near_miss (ambiguous) | 542 | 460 | 454 | **400** |
| spelling (decoder) | 37 | 42 | 36 | **34** |
| script_variant (convention) | 64 | 52 | 66 | **63** |
| inserted | 223 | 178 | 140 | **142** |
| **TOTAL scored** | **2563** | **1413** | **1240** | **1134** |

**So: training the encoder made the model broadly better, but did not do what the
diagnosis predicted.** It improved everything by roughly the same 8–9% rather than
preferentially fixing mis-hearing. Since the encoder demonstrably moved at real
magnitude, this cannot be explained away as under-training — the *theory* was
wrong, not the execution.

---

# What we know now

1. **Round 3 is the best model available** and is what should ship: Set B
   **5.19 / 2.30**, better than round 2 on every Set B bucket.
2. **Set A is fractionally worse on CER** (4.06 vs 3.96) and the cause is
   understood — more dropped words on the older corpus. Whether those are genuine
   omissions or a segmentation artifact of the older data is **not yet resolved**.
3. **The encoder-headroom theory did not survive contact with evidence.** Roughly
   half of the remaining error pile is the ambiguous `near_miss` category
   (47.8% on Set B), which has never been examined closely. That, not the
   encoder, is now the largest unexplained mass.
4. **Every round so far has improved the previous one**, but the gains are
   decelerating on Set B WER: 11.70 → 6.49 → 5.67 → 5.19.

## Open items

- Examine the `near_miss` bucket. It is the biggest remaining pile and the
  least understood; what is in it likely determines whether a round 4 is worth
  running at all.
- Resolve the Set A deletion increase — model regression or data artifact.
- ✅ Round 3 is published (private): **`mohammad-toseef059/whisper-large-v3-urdu-r3`**
  — adapter (231 MB) and merged model (6.2 GB). Still to move into the
  `models-training` org; the HF token has no write role there.
- **Round 2's adapter is still volume-only** — Modal volumes are not permanent
  storage, and nothing but a retrain recreates a LoRA adapter.
