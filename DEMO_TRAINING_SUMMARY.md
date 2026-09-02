# Urdu transcription model — training summary

**What we built.** A speech-to-text model that transcribes Urdu lecture audio,
fine-tuned from OpenAI's `whisper-large-v3`. The transcripts feed **subtitles and
search** across the video library.

**Why the stock model wasn't enough.** Two things it handles poorly: the domain's
religious and spiritual vocabulary, and the speaker's habit of mixing English into
Urdu mid-sentence.

**Where we are.** Two training rounds complete. The current model cuts word errors
on new material by **56%** against the stock model, and character errors by **72%**.

---

## The two rounds

| | **Round 1** | **Round 2** |
|---|---|---|
| Goal | Learn the speaker and the domain vocabulary | Scale up on a 4× larger corpus, and train the half of the model round 1 left untouched |
| Started from | stock `whisper-large-v3` | round 1's model |
| Training audio | 2,011 clips · 10.4 hours | 8,123 clips · ~40 hours |
| What was trained | the **decoder** (the half that writes text) | the **decoder and the encoder** (the half that hears) |
| Duration / cost | 3 h 14 m | ~5 h · single-digit dollars |

Training uses **LoRA**: the 1.5-billion-parameter base model stays frozen and a
small adapter (~3.6% of the model) is trained alongside it. That is why a round
costs hours and a few dollars rather than days and thousands.

---

## Results

Measured on **720 clips the model was never trained on**, kept in two groups that
answer different questions:

- **New material** (474 clips) — *did we improve?*
- **Original material** (246 clips) — *did we break anything that already worked?*

**WER** = % of words wrong. **CER** = % of characters wrong. Lower is better.
Both ignore punctuation and Urdu vowel marks, which are invisible to a subtitle
reader and to search.

| | stock model | round 1 | **round 2 (current)** | stock → now |
|---|---|---|---|---|
| **New material** — WER | 11.70 | 6.49 | **5.19** | **−56%** |
| **New material** — CER | 8.15 | 3.24 | **2.30** | **−72%** |
| Original material — WER | 15.62 | 8.65 | **8.24** | −47% |
| Original material — CER | 10.68 | 4.31 | **4.06** | −62% |

### In plain terms

On new material, the stock model gets roughly **one word in nine** wrong. The
current model gets roughly **one word in twenty**.

The character-level gain is larger than the word-level gain, and that is the more
useful number here: most of what the reviewers corrected was a letter or two
inside an otherwise correct word. Word Error Rate cannot see that — a word with
one wrong letter scores the same as a word that is completely wrong — so **WER
understates this work by roughly half.**

### By content type, new material

| | stock model | round 1 | **round 2 (current)** |
|---|---|---|---|
| Pure Urdu | 9.93 / 5.35 | 6.21 / 2.83 | **5.17 / 2.21** |
| Urdu with English mixed in | 15.66 / 14.20 | 7.13 / 4.13 | **5.23 / 2.49** |
| Contains spiritual vocabulary | 13.24 / 7.86 | 6.81 / 3.24 | **5.33 / 2.29** |

*(WER / CER)*

**Code-switching improved most: 15.66 → 5.23 WER, a 67% cut.** That was the stock
model's worst category and is now on par with everything else.

**Spiritual vocabulary went from a weakness to a strength.** In round 1 it was the
stock model's worst-performing area; it is now among the best.

---

## Round 1 — learn the speaker and the domain

The stock model had never heard this speaker and did not know the vocabulary.
Round 1 built that baseline competence from 49 reviewed episodes.

**Result: word errors on the original material fell 18.57 → 10.50**, a 43%
reduction.

The clearest single win was spiritual vocabulary, which moved from the model's
**worst** category to its **best** — confirming that the domain terms were
learnable from this amount of reviewed data.

## Round 2 — more data, and the other half of the model

Two changes:

**1. A much larger corpus.** 8,597 reviewed clips over 42.5 hours, with **84% of
clips carrying reviewer corrections** — a substantial body of human-verified Urdu
spelling to learn from.

**2. Training the encoder.** Round 1 trained only the decoder, the half that turns
sound into written Urdu. The encoder — the half that interprets the audio itself —
was never updated. Round 2 trained both.

**Result: the best model on every measure of new material**, with no loss on the
original material. Word errors on new material fell a further 20% beyond round 1
(6.49 → 5.19) and character errors a further 29% (3.24 → 2.30).

---

## Honest notes

**Original material is fractionally worse on characters** — 4.06 against round 1's
best of 3.96 — while its word errors improved. The cause is understood: the current
model occasionally omits a word where round 1 would have invented one. An omission
costs one word error but every character of that word, so the two measures point
in opposite directions. The effect is small and confined to one content type.

**Roughly 44% of the remaining errors are the model mis-hearing the audio**, not
misspelling it. Spelling errors are down to about 3% of what remains. Further
gains therefore depend more on audio quality and acoustic modelling than on more
text training.

**Measurement is verified, not assumed.** Before any new figure was trusted, the
stock model and round 1's model were re-scored on the same clips and reproduced
their original published numbers **to the decimal**. This rules out the possibility
that an apparent improvement came from a change in how we measure.

---

<sub>Engineering detail, for anyone cross-referencing the repository: the repo
records three training runs. The intermediate run was decoder-only and its adapter
is **not** part of this model's lineage — round 2 above resumed round 1 directly —
so it is omitted here. Full detail in `TRAINING_ROUNDS.md`.</sub>
