---
base_model: openai/whisper-large-v3
library_name: peft
language:
- ur
pipeline_tag: automatic-speech-recognition
tags:
- whisper
- urdu
- lora
- peft
- speech-recognition
---

# Whisper large-v3 — Urdu (round 3)

LoRA fine-tune of `openai/whisper-large-v3` for **Urdu** lecture transcription,
trained on human-reviewed transcripts. Output is used for **subtitles and search**.

**This is round 3, the current best model.** It is the first round in which the
model's **encoder** was trained — rounds 1 and 2 trained only the decoder.

---

## Results

Measured on **720 held-out clips** (never trained on), reported as **WER / CER**
in percent, lower is better.

**Scoring basis:** punctuation *and* Urdu diacritics (harakat) removed from both
reference and hypothesis. Diacritics are optional in written Urdu and invisible to
a subtitle reader or a search query; scoring them inflated WER by ~1.8 points for
differences no user would notice. **Figures on other bases are not comparable —
see [Scoring bases](#scoring-bases).**

| subset | n | base large-v3 | round 1 | round 2 | **round 3** |
|---|---|---|---|---|---|
| **Set B** — recent corpus | 474 | 11.70 / 8.15 | 6.49 / 3.24 | 5.67 / 2.52 | **5.19 / 2.30** |
| **Set A** — regression guard | 246 | 15.62 / 10.68 | 8.65 / 4.31 | 8.31 / 3.96 | **8.24 / 4.06** |
| Set B / pure Urdu | 337 | 9.93 / 5.35 | 6.21 / 2.83 | 5.72 / 2.45 | **5.17 / 2.21** |
| Set B / code-switched | 137 | 15.66 / 14.20 | 7.13 / 4.13 | 5.56 / 2.69 | **5.23 / 2.49** |
| Set B / spiritual vocabulary | 121 | 13.24 / 7.86 | 6.81 / 3.24 | 6.13 / 2.59 | **5.33 / 2.29** |
| Set A / pure Urdu | 194 | 14.92 / 9.61 | 7.89 / 3.60 | 7.85 / 3.57 | 8.06 / 3.95 |
| Set A / code-switched | 52 | 18.16 / 14.48 | 11.40 / 6.80 | 9.98 / 5.36 | **8.89 / 4.47** |
| Set A / spiritual vocabulary | 129 | 17.28 / 12.22 | 7.48 / 3.31 | 7.28 / 3.21 | **6.95 / 3.20** |

Against the base model on the recent corpus: **−56% WER, −72% CER.**

**Set A** is a frozen holdout from round 1, retained across rounds to detect
catastrophic forgetting. **Set B** is the holdout from the newer corpus. They are
reported separately and **must not be averaged** — they answer different questions.

### Scoring bases

The same predictions score differently depending on what is normalised away:

| basis | ignores | round 3, Set A |
|---|---|---|
| raw | nothing | 11.71 |
| legacy | punctuation | 10.31 |
| **bare** *(reported above)* | punctuation + diacritics | **8.24** |

Round 1's originally published figure of **10.50%** was on the *legacy* basis.
Comparing a figure from one basis to another is meaningless.

---

## Intended use

Batch transcription of Urdu lecture audio from one speaker/domain, for subtitle
generation and search indexing.

**Out of scope:** general-purpose Urdu ASR, other speakers or dialects, real-time
transcription, and any use where a transcription error carries safety, legal, or
medical consequences. This is a domain-adapted model measured on one corpus.

## Limitations

- **~44% of remaining errors are mis-hearings**, not misspellings — the model
  hears a different word. Spelling errors are down to ~3% of what remains, so
  further gains depend more on audio quality and acoustic modelling than on more
  text.
- **Set A CER is marginally worse than round 2** (4.06 vs 3.96) while its WER
  improved. Cause: round 3 omits a word slightly more often where round 2 would
  have invented one. A dropped word costs one word error but every character of
  that word, so the two metrics diverge. Confined to the `Set A / pure Urdu`
  bucket.
- **Music and non-speech audio** may produce a literal `موسیقی` token or
  hallucinated text, as with the base model. Trim non-speech before transcribing.
- Trained on a **single speaker**; performance on other Urdu speakers is unmeasured.

---

## How to use

### Merged model (recommended for inference)

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model = WhisperForConditionalGeneration.from_pretrained("<repo>/r3-merged")
processor = WhisperProcessor.from_pretrained("<repo>/r3-merged")
# then generate with language="ur", task="transcribe"
```

### Adapter on top of the base model

```python
from peft import PeftModel
from transformers import WhisperForConditionalGeneration

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model = PeftModel.from_pretrained(base, "<repo>/r3-lora-adapter")
```

### ⚠️ If you are resuming TRAINING from this adapter

`adapter_config.json` has **`inference_mode: true`**. Loading without
`is_trainable=True` silently freezes every LoRA parameter — training then runs to
completion, loss curves look plausible, and **the adapter does not change**:

```python
model = PeftModel.from_pretrained(base, "<repo>/r3-lora-adapter", is_trainable=True)
```

Verify afterwards that weights actually moved; do not rely on the loss curve.

---

## Training

### Data

Human-reviewed transcripts of Urdu lectures by a single speaker. Round 3 trained on
**8,123 clips (~40 hours)** from a 96-episode reviewed corpus in which **84% of
clips carried reviewer corrections**. Whole episodes were held out for evaluation,
so no episode appears in both train and eval.

### Procedure

| | |
|---|---|
| Method | LoRA — base model frozen |
| LoRA config | r=32, α=64, dropout=0.05 |
| Target modules | `q_proj, k_proj, v_proj, out_proj, fc1, fc2` |
| Trainable | 57.7M params (~3.6% of the model), 512 modules |
| Resumed from | **round 1's adapter** (not round 2's) |
| **Trained halves** | **decoder and encoder** |
| Learning rate | decoder **5e-6**, encoder **1e-5** (separate optimizer groups) |
| Steps / epochs | 762 / 3 |
| Effective batch | 32 (8 × 4 accumulation) |
| Precision | fp16, gradient checkpointing (**non-reentrant**) |
| Hardware | 1× A10G, ~5 h including in-training evaluation |

**Why two learning rates:** the halves were in opposite states. The decoder had
already trained twice and needed protecting; the encoder had never been updated and
needed to move. A single rate would have served one of them badly.

**Why `use_reentrant: False` matters:** with the reentrant gradient-checkpointing
path, the encoder's LoRA modules receive **no gradient** — the path decides whether
to build a backward graph from the checkpointed block's *inputs*, and Whisper's
encoder input is a gradient-free mel spectrogram. This is why rounds 1 and 2
trained no encoder despite intending to, and it fails **silently**: `lora_B`
initialises to zero, so an untrained module contributes exactly what a correctly
initialised one contributes at step 0. Parameter counts, loss curves and WER all
look normal.

### Verification

- **`weights moved: 1024/1024`** LoRA tensors changed during training
  (round 2: 640/1024).
- Encoder update magnitude reached **0.55×** the decoder's, measured as
  `ΔW = (α/r)·B@A` — so the encoder trained at a real magnitude, not marginally.
- **0 inert modules** (round 1 had 192 — its entire encoder).
- Before any round-3 figure was trusted, the base model and round 1's model were
  re-scored on the same clips and reproduced their published numbers **to the
  decimal** (Set A: 18.57 and 10.50, legacy basis).

---

## Lineage

```
openai/whisper-large-v3
  └── round 1  (decoder only, 10.4 h data)
        ├── round 2  (decoder only, +40 h data)      — sibling, not an ancestor
        └── round 3  (decoder + encoder, +40 h data) — THIS MODEL
```

Round 3 resumed **round 1**, not round 2, deliberately: round 2 also resumed round
1, so starting from the same point makes round 2 a clean control differing in one
concept — *the encoder now trains*.

## Licensing

Base model `openai/whisper-large-v3` is Apache-2.0. The fine-tuned weights and the
training data are internal; **confirm the intended license with the repository
owner before redistributing.**

### Framework versions

- PEFT 0.13.2
