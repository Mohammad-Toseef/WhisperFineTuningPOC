# Whisper Medium Fine-Tuning POC

## Project Goal
Fine-tune OpenAI Whisper Medium on a single Urdu/English/Arabic code-switching
speaker for accurate batch transcription (text + timestamps) of 4,000+ videos.

---

## Stack
- **Development:** Claude Code (this environment)
- **Training Platform:** Modal (modal.com)
- **Base Model:** openai/whisper-medium (300M parameters)
- **Training Framework:** HuggingFace Transformers + Seq2SeqTrainer
- **Efficient Fine-tuning:** PEFT (LoRA) — Path B
- **Production Inference:** faster-whisper (CTranslate2 runtime)
- **Timestamp Alignment:** WhisperX (wav2vec2 forced alignment)
- **Language:** Python 3.11

---

## Key Design Decisions

### Transcription Convention — Final (CONFIRMED)

| Word Type    | Script        | Handled By              | Example                            |
|--------------|---------------|-------------------------|------------------------------------|
| Urdu words   | Nastaliq      | LoRA fine-tuning        | کرنا، ہے، بات، آج، کے بارے میں   |
| Arabic words | Arabic script | LoRA fine-tuning        | الحمد لله، إن شاء الله، الربا    |
| English words| English       | Base Whisper (frozen)   | account, bank, financing, software |

**Why this works with Path B (LoRA):**
Path B freezes all base Whisper weights. Those frozen weights already contain
Whisper's pre-training on 680,000 hours of multilingual audio — including
excellent English transcription capability. The LoRA adapters only need to
learn Urdu (Nastaliq) and Arabic on top of that. English is handled
automatically by the frozen base, with zero degradation.

This would NOT work as cleanly with Path A (full fine-tuning), where
under-representing English in training data could cause the model to
forget English transcription quality.

**What a correctly transcribed training sample looks like:**
```
Speaker says:
"Aaj hum Murabaha financing ke baare mein baat karein ge,
 yeh ek Islamic finance ka concept hai, Alhamdulillah."

Correct training label:
"آج ہم Murabaha financing کے بارے میں بات کریں گے،
 یہ ایک Islamic finance کا concept ہے، الحمد لله۔"

Breakdown:
  آج ہم              → Nastaliq  (LoRA learns this)
  Murabaha financing  → English   (base model handles — no training needed)
  کے بارے میں        → Nastaliq  (LoRA learns this)
  بات کریں گے        → Nastaliq  (LoRA learns this)
  یہ ایک             → Nastaliq  (LoRA learns this)
  Islamic finance     → English   (base model handles — no training needed)
  کا concept ہے       → Nastaliq + English
  الحمد لله           → Arabic    (LoRA learns this)
```

---

### ⚠️ Arabic/Nastaliq Boundary — Resolve Before Labeling

Arabic-origin words are common in Urdu. Decide which are "Arabic" (Arabic script)
vs "naturalized Urdu" (Nastaliq) before labeling begins. Suggested rule:

- **Arabic script** → Speaker is reciting/quoting Arabic: Quranic verses, duas, hadith
- **Nastaliq** → Arabic-origin word used as part of an Urdu sentence

```
Arabic script (recitation):   الحمد لله رب العالمين، إن شاء الله، ما شاء الله
Nastaliq (used in Urdu):      رِبا، زکوٰۃ، حلال، حرام، اللہ (mid-sentence)
English (spoken in English):  Murabaha, Takaful, Ijara, Riba, Zakat (romanized)
```

Document this reference sheet and give it to every transcriber.
Zero exceptions — the model learns exactly what the labels teach it.

---

### Training Data Focus
Because English is covered by the frozen base model, training samples should
prioritise content that is Urdu-heavy and Arabic-heavy. English-dominant
samples add less value per labeling hour. When selecting which clips to
manually transcribe for training:

- ✅ Prioritise clips with dense Urdu speech + Arabic phrases
- ✅ Include clips that mix all three (natural code-switching)
- ⬇️ Deprioritise clips that are predominantly English

---

### Language Token
- **Primary language token:** Urdu (`<|ur|>`)
- Whisper's default `<|ur|>` behavior tries to output pure Nastaliq for everything.
  Fine-tuning teaches the LoRA adapters to handle the Nastaliq/Arabic/English
  distinction, while the frozen base handles English naturally.

### Audio Format
- 16kHz mono WAV
- Max chunk length: **28 seconds** (2s buffer under Whisper's 30s hard limit)
- Min chunk length: 1 second (discard shorter segments)

### Audio Format
- 16kHz mono WAV
- Max chunk length: **28 seconds** (leaves 2s buffer under Whisper's 30s hard limit)
- Min chunk length: 1 second (discard shorter segments)

### Fine-Tuning Approach: Path B (LoRA) for POC
We use **LoRA (Path B)** during the POC phase:
- Freezes base Whisper weights entirely — no catastrophic forgetting
- ~6GB VRAM vs ~12GB for full fine-tuning
- ~60MB adapter file vs ~1.5GB full checkpoint
- ~1.5x faster training — cheaper to iterate
- At 3–10 hours of training data, accuracy difference vs full fine-tuning is <2% WER

Once the POC validates the approach, optionally switch to **Path A (full fine-tuning)**
by merging the LoRA adapter or running a full training pass with the validated config.

### LoRA Configuration
```python
LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="SEQ_2_SEQ_LM"
)
# Trainable: ~15M / 300M params (5%)
```

### Timestamps: Fine-tuning + WhisperX Alignment
Whisper generates two types of timestamps:
- **Segment-level:** from decoder timestamp tokens — degrades slightly after fine-tuning
- **Word-level:** from cross-attention weights — also degrades slightly

**Solution:** Decouple transcription from alignment using WhisperX:
```
Fine-tuned Whisper  →  accurate TEXT
WhisperX (wav2vec2) →  accurate TIMESTAMPS (±50–100ms word-level)
```
This is the production architecture. Do NOT rely on raw Whisper timestamps
for subtitle or video indexing use cases.

**Timestamp accuracy by approach:**
- Base Whisper only:             ±200–400ms segment, ±300–600ms word
- Fine-tuned Whisper only:       ±300–600ms segment, ±400–800ms word
- Fine-tuned + WhisperX:         ±100–200ms segment, **±50–100ms word** ✅

**Urdu alignment note:** WhisperX Urdu support is available but less mature than English:
- English words: ~±50ms
- Urdu words:    ~±100–200ms
- Arabic terms:  ~±200–300ms (Arabic model used as fallback)

---

## Whisper vs. faster-whisper
These are NOT competing models — same weights, different runtime:
- **HuggingFace Whisper (PyTorch):** used for fine-tuning
- **faster-whisper (CTranslate2):** used for production batch inference (2–4x faster)

After training, convert once:
```bash
ct2-transformers-converter \
  --model ./model/final \
  --output_dir ./model/ct2 \
  --quantization int8 \
  --force
```

---

## Handling Unknown Words — Three-Layer Strategy

**Layer 1: Fine-tuning**
Handles speaker accent, code-switching, common domain vocabulary.

**Layer 2: initial_prompt**
Biases decoder toward rare domain terminology. Include terms in the script
they appear in output — Urdu in Nastaliq, Arabic in Arabic, English in English:
```python
domain_prompt = """
آج ہم Islamic finance کے بارے میں بات کریں گے۔
الحمد لله، إن شاء الله، ما شاء الله،
Murabaha, Takaful, Musharakah, Ijara, Istisna, Meezan Bank,
Zakat, Riba, Halal, Haram, profit margin, account
"""
model.transcribe(audio, initial_prompt=domain_prompt)
```

**Layer 3: Post-processing corrections dictionary**
Catches known recurring substitution errors identified during initial testing.
Note: corrections must match the three-script output format exactly.
```python
corrections = {
    "eastern financing":  "Istisna financing",   # English → English
    "what's up karo":     "WhatsApp کرو",         # English stays English, Urdu in Nastaliq
    "the kaful":          "Takaful",              # English → English
    "agera":              "Ijara",                # English → English
    "mezan bank":         "Meezan Bank",          # English → English
}
```
Build and expand this dictionary from the first batch of transcriptions.

---

## Modal Platform

### Resources
- **App name:** `whisper-urdu-poc`
- **Volume name:** `whisper-training-vol` (persists data + checkpoints + final model)
- **Training GPU:** A10G (24GB VRAM, ~$1.10/hr) — fits Whisper medium + LoRA comfortably
- **Inference GPU:** T4 (16GB VRAM, ~$0.59/hr) — sufficient for faster-whisper INT8

### Volume Layout
```
/data/
├── config/                          # training_config.yaml
├── processed/                       # HuggingFace dataset (.arrow format)
├── checkpoints/                     # Training checkpoints (saved every 500 steps)
│   └── whisper-medium-urdu/
├── model/
│   ├── whisper-medium-urdu-final/   # HuggingFace format (post-training)
│   └── ct2/                         # CTranslate2 format (post-conversion)
└── logs/                            # TensorBoard logs
```

### Important
Modal Volumes are **not permanent** — always download and push the final model
to HuggingFace Hub immediately after training completes.

---

## Model Portability
The fine-tuned model is standard HuggingFace format. No Modal lock-in.

**Download from Modal Volume:**
```bash
modal volume get whisper-training-vol \
  /data/model/whisper-medium-urdu-final ./model/final
```

**Push to HuggingFace Hub (do this immediately after training):**
```python
model.push_to_hub("your-org/whisper-medium-urdu", private=True)
processor.push_to_hub("your-org/whisper-medium-urdu", private=True)
```

**Use anywhere after that:**
```python
model = WhisperForConditionalGeneration.from_pretrained("your-org/whisper-medium-urdu")
```

---

## Project File Structure
```
whisper-urdu-finetuning/
├── CLAUDE.md                        ← This file
├── requirements.txt
├── modal_app.py                     ← Main Modal entry point
│                                       Functions: train, evaluate, transcribe_batch
├── src/
│   ├── data_prep.py                 ← Extract audio from videos, chunk to ≤28s, validate
│   ├── dataset_builder.py           ← Build HuggingFace dataset from manifest.json
│   ├── train.py                     ← Training logic (runs inside Modal GPU container)
│   ├── evaluate.py                  ← WER computation on held-out test set
│   └── transcribe_batch.py          ← Bulk inference: faster-whisper + WhisperX alignment
├── config/
│   └── training_config.yaml         ← All hyperparameters (lr, steps, batch size, etc.)
└── scripts/
    ├── upload_data.py               ← Push prepared data to Modal Volume
    └── download_model.py            ← Pull fine-tuned model locally + push to HF Hub
```

---

## Data Requirements

| Training Data | Hours   | Expected Outcome                                      |
|---------------|---------|-------------------------------------------------------|
| Minimal       | 1–3 hrs | Noticeable improvement on speaker accent/style        |
| POC target    | 5–10 hrs| Strong improvement on code-switching and vocabulary   |
| Solid         | 10–20 hrs| Near-optimal for this domain                         |
| Diminishing   | 20+ hrs | Marginal gains beyond this range                      |

**Quality rules:**
- Every transcription error in training labels directly teaches the model wrong output
- Nastaliq/Arabic/English boundary must be applied consistently — zero exceptions
- Prioritise Urdu-heavy and Arabic-heavy clips when selecting training samples
- English-dominant clips contribute less training value — deprioritise them
- Cover diverse topics and energy levels the speaker uses
- Match audio quality of actual videos (don't use only studio-clean samples)

---

## WER Targets
| Stage | Expected WER |
|---|---|
| Base Whisper Medium (no fine-tuning) | ~35–45% |
| After fine-tuning (POC, 5 hrs data)  | ~20–25% |
| After fine-tuning (full, 10+ hrs)    | <20%     |
| With initial_prompt + corrections    | <15%     |

---

## Commands Reference

```bash
# ── Local Data Preparation ──────────────────────────────────────────
python src/data_prep.py \
  --input_dir ./raw_videos \
  --transcript_dir ./transcripts \
  --output_dir ./data/processed

python src/dataset_builder.py

# ── Upload to Modal ─────────────────────────────────────────────────
modal volume put whisper-training-vol ./data/processed /data/processed
modal volume put whisper-training-vol ./config /data/config

# ── Training ────────────────────────────────────────────────────────
modal run modal_app.py::train                  # Streams logs live

# ── Evaluation ──────────────────────────────────────────────────────
modal run modal_app.py::evaluate               # Computes WER on test set

# ── Download Model ──────────────────────────────────────────────────
modal volume get whisper-training-vol \
  /data/model/whisper-medium-urdu-final ./model/final

# ── Convert to faster-whisper ───────────────────────────────────────
ct2-transformers-converter \
  --model ./model/final \
  --output_dir ./model/ct2 \
  --quantization int8 \
  --force

# ── Batch Transcribe All 4,000 Videos ──────────────────────────────
modal run modal_app.py::transcribe_batch

# ── Push to HuggingFace Hub (do immediately after training) ─────────
python scripts/download_model.py --push-to-hub your-org/whisper-medium-urdu
```

---

## Key Dependencies
```
transformers>=4.40.0
datasets>=2.18.0
evaluate>=0.4.0
jiwer>=3.0.3
torch>=2.2.0
torchaudio>=2.2.0
accelerate>=0.28.0
peft>=0.10.0           # LoRA (Path B)
ctranslate2>=4.0.0     # faster-whisper conversion
faster-whisper>=1.0.0  # Production inference
whisperx>=3.1.0        # Timestamp alignment
soundfile>=0.12.1
librosa>=0.10.1
tensorboard>=2.16.0
modal>=0.62.0
pyyaml>=6.0
```

---

## POC Phases
1. **Data Prep** — Manually transcribe 3–5 hrs, run data_prep.py + dataset_builder.py
2. **Baseline** — Transcribe 50 test clips with base model, document WER (~35–45%)
3. **Path B POC** — Fine-tune LoRA on 3 hrs, evaluate WER on same 50 clips
4. **Iterate** — Expand to 5–7 hrs, re-train, target WER <25%
5. **Timestamp validation** — Run WhisperX on sample clips, verify alignment quality
6. **Production** — Convert to faster-whisper + WhisperX, batch transcribe all 4,000 videos