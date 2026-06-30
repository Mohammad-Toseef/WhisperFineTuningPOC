# Whisper Medium Fine-Tuning POC

Fine-tuning OpenAI Whisper Medium (via LoRA) on a single Urdu/English/Arabic
code-switching speaker, for accurate batch transcription (text + timestamps)
of 4,000+ videos.

Full architecture, design decisions, and rationale live in [CLAUDE.MD](CLAUDE.MD).
This file is a practical entry point — what exists, what works, and how to run it.

## Status

**Data prep pipeline — working, validated on EP1–EP23.**
- `src/srt_audio_prep.py` converts an audio + SRT pair into ≤28s training
  chunks (gap-aware, silence-snapped, sentence-boundary-aware) and a `manifest.json`.
- `src/batch_srt_prep.py` runs that across a folder of multiple audio+SRT
  pairs (matched by shared YouTube ID, handles IDs containing underscores).
- `src/normalize_manifest.py` cleans transcripts in a manifest — expands
  Unicode shorthand symbols, removes invisible characters, fixes punctuation spacing.
- `src/dataset_builder.py` converts a manifest into a HuggingFace dataset.
- Validated end-to-end on EP1–EP23: **1,336 chunks · 6.92 hours**.

**Training — LoRA wired in, not yet run.**
- `modal_app.py::train()` fine-tunes via PEFT/LoRA (`config/training_config.yaml`
  → `lora:` section), saves both an adapter-only checkpoint and a merged
  standard HF checkpoint.
- Not yet validated against a real GPU run — see `sessions/session2.md` for
  the planned smoke test.

**Not yet built:**
- `evaluate()` (WER computation on a held-out set)
- `src/transcribe_batch.py` (batch inference: faster-whisper + WhisperX)
- `scripts/upload_data.py`, `scripts/download_model.py`

**Data:** EP1–EP23 processed (6.92 hours). POC target is 5–10 hours of
verified training audio — currently above the lower bound.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

The local venv only needs the lean subset for data prep (`datasets`,
`soundfile`, `librosa`). The heavy ML stack (`torch`, `transformers`, `peft`,
`whisperx`, `faster-whisper`) is intentionally left to Modal's remote
container — see the `image` definition in `modal_app.py`.

## Project layout

```
.
├── CLAUDE.MD                  ← Full design decisions, conventions, Modal reference
├── README.md                  ← This file
├── requirements.txt
├── modal_app.py                ← Modal entry point: train() implemented, evaluate()/transcribe_batch() pending
├── config/
│   └── training_config.yaml   ← Model, LoRA, training, and data hyperparameters
├── src/
│   ├── data_prep.py            ← Original video+flat-transcript chunking path
│   ├── srt_audio_prep.py       ← Audio+SRT → manifest (single pair)
│   ├── batch_srt_prep.py       ← Audio+SRT → manifest (batch, matched by YouTube ID)
│   ├── normalize_manifest.py   ← Transcript normalization (Unicode, punctuation)
│   └── dataset_builder.py      ← manifest.json → HuggingFace dataset
├── data/processed/             ← Local data prep output (gitignored)
├── samples/                    ← Raw sample audio+SRT inputs (gitignored)
└── sessions/                   ← Running session notes/log (session1.md, session2.md, ...)
```

## Typical workflow

### Step 1 — Chunk audio + SRT into training samples

**Single episode:**
```bash
python src/srt_audio_prep.py \
  --audio samples/28JulyBatch/EP1_hBK8bkFgus8.mp3 \
  --srt   samples/28JulyBatch/EP1_hBK8bkFgus8.srt \
  --output_dir ./data/processed/my_batch
```

**Full folder (recommended — merges incrementally, safe to re-run):**
```bash
python src/batch_srt_prep.py \
  --input_dir  ./samples/28JulyBatch \
  --output_dir ./data/processed/my_batch
```

Outputs: `data/processed/my_batch/manifest.json` + `audio/<episode_id>/*.wav`

Pairing is done by the 11-character YouTube ID embedded in both filenames
(e.g. `EP1_hBK8bkFgus8.mp3` ↔ `EP1_hBK8bkFgus8.srt`). YouTube IDs that
contain underscores (e.g. `EP18_o58PGx_xiIk`) are handled correctly.

---

### Step 2 — Normalize transcripts

Cleans every transcript in the manifest in-place. Run once after chunking,
before building the HuggingFace dataset.

```bash
# Preview changes without writing (recommended first pass)
python src/normalize_manifest.py \
  --manifest data/processed/my_batch/manifest.json \
  --dry-run

# Write to manifest_normalized.json (original untouched — default)
python src/normalize_manifest.py \
  --manifest data/processed/my_batch/manifest.json

# Overwrite manifest.json in-place
python src/normalize_manifest.py \
  --manifest data/processed/my_batch/manifest.json \
  --inplace
```

**What the normalizer fixes:**

| Rule | Example |
|------|---------|
| `ﷺ` (U+FDFA) → `صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ` | `حضورﷺ` → `حضور صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ` |
| `ؐ` (U+0610) → same salawat expansion | combining form of ﷺ |
| `ؑ` (U+0611) → `عَلَیْہِ السَّلَام` | after Imam names |
| `ؓ` (U+0613) → `رَضِیَ اللَّهُ عَنْہُ` | after companion names |
| U+200C ZWNJ → removed | invisible zero-width character |
| Space before `۔` removed | `کہا ۔` → `کہا۔` |
| Space added after `،` when missing | `کہا،اور` → `کہا، اور` |

All Arabic/Urdu diacritics (harakat) are preserved — including disambiguating
marks such as `اِس` (zer = "this") vs `اُس` (pesh = "that").

---

### Step 3 — Build HuggingFace dataset

```bash
python src/dataset_builder.py
```

---

See `CLAUDE.MD` for the Modal upload/train/evaluate/download commands —
those are documented but not all implemented yet (see Status above).

## Notes

- `sessions/*.md` is a running log of decisions and context per work session
  — read the latest one first if picking this up after a break.
- `CLAUDE.MD` is the source of truth for conventions (the Nastaliq/Arabic/
  English transcription boundary, LoRA config, timestamp strategy, etc.) —
  this README intentionally doesn't duplicate it.
