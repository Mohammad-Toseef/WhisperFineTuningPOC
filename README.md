# Whisper Medium Fine-Tuning POC

Fine-tuning OpenAI Whisper Medium (via LoRA) on a single Urdu/English/Arabic
code-switching speaker, for accurate batch transcription (text + timestamps)
of 4,000+ videos.

Full architecture, design decisions, and rationale live in [CLAUDE.MD](CLAUDE.MD).
This file is a practical entry point — what exists, what works, and how to run it.

## Status

**Data prep pipeline — working, validated on one sample.**
- `src/srt_audio_prep.py` converts an audio + SRT pair into ≤28s training
  chunks (gap-aware, silence-snapped, cue-bounded) and a `manifest.json`.
- `src/batch_srt_prep.py` runs that across a folder of multiple audio+SRT
  pairs (matched by shared YouTube ID) into one combined manifest.
- `src/dataset_builder.py` converts a manifest into a HuggingFace dataset.
- Validated end-to-end on one real sample (manual audio+transcript review,
  see `sessions/session1.md` for the bugs found and fixed along the way).

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

**Data:** one validated sample (~4 min). 50 plain-text transcripts (no
timestamps) have been received from the team — still need forced alignment
(WhisperX) to produce SRTs before they can feed the data prep pipeline above.
POC target is 5–10 hours of verified training audio.

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
│   └── dataset_builder.py      ← manifest.json → HuggingFace dataset
├── data/processed/             ← Local data prep output (gitignored)
├── samples/                    ← Raw sample audio+SRT inputs (gitignored)
└── sessions/                   ← Running session notes/log (session1.md, session2.md, ...)
```

## Typical workflow so far

```bash
# Single audio+SRT pair
python src/srt_audio_prep.py --audio <path> --srt <path> --output_dir ./data/processed/<name>

# Batch of audio+SRT pairs sharing YouTube IDs in their filenames
python src/batch_srt_prep.py --input_dir <folder> --output_dir ./data/processed/<name>

# Build HF dataset from the resulting manifest
python src/dataset_builder.py
```

See `CLAUDE.MD` for the Modal upload/train/evaluate/download commands —
those are documented but not all implemented yet (see Status above).

## Notes

- `sessions/*.md` is a running log of decisions and context per work session
  — read the latest one first if picking this up after a break.
- `CLAUDE.MD` is the source of truth for conventions (the Nastaliq/Arabic/
  English transcription boundary, LoRA config, timestamp strategy, etc.) —
  this README intentionally doesn't duplicate it.
