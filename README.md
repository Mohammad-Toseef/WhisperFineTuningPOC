# Whisper Urdu Fine-Tuning

Fine-tuning OpenAI Whisper **large-v3** (via LoRA) on a single Urdu/English/Arabic
code-switching speaker, for accurate batch transcription (text + timestamps)
of 4,000+ videos.

Full architecture, design decisions, and rationale live in [CLAUDE.MD](CLAUDE.MD).
This file is a practical entry point — what exists, what works, and how to run it.

## Status

**Round 1 training — complete.** Normalized WER **18.57% → 10.50%** (+8.07 pts) on a
246-clip held-out set, beating the <13% target. Per-bucket: `nastaliq_only`
18.05→9.89, `code_switch` 20.46→12.70, `spiritual_term` 20.29→**9.38** (worst→best,
validating the Tier-2 MLP LoRA targets). Trained on 49 episodes / 2,257 clips /
11.67 hours. Full write-up in [FULL_WHISPER_TRAINING_RUN.md](FULL_WHISPER_TRAINING_RUN.md).

- Model: `mohammad-toseef059/whisper-large-v3-urdu` on HF Hub (**private**)
- Also on the Modal volume `whisper-training-vol` at `/data/model/whisper-urdu-final`
- `modal_app.py` implements `train()`, `evaluate()`, and `transcribe_batch()`

**Round 2 data pipeline — built, verified end-to-end on 1 episode.** An automated
pipeline turns a spreadsheet of YouTube links into review-ready training data. See
[Batch pipeline](#batch-pipeline) below. Batch 3 is scoped at **96 videos / 43.33 hours**.

**Not yet built:**
- Phase 6 production inference: convert the HF model → faster-whisper (ct2 int8) +
  WhisperX, then batch-transcribe the 4,000 videos.

---

## Batch pipeline

Takes a spreadsheet of YouTube links to **review-ready training data**, with no manual
steps between. Built for round 2 (Batch 3) and reusable for any later batch.

### Quick start

```bash
# what's done, what's pending
python run_batch_pipeline.py --batch batch3 status

# process a range of videos, all stages
python run_batch_pipeline.py --batch batch3 run --from B3003 --to B3010

# or just take the next N pending
python run_batch_pipeline.py --batch batch3 run --next 8

# see the commands without running them
python run_batch_pipeline.py --batch batch3 run --next 8 --dry-run
```

**Everything is idempotent.** Completed work is skipped, so an interrupted run is
resumed by re-running the identical command. Nothing is lost to a network drop.

### The stages

```
Whisper_Second_Round_Training_list.xlsx  (sheet 'Batch 3')
  │
  0. validate_sheet.py ──────────► data/batch3/batch3_validated.csv
  1. download_batch.py ──────────► data/batch3/audio_raw/B3001_<videoid>.mp3
  2. batch_clean_intro_music.py ─► data/batch3/audio_trimmed/
  3+4. modal_align.py [GPU] ─────► data/batch3/timestamped_srts/*.srt
  │                                data/batch3/asr_transcripts/*.txt
  5. batch_srt_prep.py ──────────► data/processed/Batch3/{audio/,manifest.json}
  6. normalize_manifest.py ──────► data/processed/Batch3/manifest_normalized.json
  │
  7. team review (external portal)
  8. convert_reviewed_manifest.py ► manifest_reviewed.json ──► dataset_builder.py ──► training
```

Stages 7–8 involve people, so they sit outside the driver.

| Stage | Script | What it does |
|---|---|---|
| 0 | `src/srt_pipeline/validate_sheet.py` | Parses the sheet, **repairs Excel damage**, dedupes, assigns stable labels. Writes an auditable CSV that every later stage reads instead of the `.xlsx`. |
| 1 | `src/srt_pipeline/download_batch.py` | `yt-dlp` per video. Cross-checks each file's real duration against the sheet. |
| 2 | `src/srt_pipeline/batch_clean_intro_music.py` | ffmpeg-trims intro/outro music using `skip_start_seconds` + `speech_end_seconds`. |
| 3+4 | `modal_align.py::transcribe_align` | **GPU.** Fine-tuned Whisper transcribes **once**, then wav2vec2 CTC forced-alignment re-times the words. |
| 5 | `src/batch_srt_prep.py` | Cuts ≤28s silence-snapped chunks, merges into one `manifest.json`. |
| 6 | `src/normalize_manifest.py` | Expands honorific ligatures, strips ZWNJ, fixes punctuation spacing. |

### Selection options for `run`

| Flag | Effect |
|---|---|
| `--from B3003 --to B3010` | that range (bare numbers work too: `--from 3 --to 10`) |
| `--next 8` | the next 8 **incomplete** videos (finishes half-done work first) |
| `--only B3007,B3009` | specific videos |
| *(none)* | everything outstanding |
| `--stages 5,6` | only those stages — e.g. re-chunk without re-downloading |
| `--dry-run` | print commands, execute nothing |
| `--cookies-from-browser chrome` | passed to stage 1 when YouTube demands sign-in |

### Prerequisites

- **ffmpeg + ffprobe on `PATH`** — stages 2 and 5 shell out to them
- **`pip install -r requirements.txt`** — includes `yt-dlp`, `pandas`, `openpyxl`
- **Modal authenticated** (`modal token new`) with access to the `whisper-training-vol`
  volume, and the fine-tuned model present at `/data/model/whisper-urdu-final`.
  Check with `modal volume ls whisper-training-vol model`.

### The spreadsheet contract

Stage 0 reads one sheet and needs these columns:

| Column | Meaning |
|---|---|
| `Video Link` | Any YouTube URL form (`youtu.be/<id>`, `watch?v=<id>&list=...`) |
| `Video Title` | Free text, only used for logging |
| `skip_start_seconds` | Where real speech begins |
| `speech_end_timestamp` | Where real speech ends |
| `Duration` | Advisory only — stage 2 uses `ffprobe`, which is authoritative |

**Excel will corrupt your time cells, and stage 0 repairs it.** These are real defects
found in the Batch 3 sheet, each logged per-row in the CSV's `repairs` column:

| Problem | Frequency | How it's read |
|---|---|---|
| `skip_start` arrives as `datetime.time` | **80 / 99 rows** | Excel parsed typed `"0:09"` as **H:MM**; the human meant 9 **seconds**. Recovered as `hour*60 + minute`. |
| `MM.SS` decimal notation | 8 rows | `"19.17"` means 19m17s, not 19.17s |
| `0.47` in a numeric skip cell | 8 rows | Also `MM.SS` → **47 seconds**, not 0.47 |
| `;` typed for `:` | 1 row | `"15;16"` → `15:16` |
| Duplicate video IDs | 3 rows | First occurrence wins, rest logged and dropped |

Stage 0 **refuses to write** the CSV if anything is unrecoverable, rather than guessing.
It also fails loudly on a wrong sheet name (listing the available sheets) or missing columns.

### Naming: why labels matter

Stage 0 assigns each video a stable label — `B3001`, `B3002`, … — and stage 1 names files
`<label>_<youtube_id>.mp3`.

This is not cosmetic. `srt_audio_prep.make_video_id()` reuses a filename's own label only
when it's a single token matching `^[A-Za-z]+\d+$`; otherwise it falls back to
`vid{NNNN}` derived from **directory iteration order**. Only 30 of 99 Batch 3 titles carry
an episode number, so without assigned labels most videos would land on that fallback — and
a re-run could **renumber chunks and break the manifest↔audio correspondence**.

Audio and SRTs are paired by the **11-character YouTube ID**, not by filename stem, so
IDs containing underscores (`EP18_o58PGx_xiIk`) work fine.

### Two transcript directories — do not confuse them

| Directory | Contents |
|---|---|
| `raw_transcripts/` | **Round 1 only.** Human-verified text, the ground-truth *input* to forced alignment. |
| `data/<batch>/asr_transcripts/` | **Batch pipeline.** Unreviewed *model output*, written before alignment. |

The roles are inverted. Passing `asr_transcripts/` to `--require-transcript` or to
`modal_align.py::main --text-path` would treat machine output as human-verified — and both
would run without error while producing quietly wrong results.

`asr_transcripts/` is not consumed by any stage. It's kept because it's the only reference
against which alignment word-loss is measurable: an early bug silently dropped 5% of the
words while every structural check on the SRT still passed.

### Starting a new batch

Everything derives from `--batch`, defined once in
[`src/srt_pipeline/batch_paths.py`](src/srt_pipeline/batch_paths.py):

1. Add a sheet named `Batch 4` to the spreadsheet, with the columns above.
2. Run it:

```bash
python run_batch_pipeline.py --batch batch4 run --next 10
```

`batch4` derives `data/batch4/{audio_raw,audio_trimmed,timestamped_srts,asr_transcripts}`,
`data/batch4/batch4_validated.csv`, `data/processed/Batch4`, sheet name `"Batch 4"`, and
label prefix `B4`. To use a newer model for a later batch:

```bash
python run_batch_pipeline.py --batch batch4 run --next 10 \
    --model-path /data/model/whisper-urdu-round2
```

Batch names need a leading letter and at least one digit — `label_prefix_for` refuses
anything else rather than inventing a prefix that could collide with another batch.

### Troubleshooting

**`Sign in to confirm you're not a bot`** — YouTube rate-limiting. Often clears on its own
after a few minutes; otherwise pass `--cookies-from-browser chrome`. On Windows that fails
with `Could not copy Chrome cookie database` while the browser is running
([yt-dlp #7271](https://github.com/yt-dlp/yt-dlp/issues/7271)) — fully quit it, or export a
`cookies.txt` and pass `--cookies`.

**Modal prints an async-generator traceback at the end** — cosmetic client teardown noise on
Windows, after the work has completed. Check the exit code, not the traceback.

**A stage failed mid-run** — just re-run the same command. Every stage skips completed work.

**Timing looks wrong in an SRT** — check words-per-second per cue against the episode
median (~2.2 w/s for this speaker). Structural checks *cannot* catch bad timing: a broken
SRT can still have contiguous indices, monotonic timestamps, and exact end-of-file
coverage. Rate is the signal. Ignore cues under ~3 words — a 1-word cue in a 40ms sliver is
harmless and will show an absurd rate.

**Verifying a manifest** — the useful invariants are: transcript words conserved from SRT
to manifest, zero missing audio files, zero empty transcripts, no chunk over the 28s cap,
and no `.wav` on disk unreferenced by the manifest.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

The local venv only needs the lean subset for data prep (`datasets`, `soundfile`,
`librosa`, `yt-dlp`, `pandas`). The heavy ML stack (`torch`, `transformers`, `peft`,
`whisperx`, `faster-whisper`) is intentionally left to Modal's remote container — see the
`image` definitions in `modal_app.py` and `modal_align.py`.

> Pinned upper bounds in those images are load-bearing. Unbounded `>=` pulled
> transformers 5.x / torch 2.12 / numpy 2.4 and broke `Seq2SeqTrainer`.

## Project layout

```
.
├── CLAUDE.MD                       ← Full design decisions, conventions, Modal reference
├── README.md                       ← This file
├── FULL_WHISPER_TRAINING_RUN.md    ← Round-1 training run write-up + results
├── MODEL_CARD.md                   ← Pushed to the HF repo as its README
├── run_batch_pipeline.py           ← ★ Batch pipeline driver (status / run)
├── modal_app.py                    ← Modal: train(), evaluate(), transcribe_batch()
├── modal_align.py                  ← Modal: transcribe_align() [stages 3+4], run_alignment()
├── config/
│   ├── training_config.yaml        ← Model, LoRA, training, data hyperparameters
│   ├── domain_terms.json           ← Spiritual/Arabic vocabulary for eval bucketing
│   ├── initial_prompt.txt           ← Decoder-biasing prompt (not yet wired into inference)
│   └── episode_skip_start_v2.xlsx  ← Round-1 hand-verified trim points
├── src/
│   ├── srt_pipeline/               ← ★ Batch pipeline stages 0–2 + alignment core
│   │   ├── batch_paths.py          ← Batch layout, defined once (--batch derives all paths)
│   │   ├── validate_sheet.py       ← Stage 0: xlsx → validated CSV (repairs Excel damage)
│   │   ├── download_batch.py       ← Stage 1: yt-dlp per video
│   │   ├── clean_intro_music.py    ← ffmpeg trim primitives
│   │   ├── batch_clean_intro_music.py ← Stage 2: batch trim (--csv or legacy --xlsx)
│   │   └── align_to_srt.py         ← Forced-alignment core + segment-window repair
│   ├── srt_audio_prep.py           ← Audio+SRT → chunks + manifest (single pair)
│   ├── batch_srt_prep.py           ← Stage 5: batch chunking, merges one manifest
│   ├── normalize_manifest.py       ← Stage 6: transcript normalization
│   ├── dataset_builder.py          ← manifest.json → HuggingFace dataset
│   └── data_prep.py                ← Original video+flat-transcript path (round 0)
├── scripts/
│   ├── convert_reviewed_manifest.py ← Stage 8: review-portal export → manifest.json
│   ├── compare_transcribe.py       ← Base vs fine-tuned qualitative comparison
│   ├── push_to_hub.py              ← Upload merged model to HF Hub
│   └── fix_english_in_nastaliq.py  ← Latin-in-Nastaliq repair (use with care: false positives)
├── data/                           ← Pipeline output (gitignored)
└── sessions/                       ← Session-by-session decision log
```

---

## Round-1 / manual workflow

Still valid for a folder of audio+SRT pairs that already exist (skipping stages 0–4).

### Chunk audio + SRT into training samples

```bash
# single episode
python src/srt_audio_prep.py \
  --audio samples/28JulyBatch/EP1_hBK8bkFgus8.mp3 \
  --srt   samples/28JulyBatch/EP1_hBK8bkFgus8.srt \
  --output_dir ./data/processed/my_batch

# a folder (merges incrementally, safe to re-run)
python src/batch_srt_prep.py \
  --input_dir  ./samples/28JulyBatch \
  --output_dir ./data/processed/my_batch

# audio and SRTs in separate directories
python src/batch_srt_prep.py \
  --input_dir ./data/batch3/audio_trimmed \
  --srt_dir   ./data/batch3/timestamped_srts \
  --output_dir ./data/processed/Batch3
```

Outputs `manifest.json` + `audio/<video_id>/*.wav` (16 kHz mono 16-bit).

### Normalize transcripts

```bash
# preview without writing (recommended first pass)
python src/normalize_manifest.py --manifest data/processed/my_batch/manifest.json --dry-run

# write to manifest_normalized.json (original untouched — default)
python src/normalize_manifest.py --manifest data/processed/my_batch/manifest.json

# overwrite in place
python src/normalize_manifest.py --manifest data/processed/my_batch/manifest.json --inplace
```

| Rule | Example |
|------|---------|
| `ﷺ` (U+FDFA) → `صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ` | `حضورﷺ` → `حضور صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ` |
| `ؐ` (U+0610) → same salawat expansion | combining form of ﷺ |
| `ؑ` (U+0611) → `عَلَیْہِ السَّلَام` | after Imam names |
| `ؓ` (U+0613) → `رَضِیَ اللَّهُ عَنْہُ` | after companion names |
| U+200C ZWNJ → removed | invisible zero-width character |
| Space before `۔` removed | `کہا ۔` → `کہا۔` |
| Space added after `،` when missing | `کہا،اور` → `کہا، اور` |

All Arabic/Urdu diacritics (harakat) are preserved — including disambiguating marks such as
`اِس` (zer = "this") vs `اُس` (pesh = "that").

### Convert a reviewed manifest from the review portal

After reviewers correct transcripts and export (a flat JSON list with extra fields like
`audio_s3_key`, `batch_name`, `episode_label`, `youtube_video_id`, `chunk_index`, `status`),
convert it back to the standard format. The script auto-detects the matching local batch
folder by comparing `(episode_label, youtube_video_id, chunk_index)` tuples via Jaccard
similarity.

```bash
python scripts/convert_reviewed_manifest.py path/to/reviewed_export.json

# force a specific batch folder — auto-detection favours the folder with the
# most overlap, which is wrong when one batch is a superset of another
python scripts/convert_reviewed_manifest.py path/to/export.json --batch-folder Batch3
```

Output keeps only `audio_path`, `transcript`, `duration`, `language`.

> Re-running the converter regenerates entries removed by hand. Round 1 had 3 faulty clips
> deleted from `manifest_reviewed.json` only — re-do that removal after any re-convert, or
> clean the source manifests instead.

### Build the HuggingFace dataset

```bash
python src/dataset_builder.py
```

---

## Modal notes

Learned the hard way during round 1 — see `sessions/` for full context.

- **Use `modal run --detach` for long runs.** A local DNS drop killed training at step 88;
  detached runs survive client disconnects.
- **Volume paths are root-relative.** The volume mounts *at* `/data`, so
  `modal volume put vol local /config` — **not** `/data/config`, which double-nests.
- **Use PowerShell for the Modal CLI on Windows.** Git Bash rewrites `/data/...` into
  `/C:/Program Files/Git/...`. Set `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` for glyphs.
- **`modal volume get <dir>` collapses to one file** unless the local destination directory
  already exists. `New-Item -ItemType Directory` first.
- **LoRA + Whisper needs `task_type: null`.** `SEQ_2_SEQ_LM` raises
  `WhisperForConditionalGeneration.forward() got an unexpected keyword argument 'input_ids'`.

## Notes

- `sessions/*.md` is a running log of decisions and context per work session — read the
  latest first if picking this up after a break.
- `CLAUDE.MD` is the source of truth for conventions (the Nastaliq/Arabic/English
  transcription boundary, LoRA config, timestamp strategy) — this README doesn't duplicate it.
