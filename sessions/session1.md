# Session 1

## Context loaded
- Project: fine-tune Whisper medium on a single Urdu/English/Arabic code-switching
  speaker for batch transcription (text + timestamps) of 4,000+ videos.
- CLAUDE.MD was updated mid-session with major architecture decisions:
  - **LoRA (Path B)** fine-tuning via PEFT — freezes base Whisper, trains adapters only
  - **Three-script transcription convention**: Nastaliq (Urdu, LoRA-learned), Arabic
    script (Quranic/recitation, LoRA-learned), English stays English script
    (handled by frozen base, zero training needed)
  - Production pipeline: fine-tuned Whisper for text + WhisperX (wav2vec2) for
    timestamps + faster-whisper (CTranslate2 INT8) for batch inference
  - Unknown-word strategy: LoRA + `initial_prompt` biasing + corrections dictionary

## Gaps identified vs. current code (not yet done)
- `modal_app.py::train()` still does full fine-tuning, not LoRA (no `LoraConfig`/`get_peft_model`)
- No `evaluate()` function in `modal_app.py` yet
- `config/training_config.yaml` has no LoRA section
- `src/train.py`, `src/evaluate.py`, `src/transcribe_batch.py`, `scripts/` referenced
  in CLAUDE.MD's file structure don't exist yet — logic still inline in `modal_app.py`

## Data format clarified
- Audio: 16kHz mono WAV, ≤28s chunks (30s hard Whisper limit)
- Transcripts must be **human-verified ground truth** — never use Whisper's own
  output as training labels (would just reinforce existing ~35-45% WER mistakes)
- Original `data_prep.py` assumed video + single flat `.txt` transcript per file,
  with chunking-but-no-transcript-splitting as an open TODO for long files

## Sample data delivered
- Team provided one real sample: MP3 audio (44.1kHz stereo, ~270s) + matching
  `.srt` (timestamped, verified Urdu transcript)
- Built `src/srt_audio_prep.py`: parses SRT cues, greedily groups them into
  ≤28s windows using their own timestamps, slices matching audio per window,
  converts to 16kHz mono WAV, emits manifest.json compatible with `dataset_builder.py`
- Verified on the sample: 10 chunks, ~0.069 hours total
- Confirmed end-to-end: SRT → manifest → HF dataset → decoded audio array,
  correct format (`sentence`, `language: ur`, 16kHz audio array)
- Format (audio+SRT) not yet confirmed as the standard for all 4,000+ videos —
  team check is in progress; treated as a one-off test for now

## Environment / fixes made
- Created local `venv` (decided against `uv`, which wasn't installed)
- Created `requirements.txt` (full ML stack, mirrors CLAUDE.MD's dependency list)
- Local venv only has the lean subset needed for data prep: `datasets`, `soundfile`,
  `librosa` — heavy ML stack (torch, transformers, peft, whisperx, faster-whisper)
  intentionally stays in Modal's remote container only
- Fixed Windows console encoding crash (cp1252 can't print Urdu text or ✓/✅/⚠
  symbols) via `sys.stdout.reconfigure(encoding="utf-8")` in `data_prep.py`,
  `dataset_builder.py`, `srt_audio_prep.py`
- Found and fixed a real risk: `datasets>=2.18.0` with no upper bound resolves to
  v5.x, which requires `torchcodec` (and thus `torch`) just to decode audio arrays —
  breaks local lean-install design and could break Modal's container build too.
  Pinned `datasets>=2.18.0,<3.0.0` in both `requirements.txt` and `modal_app.py`'s
  image definition. (2.x still needs `librosa` installed alongside `soundfile`.)

## Sync bugs found and fixed in srt_audio_prep.py (post-build manual review)
- **Bug 1**: SRT cue timestamps don't track real audio precisely (downloaded/
  auto-aligned captions) — cutting at the raw timestamp sliced mid-word
  (confirmed via waveform RMS energy: at the nominal cut, energy was rising,
  i.e. mid-syllable, not silence). Fixed with `find_silence_boundary()` —
  snaps each chunk boundary to the nearest true low-energy point in a search
  window, falling back to the nominal timestamp if no clear silence exists.
- **Side effect caught**: snapping pushed some chunks past the 28s target
  (one hit 29.6s, too close to Whisper's 30s hard limit). Fixed by reserving
  headroom in the pre-snap grouping step + a hard per-chunk duration clamp.
- **Bug 2** (found via manual review of the manifest after Bug 1's fix):
  text assignment was still based on the *original* SRT grouping while audio
  was sliced at the *snapped* boundary — a large snap (+2.5s) could sweep an
  entire short cue's audio into the previous chunk while its text stayed
  assigned to the next chunk's transcript (e.g. "کا مطلب یہ ہے" audio ended
  up at the tail of 004.wav, but the text was still the start of 005's
  transcript). Fixed by re-deriving each chunk's text from the final snapped
  boundaries (re-walking cues against `boundaries[]`) instead of the
  pre-snap grouping.
- Takeaway: any further changes to chunk-boundary logic must keep text
  assignment and audio slicing driven by the same final boundary values —
  computing them independently is what caused both Bug 1 and Bug 2.
- **Bug 3** (found by listening — 002.wav had 7s of music at the start):
  confirmed via SRT gap analysis (a real 7.65s gap between cue 19's end at
  38.31s and cue 20's start at 45.96s) and waveform RMS energy (continuous
  low-level non-silence in that gap, i.e. music, not a pause — natural
  speech pauses elsewhere in this file are only ~1.0-1.7s). The chunking
  logic assumed adjacent cues are audio-contiguous, so it silently included
  the music-filled gap inside a chunk. Fixed: `group_cues()` now forces a
  hard chunk break on any inter-cue gap > `GAP_THRESHOLD` (3.0s), and
  `prepare_from_srt()` trims each side of that break independently and
  *drops* the gap audio entirely, rather than assigning it to either chunk.
  Caught a second real gap (3.32s) the same way on re-run.
- **Bug 4** (found by manual review — "مختلف راویوں نے، مفسرین نے..." text
  was in chunk003 but its audio was split between chunk002 and chunk003):
  root cause was a single 8.66s SRT cue (cue 25, 66.54s-75.20s) containing
  an internal comma pause. The silence search found that internal pause
  and used it as a chunk boundary -- but since the SRT only gives
  whole-cue-granularity text (no word-level timing), the cue's full text
  could only go to one chunk while its audio physically split across both.
  This also explained two earlier "suspicious" +2.50s snaps that had hit
  the search window's edge exactly. Fixed by bounding every normal-boundary
  silence search strictly to `[prev_cue.end, next_cue.start]` -- the real
  gap between two adjacent cues -- so a search can never land inside any
  single cue's own span. Re-running dropped all snap shifts to a believable
  0.24-0.86s (previously some were pegged at the old fixed 2.5s search
  limit). This is a structural limit of caption-level timing without word
  alignment -- CLAUDE.MD's planned WhisperX forced-alignment step for
  production is the real long-term fix; this heuristic is only meant to
  keep local data-prep chunks reasonably clean.

## Validation
User manually listened through all chunks + read all transcripts after
Bug 1-4 fixes — confirmed audio and transcriptions all look good. The
`srt_audio_prep.py` pipeline (SRT -> gap-aware, silence-snapped, cue-bounded
chunking -> manifest -> HF dataset) is considered solid for this sample.

## Naming convention for audio chunks + manifest
Original chunk filenames inherited the full downloaded video title
(80+ chars, e.g. "YTDown_YouTube_La-Mehdi-Illa-Isa-Ki-Haqeeqat-EP2-..._000.wav")
-- too long and not scalable across 4,000+ videos. Adopted:
- Video ID = `vid{NNNN}_{youtube_id}` (e.g. `vid0001_q1Q6B2JrY58`) --
  sequential index (always present, controlled via `--video_index`) +
  the 11-char YouTube ID auto-extracted from the source filename when
  present (traceable back to source without a separate lookup file).
- Layout: per-video subfolder, `audio/{video_id}/{video_id}_{chunk_idx:03d}.wav`
  -- avoids dumping tens of thousands of files into one flat folder at scale.
- Implemented in `srt_audio_prep.py` via `make_video_id()`; regenerated the
  sample under this convention, content unchanged, just renamed/reorganized.

## Batch processing for the next 50 files
User is bringing 50 more audio+srt pairs, each pair sharing a YouTube ID
embedded in both filenames (confirmed by user) -- not a shared filename stem.
Built `src/batch_srt_prep.py`:
- `find_pairs()` matches audio files to `.srt` files in a folder by the
  YouTube ID extracted from each filename (reuses
  `srt_audio_prep.find_youtube_id`, factored out of `make_video_id()`).
- Calls `prepare_from_srt()` per pair with an auto-incrementing
  `video_index`, collects all returned `Sample` lists, and writes ONE
  combined `manifest.json` -- `prepare_from_srt()` was refactored to return
  `samples` and take a `write_manifest=False` flag so per-video calls don't
  each overwrite a throwaway manifest.
- Per-video failures are caught and skipped (logged) rather than aborting
  the whole batch.
- Tested against a copy of the existing sample renamed to share a YouTube ID
  across audio+srt (the real sample's filenames don't follow this pattern,
  only the upcoming 50 will) -- pairing, per-video processing, and the
  combined manifest all verified working.

## Open items / next steps
- Waiting on team to confirm audio+SRT is the standard format for all videos
- Need 5-10+ hours of verified data for POC target (have 0.069 hrs from 1 sample)
- Once LoRA approach is confirmed, update `modal_app.py::train()` and
  `training_config.yaml` to match Path B (LoRA config, PEFT wrapping)
