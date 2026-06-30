# Session 7 — 2026-06-30

## Goal / Context
Convert the reviewed manifest exported from the review portal (`all_batches_reviewed_manifest (1).json`) back to the standard `manifest.json` format used by the training pipeline.

## Key Decisions
- Auto-detect matching local batch folder using Jaccard similarity on `(episode_label, youtube_video_id, chunk_index)` tuples — avoids hard-coding mappings.
- Jaccard (overlap / union) correctly penalises superset matches (e.g. `Batch1_EP10` has 618 entries vs reviewed's 535, so Batch1_EP10 scores 0.87 while `batch_28july` scores 1.00).
- Output filename is `manifest_reviewed.json` by default to avoid overwriting the existing `manifest.json` until the user is satisfied.

## Open Items / TODOs
- User may want to rename/replace `manifest.json` with `manifest_reviewed.json` once validated.
- If future reviewed manifests span multiple batch folders (different `batch_name` values), the script will need to split output per-batch.

## Summary
Created `scripts/convert_reviewed_manifest.py`. The script reads the reviewed portal manifest, auto-detects `batch_28july` as the correct local batch folder (Jaccard=1.00, 535/535 entries matched), and writes `data/processed/batch_28july/manifest_reviewed.json` with the reviewed transcripts and original audio paths.

## Next Steps
- Validate a sample of `manifest_reviewed.json` entries against audio files on disk.
- Decide whether to overwrite `manifest.json` or use `manifest_reviewed.json` as the training input.
