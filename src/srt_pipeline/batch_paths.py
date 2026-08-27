"""
Where a batch's files live — defined ONCE, imported by every pipeline stage.

Before this existed, each stage carried its own `data/batch3/...` defaults. Processing
a new batch meant passing ~10 path flags across 4 commands, and forgetting any one of
them silently wrote the new batch's output into the previous batch's directories: the
paths were valid, just wrong, so nothing errored and a finished batch got corrupted.

Now a stage takes `--batch batch4` and derives everything from it. Explicit path flags
still win when given, so one-off layouts remain possible.

Layout for batch "batch3":
    data/batch3/batch3_validated.csv     stage 0  validate_sheet.py
    data/batch3/audio_raw/               stage 1  download_batch.py
    data/batch3/audio_trimmed/           stage 2  batch_clean_intro_music.py
    data/batch3/timestamped_srts/        stage 3+4  modal_align.py
    data/batch3/vad_spans/               stage 3+4  detected speech turns (vad windows only)
    data/batch3/qa_reports/              stage qa   one timestamped report per gate run
    data/batch3/logs/                    driver     one timestamped log per run (all stages)
    data/batch3/asr_transcripts/         stage 3+4  pre-alignment MODEL output (unreviewed;
                                                    NOT round-1's human-verified
                                                    raw_transcripts/ -- see transcript_dir)
    data/batch3/hadees_snips/            downstream  scripts/hadees_snips.py -> frames + manifest
    data/processed/Batch3/               stage 5  batch_srt_prep.py -> manifest.json
"""
import re
from pathlib import Path

DATA_ROOT = Path("data")
PROCESSED_ROOT = DATA_ROOT / "processed"

# srt_audio_prep.make_video_id() only reuses a filename's own label when it is a single
# token matching this; otherwise chunk IDs fall back to directory-iteration order and
# shift between runs. Any derived prefix must satisfy it.
LABEL_RE = re.compile(r"^[A-Za-z]+\d+$")


class BatchPaths:
    """Resolved locations for one batch. Construct with the batch name, e.g. "batch3"."""

    def __init__(self, batch: str, data_root: Path | str = DATA_ROOT):
        self.batch = batch
        self.root = Path(data_root) / batch

    @property
    def validated_csv(self) -> Path:
        return self.root / f"{self.batch}_validated.csv"

    @property
    def audio_raw(self) -> Path:
        return self.root / "audio_raw"

    @property
    def audio_trimmed(self) -> Path:
        return self.root / "audio_trimmed"

    @property
    def srt_dir(self) -> Path:
        return self.root / "timestamped_srts"

    @property
    def transcript_dir(self) -> Path:
        """Stage 3+4's pre-alignment MODEL output.

        Deliberately NOT called raw_transcripts/: in round 1 that name meant the
        HUMAN-VERIFIED text that was the ground-truth *input* to forced alignment. Here the
        role is inverted -- this is machine output, unreviewed. Reusing the old name invites
        two silent mistakes: passing it to
        `batch_clean_intro_music.py --require-transcript` (which would treat unreviewed text
        as "ready for alignment"), or to `modal_align.py::main --text-path` (round 1's
        entrypoint, which force-aligns text it believes a human checked). Both would run
        without error and quietly produce wrong results.
        """
        return self.root / "asr_transcripts"

    @property
    def vad_dir(self) -> Path:
        """Stage 3+4's detected speech turns, one <stem>.vad.json per episode.

        Written only by the "vad" window path, where the VAD scores are already computed to
        cut the windows -- binarizing them into turns is a CPU post-process on a tensor that
        exists anyway, so this costs no extra GPU.

        The QA gate needs it to tell DROPPED SPEECH from a pause the speaker actually took.
        Without it the gate charges every uncovered second as loss, which measured ~2x too
        harsh on both B3001 and B3002 (see the gap-share check in qa_gate.py).

        Kept out of srt_dir on purpose: several stages glob that directory, and a sidecar
        file there is one careless `*` away from being treated as an episode.
        """
        return self.root / "vad_spans"

    @property
    def logs_dir(self) -> Path:
        """One log file per driver run, capturing the driver AND every child stage.

        Stage output used to exist only in terminal scrollback. Every defect found so far was
        diagnosed from a line in that output -- the cp1252 crash, the 403, "no supported
        JavaScript runtime", "68 segments for 67 windows", "17 windows received more than
        one". A 93-video run cannot be watched live, and the run that eventually goes wrong
        is the one nobody was looking at.
        """
        return self.root / "logs"

    @property
    def qa_reports_dir(self) -> Path:
        """One timestamped JSON per QA gate run.

        Unlike every other artifact here these are NOT regenerable: they describe the output
        of a state of the code that no longer exists once it changes. The gate on its own
        only answers "does this pass now" -- and since it re-scores every finished episode on
        each run, a change that quietly worsens an already-passing episode has nothing to be
        noticed against. That is the exact failure mode this pipeline keeps producing:
        output that is wrong while every check still reports pass.
        """
        return self.root / "qa_reports"

    @property
    def hadees_snips_dir(self) -> Path:
        """Downstream output of scripts/hadees_snips.py: still frames of quoted
        Quran/hadees plus manifest.csv and review.html.

        Batch-scoped like everything else here for one specific reason: that script
        APPENDS to manifest.csv rather than rewriting it, so a stale default would not
        just misplace files -- it would interleave two batches' rows in one manifest
        with no error and no way to tell them apart afterwards.
        """
        return self.root / "hadees_snips"

    @property
    def processed_dir(self) -> Path:
        """Stage 5 output dir, matching round 1's data/processed/Batch<N> convention."""
        return PROCESSED_ROOT / (self.batch[:1].upper() + self.batch[1:])

    def __repr__(self) -> str:
        return f"BatchPaths({self.batch!r} -> {self.root})"


def label_prefix_for(batch: str) -> str:
    """"batch3" -> "B3". First letter + the batch's digits, so the result satisfies
    LABEL_RE and stays short in filenames (B3001_<youtube_id>.mp3).

    Raises if no digits are present -- better than inventing a prefix that could collide
    with another batch's labels.
    """
    digits = "".join(re.findall(r"\d+", batch))
    if not digits or not batch[:1].isalpha():
        raise ValueError(
            f"cannot derive a label prefix from batch name {batch!r} "
            "(need a leading letter and at least one digit, e.g. 'batch4'); "
            "pass --label-prefix explicitly instead"
        )
    prefix = f"{batch[:1].upper()}{digits}"
    if not LABEL_RE.match(prefix):
        raise ValueError(f"derived label prefix {prefix!r} does not match {LABEL_RE.pattern}")
    return prefix


def add_batch_argument(parser, default: str = "batch3") -> None:
    """Attach the shared --batch flag to an argparse parser."""
    parser.add_argument(
        "--batch", default=default,
        help=f"Batch name driving all default paths under data/<batch>/ (default: {default}). "
             "Explicit path flags override the derived defaults.",
    )
