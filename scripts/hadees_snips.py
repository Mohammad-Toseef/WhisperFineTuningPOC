#!/usr/bin/env python3
"""Extract still-frame snips of NEW hadees from the episode videos.

Ported from SRTTimeStampPOC/hadees_snips.py, where it was hardcoded to one
directory layout and one skip-start spreadsheet. Here every input is either
derived from `--batch` (see src/srt_pipeline/batch_paths.py) or overridable, so
the same detector runs over any batch, any SRT variant, and any tuning.

Pipeline (full decision log: SRTTimeStampPOC/sessions/006-2026-07-12.md):

  1. Parse each SRT in the batch's timestamped_srts/. The narration is Urdu
     (Arabic script) with NO diacritics; quoted Quran/hadees carries full
     tashkeel. So a cue is a *candidate* iff it is DENSELY voweled. Consecutive
     candidate cues are merged into one block (a hadees often spans several).
  2. Classify each block by REUSING the translation-portal-ui matching logic:
       - skip if quran_lookup_service.find_quran_matches() finds a verse,
       - skip if sayings_lookup_service.find_duplicate() >= dup-threshold.
     What survives is treated as a NEW hadees iff it also passes the heuristic
     filter (tashkeel present AND >= --min-arabic-words Arabic words) -- the
     endpoints can only say "is Quran" / "already exists", never affirm "is a
     hadees", so this filter keeps out basmala / durood / short invocations.
  3. One snip per --snip-offsets entry, at floor(srt_start) + offset.
  4. IMPORTANT offset: the SRTs are aligned to the music-TRIMMED audio, but the
     YouTube video still has the intro music, so
         video_time = srt_time + skip_start_seconds[episode]
     (skip_start from the batch's *_validated.csv).
  5. Download only a small per-hadees window from YouTube with yt-dlp
     (--force-keyframes-at-cuts so file-time 0 == window start), then grab the
     frames with ffmpeg.
  6. Write <out-dir>/<LABEL>/*.jpg + <out-dir>/manifest.csv + review.html.

Usage:
    python scripts/hadees_snips.py --batch batch3 --no-download
    python scripts/hadees_snips.py --batch batch3 --only B3002,B3006
    python scripts/hadees_snips.py --batch batch3 --snip-offsets 1,4,7,10 --limit 20
    python scripts/hadees_snips.py --batch batch4 --srt-dir data/batch4/timestamped_srts_vad
"""
from __future__ import annotations

import argparse
import csv
import html
import math
import os
import re
import shutil
import string
import subprocess
import sys
import unicodedata
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

# The portal Quran matcher iterates string `set`s and caps its candidate pool,
# so its borderline results depend on PYTHONHASHSEED. Pin it for reproducibility
# by re-exec'ing once with a fixed seed before importing anything hash-sensitive.
# subprocess (not execv) because Windows execv mangles paths containing spaces;
# stdin is inherited, so the interactive per-hadees prompt still works.
if os.environ.get("PYTHONHASHSEED") != "0":
    import subprocess as _sp
    os.environ["PYTHONHASHSEED"] = "0"
    sys.exit(_sp.run([sys.executable, *sys.argv]).returncode)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "srt_pipeline"))  # -> batch_paths

from batch_paths import BatchPaths, add_batch_argument  # noqa: E402

# ── Defaults (every one is a CLI flag; see build_parser) ───────────────────────
DEFAULT_PORTAL_BACKEND = Path(
    os.environ.get(
        "PORTAL_BACKEND",
        r"C:\Users\Mohammad Touseef\PycharmProjects\translation-portal-ui\backend",
    )
)
DEFAULT_SNIP_OFFSETS = (1, 7)      # seconds after floor(srt_start)
DEFAULT_MIN_ARABIC_WORDS = 4       # heuristic: a real hadees is at least this long
DEFAULT_MIN_TASHKEEL_DENSITY = 0.30  # marks/letter: quoted Arabic ~0.5-0.8, Urdu <0.1
DEFAULT_DUP_THRESHOLD = 85.0       # >= this token_set_ratio == "already exists"
DEFAULT_MAX_HEIGHT = 720
WINDOW_PAD_END = 2                 # download a couple seconds past the last snip
MIN_ARABIC_LETTERS = 3             # ignore trivially short fragments
YTDLP_RETRIES = 3                  # transient YouTube/network failures are common
# A Quran match only counts if it's substantial: the portal matcher can align a
# common 3-word phrase (e.g. "إنه لا يحب") to a verse coincidentally. Real
# recitations span >=4 words OR cover a good fraction of the block's Arabic.
QURAN_MIN_WORDS = 4
QURAN_MIN_COVERAGE = 0.30

# Arabic diacritics (tashkeel) + Quranic annotation marks. Presence of these is
# what separates a quoted Quran/hadees from the un-voweled Urdu narration.
# Written as escapes, not literals: the class contains combining marks that any
# editor renders on top of the preceding bracket, making it unreadable and easy
# to corrupt in place.
_TASHKEEL_RE = re.compile(
    "["
    "\u064b-\u0652"  # fathatan..sukun
    "\u0670"         # superscript alef
    "\u0653-\u065f"  # maddah..wavy hamza below
    "\u06d6-\u06ed"  # Quranic annotation signs
    "\u08d3-\u08ff"  # Arabic Extended-A marks
    "\ufc5e-\ufc63"  # ligature forms carrying shadda
    "]"
)


@dataclass
class Config:
    """Everything the detector and downloader need, resolved from the CLI once.

    These were module-level constants in the original. They are a struct now
    because tuning is the whole point of this port -- Batch3's ASR vowels durood
    inline in Urdu, so the thresholds that worked for EP12/EP13 need to move,
    and a threshold you cannot set from the command line does not get swept.
    """
    srt_dir: Path
    skip_file: Path
    out_dir: Path
    portal_backend: Path
    ytdlp: str
    snip_offsets: tuple[int, ...] = DEFAULT_SNIP_OFFSETS
    min_arabic_words: int = DEFAULT_MIN_ARABIC_WORDS
    min_tashkeel_density: float = DEFAULT_MIN_TASHKEEL_DENSITY
    dup_threshold: float = DEFAULT_DUP_THRESHOLD
    max_height: int = DEFAULT_MAX_HEIGHT
    quran_any_match: bool = False

    @property
    def video_cache(self) -> Path:
        return self.out_dir / "_video_windows"

    @property
    def manifest(self) -> Path:
        return self.out_dir / "manifest.csv"

    @property
    def review_html(self) -> Path:
        return self.out_dir / "review.html"

    @property
    def window_dur(self) -> float:
        return max(self.snip_offsets) + WINDOW_PAD_END

    @property
    def snip_labels(self) -> tuple[str, ...]:
        """A, B, C... one per offset -- so --snip-offsets stays free-form."""
        return tuple(string.ascii_uppercase[: len(self.snip_offsets)])


# ── Reuse the portal's exact matching logic ────────────────────────────────────
def load_lookup_services(portal_backend: Path):
    sys.path.insert(0, str(portal_backend))
    try:
        from app.services import quran_lookup_service, sayings_lookup_service
    except Exception as exc:  # pragma: no cover - surfaced to the user
        raise SystemExit(
            f"Could not import portal lookup services from {portal_backend}\n{exc}\n"
            "Pass --portal-backend or set PORTAL_BACKEND."
        )
    return quran_lookup_service, sayings_lookup_service


# ── SRT parsing ────────────────────────────────────────────────────────────────
@dataclass
class Cue:
    index: int
    start: float  # seconds (relative to trimmed audio == SRT timeline)
    end: float
    text: str


_TS_RE = re.compile(
    r"(\d\d):(\d\d):(\d\d)[,.](\d\d\d)\s*-->\s*(\d\d):(\d\d):(\d\d)[,.](\d\d\d)"
)


def _ts_to_secs(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path: Path) -> list[Cue]:
    cues: list[Cue] = []
    blocks = re.split(r"\r?\n\r?\n", path.read_text(encoding="utf-8"))
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        m = _TS_RE.search(block)
        if not m:
            continue
        # index is the first all-digit line (may be absent); text is the rest.
        idx = 0
        text_lines = []
        for ln in lines:
            if _TS_RE.search(ln):
                continue
            if ln.strip().isdigit() and idx == 0 and not text_lines:
                idx = int(ln.strip())
                continue
            text_lines.append(ln.strip())
        start = _ts_to_secs(*m.group(1, 2, 3, 4))
        end = _ts_to_secs(*m.group(5, 6, 7, 8))
        cues.append(Cue(index=idx, start=start, end=end, text=" ".join(text_lines)))
    return cues


# ── Candidate detection + merging ──────────────────────────────────────────────
def _tashkeel_count(text: str) -> int:
    return len(_TASHKEEL_RE.findall(text))


def _is_arabic_letter(ch: str) -> bool:
    if not ch.isalpha():
        return False
    code = ord(ch)
    return (
        0x0620 <= code <= 0x064A
        or 0x0660 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
    )


def _arabic_letter_count(text: str) -> int:
    return sum(1 for ch in text if _is_arabic_letter(ch))


def tashkeel_density(text: str) -> float:
    letters = _arabic_letter_count(text)
    if letters == 0:
        return 0.0
    return _tashkeel_count(text) / letters


def arabic_word_count(text: str) -> int:
    words, has_letter = 0, False
    for ch in text:
        if _is_arabic_letter(ch):
            has_letter = True
        elif unicodedata.combining(ch) or ch == "ـ":
            continue
        else:
            if has_letter:
                words += 1
            has_letter = False
    if has_letter:
        words += 1
    return words


# ── Quoted-run analysis (the honorific / durood filter) ───────────────────────
# WHY THIS EXISTS
# ---------------
# arabic_word_count() above counts the WHOLE block, and Urdu is written in Arabic
# script, so it cannot tell quoted Arabic from the narration around it. On EP12/EP13
# that was harmless: that ASR left durood un-voweled, so a durood-only cue never
# became a candidate in the first place. Batch3's model vowels durood fully and
# inline, which breaks the assumption:
#
#     تو نبی کریم صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ کی نظرِ کیمیاں سے زید
#
# Four voweled Arabic words push the cue over the density bar, and the ten Urdu
# words around them push arabic_word_count() to 12 -- so MIN_ARABIC_WORDS=4, the
# guard specifically meant to reject durood, passes it. Measured on Batch3: 315 of
# 959 qualifying blocks (33%) were nothing but an honorific formula.
#
# The fix is to measure the QUOTED span rather than the block: take maximal runs of
# consecutive voweled words, drop the ones that are pure honorific formula, and
# count the longest thing that is left.

# Fold the orthographic variants the ASR alternates between, so one spelling of a
# formula does not slip past a stoplist keyed on another.
_NORMALIZE_MAP = str.maketrans({
    "آ": "ا", "أ": "ا", "إ": "ا",  # آ أ إ -> ا
    "ٱ": "ا",                                          # ٱ      -> ا
    "ى": "ي",                                          # ى      -> ي
    "ة": "ه", "ہ": "ه", "ھ": "ه",  # ة ہ ھ -> ه
    "ٰ": "",                                                # dagger alef
})


def _strip_marks(text: str) -> str:
    return _TASHKEEL_RE.sub("", text).replace("ـ", "")


def _normalize_word(word: str) -> str:
    """Bare consonant skeleton, for stoplist comparison only."""
    return "".join(
        ch for ch in _strip_marks(word).translate(_NORMALIZE_MAP)
        if _is_arabic_letter(ch)
    )


# Formulas that are recited constantly but are NOT the hadees being quoted:
# durood in its several lengths, the companion/imam invocations, basmala, and the
# short divine epithets. Matched per-WORD, not per-phrase, because a cue boundary
# routinely cuts a formula in half ("اللَّهُ عَلَيْهِ وَسَلَّمَ" with no صلى).
_HONORIFIC_PHRASES = (
    "صلى الله عليه وسلم",              # durood
    "صلى الله عليه وآله وسلم",
    "عليه وآله وصحبه أجمعين",
    "وبارك وسلم",
    "رضي الله تعالى عنه",              # radiallahu anhu/anha/anhum
    "رضي الله عنها عنهم عليهم",
    "عليه السلام",                     # alayhis salam
    "عليها السلام",
    "جل جلاله",                        # jalla jalaluhu
    "عز وجل",                          # azza wa jall
    "سبحانه وتعالى",                   # subhanahu wa ta'ala
    "كرم الله وجهه",                   # karamallahu wajhah
    "رحمه الله",                       # rahimahullah
    "بسم الله الرحمن الرحيم",          # basmala
)
HONORIFIC_TOKENS = {
    _normalize_word(w)
    for phrase in _HONORIFIC_PHRASES for w in phrase.split()
}
HONORIFIC_TOKENS.discard("")


def voweled_runs(text: str) -> list[list[str]]:
    """Maximal runs of consecutive words that each carry at least one tashkeel
    mark. A quoted Arabic span is one such run; un-voweled Urdu breaks it."""
    runs: list[list[str]] = []
    cur: list[str] = []
    for word in text.split():
        if _TASHKEEL_RE.search(word) and _arabic_letter_count(word):
            cur.append(word)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def quoted_runs(text: str) -> list[list[str]]:
    """Voweled runs that could be a quotation: 2+ words, not pure honorific.

    Single-word runs are dropped because in this corpus they are almost always an
    izafat kasra on an ordinary Urdu word (نظرِ، سینہِ، لطیفہِ), not Arabic at all.
    """
    return [r for r in voweled_runs(text)
            if len(r) >= 2 and not is_honorific_run(r)]


def is_honorific_run(run: list[str]) -> bool:
    """True when EVERY word of the run belongs to a stock formula, i.e. the run
    carries no quoted content of its own."""
    return bool(run) and all(_normalize_word(w) in HONORIFIC_TOKENS for w in run)


def quoted_word_count(text: str) -> int:
    """Length of the longest quotable voweled run -- the replacement for
    arabic_word_count() in the "is this long enough to be a hadees" test.

    The longest single run, not the sum over runs: summing lets several unrelated
    two-word fragments in one block add up to a passing score.
    """
    runs = quoted_runs(text)
    return max((len(r) for r in runs), default=0)


def is_candidate_cue(cue: Cue, cfg: Config) -> bool:
    """A cue holds quoted Quran/hadees if it is DENSELY voweled. Urdu narration
    carries only occasional diacritics (izafat kasra, etc.) so its tashkeel
    density stays < 0.1; fully-voweled Arabic runs 0.5-0.8. A raw-count rule
    misfires on ordinary Urdu, so we key on density instead."""
    return (
        _arabic_letter_count(cue.text) >= MIN_ARABIC_LETTERS
        and tashkeel_density(cue.text) >= cfg.min_tashkeel_density
    )


@dataclass
class Block:
    cues: list[Cue] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.cues[0].start

    @property
    def end(self) -> float:
        return self.cues[-1].end

    @property
    def index_range(self) -> str:
        return f"{self.cues[0].index}-{self.cues[-1].index}"

    @property
    def text(self) -> str:
        return " ".join(c.text for c in self.cues).strip()


def merge_candidate_blocks(cues: list[Cue], cfg: Config) -> list[Block]:
    """Group runs of consecutive candidate cues into blocks."""
    blocks: list[Block] = []
    cur: Block | None = None
    for cue in cues:
        if is_candidate_cue(cue, cfg):
            if cur is None:
                cur = Block()
            cur.cues.append(cue)
        else:
            if cur is not None:
                blocks.append(cur)
                cur = None
    if cur is not None:
        blocks.append(cur)
    return blocks


# ── Classification ─────────────────────────────────────────────────────────────
@dataclass
class Classification:
    verdict: str  # "honorific"|"quran"|"existing_hadees"|"too_short"|"new_hadees"
    detail: str


#: Verdicts that produce no snip. Only "new_hadees" is downloaded.
SKIP_VERDICTS = ("honorific", "quran", "existing_hadees", "too_short")

VERDICT_TAGS = {
    "honorific": "DUROOD",
    "quran": "QURAN ",
    "existing_hadees": "EXIST ",
    "too_short": "short ",
    "new_hadees": "NEW   ",
}


def _is_substantial_quran(text: str, matches, cfg: Config) -> bool:
    """Guard against coincidental short-phrase matches: accept a Quran verdict
    only if some match spans >= QURAN_MIN_WORDS words, or the matched spans
    cover >= QURAN_MIN_COVERAGE of the block's Arabic letters.

    With --quran-any-match the guard is off and ANY match counts, which is what
    you want when the goal is "never spend a snip on a verse" and you would
    rather lose the odd real hadees than review Quran frames. The guard exists
    because on EP13 a 3-word coincidence ("إنه لا يحب", ~21% coverage) matched
    Al-A'raf 7:31 and hid a genuine hadees -- see sessions/006-2026-07-12.md.
    """
    if not matches:
        return False
    if cfg.quran_any_match:
        return True
    if any(m.matched_words >= QURAN_MIN_WORDS for m in matches):
        return True
    # Coverage is measured against the QUOTED span, not the whole block. Using the
    # block let ordinary Urdu inflate the denominator: a fully-matched 6-word verse
    # sitting in a 24-word Urdu cue scored 6/24 = 0.25, under the 0.30 bar, and was
    # called a coincidence. That mislabelled 9 real verses in Batch3 as new hadees
    # (B3022 87:14, B3046 38:82, B3048 49:14, B3066 53:3, B3076 33:45, ...).
    # Against the quoted span the same verse scores ~1.0, while EP13's genuine
    # 3-words-of-14 coincidence stays low and is still kept.
    denom = sum(_arabic_letter_count(w) for run in quoted_runs(text) for w in run)
    if denom == 0:
        denom = _arabic_letter_count(text)
    covered = sum(_arabic_letter_count(m.transcript_text) for m in matches)
    return denom > 0 and covered / denom >= QURAN_MIN_COVERAGE


def classify_block(block: Block, quran_svc, sayings_svc, cfg: Config) -> Classification:
    text = block.text

    # Cheap local checks first: they cost microseconds and the Quran lookup costs
    # ~0.3s, so filtering the honorific blocks here removes about a third of the
    # lookups outright. Safe to precede the lookups because a block with no
    # quotable run has nothing for either service to match on. (Basmala is the one
    # verse this labels DUROOD rather than QURAN -- both skip, so nothing changes
    # downstream.)
    if not quoted_runs(text):
        return Classification(
            "honorific",
            f"no quotable run (density={tashkeel_density(text):.2f}); "
            "honorific formula or izafat only",
        )
    quoted = quoted_word_count(text)
    if quoted < cfg.min_arabic_words:
        return Classification(
            "too_short", f"longest quoted run={quoted}w < {cfg.min_arabic_words}"
        )

    quran = quran_svc.find_quran_matches(text)
    if _is_substantial_quran(text, quran, cfg):
        m = quran[0]
        return Classification(
            "quran",
            f"{m.surah_name} {m.surah_number}:{m.ayah_number} score={m.score}",
        )
    dup = sayings_svc.find_duplicate(text)
    if dup is not None and dup.score >= cfg.dup_threshold:
        return Classification("existing_hadees", f"{dup.reference} score={dup.score}")
    return Classification(
        "new_hadees",
        f"quoted={quoted}w of {arabic_word_count(text)} arabic "
        f"density={tashkeel_density(text):.2f}",
    )


# ── skip_start map ─────────────────────────────────────────────────────────────
SKIP_FILE_COLUMNS = ("audio_filename", "skip_start_seconds")


def load_skip_start(path: Path) -> dict[str, float]:
    """stem -> skip_start_seconds, from the batch's *_validated.csv.

    Keyed by the audio filename's STEM (B3001_E5NiYnR-c98.mp3 ->
    B3001_E5NiYnR-c98) because that is exactly the SRT filename's stem, so the
    two sides join without either knowing the other's extension.
    """
    if not path.exists():
        raise SystemExit(f"skip-start file not found: {path}")
    fname_col, skip_col = SKIP_FILE_COLUMNS
    out: dict[str, float] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in SKIP_FILE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"{path} is missing column(s) {missing}; found {reader.fieldnames}"
            )
        for row in reader:
            if not row.get(fname_col):
                continue
            out[Path(str(row[fname_col])).stem] = float(row[skip_col] or 0)
    return out


# ── Video download + frame extraction ──────────────────────────────────────────
def video_id_from_stem(stem: str) -> str:
    """B3001_E5NiYnR-c98 -> E5NiYnR-c98; EP13__riVrVuUdOA -> _riVrVuUdOA.
    The id is everything after the FIRST underscore, so ids that themselves
    begin with "_" survive."""
    return stem.split("_", 1)[1]


def ep_label(stem: str) -> str:
    return stem.split("_", 1)[0]  # B3001 / EP13


def resolve_ytdlp(explicit: str = "") -> str:
    """yt-dlp executable: --ytdlp wins, else this venv's Scripts/, else PATH.

    The original hardcoded another project's .venv path, which silently pinned
    every run to whatever yt-dlp version that venv happened to hold -- and
    YouTube extraction breaks often enough that the version matters.
    """
    if explicit:
        return explicit
    local = Path(sys.executable).parent / ("yt-dlp.exe" if os.name == "nt" else "yt-dlp")
    if local.exists():
        return str(local)
    found = shutil.which("yt-dlp")
    if found:
        return found
    raise SystemExit(
        "yt-dlp not found next to the running interpreter or on PATH; pass --ytdlp."
    )


def download_window(video_id: str, start_v: float, dur: float, stem: Path,
                    cfg: Config) -> Path | None:
    """Download [start_v, start_v+dur] of the video at <=max_height. yt-dlp picks
    the container (usually .mkv when merging video+audio), so we let it fill the
    extension and return whatever file it produced. --force-keyframes-at-cuts
    makes file-time 0 == start_v."""
    existing = sorted(stem.parent.glob(stem.name + ".*"))
    if existing:
        return existing[0]
    stem.parent.mkdir(parents=True, exist_ok=True)
    section = f"*{start_v:.3f}-{start_v + dur:.3f}"
    cmd = [
        cfg.ytdlp,
        "-f", f"bv*[height<={cfg.max_height}]+ba/b[height<={cfg.max_height}]/b",
        "--download-sections", section,
        "--force-keyframes-at-cuts",
        "-o", str(stem) + ".%(ext)s",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    for attempt in range(1, YTDLP_RETRIES + 1):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        produced = sorted(stem.parent.glob(stem.name + ".*"))
        if proc.returncode == 0 and produced:
            return produced[0]
        last_err = proc.stderr.strip()[-200:]
        # Clean up any partial file so the retry (and glob) starts fresh.
        for p in produced:
            p.unlink(missing_ok=True)
        print(f"    yt-dlp attempt {attempt}/{YTDLP_RETRIES} failed "
              f"({proc.returncode}): {last_err}")
    return None


def grab_frame(window_path: Path, file_time: float, out_img: Path) -> bool:
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{file_time:.3f}", "-i", str(window_path),
        "-frames:v", "1", "-q:v", "2", str(out_img),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_img.exists():
        print(f"    ffmpeg FAILED: {proc.stderr.strip()[-300:]}")
        return False
    return True


def ts_label(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    return f"{h:02d}-{m:02d}-{s:02d}_{int(round((secs - int(secs)) * 1000)):03d}"


# ── Jobs ───────────────────────────────────────────────────────────────────────
@dataclass
class Job:
    """A confirmed new-hadees candidate, with everything needed to snip it."""
    stem: str
    row: dict
    video_id: str
    window_start_v: float
    window_dur: float
    snip_srt: list[int]
    snip_vid: list[float]


def manifest_fields(cfg: Config) -> list[str]:
    """Manifest columns. The per-snip columns are generated from --snip-offsets
    rather than hardcoded to A/B, so changing the offsets cannot silently drop
    a snip's timings out of the manifest."""
    per_snip: list[str] = []
    for lbl in cfg.snip_labels:
        per_snip += [f"snip{lbl}_srt_s", f"snip{lbl}_video_s"]
    return (
        ["episode", "video_id", "skip_start_s", "cue_range", "srt_start"]
        + per_snip
        + ["quoted_words", "arabic_words", "tashkeel_density", "decision"]
        + [f"snip{lbl}_path" for lbl in cfg.snip_labels]
        + ["text"]
    )


def classify_episode(srt_path: Path, skip_map: dict, quran_svc, sayings_svc,
                     cfg: Config, counts: dict[str, int]) -> list[Job]:
    """Parse + classify one SRT, printing every verdict. Returns the new-hadees
    jobs (no download yet); `counts` accumulates verdicts across episodes."""
    stem = srt_path.stem
    ep = ep_label(stem)
    vid = video_id_from_stem(stem)
    skip = skip_map.get(stem)
    if skip is None:
        print(f"  [!] no skip_start for {stem}; assuming 0 "
              f"(snips will be off by the intro-music length)")
        skip = 0.0

    cues = parse_srt(srt_path)
    blocks = merge_candidate_blocks(cues, cfg)
    print(f"{ep} ({vid}) skip_start={skip}s  cues={len(cues)} "
          f"candidate-blocks={len(blocks)}")

    jobs: list[Job] = []
    for block in blocks:
        cls = classify_block(block, quran_svc, sayings_svc, cfg)
        counts[cls.verdict] = counts.get(cls.verdict, 0) + 1
        print(f"  [{VERDICT_TAGS[cls.verdict]}] cues {block.index_range} "
              f"@ {ts_label(block.start)} :: {cls.detail}")
        if cls.verdict != "new_hadees":
            continue

        floor_start = math.floor(block.start)
        snip_srt = [floor_start + off for off in cfg.snip_offsets]  # SRT-timeline
        snip_vid = [t + skip for t in snip_srt]                     # video-timeline
        row = {
            "episode": ep, "video_id": vid, "skip_start_s": skip,
            "cue_range": block.index_range,
            "srt_start": ts_label(block.start),
            "quoted_words": quoted_word_count(block.text),
            "arabic_words": arabic_word_count(block.text),
            "tashkeel_density": f"{tashkeel_density(block.text):.2f}",
            "text": block.text,
            "decision": "",
        }
        for lbl, st, sv in zip(cfg.snip_labels, snip_srt, snip_vid):
            row[f"snip{lbl}_srt_s"] = st
            row[f"snip{lbl}_video_s"] = sv
            row[f"snip{lbl}_path"] = ""
        jobs.append(Job(
            stem=stem, row=row, video_id=vid,
            window_start_v=floor_start + skip,
            window_dur=cfg.window_dur,
            snip_srt=snip_srt, snip_vid=snip_vid,
        ))
    return jobs


def run_snip(job: Job, cfg: Config) -> None:
    """Download the job's window and extract every frame, filling row paths."""
    win = download_window(
        job.video_id, job.window_start_v, job.window_dur,
        cfg.video_cache / f"{job.stem}_{int(job.window_start_v)}", cfg,
    )
    if not win:
        return
    ep = job.row["episode"]
    cues = job.row["cue_range"]  # e.g. "462-462" -> filename carries the hadees id
    for lbl, st, sv in zip(cfg.snip_labels, job.snip_srt, job.snip_vid):
        img = cfg.out_dir / ep / f"{ep}_cues{cues}_{ts_label(st)}_snip{lbl}.jpg"
        file_time = sv - job.window_start_v  # == the offset itself
        if grab_frame(win, file_time, img):
            try:
                job.row[f"snip{lbl}_path"] = str(img.relative_to(REPO_ROOT))
            except ValueError:  # --out-dir outside the repo
                job.row[f"snip{lbl}_path"] = str(img)


# ── Review surfaces ────────────────────────────────────────────────────────────
def _youtube_link(video_id: str, at_seconds: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(at_seconds)}s"


def print_hadees_review(jobs: list[Job]) -> None:
    """Show the detected new hadees so the user can decide before downloading.
    Terminals render RTL Arabic poorly, so the metadata is the primary content
    here (the HTML review is the readable surface); the text is wrapped in a
    First-Strong Isolate (U+2068..U+2069) so it at least stays on its own line."""
    print("\n" + "=" * 70)
    print(f"Detected {len(jobs)} NEW-hadees candidate(s) to snip:")
    print("(Arabic renders correctly in the HTML review that just opened.)")
    print("=" * 70)
    for i, job in enumerate(jobs, 1):
        r = job.row
        snips = " & ".join(f"{v:.0f}s" for v in job.snip_vid)
        print(f"\n[{i}] {r['episode']}  cues {r['cue_range']}  @ SRT {r['srt_start']}"
              f"  ({r['quoted_words']}w quoted / {r['arabic_words']}w arabic)"
              f"  → video {snips}")
        print(f"    \u2068{r['text']}\u2069")
    print("\n" + "=" * 70)


def write_review_html(jobs: list[Job], path: Path) -> None:
    """Write a right-to-left HTML review of the candidates so the Arabic/Urdu
    renders correctly (terminals mangle bidi text). One card per hadees with its
    metadata and a click-to-verify YouTube link at the first snip timestamp."""
    cards = []
    for i, job in enumerate(jobs, 1):
        r = job.row
        link = _youtube_link(r["video_id"], job.snip_vid[0])
        snips = " &amp; ".join(f"{v:.0f}s" for v in job.snip_vid)
        cards.append(f"""
    <div class="card">
      <div class="meta">
        <span class="idx">#{i}</span>
        <b>{html.escape(r['episode'])}</b> · cues {html.escape(r['cue_range'])}
        · SRT {html.escape(r['srt_start'])}
        · {r['quoted_words']}w quoted / {r['arabic_words']}w arabic
        · density {r['tashkeel_density']}
        · snips @ video {snips}
        · <a href="{link}" target="_blank">▶ verify on YouTube</a>
      </div>
      <div class="arabic" dir="rtl" lang="ar">{html.escape(r['text'])}</div>
    </div>""")
    doc = f"""<!doctype html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8">
<title>Hadees review — {len(jobs)} candidate(s)</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background:#f5f5f4;
         color:#1c1917; margin:0; padding:24px; }}
  h1 {{ font-size:20px; direction:ltr; }}
  .hint {{ direction:ltr; color:#57534e; margin-bottom:20px; font-size:14px; }}
  .card {{ background:#fff; border:1px solid #e7e5e4; border-radius:10px;
          padding:16px 20px; margin:0 0 16px; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
  .meta {{ direction:ltr; font-size:13px; color:#57534e; margin-bottom:12px;
          border-bottom:1px solid #f0efee; padding-bottom:8px; }}
  .idx {{ display:inline-block; background:#1c1917; color:#fff; border-radius:5px;
         padding:1px 8px; margin-inline-end:6px; font-weight:600; }}
  .meta a {{ color:#b45309; text-decoration:none; }}
  .meta a:hover {{ text-decoration:underline; }}
  .arabic {{ font-family:'Traditional Arabic','Amiri','Noto Naskh Arabic','Segoe UI',serif;
            font-size:30px; line-height:2.1; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background:#1c1917; color:#f5f5f4; }}
    .card {{ background:#292524; border-color:#44403c; }}
    .meta {{ color:#a8a29e; border-color:#3a3633; }}
    .idx {{ background:#f5f5f4; color:#1c1917; }}
    .meta a {{ color:#fbbf24; }}
  }}
</style></head><body>
<h1>Hadees review — {len(jobs)} candidate(s)</h1>
<div class="hint">Review the detected hadees below, then answer the y/N prompt in your terminal.
Click ▶ to jump to the moment in the source video.</div>
{''.join(cards)}
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def prompt_hadees(i: int, total: int, job: Job) -> str:
    """Show one hadees and ask what to do. Returns 'y' (snip), 'n' (skip),
    'a' (snip this + all remaining), or 'q' (skip this + quit the rest)."""
    r = job.row
    snips = " & ".join(f"{v:.0f}s" for v in job.snip_vid)
    print(f"\n[{i}/{total}] {r['episode']}  cues {r['cue_range']}  @ SRT {r['srt_start']}"
          f"  ({r['arabic_words']} words)  → video {snips}")
    print(f"    \u2068{r['text']}\u2069")
    print(f"    (card #{i} in the HTML review)")
    while True:
        try:
            ans = input("    snip this? [y]es / [n]o skip / [a]ll remaining / "
                        "[q]uit: ").strip().lower()
        except EOFError:
            return "q"
        if ans in ("y", "yes"):
            return "y"
        if ans in ("n", "no", "s", "skip"):
            return "n"
        if ans in ("a", "all"):
            return "a"
        if ans in ("q", "quit"):
            return "q"
        print("    Please enter y, n, a, or q.")


# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_offsets(raw: str) -> tuple[int, ...]:
    try:
        offs = tuple(sorted({int(x) for x in re.split(r"[,\s]+", raw.strip()) if x}))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--snip-offsets must be integers, got {raw!r}")
    if not offs:
        raise argparse.ArgumentTypeError("--snip-offsets is empty")
    if len(offs) > 26:
        raise argparse.ArgumentTypeError("at most 26 offsets (labels are A-Z)")
    if offs[0] < 0:
        raise argparse.ArgumentTypeError("--snip-offsets must be >= 0")
    return offs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_batch_argument(p)
    g = p.add_argument_group("paths (override the --batch defaults)")
    g.add_argument("--srt-dir", default="", help="Directory of *.srt to scan.")
    g.add_argument("--skip-file", default="",
                   help="CSV with audio_filename + skip_start_seconds columns "
                        "(default: the batch's *_validated.csv).")
    g.add_argument("--out-dir", default="", help="Where frames/manifest/review go.")
    g.add_argument("--portal-backend", default=str(DEFAULT_PORTAL_BACKEND),
                   help="translation-portal-ui/backend (env: PORTAL_BACKEND).")
    g.add_argument("--ytdlp", default="",
                   help="yt-dlp executable (default: this venv, then PATH).")

    s = p.add_argument_group("selection")
    s.add_argument("--only", default="",
                   help="Comma-separated episode labels, e.g. B3002,B3006. Default: all.")
    s.add_argument("--limit", type=int, default=0,
                   help="Stop after N candidates (0 = no limit). For partial runs.")

    t = p.add_argument_group("detector tuning")
    t.add_argument("--snip-offsets", type=_parse_offsets, default=DEFAULT_SNIP_OFFSETS,
                   metavar="SECS",
                   help="Seconds after floor(srt_start) to grab, comma-separated "
                        f"(default: {','.join(map(str, DEFAULT_SNIP_OFFSETS))}).")
    t.add_argument("--min-arabic-words", type=int, default=DEFAULT_MIN_ARABIC_WORDS,
                   help=f"Min Arabic words for a block to count as a hadees "
                        f"(default: {DEFAULT_MIN_ARABIC_WORDS}).")
    t.add_argument("--min-density", type=float, default=DEFAULT_MIN_TASHKEEL_DENSITY,
                   help=f"Min tashkeel marks/letter for a candidate cue "
                        f"(default: {DEFAULT_MIN_TASHKEEL_DENSITY}).")
    t.add_argument("--dup-threshold", type=float, default=DEFAULT_DUP_THRESHOLD,
                   help=f"token_set_ratio at/above which a saying already exists "
                        f"(default: {DEFAULT_DUP_THRESHOLD}).")
    t.add_argument("--max-height", type=int, default=DEFAULT_MAX_HEIGHT,
                   help=f"Max video height to download (default: {DEFAULT_MAX_HEIGHT}).")
    t.add_argument("--quran-any-match", action="store_true",
                   help="Treat ANY Quran match as a verse and skip it, instead of "
                        f"requiring >={QURAN_MIN_WORDS} matched words or "
                        f">={QURAN_MIN_COVERAGE:.0%} coverage. Never spends a snip on "
                        "a verse, at the cost of losing hadees that coincidentally "
                        "share a short phrase with one.")

    b = p.add_argument_group("behaviour")
    b.add_argument("--no-download", action="store_true",
                   help="Classify only; skip video/frames.")
    b.add_argument("--yes", "-y", action="store_true",
                   help="Skip the per-hadees prompt (non-interactive).")
    b.add_argument("--no-open", action="store_true",
                   help="Don't auto-open the HTML review in a browser.")
    return p


def config_from_args(args) -> Config:
    paths = BatchPaths(args.batch, data_root=REPO_ROOT / "data")
    return Config(
        srt_dir=Path(args.srt_dir) if args.srt_dir else paths.srt_dir,
        skip_file=Path(args.skip_file) if args.skip_file else paths.validated_csv,
        out_dir=Path(args.out_dir) if args.out_dir else paths.hadees_snips_dir,
        portal_backend=Path(args.portal_backend),
        ytdlp=resolve_ytdlp(args.ytdlp),
        snip_offsets=tuple(args.snip_offsets),
        min_arabic_words=args.min_arabic_words,
        min_tashkeel_density=args.min_density,
        dup_threshold=args.dup_threshold,
        max_height=args.max_height,
        quran_any_match=args.quran_any_match,
    )


def select_srts(cfg: Config, only: str) -> list[Path]:
    if not cfg.srt_dir.is_dir():
        raise SystemExit(f"--srt-dir is not a directory: {cfg.srt_dir}")
    srts = sorted(cfg.srt_dir.glob("*.srt"))
    if not srts:
        raise SystemExit(f"No .srt files in {cfg.srt_dir}")
    if only:
        want = {e.strip().upper() for e in only.split(",") if e.strip()}
        srts = [p for p in srts if ep_label(p.stem).upper() in want]
        found = {ep_label(p.stem).upper() for p in srts}
        for miss in sorted(want - found):
            print(f"  [!] --only {miss}: no matching SRT in {cfg.srt_dir}")
    if not srts:
        raise SystemExit("No matching SRT files.")
    return srts


def append_manifest(jobs: list[Job], cfg: Config) -> Path:
    """Append one row per candidate. Header is written only for a new file, and
    a header mismatch is refused rather than appended under the wrong columns --
    a different --snip-offsets produces a different schema."""
    fields = manifest_fields(cfg)
    manifest = cfg.manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest.exists()
    if not write_header:
        with open(manifest, encoding="utf-8-sig", newline="") as f:
            existing = next(csv.reader(f), [])
        if existing and existing != fields:
            raise SystemExit(
                f"{manifest} has a different column set (probably a different "
                f"--snip-offsets).\n  existing: {existing}\n  this run: {fields}\n"
                "Use --out-dir for this run, or move the old manifest aside."
            )
    with open(manifest, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for job in jobs:
            w.writerow(job.row)
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(args)

    print(f"srt-dir   : {cfg.srt_dir}")
    print(f"skip-file : {cfg.skip_file}")
    print(f"out-dir   : {cfg.out_dir}")
    print(f"offsets   : {','.join(map(str, cfg.snip_offsets))}s  "
          f"(window {cfg.window_dur:.0f}s, <={cfg.max_height}p)")
    print(f"thresholds: density>={cfg.min_tashkeel_density} "
          f"words>={cfg.min_arabic_words} dup>={cfg.dup_threshold}\n")

    quran_svc, sayings_svc = load_lookup_services(cfg.portal_backend)
    skip_map = load_skip_start(cfg.skip_file)
    srts = select_srts(cfg, args.only)

    # Phase 1: classify everything (no download).
    jobs: list[Job] = []
    counts: dict[str, int] = {}
    for srt in srts:
        jobs.extend(classify_episode(srt, skip_map, quran_svc, sayings_svc, cfg, counts))

    total = sum(counts.values())
    print(f"\n{total} candidate block(s) across {len(srts)} episode(s):")
    for verdict in (*SKIP_VERDICTS, "new_hadees"):
        n = counts.get(verdict, 0)
        if n:
            share = f"{100 * n / total:5.1f}%" if total else "    -"
            note = "" if verdict == "new_hadees" else "  (no snip)"
            print(f"  {VERDICT_TAGS[verdict].strip():<15} {n:5d}  {share}{note}")

    if not jobs:
        print("\nNo new-hadees candidates found. Nothing to do.")
        return

    if args.limit and len(jobs) > args.limit:
        print(f"\n--limit {args.limit}: keeping the first {args.limit} of "
              f"{len(jobs)} candidate(s); the rest are NOT in this manifest.")
        jobs = jobs[: args.limit]

    # Phase 2: show the hadees and confirm before any download. Write an RTL
    # HTML review (terminals mangle Arabic) and open it for the user to read.
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    write_review_html(jobs, cfg.review_html)
    if not args.no_open:
        try:
            webbrowser.open(cfg.review_html.resolve().as_uri())
        except Exception:
            pass
    print(f"\nReview written to: {cfg.review_html}")

    print_hadees_review(jobs)
    if args.no_download:
        print("--no-download set: classification only, skipping frames.")
        for job in jobs:
            job.row["decision"] = "not_attempted"
    else:
        # Phase 3: per-hadees decision, then download + snip the accepted ones.
        snip_all = args.yes
        quit_rest = False
        snipped = 0
        for i, job in enumerate(jobs, 1):
            if quit_rest:
                job.row["decision"] = "skipped"
                continue
            if not snip_all:
                ans = prompt_hadees(i, len(jobs), job)
                if ans == "q":
                    job.row["decision"] = "skipped"
                    quit_rest = True
                    continue
                if ans == "n":
                    job.row["decision"] = "skipped"
                    continue
                if ans == "a":
                    snip_all = True  # snip this one and everything after
            job.row["decision"] = "snipped"
            print(f"  snipping {job.row['episode']} cues {job.row['cue_range']} ...")
            run_snip(job, cfg)
            snipped += 1
        print(f"\nSnipped {snipped}, skipped {len(jobs) - snipped} of "
              f"{len(jobs)} candidate(s).")

    manifest = append_manifest(jobs, cfg)
    print(f"\nDone. {len(jobs)} new-hadees rows -> {manifest}")


if __name__ == "__main__":
    main()