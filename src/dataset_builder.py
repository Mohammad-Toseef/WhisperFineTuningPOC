"""
Build a HuggingFace DatasetDict from a training manifest, then upload to Modal.

Split strategy (changed from random per-clip 90/10):
  - WHOLE-EPISODE HOLDOUT — every clip of a held-out episode goes to eval, so no
    near-duplicate clips leak across train/eval. This reflects the real goal:
    transcribing *other* videos of the same speaker.
  - TERM-WEIGHTED eval — held-out episodes are chosen richest-first (most
    spiritual-term / code-switch clips) so the metric actually exercises the two
    training goals. Override with --eval-episodes to hand-pick.

Also writes a sidecar `<output>/eval_buckets.json` (one record per eval clip, in
eval order: {episode, sentence, buckets}) that modal_app.py::evaluate reads to
report per-bucket WER. Buckets: 'code_switch' | 'nastaliq_only' (mutually
exclusive) and 'spiritual_term' (orthogonal).

Usage:
    python src/dataset_builder.py <manifest_path> [output_path] [--eval-split 0.10]
        [--eval-episodes EP5,EP12,...] [--domain-terms config/domain_terms.json]
"""
import sys
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

from datasets import Dataset, DatasetDict, Audio

sys.stdout.reconfigure(encoding="utf-8")

# A run of >=2 Latin letters marks English code-switching in an Urdu transcript.
_LATIN_RE = re.compile(r"[A-Za-z]{2,}")


def load_domain_terms(path: str) -> list[str]:
    """Flatten spiritual_terms + arabic_phrases from domain_terms.json."""
    p = Path(path)
    if not p.exists():
        print(f"⚠️  domain terms file not found ({path}) — spiritual_term bucket disabled.")
        return []
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("spiritual_terms", [])) + list(data.get("arabic_phrases", []))


def episode_of(sample: dict) -> str:
    """Parent folder of the audio path, e.g. 'EP7_vugW7VEqLko'."""
    return Path(sample["audio_path"].replace("\\", "/")).parent.name


def buckets_for(text: str, terms: list[str]) -> list[str]:
    b = ["code_switch"] if _LATIN_RE.search(text) else ["nastaliq_only"]
    if terms and any(t in text for t in terms):
        b.append("spiritual_term")
    return b


def _ep_num(ep: str) -> int:
    m = re.match(r"EP(\d+)", ep)
    return int(m.group(1)) if m else 0


def select_eval_episodes(
    by_episode: dict[str, list[dict]],
    terms: list[str],
    eval_split: float,
    forced: list[str] | None,
) -> list[str]:
    """Pick whole episodes for eval: forced list, else richest-first to ~eval_split."""
    if forced:
        missing = [e for e in forced if e not in by_episode]
        if missing:
            print(f"ERROR: --eval-episodes not found in manifest: {missing}", file=sys.stderr)
            sys.exit(1)
        return forced

    total_clips = sum(len(v) for v in by_episode.values())
    target = max(1, round(total_clips * eval_split))

    def richness(ep: str) -> tuple[float, int]:
        clips = by_episode[ep]
        rich = sum(1 for s in clips if len(buckets_for(s["transcript"], terms)) > 1
                   or "code_switch" in buckets_for(s["transcript"], terms))
        return (rich / len(clips), rich)   # fraction first, then absolute count

    ranked = sorted(by_episode, key=lambda e: (richness(e), len(by_episode[e])), reverse=True)

    chosen, n = [], 0
    for ep in ranked:
        if n >= target and chosen:
            break
        # Never hold out every episode.
        if len(chosen) >= len(by_episode) - 1:
            break
        chosen.append(ep)
        n += len(by_episode[ep])
    return chosen


def build_dataset(
    manifest_path: str,
    output_path: str,
    eval_split: float = 0.10,
    eval_episodes: list[str] | None = None,
    domain_terms_path: str = "config/domain_terms.json",
):
    with open(manifest_path, encoding="utf-8") as f:
        samples = json.load(f)

    terms = load_domain_terms(domain_terms_path)

    by_episode: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_episode[episode_of(s)].append(s)

    eval_eps = set(select_eval_episodes(by_episode, terms, eval_split, eval_episodes))

    train_samples, eval_samples = [], []
    for ep, clips in by_episode.items():
        (eval_samples if ep in eval_eps else train_samples).extend(clips)

    # Sanity: no episode may appear in both splits (holdout invariant).
    assert not ({episode_of(s) for s in train_samples} & {episode_of(s) for s in eval_samples}), \
        "Episode leaked across train/eval"

    def to_hf(sample_list):
        return {
            "audio": [s["audio_path"] for s in sample_list],
            "sentence": [s["transcript"] for s in sample_list],
            "language": [s.get("language", "ur") for s in sample_list],
        }

    dataset = DatasetDict({
        "train": Dataset.from_dict(to_hf(train_samples)),
        "eval": Dataset.from_dict(to_hf(eval_samples)),
    })
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset.save_to_disk(output_path)

    # Sidecar bucket tags for the evaluator (aligned to eval row order).
    eval_buckets = [
        {"episode": episode_of(s),
         "sentence": s["transcript"],
         "buckets": buckets_for(s["transcript"], terms)}
        for s in eval_samples
    ]
    with open(Path(output_path) / "eval_buckets.json", "w", encoding="utf-8") as f:
        json.dump(eval_buckets, f, ensure_ascii=False, indent=2)

    _report(by_episode, eval_eps, train_samples, eval_samples, terms)
    return dataset


def _report(by_episode, eval_eps, train_samples, eval_samples, terms):
    def bucket_counts(samples):
        c = defaultdict(int)
        for s in samples:
            for b in buckets_for(s["transcript"], terms):
                c[b] += 1
        return c

    tr, ev = bucket_counts(train_samples), bucket_counts(eval_samples)
    print("\n" + "=" * 56)
    print(f"  Episodes: {len(by_episode)} total  |  {len(eval_eps)} held out for eval")
    print(f"  HELD-OUT: {sorted(eval_eps, key=_ep_num)}")
    print("-" * 56)
    print(f"  {'bucket':<16}{'train':>10}{'eval':>10}")
    for b in ("nastaliq_only", "code_switch", "spiritual_term"):
        print(f"  {b:<16}{tr[b]:>10}{ev[b]:>10}")
    print("-" * 56)
    print(f"  ✅ {len(train_samples)} train / {len(eval_samples)} eval clips "
          f"({len(eval_samples) / (len(train_samples) + len(eval_samples)):.0%} eval)")
    print("=" * 56)


def main():
    p = argparse.ArgumentParser(description="Build HF dataset with whole-episode holdout + term buckets.")
    p.add_argument("manifest", nargs="?", default="./data/processed/Batch1_EP23/manifest_reviewed.json",
                   help="Path to the training manifest JSON.")
    p.add_argument("output", nargs="?", default="./data/processed/dataset",
                   help="Output dataset dir (default: ./data/processed/dataset).")
    p.add_argument("--eval-split", type=float, default=0.10,
                   help="Approx fraction of CLIPS to hold out as eval (whole episodes). Default 0.10.")
    p.add_argument("--eval-episodes", default=None,
                   help="Comma-separated episode ids to force as eval (e.g. EP5,EP12). Overrides --eval-split.")
    p.add_argument("--domain-terms", default="config/domain_terms.json",
                   help="Path to domain_terms.json for spiritual-term bucketing.")
    args = p.parse_args()

    forced = [e.strip() for e in args.eval_episodes.split(",")] if args.eval_episodes else None
    build_dataset(args.manifest, args.output, args.eval_split, forced, args.domain_terms)


if __name__ == "__main__":
    main()
