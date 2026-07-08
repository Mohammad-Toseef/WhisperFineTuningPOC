"""
Run locally to convert your manifest.json into a HuggingFace dataset,
then upload to Modal Volume.
"""
import sys
import json
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio

sys.stdout.reconfigure(encoding="utf-8")


def build_dataset(manifest_path: str, output_path: str, 
                  eval_split: float = 0.10):
    with open(manifest_path, encoding="utf-8") as f:
        samples = json.load(f)

    # Shuffle consistently
    import random
    random.seed(42)
    random.shuffle(samples)

    split_idx = int(len(samples) * (1 - eval_split))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    def to_hf_format(sample_list):
        return {
            "audio": [s["audio_path"] for s in sample_list],
            "sentence": [s["transcript"] for s in sample_list],
            "language": [s.get("language", "ur") for s in sample_list],
        }

    dataset = DatasetDict({
        "train": Dataset.from_dict(to_hf_format(train_samples)),
        "eval": Dataset.from_dict(to_hf_format(eval_samples)),
    })

    # Cast audio column to load actual arrays
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset.save_to_disk(output_path)

    print(f"✅ Dataset saved: {len(train_samples)} train / {len(eval_samples)} eval")
    return dataset


if __name__ == "__main__":
    # POC: build from the reviewed manifest. Override via CLI args if needed:
    #   python src/dataset_builder.py <manifest_path> <output_path>
    manifest = sys.argv[1] if len(sys.argv) > 1 else \
        "./data/processed/Batch1_EP23/manifest_reviewed.json"
    output = sys.argv[2] if len(sys.argv) > 2 else "./data/processed/dataset"
    build_dataset(manifest_path=manifest, output_path=output)