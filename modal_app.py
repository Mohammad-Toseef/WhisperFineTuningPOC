"""
Main Modal app — handles training, evaluation, and batch inference.
Run with:
  modal run modal_app.py::train
  modal run modal_app.py::evaluate  
  modal run modal_app.py::transcribe_batch
"""
import modal
from pathlib import Path

# ── Modal App Definition ────────────────────────────────────────────
app = modal.App("whisper-urdu-poc")

# Persistent volume — survives between runs
volume = modal.Volume.from_name("whisper-training-vol", create_if_missing=True)
VOLUME_PATH = "/data"

# Model save locations on the volume (shared by train / evaluate / transcribe)
ADAPTER_PATH = f"{VOLUME_PATH}/model/whisper-urdu-lora-adapter"
FINAL_MODEL_PATH = f"{VOLUME_PATH}/model/whisper-urdu-final"

# Container image with all ML dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install([
        # Pinned upper bounds — unbounded >=X pulls transformers 5.x / torch 2.12
        # / numpy 2.4, which break Seq2SeqTrainer's evaluation_strategy/tokenizer
        # API and the forced_decoder_ids generate path.
        "transformers>=4.40.0,<4.46",
        "datasets>=2.18.0,<3.0.0",  # >=3.0 requires torchcodec for Audio decoding
        "evaluate>=0.4.0,<0.5",
        "jiwer>=3.0.3,<4.0",
        "torch>=2.2.0,<2.5",
        "torchaudio>=2.2.0,<2.5",
        "accelerate>=0.28.0,<1.0",
        "peft>=0.10.0,<0.14",       # LoRA (Path B)
        "numpy<2.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.1",
        "tensorboard>=2.16.0",
        "pyyaml>=6.0",
    ])
)

# ── Training Function ────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",          # 24GB VRAM — perfect for Whisper medium
    timeout=60 * 60 * 8, # 8 hour max (10hrs audio ≈ 3-4hrs training)
    volumes={VOLUME_PATH: volume},
    memory=32768,         # 32GB RAM
)
def train():
    """Fine-tune Whisper medium on the prepared dataset."""
    import yaml
    import torch
    import numpy as np
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    from datasets import load_from_disk, Audio
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from peft import LoraConfig, get_peft_model
    import evaluate

    # Load config
    with open(f"{VOLUME_PATH}/config/training_config.yaml") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    language = cfg["model"]["language"]
    task = cfg["model"]["task"]
    t_cfg = cfg["training"]
    l_cfg = cfg["lora"]

    print(f"🚀 Loading {model_name}...")
    processor = WhisperProcessor.from_pretrained(
        model_name, language=language, task=task
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Enable gradient checkpointing to reduce VRAM usage
    model.config.use_cache = False

    # ── LoRA (Path B) ──────────────────────────────────────────────
    print("🔧 Wrapping model with LoRA adapters...")
    lora_config = LoraConfig(
        r=l_cfg["r"],
        lora_alpha=l_cfg["lora_alpha"],
        target_modules=l_cfg["target_modules"],
        lora_dropout=l_cfg["lora_dropout"],
        task_type=l_cfg.get("task_type"),  # None for Whisper (see config comment)
    )
    model = get_peft_model(model, lora_config)
    # Required with gradient checkpointing + frozen base weights, otherwise
    # the backward pass has no grad_fn to reach the LoRA adapters.
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ── Data Collator ──────────────────────────────────────────────
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
            input_features = [
                {"input_features": f["input_features"]} for f in features
            ]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    # ── Load Dataset ───────────────────────────────────────────────
    print("📂 Loading dataset from volume...")
    dataset = load_from_disk(cfg["data"]["dataset_path"])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    print("⚙️  Preprocessing audio features...")
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )

    # ── Metrics ────────────────────────────────────────────────────
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # ── Training Arguments ─────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=t_cfg["output_dir"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        warmup_steps=t_cfg["warmup_steps"],
        max_steps=t_cfg["max_steps"],
        gradient_checkpointing=t_cfg["gradient_checkpointing"],
        fp16=t_cfg["fp16"],
        evaluation_strategy=t_cfg["evaluation_strategy"],
        eval_steps=t_cfg["eval_steps"],
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        metric_for_best_model=t_cfg["metric_for_best_model"],
        greater_is_better=t_cfg["greater_is_better"],
        predict_with_generate=t_cfg["predict_with_generate"],
        generation_max_length=t_cfg["generation_max_length"],
        logging_steps=t_cfg["logging_steps"],
        report_to=t_cfg["report_to"],
        logging_dir=t_cfg["logging_dir"],
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("🏋️  Starting training...")
    trainer.train()

    print("💾 Saving LoRA adapter to volume...")
    model.save_pretrained(ADAPTER_PATH)
    processor.save_pretrained(ADAPTER_PATH)

    print("🔀 Merging adapter into base model for production format...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(FINAL_MODEL_PATH)
    processor.save_pretrained(FINAL_MODEL_PATH)
    volume.commit()

    print(f"✅ Training complete. Model saved to {FINAL_MODEL_PATH}")


# ── Evaluation (baseline vs fine-tuned WER) ──────────────────────────
@app.function(
    image=image,
    gpu="A10G",           # large-v3 generate; A10G comfortably fits it
    timeout=60 * 60 * 2,
    volumes={VOLUME_PATH: volume},
    memory=32768,
)
def evaluate(which: str = "both"):
    """
    Compute WER on the held-out eval split.

    which = "base"      → frozen base model only (the baseline)
            "finetuned" → fine-tuned model only
            "both"      → run both over the SAME clips and print side by side

    Run:
      modal run modal_app.py::evaluate                 # both
      modal run modal_app.py::evaluate --which base    # baseline only
    """
    import re
    import json
    import yaml
    import torch
    from datasets import load_from_disk, Audio
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import evaluate as hf_evaluate

    with open(f"{VOLUME_PATH}/config/training_config.yaml") as f:
        cfg = yaml.safe_load(f)
    base_name = cfg["model"]["name"]
    language = cfg["model"]["language"]
    task = cfg["model"]["task"]

    # ── Load the SAME held-out eval split used in training ──────────
    dataset = load_from_disk(cfg["data"]["dataset_path"])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    eval_ds = dataset["eval"]
    references = [s for s in eval_ds["sentence"]]
    print(f"📏 Evaluating on {len(eval_ds)} held-out clips.")

    wer_metric = hf_evaluate.load("wer")

    # WER text normalizer — applied EQUALLY to base & fine-tuned so the
    # comparison is fair. Strips punctuation (Urdu + Latin) and collapses
    # whitespace; keeps diacritics (they are part of the target labels).
    _punct = r"[۔،؛؟!?.,:;\"'“”‘’()\-—…]"
    def normalize(text: str) -> str:
        text = re.sub(_punct, " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def run_model(model_path: str, label: str) -> dict:
        print(f"\n🔎 Loading {label}: {model_path}")
        processor = WhisperProcessor.from_pretrained(
            model_path, language=language, task=task
        )
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        model = model.to("cuda").eval()
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task=task
        )

        preds = []
        batch_size = 8
        for i in range(0, len(eval_ds), batch_size):
            batch = eval_ds[i : i + batch_size]
            arrays = [a["array"] for a in batch["audio"]]
            sr = batch["audio"][0]["sampling_rate"]
            feats = processor.feature_extractor(
                arrays, sampling_rate=sr, return_tensors="pt"
            ).input_features.to("cuda", dtype=model.dtype)
            with torch.no_grad():
                pred_ids = model.generate(
                    feats,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=225,
                )
            preds.extend(
                processor.batch_decode(pred_ids, skip_special_tokens=True)
            )
            print(f"   {min(i + batch_size, len(eval_ds))}/{len(eval_ds)} clips")

        raw_wer = 100 * wer_metric.compute(predictions=preds, references=references)
        norm_wer = 100 * wer_metric.compute(
            predictions=[normalize(p) for p in preds],
            references=[normalize(r) for r in references],
        )
        del model
        torch.cuda.empty_cache()
        print(f"   → {label}: raw WER {raw_wer:.2f}% | normalized WER {norm_wer:.2f}%")
        return {"label": label, "raw_wer": raw_wer, "norm_wer": norm_wer, "predictions": preds}

    import os
    results = {}
    if which in ("base", "both"):
        results["base"] = run_model(base_name, f"BASE ({base_name})")
    if which in ("finetuned", "both"):
        if not os.path.exists(FINAL_MODEL_PATH):
            print(f"⚠️  Fine-tuned model not found at {FINAL_MODEL_PATH} — run train first.")
        else:
            results["finetuned"] = run_model(FINAL_MODEL_PATH, "FINE-TUNED")

    # ── Report ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  WER RESULTS (held-out eval split)")
    print("=" * 60)
    for key in ("base", "finetuned"):
        if key in results:
            r = results[key]
            print(f"  {r['label']:<32} raw {r['raw_wer']:6.2f}%  |  norm {r['norm_wer']:6.2f}%")
    if "base" in results and "finetuned" in results:
        delta = results["base"]["norm_wer"] - results["finetuned"]["norm_wer"]
        print("-" * 60)
        print(f"  Improvement (normalized WER):   {delta:+.2f} points")
    print("=" * 60)

    # Persist results to the volume for later reference
    os.makedirs(f"{VOLUME_PATH}/logs", exist_ok=True)
    out = {
        k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
        for k, v in results.items()
    }
    out["n_clips"] = len(eval_ds)
    with open(f"{VOLUME_PATH}/logs/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    volume.commit()
    print(f"💾 Results saved to {VOLUME_PATH}/logs/eval_results.json")
    return out


# ── Batch Transcription ──────────────────────────────────────────────
@app.function(
    image=image,
    gpu="T4",             # Cheaper GPU — inference only
    timeout=60 * 60 * 6,
    volumes={VOLUME_PATH: volume},
)
def transcribe_batch(audio_paths: list[str]) -> list[dict]:
    """Transcribe a batch of audio files using the fine-tuned model."""
    import torch
    import librosa
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    processor = WhisperProcessor.from_pretrained(FINAL_MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(FINAL_MODEL_PATH)
    model = model.to("cuda")
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="ur", task="transcribe"
    )
    results = []

    for audio_path in audio_paths:
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to("cuda")

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225
            )

        transcript = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        results.append({"path": audio_path, "transcript": transcript})

    return results