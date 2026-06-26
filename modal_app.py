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

# Container image with all ML dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install([
        "transformers>=4.40.0",
        "datasets>=2.18.0,<3.0.0",  # >=3.0 requires torchcodec for Audio decoding
        "evaluate>=0.4.0",
        "jiwer>=3.0.3",
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "accelerate>=0.28.0",
        "peft>=0.10.0",       # LoRA (Path B)
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
        task_type=l_cfg["task_type"],
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
    adapter_save_path = f"{VOLUME_PATH}/model/whisper-medium-urdu-lora-adapter"
    model.save_pretrained(adapter_save_path)
    processor.save_pretrained(adapter_save_path)

    print("🔀 Merging adapter into base model for production format...")
    merged_model = model.merge_and_unload()
    model_save_path = f"{VOLUME_PATH}/model/whisper-medium-urdu-final"
    merged_model.save_pretrained(model_save_path)
    processor.save_pretrained(model_save_path)
    volume.commit()

    print(f"✅ Training complete. Model saved to {model_save_path}")


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

    model_path = f"{VOLUME_PATH}/model/whisper-medium-urdu-final"
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
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