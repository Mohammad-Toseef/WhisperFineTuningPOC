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

# ── Output path versioning (`training.run_tag`) ─────────────────────
# Round 1 wrote to the UNTAGGED paths below, and its adapter is round 2's resume
# SOURCE, so round 2 must not write there: without a tag the trainer would
# overwrite the adapter it is resuming from, partway through the run, and
# evaluate() would overwrite the eval_results.json holding round 1's 10.50%.
#
# One knob moves every artifact together. Separate per-path config keys were the
# alternative and were rejected: forgetting one of them is silent, and the one
# most likely to be forgotten (the dataset) does not clobber anything — it just
# trains round 2 on round 1's data and looks like a successful run.
#
# Unset (`""`) reproduces round 1's layout exactly.
def _tag(run_tag: str) -> str:
    return f"-{run_tag}" if run_tag else ""


def adapter_path(run_tag: str = "") -> str:
    return f"{VOLUME_PATH}/model/whisper-urdu{_tag(run_tag)}-lora-adapter"


def final_model_path(run_tag: str = "") -> str:
    return f"{VOLUME_PATH}/model/whisper-urdu{_tag(run_tag)}-final"


def eval_results_path(run_tag: str = "", label: str = "") -> str:
    """Where evaluate() writes its results.

    `label` identifies WHICH models were scored. Round 2 runs evaluate three
    times over one dataset (round 1, base, round 2) and every run would otherwise
    write the same filename, so runs 1 and 2 would be silently clobbered by run
    3 — losing the comparison baselines the round is judged against. Derived
    automatically rather than left to the operator to remember.
    """
    name = f"eval_results{_tag(run_tag)}{_tag(label)}"
    return f"{VOLUME_PATH}/logs/{name}.json"


def eval_predictions_path(run_tag: str = "", label: str = "") -> str:
    """Per-clip predictions, written BESIDE the scores rather than inside them.

    A separate file on purpose: the scores file is small enough to read by eye or
    grep, and burying ~600 transcripts in it would end that. Keeping them at all
    means later questions — rescoring under a different normalizer, finding which
    clips got worse, reading B3039's English (success criterion 4) — cost nothing
    instead of another GPU pass.
    """
    name = f"eval_predictions{_tag(run_tag)}{_tag(label)}"
    return f"{VOLUME_PATH}/logs/{name}.json"


# Round-1 (untagged) locations. Still the defaults for evaluate/transcribe, and
# the resume source for round 2.
ADAPTER_PATH = adapter_path()
FINAL_MODEL_PATH = final_model_path()

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
    # 12 h. Round 2 estimates ~6 h (4.4 h training at 3x258 steps, + 3 generate-eval
    # passes over 587 clips, + feature extraction on 8.8k clips = 3.9x round 1's,
    # + merging and writing a 6.17 GB model) — a ~30% margin on a figure INFERRED
    # from round 1's throughput, not measured for this run.
    # Raised because the overrun is total, not partial: the adapter, the merged
    # model and the only volume.commit() all happen AFTER training, so a kill at
    # 7h50m leaves no deliverable and nothing was committed during training either.
    # Modal bills time used, not the ceiling, so the margin is free.
    timeout=60 * 60 * 12,
    volumes={VOLUME_PATH: volume},
    memory=32768,         # 32GB RAM
)
def train():
    """Fine-tune Whisper large-v3 on the prepared dataset.

    Two init modes, chosen by `lora.resume_from_adapter` in training_config.yaml:
      unset  → FRESH LoRA adapter on the base model (round 1's behaviour)
      set    → CONTINUE training that existing adapter (round 2)
    """
    import os
    import json
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
    from peft import LoraConfig, PeftModel, get_peft_model
    import evaluate

    # Load config
    with open(f"{VOLUME_PATH}/config/training_config.yaml") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    language = cfg["model"]["language"]
    task = cfg["model"]["task"]
    t_cfg = cfg["training"]
    l_cfg = cfg["lora"]

    # ── Resolve where THIS run writes (see run_tag notes at module level) ──
    run_tag = str(t_cfg.get("run_tag") or "")
    out_adapter = adapter_path(run_tag)
    out_final = final_model_path(run_tag)

    # Printed up front, because "which paths did that run actually use" is not
    # answerable from a finished log otherwise, and every silent failure mode
    # here is a wrong path rather than a crash.
    print("📦 Run layout")
    print(f"   run_tag   : {run_tag or '(none — round-1 layout)'}")
    print(f"   dataset  <- {cfg['data']['dataset_path']}")
    print(f"   resume   <- {l_cfg.get('resume_from_adapter') or '(none — fresh adapter)'}")
    print(f"   adapter  -> {out_adapter}")
    print(f"   merged   -> {out_final}")
    print(f"   ckpts    -> {t_cfg['output_dir']}")
    print(f"   tb logs  -> {t_cfg['logging_dir']}")

    # The dataset is the one path a run_tag cannot protect, because reading the
    # wrong one destroys nothing and produces a plausible result. Flag the
    # specific combination that means "tagged run pointed at round 1's data".
    if run_tag and cfg["data"]["dataset_path"].rstrip("/").endswith("/processed/dataset"):
        print(f"⚠️  run_tag={run_tag} but dataset_path is the UNTAGGED "
              f"{cfg['data']['dataset_path']} — is this meant to train on round 1's dataset?")

    print(f"🚀 Loading {model_name}...")
    processor = WhisperProcessor.from_pretrained(
        model_name, language=language, task=task
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Enable gradient checkpointing to reduce VRAM usage
    model.config.use_cache = False

    # ── LoRA (Path B): FRESH adapter, or RESUME an existing one ────
    # `get_peft_model` CREATES an adapter — it never continues one. LoRA inits
    # lora_B to zeros, so a fresh adapter contributes exactly nothing at step 0
    # and the run is training the plain base model. Continuing round 1 therefore
    # requires loading its weights explicitly; nothing else in this file reads
    # ADAPTER_PATH (it is only ever written, at the end of training).
    resume_from = l_cfg.get("resume_from_adapter")

    if resume_from:
        # base large-v3 + round-1 adapter IS the round-1 model: the base weights
        # were frozen throughout round 1, so they are the substrate, not the
        # thing trained. Step 0 here reproduces round 1's outputs exactly.
        if not os.path.isdir(resume_from):
            raise FileNotFoundError(
                f"lora.resume_from_adapter = {resume_from!r} is not a directory. "
                "Refusing to fall through to a fresh adapter: that would silently "
                "train the BASE model instead of continuing round 1, and would look "
                "like a successful run."
            )
        # Resuming from the path this run saves to would destroy the source
        # adapter partway through. Compare against the RUN-SCOPED output, so
        # setting `training.run_tag` is what clears this.
        if os.path.realpath(resume_from) == os.path.realpath(out_adapter):
            raise ValueError(
                f"resume_from_adapter is the SAME path this run saves to ({out_adapter}). "
                "That would overwrite the adapter being resumed from. Set "
                "`training.run_tag` (e.g. 'r2') so this run writes to its own paths."
            )

        print(f"🔁 RESUMING LoRA adapter from {resume_from}")
        # is_trainable=True is REQUIRED. A saved adapter_config.json carries
        # "inference_mode": true, and without this flag PEFT loads the adapter
        # FROZEN — training then runs to completion, logs a loss curve, saves a
        # model, and changes nothing at all.
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)

        # The adapter's own config wins for r / alpha / target_modules, so a
        # mismatch against training_config.yaml means the yaml is describing a
        # different adapter than the one loaded. Surfaced because the cheapest
        # version of this mistake — resuming the smoke test's q/v-only adapter
        # while the yaml claims 6 Tier-2 targets — is otherwise invisible.
        active = next(iter(model.peft_config.values()))
        drift = []
        if active.r != l_cfg["r"]:
            drift.append(f"r: adapter={active.r} yaml={l_cfg['r']}")
        if active.lora_alpha != l_cfg["lora_alpha"]:
            drift.append(f"lora_alpha: adapter={active.lora_alpha} yaml={l_cfg['lora_alpha']}")
        if set(active.target_modules or []) != set(l_cfg["target_modules"]):
            drift.append(f"target_modules: adapter={sorted(active.target_modules or [])} "
                         f"yaml={sorted(l_cfg['target_modules'])}")
        if drift:
            print("⚠️  Adapter/yaml MISMATCH — the ADAPTER's values are what train:")
            for d in drift:
                print(f"      {d}")
        else:
            print(f"   adapter matches yaml: r={active.r}, alpha={active.lora_alpha}, "
                  f"{len(active.target_modules or [])} target modules")
    else:
        print("🔧 Wrapping model with a FRESH LoRA adapter...")
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

    # ── Fail LOUDLY on the two silent no-ops ───────────────────────
    # Everything that can go wrong above produces a run that looks successful:
    # a loss curve, a WER per epoch, a saved model. These two checks are the
    # only thing standing between that and hours of wasted GPU.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise RuntimeError(
            "0 trainable parameters after LoRA setup — the adapter is FROZEN, so "
            "training would run to completion and change nothing. A saved "
            'adapter_config.json carries "inference_mode": true, so a resume needs '
            "is_trainable=True."
        )
    print(f"✅ {trainable:,} trainable / {total:,} total ({100 * trainable / total:.2f}%)")

    if resume_from:
        # A freshly-initialised LoRA has lora_B == 0 in every layer (which is why
        # a fresh adapter is a no-op at step 0). Round 1's TRAINED adapter cannot
        # be all-zero, so this distinguishes "resumed round 1" from "silently
        # started from scratch" — the one failure mode a parameter count and a
        # loss curve both look identical under.
        b_tensors = [p for n, p in model.named_parameters() if "lora_B" in n]
        nonzero = sum(1 for p in b_tensors if p.detach().abs().sum().item() > 0)
        if not b_tensors:
            raise RuntimeError(
                "resumed an adapter but found no lora_B parameters — the load did not "
                "produce a LoRA model, so nothing was actually resumed."
            )
        if nonzero == 0:
            raise RuntimeError(
                f"resumed from {resume_from} but ALL {len(b_tensors)} lora_B tensors are "
                "zero, i.e. this is a freshly-initialised adapter and round 1's training "
                "was not loaded. Training would silently start from the base model."
            )
        print(f"✅ resume confirmed: {nonzero}/{len(b_tensors)} lora_B tensors carry "
              f"trained (non-zero) weights")

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

    # ── Per-source eval indices (the forgetting tripwire) ──────────
    # Round 2's eval split spans TWO corpora that measure opposite things: the
    # batch under training, and round 1's retained episodes. A single blended WER
    # can report a wash while the run trades old competence for new — improve the
    # larger half by 4 points, lose 4 on the smaller, and the average barely
    # moves. So the split is reported per source EVERY epoch, which is also the
    # earliest point the learning rate can be judged.
    #
    # Must be built BEFORE .map() below, which drops `sentence`.
    SOURCE_METRIC = {"primary": "b3", "eval_only": "r1"}
    eval_src_idx: dict[str, list[int]] = {}
    _sidecar = os.path.join(cfg["data"]["dataset_path"], "eval_buckets.json")
    if os.path.exists(_sidecar):
        with open(_sidecar, encoding="utf-8") as f:
            _bmeta = json.load(f)
        _eval_sentences = dataset["eval"]["sentence"]
        if len(_bmeta) != len(_eval_sentences):
            print(f"⚠️  eval_buckets.json has {len(_bmeta)} rows != {len(_eval_sentences)} "
                  "eval clips — per-source WER DISABLED (rebuild the dataset).")
        elif any(_bmeta[i].get("sentence") != _eval_sentences[i]
                 for i in range(len(_eval_sentences))):
            print("⚠️  eval_buckets.json is misaligned with the eval split — "
                  "per-source WER DISABLED.")
        else:
            for i, m in enumerate(_bmeta):
                # Fall back to the label prefix for datasets built before `source`
                # existed (round 1's), so this degrades rather than crashing.
                src = m.get("source") or (
                    "eval_only" if str(m.get("episode", "")).startswith("EP") else "primary"
                )
                key = SOURCE_METRIC.get(src, src)
                eval_src_idx.setdefault(key, []).append(i)
            print("   eval sources: " + ", ".join(
                f"wer_{k}={len(v)} clips" for k, v in sorted(eval_src_idx.items())))
            if len(eval_src_idx) < 2:
                print("   (single-corpus eval split — wer_* will mirror the blended wer)")
    else:
        print("⚠️  No eval_buckets.json — per-source WER DISABLED, so a Set A "
              "regression would be invisible until after the run.")

    print("⚙️  Preprocessing audio features...")
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )

    # ── Metrics ────────────────────────────────────────────────────
    wer_metric = evaluate.load("wer")
    # CER alongside WER because the PRIMARY goal of this round is Urdu spelling
    # accuracy, and WER is binary per word: a word missing one diacritic scores
    # exactly the same as a completely wrong word. CER counts characters, so it
    # shows the magnitude and can tell "nearly right" from "wrong".
    #
    # Loaded defensively on purpose. This is a reporting metric, and the failure
    # it could cause — an exception at the first eval, ~90 minutes into a ~6 h run
    # — costs far more than the number is worth. Same reasoning as the Binarize
    # catch in modal_align.py.
    try:
        cer_metric = evaluate.load("cer")
    except Exception as exc:
        cer_metric = None
        print(f"⚠️  CER metric unavailable ({type(exc).__name__}: {exc}) — "
              "training continues, reporting WER only.")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        def score(preds, refs) -> dict:
            out = {"wer": 100 * wer_metric.compute(predictions=preds, references=refs)}
            if cer_metric is not None:
                out["cer"] = 100 * cer_metric.compute(predictions=preds, references=refs)
            return out

        metrics = score(pred_str, label_str)
        # Per-corpus scores alongside the blend. `metric_for_best_model` stays on
        # the blended `wer` on purpose — a checkpoint should be good at BOTH —
        # these only make the blend's composition visible. Costs no extra GPU:
        # the predictions already exist, this slices them by index.
        for key, idxs in eval_src_idx.items():
            if not idxs:
                continue
            for name, val in score([pred_str[i] for i in idxs],
                                   [label_str[i] for i in idxs]).items():
                metrics[f"{name}_{key}"] = val
        return metrics

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

    print(f"💾 Saving LoRA adapter to {out_adapter} ...")
    model.save_pretrained(out_adapter)
    processor.save_pretrained(out_adapter)
    # Commit the ADAPTER before attempting the merge. The adapter is the only
    # irreplaceable output — the merged model is derivable from it, but nothing
    # regenerates the adapter except another full training run. Committing once at
    # the very end meant a failure in merge_and_unload() or in writing the 6.2 GB
    # merged model would discard hours of training that had already succeeded.
    volume.commit()
    print(f"   ✅ adapter committed — training is now safe even if the merge fails")

    print("🔀 Merging adapter into base model for production format...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(out_final)
    processor.save_pretrained(out_final)
    volume.commit()

    print(f"✅ Training complete. Model saved to {out_final}")


# ── Evaluation (baseline vs fine-tuned WER) ──────────────────────────
@app.function(
    image=image,
    gpu="A10G",           # large-v3 generate; A10G comfortably fits it
    timeout=60 * 60 * 2,
    volumes={VOLUME_PATH: volume},
    memory=32768,
)
def evaluate(which: str = "both", dataset_path: str = "", model_path: str = ""):
    """
    Compute WER on a held-out eval split.

    which = "base"      → frozen base model only (the baseline)
            "finetuned" → fine-tuned model only
            "both"      → run both over the SAME clips and print side by side

    dataset_path  eval set to score. Defaults to config's data.dataset_path.
                  Pass it explicitly to score a model against a set the current
                  config does not name — e.g. reproducing round 1's number on the
                  ORIGINAL /processed/dataset artifact, so a mismatch cannot be
                  blamed on a rebuilt copy. Also makes a finished run
                  self-describing instead of only interpretable alongside
                  whatever the config said at the time.
    model_path    model for the "finetuned" slot. Defaults to FINAL_MODEL_PATH
                  (round 1's). REQUIRED to score round 2, whose merged model is
                  at a run_tag-suffixed path this function would otherwise never
                  look at.

    Run:
      modal run modal_app.py::evaluate                          # both, config's dataset
      modal run modal_app.py::evaluate --which base             # baseline only
      modal run modal_app.py::evaluate --dataset-path /data/processed/dataset_r2 \
          --model-path /data/model/whisper-urdu-r2-final --which finetuned
    """
    import os
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

    # ── Load the held-out eval split ────────────────────────────────
    ds_path = dataset_path or cfg["data"]["dataset_path"]
    ft_path = model_path or FINAL_MODEL_PATH
    dataset = load_from_disk(ds_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    eval_ds = dataset["eval"]
    references = [s for s in eval_ds["sentence"]]
    print(f"📏 Evaluating on {len(eval_ds)} held-out clips.")
    print(f"   dataset : {ds_path}"
          f"{'' if dataset_path else '   (from config)'}")
    if which in ("finetuned", "both"):
        print(f"   model   : {ft_path}"
              f"{'' if model_path else '   (default — round 1)'}")

    # ── Eval subsets (from dataset_builder's sidecar) ───────────────
    # buckets  — code_switch / spiritual_term, where fine-tuning's benefit
    #            concentrates but aggregate WER hides it.
    # sources  — WHICH CORPUS a clip came from. Round 2's split holds two, and
    #            a bucket computed ACROSS them is meaningless: a `code_switch`
    #            number blending round 1's 52 clips with Batch 3's 132 is not a
    #            comparison, it is an average of two different questions. So
    #            source x bucket is reported too, and that is what criterion 3
    #            (code_switch on Set B, r1 vs r2) is actually read from.
    BUCKETS = ["nastaliq_only", "code_switch", "spiritual_term"]
    SOURCE_LABEL = {"primary": "b3", "eval_only": "r1"}
    bucket_indices: dict[str, list[int]] = {}
    source_indices: dict[str, list[int]] = {}
    source_bucket_indices: dict[str, list[int]] = {}
    # Initialised empty so the predictions sidecar below can be written even when
    # eval_buckets.json is missing or misaligned — it just loses the per-clip
    # episode/source/bucket labels rather than raising NameError at the very end,
    # after the GPU work is already paid for.
    bmeta: list[dict] = []
    sidecar = os.path.join(ds_path, "eval_buckets.json")
    if os.path.exists(sidecar):
        with open(sidecar, encoding="utf-8") as f:
            bmeta = json.load(f)
        if len(bmeta) != len(references):
            print(f"⚠️  eval_buckets.json has {len(bmeta)} rows != {len(references)} eval "
                  "clips — skipping per-bucket WER (rebuild the dataset).")
            bmeta = []
        elif any(bmeta[i].get("sentence") != references[i] for i in range(len(references))):
            print("⚠️  eval_buckets.json is misaligned with the eval split — skipping per-bucket WER.")
            # MUST discard it. The predictions sidecar decides whether to attach
            # episode/source labels by comparing LENGTHS, and a sidecar that is the
            # right length but the wrong content passes that test — so every row
            # would get a confident, wrong label. Wrong labels are worse than none:
            # they would have someone reading another episode's output believing it
            # was B3039's.
            bmeta = []
        else:
            for b in BUCKETS:
                bucket_indices[b] = [i for i, m in enumerate(bmeta) if b in m["buckets"]]
            print("   buckets: " + ", ".join(f"{b}={len(bucket_indices[b])}" for b in BUCKETS))

            for i, m in enumerate(bmeta):
                # Prefix fallback for datasets built before `source` existed.
                raw = m.get("source") or (
                    "eval_only" if str(m.get("episode", "")).startswith("EP") else "primary"
                )
                src = SOURCE_LABEL.get(raw, raw)
                source_indices.setdefault(src, []).append(i)
                for b in BUCKETS:
                    if b in m["buckets"]:
                        source_bucket_indices.setdefault(f"{src}/{b}", []).append(i)
            if len(source_indices) > 1:
                print("   sources: " + ", ".join(
                    f"{k}={len(v)}" for k, v in sorted(source_indices.items())))
            else:
                # Single corpus: the per-source number would just repeat overall.
                source_indices, source_bucket_indices = {}, {}
                print("   sources: single corpus — per-source breakdown not reported")
    else:
        print("ℹ️  No eval_buckets.json found — reporting overall WER only.")

    wer_metric = hf_evaluate.load("wer")
    # CER: the instrument for SPELLING accuracy, this round's primary goal. WER is
    # binary per word — one wrong diacritic scores the same as a wholly wrong word
    # — so it cannot show whether spelling improved, only whether words changed.
    # Degrades rather than failing: WER alone still beats losing the whole run.
    try:
        cer_metric = hf_evaluate.load("cer")
    except Exception as exc:
        cer_metric = None
        print(f"⚠️  CER metric unavailable ({type(exc).__name__}: {exc}) — WER only. "
              "Spelling quality will be much harder to read.")

    # WER text normalizer — applied EQUALLY to base & fine-tuned so the
    # comparison is fair. Strips punctuation (Urdu + Latin) and collapses
    # whitespace; keeps diacritics (they are part of the target labels).
    _punct = r"[۔،؛؟!?.,:;\"'“”‘’()\-—…]"
    def normalize(text: str) -> str:
        text = re.sub(_punct, " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # `path`, not `model_path`: the outer scope now has a `model_path` argument
    # and shadowing it here would make the two impossible to tell apart.
    def run_model(path: str, label: str) -> dict:
        print(f"\n🔎 Loading {label}: {path}")
        processor = WhisperProcessor.from_pretrained(
            path, language=language, task=task
        )
        model = WhisperForConditionalGeneration.from_pretrained(path)
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

        npreds = [normalize(p) for p in preds]
        nrefs = [normalize(r) for r in references]
        raw_wer = 100 * wer_metric.compute(predictions=preds, references=references)
        norm_wer = 100 * wer_metric.compute(predictions=npreds, references=nrefs)
        raw_cer = norm_cer = None
        if cer_metric is not None:
            raw_cer = 100 * cer_metric.compute(predictions=preds, references=references)
            norm_cer = 100 * cer_metric.compute(predictions=npreds, references=nrefs)

        # Normalized WER + CER over an arbitrary index subset.
        def subset_scores(idxs: list[int]) -> dict:
            p = [npreds[i] for i in idxs]
            r = [nrefs[i] for i in idxs]
            out = {"n": len(idxs),
                   "norm_wer": 100 * wer_metric.compute(predictions=p, references=r)}
            if cer_metric is not None:
                out["norm_cer"] = 100 * cer_metric.compute(predictions=p, references=r)
            return out

        buckets_wer = {b: subset_scores(i) for b, i in bucket_indices.items() if i}
        sources_wer = {s: subset_scores(i) for s, i in source_indices.items() if i}
        source_buckets_wer = {k: subset_scores(i) for k, i in source_bucket_indices.items() if i}

        del model
        torch.cuda.empty_cache()
        cer_txt = f" | norm CER {norm_cer:.2f}%" if norm_cer is not None else ""
        print(f"   → {label}: raw WER {raw_wer:.2f}% | normalized WER {norm_wer:.2f}%{cer_txt}")
        return {"label": label, "model_path": path,
                "raw_wer": raw_wer, "norm_wer": norm_wer,
                "raw_cer": raw_cer, "norm_cer": norm_cer,
                "buckets": buckets_wer, "sources": sources_wer,
                "source_buckets": source_buckets_wer, "predictions": preds}

    results = {}
    if which in ("base", "both"):
        results["base"] = run_model(base_name, f"BASE ({base_name})")
    if which in ("finetuned", "both"):
        if not os.path.exists(ft_path):
            # Loud, and it names the path: the likeliest cause is a run_tag'd
            # model being looked for at round 1's default location.
            print(f"⚠️  Fine-tuned model not found at {ft_path} — pass --model-path, "
                  "or run train first.")
        else:
            results["finetuned"] = run_model(ft_path, f"FINE-TUNED ({os.path.basename(ft_path)})")

    # ── Report ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  WER RESULTS (held-out eval split)")
    print("=" * 60)
    for key in ("base", "finetuned"):
        if key in results:
            r = results[key]
            cer = f"  |  CER {r['norm_cer']:6.2f}%" if r.get("norm_cer") is not None else ""
            print(f"  {r['label']:<32} raw WER {r['raw_wer']:6.2f}%  |  "
                  f"WER {r['norm_wer']:6.2f}%{cer}")
    if "base" in results and "finetuned" in results:
        print("-" * 60)
        d_wer = results["base"]["norm_wer"] - results["finetuned"]["norm_wer"]
        print(f"  Improvement, normalized WER:   {d_wer:+.2f} points")
        if results["base"].get("norm_cer") is not None and results["finetuned"].get("norm_cer") is not None:
            d_cer = results["base"]["norm_cer"] - results["finetuned"]["norm_cer"]
            print(f"  Improvement, normalized CER:   {d_cer:+.2f} points   "
                  "<- the SPELLING signal")
    print("=" * 60)

    # ── Subset tables: WER and CER side by side ────────────────────
    # One helper for all three tables. The bucket table used to have its own copy
    # of this formatting, which is how it would have quietly kept reporting WER
    # only after CER was added everywhere else.
    HDR = (f"  {'subset':<22}{'n':>6}{'WER base':>10}{'WER ft':>9}{'ΔWER':>8}"
           f"{'CER base':>11}{'CER ft':>9}{'ΔCER':>8}")

    def _row(label: str, key: str, getter: str) -> None:
        base_x = results.get("base", {}).get(getter, {}).get(key)
        ft_x = results.get("finetuned", {}).get(getter, {}).get(key)
        n = (base_x or ft_x or {}).get("n")
        if n is None:
            return
        cells = [f"  {label:<22}", f"{n:>6}"]
        for metric, widths in (("norm_wer", (10, 9, 8)), ("norm_cer", (11, 9, 8))):
            bv = base_x.get(metric) if base_x else None
            fv = ft_x.get(metric) if ft_x else None
            cells.append(f"{bv:>{widths[0]}.2f}" if bv is not None else f"{'-':>{widths[0]}}")
            cells.append(f"{fv:>{widths[1]}.2f}" if fv is not None else f"{'-':>{widths[1]}}")
            cells.append(f"{bv - fv:>+{widths[2]}.2f}" if (bv is not None and fv is not None)
                         else f"{'-':>{widths[2]}}")
        print("".join(cells))

    if bucket_indices:
        print("\n  PER-BUCKET  (nastaliq_only is the PRIMARY goal: Urdu spelling)")
        print(HDR)
        print("-" * 84)
        for b in BUCKETS:
            _row(b, b, "buckets")
        if source_indices:
            print("  ⚠️  These span BOTH corpora — see the per-source tables below. A"
                  "\n      bucket averaged across two corpora is not a comparison.")
        print("=" * 84)

    if source_indices:
        print("\n  PER-SOURCE  (never average these together)")
        print(HDR)
        print("-" * 84)
        for s in sorted(source_indices):
            _row(f"{s}  ({'round 1' if s == 'r1' else 'this batch'})", s, "sources")
        print("=" * 84)

        print("\n  SOURCE x BUCKET — the table the success criteria are read from")
        print("    b3/nastaliq_only  = criterion 1, Urdu improves (PRIMARY)")
        print("    r1/*              = criteria 2-3, nothing regressed")
        print("    b3/code_switch    = criterion 4, code-switching improves")
        print(HDR)
        print("-" * 84)
        for s in sorted(source_indices):
            for b in BUCKETS:
                _row(f"{s} / {b}", f"{s}/{b}", "source_buckets")
        print("=" * 84)

    # Persist results to the volume for later reference
    os.makedirs(f"{VOLUME_PATH}/logs", exist_ok=True)
    out = {
        k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
        for k, v in results.items()
    }
    out["n_clips"] = len(eval_ds)
    # Record WHAT was scored, not just the scores. Both inputs are now
    # overridable, so a bare number is not reproducible six weeks later —
    # the same file would otherwise be ambiguous between Set A, Set B and both.
    out["dataset_path"] = ds_path
    out["which"] = which
    out["source_clip_counts"] = {s: len(i) for s, i in source_indices.items()}
    # run_tag-scoped: round 1's eval_results.json holds the 18.57% / 10.50%
    # baseline this project is measured against, and an untagged round-2 eval
    # would overwrite it. It is also the one artifact here that cannot be
    # regenerated cheaply — it needs a GPU pass and the model that produced it.
    # The label additionally separates the three round-2 eval runs from each
    # other; without it they all write one filename and only the last survives.
    if which == "base":
        label = "base"
    elif which == "finetuned":
        label = os.path.basename(ft_path.rstrip("/"))
    else:
        label = f"base-vs-{os.path.basename(ft_path.rstrip('/'))}"
    results_file = eval_results_path(str(cfg["training"].get("run_tag") or ""), label)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ── Per-clip predictions sidecar ───────────────────────────────
    # One row per clip with every model's output side by side, so base/r1/r2 can
    # be diffed directly and criterion 4 (does B3039 come out in English?) can be
    # read straight from a file instead of costing another transcription run.
    aligned = len(bmeta) == len(references)
    clips = []
    for i, ref in enumerate(references):
        row = {"i": i, "reference": ref}
        if aligned:
            m = bmeta[i]
            row["episode"] = m.get("episode")
            row["source"] = m.get("source")
            row["buckets"] = m.get("buckets")
        for key in ("base", "finetuned"):
            if key in results:
                row[key] = results[key]["predictions"][i]
        clips.append(row)
    predictions_file = eval_predictions_path(
        str(cfg["training"].get("run_tag") or ""), label)
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump({"dataset_path": ds_path, "which": which,
                   "models": {k: results[k]["model_path"] for k in results},
                   "labels_aligned": aligned, "clips": clips},
                  f, ensure_ascii=False, indent=2)

    volume.commit()
    print(f"💾 Results saved to {results_file}")
    print(f"💾 Per-clip predictions saved to {predictions_file}"
          f"{'' if aligned else '   (WITHOUT episode/source labels — sidecar misaligned)'}")
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