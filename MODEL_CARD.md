---
license: apache-2.0
language:
- ur
- ar
- en
base_model: openai/whisper-large-v3
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- whisper
- automatic-speech-recognition
- urdu
- code-switching
- lora
---

# Whisper Large-v3 — Urdu / Arabic / English (code-switching, single speaker)

Fine-tuned [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) for
accurate transcription of a single Urdu/English/Arabic **code-switching** speaker (religious /
spiritual lectures). Trained with **LoRA (Path B)** on ~10.4 hours of manually reviewed audio,
then merged back into the base weights — so this is a **standard, standalone Whisper model**
(no PEFT/adapter loading required).

## Results (held-out eval, 246 clips, normalized WER — lower is better)

| Bucket | n | Base large-v3 | This model | Δ |
|---|---|---|---|---|
| **Overall** | 246 | 18.57% | **10.50%** | **−8.07** |
| Nastaliq only | 194 | 18.05% | 9.89% | −8.16 |
| Code-switch (has English) | 52 | 20.46% | 12.70% | −7.76 |
| Spiritual/domain terms | 129 | 20.29% | **9.38%** | **−10.91** |

The largest gain is on **spiritual/domain terminology** — the base model's weakest area became
the strongest after fine-tuning. Qualitatively the model also **reduces base-model long-form
repetition loops** and applies the transcription convention below consistently.

## Transcription convention (three-script)

The model is trained to keep each word type in its native script:

| Word type | Script | Example |
|---|---|---|
| Urdu | Nastaliq | کرنا، ہے، بات، آج |
| Arabic (recitation: Quran, hadith, durood) | Arabic, fully diacritized | صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ |
| English | Latin (kept as-is by the frozen base) | account, genocide, challenge |

Izafat (کسرہ, ِ ) constructions are preserved (e.g. ذکرِ قلب، عاشقِ رسول).

## Usage

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

repo = "mohammad-toseef059/whisper-large-v3-urdu"
processor = WhisperProcessor.from_pretrained(repo)
model = WhisperForConditionalGeneration.from_pretrained(repo).to("cuda").eval()

# 16 kHz mono audio -> features
features = processor.feature_extractor(audio_array, sampling_rate=16000,
                                       return_tensors="pt").input_features.to("cuda", torch.float16)
ids = model.generate(features, language="ur", task="transcribe", max_new_tokens=225)
text = processor.batch_decode(ids, skip_special_tokens=True)[0]
```

For long files, use the HF `pipeline` with `chunk_length_s=28, stride_length_s=(4, 2)` and
`generate_kwargs={"language": "ur", "task": "transcribe"}`.

## Training

- **Base:** openai/whisper-large-v3 (1.5B params)
- **Method:** LoRA (r=32, α=64, dropout=0.05), targets `q/k/v/out_proj + fc1/fc2`, then merged
- **Data:** 2,011 train clips (~10.4 hrs) / 246 eval clips (whole-episode holdout, no leakage)
- **Schedule:** 567 steps (~9 epochs), effective batch 32, lr 1e-5, fp16, gradient checkpointing
- **Language token:** `<|ur|>`; best checkpoint by eval WER (early convergence ~epoch 6)

## Intended use & limitations

- **Intended:** batch transcription of this speaker's Urdu/Arabic/English code-switching lectures.
- **Single-speaker domain model** — accuracy on other speakers, accents, or non-religious
  domains is not characterized and will be lower.
- **Timestamps:** raw Whisper timestamps degrade slightly after fine-tuning. For precise
  word/segment timing, pair with forced alignment (e.g. WhisperX / wav2vec2).
- **Known quirk:** occasionally over-diacritizes or garbles short Arabic fragments.
- Derived from Whisper large-v3 (Apache-2.0). Training audio is proprietary and not included.