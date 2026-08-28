# Round 3 — results

**Status:** 🟡 training in progress. Targets below were recorded **before** the run
produced a number, so the result cannot be rationalised after the fact.

---

## What round 3 has to beat

Error rates in **% WER / % CER**, punctuation *and* diacritics removed from both
sides — the reported basis since 2026-08-28, because these transcripts are for
subtitles and search where vowel marks carry nothing. Produced by
`scripts/rescore.py` from saved per-clip predictions.

| subset | n | base | round 1 | **round 2 (the bar)** |
|---|---|---|---|---|
| **Set B** — new corpus | 474 | 11.70 / 8.15 | 6.49 / 3.24 | **5.67 / 2.52** |
| **Set A** — must not regress | 246 | 15.62 / 10.68 | 8.65 / 4.31 | **8.31 / 3.96** |
| b3 / nastaliq_only ★ primary | 337 | 9.93 / 5.35 | 6.21 / 2.83 | **5.72 / 2.45** |
| b3 / code_switch ★ secondary | 137 | 15.66 / 14.20 | 7.13 / 4.13 | **5.56 / 2.69** |
| b3 / spiritual_term | 121 | 13.24 / 7.86 | 6.81 / 3.24 | **6.13 / 2.59** |
| r1 / nastaliq_only | 194 | 14.92 / 9.61 | 7.89 / 3.60 | **7.85 / 3.57** |
| r1 / code_switch | 52 | 18.16 / 14.48 | 11.40 / 6.80 | **9.98 / 5.36** |
| r1 / spiritual_term | 129 | 17.28 / 12.22 | 7.48 / 3.31 | **7.28 / 3.21** |

Legacy basis (punctuation only) is kept for continuity with round 1's published
**10.50%** on Set A; `rescore.py` prints both.

## The question this round answers

Not "is the WER lower" but **"did the mis-hearing category shrink?"** Round 2's
remaining errors on Set B, by what would fix them:

| category | round 2 | round 3 | target |
|---|---|---|---|
| encoder — mis-hearing | **544 (44.2%)** | _tbd_ | **↓ substantially** |
| ambiguous — word forms | 594 (47.9%) | _tbd_ | ↓ or flat |
| convention — Arabic/Urdu forms | 66 (5.3%) | _tbd_ | flat (a data question) |
| decoder — real spelling | 36 (2.9%) | _tbd_ | flat (nearly solved) |
| *(diacritic, not scored)* | *369* | _tbd_ | — |
| **total scored** | **1,240** | _tbd_ | ↓ |

A WER gain with a *flat* mis-hearing count would mean the encoder helped for some
other reason and the diagnosis was wrong. That is worth knowing either way.

## Success criteria, set in advance

1. ★ **`b3/nastaliq_only` CER ≤ 2.45** — the primary goal, must not go backwards.
2. ★ **Set A CER ≤ 3.96** — no forgetting. This is the one to watch: 41% more of
   the model is moving, at double the decoder's rate.
3. ★ **Mis-hearing count on Set B falls materially below 544** — the round's
   actual hypothesis.
4. `weights moved: 1024/1024` at the end of training, versus round 2's 640/1024.

## Did the encoder actually train *enough*?

Separate from whether it helped. Run after the adapter lands:

```
python scripts/inspect_adapter.py <r3-adapter> --compare <r1-adapter-backup>
```

**This is the false-negative guard.** If the encoder's weights barely moved
relative to the decoder's, then 1e-5 was too low for three epochs and the correct
conclusion is *"under-trained"*, not *"the encoder does not help"*. Those two
readings imply completely different next rounds, and only the weights can tell
them apart.

---

## Live log

| when | what |
|---|---|
| launch | `✅ 57,671,680 trainable (3.60%)` · `resume confirmed: 320/512` — 320 is correct, round 1's encoder `lora_B` is still zero |
| launch | `two LR groups — decoder 640 @ 5.0e-06 \| encoder 384 @ 1.0e-05` |
| step 254 | epoch 1 eval — see below |

### Epoch 1 (step 254) — no forgetting, ahead of round 2

In-training **RAW** metrics from `compute_metrics`. Not comparable to the
normalized figures in the table above; compare only against round 2's in-training
numbers, and Set A against round 1's raw curve.

| | r2 @ ep1 | **r3 @ ep1** | Δ |
|---|---|---|---|
| blended WER | 12.271 | **12.081** | −0.19 |
| blended CER | 5.129 | **4.977** | −0.15 |
| Set B WER | 10.469 | **10.252** | −0.22 |
| Set B CER | 4.472 | **4.213** | −0.26 |
| **Set A WER** | 15.799 | **15.664** | −0.14 |
| Set A CER | 6.401 | 6.458 | **+0.06** |
| eval_loss | 0.1978 | **0.1923** | −0.006 |

★ **The forgetting risk did not materialise.** Set A WER **15.664** is below round
1's own best of **15.71** and below round 2's 15.799 at the same point — despite
41% more of the model moving at twice the decoder's rate. That was the obvious way
this round could go wrong.

⚠️ Set A **CER** is the single metric that worsened (+0.06, +0.9% rel). Small;
watch it at epochs 2 and 3 rather than reading it now.

Round 3 at epoch 1 reaches what round 2 reached at **epoch 2** on blended WER.
Suggestive, not conclusive — the round's actual hypothesis is about the
mis-hearing category, which only the post-run triage can answer.

**Not a bug:** round 3's blended WER (12.0810098658006) is identical to 13 decimal
places to round 2's *epoch 2*. WER is an integer error count over a fixed word
count, and the two subsets moved in opposite directions relative to that run —
Set B better, Set A worse — summing to the same total. Every other metric differs.

**Eval cost is up:** 1,926 s vs round 2's 1,875 s per pass, so ~32 min each.
