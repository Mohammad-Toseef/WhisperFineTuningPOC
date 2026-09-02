r"""Report what a LoRA adapter ACTUALLY learned, per module and per half.

WHY THIS EXISTS
---------------
Round 1 finished healthy by every signal available at the time -- loss fell, WER
went 23.87% -> 15.71% raw, the merged model shipped to the Hub -- and 41% of its
adapter had never received a single gradient. Every encoder `lora_B` was still
exactly zero and every encoder `lora_A` was still at its random init.

Nothing in the training path could see this. A parameter count, a loss curve and a
WER all look identical whether the encoder is learning or inert, because `lora_B`
initialises to zeros: an untrained LoRA module contributes `B @ A = 0 @ A = 0`,
which is exactly what a *correctly* initialised one contributes at step 0.

Cause (confirmed, not inferred): `enable_input_require_grads()` hooks
`get_input_embeddings()`, which for Whisper is the DECODER's `embed_tokens`. The
encoder's input is a mel spectrogram through frozen convs -- data, so
`requires_grad=False`. With `gradient_checkpointing=True` the REENTRANT checkpoint
path decides whether to build a backward graph from the *inputs* of the
checkpointed block, and parameters inside are invisible to that decision. So the
encoder's LoRA weights get `grad=None`, silently. Backward does not raise, because
the decoder half supplies a valid graph.

The one-line fix, for whenever the encoder is meant to train:

    Seq2SeqTrainingArguments(..., gradient_checkpointing_kwargs={"use_reentrant": False})

RUN IT AFTER EVERY TRAINING ROUND:

    python scripts/inspect_adapter.py <adapter_dir>
    python scripts/inspect_adapter.py <adapter_dir> --compare <other_adapter_dir>

`--compare` diffs two adapters module-by-module, which is how you answer "did this
round actually change anything, and where" rather than trusting a WER delta. After
round 2, the useful invocation is round-1 backup vs the round-2 adapter.

Exit code: 1 if ANY module is inert, 0 if all carry trained weights — so it can
gate a pipeline. A non-zero exit here is a finding, not a crash.

Needs `safetensors`, which lives in the SRTTimeStampPOC venv / the `python` on PATH,
not in this project's `.\venv` (which has `datasets` but not `safetensors`).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("safetensors is required: pip install safetensors")

WEIGHTS = "adapter_model.safetensors"


def half_of(key: str) -> str:
    if ".encoder." in key and ".decoder." not in key:
        return "encoder"
    if ".decoder." in key:
        return "decoder"
    return "other"


def module_of(key: str) -> str:
    """base_model...layers.0.fc1.lora_B.weight -> fc1"""
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p.startswith("lora_"):
            return parts[i - 1]
    return key


def load(adapter_dir: Path) -> dict:
    path = adapter_dir / WEIGHTS
    if not path.exists():
        sys.exit(f"no {WEIGHTS} in {adapter_dir}")
    out = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def summarise(tensors: dict, label: str) -> int:
    """Prints the report. Returns the number of inert modules."""
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")

    b_keys = [k for k in tensors if "lora_B" in k]
    a_keys = [k for k in tensors if "lora_A" in k]
    if not b_keys:
        print("  no lora_B tensors found — is this a LoRA adapter?")
        return 0

    live, dead = Counter(), Counter()
    dead_modules = defaultdict(Counter)
    params = Counter()
    for k in b_keys:
        h = half_of(k)
        # lora_B == 0 means this module contributes B@A = 0: it is INERT,
        # regardless of what lora_A holds.
        if tensors[k].abs().sum().item() == 0:
            dead[h] += 1
            dead_modules[h][module_of(k)] += 1
        else:
            live[h] += 1
    for k in a_keys + b_keys:
        params[half_of(k)] += tensors[k].numel()

    total_params = sum(params.values())
    print(f"  {len(b_keys)} LoRA modules, {total_params:,} adapter parameters\n")
    print(f"  {'half':<10}{'trained':>9}{'INERT':>8}{'params':>14}{'share':>8}")
    print(f"  {'-' * 47}")
    for h in ("encoder", "decoder", "other"):
        if live[h] or dead[h]:
            share = 100 * params[h] / total_params if total_params else 0
            print(f"  {h:<10}{live[h]:>9}{dead[h]:>8}{params[h]:>14,}{share:>7.1f}%")

    n_dead = sum(dead.values())
    if not n_dead:
        print("\n  ✅ every LoRA module carries trained weights")
        return 0

    dead_params = sum(params[h] for h in dead if dead[h] and not live[h])
    print(f"\n  ⚠️  {n_dead} of {len(b_keys)} modules are INERT (lora_B == 0, so B@A == 0).")
    for h, mods in dead_modules.items():
        print(f"      {h}: " + ", ".join(f"{m}×{c}" for m, c in sorted(mods.items())))
    if dead_params:
        print(f"      ~{dead_params:,} parameters ({100 * dead_params / total_params:.1f}%) "
              "never received a gradient.")
    if dead["encoder"] and not live["encoder"]:
        print("\n      The WHOLE encoder is inert. This is the reentrant "
              "gradient-checkpointing\n      gap — see the module docstring. Fix: "
              'gradient_checkpointing_kwargs={"use_reentrant": False}')
    return n_dead


def pair_modules(tensors: dict) -> dict:
    """{module prefix: (A, B)} for every module that has both halves."""
    pairs = defaultdict(dict)
    for k, v in tensors.items():
        if ".lora_A" in k:
            pairs[k.split(".lora_A")[0]]["A"] = v
        elif ".lora_B" in k:
            pairs[k.split(".lora_B")[0]]["B"] = v
    return {k: (v["A"], v["B"]) for k, v in pairs.items() if "A" in v and "B" in v}


def delta_rms(tensors: dict, scaling: float = 1.0) -> dict:
    """Per-module RMS of the effective weight change, grouped by half.

    `changed vs identical` is binary and cannot answer the question this script is
    usually run to answer: did a half move ENOUGH, or was its learning rate too
    low? A tensor that shifts in the twelfth decimal counts as "changed".

    What a LoRA module actually applies to the frozen base weight is
    `ΔW = (alpha/r) · B @ A`. Its RMS — Frobenius norm over sqrt(numel) — is
    comparable across modules of different shapes, so encoder and decoder can be
    put side by side. An encoder two orders of magnitude below the decoder is
    under-trained, whatever the `changed` count says.
    """
    out = defaultdict(list)
    for name, (A, B) in pair_modules(tensors).items():
        dw = (B.double() @ A.double()) * scaling
        out[half_of(name)].append(dw.pow(2).mean().sqrt().item())
    return dict(out)


def report_magnitudes(tensors: dict, label: str, scaling: float = 1.0) -> None:
    rms = delta_rms(tensors, scaling)
    if not rms:
        return
    print(f"\n  effective update magnitude  ΔW = (alpha/r)·B@A   [{label}]")
    print(f"  {'half':<10}{'modules':>8}{'median RMS':>14}{'max RMS':>14}")
    print(f"  {'-' * 46}")
    for h in ("encoder", "decoder", "other"):
        if rms.get(h):
            v = sorted(rms[h])
            print(f"  {h:<10}{len(v):>8}{v[len(v) // 2]:>14.3e}{v[-1]:>14.3e}")
    enc, dec = rms.get("encoder"), rms.get("decoder")
    if enc and dec:
        e = sorted(enc)[len(enc) // 2]
        d = sorted(dec)[len(dec) // 2]
        if e == 0:
            print("\n  ⚠️  encoder ΔW is exactly zero — the encoder never trained.")
        else:
            ratio = e / d
            verdict = ("comparable — the encoder trained at a real magnitude"
                       if ratio >= 0.1 else
                       "⚠️  encoder is >10x weaker — suspect UNDER-TRAINING, not "
                       "'the encoder does not help'")
            print(f"\n  encoder / decoder median ΔW = {ratio:.2f}x — {verdict}")


def compare(a: dict, b: dict, label_a: str, label_b: str) -> None:
    print(f"\n{'=' * 74}\nCOMPARE  {label_a}  ->  {label_b}\n{'=' * 74}")
    shared = sorted(set(a) & set(b))
    only_a, only_b = set(a) - set(b), set(b) - set(a)
    if only_a or only_b:
        print(f"  ⚠️  key sets differ: {len(only_a)} only in first, "
              f"{len(only_b)} only in second — different adapter shapes")
    if not shared:
        print("  no shared tensors to compare")
        return

    moved = defaultdict(int)
    same = defaultdict(int)
    for k in shared:
        if a[k].shape != b[k].shape:
            moved[half_of(k) + " (shape changed)"] += 1
        elif a[k].double().sub(b[k].double()).abs().sum().item() > 0:
            moved[half_of(k)] += 1
        else:
            same[half_of(k)] += 1

    print(f"  {len(shared)} shared tensors\n")
    print(f"  {'half':<10}{'changed':>9}{'identical':>11}")
    print(f"  {'-' * 30}")
    for h in sorted(set(list(moved) + list(same))):
        print(f"  {h:<10}{moved[h]:>9}{same[h]:>11}")
    if not sum(moved.values()):
        print("\n  ⚠️  NOTHING changed between these two adapters.")
    else:
        print(f"\n  ✅ {sum(moved.values())} tensors changed")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("adapter", type=Path, help="adapter directory")
    p.add_argument("--compare", type=Path, default=None, metavar="DIR",
                   help="second adapter to diff against (e.g. round 1 vs round 2)")
    args = p.parse_args()

    scaling = 1.0
    cfg = args.adapter / "adapter_config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text(encoding="utf-8"))
        print(f"config: r={c.get('r')} alpha={c.get('lora_alpha')} "
              f"targets={sorted(c.get('target_modules') or [])}")
        print(f"        inference_mode={c.get('inference_mode')} "
              "(true means a resume MUST pass is_trainable=True)")
        if c.get("r"):
            scaling = c.get("lora_alpha", c["r"]) / c["r"]

    first = load(args.adapter)
    n_dead = summarise(first, str(args.adapter))
    report_magnitudes(first, args.adapter.name, scaling)
    if args.compare:
        second = load(args.compare)
        summarise(second, str(args.compare))
        report_magnitudes(second, args.compare.name, scaling)
        compare(first, second, args.adapter.name, args.compare.name)
    print()
    return 1 if n_dead else 0


if __name__ == "__main__":
    sys.exit(main())
