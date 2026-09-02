"""Pre-flight checks for the round-2 training run. READ-ONLY, no GPU, seconds to run.

WHY THIS EXISTS
---------------
Every failure this guards against is one that costs GPU time to discover. A typo'd
key in training_config.yaml, a step count that no longer matches the manifest, or
an output path that collides with round 1's all surface only *after* the container
has pulled a 6 GB model -- or worse, not at all.

RUN IT AFTER ANY EDIT TO training_config.yaml OR modal_app.py, and before launching:

    python tests/test_round2_preflight.py      # no pytest needed
    pytest tests/test_round2_preflight.py      # if pytest is installed

Tests that need the gitignored manifests SKIP rather than fail, so this still runs
on a fresh clone.

Two of the checks below are regressions for real bugs found in the 2026-08-27
pre-flight review (commit 3aed2e2), both silent:
  * a right-length/wrong-content eval_buckets.json mislabelled every prediction row
  * a merge failure discarded the adapter, the only irreplaceable output

They assert over COMMENT-STRIPPED code and the AST on purpose. The first versions
string-matched the source and reported false failures, because prose in the
comments contains the same identifiers as the statements being ordered.
"""
import ast
import json
import math
import re
import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # plain-python fallback; see __main__ below
    pytest = None

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
MODAL_APP = ROOT / "modal_app.py"
CONFIG = ROOT / "config" / "training_config.yaml"
B3_MANIFEST = ROOT / "data" / "processed" / "Batch3" / "manifest_reviewed.json"
R1_MANIFEST = ROOT / "data" / "processed" / "manifest_reviewed.json"

#: The eval holdout, pinned. Kept here as well as in --eval-episodes so a
#: drift between the two is a test failure rather than a silent mismatch.
SET_B = ["B3013", "B3017", "B3029", "B3031", "B3039", "B3051", "B3063", "B3076"]
SET_A = ["EP5_vwzNL2oziZs", "EP6_SrVnpBqd7bI", "EP34_h87EJF0Zvco", "EP41_mBtP9NKha1g",
         "EP43_m8-37sgUwUQ", "EP44_paAJQ3OKB-8", "EP47_a0NiZST0S6Q"]
EPOCHS = 3

#: Rounds already on the volume. The current round's tag comes from the config, so
#: these tests check INVARIANTS ("do not collide with anything already there",
#: "resume from a round that exists") rather than one round's literal values —
#: the previous version asserted run_tag == "r2" and failed on a correct round-3
#: config, which is worse than no test because it trains people to ignore it.
PRIOR_ROUNDS = ["", "r2"]          # "" is round 1, the untagged layout
DATASET = "/data/processed/dataset_r2"   # round 3 reuses round 2's split, on purpose


class Skipped(Exception):
    """Raised when a prerequisite is absent and pytest is not available."""


def skip(reason: str):
    if pytest is not None:
        pytest.skip(reason)
    raise Skipped(reason)


def load_cfg() -> dict:
    if yaml is None:
        skip("pyyaml not installed")
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> list:
    if not path.exists():
        skip(f"{path.relative_to(ROOT)} not present (gitignored data)")
    return json.loads(path.read_text(encoding="utf-8"))


def episode_of(entry: dict) -> str:
    return Path(entry["audio_path"].replace("\\", "/")).parent.name


def code_lines(block: str) -> list[str]:
    """Source with comments and blank lines removed.

    Ordering assertions must reason about STATEMENTS. `merge_and_unload` appears in
    a comment above the commit it is meant to be ordered after, so an index search
    over raw source finds the comment and reports a failure that does not exist.
    """
    out = []
    for line in block.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            out.append(stripped)
    return out


def func_src(src: str, name: str) -> str:
    """Source of one top-level function, via AST.

    Previously this sliced text from `def train(` to the Evaluation banner. When
    probe() was added between them, that span silently covered TWO functions, so
    ordering assertions could match statements in the wrong one. The AST cannot
    drift that way.
    """
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node)
            if seg:
                return seg
    raise AssertionError(f"function {name}() not found in modal_app.py")


def train_block(src: str) -> str:
    return func_src(src, "train")


# ── config ────────────────────────────────────────────────────────────────────
def test_round_switches_are_on():
    """A tagged run, resuming a round that exists, on the tagged dataset."""
    m = _paths_module()
    cfg = load_cfg()
    tag = cfg["training"].get("run_tag")
    assert tag, "run_tag is empty — this run would overwrite round 1's untagged paths"
    assert tag not in PRIOR_ROUNDS, (
        f"run_tag {tag!r} belongs to a round already on the volume")
    resume = cfg["lora"].get("resume_from_adapter")
    known = {m.adapter_path(t) for t in PRIOR_ROUNDS}
    assert resume in known, (
        f"resume_from_adapter {resume!r} is not a known round's adapter: {sorted(known)}")
    assert cfg["data"]["dataset_path"] == DATASET


def test_every_config_key_the_code_reads_exists():
    """The highest-value check here: a missing key dies AFTER a 6 GB model load.

    Scans modal_app.py for t_cfg["..."] / l_cfg["..."] / cfg["a"]["b"] and resolves
    each against the real yaml.
    """
    cfg = load_cfg()
    src = MODAL_APP.read_text(encoding="utf-8")
    missing = []
    for alias, section in (("t_cfg", cfg["training"]), ("l_cfg", cfg["lora"])):
        for m in re.finditer(alias + r'\["([^"]+)"\]', src):
            if m.group(1) not in section:
                missing.append(f'{alias}["{m.group(1)}"]')
    for m in re.finditer(r'cfg\["(\w+)"\]\["(\w+)"\]', src):
        a, b = m.group(1), m.group(2)
        if a not in cfg or b not in cfg[a]:
            missing.append(f'cfg["{a}"]["{b}"]')
    assert not missing, f"config keys read by the code but absent from the yaml: {sorted(set(missing))}"


def test_learning_rate_is_a_float_not_a_string():
    """YAML reads 5.0e-6 as a float but a bare 5e-6 as the STRING '5e-6', which
    would reach the optimizer as a string."""
    cfg = load_cfg()["training"]
    for key in ("learning_rate", "encoder_learning_rate"):
        lr = cfg[key]
        assert isinstance(lr, float), f"{key} parsed as {type(lr).__name__}: {lr!r}"
        assert 0 < lr < 1e-3, f"{key} = {lr!r} is not a plausible fine-tuning rate"


def test_encoder_gets_its_own_optimizer_group():
    """The encoder starts cold (lora_B all zero) while the decoder has trained
    twice, so they need different rates. Two things must hold, and the second is
    the one that silently fails: the config must set a distinct encoder rate, AND
    the code must actually build parameter groups from it. A config key nothing
    reads would leave the encoder on the decoder's rate and look configured.
    """
    cfg = load_cfg()["training"]
    assert cfg["encoder_learning_rate"] >= cfg["learning_rate"], (
        "a cold encoder should not train slower than the warm decoder: "
        f"encoder {cfg['encoder_learning_rate']} < decoder {cfg['learning_rate']}")

    src = MODAL_APP.read_text(encoding="utf-8")
    lines = code_lines(src)
    assert any('t_cfg["encoder_learning_rate"]' in l for l in lines), (
        "encoder_learning_rate is in the yaml but nothing reads it")
    # Both entry points must use the shared builder, or the probe verifies a
    # construction the real run does not use.
    for fn in ("train", "probe"):
        assert any("build_dual_rate_optimizer" in l
                   for l in code_lines(func_src(src, fn))), (
            f"{fn}() does not use build_dual_rate_optimizer — the probe would then "
            "verify a construction the real run does not use")
    assert any("optimizers=(optimizer, None)" in l for l in lines), (
        "the optimizer is built but never handed to the Trainer, or a scheduler "
        "is passed alongside it — pass None so warmup/decay is built over these "
        "groups rather than flattening them")


def test_checkpoint_selection_is_coherent():
    cfg = load_cfg()["training"]
    # compute_metrics returns "wer"; the Trainer looks for "eval_" + this name.
    assert cfg["metric_for_best_model"] == "wer", cfg["metric_for_best_model"]
    assert cfg["greater_is_better"] is False, "WER: lower is better"
    assert cfg["load_best_model_at_end"] is True
    # load_best_model_at_end cannot resolve a best checkpoint unless saves land on evals.
    assert cfg["save_steps"] % cfg["eval_steps"] == 0


def test_training_arguments_are_valid_for_the_pinned_transformers():
    """`evaluation_strategy` and `tokenizer=` were renamed in 4.46. They are correct
    under the <4.46 pin, so this asserts the PIN still matches the ARGS -- if the pin
    moves, this fails instead of the run failing."""
    src = MODAL_APP.read_text(encoding="utf-8")
    pinned_below_446 = 'transformers>=4.40.0,<4.46' in src
    uses_old_names = ("evaluation_strategy=" in src) or ("tokenizer=processor" in src)
    assert not uses_old_names or pinned_below_446, (
        "modal_app.py uses evaluation_strategy= / tokenizer= but transformers is no "
        "longer pinned below 4.46 — rename to eval_strategy= / processing_class=")


# ── step schedule vs the real data ────────────────────────────────────────────
def test_step_schedule_matches_the_manifest():
    """max_steps must be a whole number of epochs over the ACTUAL train count.

    Recomputed from the manifest rather than trusted, because changing the eval
    holdout changes the train count and therefore every step number.
    """
    cfg = load_cfg()["training"]
    rev = load_manifest(B3_MANIFEST)
    setb = set(SET_B)
    n_train = sum(1 for s in rev if episode_of(s).split("_")[0] not in setb)
    eff = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(n_train / eff)
    assert cfg["eval_steps"] == steps_per_epoch, (
        f"eval_steps {cfg['eval_steps']} != one epoch ({steps_per_epoch}) "
        f"for {n_train} train clips at effective batch {eff}")
    assert cfg["max_steps"] == steps_per_epoch * EPOCHS, (
        f"max_steps {cfg['max_steps']} != {EPOCHS} epochs ({steps_per_epoch * EPOCHS})")
    warmup_frac = cfg["warmup_steps"] / cfg["max_steps"]
    assert 0.08 <= warmup_frac <= 0.12, f"warmup is {warmup_frac:.1%} of the run"


def test_all_eval_episodes_resolve():
    """A typo in --eval-episodes is rejected by dataset_builder, but only after the
    manifests are loaded. Catch it here instead."""
    rev = load_manifest(B3_MANIFEST)
    r1 = load_manifest(R1_MANIFEST)
    b3_labels = {episode_of(s).split("_")[0] for s in rev}
    for label in SET_B:
        assert label in b3_labels, f"Set B episode {label} is not in the Batch 3 manifest"
    r1_folders = {episode_of(s) for s in r1}
    for folder in SET_A:
        assert folder in r1_folders, f"Set A episode {folder} is not in the round-1 manifest"


def test_eval_split_is_two_distinct_corpora():
    """Set A and Set B must not overlap, or the retention guard measures training data."""
    rev = load_manifest(B3_MANIFEST)
    r1 = load_manifest(R1_MANIFEST)
    b3_eval = {episode_of(s) for s in rev if episode_of(s).split("_")[0] in set(SET_B)}
    r1_eval = {episode_of(s) for s in r1 if episode_of(s) in set(SET_A)}
    assert b3_eval and r1_eval
    assert not (b3_eval & r1_eval), "an episode is in both eval sets"


# ── output paths ──────────────────────────────────────────────────────────────
def _paths_module():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        import modal_app
    except Exception as exc:  # modal not installed, or import-time failure
        skip(f"cannot import modal_app ({type(exc).__name__}: {exc})")
    return modal_app


def test_writes_nowhere_a_previous_round_lives():
    """The whole point of run_tag. A collision here would destroy an artifact an
    earlier round produced — including the one this run resumes from."""
    m = _paths_module()
    cfg = load_cfg()
    tag = cfg["training"]["run_tag"]
    previous = {"/data/processed/dataset",
                "/data/checkpoints/whisper-large-v3-urdu", "/data/logs"}
    for t in PRIOR_ROUNDS:
        previous |= {m.adapter_path(t), m.final_model_path(t)}
        if t:
            previous |= {f"/data/checkpoints/whisper-large-v3-urdu-{t}",
                         f"/data/logs/{t}"}
    current = {m.adapter_path(tag), m.final_model_path(tag),
               cfg["training"]["output_dir"], cfg["training"]["logging_dir"]}
    clash = previous & current
    assert not clash, f"round-{tag} output collides with an earlier round: {sorted(clash)}"


def test_resume_source_is_not_this_runs_output():
    """train() raises on this, but failing here costs no container start."""
    m = _paths_module()
    cfg = load_cfg()
    resume = cfg["lora"]["resume_from_adapter"]
    assert resume != m.adapter_path(cfg["training"]["run_tag"]), (
        "resume_from_adapter is this run's OWN output — training would overwrite "
        "the adapter it is continuing from")


def test_eval_result_filenames_are_distinct_per_model():
    """Three eval runs share one dataset. Without a per-model label they share one
    filename too, and only the last survives -- losing the baselines the round is
    judged against."""
    m = _paths_module()
    tag = load_cfg()["training"]["run_tag"]
    labels = ["base"] + [f"whisper-urdu{('-' + t) if t else ''}-final"
                         for t in PRIOR_ROUNDS] + [f"whisper-urdu-{tag}-final"]
    names = {m.eval_results_path(tag, lbl) for lbl in labels}
    assert len(names) == len(labels), (
        f"eval filenames collide — only the last run would survive: {sorted(names)}")
    assert m.eval_results_path() == "/data/logs/eval_results.json", \
        "the untagged default must stay round 1's filename, so it is never rewritten"


# ── regressions for the two bugs found on 2026-08-27 ──────────────────────────
def test_misaligned_sidecar_is_discarded():
    """BUG: the predictions sidecar decides whether to attach episode/source labels
    by comparing LENGTHS. A right-length/wrong-content eval_buckets.json passed that
    check, so every row got a confident WRONG label -- someone reading what they
    believed was B3039's output would have been reading another episode's.

    Both rejection branches must reset bmeta to [].
    """
    src = MODAL_APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    discards = 0
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        for branch in (node.body, node.orelse):
            warns = any(isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                        and getattr(s.value.func, "id", "") == "print" for s in branch)
            resets = any(isinstance(s, ast.Assign)
                         and any(getattr(t, "id", "") == "bmeta" for t in s.targets)
                         and isinstance(s.value, ast.List) and not s.value.elts
                         for s in branch)
            if warns and resets:
                discards += 1
    assert discards == 2, (
        f"expected both sidecar-rejection branches to reset bmeta=[], found {discards}")


def test_adapter_is_committed_before_the_merge():
    """BUG: one volume.commit(), after BOTH the adapter save and the 6.2 GB merged
    save. The adapter is the only irreplaceable output -- the merged model derives
    from it, nothing regenerates the adapter but another full training run. A failure
    in merge_and_unload() therefore discarded training that had already succeeded.
    """
    lines = code_lines(train_block(MODAL_APP.read_text(encoding="utf-8")))
    assert lines.count("volume.commit()") == 2, "expected two commits in train()"
    i_save = next(i for i, l in enumerate(lines)
                  if l == "model.save_pretrained(out_adapter)")
    i_commit = next(i for i, l in enumerate(lines)
                    if l == "volume.commit()" and i > i_save)
    i_merge = next(i for i, l in enumerate(lines) if "merge_and_unload" in l)
    assert i_save < i_commit < i_merge, (
        f"order must be adapter-save -> commit -> merge; got "
        f"save={i_save} commit={i_commit} merge={i_merge}")


def test_encoder_training_is_actually_enabled():
    """Round 3's entire point. Rounds 1 and 2 trained the decoder only because the
    REENTRANT checkpoint path builds no backward graph for a block whose inputs
    carry no gradient — and Whisper's encoder input is a mel spectrogram.

    Two halves, both required: the flag must say non-reentrant, AND the trainer
    must actually be given it. Setting the config key without wiring it through
    would look correct and change nothing — the same shape of silent no-op the
    other guards exist for.
    """
    cfg = load_cfg()["training"]
    assert cfg.get("gradient_checkpointing_use_reentrant") is False, (
        "gradient_checkpointing_use_reentrant must be false, or the encoder's LoRA "
        f"gets grad=None and 41% of the adapter trains nothing. Got "
        f"{cfg.get('gradient_checkpointing_use_reentrant')!r}")
    assert cfg.get("gradient_checkpointing") is True, (
        "the use_reentrant flag only takes effect while checkpointing is on")

    src = MODAL_APP.read_text(encoding="utf-8")
    m = re.search(r"gradient_checkpointing_kwargs\s*=\s*\{([^}]*)\}", src)
    assert m, ("Seq2SeqTrainingArguments never receives gradient_checkpointing_kwargs "
               "— the config flag is read by nobody and the encoder stays frozen")
    assert "use_reentrant" in m.group(1), m.group(1)


def test_resume_load_passes_is_trainable():
    """Without is_trainable=True the adapter loads FROZEN: training runs to
    completion, logs a loss curve, saves a model, and changes nothing."""
    src = MODAL_APP.read_text(encoding="utf-8")
    m = re.search(r"PeftModel\.from_pretrained\(([^)]*)\)", src)
    assert m, "PeftModel.from_pretrained call not found"
    assert "is_trainable=True" in " ".join(m.group(1).split())


def test_train_fails_loudly_on_the_silent_no_ops():
    """Three raises, not warnings: zero trainable params, no lora_B at all, and
    every lora_B zero (a freshly-initialised adapter, i.e. round 1 never loaded)."""
    block = train_block(MODAL_APP.read_text(encoding="utf-8"))
    segment = block[block.index("trainable = sum("):block.index("# ── Data Collator")]
    assert segment.count("raise RuntimeError") == 3, (
        f"expected 3 raises guarding the silent no-ops, found "
        f"{segment.count('raise RuntimeError')}")
    assert '"lora_B" in n' in segment, "the lora_B non-zero check is missing"


def test_weight_movement_is_checked_after_training():
    """The three raises above all fire BEFORE step 1, so they prove the adapter is
    loaded and trainable — not that it moved. Under fp16 the grad scaler skips any
    step with inf/NaN gradients; if every step is skipped the run still logs a loss
    curve and saves an adapter identical to the resume source.

    The signature must be taken before trainer.train(), and the comparison must sit
    AFTER the adapter commit (so it cannot destroy finished training) and BEFORE
    merge_and_unload (so no merged model is minted from untrained weights).
    """
    lines = code_lines(train_block(MODAL_APP.read_text(encoding="utf-8")))

    def idx(pred, what):
        i = next((i for i, l in enumerate(lines) if pred(l)), None)
        assert i is not None, f"{what} not found in train()"
        return i

    i_sig = idx(lambda l: l.startswith("pre_train_sig = {"), "the pre-training snapshot")
    i_train = idx(lambda l: l == "trainer.train()", "trainer.train()")
    i_save = idx(lambda l: l == "model.save_pretrained(out_adapter)", "the adapter save")
    i_commit = next(i for i, l in enumerate(lines) if l == "volume.commit()" and i > i_save)
    i_moved = idx(lambda l: l.startswith("moved = sum("), "the moved-weights comparison")
    i_merge = idx(lambda l: "merge_and_unload" in l, "merge_and_unload")

    assert i_sig < i_train, (
        f"the weight signature must be taken BEFORE training (sig={i_sig} "
        f"train={i_train}) — snapshotting after is a tautology")
    assert i_commit < i_moved < i_merge, (
        f"the check must sit between the adapter commit and the merge; got "
        f"commit={i_commit} check={i_moved} merge={i_merge}")

    # A print here would leave the failure exactly as silent as it is without the
    # check, and the merge would still produce a round-2 model from untrained weights.
    tail = "\n".join(lines[i_moved:i_merge])
    assert "raise RuntimeError" in tail, (
        "the moved-weights check must RAISE, not warn — otherwise the merge still runs")


def _lifted(name: str) -> str:
    """Source text of the `name = ...` assignment inside train(), via AST.

    Lifted rather than reimplemented: a test that retypes the logic proves the
    reimplementation works, which is not the question.
    """
    src = MODAL_APP.read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "train")
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == name for t in node.targets):
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{name} assignment not found in train()")


def test_moved_weights_check_answers_correctly():
    """Runs the ACTUAL snapshot/compare lines from train() against a real tiny
    module, because ordering alone does not prove the comparison is right.

    Covers the case the guard exists for (nothing moved), the cases that must NOT
    trip it (only the frozen base moved; a single tensor moved), and a regression
    for the cancelling signature this test caught -- see balanced_signs below.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        skip("torch not installed")

    sig_src, moved_src = _lifted("pre_train_sig"), _lifted("moved")

    class Tiny(nn.Module):
        """PEFT's naming: lora_A/lora_B trainable, base frozen. lora_B starts at
        zeros exactly as a fresh adapter does."""

        def __init__(self):
            super().__init__()
            # Seeded: an unseeded randn made the tiny-nudge scenario pass or fail
            # on the draw, which is how the signature's precision limit was found
            # in the first place.
            torch.manual_seed(0)
            self.lora_A = nn.Parameter(torch.randn(4, 4))
            self.lora_B = nn.Parameter(torch.zeros(4, 4))
            self.base_layer = nn.Parameter(torch.randn(4, 4))
            self.base_layer.requires_grad = False

    def moved_after(mutate, setup=None) -> tuple[int, int]:
        model = Tiny()
        if setup is not None:      # runs BEFORE the snapshot, to shape the weights
            with torch.no_grad():
                setup(model)
        env = {"model": model}
        exec(sig_src, env)  # noqa: S102 - executing our own source, on purpose
        mutate(env["model"])
        exec(moved_src, env)  # noqa: S102
        return env["moved"], len(env["pre_train_sig"])

    def real_step(model):
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=5e-6)
        ((model.lora_A.sum() + model.lora_B.sum() * 2) ** 2).backward()
        opt.step()

    def add_to(attr, eps):
        def go(model):
            with torch.no_grad():
                getattr(model, attr).add_(eps)
        return go

    # Only lora_* trainable params are watched — a frozen base tensor is not training.
    _, n_watched = moved_after(lambda m: None)
    assert n_watched == 2, f"expected lora_A + lora_B watched, got {n_watched}"

    healthy, _ = moved_after(real_step)
    assert healthy > 0, "a real AdamW step at the round-2 lr registered as no movement"

    # THE failure this guard exists for: fp16 grad scaler skipped every step.
    stalled, _ = moved_after(lambda m: None)
    assert stalled == 0, f"untouched weights reported {stalled} moved — guard is blind"

    base_only, _ = moved_after(add_to("base_layer", 1.0))
    assert base_only == 0, "movement in the FROZEN base counted as training"

    # The raise needs EVERY tensor unmoved, so partial movement must pass.
    for attr, eps in (("lora_A", 1e-7), ("lora_B", 1e-8)):
        one, _ = moved_after(add_to(attr, eps))
        assert one > 0, f"{attr} moved by {eps} went undetected"

    # REGRESSION: the signature was `.float().abs().sum()`, which is blind here.
    # With exactly balanced signs, a uniform +eps cancels term-for-term in an
    # ABSOLUTE sum (|x+e| shrinks where x<0 by what it grows where x>0), so a
    # changed tensor read as unchanged. Squares cannot cancel.
    def balanced_signs(model):
        model.lora_A.copy_(torch.tensor([[1.0, -1.0, 2.0, -2.0]] * 4))

    adversarial, _ = moved_after(add_to("lora_A", 1e-6), setup=balanced_signs)
    assert adversarial > 0, (
        "a uniform +1e-6 on a balanced-sign tensor went undetected — the signature "
        "is cancelling, which is what `.abs().sum()` did before it was squared")


# ── plain-python runner (pytest is not installed in either venv) ──────────────
if __name__ == "__main__":
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    passed = skipped = 0
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Skipped as exc:
            skipped += 1
            print(f"  SKIP {name}: {exc}")
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"  FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  ok   {name}")
    print(f"\n{passed} passed, {skipped} skipped, {len(failures)} failed "
          f"({len(tests)} checks)")
    sys.exit(1 if failures else 0)
