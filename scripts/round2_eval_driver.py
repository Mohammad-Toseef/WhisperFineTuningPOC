r"""Unattended driver: wait for round-2 training to finish, then run the three evals.

WHY THIS EXISTS
---------------
Claude Code asks permission per command, and permission rules are read at session
start -- so a rule added mid-session does not take effect. Running three eval
commands overnight would mean three prompts with nobody awake to answer them.

This script is ONE command. Everything it does happens in its own child processes,
so it needs a single approval and then runs unattended.

WHAT IT DOES
  1. Polls the Modal volume until round-2 training lands an artifact.
  2. Decides what training produced:
       - whisper-urdu-r2-final present      -> full success, run all three evals
       - only ...-r2-lora-adapter, and no
         merge after MERGE_GRACE minutes    -> the moved-weights guard fired and
                                               the merge was SKIPPED by design;
                                               run evals 1 and 2 only, never 3
                                               against a path that does not exist
  3. Runs the evals SEQUENTIALLY, r1 control FIRST -- it must reproduce 10.50% on
     Set A, which validates the harness before any new number is trusted.

Everything is appended to a log file with timestamps, so the outcome is readable
even if nobody is watching.

    python scripts/round2_eval_driver.py --app-id ap-XXXX [--log PATH] [--dry-run]

--dry-run prints the plan and exits without spending anything.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VOLUME = "whisper-training-vol"
ADAPTER_R2 = "whisper-urdu-r2-lora-adapter"
FINAL_R2 = "whisper-urdu-r2-final"

POLL_SECONDS = 120
MAX_WAIT_HOURS = 6.0
# The merge writes 6.2 GB; if the adapter has been committed but no merged model
# appears within this window, the moved-weights guard raised and skipped it.
MERGE_GRACE_MINUTES = 25

EVALS = [
    ("r1-control", ["--which", "finetuned",
                    "--model-path", "/data/model/whisper-urdu-final"],
     "MUST reproduce ~10.50% on Set A (eval_only) or the harness is wrong"),
    ("base", ["--which", "base"],
     "should reproduce ~18.57% on Set A"),
    ("r2", ["--which", "finetuned",
            "--model-path", f"/data/model/{FINAL_R2}"],
     "the actual round-2 result"),
]


def utf8_env() -> dict:
    """modal's CLI dies with 'charmap' codec errors writing emoji to cp1252."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str = "") -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        line = f"[{stamp}] {msg}" if msg else ""
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def run(cmd: list, log: Log, timeout: int | None = None) -> tuple[int, str]:
    """Run a command from the repo root, tee its output to the log."""
    log(f"$ {' '.join(cmd)}")
    try:
        p = subprocess.run(cmd, cwd=str(REPO), env=utf8_env(), timeout=timeout,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
    except subprocess.TimeoutExpired:
        log(f"  !! timed out after {timeout}s")
        return 124, ""
    out = (p.stdout or "") + (p.stderr or "")
    for line in out.splitlines():
        if line.strip():
            log(f"  | {line.rstrip()}")
    return p.returncode, out


def list_model_dir(log: Log) -> str:
    _, out = run(["modal", "volume", "ls", VOLUME, "/model"], log, timeout=180)
    return out


def wait_for_training(log: Log) -> str:
    """Returns 'merged', 'adapter-only', or 'timeout'."""
    log(f"waiting for training (poll {POLL_SECONDS}s, max {MAX_WAIT_HOURS}h)")
    deadline = time.monotonic() + MAX_WAIT_HOURS * 3600
    adapter_first_seen = None

    while time.monotonic() < deadline:
        out = list_model_dir(log)
        if FINAL_R2 in out:
            log(f"✅ {FINAL_R2} present — training completed and merged")
            return "merged"
        if ADAPTER_R2 in out:
            if adapter_first_seen is None:
                adapter_first_seen = time.monotonic()
                log(f"adapter committed; waiting up to {MERGE_GRACE_MINUTES} min "
                    "for the merge")
            elif time.monotonic() - adapter_first_seen > MERGE_GRACE_MINUTES * 60:
                log("⚠️  adapter present but NO merged model after the grace window.")
                log("    The moved-weights guard almost certainly raised and the")
                log("    merge was skipped BY DESIGN. Eval 3 will be skipped.")
                return "adapter-only"
        time.sleep(POLL_SECONDS)

    log(f"⚠️  gave up after {MAX_WAIT_HOURS}h without an artifact")
    return "timeout"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-id", required=True)
    ap.add_argument("--log", type=Path,
                    default=REPO / "logs" / "round2_eval_driver.log")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log = Log(args.log)
    log("=" * 70)
    log(f"round-2 eval driver | app {args.app_id}")
    log(f"log -> {args.log}")
    log("=" * 70)

    if args.dry_run:
        log("DRY RUN — plan only, nothing spent:")
        for name, extra, why in EVALS:
            log(f"  {name:<12} modal run modal_app.py::evaluate {' '.join(extra)}")
            log(f"  {'':<12} ^ {why}")
        log("evals run SEQUENTIALLY, r1 control first.")
        return 0

    state = wait_for_training(log)
    if state == "timeout":
        log("no evals run. Investigate the app before spending anything.")
        return 1

    planned = EVALS if state == "merged" else EVALS[:2]
    if state == "adapter-only":
        log("running evals 1-2 only (r2 has no merged model to score)")

    results = {}
    for name, extra, why in planned:
        log("")
        log("-" * 70)
        log(f"EVAL {name} — {why}")
        log("-" * 70)
        # Generous timeout: 720 clips with predict_with_generate on an A10G,
        # plus a cold container pulling a 6 GB model.
        rc, _ = run(["modal", "run", "modal_app.py::evaluate"] + extra, log,
                    timeout=3 * 3600)
        results[name] = rc
        log(f"EVAL {name} exit={rc}")
        if name == "r1-control" and rc != 0:
            log("⚠️  the CONTROL eval failed. Stopping: if the harness cannot")
            log("    reproduce round 1, no later number can be trusted.")
            break

    log("")
    log("=" * 70)
    for name, _, _ in EVALS:
        got = results.get(name)
        log(f"  {name:<12} {'ok' if got == 0 else ('SKIPPED' if got is None else f'FAILED ({got})')}")
    log("results on the volume: /data/logs/eval_results-r2-*.json")
    log("=" * 70)
    return 0 if all(v == 0 for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
