r"""Show how far a Modal training run has got — in a window, or on the console.

The trainer emits tqdm lines like

    33%|###3      | 254/762 [46:23<1:34:12, 11.12s/it]

into the app log, but they are buried among thousands of carriage-returned
redraws, and a Windows console cannot render the rest of the output anyway. This
pulls the most recent one and shows it readably.

It also CORRECTS tqdm's ETA. tqdm counts training steps only; it knows nothing
about the three in-training eval passes, which take ~31 minutes each. On a
three-epoch run its estimate is optimistic by up to an hour and a half, so the
remaining eval passes are added back here.

    python scripts/progress.py ap-ymw97lY4HQZmytWLTMU5ex            # window
    python scripts/progress.py ap-... --console                     # text, once
    python scripts/progress.py ap-... --console --watch             # text, looping

The window refreshes on its own and stays open when the run finishes, so the
final numbers are still readable. Fetching happens on a worker thread — a
`modal app logs` call takes seconds, and doing it on the UI thread would freeze
the window every refresh.
"""
import argparse
import os
import queue
import re
import subprocess
import sys
import threading
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:      # noqa: BLE001 - a redirected or exotic stream
        pass


def _can_print(sample: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        sample.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# A cp1252 console cannot encode block characters, and the resulting
# UnicodeEncodeError kills the script rather than just looking wrong.
_FANCY = _can_print("█░✅")
FILL, EMPTY, TICK = ("█", "░", "✅") if _FANCY else ("#", "-", "[done]")

# "254/762 [46:23<1:34:12, 11.12s/it]"
STEP = re.compile(r"(\d+)/(\d+)\s*\[(\d+(?::\d+)+)<(\d+(?::\d+)+|\?),\s*([\d.]+)s/it")
METRIC = re.compile(r"\{'eval_loss'.*?\}")
LOSS = re.compile(r"\{'loss':\s*([\d.]+).*?'epoch':\s*([\d.]+)\}")
WER = re.compile(r"'eval_wer_r1':\s*([\d.]+)")

EVAL_MINUTES = 32.1      # measured: 720 clips with predict_with_generate
N_EVALS = 3
REFRESH_SECONDS = 60
TRAIN_TOTAL = None


def training_total() -> int | None:
    """max_steps from the local config — used to tell the TRAINING tqdm bar apart
    from the EVAL one.

    Both emit the same `N/M [elapsed<remaining, X s/it]` shape, and the eval bar
    counts batches (180 of them at eval batch size 4). Taking the last match in
    the log therefore showed 180/180 = 100% the moment an eval pass started, while
    training was only a third done. Filtering on the known training total is what
    keeps the bar honest.
    """
    try:
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "config" / "training_config.yaml")
            .read_text(encoding="utf-8"))
        return int(cfg["training"]["max_steps"])
    except Exception:      # noqa: BLE001 - run from elsewhere, or no pyyaml
        return None


# ── data ────────────────────────────────────────────────────────────────────
def fetch(app_id: str) -> str:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        p = subprocess.run(["modal", "app", "logs", app_id], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=env,
                           timeout=180)
    except FileNotFoundError:
        raise RuntimeError(
            "`modal` is not on PATH in this shell.\n"
            "It lives in a venv — find it with:  where.exe modal") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError("`modal app logs` timed out after 180s.") from None
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0 and not out.strip():
        raise RuntimeError(f"`modal app logs {app_id}` failed (exit {p.returncode}) "
                           "with no output. Is the app id right? Try `modal app list`.")
    if "not found" in out.lower() and len(out) < 400:
        raise RuntimeError(f"Modal does not recognise app id {app_id!r}:\n"
                           f"{out.strip()[:300]}")
    return out


def _secs(clock: str) -> int:
    """'1:46:32' or '46:23' -> seconds."""
    parts = [int(x) for x in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def recent_rate(steps: list) -> "float | None":
    """Seconds per step from RECENT progress, not tqdm's running average.

    tqdm reports elapsed/steps over the whole run, so once a ~32 minute eval pass
    has happened its s/it is badly inflated — 27.9 s/step when the true rate is
    ~17 — and every remaining-time estimate built on it is wrong by hours.

    Taking the MEDIAN of consecutive-pair rates fixes it: an eval stall is one
    outlier pair among many and the median ignores it, without needing to know
    where the stall was.
    """
    rates = []
    for (c1, _t1, e1, _r1, _s1), (c2, _t2, e2, _r2, _s2) in zip(steps, steps[1:]):
        d_step, d_time = int(c2) - int(c1), _secs(e2) - _secs(e1)
        if d_step > 0 and d_time > 0:
            rates.append(d_time / d_step)
    if not rates:
        return None
    rates = sorted(rates[-60:])
    return rates[len(rates) // 2]


def hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, m = divmod(seconds // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def parse(log: str, train_total: "int | None" = None) -> dict:
    """One shape of status, shared by the window and the console renderer."""
    # Only the completion line means finished. `cur >= total` does NOT: at 762/762
    # the run still has an eval pass, the adapter save, the moved-weights check
    # and a 6.2 GB merge ahead of it.
    finished = "Training complete" in log
    info = {"finished": finished, "started": False, "pct": 100.0 if finished else 0.0,
            "evaluating": False}

    all_matches = STEP.findall(log)
    totals = [int(m[1]) for m in all_matches]
    if train_total is None and totals:
        # No config to read: the training bar is the one with the largest total
        # (762 training steps vs 180 eval batches).
        train_total = max(totals)

    steps = [m for m in all_matches if int(m[1]) == train_total]
    # An eval pass is running if the most recent bar in the log is NOT the
    # training one. During those ~32 minutes the training bar simply stops moving.
    info["evaluating"] = bool(all_matches) and int(all_matches[-1][1]) != train_total

    if steps:
        cur, total, elapsed, _remaining, sec_it = steps[-1]
        cur, total = int(cur), int(total)
        # Prefer the recent median over tqdm's run-long average, which an eval
        # pass inflates by 60%+.
        sec_it = recent_rate(steps) or float(sec_it)
        evals_done = len(METRIC.findall(log))
        train_left = (total - cur) * sec_it
        evals_left = max(0, N_EVALS - evals_done)
        info.update(
            started=True, cur=cur, total=total, sec_it=sec_it, elapsed=elapsed,
            pct=100.0 if finished else 100 * cur / total,
            evals_done=evals_done, evals_left=evals_left,
            train_left=train_left,
            eta=train_left + evals_left * EVAL_MINUTES * 60,
            finished=finished,
        )
        if info["evaluating"]:
            e_cur, e_total = int(all_matches[-1][0]), int(all_matches[-1][1])
            info["eval_progress"] = (e_cur, e_total)
    loss = LOSS.findall(log)
    if loss:
        info["loss"], info["epoch"] = loss[-1][0], float(loss[-1][1])
    evals = METRIC.findall(log)
    if evals:
        info["last_eval"] = evals[-1]
    wer = WER.findall(log)
    if wer:
        info["wer_r1"] = float(wer[-1])
    for marker in ("weights moved", "adapter committed", "Merging adapter"):
        if marker in log:
            info.setdefault("stage", marker)
    return info


# ── console ─────────────────────────────────────────────────────────────────
def render_console(info: dict) -> bool:
    if not info["started"]:
        print("  no progress line yet — the container may still be loading the "
              "6 GB model.")
        return not info["finished"]
    width = 42
    filled = int(width * info["cur"] / info["total"])
    print(f"  {FILL * filled}{EMPTY * (width - filled)}  {info['pct']:5.1f}%")
    print(f"  step {info['cur']:,} / {info['total']:,}   ·   {info['sec_it']:.1f} s/step"
          f"   ·   elapsed {info['elapsed']}")
    print(f"  evals done {info['evals_done']}/{N_EVALS}"
          f"   ·   training left {hms(info['train_left'])}"
          f"   ·   ETA {hms(info['eta'])}")
    if info.get("eval_progress"):
        e_cur, e_total = info["eval_progress"]
        print(f"  ⏸  EVALUATING now — batch {e_cur}/{e_total}. Training is paused; "
              "the bar above is correct, not stuck.")
    if "loss" in info:
        print(f"  latest loss {info['loss']}   ·   epoch {info['epoch']:.2f}")
    if "wer_r1" in info:
        print(f"  Set A raw WER {info['wer_r1']:.2f}  (round 1 = 15.71; "
              ">17 concerning)")
    if info["finished"]:
        print(f"  {TICK} training finished")
    return not info["finished"]


# ── window ──────────────────────────────────────────────────────────────────
def run_gui(app_id: str, every: int) -> int:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("tkinter is not available in this Python — falling back to --console.\n")
        return run_console(app_id, every, watch=True)

    results: "queue.Queue" = queue.Queue()
    stop = threading.Event()

    def worker():
        """Fetch on a background thread; the UI thread must never block on a
        network call or the window freezes for seconds at every refresh."""
        while not stop.is_set():
            try:
                results.put(parse(fetch(app_id), TRAIN_TOTAL))
            except Exception as exc:  # noqa: BLE001 - surface it in the window
                results.put({"error": str(exc), "started": False,
                             "finished": False, "pct": 0.0})
            for _ in range(every * 10):
                if stop.is_set():
                    return
                time.sleep(0.1)

    root = tk.Tk()
    root.title(f"Whisper training — {app_id}")
    root.geometry("560x300")
    root.minsize(520, 280)
    PAD = {"padx": 18}

    head = tk.Label(root, text="Round 3 — encoder + decoder",
                    font=("Segoe UI", 11, "bold"), anchor="w")
    head.pack(fill="x", pady=(16, 2), **PAD)
    sub = tk.Label(root, text=app_id, font=("Consolas", 8), fg="#666", anchor="w")
    sub.pack(fill="x", **PAD)

    pct_lbl = tk.Label(root, text="—", font=("Segoe UI", 30, "bold"), anchor="w")
    pct_lbl.pack(fill="x", pady=(10, 0), **PAD)

    bar = ttk.Progressbar(root, orient="horizontal", mode="determinate", maximum=100)
    bar.pack(fill="x", pady=(4, 12), **PAD)

    body = tk.Label(root, text="starting…", font=("Consolas", 9), justify="left",
                    anchor="nw")
    body.pack(fill="both", expand=True, **PAD)

    status = tk.Label(root, text="", font=("Segoe UI", 8), fg="#666", anchor="w")
    status.pack(fill="x", pady=(0, 10), **PAD)

    def poll():
        try:
            while True:
                info = results.get_nowait()
                paint(info)
        except queue.Empty:
            pass
        root.after(200, poll)

    def paint(info: dict):
        if info.get("error"):
            body.config(text=f"error:\n{info['error']}", fg="#B00")
            status.config(text=f"last try {time.strftime('%H:%M:%S')}")
            return
        body.config(fg="#111")
        if not info["started"]:
            body.config(text="No progress line yet — the container is probably "
                             "still\nloading the 6 GB model.")
            status.config(text=f"updated {time.strftime('%H:%M:%S')}")
            return

        bar["value"] = info["pct"]
        pct_lbl.config(text=f"{info['pct']:.1f}%")
        lines = [
            f"step      {info['cur']:,} / {info['total']:,}",
            f"speed     {info['sec_it']:.1f} s/step      elapsed {info['elapsed']}",
            f"evals     {info['evals_done']} of {N_EVALS} done",
            f"remaining {hms(info['train_left'])} training"
            f"  +  {info['evals_left']} eval pass"
            f"{'es' if info['evals_left'] != 1 else ''}"
            f"   =   {hms(info['eta'])}",
        ]
        if "loss" in info:
            lines.append(f"loss      {info['loss']}      epoch {info['epoch']:.2f}")
        if "wer_r1" in info:
            lines.append(f"Set A WER {info['wer_r1']:.2f} raw   "
                         f"(round 1 = 15.71, >17 concerning)")
        if info.get("eval_progress"):
            e_cur, e_total = info["eval_progress"]
            lines.append(f"EVALUATING batch {e_cur}/{e_total} — training paused")
        if info.get("stage"):
            lines.append(f"stage     {info['stage']}")
        body.config(text="\n".join(lines))

        if info["finished"]:
            bar["value"] = 100
            pct_lbl.config(text="100%", fg="#0B7A66")
            head.config(text="Round 3 — finished", fg="#0B7A66")
            status.config(text="Training complete. This window stays open — "
                               "close it when you're done.")
            stop.set()
        else:
            status.config(text=f"updated {time.strftime('%H:%M:%S')} · "
                               f"refreshing every {every}s")

    def on_close():
        stop.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    threading.Thread(target=worker, daemon=True).start()
    root.after(200, poll)
    root.mainloop()
    return 0


def run_console(app_id: str, every: int, watch: bool) -> int:
    while True:
        print(f"\n  {time.strftime('%H:%M:%S')}  {app_id}")
        try:
            live = render_console(parse(fetch(app_id), TRAIN_TOTAL))
        except RuntimeError as exc:
            sys.exit(str(exc))
        if not watch or not live:
            return 0
        time.sleep(every)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("app_id", help="Modal app id, e.g. ap-XXXX (see `modal app list`)")
    ap.add_argument("--console", action="store_true",
                    help="print to the terminal instead of opening a window")
    ap.add_argument("--watch", action="store_true",
                    help="console only: keep refreshing until the run finishes")
    ap.add_argument("--every", type=int, default=REFRESH_SECONDS,
                    help=f"seconds between refreshes (default {REFRESH_SECONDS})")
    args = ap.parse_args()

    global TRAIN_TOTAL
    TRAIN_TOTAL = training_total()

    if args.console:
        return run_console(args.app_id, args.every, args.watch)
    return run_gui(args.app_id, args.every)


if __name__ == "__main__":
    sys.exit(main())
