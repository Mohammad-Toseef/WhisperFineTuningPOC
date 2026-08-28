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

EVAL_MINUTES = 31.3      # measured: 720 clips with predict_with_generate
N_EVALS = 3
REFRESH_SECONDS = 60


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


def hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, m = divmod(seconds // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def parse(log: str) -> dict:
    """One shape of status, shared by the window and the console renderer."""
    finished = "Training complete" in log
    info = {"finished": finished, "started": False, "pct": 100.0 if finished else 0.0}

    steps = STEP.findall(log)
    if steps:
        cur, total, elapsed, _remaining, sec_it = steps[-1]
        cur, total, sec_it = int(cur), int(total), float(sec_it)
        evals_done = len(METRIC.findall(log))
        train_left = (total - cur) * sec_it
        evals_left = max(0, N_EVALS - evals_done)
        info.update(
            started=True, cur=cur, total=total, sec_it=sec_it, elapsed=elapsed,
            pct=100.0 if finished else 100 * cur / total,
            evals_done=evals_done, evals_left=evals_left,
            train_left=train_left,
            eta=train_left + evals_left * EVAL_MINUTES * 60,
            finished=finished or cur >= total,
        )
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
                results.put(parse(fetch(app_id)))
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
            live = render_console(parse(fetch(app_id)))
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

    if args.console:
        return run_console(args.app_id, args.every, args.watch)
    return run_gui(args.app_id, args.every)


if __name__ == "__main__":
    sys.exit(main())
