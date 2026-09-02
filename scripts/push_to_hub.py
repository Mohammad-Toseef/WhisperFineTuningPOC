"""
Push a round's artifacts from the Modal volume to HuggingFace Hub (PRIVATE).

WHAT AND WHY
------------
Two artifacts per round, and they are NOT equally replaceable:

  adapter  ~222 MB. The irreplaceable one — nothing regenerates it but another
           full training run, and LoRA's A/B factors cannot be cleanly recovered
           from a merged model. Until this script learned to push it, every round
           had exactly ONE copy, on a Modal volume that CLAUDE.md warns is not
           permanent. Round 1's was rescued by hand mid-session; round 2's had the
           same exposure.
  merged   ~6.2 GB. Derived — base large-v3 + the adapter reproduces it exactly.

So the adapter is the default, and `--what merged` or `--what both` opts into the
big upload.

ONE REPO PER ROUND, adapter in a subfolder:

    round 1  models-training/whisper-large-v3-urdu          (already published)
    round 2  models-training/whisper-large-v3-urdu-r2
             └── lora-adapter/

The org is `models-training` — where round 1 actually lives. It was previously
hardcoded to a personal namespace, which is how round 2's adapter first landed in
the wrong place. `--repo` overrides the whole id when a round needs somewhere else.

Pushing round 2 into round 1's repo would overwrite the published round-1 model —
the same overwrite hazard `run_tag` exists to prevent on the volume, just with a
worse blast radius. The tag decides the repo, so it cannot happen by accident.

Auth: HF write token from the Modal secret "huggingface-secret".

Run:
    modal run scripts/push_to_hub.py --run-tag r2                  # adapter only
    modal run scripts/push_to_hub.py --run-tag r2 --what both
    modal run scripts/push_to_hub.py --what merged                 # round 1, as before
    modal run scripts/push_to_hub.py --run-tag r2 --repo org/name  # explicit target
"""
import modal

app = modal.App("whisper-push-hub")

volume = modal.Volume.from_name("whisper-training-vol", create_if_missing=False)
VOLUME_PATH = "/data"
BASE_REPO = "models-training/whisper-large-v3-urdu"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["huggingface_hub>=0.23,<1.0"])
)


def _tag(run_tag: str) -> str:
    return f"-{run_tag}" if run_tag else ""


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60 * 2,
)
def push(run_tag: str = "", what: str = "adapter", private: bool = True,
         repo: str = "", card: str = ""):
    """card  path ON THE VOLUME to a model card, uploaded as the repo's root
    README.md — the Hub's front page.

    Without it, `--what adapter` uploads everything into a subfolder and the repo
    front page stays empty, so the page a colleague opens says nothing about what
    the model is or how it scored. The adapter folder's own README is PEFT's
    auto-generated stub, and it lands in the subfolder rather than at root.
    """
    import os
    from pathlib import Path
    from huggingface_hub import HfApi

    if what not in ("adapter", "merged", "both"):
        raise ValueError(f"--what must be adapter|merged|both, got {what!r}")

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not token:
        # Don't print values — just the KEY NAMES present, to debug the secret.
        keys = [k for k in os.environ if "TOKEN" in k.upper() or "HF" in k.upper()
                or "HUGG" in k.upper()]
        raise RuntimeError(
            "No HF token found in the Modal secret. Expected env var HF_TOKEN "
            f"(or HUGGING_FACE_HUB_TOKEN). Token-ish keys present: {keys or 'none'}. "
            "Set the secret's key name to HF_TOKEN."
        )

    repo_id = repo or f"{BASE_REPO}{_tag(run_tag)}"
    tag = _tag(run_tag)
    targets = []
    # Subfolders are namespaced BY ROUND, so a later round can be added to a repo
    # that already holds an earlier one without overwriting a single file. Round 1
    # keeps the root for its merged model, which is where it was already published.
    if what in ("adapter", "both"):
        targets.append((Path(f"{VOLUME_PATH}/model/whisper-urdu{tag}-lora-adapter"),
                        f"{run_tag}-lora-adapter" if run_tag else "lora-adapter",
                        "LoRA adapter"))
    if what in ("merged", "both"):
        targets.append((Path(f"{VOLUME_PATH}/model/whisper-urdu{tag}-final"),
                        f"{run_tag}-merged" if run_tag else "",
                        "merged model"))

    card_path = Path(card) if card else None
    if card_path and not card_path.is_file():
        raise FileNotFoundError(
            f"model card not found on volume: {card_path}\n"
            "  Put it there first:  modal volume put whisper-training-vol "
            "<local.md> model/<name>.md")

    # Check EVERYTHING before creating the repo, so a typo'd tag does not leave an
    # empty repo behind on the Hub.
    for src, _, label in targets:
        if not src.exists():
            raise FileNotFoundError(
                f"{label} not found on volume: {src}\n"
                f"  run_tag={run_tag!r} — is that the right round?")
        empty = [p.name for p in src.iterdir() if p.is_file() and p.stat().st_size == 0]
        if empty:
            raise RuntimeError(f"{label} has zero-length files, refusing to push: {empty}")

    api = HfApi(token=token)
    print(f"📤 repo {repo_id}  (private={private})  run_tag={run_tag or '(round 1)'}")

    # Creating a repo and writing to one are DIFFERENT permissions. A token can
    # easily be allowed to push into an existing org repo while being forbidden to
    # create new ones under that namespace — so a 403 here is only fatal if the
    # repo also does not already exist. Uploading is namespaced by round, so
    # landing in an existing repo cannot overwrite an earlier round's files.
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private,
                        exist_ok=True)
    except Exception as exc:
        try:
            api.repo_info(repo_id=repo_id, repo_type="model")
        except Exception:
            raise RuntimeError(
                f"Cannot create {repo_id} and it does not exist (or is not visible "
                f"to this token).\n  Underlying error: {exc}\n"
                "  Either grant the token write access to that namespace, or pass "
                "--repo with a namespace you can write to."
            ) from exc
        print(f"   (cannot create repos here, but {repo_id} exists — uploading into it)")

    for src, path_in_repo, label in targets:
        files = sorted(p.name for p in src.iterdir() if p.is_file())
        total = sum(p.stat().st_size for p in src.iterdir() if p.is_file())
        where = path_in_repo or "(root)"
        print(f"\n   {label}: {len(files)} files, {total / 1e6:.1f} MB -> {where}")
        print(f"   files: {files}")
        api.upload_folder(
            folder_path=str(src),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=path_in_repo,
            commit_message=f"Add {label} — Whisper large-v3 Urdu"
                           f"{f' round {run_tag}' if run_tag else ''}",
        )
        print(f"   ✅ {label} uploaded")

    if card_path:
        print(f"\n   model card: {card_path.name} -> README.md (repo root)")
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
        print("   ✅ model card uploaded")

    print(f"\n✅ https://huggingface.co/{repo_id} (private={private})")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 5,
)
def whoami():
    """Which HF identity is in the Modal secret, and what can it write to?

    A 403 on push is almost always this: the secret holds a personal token with no
    membership in the target org. Guessing at that from the error text wastes a
    round trip, so ask the Hub directly.
    """
    import os
    from huggingface_hub import HfApi

    token = (os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN")
             or os.environ.get("HUGGINGFACE_TOKEN"))
    if not token:
        raise RuntimeError("No HF token in the Modal secret 'huggingface-secret'.")
    info = HfApi(token=token).whoami()
    print(f"  account : {info.get('name')}   ({info.get('type')})")
    print(f"  token   : {info.get('auth', {}).get('accessToken', {}).get('role', '?')}")
    orgs = info.get("orgs") or []
    if not orgs:
        print("  orgs    : NONE — this token can only write to its own namespace")
    for o in orgs:
        print(f"  org     : {o.get('name')}   role={o.get('roleInOrg')}")


@app.local_entrypoint()
def main(run_tag: str = "", what: str = "adapter", private: bool = True,
         repo: str = "", who: bool = False, card: str = ""):
    # Every option must be declared HERE as well as on push(): the CLI is built
    # from this entrypoint's signature, so a parameter added only to push() is
    # rejected as "No such option".
    if who:
        whoami.remote()
        return
    push.remote(run_tag, what, private, repo, card)
