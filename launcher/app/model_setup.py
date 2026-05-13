"""Pre-TUI model discovery and download.

Runs in the parent cmd window before Textual mounts. Prints plain text
so any error is visible (no flashing-and-closing window failure mode).

Discovery order:
  1. $VLLM_MODEL_DIR, explicit override, wins.
  2. user_config.json saved path from a previous run.
  3. <writable_root>\\models\\Qwen3.6-27B-int4-AutoRound (default location).
  4. Quick scan of fixed drives for a folder of that name.

If nothing is found, prompts the user:
  [1] enter a custom path
  [2] auto-download from Hugging Face (~16 GB, public, no token needed)
  [q] quit

Auto-download uses stdlib urllib only, no huggingface_hub dependency.
The Lorbus repo is public so the resolve URLs work anonymously.

After a model dir is identified (whether discovered, entered, or freshly
downloaded) we always run ``apply_tokenizer_patch`` so the
``TokenizersBackend`` -> ``Qwen2Tokenizer`` fix is applied without the
user having to read the troubleshooting doc.
"""
from __future__ import annotations

import json
import os
import shutil
import string
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from . import paths

# Reuse the canonical patcher so the rule lives in one place. We append
# the windows_tools dir to sys.path because the embedded python ships it
# alongside launcher/, but doesn't put it on import path automatically.
_WINDOWS_TOOLS = paths.install_root() / "windows_tools"
if _WINDOWS_TOOLS.is_dir() and str(_WINDOWS_TOOLS) not in sys.path:
    sys.path.insert(0, str(_WINDOWS_TOOLS))
try:
    from patch_tokenizer import apply_tokenizer_patch  # type: ignore
except ImportError:  # pragma: no cover, missing windows_tools/, dev checkout
    def apply_tokenizer_patch(model_dir, keep_backup: bool = False) -> bool:  # type: ignore
        return False

REPO_ID = paths.DEFAULT_MODEL_REPO
HF_API_TREE = f"https://huggingface.co/api/models/{REPO_ID}/tree/main?recursive=1"
HF_RESOLVE_TMPL = f"https://huggingface.co/{REPO_ID}/resolve/main/{{path}}"


# ---------------------------------------------------------------- discovery


def _scan_fixed_drives() -> list[Path]:
    """Look for a `Qwen3.6-27B-int4-AutoRound` folder on the obvious places.

    Cheap: only checks a fixed list of likely parent directories on each
    drive, does not walk full trees. Designed to find the folder a user
    already has from a previous download without lighting the disk on
    fire. The candidate layouts cover what users actually do in practice:

      <drive>:\\<target>                                  (dropped at root)
      <drive>:\\_models\\<target>                         (common convention)
      <drive>:\\models\\<target>                          (HF / generic)
      <drive>:\\AI\\<target>                              (catch-all)
      <drive>:\\AI\\models\\<target>                      (nested catch-all)
      <drive>:\\huggingface\\<target>                     (HF cache root)
      <drive>:\\huggingface\\hub\\<target>                (HF hub default)
      <drive>:\\models\\Lorbus\\<target>                  (HF org-prefixed)

    Update this list when reports come in of a layout we missed; the
    cost of a few extra `Path.exists()` probes is negligible.
    """
    hits: list[Path] = []
    target = paths.DEFAULT_MODEL_DIRNAME
    candidates: list[Path] = []
    for letter in string.ascii_uppercase:
        root = Path(f"{letter}:\\")
        if not root.exists():
            continue
        candidates.append(root / target)
        candidates.append(root / "_models" / target)
        candidates.append(root / "models" / target)
        candidates.append(root / "AI" / target)
        candidates.append(root / "AI" / "models" / target)
        candidates.append(root / "huggingface" / target)
        candidates.append(root / "huggingface" / "hub" / target)
        candidates.append(root / "models" / "Lorbus" / target)
    for c in candidates:
        try:
            if paths.looks_like_model_dir(c):
                hits.append(c)
        except OSError:
            pass
    return hits


def _discover() -> tuple[Path, str] | None:
    """Return (model_dir, source) where source is one of:
    'env' / 'saved-config' / 'default' / 'drive-scan'.

    The source label feeds into the provenance log printed by
    ``ensure_model`` so the user can see *which* dir the launcher
    matched and why, important when multiple Qwen3.6-27B-int4-AutoRound
    folders exist on disk from different uploaders.
    """
    env = os.environ.get("VLLM_MODEL_DIR")
    if env:
        p = Path(env)
        if paths.looks_like_model_dir(p):
            return (p, "env")
    cfg = paths.load_user_config()
    saved = cfg.get("model_dir")
    if saved:
        p = Path(saved)
        if paths.looks_like_model_dir(p):
            return (p, "saved-config")
    default = paths.default_model_dir()
    if paths.looks_like_model_dir(default):
        return (default, "default")
    hits = _scan_fixed_drives()
    if len(hits) == 1:
        return (hits[0], "drive-scan")
    return None


def _provenance_repo_id(p: Path) -> str | None:
    """Best-effort upstream repo id ('<org>/<repo>') for a model dir.

    Two signals, in order:
      1. Hugging Face cache layout: ``…/hub/models--<org>--<repo>/snapshots/<sha>/``
        , any ancestor whose name starts with ``models--`` decodes cleanly.
      2. ``config.json`` ``_name_or_path`` field.
    Returns None when neither is conclusive.
    """
    for parent in [p, *p.parents]:
        name = parent.name
        if name.startswith("models--") and name.count("--") >= 2:
            parts = name.split("--", 2)
            if len(parts) == 3:
                return f"{parts[1]}/{parts[2]}"
    cfg_file = p / "config.json"
    try:
        with cfg_file.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return None
    nop = cfg.get("_name_or_path")
    if isinstance(nop, str) and "/" in nop and not nop.startswith((".", "/")) and ":" not in nop:
        return nop
    return None


# ---------------------------------------------------------------- download


def _http_get_json(url: str, timeout: float = 30.0) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "qwen36-launcher/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _list_repo_files() -> list[dict]:
    """Return [{path, size}, ...] for every blob in the repo at main."""
    data = _http_get_json(HF_API_TREE)
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected HF API response: {type(data).__name__}")
    out: list[dict] = []
    for entry in data:
        if entry.get("type") != "file":
            continue
        path = entry.get("path")
        size = entry.get("size") or (entry.get("lfs") or {}).get("size") or 0
        if path:
            out.append({"path": path, "size": int(size)})
    return out


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def _is_tty() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _download_one(rel_path: str, dest: Path, expected_size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        print(f"  [skip] {rel_path}  ({_fmt_bytes(expected_size)} already present)")
        return
    url = HF_RESOLVE_TMPL.format(path=rel_path)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    req = urllib.request.Request(url, headers={"User-Agent": "qwen36-launcher/1.0"})
    started = time.time()
    last_print = 0.0
    bytes_done = 0
    tty = _is_tty()
    # On a TTY we update an in-place \r progress bar twice a second.
    # Off a TTY (logs, CI, agents, piped output) the \r turns the whole
    # download into one mega-line, so emit a newline-terminated line every
    # ~5 s instead. Keep the cadence loose so a 2 GB shard prints ~10
    # lines, not 200.
    interval = 0.5 if tty else 5.0
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
            total = int(r.headers.get("Content-Length") or expected_size or 0)
            chunk = 1024 * 1024  # 1 MiB
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                bytes_done += len(buf)
                now = time.time()
                if now - last_print >= interval or (total and bytes_done == total):
                    pct = (bytes_done * 100 / total) if total else 0
                    elapsed = max(now - started, 1e-3)
                    rate = bytes_done / elapsed
                    if tty:
                        bar_w = 30
                        fill = int(bar_w * pct / 100) if total else 0
                        bar = "#" * fill + "." * (bar_w - fill)
                        sys.stdout.write(
                            f"\r  {rel_path[:40]:40s} [{bar}] {pct:5.1f}%  "
                            f"{_fmt_bytes(bytes_done)} / {_fmt_bytes(total)}  "
                            f"{_fmt_bytes(int(rate))}/s        "
                        )
                    else:
                        sys.stdout.write(
                            f"  {rel_path}  {pct:5.1f}%  "
                            f"{_fmt_bytes(bytes_done)} / {_fmt_bytes(total)}  "
                            f"{_fmt_bytes(int(rate))}/s\n"
                        )
                    sys.stdout.flush()
                    last_print = now
        if tty:
            sys.stdout.write("\n")
            sys.stdout.flush()
        tmp.replace(dest)
    except (urllib.error.URLError, OSError) as e:
        if tmp.exists():
            try: tmp.unlink()
            except OSError: pass
        raise RuntimeError(f"download failed for {rel_path}: {e}") from e


def _download_repo(dest_root: Path) -> None:
    print(f"\nFetching file list from huggingface.co/{REPO_ID} ...")
    files = _list_repo_files()
    total_bytes = sum(f["size"] for f in files)
    print(f"  {len(files)} files, ~{_fmt_bytes(total_bytes)} total\n")
    print(f"Destination: {dest_root}")
    # Free-space preflight on the destination drive, fail loud BEFORE
    # writing 16 GB of partial shards into a too-small disk.
    try:
        dest_root.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(dest_root.parent).free
        print(f"Free space:  {_fmt_bytes(free)} on {dest_root.anchor or dest_root.parent}")
        if free < total_bytes:
            print(f"\n[ERROR] Need ~{_fmt_bytes(total_bytes)} but only {_fmt_bytes(free)} free.")
            print("Free up space, or set $VLLM_MODELS_DIR to a larger drive and re-run.")
            sys.exit(1)
        if free < total_bytes * 1.1:
            print("[warn] Free space is tight (<10% headroom). Consider a roomier drive.")
    except OSError:
        pass
    print("Downloading. Resumable: a re-run will skip files already present at the right size.\n")
    dest_root.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f['path']}  ({_fmt_bytes(f['size'])})")
        _download_one(f["path"], dest_root / f["path"], f["size"])
    print("\nAll files downloaded.")


# ---------------------------------------------------------------- prompt


def _print_banner() -> None:
    print("=" * 70)
    print(" qwen3.6-windows-server, first-run model setup")
    print("=" * 70)
    install = paths.install_root()
    writable = paths.writable_root()
    print(f"  install root:    {install}")
    print(f"  writable root:   {writable}")
    if writable != install:
        print(f"  (install is read-only; logs/downloads go to writable root)")
    print()


def _print_scan_locations() -> None:
    """List the directory patterns _scan_fixed_drives() checks.

    Printed before --auto-download fires so a user who already has the
    weights stashed in a non-scanned path (e.g. D:\\stuff\\Qwen3.6-...)
    can ctrl-C and re-run with --model-dir instead of waiting on a 16 GB
    re-download. Keep in sync with _scan_fixed_drives() above.
    """
    target = paths.DEFAULT_MODEL_DIRNAME
    patterns = [
        f"<drive>:\\{target}",
        f"<drive>:\\_models\\{target}",
        f"<drive>:\\models\\{target}",
        f"<drive>:\\AI\\{target}",
        f"<drive>:\\AI\\models\\{target}",
        f"<drive>:\\huggingface\\{target}",
        f"<drive>:\\huggingface\\hub\\{target}",
        f"<drive>:\\models\\Lorbus\\{target}",
    ]
    print("[model] no existing weights matched any of these locations on any fixed drive:")
    for p in patterns:
        print(f"          {p}")
    print("[model] if you already have the weights elsewhere, ctrl-C now and re-run with:")
    print(f'          start.bat --model-dir "X:\\path\\to\\{target}" --snapshot start_72tps')
    print()


def _prompt_choice(extra_hits: list[Path]) -> tuple[str, Path | None]:
    """Returns (action, path) where action ∈ {'use','download','quit'}."""
    print("No valid Qwen3.6-27B-int4-AutoRound model directory was found.")
    print()
    if extra_hits:
        print("Possible candidates detected on your drives:")
        for i, h in enumerate(extra_hits, 1):
            print(f"  [{i}] {h}")
        print()
    print("Options:")
    print("  [1]  Enter a path to an existing model directory")
    print("  [2]  Download Lorbus/Qwen3.6-27B-int4-AutoRound from Hugging Face (~16 GB)")
    if extra_hits:
        print("  [3]  Use one of the candidates listed above")
    print("  [q]  Quit")
    print()
    while True:
        choice = input("Choice: ").strip().lower()
        if choice == "q":
            return ("quit", None)
        if choice == "1":
            raw = input("Full path to model directory: ").strip().strip('"')
            if not raw:
                continue
            p = Path(raw)
            if not paths.looks_like_model_dir(p):
                print(f"  ✗ {p} does not look like a model dir (need config.json + .safetensors).")
                continue
            return ("use", p)
        if choice == "2":
            return ("download", paths.download_target_dir())
        if choice == "3" and extra_hits:
            if len(extra_hits) == 1:
                return ("use", extra_hits[0])
            try:
                n = int(input(f"Which candidate (1-{len(extra_hits)})? ").strip())
                if 1 <= n <= len(extra_hits):
                    return ("use", extra_hits[n - 1])
            except ValueError:
                pass
            continue


def _autopatch(model_dir: Path) -> None:
    """Apply tokenizer patch if needed; print only when something changed."""
    try:
        if apply_tokenizer_patch(model_dir):
            print(f"  [autopatch] tokenizer_config.json: 'TokenizersBackend' -> 'Qwen2Tokenizer'")
    except Exception as e:  # noqa: BLE001, never let the patch crash the launcher
        print(f"  [autopatch] WARNING: could not patch tokenizer_config.json: {e}")


# ---------------------------------------------------------------- entrypoint


def ensure_model(
    *,
    auto_download: bool = False,
    explicit_dir: Path | None = None,
) -> Path:
    """Block until a usable model dir is identified. Sets VLLM_MODEL_DIR.

    Persists the resolved path to user_config.json so subsequent launches
    skip the prompt entirely.

    Headless flags:
      ``explicit_dir``, use this exact path; error if it doesn't validate.
      ``auto_download``, when no model is found, download instead of prompting.
    """
    if explicit_dir is not None:
        p = Path(explicit_dir)
        if not paths.looks_like_model_dir(p):
            print(f"\n[ERROR] --model-dir {p} does not look like a Qwen3.6 model dir")
            print("(needs config.json + .safetensors).")
            sys.exit(1)
        os.environ["VLLM_MODEL_DIR"] = str(p)
        cfg = paths.load_user_config()
        cfg["model_dir"] = str(p)
        paths.save_user_config(cfg)
        _autopatch(p)
        return p

    discovered = _discover()
    if discovered is not None:
        found, source = discovered
        print(f"[model] using {found}  (source: {source})")
        if source == "drive-scan":
            repo_id = _provenance_repo_id(found)
            if repo_id and repo_id.lower() != REPO_ID.lower():
                print(
                    f"[model] note: scan matched a Qwen3.6-27B-int4-AutoRound directory "
                    f"from {repo_id}, not {REPO_ID}."
                )
                print(
                    "[model]       MTP draft-head quality is only validated on the "
                    "Lorbus quant, see docs/MTP_HEAD.md."
                )
                print(
                    "[model]       Pass --model-dir <path> (or set $VLLM_MODEL_DIR) "
                    "to override."
                )
            elif repo_id is None:
                print(
                    "[model] note: could not determine the upstream HF repo for this "
                    "directory; if MTP decode looks slow, confirm it's the Lorbus "
                    "quant (see docs/MTP_HEAD.md)."
                )
        os.environ["VLLM_MODEL_DIR"] = str(found)
        cfg = paths.load_user_config()
        if cfg.get("model_dir") != str(found):
            cfg["model_dir"] = str(found)
            paths.save_user_config(cfg)
        _autopatch(found)
        return found

    _print_banner()
    _print_scan_locations()

    if auto_download:
        target = paths.download_target_dir()
        print(f"--auto-download set; pulling Lorbus/Qwen3.6-27B-int4-AutoRound to {target}")
        _download_repo(target)
        if not paths.looks_like_model_dir(target):
            print(f"\n✗ Download finished but {target} still doesn't look like a model dir.")
            sys.exit(1)
        os.environ["VLLM_MODEL_DIR"] = str(target)
        cfg = paths.load_user_config()
        cfg["model_dir"] = str(target)
        paths.save_user_config(cfg)
        _autopatch(target)
        print(f"\n✓ Model ready at: {target}\n")
        return target

    hits = _scan_fixed_drives()
    while True:
        action, path = _prompt_choice(hits)
        if action == "quit":
            print("\nAborted.")
            sys.exit(0)
        if action == "use" and path is not None:
            os.environ["VLLM_MODEL_DIR"] = str(path)
            cfg = paths.load_user_config()
            cfg["model_dir"] = str(path)
            paths.save_user_config(cfg)
            _autopatch(path)
            print(f"\n✓ Using model at: {path}\n")
            return path
        if action == "download" and path is not None:
            try:
                _download_repo(path)
            except Exception as e:  # noqa: BLE001
                print(f"\n✗ Download failed: {e}")
                print("You can retry, enter a path manually, or quit.")
                continue
            if not paths.looks_like_model_dir(path):
                print(f"\n✗ Download finished but {path} still doesn't look like a model dir.")
                continue
            os.environ["VLLM_MODEL_DIR"] = str(path)
            cfg = paths.load_user_config()
            cfg["model_dir"] = str(path)
            paths.save_user_config(cfg)
            _autopatch(path)
            print(f"\n✓ Model ready at: {path}\n")
            return path
