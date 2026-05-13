"""Writable-path resolver with automatic fallback to %LocalAppData%.

Background: the launcher must work when extracted into a read-only
location like `C:\\Program Files (x86)\\vllm\\` where the user has no
write permission. Logs, downloaded model weights, and saved user
preferences all need a writable home.

Resolution order for the writable root:
  1. Install root if it passes a write probe.
  2. %LocalAppData%\\qwen36-windows-server\\ otherwise.

All path helpers below honor explicit env-var overrides first so power
users can still pin everything to a custom drive.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

APP_NAME = "qwen36-windows-server"
DEFAULT_MODEL_REPO = "Lorbus/Qwen3.6-27B-int4-AutoRound"
DEFAULT_MODEL_DIRNAME = "Qwen3.6-27B-int4-AutoRound"


def install_root() -> Path:
    """Repo / install root. launcher/app/paths.py → repo root is two parents up."""
    return Path(__file__).resolve().parent.parent.parent


def _dir_is_writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def user_data_root() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    return Path(base) / APP_NAME


def is_install_writable() -> bool:
    return _dir_is_writable(install_root())


_writable_root_cache: Path | None = None


def writable_root() -> Path:
    """Returns the install root if writable, else %LocalAppData%\\<APP_NAME>\\.

    Memoized, UAC virtualization can make repeated probes inconsistent.
    """
    global _writable_root_cache
    if _writable_root_cache is not None:
        return _writable_root_cache
    if is_install_writable():
        _writable_root_cache = install_root()
    else:
        d = user_data_root()
        d.mkdir(parents=True, exist_ok=True)
        _writable_root_cache = d
    return _writable_root_cache


def logs_dir() -> Path:
    env = os.environ.get("VLLM_WINDOWS_LOGS")
    if env:
        d = Path(env)
    else:
        d = writable_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_parent_dir() -> Path:
    """Directory that *contains* model folders (one level above MODEL_DIR)."""
    env = os.environ.get("VLLM_MODELS_DIR")
    if env:
        d = Path(env)
    else:
        # Prefer install\models if it's writable (user may have dropped
        # weights there); otherwise fall back to the writable root so a
        # fresh download always has a guaranteed-writable destination.
        installed = install_root() / "models"
        if installed.exists() and _dir_is_writable(installed):
            d = installed
        else:
            d = writable_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_model_dir() -> Path:
    """Where the default Lorbus quant lives (or would live after download)."""
    return models_parent_dir() / DEFAULT_MODEL_DIRNAME


def download_target_dir() -> Path:
    """Always-writable destination for model auto-download.

    Independent of `default_model_dir`, that one prefers an existing
    install\\models folder when present, but a fresh download must land
    somewhere we are 100% sure we can write 16 GB to.
    """
    env = os.environ.get("VLLM_MODELS_DIR")
    if env:
        d = Path(env)
    else:
        d = writable_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d / DEFAULT_MODEL_DIRNAME


def user_config_path() -> Path:
    return writable_root() / "user_config.json"


def load_user_config() -> dict:
    p = user_config_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_user_config(cfg: dict) -> None:
    p = user_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def looks_like_model_dir(d: Path) -> bool:
    """Loose validation: directory holds an HF-style model checkpoint."""
    if not d.is_dir():
        return False
    if not (d / "config.json").is_file():
        return False
    if any(d.glob("*.safetensors")):
        return True
    if (d / "model.safetensors.index.json").is_file():
        return True
    return False
