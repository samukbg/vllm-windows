"""Detect the local NVIDIA GPU architecture.

Used by the dashboard to order configuration cards so that snapshots
matching the host's GPU architecture appear first. Pure stdlib +
``nvidia-smi`` shell-out — no torch/cuda dependency, safe to call before
the runtime is installed.

Buckets:
  - ``"blackwell"`` for sm_120 (RTX 5060/5070/5080/5090 etc.)
  - ``"ampere"``    for sm_86/sm_89 (RTX 30xx and 40xx — same wheel)
  - ``"unknown"``   when nvidia-smi is missing or the model is not
                    recognised. Treated as "show everything in default
                    order" by the dashboard.
"""
from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache

ARCH_BLACKWELL = "blackwell"
ARCH_AMPERE = "ampere"
ARCH_UNKNOWN = "unknown"


# Model-name → arch bucket. Match prefixes so we cover every SKU in the
# generation without listing every variant.
_BLACKWELL_PREFIXES = (
    "RTX 5060",
    "RTX 5070",
    "RTX 5080",
    "RTX 5090",
    "RTX PRO 6000",
    "B100",
    "B200",
)

# Ampere (sm_86) AND Ada (sm_89) share the same Windows wheel in this
# project — both labelled "ampere" for snapshot-grouping purposes.
_AMPERE_PREFIXES = (
    "RTX 3050",
    "RTX 3060",
    "RTX 3070",
    "RTX 3080",
    "RTX 3090",
    "A100",
    "A40",
    "A6000",
    "RTX A",
    # Ada / 40-series — uses the same Ampere zip in this project
    "RTX 4060",
    "RTX 4070",
    "RTX 4080",
    "RTX 4090",
    "L4",
    "L40",
)


def _classify(name: str) -> str:
    n = name.upper()
    for p in _BLACKWELL_PREFIXES:
        if p.upper() in n:
            return ARCH_BLACKWELL
    for p in _AMPERE_PREFIXES:
        if p.upper() in n:
            return ARCH_AMPERE
    return ARCH_UNKNOWN


def _query_nvidia_smi() -> list[str]:
    """Return the list of detected GPU model strings, or [] if smi unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


@lru_cache(maxsize=1)
def detect_arch() -> str:
    """Detect the local GPU architecture bucket.

    Honors ``$VLLM_WINDOWS_GPU_ARCH`` for testing — set to ``blackwell``,
    ``ampere``, or ``unknown`` to override autodetection. Falls back to
    ``"unknown"`` whenever nvidia-smi is missing or the GPU model cannot
    be classified.
    """
    override = os.environ.get("VLLM_WINDOWS_GPU_ARCH", "").strip().lower()
    if override in (ARCH_BLACKWELL, ARCH_AMPERE, ARCH_UNKNOWN):
        return override

    names = _query_nvidia_smi()
    if not names:
        return ARCH_UNKNOWN

    # If any GPU is Blackwell, prefer Blackwell (host has at least one
    # 50-series card → Blackwell zip is the relevant build). Otherwise
    # if any card is Ampere/Ada, return ampere. Else unknown.
    archs = [_classify(n) for n in names]
    if ARCH_BLACKWELL in archs:
        return ARCH_BLACKWELL
    if ARCH_AMPERE in archs:
        return ARCH_AMPERE
    return ARCH_UNKNOWN


def detected_gpu_names() -> list[str]:
    """Return the raw nvidia-smi name list (cached)."""
    return _query_nvidia_smi()


def config_arch(cfg) -> str:
    """Bucket a snapshot config to ``blackwell`` / ``ampere`` / ``unknown``.

    Source of truth is the ``arch:`` key in configs.yaml, but if missing
    we fall back to id-based heuristics so existing yaml without the
    field still classifies sensibly:

      - id starts with ``rtx5090`` → blackwell
      - everything else            → ampere
    """
    explicit = (cfg.raw or {}).get("arch")
    if explicit:
        e = str(explicit).strip().lower()
        if e in (ARCH_BLACKWELL, ARCH_AMPERE, ARCH_UNKNOWN):
            return e
    cid = getattr(cfg, "id", "") or ""
    if cid.lower().startswith("rtx5090"):
        return ARCH_BLACKWELL
    return ARCH_AMPERE


# Display labels (for cards / section headers)
ARCH_LABEL = {
    ARCH_BLACKWELL: "Blackwell",
    ARCH_AMPERE:    "Ampere/Ada",
    ARCH_UNKNOWN:   "Universal",
}
ARCH_COLOR = {
    ARCH_BLACKWELL: "#7fb3ff",   # cool blue
    ARCH_AMPERE:    "#ffbb7f",   # warm orange
    ARCH_UNKNOWN:   "#8b949e",   # neutral
}
