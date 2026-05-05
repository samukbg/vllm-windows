"""Sanity-check: vLLM install, patches applied, GPU present.

Run after install / before launch. Prints a green / yellow / red summary.

Checks:
  1. vLLM importable, version is 0.19.0+devnen.* OR 0.20.0+cu132.devnen.*
     (or the matching upstream tags).
  2. Each file in windows_patches/ matches the in-venv copy by sha256
     (patch set picked per detected vLLM major: 0.19 = full CPU-relay set,
     0.20 = reasoning-parser + serving-models only; CPU-relay was dropped
     when 0.20.0 shipped NCCL on Windows).
  3. nvidia-smi reports at least one Ampere+ GPU (sm_86 or higher).
     Blackwell (sm_120) is accepted but warns if the installed wheel is
     a cu126 build, and Ampere/Ada warn if the wheel is a cu130 build.
  4. CUDA 13 runtime shim present when running on a +cu13* wheel.
  5. MSVC `cl.exe` resolvable (warn if not — only matters for FlashInfer JIT).

Exit code 0 = all green. 1 = at least one red. 2 = warnings only.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATCHES = REPO / "windows_patches"

# vllm 0.19.x: CPU-relay fork, six patches.
PATCH_MAP_019 = {
    "parallel_state.py":            "vllm/distributed/parallel_state.py",
    "cuda_communicator.py":         "vllm/distributed/device_communicators/cuda_communicator.py",
    "base_device_communicator.py":  "vllm/distributed/device_communicators/base_device_communicator.py",
    "gpu_worker.py":                "vllm/v1/worker/gpu_worker.py",
    "qwen3_reasoning_parser.py":    "vllm/reasoning/qwen3_reasoning_parser.py",
    "serving_models.py":            "vllm/entrypoints/openai/models/serving.py",
}

# vllm 0.20.x: NCCL on Windows, CPU-relay dropped. Two source patches remain.
PATCH_MAP_020 = {
    "qwen3_reasoning_parser.py":    "vllm/reasoning/qwen3_reasoning_parser.py",
    "serving_models.py":            "vllm/entrypoints/openai/models/serving.py",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def check_vllm(venv: Path) -> tuple[str, str, str]:
    """Return (level, message, version_string). Empty version on failure."""
    py = venv / "Scripts" / "python.exe"
    if not py.exists():
        return ("RED", f"no python.exe at {py}", "")
    try:
        out = subprocess.check_output(
            [str(py), "-c", "import vllm; print(vllm.__version__)"],
            text=True, timeout=30,
        ).strip()
    except subprocess.CalledProcessError as e:
        return ("RED", f"vllm import failed: {e}", "")
    if out.startswith("0.19"):
        return ("GRN", f"vllm {out} (Ampere/Ada — cu126 / 30+40 series)", out)
    if out.startswith("0.20"):
        return ("GRN", f"vllm {out} (Blackwell-capable — cu13x / 30+40+50 series)", out)
    return ("YEL", f"unexpected version {out!r}, expected 0.19.x or 0.20.x", out)


def check_patches(venv: Path, vllm_version: str) -> list[tuple[str, str, str]]:
    sp = venv / "Lib" / "site-packages"
    rows = []
    if vllm_version.startswith("0.20"):
        patch_map = PATCH_MAP_020
    else:
        # default to 0.19 set; older/unknown versions get the conservative full check.
        patch_map = PATCH_MAP_019
    for src_name, dst_rel in patch_map.items():
        src = PATCHES / src_name
        dst = sp / dst_rel
        if not dst.exists():
            rows.append(("RED", src_name, f"missing in venv: {dst}"))
            continue
        if not src.exists():
            rows.append(("YEL", src_name, "patch source missing in repo"))
            continue
        if sha(src) == sha(dst):
            rows.append(("GRN", src_name, "applied"))
        else:
            rows.append(("RED", src_name, "DIFFERS — patches not applied"))
    return rows


def check_gpu(vllm_version: str = "") -> tuple[str, str]:
    if not shutil.which("nvidia-smi"):
        return ("RED", "nvidia-smi not on PATH")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            text=True, timeout=10,
        )
    except subprocess.CalledProcessError as e:
        return ("RED", f"nvidia-smi failed: {e}")
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if not lines:
        return ("RED", "no GPU reported")
    too_old = []
    blackwell = []
    for line in lines:
        try:
            cc = float(line.split(",")[-1].strip())
        except ValueError:
            cc = 0.0
        if cc < 8.6:
            too_old.append(line)
        elif cc >= 12.0:
            blackwell.append(line)
    if too_old:
        return ("YEL", f"non-Ampere+ GPU detected: {too_old}; this fork was tuned on sm_86")
    # Wheel/GPU mismatch warnings.
    if blackwell and vllm_version.startswith("0.19"):
        return ("YEL", f"Blackwell GPU {blackwell} but cu126 wheel installed — "
                "use the qwen3.6-windows-server-portable-x64-blackwell.zip release.")
    if not blackwell and vllm_version.startswith("0.20"):
        # Ampere/Ada on the cu13 wheel works but only with the CUDA 13 driver
        # (596+). The driver is checked indirectly via the import succeeding.
        return ("GRN", f"{' | '.join(lines)} (running cu13x wheel)")
    return ("GRN", " | ".join(lines))


def check_cuda13_shim(venv: Path, vllm_version: str) -> tuple[str, str]:
    """Confirm the launcher-built CUDA 13 shim is present when needed."""
    if not vllm_version.startswith("0.20"):
        return ("GRN", "n/a (cu126 wheel)")
    # The shim lives next to the launcher root, not the venv. Look in
    # both common spots: the repo root (dev) and the writable root sibling
    # of the venv (portable extract).
    candidates = [
        REPO / "cuda13_shim" / "bin" / "cudart64_13.dll",
        venv.parent / "cuda13_shim" / "bin" / "cudart64_13.dll",
    ]
    for c in candidates:
        if c.is_file():
            return ("GRN", f"found at {c.parent}")
    return ("YEL", "cuda13_shim/bin/cudart64_13.dll missing — "
            "launcher rebuilds this on next boot from torch/lib/")


def check_msvc() -> tuple[str, str]:
    if shutil.which("cl.exe"):
        return ("GRN", "cl.exe on PATH")
    candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools",
    ]
    for c in candidates:
        if Path(c).exists():
            return ("YEL", f"MSVC found at {c} but not on PATH (only matters for FlashInfer)")
    return ("YEL", "MSVC not found — fine for TRITON_ATTN; FlashInfer JIT would fail")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default=str(REPO / "venv"))
    args = ap.parse_args()

    venv = Path(args.venv)
    print(f"== verifying {venv} ==\n")

    rows: list[tuple[str, str, str]] = []
    vlvl, vmsg, vver = check_vllm(venv)
    rows.append(("vllm", vlvl, vmsg))
    for src, lvl, msg in check_patches(venv, vver):
        rows.append(("patch:" + src, lvl, msg))
    rows.append(("gpu",) + check_gpu(vver))
    rows.append(("cuda13_shim",) + check_cuda13_shim(venv, vver))
    rows.append(("msvc",) + check_msvc())

    bad_any = any(lvl == "RED" for _, lvl, _ in rows)
    yellow = any(lvl == "YEL" for _, lvl, _ in rows)
    width = max(len(name) for name, *_ in rows) + 2
    for name, lvl, msg in rows:
        sym = {"GRN": "OK ", "YEL": "WRN", "RED": "ERR"}[lvl]
        print(f"  [{sym}] {name.ljust(width)} {msg}")

    if bad_any:
        print("\nFAIL — fix RED items before launching.")
        return 1
    if yellow:
        print("\nOK with warnings — review WRN items.")
        return 2
    print("\nALL GREEN — ready to launch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
