"""Sanity-check: vLLM install, devnen wheel tag, GPU present.

Run after install / before launch. Prints a green / yellow / red summary.

Checks:
  1. vLLM importable, version is 0.19.0+devnen.* or 0.20.0+cu132.devnen.*.
     The PEP 440 local-version segment (`+devnen.*` / `+cu132.devnen.*`)
     is the proof that the devnen Windows patches (wildcard model name,
     reasoning parser, etc.) are baked into this wheel, they live in
     the engine fork, not as runtime overlay files.
  2. nvidia-smi reports at least one Ampere+ GPU (sm_86 or higher).
     Blackwell (sm_120) is accepted but warns if the installed wheel is
     a cu126 build, and Ampere/Ada warn if the wheel is a cu130 build.
  3. CUDA 13 runtime shim present when running on a +cu13* wheel.
  4. MSVC `cl.exe` resolvable (warn if not, only matters for FlashInfer JIT).

Exit code 0 = all green. 1 = at least one red. 2 = warnings only.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def check_vllm(venv: Path) -> tuple[str, str, str]:
    """Return (level, message, version_string). Empty version on failure."""
    # Developer venv layout: <root>/Scripts/python.exe
    # Portable embedded Python layout: <root>/python.exe (no Scripts/)
    py = venv / "Scripts" / "python.exe"
    if not py.exists():
        py = venv / "python.exe"
    if not py.exists():
        return ("RED", f"no python.exe under {venv}", "")
    try:
        out = subprocess.check_output(
            [str(py), "-c", "import vllm; print(vllm.__version__)"],
            text=True, timeout=30,
        ).strip()
    except subprocess.CalledProcessError as e:
        return ("RED", f"vllm import failed: {e}", "")
    if out.startswith("0.19"):
        return ("GRN", f"vllm {out} (Ampere/Ada, cu126 / 30+40 series)", out)
    if out.startswith("0.20"):
        return ("GRN", f"vllm {out} (Blackwell-capable, cu13x / 30+40+50 series)", out)
    return ("YEL", f"unexpected version {out!r}, expected 0.19.x or 0.20.x", out)


def check_devnen_tag(vllm_version: str) -> tuple[str, str]:
    """Confirm the wheel carries a devnen local-version tag.

    The devnen patches (wildcard `served-model-name`, qwen3 reasoning
    parser, and on 0.19 also the CPU-relay distributed shims) are baked
    into the wheel by the engine fork, not applied at install time. The
    only at-runtime evidence they're present is the PEP 440 local-version
    segment on `vllm.__version__`.
    """
    if not vllm_version:
        return ("RED", "no version string to check")
    if "+" not in vllm_version:
        return ("RED", f"upstream wheel {vllm_version!r}, devnen patches "
                "(wildcard model name, qwen3 reasoning) are NOT applied. "
                "Reinstall from the launcher zip's bundled wheel.")
    local = vllm_version.split("+", 1)[1]
    # 0.19 line: +devnen.N
    # 0.20 line: +cu132.devnen.N
    if "devnen" in local:
        return ("GRN", f"devnen wheel tag '+{local}', patches baked in")
    return ("YEL", f"local-version '+{local}' is not a known devnen tag; "
            "this wheel may be a SystemPanic upstream build without the "
            "wildcard model-name and reasoning-parser patches")


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
    if blackwell and vllm_version.startswith("0.19"):
        return ("YEL", f"Blackwell GPU {blackwell} but cu126 wheel installed, "
                "use the qwen3.6-windows-server-portable-x64-blackwell.zip release.")
    if not blackwell and vllm_version.startswith("0.20"):
        return ("GRN", f"{' | '.join(lines)} (running cu13x wheel)")
    return ("GRN", " | ".join(lines))


def check_cuda13_shim(venv: Path, vllm_version: str) -> tuple[str, str]:
    """Confirm the launcher-built CUDA 13 shim is present when needed."""
    if not vllm_version.startswith("0.20"):
        return ("GRN", "n/a (cu126 wheel)")
    candidates = [
        REPO / "cuda13_shim" / "bin" / "cudart64_13.dll",
        venv.parent / "cuda13_shim" / "bin" / "cudart64_13.dll",
    ]
    for c in candidates:
        if c.is_file():
            return ("GRN", f"found at {c.parent}")
    return ("YEL", "cuda13_shim/bin/cudart64_13.dll missing, "
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
    return ("YEL", "MSVC not found, fine for TRITON_ATTN; FlashInfer JIT would fail")


def _default_venv() -> Path:
    """Mirror snapshots/_common._resolve_vllm_exe() priority.

    The portable release installs vLLM directly into the embedded Python's
    site-packages, so the runtime root is ``REPO/python``, not ``REPO/venv``.
    Developer checkouts use a real venv. Pick whichever has python.exe.
    """
    dev = REPO / "venv"
    if (dev / "Scripts" / "python.exe").exists():
        return dev
    portable = REPO / "python"
    if (portable / "Scripts" / "python.exe").exists() or (portable / "python.exe").exists():
        return portable
    return dev  # report the missing dev venv if neither exists


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default=str(_default_venv()))
    args = ap.parse_args()

    venv = Path(args.venv)
    print(f"== verifying {venv} ==\n")

    rows: list[tuple[str, str, str]] = []
    vlvl, vmsg, vver = check_vllm(venv)
    rows.append(("vllm", vlvl, vmsg))
    rows.append(("devnen_tag",) + check_devnen_tag(vver))
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
        print("\nFAIL, fix RED items before launching.")
        return 1
    if yellow:
        print("\nOK with warnings, review WRN items.")
        return 2
    print("\nALL GREEN, ready to launch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
