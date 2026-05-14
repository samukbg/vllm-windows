"""Shared resolution for snapshot scripts.

Each ``start_*.py`` imports VENV / MODEL_PATH / VCVARS from here so users
can override paths via environment variables instead of editing every
snapshot. Resolution order:

* ``VLLM_WINDOWS_VENV`` env var, else repo-root ``venv``.
* ``VLLM_MODEL_DIR`` env var, else repo-root ``models/Qwen3.6-27B-int4-AutoRound``.
* ``VLLM_WINDOWS_VCVARS`` env var, else the most likely VS 2022 install.
* ``VLLM_WINDOWS_LOGS`` env var, else repo-root ``logs``. Created on first use.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _add_bundled_ninja_to_path() -> None:
    """Make the bundled ninja.exe (shipped via build_launcher_zip.py)
    discoverable by ``shutil.which`` and child-process subprocesses.

    ``pip install --target python\\Lib\\site-packages ninja`` puts the
    executable at ``python\\Lib\\site-packages\\bin\\ninja.exe``. That
    path is not on PATH by default, so flashinfer's JIT shells out to
    ``ninja`` and gets ``FileNotFoundError``. Prepending it here once,
    at import, propagates into every snapshot's ``env = os.environ.copy()``.
    """
    bundled = REPO_ROOT / "python" / "Lib" / "site-packages" / "bin"
    if (bundled / "ninja.exe").exists():
        current = os.environ.get("PATH", "")
        if str(bundled) not in current:
            os.environ["PATH"] = f"{bundled};{current}"


_add_bundled_ninja_to_path()

def _resolve_vllm_exe() -> tuple[Path, Path]:
    """Resolve (PYTHON_HOME, VLLM_EXE).

    Resolution order:
      1. ``VLLM_WINDOWS_VENV`` env var → expects ``Scripts/vllm.exe`` and
         ``Scripts/python.exe`` underneath (developer / external venv).
      2. ``REPO_ROOT/venv/Scripts/vllm.exe`` (developer checkout).
      3. ``REPO_ROOT/python/Scripts/vllm.exe`` (portable release: vllm
         is installed directly into the embedded Python's site-packages
         by ``launcher/app/setup.py``, which writes the entry-point exe
         under ``python/Scripts/``).
    """
    env = os.environ.get("VLLM_WINDOWS_VENV")
    if env:
        root = Path(env)
        return root, root / "Scripts" / "vllm.exe"
    dev_venv = REPO_ROOT / "venv"
    if (dev_venv / "Scripts" / "vllm.exe").exists():
        return dev_venv, dev_venv / "Scripts" / "vllm.exe"
    embedded = REPO_ROOT / "python"
    return embedded, embedded / "Scripts" / "vllm.exe"


VENV, VLLM_EXE = _resolve_vllm_exe()

def _resolve_model_path() -> str:
    """Resolve model dir using the same priority as the launcher.

    1. $VLLM_MODEL_DIR (explicit override).
    2. user_config.json["model_dir"] at repo root (written by the launcher
       once it finds/downloads the model, typically not next to the repo).
    3. Repo-root default ``models/Qwen3.6-27B-int4-AutoRound``.
    """
    env = os.environ.get("VLLM_MODEL_DIR")
    if env:
        return env
    cfg_path = REPO_ROOT / "user_config.json"
    if cfg_path.exists():
        try:
            import json
            saved = json.loads(cfg_path.read_text(encoding="utf-8")).get("model_dir")
            if saved and Path(saved).exists():
                return str(saved)
        except Exception:
            pass
    return str(REPO_ROOT / "models" / "Qwen3.6-27B-int4-AutoRound")


MODEL_PATH = _resolve_model_path()


# vswhere.exe lives under the VS Installer dir, NOT on PATH by default.
# vcvars64.bat shells out to vswhere.exe with no path qualifier, so without
# this prefix the snapshot logs start with a scary
# `'vswhere.exe' is not recognized as an internal or external command` line
# even though vcvars then falls through to a working VS install.
VS_INSTALLER_DIR = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer"


def _vswhere_vcvars() -> str | None:
    """Ask vswhere.exe for the latest VS install with the VC++ x64 toolset.

    Works for VS 2022 (17.x), VS 2026 (18.x), BuildTools, and any future
    layout Microsoft ships, since the path comes from the installer
    catalog rather than a hardcoded year. Returns None if vswhere is
    missing, returns no install, or the install lacks vcvars64.bat.
    """
    vswhere = Path(VS_INSTALLER_DIR) / "vswhere.exe"
    if not vswhere.exists():
        return None
    try:
        out = subprocess.check_output(
            [
                str(vswhere), "-latest", "-prerelease", "-format", "json",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
            ],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # vswhere with -property emits one path per line, not JSON, but with
    # -format json it emits a JSON array of objects. Handle both.
    out = out.strip()
    install_path: str | None = None
    if out.startswith("["):
        try:
            data = json.loads(out)
            if data and isinstance(data, list):
                install_path = data[0].get("installationPath")
        except (ValueError, AttributeError, IndexError):
            install_path = None
    elif out:
        install_path = out.splitlines()[0].strip()
    if not install_path:
        return None
    candidate = Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return str(candidate) if candidate.exists() else None


def _find_vcvars() -> str:
    env = os.environ.get("VLLM_WINDOWS_VCVARS")
    if env and Path(env).exists():
        return env
    found = _vswhere_vcvars()
    if found:
        return found
    # Fallback: probe the well-known VS 2022 install locations directly.
    # Only used when vswhere is missing or returns no result. Year-numbered
    # paths bake in a release schedule that bites every time Microsoft
    # ships a new version (VS 2026 = version 18 lives under \2026\, not
    # \2022\), so vswhere is preferred.
    candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return candidates[2]  # community is the most common, return so caller can warn


VCVARS = _find_vcvars()


def _count_visible_gpus() -> int:
    """Count NVIDIA GPUs reported by nvidia-smi.

    Returns 0 when nvidia-smi is missing or fails, caller should treat
    that as 'unknown, don't second-guess the snapshot defaults'.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.splitlines() if line.strip())


def print_insufficient_gpus_banner(wanted: int, visible: int) -> None:
    """Loud banner for the multi-GPU-snapshot-on-single-GPU-box case.

    Mirrors ``print_port_collision_banner`` so all hard-fail-before-vLLM
    paths share one visual style. Without this, ``CUDA_VISIBLE_DEVICES=0,1``
    on a 1-GPU host crashes vLLM ~30 s into boot with an opaque
    ``invalid device ordinal`` traceback that's easy to misread as a
    driver problem.
    """
    bar = "=" * 60
    print("", flush=True)
    print(bar, flush=True)
    print(f"  THIS SNAPSHOT NEEDS {wanted} GPUs, BUT ONLY {visible} IS VISIBLE", flush=True)
    print(bar, flush=True)
    print("  Multi-GPU snapshots (PP=2 / TP=2) require two NVIDIA GPUs.", flush=True)
    print("  On a single-GPU box, pick one of the single-GPU snapshots", flush=True)
    print("  from the launcher dashboard instead, for example:", flush=True)
    print("    start_speed   (64 tok/s @ 90k ctx, MTP n=6)", flush=True)
    print("    start_127k    (53 tok/s @ 127k ctx, MTP n=3)", flush=True)
    print("    start_gpu0_50k (GPU 0 only, conservative defaults)", flush=True)
    print("  Or edit this snapshot's TP/PP via Edit Snapshots (E) in", flush=True)
    print("  the launcher.", flush=True)
    print(bar, flush=True)


def resolve_cuda_visible_devices(preferred_single: str, world_size: int) -> str | None:
    """Pick CUDA_VISIBLE_DEVICES with a one-GPU fallback.

    The 2× 3090 snapshots default to pinning the single-card path to GPU 1
    so GPU 0 stays free for the display compositor and other work. On a
    single-GPU box (or any host where nvidia-smi reports fewer GPUs than
    the snapshot expects), that pin is wrong and vLLM dies with no useful
    error.

    Single-GPU snapshots fall back to GPU 0 (with an info note about the
    mem_util ceiling that GPU 0 + display attached implies). Multi-GPU
    snapshots have no graceful fallback, print the banner and return
    ``None`` so the caller bails before launching vLLM.

    Args:
        preferred_single: ``CUDA_VISIBLE_DEVICES`` value the snapshot
            wants when running TP=PP=1 (usually ``"1"`` or ``"0"``).
        world_size: TP * PP. Multi-GPU snapshots want ``"0,1"``.

    Returns:
        The resolved ``CUDA_VISIBLE_DEVICES`` string, or ``None`` when
        the snapshot needs more GPUs than the host has. Callers must
        treat ``None`` as a hard exit (banner has already been printed,
        and an ``input()`` pause held the WT tab open).
    """
    visible = _count_visible_gpus()
    if world_size > 1:
        if visible and visible < world_size:
            print_insufficient_gpus_banner(world_size, visible)
            try: input("Press Enter to close...")
            except EOFError: pass
            return None
        return ",".join(str(i) for i in range(max(world_size, 1)))
    try:
        wanted_idx = int(preferred_single.split(",")[0])
    except ValueError:
        wanted_idx = 0
    if visible and wanted_idx >= visible:
        print(
            f"[info] snapshot prefers GPU {wanted_idx} but only {visible} "
            f"GPU(s) visible, falling back to GPU 0. If GPU 0 has a "
            f"display attached, the safety check may trip on "
            f"--gpu-memory-utilization >= 0.95; lower to 0.92 via the "
            f"launcher's Edit Snapshots screen if you see "
            f"'Free memory on device cuda:0 ... is less than desired'.",
            file=sys.stderr,
        )
        return "0"
    return preferred_single


def _latest_vctools_version(vcvars_path: str) -> str | None:
    """Find the newest MSVC toolset version installed alongside vcvars64.bat.

    vcvars64.bat without ``VCToolsVersion`` set picks an unspecified
    default toolset, which on systems with multiple VS Build Tools
    side-by-side (e.g. 14.38 + 14.44) is frequently the older one. The
    older 14.38 ``cl.exe`` rejects empty-brace initializers (``T x = {};``)
    that Triton's generated ``__triton_launcher.c`` emits, breaking the
    fused_gdn_gating JIT compile on Qwen3-Next with:

        error C2059: syntax error: '}'

    Pinning ``VCToolsVersion`` to the highest available toolset
    sidesteps the bug. Layout:
        <install>\\VC\\Auxiliary\\Build\\vcvars64.bat   (VCVARS)
        <install>\\VC\\Tools\\MSVC\\<version>\\          (one per toolset)

    Returns the highest version directory name, or None when the layout
    isn't recognized.
    """
    try:
        vc_root = Path(vcvars_path).parents[2]
        tools_dir = vc_root / "Tools" / "MSVC"
        if not tools_dir.is_dir():
            return None
        versions = []
        for p in tools_dir.iterdir():
            if not p.is_dir():
                continue
            parts = p.name.split(".")
            if len(parts) >= 2 and all(s.isdigit() for s in parts[:2]):
                key = tuple(int(s) for s in parts if s.isdigit())
                versions.append((key, p.name))
        if not versions:
            return None
        versions.sort()
        return versions[-1][1]
    except (OSError, ValueError):
        return None


def msvc_env() -> dict:
    """Capture env vars set by vcvars64.bat so FlashInfer's ninja+cl.exe JIT works.

    Without this, fp8 KV hits a FileNotFoundError when FlashInfer tries to
    compile a new prefill kernel at first request. Best-effort: shipped
    snapshots use TRITON_ATTN, not FlashInfer JIT, so failure here is a
    warning, not a fatal error.

    Also pins ``VCToolsVersion`` to the newest installed MSVC toolset to
    avoid 14.38's stricter C parser tripping on Triton's generated
    ``__triton_launcher.c`` (issue #7).
    """
    if not Path(VCVARS).exists():
        print(f"[warn] vcvars64.bat not found at {VCVARS} - FlashInfer JIT may fail.")
        return {}
    path_prefix = f'set "PATH={VS_INSTALLER_DIR};%PATH%" && ' if Path(VS_INSTALLER_DIR).is_dir() else ""
    latest_tools = _latest_vctools_version(VCVARS)
    tools_prefix = f'set "VCToolsVersion={latest_tools}" && ' if latest_tools else ""
    try:
        out = subprocess.check_output(
            f'cmd /S /C "{path_prefix}{tools_prefix}"{VCVARS}" && set"',
            text=True, errors="replace", stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        print(
            f"[warn] vcvars64.bat invocation failed ({e.__class__.__name__}); "
            f"continuing without MSVC env. TRITON_ATTN snapshots are unaffected; "
            f"FlashInfer JIT (if used) may fail at first request.",
            file=sys.stderr,
        )
        return {}
    env = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env

def _ensure_cudart_alias(cuda_path: str | Path) -> None:
    """Create cudart64_120.dll as a copy of cudart64_12.dll if missing.

    flashinfer's ``jit/__init__.py`` does an absolute-path
    ``ctypes.CDLL("<CUDA_PATH>/bin/cudart64_120.dll")`` at import time.
    NVIDIA's CUDA 12.x toolkit ships the file as ``cudart64_12.dll`` (the
    naming changed between 11.x's ``cudart64_110.dll`` and 12.x's
    single-major form), so flashinfer crashes EngineCore on every modern
    install with:

        FileNotFoundError: Could not find module
        '...\\CUDA\\v12.6\\bin\\cudart64_120.dll'

    Best-effort fix: copy the real DLL alongside under the expected name.
    The CUDA install dir is usually under Program Files (admin write), so
    the copy can fail with PermissionError. In that case we print a
    one-line copy command the user can run from an admin prompt, same
    workaround as before, just surfaced at the right moment.
    """
    try:
        bin_dir = Path(cuda_path) / "bin"
        target = bin_dir / "cudart64_120.dll"
        if target.exists():
            return
        source = bin_dir / "cudart64_12.dll"
        if not source.exists():
            return
        try:
            shutil.copyfile(source, target)
            print(f"[info] aliased {source.name} -> {target.name} "
                  "(flashinfer hardcodes the old name)", flush=True)
        except OSError as e:
            print(
                f"[warn] flashinfer expects {target.name} but only "
                f"{source.name} is present, and the alias copy failed "
                f"({e.__class__.__name__}). Run this once from an "
                f"elevated cmd to fix permanently:\n"
                f'       copy "{source}" "{target}"',
                file=sys.stderr, flush=True,
            )
    except Exception:
        pass


def _torch_cuda_major() -> int | None:
    """Return torch.version.cuda's major version, or None when import fails."""
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    cu = (getattr(torch.version, "cuda", "") or "").split(".")
    if cu and cu[0].isdigit():
        return int(cu[0])
    return None


def _cuda13_shim_dir() -> Path | None:
    """Return the launcher-built CUDA 13 shim dir if it exists.

    setup.py builds <REPO_ROOT>/cuda13_shim/bin/cudart64_13.dll (and
    cousins) on cu13x wheel installs by copying from torch/lib. This
    helper just locates that dir so cuda_env() can prefer it over the
    system CUDA Toolkit when flashinfer needs cudart64_13.dll.
    """
    cand = REPO_ROOT / "cuda13_shim"
    if (cand / "bin" / "cudart64_13.dll").is_file():
        return cand
    return None


def cuda_env() -> dict:
    """Set CUDA_PATH / CUDA_LIB_PATH so flashinfer's import-time check passes.

    flashinfer reads CUDA_HOME / CUDA_ROOT / CUDA_PATH / CUDA_LIB_PATH (in
    that order), then loads ``<root>/bin/cudart64_<N>.dll`` where N is
    derived from ``torch.version.cuda``. On a CUDA 13 torch (cu130 wheel,
    Blackwell-supporting), N is "13", and most Windows boxes only have
    CUDA 12.x installed, so the import dies.

    Resolution order:
      1. If torch is cu13 and the launcher built a CUDA 13 shim (see
         launcher/app/setup.py:_ensure_cuda13_shim), point CUDA_PATH at
         the shim. This works without any system CUDA Toolkit installed.
      2. Existing ``CUDA_LIB_PATH`` from the environment.
      3. ``CUDA_PATH`` / ``CUDA_HOME`` from the environment, but only
         when the matching cudart DLL is present at <path>/bin/.
      4. Newest CUDA Toolkit under
         ``C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\``.
      5. Fallback: the embedded Python dir. The fake path satisfies
         flashinfer's non-empty check; JIT paths will fail loudly later.
    """
    cuda_major = _torch_cuda_major()
    expected_dll = f"cudart64_{cuda_major if cuda_major and cuda_major >= 12 else 12}.dll"

    # Step 1: prefer the launcher's CUDA 13 shim when running cu13 torch.
    if cuda_major and cuda_major >= 13:
        shim = _cuda13_shim_dir()
        if shim is not None:
            return {
                "CUDA_PATH": str(shim),
                "CUDA_LIB_PATH": str(shim),
            }

    if os.environ.get("CUDA_LIB_PATH"):
        _ensure_cudart_alias(os.environ["CUDA_LIB_PATH"])
        return {}

    def _has_expected_dll(root: str | Path) -> bool:
        return (Path(root) / "bin" / expected_dll).is_file()

    for var in ("CUDA_PATH", "CUDA_HOME"):
        val = os.environ.get(var)
        if val and Path(val).is_dir() and _has_expected_dll(val):
            _ensure_cudart_alias(val)
            return {"CUDA_LIB_PATH": val}
    nvidia_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if nvidia_root.is_dir():
        versions = sorted(
            (p for p in nvidia_root.iterdir() if p.is_dir() and p.name.startswith("v")),
            key=lambda p: p.name,
            reverse=True,
        )
        for v in versions:
            if _has_expected_dll(v):
                _ensure_cudart_alias(v)
                return {"CUDA_LIB_PATH": str(v), "CUDA_PATH": str(v)}
    print(
        f"[warn] No CUDA Toolkit found with {expected_dll} (checked CUDA_PATH, "
        r"CUDA_HOME, and C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\). "
        "Setting CUDA_LIB_PATH to a placeholder so flashinfer's import "
        "check passes. Shipped snapshots use TRITON_ATTN so this is "
        "fine, but FlashInfer JIT paths (fp8 prefill kernel build) "
        "will fail. Install CUDA Toolkit to fully clear this.",
        file=sys.stderr,
    )
    return {"CUDA_LIB_PATH": str(REPO_ROOT / "python")}


_BAD_PATH_FRAGMENTS = (
    # System NVIDIA toolkits (any version). The cu13 shim we ship has the
    # cudart/cublas/nvrtc DLLs flashinfer needs. Any other toolkit on PATH
    # can shadow them and trigger flashinfer's "SM 12.x requires CUDA >= 12.9"
    # silent fallback, which then bakes slow tactics into vLLM's compile cache.
    "NVIDIA GPU Computing Toolkit",
    "NVIDIA Corporation\\CUDA",
    # Conda/Anaconda/Miniconda's Library/bin can carry cudatoolkit DLLs.
    "Anaconda3\\Library\\bin",
    "anaconda3\\Library\\bin",
    "Miniconda3\\Library\\bin",
    "miniconda3\\Library\\bin",
    "miniforge3\\Library\\bin",
    "mambaforge\\Library\\bin",
)
_CUDA_ENV_PREFIXES = ("CUDA_", "NVCC_", "NVTOOLSEXT", "CUDNN_")


def _find_or_make_spaceless_cuda13() -> Path | None:
    """Return a Path to a CUDA 13.x toolkit at a path with no spaces.

    FlashInfer's JIT generates ``build.ninja`` with include paths inherited
    from ``CUDA_PATH``. The generated ninja passes those paths to ``cl.exe``
    UNQUOTED. If CUDA_PATH contains spaces (the Windows default install at
    ``C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.x``), cl
    splits on the space and fails to find headers (``cublasLt.h``). The
    ``trtllm_utils.dll`` build loops indefinitely on every boot (2-3 h
    observed by users), so vLLM never reaches "Application startup
    complete." See GitHub issue #18 for the full trace.

    This helper finds a real CUDA 13.x toolkit and, if it's at a spaced
    path, creates a junction at ``C:\\cuda13<N>`` so ``cl.exe`` sees a
    space-free include path. Junctions don't need admin on Windows.

    Returns the spaceless Path, or None if no CUDA 13.x toolkit is found.
    """
    # 1) Honor an existing spaceless junction first.
    for cand in (Path(r"C:\cuda132"), Path(r"C:\cuda131"), Path(r"C:\cuda13")):
        if (cand / "bin" / "nvcc.exe").is_file():
            return cand

    # 2) Find a real CUDA 13.x toolkit under the standard NVIDIA install root.
    sys_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if not sys_root.is_dir():
        return None
    real: Path | None = None
    for v in sorted(sys_root.iterdir(), reverse=True):
        if v.name.startswith("v13.") and (v / "bin" / "nvcc.exe").is_file():
            real = v
            break
    if real is None:
        return None

    # 3) If the real path has no spaces, just use it.
    if " " not in str(real):
        return real

    # 4) Otherwise create a junction at C:\cuda13<minor> for spaceless access.
    minor = real.name[len("v13."):]
    target = Path(rf"C:\cuda13{minor}")
    if target.exists():
        return target if (target / "bin" / "nvcc.exe").is_file() else None
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(real)],
            check=True, capture_output=True,
        )
    except Exception as e:
        print(f"[warn] could not create CUDA 13 junction at {target} -> {real}: {e!r}",
              file=sys.stderr)
        return None
    return target if (target / "bin" / "nvcc.exe").is_file() else None


def clean_cuda_env(base: "dict[str, str] | None" = None) -> "dict[str, str]":
    """Build a clean env for the vLLM subprocess, scrubbing system CUDA
    pollution.

    Returns a dict suitable for ``subprocess.Popen(..., env=...)``:

    - All ``CUDA_*`` / ``NVCC_*`` / ``NVTOOLSEXT*`` / ``CUDNN_*`` keys
      removed (system installer + conda set these and any one can
      poison flashinfer's runtime probe).
    - PATH filtered to drop every system NVIDIA toolkit dir and every
      conda Library/bin we recognise.
    - The bundled ``cuda13_shim/bin`` prepended to PATH (when on a cu13
      torch wheel).
    - ``CUDA_PATH`` and ``CUDA_HOME`` set to the shim.

    The four user environment classes this hardens against:

      1. Pure inference users, drivers only, no system CUDA. Already
         worked; this is a no-op for them.
      2. Devs with CUDA 12.x installed for some other tool. The
         critical one, system 12.4 was the cache-poison vector.
      3. Devs with CUDA 13.x installed. Worked accidentally before;
         we still scrub so behaviour is uniform across machines.
      4. Conda/Miniconda/Mamba users with cudatoolkit on PATH.

    See ``_local/CACHE_POISON_INCIDENT_2026-05-07.md`` for the
    incident this prevents and the runtime fingerprint.

    The legacy ``cuda_env()`` helper above is kept untouched for
    Ampere/Ada snapshots that may still depend on its add-only
    semantics. Blackwell snapshots should adopt this newer helper.
    """
    base = dict(base if base is not None else os.environ)

    # 1) Drop every CUDA-flavoured key inherited from the host.
    for k in list(base):
        if any(k.upper().startswith(p) for p in _CUDA_ENV_PREFIXES):
            del base[k]

    # 2) Filter PATH.
    path_parts = base.get("PATH", "").split(os.pathsep)
    keep = [p for p in path_parts
            if p and not any(b.lower() in p.lower() for b in _BAD_PATH_FRAGMENTS)]

    # 3) Prefer a real spaceless CUDA 13.x toolkit when available -- gives
    #    FlashInfer's JIT a working nvcc + headers AND a space-free path
    #    that cl.exe won't mis-tokenize. Fall back to the bundled cu13 shim
    #    when no real toolkit is found (inference-only setups). See
    #    _find_or_make_spaceless_cuda13() docstring + GitHub issue #18.
    cuda_major = _torch_cuda_major()
    if cuda_major and cuda_major >= 13:
        real = _find_or_make_spaceless_cuda13()
        if real is not None:
            base["CUDA_PATH"] = str(real)
            base["CUDA_HOME"] = str(real)
            real_bin = str(real / "bin")
            if real_bin not in keep:
                keep = [real_bin] + keep
        else:
            shim = _cuda13_shim_dir()
            if shim is not None:
                base["CUDA_PATH"] = str(shim)
                base["CUDA_HOME"] = str(shim)
                shim_bin = str(shim / "bin")
                if shim_bin not in keep:
                    keep = [shim_bin] + keep

    base["PATH"] = os.pathsep.join(keep)
    return base


def preflight_sm120a_or_die(env: "dict[str, str]", *, vllm_python: "Path | str") -> None:
    """Spawn a 5-second subprocess under ``env`` and verify FlashInfer
    can dispatch to ``sm_120a`` on cuda:0. Hard-exit otherwise.

    This catches the exact silent-degradation mode that bit us on
    2026-05-07: a polluted CUDA env let flashinfer log
    ``SM 12.x requires CUDA >= 12.9`` and continue, which then baked
    slow tactics into the AOT compile cache. If the env is wrong, we
    want to know in 5 seconds, not 11 minutes after warmup.

    Runs the probe in a *subprocess* (not in-process) so the launcher
    python's import state stays clean and the subprocess sees the
    exact env vLLM will see.
    """
    py = str(vllm_python)
    code = (
        "import sys\n"
        "try:\n"
        "    import torch\n"
        "    if not torch.cuda.is_available():\n"
        "        print('NO_CUDA', flush=True); sys.exit(2)\n"
        "    cap = torch.cuda.get_device_capability(0)\n"
        "    name = torch.cuda.get_device_name(0)\n"
        "    rt = torch.version.cuda\n"
        "    from flashinfer.utils import is_sm120a_supported\n"
        "    ok = bool(is_sm120a_supported(torch.device('cuda:0')))\n"
        "    print(f'PROBE cap={cap} cudart={rt} name={name!r} sm120a={ok}', flush=True)\n"
        "    sys.exit(0 if ok else 3)\n"
        "except Exception as e:\n"
        "    print(f'PROBE_FAIL {type(e).__name__}: {e}', flush=True)\n"
        "    sys.exit(4)\n"
    )
    print("[preflight] verifying FlashInfer sm_120a dispatch under clean env...", flush=True)
    try:
        out = subprocess.run(
            [py, "-c", code],
            env=env, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"[preflight] probe spawn failed ({type(e).__name__}: {e}); continuing without check.",
              file=sys.stderr, flush=True)
        return
    line = (out.stdout or out.stderr or "").strip().splitlines()
    last = line[-1] if line else ""
    if out.returncode == 0:
        print(f"[preflight] OK, {last}", flush=True)
        return
    # Failure: print everything we know, then bail.
    print("=" * 72, file=sys.stderr, flush=True)
    print("[preflight ERROR] FlashInfer sm_120a dispatch is unavailable.",
          file=sys.stderr, flush=True)
    print(f"[preflight ERROR] probe exit={out.returncode}, last_line={last!r}",
          file=sys.stderr, flush=True)
    if out.returncode == 3:
        print("[preflight ERROR] Likely cause: a system CUDA toolkit < 12.9 is",
              file=sys.stderr, flush=True)
        print("[preflight ERROR] shadowing the bundled cu13 shim. Even with the env",
              file=sys.stderr, flush=True)
        print("[preflight ERROR] scrub, an already-loaded cudart in this process tree",
              file=sys.stderr, flush=True)
        print("[preflight ERROR] can survive, close all terminals and relaunch from",
              file=sys.stderr, flush=True)
        print("[preflight ERROR] start.bat in a fresh shell.",
              file=sys.stderr, flush=True)
        print("[preflight ERROR]", file=sys.stderr, flush=True)
        print("[preflight ERROR] Diagnostic info to attach to a bug report:",
              file=sys.stderr, flush=True)
        print('[preflight ERROR]   where cudart64_13.dll', file=sys.stderr, flush=True)
        print('[preflight ERROR]   echo %CUDA_PATH%', file=sys.stderr, flush=True)
        print('[preflight ERROR]   echo %PATH%', file=sys.stderr, flush=True)
    elif out.returncode == 4:
        print("[preflight ERROR] Probe raised, check the message above.",
              file=sys.stderr, flush=True)
    print("[preflight ERROR] If this persists despite a clean shell, the wheel",
          file=sys.stderr, flush=True)
    print("[preflight ERROR] itself may not have compute_120f tactics compiled.",
          file=sys.stderr, flush=True)
    print("[preflight ERROR] See _local/CACHE_POISON_INCIDENT_2026-05-07.md.",
          file=sys.stderr, flush=True)
    print("=" * 72, file=sys.stderr, flush=True)
    sys.exit(1)


def cache_env_stamp_check(snapshot_py: "Path | None" = None) -> None:
    """Detect a stale vLLM AOT compile cache and warn loudly if found.

    vLLM's torch-inductor AOT cache (``~/.cache/vllm/``) bakes in the
    FlashInfer tactic IDs picked at graph-compile time. If the cache
    was populated under a polluted CUDA env (e.g. system CUDA 12.4 on
    PATH ahead of the cu13 shim), every later boot inherits the slow
    picks even after the env is fixed. The fingerprint is documented
    in ``_local/CACHE_POISON_INCIDENT_2026-05-07.md``.

    This check writes a small JSON stamp inside the vLLM cache the
    first time it sees a clean env, then on every subsequent boot
    compares the current env to the stamp. On mismatch it prints a
    big warning recommending ``windows_tools/wipe_caches.py``. Best
    effort: any failure is silent (we never want this to break boot).
    """
    try:
        import json as _json
        try:
            import torch  # type: ignore
            torch_cuda = getattr(torch.version, "cuda", "") or ""
            arch_list = list(torch.cuda.get_arch_list())
        except Exception:
            torch_cuda, arch_list = "", []
        try:
            import flashinfer  # type: ignore
            fi_version = getattr(flashinfer, "__version__", "")
        except Exception:
            fi_version = ""
        cuda_path = os.environ.get("CUDA_PATH", "") or os.environ.get("CUDA_LIB_PATH", "")
        snap_mtime = 0.0
        if snapshot_py is not None:
            try:
                snap_mtime = round(Path(snapshot_py).stat().st_mtime, 0)
            except OSError:
                pass

        current = {
            "torch_cuda": torch_cuda,
            "arch_list": arch_list,
            "flashinfer": fi_version,
            "cuda_path": cuda_path,
            "snapshot_py_mtime": snap_mtime,
            "version": 1,
        }

        cache_root = Path(os.environ.get("USERPROFILE", os.path.expanduser("~"))) / ".cache" / "vllm"
        stamp = cache_root / ".env_stamp.json"
        if not cache_root.is_dir():
            return  # nothing cached yet, nothing to check
        if not stamp.is_file():
            try:
                stamp.write_text(_json.dumps(current, indent=2), encoding="utf-8")
                print(f"[preflight] wrote vLLM cache env stamp: {stamp}", flush=True)
            except OSError:
                pass
            return
        try:
            previous = _json.loads(stamp.read_text(encoding="utf-8"))
        except Exception:
            return
        diffs = [k for k in current if previous.get(k) != current.get(k)]
        if not diffs:
            return  # quiet success, env matches what populated the cache
        print("=" * 72, flush=True)
        print("[preflight WARN] vLLM compile cache was populated under a different env.", flush=True)
        print(f"[preflight WARN]   stamp file: {stamp}", flush=True)
        for k in diffs:
            print(f"[preflight WARN]   {k:18s}: was={previous.get(k)!r}  now={current.get(k)!r}", flush=True)
        print("[preflight WARN]", flush=True)
        print("[preflight WARN] If you observe slow prefill (~750 tok/s) on a 47k NVFP4 prompt,", flush=True)
        print("[preflight WARN] the cache is poisoned. Recover with:", flush=True)
        print("[preflight WARN]   python windows_tools/wipe_caches.py", flush=True)
        print("[preflight WARN] See _local/CACHE_POISON_INCIDENT_2026-05-07.md for the full story.", flush=True)
        print("=" * 72, flush=True)
        # Refresh the stamp so the warning isn't repeated forever once the user accepts it.
        try:
            stamp.write_text(_json.dumps(current, indent=2), encoding="utf-8")
        except OSError:
            pass
    except Exception:
        # Preflight must never break boot.
        pass


def random_dp_rpc_port() -> int:
    """Return a random free TCP port for the data-parallel RPC handshake.

    vLLM 0.20.0 hardcodes ``data_parallel_rpc_port=29550`` in its
    ParallelConfig default. The parent binds it as ROUTER and engine-core
    children connect. When a child is orphaned (parent crashes mid-init)
    the port stays held until the orphan is killed, which makes back-to-
    back snapshot launches deterministically fail with
    ``ZMQError: Address in use (addr='tcp://127.0.0.1:29550')``.

    Snapshots pass the result of this helper to vllm via
    ``--data-parallel-rpc-port=<N>`` so each launch picks a fresh port and
    leaks aren't load-bearing.
    """
    import socket as _s
    with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def flashinfer_sampler_env(msvc_env_result: dict) -> dict:
    """Decide whether to enable flashinfer's sampler JIT path.

    vLLM 0.19's sampler can use flashinfer for top-k / top-p masking
    (small decode boost) when ``VLLM_USE_FLASHINFER_SAMPLER=1``. The
    catch: flashinfer JIT-compiles a sampling module on the first
    ``profile_run`` call, which shells out to ninja and cl.exe. If
    either is missing, EngineCore dies with ``FileNotFoundError``
    in ``run_ninja`` before the server ever accepts a request.

    Probe:
      - ``msvc_env_result`` non-empty means vcvars64.bat ran and we
        have a usable MSVC env in the subprocess (cl.exe will be on
        PATH for ninja).
      - ``ninja`` must be discoverable via ``shutil.which`` (it is
        usually pip-installed as a vLLM transitive dep, but the
        embedded Python may not have it on PATH).

    If both are satisfied, return ``{"VLLM_USE_FLASHINFER_SAMPLER": "1"}``.
    Otherwise, set it to ``"0"`` so vLLM falls back to the native
    PyTorch sampler. The fallback is slightly slower but never JIT-
    compiles anything, so boots succeed on a vanilla Windows install
    without MSVC.
    """
    if not msvc_env_result:
        print(
            "[info] MSVC env not available; setting "
            "VLLM_USE_FLASHINFER_SAMPLER=0 to skip flashinfer sampler "
            "JIT (would crash without cl.exe). Native PyTorch sampler "
            "is used instead, slightly slower but reliable.",
            file=sys.stderr,
        )
        return {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
    if not shutil.which("ninja"):
        print(
            "[info] ninja not on PATH; setting "
            "VLLM_USE_FLASHINFER_SAMPLER=0. The portable launcher "
            "ships ninja under python\\Lib\\site-packages\\bin; if "
            "you're seeing this in a developer venv, install ninja "
            "there or add the launcher's bundled path to PATH.",
            file=sys.stderr,
        )
        return {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
    return {"VLLM_USE_FLASHINFER_SAMPLER": "1"}


def _resolve_logs_dir() -> Path:
    """Return a writable logs dir.

    Honors $VLLM_WINDOWS_LOGS if set. Otherwise tries repo_root/logs, and
    falls back to %LocalAppData%\\qwen36-windows-server\\logs when the
    repo root is read-only (e.g. installs under Program Files).
    """
    env = os.environ.get("VLLM_WINDOWS_LOGS")
    if env:
        d = Path(env)
        d.mkdir(parents=True, exist_ok=True)
        return d
    candidate = REPO_ROOT / "logs"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return candidate
    except OSError:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        d = Path(base) / "qwen36-windows-server" / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d


LOGS_DIR = _resolve_logs_dir()


def log_path_for(port: int) -> Path:
    return LOGS_DIR / f"vllm_server.{port}.log"


# ---------------------------------------------------------------------------
# Runtime manifest, the launcher's source of truth for "what's running"
#
# Each snapshot writes <LOGS_DIR>/runtime/<port>.json on boot identifying
# itself by snapshot script filename. The launcher reads these to decide
# which card to light + enable Test/Unload, with no dependence on parsing
# localized netstat output or matching ambiguous --max-model-len cmdline
# fragments. Stale manifests are GC'd by the reader when port/pid checks
# fail. See launcher/app/manifest.py for the read side.
# ---------------------------------------------------------------------------

def runtime_dir() -> Path:
    d = LOGS_DIR / "runtime"
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path_for(port: int) -> Path:
    return runtime_dir() / f"{port}.json"


def write_manifest(*, snapshot_py: Path | str, port: int, wrapper_pid: int,
                   max_model_len: int | None = None,
                   mtp_n: int | None = None,
                   tp: int = 1, pp: int = 1) -> Path:
    """Atomically write <LOGS_DIR>/runtime/<port>.json with snapshot identity.

    snapshot_id is derived from the .py filename (e.g. start_speed.py ->
    "start_speed") which uniquely identifies the snapshot. The launcher
    matches this against the basename of each config's `bat`/`py` field.
    """
    import json, tempfile, datetime
    p = Path(snapshot_py)
    payload = {
        "snapshot_id": p.stem, # e.g. "start_speed"
        "snapshot_py": p.name, # e.g. "start_speed.py"
        "snapshot_bat": p.with_suffix(".bat").name, # e.g. "start_speed.bat"
        "port": int(port),
        "wrapper_pid": int(wrapper_pid),
        "max_model_len": max_model_len,
        "mtp_n": mtp_n,
        "tp": tp,
        "pp": pp,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    target = manifest_path_for(port)
    fd, tmp = tempfile.mkstemp(prefix=f".{port}.", suffix=".json", dir=str(runtime_dir()))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, target)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise
    return target


def print_port_collision_banner(port: int) -> None:
    """Print a loud, multi-line banner explaining a port-in-use exit.

    Snapshots use ``cmd /k`` so the WT tab stays open after the script
    returns, but a single ``[ERROR] Port X already in use`` line scrolls
    away in setup noise. This banner gives the user a clear "this is
    why" + "this is what to do" block right above the ``input()`` pause
    in each ``start_*.py``.
    """
    bar = "=" * 60
    print("", flush=True)
    print(bar, flush=True)
    print(f"  PORT {port} ALREADY IN USE", flush=True)
    print(bar, flush=True)
    print("  Another snapshot is already serving on this port.", flush=True)
    print("  Stop it from the launcher (Unload), or change this", flush=True)
    print("  snapshot's port via the launcher's Edit Snapshots screen.", flush=True)
    print(bar, flush=True)


def clear_manifest(port: int) -> None:
    """Remove the snapshot's runtime manifest.

    Called from the snapshot's `finally` block when the wrapper python
    exits (vLLM child died, Ctrl-C, port-collision exit). The diag log
    line lets us tell from `logs/launcher.diag.log` whether the wrapper
    ever exited and when, which was the missing data point in issue #12.
    """
    target = manifest_path_for(port)
    existed = target.exists()
    try:
        target.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
    try:
        import datetime as _dt
        line = (
            f"{_dt.datetime.now().isoformat(timespec='seconds')} "
            f"snapshot clear_manifest port={port} existed={existed}\n"
        )
        (_resolve_logs_dir() / "launcher.diag.log").open(
            "a", encoding="utf-8"
        ).write(line)
    except OSError:
        pass


# Path to the vendored Qwen3.5 enhanced jinja chat template. Lives under
# the vllm-windows repo (next to the patched wheel source). End-user
# portable installs symlink/copy it into ${VLLM_WINDOWS_TEMPLATES} so
# launcher zips can stand alone.
def enhanced_jinja_path() -> Path:
    env = os.environ.get("VLLM_WINDOWS_ENHANCED_JINJA")
    if env and Path(env).exists():
        return Path(env)
    candidates = [
        REPO_ROOT / "templates" / "qwen3.5-enhanced.jinja",
        REPO_ROOT.parent / "vllm-windows" / "templates" / "qwen3.5-enhanced.jinja",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Last resort, return the most likely portable layout (launcher next
    # to vllm-windows). Caller will print an error if it doesn't exist.
    return candidates[0]
