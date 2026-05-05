"""First-run runtime installer.

The portable zip ships only the launcher, the wheel, and an embedded Python
3.12 runtime. It does NOT pre-install vLLM and its ~150 transitive
dependencies (torch, cuda libs, transformers, ray, etc.) — that would push
the zip past the 2 GB GitHub release-asset cap.

Instead, on first run we install the bundled wheel into the embedded
Python's ``site-packages`` directly. No venv layer: the embedded runtime
is already private to this install, and skipping the venv avoids the
``ensurepip`` / ``venv``-stripped-from-embedded-python problem.

Pipeline:
  1. Try ``import vllm`` AND check the install-marker matches the bundled
     wheel's SHA256. If both pass, return.
  2. Locate ``wheels/vllm-*.whl`` (proper PEP 427 name) or legacy
     ``wheels/vllm.whl`` (back-compat with v0.1.4 and earlier zips).
  3. If pip is missing, bootstrap it with the vendored ``get-pip.py``
     (or download from https://bootstrap.pypa.io).
  4. ``pip install --extra-index-url <pytorch-cu126> <wheel>`` — pulls
     torch + ~150 deps from PyPI / pytorch.org. Several GB, several
     minutes. Resumable: rerunning skips wheels already cached.
  5. Write a marker JSON containing the wheel's SHA256 + version so a
     subsequent zip re-extract on top of an existing install correctly
     detects "different wheel, reinstall" instead of silently keeping
     the old one.

Runs in the parent cmd window before the Textual TUI mounts, so any
error is visible (no flashing-and-closing failure mode).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import paths

WHEEL_DIR_NAME = "wheels"
LEGACY_WHEEL_FILENAME = "vllm.whl"  # back-compat with v0.1.4 release zips
GET_PIP_VENDORED_NAME = "get-pip.py"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
# Default torch index. Overridden when the bundled wheel carries a cu13x
# local-version (Blackwell-supporting build), in which case we pull torch
# from the cu130 channel instead. See _torch_index_for_wheel.
TORCH_INDEX_CU126 = "https://download.pytorch.org/whl/cu126"
TORCH_INDEX_CU130 = "https://download.pytorch.org/whl/cu130"
TORCH_INDEX = TORCH_INDEX_CU126
# NuGet python package ships the C headers (Include/) and import libs
# (libs/python312.lib) that python.org's embeddable distribution strips.
# Triton's JIT needs both to compile cuda_utils.c at first model load.
# See issue #1: pre-v0.1.12 release zips ship without them.
PY_NUGET_URL_TMPL = "https://globalcdn.nuget.org/packages/python.{ver}.nupkg"

# ~6 GB of wheels download + extracted site-packages. Used for the
# preflight free-space check; not load-bearing if shutil.disk_usage fails.
EXPECTED_INSTALL_BYTES = 6 * 1024 * 1024 * 1024

MARKER_NAME = ".vllm_runtime_installed"


def _embedded_python() -> Path:
    return Path(sys.executable)


def _site_packages() -> Path:
    return _embedded_python().parent / "Lib" / "site-packages"


def _marker_path() -> Path:
    # Lives next to site-packages so it survives across re-extracts of
    # the zip (which would also wipe site-packages).
    return _site_packages() / MARKER_NAME


def _vllm_importable() -> bool:
    return importlib.util.find_spec("vllm") is not None


def _wheel_path() -> Path | None:
    """Return the bundled wheel, preferring a properly-named PEP 427 file.

    v0.1.5+ ships ``wheels/vllm-<version>-<tag>.whl`` directly. Older
    zips (v0.1.4 and earlier) shipped ``wheels/vllm.whl`` and renamed it
    at install time. Both layouts are accepted here.
    """
    wheels_dirs = [
        paths.install_root() / WHEEL_DIR_NAME,
        paths.writable_root() / WHEEL_DIR_NAME,
    ]
    # Prefer a real PEP 427 vllm-*.whl so we never need to rename.
    for d in wheels_dirs:
        if d.is_dir():
            for cand in sorted(d.glob("vllm-*.whl")):
                if cand.is_file():
                    return cand
    # Legacy fallback: vllm.whl needs renaming before pip will accept it.
    for d in wheels_dirs:
        legacy = d / LEGACY_WHEEL_FILENAME
        if legacy.is_file():
            return legacy
    return None


def _wheel_proper_filename(wheel: Path) -> Path:
    """Return a canonically-named copy of the wheel pip will accept.

    No-op when the wheel already has a PEP 427 filename. Otherwise reads
    version + tag from the wheel's METADATA / WHEEL files and writes a
    renamed copy under the writable root.
    """
    if wheel.name.startswith("vllm-") and wheel.name.endswith(".whl"):
        return wheel  # already properly named — nothing to do

    import zipfile
    version = None
    tag = None
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if name.endswith("METADATA") and version is None:
                for line in z.read(name).decode("utf-8", errors="replace").splitlines():
                    if line.startswith("Version: "):
                        version = line.split(": ", 1)[1].strip()
                        break
            if name.endswith("WHEEL") and tag is None:
                for line in z.read(name).decode("utf-8", errors="replace").splitlines():
                    if line.startswith("Tag: "):
                        tag = line.split(": ", 1)[1].strip()
                        break
            if version and tag:
                break

    if not version or not tag:
        version = version or "0.0.0"
        tag = tag or "py3-none-any"

    proper_name = f"vllm-{version}-{tag}.whl"
    dest = paths.writable_root() / WHEEL_DIR_NAME / proper_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file() or dest.stat().st_size != wheel.stat().st_size:
        shutil.copy2(wheel, dest)
    return dest


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_marker() -> dict:
    p = _marker_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_marker(payload: dict) -> None:
    try:
        _marker_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as e:
        # Non-fatal — site-packages may be read-only when install_root is
        # under Program Files. The next run will just re-import-check.
        print(f"  (note: could not write install marker: {e})")


def _pip_available() -> bool:
    return importlib.util.find_spec("pip") is not None


def _vendored_get_pip() -> Path | None:
    candidates = [
        paths.install_root() / WHEEL_DIR_NAME / GET_PIP_VENDORED_NAME,
        paths.install_root() / GET_PIP_VENDORED_NAME,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _download_get_pip() -> Path:
    dest = paths.writable_root() / GET_PIP_VENDORED_NAME
    print(f"  Downloading get-pip.py from {GET_PIP_URL} ...")
    req = urllib.request.Request(GET_PIP_URL, headers={"User-Agent": "qwen36-launcher/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as fh:
            fh.write(r.read())
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"failed to download get-pip.py: {e}") from e
    return dest


def _ensure_python_headers() -> None:
    """Ensure python\\Include\\ and python\\libs\\ exist next to the embed.

    The embeddable Python 3.12 zip from python.org strips both, but
    Triton's runtime/build.py needs Python.h (from Include/) and
    python312.lib (from libs/) to JIT-compile cuda_utils.c on the first
    model load. Without them, vLLM dies trying to inspect the model
    architecture.

    Build-time fix lives in windows_tools/build_launcher_zip.py and bakes
    the headers into the release zip directly. This runtime path
    self-heals existing v0.1.x installs that shipped before that build
    fix landed: download the NuGet python package, extract tools/include
    and tools/libs into the embed root.

    No-op when the headers are already present (re-extracted release
    zip, dev checkout against a real Python install, etc.). Best-effort
    on failure: prints a hint pointing to the manual workaround in
    docs/TROUBLESHOOTING.md and lets the caller continue. The wheel
    install can still succeed; the user just hits the JIT failure on
    first inference instead of here.
    """
    py_home = _embedded_python().parent
    have_h = (py_home / "Include" / "Python.h").is_file()
    have_lib = (py_home / "libs" / "python312.lib").is_file()
    if have_h and have_lib:
        return

    import platform
    py_ver = platform.python_version()  # e.g. "3.12.7"
    if not py_ver.startswith("3.12."):
        # Patch version doesn't have to match exactly, but the minor
        # version must. If we're somehow on a non-3.12 embed, bail
        # rather than overlay mismatched headers.
        print(f"  [warn] Python {py_ver} unexpected; skipping header self-heal.")
        return

    print()
    print("[setup] Embedded Python is missing C headers (Include/) and import libs (libs/).")
    print("        Triton needs both to JIT-compile its CUDA helpers on first model load.")
    print(f"        Downloading the NuGet python.{py_ver} package to overlay them.")

    url = PY_NUGET_URL_TMPL.format(ver=py_ver)
    cache_root = paths.writable_root() / WHEEL_DIR_NAME
    cache_root.mkdir(parents=True, exist_ok=True)
    nupkg = cache_root / f"python-{py_ver}.nupkg"

    try:
        if not nupkg.is_file() or nupkg.stat().st_size < 1_000_000:
            print(f"        GET {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "qwen36-launcher/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(nupkg, "wb") as fh:
                shutil.copyfileobj(r, fh)
    except (urllib.error.URLError, OSError) as e:
        print(f"  [warn] could not download NuGet python package: {e}")
        print("         Manual fix: see docs/TROUBLESHOOTING.md row 'Triton fails to JIT-compile'.")
        return

    import zipfile
    try:
        with zipfile.ZipFile(nupkg) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                if name.startswith("tools/include/"):
                    dst = py_home / "Include" / name[len("tools/include/"):]
                elif name.startswith("tools/libs/"):
                    dst = py_home / "libs" / name[len("tools/libs/"):]
                else:
                    continue
                try:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, open(dst, "wb") as out:
                        shutil.copyfileobj(src, out)
                except OSError as e:
                    # Hit when install root is read-only (Program Files
                    # without admin). Bail loudly so the user knows to
                    # extract elsewhere or run as admin once.
                    print(f"  [error] cannot write {dst}: {e}")
                    print("          Re-extract the launcher to a writable location")
                    print("          (Desktop, Documents, C:\\Tools\\, etc.) and try again.")
                    return
    except zipfile.BadZipFile:
        print(f"  [warn] downloaded NuGet package is corrupt: {nupkg}")
        try:
            nupkg.unlink()
        except OSError:
            pass
        return

    if (py_home / "Include" / "Python.h").is_file() and (py_home / "libs" / "python312.lib").is_file():
        print("[setup] Python headers + libs installed.")
    else:
        print("  [warn] header overlay finished but Python.h / python312.lib still missing.")


def _bootstrap_pip() -> None:
    src = _vendored_get_pip() or _download_get_pip()
    print(f"  Bootstrapping pip via {src.name} ...")
    r = subprocess.run(
        [sys.executable, str(src), "--no-warn-script-location"],
        capture_output=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"pip bootstrap failed (exit {r.returncode})")
    if not _pip_available():
        raise RuntimeError("pip bootstrap reported success but `import pip` still fails")


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def _torch_index_for_wheel(wheel: Path) -> str:
    """Pick the pytorch wheel index that matches the bundled wheel's CUDA build.

    The vLLM wheel filename's local-version suffix encodes the CUDA the wheel
    was built against:

        vllm-0.19.0+devnen.1-cp312-cp312-win_amd64.whl              -> cu126
        vllm-0.20.0+cu132-cp312-cp312-win_amd64.whl                 -> cu130
        vllm-0.20.0+cu132.devnen.1-cp312-cp312-win_amd64.whl        -> cu130

    Pulling torch from the wrong channel installs binaries that won't load
    against the wheel's CUDA runtime. cu130 torch is required for Blackwell
    (sm_120) anyway — the cu126 build has no sm_120 kernels.
    """
    name = wheel.name.lower()
    if "+cu13" in name:
        return TORCH_INDEX_CU130
    return TORCH_INDEX_CU126


def _install_wheel(wheel: Path) -> None:
    torch_index = _torch_index_for_wheel(wheel)
    print()
    print("  Installing vLLM and its ~150 transitive dependencies.")
    print(f"  torch index: {torch_index}")
    print("  Total download is multiple GB (torch + CUDA wheels + Python deps).")
    print("  Expect 5–15 minutes on a fast connection. Progress prints below.")
    print()
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--extra-index-url", torch_index,
        "--no-warn-script-location",
        str(wheel),
    ]
    r = subprocess.run(cmd, capture_output=False)
    if r.returncode != 0:
        raise RuntimeError(f"pip install failed (exit {r.returncode})")
    # vLLM 0.19.0+devnen.1's METADATA gates llguidance + xgrammar on
    # ``platform_machine == "x86_64"`` — Windows reports "AMD64" instead,
    # so pip's marker evaluator skips both. Install them explicitly so
    # the structured-output backend can import.
    print()
    print("  Installing Windows-marker-skipped extras (llguidance, xgrammar) ...")
    extras_cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-warn-script-location",
        "llguidance>=1.3.0,<1.4.0",
        "xgrammar>=0.1.32,<1.0.0",
    ]
    r = subprocess.run(extras_cmd, capture_output=False)
    if r.returncode != 0:
        raise RuntimeError(f"extras install failed (exit {r.returncode})")


def _print_banner() -> None:
    print("=" * 70)
    print(" qwen3.6-windows-server — first-run runtime install")
    print("=" * 70)
    print(f"  python:        {sys.executable}")
    print(f"  install root:  {paths.install_root()}")
    print(f"  writable root: {paths.writable_root()}")
    # Free-space preflight: warn loudly if the embedded python's drive
    # doesn't have room for the ~6 GB of wheels we're about to fetch and
    # extract into site-packages.
    try:
        target = _site_packages()
        target.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(target).free
        print(f"  disk free:     {_fmt_bytes(free)} on {Path(target.anchor or target).drive or target}")
        print(f"  expected use:  ~{_fmt_bytes(EXPECTED_INSTALL_BYTES)} (vLLM + torch + CUDA wheels)")
        if free < EXPECTED_INSTALL_BYTES:
            print()
            print(f"  [ERROR] Need ~{_fmt_bytes(EXPECTED_INSTALL_BYTES)} free, only {_fmt_bytes(free)} available.")
            print("          Free up space and re-run, or move the install to a roomier drive.")
            sys.exit(1)
        if free < EXPECTED_INSTALL_BYTES * 1.25:
            print("  [warn]         Disk is tight (<25% headroom). Pip download cache may fail mid-install.")
    except OSError:
        pass
    print(f"  est. time:     5–15 min on a fast (>200 Mbps) connection")
    print()


CUDA13_SHIM_DIRNAME = "cuda13_shim"
# Files we need from torch/lib/ to satisfy flashinfer's CDLL load on Windows
# when no real CUDA 13 toolkit is installed. Only cudart64_13.dll is
# load-bearing for the import-time check; the others are added defensively
# so JIT paths that go via cublas / nvrtc don't trip later.
CUDA13_SHIM_FILES = (
    "cudart64_13.dll",
    "cublas64_13.dll",
    "cublasLt64_13.dll",
    "nvrtc64_130_0.dll",
    "nvrtc-builtins64_130.dll",
)


def _is_cu13_torch_installed() -> bool:
    """Return True when the installed torch was built against CUDA 13."""
    try:
        import torch  # type: ignore
    except ImportError:
        return False
    cu = (getattr(torch.version, "cuda", "") or "").split(".")
    if not cu or not cu[0].isdigit():
        return False
    return int(cu[0]) >= 13


def _ensure_cuda13_shim(torch_lib: Path | None = None) -> Path | None:
    """Build a tiny CUDA 13 toolkit shim from torch's bundled DLLs.

    Why: vLLM 0.20.0 imports flashinfer at engine startup, which does
    ``ctypes.CDLL("$CUDA_PATH/bin/cudart64_13.dll")`` against ``CUDA_PATH``
    (or ``CUDA_HOME`` / ``CUDA_ROOT`` / ``CUDA_LIB_PATH``). A box that has
    only CUDA 12.x installed will fail this check even though the cu130
    torch wheel ships ``cudart64_13.dll`` under
    ``site-packages/torch/lib/``. Rather than ask end users to install the
    full CUDA 13 toolkit (~3 GB), copy the handful of CUDA 13 DLLs torch
    already ships into a dedicated shim folder, then point ``CUDA_PATH``
    at that folder at snapshot launch time (see snapshots/_common.py).

    No-op when:
      - torch is not yet importable (called too early in install).
      - torch is a cu12 build (existing 0.19 wheel; old launcher behavior
        applies).
      - the shim is already populated.

    Returns the shim directory (containing ``bin/<dlls>``) on success,
    or ``None`` when no shim is needed.
    """
    if not _is_cu13_torch_installed():
        return None
    if torch_lib is None:
        try:
            import torch  # type: ignore
        except ImportError:
            return None
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
    shim = paths.writable_root() / CUDA13_SHIM_DIRNAME
    bin_dir = shim / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for fname in CUDA13_SHIM_FILES:
        src = torch_lib / fname
        dst = bin_dir / fname
        if dst.is_file() and dst.stat().st_size == (src.stat().st_size if src.is_file() else 0):
            continue
        if not src.is_file():
            # Defensive: torch packaging changes occasionally renamed DLLs.
            # Skip if a non-essential one is missing; only cudart64_13 is
            # actually load-bearing.
            if fname == "cudart64_13.dll":
                print(f"  [warn] {src.name} missing in torch/lib; flashinfer will fail on first request.")
            continue
        shutil.copy2(src, dst)
        copied.append(fname)
    if copied:
        print(f"[setup] CUDA 13 shim built at {shim} ({len(copied)} DLLs copied from torch/lib)")
    else:
        print(f"[setup] CUDA 13 shim already populated at {shim}")
    return shim


def _check_gpu_supports_wheel(wheel: Path) -> tuple[str, str] | None:
    """Preflight: refuse install if GPU class doesn't match bundled wheel.

    Returns ``(level, message)`` to print, or ``None`` for a clean pass.
    Level is one of ``"FATAL"`` (refuse, exit 1) or ``"WARN"`` (proceed
    but tell the user what they're getting).

    Decision matrix:
      - bundled cu126 (Ampere/Ada zip), GPU is sm_120 (Blackwell): FATAL.
        wheel has no sm_120 kernels; engine fails at boot.
      - bundled cu132/cu130 (Blackwell zip), GPU is sm < 8.6: FATAL.
        too old regardless of wheel choice.
      - bundled cu132/cu130 (Blackwell zip), GPU is sm 8.6 / 8.9
        (Ampere/Ada): WARN. wheel works fine but user installed a CUDA-13
        driver upgrade for nothing — the cu126 zip would have been simpler.
      - GPU sm < 8.6: FATAL on any wheel.
    """
    name = wheel.name.lower()
    is_cu13_wheel = "+cu13" in name
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ("WARN", "could not run nvidia-smi to verify GPU; proceeding anyway.")
    gpus = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            n, cap = line.rsplit(",", 1)
            gpus.append((n.strip(), float(cap.strip())))
        except ValueError:
            continue
    if not gpus:
        return None
    max_cap = max(c for _, c in gpus)
    min_cap = min(c for _, c in gpus)
    summary = " | ".join(f"{n} (sm_{int(c*10)})" for n, c in gpus)
    if min_cap < 8.6:
        return ("FATAL",
                f"detected GPU with compute capability {min_cap} (Pascal/Turing/older). "
                f"This launcher requires Ampere (sm_86) or newer. {summary}\n"
                f"  See docs/HARDWARE.md for the supported card list.")
    if not is_cu13_wheel and max_cap >= 12.0:
        return ("FATAL",
                f"this zip ships an Ampere/Ada wheel (CUDA 12.6) which has no Blackwell (sm_120) kernels.\n"
                f"  Detected: {summary}.\n"
                f"  Download the -blackwell zip from\n"
                f"    https://github.com/devnen/qwen3.6-windows-server/releases\n"
                f"  and re-extract on top of this install.")
    if is_cu13_wheel and max_cap < 9.0:
        return ("WARN",
                f"this zip ships the Blackwell wheel (CUDA 13). Your GPU works on it but the\n"
                f"  -ampere zip would not require a CUDA 13 driver update. {summary}")
    return None


def ensure_runtime() -> None:
    """Block until vllm + the bundled wheel's exact build are installed.

    Idempotent: a no-op when the marker file's wheel SHA matches the
    bundled wheel and ``import vllm`` works. When a user re-extracts a
    new release zip on top of an existing install we detect the SHA
    mismatch and reinstall — without that check, the old vLLM stays in
    site-packages forever.
    """
    # Self-heal missing Python C headers / import libs before doing
    # anything else. Cheap when already present; one HTTP GET + a small
    # extract on existing v0.1.x installs that shipped without them.
    _ensure_python_headers()

    wheel = _wheel_path()
    bundled_sha = _sha256_file(wheel) if wheel and wheel.is_file() else None
    marker = _read_marker()

    # Fast path: marker matches the bundled wheel and vllm imports cleanly.
    if (
        bundled_sha
        and marker.get("wheel_sha256") == bundled_sha
        and _vllm_importable()
    ):
        # Idempotent: rebuild the CUDA 13 shim if needed (no-op when
        # already populated; also a no-op for cu126 wheels).
        try:
            _ensure_cuda13_shim()
        except Exception as e:
            print(f"  [warn] CUDA 13 shim rebuild failed: {e}")
        return

    # Older install (no marker / legacy "ok" string) but vllm imports.
    # If we can't see a bundled wheel either, leave it alone — the user
    # may be running against their own venv.
    if marker and marker.get("wheel_sha256") is None and _vllm_importable() and not wheel:
        return

    if wheel is None:
        print(
            "[setup] Could not find the bundled wheel. Expected one of:\n"
            f"  {paths.install_root() / WHEEL_DIR_NAME / 'vllm-*.whl'}\n"
            f"  {paths.install_root() / WHEEL_DIR_NAME / LEGACY_WHEEL_FILENAME}\n"
            "Re-extract the release zip or drop the wheel into the wheels/ folder."
        )
        sys.exit(1)

    _print_banner()

    if marker.get("wheel_sha256") and marker.get("wheel_sha256") != bundled_sha:
        print("  [reinstall] Bundled wheel changed since last install — reinstalling vLLM.")
        print(f"             marker: {marker.get('wheel_sha256', '<none>')[:12]}...")
        print(f"             bundle: {bundled_sha[:12]}...")
        print()

    proper_wheel = _wheel_proper_filename(wheel)

    # GPU compatibility preflight. Catches wrong-zip-for-this-GPU before
    # a 6 GB wheel install fails 30 s into model load with an obscure
    # cudaErrorNoKernelImageForDevice traceback.
    gpu_check = _check_gpu_supports_wheel(proper_wheel)
    if gpu_check is not None:
        level, msg = gpu_check
        if level == "FATAL":
            print()
            print("=" * 70)
            print(f"  [GPU mismatch] {msg}")
            print("=" * 70)
            sys.exit(1)
        else:
            print(f"  [warn] {msg}")
            print()

    if not _pip_available():
        _bootstrap_pip()

    _install_wheel(proper_wheel)

    # Build the CUDA 13 shim AFTER torch is installed, so torch.__file__ is
    # resolvable. No-op for cu126 wheels.
    try:
        _ensure_cuda13_shim()
    except Exception as e:
        print(f"  [warn] CUDA 13 shim build failed: {e}")
        print("         flashinfer may fail at first model load.")

    payload = {
        "wheel_sha256": bundled_sha,
        "wheel_filename": wheel.name,
    }
    # Best-effort: read the version out of the wheel name for human eyes.
    if proper_wheel.name.startswith("vllm-"):
        try:
            payload["version"] = proper_wheel.name.split("-", 2)[1]
        except IndexError:
            pass
    _write_marker(payload)

    print()
    print("[setup] vLLM runtime installed.")
    print()
