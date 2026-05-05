"""Developer-side: build the portable launcher zip for a release.

Run on the developer machine ONCE per tag. Produces
``qwen3.6-windows-server-portable-x64.zip`` with embeddable Python, all
launcher dependencies preinstalled, the launcher source, snapshot/tool
scripts, and (optionally) the patched vLLM wheel. End users unzip and
double-click ``start.bat`` — they NEVER run pip.

Pass ``--wheel <path>`` to embed the patched vLLM wheel into the zip
under ``wheels/`` so the launcher can install it on first run without
network access.

Steps:
  1. Download python-3.12.X-embed-amd64.zip into a temp dir.
  2. Extract to <build>/python/.
  3. Edit python312._pth to enable site-packages.
  4. Bootstrap pip via get-pip.py (one-shot, into the embed).
  5. pip install textual textual-serve rich httpx pyyaml ninja --target python\\Lib\\site-packages.
     ninja's executable lands at python\\Lib\\site-packages\\bin\\ninja.exe;
     snapshots/_common.py prepends that dir to PATH at module import so
     flashinfer's JIT and ``shutil.which("ninja")`` both find it.
  6. Strip pip / setuptools / wheel / Scripts / __pycache__ from the embed.
  7. Copy launcher/, snapshots/, windows_tools/, docs/, README + LICENSE,
     and the wheel (if --wheel given) into the build dir.
  8. Zip the whole build dir.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

PY_VERSION = "3.12.7"  # last 3.12 release with embed-amd64 zip
PY_EMBED_URL = f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
# NuGet python package ships tools/include/ + tools/libs/ that the embed zip
# omits. Triton's JIT (runtime/build.py) needs Python.h and python312.lib to
# compile cuda_utils.c on first model load; without them vLLM dies on the
# first request with `include file 'Python.h' not found`. See issue #1.
PY_NUGET_URL = f"https://globalcdn.nuget.org/packages/python.{PY_VERSION}.nupkg"

LAUNCHER_DEPS = ["textual>=0.86", "textual-serve", "rich", "httpx", "pyyaml", "ninja"]
TOP_FILES = ["LICENSE", "README.md", "CITATION.cff"]
TOP_DIRS = ["launcher", "snapshots", "windows_tools", "docs", "terminal", "templates"]

REPO = Path(__file__).resolve().parent.parent


def download(url: str, dst: Path) -> None:
    print(f"[build] download {url} -> {dst.name}")
    urllib.request.urlretrieve(url, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="Output zip path. If omitted, uses dist/qwen3.6-windows-server-portable-x64[-VARIANT].zip.")
    ap.add_argument("--workdir", default=None,
                    help="Build scratch dir. Defaults to .build_launcher[-VARIANT] under the repo.")
    ap.add_argument("--wheel", default=None,
                    help="Path to vllm-...+devnen.X-...whl to bundle into wheels/")
    ap.add_argument("--variant", default=None, choices=[None, "ampere", "blackwell"],
                    help="Release variant. 'ampere' = current 30/40-series zip (cu126 torch, "
                         "vllm 0.19.x). 'blackwell' = 50-series zip (cu130 torch, vllm 0.20.x, "
                         "CUDA 13 shim auto-built at first run). Affects default --out filename. "
                         "Omit for legacy single-zip builds.")
    args = ap.parse_args()

    variant = args.variant
    if args.out:
        out_zip = Path(args.out)
    elif variant:
        out_zip = REPO / "dist" / f"qwen3.6-windows-server-portable-x64-{variant}.zip"
    else:
        out_zip = REPO / "dist" / "qwen3.6-windows-server-portable-x64.zip"

    if args.workdir:
        work = Path(args.workdir)
    else:
        suffix = f"-{variant}" if variant else ""
        work = REPO / f".build_launcher{suffix}"

    if variant == "blackwell" and args.wheel:
        wheel_name = Path(args.wheel).name
        if "+cu13" not in wheel_name:
            print(f"[build] WARNING: --variant=blackwell but wheel '{wheel_name}' "
                  f"has no '+cu13*' local-version tag. Launcher autodetect "
                  f"will install cu126 torch — Blackwell GPUs will fail to boot.",
                  file=sys.stderr)
    if variant == "ampere" and args.wheel:
        wheel_name = Path(args.wheel).name
        if "+cu13" in wheel_name:
            print(f"[build] WARNING: --variant=ampere but wheel '{wheel_name}' "
                  f"is a CUDA 13 wheel. 30/40-series users on default cu126 "
                  f"drivers will fail to import torch.",
                  file=sys.stderr)

    if work.exists():
        shutil.rmtree(work)
    build = work / "qwen3.6-windows-server"
    build.mkdir(parents=True)
    py_dir = build / "python"
    py_dir.mkdir()

    # 1-2. embed python
    embed_zip = work / "embed.zip"
    download(PY_EMBED_URL, embed_zip)
    with zipfile.ZipFile(embed_zip) as zf:
        zf.extractall(py_dir)

    # 2b. Overlay Include/ and libs/ from the NuGet python package. The
    # embed zip strips both, but Triton's JIT compile of cuda_utils.c
    # needs Python.h (Include/) and python312.lib (libs/). Without this,
    # the first model load fails with the registry-inspect error chain
    # documented in issue #1.
    nuget_pkg = work / "python-nuget.nupkg"
    download(PY_NUGET_URL, nuget_pkg)
    with zipfile.ZipFile(nuget_pkg) as zf:
        for name in zf.namelist():
            # tools/include/* -> python/Include/*
            # tools/libs/*    -> python/libs/*
            if name.startswith("tools/include/") and not name.endswith("/"):
                rel = name[len("tools/include/"):]
                dst = py_dir / "Include" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
            elif name.startswith("tools/libs/") and not name.endswith("/"):
                rel = name[len("tools/libs/"):]
                dst = py_dir / "libs" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
    # Sanity: refuse to ship a zip without these.
    if not (py_dir / "Include" / "Python.h").is_file():
        raise RuntimeError("NuGet overlay failed: python/Include/Python.h missing")
    if not (py_dir / "libs" / "python312.lib").is_file():
        raise RuntimeError("NuGet overlay failed: python/libs/python312.lib missing")
    print("[build] overlaid Python Include/ + libs/ from NuGet package")

    # 3. enable site-packages in ._pth + add ..\launcher so `python -m app`
    # works without relying on PYTHONPATH (embedded distros ignore it).
    pth = next(py_dir.glob("python*._pth"))
    txt = pth.read_text(encoding="utf-8")
    if "Lib\\site-packages" not in txt:
        txt = txt.rstrip() + "\nLib\\site-packages\n"
    # Embedded python ._pth fully owns sys.path — the script's directory
    # is NOT auto-added the way it is for a normal Python install. We
    # need ..\launcher (so `python -m app` works) AND ..\snapshots
    # (so snapshot scripts can `from _common import ...`).
    for extra in ("..\\launcher", "..\\snapshots"):
        if extra not in txt:
            lines = txt.splitlines()
            for i, line in enumerate(lines):
                if line.strip() == ".":
                    lines.insert(i + 1, extra)
                    break
            else:
                lines.append(extra)
            txt = "\n".join(lines) + "\n"
    if "import site" in txt and "#import site" in txt:
        txt = txt.replace("#import site", "import site")
    elif "import site" not in txt:
        txt = txt.rstrip() + "\nimport site\n"
    pth.write_text(txt, encoding="utf-8")

    # 4. get-pip
    get_pip = work / "get-pip.py"
    download(GET_PIP_URL, get_pip)
    py_exe = py_dir / "python.exe"
    subprocess.check_call([str(py_exe), str(get_pip), "--no-warn-script-location"])

    # 5. install deps
    sp = py_dir / "Lib" / "site-packages"
    sp.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [str(py_exe), "-m", "pip", "install",
         "--no-warn-script-location",
         "--target", str(sp), *LAUNCHER_DEPS]
    )

    # 6. strip pip + tools + scripts (users don't need them)
    for victim in ["pip", "pip.exe", "setuptools", "wheel"]:
        for p in sp.glob(victim):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    for d in [py_dir / "Scripts"]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    for pyc in py_dir.rglob("__pycache__"):
        shutil.rmtree(pyc, ignore_errors=True)

    # 7. copy launcher + snapshots + tools + docs + readme
    for f in TOP_FILES:
        src = REPO / f
        if src.exists():
            shutil.copy2(src, build / f)
    for d in TOP_DIRS:
        src = REPO / d
        if src.exists():
            shutil.copytree(src, build / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # ship the user-facing start.bat + update.bat at top level
    shutil.copy2(REPO / "launcher" / "start.bat", build / "start.bat")
    update_bat = REPO / "launcher" / "update.bat"
    if update_bat.is_file():
        shutil.copy2(update_bat, build / "update.bat")

    # 7b. optionally bundle the patched vLLM wheel under wheels/.
    # Always ship a normalized "vllm.whl" plus a vendored get-pip.py so
    # launcher/app/setup.py can bootstrap the runtime offline-ish (only
    # the actual torch + dep wheels need to be fetched at first run).
    wheels_dir = build / "wheels"
    wheels_dir.mkdir(exist_ok=True)
    # Reuse the get-pip we already downloaded in step 4, plus copy the
    # repo-tracked one if a developer dropped it under wheels/.
    repo_get_pip = REPO / "wheels" / "get-pip.py"
    if repo_get_pip.exists():
        shutil.copy2(repo_get_pip, wheels_dir / "get-pip.py")
    elif get_pip.exists():
        shutil.copy2(get_pip, wheels_dir / "get-pip.py")
    if args.wheel:
        wheel_src = Path(args.wheel)
        if not wheel_src.exists():
            print(f"[build] WARNING: --wheel {wheel_src} not found; skipping bundle", file=sys.stderr)
        else:
            # Ship the wheel under its real PEP 427 filename (e.g.
            # ``vllm-0.19.0+devnen.1-cp312-cp312-win_amd64.whl``). The
            # launcher's setup.py looks for ``wheels/vllm-*.whl`` first
            # and only falls back to the legacy ``vllm.whl`` for back-compat
            # with v0.1.4 zips. Shipping the proper name removes the
            # rename-at-install-time dance and lets pip identify the wheel
            # without us having to crack open the METADATA at runtime.
            target_name = wheel_src.name
            if not (target_name.startswith("vllm-") and target_name.endswith(".whl")):
                # Defensive: caller passed a "vllm.whl"-style file. Fall
                # back to legacy name so setup.py's compat path picks it up.
                target_name = "vllm.whl"
            shutil.copy2(wheel_src, wheels_dir / target_name)
            print(f"[build] bundled wheel: {wheel_src.name} -> wheels/{target_name}")

    # 8. zip
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"[build] writing {out_zip}")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in build.rglob("*"):
            zf.write(p, p.relative_to(build.parent))
    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"[build] done. {out_zip.name} = {size_mb:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
