"""In-place updater for the qwen3.6-windows-server portable launcher.

Detects the current install variant (ampere = cu126 / 30+40 series, or
blackwell = cu13x / 50 series) by inspecting the bundled vLLM wheel
filename. Hits the GitHub releases API, downloads the matching zip,
verifies SHA256, replaces the launcher source tree in place, and
preserves user state.

Always preserved (script never touches them):
    user_config.json, models/, logs/, venv/, cuda13_shim/

Preserved by default, can be overwritten on prompt:
    launcher/configs.yaml  (snapshot CRUD writes lives here)

Everything else is replaced wholesale: launcher/ source, snapshots/,
windows_tools/, docs/, templates/, terminal/, wheels/, python/ embed,
README.md, LICENSE, CITATION.cff, start.bat, update.bat.

The launcher's ensure_runtime() repairs the venv on next boot when the
bundled wheel changes (e.g. ampere zip 0.19.0 -> 0.19.1, or a switch to
the blackwell zip).

Usage:
    update.bat                 # interactive, double-clickable wrapper
    python windows_tools\\update.py
    python windows_tools\\update.py --yes --launch
    python windows_tools\\update.py --dry-run
    python windows_tools\\update.py --variant blackwell    # force-switch
    python windows_tools\\update.py --zip path\\to\\local.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = "devnen/qwen3.6-windows-server"
REPO_API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"

# windows_tools/update.py -> install root is parent
INSTALL_ROOT = Path(__file__).resolve().parent.parent

# Top-level entries the script never touches.
PRESERVE = {
    "models", "logs", "venv", "cuda13_shim", "user_config.json",
}

# Files inside replaced trees that the user can choose to keep or
# overwrite. (relative path under install root, default-keep).
ASK_PRESERVE = [
    ("launcher/configs.yaml", True),
]


# ---- helpers ----------------------------------------------------------

def yn(prompt: str, default: bool = True, assume_yes: bool = False) -> bool:
    if assume_yes:
        return default if default is not None else True
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        try:
            ans = input(prompt + suffix).strip().lower()
        except EOFError:
            return default
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")


def detect_variant(root: Path) -> str:
    wheels = root / "wheels"
    if wheels.is_dir():
        for whl in wheels.glob("vllm-*.whl"):
            return "blackwell" if "+cu13" in whl.name else "ampere"
    return "ampere"


def detect_current_version(root: Path) -> str:
    v = root / "VERSION"
    if v.is_file():
        return v.read_text(encoding="utf-8").strip()
    return "unknown"


def fetch_latest_release() -> dict:
    req = urllib.request.Request(
        REPO_API_LATEST,
        headers={
            "User-Agent": "qwen3.6-update",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def find_zip_asset(release: dict, variant: str) -> dict | None:
    blackwell_suffix = "portable-x64-blackwell.zip"
    ampere_suffix = "portable-x64.zip"
    for a in release.get("assets", []):
        name = a["name"]
        if variant == "blackwell" and name.endswith(blackwell_suffix):
            return a
        if (variant == "ampere"
                and name.endswith(ampere_suffix)
                and not name.endswith(blackwell_suffix)):
            return a
    return None


def find_sha_asset(release: dict) -> dict | None:
    for a in release.get("assets", []):
        if a["name"] == "SHA256SUMS.txt":
            return a
    return None


def download(url: str, dst: Path, label: str) -> None:
    print(f"[update] downloading {label}")
    req = urllib.request.Request(url, headers={
        "User-Agent": "qwen3.6-update",
        "Accept": "application/octet-stream",
    })
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        last = 0.0
        with open(dst, "wb") as f:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if total and (now - last) > 0.5:
                    pct = done / total * 100
                    sys.stdout.write(
                        f"\r[update]   {done/1e6:7.1f} / {total/1e6:7.1f} MiB"
                        f"  ({pct:5.1f}%)"
                    )
                    sys.stdout.flush()
                    last = now
    if total:
        sys.stdout.write("\n")


def verify_sha(zip_path: Path, sha_path: Path) -> bool:
    expected = None
    target = zip_path.name
    for line in sha_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == target:
            expected = parts[0]
            break
    if not expected:
        print(f"[update] WARNING: {target} not in SHA256SUMS.txt; "
              f"skipping checksum verification.")
        return True
    h = hashlib.sha256()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got.lower() != expected.lower():
        print(f"[update] CHECKSUM MISMATCH: expected {expected}, got {got}")
        return False
    print(f"[update] sha256 verified: {target}")
    return True


def stop_running(root: Path) -> None:
    bat = root / "snapshots" / "stop_vllm.bat"
    if bat.is_file():
        print("[update] stopping any running snapshot...")
        try:
            subprocess.run([str(bat)], cwd=str(root), timeout=30, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[update]   stop_vllm.bat returned: {e}")


def replace_top_level(staging: Path, root: Path, dry_run: bool) -> int:
    """Walk staging top-level entries; replace each into root unless
    preserved. Returns count replaced."""
    n = 0
    for entry in sorted(staging.iterdir()):
        name = entry.name
        if name in PRESERVE:
            print(f"[update]   keep    {name}/  (preserved)")
            continue
        target = root / name
        verb = "would-replace" if dry_run else "replace"
        size_hint = ""
        if entry.is_dir():
            size_hint = "/"
        print(f"[update]   {verb:14s} {name}{size_hint}")
        n += 1
        if dry_run:
            continue
        if target.exists():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)
    return n


def find_inner_root(staging: Path) -> Path:
    """Release zips have a single top-level dir 'qwen3.6-windows-server/'.
    Tolerate flat zips too."""
    entries = list(staging.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return staging


# ---- main -------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Update qwen3.6-windows-server in place from the latest GitHub release.",
    )
    ap.add_argument("--variant", choices=["ampere", "blackwell"],
                    help="Override autodetected variant. Use to switch a 30/40-series "
                         "install onto the Blackwell zip (or vice versa).")
    ap.add_argument("--yes", action="store_true",
                    help="Accept all defaults; non-interactive.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan without modifying anything.")
    ap.add_argument("--launch", action="store_true",
                    help="Run start.bat after the update completes.")
    ap.add_argument("--no-launch", action="store_true",
                    help="Skip the post-update start.bat prompt entirely.")
    ap.add_argument("--zip", default=None,
                    help="Use a local zip instead of downloading from GitHub.")
    args = ap.parse_args()

    root = INSTALL_ROOT
    print(f"[update] install root:    {root}")
    print(f"[update] python:          {sys.executable}")

    detected_variant = detect_variant(root)
    variant = args.variant or detected_variant
    if args.variant and args.variant != detected_variant:
        print(f"[update] variant override: {detected_variant} -> {variant} "
              f"(install will switch wheels)")
    else:
        print(f"[update] variant:         {variant}")
    cur_ver = detect_current_version(root)
    print(f"[update] current version: {cur_ver}")

    new_ver = "(local zip)"
    asset = None
    sha_asset = None
    rel = None

    if args.zip:
        zip_local = Path(args.zip)
        if not zip_local.is_file():
            print(f"[update] ERROR: --zip {zip_local} not found")
            return 1
    else:
        print("[update] querying GitHub for the latest release...")
        try:
            rel = fetch_latest_release()
        except urllib.error.URLError as e:
            print(f"[update] ERROR: cannot reach GitHub: {e}")
            print(f"[update]   try downloading the zip manually and re-run with --zip <path>.")
            return 2
        new_ver = rel.get("tag_name", "unknown")
        asset = find_zip_asset(rel, variant)
        if not asset:
            print(f"[update] ERROR: no '{variant}' zip on release {new_ver}.")
            avail = [a["name"] for a in rel.get("assets", []) if a["name"].endswith(".zip")]
            if avail:
                print(f"[update]   available zips: {avail}")
            return 3
        sha_asset = find_sha_asset(rel)
        size_mb = asset.get("size", 0) / 1e6
        print(f"[update] latest release:  {new_ver}")
        print(f"[update] target asset:    {asset['name']}  ({size_mb:.1f} MiB)")

    print()
    print("[update] preservation plan:")
    print("[update]   ALWAYS preserved: user_config.json, models/, logs/, "
          "venv/, cuda13_shim/")
    keep_configs = yn(
        "[update] Keep your current launcher\\configs.yaml? "
        "(snapshot edits and Blackwell overrides live here)",
        default=True, assume_yes=args.yes)

    if args.no_launch:
        auto_launch = False
    elif args.launch:
        auto_launch = True
    else:
        auto_launch = yn("[update] Run start.bat after the update?",
                         default=True, assume_yes=args.yes)

    if not args.dry_run:
        if not yn(f"\n[update] Proceed with update to {new_ver} ({variant})?",
                  default=True, assume_yes=args.yes):
            print("[update] cancelled.")
            return 0

    stop_running(root)

    with tempfile.TemporaryDirectory(prefix="qwen36-update-") as td:
        tmp = Path(td)

        if args.zip:
            zip_path = Path(args.zip)
        else:
            zip_path = tmp / asset["name"]
            download(asset["browser_download_url"], zip_path, asset["name"])
            if sha_asset:
                sha_path = tmp / "SHA256SUMS.txt"
                download(sha_asset["browser_download_url"], sha_path, "SHA256SUMS.txt")
                if not verify_sha(zip_path, sha_path):
                    print("[update] aborting due to checksum failure.")
                    return 4

        staging = tmp / "staging"
        staging.mkdir()
        print(f"[update] extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(staging)
        inner = find_inner_root(staging)

        # Save preserved-by-prompt files before the wholesale replace.
        saved: dict[str, bytes] = {}
        for rel_path, default_keep in ASK_PRESERVE:
            keep = keep_configs if rel_path == "launcher/configs.yaml" else default_keep
            if keep:
                p = root / rel_path
                if p.is_file():
                    saved[rel_path] = p.read_bytes()

        print()
        print("[update] applying changes:")
        replace_top_level(inner, root, dry_run=args.dry_run)

        if not args.dry_run:
            for rel_path, data in saved.items():
                p = root / rel_path
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
                print(f"[update]   restored {rel_path}")
            if not args.zip:
                (root / "VERSION").write_text(new_ver + "\n", encoding="utf-8")
                print(f"[update]   wrote VERSION -> {new_ver}")

    print()
    if args.dry_run:
        print(f"[update] dry-run complete. No files modified.")
        return 0

    print(f"[update] done. Now on {new_ver} ({variant}).")
    if auto_launch:
        bat = root / "start.bat"
        if bat.is_file():
            print(f"[update] launching {bat.name}...")
            subprocess.Popen(["cmd", "/c", "start", "", str(bat)], cwd=str(root))
        else:
            print(f"[update] WARNING: start.bat not found at {bat}.")
    else:
        print("[update] skip start.bat. Re-run start.bat manually when ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
