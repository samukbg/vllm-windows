"""Backup-and-wipe the four caches that, when poisoned, cause the
NVFP4 prefill regression documented in
``_local/CACHE_POISON_INCIDENT_2026-05-07.md``.

The fingerprint for poisoning: prefill ~750 tok/s on a 47k NVFP4 prompt,
power 200-270W, mem-BW% near 0 — i.e. silicon idle despite NVFP4 being
loaded. First-line response is to run this script.

By default everything is *moved* to a sibling ``.bak.<TIMESTAMP>`` dir
so a forensic comparison is possible. ``--no-backup`` deletes outright.
``--dry-run`` reports sizes without touching anything.

Caches (in priority order — the first one is the load-bearing fix):

1. ``%USERPROFILE%\\.cache\\vllm\\`` — vLLM torch-inductor AOT compile
   cache (typically multi-GB after sustained use). Compiled graphs hold
   baked-in references to whichever FlashInfer tactic the autotuner
   picked at graph-compile time. If those picks were made under a
   polluted CUDA env, every later boot inherits the slow choices.
2. ``%LOCALAPPDATA%\\Temp\\torchinductor_<user>\\`` — torch's separate
   temp inductor cache. Nuked together with #1.
3. ``%USERPROFILE%\\.cache\\torch\\`` — torch's general kernel cache.
4. ``%USERPROFILE%\\.cache\\flashinfer\\`` — flashinfer's per-shape
   autotune scoreboard. Wiping this alone never fixes the regression
   because the vLLM AOT cache hits before flashinfer is reconsulted —
   but wipe it together with #1 so the re-autotune produces fresh picks.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


def _userprofile() -> Path:
    return Path(os.environ.get("USERPROFILE", os.path.expanduser("~")))


def _localappdata() -> Path:
    val = os.environ.get("LOCALAPPDATA")
    if val:
        return Path(val)
    return _userprofile() / "AppData" / "Local"


def _username() -> str:
    return os.environ.get("USERNAME", "")


def cache_targets() -> list[tuple[str, Path]]:
    """Return ``[(label, path), ...]`` of cache dirs to handle, in order.

    Missing dirs stay in the list so the report is honest about what
    was and wasn't found.
    """
    home = _userprofile()
    lad = _localappdata()
    user = _username()
    inductor_dirname = f"torchinductor_{user}" if user else "torchinductor_*"
    targets: list[tuple[str, Path]] = [
        ("vllm AOT compile cache (LOAD-BEARING)", home / ".cache" / "vllm"),
        ("torch inductor temp cache", lad / "Temp" / inductor_dirname),
        ("torch general kernel cache", home / ".cache" / "torch"),
        ("flashinfer autotune scoreboard", home / ".cache" / "flashinfer"),
    ]
    # Glob fallback for inductor when USERNAME wasn't available.
    if not user:
        matches = sorted((lad / "Temp").glob("torchinductor_*"))
        if matches:
            targets[1] = (targets[1][0], matches[0])
    return targets


def dir_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def fmt_size(n: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:6.1f} {u}"
        f /= 1024
    return f"{n} B"


def report(targets: list[tuple[str, Path]]) -> None:
    print("=" * 72)
    print("Cache scan")
    print("=" * 72)
    for label, path in targets:
        if path.is_dir():
            size = dir_size(path)
            print(f"  EXISTS  {fmt_size(size)}   {path}")
            print(f"          {label}")
        else:
            print(f"  absent              {path}")
            print(f"          {label}")
    print("=" * 72)


def wipe_one(label: str, path: Path, *, no_backup: bool, ts: str) -> tuple[str, str]:
    """Return ``(action, message)``. action ∈ {moved, deleted, missing, failed}."""
    if not path.is_dir():
        return ("missing", f"  -    not present: {path}")
    if no_backup:
        try:
            shutil.rmtree(path)
            return ("deleted", f"  rm   {path}")
        except OSError as e:
            return ("failed", f"  X    rm failed: {path}: {e}")
    backup = path.with_name(f"{path.name}.bak.{ts}")
    try:
        path.rename(backup)
        return ("moved", f"  mv   {path}  ->  {backup.name}")
    except OSError as e:
        return ("failed", f"  X    mv failed: {path}: {e}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Just report sizes, do not touch.")
    p.add_argument("--no-backup", action="store_true", help="rmtree instead of rename to .bak.<TS>.")
    p.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt.")
    args = p.parse_args()

    targets = cache_targets()
    report(targets)

    if args.dry_run:
        print("\n--dry-run: nothing modified.")
        return 0

    # Confirm.
    if not args.yes:
        op = "DELETE OUTRIGHT" if args.no_backup else "MOVE TO .bak.<timestamp>"
        try:
            ans = input(f"\nProceed to {op} all four caches above? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("aborted.")
            return 1

    ts = time.strftime("%Y%m%d_%H%M%S")
    print()
    print("=" * 72)
    print(f"Wiping (timestamp {ts}, no_backup={args.no_backup})")
    print("=" * 72)

    counts = {"moved": 0, "deleted": 0, "missing": 0, "failed": 0}
    for label, path in targets:
        action, msg = wipe_one(label, path, no_backup=args.no_backup, ts=ts)
        counts[action] += 1
        print(msg)

    print("=" * 72)
    print(f"summary: {counts['moved']} moved, {counts['deleted']} deleted, "
          f"{counts['missing']} missing, {counts['failed']} failed")
    if counts["failed"]:
        print("\nNOTE: at least one cache could not be wiped. Likely a process "
              "is still holding a file open. Stop vLLM first:")
        print("  python snapshots/stop_vllm.py")
        return 2
    print("\nNext steps:")
    print("  1. python snapshots/stop_vllm.py        # if a server is still running")
    print("  2. python snapshots/start_5090_nvfp4.py # cold rebuild ~11 min")
    print("  3. re-bench at 47k unique prompt — expect 5,200+ tok/s, ~580W peak.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
