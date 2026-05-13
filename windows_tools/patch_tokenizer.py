"""Patch a Lorbus AutoRound tokenizer_config.json so transformers loads it.

Lorbus ships ``tokenizer_class: TokenizersBackend`` which transformers 4.57
on Windows does not recognize (raises ValueError at engine boot). The fix is
to switch it to ``Qwen2Tokenizer``.

As of v0.1.5 the launcher applies this patch automatically the first time
it sees a model dir, so end users no longer need to run this script
manually. It remains here for power users who download the weights with
their own tooling, and as the canonical source of truth for the patch.

Usage:
    python windows_tools/patch_tokenizer.py G:\\_models\\Qwen3.6-27B-int4-AutoRound
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def apply_tokenizer_patch(model_dir: Path | str, keep_backup: bool = False) -> bool:
    """Apply the TokenizersBackend -> Qwen2Tokenizer fix in-place.

    Idempotent: returns False if the file is already patched, missing, or
    unreadable. Returns True only when a real change was written.

    Safe to call on every boot. The launcher does exactly that so re-downloads
    and fresh extracts heal themselves without user intervention.

    ``keep_backup``: when True, write a ``tokenizer_config.json.bak`` next to
    the original on first patch. Default False, the upstream HF copy is
    canonical, and the .bak is noise for downstream hash validators.
    """
    cfg = Path(model_dir) / "tokenizer_config.json"
    if not cfg.is_file():
        return False
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    cur = data.get("tokenizer_class")
    if cur != "TokenizersBackend":
        return False
    if keep_backup:
        bak = cfg.with_suffix(".json.bak")
        if not bak.exists():
            try:
                shutil.copy2(cfg, bak)
            except OSError:
                pass
    data["tokenizer_class"] = "Qwen2Tokenizer"
    cfg.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="path to the model directory containing tokenizer_config.json")
    ap.add_argument("--keep-backup", action="store_true",
                    help="write tokenizer_config.json.bak before patching (off by default)")
    args = ap.parse_args()

    md = Path(args.model_dir)
    cfg = md / "tokenizer_config.json"
    if not cfg.exists():
        print(f"[patch_tokenizer] no tokenizer_config.json at {cfg}", file=sys.stderr)
        return 1

    cur = json.loads(cfg.read_text(encoding="utf-8")).get("tokenizer_class")
    if cur == "Qwen2Tokenizer":
        print(f"[patch_tokenizer] already patched ({cur}); nothing to do.")
        return 0
    if cur != "TokenizersBackend":
        print(f"[patch_tokenizer] tokenizer_class is {cur!r}; nothing to do.")
        return 0

    apply_tokenizer_patch(md, keep_backup=args.keep_backup)
    suffix = " (with .bak)" if args.keep_backup else ""
    print(f"[patch_tokenizer] {cur!r} -> 'Qwen2Tokenizer' in {cfg}{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
