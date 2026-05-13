from __future__ import annotations
import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path
from . import model_setup, paths, setup as runtime_setup
from .app import LauncherApp


SNAPSHOT_NAMES = (
    "start_72tps",
    "start_speed",
    "start_127k",
    "start_mtp4",
    "start_pp2_160k",
    "start_gpu0_50k",
)

DEFAULT_HEADLESS_SNAPSHOT = "start_72tps"


def _resolve_snapshot(name: str) -> Path:
    """Map a snapshot name to its .py launcher under <install>/snapshots/."""
    if not name.startswith("start_"):
        name = "start_" + name
    snap_dir = paths.install_root() / "snapshots"
    candidate = snap_dir / f"{name}.py"
    if not candidate.is_file():
        print(f"\n[ERROR] No such snapshot: {snap_dir / (name + '.py')}")
        print("Available snapshots:")
        for s in SNAPSHOT_NAMES:
            print(f"  {s}")
        sys.exit(1)
    return candidate


def _exec_snapshot(snapshot_py: Path) -> int:
    """Run a snapshot .py with the embedded python and inherit stdio."""
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = [sys.executable, "-u", str(snapshot_py)]
    print(f"[launcher] exec: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, env=env)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="vllm_launcher",
        description="vLLM config launcher TUI (and headless driver)",
    )
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="Run web UI only (textual-serve)")
    s.add_argument("--host", default="localhost")
    s.add_argument("--port", type=int, default=8765)

    # Headless / scripted-install knobs. All are no-ops for the default
    # interactive TUI flow; they take effect only when paired with
    # --snapshot (or --headless without one, which still skips the TUI).
    p.add_argument("--snapshot", metavar="NAME",
                   help=f"Skip TUI and launch one of: {', '.join(SNAPSHOT_NAMES)}")
    p.add_argument("--headless", action="store_true",
                   help=("Skip the TUI. Without --snapshot or --setup-only, "
                         f"runs the default snapshot ({DEFAULT_HEADLESS_SNAPSHOT})."))
    p.add_argument("--setup-only", action="store_true",
                   help="Run runtime + model setup checks then exit. Implies --headless.")
    p.add_argument("--auto-download", action="store_true",
                   help="When the model is missing, download from Hugging Face without prompting.")
    p.add_argument("--model-dir", metavar="PATH",
                   help="Explicit path to the Qwen3.6-27B-int4-AutoRound directory.")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Assume yes for any preflight confirmations (reserved).")

    p.add_argument("--skip-model-check", action="store_true",
                   help="Skip the pre-TUI model discovery prompt.")
    p.add_argument("--skip-runtime-check", action="store_true",
                   help="Skip the pre-TUI vLLM runtime install check.")
    args = p.parse_args()

    if args.cmd == "serve":
        from textual_serve.server import Server
        cmd = f'"{sys.executable}" -m app'
        Server(command=cmd, host=args.host, port=args.port, title="vLLM Launcher").serve()
        return

    # In headless / snapshot / setup-only modes the user is running
    # unattended, bail loudly on missing prereqs instead of dropping to
    # the TUI fallback.
    headless = args.headless or bool(args.snapshot) or args.setup_only

    if not args.skip_runtime_check:
        try:
            runtime_setup.ensure_runtime()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(130)
        except Exception:  # noqa: BLE001
            print("\n[start] runtime install failed:\n")
            traceback.print_exc()
            if not headless:
                input("\nPress Enter to exit...")
            sys.exit(1)

    if not args.skip_model_check:
        try:
            model_setup.ensure_model(
                auto_download=args.auto_download,
                explicit_dir=Path(args.model_dir) if args.model_dir else None,
            )
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(130)
        except Exception:  # noqa: BLE001
            print("\n[start] model setup failed:\n")
            traceback.print_exc()
            if not headless:
                input("\nPress Enter to exit...")
            sys.exit(1)

    if args.setup_only:
        print("\n[launcher] --setup-only: runtime + model are ready. Exiting.")
        return

    if args.snapshot:
        snap = _resolve_snapshot(args.snapshot)
        sys.exit(_exec_snapshot(snap))

    if args.headless:
        print(f"\n[launcher] --headless without --snapshot; running default "
              f"snapshot {DEFAULT_HEADLESS_SNAPSHOT}. "
              f"Pass --setup-only to run setup checks then exit.")
        snap = _resolve_snapshot(DEFAULT_HEADLESS_SNAPSHOT)
        sys.exit(_exec_snapshot(snap))

    LauncherApp().run()


if __name__ == "__main__":
    main()
