# Snapshots

Each `start_*.py` is a frozen, validated config: one knob set, one set
of measured numbers. Each `start_*.bat` is a one-line wrapper that
resolves the venv via `%VLLM_WINDOWS_VENV%` and launches the
corresponding `.py`.

Full user-facing guide (CRUD editor, keyboard shortcuts, flag
invariants, hand-edit fallback): [`docs/SNAPSHOTS.md`](../docs/SNAPSHOTS.md).

## When to use which

All snapshots ship with the tool-calling fix (patched wheel PR #35687
plus #40861, `qwen3.5-enhanced.jinja`, `preserve_thinking=false`),
verified 8/8 on `windows_tools\test_toolcall.py`. Pick one based on
speed / context tradeoff; any of them works for Cline, Cursor, Codex
CLI, OpenWebUI, Claude Code.

| Snapshot | Decode tok/s | Ctx | When |
|---|---|---|---|
| `start_speed`     | **64.5** (long prompt) | 90 k  | Default for daily use. MTP n=6, GPU1, 350 W. |
| `start_127k`      | 53.4                   | 127 k | When you need the largest single-GPU ctx. MTP n=3. |
| `start_mtp4`      | 58.3                   | 120 k | Mid-balance speed vs ctx. |
| `start_72tps`     | ~72 short prompts      | 32 k  | Original short-prompt baseline. |
| `start_pp2_160k`  | 43.5                   | 160 k | Both GPUs, no MTP. Use only when 127 k isn't enough. |
| `start_gpu0_50k`  | volatile               | ~9 to 50 k | Single-GPU users with the display attached. Read [`docs/HARDWARE.md`](../docs/HARDWARE.md). |

## How they resolve paths

Every snapshot imports from `_common.py`:

```python
from _common import VENV, VLLM_EXE, MODEL_PATH, VCVARS, log_path_for
```

`_common.py` reads four env vars with sensible defaults:

| Variable | Default |
|---|---|
| `VLLM_WINDOWS_VENV`   | `<repo>\venv` |
| `VLLM_MODEL_DIR`      | `<repo>\models\Qwen3.6-27B-int4-AutoRound` |
| `VLLM_WINDOWS_VCVARS` | First found among VS 2022 Community / Pro / Enterprise / BuildTools |
| `VLLM_WINDOWS_LOGS`   | `<repo>\logs` |

So a typical install just sets `VLLM_MODEL_DIR` once (or drops the
model into the default location) and everything else resolves
automatically.

## Adding your own (preferred path: in-TUI editor)

1. Launch the TUI (`start.bat`).
2. Press `e` on the dashboard.
3. Press `Ctrl+N` for a blank entry, or `Ctrl+D` to duplicate the
   highlighted snapshot as a starting point.
4. Fill the form (id, tagline, port, mem-util, ctx, MTP n, etc.).
5. `Ctrl+S` to save. The editor writes `launcher\configs.yaml` and
   `snapshots\start_<id>.py` together, copying flag invariants from
   `start_speed.py` so they don't drift.
6. Esc back to the dashboard. New card appears immediately.
7. Validate it with [`windows_tools\check_coherence.py`](../windows_tools/check_coherence.py)
   before trusting any tok/s number.

Full keyboard reference and field-by-field walkthrough:
[`docs/SNAPSHOTS.md`](../docs/SNAPSHOTS.md).

## Adding your own (hand-edit fallback)

For changes the form doesn't cover (a new vLLM flag, a different
attention backend on a test branch, env vars beyond what `_common.py`
already sets):

1. Copy `start_speed.py` to `start_<myname>.py`.
2. Edit the constants block (CTX, TP, PP, MTP_N, GPU_MEM_UTIL, etc.)
   and any new flags. Keep the flag invariants from
   [`docs/SNAPSHOTS.md`](../docs/SNAPSHOTS.md).
3. Copy `start_speed.bat` to `start_<myname>.bat`, change the `.py`
   filename inside.
4. Add a card under `windows.configs[]` in `launcher\configs.yaml`,
   or open the in-TUI editor and let it pick up the new `.py` on next
   launch so you can fill the YAML side from the form.
5. Bench, validate coherence, commit.

Don't parameterise via env vars within a snapshot. The whole point is
that each one is self-describing and rollback-able. A new config is a
new file.

## Running outside the launcher

Every `.bat` is double-clickable. From cmd:

```cmd
set VLLM_MODEL_DIR=D:\models\Qwen3.6-27B-int4-AutoRound
snapshots\start_speed.bat
```

To stop all servers:

```cmd
snapshots\stop_vllm.bat
```
