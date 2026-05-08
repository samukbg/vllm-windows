# Managing snapshots

A "snapshot" is one validated set of vLLM flags for one hardware/model combo.
The launcher ships with six (`start_speed`, `start_127k`, `start_72tps`,
`start_mtp4`, `start_pp2_160k`, `start_gpu0_50k`); you can add, edit,
duplicate, and delete your own from inside the TUI without ever opening
a text editor.

## TL;DR

- Press `e` on the dashboard to open the snapshot editor.
- New / Duplicate / Edit / Delete work on the list at the left, the form
  on the right edits whatever's highlighted.
- Save (`Ctrl+S`) writes both `launcher\configs.yaml` and
  `snapshots\start_<id>.py` in one go.
- Esc returns to the dashboard. The card grid rebuilds itself, no
  restart needed.

## Why two files per snapshot

Each snapshot lives in two places that have to stay in sync:

1. **`launcher\configs.yaml`**: the descriptive entry (id, tagline, tier,
   measured tok/s, notes). The dashboard reads this to draw cards.
2. **`snapshots\start_<id>.py`**: the actual `vllm serve ...` argv. This
   is what runs when you press Load.

Hand-editing one and forgetting the other is the most common way to
break a config. The in-TUI editor writes both at once, so you can't
desync them.

## Keyboard reference

### Dashboard

| Key | Action |
|---|---|
| Arrow keys | Move focus between cards |
| Enter | Open the highlighted card's detail screen |
| `e` | Open the snapshot editor |
| `h` | Help |
| `r` | Refresh (re-read configs.yaml from disk) |
| `q` | Quit |

Since v1.2.4 the dashboard shells out to `nvidia-smi` at startup,
classifies the host as `blackwell` (RTX 50-series), `ampere` (RTX
30/40-series + datacentre A/L), or `unknown`, and uses that to:

- Show a banner at the top of the snapshots pane with the detected
  GPU and arch bucket.
- Group active cards into arch sections, putting the host's
  architecture first under a `Recommended for your <Arch> GPU`
  header (blue) and the other arch underneath under a neutral header.
- Tag every card with an arch chip (`[Blackwell]` blue, `[Ampere/Ada]`
  orange) plus a colored top border.

The arch bucket is read from the optional `arch:` key on each entry
in `configs.yaml`. The shipped `rtx5090*` snapshots are explicitly
tagged `arch: blackwell`. Anything else falls through a heuristic
(`id` starts with `rtx5090` -> blackwell, otherwise ampere), so
existing user snapshots don't need editing.

### Detail screen (one snapshot)

| Key | Action |
|---|---|
| `l` | Load (start the server). Disabled if it's already running. |
| `u` | Unload (stop the server). Disabled if it's not running. |
| `t` | Test (send a small chat completion). Disabled while not running. |
| `e` | Edit this snapshot in the CRUD editor (preselected). |
| Esc | Back to the dashboard |

The Load button is gated on live state. While a snapshot is booting
(LOADING banner is up), all three action buttons are disabled so you
can't double-load it. Once `/v1/models` answers, Unload and Test light
up; Load stays disabled until you Unload.

### Snapshot editor (CRUD)

| Key | Action |
|---|---|
| Arrow keys / Tab | Move between the list and the form |
| `Ctrl+N` | New (blank entry, ready to fill in) |
| `Ctrl+D` | Duplicate the highlighted entry |
| `Ctrl+S` | Save |
| Esc | Back to the dashboard |

Delete sits on a button in the list panel rather than on a key, so a
stray keypress can't wipe a config. Confirmation modal before anything
is removed.

### Editing inside text inputs (WindowsInput)

The text inputs in the form follow Windows / Notepad conventions, not
the emacs/readline conventions Textual ships by default:

| Key | Action |
|---|---|
| `Ctrl+A` | Select all |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` or `Ctrl+Shift+Z` | Redo |
| `Ctrl+C` / `Ctrl+X` / `Ctrl+V` | Copy / cut / paste |
| `Shift+Delete` | Cut (legacy alias) |
| `Shift+Insert` | Paste (legacy alias) |
| `Ctrl+Insert` | Copy (legacy alias) |
| `Ctrl+Backspace` | Delete word to the left |
| `Ctrl+Delete` | Delete word to the right |
| `Ctrl+Left` / `Ctrl+Right` | Jump one word |
| `Ctrl+Shift+Left` / `Ctrl+Shift+Right` | Select one word |
| `Shift+Home` / `Shift+End` | Extend selection to line start / end |
| `Ctrl+Shift+Home` / `Ctrl+Shift+End` | Select to start / end |
| Single click | Position cursor |
| Double click | Select word under cursor |
| Triple click | Select all |
| Click and drag | Select range |
| Shift + click | Extend the current selection |

Undo history is per input, lives in memory only, and resets when you
leave the screen.

## What the editor does on Save

When you hit `Ctrl+S` (or click Save), the editor:

1. Validates the form (id is unique, port is an integer, tier and
   status are one of the allowed values).
2. Rewrites `launcher\configs.yaml` in place via PyYAML. Placeholders
   like `${SNAPSHOTS_DIR}\start_speed.bat` stay portable; multi-line
   notes use literal block style so they don't get mangled.
3. Rewrites the editable constants in `snapshots\start_<id>.py` with a
   regex pass: `PORT`, `TP`, `PP`, `USE_MTP`, `NUM_SPEC_TOKENS`, `CTX`,
   `GPU_MEM_UTIL`, `MAX_NUM_BATCHED_TOKENS`. The flag invariants
   (attention backend, KV dtype, chat template, tool-call parser, env
   vars for Gloo stability) stay frozen because they're not in the
   editable set.
4. For New entries: copies `start_speed.py` to `start_<id>.py` first,
   so all the invariants come along for free, then runs the regex
   rewrite. Generates a thin `start_<id>.bat` wrapper next to it.
5. For Rename: moves the `.py` and `.bat` together to the new basename
   before the constants rewrite.
6. For Delete: removes `.py`, `.bat`, and the YAML entry after a
   confirmation modal.

The dashboard rebuilds itself when you press Esc, so renames and
deletes show up immediately.

## What the form covers

The form exposes every YAML field plus the `.py` constants you actually
change between configs. This is the full set:

| Field | Where it lives | Notes |
|---|---|---|
| ID | yaml + .py filename | The snapshot's stable name. Renaming this moves the .py and .bat. |
| Tagline | yaml | Shown on the dashboard card. |
| Tier | yaml | `active` / `legacy` / `blocked`. Picks which dashboard section the card lands in. |
| Status | yaml | `recommended` / `experimental` / `conditional` / `superseded` / `blocked`. |
| GPU | yaml | `GPU0`, `GPU1`, or `GPU0+1`. The Python falls back to GPU0 on a single-GPU host. |
| Port | .py (`PORT`) | Two snapshots can share a port; the launcher disambiguates by the runtime manifest written at boot. |
| TP / PP | .py | Tensor / pipeline parallel. TP=2 is unusable on Windows (~7 tok/s). |
| GPU mem-util | .py (`GPU_MEM_UTIL`) | 0.92 is conservative for a card with the display attached, 0.948 is the boot-quiet ceiling. |
| MTP n | .py (`USE_MTP` + `NUM_SPEC_TOKENS`) | Blank disables spec-decode. |
| Context | .py (`CTX`) | Tokens. If you push past what the KV pool can hold, vLLM raises `ValueError: ... estimated maximum model length is N` at boot before any weights load. Read N from the error and re-save. See [`docs/TUNING.md`](TUNING.md#context) for the OOM-oracle and the headroom-from-successful-boot pattern. |
| Decode tok/s | yaml | Measured number you've validated. |
| Prefill tok/s cold | yaml | Cold-cache prefill, optional. |
| Power cap | yaml | Watts, optional. |
| Notes | yaml | Free text. Multi-line is fine. |
| Arch | yaml (`arch:`) | Optional. `blackwell` or `ampere`. Drives dashboard grouping and the colored chip on each card. Defaults to a heuristic if missing (id starts with `rtx5090` -> blackwell, otherwise ampere). |

Anything else (attention backend, KV dtype, chat template path, env
vars, etc.) is a flag invariant and is not editable through the form
on purpose. See "When you have to hand-edit" below.

## Flag invariants (don't break these)

These are baked into every shipped snapshot and you should not change
them lightly. Full reasoning is in
[`docs/HALLUCINATED_FLAGS.md`](HALLUCINATED_FLAGS.md) and
[`docs/COHERENCE.md`](COHERENCE.md):

- `--quantization=auto-round` for Lorbus AutoRound INT4 weights (every Ampere/Ada snapshot, plus `rtx5090` / `rtx5090_max`); `--quantization=compressed-tensors` for the NVFP4 snapshots `rtx5090_nvfp4` (text default since v1.3.0) and `rtx5090_nvfp4_vision` (image/video input on the same NVFP4 weights — the Peutlefaire quant preserves the unquantized visual tower, so vision is a CLI flag flip rather than a different model). See [`BLACKWELL.md`](BLACKWELL.md) and [`SM120_GDN_CEILING.md`](SM120_GDN_CEILING.md).
- `--attention-backend=TRITON_ATTN` plus
  `VLLM_ATTENTION_BACKEND=TRITON_ATTN`. FLASHINFER trips MAX_PATH on
  Windows; nothing else is supported by the Qwen3-Next hybrid arch on
  vLLM 0.19.0.
- `--kv-cache-dtype=fp8_e4m3`. TRITON_ATTN rejects `fp8_e5m2`.
- `--chat-template=qwen3.5-enhanced.jinja` plus
  `default-chat-template-kwargs={"preserve_thinking": false}` plus
  `--tool-call-parser=qwen3_coder` plus `--reasoning-parser=qwen3`.
  This is the tool-calling fix from PR 35687 and PR 40861. Removing it
  breaks Cline / Claude Code / Codex.
- `USE_LIBUV=0`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=0`,
  `NCCL_ASYNC_ERROR_HANDLING=0`. Windows Gloo stability.

The CRUD form leaves all of these alone. New entries inherit them by
copying `start_speed.py` as the template.

## Compatibility matrix (also do not break)

- **MTP + PP=2**: not supported on Qwen3-Next on this wheel
  (`SupportsPP NotImplementedError`). The shipped `start_pp2_160k`
  disables MTP for that reason. If you set PP > 1 in the form, set
  MTP n to blank.
- **PP=2 requires the `+devnen.3` Ampere wheel** (or any Blackwell
  wheel). v1.3.3 added the ZMQ ipc -> tcp fallback (`+devnen.2`); v1.3.4
  added the `os.sched_yield` Windows guard (`+devnen.3`). Older zips
  crash at boot with `ZMQError: Protocol not supported` (pre-v1.3.3) or
  `AttributeError: module 'os' has no attribute 'sched_yield'`
  (v1.3.3 itself). Run `update.bat` to v1.3.4+.
- **Draft-model spec-decode**: blocked entirely on Qwen3.6-27B
  (vocab=248320, no compatible drafter). Only MTP works.
- **TP=2 on Windows**: technically loads, decodes at ~7 tok/s because
  CPU-relay allreduce eats every layer. Don't.

Full table at [`docs/SPEC_DECODE_MATRIX.md`](SPEC_DECODE_MATRIX.md).

## When you have to hand-edit

The CRUD form covers the constants you actually change between configs.
For anything else (a new vLLM flag, a different attention backend on a
test branch, env vars beyond what `_common.py` already sets), you have
to edit the `.py` directly. The path:

1. Copy `snapshots\start_speed.py` to `snapshots\start_<myname>.py`.
2. Change whatever you need. Keep the flag invariants above.
3. Copy `snapshots\start_speed.bat` to `snapshots\start_<myname>.bat`,
   change the `.py` filename inside.
4. Add a card under `windows.configs[]` in `launcher\configs.yaml` so
   the launcher can list it. Or open the in-TUI editor, which will
   pick up your new `.py` on next launch and let you fill the YAML
   side from the form.
5. Run [`windows_tools\check_coherence.py`](../windows_tools/check_coherence.py)
   against the new snapshot. Coherence-validated TPS is the bar; a
   coherence-failing 80 tok/s is fraud.

Don't parameterise via env vars within a snapshot. The whole point is
that each one is self-describing and rollback-able. A new config is a
new file.

## Running outside the launcher

Every `.bat` is double-clickable, and the `.py` runs under the embedded
Python directly. From cmd:

```cmd
set VLLM_MODEL_DIR=D:\models\Qwen3.6-27B-int4-AutoRound
snapshots\start_speed.bat
```

To stop everything:

```cmd
snapshots\stop_vllm.bat
```

## How "running" detection works

When a snapshot boots it writes `logs\runtime\<port>.json` with its id.
The launcher reads these files to drive the dashboard, so the right
card lights up even when two snapshots share a port (e.g. `start_speed`
and `start_127k` both bind 5001). The probe is locale-free
(`socket.connect_ex` plus a wrapper-pid check), so non-en-US Windows
isn't a problem.

If a card looks stuck, `logs\runtime\` is the place to look. A stale
`<port>.json` with a dead process gets cleaned up on the next poll.

## Sharing your configs

If you've validated something faster (or with more context) than what
ships, please send a PR. The bar is the
[3-tier coherence check](COHERENCE.md). Configs I'd love to see:

- Other Qwen3.6-27B quants (FP8, NVFP4, smaller AutoRound variants).
- Smaller Qwen models (14B, 8B, 4B) for 16 GB cards.
- 4090 / 5090 / 5060 Ti / A6000 tunings.
- New parallelism or KV-cache combos as vLLM adds them.
