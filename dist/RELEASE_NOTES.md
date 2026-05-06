Portable Windows launcher for Qwen3.6-27B inference. Unzip, double-click `start.bat`, you're serving on `http://127.0.0.1:5001/v1`.

## What's in the zip

- Embedded Python 3.12.7 with textual / rich / httpx / pyyaml preinstalled
- Patched vLLM wheel (`0.19.0+devnen.<N>`) bundled under `wheels/`
- 12 validated snapshot configs:
  - `start_72tps`, **~72 tok/s** short-prompt baseline (~200-token chat, 32 k ctx, MTP n=3)
  - `start_speed`, **64.5 tok/s** long-prompt default (~100 KB / ~24 k-token Python source, 90 k ctx, MTP n=6)
  - `start_127k`, 53.4 tok/s, max context on a single 3090
  - `start_mtp4`, 58.3 tok/s, mid-balance speed vs context
  - `start_pp2_160k`, 43.5 tok/s on **2× 3090 PP=2**, 160 k context
  - plus `start_gpu0_50k` and short-prompt variants (full table in the README)
- Helper tools: `bench`, `bench_summarize`, `check_coherence`, `verify_install`, `patch_tokenizer`, `probe_max_ctx`
- Full docs: HARDWARE / COHERENCE / TUNING / TROUBLESHOOTING / MTP_HEAD

## Install

1. Download `qwen3.6-windows-server-portable-x64.zip`.
2. Verify SHA256 (recommended, see below).
3. Extract anywhere, no admin needed, **including `Program Files` / `Program Files (x86)`**.
4. Double-click `start.bat`. On first run the launcher auto-discovers existing weights or offers to download Lorbus/Qwen3.6-27B-int4-AutoRound from Hugging Face (~16 GB, public, no token).

## What's new in v1.2.2 — important decode-tps fix

**TL;DR:** if you ever sent a long prompt (e.g. summarising a code file) and noticed every following request was suddenly slower, **this release fixes that.** Just `update.bat`.

### The bug

Every snapshot in v1.2.1 and earlier shipped with `--enable-prefix-caching` enabled. Qwen3.6-27B is a hybrid Mamba/SSM model, and per [vLLM issue #17140](https://github.com/vllm-project/vllm/issues/17140) prefix caching is **incompatible** with SSM state management — the SSM state is not tracked by the block manager, so prefix-cached blocks left Mamba state in a "dirty" mode after long-context requests. Symptom on Blackwell (RTX 5090, measured this release):

- Fresh server, short prompts: 130–160 tok/s (matches docs).
- After one ~24 k-token request: drops to 88–111 tok/s (-30 %), **stays stuck**.
- After a second long request: drops to 40–44 tok/s (-70 %), **stays stuck**.
- GPU clocks healthy, KV cache at 9.5 % — pure software regression.
- Pre-fix workaround was to restart the server between long-context turns.

The 3090 was hit by the same bug but less obviously (smaller absolute drop, masked by existing MTP-acceptance variability across prompts).

### The fix

All 12 shipped snapshots now pass `--no-enable-prefix-caching`. Measured post-fix on the same 5090 across 5 alternating long+short rounds:

| Round | Long-prompt decode | Short-prompt decode |
|---|---|---|
| 1 | 96 tok/s | 129 / 135 / 137 |
| 2 | 104 tok/s | 129 / 135 / 137 |
| 5 | 104 tok/s | 129 / 135 / 137 |

Zero degradation across rounds. Decode stays at documented speed across mixed long+short workloads, indefinitely.

### What you give up

The warm-prefix TTFT speedup on identical repeat prompts. Negligible in practice — every shipped snapshot uses `--max-num-seqs=1` (single user), so repeat-prefix hits are rare to begin with.

### Files changed

- All 12 `snapshots/start_*.py` (3090 + 5090 snapshots both).
- `launcher/configs.yaml` `shared_defaults.enable_prefix_caching: false` (informational; the flag the engine actually reads is in the `.py`).
- `docs/TROUBLESHOOTING.md`, `docs/TUNING.md`, `docs/UPGRADING.md`, README — full bug write-up and upgrade notes.

### How to upgrade

`update.bat`. Done.

## What's new in v1.1

Three new ready-to-go snapshots that pair Unsloth's recommended sampler settings with the right thinking-mode toggle, so you don't have to hand-edit a JSON or a snapshot file to get the workload-tuned defaults:

- **`start_instruct_general`** — non-thinking mode, temp 0.7 / top_p 0.8 / top_k 20 / min_p 0.0 (Unsloth's Instruct-general row). 127k context. For chat, Q&A, summarisation, general writing where you don't want the `<think>` block.
- **`start_instruct_coding`** — non-thinking mode, temp 0.6 / top_p 0.95 / top_k 20 / min_p 0.0 (Unsloth's Instruct-coding row). 127k context. For tool-heavy coding agents (Cline, Continue, Codex, Roo, Cursor) that want crisp, low-randomness output.
- **`start_thinking_coding`** — thinking mode left on, temp 0.6 / top_p 0.95 / top_k 20 / min_p 0.0 (Unsloth's Thinking-precise-coding row). 127k context. For code review, debugging, and architecture-level reasoning where the `<think>` trace adds value but you still want tighter sampling than the default thinking-mode temp 1.0.

All three use `--override-generation-config` to bake the sampler defaults into the running server, and the non-thinking pair uses `--default-chat-template-kwargs={"preserve_thinking": false, "enable_thinking": false}` to flip the model into Instruct mode at the template level. Per-request sampler params from your client (Cline / Cursor / Codex, etc.) still override these defaults.

Also in this release, two reliability fixes:

- **MSVC detection now uses `vswhere.exe` instead of hardcoded `\Microsoft Visual Studio\2022\` paths.** VS 2026 (= internal version 18.x, installed under `\2026\`) and any non-default install layout were silently missed by the old probe, which then printed a `vcvars64.bat not found` warning and disabled the FlashInfer sampler boost even when MSVC was actually available. New probe shells out to vswhere with `-requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64` and falls back to the hardcoded VS 2022 list when vswhere is missing.
- **TROUBLESHOOTING.md adds rows for the vswhere fix and the `cudart64_120.dll` mismatch workaround** (copy `cudart64_12.dll`, rename to `cudart64_120.dll`, place side by side) reported in the wild by an ev8siv3-style setup with VS 2026 and a non-default CUDA Toolkit layout.

Docs also gained: a Windows-paths tool-call gotcha in `docs/CLAUDE_CODE.md`, an LM Studio streaming-penalty section in `docs/COMPARISON.md`, and a 3x-small-card guidance section in `docs/HARDWARE.md`.

## What's new in v0.1.20

- **Copy button on the result modal.** A new green "Copy (C)" button next to "Close" copies the title + body to the system clipboard with rich-text markup stripped, so it pastes cleanly into a GitHub issue, Reddit comment, or Discord message. Press `c` to copy, `Esc` or `q` to dismiss.
- **Modal grew to 95% width × 90% height.** The v0.1.18 / v0.1.19 modal at fixed 90 cols clipped the response preview on Windows Terminal default sizes; the new percentage-based size scales with the terminal and the body fills the available height. Body is rendered inside a darker bordered scroll container so the title and button row are always visible.

## What's new in v0.1.19

- **Result modal is closable.** The benchmark result body is now long enough to push the Close button off-screen on shorter terminals, and the modal had no `Esc` binding. The body is now wrapped in a scroll container, the Close button stays pinned to the bottom, and `Esc` / `Enter` / `q` all dismiss the modal. Same fix applies to every other modal that uses `ResultModal` (Wake-on-LAN result, etc.).

## What's new in v0.1.18

Two TUI fixes so users don't have to drop to a command line for things the launcher should already do:

- **Test button now runs the same benchmark as `windows_tools/bench.py`.** Previously it sent a one-line "What is 2+2?" with `max_tokens=50`, which gives a noisy decode tok/s number that doesn't match anything published. The Test button now uses the 300-word transformer-attention prose prompt with `max_tokens=300` and reports TTFT, decode window, decode tok/s, and wall tok/s, the same fields `bench.py` prints. A live progress toast updates as tokens stream in (`Benchmarking... N/300 tokens decoded`), and the result modal includes a response preview plus a note that the number is directly comparable to the published headlines. No need to open cmd, set `VLLM_BENCH_BASE`, and run `python windows_tools\bench.py` separately.
- **Web UI (Ctrl-W) actually works.** `textual-serve` is now in `LAUNCHER_DEPS`, so the embedded Python ships with it instead of erroring out with "textual-serve not installed". Also fixed the spawned-child command in `app.py` and `__main__.py`, both said `-m vllm_launcher` (a module that does not exist) instead of `-m app`. The Web UI binding now opens a browser on `http://localhost:8765` and serves the same TUI to a remote machine.

## What's new in v0.1.17

Two follow-ups to v0.1.16's flashinfer-sampler fix:

- **ninja now ships inside the launcher zip.** Added to `LAUNCHER_DEPS` in `windows_tools/build_launcher_zip.py`. `pip install --target` puts the executable at `python\Lib\site-packages\bin\ninja.exe`; `snapshots/_common.py` prepends that dir to `PATH` at module import so `shutil.which("ninja")` and flashinfer's JIT child processes both find it. Users no longer need to `pip install ninja` into the embedded Python to enable the flashinfer sampler boost.
- **README.md gains an "Optional: install MSVC 2022 for the small decode boost" section.** Calls out that the launcher works without MSVC, and explains what installing the free Build Tools ("Desktop development with C++" workload) buys: the snapshots auto-detect MSVC and flip on the flashinfer sampler path. Cost is the ~7 GB Build Tools download plus a one-time JIT compile on first launch. With ninja bundled, the Build Tools install is now the only extra step.

TUNING.md and TROUBLESHOOTING.md updated to match.

## What's new in v0.1.16

The other half of the flashinfer story v0.1.15 missed:

- **`FileNotFoundError [WinError 2]` from `flashinfer/jit/cpp_ext.py:run_ninja` during `profile_run`.** vLLM's sampler honors `VLLM_USE_FLASHINFER_SAMPLER=1` by JIT-compiling a flashinfer sampling module on the first `profile_run` call. That JIT shells out to ninja, then to `cl.exe`. If MSVC 2022 or ninja is missing, EngineCore dies before the server boots. v0.1.15 set the flashinfer env var unconditionally to `1`, which crashed any setup without MSVC. v0.1.16 adds `flashinfer_sampler_env()` in `snapshots/_common.py` that probes whether MSVC env is usable AND ninja is on PATH, and toggles `VLLM_USE_FLASHINFER_SAMPLER` accordingly. The PyTorch fallback sampler is slightly slower but never JIT-compiles anything, so boots succeed on a vanilla Windows install. Install MSVC 2022 Build Tools + `pip install ninja` if you want the small boost back.

Reported by kadeshar in issue #2; thanks for the full log.

TUNING.md no longer lists `VLLM_USE_FLASHINFER_SAMPLER=1` as a "no downside" env var. TROUBLESHOOTING.md gets a new row for the run_ninja failure.

## What's new in v0.1.15

One bug, exposed by v0.1.14 finally letting boot get this far:

- **`ValueError: CUDA_LIB_PATH is not set` from flashinfer at EngineCore init.** vLLM 0.19 unconditionally imports flashinfer in `topk_topp_sampler.py`, regardless of which attention backend you select. flashinfer's Windows path raises at import time if `CUDA_LIB_PATH` is missing, even though the shipped snapshots use TRITON_ATTN and never trigger flashinfer JIT. New `cuda_env()` in `snapshots/_common.py` probes `CUDA_PATH`, `CUDA_HOME`, and standard NVIDIA Toolkit install dirs and sets `CUDA_LIB_PATH` before launching vLLM. If no Toolkit is installed, a placeholder path is set so the import check passes; TRITON_ATTN never needs the real libs.

Reported by Shustrik116 in the launch thread.

## What's new in v0.1.14

Three bugs reported by Shustrik116 in the launch thread:

- **`msvc_env()` no longer crashes the snapshot on a vcvars64.bat parse failure.** Wrapped the `subprocess.check_output` call in try/except and downgraded failure to a warning. Shipped snapshots use TRITON_ATTN, not FlashInfer JIT, so an MSVC env failure should never stop boot. The previous build crashed all 4 single-GPU snapshots on machines where vcvars64.bat's call to vcvarsall.bat tripped a cmd quoting glitch.
- **`start_pp2_160k.bat` now calls `start_pp2.py` instead of the non-existent `start_pp2_ngram.py`.** Old name was a leftover from an earlier ngram-spec-decode experiment. The PP=2 + ngram path was removed; only PP=2 with no spec-decode ships.
- **Removed the `dflash` config from `configs.yaml`.** It was a personal dev-tree config pointing at `C:\_projects\luce-dflash\`, which obviously does not exist on user machines. Should never have shipped.

## What's new in v0.1.9

Docs-only cleanup ahead of the public launch:

- **README:** the [`docs/AGENT_INSTALL_PROMPT.md`](docs/AGENT_INSTALL_PROMPT.md) hand-off prompt is now linked prominently from the Install section. Edit one line, paste into Claude Code / Cursor / Codex CLI / any agent with shell access, and it does the entire install + smoke test hands-off.
- **README:** broadened the "works with" client list, any OpenAI-compatible client (Continue, LM Studio, OpenWebUI, etc.), not just agent CLIs.
- **RELEASING.md:** `devnen/vllm-windows` is public now, so no PAT setup is required to cut a release. The workflow uses `${{ secrets.GITHUB_TOKEN }}` directly.
- **WINDOWS_VRAM_HEADLESS.md:** genericized the example `taskkill` list.
- **HALLUCINATED_FLAGS.md / TUNING.md:** softened phrasing about outdated online recipes.

No code or wheel changes, the launcher zip is byte-identical to v0.1.8 except for the bundled docs.

## What's new in v0.1.4

- **Snapshot logs no longer start with `'vswhere.exe' is not recognized`.** `vcvars64.bat` shells out to `vswhere.exe` with no path qualifier; the VS Installer dir isn't on `PATH` by default. `_common.msvc_env()` now prefixes `C:\Program Files (x86)\Microsoft Visual Studio\Installer` to `PATH` before calling `vcvars64.bat`. Cosmetic fix, the previous version still worked, but the scary first line of every snapshot log made it look broken.
- **De-duplicated `msvc_env()` across all six snapshots.** Was copy-pasted into each `start_*.py`; now lives once in `snapshots/_common.py`.

## What's new in v0.1.3

- **First-run runtime installer.** The portable zip ships the launcher, the patched vLLM wheel (~200 MB), a vendored `get-pip.py`, and an embedded Python, but NOT the ~6 GB of transitive deps (torch + CUDA wheels + ~150 Python packages). On first launch, `setup.py` bootstraps pip, then installs the bundled wheel + deps into the embedded Python's `site-packages`. One-time, ~5–15 min, several-GB download. A marker file makes subsequent launches no-ops. Honest scope: previous releases claimed "every dependency preinstalled", that wasn't true and pressing Enter on a snapshot would crash with `ModuleNotFoundError: vllm`.
- **Snapshot scripts now find the embedded Python.** `_common.py` resolves `VLLM_EXE` to `<install>\python\Scripts\vllm.exe` when no developer venv is present. Each `start_*.bat` falls back to `..\python\python.exe` when `..\venv\` doesn't exist.
- **WT relaunch fixed for installs under `Program Files (x86)`.** `start.bat` now matches the working `portable-launcher` pattern (delayed expansion + triple-quote `cmd /c` + `activate_wt.py` foreground bring-up + `ShowWindow(0)` to hide the parent cmd). The previous space-padded form interacted badly with parens in the install path.
- **Bundled jinja chat template.** `templates/qwen3.5-enhanced.jinja` now ships in the zip, earlier builds shipped only when paired with the `vllm-windows` repo checkout.
- **Removed the `start_toolcall` snapshot + harness.** Every snapshot now ships with the tool-calling fix baked in (`qwen3.5-enhanced.jinja` + `preserve_thinking=false`), so a separate config is redundant.

## What's new in v0.1.2

- **Bundled Windows Terminal.** A portable Windows Terminal is shipped under `terminal/` and `start.bat` automatically launches the TUI inside it, no separate install required, no Microsoft Store, works on Windows 10. Falls back gracefully to plain `cmd` if the bundle is missing.

## What's new in v0.1.1

- **Truly portable.** The launcher now works from any path including `Program Files (x86)` (parens-safe `start.bat`, embedded-Python `_pth` correctly wired).
- **Auto-discover model.** Checks env var → saved config → install/models → drive scan; only prompts when nothing is found.
- **Auto-download.** When prompted, the launcher can fetch the Lorbus quant directly from Hugging Face using only stdlib `urllib`, no token, no `huggingface_hub` install needed. Resume-safe.
- **Writable-path fallback.** When the install dir is read-only (e.g. real `Program Files`), logs / downloaded models / saved config auto-route to `%LocalAppData%\qwen36-windows-server\`.
- **Visible errors.** A failure during startup no longer just flashes the cmd window and closes, the error stays on screen with a `Press a key` prompt.

## SHA256

See `SHA256SUMS.txt` in this release.

PowerShell: `Get-FileHash qwen3.6-windows-server-portable-x64.zip -Algorithm SHA256 | Format-List`.

## Tested on

Windows 10 Enterprise 22H2, 2x RTX 3090. Should work on any Ampere/Ada/Blackwell NVIDIA + Windows 10/11. See [`docs/HARDWARE.md`](https://github.com/devnen/qwen3.6-windows-server/blob/main/docs/HARDWARE.md).

## License

Apache-2.0. No telemetry. No analytics. No phone-home. Everything runs on your machine.
