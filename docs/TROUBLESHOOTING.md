# Troubleshooting

Every failure mode we've actually hit, with the fix. Sorted roughly by
how often it bites.

| Symptom | Likely cause | Fix |
|---|---|---|
| `OSError: free memory < required` at startup despite a 24 GB card | `--gpu-memory-utilization >= 0.95` on a card with the display attached | Drop to 0.92 (or use `start_gpu0_50k`) |
| `ValueError: To serve at least one request with the model's max seq len (X), N GiB KV cache is needed ...` | `--max-model-len` is higher than what fits in the available KV pool | Read `estimated maximum model length is M` from the same error and set `--max-model-len ≈ 0.99 × M`. Or run `python windows_tools\probe_max_ctx.py --snapshot snapshots\start_speed.py` |
| `TRITON_ATTN only accepts {"fp8","fp8_e4m3"}` | `fp8_e5m2` copied from a Linux recipe | Change to `fp8_e4m3`. Linux features that ship `fp8_e5m2` and TurboQuant 3-bit don't apply to this wheel. |
| `'GPUModelRunner' object has no attribute 'drafter'` at boot | ngram spec-decode + PP > 1 on vLLM 0.19.0 | Disable spec-decode for any PP > 1 config. See [`SPEC_DECODE_MATRIX.md`](SPEC_DECODE_MATRIX.md) |
| `NotImplementedError: Pipeline parallelism is not supported for this model` | MTP + PP on Qwen3-Next | Pick: TP=1 with MTP, *or* PP=2 with no spec-decode. There's no middle ground on this wheel. |
| `ValidationError: Target and draft model should have the same vocabulary size` | Vocab mismatch (Qwen3 drafter under Qwen3.5/3.6 target) | Qwen3.6-27B is vocab=248320; no small (≤2 B) Qwen3 drafter has that vocab. Don't try to use draft-model spec-decode on this model class. |
| `FileNotFoundError` during first request after fresh boot | FlashInfer JIT tripping ninja MAX_PATH | Use TRITON_ATTN. Pass `--attention-backend=TRITON_ATTN` as a CLI flag (the env var alone is ignored on 0.19.0). |
| TP=2 loads fine but decodes at ~7 tok/s | CPU-relay allreduce dominating per-layer cost | Don't use TP=2 on Windows. Use PP=2 or TP=1. |
| Boot hangs forever in worker | Wheel is unpatched upstream SystemPanic build (no CPU-relay shim on 0.19, no wildcard model name) | `python windows_tools\verify_install.py --venv venv`. If `devnen_tag` row is RED, reinstall from the launcher zip's bundled `wheels\` directory — the patches are baked into that wheel and there's nothing to apply on the side. |
| Port 5001 in use | Prior server didn't exit cleanly | `python windows_tools\tune_restart.py --port 5001` sweeps PIDs from the log file and re-launches |
| `zmq.error.ZMQError: Address in use (addr='tcp://127.0.0.1:459NN')` | Orphan EngineCore from previous run still holds an ephemeral ZMQ port | Same, `tune_restart.py` walks every `EngineCore pid=N` line in the log |
| Output appears in `reasoning` field with `content=""` and `finish_reason=length` | `max_tokens` ran out before `</think>` | Raise `max_tokens`, or append `/no_think` to the prompt, or drop `--reasoning-parser qwen3` for that workload |
| `vllm: error: unrecognized arguments: --cuda-graph-sizes ...` | Wrong flag name | It's `--cudagraph-capture-sizes` (no internal hyphen between cuda and graph) |
| `UnicodeEncodeError: 'charmap' codec can't encode character '\u2588'` | Detached launch → stdout falls back to cp1252; vLLM emits progress-bar chars | Already handled in shipped snapshots. If you wrote a custom one: `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at the top of the tee thread |
| Boot wait times out at 120 s | vLLM 27B INT4 takes ~90–110 s to first `Application startup complete` on a 3090 | Increase wait. The launcher polls every 2 s for ~3 minutes by default. |
| Mid-boot warning `decorators.py:315 ... Compiling model again due to a load failure from C:\Users\<user>\.cache\vllm\torch_compile_cache\... reason: Source code has changed since the last compilation. Recompiling the model.` | Stale `torch_compile_cache` from a previous vLLM version on the same Windows account; the cache is keyed per-user and invalidates whenever the wheel changes | Benign. The recompile is automatic and adds ~30 s to cold boot. No action needed. To suppress on subsequent boots, leave the cache alone, it'll repopulate. To force a fresh compile, delete `%USERPROFILE%\.cache\vllm\torch_compile_cache\`. |
| `Available KV cache memory: -X.XX GiB` (negative) | Trying to serve on a card where free < model + ~5 GiB activations | This is the GPU0-with-desktop case. Switch to GPU1, or shrink the model, or close everything. Lowering `--max-num-batched-tokens` to 512 saves ~2 GiB activation but rarely enough for 27B. |
| Coherent for 30 tokens then "the the the" mid-sentence | KV-dtype too aggressive for this model class | Drop to BF16 baseline, then step back up. See [`COHERENCE.md`](COHERENCE.md). |
| Tokenizer load fails with "tokenizer_class 'TokenizersBackend' is not recognised" | Lorbus AutoRound's custom class name | **Auto-fixed since v0.1.5**, the launcher patches `tokenizer_config.json` on every boot. Manual recovery (e.g. when running snapshots without the launcher): `python windows_tools\patch_tokenizer.py G:\_models\Qwen3.6-27B-int4-AutoRound`. |
| Coherent output but `draft_acceptance_rate ~ 0.0` | MTP head was quantised to INT4 by the quant author and silently skipped | Use `Lorbus/Qwen3.6-27B-int4-AutoRound` specifically. See [`MTP_HEAD.md`](MTP_HEAD.md). |
| Launcher silently picked the wrong `Qwen3.6-27B-int4-AutoRound` directory (you have several on disk) | The drive scan matches by folder name only | Since v0.1.7 the launcher prints `[model] using <path>  (source: …)` at boot and warns when a drive-scan match isn't from `Lorbus/...`. To force a specific dir: `start.bat --model-dir "X:\path\to\Lorbus\Qwen3.6-27B-int4-AutoRound" --snapshot start_72tps`, or set `$VLLM_MODEL_DIR`. |
| Launcher TUI looks broken in legacy cmd | Console is too old for VT sequences | Install Windows Terminal (free in the Microsoft Store). The launcher tries to relaunch into it automatically. |
| `start.bat` opens a cmd window that flashes and closes immediately, no TUI ever appears | The launcher hides the original cmd window before relaunching into the bundled `terminal\WindowsTerminal.exe`. If WT silently fails to start (antivirus quarantined the exe during extraction, missing VC++ 2015-2022 runtime, zero-byte WT exe), the cmd is already hidden and just exits. | Open a cmd or PowerShell window first, `cd` into the install folder, set `VLLM_NO_WT=1` (PowerShell: `$env:VLLM_NO_WT=1`), then run `start.bat`. The output stays in your window. If it boots cleanly that way, the WT relaunch is the issue: confirm `terminal\WindowsTerminal.exe` exists and is not zero bytes, double-click it to confirm it opens, install the [VC++ 2015-2022 redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist), and check your antivirus quarantine. |
| Boot fails with `cudaErrorNoKernelImageForDevice` on `torch.zeros` (or any early CUDA call) on RTX 50-series | You extracted the **default** zip on a Blackwell GPU. The default zip bundles `vllm-0.19.0+devnen.1` (CUDA 12.6 / cu126), which has no sm_120 kernels. | Re-download the **`-blackwell`** release zip (`qwen3.6-windows-server-portable-x64-blackwell.zip`). It bundles `vllm-0.20.0+cu132.devnen.1` against cu130 torch and the launcher autodetects the matching torch index from the wheel filename. Driver 596+ required. |
| Boot fails on Blackwell with `OSError: cannot load library 'cudart64_13.dll'` from flashinfer | The Blackwell zip's CUDA 13 runtime shim (`cuda13_shim/bin/`) was deleted, never built, or `CUDA_PATH` was overridden. The launcher rebuilds it from `<venv>/Lib/site-packages/torch/lib/` on every boot, but only if torch is already installed. | Re-run the launcher; first boot recreates the shim. If a custom `CUDA_PATH` env var is exported in your shell, unset it (the launcher's snapshots then point `CUDA_PATH` at `<install>/cuda13_shim/`). System CUDA 13 toolkit install also works as a fallback. |
| Blackwell boot succeeds but `start_speed` / `start_127k` snapshots fail with "Unknown vLLM environment variable: VLLM_ATTENTION_BACKEND" | vLLM 0.20.0 dropped the env-var form; only the `--attention-backend` CLI arg is read. | Harmless warning if you're on the Blackwell zip's `start_5090` snapshot (it doesn't set the env var). For the legacy snapshots running on 0.20, the env var is ignored — backend is still picked from the CLI flag. To silence the warning, drop the `VLLM_ATTENTION_BACKEND` line from your custom snapshot. |
| Back-to-back snapshot launches fail with "Address in use 127.0.0.1:29550" only on Blackwell zip | vLLM 0.20.0 hardcoded `data_parallel_rpc_port=29550`. Orphaned engine cores from a crashed parent hold the port across runs. | Shipped snapshots in this project pass `--data-parallel-rpc-port=<random>` via the `random_dp_rpc_port()` helper, so the launcher path is unaffected. If you launch vLLM directly without the helper: `netstat -ano -p tcp \| grep ":29550"` then `taskkill /F /PID <pid>`. |
| Triton fails to JIT-compile (`cuda_utils.c` errors, `cl.exe exited with status 2`) even after MSVC 2022 is installed | Triton couldn't find the embedded Python's `Include/` directory where it expected. Symptom reported on first request after fresh install. | Workaround: copy the Python `Include/` and `libs/` directories from a system Python 3.12 install into the launcher's `python\` directory, mirroring the same subpaths. The proper fix is in the runtime installer (tracked); if you hit this, please open an issue with your exact source and destination paths so the launcher can do this for the next user. |
| Boot fails with `ValueError: CUDA_LIB_PATH is not set` from `flashinfer/jit/__init__.py` during EngineCore init | vLLM 0.19 unconditionally imports flashinfer in `topk_topp_sampler.py` regardless of `--attention-backend`. flashinfer's Windows path raises at import time if `CUDA_LIB_PATH` is missing. | **Auto-fixed since v0.1.15**, the snapshots probe `CUDA_PATH`, `CUDA_HOME`, and standard NVIDIA install dirs and set `CUDA_LIB_PATH` before launching vLLM. Manual fallback if you're on an older release: `set CUDA_LIB_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4` (adjust path) before running `start.bat`. |
| Boot fails with `FileNotFoundError [WinError 2]` from `flashinfer/jit/cpp_ext.py:run_ninja` during `profile_run`. Trace passes through `flashinfer_sample` -> `top_k_mask_logits` -> `gen_sampling_module().build_and_load()`. | The flashinfer sampler path is enabled (`VLLM_USE_FLASHINFER_SAMPLER=1`) but ninja or `cl.exe` is missing on the system. flashinfer JIT-compiles a sampling module on the first `profile_run` call, which shells out to ninja then to `cl.exe`. Either step missing kills EngineCore. | **Auto-fixed since v0.1.16**, snapshots probe MSVC env (`vcvars64.bat` succeeded) plus `shutil.which("ninja")` and force `VLLM_USE_FLASHINFER_SAMPLER=0` when either is missing. The PyTorch fallback sampler is slightly slower but never JIT-compiles anything. Since v0.1.17 the launcher zip ships ninja itself, so installing Visual Studio 2022 Build Tools (free, "Desktop development with C++" workload) is enough to get the boost back, no extra `pip install`. Manual fallback for users on older releases: `set VLLM_USE_FLASHINFER_SAMPLER=0` before `start.bat`. |
| `[warn] vcvars64.bat not found at ...\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat` even though Visual Studio is installed | The snapshot's MSVC probe used to hardcode `\Microsoft Visual Studio\2022\<edition>\` paths only, which misses VS 2026 (version 18.x, installed under `\2026\`) and any non-default install location. | **Auto-fixed in the latest launcher**, the probe now shells out to `vswhere.exe` first (which the VS Installer always drops at `C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe`) and asks for the latest install with the C++ x64 toolset. Falls back to the hardcoded VS 2022 list only when vswhere is missing. If you still see the warning, set `VLLM_WINDOWS_VCVARS` to the absolute path of `vcvars64.bat` for your install. |
| Boot fails with Triton `error C2059: syntax error: '}'` while compiling `__triton_launcher.c`, traceback passes through `fused_gdn_gating` and `triton/runtime/build.py` | `vcvars64.bat` activated the older MSVC 14.38 toolset whose `cl.exe` rejects the empty-brace initializers (`T x = {};`) Triton emits. Hits systems with multiple VS Build Tools side-by-side (e.g. 14.38 + 14.44). | **Auto-fixed in the latest launcher**, `msvc_env()` in `snapshots/_common.py` now scans `<install>\VC\Tools\MSVC\` and pins `VCToolsVersion` to the newest installed toolset before invoking `vcvars64.bat`. Manual fallback for older releases: add `set VCToolsVersion=14.44.35207` (or whichever toolset version is in `C:\Program Files\Microsoft Visual Studio\2022\<edition>\VC\Tools\MSVC\`) at the top of `start.bat`. Long-term cleanest fix is to uninstall the older toolset via the VS Installer. |
| DLL load fails with `cudart64_120.dll not found` on a machine with `cudart64_12.dll` already present | flashinfer's `jit/__init__.py` does an absolute-path `ctypes.CDLL("<CUDA_PATH>/bin/cudart64_120.dll")` at import time. NVIDIA's CUDA 12.x toolkit ships the runtime as `cudart64_12.dll` (the naming changed between 11.x's `cudart64_110.dll` and 12.x's single-major form), so flashinfer crashes EngineCore on every modern Toolkit install. | **Auto-fixed in the latest launcher**, `cuda_env()` in `snapshots/_common.py` probes the CUDA bin dir and copies `cudart64_12.dll` to `cudart64_120.dll` on first launch. The CUDA install dir is usually under Program Files, so the copy can fail with `PermissionError`. In that case the snapshot prints a one-line `copy "..." "..."` command to run once from an elevated cmd; that fixes it permanently. |

## Reading the logs

Three log locations matter, in order of how often you'll touch them.

### `logs\vllm_server.<port>.log` — the engine log

The vLLM serving process tees its stdout here. Most failures show up
in this file. Useful greps:

```powershell
# Did boot complete?
Get-Content logs\vllm_server.5001.log | Select-String "Application startup complete"

# How big is the KV pool, and how much headroom do you have?
Get-Content logs\vllm_server.5001.log | Select-String "GPU KV cache size|Maximum concurrency"

# MTP working?
Get-Content logs\vllm_server.5001.log | Select-String "draft_acceptance|system_efficiency"

# Tail in real time
Get-Content logs\vllm_server.5001.log -Wait
```

The two oracle lines for context tuning live here:

```
INFO ... [kv_cache_utils.py:1319] GPU KV cache size: N tokens
INFO ... [kv_cache_utils.py:1325] Maximum concurrency for X tokens per request: Y.YYx
```

Trust the `Maximum concurrency` line. `safe_max_ctx ≈ X × Y`. The
`GPU KV cache size` line is a derived ceiling on this wheel, not the
physical pool. See [`TUNING.md`](TUNING.md#context) for the full
oracle workflow.

### `logs\runtime\<port>.json` — the runtime manifest

Each running snapshot writes one file here at boot. The launcher's
dashboard reads these to decide which card lights up. Schema:

```json
{
  "id": "speed",
  "port": 5001,
  "wrapper_pid": 12345,
  "engine_pid": 12356,
  "started_at": "2026-05-05T20:30:00",
  "snapshot_py": "C:\\...\\start_speed.py"
}
```

If the dashboard shows the wrong card running, look here first. A
stale `<port>.json` from a crash that skipped the finally block can
confuse the dashboard for one poll tick (~2 s); the next poll GCs it.
If a `<port>.json` points at a dead pid AND the dashboard still
mis-reports, that's a bug worth a GitHub issue with the manifest
contents and a `tasklist /fi "pid eq <wrapper-pid>"` output.

### `logs\runtime\` deletes itself when needed

Stop hooks delete the manifest when a snapshot exits cleanly. Crashed
snapshots leave a stale manifest behind; the launcher detects it via
a probe (`socket.connect_ex` on the port plus a process-alive check
on the wrapper pid) and either GCs the file or surfaces the orphan.

### Finding orphan PIDs to kill

If a port is wedged after a crash:

```powershell
# Who's holding port 5001?
netstat -ano -p tcp | Select-String ":5001 "

# Who's holding the DP RPC port (0.20.0 hardcoded 29550)?
netstat -ano -p tcp | Select-String ":29550 "

# Kill by pid
taskkill /F /PID <pid>
```

Or use the bundled cleanup tool:

```powershell
python windows_tools\tune_restart.py --port 5001
```

`tune_restart.py` regex-parses the engine log for every
`EngineCore pid=N` line and kills each one, then relaunches the
snapshot. Useful between bench sweeps.

### Detached / cmd-flashes-and-closes log

If `start.bat` flashes a cmd window and disappears, no engine log
gets written. The cause is the WT relaunch detaching while the
launcher's PowerShell hide-window call has already hidden the cmd.
See the `start.bat opens a cmd window that flashes and closes`
row in the table above for the exact recovery (set `VLLM_NO_WT=1`,
run from an open shell).

## When opening an issue

Please include:

1. GPU model + driver version (`nvidia-smi -q | head -25`)
2. Windows build (`winver`)
3. The snapshot you launched
4. The relevant slice of `logs\vllm_server.<port>.log`, the boot section
   plus 50 lines around the failure
5. Output of `python windows_tools\verify_install.py`
6. Whether the same prompt works on a known-good config (e.g. drop to
   `--enforce-eager`, MTP off, ctx=8000)

The [bug report template](../.github/ISSUE_TEMPLATE/bug_report.md) prompts
for these.
