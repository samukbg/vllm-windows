# RTX 50-series (Blackwell) on this launcher

Single landing page for everything Blackwell. If you have an RTX 5060,
5070, 5080, or 5090 on Windows and want to run Qwen3.6-27B natively
(no WSL, no Docker), this is the page.

## TL;DR

1. Download `qwen3.6-windows-server-portable-x64-blackwell.zip` from
   the [Releases page](https://github.com/devnen/qwen3.6-windows-server/releases).
   **Not** the default zip — the default is for 30/40-series only.
2. Make sure your NVIDIA driver is **596 or newer** (CUDA 13 is
   required). `nvidia-smi` shows the driver version.
3. Extract anywhere, double-click `start.bat`, pick a 5090 snapshot
   — `rtx5090_speed` (120k ctx, fastest), `rtx5090` (200k ctx,
   balanced), or `rtx5090_max` (280k ctx).

That's it. The launcher autodetects the bundled wheel as a CUDA 13
build and installs the right torch index (cu130) plus a runtime shim
on first boot. No CUDA Toolkit install required.

## Why two zips

The default `qwen3.6-windows-server-portable-x64.zip` ships
`vllm-0.19.0+devnen.1` against CUDA 12.6 / PyTorch cu126. That torch
build has no `sm_120` kernels, so on Blackwell it boots cleanly to the
first `torch.zeros` call and dies with
`cudaErrorNoKernelImageForDevice`. There is no wheel-side workaround.

The Blackwell zip ships `vllm-0.20.0+cu132.devnen.1` against CUDA 13.2
/ PyTorch cu130, which has `sm_120` kernels. Same launcher, same
snapshots, different wheel.

We keep them as separate releases because forcing every existing
30/40-series user to install a CUDA 13 driver is a breaking change
for installs that work today. The Blackwell zip will run on Ampere
and Ada too if the host has a 596+ driver, but the default zip is the
recommended path for non-Blackwell users.

## What's verified

End-to-end on a single RTX 5090 (driver 596.36, sm_120, 32 GB) on
2026-05-05:

| Check | Result |
|---|---|
| Wheel boots, model loads | yes; ~17 s to load 17 GB AutoRound INT4 |
| `/v1/chat/completions`, `/v1/messages`, `/v1/responses` | yes |
| Marlin sm_120 + AutoRound INT4 | works; Marlin selects `MarlinLinearKernel` for `GPTQMarlinLinearMethod` on first load. The `scalar_types.int4` Marlin sm_120 bug from older vLLM versions is **fixed** in 0.20.0. |
| TP=1 + MTP n=6 | works |
| Decode tok/s | **130.9 tok/s** on `rtx5090_speed` (ctx 120k, MTP n=6, mem_util 0.95, 39-token prompt, 300-token completion). 24k-prompt prefill ~2700 tok/s, sustained decode ~89 tok/s. KV pool 80k–95k tokens, max concurrency 2.0–2.4×. |
| CUDA 13 toolkit on host | **not required**. The launcher copies torch's bundled `cudart64_13.dll`, `cublas64_13.dll`, etc. from `venv\Lib\site-packages\torch\lib\` into a writable `cuda13_shim\bin\` and points `CUDA_PATH` there so flashinfer's import-time `CDLL` succeeds. |

## What's NOT yet validated on Blackwell

These rows in
[`SPEC_DECODE_MATRIX.md`](SPEC_DECODE_MATRIX.md) are 0.19-era and have
not been re-tested on the 0.20 wheel that ships in the Blackwell zip:

- **PP=2 + MTP**: was blocked on 0.19 with `Qwen3NextMTP` /
  `SupportsPP NotImplementedError`. May or may not be fixed on 0.20.
- **PP=2 + ngram**: was blocked on 0.19 with the missing `drafter`
  attribute. May or may not be fixed.
- **TP=2 numbers**: 0.19 was ~7.5 tok/s (unusable) because we ran the
  CPU-relay patch. 0.20 ships NCCL on Windows (experimental), which
  removes the CPU-relay floor — TP=2 numbers may be very different.
- **MTP-on-Blackwell tuning**: the 0.19 sweep peaked at MTP n=6 ctx
  90k for 64.5 tok/s on a 3090. The 5090 has more memory bandwidth
  and a different cudagraph profile; the optimum almost certainly
  shifts. Bench yours and post numbers.
- **Async scheduling**: on by default in 0.20.0 (was opt-in on 0.19).
  The current decode numbers were taken with the default; cross-zip
  comparisons (Ampere zip vs Blackwell zip on a 3090) need a fresh
  bench.

If you have a 2× 5090 box or a 5090 + 3090 box, please boot the
Blackwell zip on it, run the `pp2_160k` snapshot, and post numbers.

## The 5090 snapshots

The Blackwell zip ships three single-card 5090 snapshots, all GPU0,
all port 5001, all attention backend TRITON_ATTN, all KV dtype
fp8_e4m3, all with a randomised `--data-parallel-rpc-port` (see
"RPC port leak" below) and **no** `VLLM_ATTENTION_BACKEND` env var
(deprecated in 0.20.0; the CLI flag still works):

| Snapshot         | ctx  | MTP n | mem_util | Short decode | 24k decode | 24k prefill | Use it when |
|------------------|------|-------|----------|--------------|------------|-------------|-------------|
| `rtx5090_speed`  | 120k | 6     | 0.95     | **130.9 tok/s** | ~89 tok/s  | ~2700 tok/s | Default — 120k is enough for almost everything and you want max decode and 2× KV concurrency for batching. |
| `rtx5090`        | 240k | 6     | 0.95     | 124.9 tok/s     | 89.3 tok/s | 2796 tok/s  | Balanced: bigger ctx than `_speed`, MTP n=6 still on. v1.2.1 upgrade replaces the v1.2.0 200k/0.93 build. |
| `rtx5090_max`    | 280k | 3     | 0.95     | **138.0 tok/s** | 81.0 tok/s | 2818 tok/s  | Largest single-card context. n=3 frees KV-pool budget so 280k still fits with MTP on. Edges out n=6 on short prompts. |

All three beat every 3090 snapshot on context size at the same MTP n.

If you want a different combo, `e` on the dashboard opens the
snapshot editor. Duplicate any of the three, edit, save. The launcher
rewrites both the YAML and the `.py` for you.

## Driver and toolkit requirements

| Component | Requirement |
|---|---|
| GPU | RTX 5060 / 5070 / 5080 / 5090 (any sm_120) |
| NVIDIA driver | 596 or newer |
| CUDA Toolkit | not required on the host |
| Visual Studio Build Tools | optional (small flashinfer-sampler decode boost; otherwise the launcher uses the PyTorch fallback sampler) |
| Disk | ~10 GB for the launcher install + ~17 GB for the model weights |
| RAM | 16 GB+ recommended |
| Windows | 10 22H2 or 11 |

The launcher refuses to boot on a Blackwell GPU with the cu126 wheel
(it surfaces a preflight error pointing at this page) and refuses to
boot on a non-Blackwell GPU with a forced cu126 install via
`--variant ampere` if the host driver is too old.

## Switching from the default zip to the Blackwell zip

If you already have the default zip extracted and want to switch
without losing your `models\`, `logs\`, custom snapshots:

```
update.bat --variant blackwell
```

See [`UPGRADING.md`](UPGRADING.md) for the full updater story.

## The CUDA 13 runtime shim, in detail

flashinfer 0.4.x does an absolute-path `CDLL` of `cudart64_13.dll` at
import time. NVIDIA's CUDA 13 toolkit ships that DLL in
`<CUDA_PATH>\bin\`, but installing the toolkit just to satisfy a
`CDLL` call would be silly when torch already bundles every CUDA 13
runtime DLL it needs.

What the launcher does on first boot:

1. Looks at `venv\Lib\site-packages\torch\lib\` and finds
   `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`,
   `cudnn64_*.dll`, etc.
2. Creates `cuda13_shim\bin\` next to the launcher and copies the
   DLLs there.
3. Sets `CUDA_PATH=<install>\cuda13_shim\` for the snapshot
   subprocess.
4. flashinfer's import-time `CDLL` succeeds.

This is idempotent (runs every boot, skips DLLs already present),
cheap (~5 MiB of file copies), and doesn't touch your system CUDA
install. If you ever delete `cuda13_shim\`, the next launcher boot
recreates it from torch's `lib\`.

## The 29550 RPC port leak

vLLM 0.20.0 hardcodes `data_parallel_rpc_port=29550` in its
`ParallelConfig` default. When an engine-core child orphans (parent
crash, ctrl-C during boot), the port stays held until the orphan is
killed. Back-to-back snapshot launches then deterministically fail
with `Address in use 127.0.0.1:29550`.

The shipped snapshots in this project pass
`--data-parallel-rpc-port=<random>` via a helper in
`snapshots/_common.py`, so you won't hit this on the launcher path.

If you're invoking vLLM directly without the helper:

```powershell
netstat -ano -p tcp | Select-String ":29550"
taskkill /F /PID <pid>
```

## `VLLM_ATTENTION_BACKEND` is gone in 0.20

On 0.19.0, the env var was silently ignored and the CLI flag was
load-bearing. On 0.20.0, the env var is genuinely unrecognised and
emits a one-line `Unknown vLLM environment variable:
VLLM_ATTENTION_BACKEND` warning. The CLI flag
(`--attention-backend=TRITON_ATTN`) is the right answer either way.

The three shipped `rtx5090*` snapshots do not set the env var. The
other snapshots (carried over from the 0.19 path) still set it; on
the Blackwell zip those produce the warning above and otherwise
behave identically. To silence the warning, drop the env-var line
from the snapshot or rebuild it via the in-TUI editor.

## When to keep using WSL2 / the Blackwell guide instead

The community
[vllm-blackwell-guide](https://github.com/lastloop-ai/vllm-blackwell-guide)
has reported up to 120 tok/s on 27B and 200 tok/s on the 35B MoE on a
5090, on tuned upstream vLLM running in WSL2. That stack:

- ships pure-upstream vLLM (this project's wheel is mostly upstream
  with a small reasoning-parser tweak and a wildcard model name;
  features like NVFP4 KV that landed in upstream after 0.20 are not
  in our wheel),
- pays the WSL2 tax on the GPU (one community measurement: 85 tok/s
  in WSL vs 160 tok/s in native Ubuntu).

Pick that route if you specifically need an upstream feature that
isn't in our 0.20 base, or if you're already comfortable in Linux
and would rather pay the WSL tax. Pick this launcher if you want
native Windows decode with Anthropic-API-compatible serving and the
shipped tool-calling fixes.

## Reporting Blackwell numbers

If you bench the Blackwell zip, please post:

- GPU model + driver version (`nvidia-smi --query-gpu=name,driver_version --format=csv`)
- Snapshot id (`rtx5090_speed`, `rtx5090`, `rtx5090_max`, or your custom one)
- Output of `windows_tools\check_coherence.py --port 5001` (decode
  tok/s without coherence is meaningless)
- Output of `windows_tools\bench_summarize.py` (a single TSV row
  with prefill / decode / TTFT)
- The `Maximum concurrency for X tokens per request: Y.YYx` line
  from `logs\vllm_server.5001.log` (tells us how much KV headroom
  you had)

A Reddit reply, a GitHub issue, or a PR with a new
`launcher\configs.yaml` row are all welcome.

## Related docs

- [`HARDWARE.md`](HARDWARE.md), full GPU compatibility table.
- [`INSTALL.md`](INSTALL.md), full install procedure including
  picking the right zip.
- [`UPGRADING.md`](UPGRADING.md), in-place updater and variant
  switching.
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md), Blackwell-specific
  failure rows are at the bottom of the table.
- [`SPEC_DECODE_MATRIX.md`](SPEC_DECODE_MATRIX.md), parallelism /
  spec-decode combos. The 0.19-era results need re-validation on 0.20.
- [`HALLUCINATED_FLAGS.md`](HALLUCINATED_FLAGS.md), the
  `VLLM_ATTENTION_BACKEND` env-var deprecation note for 0.20.
