# qwen3.6-windows-server

> **One-click [Qwen3.6-27B](https://huggingface.co/Qwen) inference on Windows.**
> Unzip, double-click, you're serving on `http://127.0.0.1:5001/v1`.
> No WSL, no Docker, no conda, no pip, no admin. **Everything runs on
> your machine. No telemetry. No analytics. No phone-home.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Made for Windows](https://img.shields.io/badge/OS-Windows%2010%2F11-0078d6.svg)](https://www.microsoft.com/windows)
[![GPU](https://img.shields.io/badge/tested-RTX%203090-76b900.svg)](https://www.nvidia.com/)
[![Local AI](https://img.shields.io/badge/100%25-local%20%2F%20offline-success.svg)](#the-local-ai-ethos)

---

> ## v1.3.3 — PP=2 fixed on Ampere, real long-prompt bench fixture
>
> Bug-fix release. `pp2_160k` (Both-GPU big-ctx) failed to boot on the
> public Ampere zip with `ZMQError: Protocol not supported (addr='ipc://...')`
> because pyzmq has no `ipc://` transport on Windows. New patched wheels
> (`vllm-0.19.0+devnen.2` Ampere, `vllm-0.20.0+cu132.devnen.2` Blackwell)
> add a Windows-only ipc -> tcp fallback plus an Ampere-only worker-pipe
> `_ConnectionBase` widening. PP=2 now boots, is coherent, and decodes
> within ~10 % of the documented 40.3 tok/s on 2× RTX 3090.
>
> Also: `bench_summarize.py` runs from a stock install (the embedded
> `python312._pth` now includes `..\windows_tools` so `import bench`
> resolves), and `bench_prompt_sample.py` is a real ~25 k-token fixture
> (CPython 3.12 `inspect.py`) instead of a 670-token stub. Documented
> `decode_tps` numbers are now reproducible from a clean install.
>
> See [`docs/UPGRADING.md`](docs/UPGRADING.md). Single-GPU users see no
> functional change.

---

> ## v1.3.2 — Blackwell env hardening + cache-poison prevention
>
> Hotfix for a slow-prefill regression that bit RTX 5090 NVFP4 users
> when system CUDA installs (or conda `cudatoolkit`) leaked into the
> launch env. New `clean_cuda_env()` scrubs `CUDA_*`/`NVCC_*`/`CUDNN_*`
> env vars and filters NVIDIA-toolkit + conda `Library/bin` from PATH;
> new `preflight_sm120a_or_die()` 5-second probe hard-exits before the
> 11-minute warmup if FlashInfer can't dispatch sm_120a; new
> `windows_tools/wipe_caches.py` recovers the four caches that get
> poisoned, with `mv → .bak.<timestamp>` for forensics. If you upgraded
> from v1.3.0 / v1.3.1 and saw slow prefill (~750 tok/s on a 47 k NVFP4
> prompt), run `python windows_tools\wipe_caches.py` once after
> `update.bat`. See [`docs/UPGRADING.md`](docs/UPGRADING.md) and
> [`docs/BLACKWELL.md`](docs/BLACKWELL.md).

---

> ## v1.2.5 — prefix caching back on, big prefill speedup
>
> v1.2.2 turned prefix caching **off** in all 12 snapshots to dodge a
> stepwise decode regression on long-context requests
> ([vLLM issue #17140](https://github.com/vllm-project/vllm/issues/17140)),
> a real bug in Qwen3-Next's hybrid Mamba/SSM state handling.
>
> That bug is fixed upstream by
> [vLLM PR #25752](https://github.com/vllm-project/vllm/pull/25752)
> (Mamba2 Automatic Prefix Caching, merged 2025-10-04), which is in the
> wheel both release zips ship. With prefix caching enabled, vLLM
> auto-sets `mamba_cache_mode='align'` for Qwen3_5 so SSM state is
> tracked across cache blocks. **v1.2.5 turns prefix caching back on in
> every snapshot.**
>
> What you get on the 5090 path (verified, see
> [`docs/TUNING.md`](docs/TUNING.md)):
>
> - **Prefill 12k**: 686 → ~2,150 tok/s (3.1x)
> - **Prefill 16k**: 559 → ~2,030 tok/s (3.6x)
> - **Prefill 24k+**: was timing out → 2-4k tok/s
> - **Decode after 2x 24k hits**: stable, no regression (the old #17140
>   pattern of 130 → 90 → 40 tok/s does not reproduce)
> - **KV pool**: +18 % headroom (94,656 vs 79,968 tokens at ctx=240k)
> - **Warm-prefix TTFT** on a 24k re-hit: ~1.6 s vs ~42 s cold
>
> The same wheel-side fix ships in the Ampere/Ada zip's
> `vllm-0.19.0+devnen.1` build (the Mamba2 APC code is present in the
> source tree the wheel is built from). 3090/4090 snapshots are flipped
> on the same assumption; if you hit the old regression on Ampere
> hardware, please open an issue.
>
> **To get the fix:** `update.bat` (or download the new release zip and
> re-extract). See [`docs/UPGRADING.md`](docs/UPGRADING.md) for details
> and [`docs/TUNING.md`](docs/TUNING.md) for the bench tables.

---

## What this is

A small portable Windows app that gives you an OpenAI-compatible API
serving Qwen3.6-27B locally, with config presets that I actually
measured myself. The launcher is a Textual TUI: arrow keys, Enter
to start a snapshot, Esc to stop. Press `e` to add, edit, duplicate, or
delete your own snapshot configs from inside the TUI, no hand-editing
files. That's the whole UX.

It is the matching launcher for the [`devnen/vllm-windows`](https://github.com/devnen/vllm-windows)
patched wheel, but you don't need to know or care about that. The wheel
ships inside the launcher zip.

## What you get

On a single RTX 3090 (24 GB), running [Lorbus AutoRound INT4](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound):

Every snapshot below has the tool-calling fix baked in (PR #35687 + #40861 + `qwen3.5-enhanced.jinja` + `preserve_thinking=false`), so any one of them works with any OpenAI-compatible client, Claude Code, Cline, Cursor, Codex, OpenCode, KiloCode, LM Studio, etc. Just point it at the listed port.

| Snapshot              | Decode tok/s | Prompt class      | Context | Use it when |
|-----------------------|--------------|-------------------|---------|-------------|
| `start_72tps`         | **~72**      | short (~200 tok)  | 32 k    | Short-prompt / chat baseline. MTP n=3. |
| `start_speed`         | **64.5**     | long (100 KB)     | 90 k    | Default for long prompts. MTP n=6, see note below. |
| `start_127k`          | 53.4         | long (100 KB)     | 127 k   | Maximum context on a single 3090. |
| `start_mtp4`          | 58.3         | long (100 KB)     | 120 k   | Mid-balance speed vs context. |
| `start_pp2_160k` (2 GPU) | 43.5      | long (100 KB)     | 160 k   | Pipeline-parallel for the largest contexts. |
| `start_gpu0_50k`      | 56.9         | mixed             | 9–50 k  | Single-GPU + display, fallback when you can't boot-quiet. |

> **GPU index note.** `start_72tps`, `start_speed`, `start_127k`, and
> `start_mtp4` pin to **GPU 1** so GPU 0 stays free for the desktop
> compositor and other apps on a 2× 3090 box. On a single-GPU host the
> snapshot detects that via `nvidia-smi` and falls back to GPU 0 with a
> warning. `start_pp2_160k` requires two GPUs.

> **Single 3090 with display attached.** You can run the full
> `start_speed` snapshot at 90 k context if you close heavy GPU apps
> (Chrome, Discord, Slack, video playback) **during boot**. Once vLLM
> has reserved its KV pool, the driver schedules everything else around
> what vLLM already owns, so you can reopen those apps and they'll
> behave normally. The danger is reopening them before boot finishes,
> mid-allocation OOM is what kills runs. If you can't or won't
> boot-quiet, `start_gpu0_50k` is the conservative fallback (`mem_util
> 0.92`, ~50 k ctx, same decode tok/s).

Long-prompt rows were measured on a ~100 KB / ~24 k-token Python
source-summary prompt (a real Windows-service module fed to
`windows_tools\bench_summarize.py`). The short-prompt row was measured
on a ~200-token chat turn via `windows_tools\bench.py`. All numbers
[coherence-validated](docs/COHERENCE.md), TPS without coherence is a
lie.

> **Why MTP n=6 on `start_speed`?** n=3 is the universal *short-prompt*
> sweet spot and ships as `start_72tps`. On long, dense Python source
> the acceptance curve shifts later, n=6 won my coherence sweep
> (n=3 / 4 / 5 / 6 / 7 / 8 → 53.4 / 58.3 / 62.8 / 64.5 / 61.5 / 58.0
> tok/s; full sweep in [`docs/TUNING.md`](docs/TUNING.md)). Always
> re-sweep on a representative prompt for your workload.

> **Honest framing:** these are not r/LocalLLaMA records. Community has
> hit 80–82 tok/s on a 3090 with TurboQuant 3-bit KV, and 160 tok/s on a
> 5090. The unique angle here is **native Windows, no WSL**. Same
> recipe, no virtualization tax, one community member measured the
> same hardware going from **85 tok/s in WSL to 160 tok/s in native
> Ubuntu** ([reported here](https://www.reddit.com/r/LocalLLaMA/comments/1sw21op/comment/oid8d9n/)).
> This launcher closes that gap on Windows.

## Why this exists

Most fast Qwen3.6-27B recipes on r/LocalLLaMA assume Linux + Docker, or
Linux-in-WSL. Windows users either pay the WSL tax, dual-boot, or skip
inference entirely. None of those is great if your daily driver is
Windows.

This launcher is the third option:

- **Native Windows.** Runs as a normal Windows process. No virtualization layer.
- **Portable.** Unzip the launcher, drop your model into a folder, double-click. That's it.
- **Validated.** Every config in here was measured against a coherence battery before being checked in. No copy-pasted Reddit recipes that look fast but emit `* * * *`.
- **Local-only.** No outbound calls except when you explicitly ask the launcher to download a model from HuggingFace. No telemetry of any kind, ever.

## Install

**TL;DR for CI / agents / scripted installs**, one line, no TUI:

```powershell
start.bat --auto-download --snapshot start_72tps
```

That installs the runtime, downloads the model if missing, and starts
serving on `http://127.0.0.1:5001/v1`. See
[Headless / scripted install](#headless--scripted-install) below for
all the flags.

**Hand the install to a coding agent**, copy/paste prompt at
[`docs/AGENT_INSTALL_PROMPT.md`](docs/AGENT_INSTALL_PROMPT.md). Edit the
one `INSTALL_DIR` line, paste into Claude Code / Cursor / Codex CLI /
any agent with shell access, and it does the download + extract +
runtime install + model fetch + smoke test end-to-end while you do
something else.

**Interactive path:**

1. Download [`qwen3.6-windows-server-portable-x64.zip`](../../releases/latest)
   from the latest Release. Extract anywhere (no admin needed).
2. Double-click `start.bat`. The first run does two one-time steps,
   then drops you in the TUI:
   - **Runtime install** (~5–15 min, several GB). The bundled vLLM
     wheel + ~150 transitive deps (torch, CUDA wheels, transformers,
     etc.) install into the embedded Python's `site-packages`. A
     marker file is written so subsequent launches skip this entirely.
   - **Model setup.** Looks for `Qwen3.6-27B-int4-AutoRound` weights
     on your fixed drives (scans `<drive>:\`, `_models\`, `models\`,
     `AI\`, `AI\models\`, `huggingface\`, `huggingface\hub\`,
     `models\Lorbus\`). If it doesn't find them, offers to
     **auto-download from Hugging Face** (~16 GB, public, no token)
     or accepts a path to weights you already have. If your weights
     live somewhere else, pass `--model-dir <path>` to skip the scan.
3. Pick a snapshot, press Enter, you're serving on
   `http://127.0.0.1:5001/v1`.

The portable zip ships with an embedded Python 3.12 runtime, the
patched vLLM wheel, the launcher TUI, a portable Windows Terminal,
and a vendored `get-pip.py`. No conda, no system-Python install, no
registry changes, no admin prompts. The runtime install on first run
is the only network-dependent step besides the model download.

Don't have the model yet? See [`docs/MTP_HEAD.md`](docs/MTP_HEAD.md),
**use the Lorbus AutoRound quant**, the others won't draft.

Detailed install (including the wheel-only path for users who already
have their own venv): [`docs/INSTALL.md`](docs/INSTALL.md).

## Optional: install MSVC 2022 for the small decode boost

The launcher works on a vanilla Windows install, no MSVC required.
But if you install **Visual Studio 2022 Build Tools** (free, no full
IDE) with the **"Desktop development with C++"** workload, the
snapshots auto-detect it and turn on vLLM's flashinfer sampler path,
which JIT-compiles a faster top-k / top-p kernel on first launch.

What it costs:
- ~7 GB download, one-time install.
- Extra 30 to 60 s on the first `profile_run` of each new snapshot
  while the kernel compiles. Subsequent boots reuse the compiled
  cache.

What you get:
- A small but measurable decode boost on the sampler path.

Without MSVC, the snapshots transparently fall back to the PyTorch
sampler, which never JIT-compiles anything. Boot is faster and the
server is reliable; you just leave a few percent of decode tok/s on
the table. The launcher prints a one-line `[info]` at startup telling
you which path it picked.

Get the Build Tools installer here (official Microsoft `aka.ms`
shortlink, pinned to VS 2022 / 17.x so it stays on the right product
even after VS 2026 ships):
https://aka.ms/vs/17/release/vs_buildtools.exe

ninja (the other half of the JIT toolchain) ships inside the
launcher zip, you don't need to install it separately.

## Test it

Once the server is up:

```powershell
curl http://127.0.0.1:5001/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"any\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France?\"}],\"max_tokens\":2000}"
```

Note the `"model": "any"`, the patched wheel accepts any value. You
don't have to know what the model is called.

> **Why `max_tokens: 2000`?** Qwen3.6 is a thinking model: it spends the
> first chunk of its budget reasoning inside `<think>...</think>` and
> only then writes the answer to `content`. With `max_tokens: 50` the
> entire budget gets eaten by the thinking phase and you'll see
> `content: null` plus `finish_reason: "length"`, the server is fine,
> the budget was just too small. 1500–2000 is a safe floor for short Q&A.

> **Where's the answer in the response?** The final answer lands in
> `choices[0].message.content`. The chain-of-thought lands in a separate
> `choices[0].message.reasoning` field, that's the `--reasoning-parser=qwen3`
> wheel patch doing its job, not a bug. Most chat clients show
> `content` and ignore `reasoning`; if yours doesn't, point it at
> `content`.

> **If the request hangs**, tail `logs\vllm_server.<port>.log` for vLLM's
> own stdout, the parent launcher logs only the boot banner; the
> serving process tees its progress to that file.

## Headless / scripted install

End-to-end automated install (no TUI, no prompts), useful for CI,
remote machines, agent installers, or just keeping a repeatable
recipe:

```powershell
start.bat --auto-download --snapshot start_72tps
```

The launcher runs the first-run setup (vLLM wheel + ~150 deps),
auto-downloads the Lorbus quant from Hugging Face if it's missing,
applies the tokenizer patch automatically, and execs the chosen
snapshot, all without opening the TUI. Other useful flags:

```powershell
start.bat --model-dir G:\_models\Qwen3.6-27B-int4-AutoRound --snapshot start_speed
start.bat --headless     :: skip TUI, run the default snapshot (start_72tps)
start.bat --setup-only   :: install runtime + model, then exit (no serving)
```

`--headless` without `--snapshot` now runs the default snapshot
(`start_72tps`) instead of exiting after setup checks. To run only
the setup checks (the old `--headless` behavior), pass `--setup-only`.

The launcher also stays in the parent terminal, instead of detaching
into a new Windows Terminal window, when it sees any of `WT_SESSION`,
`VLLM_NO_WT`, `CI`, `GITHUB_ACTIONS`, `MSYSTEM`, or `TERM` in the
environment. That covers GitHub Actions, git-bash, MSYS, agent
runners, and anything that exports `TERM`. So your captured stdout
won't go missing.

For benchmark numbers like the table above, use the bundled tools:

```powershell
windows_tools\bench.bat              :: short prompt, decode-only TPS
windows_tools\bench_summarize.bat    :: ~100 KB / ~24 k-token prompt, prefill + decode + KV
windows_tools\check_coherence.bat    :: 3-tier coherence validator
```

## Hardware reality

Tuned and measured on:

- Windows 10 Enterprise 22H2
- 2× NVIDIA RTX 3090 (Ampere `sm_86`), no NVLink, PCIe Gen 4
- 350 W power cap (250 W also benchmarked, see [`docs/TUNING.md`](docs/TUNING.md))

Should also work on any Ampere or Ada NVIDIA GPU running Windows 10/11,
3090, 4090, A6000, etc. **Will not work** on Pascal, Turing, Intel Arc,
or any AMD card. **Single GPU with the display attached** loses 1–3 GiB
of VRAM to the desktop compositor and another 2–5 GiB to running apps,
but you can still run the full `start_speed` snapshot at 90 k context
by closing heavy GPU apps (Chrome, Discord, Slack, video playback)
during boot, then reopening them after vLLM finishes booting. If you
can't boot-quiet, fall back to `start_gpu0_50k`. Either path is
covered in [`docs/WINDOWS_VRAM_HEADLESS.md`](docs/WINDOWS_VRAM_HEADLESS.md).

> **RTX 50-series (Blackwell, 5060 / 5070 / 5080 / 5090): supported via
> the Blackwell zip.** Download
> `qwen3.6-windows-server-portable-x64-blackwell.zip` instead of the
> default zip. It bundles `vllm-0.20.0+cu132.devnen.1` against CUDA 13.2
> / PyTorch cu130 with `sm_120` kernels. **v1.3.0 ships NVFP4 as the new
> default** (`rtx5090_nvfp4`, port 5001) using the
> [`Peutlefaire/Qwen3.6-27B-NVFP4`](https://huggingface.co/Peutlefaire/Qwen3.6-27B-NVFP4)
> weights — these route FFN GEMMs through FlashInfer's sm_120 native
> FP4 tensor cores, escaping the 170W prefill ceiling that AutoRound INT4
> hits on consumer Blackwell. Measured on a single RTX 5090 at 575W:
> **~5,300 tok/s prefill @ 47k prompt (5x AutoRound), ~92 tok/s decode**
> at 200k context. AutoRound INT4 snapshots (`rtx5090`, `rtx5090_max`)
> still ship as alternatives for the 240k/280k context profiles. NVIDIA
> driver 596+ required. See [`docs/BLACKWELL.md`](docs/BLACKWELL.md) for
> the full story and [`docs/SM120_GDN_CEILING.md`](docs/SM120_GDN_CEILING.md)
> for the prefill-ceiling investigation.

If you're on a 4090, expect slightly higher numbers than mine. If
you're on something more exotic, nothing here is going to work without
your own tuning, that's fine, please share what you find.

> **Scope.** This launcher serves Qwen3.6-27B specifically through a
> fixed set of validated snapshots. It is not a general vLLM server you
> can point at any model. Adding configs for smaller Qwen variants is
> straightforward (see [`docs/SNAPSHOTS.md`](docs/SNAPSHOTS.md));
> running unrelated models like ACE-Step, Stable Diffusion, or other
> diffusion / multimodal stacks is out of scope.

## The local-AI ethos

Everything runs on your machine. No telemetry. No analytics. No
phone-home. No cloud inference. No model weights downloaded behind your
back. The launcher never opens an outbound connection except when you
explicitly ask it to (downloading a model from HuggingFace via your own
browser/`huggingface-cli`). This is in the spirit of
[r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/): your hardware,
your weights, your prompts, your business.

The launcher and every script are Apache-2.0. The bundled wheel inherits
upstream vLLM's Apache-2.0 license. SHA256 of every release asset is
published next to the release, verify before extracting.

## What's under the hood

The wheel that powers this launcher is
[`devnen/vllm-windows`](https://github.com/devnen/vllm-windows): a
patched native-Windows build of [vLLM](https://github.com/vllm-project/vllm),
with three Windows-specific fixes (CPU-relay for Gloo collectives, Qwen3
reasoning-parser fix mirrored from PR #35687, hardwired wildcard model
name). The full diff is at
[`CHANGES_VS_SYSTEMPANIC.md`](https://github.com/devnen/vllm-windows/blob/main/CHANGES_VS_SYSTEMPANIC.md)
in that repo. You don't have to download it separately, it's bundled
inside this launcher's portable zip.

## Documentation

- [`docs/INSTALL.md`](docs/INSTALL.md), full install + the bring-your-own-venv path.
- [`docs/UPGRADING.md`](docs/UPGRADING.md), in-place updater (`update.bat`), preserve list, variant switching.
- [`docs/BLACKWELL.md`](docs/BLACKWELL.md), single landing page for RTX 50-series users.
- [`docs/MODELS.md`](docs/MODELS.md), swapping in other quants / model sizes / Qwen variants.
- [`docs/CLAUDE_CODE.md`](docs/CLAUDE_CODE.md), point Claude Code at the local server (native `/v1/messages`, no proxy).
- [`docs/CODEX.md`](docs/CODEX.md), use OpenAI Codex CLI with this server (Responses API, `developer`-role template fix).
- [`docs/OPENCODE.md`](docs/OPENCODE.md), point OpenCode at the local server (custom OpenAI-compatible provider, AGENTS.md path-handling rule).
- [`docs/QWEN_CLI.md`](docs/QWEN_CLI.md), use Alibaba's official Qwen Code agent against this server.
- [`docs/PI.md`](docs/PI.md), use the Pi coding agent with this server (custom provider extension, auto-compaction notes).
- [`docs/UNINSTALL.md`](docs/UNINSTALL.md), clean removal (it's portable, so just delete folders).
- [`docs/HARDWARE.md`](docs/HARDWARE.md), what works, what doesn't, and why.
- [`docs/COMPARISON.md`](docs/COMPARISON.md), how this stacks up against Ollama, LM Studio, llama.cpp, Docker, and WSL2.
- [`docs/COHERENCE.md`](docs/COHERENCE.md), degenerate-output guide and the 3-tier validator.
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md), every failure mode I've hit.
- [`docs/TUNING.md`](docs/TUNING.md), the lever set, anti-levers, how to sweep your own configs.
- [`docs/MTP_HEAD.md`](docs/MTP_HEAD.md), why Lorbus AutoRound is the only INT4 quant that works (the NVFP4 weights ship a separate MTP head and are documented in [`docs/BLACKWELL.md`](docs/BLACKWELL.md) and [`docs/SM120_GDN_CEILING.md`](docs/SM120_GDN_CEILING.md)).
- [`docs/SPEC_DECODE_MATRIX.md`](docs/SPEC_DECODE_MATRIX.md), what spec-decode + parallelism combos work.
- [`docs/SNAPSHOTS.md`](docs/SNAPSHOTS.md), managing snapshots from inside the TUI: keyboard shortcuts, the CRUD editor, flag invariants, hand-edit fallback.
- [`docs/WINDOWS_VRAM_HEADLESS.md`](docs/WINDOWS_VRAM_HEADLESS.md), free VRAM on Windows for single-GPU.
- [`docs/HALLUCINATED_FLAGS.md`](docs/HALLUCINATED_FLAGS.md), flags from web search results that don't exist on this wheel.
- [`docs/CREDITS.md`](docs/CREDITS.md), vLLM team, SystemPanic, Lorbus, the community.

## Contributing

Bug reports welcome, please include GPU model, driver version, Windows
build, and the relevant slice of `logs\vllm_server.<port>.log`. The
[issue template](.github/ISSUE_TEMPLATE/bug_report.md) walks you
through it.

**Share your configs.** Each snapshot in `snapshots/` is just a
validated set of vLLM flags for one hardware/model combo, plus a card
in `launcher/configs.yaml` so the launcher can list it. If you've got
a config that runs coherent and faster (or with more context) than
what's in here, please send a PR. The bar is the
[3-tier coherence check](docs/COHERENCE.md), TPS without coherence
won't be merged.

Configs I'd love to see:

- Other Qwen3.6-27B quants (FP8, additional NVFP4 variants, smaller AutoRound variants)
- Smaller Qwen models (14B, 8B, 4B) for 16 GB cards
- 4090 / 5090 / 5060 Ti / A6000 tunings
- New parallelism or KV-cache combos as vLLM adds them

How to add a snapshot: [`docs/SNAPSHOTS.md`](docs/SNAPSHOTS.md) (in-TUI editor and hand-edit fallback).

This project is intentionally narrow scope: **Windows + Ampere/Ada/Blackwell
NVIDIA**. PRs for other operating systems or GPU vendors are politely
out of scope, please go upstream.

## Credits

- [vLLM](https://github.com/vllm-project/vllm), the engine.
- [SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows), the upstream Windows wheel build infrastructure.
- [Lorbus](https://huggingface.co/Lorbus), the AutoRound INT4 quant of Qwen3.6-27B that makes the Ampere/Ada path fast.
- [Peutlefaire](https://huggingface.co/Peutlefaire), the NVFP4 quant that unlocks consumer Blackwell's full prefill throughput on the 5090.
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/), the configs in here started from recipes posted on the subreddit, and got refined by the honest feedback in the comments.
