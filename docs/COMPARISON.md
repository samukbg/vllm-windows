# How this compares to other ways to run Qwen3.6-27B on Windows

Honest framing of where this project sits versus the alternatives. If
another tool fits your workflow better, use that, no offence taken.

## Quick table

| Option | Backend | Decode tok/s (27B INT4) | MTP spec-decode | Tool calling | Multi-request | UI |
|---|---|---|---|---|---|---|
| **This project (Ampere/Ada zip on 3090)** | patched vLLM 0.19.0+devnen.1, native Windows | **64.5 to 72** (start_speed / start_72tps) | yes, MTP n=3 to n=6 | yes, OpenAI + Anthropic, baked in | yes | TUI |
| **This project (Blackwell zip on 5090, 575W)** | patched vLLM 0.20.0+cu132.devnen.1, native Windows | **158.1** (rtx5090, ctx 240k, MTP n=6) / 154.3 (rtx5090_max, ctx 280k, MTP n=3) | yes, MTP n=3 to n=6 | yes, OpenAI + Anthropic, baked in | yes | TUI |
| Ollama | llama.cpp | 30 to 40 (Q4) | no, GGUF strips the head | beta | one at a time | CLI + GUI clients |
| LM Studio | llama.cpp | 30 to 40 (Q4) | no | beta | one at a time | full GUI |
| llama.cpp direct | llama.cpp | 30 to 40 (Q4) | no native MTP, only draft-model | partial | one at a time | CLI |
| vLLM in Docker on Windows | upstream vLLM via Docker Desktop | 35 to 60 (depends on quant) | yes | yes | yes | terminal |
| WSL2 + native vLLM | upstream vLLM | 50 to 80, less than native Linux | yes | yes | yes | terminal |
| WSL2 + Blackwell guide (5090 only) | tuned vLLM cu128, see [vllm-blackwell-guide](https://github.com/lastloop-ai/vllm-blackwell-guide) | 120 (27B) / 200 (35B MoE) on 5090 | yes | yes | yes | terminal |
| Native Linux | upstream vLLM | 80 to 160 depending on stack | yes | yes | yes | terminal |

The 30 to 40 tok/s figure for llama.cpp on a 3090 is consistent across
multiple r/LocalLLaMA reports for 27B Q4 quants at short to medium
context. Decode falls off sharply with context (one user reported 50
to 60 tok/s at 4 k context dropping to 5 tok/s at 8 k due to KV cache
pressure), so the headline number is best-case.

A Reddit user (Kadeshar) posted a same-machine head-to-head on a 3090
at 90 k context: this server delivered 36 to 55 tok/s consistently
across streaming and non-streaming, both at 390 W and at a 280 W power
cap. LM Studio on the same hardware delivered 36 to 38 tok/s
non-streaming but only 5 to 13 tok/s when streaming was enabled, a 3x
to 7x streaming penalty. If you compare the two tools and only look at
non-streaming numbers, the gap looks smaller than it really is for an
agent workflow that always streams.

## Why this is faster than Ollama / LM Studio / llama.cpp

Three reasons, all stack-level not config-level:

1. **MTP spec-decode.** Qwen3.6 ships a multi-token-prediction head
   that lets the model speculate several tokens per forward pass and
   verify them in one shot. vLLM uses it. llama.cpp's GGUF conversion
   strips it. So llama.cpp tops out at the un-speculated decode rate
   while vLLM gets a real multiplier on top.
2. **AutoRound INT4 with Marlin kernels.** AutoRound preserves more
   accuracy than naive INT4 at the same bit width, and the Marlin
   kernels are dramatically faster than llama.cpp's INT4 path on
   Ampere and Ada.
3. **Continuous batching.** vLLM serves multiple requests concurrently
   from one model. llama.cpp serializes. For single-user chat the
   difference is invisible, but for an agent that fires multiple tool
   calls in flight it matters.

## When llama.cpp is the right answer instead

- **You have a 16 GB card or smaller.** Qwen3.6-27B INT4 weights are
  16.96 GiB on disk, so vLLM cannot fit them on a 16 GB card after
  activations and KV. llama.cpp can spill some layers to RAM. Use
  llama.cpp + Q4 there. See [`HARDWARE.md`](HARDWARE.md).
- **You have a Pascal or Turing card.** Marlin INT4 needs sm_80+,
  Pascal has no BF16 in hardware, vLLM is a hard no on those. llama.cpp
  works, just slower.
- **You have RTX 50-series and want llama.cpp / Ollama instead.** This
  project now ships a Blackwell-capable build
  (`qwen3.6-windows-server-portable-x64-blackwell.zip`, vLLM
  0.20.0+cu132.devnen.1 against cu130 torch) and AutoRound INT4 + Marlin
  is verified working on sm_120, so this column is no longer "must
  switch tools" — pick llama.cpp / Ollama if you prefer their UX or
  need partial-RAM offload, but vLLM is no longer Windows-blocked on
  Blackwell. WSL2 +
  [jaMMint's vllm-blackwell-guide](https://github.com/lastloop-ai/vllm-blackwell-guide)
  remains a valid alternative if you want pure-upstream vLLM with NVFP4
  or other Linux-only features.
- **You want a polished GUI.** LM Studio is good. This project ships
  a terminal TUI; you pick a snapshot and that is it.
- **You have an AMD card.** ROCm vLLM does not ship in this Windows
  wheel. Use llama.cpp with Vulkan or ROCm.

## When WSL2 / Docker on Windows is worth it

If you are already on Linux with Docker, Docker overhead is essentially
zero (it is just kernel namespaces, not a VM). Stick with what you have.

On Windows it is different. Docker Desktop runs through WSL2, which is
a real VM on Hyper-V. CUDA goes through GPU-PV paravirtualisation, the
Windows host driver still owns the GPU and DWM keeps its allocation.
Same hardware, same model, one community member measured **85 tok/s in
WSL vs 160 tok/s in native Ubuntu**
([reported here](https://www.reddit.com/r/LocalLLaMA/comments/1sw21op/comment/oid8d9n/)).
WSL 2.7.3 closes some of that gap (115 vs 160), not all.

Pick WSL2 / Docker when:

- You need an upstream vLLM feature that isn't in either of this
  project's wheels (`0.19.0+devnen.1` for the default zip,
  `0.20.0+cu132.devnen.1` for the Blackwell zip), e.g. NVFP4 / MXFP4
  KV, ROCm, or Linux-only patches that landed after our base tags.
- You are already comfortable in Linux and are happy paying the WSL
  tax.

Otherwise, native Windows wins on tok/s.

## When dual-boot Linux is worth it

If you are running 24/7 inference on dedicated hardware, the
single-digit-percent speedup is real and the ~70 tok/s gap (160 vs 90
on a 5090, or roughly 90 vs 65 on a 3090) compounds over thousands of
prompts. Honestly the sweet spot is a separate Ubuntu box with two
3090s running 27B locally; that is the most cost-efficient setup I
have found.

This project exists for the people who already have Windows + a
working NVIDIA GPU and do not want to dual-boot just to run a model.

## When this project is the right answer

- You are on Windows 10 or 11 with an Ampere / Ada / Blackwell card
  (3090 / 4090 / A6000 / 5070 / 5080 / 5090). Pick the default zip for
  30/40-series, or the `-blackwell` zip for 50-series; see
  [`HARDWARE.md`](HARDWARE.md) and [`INSTALL.md`](INSTALL.md).
- You want OpenAI-compatible AND Anthropic-compatible endpoints out of
  the box for Claude Code, Cline, Cursor, Codex CLI, OpenCode,
  KiloCode, etc.
- You want tool calling that just works, including the streaming-path
  fixes that catch dropped parameters and split tags under spec-decode.
- You do not want to install Python, conda, pip, MSVC (mostly), or any
  global runtime. The portable zip drops in and runs.
- You like reading exactly what flags every snapshot uses and being
  able to verify a config end-to-end with `check_coherence.py` before
  trusting any tok/s number.

If two or more of those line up, this is probably the right tool.