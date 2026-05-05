# Hardware reality

Honest answers about what works on what.

## Tested

- Windows 10 Enterprise 22H2, 19044.x
- 2× NVIDIA RTX 3090, 24 GB each, Ampere sm_86, no NVLink, PCIe Gen 4 ×16
- Power cap up to 350 W per card (250 W also benchmarked, see TUNING.md)
- 256 GB DDR4 (model weights stream from disk, RAM hardly matters)
- Models live on a separate NVMe; no measurable load-time difference vs system disk

## Should work, untested

- RTX 4090 / 4080 (Ada, sm_89), same code path; expect higher numbers
- Single 3090 / 4090, but see the display-attached caveat below
- A6000 / A40 / data-centre Ampere, in theory; nobody has tested

## RTX 50-series (Blackwell, sm_120) — supported via the Blackwell zip

We ship two release zips. The default
`qwen3.6-windows-server-portable-x64.zip` (Ampere/Ada) bundles
`vllm-0.19.0+devnen.1` against CUDA 12.6 / PyTorch cu126 — kernels go up
to sm_90, so on RTX 5060 / 5070 / 5080 / 5090 it would fail at boot with
`cudaErrorNoKernelImageForDevice`. **Use
`qwen3.6-windows-server-portable-x64-blackwell.zip` for any 50-series
GPU.** That variant bundles `vllm-0.20.0+cu132.devnen.1` against CUDA
13.2 / PyTorch cu130, and the launcher auto-detects which torch index
to install from based on the bundled wheel's filename
(`+cu13*` → cu130).

Verified end-to-end on a single RTX 5090 (driver 596.36, sm_120) on
2026-05-05:

- Lorbus AutoRound INT4 27B loads in ~17 s and serves on
  `/v1/chat/completions`, `/v1/messages`, `/v1/responses`.
- Decode at ~36 tok/s eager mode + reasoning chain (initial
  measurement; MTP and async-scheduler tuning still TBD on Blackwell).
- **Marlin sm_120 + AutoRound INT4 works.** The
  `scalar_types.int4` bug previously reported on older vLLM versions is
  resolved in 0.20.0; no AWQ repackaging needed. Marlin selects
  `MarlinLinearKernel` for `GPTQMarlinLinearMethod` on first load.
- CUDA 13 toolkit is **not** required on the user's machine — the
  launcher copies `cudart64_13.dll`, `cublas64_13.dll`, etc. from
  torch's `site-packages/torch/lib/` into a writable
  `cuda13_shim/bin/` and points `CUDA_PATH` at it so flashinfer's
  import-time `CDLL` succeeds. Driver 596+ remains the only host
  requirement.

The default 5090 snapshot is `start_5090` (single-card, ctx 200k,
mem_util 0.93, MTP n=6). vLLM 0.20.0 hardcodes
`data_parallel_rpc_port=29550` which leaks across orphaned engine
cores; snapshots in this project pass a randomised
`--data-parallel-rpc-port` to dodge the leak.

NCCL TP/PP on Windows is experimental in 0.20.0 — the multi-card
snapshots in the Blackwell zip are still the existing
`start_pp2_160k` path. We have no multi-card 5090 box, so re-bench
multi-GPU on the Blackwell zip on a 2× 3090 host before relying on it.

WSL2 + Docker (e.g. jaMMint's
[vllm-blackwell-guide](https://github.com/lastloop-ai/vllm-blackwell-guide))
remains a valid alternative for users who want pure-upstream vLLM, but
pays the WSL tax (see below).

## Probably won't work without effort

- Pascal / Turing GPUs, sm_86 minimum. Pascal lacks BF16 in hardware
  (the Lorbus AutoRound MTP head is BF16, won't load) and INT4 Marlin
  kernels need compute capability 8.0 or higher. The wheel itself
  may build kernels for older arches but TRITON_ATTN code paths
  haven't been validated.
- WSL2, works in principle (you'd just install upstream vLLM there) but
  pays a real virtualisation tax. One community member measured the
  same hardware at **85 tok/s in WSL vs 160 tok/s in native Ubuntu**
  ([reported here](https://www.reddit.com/r/LocalLLaMA/comments/1sw21op/comment/oid8d9n/)).
  Updating WSL to 2.7.3 closes some of the gap (115 vs 160) but not
  all. WSL2 runs on Hyper-V (Type-1), CUDA goes through GPU-PV
  paravirtualisation, the Windows host driver still owns the GPU and
  DWM keeps its allocation. Use native Linux if you have the option.
- Hyper-V / DDA passthrough into a Linux VM, not tested; if you do, please
  open an issue with your numbers

## Will not work

- AMD GPUs (RX 6000/7000/9000, Instinct), vLLM ROCm path doesn't ship in
  this Windows wheel. Use upstream vLLM on Linux.
- Intel Arc / Battlemage, same.
- Apple Silicon, wrong universe; use mlx-lm.
- 16 GB cards (RTX 4060 Ti 16G, 5060 Ti 16G), Qwen3.6-27B INT4 weights
  alone are 16.96 GiB; you'd need a smaller model. Try Qwen3-14B or
  smaller variants.

## Mixed-card multi-GPU (PP=2)

PP splits transformer layers evenly across the two cards, so the
**smaller card sets the upper bound** for half the model plus
activations and KV cache. Worked example: a 4080 Super (16 GB) plus
3060 (12 GB) cannot fit Qwen3.6-27B because the 3060 side has roughly
8.5 GiB of weights to hold, plus activations, plus enough KV for any
useful context. Expect small context, no MTP (PP+MTP is broken on this
wheel), and roughly 30 to 40 tok/s decode if it boots at all.

Often it is better to run a smaller model on the larger card alone
than to split 27B unevenly. Mixed-arch combos (Ada + Ampere, Blackwell
+ Blackwell) are theoretically fine for PP but untested. Always boot
single-card first to confirm the wheel loads, then add the second.

## Three or more identical small Ampere cards (e.g. 3x RTX 3060 12 GB)

PP=3 across three 12 GB cards fits the 27B INT4 weights on paper
(roughly 5.7 GiB per card for the model, plus activations and KV), but
this combo is **untested** on this launcher and the shipped snapshots
only cover single-GPU and 2-GPU PP. Two practical caveats even if you
hand-edit a snapshot to try it:

- **No MTP.** PP + MTP is broken on this wheel
  (`SupportsPP NotImplementedError`), so you lose the speculative
  decoding multiplier. Decode caps somewhere below `start_pp2_160k`'s
  43 tok/s, possibly well below it once you add a third pipeline
  hop's worth of cross-card hand-off.
- **PCIe lane constraints matter.** Three cards on x4 / x1 risers
  push more data over slower links per token, which compounds the
  PP hand-off cost.

For a typical 3x 3060 setup, running a smaller model (Qwen3-14B INT4)
on a single card is usually a faster, less fiddly experience than
splitting 27B three ways. The other two cards are then free for other
workloads (image gen, a second model, etc.). If you do try the PP=3
path and get coherent output, please post numbers, validated configs
for >2-card setups would be welcome PRs.

## How to read the headline tok/s numbers

The 64.5 and 72 tok/s figures are **single-card decode**. The model and
KV cache live entirely on one 3090. The reason there are two cards in
the reference rig is the Windows display tax, the second card is for
the desktop. With one 3090 driving your monitor you get the same decode
numbers.

You can also run the full `start_speed` snapshot (90 k ctx) on that
single display-attached 3090 if you close heavy GPU apps during boot
and reopen them afterward, see "Boot quiet then reopen" in the next
section. The fallback is `start_gpu0_50k` for users who can't or won't
boot-quiet (lower mem_util, capped near 50 k ctx, same decode tok/s).

The only snapshot that actually uses both GPUs for inference is
`start_pp2_160k` (43.5 tok/s, 160 k context).

## The Windows desktop VRAM tax

The GPU that drives your monitor loses **1–3 GiB** to the Windows desktop
compositor (DWM) before any app is open. Common apps eat more:

| Workload | Extra VRAM |
|---|---|
| Single 1440p SDR monitor, idle desktop | ~0.6–1.0 GiB |
| Dual 4K HDR monitors | ~1.5–2.5 GiB |
| Chrome (10 tabs, HW accel on) | +0.3–0.7 GiB |
| Microsoft Teams, Discord, Outlook | +0.4–1.0 GiB combined |
| dbForge / heavy IDE | +0.5–1.5 GiB |
| 4K YouTube playing | +1.0–1.5 GiB |
| Realistic "office workload" total | ~3–5 GiB |
| Heavy: + 4K media + Snagit | ~5–7 GiB |

So a 24 GiB card with the display attached and a typical workload has
~17–20 GiB *actually free* for vLLM at idle, not 24. Qwen3.6-27B INT4
weights are 16.96 GiB, plus ~5 GiB of activations, plus you want some
KV pool, the math is tight at 24 GiB.

**Boot quiet, then reopen apps.** What matters for vLLM is the VRAM
free **at boot**, not the steady-state load. Close Chrome, Discord,
Slack, video playback, and other heavy GPU apps before launching, then
reopen them after `Application startup complete`. Once vLLM has
reserved its KV pool, the NVIDIA driver schedules everything else
around what vLLM already owns. With this pattern, the default
`start_speed` snapshot (`mem_util=0.948`, 90 k ctx) runs cleanly on a
single display-attached 3090.

**Fallback if you can't boot-quiet:** [`start_gpu0_50k`](../snapshots/start_gpu0_50k.py)
keeps `mem_util=0.92`, leaving headroom for whatever the desktop grabs
post-boot. Same decode tok/s, ~50 k ctx ceiling instead of 90 k.

For permanent VRAM relief on a single-GPU system, see
[`WINDOWS_VRAM_HEADLESS.md`](WINDOWS_VRAM_HEADLESS.md). Short version:
plug a $30 GT 1030 into your monitor, leave the 3090 compute-only.
Or on Intel desktop CPUs, route the display to the iGPU.

## GPU0 vs GPU1 (dual-GPU systems)

If you have two cards and one drives the display:

- **GPU0** (display), display tax applies. Use `mem_util ≤ 0.92`.
- **GPU1** (no display), full ~22.76 GiB free after CUDA context init.
  Default snapshots use `mem_util = 0.948` here; 0.95 trips vLLM's safety
  check by ~40 MiB.

The `start_*` snapshots (other than `pp2`) all pin GPU1. If your headless
card is GPU0 instead, edit the snapshot's `CUDA_VISIBLE_DEVICES`
assignment, or set the `CUDA_VISIBLE_DEVICES` env var before launching.

## Power cap

We measured 250 W → 350 W on these cards:
- Prefill: 845 → 983 tok/s (+16 %)
- Decode: unchanged (decode is memory-bandwidth-bound at batch=1)

Default snapshots assume 350 W. Set with `nvidia-smi -pl 350`. **Don't
exceed your PSU's headroom**, two 3090s at 350 W draw ~750 W from the
12V rails alone before CPU and the rest. We're using a 1300 W Gold PSU.

## Tensor vs pipeline parallelism

- **TP=2 on Windows: don't.** Even with the CPU-relay patch, allreduce
  fires every transformer layer and dominates the per-token cost (~7.5
  tok/s on Qwen3.6-27B). PP=2 is far better.
- **PP=2: usable for big context.** ~43 tok/s, ctx up to 160 k. The
  hidden-state hand-off is the only thing crossing CPU per layer.
- **TP=1: the throughput champion** when one card is enough. MTP works.
  64.5 tok/s on the recommended `start_speed` snapshot.

You cannot have MTP and PP at the same time on this wheel, see
[`SPEC_DECODE_MATRIX.md`](SPEC_DECODE_MATRIX.md).
