# Hardware reality

Honest answers about what works on what.

## Compatibility table

Skim this first, prose follows.

| GPU class | Arch / sm | Which zip | Status | Notes |
|---|---|---|---|---|
| RTX 3090 (24 GB) | Ampere / sm_86 | default zip | ✅ tested, reference rig | 64.5 tok/s decode (start_speed). Headline numbers measured here. |
| RTX 3080, A40, A6000, A5000, A100 | Ampere / sm_86 / sm_80 | default zip | 🟡 should work, untested | Same code path as 3090. Please post numbers. |
| RTX 4090, 4080, 4070 Ti Super | Ada / sm_89 | default zip | 🟡 should work, untested | Same code path as 3090; expect higher numbers. |
| RTX 4060 Ti 16 GB, 4070 12 GB | Ada / sm_89 | default zip | 🟡 tight on VRAM | 27B INT4 weights are 16.96 GiB; needs boot-quiet + small ctx, or step down to Qwen3-14B. |
| RTX 5090 | Blackwell / sm_120 | **`-blackwell` zip** | ✅ tested | **`rtx5090_nvfp4` (NVFP4) is the default since v1.3.0**: ~5,300 tok/s prefill at 47 k prompt, ~92 tok/s decode at 200 k ctx (escapes the 170 W AutoRound prefill ceiling on consumer Blackwell). `rtx5090_nvfp4_vision` adds image and video input (180 k ctx, experimental). NVFP4 is the only 5090 path since v1.3.7; the AutoRound INT4 5090 snapshots were removed since they cannot escape the 170W ceiling. See [`BLACKWELL.md`](BLACKWELL.md) and [`SM120_GDN_CEILING.md`](SM120_GDN_CEILING.md). |
| RTX PRO 5000 (48 GB) | Blackwell / sm_120 | **`-blackwell` zip** | 🟡 community-validated on Linux, untested on Windows | Independent reproduction by @chorious on Linux + upstream vLLM 0.20.2 + FlashInfer 0.6.8.post1 hit 4/4 needles up to 252,510 prompt tokens with NVFP4 + fp8 KV, MTP n=3 87.8% accept (97.8 tok/s decode), MTP n=6 78.2% accept (120.9 tok/s decode). Full report and raw data: [`pro5000-linux-nvfp4/`](pro5000-linux-nvfp4/). The Windows launcher snapshots are unchanged; a 48 GB Blackwell user can raise `rtx5090_nvfp4`'s ctx via the TUI's Edit screen. |
| RTX 5070, 5080, 5060 | Blackwell / sm_120 | **`-blackwell` zip** | 🟡 should work, untested | Same wheel, same snapshots. 5060 (8 GB) won't fit 27B; use a smaller model. |
| GTX 1080 Ti, 1080, GT 1030 | Pascal / sm_61 | none | ❌ won't work | No BF16 in hardware; Marlin INT4 needs sm_80+. Use llama.cpp. |
| RTX 2080 Ti, 2070 Super | Turing / sm_75 | none | ❌ won't work | Marlin INT4 needs sm_80+. Use llama.cpp. |
| Intel Arc, Battlemage | Xe | none | ❌ won't work | vLLM has no working Windows path for Intel. |
| AMD Radeon (RX 6000/7000/9000) | RDNA | none | ❌ won't work | ROCm vLLM doesn't ship in this Windows wheel. Use llama.cpp Vulkan/ROCm. |
| Apple Silicon | M-series | none | ❌ wrong universe | Use mlx-lm. |

Driver requirements: 596+ for the Blackwell zip (CUDA 13). Any modern
driver (550+) for the default zip. The default zip auto-installs cu126
torch; the Blackwell zip auto-installs cu130 torch and a CUDA 13
runtime shim.

## Tested

Two reference rigs.

**Ampere (original launch rig):**

- Windows 10 Enterprise 22H2, 19044.x
- 2× NVIDIA RTX 3090, 24 GB each, sm_86, no NVLink, PCIe Gen 4 ×16
- Power cap up to 350 W per card (250 W also benchmarked, see TUNING.md)
- 256 GB DDR4
- Headline: 64.5 tok/s on `start_speed`, 90 k ctx, single-card decode

**Blackwell (current dev rig as of 2026-05-06):**

- Windows 10 Enterprise 22H2
- 1× NVIDIA RTX 5090, 32 GB, sm_120, driver 596.36
- Power cap 575 W (500 W also benchmarked, see TUNING.md / BLACKWELL.md)
- Headline: ~92 tok/s decode at 200k ctx, ~5,300 tok/s prefill @ 47k prompt on `rtx5090_nvfp4` (NVFP4, MTP n=6, 575 W)
- The 0.20.0 wheel ships from this box; the 0.19.0 / Ampere zip is
  still the recommended path for non-Blackwell users.

Models live on a separate NVMe; no measurable load-time difference vs
the system disk on either rig.

## Should work, untested

- RTX 4090 / 4080 (Ada, sm_89), same code path; expect higher numbers
- Single 3090 / 4090, but see the display-attached caveat below
- A6000 / A40 / data-centre Ampere, in theory; nobody has tested

## RTX 50-series (Blackwell, sm_120), supported via the Blackwell zip

We ship two release zips. The Ampere/Ada zip
`qwen3.6-windows-server-portable-x64-ampere.zip` bundles
`vllm-0.19.0+devnen.3` against CUDA 12.6 / PyTorch cu126, kernels go up
to sm_90, so on RTX 5060 / 5070 / 5080 / 5090 it would fail at boot with
`cudaErrorNoKernelImageForDevice`. **Use
`qwen3.6-windows-server-portable-x64-blackwell.zip` for any 50-series
GPU.** That variant bundles `vllm-0.20.0+cu132.devnen.2` against CUDA
13.2 / PyTorch cu130, and the launcher auto-detects which torch index
to install from based on the bundled wheel's filename
(`+cu13*` → cu130).

Verified end-to-end on a single RTX 5090 (driver 596.36, sm_120) on
2026-05-05:

- Peutlefaire NVFP4 27B (~20 GB) loads in ~25 s and serves on
  `/v1/chat/completions`, `/v1/messages`, `/v1/responses` via
  `rtx5090_nvfp4`. The vision twin `rtx5090_nvfp4_vision` reuses the
  same weights with the unquantized visual tower loaded.
- Decode at **~92 tok/s** at 200k context (MTP n=6, mem_util 0.95,
  575W), prefill **~5,300 tok/s @ 47k prompt** and **~7,460 tok/s @
  24k prompt** via FlashInfer's sm_120 native FP4 tensor cores. This
  bypasses the 170W prefill ceiling that AutoRound INT4 hits on
  consumer Blackwell, which is why the AutoRound INT4 5090 snapshots
  were removed in v1.3.7.
- **Marlin sm_120 + AutoRound INT4 works** on Ampere/Ada and is the
  default path there. The `scalar_types.int4` bug previously reported
  on older vLLM versions is resolved in 0.20.0. Marlin selects
  `MarlinLinearKernel` for `GPTQMarlinLinearMethod` on first load.
- CUDA 13 toolkit is **not** required on the user's machine, the
  launcher copies `cudart64_13.dll`, `cublas64_13.dll`, etc. from
  torch's `site-packages/torch/lib/` into a writable
  `cuda13_shim/bin/` and points `CUDA_PATH` at it so flashinfer's
  import-time `CDLL` succeeds. Driver 596+ remains the only host
  requirement.

The Blackwell zip ships two single-card 5090 snapshots since v1.3.7.
**`rtx5090_nvfp4`** (NVFP4, text, default since v1.3.0) and
**`rtx5090_nvfp4_vision`** (NVFP4, image and video input,
experimental).

| Snapshot                | Quant | ctx  | MTP n | mem_util | Short decode | 24k decode | 24k prefill |
|-------------------------|-------|------|-------|----------|--------------|------------|-------------|
| `rtx5090_nvfp4`         | NVFP4 (`Peutlefaire/Qwen3.6-27B-NVFP4`, `--quantization=compressed-tensors`) | 200k | 6 | 0.95 | see [`BLACKWELL.md`](BLACKWELL.md) | ~92 tok/s @ 200k | ~5,300 tok/s @ 47k |
| `rtx5090_nvfp4_vision`  | NVFP4 (same weights, visual tower loaded) | 180k | 6 | 0.95 | not yet benched | not yet benched | not yet benched |

(575W power cap. Historical AutoRound INT4 5090 snapshots
were removed in v1.3.7. Their 575W numbers, kept here for reference:
`rtx5090` (240k, MTP n=6) hit 158.1 tok/s short decode, 107.8 tok/s
24k decode, 3,100-3,300 tok/s 24k prefill capped at 170W on long
unique-word prompts; `rtx5090_max` (280k, MTP n=3) hit 154.3 / 90.2
under the same cap. NVFP4 beats both on prefill by 5-7x at full TDP,
which is why the AutoRound 5090 path was dropped. See
[`BLACKWELL.md`](BLACKWELL.md).)

The NVFP4 snapshots beat every 3090 snapshot on context size at the
same MTP n.
vLLM 0.20.0 hardcodes `data_parallel_rpc_port=29550` which leaks
across orphaned engine cores; snapshots in this project pass a
randomised `--data-parallel-rpc-port` to dodge the leak.

NCCL TP/PP on Windows is experimental in 0.20.0, the multi-card
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

**3090 (Ampere):** 250 W → 350 W:
- Prefill: 845 → 983 tok/s (+16 %)
- Decode: unchanged (decode is memory-bandwidth-bound at batch=1)

Default Ampere/Ada snapshots assume 350 W. Set with `nvidia-smi -pl 350`.
**Don't exceed your PSU's headroom**, two 3090s at 350 W draw ~750 W from
the 12V rails alone before CPU and the rest. The launch rig used a 1300 W
Gold PSU.

**5090 (Blackwell):** 500 W → 575 W (numbers below are from the
historical AutoRound `rtx5090` snapshot at 240k / MTP n=6, kept as a
power-scaling reference; NVFP4 prefill at full TDP is ~5,300 tok/s @
47k, which AutoRound could not match at any cap on consumer
Blackwell):
- Short decode: 124.9 → 158.1 tok/s (+27 %)
- 24k-token decode: 89.3 → 107.8 tok/s (+21 %)
- 24k-token prefill: ~2,800 → 3,100-3,300 tok/s (+10-18 %)

Unlike the 3090, **decode itself moves with power on Blackwell**. The
5090 has compute headroom even at batch=1 / max-num-seqs=1, so the
bandwidth-bound assumption no longer holds. If your PSU and cooling
allow it, the 575 W cap pays out. The shipped `rtx5090_nvfp4*`
snapshots assume 575 W. Drop to 500 W if your PSU is tight.

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
