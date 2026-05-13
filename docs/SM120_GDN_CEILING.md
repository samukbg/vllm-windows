# RTX 5090 (sm_120) GDN prefill power ceiling, solved by NVFP4

**Status (2026-05-06):** SOLVED for users who can switch quantization.
The historical ceiling described below applies to the AutoRound INT4
weights we shipped originally. **Switching to NVFP4 weights
(`Peutlefaire/Qwen3.6-27B-NVFP4`, snapshot `start_5090_nvfp4`) lifts
prefill from ~1100 tok/s @ 170W to ~5300 tok/s @ 580W on a 47k prompt
, 5x throughput, full TDP utilization, mem-BW% climbs from 0% to 35%.**
Decode also improves: ~92 tok/s (vs 73 baseline).

The GDN linear-attention layer ceiling is still real for its share of
the work (10/40 layers go through FLA Triton), but FFN GEMMs dominate
prefill FLOPs and routing them through FlashInfer's sm_120 native FP4
tensor-core kernels (autotuned `fp4_gemm`) escapes the bottleneck
entirely.

### Validation matrix (NVFP4 snapshot, 2026-05-06)

| Check | Tool | Result |
|---|---|---|
| 3-tier short coherence | `windows_tools/check_coherence.py` | PASS |
| Long-ctx coherence + 2-needle retrieval @ 50k / 100k / 177k tokens | `windows_tools/test_long_ctx_coherence.py` | PASS at all 3 depths |
| MTP acceptance @ 50k ctx | `windows_tools/probe_mtp_acceptance.py` | 81.9% (4.91 of 6 avg) |
| MTP acceptance @ 150k ctx | `windows_tools/probe_mtp_acceptance.py` | 73.9% (4.43 of 6 avg) |
| Tool-calling tier-1 (emit tool_call) | `windows_tools/test_tool_calling.py` | PASS |
| Tool-calling tier-2 (synthesize tool result) | `windows_tools/test_tool_calling.py` | PASS |
| Tool-calling tier-3 (developer-role alias) | `windows_tools/test_tool_calling.py` | PASS\* |
| Coding quality (12-problem slice) vs AutoRound | `windows_tools/eval_humaneval_slice.py` | 12/12 vs 12/12, tied 100%\*\* |

\* Required a 1-line patch to `templates/qwen3.5-enhanced.jinja` so the
template prelude (lines 96/104) accepts `developer` as well as `system`.
The previously-existing alias in the per-message loop only fires *after*
the system header is rendered, so a developer message at `messages[0]`
was being silently dropped. This fix benefits both AutoRound and NVFP4
(both failed tier-3 before, both pass after).

\*\* **Eval slice is small.** 12 hand-curated coding problems is enough
to detect a catastrophic regression but not a small one. Tied at 100%
means we cannot resolve a few-percent quality gap, not that none
exists. NVFP4 weights are static post-training quant (not QAT), so a
real long-tail accuracy comparison would require a much larger eval
(MMLU, full HumanEval+, GSM8K, multi-turn agentic benchmarks). We have
not run those. Treat the equivalent score as *not catastrophically
worse*, not as *proven equal*.

**Original ceiling for reference (AutoRound INT4 only):** prefill on a
single RTX 5090 hits a **170W power ceiling at SM=100% / mem-BW≈0%**.
Every available FlashInfer GDN prefill kernel is hard-locked either to
Hopper sm_90a or datacenter Blackwell sm_100a, so AutoRound INT4
(Marlin GEMM path) falls back to FLA pure-Triton. That path uses
tensor cores for its main GEMMs, but the chunk-wise recurrent state
update is algorithmically serial-with-small-working-set and doesn't
push power. NVFP4 sidesteps this by routing the FFN/QKV/proj GEMMs
through a different, sm_120-native code path.

This document is written so that another engineer (or LLM in a future
session) can pick up exactly where we left off without re-walking the
dead ends.

---

## Symptom and characterization

| Phase                | Power   | SM   | Mem-BW | tok/s |
|----------------------|---------|------|--------|-------|
| Decode (1.2k tok)    | 340W    | 96%  | 40%    | 73    |
| Prefill (30k tok)    | 170W    | 100% | ~0%    | ~660  |
| Prefill (60k tok)    | 170W    | 100% | ~0%    | ~1100 |

Decode is healthy, power and mem-BW utilization are exactly what you
want for a memory-bandwidth-bound INT4-weight inference path.

Prefill shows the diagnostic signature: SM at 100% but VRAM untouched.
That means kernels are spinning, but on a small working set, not
streaming weights or KV from HBM. Classic recurrent-state-update
profile.

To reproduce the measurement:

```bash
# Server up at port 5001 (AutoRound 5090 snapshot, removed in v1.3.7;
# reproduce on the equivalent custom snapshot if you want to re-measure
# the AutoRound ceiling)
nvidia-smi.exe dmon -s pucvmet -i 0 -d 1 -c 25 > /tmp/dmon.log 2>&1 &
# Fire a long prompt, 1-token output (prefill-heavy)
curl -sS http://127.0.0.1:5001/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.6-27b-autoround","prompt":"<~30000 tokens of text>","max_tokens":1,"temperature":0.0}'
# After it returns:
awk 'NR>2 && /^ *0/ && $5 > 50 {p+=$2; s+=$5; m+=$6; n++} END{printf "AVG: power=%.0fW SM=%.0f%% mem=%.0f%%\n",p/n,s/n,m/n}' /tmp/dmon.log
```

---

## Why three independent kernel paths are all dead

vLLM's `ChunkGatedDeltaRule` op
(`vllm/model_executor/layers/mamba/gdn_linear_attn.py:120`) gates on
`current_platform.is_device_capability(90)` and dispatches either to
**FlashInfer** (`forward_cuda`, when supported) or **FLA Triton**
(`forward_native`, fallback). The 5090 falls through to FLA. Three
underlying CUDA paths exist; none reach our GPU.

### Path 1, FlashInfer SM100 (CuTe-DSL Blackwell datacenter)

- **Source:** `flashinfer/gdn_kernels/blackwell/gdn_prefill.py`,
  exposed as `chunk_gated_delta_rule_sm100`.
- **Landed in:** flashinfer-ai/flashinfer **PR #3001** (merged
  2026-04-13, "Add Blackwell GDN prefill kernel"). All benchmarks in
  that PR were on **B200 (sm_100a)**. No 5090 numbers.
- **Gate:** `is_sm100a_supported(device)` →
  `major == 10 and CUDA >= 12.8`. RTX 5090 is `(12, 0)`. Hard-rejected.
- **Why bypassing the gate fails (we tested):** the kernel uses
  `tcgen05` MMA + TMEM-backed accumulation + `cvt.rs.f16x2.f32`
  (stochastic rounding fp32→fp16x2). FlashInfer's own
  `flashinfer/utils.py:566` warns that **SM120 does NOT support
  `cvt.rs.f16x2.f32`**. CUTLASS docs explicitly state sm_100a kernels
  are **incompatible with RTX 50-series**. Forcing the path on sm_120
  would either fail at JIT (CuTe DSL `admissible_archs = ['sm_100a',
  'sm_100f']`) or, if forced through, produce illegal-instruction
  faults at runtime.
- **Status of the underlying ask:** flashinfer-ai/flashinfer **#2340**
  ("`chunk_gated_delta_rule` for Blackwell") is OPEN. Maintainers say
  PR #3001 closed it, but #3001 is sm_100 only. The sm_120 case is
  **not on a named roadmap milestone** as of 2026-05.
- **Related:** flashinfer-ai/flashinfer **#3170** (DGX Spark / SM121
  audit) is the canonical reference for the sm_100 / sm_120 / sm_121
  distinction. Confirms `compute_120f` is a subset of `120a` and that
  the prebuilt wheels (`12.0a` for CUDA <12.9, `12.0f` for >=12.9)
  don't ship `121a`-targeted code at all.

### Path 2, FlashInfer SM90 (Hopper C++ JIT)

- **Source:** `flashinfer/data/csrc/prefill_kernel_delta_rule_sm90.cu`
  + headers under `flashinfer/data/include/flashinfer/flat/prefill/`.
  Built by `flashinfer/jit/gdn.py:gen_gdn_prefill_sm90_module`.
- **Build flags** (from `flashinfer/jit/gdn.py:87`):
  `sm90a_nvcc_flags + ["-DFLAT_SM90A_ENABLED", "-std=c++20"]`. The
  `sm90a_nvcc_flags` is defined in `flashinfer/jit/core.py:127` as
  `["-gencode=arch=compute_90a,code=sm_90a", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED"]`.
- **Why PTX forward-compat doesn't save us:** per the NVIDIA Hopper
  Compatibility Guide, plain `compute_90` PTX **does** JIT to sm_120
  via the driver. But `compute_90a` PTX (the `a` suffix is required
  for `wgmma.mma_async`) has **no forward or backward compatibility
  guarantee**, it's hard-failed on sm_120.
- **The kernel itself uses Hopper-only features:**
  `prefill_kernel_delta_rule_sm90.cuh:33` includes
  `flashinfer/flat/hopper/device/device_universal.hpp` and uses
  `cutlass::gemm::KernelTmaWarpSpecializedCooperative`, both Hopper
  warp-specialized async pipelines that only exist on `sm_90a`. The
  whole kernel body is gated on `#if defined(FLAT_SM90A_ENABLED)`.
- **Verdict:** even if we patched the build flags from `sm_90a` to
  plain `sm_90`, the kernel source wouldn't compile (the CUTLASS
  templates would refuse to instantiate `KernelTmaWarpSpecializedCooperative`
  without sm_90a). And on this Windows install, the JIT fails earlier
  anyway: `ninja` is not on PATH and `nvidia-cutlass-dsl` is not
  installed (see install state notes below), so even Hopper users
  using our packaged FlashInfer 0.6.8 would never get this kernel
  built.

### Path 3, FLA Triton (`forward_native`, the actual hot path)

This is what runs on the 5090. Every FLA op file under
`vllm/model_executor/layers/fla/ops/` is a vendored copy from
`fla-org/flash-linear-attention`. The forward chunked path used by
`ChunkGatedDeltaRule.forward_native` is a chain of:

1. `cumsum.chunk_local_cumsum`, small pre-scan
2. `chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd`, `K @ K^T` per chunk
3. `solve_tril.solve_tril`, small triangular solve (18 `tl.dot`s,
   gated by `FLA_TRIL_PRECISION`, defaults to `"ieee"` = scalar fp32)
4. `wy_fast.recompute_w_u_fwd`, produces W and U tensors for the chunk
5. `chunk_delta_h.chunk_gated_delta_rule_fwd_h`, **the recurrent
   state update**. Iterates chunks serially, h_t = h_{t-1} + outer(k,v)
   with delta-rule gating. 8 `tl.dot` calls on bf16. This is the
   dominant kernel.
6. `chunk_o.chunk_fwd_o`, output projection, 3 `tl.dot` on bf16.

The big kernels (#5, #6) use plain `tl.dot` on bf16. **Triton 3.6.0 +
CUDA 13.0 + sm_120 lowers these to native `mma.sync` tensor-core
instructions correctly** (the older `getMMAVersionSafe` assertion
crash that hit Triton 3.2.x on sm_120, triton-lang/triton #6087 ,
was fixed by 3.5). Tensor cores are active. The 170W ceiling is
*not* a "tensor cores aren't lit up" bug.

The dependency chain in step 5 is the actual bottleneck. Each chunk's
state h_t depends on h_{t-1}, so within a single sequence the
state-update can't parallelize across chunks. The working set per
chunk (16 q-heads × 48 v-heads × 128 head-dim ≈ a few MB) fits in L2,
so VRAM bandwidth never saturates. SMs spin at 100% executing
small-tile MMAs and tile-load latency, but nothing pushes the chip
into its high-power regime.

---

## What we tried, and how it failed

All experiments below were on `C:\Temp\qwen36-27b\qwen3.6-windows-server\`
(test install mirror of the source repo). Patches were rolled back
after measurement.

### Experiment 1, set `FLA_TRIL_PRECISION=tf32`

`solve_tril.py:21` defaults to `"ieee"` (scalar fp32). Setting to
`"tf32"` makes the 18 `tl.dot`s in `solve_tril` use TF32 tensor cores.

- Result: power went 170W → 219W (+30%) and mem-BW 0% → 5%, **but
  prefill tok/s did not improve** (~570 tok/s on 60k prompts vs ~1100
  tok/s baseline). solve_tril is NOT the dominant kernel, it does
  small triangular solves, the GEMM-heavy work is in `chunk_delta_h`
  and `chunk_o` which already use tensor cores by default.

### Experiment 2, remove `allow_tf32=False` in two FLA kernels

`wy_fast.py:93` and `cumsum.py:156` had explicit `allow_tf32=False`,
forcing scalar fp32. Removed both.

- Result: no measurable change. These kernels do small contractions on
  the float32 cumulative-g and beta tensors; not bandwidth-binding.

### Experiment 3, force the SM100 CuTe-DSL kernel on sm_120

Patched `flashinfer.utils.is_sm100a_supported` to also return True for
`major == 12`, then called `chunk_gated_delta_rule(...)` on a synth
input.

- Result: **`_has_blackwell_prefill = False`** at module load, the
  CuTe-DSL kernel was never importable because `nvidia-cutlass-dsl` is
  not installed. Falls through to the SM90 branch, which then fails to
  JIT because **`ninja` is not on PATH**. So on this install, both
  FlashInfer GDN paths are inoperable regardless of which gate we
  patch.
- Even installing both deps wouldn't help, because the underlying
  kernel uses `tcgen05` etc. (see Path 1 above). This experiment
  conclusively confirmed the upstream-fix-required posture.

### Experiment 4, widen `chunk_delta_h.py` autotune

The kernel has a hardcoded
`for num_warps in [2, 4] for BV in [32, 64]` autotune (12 configs),
even though `NUM_WARPS = [2, 4, 8, 16]` is defined at the top of the
file but unused. Other FLA kernels (`chunk_o.py`) extend to
`[2, 4, 8]` warps for non-Hopper but `chunk_delta_h.py` doesn't. We
tried widening to `[2, 4, 8] × [32, 64, 128]` (27 configs) on the
hypothesis that the 5090's 170 SMs and 128 cores/SM need more warps
than `chunk_delta_h`'s default search allowed.

- Result: **slower** (540 tok/s @ 228W vs baseline 1100 tok/s @ 170W).
  Triton autotune picked a high-warp config that micro-benchmarks well
  (the autotuner times each config in isolation with hot caches) but
  causes register spilling and worse cache behavior in the real
  end-to-end prefill workload. Classic per-call vs sustained
  throughput mismatch.
- This is consistent with the failure mode in **triton-lang/triton
  #9933** ("SM100: ptxas C7907 internal compiler error eliminates most
  autotuner configs for `mamba3_siso_bwd_kernel_dqkv`, causing 38.7x
  regression vs SM90"), where on Blackwell `num_warps=4`/`8` configs
  silently produce trap stubs and only `num_warps=2` survives. We may
  be hitting a milder version of the same: high-warp configs compile
  but spill heavily, mis-rank the autotune.
- Reverted. The narrow `[2, 4]` × `[32, 64]` default is correct for
  sm_120 on this kernel.

### Experiment 5, env var experiments

Tested `CUDA_FORCE_PTX_JIT=1` (forces driver to ignore embedded cubins
and JIT all PTX), and various `FLA_USE_TMA` toggles. No path that
fundamentally lacks an sm_120 cubin or doesn't compile from the C++
source was rescued by these.

---

## Related upstream issues (the canonical reading list)

If a future engineer wants to pick this up, these are the load-bearing
upstream issues. Read in this order.

**FlashInfer:**
- [#2340](https://github.com/flashinfer-ai/flashinfer/issues/2340) ,
  "`chunk_gated_delta_rule` for Blackwell". OPEN. Original ask.
  Maintainers say closed by #3001, but #3001 is sm_100 only. **The
  RTX 5090 / sm_120 case is what's actually missing.**
- [#3001](https://github.com/flashinfer-ai/flashinfer/pull/3001) ,
  MERGED. Adds the CuTe-DSL Blackwell kernel that's sm_100/sm_100a
  only. All benchmarks on B200.
- [#2555](https://github.com/flashinfer-ai/flashinfer/issues/2555) ,
  "SM120 attention kernels exist but are blocked by wiring issues". A
  whole separate set of patchwork PRs (#2598, #2689, #2885, #3016) for
  attention/MLA/MoE on sm_120. None target GDN.
- [#3170](https://github.com/flashinfer-ai/flashinfer/issues/3170) ,
  DGX Spark (SM121) audit. The canonical reference for sm_100 vs
  sm_120 vs sm_121 differences. Confirms the wheel build matrix
  doesn't ship `121a` cubins.
- [#2649](https://github.com/flashinfer-ai/flashinfer/issues/2649) ,
  Compile for `sm_120f` (family) instead of just `sm_120a`.
- [#1147](https://github.com/flashinfer-ai/flashinfer/issues/1147) ,
  Original "Does FlashInfer support SM120?" question, 10mo old.

**vLLM:**
- [#36598](https://github.com/vllm-project/vllm/issues/36598), CLOSED.
  "Triton autotuner OOM on Qwen3.5/Qwen3-Next GDN layers (non-SM90
  GPUs)". Confirms our diagnosis: *"non-SM90 GPUs (e.g. RTX 5090,
  SM120) [run] the Triton-based forward_native path"*. Fix landed was
  a profile-time KV-cache reservation tweak, NOT a new kernel.
- [#36973](https://github.com/vllm-project/vllm/issues/36973), OPEN,
  active (23 comments). `_warmup_prefill_kernels` in `qwen3_next.py`
  leaks ~3.4 GiB on the Triton path. Adjacent issue, not a fix for us.
- [#34948](https://github.com/vllm-project/vllm/issues/34948), OPEN.
  "Qwen3.5 CUDA Illegal Memory Access in GDN Kernel".
- [#39287](https://github.com/vllm-project/vllm/issues/39287), OPEN
  RFC. "Handle GDN prefill kernel JIT compilation failures".
- [#36450](https://github.com/vllm-project/vllm/issues/36450), OPEN.
  "Qwen3.5 AWQ models crash during inference on RTX 5090 (Blackwell)
  with Triton OOM in `solve_tril`".

**Triton:**
- [#9933](https://github.com/triton-lang/triton/issues/9933), OPEN.
  "SM100 (Blackwell): ptxas C7907 ICE eliminates most autotuner
  configs for `mamba3_siso_bwd_kernel_dqkv`, causing 38.7x regression
  vs SM90". Important context for why widening autotune on Blackwell
  is dangerous: failed configs become trap stubs that mis-rank.
- [#5950](https://github.com/triton-lang/triton/issues/5950), OPEN,
  1y old. "Does Triton support new features of Blackwell for RTX5090
  and 5080?" Tracking issue.
- [#6087](https://github.com/triton-lang/triton/issues/6087) ,
  Resolved by 3.5+. The `getMMAVersionSafe` assertion that hit
  sm_120 on Triton 3.2.x. Why Triton 3.5+ is required for any GDN
  on a 5090.
- [#7550](https://github.com/triton-lang/triton/issues/7550) ,
  `tl.dot_scaled` actually using fp16 mma on RTX 5090 (sm_120
  scaled-dot lowering still incomplete).
- [#8695](https://github.com/triton-lang/triton/issues/8695) ,
  "GatedDeltaNet backward error on Blackwell". Pipeliner bug
  underlying multiple FLA correctness issues.

**FLA (`fla-org/flash-linear-attention`):**
- [#790](https://github.com/fla-org/flash-linear-attention/issues/790)
 , OPEN. "Incorrect outputs for
  `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` on Blackwell for
  certain autotune configs (`num_warps=4 num_stages∈{2,3}`)". Fixed
  in commit
  [`02af88e`](https://github.com/fla-org/flash-linear-attention/commit/02af88ef8aa7f7043a899c9ca6fde168e1cf8c7e)
  by switching to `safe_dot`. **vLLM's vendored FLA at the moment of
  vLLM 0.20.0 release may not include this fix.** Worth verifying, if
  it doesn't, we may have a silent correctness drift in long
  prefills, not just the perf ceiling.
- [#638](https://github.com/fla-org/flash-linear-attention/issues/638) ,
  CLOSED `wontfix triton-bug`. Underlying cause of #790, points at
  Triton #8695.
- [#609](https://github.com/fla-org/flash-linear-attention/issues/609) ,
  CLOSED. Disable TMA on Blackwell. Already fixed.

---

## Concrete handoff notes for the next attempt

If someone wants to try to fix this: the "expensive but realistic"
options, ordered by tractability.

**Option A, Switch to SGLang on Linux/WSL (escape hatch).**

Community signal (perplexity social mode, 2026-05) says SGLang's
sm_120 GDN path is more active than vLLM's; `voipmonitor/sglang:cu130`
Docker image is referenced. SGLang has its own linear-attention impl
that's apparently been ported with sm_120 in mind. This is outside the
scope of `qwen3.6-windows-server` (we're a vLLM-on-native-Windows
project) but is a real escape hatch for users who hit the wall.
Compare numbers and document.

**Option B, Backport / write a native sm_120 GDN prefill kernel
for FlashInfer.**

This is what FlashInfer #2340 is asking for. Effort: multi-week.
Approach:
1. Start from `flashinfer/data/include/flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh`.
2. Replace `KernelTmaWarpSpecializedCooperative` (Hopper TMA + warp
   specialization) with a simpler `cute::Sm80` style scheduler that
   uses standard `mma.sync.aligned.m16n8k16` (forward-compat across
   Ampere/Ada/Blackwell consumer).
3. Remove `wgmma.mma_async`. Use synchronous `cute::TiledMma` with
   sm_120's native bf16 MMA shape.
4. Verify TMA can stay (sm_120 supports TMA per #3170 audit) or
   replace with cp.async-bulk equivalent.
5. Build with `-gencode=arch=compute_120,code=sm_120` (no `a` ,
   plain so PTX JITs forward).
6. Submit to FlashInfer.

**Option C, Replace FLA's chunk-based scan with TFLA (Tiled Flash
Linear Attention).**

The arxiv paper [2503.14376](https://arxiv.org/abs/2503.14376) ("More
Efficient Linear RNN and xLSTM Kernels", "Tiled Flash Linear
Attention") describes a re-tiling of the chunk-based scan that's more
parallel, instead of strictly serial chunk-to-chunk dependency, it
processes blocks of chunks in a flash-attention style. Would need a
Triton port targeting sm_120 specifically. Effort: 1-2 weeks for
someone who knows Triton + flash attention well.

**Option D, Wait.**

Issue tracking `flashinfer-ai/flashinfer #2340`. If/when sm_120 GDN
ships in a FlashInfer release, our existing
`vllm/model_executor/layers/mamba/gdn_linear_attn.py:128` gate will
need to be updated to `is_device_capability(90) or
is_device_capability(120)`. That's a one-line vLLM patch. Still
requires `ninja` and `nvidia-cutlass-dsl` to be in the install (see
"State of this install" below).

---

## State of this install (relevant to debugging)

These are environmental quirks of the
`devnen/qwen3.6-windows-server` portable build that affect any future
GDN-on-FlashInfer work on Windows:

- **`ninja` is NOT on PATH** in the bundled embedded Python. FlashInfer's
  JIT C++ kernel build always fails. We've been silently in
  Triton-only mode the whole time.
- **`nvidia-cutlass-dsl` is NOT installed.** FlashInfer's CuTe-DSL
  kernels (decode and prefill) are inert
  (`_has_blackwell_prefill = False`).
- **`CUDA_HOME` is not set** by default. FlashInfer needs a directory
  containing `cudart64_13.dll`, we've shimmed this elsewhere
  (`C:\Temp\cuda_shim\bin\` with the dll copied from
  `torch/lib/cudart64_13.dll`) and it works, but it's not yet baked
  into the snapshot env.
- **Triton 3.6.0 + CUDA 13.0** is what's bundled. This is recent enough
  to have the `getMMAVersionSafe` fix and correct sm_120 lowering for
  bf16 `tl.dot`. **Not** affected by the Triton 3.2.x assertion bug.
- The Qwen3.6-27B AutoRound model has K=128 (head_dim) and uses 2 of
  the 4 possible state slabs in `chunk_delta_h` (h1 and h2; h3/h4
  branches at `K > 128` and `K > 192` are dead code for our model).

---

## Summary table for future readers

| Question | Answer |
|---|---|
| Is the 170W prefill ceiling a configuration bug? | **No.** Best-known config. |
| Does Triton emit MMA on sm_120 for bf16? | **Yes** (3.5+). Already happening. |
| Is FLA `chunk_delta_h` using tensor cores? | **Yes**, but it's recurrent and the working set is small. |
| Can we coerce FlashInfer's SM90 path to sm_120? | **No.** sm_90a + wgmma + Hopper TMA. Hard fail. |
| Can we coerce FlashInfer's SM100 path to sm_120? | **No.** tcgen05 + cvt.rs.f16x2.f32 absent on sm_120 silicon. |
| Is upstream working on sm_120 GDN? | Slowly. FlashInfer #2340 open with no PR. |
| Is decode affected? | **No.** Decode is healthy at 340W / 73 tok/s. |
| Is the prefill output correct? | **Yes**, coherent (we run `check_coherence.py` after every change). |
| What's the realistic prefill throughput? | ~660-1100 tok/s depending on prompt length. |

---

## Open angles, not yet tested

The patches-at-our-layer search space above is exhausted, but two
adjacent angles remain untested as of 2026-05-06. Either could
materially move the prefill ceiling without touching the GDN kernel
itself.

### 1. NVFP4 weights (different prefill code path)

**Hypothesis:** GDN layers are only 30/40 of the model. The other 10
attention layers + every FFN GEMM run at decode-style efficiency
during prefill. NVFP4 weights would route those through FlashInfer's
sm_120-native FP4 tensor-core kernels (the path wired via
`is_sm120a_supported` in `flashinfer/utils.py:566`), which is a
completely different code path from GDN. Even if GDN stays at the
170W ceiling for its share of layers, the rest of the model could
prefill faster, potentially raising overall throughput substantially.

**Evidence it's worth testing:**
- u/Maheidem on r/LocalLLaMA (thread `1t5dya8`) reports running
  `Peutlefaire/Qwen3.6-27B-NVFP4` on a single 5090, 200k context,
  vLLM 0.20.1.dev / Torch 2.13.dev / CUDA 13.0, with MTP enabled.
- u/rpkarma in the same thread: NVFP4 needs the right consumer
  Blackwell kernels, "even more specific ones if you're on SM121."
  Implies sm_120 vs sm_121 dispatch matters; check before assuming
  it Just Works.

**Concrete test plan:**
1. `huggingface-cli download Peutlefaire/Qwen3.6-27B-NVFP4
   --local-dir D:\models\Qwen3.6-27B-NVFP4`
2. (Historical, executed in v1.3.0: cloned the AutoRound 5090 snapshot
   to `start_5090_nvfp4.py`, swapped `--quantization auto-round` for
   `--quantization compressed-tensors`, pointed `MODEL_PATH` at the
   NVFP4 dir.)
3. Run `check_coherence.py --port 5001` first, must pass.
4. Bench prefill under `nvidia-smi dmon -s pucvmet`. Compare to
   baseline 170W / ~1100 tok/s @ 60k.
5. If prefill power climbs past 170W AND tok/s improves materially:
   ship as a second snapshot, soften this doc, skip the upstream
   issue comment.

**Caveat for whoever picks this up:** the bundled Torch 2.11.0 ships
arch list `['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`
, note **`sm_120` but NOT `sm_120a`**. NVFP4 FP4 MMA instructions
may need the `a`-suffix variant. If NVFP4 fails to compile or
JIT-load, this is a likely cause. See angle #2.

### 2. Wheel `TORCH_CUDA_ARCH_LIST` audit

**Confirmed 2026-05-06:** `torch.cuda.get_arch_list()` returns
`['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']` on the
shipped wheel. **`sm_120a` is missing.** Architecture-specific
suffixed targets (`sm_90a`, `sm_100a`, `sm_120a`) enable
generation-specific instructions that plain non-suffixed targets
don't have access to:

- `sm_120a`: native FP4 MMA, `cvt.rs.f16x2.f32`, and other
  consumer-Blackwell-specific instructions
- Without `sm_120a` in the build matrix, FlashInfer kernels gated on
  `is_sm120a_supported` will fall back to slower paths or fail at
  load time

**This may be a separate finding from the GDN ceiling**, even if
NVFP4 doesn't help GDN, the missing `sm_120a` may hurt other
Blackwell-specific paths. Worth reporting upstream to
`devnen/vllm-windows` (and ultimately `SystemPanic/vllm-windows`)
as a wheel-build improvement: add `120a` to `TORCH_CUDA_ARCH_LIST`
in the wheel build script.

**Note:** Adding `120a` is a wheel rebuild, not a runtime patch. If
NVFP4 (angle #1) needs `120a` to work, this becomes a blocker for
shipping NVFP4, would need to coordinate with devnen on a wheel
rebuild that includes `12.0+PTX` or `12.0a` in the arch list.

### 3. TurboQuant KV cache (adjacent, not central)

vLLM PR #39931 (just merged) enables TurboQuant KV-cache
quantization for Qwen3.5/3.6. Different lever, KV cache, not
prefill speed. Lets users get more context at the cost of (probably
small) PPL hit. Per Reddit discussion (`1t3zu7u`):

- Flag: `--kv-cache-dtype turboquant_k8v4` (most conservative
  variant, also `_4bit_nc`, `_k3v4_nc`, `_3bit_nc`)
- Requires `--max-num-batched-tokens >= 4096` for chunked-prefill
  + mamba-align
- Quality is contested, only `_k8v4` claimed comparable to Q4_0

Not a prefill-speed fix, but a useful adjacent option. Would land
as a separate snapshot variant (`start_5090_turboquant.py`) if it
proves stable.

