# Speculative-decoding × parallelism matrix

What works on the SystemPanic 0.19.0 wheel + the devnen patches
(`qwen3.6-windows-server-portable-x64-ampere.zip`, current devnen tag
`+devnen.3`).

The Blackwell zip ships vLLM 0.20.0+cu132.devnen.2. The matrix below is
the 0.19 reference; 0.20 results are empirical and only partially
validated. Confirmed on a single RTX 5090 (sm_120) on 2026-05-05:
**TP=1 + MTP works** on both AutoRound INT4 (Marlin sm_120) and NVFP4
(FlashInfer sm_120 native FP4 tensor cores).

Since v1.3.0 the Blackwell default snapshot is `rtx5090_nvfp4`
(`Peutlefaire/Qwen3.6-27B-NVFP4`, `--quantization=compressed-tensors`).
As of v1.3.7 this and `rtx5090_nvfp4_vision` are the only 5090 paths;
the AutoRound INT4 5090 snapshots were removed because they cannot
escape the 170W prefill ceiling on consumer Blackwell. NVFP4 has its
own bundled MTP head (separate from the official Qwen MTP head Lorbus
AutoRound preserves) and the same TP=1 + MTP row applies. See
[`BLACKWELL.md`](BLACKWELL.md) and
[`SM120_GDN_CEILING.md`](SM120_GDN_CEILING.md).

**PP=2 status (2026-05-07).** The Ampere `pp2_160k` snapshot was broken
on every release prior to v1.3.3, the `+devnen.1` wheel didn't carry
the Windows ZMQ ipc -> tcp swap, so PP=2 crashed at engine init with
`ZMQError: Protocol not supported`. Fixed in `+devnen.2` (Ampere), and
the matching `+cu132.devnen.2` Blackwell wheel carries the same fix.
Verified clean boot, coherence, and ~10 % of documented 40.3 tok/s on
2× RTX 3090 (reference box). PP=2 + MTP / PP=2 + ngram
remain blocked by the upstream `Qwen3NextMTP.SupportsPP` /
`'GPUModelRunner' object has no attribute 'drafter'` bugs and have not
been re-tested on 0.20.

| Combo | Result |
|---|---|
| TP=1 + MTP (n=3..6) | **Works.** 53–72 tok/s on Qwen3.6-27B INT4 depending on N and prompt class. The headline `start_speed` config. |
| TP=1 + draft-model | Works *if* vocab matches target. Qwen3.6-27B vocab=248320; no small (≤2 B) Qwen3.5/3.6 drafter exists with that vocab. Qwen3-0.6B has vocab=151936 and fails at boot with pydantic `ValidationError`. Opt-in shell `start_draft.py` is reserved for when a vocab-matched drafter ships. |
| PP=2 + MTP | `NotImplementedError: Pipeline parallelism is not supported for this model` on Qwen3-Next at engine init. Documented in vLLM upstream, no workaround on 0.19.0. |
| PP=2 + ngram | `RuntimeError: 'GPUModelRunner' object has no attribute 'drafter'` at worker rank during `determine_available_memory`. vLLM 0.19.0 bug. |
| PP=2 + draft-model | Unsupported since vLLM 0.15 (hard block in upstream). |
| **PP=2, no spec-decode** | **Works.** 43.5 tok/s, ctx up to 160 k. The `start_pp2_160k` config, use only when 127 k of single-GPU context isn't enough. |
| TP=2 + MTP | Works after the CPU-relay patch but ~7.5 tok/s, the CPU-relay allreduce dominates per-layer cost. **Don't.** |
| TP=2 + ngram / draft | Same, TP=2 itself is the wrong config on Windows. |

**Bottom line:** pick **either** speed (MTP on a single GPU) **or**
context (PP=2 across both GPUs with no spec-decode). You cannot have both
on this wheel.

If MTP+PP support lands in a future 0.19.x release that SystemPanic ships
a wheel for, this matrix changes. We'll re-bench when that happens.
