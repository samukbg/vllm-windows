# Speculative-decoding × parallelism matrix

What works on the SystemPanic 0.19.0 wheel + the devnen patches
(default `qwen3.6-windows-server-portable-x64.zip`).

The Blackwell zip ships vLLM 0.20.0+cu132.devnen.1. The matrix below is
the 0.19 reference; 0.20 results are empirical and only partially
validated. Confirmed on a single RTX 5090 (sm_120) on 2026-05-05:
**TP=1 + MTP works** — Marlin sm_120 + AutoRound INT4 boots, serves,
and decodes correctly. Verified single-card decode is **124.8 tok/s**
on `rtx5090_speed` (ctx 120k, MTP n=6, mem_util 0.95). MTP-on-Blackwell
sweep across n=3..n=8 is still TBD; the value chosen is the
3090-tuned long-prompt peak rather than re-swept on Blackwell.
PP=2 + MTP
and PP=2 + ngram have not been re-tested on 0.20 yet — the
`Qwen3NextMTP.SupportsPP` block and the ngram drafter bug were 0.19-era
and may or may not be fixed. NCCL is the default on Windows in 0.20
(experimental), so TP=2 numbers may also change. Re-bench on a 2× 3090
host running the Blackwell zip before relying on PP=2 there.

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
