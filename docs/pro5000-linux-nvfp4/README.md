# RTX PRO 5000 Blackwell Linux NVFP4 256K validation

This directory contains an independent Linux reproduction run for
Qwen3.6-27B-Text-NVFP4-MTP on an NVIDIA RTX PRO 5000 Blackwell 48 GB
card. It is not a shipped Windows snapshot; it is a community validation
of the same vLLM / FlashInfer / NVFP4 direction on Linux.

## Environment

- GPU: NVIDIA RTX PRO 5000 Blackwell 48 GB, sm_120
- Driver: 580.126.09
- vLLM: 0.20.2
- torch: 2.11.0+cu130
- flashinfer-python: 0.6.8.post1
- Model: Qwen3.6-27B-Text-NVFP4-MTP, ModelOpt NVFP4
- KV cache: fp8_e4m3
- Long-context mode: chunked prefill, prefix caching, text-only

Critical launch details:

```bash
FLASHINFER_CUDA_ARCH_LIST=12.0
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OMP_NUM_THREADS=1

--max-num-seqs 1
--max-num-batched-tokens 4128
--block-size 32
--enable-chunked-prefill
--enable-prefix-caching
--kv-cache-dtype fp8_e4m3
--language-model-only
```

The `--language-model-only` flag was required because this checkpoint
resolved through a Qwen3.5 conditional-generation path that otherwise
tried to initialize a missing image processor.

## Summary

The full report is
[`REPORT_PRO5000_NVFP4_256K_2026-05-13.md`](REPORT_PRO5000_NVFP4_256K_2026-05-13.md).

Hard gates passed:

| Gate | Result |
|---|---|
| 47K health check | 46,855 prompt tokens, 4/4 needles, roughly 5,800 tok/s prefill |
| 177K reproduction | 177,738 prompt tokens, 4/4 needles |
| 200K target | 197,391 prompt tokens, 4/4 needles |
| 224K stretch | 221,014 prompt tokens, 4/4 needles |
| 256K stretch | 252,510 prompt tokens, 4/4 needles after raising output budget |
| Kernel path | FlashInferCutlassNvFp4LinearKernel, Triton/FLA GDN prefill, FlashInfer attention |
| MTP n=3 | 87.8% acceptance, 97.8 tok/s engine decode |
| MTP n=6 | 78.2% acceptance, 120.9 tok/s engine decode |

Startup reported a 262,144-token configuration with no OOM at
`gpu_memory_utilization=0.92`. The 262K no-MTP launch reported 714,116
GPU KV-cache tokens and 2.72x theoretical concurrency for 262,144-token
requests. This is vLLM capacity accounting with `--max-num-seqs 1`; it
is not a measured multi-request concurrency benchmark.

## Raw data

- [`needle_ladder_200k.json`](needle_ladder_200k.json)
- [`needle_ladder_256k.json`](needle_ladder_256k.json)
- [`needle_256k_retry.json`](needle_256k_retry.json)
- [`needle_ladder_mtp_n3.json`](needle_ladder_mtp_n3.json)
- [`needle_ladder_mtp_n6.json`](needle_ladder_mtp_n6.json)
- [`mtp_n3_decode_test.json`](mtp_n3_decode_test.json)
- [`metrics_mtp_n3.txt`](metrics_mtp_n3.txt)
- [`metrics_mtp_n6.txt`](metrics_mtp_n6.txt)

## Notes

The prefill figures in the report are client-side estimates from
end-to-end request latency, not isolated TTFT/prefill metrics. The MTP
decode rates are taken from vLLM `/metrics` request decode timing.

The first 256K no-MTP run used `max_tokens=200` and only surfaced 1/4
markers because the thinking template consumed the short output budget.
A retry with `max_tokens=1600` surfaced all 4/4 markers. The MTP n=3 and
n=6 256K runs also hit 4/4.
