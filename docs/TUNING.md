# Tuning, the levers that actually move the numbers

The lever set on this wheel is narrower than upstream-Linux because TurboQuant
KV, FlashInfer, and a few Genesis patches are unavailable. What remains:

## Decode TPS

1. **MTP num_speculative_tokens.** The widely-repeated "n=3 is the sweet spot"
   conventional wisdom is for *short prompts*. On long-prompt dense code
   (~100 KB / ~24 k tokens of Python source, our bench harness) the
   acceptance curve shifts later:
   ```
   n=3 / 4 / 5 / 6 / 7 / 8  →  53.4 / 58.3 / 62.8 / 64.5 / 61.5 / 58.0 tok/s
   ```
   **n=6 is the long-prompt peak**, dropping a cliff at n=7. Always re-sweep
   on a representative prompt for a new workload, don't trust a single
   fixed number.

   On RTX 5090 (Blackwell zip, vLLM 0.20.0+cu132.devnen.1) the picture
   shifts:

   ```
   ctx 120k MTP n=6 (rtx5090_speed) -> short 130.9 / 24k decode ~89 tok/s
   ctx 240k MTP n=6 (rtx5090)       -> short 124.9 / 24k decode  89.3 tok/s
   ctx 280k MTP n=3 (rtx5090_max)   -> short 138.0 / 24k decode  81.0 tok/s
   ```

   **Two surprises** on Blackwell that don't show on Ampere:

   - On **short prompts**, MTP n=3 actually edges out n=6 (138.0 vs
     130.9) because the 5090's higher memory bandwidth makes wasted
     draft tokens cheaper than the cudagraph-replay overhead n=6 adds.
     n=6 still wins on long prompts (89.3 vs 81.0), so n=6 remains
     the default for the balanced and speed configs.
   - **Bigger ctx does NOT slow decode at fixed MTP** the way it does
     on Ampere — going 120k → 240k at MTP n=6 costs 5–6 short-prompt
     tok/s but is *equal* on long-prompt decode. The Blackwell KV-pool
     attention bandwidth has more headroom than Ampere's. We have not
     yet swept n=4 / n=5 / n=7 on the 5090; conventional wisdom from
     the 3090 sweep suggests n=6 is still the long-prompt peak.

2. **Power cap.** 250 W → 350 W: prefill +16 %, decode unchanged (decode
   is memory-bandwidth-bound at batch=1 / max-num-seqs=1). If you want
   decode speed, more power doesn't help; if you want short-prompt TTFT,
   it does.

3. **Bigger ctx slows decode at fixed MTP.** Measured at MTP n=6:
   ctx=90 k → 64.5, ctx=112 k → 60.4 (–6 %). Likely KV-pool memory
   bandwidth during attention. If a snapshot doesn't need the full ctx
   headroom, don't pay for it, split into a "speed" snapshot (smaller
   ctx) and a "max-ctx" snapshot. We have both.

4. **Cudagraphs on.** `--enforce-eager` costs ~55 % on Linux; untested on
   Windows but almost certainly similar. Don't disable.

5. **Attention backend.** `TRITON_ATTN` only, FlashInfer JIT is broken
   on Windows because ninja trips MAX_PATH inside flashinfer cache dirs.
   Pass `--attention-backend=TRITON_ATTN` as a CLI flag (the env var alone
   is ignored on 0.19.0).

6. **Small-boost env vars:**
   ```
   VLLM_ENABLE_CUDAGRAPH_GC=1     # no downside
   VLLM_MARLIN_USE_ATOMIC_ADD=1   # no downside
   VLLM_USE_FLASHINFER_SAMPLER=1  # ONLY if MSVC 2022 + ninja are present
   ```
   `VLLM_USE_FLASHINFER_SAMPLER=1` JIT-compiles a sampling module on the
   first `profile_run`, which shells out to ninja and `cl.exe`. Without
   MSVC, EngineCore dies with `FileNotFoundError` in `run_ninja` before
   the server boots. The shipped snapshots auto-detect MSVC + ninja and
   set this to `0` (PyTorch fallback) when either is missing. ninja
   ships in the launcher zip since v0.1.17, so installing the free
   Visual Studio 2022 Build Tools ("Desktop development with C++"
   workload) is enough to flip the snapshot probe to `=1` automatically.
   See the README "Optional: install MSVC 2022" section.

## Prefill TPS

1. **`--max-num-batched-tokens` peak is around 4128, non-monotonic.** Lowering
   to 2048 *shrinks* the available KV from 4.74 GiB → 2.99 GiB; raising to
   8192 shrinks it to 2.64 GiB. Leave at 4128. (The chunked-prefill scratch
   buffer scales weirdly with this number on this wheel.)

2. `--enable-chunked-prefill` and `--enable-prefix-caching` are on in every
   shipped snapshot.

## Context

1. **Just raise `--max-model-len`.** On a fresh `start_72tps` snapshot the
   KV pool already holds ~75 k tokens of physical KV but `--max-model-len=32000`
   caps it at 32 k. Bumping to 121 k costs nothing (measured: prefill 845 vs
   835 tok/s, decode within noise on the ~25 k-token bench prompt). The 32 k
   conservatism is a copy-from-Linux artifact, not a hardware limit.

   **Easiest path: clone an existing snapshot and bump the context.** Press
   `e` on the dashboard, Duplicate the closest match (e.g. `gpu0_50k` or
   `start_speed`), edit the Context field, Save. The launcher rewrites
   both `configs.yaml` and `start_<id>.py` for you. If you push past what
   fits, vLLM raises a clear `ValueError` at boot before any model weights
   load and prints the safe ceiling (see the OOM oracle below). Nothing
   gets corrupted, just relaunch with a smaller value. See
   [`docs/SNAPSHOTS.md`](SNAPSHOTS.md) for the full editor reference.

2. **GPU1 mem-util ceiling is 0.948** (display-free). When GPU0 is idle,
   vLLM sees free=22.76/24 GiB on GPU1 *after* CUDA context init reserves
   ~1.5 GiB. 0.948 passes; 0.95 trips by ~40 MiB. The 0.92 → 0.94 → 0.948
   progression: each step adds ~0.2 GiB KV / ~6.4 k tokens of ctx ceiling.
   GPU0 with display: 0.948 is also reachable if you boot-quiet (close
   heavy GPU apps before launching, reopen after `Application startup
   complete`). If you can't boot-quiet, stay at 0.92 max, that's the
   `start_gpu0_50k` path.

3. **Use vLLM's auto-profiler as a context oracle.** Two ways:

   a. **OOM probe.** Set `--max-model-len=200000` (deliberately too high)
      and read the engine's
      ```
      ValueError: To serve at least one request with the models's max seq len
      (200000), (X GiB KV cache is needed, which is larger than the available
      KV cache memory (Y GiB). Based on the available memory, the estimated
      maximum model length is N. ...
      ```
      That's the exact ceiling for the current config. Push max-model-len to
      ~99 % of N. Wrapped as a tool: `python windows_tools\probe_max_ctx.py`.
      Source: `vllm/v1/core/kv_cache_utils.py:641`.

   b. **Headroom from a successful boot.** When a snapshot boots cleanly,
      vLLM logs (`vllm/v1/core/kv_cache_utils.py:1319 / :1325`):
      ```
      INFO ... GPU KV cache size: N tokens
      INFO ... Maximum concurrency for X tokens per request: Y.YYx
      ```
      `Y` is the multiplier of unused KV. If a 90 k snapshot reports
      concurrency 1.5x, you have ~50 % headroom and could push ctx to
      ~135 k (`X * Y`) without re-probing. Trust the `Maximum concurrency`
      line, not the `GPU KV cache size` (the latter is a derived ceiling on
      this wheel, not the physical pool).

4. **PP=2** when you need >121 k. ~2× the KV pool but kills MTP, capping
   decode at ~43 tok/s. Use [`start_pp2_160k`](../snapshots/start_pp2_160k.bat)
   for this case only.

5. `--kv-cache-dtype=fp8_e4m3` always. fp8_e5m2 is rejected by TRITON_ATTN.

6. `--max-num-seqs=1` always, single-user only.

## Reading the KV pool from logs

Quick reference for the two oracle log lines (full discussion in Context
item 3 above). After a successful boot, in `logs\vllm_server.<port>.log`:

```
INFO ... [kv_cache_utils.py:1319] GPU KV cache size: N tokens
INFO ... [kv_cache_utils.py:1325] Maximum concurrency for X tokens per request: Y.YYx
```

`safe_max_ctx ≈ X × Y`. The `GPU KV cache size: N tokens` line is *not*
the physical pool on this wheel, it's a derived ceiling. Trust the
`Maximum concurrency` line.

When ctx is too high, vLLM raises the `estimated maximum model length is N`
ValueError before any weights load. Read N, set `--max-model-len` just
under, re-launch. The CRUD editor (press `e` on the dashboard) is the
fastest way to apply the new value.

## Sweeping your own configs

`windows_tools\bench_summarize.py` runs a ~100 KB / ~24 k-token Python
source-summary prompt against any port and appends one TSV row per run
to `runs.tsv`.
Schema:

```
label  kv_pool_GiB  ctx  mtp_n  prefill_tok_s  decode_tok_s  ttft_s  wall_tok_s  prompt_tokens  completion_tokens  notes
```

Convention: cold rows labeled `<config>`, warm rerun rows `<config>_run2`,
`<config>_run3`. Cold runs hit fresh prefix cache (TTFT ~25 s on the
~25 k-token prompt); warm runs report TTFT ~4 s / prefill ~5 900 tok/s
. That warm
prefill is *bogus* for cross-config comparison. Decode TPS *is* meaningful
on warm runs and useful for run-to-run noise.

`windows_tools\tune_restart.py --port 5001` kills any orphan EngineCore /
APIServer / ZMQ listener tied to that port (regex-parsing the log) and
relaunches the snapshot. Use this between sweeps so port 5001 isn't
held by yesterday's process.
