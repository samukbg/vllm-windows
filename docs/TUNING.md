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
   500W power cap (v1.2.0–v1.2.2 baseline, three snapshots shipped):
   ctx 120k MTP n=6 (rtx5090_speed) -> short 130.9 / 24k decode ~89  tok/s   [retired in v1.2.3]
   ctx 240k MTP n=6 (rtx5090)       -> short 124.9 / 24k decode  89.3 tok/s
   ctx 280k MTP n=3 (rtx5090_max)   -> short 138.0 / 24k decode  81.0 tok/s

   575W power cap (v1.2.3 re-bench, --no-enable-prefix-caching, two snapshots):
   ctx 240k MTP n=6 (rtx5090)       -> short 158.1 / 24k decode 107.8 / 24k prefill 3,100-3,300
   ctx 280k MTP n=3 (rtx5090_max)   -> short 154.3 / 24k decode  90.2 / 24k prefill 3,100-3,300

   v1.2.5 (--enable-prefix-caching ON, vLLM PR #25752 / Mamba2 APC in wheel):
   ctx 240k MTP n=6 (rtx5090)       -> short ~125 / KV pool 94,656 (+18%)
                                       prefill 12k 686 -> 2147 tok/s (3.1x)
                                       prefill 16k 559 -> 2034 tok/s (3.6x)
                                       prefill 24k+   timeout -> 2-4k tok/s
                                       decode after 2x 24k hits: stable
   ```

   **Why two snapshots, not three.** v1.2.3 retired `rtx5090_speed`.
   The 575W re-bench showed it tied `rtx5090` on short decode (158 vs
   158) but was slower on long decode (103 vs 108) AND had a
   reproducible long-prompt prefill regression (~343 tok/s vs ~3,200
   for the other two). Same MTP n=6 + chunked-prefill flags — cause
   not yet root-caused, but the practical conclusion was that it
   offered no inference advantage and a real prefill loss.

   **Three Blackwell observations from the 575W re-bench:**

   - **Power cap matters for decode** on Blackwell — 500W → 575W adds
     ~20-30% short decode and ~10-20% long decode. (On Ampere the
     equivalent 250→350W bump moved only prefill, never decode, because
     decode was bandwidth-bound. Blackwell evidently has compute
     headroom even at batch=1.) If your PSU and cooling can handle it,
     the cap pays out.
   - **MTP n=3 vs n=6 short-prompt advantage flips** with power. At
     500W n=3 edged out n=6 (138 vs 131); at 575W n=6 wins (158 vs 154).
     The "n=3 sweet spot for short prompts" claim is power-cap-dependent.
   - **Bigger ctx does NOT slow decode at fixed MTP** on Blackwell.
     The 240k snapshot lands the headline 158 tok/s short-decode number.
     The 280k snapshot is close behind (154) only because its MTP is
     n=3, not n=6.

2. **Power cap.** 250 W → 350 W: prefill +16 %, decode unchanged (decode
   is memory-bandwidth-bound at batch=1 / max-num-seqs=1). If you want
   decode speed, more power doesn't help; if you want short-prompt TTFT,
   it does.

3. **Bigger ctx slows decode at fixed MTP (3090 / Ampere only).**
   Measured at MTP n=6: ctx=90 k → 64.5, ctx=112 k → 60.4 (–6 %).
   Likely KV-pool memory bandwidth during attention. If a snapshot
   doesn't need the full ctx headroom, don't pay for it, split into a
   "speed" snapshot (smaller ctx) and a "max-ctx" snapshot. We have
   both.

   **This does NOT hold on Blackwell.** The 5090 re-bench at 575W
   showed `rtx5090` (240k, MTP n=6) hits 158.1 tok/s short decode
   while `rtx5090_max` (280k, MTP n=3) hits 154.3 tok/s. The 4 % gap
   is the MTP n drop, not the ctx ceiling. Bigger ctx is effectively
   free on Blackwell at fixed MTP n. See [`BLACKWELL.md`](BLACKWELL.md)
   for the full table.

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

2. `--enable-chunked-prefill` is on in every shipped snapshot.

3. **`--enable-prefix-caching` is ON in every shipped snapshot** (since v1.2.5,
   2026-05-06). v1.2.2 had it off as a defensive workaround for
   [vLLM issue #17140](https://github.com/vllm-project/vllm/issues/17140), the
   stepwise decode regression on Qwen3-Next where short-prompt decode dropped
   30 % after one 24 k-token request and a further 30 % after a second
   (130 → 90 → 40 tok/s, stuck) because SSM state was not tracked across
   prefix-cached KV blocks. That bug is fixed upstream by
   [vLLM PR #25752](https://github.com/vllm-project/vllm/pull/25752) (Mamba2
   Automatic Prefix Caching, merged 2025-10-04). Both shipped wheels include
   the fix in their source tree, and with prefix caching enabled vLLM
   auto-sets `mamba_cache_mode='align'` for `Qwen3_5ForConditionalGeneration`
   at `vllm/model_executor/models/config.py:367`, so SSM state aligns with
   cache-block boundaries.

   **What we measured on the 5090** (`start_5090`, ctx 240k, MTP n=6,
   `max_num_batched_tokens=4128`) re-bench at 575 W:

   | Prompt size | OFF (v1.2.2-v1.2.4) | ON (v1.2.5) | Speedup    |
   |-------------|---------------------|-------------|------------|
   | 4 k         | 3744 tok/s          | 3375 tok/s  | similar    |
   | 8 k         | 3019 tok/s          | 3497 tok/s  | +16 %      |
   | 12 k        | 686 tok/s           | 2147 tok/s  | **3.1x**   |
   | 16 k        | 559 tok/s           | 2034 tok/s  | **3.6x**   |
   | 20 k        | timeout             | 2473 tok/s  | unblocks   |
   | 24 k        | timeout             | 4333 tok/s  | unblocks   |
   | 32 k        | timeout             | 2479 tok/s  | unblocks   |
   | 48 k        | timeout             | 2444 tok/s  | unblocks   |

   Decode regression test (`windows_tools/repro_17140.py`, the exact
   protocol from the v1.2.2 bug write-up: 3 short → one 24k hit → 3
   short → one 24k hit → 3 short):

   | Phase                    | tok/s   | drift   |
   |--------------------------|---------|---------|
   | baseline (3 short)       | 119.3   | —       |
   | after 1st 24k hit        | 122.4   | +2.6 %  |
   | after 2nd 24k hit        | 122.0   | +2.3 %  |

   No stepwise drop. The old 130 → 90 → 40 pattern does not reproduce on
   the 0.20.0 wheel with prefix caching on. KV pool also gains ~18 %
   headroom (94,656 vs 79,968 tokens at the same ctx=240k). Warm-prefix
   TTFT on a 24 k re-hit drops from ~42 s cold to ~1.6 s, useful for
   agents that re-send long context.

   Why the 3090 snapshots got the same flip without a separate hardware
   test: the `vllm-0.19.0+devnen.1` source tree the Ampere/Ada wheel is
   built from contains the same Mamba2 APC code (`alignment_tokens`,
   `MambaSpec`, the `config.py:367` auto-default), verified via the
   `devnen/vllm-windows` GitHub API at tag `v0.19.0-devnen.1`. If the
   regression resurfaces on a real 3090 / 4090, please open an issue
   with the `repro_17140.py` output.

   Reproducing the bench yourself:
   ```powershell
   python windows_tools\bench_prefill_quick.py `
       --sizes 4096,8000,12000,16000,20000,24000,32000,48000 --timeout 60
   python windows_tools\repro_17140.py
   ```

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
