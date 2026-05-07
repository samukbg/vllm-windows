# qwen3.6-windows-server v1.3.2

Hotfix release. Hardens the RTX 5090 NVFP4 boot path against a class
of cache poisoning that caused a 7x prefill regression (~750 vs ~5300
tok/s on 47k NVFP4 prompts) when system CUDA installs or conda
cudatoolkit leaked into the launch env.

## What changed

- **`clean_cuda_env()`** in `snapshots/_common.py` — the NVFP4
  snapshot now builds its subprocess env from scratch, not from
  `os.environ.copy()`. Strips every `CUDA_*` / `NVCC_*` /
  `NVTOOLSEXT*` / `CUDNN_*` key inherited from the host, filters
  NVIDIA-toolkit dirs and conda `Library/bin` out of PATH, and pins
  PATH + `CUDA_PATH` / `CUDA_HOME` at the bundled cu13 shim.
  Protects against four user environment classes uniformly — pure
  inference users, devs with system CUDA 12.x, devs with system CUDA
  13.x, and Conda/Mamba users with cudatoolkit on PATH.
- **`preflight_sm120a_or_die()`** — a 5-second subprocess probe runs
  before the ~11-minute warmup. Hard-exits with a diagnostic message
  if FlashInfer can't dispatch to `sm_120a` under the cleaned env,
  instead of silently running degraded.
- **`windows_tools/wipe_caches.py`** — single-command recovery
  utility for the four caches that, when poisoned, cause the
  fingerprint above (`~/.cache/vllm/`, the torchinductor temp dir,
  `~/.cache/torch/`, `~/.cache/flashinfer/`). Defaults to
  move-to-`.bak.<timestamp>` for forensic safety.
- **`cache_env_stamp_check()`** — every boot writes / verifies an env
  fingerprint at `~/.cache/vllm/.env_stamp.json` and prints a loud
  `[preflight WARN]` block on mismatch (e.g. wheel upgrade that
  changes the dispatch surface).

## Who is affected

Only the `rtx5090_nvfp4` snapshot. Ampere/Ada paths keep the legacy
`cuda_env()` add-only semantics — they don't run on the cu13 wheel
and aren't exposed to this class of bug.

## Upgrading

Use `update.bat` from inside an existing v1.3.x install. No model
re-download is needed; NVFP4 weights at `g:\_models\Qwen3.6-27B-NVFP4`
remain valid.

## If you still see slow prefill after upgrading

If after upgrade you observe prefill below ~3,000 tok/s on a 30k+
NVFP4 prompt, the cache from before the upgrade may already be
poisoned. Recover with:

```
python windows_tools/wipe_caches.py
python snapshots/stop_vllm.py
python snapshots/start_5090_nvfp4.py    # ~11 min cold rebuild
```

After the rebuild expect ~5,300 tok/s prefill at 580W on a 47k unique
prompt with `max_tokens=1`.
