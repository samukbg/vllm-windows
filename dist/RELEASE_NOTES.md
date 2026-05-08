# qwen3.6-windows-server v1.3.4

Bug-fix release. Closes the second wave of `pp2_160k` boot failures on the
public Ampere zip, surfaced after v1.3.3 unblocked the prior `ZMQError:
Protocol not supported` crash.

## What changed

- **New Ampere wheel: `vllm-0.19.0+devnen.3`.** Adds a one-line Windows
  guard to `vllm/distributed/utils.py` so the `sched_yield()` wrapper
  takes the `time.sleep(0)` fallback on Windows instead of calling the
  POSIX-only `os.sched_yield()`. The bug only fires on multi-worker
  paths (PP>1 / TP>1) via `shm_broadcast.acquire_read`'s spin-wait, so
  single-card snapshots are unaffected. See
  [devnen/vllm-windows v0.19.0-devnen.3](https://github.com/devnen/vllm-windows/releases/tag/v0.19.0-devnen.3)
  for the patched wheel and diff.
- **Blackwell wheel unchanged at `vllm-0.20.0+cu132.devnen.2`.** The
  upstream v0.20.0 source already inherited the same Windows guard from
  `vllm-project/vllm` master, so 50-series users were never exposed.
- **`pp2_160k` boots cleanly on the public Ampere zip.** Verified on a
  2× RTX 3090 box: `Application startup complete`, KV pool 169,344
  tokens, all 3 coherence tiers pass, decode 41.4 tok/s (95 % of the
  documented 43.5 tok/s).

## Who is affected

- 2-GPU users on the Ampere zip who clicked `pp2_160k` and hit
  `AttributeError: module 'os' has no attribute 'sched_yield'` (the
  most recent crash, after v1.3.3 fixed the previous `ZMQError`).
  Reported in
  [issue #14](https://github.com/devnen/qwen3.6-windows-server/issues/14).
- Single-GPU users see no functional change. Blackwell users see no
  functional change.

## Upgrading

```
update.bat
```

The Ampere variant pulls the new `+devnen.3` wheel and re-runs
`setup.ensure_runtime()` automatically.

## Verification

After upgrading, on a 2× GPU box:

```
snapshots\start_pp2_160k.bat
```

Wait for `Application startup complete`, then:

```
python windows_tools\check_coherence.py --port 5002
```

Expect `COHERENT` (3/3 tiers).

## Files

- `qwen3.6-windows-server-portable-x64-ampere.zip` (and the unsuffixed
  alias for legacy in-place updates from pre-v1.2.3 installs)
- `qwen3.6-windows-server-portable-x64-blackwell.zip`
- `SHA256SUMS.txt`
