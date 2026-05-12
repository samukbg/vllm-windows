# Upgrading

> **v1.3.3 — PP=2 fixed on Ampere, real long-prompt bench fixture.**
> The `pp2_160k` (Both-GPU big-ctx) snapshot failed to boot on every
> public release zip prior to v1.3.3 with
> `ZMQError: Protocol not supported (addr='ipc://...')` because pyzmq
> has no `ipc://` transport on Windows. New patched wheels —
> `vllm-0.19.0+devnen.3` (Ampere, CUDA 12.6) and
> `vllm-0.20.0+cu132.devnen.2` (Blackwell, CUDA 13.2) — add a
> Windows-only ipc -> tcp fallback in `vllm/utils/network_utils.py`,
> plus a worker-pipe `_ConnectionBase` widening on the Ampere wheel
> (the Blackwell wheel inherited that piece from upstream 0.20.0).
> PP=2 boots cleanly and decodes within ~10 % of the documented
> 40.3 tok/s on 2× RTX 3090 (verified on a real 2× RTX 3090 reference
> box, 2026-05-07).
>
> Two more bench-side fixes:
>
> - `windows_tools/bench_summarize.py` now runs from a stock install
>   without a wrapper. The embedded Python's `python312._pth` adds
>   `..\windows_tools` so `import bench` resolves. Embedded Python
>   ignores `cwd` and `PYTHONPATH`, so the `_pth` line is the only
>   fix.
> - `windows_tools/bench_prompt_sample.py` is now a real ~130 KB /
>   ~25 k-token fixture (verbatim copy of CPython 3.12's `Lib/inspect.py`
>   under the PSF Agreement). Replaces the 670-token stub. Documented
>   `decode_tps` numbers are reproducible from a clean install.
>
> Single-GPU users see no functional change — the new wheel is a
> strict superset of `+devnen.1`.

> **v1.3.2 — Blackwell env hardening + cache-poison prevention.** A
> hotfix for a class of slow-prefill regression that bit RTX 5090 NVFP4
> users when system CUDA installs (or conda `cudatoolkit`) leaked into
> the launch env. New `clean_cuda_env()` in `snapshots/_common.py`
> builds the `rtx5090_nvfp4` subprocess env from scratch (drops every
> `CUDA_*` / `NVCC_*` / `CUDNN_*` key inherited from the host, filters
> NVIDIA-toolkit and conda `Library/bin` from PATH, pins the cu13
> shim). New `preflight_sm120a_or_die()` 5-second probe hard-exits
> before the 11-minute warmup if FlashInfer can't dispatch sm_120a.
> New `windows_tools/wipe_caches.py` recovers the four caches that get
> poisoned (`~/.cache/vllm/`, torchinductor temp, `~/.cache/torch/`,
> `~/.cache/flashinfer/`) with mv-to-`.bak.<timestamp>` for forensics.
> New `cache_env_stamp_check()` writes
> `~/.cache/vllm/.env_stamp.json` and warns on mismatch.
>
> **If you're on v1.3.0 or v1.3.1 and saw slow prefill** (~750 tok/s
> on a 47 k NVFP4 prompt, SM=100 %, mem-BW≈0 %, ~200 W during load),
> the cache from before the upgrade may already be poisoned. After
> running `update.bat`, also run **once**:
>
> ```powershell
> python windows_tools\wipe_caches.py
> ```
>
> Then relaunch. Cold rebuild takes ~11–25 min. Subsequent boots are
> back to the documented 3–8 min. Full forensic write-up in
> internal forensic notes. Hygiene-only edit on
> Ampere/Ada paths; they keep the legacy `cuda_env()` semantics and
> aren't exposed to this class of bug.

> **v1.2.5 — prefix caching back on, big prefill speedup.** Re-enables
> `--enable-prefix-caching` in all 12 snapshots. The v1.2.2-era
> stepwise decode regression
> ([vLLM issue #17140](https://github.com/vllm-project/vllm/issues/17140))
> was fixed upstream by
> [vLLM PR #25752](https://github.com/vllm-project/vllm/pull/25752)
> (Mamba2 Automatic Prefix Caching, merged 2025-10-04), which is in
> both shipped wheels. With prefix caching on, vLLM auto-sets
> `mamba_cache_mode='align'` for Qwen3_5 so SSM state is tracked
> across cache blocks. Net effect on the verified 5090 path: 3-4x
> faster prefill at 12-16k tokens, 24k+ prompts no longer time out,
> +18 % KV pool headroom, and no decode regression after repeated
> long-context hits. Same fix flipped for the 3090/4090 snapshots on
> the assumption that the same upstream code path is in the
> `vllm-0.19.0+devnen.1` source tree (it is — verified via gh API).
> If you upgrade with `update.bat` you keep your `launcher\configs.yaml`
> by default, which is fine — the snapshot `.py` files carry the
> actual flag and are replaced. Full write-up and bench tables in
> [`docs/TUNING.md`](TUNING.md).
>
> **History note:** v1.2.2 (released a few days earlier) shipped with
> prefix caching **off** as a defensive workaround for the same
> regression. Both v1.2.2 and v1.2.5 produce coherent, stable output;
> v1.2.5 is faster and uses less VRAM for the same context window.
> Users on v1.2.x can upgrade in place without losing custom snapshots
> or model weights.

This launcher is fully portable and ships an in-place updater. The
short version: double-click `update.bat`, accept the defaults, done.

## TL;DR

```
update.bat
```

That:

1. Detects whether you're on the Ampere/Ada zip (3090, 4090, A6000) or
   the Blackwell zip (5060, 5070, 5080, 5090) by looking at the bundled
   wheel filename.
2. Hits the GitHub Releases API and finds the matching latest zip.
3. Downloads it and verifies the SHA256 against `SHA256SUMS.txt` from
   the same release.
4. Stops any running snapshot.
5. Replaces every part of the install that should be replaced, and
   leaves the parts that hold your data alone.
6. Asks if you want to relaunch `start.bat`. Default yes.

The embedded Python (`python\`) is replaced too, even though the
running updater is itself the embedded interpreter. That works via a
detached `_update_finalize.bat` spawned just before `update.py`
exits, which waits for the parent PID, atomically renames
`python.new\` → `python\`, then self-deletes. You may briefly see
that file appear next to `start.bat` during the swap; it's expected.

The whole thing is one prompt to keep `launcher\configs.yaml` (default
yes) and one prompt to relaunch (default yes). Holding Enter through
both does the right thing.

## What is preserved

These are never touched by the updater:

| Entry | Why |
|---|---|
| `user_config.json` | Your saved model directory and any per-install settings. |
| `models\` | The 16 GB of model weights you already downloaded. |
| `logs\` | Including `logs\runtime\<port>.json` manifests so the dashboard keeps state. |
| `venv\` | The 6 GB vLLM runtime env. The launcher's `ensure_runtime` repairs it on next boot when the bundled wheel changes. |
| `cuda13_shim\` | Auto-rebuilt on next boot from `venv\Lib\site-packages\torch\lib\` when running on the Blackwell zip. |
| `launcher\configs.yaml` | Preserved by default; you'll be prompted. The snapshot CRUD editor writes here, so any custom snapshots you added would be lost otherwise. |

Everything else in the install (launcher source, snapshots, docs,
templates, terminal, wheels, embedded Python, `start.bat`,
`update.bat`, README) is replaced wholesale. That's the safe choice
for a bug fix release: any changes you might have made to those files
get overwritten, but you also pick up every fix.

## What changes when the wheel changes

The bundled wheel under `wheels\vllm-*.whl` is replaced on every
upgrade. If the new wheel's filename differs from what's installed in
`venv\`, the launcher's first boot after the upgrade will reinstall
the runtime (~5 to 15 minutes, same as a fresh install). You'll see
`[setup] vLLM runtime install...` in the launcher TUI.

This happens transparently. You don't need to delete `venv\` by hand.

## Switching variants (Ampere/Ada zip ↔ Blackwell zip)

If you bought a 5090, ran the Ampere zip on it (and got
`cudaErrorNoKernelImageForDevice`), and want to switch to the Blackwell
zip without nuking your install:

```
update.bat --variant blackwell
```

The script overrides the autodetected variant, downloads the Blackwell
zip, replaces the wheel, and the next boot installs the cu130 torch and
builds the CUDA 13 runtime shim.

To switch the other way (you sold the 5090, back to a 3090):

```
update.bat --variant ampere
```

Variant switching keeps your `user_config.json`, `models\`, and
`launcher\configs.yaml` (with the prompt) intact. The shipped
`rtx5090_nvfp4` and `rtx5090_nvfp4_vision` snapshots will still be in
`launcher\configs.yaml` after switching to ampere, but they won't
show up on the dashboard if no 50-series card is detected; harmless.
You can delete them from the snapshot editor (`e` on the dashboard)
if you want a clean list.

## Re-running the updater offline / against a custom zip

If your machine can't reach `api.github.com` (locked-down corporate
network, etc.), download the zip manually from the
[Releases page](https://github.com/devnen/qwen3.6-windows-server/releases),
copy it to the install machine, and:

```
update.bat --zip "C:\path\to\qwen3.6-windows-server-portable-x64.zip"
```

The variant is still autodetected from the existing install. The
`--variant` override and the SHA256 check (skipped when no
`SHA256SUMS.txt` is alongside the zip) work the same way.

## Headless / CI

For an agent or CI driving the upgrade hands-off:

```
update.bat --yes --launch
```

`--yes` accepts every default (keep configs.yaml, proceed with
update). `--launch` skips the post-update launch prompt and starts the
launcher TUI. Equivalent: `--no-launch` to update and exit.

`--dry-run` prints the plan (which entries would be replaced, which
preserved) without modifying anything; useful for checking what an
upgrade is about to touch before committing.

## Manual upgrade (no script)

The script is just a wrapper. The manual procedure is:

1. From the launcher TUI, stop any running snapshot. Or run
   `snapshots\stop_vllm.bat`.
2. Download the right zip from the Releases page (Ampere/Ada zip or
   Blackwell zip).
3. Extract it on top of your existing install folder. Windows will
   ask whether to overwrite; say Yes to All.
4. The extracted zip does not contain `models\`, `logs\`, `venv\`, or
   `cuda13_shim\`, so those are untouched. It does contain a fresh
   `launcher\configs.yaml`, which will overwrite any custom snapshots
   you added; if you want to keep them, copy your existing
   `launcher\configs.yaml` out before extracting and copy it back
   after.
5. Re-run `start.bat`.

The script automates all of this and adds the SHA256 check, the
variant detection, and the `configs.yaml` preservation prompt.

## When a clean reinstall is the right answer

`update.bat` is for normal in-place upgrades. Reach for a clean
reinstall (delete the install folder, extract a fresh zip) when:

- The launcher won't boot at all and you're not sure why. The
  `~6 GB venv\` folder is the most common source of "weird state",
  and a clean reinstall costs only the runtime install time.
- You want to free disk space (the venv accumulates wheel caches over
  time, ~1 to 2 GiB on a busy install).
- You're switching to a wildly different wheel (e.g. someone
  publishes a 0.21.x wheel and the in-place upgrade has issues).

Before deleting the install folder, copy these out so you don't have
to redo them:

```powershell
Copy-Item -Recurse `
  "user_config.json", "launcher\configs.yaml", "models" `
  -Destination "C:\backup\qwen36-state\"
```

After extracting the new zip, copy them back in. The model dir is
optional; the launcher will rediscover it via drive scan or via
`VLLM_MODEL_DIR` if you set it.

## Troubleshooting the updater

| Symptom | Fix |
|---|---|
| `cannot reach GitHub: ...` | Network blocked. Download the zip on another machine, copy it over, run `update.bat --zip <path>`. |
| `no 'blackwell' zip on release vX.Y.Z` | The release dropped one of the two variants. Run with the other variant (`--variant ampere`) or wait for the next release. |
| `CHECKSUM MISMATCH` | The download corrupted (or you're pointing at an asset from a different release). Re-run; the next download is into a fresh temp dir. |
| Extraction completes but `start.bat` fails to launch | Open a cmd or PowerShell, `cd` into the install folder, run `start.bat` directly to see the real error message. The launch path uses `cmd /c start` to detach, which hides crashes. |
| `update.bat` works but the dashboard still shows old version metadata | The card layout is read from `launcher\configs.yaml`, which is preserved by default. Re-run `update.bat` and answer **n** to the keep-configs prompt to pick up the shipped layout, or re-edit your customisations on top of the new file. |

## Why the in-place updater exists

Re-extracting a zip on top of an existing install works, but it has
three problems an updater can solve cleanly:

1. **Manual zip-extract overwrites `launcher\configs.yaml`** even when
   the user has customised snapshots. The script asks first.
2. **Manual download + extract has no checksum verification step**.
   The script always verifies SHA256 against the release's
   `SHA256SUMS.txt`.
3. **Variant switching is fiddly.** Going from the Ampere zip to the
   Blackwell zip by hand requires picking the right asset name, which
   is easy to get wrong. The script just takes `--variant blackwell`.

If you prefer to do it by hand anyway, the manual procedure above
still works. Both paths produce an identical install state.
