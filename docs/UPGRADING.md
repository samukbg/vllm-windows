# Upgrading

> **v1.2.2 — important decode-tps fix.** Disables `--enable-prefix-caching`
> in all 12 snapshots. Pre-v1.2.2 builds had a stepwise decode slowdown
> after long-context requests because Qwen3-Next's Mamba/SSM state is
> incompatible with prefix caching ([vLLM issue #17140](https://github.com/vllm-project/vllm/issues/17140)).
> If you upgrade with `update.bat` you keep your `launcher\configs.yaml`
> by default, which is fine — the snapshot `.py` files (which carry the
> actual flag) are replaced. The configs.yaml `enable_prefix_caching`
> field is informational only; the launcher reads the flag from the `.py`
> file at boot. Re-launch your snapshot after upgrading and decode will
> stay at documented speed across mixed workloads. Full write-up in
> [`docs/TUNING.md`](TUNING.md).

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
`rtx5090_speed`, `rtx5090`, and `rtx5090_max` snapshots will still be
in `launcher\configs.yaml` after switching to ampere, but they won't
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
