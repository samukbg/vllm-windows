# Uninstalling

The launcher is fully portable. There's no installer, no registry
entries, no system services, no admin rights involved at any point.
Uninstall is just deleting folders.

## Step 1, Stop the server

If a snapshot is running, stop it first:

- From the launcher TUI: select the running snapshot, press the Stop
  button (or Esc).
- From a shell:

  ```powershell
  snapshots\stop_vllm.bat
  ```

  This kills any orphan vLLM, EngineCore, or APIServer processes
  bound to the snapshot ports.

## Step 2, Delete the install folder

Wherever you extracted the portable zip (`qwen3.6-windows-server-portable-x64-ampere.zip`
or `-blackwell.zip`),
delete that folder. It contains:

- The embedded Python 3.12 runtime
- The bundled vLLM wheel and all installed dependencies (~6 GB after
  first-run setup)
- The launcher TUI source
- The portable Windows Terminal
- The bundled chat templates
- Logs (under `logs/`, unless your install dir was read-only, see
  step 3)

That's the entire main install. After this step the project is gone
from your machine *except* for the optional caches and the model
weights, both of which live outside the install folder.

## Step 3, Optional cleanup outside the install folder

These directories are only touched if certain paths were used during
the run. Delete whichever apply to you:

| Path | What's there | Size |
|---|---|---|
| `%LocalAppData%\qwen36-windows-server\` | Logs and saved config when the install dir was read-only (e.g. `Program Files` installs). | A few MB to a few hundred MB |
| `%USERPROFILE%\.cache\vllm\torch_compile_cache\` | Torch compile cache built lazily on first request. Speeds up subsequent boots. | 200–800 MB |
| Your model weights folder | Wherever you downloaded `Qwen3.6-27B-int4-AutoRound`. Common locations: `<drive>:\models\Lorbus\Qwen3.6-27B-int4-AutoRound`, `<drive>:\_models\`, `<drive>:\AI\models\`, `<install>\models\`. | ~16 GB |
| `%USERPROFILE%\.cache\huggingface\` | Hugging Face cache. Only delete if you don't use HF for other tools, since it's shared across all your HF downloads. | varies |

PowerShell one-liner that removes everything (review before running):

```powershell
Remove-Item -Recurse -Force `
  "$env:LocalAppData\qwen36-windows-server", `
  "$env:UserProfile\.cache\vllm\torch_compile_cache" `
  -ErrorAction SilentlyContinue
```

## What's NOT touched

For peace of mind, here's what the launcher never modifies:

- **Registry.** No keys created or modified.
- **System Python.** The embedded Python 3.12 lives entirely inside
  the install folder. Your `python.exe` on PATH is untouched.
- **PATH and environment variables.** No system-wide changes. The
  only env vars used (`VLLM_MODEL_DIR`, `VLLM_WINDOWS_LOGS`, etc.)
  are read at runtime, not set globally.
- **Services.** Nothing installed as a Windows service.
- **Drivers.** No driver changes. Your NVIDIA driver is the only one
  needed, and the launcher doesn't touch it.
- **Firewall.** The server binds to `127.0.0.1` by default, no
  inbound firewall rule is added.

So once the install folder is deleted and the optional caches are
cleared, your machine is in exactly the state it was in before you
extracted the zip.

## Reinstalling later

If you delete and want to come back, just download the latest release
zip and extract again. The launcher will find your existing model
weights on first run if they're still on disk and skip the download.
