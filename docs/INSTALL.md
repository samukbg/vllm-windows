# Install

Three paths, in order of how much you have to do.

## 1. Portable launcher zip, recommended

For users who just want it to run.

1. Open the latest [Release](../../../releases). Pick the right zip for your GPU:
   - **30-series / 40-series (Ampere, Ada):**
     `qwen3.6-windows-server-portable-x64.zip`
     (vLLM 0.19.0+devnen.1, CUDA 12.6 / cu126 torch). This is the default.
   - **50-series (Blackwell — 5060/5070/5080/5090):**
     `qwen3.6-windows-server-portable-x64-blackwell.zip`
     (vLLM 0.20.0+cu132.devnen.1, CUDA 13.2 / cu130 torch, plus an
     auto-built CUDA 13 runtime shim). Requires NVIDIA driver 596 or
     newer. The Blackwell zip also runs on Ampere/Ada if the host has a
     CUDA 13 driver, but the default zip is the recommended path for
     non-Blackwell users today.
   - `SHA256SUMS.txt`.
2. Verify checksums (optional but recommended):
   ```powershell
   Get-FileHash *.zip, *.whl -Algorithm SHA256
   ```
   Compare against `SHA256SUMS.txt`.
3. Extract the launcher zip anywhere, no admin needed, fully relocatable.
4. Either set `VLLM_MODEL_DIR` to point at your existing Qwen3.6 weights, or
   download the model into the bundled `models\Qwen3.6-27B-int4-AutoRound\`
   folder.
5. Double-click `start.bat` at the top of the extracted folder.
6. The TUI walks you through:
   - Detecting your GPUs and warning if any are below sm_86.
   - Asking which snapshot to launch.
   - First run: offers to install the wheel into a sibling `venv\` and apply
     the patches automatically.
   - Optional one-click coherence check after the server boots.

## 2. Wheel-only, for users with their own venv

If you already manage Python environments and just want the patched wheel:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install <url-to-wheel-from-Release>
```

Then apply the Windows patches if your wheel was the upstream
SystemPanic build (the devnen wheel already has them baked in):

```powershell
python windows_tools\apply_patches.py --venv venv
python windows_tools\verify_install.py --venv venv
```

You can launch any snapshot directly:

```powershell
$env:VLLM_WINDOWS_VENV = "$PWD\venv"
$env:VLLM_MODEL_DIR    = "G:\_models\Qwen3.6-27B-int4-AutoRound"
.\snapshots\start_speed.bat
```

## 3. From source, only if you must

The patched source tree in this fork is what produces the wheel. Two
toolchains depending on which release line you target:

- **0.19.x (Ampere/Ada line):** CUDA 12.6, MSVC 2022, PyTorch
  2.11.0+cu126. Build follows SystemPanic's
  [original instructions](https://github.com/SystemPanic/vllm-windows#building-from-source)
  verbatim. Expect 2–4 hours on a 5950X-class machine.
- **0.20.x (Blackwell line):** CUDA 13.2, MSVC 2022, PyTorch cu130. The
  bundled `vllm-0.20.0+cu132.devnen.1` wheel is produced by overlay
  rather than full rebuild — `windows_patches/repackage_wheel.py` patches
  the Python-only files (reasoning parser, serving) on top of
  SystemPanic's prebuilt `vllm-0.20.0+cu132` wheel. No CMake build
  required for Python-level patches; reach for the full source build
  only if you need to touch CUDA kernels.

## After install: first-run sanity

Whichever path you took, run the install verifier:

```powershell
python windows_tools\verify_install.py --venv .\venv
```

Green = good. Yellow = warnings (usually missing MSVC, only matters for
FlashInfer JIT which we don't use). Red = something is broken; fix before
launching.

## Model weights

The default model is
[`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound).
Download with `huggingface-cli` or `snapshot_download`:

```powershell
$env:HF_HOME = "G:\_hf_cache"
huggingface-cli download Lorbus/Qwen3.6-27B-int4-AutoRound `
    --local-dir G:\_models\Qwen3.6-27B-int4-AutoRound
```

After downloading, **always** verify shard SHAs and patch the tokenizer:

```powershell
python windows_tools\verify_model_sha.py G:\_models\Qwen3.6-27B-int4-AutoRound
python windows_tools\patch_tokenizer.py  G:\_models\Qwen3.6-27B-int4-AutoRound
```

`verify_model_sha.py` catches torrent-like corruption that produces
fast-but-degenerate output. `patch_tokenizer.py` flips the `tokenizer_class`
from Lorbus's custom `TokenizersBackend` (which transformers 4.57 doesn't
recognise on Windows) to `Qwen2Tokenizer`. A `.bak` is preserved; the patch
is idempotent. **Re-run after every fresh download**, HF redownloads
overwrite the patched copy.
