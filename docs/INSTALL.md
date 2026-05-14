# Install

Three paths, in order of how much you have to do.

## 1. Portable launcher zip, recommended

For users who just want it to run.

1. Open the latest [Release](../../../releases). Pick the right zip for your GPU:
   - **30-series / 40-series (Ampere, Ada):**
     `qwen3.6-windows-server-portable-x64-ampere.zip`
     (vLLM 0.19.0+devnen.3, CUDA 12.6 / cu126 torch). This is the default.
   - **50-series (Blackwell, 5060/5070/5080/5090):**
     `qwen3.6-windows-server-portable-x64-blackwell.zip`
     (vLLM 0.20.0+cu132.devnen.2, CUDA 13.2 / cu130 torch, plus an
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
   - First run: bootstraps the embedded Python with `setuptools`,
     `wheel`, and `pybind11` (needed because the Blackwell wheel pulls
     fastsafetensors from a SystemPanic source tarball whose
     `pyproject.toml` imports pybind11 at module load and pip's
     build-isolation env doesn't always propagate on embedded Python),
     then installs the bundled vLLM wheel into a sibling `venv\`. The
     devnen patches are baked into the wheel, there's nothing to apply.
   - Optional one-click coherence check after the server boots.

## 2. Wheel-only, for users with their own venv

If you already manage Python environments and just want the patched wheel:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install <url-to-wheel-from-Release>
```

The devnen wheel has all Windows patches (wildcard `served-model-name`,
qwen3 reasoning parser, on 0.19 also the CPU-relay distributed shims)
baked in via the engine fork, there's nothing to apply at install
time. Confirm the wheel is the right one:

```powershell
python windows_tools\verify_install.py --venv venv
```

If you installed a SystemPanic upstream wheel by accident,
`verify_install.py` will flag the missing `+devnen` local-version tag
in red. Reinstall from the wheel bundled in the launcher zip's
`wheels\` directory.

You can launch any snapshot directly:

```powershell
$env:VLLM_WINDOWS_VENV = "$PWD\venv"
$env:VLLM_MODEL_DIR    = "D:\models\Qwen3.6-27B-int4-AutoRound"
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
  bundled `vllm-0.20.0+cu132.devnen.2` wheel is produced from the
  `vllm-for-windows-0.20.0` branch of the
  [`devnen/vllm-windows`](https://github.com/devnen/vllm-windows) engine
  fork (4 commits on top of upstream v0.20.0: reasoning parser mirror,
  hardwired wildcard `served-model-name` in `serving.py`, the
  Windows ZMQ ipc -> tcp fallback in `network_utils.py` added in
  `+devnen.2`, and a generalized `repackage_wheel.py`). The repackage script overlays the
  Python-only patches onto SystemPanic's prebuilt `vllm-0.20.0+cu132`
  wheel without re-running CMake; reach for the full source build only
  if you need to touch CUDA kernels.

## After install: first-run sanity

Whichever path you took, run the install verifier:

```powershell
python windows_tools\verify_install.py --venv .\venv
```

Green = good. Yellow = warnings. Red = something is broken; fix before
launching.

What each row means:

| Row | What it checks | Common causes of yellow / red |
|---|---|---|
| `vllm` | vllm imports and version starts with 0.19.x or 0.20.x | RED if the venv's pip install never finished (re-run `start.bat` to repair). YELLOW if the wheel is some other version (this fork has only validated 0.19.x and 0.20.x). |
| `devnen_tag` | The wheel's PEP 440 local-version segment is `+devnen.*` (0.19 line) or `+cu132.devnen.*` (0.20 line). This is the only at-runtime evidence that the devnen patches (wildcard `served-model-name`, qwen3 reasoning parser, on 0.19 also the CPU-relay distributed shims) are present, they're baked into the wheel by the engine fork, not applied as runtime overlays. | RED if you ran `pip install --upgrade vllm` and pulled an upstream wheel without the local-version tag. Reinstall from the launcher zip's bundled `wheels\` directory. |
| `gpu` | `nvidia-smi` enumerates at least one GPU, and the wheel and GPU's compute capability agree | RED if no GPU. YELLOW if the wheel and GPU mismatch: cu126 wheel + Blackwell GPU (use the `-blackwell` zip), or cu130 wheel + Ampere/Ada (works with driver 596+; harmless). |
| `cuda13_shim` | `cuda13_shim\bin\cudart64_13.dll` exists (only checked for 0.20 wheels) | YELLOW if missing on a Blackwell install. The launcher rebuilds it from `venv\Lib\site-packages\torch\lib\` on the next boot, so this self-heals, usually means the install is brand new and hasn't booted yet. |
| `msvc` | `cl.exe` is on PATH or a known VS install path exists | YELLOW always-OK. Only matters for the flashinfer-sampler decode boost; the PyTorch fallback sampler works without MSVC. See [`TUNING.md`](TUNING.md). |

## Model weights

The default model on Ampere/Ada is
[`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound).
On Blackwell (RTX 5090) the default since v1.3.0 is
[`Peutlefaire/Qwen3.6-27B-NVFP4`](https://huggingface.co/Peutlefaire/Qwen3.6-27B-NVFP4),
loaded by the `rtx5090_nvfp4` snapshot via the separate
`VLLM_NVFP4_MODEL_DIR` env var. The `rtx5090_nvfp4_vision` snapshot
(experimental) reuses the same weights with the unquantized visual
tower loaded for image and video input. As of v1.3.7 these are the
only two 5090 snapshots; the AutoRound INT4 5090 snapshots were
removed since they cannot escape the 170W prefill ceiling on consumer
Blackwell. AutoRound INT4 remains the path for Ampere/Ada (3090,
4090). See [`BLACKWELL.md`](BLACKWELL.md).
Download with `huggingface-cli` or `snapshot_download`:

```powershell
$env:HF_HOME = "D:\hf_cache"
huggingface-cli download Lorbus/Qwen3.6-27B-int4-AutoRound `
    --local-dir D:\models\Qwen3.6-27B-int4-AutoRound
```

After downloading, **always** verify shard SHAs and patch the tokenizer:

```powershell
python windows_tools\verify_model_sha.py D:\models\Qwen3.6-27B-int4-AutoRound
python windows_tools\patch_tokenizer.py  D:\models\Qwen3.6-27B-int4-AutoRound
```

`verify_model_sha.py` catches torrent-like corruption that produces
fast-but-degenerate output. `patch_tokenizer.py` flips the `tokenizer_class`
from Lorbus's custom `TokenizersBackend` (which transformers 4.57 doesn't
recognise on Windows) to `Qwen2Tokenizer`. A `.bak` is preserved; the patch
is idempotent. **Re-run after every fresh download**, HF redownloads
overwrite the patched copy.
