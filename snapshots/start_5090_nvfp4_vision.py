"""Launch vLLM serving Qwen3.6-27B (Peutlefaire NVFP4) with vision enabled.

Vision twin of start_5090_nvfp4.py.

The Peutlefaire NVFP4 quant deliberately keeps the visual tower unquantized
(see HF card: ignore=["re:visual.*", "re:model.visual.*"]), the safetensors
already contain the BF16/F32 vision encoder weights alongside the NVFP4 LM
body. Loading them as multimodal is a flag flip, not a different model.

VRAM cost vs the text-only twin (start_5090_nvfp4.py):
  - vision encoder weights resident on GPU: ~1.5-2.0 GiB (BF16)
  - per-image visual tokens consume KV cache like text tokens
    (Qwen3.6 visual tokenizer: ~256-1280 tokens per image depending on res)
  - FlashInfer also has to autotune additional GEMM shapes for the vision
    tower at first boot, expect a longer warmup than the text twin

To stay inside the 32 GB envelope, CTX is dropped from 200k -> 120k. If
you need more, lower --limit-mm-per-prompt or drop mem_util.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from _common import (
    VENV, VLLM_EXE, VCVARS,
    msvc_env, cuda_env, flashinfer_sampler_env, log_path_for,
    enhanced_jinja_path, resolve_cuda_visible_devices,
    print_port_collision_banner, random_dp_rpc_port,
    cache_env_stamp_check, clean_cuda_env, preflight_sm120a_or_die,
)

MODEL_PATH = os.environ.get(
    "VLLM_NVFP4_MODEL_DIR",
    r"g:\_models\Qwen3.6-27B-NVFP4",
)

SERVED_NAME = "qwen3.6-27b-nvfp4-vision"
HOST = "0.0.0.0"
PORT = 5004  # 5001 = text NVFP4, 5002 = pp2_160k, 5003 = draft (blocked), 5004 = vision NVFP4

# ---- Parallelism ------------------------------------------------------------
TP = 1
PP = 1
USE_MTP = True
NUM_SPEC_TOKENS = 6

# ---- Memory + context -------------------------------------------------------
# Vision encoder weights (~2 GiB unquantized) + the 16k-token encoder cache
# trim the KV pool vs the text twin. Measured at boot: with ctx=120000 the
# engine reported KV=66,912 tokens / max-concurrency 1.68x, plenty of slack
# at max_num_seqs=1. ctx 180000 keeps ~1.12x slack, still safe.
CTX = 180000
GPU_MEM_UTIL = 0.95
KV_CACHE_DTYPE = "fp8_e4m3"
MAX_NUM_BATCHED_TOKENS = 4128

ENFORCE_EAGER = False
ENABLE_VISION = True
MM_IMAGE_LIMIT = 4
MM_VIDEO_LIMIT = 1
GPU_INDEX = "0"


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host if host != "0.0.0.0" else "127.0.0.1", port))
            return True
        except OSError:
            return False


def main() -> int:
    # Set FLASHINFER_CUDA_ARCH_LIST before any flashinfer import (including
    # the in-process import done by cache_env_stamp_check below). Otherwise
    # flashinfer.compilation_context's module-level init calls
    # torch.cuda.get_device_capability() which on the cu130 wheel logs
    # "Failed to get device capability: SM 12.x requires CUDA >= 12.9."
    # twice on stderr before falling back. RTX 5090 is sm_120; the wheel
    # is built with compute_120 (non-suffixed) per docs/SM120_GDN_CEILING.md.
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0")

    if not VLLM_EXE.exists():
        print(f"[ERROR] vllm.exe not found at {VLLM_EXE}", file=sys.stderr)
        return 1
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] NVFP4 model dir not found: {MODEL_PATH}", file=sys.stderr)
        print("        Download with:", file=sys.stderr)
        print("        huggingface-cli download Peutlefaire/Qwen3.6-27B-NVFP4 \\", file=sys.stderr)
        print(f"          --local-dir {MODEL_PATH}", file=sys.stderr)
        return 1
    if port_in_use(HOST, PORT):
        print_port_collision_banner(PORT)
        try: input("Press Enter to close...")
        except EOFError: pass
        return 1

    cache_env_stamp_check(snapshot_py=Path(__file__))

    env = clean_cuda_env(os.environ)
    # FLASHINFER_CUDA_ARCH_LIST was set at the top of main() and is
    # in os.environ; clean_cuda_env preserves it on the way through.
    _msvc = msvc_env()
    env.update(_msvc)
    env.update(flashinfer_sampler_env(_msvc))

    _probe_py = VENV / "Scripts" / "python.exe"
    if not _probe_py.exists():
        _probe_py = VENV / "python.exe"
    preflight_sm120a_or_die(env, vllm_python=_probe_py)
    ENHANCED_JINJA = enhanced_jinja_path()
    if not Path(ENHANCED_JINJA).exists():
        print(f"[ERROR] enhanced jinja template not found: {ENHANCED_JINJA}", file=sys.stderr)
        return 1
    _world = TP * PP
    _cvd = resolve_cuda_visible_devices(GPU_INDEX, _world)
    if _cvd is None:
        return 1
    env["CUDA_VISIBLE_DEVICES"] = _cvd
    env["VLLM_SLEEP_WHEN_IDLE"] = "1"
    env["VLLM_ENABLE_CUDAGRAPH_GC"] = "1"
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    env["RAY_memory_monitor_refresh_ms"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["USE_LIBUV"] = "0"
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    env["PYTHONFAULTHANDLER"] = "1"

    args = [
        str(VLLM_EXE), "serve", MODEL_PATH,
        f"--served-model-name={SERVED_NAME}",
        "--quantization=compressed-tensors",
        f"--max-model-len={CTX}",
        "--max-num-seqs=1",
        f"--max-num-batched-tokens={MAX_NUM_BATCHED_TOKENS}",
        "--block-size=32",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-auto-tool-choice",
        "--tool-call-parser=qwen3_coder",
        "--reasoning-parser=qwen3",
        f"--chat-template={ENHANCED_JINJA}",
        '--default-chat-template-kwargs={"preserve_thinking": false}',
        f"--kv-cache-dtype={KV_CACHE_DTYPE}",
        f"--tensor-parallel-size={TP}",
        f"--pipeline-parallel-size={PP}",
        f"--gpu-memory-utilization={GPU_MEM_UTIL}",
        "--trust-remote-code",
        "--attention-backend=TRITON_ATTN",
        "--no-use-tqdm-on-load",
        f"--host={HOST}",
        f"--port={PORT}",
        f"--data-parallel-rpc-port={random_dp_rpc_port()}",
    ]
    if ENFORCE_EAGER:
        args.append("--enforce-eager")
    if ENABLE_VISION:
        args.append(
            f'--limit-mm-per-prompt={{"image":{MM_IMAGE_LIMIT},"video":{MM_VIDEO_LIMIT}}}'
        )
    else:
        args.append('--limit-mm-per-prompt={"image":0,"video":0}')
    if _world > 1:
        args.append("--distributed-executor-backend=mp")
    if USE_MTP:
        args.append(
            f'--speculative-config={{"method":"mtp","num_speculative_tokens":{NUM_SPEC_TOKENS}}}'
        )

    print("=" * 60)
    print(f"vLLM serve: {SERVED_NAME}  (Blackwell NVFP4 + VISION)")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Ctx     : {CTX}  |  TP: {TP}  |  PP: {PP}")
    print(f"  KV dtype: {KV_CACHE_DTYPE}  |  MTP: {USE_MTP} (n={NUM_SPEC_TOKENS})")
    print(f"  Vision  : ENABLED (image<={MM_IMAGE_LIMIT} video<={MM_VIDEO_LIMIT})")
    print(f"  Listen  : http://{HOST}:{PORT}")
    print("=" * 60)
    print("[NOTE] FlashInfer autotune covers vision-tower GEMMs in addition to")
    print("       the LM body, so first-boot warmup is longer than the text-only")
    print("       NVFP4 snapshot (expect 10-15 min cold).")
    print("=" * 60)
    print(" ".join(args))
    print("=" * 60, flush=True)

    log_path = log_path_for(PORT)
    log_f = open(log_path, "w", encoding="utf-8", buffering=1)
    print(f"[launcher] tee stdout -> {log_path} (also streaming to this terminal)")
    proc = subprocess.Popen(
        args, env=env, cwd=str(VENV),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        text=True, encoding="utf-8", errors="replace",
    )

    try:
        from _common import write_manifest
        _mf = write_manifest(
            snapshot_py=Path(__file__),
            port=PORT, wrapper_pid=os.getpid(),
            max_model_len=CTX, mtp_n=NUM_SPEC_TOKENS if USE_MTP else None,
            tp=TP, pp=PP,
        )
        print(f"[launcher] runtime manifest -> {_mf}")
    except Exception as _mfe:
        print(f"[launcher] manifest write failed (non-fatal): {_mfe}", file=sys.stderr)

    import threading
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    def _tee():
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except Exception:
                pass
            log_f.write(line)
    threading.Thread(target=_tee, daemon=True).start()

    def _forward(sig, _frame):
        proc.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)

    try:
        try:
            _rc = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            _rc = proc.wait()
        return _rc
    finally:
        try:
            from _common import clear_manifest
            clear_manifest(PORT)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
