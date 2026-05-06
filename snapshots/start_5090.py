"""Launch vLLM serving Qwen3.6-27B (Lorbus AutoRound INT4) on a single RTX 5090.

Blackwell-tuned counterpart to start_speed.py. Differences from the 3090
snapshot:

  - Built against vLLM 0.20.0+cu132.devnen.1 (CUDA 13, sm_120 kernels).
  - VLLM_ATTENTION_BACKEND env var is no longer read in 0.20.0; only
    the --attention-backend CLI arg is honored. Setting the env var
    triggers a "Unknown vLLM environment variable" warning.
  - 32 GB VRAM gives ~13 GB of KV-cache headroom after weights, so
    ctx default jumps from 90k (3090) to 200k.
  - GPU pin defaults to "0" since most 5090 boxes ship single-card.

The 3090 snapshots in this folder remain unchanged. Pick the right
snapshot for your card from the launcher dashboard, or via
``configs.yaml -> blackwell`` when running on a 5090-class GPU.
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
    VENV, VLLM_EXE, MODEL_PATH, VCVARS,
    msvc_env, cuda_env, flashinfer_sampler_env, log_path_for,
    enhanced_jinja_path, resolve_cuda_visible_devices,
    print_port_collision_banner, random_dp_rpc_port,
)

SERVED_NAME = "qwen3.6-27b-autoround"
HOST = "0.0.0.0"
PORT = 5001

# ---- Parallelism ------------------------------------------------------------
TP = 1
PP = 1
USE_MTP = True
NUM_SPEC_TOKENS = 6

# ---- Memory + context -------------------------------------------------------
# 5090: 32 GB VRAM. Weights 16.96 GB, leaves 13.5 GB after profile/activations.
# Verified: ctx 240k + mem_util 0.95 boots with KV pool 79,968 tokens at 1.14x
# concurrency. Same decode rate as 200k (89 tok/s on 24k prompt) with +40k ctx.
CTX = 240000
GPU_MEM_UTIL = 0.95
KV_CACHE_DTYPE = "fp8_e4m3"  # TRITON_ATTN only accepts fp8_e4m3 on Windows.
MAX_NUM_BATCHED_TOKENS = 4128

# ---- Misc -------------------------------------------------------------------
ENFORCE_EAGER = False     # cudagraphs on
ENABLE_VISION = False     # MoonViT off; Windows c10d allreduce instability
GPU_INDEX = "0"           # most 5090 boxes are single-card; override via CUDA_VISIBLE_DEVICES


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host if host != "0.0.0.0" else "127.0.0.1", port))
            return True
        except OSError:
            return False


def main() -> int:
    if not VLLM_EXE.exists():
        print(f"[ERROR] vllm.exe not found at {VLLM_EXE}", file=sys.stderr)
        return 1
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Model dir not found: {MODEL_PATH}", file=sys.stderr)
        return 1
    if port_in_use(HOST, PORT):
        print_port_collision_banner(PORT)
        try: input("Press Enter to close...")
        except EOFError: pass
        return 1

    env = os.environ.copy()
    _msvc = msvc_env()
    env.update(_msvc)
    env.update(cuda_env())
    env.update(flashinfer_sampler_env(_msvc))
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
    env["VLLM_MARLIN_USE_ATOMIC_ADD"] = "1"
    env["RAY_memory_monitor_refresh_ms"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # vLLM 0.20.0 ignores VLLM_ATTENTION_BACKEND env var; only the
    # --attention-backend CLI arg is honored. We do not set the env var
    # here so the snapshot doesn't trip the "Unknown vLLM environment
    # variable" warning.
    # Windows torch.distributed stability:
    env["USE_LIBUV"] = "0"
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    env["PYTHONFAULTHANDLER"] = "1"

    args = [
        str(VLLM_EXE), "serve", MODEL_PATH,
        f"--served-model-name={SERVED_NAME}",
        "--quantization=auto-round",
        f"--max-model-len={CTX}",
        "--max-num-seqs=1",
        f"--max-num-batched-tokens={MAX_NUM_BATCHED_TOKENS}",
        "--block-size=32",
        # Prefix caching: re-enabled in v1.2.4. The original v1.2.2 doc cited
        # vLLM #17140 (130 -> 90 -> 40 tok/s stepwise decode regression after
        # 24k-token requests) — that bug was fixed by PR #25752 / Mamba2 APC
        # (merged 2025-10-04), included in our vLLM 0.20.0 wheel. With prefix
        # caching enabled, vLLM auto-sets mamba_cache_mode='align' for
        # Qwen3_5 (vllm/model_executor/models/config.py:367), so SSM state
        # is properly tracked across cache blocks.
        # Verified: (a) no decode regression across 2x 24k-token hits
        # (119.3 -> 122.4 -> 122.0 tok/s, +2.6% / +2.3% drift, see
        # windows_tools/repro_17140.py); (b) chunked-prefill at small
        # max_num_batched_tokens no longer hits the 5x cliff at 12-16k tokens;
        # (c) 24k+ prompts that previously timed out now run at ~2-4k tok/s.
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
    if not ENABLE_VISION:
        args.append('--limit-mm-per-prompt={"image":0,"video":0}')
    if _world > 1:
        args.append("--distributed-executor-backend=mp")
    if USE_MTP:
        args.append(
            f'--speculative-config={{"method":"mtp","num_speculative_tokens":{NUM_SPEC_TOKENS}}}'
        )

    print("=" * 60)
    print(f"vLLM serve: {SERVED_NAME}  (Blackwell snapshot)")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Ctx     : {CTX}  |  TP: {TP}  |  PP: {PP}")
    print(f"  KV dtype: {KV_CACHE_DTYPE}  |  MTP: {USE_MTP} (n={NUM_SPEC_TOKENS})")
    print(f"  Listen  : http://{HOST}:{PORT}")
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
