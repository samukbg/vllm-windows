"""Launch vLLM serving Qwen3.6-27B (Peutlefaire NVFP4) on a single RTX 5090.

EXPERIMENTAL — NVFP4 prefill-ceiling test.

Background: AutoRound INT4 prefill on the 5090 hits a 170W ceiling because
GDN kernels fall back to FLA Triton (FlashInfer GDN paths are sm_90a or
sm_100a only — neither maps to consumer Blackwell sm_120). NVFP4 weights
route the FFN GEMMs and non-GDN attention through FlashInfer's native
sm_120 FP4 tensor-core path (the one wired via is_sm120a_supported in
flashinfer/utils.py). GDN layers still hit the same ceiling for their
share, but the rest of the model could prefill faster.

See docs/SM120_GDN_CEILING.md "Open angles" section #1 for the test
plan and what to measure.

Reference: u/Maheidem on r/LocalLLaMA (1t5dya8) reports this model
running on a single 5090 with 200k context, vLLM 0.20.1.dev,
Torch 2.13.dev, CUDA 13.0, MTP enabled.
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
)

# NVFP4 weights live separately from the AutoRound default. Override.
MODEL_PATH = os.environ.get(
    "VLLM_NVFP4_MODEL_DIR",
    r"g:\_models\Qwen3.6-27B-NVFP4",
)

SERVED_NAME = "qwen3.6-27b-nvfp4"
HOST = "0.0.0.0"
PORT = 5001  # canonical default port; mutually exclusive with other 5090 snapshots

# ---- Parallelism ------------------------------------------------------------
TP = 1
PP = 1
USE_MTP = True
NUM_SPEC_TOKENS = 6

# ---- Memory + context -------------------------------------------------------
# NVFP4 weights are ~4 bits/param like AutoRound INT4, so VRAM budget is similar.
# Start at 200k ctx (Maheidem reports this works on single 5090); revisit after
# coherence + bench passes.
CTX = 200000
GPU_MEM_UTIL = 0.95
KV_CACHE_DTYPE = "fp8_e4m3"
MAX_NUM_BATCHED_TOKENS = 4128

ENFORCE_EAGER = False
ENABLE_VISION = False
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
    # NB: VLLM_MARLIN_USE_ATOMIC_ADD removed — Marlin is the AutoRound INT4
    # path, not used for NVFP4. Compressed-tensors NVFP4 routes through
    # FlashInfer FP4 kernels.
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
    if not ENABLE_VISION:
        args.append('--limit-mm-per-prompt={"image":0,"video":0}')
    if _world > 1:
        args.append("--distributed-executor-backend=mp")
    if USE_MTP:
        args.append(
            f'--speculative-config={{"method":"mtp","num_speculative_tokens":{NUM_SPEC_TOKENS}}}'
        )

    print("=" * 60)
    print(f"vLLM serve: {SERVED_NAME}  (Blackwell NVFP4 EXPERIMENTAL)")
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
