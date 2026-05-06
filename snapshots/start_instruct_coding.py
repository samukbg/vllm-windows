"""Launch vLLM serving Qwen3.6-27B in non-thinking (Instruct) mode tuned
for coding workloads at 127k context.

Sampler defaults match Unsloth's recommended Instruct-coding row
(temperature 0.6, top_p 0.95, top_k 20, min_p 0.0). Thinking is disabled
in the chat template so the model goes straight to the answer with no
<think> block. Per-request sampler params still override these defaults.

Use this snapshot for tool-heavy coding agents (Cline, Continue, Codex,
Roo, Cursor) where you want crisp, low-randomness output and don't need
the reasoning trace.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from _common import VENV, VLLM_EXE, MODEL_PATH, VCVARS, msvc_env, cuda_env, flashinfer_sampler_env, log_path_for, enhanced_jinja_path, resolve_cuda_visible_devices, print_port_collision_banner
SERVED_NAME = "qwen3.6-27b-autoround"
HOST = "0.0.0.0"
PORT = 5001

TP = 1
PP = 1
USE_MTP = True
NUM_SPEC_TOKENS = 3

CTX = 127000
GPU_MEM_UTIL = 0.948
KV_CACHE_DTYPE = "fp8_e4m3"
MAX_NUM_BATCHED_TOKENS = 4128

ENFORCE_EAGER = False
ENABLE_VISION = False

# Unsloth's Instruct-coding sampler row.
SAMPLER_OVERRIDE = '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0}'
# preserve_thinking=false strips prior <think> blocks from history;
# enable_thinking=false flips the model into Instruct (non-thinking) mode.
CHAT_TEMPLATE_KWARGS = '{"preserve_thinking": false, "enable_thinking": false}'
MODE_LABEL = "Instruct (non-thinking), coding"


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
    _cvd = resolve_cuda_visible_devices("1", _world)
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
    env["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
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
        # Prefix caching re-enabled in v1.2.5: vLLM PR #25752 (Mamba2 APC,
        # merged 2025-10-04) ships in our wheel and auto-sets
        # mamba_cache_mode='align' for Qwen3_5
        # (vllm/model_executor/models/config.py:367), fixing the v1.2.2-era
        # #17140 stepwise decode regression at the source. See
        # snapshots/start_5090.py for the full rationale and bench evidence.
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-auto-tool-choice",
        "--tool-call-parser=qwen3_coder",
        "--reasoning-parser=qwen3",
        f"--chat-template={ENHANCED_JINJA}",
        f"--default-chat-template-kwargs={CHAT_TEMPLATE_KWARGS}",
        f"--override-generation-config={SAMPLER_OVERRIDE}",
        f"--kv-cache-dtype={KV_CACHE_DTYPE}",
        f"--tensor-parallel-size={TP}",
        f"--pipeline-parallel-size={PP}",
        f"--gpu-memory-utilization={GPU_MEM_UTIL}",
        "--trust-remote-code",
        "--attention-backend=TRITON_ATTN",
        "--no-use-tqdm-on-load",
        f"--host={HOST}",
        f"--port={PORT}",
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
    print(f"vLLM serve: {SERVED_NAME}")
    print(f"  Mode    : {MODE_LABEL}")
    print(f"  Sampler : {SAMPLER_OVERRIDE}")
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
