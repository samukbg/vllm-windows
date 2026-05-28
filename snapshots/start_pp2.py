"""PP=2 variant, uses BOTH 3090s so KV cache fits a much larger context.

Measured on 2x RTX 3090, Qwen3.6-27B Lorbus AutoRound INT4, fp8_e4m3 KV,
TRITON_ATTN, gpu-memory-utilization=0.92:
  - KV pool: 169,344 tokens total.
  - ctx=160000: 43.5 tok/s decode, TTFT 0.76 s (committed snapshot).
  - ctx=128000: 43.2 tok/s decode, TTFT 0.87 s.
  - ctx=96000:  43.1 tok/s decode, TTFT 2.22 s (first-run cold cache).

Spec-decode disabled on vLLM 0.19.0 because every option breaks with PP>1:
  - MTP          -> NotImplementedError on Qwen3-Next (documented).
  - ngram        -> 'GPUModelRunner' object has no attribute 'drafter'.
  - draft-model  -> unsupported with PP>1 since vLLM 0.15.

Port 5002 so it can coexist with the 72-tok/s MTP baseline on port 5001.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from _common import VENV, VLLM_EXE, PYTHON_EXE, VLLM_BASE_CMD, MODEL_PATH, VCVARS, msvc_env, cuda_env, flashinfer_sampler_env, log_path_for, enhanced_jinja_path, resolve_cuda_visible_devices, print_port_collision_banner
SERVED_NAME = "qwen3.6-27b-autoround"
HOST = "0.0.0.0"
PORT = 5002  # baseline 72-tok/s server owns 5001

# ---- Parallelism ------------------------------------------------------------
TP = 1
PP = 2  # split model across GPU0 + GPU1 -> free up KV space for context
USE_NGRAM = False  # Drafter pre-init patch unblocks boot but PP+ngram trips an
                    # assertion in _prepare_inputs (total_num_scheduled_tokens>0)
                    #, structural scheduler bug, not patchable inline.
NGRAM_NUM_SPEC_TOKENS = 5
NGRAM_PROMPT_LOOKUP_MAX = 4
NGRAM_PROMPT_LOOKUP_MIN = 2

# ---- Memory + context -------------------------------------------------------
# KV pool at these settings holds ~169k tokens. 160k leaves ~9k headroom for
# the prompt. If you change GPU_MEM_UTIL, KV_CACHE_DTYPE, or the model, boot
# once and re-read the "GPU KV cache size" line from the log before raising.
CTX = 160000
GPU_MEM_UTIL = 0.92  # PP=2 baseline; 0.93 saved measurably nothing (KV pool same)
KV_CACHE_DTYPE = "fp8_e4m3"
MAX_NUM_BATCHED_TOKENS = 4128

ENFORCE_EAGER = False
ENABLE_VISION = False
def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host if host != "0.0.0.0" else "127.0.0.1", port))
            return True
        except OSError:
            return False


def main() -> int:
    if not PYTHON_EXE.exists():
        print(f"[ERROR] python.exe not found at {PYTHON_EXE}", file=sys.stderr)
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
    # vLLM 0.19 unconditionally imports flashinfer in the sampler;
    # flashinfer's Windows path raises if CUDA_LIB_PATH is unset.
    env.update(cuda_env())
    # Toggle the flashinfer sampler based on MSVC + ninja availability,
    # since flashinfer JIT-compiles a sampling module at first profile_run.
    env.update(flashinfer_sampler_env(_msvc))
    ENHANCED_JINJA = enhanced_jinja_path()
    if not Path(ENHANCED_JINJA).exists():
        print(f"[ERROR] enhanced jinja template not found: {ENHANCED_JINJA}", file=sys.stderr)
        return 1
    _cvd = resolve_cuda_visible_devices("0,1", 2)
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
    env["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    env["PYTHONFAULTHANDLER"] = "1"
    # Reddit Splinter2121 envs were measured on PP=2 2026-04-25, regressed
    # decode 40.3 → 39.2 tok/s. Probably Gloo-CPU-relay bound, not compute/mem.

    args = [
        *VLLM_BASE_CMD, "serve", MODEL_PATH,
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
        "--no-scheduler-reserve-full-isl", # +5k KV pool; decode within noise
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
        "--distributed-executor-backend=mp",
        "--no-use-tqdm-on-load",
        f"--host={HOST}",
        f"--port={PORT}",
    ]
    if ENFORCE_EAGER:
        args.append("--enforce-eager")
    if not ENABLE_VISION:
        args.append('--limit-mm-per-prompt={"image":0,"video":0}')
    if USE_NGRAM:
        spec = (
            '{"method":"ngram",'
            f'"num_speculative_tokens":{NGRAM_NUM_SPEC_TOKENS},'
            f'"prompt_lookup_max":{NGRAM_PROMPT_LOOKUP_MAX},'
            f'"prompt_lookup_min":{NGRAM_PROMPT_LOOKUP_MIN}'
            '}'
        )
        args.append(f"--speculative-config={spec}")

    print("=" * 60)
    print(f"vLLM serve: {SERVED_NAME}  (PP=2 + ngram variant)")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Ctx     : {CTX}  |  TP: {TP}  |  PP: {PP}")
    print(f"  KV dtype: {KV_CACHE_DTYPE}  |  ngram: {USE_NGRAM} (n={NGRAM_NUM_SPEC_TOKENS})")
    print(f"  Listen  : http://{HOST}:{PORT}")
    print("=" * 60)
    print(" ".join(args))
    print("=" * 60, flush=True)

    log_path = HERE / "vllm_server_pp2.log"
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
            max_model_len=CTX, mtp_n=NGRAM_NUM_SPEC_TOKENS if USE_NGRAM else None,
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
