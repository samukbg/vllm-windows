"""Run a sustained workload while parent samples nvidia-smi."""
import argparse, json, os, time, urllib.request, threading, subprocess

BASE = os.environ.get("VLLM_BENCH_BASE", "http://127.0.0.1:5001")
MODEL = os.environ.get("VLLM_BENCH_MODEL", "qwen3.6-27b-autoround")

def fire(prompt, max_tokens):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": False, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        obj = json.load(r)
    dt = time.perf_counter() - t0
    u = obj.get("usage", {})
    return dt, u

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["decode", "prefill"], default="decode")
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=1500)
    args = ap.parse_args()

    SEED = "The quick brown fox jumps over the lazy dog. "
    chars = max(1, args.prompt_tokens) * 4
    prompt = (SEED * (chars // len(SEED) + 1))[:chars]

    print(f"mode={args.mode} prompt_tokens~{args.prompt_tokens} max_tokens={args.max_tokens}")
    dt, u = fire(prompt, args.max_tokens)
    pt = u.get("prompt_tokens", 0); ct = u.get("completion_tokens", 0)
    print(f"wall_s={dt:.2f} prompt_tokens={pt} completion_tokens={ct}")
    if args.mode == "decode" and ct > 1:
        print(f"decode_tok_s~{ct/dt:.1f}")
    if args.mode == "prefill" and pt:
        print(f"prefill_tok_s~{pt/dt:.1f}")

if __name__ == "__main__":
    main()
