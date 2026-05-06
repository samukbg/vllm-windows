"""Fast prefill probe with per-request timeout. Aborts slow runs."""
from __future__ import annotations
import argparse, json, os, time, urllib.request, urllib.error, socket

BASE = os.environ.get("VLLM_BENCH_BASE", "http://127.0.0.1:5001")
MODEL = os.environ.get("VLLM_BENCH_MODEL", "qwen3.6-27b-autoround")
SEED = "The quick brown fox jumps over the lazy dog. "

def run(target_tokens: int, timeout_s: float = 8.0) -> dict:
    chars = target_tokens * 4
    prompt = (SEED * (chars // len(SEED) + 1))[:chars]
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": 1,
        "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            obj = json.loads(r.read())
        wall = time.perf_counter() - t0
        pt = obj.get("usage", {}).get("prompt_tokens")
        return {"prompt_tokens": pt, "wall_s": round(wall, 3),
                "prefill_tok_s": round(pt / wall, 1) if pt and wall else None,
                "status": "ok"}
    except (urllib.error.URLError, socket.timeout, TimeoutError):
        wall = time.perf_counter() - t0
        return {"prompt_tokens": None, "wall_s": round(wall, 3),
                "prefill_tok_s": None, "status": f"TIMEOUT>{timeout_s}s"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,1024,1500,2000,2500,3000,3500,4000,4096,4500")
    ap.add_argument("--timeout", type=float, default=8.0)
    args = ap.parse_args()
    print(f"server={BASE} model={MODEL} timeout={args.timeout}s")
    print(f"{'target':>7} {'prompt':>7} {'wall_s':>7} {'tok/s':>9} status")
    for n in (int(s) for s in args.sizes.split(",") if s.strip()):
        r = run(n, args.timeout)
        print(f"{n:>7} {str(r['prompt_tokens']):>7} {r['wall_s']:>7} {str(r['prefill_tok_s']):>9} {r['status']}")

if __name__ == "__main__":
    main()
