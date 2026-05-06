"""Sweep prompt sizes to isolate prefill cost.

Reports TTFT and prefill tok/s for prompts of varying length on the
same server, max_tokens=8 so decode contribution is minimal.
"""
from __future__ import annotations

import argparse, json, os, time, urllib.request

BASE = os.environ.get("VLLM_BENCH_BASE", "http://127.0.0.1:5001")
MODEL = os.environ.get("VLLM_BENCH_MODEL", "qwen3.6-27b-autoround")

# Build prompts roughly of N tokens by repeating a short string. ~4 chars/token.
SEED = "The quick brown fox jumps over the lazy dog. "

def run(target_tokens: int) -> dict:
    chars = target_tokens * 4
    prompt = (SEED * (chars // len(SEED) + 1))[:chars]
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        obj = json.loads(r.read())
    wall = time.perf_counter() - t0
    pt = obj.get("usage", {}).get("prompt_tokens")
    return {
        "target_tokens": target_tokens,
        "prompt_tokens": pt,
        "wall_s": round(wall, 3),
        "prefill_tok_s": round(pt / wall, 1) if pt and wall else None,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,256,1024,4096,8192,16384")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    print(f"server={BASE} model={MODEL}")
    print(f"{'target':>8} {'prompt':>8} {'wall_s':>8} {'prefill_tok_s':>14}")
    for n in sizes:
        try:
            r = run(n)
        except Exception as e:
            print(f"{n:>8} ERROR {e}")
            continue
        print(f"{n:>8} {str(r['prompt_tokens']):>8} {r['wall_s']:>8} {str(r['prefill_tok_s']):>14}")

if __name__ == "__main__":
    main()
