"""Reproduce the documented #17140 long-context decode regression.

Pattern from docs/TROUBLESHOOTING.md:
  "Decode tok/s drops 30-70% after a long-context (~24k token) request and
   stays slow for every subsequent request, even short ones, until the
   server is restarted."

Procedure:
  1. Warm: 3 short decode runs, record baseline decode tok/s.
  2. Hit: one ~24k-token prompt, decode 200 tokens.
  3. After: 3 short decode runs, record post-hit decode tok/s.

Pass criteria: post-hit decode is within 15% of baseline.
"""
from __future__ import annotations
import argparse, json, os, time, urllib.request

BASE = os.environ.get("VLLM_BENCH_BASE", "http://127.0.0.1:5001")
MODEL = os.environ.get("VLLM_BENCH_MODEL", "qwen3.6-27b-autoround")
SEED = "The quick brown fox jumps over the lazy dog. "

SHORT_PROMPT = (
    "Write a 300-word plain-text explanation of how transformer attention "
    "works. No lists, no code, no markdown. Single flowing prose."
)

def stream_decode(prompt: str, max_tokens: int = 200) -> dict:
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    last = t0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            text = obj.get("choices", [{}])[0].get("text", "")
            if text:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_tokens += 1
                last = time.perf_counter()
    wall = last - t0
    decode_window = wall - (ttft or 0)
    return {
        "wall_s": round(wall, 3),
        "TTFT_s": round(ttft or 0, 3),
        "decoded_chunks": n_tokens,
        "decode_tok_s": round((n_tokens - 1) / decode_window, 1) if decode_window > 0 and n_tokens > 1 else None,
    }

def long_prompt(target_tokens: int) -> str:
    chars = target_tokens * 4
    return (SEED * (chars // len(SEED) + 1))[:chars]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-tokens", type=int, default=24000)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    print(f"server={BASE} model={MODEL} long_prompt_tokens={args.long_tokens}")
    print()
    print("=== BEFORE long-context request ===")
    pre = []
    for i in range(args.rounds):
        r = stream_decode(SHORT_PROMPT, max_tokens=200)
        print(f"  short[{i+1}] decode={r['decode_tok_s']} TTFT={r['TTFT_s']}s wall={r['wall_s']}s")
        pre.append(r["decode_tok_s"])

    print()
    print(f"=== HIT: ~{args.long_tokens}-token prompt ===")
    r = stream_decode(long_prompt(args.long_tokens), max_tokens=200)
    print(f"  long decode={r['decode_tok_s']} TTFT={r['TTFT_s']}s wall={r['wall_s']}s")

    print()
    print("=== AFTER long-context request ===")
    post = []
    for i in range(args.rounds):
        r = stream_decode(SHORT_PROMPT, max_tokens=200)
        print(f"  short[{i+1}] decode={r['decode_tok_s']} TTFT={r['TTFT_s']}s wall={r['wall_s']}s")
        post.append(r["decode_tok_s"])

    pre_avg = sum(pre) / len(pre)
    post_avg = sum(post) / len(post)
    delta = (post_avg - pre_avg) / pre_avg * 100
    print()
    print(f"=== VERDICT ===")
    print(f"  pre  avg decode tok/s: {pre_avg:.1f}")
    print(f"  post avg decode tok/s: {post_avg:.1f}")
    print(f"  delta: {delta:+.1f}%")
    if delta < -15:
        print("  WARN: REGRESSION (>15% drop) -- issue #17140 reproduces")
    else:
        print("  OK: No regression")

if __name__ == "__main__":
    main()
