"""Repeated long-context hits to test stepwise decode regression.

Sequence: short, short, [long, short, short] x N. Watch decode tok/s on
short requests after each long hit.
"""
from __future__ import annotations
import argparse, json, os, time, urllib.request

BASE = os.environ.get("VLLM_BENCH_BASE", "http://127.0.0.1:5001")
MODEL = os.environ.get("VLLM_BENCH_MODEL", "qwen3.6-27b-autoround")
SEED = "The quick brown fox jumps over the lazy dog. "

SHORT_PROMPT = (
    "Write a 300-word plain-text explanation of how transformer attention "
    "works. No lists, no code, no markdown."
)

def stream_decode(prompt, max_tokens=200):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None; n = 0; last = t0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: "): continue
            payload = line[6:].strip()
            if payload == b"[DONE]": break
            try: obj = json.loads(payload)
            except: continue
            text = obj.get("choices", [{}])[0].get("text", "")
            if text:
                if ttft is None: ttft = time.perf_counter() - t0
                n += 1; last = time.perf_counter()
    decode_window = (last - t0) - (ttft or 0)
    return round((n - 1) / decode_window, 1) if decode_window > 0 and n > 1 else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-tokens", type=int, default=20000)
    ap.add_argument("--cycles", type=int, default=4)
    args = ap.parse_args()
    long = (SEED * (args.long_tokens * 4 // len(SEED) + 1))[:args.long_tokens * 4]

    print(f"server={BASE} long_tokens~{args.long_tokens} cycles={args.cycles}")
    base = []
    print("Baseline (3 short):")
    for i in range(3):
        d = stream_decode(SHORT_PROMPT)
        print(f"  short -> {d} tok/s")
        base.append(d)
    base_avg = sum(base)/len(base)

    history = [base_avg]
    for c in range(args.cycles):
        print(f"\nCycle {c+1}: long-context hit ...")
        long_d = stream_decode(long, max_tokens=100)
        print(f"  long({args.long_tokens}) -> {long_d} tok/s")
        post = []
        for i in range(2):
            d = stream_decode(SHORT_PROMPT)
            post.append(d)
            print(f"  short -> {d} tok/s")
        post_avg = sum(post)/len(post)
        delta = (post_avg - base_avg)/base_avg*100
        history.append(post_avg)
        print(f"  cycle{c+1} avg post-long short decode: {post_avg:.1f} ({delta:+.1f}% vs baseline)")

    print("\n=== Trajectory ===")
    print("baseline:", history[0])
    for i, v in enumerate(history[1:], 1):
        delta = (v - history[0])/history[0]*100
        print(f"after cycle {i}: {v:.1f} ({delta:+.1f}% vs baseline)")

if __name__ == "__main__":
    main()
