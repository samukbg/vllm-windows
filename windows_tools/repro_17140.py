"""Reproduce the documented #17140 stepwise decode regression.

From docs/TUNING.md:
  "short-prompt decode dropped 30% after one 24k-token request and a further
   30% after a second (130 -> 90 -> 40 tok/s, stuck)"

Protocol:
  Phase A: 3x bench.py-style short decode -> baseline
  Phase B: one 24k-token prefill+decode hit
  Phase C: 3x short decode -> after first hit
  Phase D: another 24k-token hit
  Phase E: 3x short decode -> after second hit

Uses the SAME methodology as bench.py: non-streaming, decode_tok_s from
usage.completion_tokens / decode_window. Reports decode_tok_s.
"""
from __future__ import annotations
import json, os, time, urllib.request

BASE = os.environ.get("VLLM_BENCH_BASE", "http://127.0.0.1:5001")
MODEL = os.environ.get("VLLM_BENCH_MODEL", "qwen3.6-27b-autoround")
SEED = "The quick brown fox jumps over the lazy dog. "

SHORT_PROMPT = (
    "Write a 300-word plain-text explanation of how transformer attention "
    "works. No lists, no code, no markdown. Single flowing prose."
)

def bench_one(prompt: str, max_tokens: int) -> dict:
    """Mirror windows_tools/bench.py: stream + measure TTFT and decode tok/s."""
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    completion_tokens = 0
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
            text = obj.get("choices", [{}])[0].get("text", "") if obj.get("choices") else ""
            if text and ttft is None:
                ttft = time.perf_counter() - t0
            if obj.get("usage"):
                completion_tokens = obj["usage"].get("completion_tokens", completion_tokens)
            last = time.perf_counter()
    wall = last - t0
    decode_window = wall - (ttft or 0)
    return {
        "wall_s": round(wall, 3),
        "TTFT_s": round(ttft or 0, 3),
        "completion_tokens": completion_tokens,
        "decode_tok_s": round((completion_tokens - 1) / decode_window, 1)
                        if decode_window > 0 and completion_tokens > 1 else None,
    }

def long_prompt(target_tokens: int) -> str:
    chars = target_tokens * 4
    return (SEED * (chars // len(SEED) + 1))[:chars]

def phase(name: str, n: int):
    print(f"\n=== {name} ({n} short bench runs) ===")
    out = []
    for i in range(n):
        r = bench_one(SHORT_PROMPT, max_tokens=300)
        print(f"  [{i+1}] decode={r['decode_tok_s']} tok/s  TTFT={r['TTFT_s']}s  comp={r['completion_tokens']}")
        out.append(r["decode_tok_s"])
    return out

def long_hit(target: int = 24000):
    print(f"\n=== HIT: {target}-token prefill+decode ===")
    r = bench_one(long_prompt(target), max_tokens=200)
    print(f"  decode={r['decode_tok_s']} tok/s  TTFT={r['TTFT_s']}s  comp={r['completion_tokens']}")

def main():
    print(f"server={BASE} model={MODEL}")
    A = phase("Phase A: baseline", 3)
    long_hit(24000)
    C = phase("Phase C: after 1st 24k hit", 3)
    long_hit(24000)
    E = phase("Phase E: after 2nd 24k hit", 3)

    def avg(xs): return sum(x for x in xs if x) / max(1, len([x for x in xs if x]))
    a, c, e = avg(A), avg(C), avg(E)
    print("\n=== TRAJECTORY ===")
    print(f"  baseline avg:           {a:.1f} tok/s")
    print(f"  after 1st 24k hit avg:  {c:.1f} tok/s  ({(c-a)/a*100:+.1f}% vs baseline)")
    print(f"  after 2nd 24k hit avg:  {e:.1f} tok/s  ({(e-a)/a*100:+.1f}% vs baseline)")
    drop1 = (a - c) / a * 100
    drop2 = (a - e) / a * 100
    print()
    if drop1 > 15 or drop2 > 15:
        print(f"  RESULT: REGRESSION REPRODUCES (drop1={drop1:.1f}%, drop2={drop2:.1f}%)")
    else:
        print(f"  RESULT: no regression (drop1={drop1:.1f}%, drop2={drop2:.1f}%)")

if __name__ == "__main__":
    main()
