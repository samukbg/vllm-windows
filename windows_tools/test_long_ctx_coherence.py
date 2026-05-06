"""Long-context coherence + needle-in-haystack test.

Three context sizes: 50k, 100k, 180k. For each, builds a long prompt by
repeating filler (a public-domain text), injects 1-2 NEEDLES at known
positions (an arbitrary fact), and asks a question about each needle.

A coherent server should:
  1. Not produce degenerate output (the standard check_coherence guards)
  2. Successfully retrieve the needle facts (NIAH-style)

NVFP4 without QAT can drift at long ctx: weight quantization errors
accumulate in the long key/value caches, causing degeneration or
needle-miss. This test catches that.

Usage:
    python windows_tools/test_long_ctx_coherence.py --port 5002 \
        --model qwen3.6-27b-nvfp4 --sizes 50000,100000,180000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request

DEGENERATE_PATTERNS = [
    (r"(\*\s*){5,}",                "asterisk attractor"),
    (r"(\bthe\s+){5,}",             "the-the loop"),
    (r"(\*\*:){3,}",                "delim loop"),
    (r"\n{6,}",                     "newline loop"),
    (r"(\b\w+\b\s+)\1{4,}",         "single-token loop"),
]

# Filler: a small, neutral, repeatable paragraph. ~120 tokens per chunk.
FILLER = (
    "The early history of cartography is closely tied to the development of mathematics, "
    "astronomy, and trade. Greek philosophers debated whether the earth was flat or "
    "spherical for centuries before Eratosthenes calculated its circumference with "
    "impressive accuracy around 240 BCE. Later, Ptolemy's Geographia synthesized the "
    "geographic knowledge of the classical world into a coordinate system using lines "
    "of latitude and longitude, an approach that would persist in modified forms for "
    "more than a thousand years. Medieval European mapmakers, by contrast, often "
    "produced T-O maps that prioritized theological symbolism over geographic accuracy, "
    "while Islamic and Chinese cartographers maintained more empirical traditions. "
)

NEEDLES = [
    {"fact": "The blue penguin's name is Quentin and he was born on a Tuesday in 1987.",
     "question": "What is the blue penguin's name and when was he born?",
     "must_contain": ["quentin", "1987"]},
    {"fact": "The secret password for the Ravenfield laboratory is THISTLEDOWN-93-ALPHA.",
     "question": "What is the secret password for the Ravenfield laboratory?",
     "must_contain": ["thistledown-93-alpha"]},
]


def chat(port: int, model: str, prompt: str, max_tokens: int = 400, host: str = "127.0.0.1") -> tuple[str, dict]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=data, headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        obj = json.loads(r.read().decode("utf-8"))
    dt = time.time() - t0
    msg = obj["choices"][0]["message"]
    text = (msg.get("content") or "")
    if not text:
        # Fall back to reasoning if model used all budget on it
        text = msg.get("reasoning") or ""
    usage = obj.get("usage") or {}
    return text, {"wall_s": round(dt, 2), "prompt_tokens": usage.get("prompt_tokens", 0),
                  "completion_tokens": usage.get("completion_tokens", 0),
                  "finish": obj["choices"][0].get("finish_reason")}


def build_prompt(target_tokens: int) -> tuple[str, list[dict]]:
    """Build a ~target_tokens prompt with two needles at ~25% and ~75%."""
    # Empirically: each FILLER chunk tokenizes to ~128 tokens for this model.
    # Use 130 to leave a small headroom margin so we don't blow ctx.
    chunks_needed = max(1, target_tokens // 130)
    buf = []
    needle_a_pos = chunks_needed // 4
    needle_b_pos = (3 * chunks_needed) // 4
    for i in range(chunks_needed):
        buf.append(FILLER)
        if i == needle_a_pos:
            buf.append(f"\n\n[IMPORTANT FACT TO REMEMBER]: {NEEDLES[0]['fact']}\n\n")
        if i == needle_b_pos:
            buf.append(f"\n\n[IMPORTANT FACT TO REMEMBER]: {NEEDLES[1]['fact']}\n\n")
    body = "".join(buf)
    questions = (
        "\n\n---\nNow, based on the text above, answer these two questions concisely:\n"
        f"1. {NEEDLES[0]['question']}\n"
        f"2. {NEEDLES[1]['question']}\n"
    )
    return body + questions, NEEDLES


def grade_response(text: str, needles: list[dict]) -> dict:
    issues = []
    if not text.strip():
        issues.append("empty response")
    for pat, name in DEGENERATE_PATTERNS:
        if re.search(pat, text):
            issues.append(name)
    text_lower = text.lower()
    needle_results = []
    for i, n in enumerate(needles):
        hits = [s for s in n["must_contain"] if s in text_lower]
        ok = len(hits) == len(n["must_contain"])
        needle_results.append({"needle": i, "ok": ok, "hits": hits, "want": n["must_contain"]})
        if not ok:
            issues.append(f"needle{i} miss")
    return {"degenerate_or_miss": bool(issues), "issues": issues, "needles": needle_results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sizes", default="50000,100000,180000")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    all_results = []
    overall_ok = True
    for sz in sizes:
        prompt, needles = build_prompt(sz)
        print(f"\n=== ctx~{sz} (built prompt {len(prompt)} chars) ===")
        try:
            text, info = chat(args.port, args.model, prompt, max_tokens=600, host=args.host)
        except Exception as e:
            print(f"  HTTP_ERROR: {type(e).__name__}: {e}")
            all_results.append({"size": sz, "ok": False, "error": str(e)})
            overall_ok = False
            continue
        grade = grade_response(text, needles)
        ok = not grade["degenerate_or_miss"]
        print(f"  prompt_tokens={info['prompt_tokens']} wall_s={info['wall_s']} finish={info['finish']}")
        print(f"  preview: {text[:200]!r}")
        for nr in grade["needles"]:
            print(f"  needle{nr['needle']}: {'HIT' if nr['ok'] else 'MISS'} hits={nr['hits']} want={nr['want']}")
        if grade["issues"]:
            print(f"  issues: {grade['issues']}")
        print(f"  result: {'PASS' if ok else 'FAIL'}")
        all_results.append({
            "size": sz, "ok": ok, "info": info, "grade": grade,
            "preview": text[:300],
        })
        if not ok:
            overall_ok = False

    out_path = f"long_ctx_{args.label}.json"
    with open(out_path, "w") as f:
        json.dump({"label": args.label, "model": args.model,
                   "results": all_results, "overall_ok": overall_ok}, f, indent=2)
    print(f"\nwrote {out_path}")
    print("\nLONG_CTX_OK" if overall_ok else "\nLONG_CTX_DEGRADED")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
