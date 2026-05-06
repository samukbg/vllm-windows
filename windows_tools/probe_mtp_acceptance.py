"""Probe MTP draft acceptance rate at varying context sizes.

vLLM exposes per-engine spec-decode metrics on /metrics (Prometheus
format). The relevant gauges are:

    vllm:spec_decode_num_drafts_total
    vllm:spec_decode_num_draft_tokens_total
    vllm:spec_decode_num_accepted_tokens_total

Acceptance rate = accepted / drafted.

This script:
  1. Snapshots /metrics
  2. Sends a chat request with a target context size
  3. Snapshots /metrics again
  4. Computes the per-request acceptance rate

For long-ctx degradation: if MTP-head acceptance falls below ~50% at
long ctx, that's a signal the weights have drifted enough that the
draft head can no longer agree with the main head. AutoRound INT4
typically holds 65%+ acceptance at 200k (per Otherwise-Director17 on
Reddit thread 1t46klu).

Usage:
    python windows_tools/probe_mtp_acceptance.py --port 5002 \
        --model qwen3.6-27b-nvfp4 --label nvfp4 --sizes 50000,150000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request

METRIC_KEYS = [
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
]


def fetch_metrics(port: int, host: str = "127.0.0.1") -> dict[str, float]:
    with urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout=10) as r:
        body = r.read().decode("utf-8")
    out: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # Format: metric_name{labels} value [timestamp]
        m = re.match(r"([\w:]+)(\{[^}]*\})?\s+([0-9eE.+-]+)", line)
        if not m:
            continue
        name = m.group(1)
        if name in METRIC_KEYS:
            # Sum across label sets (one engine here, but be safe)
            out[name] = out.get(name, 0.0) + float(m.group(3))
    return out


def chat(port: int, model: str, prompt: str, max_tokens: int = 200, host: str = "127.0.0.1") -> dict:
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
    with urllib.request.urlopen(req, timeout=900) as r:
        obj = json.loads(r.read().decode("utf-8"))
    dt = time.time() - t0
    usage = obj.get("usage") or {}
    msg = obj["choices"][0]["message"]
    text = (msg.get("content") or msg.get("reasoning") or "")
    return {
        "wall_s": round(dt, 2),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "preview": text[:200],
    }


FILLER = (
    "The early history of cartography is closely tied to the development of mathematics, "
    "astronomy, and trade. Greek philosophers debated whether the earth was flat or "
    "spherical for centuries before Eratosthenes calculated its circumference with "
    "impressive accuracy around 240 BCE. Later, Ptolemy's Geographia synthesized the "
    "geographic knowledge of the classical world into a coordinate system using lines "
    "of latitude and longitude, an approach that would persist in modified forms for "
    "more than a thousand years. "
)


def build_prompt(target_tokens: int) -> str:
    chunks = target_tokens // 100
    return (FILLER * chunks) + "\n\n---\nSummarize the above in three bullet points:"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sizes", default="50000,150000")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    results = []
    for sz in sizes:
        prompt = build_prompt(sz)
        # Warmup small request to ensure metrics path is alive
        before = fetch_metrics(args.port, args.host)
        if not before:
            print("ERROR: /metrics returned no spec_decode keys — is MTP enabled?")
            return 1
        info = chat(args.port, args.model, prompt, max_tokens=200, host=args.host)
        after = fetch_metrics(args.port, args.host)

        d_drafts = after.get("vllm:spec_decode_num_drafts_total", 0) - before.get("vllm:spec_decode_num_drafts_total", 0)
        d_drafted = after.get("vllm:spec_decode_num_draft_tokens_total", 0) - before.get("vllm:spec_decode_num_draft_tokens_total", 0)
        d_accepted = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - before.get("vllm:spec_decode_num_accepted_tokens_total", 0)
        accept_rate = (d_accepted / d_drafted) if d_drafted > 0 else 0.0
        avg_per_draft = (d_accepted / d_drafts) if d_drafts > 0 else 0.0

        print(f"\n=== ctx≈{sz} ===")
        print(f"  prompt_tokens   : {info['prompt_tokens']}")
        print(f"  completion_toks : {info['completion_tokens']}")
        print(f"  wall_s          : {info['wall_s']}")
        print(f"  drafts          : {int(d_drafts)}")
        print(f"  drafted tokens  : {int(d_drafted)}")
        print(f"  accepted tokens : {int(d_accepted)}")
        print(f"  acceptance rate : {accept_rate*100:.1f}%")
        print(f"  avg accepted/draft : {avg_per_draft:.2f}")
        print(f"  preview         : {info['preview'][:150]!r}")
        results.append({
            "size": sz, "info": info,
            "drafts": int(d_drafts), "drafted": int(d_drafted),
            "accepted": int(d_accepted), "accept_rate": round(accept_rate, 4),
            "avg_per_draft": round(avg_per_draft, 3),
        })

    out_path = f"mtp_accept_{args.label}.json"
    with open(out_path, "w") as f:
        json.dump({"label": args.label, "model": args.model,
                   "results": results}, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
