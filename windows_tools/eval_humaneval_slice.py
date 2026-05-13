"""Quality eval: small HumanEval-style coding slice for side-by-side
comparison of two vLLM endpoints (e.g. AutoRound vs NVFP4).

Each problem ships with a function signature, docstring, and a small
test suite that gets exec'd against the model's completion. Pass = all
asserts hold; fail = exception or wrong output.

Runs the same N problems against two ports/models, records pass@1 for
each, prints per-problem pass/fail and an overall delta.

This is NOT the full HumanEval (164 problems); it's a 12-problem slice
chosen for: clearly-defined tests, no string-edit-distance grading,
runs fast (<2 min per side at 92 tok/s), exercises a mix of arithmetic,
control flow, dict/list manipulation, and string handling.

Usage:
    # baseline only
    python windows_tools/eval_humaneval_slice.py \
        --port 5001 --model qwen3.6-27b-autoround --label baseline

    # nvfp4 only (separate run, save JSON)
    python windows_tools/eval_humaneval_slice.py \
        --port 5002 --model qwen3.6-27b-nvfp4 --label nvfp4

Each run writes ``humaneval_<label>.json`` for cross-comparison.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import time
import traceback
import urllib.request
from typing import Any

# 12 self-contained coding problems with embedded tests.
# Each test_fn takes the candidate function and returns (ok, info).
PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "is_palindrome",
        "prompt": (
            "Write a Python function `is_palindrome(s: str) -> bool` that returns True "
            "if `s` reads the same forwards and backwards (case-insensitive, ignoring "
            "non-alphanumeric chars). Return only the function definition, no commentary."
        ),
        "tests": lambda f: all([
            f("racecar") is True,
            f("A man, a plan, a canal: Panama") is True,
            f("hello") is False,
            f("") is True,
            f("Was it a car or a cat I saw?") is True,
        ]),
    },
    {
        "id": "fizzbuzz",
        "prompt": (
            "Write `fizzbuzz(n: int) -> list[str]` returning a list of length n where "
            "index i (0-based) is 'FizzBuzz' if (i+1) divisible by 15, 'Fizz' if 3, "
            "'Buzz' if 5, else str(i+1). Return only the function definition."
        ),
        "tests": lambda f: f(15) == ['1','2','Fizz','4','Buzz','Fizz','7','8','Fizz','Buzz','11','Fizz','13','14','FizzBuzz'],
    },
    {
        "id": "flatten",
        "prompt": (
            "Write `flatten(xs: list) -> list` that flattens an arbitrarily-nested "
            "list of ints into a flat list, preserving order. Return only the function definition."
        ),
        "tests": lambda f: all([
            f([1, [2, [3, 4], 5], 6]) == [1, 2, 3, 4, 5, 6],
            f([]) == [],
            f([[[1]]]) == [1],
            f([1, 2, 3]) == [1, 2, 3],
        ]),
    },
    {
        "id": "two_sum",
        "prompt": (
            "Write `two_sum(nums: list[int], target: int) -> tuple[int, int] | None` "
            "that returns the indices (i, j) with i < j of two distinct numbers in nums "
            "that sum to target. Return None if no such pair exists. Return only the function."
        ),
        "tests": lambda f: all([
            f([2, 7, 11, 15], 9) == (0, 1),
            f([3, 2, 4], 6) == (1, 2),
            f([1, 2, 3], 7) is None,
        ]),
    },
    {
        "id": "word_count",
        "prompt": (
            "Write `word_count(s: str) -> dict[str, int]` that counts case-insensitive "
            "words in s, splitting on whitespace and stripping punctuation. Return only the function."
        ),
        "tests": lambda f: f("The quick brown fox. The lazy dog!") == {"the": 2, "quick": 1, "brown": 1, "fox": 1, "lazy": 1, "dog": 1},
    },
    {
        "id": "factorial_iter",
        "prompt": (
            "Write `factorial(n: int) -> int` that returns n! iteratively. Raise "
            "ValueError for negative input. Return only the function definition."
        ),
        "tests": lambda f: all([f(0) == 1, f(1) == 1, f(5) == 120, f(10) == 3628800]) and _raises(f, ValueError, -1),
    },
    {
        "id": "rle_encode",
        "prompt": (
            "Write `rle_encode(s: str) -> list[tuple[str, int]]` that returns the "
            "run-length encoding of s. Empty string → empty list. Return only the function."
        ),
        "tests": lambda f: all([
            f("aaabbc") == [("a", 3), ("b", 2), ("c", 1)],
            f("") == [],
            f("a") == [("a", 1)],
        ]),
    },
    {
        "id": "is_prime",
        "prompt": (
            "Write `is_prime(n: int) -> bool` that returns True iff n is prime. "
            "0 and 1 are not prime; negatives are not prime. Return only the function."
        ),
        "tests": lambda f: all([
            f(2), f(3), f(5), f(7), f(11), f(97),
            not f(0), not f(1), not f(4), not f(9), not f(15), not f(-7),
        ]),
    },
    {
        "id": "merge_sorted",
        "prompt": (
            "Write `merge_sorted(a: list[int], b: list[int]) -> list[int]` that merges "
            "two sorted (ascending) lists into one sorted list. Return only the function."
        ),
        "tests": lambda f: all([
            f([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6],
            f([], [1, 2]) == [1, 2],
            f([1, 2], []) == [1, 2],
            f([], []) == [],
        ]),
    },
    {
        "id": "reverse_words",
        "prompt": (
            "Write `reverse_words(s: str) -> str` that reverses the order of words "
            "(separated by single spaces) but keeps each word itself unreversed. "
            "Return only the function."
        ),
        "tests": lambda f: all([
            f("hello world foo") == "foo world hello",
            f("a b c") == "c b a",
            f("single") == "single",
            f("") == "",
        ]),
    },
    {
        "id": "gcd",
        "prompt": (
            "Write `gcd(a: int, b: int) -> int` returning the greatest common divisor "
            "of two non-negative integers. gcd(0, 0) = 0. Return only the function."
        ),
        "tests": lambda f: all([f(12, 18) == 6, f(7, 13) == 1, f(100, 75) == 25, f(0, 5) == 5, f(0, 0) == 0]),
    },
    {
        "id": "balanced_parens",
        "prompt": (
            "Write `balanced(s: str) -> bool` that returns True iff brackets in s "
            "are balanced. Brackets: () [] {}. Other chars ignored. Return only the function."
        ),
        "tests": lambda f: all([
            f("(){}[]") is True,
            f("([{}])") is True,
            f("(]") is False,
            f("((") is False,
            f("") is True,
            f("foo(bar[baz]){}") is True,
        ]),
    },
]


def _raises(fn, exc_type, *args) -> bool:
    try:
        fn(*args)
        return False
    except exc_type:
        return True
    except Exception:
        return False


def chat(port: int, model: str, prompt: str, host: str = "127.0.0.1") -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise coding assistant. Output only Python code, no markdown fences, no commentary."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 3000, # generous: thinking eats budget, then code
        "temperature": 0,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=data, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        obj = json.loads(r.read().decode("utf-8"))
    msg = obj["choices"][0]["message"]
    # Prefer content (the post-thinking answer); fall back to reasoning only
    # if content is empty (max_tokens exhausted by thinking).
    content = msg.get("content") or ""
    if content.strip():
        return content
    return msg.get("reasoning") or ""


# Strip markdown fences and any prose before the function.
FENCE_RE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)
DEF_RE = re.compile(r"^(?:from .+\n|import .+\n)*\s*def \w+\(", re.MULTILINE)


def extract_code(text: str) -> str:
    # Try fenced first
    m = FENCE_RE.search(text)
    if m:
        block = m.group(1)
    else:
        # Else find first def and take from there
        m = DEF_RE.search(text)
        if m:
            block = text[m.start():]
        else:
            block = text
    # The model sometimes emits code inside an indented "drafting" block in
    # its reasoning. Dedent so the top-level def/import starts at column 0.
    return textwrap.dedent(block).strip()


def grade(code: str, fn_name: str, test_fn) -> tuple[bool, str]:
    ns: dict[str, Any] = {}
    try:
        exec(code, ns)
    except Exception as e:
        return False, f"compile/exec error: {type(e).__name__}: {e}"
    if fn_name not in ns:
        # Maybe the model used a different name; try to find any def
        defs = [k for k, v in ns.items() if callable(v) and not k.startswith("_")]
        if not defs:
            return False, "no callable produced"
        fn = ns[defs[0]]  # pick first
    else:
        fn = ns[fn_name]
    try:
        ok = test_fn(fn)
    except Exception as e:
        return False, f"test exception: {type(e).__name__}: {e}"
    return bool(ok), "OK" if ok else "asserts failed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="output filename suffix")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    results = []
    pass_count = 0
    t0 = time.time()
    for i, p in enumerate(PROBLEMS):
        t_start = time.time()
        try:
            raw = chat(args.port, args.model, p["prompt"], args.host)
        except Exception as e:
            print(f"[{i+1:2d}/{len(PROBLEMS)}] {p['id']:20s} HTTP_ERROR: {e}")
            results.append({"id": p["id"], "ok": False, "info": str(e), "wall_s": 0, "code": ""})
            continue
        wall_s = time.time() - t_start
        code = extract_code(raw)
        ok, info = grade(code, p["id"], p["tests"])
        if ok:
            pass_count += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{i+1:2d}/{len(PROBLEMS)}] {p['id']:20s} {status:5s} ({wall_s:5.1f}s)  {info}")
        results.append({
            "id": p["id"], "ok": ok, "info": info,
            "wall_s": round(wall_s, 2), "code": code,
        })

    total_s = time.time() - t0
    pass_pct = 100 * pass_count / len(PROBLEMS)
    print(f"\n=== {args.label} pass@1 = {pass_count}/{len(PROBLEMS)} ({pass_pct:.1f}%), total {total_s:.0f}s ===")

    out_path = f"humaneval_{args.label}.json"
    with open(out_path, "w") as f:
        json.dump({
            "label": args.label, "port": args.port, "model": args.model,
            "pass_count": pass_count, "total": len(PROBLEMS),
            "pass_pct": pass_pct, "total_s": round(total_s, 1),
            "results": results,
        }, f, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
