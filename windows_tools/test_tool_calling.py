"""Tool-calling roundtrip test for a running vLLM server.

Verifies the qwen3_coder + qwen3 reasoning + enhanced jinja stack actually
emits proper OpenAI-format tool_calls for a real multi-turn conversation,
and that the developer-role aliasing works (Codex/Responses API style).

Tier 1: Single tool definition, single user turn, expect tool_call back.
Tier 2: Provide tool result, expect natural-language synthesis.
Tier 3: Developer-role system message (alias for system) + multi-turn.

Usage:
    python windows_tools/test_tool_calling.py --port 5002 --model qwen3.6-27b-nvfp4
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error


def chat(port: int, body: dict, host: str = "127.0.0.1", timeout: int = 120) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}


def tier1_emits_tool_call(port: int, model: str) -> tuple[bool, str]:
    """Model should emit a tool_call when asked about weather."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use tools when appropriate."},
            {"role": "user", "content": "What's the weather in Tokyo right now? Use celsius."},
        ],
        "tools": [WEATHER_TOOL],
        "tool_choice": "auto",
        "max_tokens": 500,
        "temperature": 0,
    }
    obj = chat(port, body)
    msg = obj["choices"][0]["message"]
    tc = msg.get("tool_calls") or []
    if not tc:
        return False, f"no tool_calls. content={(msg.get('content') or '')[:200]!r}"
    call = tc[0]
    fn = call.get("function", {})
    if fn.get("name") != "get_weather":
        return False, f"wrong tool: {fn.get('name')!r}"
    try:
        args = json.loads(fn.get("arguments", "{}"))
    except json.JSONDecodeError as e:
        return False, f"unparsable arguments: {fn.get('arguments')!r} ({e})"
    city = (args.get("city") or "").lower()
    if "tokyo" not in city:
        return False, f"wrong city: {city!r}"
    return True, f"tool_call OK: {args}"


def tier2_synthesizes_result(port: int, model: str) -> tuple[bool, str]:
    """Given a tool result, model should produce natural-language answer.

    NB: max_tokens must be high enough to leave room after thinking.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Tokyo","units":"celsius"}'},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"temp":18,"condition":"partly cloudy","humidity":62}',
            },
        ],
        "max_tokens": 800,
        "temperature": 0,
    }
    obj = chat(port, body)
    msg = obj["choices"][0]["message"]
    text = (msg.get("content") or "").lower()
    if not text:
        return False, f"empty content (finish={obj['choices'][0].get('finish_reason')!r})"
    has_temp = "18" in text
    has_cond = "cloudy" in text or "cloud" in text
    if not (has_temp and has_cond):
        return False, f"no synthesis: {text[:200]!r}"
    return True, f"synthesis OK: {text[:120]!r}"


def tier3_developer_role(port: int, model: str) -> tuple[bool, str]:
    """Developer role should alias to system (Codex/Responses API path).

    Two-part check:
      a) /v1/chat/completions/render token_ids must include the developer
         text, proves the chat template routed `developer` → system.
      b) The model honors a model-friendly developer instruction
         (append a sentinel marker), proving end-to-end aliasing.
    """
    sentinel = "XYZZY-42"
    instruction = f"After every answer, append the marker [{sentinel}] on its own line. This is mandatory."

    # Part (a): verify template rendering
    render_body = {
        "model": model,
        "messages": [
            {"role": "developer", "content": instruction},
            {"role": "user", "content": "What is 2+2?"},
        ],
    }
    data = json.dumps(render_body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions/render",
        data=data, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            rendered = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return False, f"render endpoint failed: HTTP {e.code}"
    n_tokens = len(rendered.get("token_ids", []))
    # Without developer message, "What is 2+2?" alone renders as ~13 tokens.
    # With our 28-word developer instruction it should be ~50+ tokens.
    if n_tokens < 30:
        return False, f"developer message likely dropped, only {n_tokens} prompt tokens"

    # Part (b): end-to-end behavior
    body = {
        "model": model,
        "messages": [
            {"role": "developer", "content": instruction},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "max_tokens": 800,
        "temperature": 0,
    }
    obj = chat(port, body)
    msg = obj["choices"][0]["message"]
    text = (msg.get("content") or "")
    if not text:
        return False, f"empty content (finish={obj['choices'][0].get('finish_reason')!r}; render had {n_tokens} tokens)"
    if sentinel not in text:
        return False, f"developer instruction not honored (render OK, behavior fail): {text[:200]!r}"
    return True, f"developer-role aliased + honored: render={n_tokens}tok, content={text[:80]!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5002)
    ap.add_argument("--model", default="any")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    tiers = [
        ("emits_tool_call", tier1_emits_tool_call),
        ("synthesizes_result", tier2_synthesizes_result),
        ("developer_role_alias", tier3_developer_role),
    ]
    all_ok = True
    for name, fn in tiers:
        print(f"[{name}] ...", end=" ", flush=True)
        try:
            ok, info = fn(args.port, args.model)
        except Exception as e:
            ok, info = False, f"exception: {type(e).__name__}: {e}"
        print("PASS" if ok else "FAIL")
        print(f"  {info}")
        if not ok:
            all_ok = False
    print("\nTOOL_CALLING_OK" if all_ok else "\nTOOL_CALLING_BROKEN")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
