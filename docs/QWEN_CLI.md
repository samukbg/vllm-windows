# Using Qwen Code (qwen-code) with this server

[Qwen Code](https://github.com/QwenLM/qwen-code) is Alibaba's
official agentic CLI for Qwen models, forked from `gemini-cli` and
adapted for Qwen3 thinking, tool calling, and the OpenAI-compatible
API. It talks to this server's `/v1/chat/completions` endpoint
directly. No proxy needed.

If you don't specifically want Qwen Code, the easier path is one of
these clients, all of which work with this server out of the box:

- Claude Code: see [`CLAUDE_CODE.md`](CLAUDE_CODE.md). Native
  `/v1/messages`, no setup beyond a base URL.
- OpenCode: see [`OPENCODE.md`](OPENCODE.md). Same OpenAI-compat
  endpoint as Qwen Code, with a polished agent UX.
- Cline, Cursor, Continue, KiloCode: any string that ends in "code"
  works against `/v1` with a base URL and an empty key.
- Codex CLI: [`CODEX.md`](CODEX.md). Slightly fiddlier (Responses API).

The rest of this page is for users who specifically want Qwen Code.

## Step 1: install Qwen Code

The official install is via npm:

```bash
npm install -g @qwen-code/qwen-code@latest
```

Or grab the latest binary release from the
[Qwen Code releases page](https://github.com/QwenLM/qwen-code/releases).

If you don't have Node, install Node 20+ first
([nodejs.org](https://nodejs.org)). Qwen Code runs on Windows, macOS,
and Linux; the server only needs to be reachable on
`127.0.0.1:5001` (or whichever port your snapshot uses), so Qwen Code
can run on the same machine or a different one that can reach the
server's port.

## Step 2: point Qwen Code at this server

Qwen Code reads the standard OpenAI environment variables. Set them
before launching:

```powershell
# PowerShell
$env:OPENAI_API_KEY  = "qwen-local"   # any non-empty string works
$env:OPENAI_BASE_URL = "http://127.0.0.1:5001/v1"
$env:OPENAI_MODEL    = "qwen3.6-27b-autoround"
```

```cmd
:: cmd.exe
set OPENAI_API_KEY=qwen-local
set OPENAI_BASE_URL=http://127.0.0.1:5001/v1
set OPENAI_MODEL=qwen3.6-27b-autoround
```

```bash
# bash / git-bash / WSL
export OPENAI_API_KEY=qwen-local
export OPENAI_BASE_URL=http://127.0.0.1:5001/v1
export OPENAI_MODEL=qwen3.6-27b-autoround
```

`OPENAI_MODEL` matches the `--served-model-name` set by every shipped
snapshot. The wheel's served-model-name is also a wildcard, so
literally any string works in `OPENAI_MODEL`. Match the snapshot's
declared name if you want to be explicit.

To make the env vars persistent on Windows, set them via System
Settings → Environment Variables, or drop them into a per-project
`.env` file Qwen Code can read.

## Step 3: launch a server snapshot, then Qwen Code

From this launcher, pick a snapshot. `start_speed` (90 k ctx) and
`start_127k` (127 k ctx) are good defaults for code work. Wait until
the log shows `Application startup complete.`.

Then in any project directory:

```bash
qwen
```

Ask it to read or modify a file. The first request hits
`/v1/chat/completions`. If you see a normal response and tool calls
work, you're done.

## Sampler defaults

Qwen Code does not override sampler params unless you ask it to. The
shipped snapshots use Unsloth's recommended Qwen3 sampling for thinking
mode (temperature 0.6, top_p 0.95, top_k 20, min_p 0.0). For coding-
specific sampler defaults baked into the snapshot:

- Use `start_thinking_coding` for thinking-mode coding (precise debug
  / architecture work).
- Use `start_instruct_coding` for non-thinking coding (faster, no
  `<think>` block).

## Reasoning / thinking output

The shipped chat template defaults to thinking ON for snapshots that
don't set `chat_template_kwargs.enable_thinking=false`. Qwen Code
displays the thinking content separately from the final answer; the
`reasoning` field comes back populated, the `content` field has the
post-thinking response.

If `content` comes back empty, the thinking block ate your
`max_tokens`. Set a higher budget. Qwen3.6 thinking can run 200-2000
tokens before answering; 4096+ is safe for short Q&A, 8000+ for
non-trivial reasoning.

To force non-thinking on a per-request basis, append `/no_think` to
the user prompt or use the `start_instruct_*` snapshots which
disable thinking via `chat_template_kwargs.enable_thinking=false` in
the snapshot args.

## Tool calling

Every snapshot ships the tool-calling fix baked in:

- vLLM PR [#35687](https://github.com/vllm-project/vllm/pull/35687):
  treats `<tool_call>` as an implicit `</think>`.
- vLLM PR [#40861](https://github.com/vllm-project/vllm/pull/40861):
  streaming-path fixes for split tags, dropped parameters,
  multi-call drops under speculative decoding, and structural
  delimiters appearing as literal text inside parameter values.
- `qwen3.5-enhanced.jinja` chat template under `templates\`.
- `--tool-call-parser=qwen3_coder` and `--reasoning-parser=qwen3`.

So Qwen Code's read-file / edit / shell tool calls work without
per-snapshot tweaking.

## Windows path-handling rule

If Qwen Code emits backslash paths inside tool-call JSON arguments
without escaping (`C:\Users\...` instead of `C:\\Users\\...`), the JSON
parse fails and the tool call drops. The cheap fix is a one-liner in
your project's `QWEN.md` (or any system-prompt file Qwen Code reads):

> "I am on a Windows system, so properly escape directory backslashes
> to keep from breaking JSON."

This is a model-side issue, not a Qwen Code or server bug, and it
also reproduces on Linux vLLM and llama.cpp.

## Verifying it works

1. Server up: visit `http://127.0.0.1:5001/v1/models` in a browser.
   You should see a JSON `data` array.
2. Qwen Code reaches the server: `qwen` should not hang on first
   request. If it does, check the env vars and the base URL.
3. Reasoning is on: ask a non-trivial question. You should see a
   thinking block (Qwen Code renders it inline by default).
4. Tools work: ask Qwen Code to read a file. The tool call should
   round-trip cleanly. If it fails with a JSON parse error, see
   "Windows path-handling" above.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `404 Not Found` on `/v1/chat/completions` | Wrong port (or `OPENAI_BASE_URL` includes the wrong path). Confirm the server is on the port you think it is. |
| `qwen` hangs on first request | Check `nvidia-smi` to confirm vLLM hasn't OOM'd; tail `logs\vllm_server.5001.log` for the real error. |
| Empty `content`, `finish_reason=length` | `max_tokens` ate the thinking phase. Raise to 8000+, or use an `instruct_*` snapshot. |
| Tool call returned but the file path wasn't found | Path-escape issue. Add the `QWEN.md` rule above. |
| `Unexpected message role.` | Qwen Code is sending a role the chat template doesn't handle. The shipped template aliases `developer` to `system` since v1.0.1. If you're on v1.0 or older, see [`CODEX.md`](CODEX.md) for the Option A four-line patch. |

## Why this is documented separately from OpenCode

Qwen Code's UX, default model selection, and tool catalog are
specific to Alibaba's Qwen-first agent flow (it knows about Qwen
thinking, defaults to Qwen tool conventions, and ships Qwen-specific
prompts). Configuration on the server side is identical to OpenCode
because both speak `/v1/chat/completions`. If you bounce between
OpenCode and Qwen Code, the same env vars work for both.

## Related

- [`OPENCODE.md`](OPENCODE.md), the OpenCode setup, mostly identical
  to this one.
- [`CLAUDE_CODE.md`](CLAUDE_CODE.md), the easiest integration overall.
- [`COHERENCE.md`](COHERENCE.md), the validator to run if Qwen Code
  sees garbage output (almost always a server-side problem, not the
  client).
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md), every failure mode
  I've hit on the server side.
