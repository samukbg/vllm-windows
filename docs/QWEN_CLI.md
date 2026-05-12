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
  endpoint as Qwen Code, polished agent UX.
- Cline, Cursor, Continue, KiloCode: any string that ends in "code"
  works against `/v1` with a base URL and an empty key.
- Codex CLI: [`CODEX.md`](CODEX.md). Slightly fiddlier (Responses API).

The rest of this page is for users who specifically want Qwen Code.

## Step 1: install Qwen Code

The npm package is `@qwen-code/qwen-code`. Node 20+ required.

```powershell
npm install -g @qwen-code/qwen-code@latest
qwen --version
```

Latest as of writing is v0.15.9 (May 2026). The release cadence is
fast (multiple releases per week, plus nightlies and previews); if
something behaves oddly, bump to the latest stable before debugging.

If you don't have Node, install Node 20+ first
([nodejs.org](https://nodejs.org)). Or grab a binary from the
[releases page](https://github.com/QwenLM/qwen-code/releases) and
skip npm.

## Step 2: point Qwen Code at this server

The cleanest path is a `~/.qwen/.env` file. Qwen Code reads it on
every start and it keeps OpenAI keys out of your global shell env.

Create `%USERPROFILE%\.qwen\.env` (Windows) or `~/.qwen/.env`
(macOS/Linux):

```ini
OPENAI_API_KEY=qwen-local
OPENAI_BASE_URL=http://127.0.0.1:5001/v1
OPENAI_MODEL=any
```

Search order Qwen Code uses for `.env` files (stops at first match):

1. `.qwen/.env` in the current directory (walking up toward root)
2. `.env` in the current directory (walking up)
3. `~/.qwen/.env`
4. `~/.env`

So a project-local `.qwen/.env` overrides the user-global one. That's
useful if one project should hit `start_speed` on 5001 and another
should hit `start_pp2_160k` on 5002.

If you'd rather use shell exports (also fine):

```powershell
# PowerShell
$env:OPENAI_API_KEY  = "qwen-local"
$env:OPENAI_BASE_URL = "http://127.0.0.1:5001/v1"
$env:OPENAI_MODEL    = "any"
```

```cmd
:: cmd.exe
set OPENAI_API_KEY=qwen-local
set OPENAI_BASE_URL=http://127.0.0.1:5001/v1
set OPENAI_MODEL=any
```

```bash
# bash / git-bash / WSL
export OPENAI_API_KEY=qwen-local
export OPENAI_BASE_URL=http://127.0.0.1:5001/v1
export OPENAI_MODEL=any
```

The patched wheel uses a wildcard served-model-name, so `OPENAI_MODEL`
can be literally `any`. If you'd rather match the snapshot exactly,
use `qwen3.6-27b-autoround` (every Lorbus AutoRound snapshot) or
`qwen3.6-27b-nvfp4` (the Blackwell NVFP4 snapshot).

`OPENAI_API_KEY` must be non-empty. vLLM doesn't validate it.

## Step 3: pick auth mode

Qwen Code supports three auth methods. Inside a running session,
type `/auth` to switch between them; from outside, run `qwen auth`.

- **Qwen OAuth**: browser login on `qwen.ai`. The free tier was
  retired April 2026; this is now a paid path.
- **Alibaba Cloud Coding Plan**: paid subscription, higher quotas.
- **OpenAI-Compatible API Key**: what we want. Picks up
  `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` from env or
  `.env` and routes everything through `/v1/chat/completions`.

If you set the env vars from Step 2, the OpenAI option should already
be selected the first time you launch. If not, run `/auth` and pick
"OpenAI" from the menu.

To pin the choice non-interactively (useful for CI or just to skip
the menu on first run), drop a `~/.qwen/settings.json`:

```json
{
  "security": {
    "auth": {
      "selectedType": "openai"
    }
  }
}
```

Qwen Code reads settings from, in priority order:

1. `.qwen/settings.json` in project root (per-repo override)
2. `~/.qwen/settings.json` (user-global)
3. System defaults (Linux only)

## Step 4: launch a server snapshot, then Qwen Code

From this launcher, pick a snapshot. On the Ampere/Ada zip,
`start_speed` (90 k ctx) and `start_127k` (127 k ctx) are good
defaults for code work. On the Blackwell zip, `rtx5090_nvfp4`
(NVFP4, 200 k ctx) is the default since v1.3.0 and the only 5090 text
path since v1.3.7. The experimental `rtx5090_nvfp4_vision` (180 k ctx,
port 5004) adds image and video input on the same weights. See
[`BLACKWELL.md`](BLACKWELL.md). Wait until the log shows
`Application startup complete.`.

Then in any project directory:

```powershell
qwen
```

Ask it to read or modify a file. The first request hits
`/v1/chat/completions`. If you see a normal response and tool calls
work, you're done.

## Sampler defaults

Qwen Code does not override sampler params unless you ask it to. The
shipped snapshots use Unsloth's recommended Qwen3 sampling for
thinking mode (temperature 0.6, top_p 0.95, top_k 20, min_p 0.0). For
coding-specific defaults baked into the snapshot:

- `start_thinking_coding` for thinking-mode coding (precise debug /
  architecture work).
- `start_instruct_coding` for non-thinking coding (faster, no
  `<think>` block).

## Reasoning / thinking output

The shipped chat template defaults to thinking ON for snapshots that
don't set `chat_template_kwargs.enable_thinking=false`. Qwen Code
displays the thinking content separately from the final answer; the
`reasoning` field comes back populated, the `content` field has the
post-thinking response.

If `content` comes back empty, the thinking block ate `max_tokens`.
Raise the budget. Qwen3.6 thinking can run 200-2000 tokens before
answering; 4096+ is safe for short Q&A, 8000+ for non-trivial
reasoning.

To force non-thinking on a per-request basis, append `/no_think` to
the user prompt, or use the `start_instruct_*` snapshots which
disable thinking via `chat_template_kwargs.enable_thinking=false`.

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
without escaping (`C:\Users\...` instead of `C:\\Users\\...`), the
JSON parse fails and the tool call drops. The cheap fix is a one-
liner in your project's `QWEN.md` (or any system-prompt file Qwen
Code reads):

> "I am on a Windows system, so properly escape directory backslashes
> to keep from breaking JSON."

This is a model-side issue, not a Qwen Code or server bug, and it
also reproduces on Linux vLLM and llama.cpp.

## Other Windows quirks worth knowing

These are upstream Qwen Code issues, not server-side problems, but
they show up when you run the CLI on Windows:

- **Slow first paint on Windows Terminal / PowerShell.** First
  `qwen` invocation can take 5-15s to render the TUI on cold cache
  (issues [#2386](https://github.com/QwenLM/qwen-code/issues/2386),
  [#706](https://github.com/QwenLM/qwen-code/issues/706)). Subsequent
  launches in the same session are fast.
- **`/quit` hang with `ansiRegex3 is not a function`**
  ([#3185](https://github.com/QwenLM/qwen-code/issues/3185)). If
  `/quit` hangs, just close the terminal. Tracked upstream.
- **Default shell is `cmd.exe`, not PowerShell**
  ([#2907](https://github.com/QwenLM/qwen-code/issues/2907),
  [#2909](https://github.com/QwenLM/qwen-code/issues/2909)). If your
  system prompt says "use PowerShell", Qwen Code still spawns shell
  tools through `cmd.exe`. Until upstream lands a setting, write
  prompts that work in either shell, or wrap the command yourself
  (`pwsh -NoProfile -Command "..."`).

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
| `404 Not Found` on `/v1/chat/completions` | `OPENAI_BASE_URL` is missing the trailing `/v1`, or wrong port. |
| `qwen` hangs on first request | Check `nvidia-smi` to confirm vLLM hasn't OOM'd; tail `logs\vllm_server.5001.log` for the real error. |
| `qwen` ignores my env vars | Something earlier in the search order is winning. Qwen Code reads `.qwen/.env` in cwd first, then plain `.env`, then `~/.qwen/.env`, then `~/.env`. Delete or fix the one ahead of yours. |
| Empty `content`, `finish_reason=length` | `max_tokens` ate the thinking phase. Raise to 8000+, or use an `instruct_*` snapshot. |
| Tool call returned but the file path wasn't found | Path-escape issue. Add the `QWEN.md` rule above. |
| `Unexpected message role.` | Qwen Code is sending a role the chat template doesn't handle. The shipped template aliases `developer` to `system` since v1.0.1. If you're on v1.0 or older, see [`CODEX.md`](CODEX.md) for the Option A four-line patch. |

## Why this is documented separately from OpenCode

Qwen Code's UX, default model selection, and tool catalog are
specific to Alibaba's Qwen-first agent flow (it knows about Qwen
thinking, defaults to Qwen tool conventions, ships Qwen-specific
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
  hit on the server side.
- [Qwen Code authentication docs](https://github.com/QwenLM/qwen-code/blob/main/docs/cli/authentication.md)
  and [configuration docs](https://qwenlm.github.io/qwen-code-docs/en/users/configuration/settings/).
