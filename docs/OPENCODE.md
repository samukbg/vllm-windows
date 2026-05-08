# Using OpenCode with this server

[OpenCode](https://opencode.ai) ([sst/opencode](https://github.com/sst/opencode))
is a terminal AI coding agent. It talks to this server's
`/v1/chat/completions` endpoint directly via the Vercel AI SDK
(`@ai-sdk/openai-compatible`). No proxy, no LiteLLM, no translation
layer.

## Which OpenCode

There are two repos with confusingly similar names. Use the first one.

- **`sst/opencode`** (this doc) — actively maintained by the SST team.
  Latest is v1.14.x (May 2026). Docs at [opencode.ai](https://opencode.ai).
  npm package: `opencode-ai`. Yes, the npm name reuses the older
  fork's slug; the GitHub repo is what tells you which project you're
  on.
- `opencode-ai/opencode` (the older Go-based fork acquired by Charm) —
  not what this doc covers. If you cloned that repo, the config schema
  below will not match.

If you don't specifically want OpenCode, the easier path is one of
these clients, all of which work with this server out of the box:

- Claude Code: see [`CLAUDE_CODE.md`](CLAUDE_CODE.md). Native
  `/v1/messages`, no setup beyond a base URL.
- Cline, Cursor, Continue, KiloCode: point the OpenAI-compatible base
  URL at `http://127.0.0.1:5001/v1` and pick any model name.
- Codex CLI: see [`CODEX.md`](CODEX.md). Slightly fiddlier (Responses
  API, role aliasing).
- Qwen Code: see [`QWEN_CLI.md`](QWEN_CLI.md).

## Step 1: install OpenCode

Pick one. All resolve to the same binary.

```powershell
# Windows: Scoop is the cleanest path, handles PATH automatically
scoop install extras/opencode

# Or npm (any platform with Node 20+)
npm install -g opencode-ai@latest

# Or Chocolatey
choco install opencode
```

The `curl -fsSL https://opencode.ai/install | bash` one-liner is
macOS/Linux only; on Windows use one of the above. If a package
manager ships an older version, `npm install -g opencode-ai@latest`
forces the current release. Confirm with `opencode --version`.

## Step 2: launch a snapshot

Pick any snapshot in the launcher (the default is `start_speed` on
port 5001), or run headless:

```powershell
start.bat --headless --snapshot start_speed
```

Wait until the log shows `Application startup complete.` The
single-GPU snapshots all listen on port 5001. Only `start_pp2_160k`
uses port 5002.

Verify the server's served-model-name with:

```powershell
curl http://127.0.0.1:5001/v1/models
```

The patched wheel uses a wildcard served-model-name, so any string
works as the model id in `opencode.json`. The shipped snapshots
declare `qwen3.6-27b-autoround` (or `qwen3.6-27b-nvfp4` on the
Blackwell NVFP4 snapshot) as the canonical id, but `"any"` also
works.

## Step 3: configure the provider

OpenCode reads config from one of:

- Project root: `./opencode.json` (per-repo override).
- Global: `%USERPROFILE%\.config\opencode\opencode.json` on Windows,
  `~/.config/opencode/opencode.json` on macOS/Linux.

If the global directory does not exist, run `opencode` once to let it
create the parent dirs, then drop this file in:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "vllm-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local vLLM",
      "options": {
        "baseURL": "http://127.0.0.1:5001/v1",
        "apiKey": "sk-no-key-required"
      },
      "models": {
        "qwen3.6-27b-autoround": {
          "name": "Qwen3.6 27B (local vLLM)",
          "limit": {
            "context": 90000,
            "output": 8192
          }
        }
      }
    }
  },
  "model": "vllm-local/qwen3.6-27b-autoround"
}
```

A few things that bite people:

- **`baseURL` must end in `/v1`.** Without the path, the AI SDK
  builds wrong URLs and every request 404s.
- **The model key under `models` must match what `GET /v1/models`
  returns** (or you can use `"any"` since the wildcard accepts any
  string). A mismatch fails silently rather than logging a useful
  error.
- **`apiKey` cannot be empty.** vLLM does not validate it, but the
  SDK requires the field non-empty. Any string works
  (`sk-no-key-required`, `none`, `dummy`, etc.). You can also use
  `"$ENV_VAR"` to read from the environment.
- **`limit.context`** should match your snapshot's `--max-model-len`.
  On the Ampere/Ada zip: `start_72tps` 32000, `start_speed` 90000,
  `start_127k` / Unsloth tunes 127000, `start_pp2_160k` 160000,
  `start_gpu0_50k` 50000. On the Blackwell zip: `rtx5090` 240000,
  `rtx5090_max` 280000.

`@ai-sdk/openai-compatible` is the right adapter here because the
shipped snapshots serve `/v1/chat/completions`. The other adapter,
`@ai-sdk/openai`, is for the Responses API (`/v1/responses`) and is
what Codex CLI and OpenCode Zen's Codex models route through. The
bundled wheel implements both endpoints, but for OpenCode the
chat-completions path is simpler and avoids the `developer`-role
issues described in [`CODEX.md`](CODEX.md).

## Step 4: launch

From any project directory:

```powershell
opencode
```

The default model from `opencode.json` should be selected. If not,
press `/` to open the picker and pick `vllm-local/qwen3.6-27b-autoround`,
or run `/model vllm-local/qwen3.6-27b-autoround` to pin it for the
session.

## Tool calling just works

Every snapshot ships the tool-calling fix baked in:

- vLLM PR [#35687](https://github.com/vllm-project/vllm/pull/35687):
  treats `<tool_call>` as an implicit `</think>`.
- vLLM PR [#40861](https://github.com/vllm-project/vllm/pull/40861):
  streaming-path fixes for split tags, dropped parameters, multi-call
  drops under speculative decoding, and structural delimiters
  appearing as literal text inside parameter values.
- `qwen3.5-enhanced.jinja` chat template under `templates/`.
- `--tool-call-parser=qwen3_coder` and `--reasoning-parser=qwen3`.

So OpenCode's read / edit / shell tool calls round-trip cleanly
without per-snapshot tweaking. Reports of OpenCode tool calls
breaking against local LLMs are usually Ollama-side parsing bugs
(Ollama serialises tool calls as plain text under some chat
templates); none of that applies here because vLLM hands back
proper structured `tool_calls` JSON.

## AGENTS.md (project rules)

OpenCode reads `AGENTS.md` for persistent, project-specific
instructions. It also falls back to `CLAUDE.md` if no `AGENTS.md`
exists, which is handy if you already wrote rules for Claude Code.

Two scopes:

- **Project-local** (recommended): drop an `AGENTS.md` in your
  project root. Only applies when OpenCode is run from that
  directory or below.
- **Global**: `%USERPROFILE%\.config\opencode\AGENTS.md` applies to
  every project on the machine.

To bootstrap one, run `/init` inside OpenCode. It scans the repo and
either creates a starter file or improves the existing one in place
(it does not overwrite).

### Windows path-handling rule

Qwen3.6 sometimes emits a single backslash inside JSON tool-call
arguments (`C:\Users\...` instead of `C:\\Users\\...`), which makes
the call fail to parse. This is a model-side issue, not OpenCode-
or Windows-specific (it also reproduces on Linux vLLM and llama.cpp),
but Windows users hit it more because their paths are full of
backslashes.

The cheap fix is a one-liner in `AGENTS.md`:

```
On Windows, always use Windows-style paths (C:\path\to\file) in tool
calls. When a path appears inside a JSON argument, escape backslashes
as needed (C:\\path\\to\\file).
```

Once the model knows the target, it escapes correctly. Reported and
confirmed by a Reddit user running OpenCode against this server.

## Troubleshooting

- **"model not found"** — the model key in `opencode.json` doesn't
  match what `/v1/models` returns. Re-run `curl http://127.0.0.1:5001/v1/models`
  and copy the `id` field verbatim, or use `"any"`.
- **Every request 404s** — `baseURL` is missing the `/v1` path.
- **`/connect` asks for an API key** — vLLM doesn't validate it, but
  the SDK requires the field non-empty. Any string works.
- **Empty response, finish_reason length** — the thinking block ate
  `max_tokens`. Qwen3.6 is a thinking model; with the budget under
  about 1500 the entire allowance can go into `<think>`. OpenCode
  defaults are fine, but if you script your own calls, set
  `max_tokens` to 2000+ for short Q&A.
- **Tool calls fail or stop mid-task** — almost always a server-side
  issue rather than OpenCode. Run
  `python windows_tools\check_coherence.py --port 5001` to confirm
  the server itself is healthy. If coherence passes but tool calls
  still fail, capture the request/response and open an issue.
- **First request after boot takes longer.** vLLM compiles attention
  kernels lazily on first request. Subsequent requests are fast.
- **Wrong model selected** — pin per-session with
  `/model vllm-local/qwen3.6-27b-autoround`, or set the top-level
  `"model"` field in `opencode.json` as shown above.

## Related

- [`CLAUDE_CODE.md`](CLAUDE_CODE.md), the easiest integration overall.
- [`CODEX.md`](CODEX.md), Responses-API client notes.
- [`QWEN_CLI.md`](QWEN_CLI.md), Qwen Code (Alibaba's fork of
  gemini-cli) setup.
- [`PI.md`](PI.md), Pi coding agent setup.
- [`COHERENCE.md`](COHERENCE.md), the validator to run if OpenCode
  sees garbage output.
- [OpenCode docs: Config](https://opencode.ai/docs/config),
  [Rules](https://opencode.ai/docs/rules), and
  [Providers](https://opencode.ai/docs/providers).
