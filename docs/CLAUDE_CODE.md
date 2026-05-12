# Using Claude Code with this server

The bundled vLLM wheel serves Anthropic's `/v1/messages` endpoint
natively. That means Claude Code talks to your local server the same
way it talks to api.anthropic.com. No proxy, no translation layer,
no LiteLLM, no `claude-code-router`, no `claude-bridge`. Just an
`ANTHROPIC_BASE_URL` env var.

If you have used local LLMs with Claude Code before and ended up
stacking a proxy in front of an OpenAI-compatible server, you do
not need that here. Skip straight to step 2.

## Quick start

1. Install Claude Code (Node 20+ required):

   ```powershell
   npm install -g @anthropic-ai/claude-code
   ```

   Confirm with `claude --version`. On Windows, Anthropic's docs
   recommend running inside Git Bash or WSL2; native PowerShell
   works for the `claude` command itself but a few of its built-in
   bash-flavoured tools assume a POSIX-ish shell.

2. Start the server. Pick any snapshot in the launcher (the default
   is `start_speed` on port 5001 for Ampere/Ada, `rtx5090_nvfp4` on
   port 5001 for Blackwell), or run headless:

   ```powershell
   start.bat --headless --snapshot start_speed
   ```

   Wait until the log shows `Application startup complete.`

3. Point Claude Code at the server. Easiest is to put this in
   `%USERPROFILE%\.claude\settings.json` (or `~/.claude/settings.json`
   on Linux/macOS):

   ```json
   {
     "env": {
       "ANTHROPIC_BASE_URL": "http://127.0.0.1:5001",
       "ANTHROPIC_AUTH_TOKEN": "dummy",
       "ANTHROPIC_API_KEY": "dummy",
       "ANTHROPIC_MODEL": "any",
       "ANTHROPIC_SMALL_FAST_MODEL": "any",
       "ANTHROPIC_DEFAULT_OPUS_MODEL": "any",
       "ANTHROPIC_DEFAULT_SONNET_MODEL": "any",
       "ANTHROPIC_DEFAULT_HAIKU_MODEL": "any"
     }
   }
   ```

   Or export the same vars in your shell before running `claude`.
   The `env` block is a plain string-to-string map; Claude Code
   injects these into the session at startup.

4. Run `claude` in your project. It will hit your local server.

### Why so many model env vars

Claude Code routes different internal task types (main reasoning,
background summarisation, fast tool-arg generation) to different
model tiers. It looks up the actual model name in this order:

- `ANTHROPIC_MODEL` for the primary model.
- `ANTHROPIC_SMALL_FAST_MODEL` for the background/quick-task slot.
- `ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`,
  `ANTHROPIC_DEFAULT_HAIKU_MODEL` for the per-tier overrides used
  when a request asks for a specific tier by name.

I set all of them to `"any"` so every code path lands on the same
local model regardless of which slot Claude Code is filling. The
patched wheel uses a wildcard served-model-name (see below) so the
literal string does not need to match anything on the server.

## Why the model name is `any`

The patched wheel uses a wildcard served-model-name. Claude Code
will sometimes send model names like `claude-sonnet-4-5` or
`claude-haiku-4-5` that do not match what vLLM loaded. The wildcard
accepts whatever the client sends, so I do not have to coordinate
names. `"any"` is just the convention used in these docs. Any
non-empty string works.

## What about claude-code-router or claude-bridge

I do not use either with this server. They are real, maintained
projects (`musistudio/claude-code-router` is the larger one), but
they exist to solve two problems this server does not have:

- **`claude-code-router`** routes Claude Code's requests across
  multiple backend models based on task type (e.g., heavy reasoning
  to a 70B, background tasks to a 4B). If you have one local model
  serving everything, you do not need it.
- **`claude-bridge`** translates between Anthropic's `/v1/messages`
  schema and OpenAI's `/v1/chat/completions`. This server speaks
  `/v1/messages` natively (that is the point of the patched wheel),
  so the translation layer is redundant.

If you genuinely want per-task routing across multiple local models,
`claude-code-router` will work in front of this server; just point
its upstream at `http://127.0.0.1:5001`. For everything else, env
vars are simpler.

## Why tool calling just works

Every snapshot ships the tool-calling fix baked in:

- vLLM PR [#35687](https://github.com/vllm-project/vllm/pull/35687): treats `<tool_call>` as an implicit `</think>`.
- vLLM PR [#40861](https://github.com/vllm-project/vllm/pull/40861): streaming-path fixes for split tags, dropped parameters, multi-call drops under speculative decoding, and structural delimiters appearing as literal text inside parameter values.
- `qwen3.5-enhanced.jinja` chat template, vendored under `templates/`.
- `--tool-call-parser=qwen3_coder`, `--reasoning-parser=qwen3`, `default-chat-template-kwargs={"preserve_thinking": false}`.

So Claude Code's tool calls (Read, Edit, Bash, Grep, etc.) work out of
the box without per-snapshot tweaking.

## Which snapshot to pick

Snapshot ids depend on which zip you installed (Ampere/Ada vs Blackwell).
The dashboard tags every card with `[Blackwell]` or `[Ampere/Ada]` and
groups them by your detected GPU since v1.2.4, so the right ones float
to the top automatically.

**Ampere / Ada zip (RTX 3090, 4090, A6000):**

| Use case | Snapshot | Port |
|---|---|---|
| Daily Claude Code on a 3090 with no display attached | `start_speed` (90k ctx) | 5001 |
| Short prompts, max tok/s | `start_72tps` (32k ctx) | 5001 |
| Long Claude Code sessions that need 127k context | `start_127k` | 5001 |
| Single GPU, display attached | `start_gpu0_50k` | 5001 |
| Need 160k context, have 2 GPUs | `start_pp2_160k` | 5002 |

**Blackwell zip (RTX 5060, 5070, 5080, 5090):**

| Use case | Snapshot | Port |
|---|---|---|
| Daily Claude Code on a 5090, default | `rtx5090_nvfp4` (NVFP4, 200k ctx) | 5001 |
| Image and video input (experimental) | `rtx5090_nvfp4_vision` (NVFP4, 180k ctx) | 5004 |

NVFP4 is the only 5090 path since v1.3.7 because it routes FFN GEMMs
through FlashInfer's sm_120 native FP4 tensor cores and bypasses the
170W prefill ceiling AutoRound hits on consumer Blackwell. The
AutoRound INT4 5090 snapshots were removed in v1.3.7.

If you change the port, update `ANTHROPIC_BASE_URL` to match.

## Verifying the connection

A quick sanity check before launching Claude Code:

```powershell
curl http://127.0.0.1:5001/v1/messages `
  -H "Content-Type: application/json" `
  -H "anthropic-version: 2023-06-01" `
  -d "{\"model\":\"any\",\"max_tokens\":200,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}"
```

You should see a JSON response with a `content` array. If you get a
404, the server is up but on a different port. If you get a connection
refused, the server isn't ready yet.

## Common gotchas

- **The thinking budget eats short replies.** Qwen3.6 is a thinking
  model. With `max_tokens` under about 1500 the entire budget can go
  into the `<think>` block and you get an empty `content`. Claude Code
  defaults are fine, but if you script your own calls, set
  `max_tokens` to 2000 or higher for short Q&A.
- **First request after boot takes longer.** vLLM compiles attention
  kernels lazily on first request. Subsequent requests are fast.
- **Don't run two snapshots on the same port.** If you switch
  snapshots, stop the previous one first (the launcher does this for
  you, or use `snapshots\stop_vllm.bat`).
- **Windows paths in tool-call arguments can break JSON parsing.**
  Qwen3.6 sometimes emits a single backslash inside a JSON string when
  file paths land in tool arguments (`C:\Users\...` instead of
  `C:\\Users\\...`), which makes the call fail to parse. This is a
  model-side issue, not Windows-specific (it also reproduces on Linux
  vLLM and llama.cpp), but Windows users hit it more because their
  paths are full of backslashes. The cheap fix is a one-liner in your
  agent's system prompt: *"I am on a Windows system, so properly
  escape directory backslashes to keep from breaking JSON."* Once the
  model knows the target, it escapes correctly. Reported and confirmed
  by a Reddit user running OpenCode against this server.

## Other clients

The same `/v1/messages` endpoint works with anything that speaks the
Anthropic API. Cline, Cursor, and Codex CLI all use the same
`ANTHROPIC_BASE_URL` env var. For OpenAI-format clients (Continue,
LM Studio's external server, OpenWebUI), point them at
`http://127.0.0.1:5001/v1` instead, which is the standard OpenAI
endpoint that vLLM also serves.

## Reference

- vLLM's official Claude Code integration page: https://docs.vllm.ai/en/stable/serving/integrations/claude_code/
- Snapshot list and ports: [`snapshots/README.md`](../snapshots/README.md)
- Tool-calling patch details: [`devnen/vllm-windows` CHANGES_VS_SYSTEMPANIC.md](https://github.com/devnen/vllm-windows/blob/main/CHANGES_VS_SYSTEMPANIC.md)
