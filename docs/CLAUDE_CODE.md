# Using Claude Code with this server

The bundled vLLM wheel serves Anthropic's `/v1/messages` endpoint
natively. That means Claude Code talks to your local server the same
way it talks to api.anthropic.com. No proxy, no LiteLLM, no
translation layer.

## Quick start

1. Start the server. Pick any snapshot in the launcher (the default is
   `start_72tps` on port 5001), or run headless:

   ```powershell
   start.bat --headless --snapshot start_72tps
   ```

   Wait until the log shows `Application startup complete.`

2. Point Claude Code at the server. Easiest is to put this in
   `~/.claude/settings.json` (or `%USERPROFILE%\.claude\settings.json`
   on Windows):

   ```json
   {
     "env": {
       "ANTHROPIC_BASE_URL": "http://127.0.0.1:5001",
       "ANTHROPIC_API_KEY": "dummy",
       "ANTHROPIC_AUTH_TOKEN": "dummy",
       "ANTHROPIC_DEFAULT_OPUS_MODEL": "any",
       "ANTHROPIC_DEFAULT_SONNET_MODEL": "any",
       "ANTHROPIC_DEFAULT_HAIKU_MODEL": "any",
       "ANTHROPIC_DEFAULT_HAIKU_BACKGROUND_MODEL": "any"
     }
   }
   ```

   Or export the same vars in your shell before running `claude`.

3. Run `claude` in your project. It will hit your local server.

## Why the model name is `any`

The patched wheel uses a wildcard served-model-name. Claude Code (and
Cline, Cursor, Codex CLI, OpenWebUI) often picks model names like
`claude-sonnet-4-5` or `claude-haiku-4-5` that don't match what vLLM
loaded. The wildcard accepts whatever the client sends, so you don't
have to coordinate names. `any` is the convention used in the docs;
literally any string works.

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
| Daily Claude Code on a 3090 with no display attached | `start_speed` | 5001 |
| Short prompts, max tok/s | `start_72tps` | 5001 |
| Long Claude Code sessions that need 127k context | `start_127k` | 5001 |
| Single GPU, display attached | `start_gpu0_50k` | 5001 |
| Need 160k context, have 2 GPUs | `start_pp2_160k` | 5002 |

**Blackwell zip (RTX 5060, 5070, 5080, 5090):**

| Use case | Snapshot | Port |
|---|---|---|
| Daily Claude Code on a 5090, default | `rtx5090` (240k ctx) | 5001 |
| Need >240k context for whole-codebase sessions | `rtx5090_max` (280k ctx) | 5001 |

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
