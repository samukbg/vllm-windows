# Using OpenCode with this server

[OpenCode](https://opencode.ai) is an OpenAI-compatible coding agent
that talks to this server's `/v1/chat/completions` endpoint directly.
No proxy, no LiteLLM, no translation layer.

## Quick start

1. **Start the server.** Pick any snapshot in the launcher (the default
   is `start_speed` on port 5001), or run headless:

   ```powershell
   start.bat --headless --snapshot start_speed
   ```

   Wait until the log shows `Application startup complete.`. The other
   single-GPU snapshots (`start_72tps`, `start_127k`, `start_mtp4`,
   `start_gpu0_50k`) also listen on port 5001. Only `start_pp2_160k`
   uses port 5002.

2. **Find the served model id.** Our snapshots set
   `--served-model-name=qwen3.6-27b-autoround`. OpenCode requires the
   model key in `opencode.json` to match exactly what
   `GET /v1/models` returns, so verify with:

   ```powershell
   curl http://localhost:5001/v1/models
   ```

   You should see `"id": "qwen3.6-27b-autoround"` in the response.

3. **Create `opencode.json`.** The global path on Windows is
   `%USERPROFILE%\.config\opencode\opencode.json` (which expands the
   same as `~/.config/opencode/opencode.json` on macOS/Linux). If the
   directory does not exist, run `opencode` once first to let it
   create the parent dirs, then drop this file in:

   ```json
   {
     "$schema": "https://opencode.ai/config.json",
     "provider": {
       "vllm-local": {
         "npm": "@ai-sdk/openai-compatible",
         "name": "Local vLLM",
         "options": {
           "baseURL": "http://localhost:5001/v1",
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

   Adjust `"context"` to match your snapshot's `--max-model-len`:
   `start_72tps` 32000, `start_speed` 90000, `start_127k`/Unsloth
   tunes 127000, `start_pp2_160k` 160000, `start_gpu0_50k` 50000.
   You can also drop this file in your project root as `opencode.json`
   to override per-repo.

4. **Launch.** From any project directory:

   ```powershell
   opencode
   ```

   The default model from `opencode.json` should be selected. If not,
   use `/connect`, scroll to **Other**, and enter `vllm-local` as the
   provider id. Any non-empty string works as the API key.

## Optional: Windows path-handling rule for tool calls

Some agentic loops on Windows trip over forward-slash paths the model
emits in tool calls. The community fix is a one-liner in `AGENTS.md`.
Two scopes:

- **Project-local** (recommended): drop an `AGENTS.md` in your
  project root. Only applies when OpenCode is run from that
  directory or below — won't bleed into your macOS/Linux sessions.
- **Global**: `%USERPROFILE%\.config\opencode\AGENTS.md` applies to
  every project on this machine.

Suggested content:

```
On Windows, always use Windows-style paths (C:\path\to\file) in tool
calls. Do not use forward slashes. When a path appears inside a JSON
argument, escape backslashes as needed (C:\\path\\to\\file).
```

If you already have an `AGENTS.md`, run `/init` inside OpenCode and it
will improve the existing file in place rather than overwriting.

## Troubleshooting

- **OpenCode shows "model not found"** — the model key in
  `opencode.json` doesn't match the served-model-name. Re-run
  `curl http://localhost:5001/v1/models` and copy the `id` field
  verbatim.
- **`/connect` asks for an API key** — vLLM doesn't validate it, but
  OpenCode requires the field non-empty. Any string works
  (`sk-no-key-required`, `none`, etc.).
- **Tool calls fail or stop mid-task** — almost always a server-side
  issue rather than OpenCode. Run
  `python windows_tools\check_coherence.py --port 5001` to confirm the
  server itself is healthy. If coherence passes but tool calls still
  fail, capture the request/response and open an issue.
- **OpenCode picks the wrong model** — you can pin per-session with
  `/model vllm-local/qwen3.6-27b-autoround`, or set the top-level
  `"model"` field in `opencode.json` as shown above.

## Related

- [`CLAUDE_CODE.md`](CLAUDE_CODE.md), the easiest integration overall.
- [`CODEX.md`](CODEX.md), Responses-API client notes.
- [`PI.md`](PI.md), Pi coding agent setup.
- [`COHERENCE.md`](COHERENCE.md), the validator to run if OpenCode
  sees garbage output.
- [OpenCode docs: Config](https://opencode.ai/docs/config),
  [Rules](https://opencode.ai/docs/rules), and
  [Custom providers](https://opencode.ai/docs/providers).
