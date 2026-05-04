# Using Pi coding agent with this server

[Pi](https://github.com/badlogic/pi-mono) (`@mariozechner/pi-coding-agent`)
is a TypeScript coding agent that talks to any provider you register.
There is no built-in vLLM provider in Pi, but Pi supports custom
providers via a tiny extension dropped into
`~/.pi/agent/extensions/`. The extension below registers this server's
local OpenAI-compatible endpoint as a Pi provider named `qwen-local`,
with the right Qwen3 thinking flags so reasoning works end-to-end.

If you do not specifically want Pi, the easier path is one of these
clients, all of which work with this server out of the box:

- Claude Code: see [`CLAUDE_CODE.md`](CLAUDE_CODE.md). Native
  `/v1/messages`, no setup beyond a base URL.
- OpenCode, Cline, Cursor, Continue, KiloCode: point their
  OpenAI-compatible base URL at `http://127.0.0.1:5001/v1` and pick
  any model name. They all use `/v1/chat/completions`.
- OpenAI Codex CLI: see [`CODEX.md`](CODEX.md). Slightly fiddlier
  because of the Responses API.

The rest of this page is for users who specifically want Pi.

## Why a custom extension is needed

Pi's [`stakira/pi-lmstudio`](https://github.com/stakira/pi-lmstudio)
extension targets LM Studio's `/api/v1/models` endpoint, which vLLM
does not serve. Its sibling `pi-llama-server` extension targets
llama-server's `/v1/models` and reads `--ctx-size` out of the
response's `status.args` array, which vLLM also does not serve. Both
silently catch fetch failures and register the provider with an empty
model list, which means Pi's `/model` picker has nothing to select
and you end up entering the model id by hand each session.

A 25-line provider extension (below) avoids both problems and only
takes one paste.

## Step 1: install Pi

```bash
npm install -g @mariozechner/pi-coding-agent
```

If you do not have Node, install Node 20+ first
([nodejs.org](https://nodejs.org)). Pi runs on Windows, macOS, and
Linux; the server only needs to be reachable on `127.0.0.1:5001`
(or whichever port your snapshot uses), so Pi can run on the same
machine or on another box that can reach this server's port.

## Step 2: drop in the extension

Save the snippet below to
`%USERPROFILE%\.pi\agent\extensions\qwen-local.ts` (Windows) or
`~/.pi/agent/extensions/qwen-local.ts` (macOS/Linux). Pi
auto-discovers extensions from this path on next launch.

```typescript
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

const BASE_URL = process.env.QWEN_LOCAL_URL ?? "http://127.0.0.1:5001/v1";
// Match this to the --max-model-len of the snapshot you are running.
// start_speed = 90000, start_127k / unsloth_127k_* = 127000,
// pp2_160k = 160000, start_gpu0_50k = 50000.
const CONTEXT_WINDOW = Number(process.env.QWEN_LOCAL_CTX ?? 90000);

export default function (pi: ExtensionAPI) {
  pi.registerProvider("qwen-local", {
    name: "Qwen3.6 (local vLLM)",
    baseUrl: BASE_URL,
    apiKey: "qwen-local",        // vLLM does not check the key, any string works
    api: "openai-completions",
    models: [
      {
        id: "qwen3.6-27b",
        name: "Qwen3.6 27B (local)",
        reasoning: true,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: CONTEXT_WINDOW,
        maxTokens: CONTEXT_WINDOW,
        compat: {
          // The shipped chat template reads chat_template_kwargs.enable_thinking.
          thinkingFormat: "qwen-chat-template",
          // Belt-and-braces: this server's template aliases developer -> system
          // since v1.0.1, but tell Pi to send "system" anyway so older builds
          // do not 400.
          supportsDeveloperRole: false,
          maxTokensField: "max_tokens",
        },
      },
    ],
  });
}
```

The model `id` (`qwen3.6-27b` above) is a label local to Pi; vLLM's
served-model-name is wildcard, so any string accepts. If you change
snapshots and the context window changes, restart Pi (or just edit
`CONTEXT_WINDOW` and `/reload`).

## Step 3: launch and select the model

Start the snapshot from this server's launcher (`start.bat`,
double-click `start_speed`, etc.). Then in any project directory:

```bash
pi
```

Inside Pi:

- Run `/model` and pick `Qwen3.6 27B (local)`, **or**
- Press `Ctrl+P` and type `qwen` to search for it.

Ask Pi to do something. The first request hits
`http://127.0.0.1:5001/v1/chat/completions`. If you see normal
output and tool calls work, the extension is wired up correctly.

## Optional: point Pi at a different host or port

The snippet above honors two env vars before falling back to its
defaults:

```bash
# different port (e.g. pp2_160k snapshot listens on 5002)
export QWEN_LOCAL_URL=http://127.0.0.1:5002/v1
export QWEN_LOCAL_CTX=160000

# remote box on the same LAN
export QWEN_LOCAL_URL=http://192.168.1.50:5001/v1
```

On Windows, `set QWEN_LOCAL_URL=...` in cmd or
`$env:QWEN_LOCAL_URL = "..."` in PowerShell before `pi`.

## Verifying it works

1. Server up: visit `http://127.0.0.1:5001/v1/models` in a browser.
   You should see a JSON `data` array with one entry.
2. Pi picks up the extension: launch `pi` and run `/model`. The
   list should include `Qwen3.6 27B (local)`.
3. Reasoning is on: ask Pi a question that triggers thinking
   (`pi --think medium "what is 23 * 47, show your reasoning"`).
   The response should include a thinking block.
4. Tools work: ask Pi to read a file in your project. The tool call
   should round-trip without `Unexpected message role.` or
   `Invalid argument` errors. If you see those, run
   `windows_tools/check_coherence.py --port 5001` to confirm the
   server itself is healthy, then check that the snapshot has
   `--tool-call-parser=qwen3_coder` and `--reasoning-parser=qwen3`
   in its argv (all shipped snapshots do).

## Auto-compaction

Pi's [auto-compaction](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/compaction.md)
is the main reason users land here over Cline / Continue. With this
server, compaction triggers when Pi's running token count crosses
the model's `contextWindow`. Set `CONTEXT_WINDOW` in the extension
to match your snapshot's `--max-model-len`, otherwise Pi will either
compact too early (wasting context) or send requests the server
rejects with `This model's maximum context length is N`.

## Why this is not shipped as a built-in Pi provider

Pi ships a provider list per release; adding a `qwen-local` entry
upstream would require Pi to know about port 5001 and the Qwen3.6
chat template, which is product-specific. The custom-extension path
is the documented integration point for self-hosted endpoints, and
keeps the configuration where the user controls it. If you already
run another Pi extension and prefer to extend it, the same
`pi.registerProvider("qwen-local", { ... })` block drops directly
into your existing extension's default export.

## Related

- [`CLAUDE_CODE.md`](CLAUDE_CODE.md), the easiest integration overall.
- [`CODEX.md`](CODEX.md), the Responses-API client notes that
  motivated the v1.0.1 `developer`-role alias also used here.
- [`COHERENCE.md`](COHERENCE.md), the validator to run if Pi sees
  garbage output (almost always a server-side problem, not Pi).
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md), every failure mode I've
  hit on the server side.
