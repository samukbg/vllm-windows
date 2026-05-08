# Using Pi coding agent with this server

[Pi](https://github.com/badlogic/pi-mono) (`@earendil-works/pi-coding-agent`,
formerly `@mariozechner/pi-coding-agent` before the August 2026 rename) is a
TypeScript coding agent that talks to any provider you register. There is no
built-in vLLM provider in Pi, but Pi supports custom providers via a tiny
extension dropped into `~/.pi/agent/extensions/`. The extension below
registers this server's local OpenAI-compatible endpoint as a Pi provider
named `qwen-local` and **auto-detects the active snapshot's context window
at startup**, so Pi stays in sync no matter which snapshot you launched.

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

A small async provider extension (below) avoids both problems, only
takes one paste, and fetches the live `max_model_len` from the
running snapshot so you never edit a number by hand again.

## Step 1: install Pi

The official installer is the shortest path:

```bash
curl -fsSL https://pi.dev/install.sh | sh
```

Or via npm (Node 20+):

```bash
npm install -g @earendil-works/pi-coding-agent
```

Pi runs on Windows, macOS, and Linux. The server only needs to be
reachable on `127.0.0.1:5001` (or whichever port your snapshot uses),
so Pi can run on the same machine or on another box that can reach
this server's port.

## Step 2: drop in the extension

Save the snippet below to
`%USERPROFILE%\.pi\agent\extensions\qwen-local.ts` (Windows) or
`~/.pi/agent/extensions/qwen-local.ts` (macOS/Linux). Pi
auto-discovers extensions from this path on next launch.

```typescript
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const BASE_URL = process.env.QWEN_LOCAL_URL ?? "http://127.0.0.1:5001/v1";
// Used only when /v1/models is unreachable at startup. Match the smallest
// snapshot you ever launch so Pi never overshoots the active context.
const FALLBACK_CTX = Number(process.env.QWEN_LOCAL_CTX ?? 90000);

type VLLMModelCard = {
  id: string;
  // vLLM exposes the active --max-model-len on every /v1/models entry.
  max_model_len?: number;
};

async function fetchActiveContext(): Promise<{ id: string; ctx: number }> {
  try {
    const response = await fetch(`${BASE_URL}/models`, {
      signal: AbortSignal.timeout(2000),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = (await response.json()) as { data: VLLMModelCard[] };
    const card = payload.data?.[0];
    if (card?.max_model_len) {
      return { id: card.id, ctx: card.max_model_len };
    }
  } catch {
    // Server not up yet, or different shape. Fall through to defaults.
  }
  return { id: "qwen3.6-27b", ctx: FALLBACK_CTX };
}

export default async function (pi: ExtensionAPI) {
  const { id, ctx } = await fetchActiveContext();

  pi.registerProvider("qwen-local", {
    name: "Qwen3.6 (local vLLM)",
    baseUrl: BASE_URL,
    apiKey: "qwen-local",        // vLLM does not check the key, any string works
    api: "openai-completions",
    models: [
      {
        id,                       // mirrors what /v1/models reports
        name: `Qwen3.6 27B (local, ${ctx.toLocaleString()} ctx)`,
        reasoning: true,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: ctx,
        maxTokens: ctx,
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

Two things to note about this snippet:

- **The factory is `async`**, which Pi explicitly supports. Pi awaits
  the Promise before flushing provider registrations, so the live
  `/v1/models` fetch lands before the `/model` picker reads the list.
  Pattern documented in
  [`pi-mono/packages/coding-agent/docs/extensions.md`](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md#async-factory-functions).
- **`max_model_len` comes straight from vLLM's `/v1/models`
  response**. The model server already knows the active snapshot's
  `--max-model-len` and reports it on every model card, so the
  extension never has to guess and there is nothing to edit when you
  switch snapshots.

## Step 3: launch and select the model

Start the snapshot from this server's launcher (`start.bat`, double-click
`start_speed`, etc.). **Then** start Pi (the order matters, see "Auto-sync
with the active snapshot" below for why):

```bash
pi
```

Inside Pi:

- Run `/model` and pick `Qwen3.6 27B (local, NNN,NNN ctx)`, **or**
- Press `Ctrl+P` and type `qwen` to search for it.

Ask Pi to do something. The first request hits
`http://127.0.0.1:5001/v1/chat/completions`. If you see normal
output and tool calls work, the extension is wired up correctly.

## Auto-sync with the active snapshot

This was the main pain point reported on
[issue #6](https://github.com/devnen/qwen3.6-windows-server/issues/6):
every time you launch a different snapshot, Pi's `contextWindow` has to
match the new `--max-model-len` or compaction triggers wrong. The
async-fetch pattern above pulls the right number from `/v1/models`
every time Pi starts, so you don't maintain a snapshot-to-CTX map by
hand.

The one rule: **launch the snapshot first, then start Pi.** The
extension reads `/v1/models` exactly once at Pi startup. If you swap
snapshots while Pi is running:

- `/reload` inside Pi re-runs the extension factory and picks up the
  new `max_model_len`.
- A full Pi restart works too.

If `/v1/models` is unreachable when Pi starts (server still booting,
no snapshot launched, port wrong), the extension falls back to
`QWEN_LOCAL_CTX` (default 90000, same as `start_speed`). You'll see
the registered model name still shows the fallback context, which is
your cue to `/reload` once the server is up.

## Optional: point Pi at a different host or port

The snippet honors two env vars before falling back to its defaults:

```bash
# different port (e.g. pp2_160k snapshot listens on 5002)
export QWEN_LOCAL_URL=http://127.0.0.1:5002/v1

# remote box on the same LAN
export QWEN_LOCAL_URL=http://192.168.1.50:5001/v1

# offline override (only used if /v1/models cannot be fetched)
export QWEN_LOCAL_CTX=160000
```

On Windows, `set QWEN_LOCAL_URL=...` in cmd or
`$env:QWEN_LOCAL_URL = "..."` in PowerShell before `pi`.

## Verifying it works

1. Server up: visit `http://127.0.0.1:5001/v1/models` in a browser.
   You should see a JSON `data` array with one entry whose
   `max_model_len` matches the active snapshot's `--max-model-len`
   (90000 for `start_speed`, 127000 for `start_127k`, 240000 for
   `rtx5090`, 280000 for `rtx5090_max`, 160000 for `pp2_160k`, etc.).
2. Pi picks up the extension: launch `pi` and run `/model`. The list
   should include `Qwen3.6 27B (local, ... ctx)` and the number in
   parens should match step 1.
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
is the main reason users land here over Cline / Continue. Compaction
triggers when Pi's running token count crosses the model's
`contextWindow`. With the async-fetch extension above,
`contextWindow` always equals the running snapshot's
`--max-model-len`, so compaction kicks in at the right moment whether
you're on `start_speed` (90 k), `start_127k` (127 k), `pp2_160k`
(160 k), or `rtx5090_max` (280 k).

If you hand-edited the extension to use a static
`contextWindow`, Pi will either compact too early (wasting context)
or send requests the server rejects with
`This model's maximum context length is N`.

## Why this is not shipped as a built-in Pi provider

Pi ships a provider list per release; adding a `qwen-local` entry
upstream would require Pi to know about port 5001 and the Qwen3.6
chat template, which is product-specific. The custom-extension path
is the documented integration point for self-hosted endpoints and
keeps the configuration where the user controls it. If you already
run another Pi extension and prefer to extend it, the same
`pi.registerProvider("qwen-local", { ... })` block drops directly
into your existing extension's default export (just `await
fetchActiveContext()` first).

## Related

- [`CLAUDE_CODE.md`](CLAUDE_CODE.md), the easiest integration overall.
- [`CODEX.md`](CODEX.md), the Responses-API client notes that
  motivated the v1.0.1 `developer`-role alias also used here.
- [`COHERENCE.md`](COHERENCE.md), the validator to run if Pi sees
  garbage output (almost always a server-side problem, not Pi).
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md), every failure mode I've
  hit on the server side.
