# Using OpenAI Codex CLI with this server

Codex CLI talks to OpenAI's **Responses API** (`/v1/responses`)
rather than the older Chat Completions API (`/v1/chat/completions`).
The bundled vLLM wheel implements both endpoints, and the shipped
chat template aliases the `developer` role (which Responses API
clients send for system-tier instructions) to `system` so this all
works out of the box since v1.0.1.

If you are on v1.0 or older you will hit the `Unexpected message role.`
error described below. Upgrade to v1.0.1 or apply Option A by hand.

If you do not specifically want Codex CLI, the easier path is one of
these clients, all of which work with this server out of the box:

- Claude Code: see [`CLAUDE_CODE.md`](CLAUDE_CODE.md).
- OpenCode, Cline, Cursor, Continue, KiloCode: point their
  OpenAI-compatible base URL at `http://127.0.0.1:5001/v1` and pick
  any model name. They all use `/v1/chat/completions`, which has
  none of the role-mapping issues below.

The rest of this page is for users who specifically want Codex CLI.

## The error you will see (v1.0 and older only)

When Codex sends its first inference request on v1.0 or older, vLLM
logs:

```
ERROR ... [hf.py:502] An error occurred in `transformers` while
applying chat template
ERROR ... jinja2.exceptions.TemplateError: Unexpected message role.
INFO:     127.0.0.1:NNNNN - "POST /v1/responses HTTP/1.1" 400 Bad Request
```

The traceback ends inside `qwen3.5-enhanced.jinja` at the role
dispatch. The shipped template only branches on `system`, `user`,
`assistant`, and `tool`. Codex sends a fifth role, `developer`,
which OpenAI introduced as a system-tier role for the Responses API.
The template falls through to `raise_exception('Unexpected message
role.')` and vLLM returns 400.

## The fix

If you are on v1.0.1 or newer, there is no fix needed; the alias is
shipped. If you are on v1.0 or older and cannot or will not upgrade,
pick one of these.

### Option A: patch the chat template to accept `developer`

Open `templates/qwen3.5-enhanced.jinja` and add a four-line alias at
the very top of the message loop, before the first `{%- if
message.role == "system" -%}` check. The block converts a
`developer` role into a `system` role for the rest of the template:

```jinja
{%- if message.role == "developer" -%}
    {%- set message = dict(message, role="system") -%}
{%- endif -%}
```

Save the file and restart the snapshot. Codex CLI's first request
will now go through. This is a tiny patch with no effect on Claude
Code, OpenCode, Cline, or any other client that already uses the
four standard roles.

### Option B: replace the template with froggeric/Qwen-Fixed-Chat-Templates

[`froggeric/Qwen-Fixed-Chat-Templates`](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
on Hugging Face is a community-maintained drop-in replacement for
the official Qwen3.5 / Qwen3.6 templates. Same `developer` to
`system` alias, plus five other fixes:

- `|items` iteration that breaks on llama.cpp / LM Studio / MLX.
- Empty `<think/>` blocks wasting context tokens on every history
  turn.
- `</thinking>` hallucination on Qwen3.6.
- Arguments serialised with `|tojson` crashing when the value is
  already a string.
- `raise_exception('No user query found')` hard-crashing agentic
  tool loops.

Download the relevant `.jinja` and point the snapshot at it via
`--chat-template`. The bundled snapshots use
`templates/qwen3.5-enhanced.jinja` by default, so the simplest
swap is to overwrite that file.

I have not validated the `froggeric` template end-to-end against
this snapshot stack yet, so if you take this path, run
`windows_tools/check_coherence.py --port 5001` afterwards to
confirm output is still clean.

## Installing Codex CLI on Windows

Two paths, pick one. Both land you on the same Rust binary (the old
TypeScript-based npm package was retired during the rewrite). Latest
stable as of this writing is `0.129.0` (versioned `rust-v0.129.0` on
the GitHub releases page).

```powershell
# winget, the cleanest Windows-native install
winget install --id OpenAI.Codex

# or npm, requires Node 20+
npm install -g @openai/codex
```

Confirm with `codex --version`.

## Codex CLI configuration

Codex does not respect `OPENAI_BASE_URL` or `OPENAI_API_KEY`
environment variables for custom providers. You have to declare the
provider in `%USERPROFILE%\.codex\config.toml` (Windows-native) or
`~/.codex/config.toml` (WSL / macOS / Linux):

```toml
[model_providers.local_vllm]
name = "Local vLLM"
base_url = "http://127.0.0.1:5001/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
stream_idle_timeout_ms = 600000

[profiles.qwen]
model_provider = "local_vllm"
model = "any"
```

Then export a dummy key (vLLM does not check it but the env var must
exist) and launch Codex with the profile:

```powershell
$env:OPENAI_API_KEY = "dummy"
codex --profile qwen
```

Notes:

- The `model` field can be literally `any`; the wheel uses a
  wildcard served-model-name so any string accepts.
- `wire_api` MUST live under `[model_providers.<id>]`. Putting it
  under `[profiles.<name>]` silently does nothing and Codex falls
  back to the default, which on current versions is a hard error.
- `stream_idle_timeout_ms` defaults to 300000 (5 minutes). On a long
  thinking response or a cold-kernel first request, Qwen3.6 can sit
  quiet long enough to trip that. Bumping it to 10 minutes avoids
  spurious mid-generation disconnects.
- Optional fields the provider block accepts if you need them:
  `query_params` (extra query string pairs, e.g. `api-version` for
  Azure-shaped endpoints), `http_headers` and `env_http_headers`
  (static or env-sourced extra headers), `request_max_retries`,
  `stream_max_retries`.
- Do not use `codex --oss`. That mode hardcodes Ollama-only
  endpoints (`/api/tags`, `/api/pull`) which do not exist on vLLM
  and you will get 404s during model discovery.
- `wire_api = "responses"` is required. Codex 0.80 and earlier
  accepted `wire_api = "chat"` which routed through
  `/v1/chat/completions` and avoided the `developer` role problem
  entirely, but that path was deprecated in December 2025 and
  removed around February 2026. Current versions hard-error if you
  set it.

## Verifying it works

After configuring Codex (and patching the template if you are on
v1.0 or older):

1. Restart the snapshot so the template is loaded.
2. Run `codex --profile qwen` in any project directory.
3. Ask Codex to read a file. The first request hits
   `/v1/responses`. If you see normal output instead of a 400, the
   wiring is good.

To skip the `--profile` flag every launch, add `profile = "qwen"` at
the top of `config.toml` (above any `[...]` section). Codex picks it
up as the default.

If you still see `Unexpected message role.` in the vLLM log on v1.0
or older, the snapshot is loading a different template than the one
you patched. Check the `--chat-template` flag in your snapshot file
matches the file you edited.

## Why the patch is now shipped by default

OpenAI's API itself bidirectionally aliases `system` and `developer`
(it casts one to the other depending on which model you target), so
the two roles are functionally equivalent at the model level. The
documented harm in the wild flows entirely from *not* aliasing:
silent message drops on MiniMax M2.5, literal `<|developer|>` token
corruption on GLM-4.7, HTTP 500 crashes on Qwen3.5/3.6 with stock
templates on llama.cpp.

Multiple clients beyond Codex CLI hit this same wall:

- **PI agent** (`badlogic/pi-mono`) sends `developer` by default for
  reasoning-capable model configs. The `compat.supportsDeveloperRole:
  false` flag in PI's config is a client-side workaround that does
  the same role mapping on the way out.
- **OpenCode** sends `system` by default, but routes through the
  Responses API (with `developer`) when the provider is configured
  via the official `@ai-sdk/openai` adapter rather than the generic
  `@ai-sdk/openai-compatible` one.
- Any future Responses-API client targeting o-series or GPT-5
  semantics.

Given that, baking the alias in trades nothing (no client sends
`developer` content with semantics that need different treatment from
`system`) for compatibility with several real-world clients out of
the box.

## Related

- [`CLAUDE_CODE.md`](CLAUDE_CODE.md), the supported integration.
- [`COHERENCE.md`](COHERENCE.md), the validator to run after any
  template change.
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md), other vLLM-side
  failure modes.
