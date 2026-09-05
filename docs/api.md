# OpenAI-compatible API, KV contexts & web UI

## `coli serve`

`coli serve` keeps one model process loaded and exposes a text-only
OpenAI-compatible HTTP API. The gateway uses only the Python standard library;
inference still runs in the same dependency-free C engine.

```bash
cd c
COLI_MODEL=/nvme/glm52_i4 COLI_API_KEY=local-secret ./coli serve \
  --host 127.0.0.1 --port 8000 --model-id glm-5.2-colibri

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer local-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-5.2-colibri",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Implemented endpoints are `GET /v1/models`, `GET /v1/models/{model}`,
`POST /v1/chat/completions`, and legacy `POST /v1/completions`. Chat and
completion requests support JSON responses, SSE streaming, usage counts,
`max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, and up to four
custom `stop` sequences. Stop sequences are removed from the response and end
generation early in both JSON and streaming modes. The extension
`x_colibri_ignore_leading_stop: true` discards leading stop sequences until
the first non-whitespace response content, which is useful for local templates
that occasionally emit a role marker before the answer; strict OpenAI stop
behavior remains the default for client-provided sequences. GLM chat requests
with no client `stop` automatically use the template's `<|user|>` and
`<|observation|>` role markers, patiently ignoring only leading markers; this
prevents a model-completed turn from silently generating a new user or tool
turn. Inkling chat and legacy completion requests receive no implicit GLM stop
sequences. The extension
`enable_thinking: true` enables GLM-5.2's reasoning block; the standard
`reasoning_effort` field also enables it unless set to `none`.

The server serves one generation at a time: the model stays in one persistent
process, so concurrent HTTP requests queue instead of loading duplicate model
copies. Tool calling depends on the active engine; see the support matrix below.
Images and token penalties return an explicit error rather than being silently
ignored, with one documented exception: `seed` is accepted and ignored rather
than rejected (see below). Log probabilities are served on the glm engine (see
below) and refused with a named error on every other engine, never silently
ignored. Audio is accepted only by Inkling checkpoints with audio support. The
default bind address is localhost; set `COLI_API_KEY` before exposing the
server beyond the machine.

### `seed`

`seed` is accepted (not rejected) for OpenAI-API request-shape compatibility.
It currently has **no effect on any code path, at any temperature**: no
engine, and no field on the wire protocol, reads a per-request seed. The
`glm` and `inkling` engines seed their process-global RNG once from the
`SEED` environment variable at launch, never per request; no other engine
reads `SEED` or any per-request seed at all. Either way the request's
`seed` value goes nowhere. At `temperature: 0` this is moot anyway (greedy
decoding has no distribution to seed), but the same "no effect" is equally
true at `temperature > 0`, where a client might otherwise expect the value
to matter. A true per-request seed is out of scope for this build
regardless.

### Log probabilities and prompt echo (glm engine only)

`/v1/completions` accepts the legacy integer `logprobs` (**1–32**; the range
is bound to the engine's top-32 read-out interface, and anything above 32 is
a named 400) and boolean `echo`; `/v1/chat/completions` accepts boolean
`logprobs` plus integer `top_logprobs` (0–32) and returns
`choices[].logprobs.content[]` (`{token, logprob, bytes, top_logprobs}` per
generated token) — chat has no `echo` concept and rejects one with a 400. A
non-boolean `echo` is a named 400 (`invalid_value`) on both endpoints,
independent of whether `logprobs` is requested at all. On chat,
`top_logprobs` is type- and range-checked even when `logprobs` is
false or absent, so a malformed `top_logprobs` is a named 400 whether or
not the gate it would feed is open; a valid `top_logprobs` with `logprobs`
off remains a documented no-op.

The zero semantics are explicit, not a truthiness accident: on
`/v1/completions`, `logprobs: 0`, `false`, and `null` all mean **no log
probabilities** (the request succeeds with `choices[].logprobs: null`,
exactly as if the field were omitted), while boolean `true` is a named 400 —
the legacy field is an integer count, and a boolean carries no count. On
`/v1/chat/completions` the field is a boolean gate (`null` behaves like
`false`; any integer is a named 400).

`/v1/completions` with `echo: true` returns the full legacy `logprobs` object
(`tokens`, `token_logprobs`, `top_logprobs`, `text_offset`) covering the
echoed prompt plus any generated tokens, and `text` itself is the
reconstructed prompt followed by the completion (the standard OpenAI legacy
behavior for `echo: true`) rather than the completion alone; `echo` without
`logprobs` is a documented no-op. `text_offset` is a character offset into
that same returned `text` string, always counted from 0 — including when
`echo` is false, where `text` holds only the completion and the offsets
describe only that text, not a position within the (unreturned) prompt. The
requested top-k table is **unsorted** on the wire — do not assume the first
entry is the argmax. Per-token values are printed by the engine to six
decimal digits of precision. Non-finite values (a degenerate all-`-inf`
logit row, say) serialize as JSON `null`, never a clamped number.

Only the glm engine implements this channel; every other engine returns a
named 400 rather than silently ignoring the request.

Known limitations, current build:

- **Cost.** Requesting `logprobs` at all — completions or chat, `echo` or
  not — makes the engine re-run the ENTIRE prompt through a full read-out
  pass to score every position (the wire has one opt-in bit, not a separate
  "echo" bit), forfeiting prefix-cache reuse for that request. There is no
  long-echo cap; a very long prompt pays a correspondingly large one-shot
  activation buffer.
- **Cancellation.** `CANCEL` is not honored while a logprobs-opted-in request
  is inside its prefill read-out — the un-cancellable window widens by the
  read-out's own wall time on long prompts.
- **Alternative-token labels.** `top_logprobs` entries for candidate token
  ids other than the position's own actual token are not decoded text (no
  server-side tokenizer exists, by design) — they are labeled
  `<token_id:N>`. Only the position's own token (identified by an exact
  logprob match, not by id) gets its real decoded text.
- **The sampled token is not guaranteed to appear in its own
  `top_logprobs` table.** The engine's numeric channel reports the top-k
  candidates by its own read-out; if the actually chosen token falls
  outside that table, no entry represents it, and the response's
  `token_logprobs`/`logprob` field is still the chosen token's own value
  read from the DATA/ECHO frame directly, not looked up in the table.
- **A filtered stop token's own record is dropped; a reasoning/tool-call
  split's is not.** A matched `stop` sequence is withheld from the
  returned text, and its own logprob record is dropped along with it, but
  chat's `<think>`/answer split and tool-call parsing can still remove or
  rewrite text that a generated-token logprob record continues to
  describe — the two are not realigned in this build.
- **An engine build older than this server's per-token logprobs extension
  is refused, not silently ignored, but only after a bounded wait.** Such
  an engine rejects the whole opted-in request at the wire level in a way
  this server cannot see as a rejection of THIS specific request; after
  `COLI_LOGPROBS_ACCEPT_TIMEOUT` seconds (default 30) with no
  acknowledgment, the request fails with a named 503 rather than hanging.
- **Server-side buffering.** The gateway holds a logprobs-opted-in request's
  full echo table in memory for the whole request lifetime (no streaming is
  allowed together with `logprobs` — the combination is a named 400).

### Tool-calling support

| Engine | OpenAI `tools` | Anthropic `tool_use` | Native format |
|---|---|---|---|
| GLM-5.2 (`colibri`) | yes | yes | `<tool_call>` blocks |
| DeepSeek V4 | yes | yes | native DSML tool-call blocks |
| Inkling | no | no | active tool declarations/choices return HTTP 400 |
| Kimi K3 | yes | yes | native XTML `tools`/`call`/`argument` blocks (#1143) |
| Qwen3.8-Flash-Next | no | no | active tool declarations/choices return HTTP 400 |
| OLMoE | no | no | active tool declarations/choices return HTTP 400 |

On supported engines, pass OpenAI `tools` and optionally `tool_choice` to
`/v1/chat/completions`. The Anthropic endpoint translates `tools`,
`tool_use`/`tool_result`, and the `auto`, `any`, `none`, and forced-tool choice
modes into the active engine's native prompt and back into protocol responses.
Protocol support does not guarantee that every quantized model emits valid
tool syntax; `COLI_TOOL_SALVAGE=1` is an opt-in recovery path for malformed GLM
int4 tool calls. DeepSeek V4 uses its strict native DSML parser instead.

When a reverse proxy or MagicDNS hostname preserves a public `Host` header,
trust that exact hostname with repeatable `--allowed-host` options. The
comma-separated `COLI_ALLOWED_HOSTS` environment variable is equivalent:

```bash
COLI_ALLOWED_HOSTS=llm.example.com ./coli serve --model /nvme/glm52_i4
# or: ./coli serve --model /nvme/glm52_i4 --allowed-host llm.example.com
```

Only configure hostnames or IP addresses you control; there is no wildcard.
This setting extends the DNS-rebinding allowlist and is independent of CORS and
API-key authentication.

Browser access from the Vite development server and Tauri local origins is
enabled by default. Repeat `--cors-origin https://your-ui.example` to allow
another exact origin, or use `--cors-origin '*'` only on a trusted local
network.

The engine owns its KV contexts, so HTTP generation uses a bounded FIFO
admission queue instead of pretending to run unsafe parallel sequences.
Configure it with `--max-queue N` (default 8) and `--queue-timeout SECONDS`
(default 300), or the `COLI_MAX_QUEUE` / `COLI_QUEUE_TIMEOUT` environment
variables. Saturated and timed-out requests receive OpenAI-shaped HTTP 429
errors before streaming headers are sent. `GET /health` exposes
active/queued/completed/rejected counters, and successful generation responses
include `x-colibri-queue-wait-ms`.

## Anthropic-protocol endpoint (`/v1/messages`)

The same server also speaks the **Anthropic Messages API**, so clients that only talk
to Anthropic endpoints — Claude Code, the Anthropic SDKs — work against colibri
without a shim. Nothing to enable: `/v1/messages` is served alongside
`/v1/chat/completions` on the same port.

```bash
curl http://127.0.0.1:8000/v1/messages \
  -H 'x-api-key: local' -H 'content-type: application/json' \
  -d '{"model":"glm-5.2-colibri","max_tokens":128,
       "messages":[{"role":"user","content":"Hello"}]}'
```

For Claude Code, point it at the server and give it any non-empty key:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=local            # only enforced if you set COLI_API_KEY
export ANTHROPIC_MODEL=glm-5.2-colibri
claude
```

Supported on every served architecture: system prompts (string or text blocks),
multi-turn `user`/`assistant` messages, streaming with the full named-event
sequence (`message_start` → `content_block_*` → `message_delta` → `message_stop`,
plus protocol `ping` keepalives during long prefills), `stop_reason`, Anthropic
`usage` field names, and `x-api-key` authentication (`Authorization: Bearer`
also works). The gateway renders each request with the active engine's native
chat template; GLM, Inkling, Kimi K3, Qwen3.8, OLMoE, and DeepSeek V4 prompts are not
interchangeable. Where the engine exposes a reasoning mode, extended thinking
is enabled with `{"thinking": {"type": "enabled"}}` and translated to that
architecture's reasoning protocol; OLMoE disables it explicitly.

Tool use follows the per-engine matrix above. Unsupported engines reject active
tool declarations and choices explicitly instead of feeding another
architecture's markers to an incompatible tokenizer.

Not supported, and refused explicitly rather than ignored: `stop_sequences`,
`top_k`, and non-text content blocks (images, documents). Errors use Anthropic's
own `{"type":"error","error":{...}}` envelope on this path. Architecture-local
features that have not been wired to this protocol are likewise rejected with
an explicit error.

Streaming commits its HTTP 200 only once the engine has accepted the prompt,
the same rule the OpenAI-protocol endpoints follow: a refusal discovered
before acceptance (an oversized prompt over the context limit, say, which
is reported as HTTP 400) surfaces with its own mapped HTTP status in the
Anthropic error envelope, not a committed 200 whose event stream then ends
abruptly. On a healthy engine nothing observable changes — the SSE framing
and event order are exactly as documented above. Until the engine accepts,
no bytes are sent at all — a request queued behind another generation
waits silently, exactly as the OpenAI-style streaming path already does.

> The prefill warning below applies here too, and applies *hardest* to Claude Code:
> its system prompt and tool catalog are large, and on a disk-streaming CPU path
> that is a long silent wait before the first token. Read it before you connect.

## Connect a coding CLI or editor

The API is OpenAI-compatible, so most coding CLIs and editor extensions work by
pointing them at Colibri as an *OpenAI-compatible* provider. Three settings:

- **Base URL** — `http://localhost:8000/v1`
- **Model** — `glm-5.2-colibri` (or whatever you pass to `--model-id`)
- **API key** — any non-empty string, e.g. `local`

Colibri needs **no** API key by default, but many clients refuse to start without
one — give them any dummy value. The key is only enforced if you set `COLI_API_KEY`.

Smoke-test the endpoint first (no key needed unless you set one):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2-colibri","messages":[{"role":"user","content":"hi"}]}'
```

**aider**

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=local
aider --model openai/glm-5.2-colibri     # the openai/ prefix routes to OPENAI_API_BASE
```

**crush** — add a provider to `crush.json` (`~/.config/crush/crush.json`, or
`%USERPROFILE%\AppData\Local\crush\crush.json` on Windows):

```json
{
  "$schema": "https://charm.land/crush.json",
  "providers": {
    "colibri": {
      "name": "Colibri",
      "type": "openai-compat",
      "base_url": "http://localhost:8000/v1/",
      "api_key": "local",
      "models": [
        { "name": "GLM-5.2 (Colibri)", "id": "glm-5.2-colibri",
          "context_window": 131072, "default_max_tokens": 1024 }
      ]
    }
  }
}
```

The `"api_key": "local"` dummy is what satisfies clients that demand a key.
`context_window` is only the client's budget display — set it to whatever your
KV configuration actually allows.

**Continue, Cline / Roo, `llm`, the OpenAI SDKs, …** — set the provider's base
URL to `http://localhost:8000/v1`, the model to `glm-5.2-colibri`, and any dummy
key (`OPENAI_API_KEY` / `OPENAI_BASE_URL` for env-based tools).

> **Set your expectations before connecting an agentic CLI.** Two costs dominate,
> and the first one is invisible until you know it's there:
>
> 1. **Prefill.** Coding agents (crush, aider in repo-map mode, Cline, …) send a
>    large system prompt plus tool definitions — often 10–20k tokens — *before
>    your first word*. Prefill on the CPU-streaming path runs at a few tokens per
>    second (it is attention-bound, see #153), so a 15k-token agent preamble is
>    **an hour of silent "thinking" before the first output token**. The client
>    looks hung; it isn't. Smoke-test with the tiny `curl` above first — if that
>    answers in about a minute, the pipeline works and what you're paying for is
>    prompt size.
> 2. **Decode.** Roughly 1 tok/s for a large model, so multi-turn agent loops
>    (which re-pay the growing context every turn) compound the cost.
>
> Practical guidance: single surgical asks with a short context work; iterative
> agent sessions against a disk-streaming 744B model do not resemble a hosted
> API and mostly won't be worth the wait. If your client lets you trim or disable
> its system preamble and tool catalog, do it.

## Isolated KV contexts

`coli serve --kv-slots N` allocates up to 16 independent sequence contexts.
Requests select one with the optional integer `cache_slot` field; ordinary
OpenAI clients omit it and keep the original slot 0 behavior.

```json
{
  "model": "glm-5.2-colibri",
  "messages": [{"role": "user", "content": "Continue this conversation"}],
  "cache_slot": 1
}
```

Each slot owns its token history, compressed MLA/DSA KV memory, MTP window, and
crash-safe persistence file (`.coli_kv`, `.coli_kv.1`, ...). The engine matches
each request's tokenized prompt against the slot's history and reuses the common
KV prefix, so stateless HTTP turns keep their cache across requests and even
across engine restarts. Use `COLI_KV_SLOTS=N` as the environment equivalent.
Start small: at the default 4096-token context, every slot costs hundreds of MB.

## Web dashboard

One command serves the OpenAI-compatible API **and** the web console on the
same port, then opens your browser when the engine is ready:

```bash
cd web && npm install && npm run build   # once
./coli web --model <model-dir>
```

`coli web` differs from `coli serve` only in opening a browser — both serve the
dashboard on the same port. On a headless host (no display, often no GPU at all)
use `coli serve`, or `coli web --no-browser`, and point a browser at it from
another machine. Nothing in the dashboard needs a desktop session on the host.

What you get:

- **Chat** with live metrics: a flashing token counter while generating, then
  tok/s, time-to-first-token, prompt→completion counts and queue wait;
- **Runtime panel**: your hardware (CPU, GPUs + VRAM, RAM, cores), the
  scheduler, and the live expert-tier bar — how many of the 19,456 experts sit
  in VRAM / RAM / disk right now;
- **Brain**: the whole model as a 76×256 cortex, one cell per expert. Colour =
  tier, brightness = routing heat, and the experts routed in each turn flash
  white and decay — you watch the model think. Hover any cell for its tier,
  heat and [measured topic affinity](https://github.com/JustVugg/colibri/issues/175);
- **Atlas**: the measured expert atlas as a 3-D galaxy (publish `experts.json`
  from `tools/expert_atlas/analyze.py --web`).

The dashboard talks to the engine over a small line protocol and plain JSON
endpoints — nothing heavier than the engine itself. `web/` is a pure OpenAI-API
client (React + TypeScript) and also works against any other compatible
endpoint; the terminal `coli chat` remains the first-class interface.

The layout is responsive down to phone widths, and the sidebar carries the full
telemetry stack — hardware, scheduler, tier bar, per-turn time breakdown, tok/s
trend and per-GPU expert counts:

<p align="center">
  <img src="media/colibri-mobile.png" width="270" alt="the dashboard on a phone-sized viewport" />
  &nbsp;&nbsp;
  <img src="media/colibri-metrics.png" width="300" alt="the telemetry sidebar" />
</p>
