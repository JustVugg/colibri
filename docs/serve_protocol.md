# The serve protocols — engine ⇄ server wire format

The engine speaks two line-oriented protocols over stdin/stdout. Both are plain text
plus byte-counted payload frames; every outbound line is written with a trailing
`fflush`, one line per write. On Windows both ends of the pipe are switched to binary
mode at startup — the CRT's CRLF translation otherwise corrupts the sentinels and
stalls byte-counted reads (#195).

| protocol | entry | selected by | used by |
|---|---|---|---|
| **mux** (continuous batching, up to 16 KV slots) | `run_serve_mux` | `SERVE_BATCH=1` | `openai_server.py`, `coli web` |
| **legacy** (single slot, interactive) | `run_serve` | `SERVE=1` (without `SERVE_BATCH`) | `coli chat` |

This document is the reference for the **mux** protocol; the legacy protocol is
summarized at the end. Line formats below are quoted from the emitting `printf`s in
`glm.c` — if this document and the code disagree, the code wins and this file needs a PR.

## Startup handshake (engine → server)

```
\x01\x01READY\x01\x01
CAPS vision=<0|1>
STAT 0 0.00 0.0 <rss_gb>
HWINFO <cores> <ram_total_gb> <ram_avail_gb> <ngpu> <vram_total_gb> <cpu_name>|<gpu_name>
TIERS <vram_experts> <ram_experts> <disk_experts> <vram_gb> <ram_gb>
EMAP <rows> <cols> <hex>
```

The server must not send requests before `READY`. `HWINFO`/`TIERS`/`EMAP` are
telemetry (see below) and may grow — **servers must ignore line kinds they do not
recognize**; that is the protocol's forward-compatibility rule.

`CAPS key=value ...` is optional and sits *between* `READY` and `STAT`: it says what
the engine actually loaded, and the server reads it while waiting for `STAT`, so the
served modalities are known before the first request. Today the one key is `vision`
(`1`: a vision tower is loaded; `0`: none — a text-only checkpoint, or a config that
declares a tower whose tensors are not in the container). Engines that say nothing
leave the server's flag unknown. glm53, qwen38, qwen36 and deepseek_v41 emit it;
`openai_server.py` derives `/v1/models` `input_modalities` from it.

Two more keys say what kind of engine answers. `decide=1`: the engine takes
`DECIDE` (below), and the gateway sends `POST /v1/systemone` to it as one record
instead of scoring options through the logprob channel. `chat=0`: it has nothing
else, so the generating endpoints answer 400 with a pointer to `/v1/systemone`.
A decision engine (laya, gliner_decide) says `decide=1 chat=0`; an engine that chats and also
decides natively says `decide=1` alone (qwen36 with Clef's head). `decide_record=raw`
asks for the record in its raw form, the caller's own values (docs/systemone.md,
"Decision engines").

An engine that cannot load its model says why on this channel, instead of
`READY`, and exits:

```
LOAD_FAIL kind=<nomem|io|format|unsupported> <detail>
```

It is the last line such an engine writes. `nomem`: the host refused memory
(ENOMEM/EAGAIN from malloc, mmap or mlock); `io`: the file could not be reached or
read (ENOENT, EACCES, EIO, ...); `format`: the file was read but is not what it
claims (short read, bad header); `unsupported`: well-formed, but a dtype or geometry
this engine does not serve. The detail is the same text the engine wrote to
stderr. `openai_server.py` raises `EngineLoadError(kind, detail)` from it and logs
`[gateway] engine load failed: kind=<kind> <detail>`; an engine that exits with no
such line is still reported as "colibri engine exited unexpectedly". The shared
safetensors path (`st.h`, every family) emits it through `coli_load_fail()`
(`load_fail.h`); a family's own load-time refusals adopt the same call as they are
touched.

## Requests (server → engine)

```
SUBMIT <id> <slot> <bytes> <max_tokens> <temperature> <top_p>\n<payload>\n
IMAGE <id> <bytes> <grid_h> <grid_w>\n<payload>\n
STOP <id>\n
CANCEL <id>\n
DECIDE <id> <slot> <bytes>\n<payload>\n
```

- `id` — non-zero u64, unique among in-flight requests.
- `slot` — KV slot index, `0 … KV_SLOTS-1` (`KV_SLOTS` env, 1–16, default 1). A slot
  holds one conversation's KV; the engine matches the tokenized payload against the
  slot's history and reuses the common prefix (truncate-and-extend), so stateless
  HTTP turns keep their cache. With `KV_SLOTS` above 1 every text engine takes a
  `SUBMIT` on a free slot while others decode: its prompt is prefilled at once, and
  from then on each decode step runs one row for every active request. The frames of
  the requests interleave, each tagged with its `id`. A `SUBMIT` on a slot whose
  request is still running gets `ERROR <id> SLOT_BUSY`; a slot out of range,
  `ERROR <id> invalid cache slot`.
- `bytes` — exact byte length of `payload` (UTF-8, may contain newlines). The engine
  reads exactly that many bytes after the header line, then one trailing `\n`.
- `payload` — the fully rendered prompt (the server owns the chat template).
- `IMAGE` (GLM-5.3-Flash) announces the pre-extracted patches of one image for
  the request with the same `id`, and must arrive immediately before its
  `SUBMIT`. `payload` is `bytes` of little-endian f32 in the tower's patch
  order; the server owns the preprocessing (`c/tools/glm53_image.py`). The
  prompt must already contain one `<|image|>` placeholder per output token,
  i.e. `(grid_h/merge) x (grid_w/merge)` of them: the engine refuses a mismatch
  rather than answering about a different picture. An engine holds one pending
  image and discards an earlier one.
- `STOP` ends generation through the normal successful `DONE` path. Statistics,
  usage history, and KV state are persisted; the HTTP gateway uses it after a
  client-provided stop sequence matches.
- `CANCEL` aborts a request after its client disconnects and returns `CANCELLED`.
- `DECIDE` (only to an engine that announced `decide=1`) carries one decision
  record, UTF-8 JSON: the state and the typed questions with their options in
  order. The engine answers with one `DECISION` frame and `DONE`, or with
  `ERROR <id> DECIDE_INVALID <reason>` for a record it refuses (the client's
  422; the reason starts with the field, `questions.<id>: ...`) and
  `ERROR <id> DECIDE_FAILED <reason>` for its own failure. The record and the
  answer are specified in [docs/systemone.md, Decision engines](systemone.md#decision-engines);
  `c/decide_serve.h` parses one and writes the other.
- EOF on stdin = graceful shutdown: in-flight requests finish first.

Prefill is serial; decode is continuously batched — every active slot contributes
one row per forward.

## Responses (engine → server)

Per request, in order:

```
DATA <id> <n>\n<n bytes of UTF-8>\n        # a decoded token's text; repeated
TOOL <id> <n>\n<n bytes of UTF-8>\n        # engine-authenticated tool structure; Kimi K3 only
TOPK <id> 5 <logprob> <hextext> ... ×5     # candidates for the sampled token (SERVE_TOPK=1)
HITS <rows> <cols> <hex>                   # ~every 6 tokens: routed-expert bitmap since last HITS
REPIN <layer> <eid> <old_tier> <gpu>       # live re-pin swap events, as they happen
...
DONE <id> STAT <emitted> <tok_s> <hit_pct> <rss_gb> <prompt_tokens> <length_limited>
```

and for `DECIDE`:

```
DECISION <id> <n>\n<n bytes of JSON>\n    # once
DONE <id> STAT 0 <tok_s> 0.0 <rss_gb> <tokens_read> 0
```

Kimi K3 emits a zero-byte `TOOL` frame immediately after `ACCEPT` for every
chat request. That frame declares the sideband authoritative even when no tool
call follows. Real K3 special-token structure and its enclosed tool payload use
subsequent `TOOL` frames; ordinary decoded text remains on `DATA`, so text that
only resembles an XTML marker cannot be promoted into a client tool call.

Errors replace the stream: `ERROR <id> <CODE>` with codes `BAD_FRAME`, `BAD_REQUEST`,
`SLOT_BUSY`, `DUPLICATE_ID`, `EMPTY_PROMPT`, `NOT_FOUND` (CANCEL of unknown id),
`CANCELLED`, and for decision engines `DECIDE_INVALID`, `DECIDE_FAILED` and
`NOT_SUPPORTED` (a `SUBMIT` to an engine that does not generate). A `CANCEL` is acknowledged by `ERROR <id> CANCELLED` after the slot's KV
is persisted.

Immediately before each `DONE` the engine emits a telemetry block for the finished
turn: `HWINFO`, `PERF`, `ENTROPY`, `GPUS`, `TIERS`, `EMAP`, `HITS` (formats below).
`.coli_usage` is persisted at every turn end, not only at exit.

## Telemetry lines

| line | format | meaning |
|---|---|---|
| `TIERS` | `TIERS <vram> <ram> <disk> <vram_gb> <ram_gb>` | expert count per tier + resident bytes |
| `HWINFO` | `HWINFO <cores> <ram_total> <ram_avail> <ngpu> <vram_total> <cpu>\|<gpu>` | host snapshot (GBs are floats) |
| `CAPS` | `CAPS key=value ...` | handshake only, between `READY` and `STAT`: what the engine loaded (`vision=0\|1`) |
| `LOAD_FAIL` | `LOAD_FAIL kind=<kind> <detail>` | handshake only, instead of `READY`: why the model did not load (`nomem`, `io`, `format`, `unsupported`); the engine exits after it |
| `EMAP` | `EMAP <rows> <cols> <hex>` | one byte per expert, row-major over `rows×cols` (sparse layers +MTP × experts): `byte = (tier<<6) \| heat` — 2-bit tier (0 disk / 1 RAM / 2 VRAM), 6-bit log₂-bucketed usage heat |
| `HITS` | `HITS <rows> <cols> <hex>` | 1 bit per expert, experts routed since the previous `HITS` |
| `PERF` | `PERF <id> <dt> <t_edisk> <t_ewait> <t_emm> <t_attn> <t_kvb> <t_head>` | this turn's PROFILO deltas, seconds |
| `ENTROPY` | `ENTROPY <h0> <h1> …` | per-sparse-layer routing entropy of the turn, bits |
| `GPUS` | `GPUS <n> (<used_gb> <total_gb> <experts>)×n` | per-device VRAM + resident expert count (CUDA builds) |
| `TOPK` | `TOPK <id> 5 (<logprob> <hextext>)×5` | token text hex-encoded so the line stays line-shaped |
| `REPIN` | `REPIN <layer> <eid> <old_tier> <gpu>` | one line per hot-store swap (`REPIN=n` mode) |

All telemetry is advisory: servers render what they know and skip the rest.

## HTTP surface (`openai_server.py`)

- `POST /v1/chat/completions` — OpenAI-compatible; streaming responses emit one extra
  SSE frame `data: {"colibri": {stats, perf, topk, entropy, gpus, repin}}` immediately
  before `data: [DONE]`; non-streaming responses attach the same object as a
  `"colibri"` field.
- `GET /experts` — the latest `EMAP`/`HITS` state: `{rows, cols, map, hits, seq,
  gpus, entropy, repin}`.
- `GET /*` — static hosting of `web/dist` (SPA fallback, path-traversal-safe), plus
  `experts.json` if published there (the measured expert atlas, #175/#218).

## Legacy protocol (`run_serve`, `coli chat`)

Interactive lines are prompts; control frames: `\x02RESET` (clear history),
`\x02MORE` (continue an NGEN-truncated answer), and
`\x02PROMPT <bytes> <max_tokens> <temperature> <top_p> [kv_slot]\n<prompt>\n`
(the pre-mux API mode). Responses are raw text terminated by `\x01\x01END\x01\x01`
plus a `STAT` line. New integrations should use the mux protocol.
