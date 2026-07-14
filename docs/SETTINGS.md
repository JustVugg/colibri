# CLI & Settings Reference

Command-line settings for the two user-facing programs: the **`coli`** CLI and the **`openai_server.py`** server. The underlying `glm` engine is driven by environment variables — see [ENVIRONMENT.md](ENVIRONMENT.md).

**Generated from `upstream/dev @ 6d3ed7e`** (argparse definitions in `c/coli` and `c/openai_server.py`). See [MAINTAINING-DOCS.md](MAINTAINING-DOCS.md) to regenerate.

---

## `coli` — the CLI

```
coli <subcommand> [flags]
```

Flags may also be given **after** the subcommand. Most flags map onto an engine environment variable before `glm` is launched (see the mapping table at the bottom).

### Subcommands

| Subcommand | Purpose |
|---|---|
| `build` | Build/prepare the engine. |
| `info` | Print model / build info. |
| `plan` | Show the computed RAM/VRAM placement plan (`--json` for machine-readable). |
| `doctor` | Environment/health check (`--json` for a versioned report). |
| `run "<prompt>"` | One-shot generation for the given prompt (positional, may be multi-word). |
| `chat` | Interactive REPL chat. |
| `serve` | Start the OpenAI-compatible HTTP server. |
| `bench [tasks]` | Run benchmark tasks (`--limit`, `--data`). |
| `convert` | Convert an FP8 repo to a colibrì int4 snapshot. |

### Common flags (all subcommands)

| Flag | Default | Maps to | Meaning |
|---|---|---|---|
| `--model` | `$COLI_MODEL` or built-in path | `SNAP` | Model snapshot directory. |
| `--ram` | `0` (auto ≈ 88% free) | `RAM_GB` | RAM budget in GB for the expert working set. |
| `--ctx` | `0` (auto) | `CTX` | Context length. |
| `--cap` | `8` | `<cap>` argv | Expert-cache cap (starting point; see `CAP_RAISE`). |
| `--ngen` | `1024` | `NGEN` | Max tokens to generate. |
| `--temp` | none (`0`=greedy; engine default 1.0) | `TEMP` | Sampling temperature. |
| `--topp` | `0` | `TOPP` | Top-p filter. |
| `--topk` | `0` | `TOPK` | Top-k filter. |
| `--repin` | `0` | `REPIN` | Re-pin experts every N tokens. |
| `--policy` | `quality` | `COLI_POLICY` | `quality` \| `balanced` \| `experimental-fast`. |
| `--gpu` | `None` | `COLI_GPU(S)` | `auto`, `none`, or a device list like `0,1`. |
| `--vram` | `0` (auto) | CUDA plan | Total VRAM budget in GB. |
| `--auto-tier` | off | resource plan | Automatically apply the RAM/VRAM placement plan. |

### Subcommand-specific flags

**`serve`**

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8000` | Port. |
| `--model-id` | `$COLI_MODEL_ID` or `glm-5.2-colibri` | Model id reported by the API. |
| `--api-key` | `$COLI_API_KEY` | Require this bearer token. |
| `--cors-origin` | none (repeatable) | Allowed CORS origin(s). |
| `--max-queue` | `$COLI_MAX_QUEUE` or `8` | Max queued requests. |
| `--queue-timeout` | `$COLI_QUEUE_TIMEOUT` or `300` | Seconds a request may wait. |
| `--kv-slots` | `$COLI_KV_SLOTS` or `1` | Independent KV conversation slots (→ `KV_SLOTS`). |

**`convert`**

| Flag | Default | Meaning |
|---|---|---|
| `--repo` | `zai-org/GLM-5.2-FP8` | Source FP8 repo. |
| `--ebits` | `4` | Streamed-expert bit width. |
| `--io-bits` | `8` | Resident (attention/dense/embed) bit width. |
| `--xbits` | `0` | Extra/override bit width. |
| `--no-mtp` | off | Skip the MTP speculative-draft head. |

**`bench`**: `[tasks...]` (positional), `--limit 40`, `--data <bench dir>`.
**`plan` / `doctor`**: `--json`.

---

## `openai_server.py` — the HTTP server

Run directly (or via `coli serve`). OpenAI-compatible `/v1/chat/completions`.

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `$COLI_MODEL` (required if unset) | Model snapshot directory. |
| `--engine` | `./glm` | Path to the engine binary. |
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8000` | Port. |
| `--model-id` | `$COLI_MODEL_ID` or `glm-5.2-colibri` | Model id in API responses. |
| `--api-key` | `$COLI_API_KEY` | Required bearer token. |
| `--cors-origin` | none (repeatable) | Allowed CORS origin(s). |
| `--cap` | `8` | Expert-cache cap. |
| `--max-tokens` | `1024` | Default max completion tokens. |
| `--max-queue` | `$COLI_MAX_QUEUE` or `8` | Max queued requests. |
| `--queue-timeout` | `$COLI_QUEUE_TIMEOUT` or `300` | Request queue timeout (s). |
| `--kv-slots` | `$COLI_KV_SLOTS` or `1` | KV conversation slots. |

Tool calling (`tools` in the request) is supported; the opt-in `COLI_TOOL_SALVAGE=1` env var recovers malformed int4 tool calls. Server-relevant env vars: `COLI_METAL`, `PIPE`, `DIRECT`, `COLI_NO_OMP_TUNE`, `RAM_GB`, `CTX`, `KVSAVE` (all from [ENVIRONMENT.md](ENVIRONMENT.md)) apply because the server launches the same `glm` engine.

---

## Flags vs environment variables

A flag and its mapped environment variable are two routes to the same engine knob. Precedence and coverage:

- For knobs with a flag (`--temp`, `--ctx`, `--ram`, `--topk`, `--topp`, `--repin`, `--cap`, `--ngen`, `--policy`), prefer the flag — it's the supported surface.
- For knobs with **no** flag (`COLI_METAL`, `PIPE`, `DIRECT`, `COLI_NO_OMP_TUNE`, `MLOCK`, `CAP_RAISE`, `KVSAVE`, `SEED`, `NUCLEUS`, …), export the environment variable.
- The CLI copies your whole environment through to `glm`, so any variable you export is honored unless a flag explicitly overrides it.
