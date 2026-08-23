# CLI & Settings Reference

Command-line settings for the two user-facing programs: the **`coli`** CLI and the **`openai_server.py`** server. The underlying `glm` engine is driven by environment variables — see [ENVIRONMENT.md](ENVIRONMENT.md).

**Updated for the contribution based on `upstream/dev @ 21e7a35`** (argparse definitions in `c/coli` and `c/openai_server.py`). See [MAINTAINING-DOCS.md](MAINTAINING-DOCS.md) to regenerate.

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
| `doctor` | Environment/health check (`--json` report, `--deep` strict preflight). |
| `tune` | Measure and save the fastest quality-preserving execution profile for this machine/model. |
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
| `--cap` | `0` (auto) | `<cap>` argv | Expert-cache cap (starting point; see `CAP_RAISE`). `0` lets the engine pick: `8` historically, `1` on Metal + macOS when the model volume measures fast (F_NOCACHE probe ≥ `COLI_SSD_FAST_GBS`, cached in `<model>/.coli_ssd` — #379). An explicit value always wins. |
| `--ngen` | `1024` | `NGEN` | Max tokens to generate. |
| `--temp` | none (`0`=greedy; engine default 1.0) | `TEMP` | Sampling temperature. |
| `--topp` | `0` | `TOPP` | Top-p filter. |
| `--topk` | `0` | `TOPK` | Top-k filter. |
| `--repin` | `0` | `REPIN` | Re-pin experts every N tokens. |
| `--policy` | `quality` | `COLI_POLICY` | `quality` \| `balanced` \| `experimental-fast`. |
| `--gpu` | `None` | `COLI_GPU(S)` | `auto`, `none`, or a device list like `0,1`. |
| `--vram` | `0` (auto) | CUDA plan | Total VRAM budget in GB. |
| `--auto-tier` | off | resource plan | Automatically apply the RAM/VRAM placement plan. |
| `--no-tune-profile` | off | profile loader | Ignore a saved measured profile. |

### Subcommand-specific flags

**`serve`**

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8000` | Port. |
| `--model-id` | `$COLI_MODEL_ID` or `glm-5.2-colibri` | Model id reported by the API. |
| `--api-key` | `$COLI_API_KEY` | Require this bearer token. |
| `--cors-origin` | none (repeatable) | Allowed CORS origin(s). |
| `--allowed-host` | `$COLI_ALLOWED_HOSTS` or none (repeatable) | Additional Host header accepted by the DNS-rebinding guard. |
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

**`tune`**: `--prompt <text>`, `--tokens 16`, `--repeats 2`,
`--timeout 900`, `--min-gain 0.03`. The command uses fixed-token replay and
only tests quality-preserving execution scheduling.

**`doctor`**: `--deep` strictly checks every safetensors header and tensor
layout, filename-declared shard completeness, required core tensors, an
optional model index, and runtime-equivalent size/header admission for
`COLI_MODEL_MIRROR`. It does not hash tensor payloads or load the engine.

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
| `--allowed-host` | `$COLI_ALLOWED_HOSTS` or none (repeatable) | Additional Host header accepted by the DNS-rebinding guard. |
| `--cap` | `0` (auto) | Expert-cache cap; `0` = engine default (`8`, or `1` on Metal + macOS + fast model volume — #379). |
| `--max-tokens` | `1024` | Default max completion tokens. |
| `--max-queue` | `$COLI_MAX_QUEUE` or `8` | Max queued requests. |
| `--queue-timeout` | `$COLI_QUEUE_TIMEOUT` or `300` | Request queue timeout (s). |
| `--kv-slots` | `$COLI_KV_SLOTS` or `1` | KV conversation slots. |

Tool calling is supported by GLM and DeepSeek V4; Inkling, Kimi K3, and OLMoE reject active tool declarations and choices explicitly. See the [per-engine API matrix](api.md#tool-calling-support). The opt-in `COLI_TOOL_SALVAGE=1` env var recovers malformed GLM int4 tool calls; V4 uses its native DSML parser. Engine-specific runtime variables are listed in [ENVIRONMENT.md](ENVIRONMENT.md); the server passes the environment through to the selected engine.

---

## Flags vs environment variables

A flag and its mapped environment variable are two routes to the same engine knob. Precedence and coverage:

- For knobs with a flag (`--temp`, `--ctx`, `--ram`, `--topk`, `--topp`, `--repin`, `--cap`, `--ngen`, `--policy`), prefer the flag — it's the supported surface.
- For knobs with **no** flag (`COLI_METAL`, `PIPE`, `DIRECT`, `COLI_NO_OMP_TUNE`, `MLOCK`, `CAP_RAISE`, `KVSAVE`, `SEED`, `NUCLEUS`, …), export the environment variable.
- The CLI copies your whole environment through to `glm`, so any variable you export is honored unless a flag explicitly overrides it.
