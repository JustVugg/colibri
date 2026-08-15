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
| `ramdisk` | Run the headless Linux NUMA-aware RAM-workspace lifecycle. |
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

### `ramdisk` (Linux only)

The RAM-workspace control plane exposes exactly these actions:

| Action | Contract |
|---|---|
| `plan --json` | Emit the exact staging/reserve plan plus its `plan_token`; returns 2 when the plan has blockers. |
| `stage --plan-token TOKEN --yes --json` | Mount, stage, and validate only the reviewed plan; returns `colibri.ramdisk.stage.v1`. |
| `prepare ...` | Exact alias of `stage`, including token enforcement and JSON schema. |
| `verify --json` | Deeply revalidate source identity, mounts, namespaces, and processes; returns 2 when absent or unverified. |
| `status --json` | Observational status; includes a `deployment_token` when a manifest is present. |
| `status --runtime --json` | Add advisory serving/GPU telemetry after revalidating the same deployment snapshot. |
| `start [--base-port PORT] --deployment-token TOKEN --yes --json` | Start contained managed engines; omission of `--base-port` reuses the prepared deployment's persisted port. Returns `colibri.ramdisk.start.v1` and a fresh token. |
| `stop --deployment-token TOKEN --yes --json` | Stop contained engines and merge exact usage transactions; returns `colibri.ramdisk.stop.v1` and a fresh token. |
| `benchmark ... --json` | Run the fixed causal RAMMAP protocol with append-only evidence while managed engines are stopped. |
| `destroy --deployment-token TOKEN --yes --json` | Stop internal recovery work as needed and unmount only the reviewed deployment; returns `colibri.ramdisk.destroy.v1`. |

Tokens are lowercase 64-character SHA-256 identities. Missing, malformed, or
preflight-stale tokens fail before creating lifecycle state. If a reviewed plan
changes after preflight, the serialized under-lock check still refuses it
before mounts, processes, or manifest writes; the deliberately durable
lifecycle lock may remain as synchronization metadata. `prepare`/`stage` bind
model identity, hardware-mask provenance, placement, selected shards,
reserves, mounts, runtime knobs, accelerator policy, and preset. Deployment
tokens used by `start`, `stop`, and `destroy` additionally bind the deployment,
mount identities, process identities, state, and persisted endpoint.

The `start.v1` success object has exactly `schema`, `version`, `state`,
`deployment_id`, `deployment_token`, `ports`, `endpoints`,
`containment_mode`, `usage_merge_summary`, and `recovery_attention`.
The `stop.v1` object replaces `ports` and `endpoints` with `stopped_count`.
Usage-merge fields are counts for that invocation, not lifetime totals.
`containment_mode` is either `cgroup-v2` or `legacy-process-group`.

```sh
coli ramdisk plan --model /models/glm --memory-nodes 0,2 \
  --cpu-list 0-31,64-95 --json
coli ramdisk stage --model /models/glm \
  --plan-token <token-from-plan> --yes --json
coli ramdisk verify --json
coli ramdisk status --json
coli ramdisk start --deployment-token <token-from-status> --yes --json
coli ramdisk stop --deployment-token <token-from-start> --yes --json
coli ramdisk destroy --deployment-token <token-from-stop-or-status> --yes --json
```

Common planning flags are `--mode full|partial`,
`--topology interleaved|per-node`, `--memory-nodes`, `--cpu-list`,
`--capacity-gb`, `--profile`, `--mount-root`, `--allow-swappable`, `--thp`,
`--prefault`, `--parallel`, `--ctx`, `--gpu`, and `--gpu-layout`. Planning,
status, and verification are unprivileged. Staging and destruction use only
the narrowly reviewed mount/unmount privilege path. A bare `coli ramdisk`
prints this headless command surface and exits 2.

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

Tool calling (`tools` in the request) is supported; the opt-in `COLI_TOOL_SALVAGE=1` env var recovers malformed int4 tool calls. Server-relevant env vars: `COLI_METAL`, `PIPE`, `DIRECT`, `COLI_NO_OMP_TUNE`, `RAM_GB`, `CTX`, `KVSAVE` (all from [ENVIRONMENT.md](ENVIRONMENT.md)) apply because the server launches the same `glm` engine.

---

## Flags vs environment variables

A flag and its mapped environment variable are two routes to the same engine knob. Precedence and coverage:

- For knobs with a flag (`--temp`, `--ctx`, `--ram`, `--topk`, `--topp`, `--repin`, `--cap`, `--ngen`, `--policy`), prefer the flag — it's the supported surface.
- For knobs with **no** flag (`COLI_METAL`, `PIPE`, `DIRECT`, `COLI_NO_OMP_TUNE`, `MLOCK`, `CAP_RAISE`, `KVSAVE`, `SEED`, `NUCLEUS`, …), export the environment variable.
- The CLI copies your whole environment through to `glm`, so any variable you export is honored unless a flag explicitly overrides it.
