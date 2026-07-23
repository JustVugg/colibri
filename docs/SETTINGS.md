# CLI & Settings Reference

Command-line settings for the two user-facing programs: the **`coli`** CLI and the **`openai_server.py`** server. The underlying Colibri engine is driven by environment variables — see [ENVIRONMENT.md](ENVIRONMENT.md).

**Updated for the contribution based on `upstream/dev @ 21e7a35`** (argparse definitions in `c/coli` and `c/openai_server.py`). See [MAINTAINING-DOCS.md](MAINTAINING-DOCS.md) to regenerate.

---

## `coli` — the CLI

```
coli <subcommand> [flags]
```

Flags may also be given **after** the subcommand. Most flags map onto an engine environment variable before Colibri is launched (see the mapping table at the bottom).

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
| `stop` | Stop a server on the selected port. |
| `ramdisk` | Open the Linux NUMA-aware RAM-disk TUI or run a scriptable lifecycle action. |
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

**`ramdisk`** (Linux only)

Run `coli ramdisk` in a terminal for the guided Textual server console, or select
one of the equivalent scriptable actions. Source and flat-archive users can
install the supported interface with
`python3 -m pip install -r c/requirements-tui.txt` from a source checkout, or
`python3 -m pip install -r requirements-tui.txt` from an unpacked release
archive. `COLI_RAMDISK_UI=auto` is the default and falls back to the
one-release curses interface only when Textual is unavailable;
`COLI_RAMDISK_UI=textual` or `curses` selects one explicitly.

The console walks through Inspect → Placement → Capacity → Runtime → Review →
Operate. A live backplane rail always shows the resulting copy count, memory
nodes, engine count, and ports. The normal `interleaved` plan is shown as **one
shared model copy and one engine** whose RAM pages span the selected NUMA nodes.
Replica placement cannot be enabled by an accidental TUI toggle; request
`--topology per-node` explicitly to review its full-copy and endpoint multiplier.
Memory-node and whole-core CPU range lists are validated against the invoking
process's effective cgroup/cpuset masks. Once prepared, the persisted placement
is shown as the active deployment and its weight settings stay locked until it
is destroyed. Mount policies use Linux static-node IDs, and Start/Benchmark
refuse to run if the effective CPU or memory-node masks changed after Prepare.
Capacity admission is also cgroup-aware: on cgroup v2 it uses the tightest
`memory.max - memory.current` headroom across the current cgroup and every
limiting ancestor; cgroup v1 memory limits receive the equivalent check.
`memory.high` is reported as a reclaim/throttling warning rather than a hard
limit. Host/NUMA availability and cgroup headroom are both enforced, using the
smaller value.

Linux memory policy addresses NUMA nodes, not individual DIMMs or channels.
DIMM/channel data is therefore informational: populate symmetric firmware-
recommended DIMM/channel pairs, then use `--memory-nodes` to select the NUMA
domains they expose. Colibri never rewrites the host-global weighted-interleave
weights.
Prepare, Start, and Benchmark remain navigable while they run and can be
cancelled with `c`; Colibri waits for a rollback/cleanup checkpoint before
returning control. `q` requests the same safe cancellation and exits only after
cleanup succeeds; cleanup failures remain visible in the TUI.

```sh
coli ramdisk plan --model /models/glm --memory-nodes 0,2 --cpu-list 0-31,64-95 --json
coli ramdisk prepare --model /models/glm --yes
coli ramdisk start --base-port 8000
coli ramdisk status --json
coli ramdisk benchmark --json
coli ramdisk stop
coli ramdisk destroy --yes
```

`benchmark` requires managed engines to be stopped so SSD, tmpfs/slab, direct
RAM-map, and per-node aggregate runs all use the same deterministic controls.
The best safe I/O knobs are saved per topology and reused by `start`; managed
engines use exactly the reviewed whole-core CPU masks. The engine preserves
those masks through its OpenMP self-reexec instead of widening back to every
online CPU.
Partial plans also show the profile coverage of shard-closure staging beside a
same-budget hot-expert `PIN` estimate, making shard-granularity waste explicit.
System scorecards report mount-specific tmpfs allocation; tmpfs does not expose
per-mount THP counters, so huge-page coverage is labeled as host-global.
Saved benchmark reports include the source revision and dirty-tree state, exact
command, hardware and storage details, warm-up count, measured-run count, and
median throughput required for reproducible comparisons.
Linux physical-read accounting carries an explicit validity bit: unavailable
`/proc/self/io` data is shown as `n/a` and cannot satisfy the full-mode
zero-SSD-read acceptance check.

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `full` | `full` copies every shard; `partial` requires a compatible usage profile and stages complete shard closures. |
| `--topology` | `interleaved` | One shared copy/engine, or `per-node`: one complete staged copy and independent engine per NUMA node (replication, not sharding). |
| `--memory-nodes` | effective CPU-bearing NUMA nodes | Linux NUMA range list such as `0-3,8`. Shared mode requests equal interleave over a multi-node mask and a strict bind for one selected node, then verifies initial page placement; per-node mode creates replicas only on these nodes. |
| `--cpu-list` | effective CPUs on selected nodes | Linux CPU range list for managed engines. Selections must contain whole effective physical-core sibling groups. Shared plans flag CPUs outside the memory-node mask as intentional remote access; replica plans reject them. |
| `--capacity-gb` | full model size | Staging budget; required and strictly enforced in partial mode. |
| `--profile` | `<model>/.coli_usage` | Explicit compatible text/JSON expert-usage profile for partial mode. |
| `--mount-root` | `/mnt/colibri-ram` | Managed tmpfs mount root. V1 accepts only non-symlink paths below `/mnt`; existing paths must be empty and not writable by the invoking user. |
| `--allow-swappable` | off | Explicitly permit tmpfs without `noswap` on older kernels; Colibri never runs `swapoff`. |
| `--thp` | `auto` | tmpfs THP policy: `auto` prefers `within_size`, or select `within_size`/`advise` explicitly. Unsupported `within_size` mounts retry with `advise`. |
| `--prefault` | `1` for managed full mode, otherwise `0` | Prefault direct mappings at engine startup. |
| `--parallel` | `2` | Bounded shard-copy worker count. |
| `plan/status/benchmark --json` | off | Emit a versioned machine-readable report. |
| `prepare/destroy --yes` | off | Confirm reviewed mount or cleanup work in non-interactive scripts. |
| `start --base-port` | prepared value (`8000` initially) | Interleaved port, or base plus NUMA node id for replicas; omitted restarts preserve the previous value. |

Planning and status are unprivileged. In the TUI, preparation and destruction
validate sudo once in the foreground; their background workers then use
non-interactive sudo only for the exact `mount`/`umount` commands, so a password
prompt can never collide with the interface. Colibri verifies that the ticket
is reusable before mounting and refreshes it without prompting while a long
copy is active, preserving rollback authority. `SIGHUP`/`SIGTERM` request
cooperative rollback for Prepare/Start/Benchmark; Stop/Destroy finish their
verified cleanup transaction before returning the signal exit code. Model files and durable
`.coli_usage`/`.coli_kv*` state remain on SSD and are never modified by staging.
Linux interleave can fall back to other allowed nodes under severe pressure;
Colibri reduces that risk with an admission reserve and rejects staging when its
post-copy page sample materially escapes the reviewed mask. A cgroup/cpuset remains the
hard host-level boundary when strict isolation is required.
Set `XDG_STATE_HOME` to an absolute SSD-backed path: tmpfs/ramfs state locations
and locations beneath the weight mount are rejected. `X-mount.mkdir=0755` may
leave an empty root-owned directory below `/mnt` after unmount; the mounted tmpfs
itself is still private (`mode=0700`) and that empty directory is safe to reuse.

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
| `--engine` | `./colibri` | Path to the engine binary. |
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

Tool calling (`tools` in the request) is supported; the opt-in `COLI_TOOL_SALVAGE=1` env var recovers malformed int4 tool calls. Server-relevant env vars: `COLI_METAL`, `PIPE`, `DIRECT`, `COLI_NO_OMP_TUNE`, `RAM_GB`, `CTX`, `KVSAVE` (all from [ENVIRONMENT.md](ENVIRONMENT.md)) apply because the server launches the same Colibri engine.

---

## Flags vs environment variables

A flag and its mapped environment variable are two routes to the same engine knob. Precedence and coverage:

- For knobs with a flag (`--temp`, `--ctx`, `--ram`, `--topk`, `--topp`, `--repin`, `--cap`, `--ngen`, `--policy`), prefer the flag — it's the supported surface.
- For knobs with **no** flag (`COLI_METAL`, `PIPE`, `DIRECT`, `COLI_NO_OMP_TUNE`, `MLOCK`, `CAP_RAISE`, `KVSAVE`, `SEED`, `NUCLEUS`, …), export the environment variable.
- The CLI copies your whole environment through to Colibri, so any variable you export is honored unless a flag explicitly overrides it.
