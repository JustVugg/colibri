# Changelog

All notable changes to colibrì are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [1.6.2] — 2026-08-14
Security release: **six privately-reported memory-safety issues fixed**, all reachable
from attacker-controlled input (malicious model file / `config.json`, or the kimi_k3
SERVE stdin). Every fix validates at the trust boundary — no behavioural change on
well-formed models or requests. Advisories: GHSA-gf38-c8fx-ppvv (kimi_k3 SERVE OOB
write), GHSA-2qrj-xjmh-mv74 (json.h OOB read), GHSA-w696-h9p7-6rgc (inkling audio
OOB), GHSA-7654-r78q-vc3r (deepseek_v4 indexer OOB R/W), and two more in the same
class. See the GitHub Security Advisories for details.

## [1.6.1] — 2026-08-13
- `--allowed-host '*'` lets an operator reach a public bind deliberately (#990, #993)
- OLMoE streaming no longer drops the answer into the reasoning channel (#984, #985)
- chat reasoning-channel fixes and pty test hardening (#980)

## [1.6.0] — 2026-08-12
**If you are on v1.5.0, update:** it shipped a performance regression on GLM-5.2
(#856), left up to ~60 GB of RAM unused with a 13-point expert hit-rate loss (#885),
and broke Kimi K3 outright on some machines (#888).
- #869 — the planner priced every expert row at the container's *widest* width;
  mixed-width containers had their cache silently halved
- #914 (bherald) — pin budgets accounted correctly
- prefill batch-union: each distinct expert is read **once** per chunk

## [1.5.0] — 2026-08-05
51 pull requests from 15 contributors.
- **Fifth engine: DeepSeek V4 Flash** (@DrewZt, #165) — MLA + DSA sparse attention,
  43 layers, 256 routed experts + 1 shared, top-6; official checkpoint streams with
  no conversion (fp4 experts, fp8-e4m3 dense with UE8M0 block scales)
- #839 — the rows16 fp4 fast path no longer requires AVX-512: consumer Intel/AMD
  CPUs since Alder Lake get the fast path

## [1.4.0] — 2026-08-01
162 commits, 29 pull requests (25 from contributors).
- **Release archives now contain every engine** — v1.3.0 archives shipped `c/colibri`
  alone while the README promised four families (#720); `inkling` and `kimi_k3` are
  built, packaged and smoke-tested per platform
- Third GPU backend lands

## [1.3.0] — 2026-07-29
Three MoE families on one engine, 744B → 2.8T.
- **Kimi K3** (#676) — 2.8T/104B active: KDA + gated-NoPE-MLA + AttnRes + LatentMoE,
  streams Moonshot's QAT MXFP4 experts straight from the original HF shards
- **Inkling** — a 975B model answers on a 25 GB machine

## [1.2.0] — 2026-07-28
- GB10 / DGX Spark unified-memory OOM fix (#653); AMD/ROCm recognized by `doctor`
  and `resource_plan` (#662, #663)
- AVX2 `matmul_e8` — fmt=6 was 92% of decode on the scalar kernel (#654); native
  SIMD fmt=6 encoder, 15× over numpy
- chat stop-set fix (#633/#381), async packed-int4 parity (#632), stable KV slot
  per conversation (#634, #639)

## [1.1.1] — 2026-07-23

A same-day patch release. **Windows users on v1.1.0 should upgrade**: Microsoft
Defender flags the v1.1.0 Windows binary, and the cause was ours.

### Fixed

- **107 KB of zeros were shipped inside every binary** (#527, #532) — and that is
  what antivirus ML heuristics were reacting to. `static GrDraft g_grd={.max=24};`
  looks harmless, but `GrDraft` is ~107 KB (the grammar's 1024 static rules plus the
  PDA walker) and **any** initializer moves the whole struct out of `.bss` and into
  `.data`, writing 106,848 bytes of near-zero-entropy data into the file — in a
  *writable* section, which is the classic shape of an unpacking buffer for a packed
  payload. Section forensics against v1.0.0 (clean on the same Defender definitions)
  isolated it: identical toolchain, identical PE layout, `.data` 1,840 → 108,752
  bytes. A Windows build with the fix scans clean where v1.1.0 does not. Every
  platform's binary also gets smaller: the Linux engine drops 474,904 → 368,016
  bytes, **-22.5%**.
- **`python3 openai_server.py` was broken on a clean checkout** (#526) — the gateway
  still looked for an engine named `glm` after the #391 rename. It resolves
  `colibri`/`colibri.exe` first now, falling back to `glm` for older trees.
  `coli serve` was unaffected. Spotted by @RDouglasSharp while debugging #488.

### Added

- **Anthropic Messages API** on `/v1/messages` (#343, #525) — clients that only speak
  to Anthropic endpoints, Claude Code above all, now work against colibri with no
  shim: same port, nothing to enable. It is a translation layer over the same
  generation path, so tools, streaming and the KV cache behave exactly as on
  `/v1/chat/completions`. Covers system prompts, `text`/`tool_use`/`tool_result`
  blocks, `input_schema` tools, every `tool_choice` mode, the full named-event SSE
  sequence with protocol `ping` keepalives, `stop_reason` mapping, extended thinking,
  and `x-api-key` auth (`Bearer` still works). `stop_sequences`, `top_k` and non-text
  blocks are refused explicitly rather than silently ignored.
- **`SHA256SUMS.txt` published with every release** (#530) — verify a download is
  exactly what CI built from the tagged source.
- **The Windows engine is uploaded as a CI artifact** (#532) — an antivirus report can
  now be verified on a pull request instead of only after a release is published.

### Changed

- **Docs: "Get started" now starts by getting the program** (#521). The README told
  newcomers to download the 372 GB model *before* it told them how to obtain colibri —
  and for Linux/macOS it never told them at all. New order: get colibri (prebuilt
  archive or build from source) → get the model (372 GB stated up front) → run it.
  The obsolete "rename the engine to `glm.exe`" step is gone; archives have shipped a
  plainly-named `colibri.exe` since #508. Applied in all four languages.

## [1.1.0] — 2026-07-22

A community release. 27 pull requests from more than 20 contributors, 216 commits since
v1.0.0. Most of what follows was found, measured, or fixed by people who do not work on
this project and had nothing to gain from it.

### Added

- **AMD GPU support (HIP/ROCm)** (#339) — single-source `backend_gpu_compat.h` with a WMMA
  dispatch gate, so one codebase builds for CUDA and HIP. Validated on an RX 9070 XT
  (RDNA4, ROCm 7.2): token-exact against CPU on a real fmt=4 gs64 container, with resident
  dense *and* with routed experts in VRAM, plus a fail-injection control proving the GPU
  actually executed the work.
- **Dual-SSD streaming** (`COLI_MODEL_MIRROR`, #421) — read the model from two drives at
  once, roughly doubling streaming bandwidth on a disk-bound host.
- **N-drive shard split** (`COLI_MODEL_DIRS`, #469) — capacity aggregation: run a container
  no single drive can hold, spread across several with no duplication.
- **fmt=5 (int3-g64)** (#168) — 3-bit weights with per-64 group scales: measured 3.3x lower
  outlier-row error than per-row int4 at 25% fewer bytes.
- **fmt=6 (E8/IQ3 lattice)** (#465) — CPU decode kernel and dispatch; index codec tooling (#458).
- **`tools/try_tool_calling.py`** — dependency-free two-turn tool-calling probe that doubles
  as a smoke test.

### Fixed

- **Tool calling in coding clients** (#401), root cause found, in two parts:
  - **#506** — the engine capped prompt encoding at `CTX-2`, and the tokenizer stops dead at
    its limit *without reporting anything*. A prompt longer than the context was therefore
    silently truncated to its first `CTX-2` tokens and answered anyway. With the 4096 default
    that is 4094 — exactly the `prefill 4094` in the field report. The dropped tail was the
    tool instructions and the user's actual turn, so the model emitted a bare `<` and stopped;
    and because clients append to the *end* while truncation keeps the *head*, every retry
    re-sent a byte-identical prompt. Now refused with a 400 `context_length_exceeded`.
  - **#505** — a tool call whose closing `</tool_call>` never arrived was dropped whole,
    because the parser required both tags. Now recovered when unambiguous, on both the
    streamed and non-streamed paths.
  - **#437** — non-EOS role markers were armed as hard stops in serve mode and cut generation
    the instant a tool block started.
- **Grouped-int4 (fmt=4) produced garbage output on CUDA** with `CUDA_DENSE=1` (#298) — the
  dense and attention kernels applied per-group scales as if they were per-row. Hardware-verified.
- **OpenMP tuning re-exec preserved the CPU affinity mask** (#476), jailing every thread onto
  one core when `OMP_PROC_BIND`/`OMP_PLACES` were set: roughly a 20x slowdown.
- **Pilot eviction guard dropped ~100% of speculations** once the cache filled (#497),
  collapsing `PILOT_REAL` to a hint-only path.
- **Silent budget clamp** capped the CUDA expert tier at ~109 experts regardless of
  `CUDA_EXPERT_GB` (#495).
- fmt=4 guard at the per-row-only CUDA entry points (#464/#470); `COLI_CUDA_MTP=1` and
  `COLI_CUDA=0` are now honoured over implicit defaults (#468).

### Security

Threat model: model files come from mirrors that are not trusted.

- **#368** — server hardening, JSON and tokenizer parser hardening, build flags, downloader
  and dependency pinning.
- **#413** — the quant layout is resolved *and* validated against the on-disk byte counts
  (unknown layouts are refused rather than falling through to int2), shape-product overflow
  is rejected, and the olmoe dtype-3 path no longer trusts a crafted `nbytes` (heap overflow).

### Performance — all byte-identical

- **#481** +4.7x on the MLA-absorb score and value-mix reductions
- **#477** +13% decode on AVX-512 (`qt_addrow` / `qt_matvec_rows`)
- **#475** +11.6% with opt-in `XEXP=1` (one OpenMP region per expert block at S=1 full residency)
- **#473** +5.5% int4 IDOT at S=1 on AVX-512 VNNI

### Changed

- `glm.c` is now `colibri.c` plus header modules (#391); `make glm` remains as an alias.
- Serve stage 2 (#192): `response_format`, per-request grammars, grammar-forced drafts.

### Upgrade notes

- **`CTX` still defaults to 4096.** Coding clients send far more than that in a single system
  prompt. Use `CTX=32768`. Before this release an over-long prompt was silently truncated;
  now you get a clear 400 instead.

## [1.0.0] — 2026-07-19

First tagged release. The engine has been running in production since late June
2026; this tag marks the baseline for semantic versioning going forward.

### Highlights

- **GLM-5.2 (744B MoE)** runs on ~25 GB RAM in pure C, streaming experts from disk
- **Three-tier placement**: VRAM (hot) / RAM (warm) / NVMe (cold), with a learning
  cache that pins your workload's hottest experts automatically
- **CUDA backend**: multi-GPU expert tier, dense tensor distribution, batched
  ragged attention, resident pipeline (`COLI_CUDA_PIPE=2`)
- **Metal backend** (Apple Silicon): batched expert SwiGLU + fused decode attention
  on unified memory GPU
- **MTP speculation**: native GLM-5.2 draft heads, grammar-forced drafts, kernel-
  pinned verification (`SPEC_PIN=1`)
- **OpenAI-compatible API**: `coli serve` with SSE streaming, KV slots, bounded
  queue, web dashboard (`coli web`)
- **Web UI**: chat with live metrics, expert cortex brain page, profiling breakdown,
  expert atlas 3-D galaxy
- **Cross-platform**: Linux, macOS, Windows 11 (native MinGW), PowerPC; CI on all three
- **Auto-tune**: `coli plan --auto-tier` classifies the bottleneck and derives
  MTP/PIPE/NUMA/PIN settings with explanations

### Engine

- Token-exact validation against `transformers` oracle (teacher-forcing 32/32)
- Compressed MLA KV cache (576 floats/token, 57× smaller), persisted across
  restarts (`.coli_kv`, zero re-prefill)
- DSA sparse attention (lightning indexer), faithfully implemented
- Router-lookahead prefetch (`PILOT=1`, 71.6% predictive)
- Async expert I/O pool (`PIPE=1`), io_uring batching (`URING=1`)
- NUMA-aware expert placement (`COLI_NUMA=1`, +13–40% on multi-socket)
- AVX2 / AVX-512 / AVX-VNNI / ARM NEON / NEON-i8mm / POWER VSX kernels
- int4 / int8 / int2 / grouped-int4 (fmt=4) quantization formats

### Tools

- `coli convert` — FP8→int4 one-shard-at-a-time converter
- `coli doctor` — read-only setup diagnostics
- `coli plan` — resource planner with auto-tune prescription
- `coli bench` — MMLU / HellaSwag / ARC quality benchmarks
- Expert atlas (`tools/analyze.py --web`) — measured topic affinity for 19,456 experts

### Community

- 30+ hardware datapoints in the benchmark tracker
- Contributions from 20+ authors across engine, docs, tooling, and ports
