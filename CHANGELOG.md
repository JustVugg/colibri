# Changelog

All notable changes to colibrì are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [1.7.0] — 2026-08-19
71 pull requests since v1.6.2. A sixth model family with its GPU tier, a
rebuilt expert-matmul path, and the CI that would have caught the class of bug
we shipped twice.

### A sixth engine: Qwen3.6-35B-A3B — CPU and GPU
- **#712** (@kreuzzelg) — hybrid Gated Attention + Gated DeltaNet + streaming
  MoE, in `c/qwen36.c`. Pre-converted containers published (int4-gs64
  recommended: cosine to the int8 anchor 0.98777 → 0.99313, KL 0.109 → 0.080
  vs per-row). The engine takes any architecture-identical checkpoint
  unchanged — KAT-Coder v2.5 runs on it with no code path of its own.
- **#713** (@kreuzzelg) — CUDA VRAM expert tier with heat-based placement
  across GPUs: **1.44 → 10.05 tok/s (7.0×) on 2× 8 GB cards, output
  bit-identical to CPU** (`cmp` over the full 200-token generation), measured
  cold with no heat table. A `qt_ready()` gate keeps CPU-only builds from
  allocating the packed int4 buffers they never read: **7.33 GB saved**.

### The expert matmul path, rebuilt (all bit-identical)
- **#1071 / #1075 / #1076 / #1077** — activation quantization hoisted to layer
  level across GLM, Kimi K3 and DeepSeek V4: the same vector was being
  re-quantized ~16× per layer, serially. Removes ~5.2 ms/token of serial time
  and every per-call `malloc` from the hot path.
- **#1079 / #1086** — **K1: plane-nibble int4 layout + unsigned-VNNI dot.**
  Storing element *k* and *k+32* in one byte deletes the unpack, and since
  nibbles are stored unsigned, `dot(v,x) = dot(u,x) − 8·Σx` feeds `vpdpbusd`
  natively. 8 uops per 64 MACs instead of 32: **1.45–2.65× on the IDOT
  kernels**, zero bytes added.
- **#1088** — **K2: 1×4 union tile.** The prefill union hands each expert
  2–16 rows; the weight block's load+mask is now paid once per four rows
  instead of once per row: **2.67–3.10× at S=4** (peak 253 GMAC/s).
- **#1093** — parallel silu and the down-side activation hoist.
- **#1094** — **K1b: grouped planar IDOT for gs64 containers** (`IDOT_GS=1`,
  opt-in): the recommended container format could not reach the integer
  kernels at any batch size before this.
- **#1082** (@outtodata) — `FUSED3=1` opt-in fused AVX2 expert matmul.

### Streaming and I/O
- **#1097** — DeepSeek V4 expert-loader pool default 3 → 9 lanes:
  **1.41× decode** on the real V4-Flash checkpoint (8 interleaved runs on a
  quiet 25 GB box). `V4_LOADER_LANES` still overrides.
- **#1056** (@dcutugno) — DeepGEMM sm120 headers fetched at a pinned commit on
  first build: 2.5× prefill on sm120 with nothing vendored in-tree.
- **#988 / #1054 / #1055** — DeepSeek V4 CUDA tier and dual-SSD mirror.

### Correctness and CI
- **#1083** — **ARM CI job** (`ubuntu-24.04-arm`) plus an integer-kernel
  bit-exactness gate that runs on both ISAs. Every tiny-oracle job ran on x86
  before this, so NEON-divergent paths were invisible — which is how the IDOT
  defaults below shipped. Closes #1081.
- **#1044 / #1080** — IDOT made opt-in in olmoe and inkling: the fast path is
  x86-only and quantizes activations, so the same model produced different
  tokens on x86 and ARM by default.
- **#1109** (@SebaWag) — ARM64 dotprod probed by *compiling* the intrinsic:
  GCC 11 defines `__ARM_FEATURE_DOTPROD` for a base it cannot emit
  `vdotq_s32` for. Fixes #1104.
- **#1111** — `__syncwarp()` after `grouped_s4_wmma`'s store (reported by
  @monotophic with `compute-sanitizer` evidence). Fixes #1099.
- **#1073 / #1074** (@bherald) — Kimi K3 cancels prefill between layers
  instead of holding the engine for a minutes-long prompt; cancelled requests
  no longer count as completed.

### Apple Silicon
- **#790 → #1113** (@RDouglasSharp) — **Metal backend for Kimi K3**: KDA state and
  window buffers aligned, wrap-once buffer cache, CPU-side MLA KV cache. 1.7×/2.4×
  on the compute-bound phases (KDA attention + projections dispatched to the GPU);
  MoE experts stay on the CPU. Kimi K3's first GPU backend.

### More correctness fixes
- **#1098** (@monotophic) — `__syncthreads()` missing from the absorb softmax
  reduction, with a determinism test that reproduces the hazard.
- **#1101** (@monotophic) — allocation and `snprintf` results checked on the
  checkpoint-load path (#798).
- **#1100 / #1108** (@monotophic) — fmt=8/fmt=6 scale-byte accounting in
  `tensor_bytes`/`tensor_free`, and `weights_owned` set before the host-to-device
  copy so a failed upload frees its buffer. Each ships with its own regression
  test; all four of this contributor's CUDA fixes landed in this release.
- **#1122** (@ZacharyZcR) — `USAGE_SAVE=0` honoured in every engine (#1039): the
  history was loaded but written back anyway, which quietly contaminated any A/B
  that shared a usage file between arms.
- **#1121** (@ZacharyZcR) — LRU victim selection now respects a lowered `ecap`
  (#1034): after an RSS-guard reduction the cache kept evicting against the old
  capacity.
- **#1123** (@ZacharyZcR) — the v1.6.2 warning-cleanup patches landed (#1032).
- **#1106** (@monotophic) — duplicate tensor names across indexed shards are now
  refused rather than silently resolved to one of them (untrusted containers).

### Interfaces
- **#829** (@aaristov) — GPU-vs-fallback counters and chat status made visible in
  `coli serve`: the tier's behaviour is now observable instead of inferred.
- **#1095** (@benmaster82) — OLMoE planner geometry adapter, and **#1103**
  (@SebaWag) — Kimi K3, Inkling and DeepSeek V4 adapters with 23 tests: every
  family now has real planner geometry, so `coli plan` stops guessing (#1066).
- **#1096** (@terrizoaguimor) — DeepSeek V4 serve framing on the shared codec,
  completing the codec migration across OLMoE, Kimi K3 and V4.
- **#1063 / #1068** (@terrizoaguimor) — model families are registry-owned:
  `coli`, the gateway, `doctor` and the planner read one descriptor table.
- **#1087 / #1090 / #1096 / #1116** (@terrizoaguimor) — a shared serve framing
  codec, now adopted by **every engine**: OLMoE, Kimi K3, DeepSeek V4 and
  Inkling (whose audio payload rides as an opaque extension). Each migration
  landed behind a byte-exact wire-transcript freeze, so the gateway contract is
  provably unchanged. Byte framing had been duplicated five times, which is how
  Windows binary mode silently disappeared from sibling engines (#748).
- **#1036** (@lineape) — distributed expert workers (LAN, opt-in via
  `CLUSTER_WORKERS`).
- Planner: DeepSeek V4 expert naming now recognized, so `coli plan` and
  `coli doctor` stop counting every routed expert as dense (fixes #1110).

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
