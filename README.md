<p align="center">
  <img src="assets/colibri.svg" width="500" alt="colibrì — tiny engine, immense model">
</p>

<p align="center">
  English · <a href="README.zh-CN.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.it.md">Italiano</a>
</p>

**Tiny engine, immense model.** Run **GLM-5.2 (744B-parameter MoE)** on a consumer machine with ~25 GB of RAM — in pure C, with zero dependencies, by streaming experts from disk.

Colibrì is a lightweight, quality-preserving MoE runtime that treats VRAM, RAM,
and storage as one managed memory hierarchy. Insufficient fast memory may reduce
speed, but the default policy **never silently changes model precision or router
semantics**.

```
$ ./coli chat
  🐦 colibrì v1.0 — GLM-5.2 · 744B MoE · int4 · streaming CPU
  ✓ ready in 32s · resident 9.9 GB
  › ciao!
  ◆ Ciao! 😊 Come posso aiutarti oggi?
```

## See it running

<p align="center">
  <img src="docs/media/colibri-dashboard.png" width="900" alt="colibrì web dashboard — live metrics, hardware panel, expert tiers">
</p>
<p align="center"><em>The web dashboard (<code>./coli web</code>): a 744B model at <strong>4 tok/s, TTFT 1.6 s, disk 0</strong> —
full expert residency on 6× RTX 5090, with live token metrics, the per-turn time breakdown,
the VRAM/RAM/disk tier bar and the live mini-brain in the corner.</em></p>

<p align="center">
  <img src="docs/media/colibri-brain.png" width="900" alt="the Brain page — 19,456 experts as a live cortex">
</p>
<p align="center"><em>The <strong>Brain</strong> page: all 19,456 experts as a living cortex — colour is the storage tier,
brightness is routing heat, and every expert routed in a turn flashes white. Hovering shows the expert's
<a href="https://github.com/JustVugg/colibri/issues/175">measured topic affinity</a>.</em></p>

<p align="center">
  <img src="docs/media/colibri-atlas.png" width="900" alt="the Atlas page — the measured expert atlas as a 3-D galaxy">
</p>
<p align="center"><em>The <strong>Atlas</strong> page: the <a href="https://github.com/JustVugg/colibri/issues/175">measured expert atlas</a>
as a 3-D galaxy — 13,260 characterised experts, 1,041 replicated specialists clustering by topic
(poetry, law, Chinese, SQL…). Position is measured routing affinity, not a learned embedding. Drag to spin.</em></p>

## The idea

A 744B Mixture-of-Experts model activates only ~40B parameters per token — and
only ~11 GB of those change from token to token (the routed experts):

<p align="center">
  <img src="docs/media/sparse.png" width="880" alt="only ~5.4% of parameters are active per token">
</p>

So the model doesn't need to *fit* in fast memory — it needs to be **placed**:

- the **dense part** (attention, shared experts, embeddings — ~17B params) stays
  **resident in RAM at int4** (~9.9 GB);
- the **19,456 routed experts** (75 MoE layers × 256 + the MTP head, ~19 MB each
  at int4) live **on disk** (~370 GB) and are **streamed on demand**, with a
  per-layer LRU cache, a learned pinned hot-store, and an optional VRAM tier.

The engine is a single C file (`c/glm.c`) plus small headers. No BLAS, no Python
at runtime, no GPU required.

## How it works

### The per-token path

<p align="center">
  <img src="docs/media/token-path.png" width="880" alt="route → union → place → overlap → learn">
</p>

Every layer of every token walks the same five steps. The design goal is that
**placement only ever decides speed** — the router's decisions and the weights'
precision are the same whether an expert answered from VRAM or from disk.

### One memory hierarchy instead of one memory requirement

<p align="center">
  <img src="docs/media/tiers.png" width="880" alt="VRAM / RAM / NVMe three-tier expert residency">
</p>

The same engine spans the whole range: on a 25 GB laptop everything streams from
disk (slow but correct); on a large host the entire expert set becomes resident
(`CUDA_EXPERT_GB=auto PIN_GB=all`) and disk drops out of the decode path
entirely. Between the tiers sits a **learning cache**: the engine records which
experts *your* workload routes to (`.coli_usage`, updated every turn) and pins
the hottest ones automatically — colibrì literally gets faster the more you use
it. On multi-socket hosts, `COLI_NUMA=1` interleaves the resident weights across
memory controllers ([#82](https://github.com/JustVugg/colibri/issues/82)).

### Never wait for the disk twice

Misses are expensive, so the engine spends most of its cleverness avoiding and
overlapping them: each expert's three matrices are stored adjacent and read in
one `pread`; a bounded async I/O pool (`PIPE=1`, default) loads missing experts
while resident ones compute; batched positions read each unique expert once
(**batch-union**); and a router-lookahead thread (`PILOT=1`) prefetches the next
layer's experts — routing is measurably **71.6% predictable one layer ahead**.
On GPUs, the resident pipeline (`COLI_CUDA_PIPE=2`) keeps the residual stream
on-device across layers so the CPU expert loop runs uninterrupted; on Apple
Silicon an experimental [Metal backend](docs/metal.md) does the batched expert
math on the unified-memory GPU.

### Faithful model, compressed state

The forward pass is validated **token-exact against a `transformers` oracle**
(teacher-forcing 32/32). MLA attention stores a compressed KV state — 576
floats/token instead of 32,768 (**57× smaller**) — and persists it across
restarts (`.coli_kv`): conversations reopen warm with zero re-prefill,
byte-identical to an uninterrupted session. DSA sparse attention (GLM-5.2's
lightning indexer) is implemented faithfully and validated by forcing full-key
selection to reproduce dense attention exactly.

### Speculative decoding, honestly

GLM-5.2's native MTP head drafts tokens that the main model verifies in one
batched forward — 2.2–2.8 tokens/forward when it pays. Two hard-won rules ship
as defaults: the MTP head must be **int8** (int4 heads collapse to 0–4%
acceptance, [#8](https://github.com/JustVugg/colibri/issues/8)), and draft and
verify must compute **the same function** — `SPEC_PIN=1` pins both to one
kernel family ([#163](https://github.com/JustVugg/colibri/issues/163) is the
full forensic story). Grammar-forced drafts
([`GRAMMAR=file.gbnf`](docs/grammar-draft.md)) add ~free acceptance on
constrained JSON output. Whether speculation is a net win depends on your
cache temperature — measure, and use `DRAFT=0` when it doesn't pay.

## What it achieves

<p align="center">
  <img src="docs/media/ladder.png" width="880" alt="measured decode speed by hardware class">
</p>

Same engine, same int4 container — the hardware only changes where the experts
live. Highlights from the [full benchmark tables](docs/benchmarks.md):

- **6× RTX 5090, full residency:** 5.8–6.8 tok/s decode, TTFT ~13 s
  ([experiment log](docs/experiments/glm52-6x5090-2026-07-12.md));
- **128 GB CPU-only desktop:** ~1.8 tok/s warm ([#200](https://github.com/JustVugg/colibri/issues/200));
- **single RTX 5070 Ti laptop-class box:** 1.07 tok/s via the GPU-resident
  pipeline ([#273](https://github.com/JustVugg/colibri/issues/273));
- **25 GB dev box:** 0.05–0.1 tok/s cold — the proven floor where this project
  started, and still the honest baseline.

Quality is measured, not assumed: the int4 container's quantization cost and the
scale-granularity/rotation ablations live in
[docs/benchmarks.md](docs/benchmarks.md#quality-benchmark) and
[#108](https://github.com/JustVugg/colibri/issues/108)/[#81](https://github.com/JustVugg/colibri/issues/81).

## Get started

> **New here?** The [Quick Start guide](docs/quickstart.md) walks through
> install → build → model → first chat step by step for Linux, Windows, and
> macOS, with copy-paste commands and no assumed background.

### 1. Get the model

A pre-converted **GLM-5.2 int4** container is on Hugging Face — **use the
version with the int8 MTP heads**:

**https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp**

> ⚠️ The original mirror ships int4 MTP heads → 0% draft acceptance
> ([#8](https://github.com/JustVugg/colibri/issues/8)). Check yours:
> `ls -l <model>/out-mtp-*` — int8 (correct) is `3527131672 / 5366238584 / 1065950496`.

Or convert from the FP8 source yourself — one resumable command that never needs
the full 756 GB on disk at once:

```bash
cd c && ./setup.sh                        # checks gcc/OpenMP, builds, self-tests
./coli convert --model /nvme/glm52_i4     # download+convert shard by shard (python, one-time)
```

### 2. Run it

```bash
COLI_MODEL=/nvme/glm52_i4 ./coli chat     # RAM budget, cache and MTP auto-detected
COLI_MODEL=/nvme/glm52_i4 ./coli plan     # inspect the planned VRAM/RAM/disk placement
COLI_MODEL=/nvme/glm52_i4 ./coli doctor   # read-only readiness check
./coli web  --model /nvme/glm52_i4        # API + web dashboard on one port
./coli serve --model /nvme/glm52_i4       # OpenAI-compatible API only
```

The engine at runtime is pure C — python is only used by the one-time converter
and the optional API gateway.

Prefer a `coli` command on your PATH? From a checkout, `pip install -e .`
registers it (the engine itself still lives in `c/` — this is an editable
install from the clone, not a standalone wheel).

### 3. Go deeper

| topic | doc |
|---|---|
| Benchmarks, community datapoints, quality measurements | [docs/benchmarks.md](docs/benchmarks.md) |
| Tuning knobs, policies, the learning cache, prefetch | [docs/tuning.md](docs/tuning.md) |
| Windows 11 native build (+ CUDA DLL) | [docs/windows.md](docs/windows.md) |
| CUDA backend, VRAM expert tier, full residency | [docs/cuda.md](docs/cuda.md) |
| Apple Silicon Metal backend | [docs/metal.md](docs/metal.md) |
| OpenAI-compatible API, KV slots, web dashboard | [docs/api.md](docs/api.md) |
| Grammar-forced drafts (structured output) | [docs/grammar-draft.md](docs/grammar-draft.md) |
| Environment variable inventory | [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) |

## Supporting the project

colibrì started as a one-person project on a 12-core laptop with 25 GB of RAM;
today its numbers come from a community of real machines. If it's useful to you:

- ⭐ star the repo and share it;
- 🐛 open issues with benchmark numbers from your hardware — datapoints move
  this project more than anything else;
- 💬 reach out via GitHub issues to sponsor development or donate hardware.

## Repo layout

```
Makefile                  root build/check entry point
c/
├── glm.c                 single-file GLM engine
├── st.h, tok.h, json.h   runtime headers
├── backend_cuda.*        optional CUDA tier
├── Makefile              build and local checks
├── coli                  user-facing CLI
├── openai_server.py      OpenAI-compatible HTTP gateway
├── setup.sh              one-command local setup
├── tools/                offline conversion, fixtures and benchmarks
├── scripts/              long-running conversion helpers
└── tests/                dependency-free C and Python tests
web/                      browser UI (pure OpenAI-API client)
desktop/                  Tauri v2 desktop shell wrapping the web UI
docs/                     reference docs, experiments, media
```

The runtime path intentionally stays flat and readable: `glm.c` plus its small
headers. From the repository root, `make`, `make check`, and `make clean`
delegate to the engine Makefile.

## Why "colibrì"

The hummingbird weighs a few grams, hovers in place, and visits a thousand
flowers a day. This engine keeps a 744-billion-parameter giant alive on
hummingbird rations: 25 GB of RAM, twelve CPU cores, and a lot of disk patience.

## License

Apache 2.0. GLM-5.2 weights are released by Z.ai under MIT.
