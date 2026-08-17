<p align="center">
  <img src="assets/colibri-logo.svg" width="560" alt="colibrì — tiny engine, immense model">
</p>

<p align="center">
  <a href="https://justvugg.github.io/colibri"><img src="https://img.shields.io/badge/website-justvugg.github.io%2Fcolibri-1f6feb" alt="Website"></a>
  <a href="https://github.com/JustVugg/colibri/releases"><img src="https://img.shields.io/github/v/release/JustVugg/colibri?color=2ea043" alt="Latest release"></a>
</p>

<p align="center">
  <a href="https://justvugg.github.io/colibri"><b>Website</b></a> ·
  <a href="https://discord.gg/MAaKtQRc"><b>Discord</b></a> ·
  English · <a href="README.zh-CN.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.it.md">Italiano</a>
</p>

**Tiny engine, immense model.** Run **frontier MoE models — 744B to 2.8T
parameters** — on consumer and heterogeneous hardware, in pure C with zero
engine dependencies, by treating storage, RAM, and VRAM as a single inference
hierarchy (AI memory multitiering).

Five families run today: **GLM-5.2** (744B), **Inkling** (975B), **Kimi K3**
(2.8T), **DeepSeek V4 Flash** (284B) and **OLMoE** (7B) — one C file each, the
same `coli chat` / `coli serve` / `coli web` front end.
[Full roster ↓](#other-supported-models)

> **Colibrì is an inference engine you can run today, and an open research
> platform.** Its primary goal is to pursue inference-side performance across
> the entire software/hardware boundary — model formats, memory hierarchy,
> storage I/O, placement, scheduling, kernels, speculation, and CPU/GPU
> overlap — so large models depend less on scarce hardware and cost less to run.

Colibrì treats VRAM, RAM, and storage as a single multitier hierarchy, and it is
deliberately a place to test aggressive systems ideas — so there is **no SLA on
speed, and a hard guarantee on semantics**: experiments must earn their place
through reproducible end-to-end measurements, and the default policy **never
silently changes model precision or router semantics**. Insufficient fast memory
may reduce speed; it must not quietly redefine the model.

```
$ ./coli chat
  🐦 colibri v1.4.0 — GLM-5.2 · 744B MoE · int4 · streaming CPU
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

## The research mission

With Colibrì, private frontier model access is not limited by availability of hyperscaler-class hardware.

With its multitiering features Colibrì **removes proprietary hardware dependencies aggressively 
optimizing functional inference engine pipelines**.

Our operational mission includes changing how weights are represented and moved, deciding what
lives in VRAM, RAM, or storage, overlapping heterogeneous compute, reducing
launch and synchronization overhead, exploiting sparsity and reuse, and testing
new decoding algorithms. Nothing is protected merely because it is conventional;
nothing is adopted merely because a microbenchmark looks fast. The deciding
result is end-to-end inference on real machines, with correctness and quality
measured alongside throughput, latency, memory, and cost.

The practical consequence is **accessibility**: run a 744B-parameter model on
hardware you already own, watch every expert fire in real time, and change the
code that does it. Not renting intelligence behind an API — *holding* it:
probing it, measuring it, improving it. The engine is deliberately small enough
that the next useful optimization can come from anyone willing to measure it.

## Core techniques and measured findings

- **One hierarchy, not limited by tier capacity.** VRAM, RAM, and NVMe are placement
  tiers for the same weights; limited fast memory changes speed, not model semantics.
- **A JIT for weights.** Measured routing heat drives a per-layer LRU, a learned
  pinned hot-store, and one-layer-ahead prefetch instead of loading every expert.
  It wins on repeatable workloads; history can overfit, and lookahead can lose on
  some hosts, so both remain measurable policies rather than promises.
- **I/O is part of the engine.** Batched expert unions, overlapped reads and
  compute, `O_DIRECT`, and weighted dual-SSD striping attack the streaming path
  rather than pretending storage latency is free. `O_DIRECT` is drive-dependent,
  and dual-SSD still needs broader end-to-end community A/Bs.
- **Heterogeneous execution.** CPU, CUDA, Metal, NUMA memory, and partial or full
  expert residency share one runtime and can be combined according to the machine;
  the profitable combination depends on compute, bandwidth, residency, and workload.
- **Compressed state without a different model.** Token-exact forward validation,
  57× smaller MLA KV state, persistent warm conversations, and faithful DSA keep
  optimization tied to correctness. These are memory, latency, and correctness
  properties — not a blanket throughput claim.
- **Speculation that must earn its keep.** Native MTP and grammar-forced drafts
  are measured end to end and can be disabled when acceptance does not repay verification.

## Open hypotheses, experiments, and how to help

Colibrì treats an optimization as a hypothesis until a controlled end-to-end A/B
shows otherwise. These are the main questions now:

| hypothesis | evidence so far | experiment still needed |
|---|---|---|
| Routing history can place experts better than plain LRU | learned pins improve repeated workloads, but can overfit a prompt | held-out, cross-session A/Bs across coding, chat, multilingual, and long-context workloads |
| Multiple SSDs can turn independent bandwidth into decode speed | weighted mirror/split routing is implemented and validated; the bandwidth model is sound | cold-cache one-drive vs two-drive GLM-5.2 runs on real, independent controllers |
| A hardware-aware planner can approach each machine's best configuration automatically | RAM/VRAM budgets and several backends are detected today | compare the generated plan with a controlled parameter sweep across laptops, workstations, NUMA hosts, and multi-GPU systems |
| Lossless or quality-bounded representations can reduce weight movement enough to matter | format and quantization ablations exist, with correctness/quality gates | reproduce quality, bytes moved, latency, and cost per useful token together — not compression ratio alone |
| Routing-aware speculation can pay before near-full residency | MTP and grammar drafts work, but MTP has also measured a 32% loss around 85% expert hit | map the break-even surface across acceptance, expert hit rate, batch union, and draft depth |
| CPU/GPU overlap can hide transfer and synchronization rather than merely move the bottleneck | CUDA and Metal wins exist, but fast CPUs and low residency can erase them | per-stage profiles and one-variable A/Bs across PCIe, unified-memory, and full-resident machines |

Want to help? Pick one row and publish the negative results too. Record the
hardware, commit, model/container, exact command, prompt, cache state, throughput,
TTFT, expert hit rate, bytes read, and quality check; change one variable, repeat
the run, and attach raw logs. Start with
[CONTRIBUTING.md](CONTRIBUTING.md), compare against
[the benchmark protocol](docs/benchmarks.md), then
[open an experiment issue](https://github.com/JustVugg/colibri/issues/new).
A well-controlled failure is more valuable here than an unexplained fast number.

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

Think of the core algorithm as **a JIT, but for weights**. A compiler JIT never
compiles the whole program — it watches what actually runs and compiles the hot
paths, just in time. colibrì makes the same bet about a 744B parameter space:
parameters are not resident state to be held, they are **data to be staged**
across a heterogeneous storage hierarchy (VRAM / RAM / NVMe), exactly when the
router proves they are needed. Measured routing heat decides which experts earn
which tier, the router runs a layer ahead so prefetch hides the staging latency,
and — like a JIT — the engine learns your workload: the more you run, the hotter
the right experts get. It works because routing has measurable structure (see
the [expert atlas](https://github.com/JustVugg/colibri/issues/175)) — and
structure is cacheable.

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

### Dual-SSD: two copies of the model, twice the read bandwidth

Decode is disk-bound on most machines, and expert reads are read-only — so if you have a **second SSD**, put a full copy of the model on it and let the engine stream from both drives at once:

```bash
COLI_MODEL=/fast/glm52_i4 COLI_MODEL_MIRROR=/second/glm52_i4 ./coli chat
COLI_DISK_WEIGHTS=9,3 ...   # optional: primary,mirror bandwidth ratio (else measured at startup)
```

Each expert is routed to one drive by a deterministic hash, weighted by the two drives' measured (or declared) bandwidth, so readahead/PILOT prefetch and the demand read always hit the same drive and nothing is cached twice. The aggregate bandwidth is the sum of both drives — a 9 GB/s + 3 GB/s pair reads experts ~33% faster than the fast drive alone, and the OMP-parallel pin/warmup load streams from both. Details worth knowing:

- the mirror is **validated at startup** (per-file size + safetensors header must be byte-identical to the primary); divergent or missing files silently stay on the primary, so a **partial mirror is fine** — a smaller second SSD holding only some shards still helps;
- the mirror is **never written**: `.coli_usage`, `.coli_kv` and all sidecars stay on the primary;
- a read error on the mirror falls back to the primary (one warning, no crash), so unplugging the second drive mid-run degrades instead of killing the server;
- routing never changes tokens — both copies are byte-identical, and the per-run `MIRROR:` stats line shows GB served per drive.

The same engine spans the whole range: on a 25 GB laptop everything streams from
disk (slow but correct); on a large host the entire expert set becomes resident
(`CUDA_EXPERT_GB=auto PIN_GB=all`) and disk drops out of the decode path
entirely. Between the tiers sits a **learning cache**: the engine records which
experts *your* workload routes to (`.coli_usage`, updated every turn) and pins
the hottest ones automatically — colibrì literally gets faster the more you use
it. On multi-socket hosts, `COLI_NUMA=1` interleaves the resident weights across
memory controllers ([#82](https://github.com/JustVugg/colibri/issues/82)).

For a second drive that cannot hold the whole model, Colibri can rank a partial
mirror from the expert history it already learns. Run a few representative
prompts first so `.coli_usage` reflects the workload, then plan, stage, and
verify the mirror:

```bash
./c/coli mirror plan  --model /fast/glm52_i4 --mirror /second/glm52_i4 \
  --budget-gib 200 --reserve-gib 20
./c/coli mirror stage --model /fast/glm52_i4 --mirror /second/glm52_i4 \
  --budget-gib 200 --reserve-gib 20
./c/coli mirror verify --model /fast/glm52_i4 --mirror /second/glm52_i4
```

The planner reads safetensors headers directly, follows split-model directories
from `COLI_MODEL_DIRS`, and prioritizes shards that can serve the hottest routed
experts. Staging never changes the primary model: it copies through temporary
files, preserves the requested free-space reserve, verifies every shard with
SHA-256, never deletes an existing mirror shard, and atomically publishes a
receipt only after the selected mirror is ready.

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
math on the unified-memory GPU; and a [Vulkan backend](docs/vulkan.md) brings
the expert tier, dense projections, and the MLA attention core to any GPU with
a Vulkan 1.2 driver — including AMD cards via Mesa/RADV (the only backend for
cards the vendor stacks no longer support, like the RX 580, and competitive
with ROCm on RDNA4 — see [the benchmarking notes](docs/vulkan.md)).

> **On real NVMe, measure `DIRECT=1`.** O_DIRECT bypasses the page cache and is
> often a large win on drives with DRAM cache and bandwidth headroom (+34%
> decode measured with `PIPE=1` on a Blackwell/Windows box; 4.25→9.69 GB/s in
> iobench on a GB10) — but it is drive-dependent: QLC/DRAM-less or virtualised
> disks can be neutral to negative. Try it first; keep what your hardware
> rewards.

### Faithful model, compressed state

The forward pass is validated against a `transformers` oracle (teacher-forcing
typically 30-32/32; two tiny-oracle positions are floating-point near-ties and
toolchain-dependent). MLA attention stores a compressed KV state — 576
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

You need two things: **the program** (a few hundred KB) and **the model**
(372 GB). Step-by-step for every platform in the
[Quick Start guide](docs/quickstart.md).

### 1. Get colibri

**Download a prebuilt release** — Linux, macOS and Windows, no compiler needed.
Take the archive for your platform from
[Releases](https://github.com/JustVugg/colibri/releases) and unpack it:

```bash
mkdir colibri && tar xzf colibri-v1.1.0-linux-x86_64.tar.gz -C colibri && cd colibri
python3 coli info                         # engine ready ✓
```

Inside you get the engine (`colibri`, `colibri.exe` on Windows), the `coli`
launcher and its Python helpers. Nothing to rename or configure — `coli` finds
the engine next to itself. You only need
[Python 3](https://www.python.org/downloads/) installed: the launcher and the
API gateway are Python scripts, while the engine itself is pure C with zero
dependencies.

**Or build from source** — needs `gcc` (or clang) with OpenMP:

```bash
git clone https://github.com/JustVugg/colibri && cd colibri/c
./setup.sh                                # checks gcc/OpenMP, builds, self-tests
```

Want `coli` on your PATH? From a checkout, `pip install -e .` registers it (the
engine still lives in `c/` — an editable install from the clone, not a wheel).

### 2. Get the model

A pre-converted **GLM-5.2 int4** container is on Hugging Face — use the
**group-scaled (gs64)** build with the **int8 MTP head**. It is about **372 GB**,
so put it on a disk with the room, ideally a fast one:

**https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp**

> ⚠️ Use the **gs64** container above, not the older per-row int4 mirrors
> (`mateogrgic/…`, `jlnsrk/…`): those measure ~9pp worse on quality and are the
> root cause of the original think-mode loops and never-terminating generations
> in [#455](https://github.com/JustVugg/colibri/issues/455). The gs64 container
> fixed those controlled per-row A/Bs, but it is not a general repetition or
> EOS-starvation guard. The MTP head must also be **int8, not int4**
> (int4 → 0% draft acceptance, [#8](https://github.com/JustVugg/colibri/issues/8)):
> `ls -l <model>/out-mtp-*` — int8 (correct) is `3527131672 / 5366238584 / 1065950496`.

Or convert from the FP8 source yourself — one resumable command that never needs
the full 756 GB on disk at once:

```bash
./coli convert --model /nvme/glm52_i4     # download+convert shard by shard (python, one-time)
```

#### Other supported models

GLM-5.2 is the reference model, but the same streaming approach runs three more
families. Each is a **sibling engine** — one C file, its own architecture, the same
`coli chat` / `coli serve` / `coli web` front end (the launcher picks the binary from
the model's `config.json`):

> **What each one needs.** These differ a lot, and reading two of them together
> has confused people into thinking the requirements contradict each other
> ([#191](https://github.com/JustVugg/colibri/issues/191)). They do not — they
> are different models. **None of them needs a GPU.**
>
> | Model | Disk for the weights | RAM | GPU |
> |---|---|---|---|
> | **OLMoE** | ~7 GB (int8 container) | 8 GB | not needed |
> | **GLM-5.2** | ~372 GB | 16 GB min, 24 GB comfortable | not needed |
> | **Inkling** | ~469 GB | 25 GB with the int4 dense container, ~120 GB without | not needed |
> | **Kimi K3** | ~1.6 TB | 32 GB+ | not needed |
> | **DeepSeek V4 Flash** | ~167 GB | 16 GB min, 32 GB comfortable | optional; an NVIDIA card (any sm_80+, best on RTX 50) makes prefill 5-10x and decode ~2.5x faster |
>
> A GPU only ever makes it faster. Speed is set by your disk, because the experts
> are streamed from it — expect a fraction of a token per second on a slow drive
> and a few per second on a fast one with the cache warm.

| Family | Total / active | Weights | Build | Docs |
|---|---|---|---|---|
| **GLM-5.2** | 744B / 40B | [`mastouri/…-int4-g64-with-int8-mtp`](https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp) (372 GB) | `make -C c glm` | this page |
| **Inkling** (Thinking Machines) | 975B / 41B | [`nbeerbower/Inkling-colibri-int4`](https://huggingface.co/nbeerbower/Inkling-colibri-int4) (469 GB) | `make -C c inkling` | [inkling.md](docs/inkling.md) |
| **Kimi K3** (Moonshot) | 2.8T / 104B | [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) — original checkpoint, routed experts stay **native MXFP4** | `make -C c kimi_k3` | [kimi_k3.md](docs/kimi_k3.md) |
| **DeepSeek V4 Flash** | 284B / 13B | official sharded checkpoint — routed experts stay **native fp4**, dense stays fp8-e4m3 | `make -C c deepseek-v4` | [deepseek-v4.md](docs/deepseek-v4.md) |
| **OLMoE** (AI2) | 7B / 1B | converted with `c/tools/convert_olmoe_merged.py` — **int8** container, ~7 GB | `make -C c olmoe` | — |

Kimi K3 needs no conversion: its QAT-trained MXFP4 experts are streamed straight from
the original Hugging Face shards, and the bf16 dense set is quantized at load time.
Inkling ships int4 experts but **bf16 dense weights** (49.4 GB resident); on a host
that cannot hold those, [inkling.md](docs/inkling.md) has a one-pass tool that brings
the dense set to 15.3 GB and lets the 975B run on a 25 GB box — with the honest
trade-off written down.

### 3. Run it

```bash
COLI_MODEL=/nvme/glm52_i4 ./coli chat     # RAM budget, cache and MTP auto-detected
COLI_MODEL=/nvme/glm52_i4 ./coli plan     # inspect the planned VRAM/RAM/disk placement
COLI_MODEL=/nvme/glm52_i4 ./coli doctor   # read-only readiness check
COLI_MODEL=/nvme/glm52_i4 ./coli doctor --deep  # strict tensors/shards/index/mirror preflight
COLI_MODEL=/nvme/glm52_i4 ./coli tune     # measure and save this machine's fastest safe execution profile
./coli web  --model /nvme/glm52_i4        # API + dashboard, and opens a browser
./coli serve --model /nvme/glm52_i4       # API + dashboard, no browser (headless)
```

On Windows the same commands work with `python coli chat --model D:\glm52_i4`.
The engine at runtime is pure C — python is only used by the one-time converter
and the optional API gateway.

#### The same commands run any of the models

`coli` reads the model's `config.json`, picks the matching engine binary, and
renders that family's chat template — so **nothing about the command line
changes between models**. Build the engine you want once, then just point
`COLI_MODEL` at the right directory:

```bash
make -C c glm                                     # GLM-5.2
make -C c inkling                                 # Inkling
make -C c kimi_k3                                 # Kimi K3

COLI_MODEL=/nvme/glm52_i4      ./coli chat        # TUI
COLI_MODEL=/nvme/inkling_i4    ./coli chat
COLI_MODEL=/nvme/kimi_k3       ./coli chat

./coli web --model /nvme/inkling_i4               # API + dashboard, opens a browser
./coli web --model /nvme/kimi_k3
./coli serve --model /nvme/inkling_i4             # API + dashboard, no browser
```

For the non-GLM engines `coli chat` starts the gateway locally and attaches the
TUI to it, so the TUI, the API and the dashboard all go through the same
arch-aware chat template — you never have to pass the template yourself.

Two things that differ per model, both documented in the per-model page:

- **Inkling on a RAM-tight host** needs the int4 dense container and a small
  expert cache: `./coli chat --model /nvme/inkling_i4 --cap 2`
  (see [inkling.md](docs/inkling.md) — the default `--cap 8` wants ~14 GB of
  cache on top of the resident set).
- **Kimi K3** streams its MXFP4 experts from the original checkpoint, so there
  is nothing to convert — but the snapshot is ~1.6 TB
  (see [kimi_k3.md](docs/kimi_k3.md)).

### 4. Go deeper

| topic | doc |
|---|---|
| Benchmarks, community datapoints, quality measurements | [docs/benchmarks.md](docs/benchmarks.md) |
| Tuning knobs, policies, the learning cache, prefetch | [docs/tuning.md](docs/tuning.md) |
| Windows 11 native build (+ CUDA DLL) | [docs/windows.md](docs/windows.md) |
| CUDA backend, VRAM expert tier, full residency | [docs/cuda.md](docs/cuda.md) |
| Vulkan backend (any GPU: AMD via RADV, incl. cards ROCm dropped) | [docs/vulkan.md](docs/vulkan.md) |
| Apple Silicon Metal backend | [docs/metal.md](docs/metal.md) |
| OpenAI-compatible API, KV slots, web dashboard | [docs/api.md](docs/api.md) |
| Grammar-forced drafts (structured output) | [docs/grammar-draft.md](docs/grammar-draft.md) |
| Environment variable inventory | [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) |

## DeepSeek V4

**DeepSeek V4 Flash** streams the official checkpoint with no conversion: routed
experts stay **native fp4**, the dense set stays **fp8-e4m3** with UE8M0 block
scales. MLA + DSA sparse attention, 43 layers, 256 routed experts plus one
shared, top-6. Supported on x86-64/aarch64 Linux and Windows/MSYS2 (CPU), with
an optional CUDA tier (Windows runtime DLL; Linux `CUDA=1` direct link,
verified under WSL2) that keeps every stage CPU-canonical and falls back per stage.

```bash
cd c
make deepseek-v4
python ./coli chat --model /path/to/DeepSeek-V4-Flash --ram 32
# also: coli run / coli serve / coli web
# Windows CUDA tier: make cuda-dsv4-dll CUDA_ARCH=portable  (+ make cuda-dsv4-dg-dll on RTX 50)
```

Greedy decode and one KV slot. Tool calling is wired through the HTTP gateway
with V4's native prompt and DSML call blocks; grammar is not supported. See the
[per-engine API matrix](docs/api.md#tool-calling-support). Prefix checkpoints
(in memory + on disk) make agent sessions and follow-up turns start in seconds
after the first prefill of a system prompt. Measured on an RTX 5080 + 2 NVMe:
3324-token prefill 90 s, 8.3k-token first turn ~4 min once, later
sessions/turns 6-9 s, decode ~1.6 tok/s at 3k context — see
[docs/deepseek-v4.md](docs/deepseek-v4.md).

**Give it RAM.** 43 × 256 routed experts are ~137 GiB on disk and a token
touches 301 of them, so the expert cache hit rate is what sets tok/s — `--ram`
is the single most valuable knob, and it changes speed only, never output.

**Speculative drafting exists and is off.** DSpark's markov drafter and full MTP
are both implemented and verified: a draft can save forward passes but never
change a token, because every accepted token is still the target's own argmax.
Measured on real multi-turn chat, they accepted 1 in 15 and 10 in 24, and the
rejected-suffix replay of this engine's recurrent attention state cost more than
the drafts saved — one 14-token answer took 495 seconds. So `V4_DRAFT` and
`V4_MTP` default to `0` and the code stays, with the numbers beside it, for
whoever retries this on faster storage.

See [docs/deepseek-v4.md](docs/deepseek-v4.md) for the CUDA tier (build, DLL
selection, GPU coverage), the environment reference, performance numbers,
checkpoint validation, and the generated tiny independent oracle.

## What's next

- **Inference-systems research is the product.** The current hierarchy is LRU +
  a learned pin set; active work spans model formats, compression, placement,
  scheduling, I/O, CPU/GPU kernels, heterogeneous overlap, KV state, and
  routing-aware speculation. The objective is lower hardware requirements and
  lower cost per useful token. Everything lands the way this project works:
  measured end to end, reviewed, and developed in the open.
- **More open models.** The tiering algorithm is model-agnostic: any MoE with
  routed experts can be staged the same way. GLM-5.2 and OLMoE run today;
  support for more open-weight families — **Kimi K2** (Moonshot AI),
  **Qwen3 MoE** (Alibaba), **MiniMax** — is on the roadmap.

## Supporting the project

colibrì started as a one-person project on a 12-core laptop with 25 GB of RAM;
today its numbers come from a community of real machines. If it's useful to you:

- ⭐ star the repo and share it;
- 🐛 open issues with benchmark numbers from your hardware — datapoints move
  this project more than anything else;
- 💬 join the [Discord community](https://discord.gg/MAaKtQRc) to discuss
  experiments, hardware results, and research directions;
- 💬 reach out via GitHub issues to sponsor development or donate hardware.

## Repo layout

```
Makefile                  root build/check entry point
c/
├── colibri.c             GLM-5.2 engine  (make glm)
├── inkling.c             Inkling engine  (make inkling)
├── kimi_k3.c             Kimi K3 engine  (make kimi_k3)
├── deepseek_v4.c         DeepSeek V4 Flash engine  (make deepseek-v4)
├── olmoe.c               OLMoE engine  (make olmoe)
│
├── st.h                  safetensors index and range reads
├── quant.h               canonical container decoders
├── tok.h, json.h         tokenizer and JSON parser
├── compat.h              Windows/macOS shims (POSIX names, one place)
├── expert_store.h        streaming expert cache
├── route_trace.h         routing telemetry and .coli_usage, engine-agnostic
├── kv_prefix.h           KV prefix reuse across turns
│
├── backend_cuda.*        optional CUDA tier   (CUDA=1)
├── backend_metal.*       optional Metal tier  (METAL=1)
├── backend_vulkan.*      optional Vulkan tier (VULKAN=1)
│
├── Makefile              build and local checks
├── coli                  user-facing CLI
├── openai_server.py      OpenAI-compatible HTTP gateway
├── resource_plan.py      RAM/VRAM planner behind `coli plan` and `coli doctor`
├── tools/                offline conversion, fixtures and benchmarks
├── scripts/              long-running conversion helpers
└── tests/                dependency-free C and Python tests
web/                      browser UI (pure OpenAI-API client)
desktop/                  Tauri v2 desktop shell wrapping the web UI
docker/                   container images
docs/                     reference docs, experiments, media
```

**One `.c` per model family, over shared single headers.** An engine owns its
architecture and nothing else; anything two engines both need — the safetensors
reader, the container decoders, the tokenizer, the expert cache — lives in a
header they both include, so a fix reaches all of them at once. That rule is not
decorative: the defects that keep recurring here are the ones where a mechanism
landed in one engine and never reached its siblings.

From the repository root, `make`, `make check` and `make clean` delegate to the
engine Makefile.

## Why "colibrì"

The hummingbird weighs a few grams, hovers in place, and visits a thousand
flowers a day. This engine keeps a 744-billion-parameter giant alive on
hummingbird rations: 25 GB of RAM, twelve CPU cores, and a lot of disk patience.

## Acknowledgements

colibrì is an engine; the minds it runs are a gift. Thank you to the teams
releasing frontier-class weights in the open — **Z.ai** (GLM), **Moonshot AI**
(Kimi), **Alibaba Qwen**, **MiniMax**, and **Allen AI** (OLMoE) — and to every
contributor who benchmarked, bisected, replicated an atlas run, or sent a patch.
This project is proof of what open weights make possible.

The project's expert placement, compression, and routing experiments also build
on ideas and evidence from the following open research and systems work:

- [REAP](https://github.com/CerebrasResearch/reap) and
  [EASY-EP](https://github.com/RUCAIBox/EASYEP) for output-aware and
  domain-specific expert importance.
- [SERE](https://github.com/JL-Cheng/SERE) for similarity-based expert
  re-routing, and [ReMoE](https://github.com/BUAA-OSCAR/ReMoE) for
  cache-locality-aware router fine-tuning.
- [MC-SMoE](https://github.com/UNITES-Lab/MC-SMoE) for routing-guided expert
  merging and compression.
- [MoBE](https://github.com/inclusionAI/MoBE) and
  [D²-MoE](https://github.com/lliai/D2MoE) for shared expert bases and
  low-rank expert deltas.
- [HybriMoE](https://github.com/PKU-SEC-Lab/HybriMoE) for hybrid CPU/GPU expert
  scheduling, [ScMoE](https://arxiv.org/abs/2404.05019) for overlapping expert
  communication with computation, and
  [OD-MoE](https://arxiv.org/abs/2512.03927) for distributed on-demand expert
  loading.
- [vLLM](https://github.com/vllm-project/vllm),
  [llama.cpp](https://github.com/ggml-org/llama.cpp), and
  [kTransformers](https://github.com/kvcache-ai/ktransformers) for the open
  inference systems and expert-offload work that make comparisons reproducible.

The engine also stands on concrete engineering work, not only ideas. Each of
these is used or reimplemented in the tree today:

- [safetensors](https://github.com/huggingface/safetensors) — the container
  every engine reads (`c/st.h`), including its fp8 and I64 dtypes.
- [tiktoken](https://github.com/openai/tiktoken) — `c/tok.h` reimplements its
  `byte_pair_encode` exactly, merging the adjacent pair whose concatenation has
  the lowest vocab id, so a tiktoken-derived vocabulary needs no merges list.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — the GBNF grammar subset
  in `c/grammar.h` follows its syntax and its set-of-stacks PDA, and the Metal
  path borrows its `newBufferWithBytesNoCopy` residency trick.
- [vLLM](https://github.com/vllm-project/vllm) — the reference for output
  semantics the engine matches position by position (e.g. where the final norm
  lands relative to the LM head).
- [transformers](https://github.com/huggingface/transformers) — the oracle:
  CI reproduces a random-init model token for token against it.
- [DietGPU](https://github.com/facebookresearch/dietgpu) — the GPU ANS codec
  behind the experimental compressed expert tier (`COLI_ANS`).
- [rocWMMA](https://github.com/ROCm/rocWMMA) — the HIP backend maps CUDA's
  `nvcuda::wmma` fragment/mma_sync API onto it (`c/backend_gpu_compat.h`), which
  is what lets one .cu source compile for both vendors.

## License

Apache 2.0. GLM-5.2 weights are released by Z.ai under MIT.
