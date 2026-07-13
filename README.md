<p align="center">
  <img src="assets/colibri.svg" width="500" alt="colibrì — tiny engine, immense model">
</p>

**Tiny engine, immense model.** Run **GLM-5.2 (744B-parameter MoE)** on a consumer machine with ~25 GB of RAM — in pure C, with zero dependencies, by streaming experts from disk.

Colibrì is a lightweight, quality-preserving MoE runtime that treats VRAM,
RAM, and storage as one managed memory hierarchy. Insufficient fast memory may
reduce speed, but the default policy never silently changes model precision or
router semantics.

```
$ ./coli chat
  🐦 colibrì v1.0 — GLM-5.2 · 744B MoE · int4 · streaming CPU
  ✓ ready in 32s · resident 9.9 GB
  › ciao!
  ◆ Ciao! 😊 Come posso aiutarti oggi?
```

## The idea

A 744B Mixture-of-Experts model activates only ~40B parameters per token — and only ~11 GB of those change from token to token (the routed experts). So:

- the **dense part** (attention, shared experts, embeddings — ~17B params) stays **resident in RAM at int4** (~9.9 GB);
- the **21,504 routed experts** (75 MoE layers × 256 experts + the MTP head, ~19 MB each at int4) live **on disk** (~370 GB) and are **streamed on demand**, with a per-layer LRU cache, an optional pinned hot-store, and the OS page cache as a free L2.

The engine is a single C file (`c/glm.c`, ~2,400 lines) plus small headers. No BLAS, no Python at runtime, no GPU required (an opt-in CUDA tier for pinned experts exists — see below).

## What's implemented

- **Faithful GLM-5.2 (`glm_moe_dsa`) forward** — validated token-exact against a `transformers` oracle (teacher-forcing 32/32, greedy 20/20 on a tiny-random model with the real architecture).
- **MLA attention** (q/kv-LoRA, interleaved partial RoPE) with **compressed KV-cache**: 576 floats/token instead of 32,768 (57× smaller — GLM-5.2 has 64 heads and no GQA).
- **DeepSeek-V3-style sigmoid router** (noaux_tc, routed_scaling_factor), shared expert, first-3-dense layers.
- **Native MTP speculative decoding** — GLM-5.2's own multi-token-prediction head (layer 78) drafts tokens that the main model verifies in one batched forward. **The head must be int8** (the converter does this by default): at int4 draft acceptance collapses to 0–4% and speculation never engages; at int8 it's 39–59% acceptance, **2.2–2.8 tokens/forward** (community-measured, [#8](https://github.com/JustVugg/colibri/issues/8)). Lossless *in exact arithmetic* — but **not byte-identical to non-speculative greedy in practice** ([#100](https://github.com/JustVugg/colibri/issues/100)). This isn't MTP-specific: colibrì's quantized integer kernels are shape-dependent, so any batched (S>1) or GPU forward rounds slightly differently from the single-token path, and int4 GLM-5.2 sits close enough to argmax ties that such a rounding change can flip a token. MTP, the CUDA expert tier, and batched prefill are three different ways to trip the same sensitivity (community-confirmed in #100: swapping only the kernel family forks greedy output on 3/5 prompts, with **zero speculation**). Every emitted token is still the argmax of a *valid* forward — the continuation stays correct — it just isn't the same stream. For byte-exact reproducibility: `DRAFT=0` (no speculation), plus `IDOT=0 COLI_CUDA=0` if you also want kernel-family/GPU independence. Under sampling, rejection sampling keeps the distribution correct. Honest caveat from the same measurement: on a **cold** cache each verified draft routes to extra experts (~660 → ~1100 expert-loads/token), so speculation can be a net *time* loss until the cache/pin warms up.
- **Grammar-forced speculative drafts** (`GRAMMAR=file.gbnf`, [#48](https://github.com/JustVugg/colibri/issues/48)) — on constrained-output workloads (JSON/NDJSON, function calling, structured extraction) the grammar itself is a third draft source: wherever it admits exactly **one** legal byte (braces, quotes, key names, enum bodies), that forced span is tokenized and injected as pre-accepted drafts with ~1.0 acceptance — no draft head, no lookup table, and it engages even with the int4 MTP head from [#8](https://github.com/JustVugg/colibri/issues/8). It never constrains sampling: forced spans are verified in the same batch-union forward as any draft, so a wrong or out-of-sync grammar cannot change the output — worst case is rejected drafts, and an adaptive guard turns the source off below 50% acceptance. Byte-level GBNF subset (literals, char classes, `| ( ) ? * +`, comments); `GRAMMAR_DRAFT=n` caps the forced span per forward (default 24). Composes with `DRAFT`/MTP, which fill the free-text gaps between forced spans.
- **True sampling** — temperature + nucleus, defaults tuned for int4 reality (0.7 / 0.90; the official 1.0 / 0.95 samples quantization noise from the tail).
- **Integer-dot kernels** (Q8_0-style int8 activations, AVX2 `maddubs`): int8 matmuls 1.4–2.5× faster (119 GFLOP/s measured), int4 1.8× in batch — routing decided per shape by measurement (int4 single-row stays f32: it measured slower).
- **MLA weight absorption** (DeepSeek trick) for decode: no per-token k/v reconstruction — the query absorbs `kv_b`, context is projected after attention. Validated exact: TF 32/32 and generation 20/20 with absorption forced everywhere.
- **Async expert readahead**: while one block of experts is being multiplied, the kernel is already reading the next (`WILLNEED`).
- **Quantization kernels**: int8 / packed int4 / packed int2, per-row scales, AVX2, dequant-on-use. Packing validated bit-identical to the int8 container.
- **DSA sparse attention** — GLM-5.2's lightning indexer, faithful to the reference `glm_moe_dsa` modeling: per-layer top-2048 causal key selection (full/shared indexer layers), auto-detected from the `out-idx-*` weights (`--indexer` converter mode, ~189 MB extracted from the FP8 repo). Validated exact: forcing the selection to keep every key reproduces dense attention token-for-token. `DSA=0` disables, `DSA_TOPK` overrides.
- **KV-cache persistence** — conversations reopen **warm** across engine restarts: serve mode appends the compressed MLA KV to `.coli_kv` after every turn (~182 KB/token, crash-safe) and resumes it at startup with zero re-prefill. Validated byte-identical to an uninterrupted session. `KVSAVE=0` disables.
- **Router-lookahead prefetch** (`PILOT=1`, experimental) — the next layer's routing is 71.6% predictable from the current layer's post-attention state (measured); a dedicated I/O thread prefetches those experts while the current layer computes.
- **Batch-union MoE**: in prefill (and MTP verification), each unique expert of the batch is read once and applied to every position that routes to it.
- **Byte-level BPE tokenizer in C** (GPT-2-style with Unicode-property regex, 320k merges).
- **RAM safety**: the expert cache is auto-sized from `MemAvailable` at startup — an honest peak projection (working set, KV, MTP row, reconstruction buffers) so the kernel OOM-killer never fires.
- **Offline FP8→int4 converter** (`c/tools/convert_fp8_to_int4.py`): downloads one shard at a time (~5 GB), dequants (128×128 block scales), requantizes to the engine's container, deletes the shard — the 756 GB FP8 checkpoint never needs to exist on disk at once. Resumable.

## Honest numbers (WSL2, 12 cores, 25 GB RAM, NVMe via VHDX)

Detailed GPU experiment: [GLM-5.2 on 6x RTX 5090](docs/experiments/glm52-6x5090-2026-07-12.md) — full expert residency across VRAM+RAM reaches 6.84 tok/s single-request decode.

| metric | value |
|---|---|
| model on disk (int4 container) | ~370 GB |
| resident RAM (dense, int4) | 9.9 GB |
| load time | ~30 s |
| peak RSS during chat | ~20 GB (auto-capped) |
| cold decode cost | ~11 GB disk reads/token (75 layers × 8 experts) |
| disk ceiling (this dev box's drive) | ~1 GB/s → ~0.05–0.1 tok/s cold |
| MTP speculation (int8 head) | 2.2–2.8 tok/forward measured ([#8](https://github.com/JustVugg/colibri/issues/8)) |

This is not fast. It is a 744B frontier-class model **answering correctly on a machine that costs less than one H100 fan**. Warm cache, pinned hot experts and MTP push the useful-response latency down considerably; the physics of the disk does the rest.

### SSD note
Cold starts are heavy on random reads (~11 GB/token), but reads don't meaningfully wear an SSD — colibrì's streaming is read-only. The real concerns under heavy use are (1) **swap traffic** if the system runs out of RAM (writes do wear the drive — keep a sane `--ram` budget; colibrì's auto-budget is designed to stay clear of swap) and (2) **sustained thermals**: hours at full read duty cycle will heat cheaper drives. Monitor drive temperature and health.

## Download the model

A pre-converted **GLM-5.2 int4** model for colibrì is available on Hugging Face — **use the version with the int8 MTP heads** (matey-0's clone):

**https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp**

> ⚠️ **The MTP head must be int8.** The original mirror ([jlnsrk/GLM-5.2-colibri-int4](https://huggingface.co/jlnsrk/GLM-5.2-colibri-int4)) ships **int4** MTP heads, which give **0% draft acceptance** — speculation silently never engages and you lose the ~2× MTP lever. This is the single most common "why is MTP stuck at 0%?" report ([#8](https://github.com/JustVugg/colibri/issues/8), [#102](https://github.com/JustVugg/colibri/issues/102)). The int8 head gives the measured **39–59% acceptance**. matey-0's clone above is the original int4 model with the three `out-mtp-*` files already swapped to int8 — download that one and you're done.
>
> Check what you have: `ls -l <model>/out-mtp-*`
> · **int8 (correct):** `3527131672 / 5366238584 / 1065950496`
> · **int4 (0% acceptance):** `1765523544 / 2686077736 / 536747200` — if you see these, replace just those three files from the int8 mirror.

Download the repository and point `COLI_MODEL` to its directory:

```bash
COLI_MODEL=/path/to/GLM-5.2-colibri-int4-with-int8-mtp ./coli chat
```

This skips the FP8 → int4 conversion step entirely. Thanks to DatPat for the original mirror and matey-0 for the int8-head clone.

### Quick start

```bash
cd c
./setup.sh                      # checks gcc/OpenMP, builds, self-tests

# ONE command does everything model-side: downloads GLM-5.2-FP8 shard by shard
# (never needs the full 756 GB at once), converts to the int4 container, then
# converts the MTP head for speculative decoding. Resumable at any point.
# Conversion (only) needs python with: pip install torch safetensors huggingface_hub numpy
./coli convert --model /nvme/glm52_i4     # ~400 GB free on a real ext4/NVMe path

# chat — RAM budget, expert cache and MTP are all detected automatically:
COLI_MODEL=/nvme/glm52_i4 ./coli chat
```

Inspect the planned storage hierarchy before loading the model:

```bash
COLI_MODEL=/nvme/glm52_i4 ./coli plan
COLI_MODEL=/nvme/glm52_i4 ./coli plan --gpu 0,1 --ram 128 --vram 48 --json

# apply the bounded plan to the normal runner
COLI_MODEL=/nvme/glm52_i4 ./coli chat --auto-tier
```

`coli plan` reads only safetensors headers and reports the model's exact dense/expert
footprint, runtime RAM reserve, safe expert-cache cap, and bounded VRAM hot tier. Its
versioned JSON output is intended to be shared by the CLI, API server, Web UI, and
desktop shell; it does not allocate model tensors or start inference.
`--auto-tier` applies the same plan to `chat`, `run`, `serve`, and benchmarks. It
sets the RAM budget and context immediately; the VRAM tier is enabled only when
the current `glm` binary is linked with CUDA. Explicit flags and environment
variables keep precedence over automatic values.

Before loading the model, `coli doctor` performs a read-only readiness check and
explains whether the selected Disk/RAM/VRAM placement is runnable:

```bash
COLI_MODEL=/nvme/glm52_i4 ./coli doctor
COLI_MODEL=/nvme/glm52_i4 ./coli doctor --gpu 0 --ram 128 --json
```

Doctor validates the model directory, config, tokenizer, safetensors headers,
engine executable, available RAM, requested NVIDIA devices, CUDA linkage, and the
same placement budget used by `coli plan`. It never starts `glm`, reads tensor
payloads, imports a model framework, or creates a CUDA context. The versioned JSON
report uses stable check IDs for automation. Warnings keep exit status 0; missing
requirements or an unsafe RAM projection return 1, while invalid CLI values return 2.

The engine at runtime is pure C — python is only used by the one-time converter.

### Windows 11 (native, no WSL)

colibrì builds and runs natively on Windows 11 x86-64 with MinGW-w64. The port adds
a `_WIN32` compatibility layer in `c/compat.h` that maps POSIX I/O to the Windows API
(pread → ReadFile+OVERLAPPED, posix_fadvise no-op, aligned allocation, MoveFileEx rename,
GlobalMemoryStatusEx RAM detection). All platform differences stay in `compat.h`; the
engine source is unchanged.

**Toolchain:** GCC via [winlibs](https://winlibs.com/) or MSYS2 MinGW-w64. Tested with
GCC 16.1.0 (x86_64-ucrt-posix-seh).

```powershell
# One-time toolchain install (pick one):
scoop install mingw-winlibs                    # portable, no shell needed
# or: pacman -S mingw-w64-x86_64-gcc make     # via MSYS2

# Build (from c/ directory):
make glm.exe            # GLM-5.2 engine (static, no DLL dependencies)
make olmoe.exe          # OLMoE engine (same shims)
make iobench.exe        # disk I/O benchmark
make test-c             # run C tests
make test-python        # run Python tests (requires python)

# AVX-VNNI: Intel Alder Lake+ (and Meteor Lake+) CPUs have a 128-bit int8
# dot-product instruction (VPDPBUSD) the engine can use for ~1.3x faster
# quantized matmul. The x86-64-v3 default (portable AVX2) compiles it out;
# build for THIS machine to enable it:
make glm.exe ARCH=native                       # banner prints "idot: avx-vnni"

# Verify (tiny model, 2.4 MB):
pip install torch transformers safetensors huggingface_hub
python tools/make_glm_oracle.py                # generate tiny oracle
SNAP=./glm_tiny TF=1 ./glm.exe 64 16 16        # expect "32/32 positions"

# Run with real model:
SNAP=D:\glm52_i4 ./glm.exe 64 4 16            # batch inference
python coli chat --model D:\glm52_i4            # interactive chat
python coli serve --model D:\glm52_i4            # OpenAI-compatible API
```

**Warmup (overnight cache priming):** the engine's expert cache learns from
your workload. The included `warmup.ps1` script runs `coli run` in a loop with
diverse prompts to build the `.coli_usage` histogram unattended, so the next
real session starts with a large, accurate hot-expert pin. Each run saves usage
atomically on clean completion.

```powershell
.\warmup.ps1 -Rounds 1 -Ngen 32               # ~60-90 min, durable progress
```

**NVIDIA GPU (optional, via runtime DLL):** on Windows the engine is built with
MinGW gcc but CUDA kernels require MSVC + nvcc. The split is clean: build the
CUDA backend into a standalone `coli_cuda.dll` (nvcc + MSVC), then the host
`glm.exe` loads it at runtime via `LoadLibrary` (`c/backend_loader.c`). The host
never links cudart directly; if the DLL is absent the engine falls back to CPU
without error.

```powershell
# Prerequisites: CUDA Toolkit + MSVC Build Tools (cl.exe) + nvcc on PATH.
# Build the DLL from a shell with the MSVC environment set (vcvars64.bat or
# "x64 Native Tools Command Prompt for VS"):
make cuda-dll CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" CUDA_ARCH=sm_120

# Build the host with the runtime loader (CUDA_DLL=1 adds -DCOLI_CUDA and
# links backend_loader.o instead of cudart):
make glm.exe CUDA_DLL=1 ARCH=native

# Run with the GPU expert tier (8 GB VRAM budget here; scale to your free VRAM):
$env:COLI_CUDA="1"; $env:COLI_GPU="0"; $env:CUDA_EXPERT_GB="8"
python coli chat --model D:\glm52_i4 --topp 0.7
```

The DLL exports 11 `extern "C"` symbols (`coli_cuda_init`, `coli_cuda_matmul`,
etc.); `backend_loader.c` resolves them via `GetProcAddress` on first use.
`ColiCudaTensor*` is opaque to the host (stored, never dereferenced), so the
MSVC-allocated struct is safe across the ABI boundary. `CUDA_ARCH` must match
your GPU's compute capability (e.g. `sm_120` for Blackwell / RTX 50-series,
`sm_89` for Ada / RTX 40-series).

**Status:** Phase 1 complete (compiles, correct, static-linked). The Windows
GPU tier (runtime `coli_cuda.dll` via `LoadLibrary`) is implemented and
verified on RTX 50-series (sm_120). O_DIRECT (Phase 2) and full-model
validation against the transformers oracle remain separate workstreams.

### OpenAI-compatible API

`coli serve` keeps one model process loaded and exposes a text-only OpenAI-compatible
HTTP API. The gateway uses only the Python standard library; inference still runs in
the same dependency-free C engine.

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
`max_tokens`/`max_completion_tokens`, `temperature`, and `top_p`. The extension
`enable_thinking: true` enables GLM-5.2's reasoning block; the standard
`reasoning_effort` field also enables it unless set to `none`.

The first version is deliberately text-only and serves one generation at a time:
the 744B model stays in one persistent process, so concurrent HTTP requests queue
instead of loading duplicate model copies. Tools, image/audio input, custom stop
sequences, log probabilities, and token penalties return an explicit error rather
than being silently ignored. The default bind address is localhost; set
`COLI_API_KEY` before exposing the server beyond the machine.

Browser access from the Vite development server and Tauri local origins is enabled
by default. Repeat `--cors-origin https://your-ui.example` to allow another exact
origin, or use `--cors-origin '*'` only on a trusted local network.

The engine owns one mutable KV context, so HTTP generation uses a bounded FIFO
admission queue instead of pretending to run unsafe parallel sequences. Configure it
with `--max-queue N` (default 8) and `--queue-timeout SECONDS` (default 300), or the
`COLI_MAX_QUEUE` / `COLI_QUEUE_TIMEOUT` environment variables. Saturated and timed-out
requests receive OpenAI-shaped HTTP 429 errors before streaming headers are sent.
`GET /health` exposes active/queued/completed/rejected counters, and successful
generation responses include `x-colibri-queue-wait-ms`.

### Isolated KV contexts

`coli serve --kv-slots N` allocates up to 16 independent sequence contexts. Requests
select one with the optional integer `cache_slot` field; ordinary OpenAI clients omit
it and keep the original slot 0 behavior.

```json
{
  "model": "glm-5.2-colibri",
  "messages": [{"role": "user", "content": "Continue this conversation"}],
  "cache_slot": 1
}
```

Each slot owns its token history, compressed MLA/DSA KV memory, MTP window, and
crash-safe persistence file (`.coli_kv`, `.coli_kv.1`, ...). The engine still executes
one sequence at a time; this establishes explicit KV ownership without pretending that
threaded HTTP is continuous batching. RAM admission accounts for every configured slot.
Use `COLI_KV_SLOTS=N` as the environment equivalent. Start with a small value: at the
default 4096-token context, every slot costs hundreds of MB.

### Experimental Metal backend (Apple Silicon)

On Apple Silicon the decode profile is matmul-bound, and unified memory removes the
PCIe copy tax that keeps CUDA's streaming experts on the CPU — so colibrì has an
opt-in Metal backend that runs the **routed-expert SwiGLU (batched, zero-copy from
the RAM slabs)**, the **fused decode attention** (full MLA layer in one command
buffer, S≤4), and **prefill's large GEMMs** on the GPU. Token-exact vs the CPU path.

```bash
cd c
make glm METAL=1          # macOS only; no Xcode needed (shader compiles at runtime)
make metal-test           # standalone kernel/attention correctness vs CPU reference
COLI_METAL=1 COLI_MODEL=/path/glm52_i4 ./coli chat --ram 96
```

Measured on an M4 Max (128 GB, warm cache, MTP on): CPU 0.30 → Metal **0.42 tok/s (~1.4×)**
(best config adds `DIRECT=1`; ~3× vs this machine's first cold run).
Key design points: Metal's ~5 ms submit latency makes per-matmul dispatch a loss —
everything is batched into few command buffers per layer, and the resident experts'
GPU work is submitted *before* the missed experts' disk reads so I/O and compute
overlap. `COLI_METAL_GEMM_MIN` tunes the prefill GEMM row threshold (default 16).
Streaming, cache, MTP, DSA and the persistence formats are unchanged; every GPU
path falls back to the CPU per-block on any fault. Numerics are dequant→f32-MAC
(same as the CUDA tier); greedy outputs are byte-identical to the CPU engine.

### Experimental resident CUDA backend

colibrì includes an opt-in CUDA backend for model-resident tensors. Streaming
experts deliberately remain on the original CPU path for now: copying an expert
from NVMe to the GPU on every use would only replace the disk bottleneck with a
PCIe bottleneck. Resident quantized tensors are uploaded lazily once and reused.

```bash
cd c
make cuda-test CUDA=1                  # q8/q4/q2/f32 kernel correctness
make CUDA=1
# optional dense-path experiment (hot experts are configured below)
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 SNAP=/nvme/glm52_i4 ./glm 64 4 4
```

Requirements: Linux, an NVIDIA driver, and a CUDA Toolkit under
`/usr/local/cuda` (override with `CUDA_HOME=/path/to/cuda`). `CUDA_ARCH=native`
builds for the GPU in the current machine; set an explicit architecture when
cross-compiling. Requesting CUDA with a CPU-only binary, an invalid device, or
an unavailable runtime fails at startup instead of silently falling back.

The normal `make` build and runtime behavior are unchanged. CUDA defaults to an
expert-only accelerator. `CUDA_DENSE=1` additionally distributes resident
dense/attention projection tensors round-robin across the selected devices;
their projected footprint is reserved before the expert tier is placed. On six
RTX 5090s with a 150 GB expert tier, a warmed two-request/64-token GLM-5.2 run
improved from 1.650 to 2.157 aggregate tok/s (+30.8%) while retaining the full
expert tier. Treat this as an opt-in until the projected dense set and the 2 GB
per-device runtime reserve fit the target GPUs.
A measured `PIN` profile can promote its hottest experts into the persistent
VRAM tier while keeping the rest in RAM:

```bash
STATS=stats.txt SNAP=/nvme/glm52_i4 ./glm 64 4 4   # collect routing frequencies first
COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=16 \
PIN=stats.txt PIN_GB=160 SNAP=/nvme/glm52_i4 ./glm 64 4 4
# multi-GPU expert tier, 150 GB total budget across six 32 GB devices
COLI_CUDA=1 COLI_GPUS=0,1,2,3,4,5 CUDA_EXPERT_GB=150 \
CUDA_DENSE=1 PIN=stats.txt PIN_GB=300 RAM_GB=226 \
SNAP=/nvme/glm52_i4 ./glm 64 4 4
```

Selected experts are uploaded during startup, so capacity failures occur before
inference and the log reports their exact tensor footprint. The budget is clamped
against free VRAM after reserving the projected dense resident set and 2 GB of
runtime headroom per selected device. With `COLI_GPUS`, `CUDA_EXPERT_GB` is a
total budget across the device set; experts are assigned whole to the
least-loaded device that can hold them. Multi-GPU runs also default to
`PIN_FILL=1`: the measured hot set is placed first, then unused VRAM is filled
with zero-heat experts. `CUDA_RELEASE_HOST=1` (the multi-GPU default) releases
the RAM copy after a successful upload and reloads it from disk only if CUDA
later fails. Set either variable to `0` to restore the conservative behavior.
When host backing is released, placement is disjoint and staged: the hottest
prefix is loaded, uploaded to VRAM, and freed before the next-ranked suffix is
loaded into RAM. `PIN_GB` therefore describes the combined ranked set rather
than duplicate RAM and VRAM copies. On a 256 GB dual-socket host, moving from a
150 GB VRAM + 130 GB RAM placement to 150 GB VRAM + 150 GB RAM raised fixed-token
replay from 1.87 to 2.16 tok/s (+15.7%), reduced expert disk wait from 5.144s to
3.948s, and kept the projected RAM peak below `RAM_GB=226`. The cache cap adjusts
down automatically (54 to 40 in that run) so the larger pinned tier does not exceed
the process budget. Start lower on hosts with less available RAM.
MTP speculation defaults off on CUDA because cold draft routes increase expert
traffic; an explicit `DRAFT=n` still overrides the default.

On six RTX 5090 32 GB cards with GLM-5.2 int4, a 150 GB hot-first tier sustained
0.94 token/s over a 64-token varied prompt (87.8% expert hit rate), and reached
1.64 token/s on a warmed short prompt (99.3% hit rate). The same capacity filled
without routing heat managed only 0.29 token/s, so profile quality matters more
than raw VRAM capacity. These are single-run engineering measurements, not a
portable performance guarantee.

Current limitations: devices use independent contexts and synchronous
host-staged activation copies—there is no P2P/NCCL dependency yet. Independent
expert groups execute concurrently across devices, but a single expert is not
sharded. The kernels are correctness-first custom kernels rather than
cuBLAS/Tensor Core kernels.

For a reproducible backend A/B without the full checkpoint, generate the
deterministic 313M-parameter `glm_moe_dsa` fixture and run fixed-token replay:

```bash
cd c
python tools/make_glm_bench_model.py --output /nvme/colibri-bench-medium --device cuda
python tools/benchmark_cuda_fixture.py --model /nvme/colibri-bench-medium --gpu 0
```

The fixture has random weights and is not a language model. It exists only to
preserve the real MLA/MoE/streaming shapes and compare CPU streaming, dense-only
CUDA, CPU hot-store, and CUDA hot-expert execution with identical replay tokens.

### Web interface

`web/` contains a community-contributed browser UI (React + TypeScript, ~390
lines of source, a pure API client — it never touches the engine directly):

```bash
cd web
npm ci && npm run dev        # then point it at an OpenAI-compatible endpoint
```

It speaks the standard OpenAI Chat Completions protocol with SSE streaming, so it
works against the colibrì OpenAI-compatible server (in review, #21) or any other
compatible endpoint. Nothing leaves the endpoint you configure. The terminal
`coli chat` remains the first-class interface.

Useful knobs (env or flags): `--temp T` token sampling temperature (default 0.7 + nucleus 0.90 — tuned for int4; 0 = greedy), `--topp 0.7` adaptive expert top-p (30–40% less disk), `--ngen N` max tokens per answer (`:more` in chat continues a truncated one), `--repin N` adapt RAM/VRAM hot experts every N emitted tokens, `AUTOPIN=0` disable the learning cache's auto-pin, `THINK=1` enable GLM-5.2's reasoning block, `DRAFT=n` MTP draft depth, `GRAMMAR=g.gbnf` grammar-forced drafts for constrained JSON/NDJSON output (`GRAMMAR_DRAFT=n` caps the forced span), `TF=1` teacher-forcing validation, `PILOT=1` router-lookahead disk prefetch (experimental — see below), `CAP_RAISE=0` don't auto-grow the expert cache.

### Resource policy

`coli plan` reports the planned hot (VRAM), warm (RAM), and cold backing
(disk) tiers, the reason for each placement, and the expected bottleneck. The
default `--policy quality` and `--policy balanced` modes preserve checkpoint
quantization and router decisions unless `--topk` or `--topp` is passed; those
explicit lossy overrides print a warning and proceed.

Auto-tier plans size OpenMP from physical cores and bind workers across cores.
Memory-bound quantized kernels can regress sharply when SMT siblings compete
for limited memory channels; explicit `OMP_*` settings always take precedence.

```bash
coli plan --model /models/glm52_i4 --policy quality
coli run --auto-tier --policy quality "Explain MoE offloading"
# Explicit research-only router reduction:
coli run --policy experimental-fast --topk 4 "Benchmark prompt"
```

Disk is an immutable recovery source, not a normal decode target. If the plan
leaves cold expert bytes on disk, speed depends on cache hit rate; output
quality does not.

Cold expert reads use a deferred pipeline: resident RAM/VRAM experts execute
while missing experts are loaded in a bounded background I/O pool, then the
cold results join before the layer completes. `IO_THREADS=n` overrides the
default eight loader threads when foreground work exists. Profiling reports
both disk service time and the smaller foreground-visible wait time so overlap
is explicit rather than credited as unexplained speedup.

`--policy balanced` enables lossless live placement (`REPIN=64`). At safe
request boundaries, a per-layer LFRU score combines decaying session frequency
with recent access and replaces at most four sufficiently colder pinned
experts. `--policy quality` leaves live replacement off by default; `REPIN=0`
always disables it. Persistent `.coli_usage` history and session-local LFRU
state remain separate.

For single-token q4 CPU experts, gate and up projections share one OpenMP
dispatch while retaining the same per-row AVX2/NEON arithmetic. This removes
one thread-team launch per RAM expert without activation requantization or a
lower-precision fallback. It is a stepping stone toward a persistent native
CPU expert pool, not a replacement for one.

**The expert cache auto-sizes to your RAM** (since 2026-07-10): the engine now *raises* the LRU cap to fill your `--ram` budget instead of only lowering it. Before this fix a 128 GB machine ran with the same 8-experts/layer cache as a 16 GB one (issue #12) — **if you benchmarked colibrì before this date, rerun: your numbers were capped.**

**Router-lookahead prefetch** (`PILOT=1`, experimental): GLM-5.2's expert routing is measurably predictable *ahead of time* — applying layer L+1's router to layer L's post-attention state recalls **71.6%** of the true top-8 (vs 41.3% for "same experts as last token"). `PILOT=1` uses this to issue next-layer expert readahead from a dedicated I/O thread while the current layer computes. On our dev box the disk is already ~80% saturated, so it measures neutral; on machines where compute and disk are balanced (like the Ryzen AI 9 in issue #12: 43% disk / 46% matmul) it should overlap real work — measurements welcome.

**The learning cache**: the engine records which experts your usage actually routes to (`.coli_usage` next to the model, updated every turn) and at startup automatically pins the hottest ones in spare RAM. colibrì literally gets faster the more you use it.

**Live tier adaptation** (`--repin N`, opt-in): at safe turn boundaries, a decaying
session heat map replaces cold pinned experts with hotter streamed experts. Replacement
loads the expert from disk into the existing RAM slot; GPU-backed slots immediately
refresh the same VRAM tier budget. A 25% hysteresis and a four-swap limit prevent tier
thrashing. Persistent `.coli_usage` remains the long-term signal and is not decayed.

**Conversations reopen warm** (`.coli_kv`, since 2026-07-10): `coli chat` persists the compressed MLA KV-cache to disk after every turn (~182 KB/token, appended incrementally, crash-safe). Close the chat, reopen it tomorrow — the model still remembers the whole conversation and **zero re-prefill happens**: validated byte-identical to an uninterrupted session. `:reset` clears it, `KVSAVE=0` disables it.

## Got a better machine? Try it — here's what to expect

colibrì was built on deliberately humble hardware (12 cores, 25 GB RAM, an older DRAM-less NVMe behind a WSL2 VHDX that measured ~1 GB/s random on *this* drive — note WSL2 VHDX is not inherently slow: a community 5090 box measured 10.5 GB/s O_DIRECT through one, [#101](https://github.com/JustVugg/colibri/issues/101)). **Every one of those constraints is a knob your machine can turn up.** The engine needs: Linux (or WSL2), macOS, or **Windows 11 natively (MinGW-w64)**; gcc with OpenMP, AVX2, ≥16 GB RAM, and the ~370 GB int4 model on a local NVMe (ext4/NTFS — never a network/9p mount).

**How to test it, in order:**

```bash
cd c && ./setup.sh                 # build + architecture self-test (expects 32/32)

# 1) measure YOUR disk the way the engine uses it (parallel 19 MB random reads):
gcc -O2 -fopenmp iobench.c -o iobench
./iobench /path/to/glm52_i4/out-00069.safetensors 19 64 8 0   # buffered, 8 threads
./iobench /path/to/glm52_i4/out-00069.safetensors 19 64 8 1   # O_DIRECT (bypass cache)
# Caveat (#86): iobench reads a bounded ~1 GB shard, so buffered reads on a big-RAM box
# report the PAGE CACHE, not the disk. Use the O_DIRECT run (arg 1) for a true number, and
# run it on a shard you haven't touched this session (a prior buffered run caches its pages).
# On macOS there is no O_DIRECT — iobench uses F_NOCACHE, which stops *new* caching but can't
# evict pages a prior buffered run already resident-mapped, so a macOS "O_DIRECT" figure right
# after a buffered run still reads cache. Reboot or use a fresh shard for a real cold read.

# 2) chat; watch the per-turn stats line (tok/s, expert hit-rate, RSS):
COLI_MODEL=/path/to/glm52_i4 ./coli chat

# 3) record expert usage, then pin the hottest experts in your spare RAM:
STATS=stats.txt ./coli chat
PIN=stats.txt PIN_GB=20 ./coli chat        # scale PIN_GB to your free RAM

# 4) quality benchmarks (MMLU/HellaSwag/ARC):
./coli bench
```

**Back-of-envelope predictions** (decode is disk-bound: a cold token costs ~11.4 GB of expert reads; MTP speculation roughly halves the effective cost *once the cache is warm*; RAM turns cold reads into free cache hits):

| machine | expected |
|---|---|
| this dev box (WSL2 VHDX, ~1 GB/s, 25 GB RAM) | ~0.05–0.1 tok/s cold — proven baseline |
| native Linux, PCIe4 NVMe (~3–5 GB/s random), 32 GB | ~0.5–1 tok/s |
| PCIe5 NVMe or 2×NVMe RAID0 (~8–12 GB/s), 64 GB (PIN ~40 GB of hot experts) | ~2–4 tok/s |
| 128–256 GB RAM, 12 cores (hot experts cached) | ~2–4 tok/s — matmul-bound: ~80 GFLOP/token vs ~250 GFLOP/s of our AVX2 kernels |
| same RAM + 24–32 cores, or AVX-512/VNNI kernels | ~5–15 tok/s — interactive; kernel work is the multiplier |

These are estimates, not measurements — if you run colibrì on serious hardware, **please open an issue with your numbers**: real datapoints from better machines are exactly what this project needs next.

### Community benchmarks (measured)

Real numbers from real machines, stock build (`setup.sh`, gcc 13), greedy decoding, `--ngen 32`, MTP active:

| machine | disk (iobench, 19 MB × 64, 8 threads) | config | measured |
|---|---|---|---|
| Intel Core Ultra 7 270K Plus (24 threads) · WSL2 · 24 GB RAM · NVMe VHDX ([#2](https://github.com/JustVugg/colibri/issues/2)) | 1.96 GB/s buffered · 2.74 GB/s O_DIRECT | default | 0.07 tok/s · expert hit 3–4% · RSS 14.1 GB |
| 〃 | 〃 | `--topp 0.7` | **0.11 tok/s** · expert hit 11% · RSS 14.7 GB |
| Apple M5 Max (18 cores) · macOS · 128 GB unified · internal SSD ([#4](https://github.com/JustVugg/colibri/issues/4), [#5](https://github.com/JustVugg/colibri/issues/5)) | ~4 GB/s cold (the 14.2 GB/s reading was cache-influenced — see note) | default, MTP off | **1.06 tok/s** · expert hit 23% · RSS 21.8 GB |
| Apple M5 Max · macOS · 128 GB unified · 2 TB SSD · **Metal backend** ([#72](https://github.com/JustVugg/colibri/pull/72), [#87](https://github.com/JustVugg/colibri/issues/87)) | (macOS O_DIRECT figure unreliable — see note) | Metal on · `--ram 96` · 39.7 GB warm pin · MTP off | **1.83 tok/s** · expert hit 66% · warmed 1.11 → 1.83 over the run |
| 〃 · 46.9 GB pin (2.94M-selection history) · `--ram 110`, 1024-token run ([#103](https://github.com/JustVugg/colibri/issues/103)) | 〃 | Metal on (experts + attention) · MTP off | **2.06 tok/s** · hit 72.5% · coherent output · fastest datapoint yet (still on the pre-rebase Metal branch) |
| Mac Mini M4 Pro · macOS · **48 GB** unified · **Metal backend** ([#107](https://github.com/JustVugg/colibri/issues/107)) | 6.59 GB/s F_NOCACHE (fresh shard) | Metal on · `--ram 38` | **0.30 tok/s** (vs 0.18 CPU-only) — entry Apple Silicon on a third the RAM beats the 32-core 9950X row |
| Epyc 9654 ES · Linux · 4x16GB DDR5-4800-rdimm · Samsung PCIe Gen3 x4 NVME SSD | — | `MTP=1 DIRECT=1` | 0.31 tok/s · expert hit 35% · RSS 21.52 GB |
| Ryzen AI 9 HX 370 (Framework 13) · Arch Linux · 128 GB · WD SN850X, BTRFS zstd ([#12](https://github.com/JustVugg/colibri/issues/12)) | — | int8 MTP head · `--cap 32` · 46.7 GB auto-learned PIN | **0.37 tok/s** · expert hit 66% · MTP acceptance 52% (2.59 tok/fw) · RSS 105 GB |
| Ryzen 9 9950X (32 threads) · Linux · 123 GB · Crucial P3 QLC Gen3 ([#31](https://github.com/JustVugg/colibri/issues/31)) | 1.51 GB/s buffered | default, 2 runs from cold | 0.10 tok/s · hit 53% · profile 66% disk |
| 〃 same machine, model moved to a Samsung 9100 PRO PCIe 5.0 ([#31](https://github.com/JustVugg/colibri/issues/31)) | **8.81 GB/s** O_DIRECT | 〃 (usage history retained) | **0.28 tok/s** · hit 57% · profile flips: 32% disk / **57% matmul** |
| Ryzen AI Max+ 395 (Framework Desktop) · Ubuntu · 128 GB LPDDR5x · Intel Optane 905p PCIe 3.0 ([#39](https://github.com/JustVugg/colibri/issues/39)) | 3.27 GB/s buffered | int8 MTP head · fresh history (pure LRU, auto-raised cap 65) | 0.16 tok/s · hit 57% · profile 49% disk / 47% matmul |
| 〃 five runs later — learned pin 47.6 GB ([#39](https://github.com/JustVugg/colibri/issues/39)) | 〃 | `--temp 0.7 --topp 0.7` | **0.40 tok/s** · hit 71% · fastest non-Apple datapoint |
| Ryzen 7 9800X3D (16T) · WSL2 · 70 GB RAM · Samsung 9100 PRO PCIe 5.0 · RTX 5090 ([#101](https://github.com/JustVugg/colibri/issues/101)) | **10.51 GB/s** O_DIRECT | MTP off · learned pin 24 GB · hit 54% · OMP hot-team on | **0.41 tok/s** · disk-bound (36.5 s disk vs 24.0 s matmul) · **CUDA expert tier ≈ 0%** (AVX-512 CPU matches the 5090) · `--topp 0.7` → **0.52 tok/s** |
| EPYC 7443 (24C/48T, Zen3 AVX2) · Linux · **430 GB RAM** · NVMe RAID-Z1 via TrueNAS VM ([#104](https://github.com/JustVugg/colibri/issues/104)) | ~1 GB/s (VM overhead) | 77.5 GB pin · cap auto-raised to 194/layer · MTP off | **1.00 tok/s** · **hit 98%** · disk eliminated → **RAM-bandwidth + matmul bound** (no AVX-512/VNNI on Zen3) |
| Intel i5-12600K (10C/16T, AVX2) · **native Windows 11, no WSL** · 32 GB · MinGW GCC 16.1 ([#113](https://github.com/JustVugg/colibri/issues/113)) | buffered (no O_DIRECT on MinGW) | int8 MTP head · cold, small-RAM (cap ~2/layer) | **0.08 tok/s** · hit 3.7% · **MTP 57% acceptance** — first native-Windows datapoint, port validated |
| Ryzen 9 9950X3D2 (16C/32T, avx512-vnni) · native Linux · 121 GB · Samsung 9100 PRO **PCIe Gen5** · RTX 5090 (28 GB expert tier, 1475 pinned) ([#120](https://github.com/JustVugg/colibri/issues/120)) | **11.48 GB/s** O_DIRECT | `MTP=0 DIRECT=1 PIPE_WORKERS=16 PREFETCH=1` | **1.23 tok/s** · MTP-off wins disk-bound · fastest x86 datapoint yet |
| Ryzen AI Max+ 395 (Strix Halo, 16C/32T Zen5, avx512-vnni) · Arch Linux · 128 GB unified LPDDR5x · SK hynix P41 PCIe 4.0 ([#124](https://github.com/JustVugg/colibri/issues/124)) | — | `DIRECT=1 PIPE=1 --topp 0.7` · auto-pin | 0.06 cold → **1.10 tok/s** sustained · first Strix Halo / gfx1151 datapoint (unified memory: no discrete VRAM tier) |

Takeaways: with 24 GB of RAM the engine auto-caps the expert cache to 2 slots/layer, so decode stays cold even on a disk 2–2.7× faster than the dev box — **on small-RAM machines the RAM cap, not the disk, is the binding constraint**, exactly as the table above predicts; `--topp 0.7` alone bought a clean 1.6× end-to-end speedup. The M5 Max datapoint lands right on the table's second row: **~1 tok/s of a 744B model on a laptop SSD** — and its 14 GB/s disk shifts the bottleneck back to RAM budget and kernels. The Framework 13 rows are the cache thesis proven end-to-end on one machine: 0.29 → 0.37 tok/s (hit 28% → 66%, speculation finally engaging at 52% acceptance) just by giving the cache its RAM — int8 MTP head + a bigger cap + the learned pin. The cap part is now automatic (cap auto-raise, 2026-07-10). The 9950X pair is the cleanest bottleneck experiment yet — same machine, same history, only the disk swapped: ×5.8 disk bandwidth bought ×2.9 tokens, and the profile **flipped from 66% disk to 57% matmul**. But the crossover depends on the CPU kernel: the 9800X3D row ([#101](https://github.com/JustVugg/colibri/issues/101)) shows that with the OMP hot-team tuning on, the AVX-512 CPU matmul is fast enough that even a **10 GB/s NVMe stays disk-bound** — and there the **CUDA expert tier buys ≈ 0%**, because the CPU already matches the 5090 on expert matmul. The GPU tier earns its VRAM only when the CPU is the weak link, not by default. (Honest correction from #101: an earlier version of that report ran with the OMP tuning off, which manufactured a false matmul-bound crossover and a false +14% for CUDA — neither survived a clean re-run.)

## Quality benchmark — help wanted

**First measurement is in** ([#108](https://github.com/JustVugg/colibri/issues/108), thanks dnnspaul): the int4 container scored **62.5% mean acc_norm** on hellaswag/arc/mmlu (0-shot log-likelihood, n=40) — below the 85–95% published for full-precision GLM-5.2, but **the gap is not yet attributable to quantization.** Two confounds sit in the way: (1) 0-shot log-likelihood MC scoring badly underserves a *reasoning* model like GLM-5.2 (it never gets to think), so a large gap is expected even at fp16; (2) n=40 is ±14pp. The **decisive experiment** is the OLMoE fp16-vs-int4 A/B under this same harness (small enough to run both precisions) — that delta *is* the quantization cost with the scoring protocol cancelled out. Until it's run, 62.5% is a datapoint, not a verdict.

The code is here and ready; one command runs it end to end (it auto-downloads the datasets on first use):

```bash
cd c
pip install tokenizers datasets                # in addition to the convert deps above
./coli bench                                   # hellaswag, arc_challenge, mmlu — 40 questions each
./coli bench hellaswag --limit 200             # one task, more questions
./coli bench mmlu arc_challenge --ram 100      # pick tasks, set a RAM budget
```

It prints per-task accuracy (log-likelihood scoring, EleutherAI-harness style). **If you can run the OLMoE fp16-vs-int4 A/B (or a large-n GLM run), please open an issue with the numbers** — it's the measurement that turns 62.5% into either "int4 is fine, scoring artifact" or "quantization is the ceiling, grouped-scale is the priority."

## Supporting the project

colibrì is a one-person project, written and tested entirely on a 12-core laptop with 25 GB of RAM — the numbers above are the ceiling of what I can measure at home. If this project is useful or interesting to you and you'd like to support its development (better test hardware translates *directly* into a faster engine for everyone: real NVMe scaling data, bigger pinned caches, int2/int3 quality sweeps on real benchmarks), you can:

- ⭐ star the repo and share it;
- 🐛 open issues with benchmark numbers from your hardware;
- 💬 reach out via GitHub issues if you'd like to sponsor development or donate hardware.

Every contribution, from a datapoint to a disk, moves the ceiling.

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
web/                      browser UI (pure OpenAI-API client, community-maintained)
```

The runtime path intentionally stays flat and readable: `glm.c` plus its small
headers. Auxiliary Python and shell tooling is grouped separately and is never a
runtime dependency of the engine.

From the repository root, `make`, `make check`, and `make clean` delegate to the
engine Makefile. Existing commands run from `c/` continue to work unchanged.

## Why "colibrì"

The hummingbird weighs a few grams, hovers in place, and visits a thousand flowers a day. This engine keeps a 744-billion-parameter giant alive on hummingbird rations: 25 GB of RAM, twelve CPU cores, and a lot of disk patience.

## License

Apache 2.0. GLM-5.2 weights are released by Z.ai under MIT.
