<p align="center">
  <img src="assets/colibri.svg" width="500" alt="colibrì — piccolo motore, modello immenso">
</p>

**Tiny engine, immense model.** Run **GLM-5.2 (744B-parameter MoE)** on a consumer machine with ~25 GB of RAM — in pure C, with zero dependencies, by streaming experts from disk.

```
$ ./coli chat
  🐦 colibrì v1.0 — GLM-5.2 · 744B MoE · int4 · streaming CPU
  ✓ pronto in 32s · residente 9.9 GB
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
- **Native MTP speculative decoding** — GLM-5.2's own multi-token-prediction head (layer 78) drafts tokens that the main model verifies in one batched forward. **The head must be int8** (the converter does this by default): at int4 draft acceptance collapses to 0–4% and speculation never engages; at int8 it's 39–59% acceptance, **2.2–2.8 tokens/forward** (community-measured, [#8](https://github.com/JustVugg/colibri/issues/8)). Lossless — *and stays lossless under sampling* via rejection sampling. Honest caveat from the same measurement: on a **cold** cache each verified draft routes to extra experts (~660 → ~1100 expert-loads/token), so speculation can be a net *time* loss until the cache/pin warms up — the adaptive guard and `DRAFT=0` are there for that.
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

| metric | value |
|---|---|
| model on disk (int4 container) | ~370 GB |
| resident RAM (dense, int4) | 9.9 GB |
| load time | ~30 s |
| peak RSS during chat | ~20 GB (auto-capped) |
| cold decode cost | ~11 GB disk reads/token (75 layers × 8 experts) |
| disk ceiling (VHDX random) | ~1 GB/s → ~0.05–0.1 tok/s cold |
| MTP speculation (int8 head) | 2.2–2.8 tok/forward measured ([#8](https://github.com/JustVugg/colibri/issues/8)) |

This is not fast. It is a 744B frontier-class model **answering correctly on a machine that costs less than one H100 fan**. Warm cache, pinned hot experts and MTP push the useful-response latency down considerably; the physics of the disk does the rest.

### SSD note
Cold starts are heavy on random reads (~11 GB/token), but reads don't meaningfully wear an SSD — colibrì's streaming is read-only. The real concerns under heavy use are (1) **swap traffic** if the system runs out of RAM (writes do wear the drive — keep a sane `--ram` budget; colibrì's auto-budget is designed to stay clear of swap) and (2) **sustained thermals**: hours at full read duty cycle will heat cheaper drives. Monitor drive temperature and health.

## Download the model

A pre-converted **GLM-5.2 int4** model for colibrì is available on Hugging Face:

**https://huggingface.co/jlnsrk/GLM-5.2-colibri-int4**

If the MTP files there are still the int4 head (see [#8](https://github.com/JustVugg/colibri/issues/8) — sizes `1765523544/2686077736/536747200` = int4, unusable), grab the **int8 MTP heads** from the community clone by matey-0: **https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp**

Download the repository and point `COLI_MODEL` to its directory:

```bash
COLI_MODEL=/path/to/GLM-5.2-colibri-int4 ./coli chat
```

This skips the FP8 → int4 conversion step entirely.

Thanks DatPat for your help!

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

The engine at runtime is pure C — python is only used by the one-time converter.

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
expert-only accelerator: resident dense/attention tensors stay on CPU because
fixture measurements show that moving them does not help while expert I/O is
the bottleneck. `CUDA_DENSE=1` keeps the earlier all-resident experimental path.
A measured `PIN` profile can promote its hottest experts into the persistent
VRAM tier while keeping the rest in RAM:

```bash
STATS=stats.txt SNAP=/nvme/glm52_i4 ./glm 64 4 4   # collect routing frequencies first
COLI_CUDA=1 COLI_GPU=0 CUDA_EXPERT_GB=16 \
PIN=stats.txt PIN_GB=160 SNAP=/nvme/glm52_i4 ./glm 64 4 4
# multi-GPU expert tier, 96 GB total budget across six devices
COLI_CUDA=1 COLI_GPUS=0,1,2,3,4,5 CUDA_EXPERT_GB=96 \
PIN=stats.txt PIN_GB=160 SNAP=/nvme/glm52_i4 ./glm 64 4 4
```

Selected experts are uploaded during startup, so capacity failures occur before
inference and the log reports their exact tensor footprint. The budget is clamped
against free VRAM after reserving the projected dense resident set and 2 GB of
runtime headroom per selected device. With `COLI_GPUS`, `CUDA_EXPERT_GB` is a
total budget across the device set; experts are assigned whole to the
least-loaded device that can hold them. A NUMA-local RAM backing store is not
implemented yet.

Current limitations: devices use independent contexts and synchronous
host-staged activation copies—there is no P2P/NCCL dependency yet. The kernels
are correctness-first custom kernels rather than cuBLAS/Tensor Core kernels.
This draft intentionally makes no end-to-end speedup claim before the full model
is benchmarked.

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

Useful knobs (env or flags): `--temp T` token sampling temperature (default 0.7 + nucleus 0.90 — tuned for int4; 0 = greedy), `--topp 0.7` adaptive expert top-p (30–40% less disk), `--ngen N` max tokens per answer (`:piu` in chat continues a truncated one), `--repin N` adapt RAM/VRAM hot experts every N emitted tokens, `AUTOPIN=0` disable the learning cache's auto-pin, `THINK=1` enable GLM-5.2's reasoning block, `DRAFT=n` MTP draft depth, `TF=1` teacher-forcing validation, `PILOT=1` router-lookahead disk prefetch (experimental — see below), `CAP_RAISE=0` don't auto-grow the expert cache.

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

colibrì was built on deliberately humble hardware (12 cores, 25 GB RAM, NVMe behind a WSL2 VHDX that caps random reads at ~1 GB/s). **Every one of those constraints is a knob your machine can turn up.** The engine needs: Linux (or WSL2), gcc with OpenMP, AVX2, ≥16 GB RAM, and the ~370 GB int4 model on a local NVMe (ext4 — never a network/9p mount).

**How to test it, in order:**

```bash
cd c && ./setup.sh                 # build + architecture self-test (expects 32/32)

# 1) measure YOUR disk the way the engine uses it (parallel 19 MB random reads):
gcc -O2 -fopenmp iobench.c -o iobench
./iobench /path/to/glm52_i4/out-00069.safetensors 19 64 8 0   # buffered, 8 threads
./iobench /path/to/glm52_i4/out-00069.safetensors 19 64 8 1   # O_DIRECT

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
| Apple M5 Max (18 cores) · macOS · 128 GB unified · internal SSD ([#4](https://github.com/JustVugg/colibri/issues/4), [#5](https://github.com/JustVugg/colibri/issues/5)) | 14.2 GB/s O_DIRECT | default, MTP off | **1.06 tok/s** · expert hit 23% · RSS 21.8 GB |
| Epyc 9654 ES · Linux · 4x16GB DDR5-4800-rdimm · Samsung PCIe Gen3 x4 NVME SSD | — | `MTP=1 DIRECT=1` | 0.31 tok/s · expert hit 35% · RSS 21.52 GB |
| Ryzen AI 9 HX 370 (Framework 13) · Arch Linux · 128 GB · WD SN850X, BTRFS zstd ([#12](https://github.com/JustVugg/colibri/issues/12)) | — | int8 MTP head · `--cap 32` · 46.7 GB auto-learned PIN | **0.37 tok/s** · expert hit 66% · MTP acceptance 52% (2.59 tok/fw) · RSS 105 GB |

Takeaways: with 24 GB of RAM the engine auto-caps the expert cache to 2 slots/layer, so decode stays cold even on a disk 2–2.7× faster than the dev box — **on small-RAM machines the RAM cap, not the disk, is the binding constraint**, exactly as the table above predicts; `--topp 0.7` alone bought a clean 1.6× end-to-end speedup. The M5 Max datapoint lands right on the table's second row: **~1 tok/s of a 744B model on a laptop SSD** — and its 14 GB/s disk shifts the bottleneck back to RAM budget and kernels. The Framework 13 rows are the cache thesis proven end-to-end on one machine: 0.29 → 0.37 tok/s (hit 28% → 66%, speculation finally engaging at 52% acceptance) just by giving the cache its RAM — int8 MTP head + a bigger cap + the learned pin. The cap part is now automatic (cap auto-raise, 2026-07-10).

## Quality benchmark — help wanted

We have never measured how much the int4 quantization costs in accuracy — the harness is built and wired, but scoring is one forward per answer option, and on the dev box's ~1 GB/s disk a full run takes the better part of a day. **This is the single most valuable thing a faster machine can contribute.** The code is here and ready; one command runs it end to end (it auto-downloads the datasets on first use):

```bash
cd c
./coli bench                                   # hellaswag, arc_challenge, mmlu — 40 questions each
./coli bench hellaswag --limit 200             # one task, more questions
./coli bench mmlu arc_challenge --ram 100      # pick tasks, set a RAM budget
```

It prints per-task accuracy (log-likelihood scoring, EleutherAI-harness style). Published full-precision GLM-5.2 scores on these tasks sit around 85–95%; if our int4 container lands within a few points, the quantization is validated — if it doesn't, we know to invest in mixed / grouped-scale quantization. **If you have the hardware to run this, please open an issue with the numbers** — it's the measurement the project is missing.

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
