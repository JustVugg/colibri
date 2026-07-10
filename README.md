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

The engine is a single C file (`c/glm.c`, ~1,300 lines) plus small headers. No BLAS, no Python at runtime, no GPU.

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
- **DSA sparse attention: in progress** — the lightning-indexer weights (a ~108 GB extraction from the FP8 repo, `--indexer` converter mode) are downloading; the indexer forward lands next. Until then attention is dense and exact for contexts ≤ 2048 tokens.
- **Batch-union MoE**: in prefill (and MTP verification), each unique expert of the batch is read once and applied to every position that routes to it.
- **Byte-level BPE tokenizer in C** (GPT-2-style with Unicode-property regex, 320k merges).
- **RAM safety**: the expert cache is auto-sized from `MemAvailable` at startup — an honest peak projection (working set, KV, MTP row, reconstruction buffers) so the kernel OOM-killer never fires.
- **Offline FP8→int4 converter** (`c/convert_fp8_to_int4.py`): downloads one shard at a time (~5 GB), dequants (128×128 block scales), requantizes to the engine's container, deletes the shard — the 756 GB FP8 checkpoint never needs to exist on disk at once. Resumable.

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

Useful knobs (env or flags): `--temp T` token sampling temperature (default 0.7 + nucleus 0.90 — tuned for int4; 0 = greedy), `--topp 0.7` adaptive expert top-p (30–40% less disk), `--ngen N` max tokens per answer (`:piu` in chat continues a truncated one), `AUTOPIN=0` disable the learning cache's auto-pin, `THINK=1` enable GLM-5.2's reasoning block, `DRAFT=n` MTP draft depth, `TF=1` teacher-forcing validation.

**The learning cache**: the engine records which experts your usage actually routes to (`.coli_usage` next to the model, updated every turn) and at startup automatically pins the hottest ones in spare RAM. colibrì literally gets faster the more you use it.

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
| Epyc 9654 ES · Linux · 4x16GB DDR5-4800-rdimm · Samsung PCIe Gen3 x4 NVME SSD | MTP=1 DIRECT=1 | default | 0.31 tok/s · expert hit 35% · RSS 21.52 GB |

Takeaways: with 24 GB of RAM the engine auto-caps the expert cache to 2 slots/layer, so decode stays cold even on a disk 2–2.7× faster than the dev box — **on small-RAM machines the RAM cap, not the disk, is the binding constraint**, exactly as the table above predicts; `--topp 0.7` alone bought a clean 1.6× end-to-end speedup. The M5 Max datapoint lands right on the table's second row: **~1 tok/s of a 744B model on a laptop SSD** — and its 14 GB/s disk shifts the bottleneck back to RAM budget and kernels.

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
c/glm.c          the engine (GLM-5.2 forward, streaming MoE, MTP, serve mode)
c/st.h           safetensors reader: pread + fadvise, no mmap (RSS stays flat)
c/tok.h          byte-level BPE tokenizer in C
c/coli           CLI: chat / run / bench / convert / info
c/iobench.c      parallel disk microbenchmark (measures what the engine feels)
c/convert_fp8_to_int4.py   disk-safe FP8 → int4 converter
c/make_glm_oracle.py       tiny-random oracle generator for validation
c/olmoe.c        stage-A engine (OLMoE), first validation target
```

## Why "colibrì"

The hummingbird weighs a few grams, hovers in place, and visits a thousand flowers a day. This engine keeps a 744-billion-parameter giant alive on hummingbird rations: 25 GB of RAM, twelve CPU cores, and a lot of disk patience.

## License

Apache 2.0. GLM-5.2 weights are released by Z.ai under MIT.
