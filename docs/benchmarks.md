# Benchmarks & measured numbers

Everything on this page is a measurement, not a promise. If you run colibrì on
hardware not listed here, **please open an issue with your numbers** — real
datapoints are what move this project.

## Reference numbers (the original dev box: WSL2, 12 cores, 25 GB RAM, NVMe via VHDX)

Detailed GPU experiment: [GLM-5.2 on 6× RTX 5090](experiments/glm52-6x5090-2026-07-12.md) —
full expert residency across VRAM+RAM reaches **6.84 tok/s** single-request decode.

| metric | value |
|---|---|
| model on disk (int4 container) | ~370 GB |
| resident RAM (dense, int4) | 9.9 GB |
| load time | ~30 s |
| peak RSS during chat | ~20 GB (auto-capped) |
| cold decode cost | ~11 GB disk reads/token (75 layers × 8 experts) |
| disk ceiling (this dev box's drive) | ~1 GB/s → ~0.05–0.1 tok/s cold |
| MTP speculation (int8 head) | 2.2–2.8 tok/forward measured ([#8](https://github.com/JustVugg/colibri/issues/8)) |

This is not fast. It is a 744B frontier-class model **answering correctly on a
machine that costs less than one H100 fan**. Warm cache, pinned hot experts and
MTP push the useful-response latency down considerably; the physics of the disk
does the rest.

### SSD note

Cold starts are heavy on random reads (~11 GB/token), but reads don't
meaningfully wear an SSD — colibrì's streaming is read-only. The real concerns
under heavy use are (1) **swap traffic** if the system runs out of RAM (writes
do wear the drive — keep a sane `--ram` budget; colibrì's auto-budget is designed
to stay clear of swap) and (2) **sustained thermals**: hours at full read duty
cycle will heat cheaper drives. Monitor drive temperature and health.

## Test your machine, in order

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

## Back-of-envelope predictions

Decode is disk-bound: a cold token costs ~11.4 GB of expert reads; MTP
speculation roughly halves the effective cost *once the cache is warm*; RAM
turns cold reads into free cache hits.

| machine | expected |
|---|---|
| the dev box (WSL2 VHDX, ~1 GB/s, 25 GB RAM) | ~0.05–0.1 tok/s cold — proven baseline |
| native Linux, PCIe4 NVMe (~3–5 GB/s random), 32 GB | ~0.5–1 tok/s |
| PCIe5 NVMe or 2×NVMe RAID0 (~8–12 GB/s), 64 GB (PIN ~40 GB of hot experts) | ~2–4 tok/s |
| 128–256 GB RAM, 12 cores (hot experts cached) | ~2–4 tok/s — matmul-bound: ~80 GFLOP/token vs ~250 GFLOP/s of our AVX2 kernels |
| same RAM + 24–32 cores, or AVX-512/VNNI kernels | ~5–15 tok/s — interactive; kernel work is the multiplier |

These are estimates, not measurements.

## Community benchmarks (measured)

Real numbers from real machines, stock build (`setup.sh`, gcc 13), greedy decoding, `--ngen 32`, MTP active:

| machine | disk (iobench, 19 MB × 64, 8 threads) | config | measured |
|---|---|---|---|
| Intel Core Ultra 7 270K Plus (24 threads) · WSL2 · 24 GB RAM · NVMe VHDX ([#2](https://github.com/JustVugg/colibri/issues/2)) | 1.96 GB/s buffered · 2.74 GB/s O_DIRECT | default | 0.07 tok/s · expert hit 3–4% · RSS 14.1 GB |
| 〃 | 〃 | `--topp 0.7` | **0.11 tok/s** · expert hit 11% · RSS 14.7 GB |
| Apple M5 Max (18 cores) · macOS · 128 GB unified · internal SSD ([#4](https://github.com/JustVugg/colibri/issues/4), [#5](https://github.com/JustVugg/colibri/issues/5)) | ~4 GB/s cold (the 14.2 GB/s reading was cache-influenced — see note) | default, MTP off | **1.06 tok/s** · expert hit 23% · RSS 21.8 GB |
| Apple M5 Max · macOS · 128 GB unified · 2 TB SSD · **Metal backend** ([#72](https://github.com/JustVugg/colibri/pull/72), [#87](https://github.com/JustVugg/colibri/issues/87)) | (macOS O_DIRECT figure unreliable — see note) | Metal on · `--ram 96` · 39.7 GB warm pin · MTP off | **1.83 tok/s** · expert hit 66% · warmed 1.11 → 1.83 over the run |
| 〃 · 46.9 GB pin (2.94M-selection history) · `--ram 110`, 1024-token run ([#103](https://github.com/JustVugg/colibri/issues/103)) | 〃 | Metal on (experts + attention) · MTP off | **2.06 tok/s** · hit 72.5% · coherent output |
| Mac Mini M4 Pro · macOS · **48 GB** unified · **Metal backend** ([#107](https://github.com/JustVugg/colibri/issues/107)) | 6.59 GB/s F_NOCACHE (fresh shard) | Metal on · `--ram 38` | **0.30 tok/s** (vs 0.18 CPU-only) |
| Epyc 9654 ES · Linux · 4x16GB DDR5-4800-rdimm · Samsung PCIe Gen3 x4 NVME SSD | — | `MTP=1 DIRECT=1` | 0.31 tok/s · expert hit 35% · RSS 21.52 GB |
| Ryzen AI 9 HX 370 (Framework 13) · Arch Linux · 128 GB · WD SN850X, BTRFS zstd ([#12](https://github.com/JustVugg/colibri/issues/12)) | — | int8 MTP head · `--cap 32` · 46.7 GB auto-learned PIN | **0.37 tok/s** · expert hit 66% · MTP acceptance 52% (2.59 tok/fw) · RSS 105 GB |
| Ryzen 9 9950X (32 threads) · Linux · 123 GB · Crucial P3 QLC Gen3 ([#31](https://github.com/JustVugg/colibri/issues/31)) | 1.51 GB/s buffered | default, 2 runs from cold | 0.10 tok/s · hit 53% · profile 66% disk |
| 〃 same machine, model moved to a Samsung 9100 PRO PCIe 5.0 ([#31](https://github.com/JustVugg/colibri/issues/31)) | **8.81 GB/s** O_DIRECT | 〃 (usage history retained) | **0.28 tok/s** · hit 57% · profile flips: 32% disk / **57% matmul** |
| Ryzen AI Max+ 395 (Framework Desktop) · Ubuntu · 128 GB LPDDR5x · Intel Optane 905p PCIe 3.0 ([#39](https://github.com/JustVugg/colibri/issues/39)) | 3.27 GB/s buffered | int8 MTP head · fresh history (pure LRU, auto-raised cap 65) | 0.16 tok/s · hit 57% · profile 49% disk / 47% matmul |
| 〃 five runs later — learned pin 47.6 GB ([#39](https://github.com/JustVugg/colibri/issues/39)) | 〃 | `--temp 0.7 --topp 0.7` | **0.40 tok/s** · hit 71% |
| Ryzen 7 9800X3D (16T) · WSL2 · 70 GB RAM · Samsung 9100 PRO PCIe 5.0 · RTX 5090 ([#101](https://github.com/JustVugg/colibri/issues/101)) | **10.51 GB/s** O_DIRECT | MTP off · learned pin 24 GB · hit 54% · OMP hot-team on | **0.41 tok/s** · disk-bound (36.5 s disk vs 24.0 s matmul) · **CUDA expert tier ≈ 0%** (AVX-512 CPU matches the 5090) · `--topp 0.7` → **0.52 tok/s** |
| EPYC 7443 (24C/48T, Zen3 AVX2) · Linux · **430 GB RAM** · NVMe RAID-Z1 via TrueNAS VM ([#104](https://github.com/JustVugg/colibri/issues/104)) | ~1 GB/s (VM overhead) | 77.5 GB pin · cap auto-raised to 194/layer · MTP off | **1.00 tok/s** · **hit 98%** · disk eliminated → **RAM-bandwidth + matmul bound** |
| Intel i5-12600K (10C/16T, AVX2) · **native Windows 11, no WSL** · 32 GB · MinGW GCC 16.1 ([#113](https://github.com/JustVugg/colibri/issues/113)) | buffered (no O_DIRECT on MinGW) | int8 MTP head · cold, small-RAM (cap ~2/layer) | **0.08 tok/s** · hit 3.7% · **MTP 57% acceptance** — first native-Windows datapoint |
| Ryzen 9 9950X3D2 (16C/32T, avx512-vnni) · native Linux · 121 GB · Samsung 9100 PRO **PCIe Gen5** · RTX 5090 (28 GB expert tier, 1475 pinned) ([#120](https://github.com/JustVugg/colibri/issues/120)) | **11.48 GB/s** O_DIRECT | `MTP=0 DIRECT=1 PIPE_WORKERS=16 PREFETCH=1` | **1.23 tok/s** |
| Ryzen AI Max+ 395 (Strix Halo, 16C/32T Zen5, avx512-vnni) · Arch Linux · 128 GB unified LPDDR5x · SK hynix P41 PCIe 4.0 ([#124](https://github.com/JustVugg/colibri/issues/124)) | — | `DIRECT=1 PIPE=1 --topp 0.7` · auto-pin | 0.06 cold → **1.10 tok/s** sustained · later **1.83 tok/s** on current dev with `DIRECT=1 PIPE=1 PILOT_REAL=1 PILOT_TWO=1` ([#200](https://github.com/JustVugg/colibri/issues/200)) |
| Intel Core Ultra 9 185H (16C/22T, avx-vnni) · **native Windows 11, no WSL** · 32 GB · Crucial P3 QLC NTFS · RTX 5070 Ti ([#128](https://github.com/JustVugg/colibri/issues/128), [#273](https://github.com/JustVugg/colibri/issues/273)) | — | int8 MTP head · warm cache · GPU-resident pipeline at decode | 0.03 cold → 0.5 warm CPU → **1.07 tok/s** with the pipe2 decode gate (#274) |
| Dell Pro Max GB10 (DGX Spark: Grace, **aarch64 i8mm/sve2**) · Linux · 121 GB unified LPDDR5x · GB10 sm_121 ([#136](https://github.com/JustVugg/colibri/issues/136), [#161](https://github.com/JustVugg/colibri/issues/161)) | **5.58 GB/s** O_DIRECT | int8 MTP head · warm cache | 0.50 tok/s warm · **2.4 tok/s full-k8**, **3.33 tok/s** with `CACHE_ROUTE` (#199) |
| **6 × RTX 5090 · dual Xeon Silver 4510 · 251 GB** (author's rig, [experiment log](experiments/glm52-6x5090-2026-07-12.md)) | NVMe | `CUDA_EXPERT_GB=auto PIN_GB=all` full residency · `COLI_CUDA_PIPE=2 TC_W4A16` · DRAFT=0 | **5.8–6.8 tok/s** decode · TTFT ~13 s · hit 89–100% |

### Takeaways

With 24 GB of RAM the engine auto-caps the expert cache to 2 slots/layer, so
decode stays cold even on a fast disk — **on small-RAM machines the RAM cap, not
the disk, is the binding constraint**; `--topp 0.7` alone bought a clean 1.6×
end-to-end speedup. The 9950X pair is the cleanest bottleneck experiment: same
machine, same history, only the disk swapped — ×5.8 disk bandwidth bought ×2.9
tokens, and the profile **flipped from 66% disk to 57% matmul**. But the
crossover depends on the CPU kernel: with OMP hot-team tuning on, an AVX-512 CPU
can match an RTX 5090 on expert matmul ([#101](https://github.com/JustVugg/colibri/issues/101)),
so **the GPU tier earns its VRAM only when the CPU is the weak link**. On
multi-socket hosts, NUMA placement is a further lever: interleaving the resident
weights across nodes measured **+13% (2-socket) and +40% (4-socket CPU-only)**
([#82](https://github.com/JustVugg/colibri/issues/82)). On a 2-socket Xeon Silver
4510 host with 6× RTX 5090, selective `COLI_NUMA=1` raised effective CPU-expert
bandwidth from **42.42 to 58.26/65.89 GB/s** and greedy decode from **7.66 to
9.02/9.17 tok/s** (64 tokens, `TEMP=0 DRAFT=0`, byte-identical output). Do not
blanket-interleave a GPU host: it also spreads DMA staging pages and has measured
up to a 10× regression; generated plans enable only the selective slab policy.

## Quality benchmark

**Measured** ([#108](https://github.com/JustVugg/colibri/issues/108)): the int4
container scored **62.5% mean acc_norm** on hellaswag/arc/mmlu (0-shot
log-likelihood, n=40) — but 0-shot MC scoring underserves a reasoning model, and
the OLMoE fp16-vs-int4 A/B under the same harness measured the pure quantization
cost at **-8.2pp**, concentrated on the hardest task (per-row int4 scales erode
the small logit margins hard questions depend on — grouped scales recover ~63%
of that loss, see [#225](https://github.com/JustVugg/colibri/issues/225)). The
scale-granularity/rotation/lattice ablation lives in
`tools/quant_ablation.py` ([#81](https://github.com/JustVugg/colibri/issues/81)).

```bash
cd c
pip install tokenizers datasets
./coli bench                                   # hellaswag, arc_challenge, mmlu — 40 questions each
./coli bench hellaswag --limit 200             # one task, more questions
./coli bench mmlu arc_challenge --ram 100      # pick tasks, set a RAM budget
```
