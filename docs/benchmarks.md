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
cd c && ./setup.sh                 # build + architecture self-test (expects ~30-32/32)

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

# 3) full automated datapoint — machine info + cold/warm decode + disk, one command:
python tools/datapoint.py --snap /path/to/model --shard /path/to/container/model-00000.safetensors
# (stdlib-only; auto-selects GLM, Inkling, Kimi K3, OLMoE, Qwen3.6, or DeepSeek V4
#  from config.json; evicts the page cache before engine load; then keeps one
#  SERVE=1 engine alive for cold, warm-identical, and rotating-prompt measurements)

# 4) record expert usage, then pin the hottest experts in your spare RAM:
STATS=stats.txt ./coli chat
PIN=stats.txt PIN_GB=20 ./coli chat        # scale PIN_GB to your free RAM

# 5) quality benchmarks (MMLU/HellaSwag/ARC):
./coli bench
```

The default datapoint measures serving behavior, not repeated startup: the same
engine process and cache slot are used for one cold request, one repeated-prompt
warm request, and four requests drawn in fixed order from a diverse built-in
prompt suite. The **rotating-prompt median is the primary result** because the
expert cache must keep adapting as requests change. Warm-identical remains in
the report only as a useful cache/KV upper bound; it is not representative of a
mixed chat workload. The reported completion count, decode tok/s, expert hit
rate, and RSS come from each engine's `DONE` frame.

Use repeated `--rotate-prompt '...'` arguments to replace the built-in suite
(at least two distinct prompts), or `--rotating-runs N` to run more fixed-order
samples. Use `--mode fresh-process` only when measuring startup or OS page-cache
effects; that compatibility mode launches a new engine for every row and
therefore does not retain in-process caches.

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
| Apple M1 Ultra (20C, 48-core GPU) · Mac Studio · macOS · 128 GB unified · internal SSD · **Metal backend** · fmt=2 per-row container ([report](METAL-M1ULTRA-FMT2-REPORT.md)) | 6.89 GB/s F_NOCACHE · 8.93 GB/s buffered | Metal on (fmt=2) · `--ram 125` · `--cap 33` · 46.9 GB frozen pin · `NO_OMP`+`PIPE` · MTP off · 1024-token run | **1.50 tok/s** · hit 78.7% · RSS 104.8 GB · disk wait 57% of decode, SSD at ~93% of its iobench ceiling (1.31 at default flags, `--ram 110`) |
| Mac Mini M4 Pro · macOS · **48 GB** unified · **Metal backend** ([#107](https://github.com/JustVugg/colibri/issues/107)) | 6.59 GB/s F_NOCACHE (fresh shard) | Metal on · `--ram 38` | **0.30 tok/s** (vs 0.18 CPU-only) |
| Apple M3 (base, 4P+4E) · macOS · **16 GB unified** · internal Apple SSD ([#949](https://github.com/JustVugg/colibri/issues/949)) | 3.18 GB/s cold F_NOCACHE (post-eviction) · 7.27 GB/s buffered — no O_DIRECT on macOS, caveat #86 | **OLMoE int8** · cap 16 · TEMP=0 · CPU-only | **3.69 tok/s cold → 4.18 tok/s warm** · RSS 1.5–1.81 GB · load 0.7 s |
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
| Intel i5-13600K (14C/20T, avx-vnni) · native Linux · 62 GB · Samsung 980 PRO PCIe 4.0 (NTFS/ntfs3) · RTX 5070 Ti ([#605](https://github.com/JustVugg/colibri/issues/605)) | **5.90 GB/s** O_DIRECT | MTP off · `PIN=auto PIN_GB=20 DIRECT=1 PIPE=1 --ram 50 --cap 32 --topp 0.7` · 750 VRAM + 307 RAM pinned (5.8 GB) | **0.98 tok/s** (peak 1.07) · hit 54.7% (pin 37.9 + lru 16.7) · 360 experts/token · RSS 42.0 GB — up from 0.56 tok/s / hit 45.4% untuned |
| **6 × RTX 5090 · dual Xeon Silver 4510 · 251 GB** (author's rig, [experiment log](experiments/glm52-6x5090-2026-07-12.md)) | NVMe | `CUDA_EXPERT_GB=auto PIN_GB=all` full residency · `COLI_CUDA_PIPE=2 TC_W4A16` · DRAFT=0 | **5.8–6.8 tok/s** decode · TTFT ~13 s · hit 89–100% |
| Apple M5 Max (6P+12E) · macOS 26.5 · 128 GB unified · internal SSD · **Metal** ([#387](https://github.com/JustVugg/colibri/issues/387)) | macOS figures cache-influenced (caveat #86) | `MTP=0 CAP_RAISE=0 AUTOPIN=0 --ram 90 --cap 1` | **0.4 → 2.0 tok/s at identical output** — two additive levers, no routing substitution, no expert pruning |
| Ryzen 9 5950X (16C/32T, Zen 3, AVX2 only) · Ubuntu 26.04 · 62.7 GB · Samsung 990 PRO Gen4 · RX 7900 XTX, ROCm 7.2.4 ([#680](https://github.com/JustVugg/colibri/issues/680)) | 4.88 GB/s buffered · **5.87 GB/s** O_DIRECT | CPU-tuned | **0.77 tok/s** — the HIP expert tier underperformed the AVX2 CPU on this box |
| Ryzen AI 365 · native Windows · 32 GB DDR5-5600 · NVMe · **Vulkan** ([#999](https://github.com/JustVugg/colibri/issues/999)) | 3.32 GB/s buffered · **1.53 GB/s** O_DIRECT | `COLI_VULKAN=1 COLI_VK_DENSE=1 COLI_VK_ATTN=1` · `PIN_GB=0` | **0.19–0.24 tok/s** |
| Apple M1 Max (8P+2E, 32-core GPU) · macOS 26.6 · 64 GB unified · internal 2 TB SSD · **Metal** ([#1030](https://github.com/JustVugg/colibri/issues/1030)) | 4.4 GB/s (engine probe) | Metal on, settings-tuned | **0.13 → 0.61 tok/s from settings alone** · Metal 2.5× over CPU · **a larger `--ram` was consistently slower** |
| Ryzen 7 5800X (8C/16T, Zen 3) · Ubuntu 22.04 · 32 GB DDR4-3200 · Samsung 980 PRO ([#1070](https://github.com/JustVugg/colibri/issues/1070)) | 10.02 GB/s buffered · **2.50 GB/s** O_DIRECT | **DeepSeek V4 Flash** · `--ram 22` | **0.93 tok/s** · hit 52.5% · TTFT 16.9 s |
| Ryzen 7 9850X3D (8C/16T) · native Windows 11 · 64 GB DDR5 · WD_Black SN8100 · RTX 5090 sm_120 ([#1091](https://github.com/JustVugg/colibri/issues/1091)) | up to **10.64 GB/s** | `COLI_CUDA=1 CUDA_DENSE=1 CUDA_EXPERT_GB=auto PIN_GB=20 DIRECT=1` | **0.89 tok/s median** |
| Ryzen 9 5950X (16C/32T, Zen 3, **AVX2 only**) · Linux · 62.7 GB ([#1119](https://github.com/JustVugg/colibri/issues/1119)) | ~2.7 GB/s O_DIRECT | **DeepSeek V4 Flash** · CPU-only | **1.24 tok/s** — no I/O flag and no +40% expert cache moved it, which points at the AVX2 kernels rather than storage |
| EPYC 7282 (16C, Zen 2 AVX2) · Ubuntu 24.04 · **128 GB** DDR4-3200 8-channel · 2× Micron 3400 RAID 0 ([#1154](https://github.com/JustVugg/colibri/issues/1154)) | — | **DeepSeek V4 Flash** · `COLI_TEMP=0`, 128 tokens | **0.68 cold / 0.69 warm tok/s** · RSS 123.8 GB · load 30 s |
| Ryzen 7 4800H · Linux · 64 GB · NVMe ([#1173](https://github.com/JustVugg/colibri/issues/1173)) | 1.65 GB/s buffered · **1.37 GB/s** O_DIRECT | **DeepSeek V4 Flash** · rotating prompts, n=4 | **0.47 tok/s** (ondemand, 2.9 GHz cap) → **0.60 tok/s** (performance governor) · hit 81% · CPU 56.8 → 92.4 °C |
| i9-14900K (24C/32T) · native Windows 11 · 128 GB DDR5 · Samsung 990 Pro, SN850P partial mirror · RTX 3090 24 GB ([#1183](https://github.com/JustVugg/colibri/issues/1183)) | — | GLM-5.2 int4 · CUDA expert tier | **0.71 tok/s median** — the 24 GB VRAM point on the curve |
| Apple M4 Max (16C, 40-core GPU) · macOS · 128 GB unified · **XPG MARS 980 Gen5 over Thunderbolt 5** ([#1210](https://github.com/JustVugg/colibri/issues/1210)) | — (TB5 enclosure) | `COLI_METAL=1 DIRECT=1 MTP=0 PIPE_WORKERS=8` · `PIN_GB=90` | 0.88 tok/s at `--ram 90` → **1.06–1.17 tok/s** pinned · hit 68–73% · RSS 96.7 GB |
| Threadripper PRO 7965WX (24C/48T, Zen 4 avx512-vnni) · Linux · 123 GB · **two NVMe on independent controllers** (990 PRO 4 TB + 9100) ([#1249](https://github.com/JustVugg/colibri/issues/1249)) | 6.70 / 8.05 GB/s single-drive O_DIRECT | `DIRECT=1`, CPU-only (RTX 5090 present, not engaged) | one drive **0.80** → both **1.10 tok/s**, **+37.5%** · with `DIRECT=0` the same split is worth only +16% · expert wait 98.9 → 55.5 s |

### Two datapoints that are not rows

**16 GB is below the floor for the large models** ([#923](https://github.com/JustVugg/colibri/issues/923)):
an i5-12450H with 16 GB DDR5 and a DRAM-less QLC drive builds and passes the
self-test on native Windows, but there is no decode number to report, because a
cold token needs ~11 GB of expert reads and the RAM cap leaves almost nothing
resident. On machines this size, run OLMoE or Qwen3.6 rather than GLM-5.2: the
M3 row above is 16 GB unified and reaches 3.69–4.18 tok/s on OLMoE.

**Vulkan beat ROCm/HIP on RDNA4** ([#523](https://github.com/JustVugg/colibri/issues/523)):
on an RX 9070 XT, Vulkan measured 19–24% faster than the HIP path. The first
version of that comparison was confounded and the reporter corrected it
themselves, which is why the number is worth carrying: it is the corrected one.

### Takeaways

With 24 GB of RAM the engine auto-caps the expert cache to 2 slots/layer, so
decode stays cold even on a fast disk — **on small-RAM machines the RAM cap, not
the disk, is the binding constraint**; `--topp 0.7` alone bought a clean 1.6×
end-to-end speedup. The 9950X pair is the cleanest bottleneck experiment: same
machine, same history, only the disk swapped — ×5.8 disk bandwidth bought ×2.9
tokens, and the profile **flipped from 66% disk to 57% matmul**. But the
crossover depends on the CPU kernel: with OMP hot-team tuning on, an AVX-512 CPU
can match an RTX 5090 on expert matmul ([#101](https://github.com/JustVugg/colibri/issues/101)),
so **the GPU tier earns its VRAM only when the CPU is the weak link**. The
M1 Ultra ↔ M5 Max Metal pair is the same lesson on Apple Silicon: near-equal
GPU core counts (48 vs 40) but −33% tok/s (1.50 vs 2.24), because 57% of the
M1 Ultra decode wall is serial SSD wait at ~93% of the drive's measured
ceiling — **when experts stream from disk, the drive, not the GPU, sets the
rate**. On multi-socket hosts, NUMA placement is a further lever:
interleaving the resident weights across nodes measured **+13% (2-socket) and +40% (4-socket CPU-only)**
([#82](https://github.com/JustVugg/colibri/issues/82)). On a 2-socket Xeon Silver
4510 host with 6× RTX 5090, selective `COLI_NUMA=1` raised effective CPU-expert
bandwidth from **42.42 to 58.26/65.89 GB/s** and greedy decode from **7.66 to
9.02/9.17 tok/s** (64 tokens, `TEMP=0 DRAFT=0`, byte-identical output). Do not
blanket-interleave a GPU host: it also spreads DMA staging pages and has measured
up to a 10× regression; generated plans enable only the selective slab policy.

Three lessons from the 2026-08 datapoints. **A second drive on an independent
controller is real bandwidth**: the Threadripper pair measured +37.5% from
adding one, and the gain more than doubles under `DIRECT=1`, because buffered
reads spend the second controller's bandwidth refilling a page cache the engine
does not need ([#1249](https://github.com/JustVugg/colibri/issues/1249)). That is
the README's multi-SSD hypothesis measured rather than assumed. **On AVX2-only
CPUs the kernels bind before the disk does**: the Zen 3 DeepSeek V4 Flash run
did not move for any I/O flag or for 40% more expert cache
([#1119](https://github.com/JustVugg/colibri/issues/1119)), which is the
opposite of the small-RAM machines above and says the AVX-512 kernels are where
that box's speed is. And **settings are worth more than hardware on Apple
Silicon**: the M5 Max went 0.4 → 2.0 and the M1 Max 0.13 → 0.61 at identical
output, with a *larger* `--ram` measuring consistently slower on the M1
([#387](https://github.com/JustVugg/colibri/issues/387),
[#1030](https://github.com/JustVugg/colibri/issues/1030)).

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
