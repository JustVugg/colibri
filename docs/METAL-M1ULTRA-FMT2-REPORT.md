# Metal backend on M1 Ultra (fmt=2) — performance report: the SSD, not the GPU, sets the speed

**TL;DR:** Best measured **1.50 tok/s** (`--ram 125`, tuned) vs the M5 Max's 2.24 tok/s — **−33% with near-equal GPU core counts (48 vs 40)**. The gap is not in the GPU: **disk wait is 55–60% of the decode wall in every config, the decode budget is fully serial, and the drive is already running at ~93% of its measured ceiling** (6.4 GB/s effective in-decode vs 6.89 GB/s iobench F_NOCACHE). On this workload the SSD contains the GPU: more GPU cores buy nothing. Secondary findings: the M5 Max OMP active-spin trap does **not** reproduce on M1 Ultra; `PIPE` is the only tuning lever that matters (+6.9%), `PIPE_WORKERS=8` is the sweet spot; MTP is a strict loss at 128 GB (memory knee). Companion to the [M5 Max report](METAL-M5MAX-PERF-REPORT.md) (#72, #103, #116) — same methodology, same prompt, same frozen expert history.

## Setup (held fixed across every run)

- **Hardware:** Mac Studio, Apple M1 Ultra — 20 CPU cores (16 P + 4 E), 48-core GPU, 128 GB unified memory, macOS 26.5.2, internal SSD; `sudo sysctl iogpu.wired_limit_mb=120832` (118 GB)
- **Engine:** colibri dev @ `c024a46` (v1.4.0, post-rebase); the pre-rebase comparison runs used the v1.2.0-era build (see "Rebase check")
- **Model:** GLM-5.2 int4 (744B MoE, 78 layers, 256 experts, topk=8) converted to **fmt=2 (per-row int4)** from the official fmt=4 (g64) container with `c/tools/convert_fmt4_to_fmt2.py`. fmt=2 is the format the Metal backend fully supports (fmt=4 Metal dispatch is still open: #585/#587). **Quality caveat:** per-row scales cost ~9pp vs grouped (see the benchmarks.md quality section) — fmt=2 here is a Metal-compatibility vehicle, not a quality recommendation.
- **Workload:** `./coli run "Compare the myths of Lucifer and Prometheus"`, 1024 tokens generated (warmup 128). Prefill ~11 s, excluded from tok/s.
- **Constant flags:** `COLI_METAL=1 DIRECT=1 MTP=0 --cap 33`
- **Expert history frozen:** the same 6.37M-selection usage snapshot is restored before every run → identical pin (2478 experts / 46.9 GB mlock'd, no compression) in all runs except `ram105` (auto-capped to 41.9 GB to preserve LRU room).
- **Disk (iobench, 19 MB × 64, 8 threads — benchmarks.md methodology):** **6.89 GB/s F_NOCACHE**, 8.93 GB/s buffered (cache-influenced).
- **Single run per config**, warm page cache. The engine is non-deterministic run-to-run (see the M5 Max report's caveats); treat ~±0.05 tok/s as noise.

## Results

### Tuning at `--ram 110` (mirrors M5 Max configs A–E)

| config | tok/s | decode (s) | hit % | disk wait (s) | expert-matmul (s) | attention (s) | RSS (GB) |
|---|---|---|---|---|---|---|---|
| **A** — baseline | 1.31 | 781.2 | 76.5 | 431.7 | 188.3 | 149.6 | 97.9 |
| **C** — + `PIPE=1 PIPE_WORKERS=8` | 1.40 | 732.2 | 76.2 | 433.1 | 164.2 | 123.7 | 97.9 |
| **D** — + `COLI_NO_OMP_TUNE=1` | 1.32 | 775.0 | 76.8 | 424.8 | 189.2 | 149.4 | 97.9 |
| **E** — `NO_OMP` + `PIPE` | 1.40 | 732.8 | 76.1 | 432.5 | 164.7 | 124.4 | 97.9 |
| E with `PIPE_WORKERS=12` | 1.41 | 726.8 | 76.6 | 426.5 | 165.5 | 123.7 | 97.9 |
| E with `PIPE_WORKERS=16` | 1.37 | 745.0 | 75.9 | 441.5 | 167.6 | 124.9 | 97.9 |

### RAM sweep (E config)

| `--ram` | tok/s | hit % | disk wait (s) | RSS (GB) |
|---|---|---|---|---|
| 105 | 1.35 | 74.9 | 455.2 | 93.3 |
| 112 | 1.38 | 76.0 | 439.4 | 98.8 |
| 115 | 1.37 | 75.8 | 441.2 | 100.2 |
| 118 | 1.44 | 77.5 | 411.6 | 101.6 |
| 120 | 1.45 | 77.6 | 375.6 | 102.5 |
| **125** | **1.50** | **78.7** | **390.2** | **104.8** |

Winner command:

```bash
COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8 \
  ./coli run --model /path/to/GLM-5.2-colibri-int4-perrow \
  "Compare the myths of Lucifer and Prometheus" --ram 125 --cap 33 --ngen 1024
```

## Why the SSD, not the GPU, sets the speed

**The decode budget is fully serial.** At `--ram 125`: 390.2 s disk wait + 162.1 s expert-matmul + 121.1 s attention + 11.1 s other = 684.4 s ≈ the 684.55 s measured decode wall. Wait and compute do not overlap, so every second of disk wait is a second of throughput, and the fixed compute (~294 s) bounds the unreachable zero-miss asymptote at ~3.5 tok/s.

**The drive is already at its ceiling.** ~128 misses/token × ~19 MB/expert ≈ 2.4 GB of expert reads per token, served in 0.381 s of wait per token ≈ **6.4 GB/s effective — 93% of the 6.89 GB/s iobench F_NOCACHE measurement.** There is no idle disk left to feed a faster GPU.

**More RAM barely helps.** Across the sweep, each +5 GB of pin budget buys only ~0.5–1 pp hit rate (misses/token moves 150 → 128 from ram 105 → 125). You cannot pin your way out of a 744B model with 128 GB.

**Consequence: GPU core count is contained by storage.** M1 Ultra has 48 GPU cores vs the M5 Max's 40 (+20%), yet delivers 1.50 vs 2.24 tok/s (−33%). The M5 Max's storage path feeds its GPU; the M1 Ultra's drive is the wall.

**Secondary (and it doesn't change the verdict): the M1 Ultra GPU is also per-core slower at this work.** Expert GPU kernel time is 71.4 s vs ~34.5 s on the M5 Max for identical work (618k experts-on-GPU, 2.07×); attention kernel 90.4 s vs 79 s. Plausibly M1-era cores plus dual-die fabric overhead — but even deleting *all* GPU kernel time would only approach the ~3.5 tok/s zero-miss asymptote. The disk term dominates every realistic config.

## What happened, phase by phase (tuning at `--ram 110`)

**The M5 Max OMP active-spin trap does not reproduce (A vs D).** On the M5 Max, the #77 hot-team spin steals the shared SoC power budget and triples attention GPU kernel time (76 → 223 s). Here, attention GPU time is identical between A and D (149.56 vs 149.36 s) and `COLI_NO_OMP_TUNE` alone is neutral (+0.8%). Observed, not fully explained — likely the Mac Studio's desktop power/thermal headroom leaves the GPU unthrottled by the spin. `NO_OMP` remains harmless insurance and is kept in the winner config.

**`PIPE` is the only lever that matters (A → C).** +6.9% (1.31 → 1.40), trimming matmul 188 → 164 s and attention 150 → 124 s by keeping experts streaming. E equals C within noise — once `PIPE` is present, `NO_OMP` adds nothing.

**`PIPE_WORKERS=8` is the sweet spot.** 12 → 1.41 (noise), 16 → 1.37. Extra workers cannot help once the SSD is saturated; they only add contention.

## Rebase check (v1.2.0 → v1.4.0): performance-neutral

Same machine, same usage snapshot, cap-matched at `--ram 110` (33 = v1.2.0's auto-raised value; v1.4.0's fast-SSD default cap=1 measured 0.98 tok/s on the warmup and was overridden):

| config | v1.2.0 tok/s | v1.4.0 tok/s | Δ |
|---|---|---|---|
| warmup (128 tok) | 1.45 | 1.45 | 0% |
| A | 1.33 | 1.31 | −1.5% |
| C | 1.39 | 1.40 | +0.7% |
| D | 1.30 | 1.32 | +1.5% |
| E | 1.43 | 1.40 | −2.1% |
| E pipe12 | 1.38 | 1.41 | +2.2% |
| E pipe16 | 1.39 (970 tok) | 1.37 | −1.4% |

All configs land within the noise band; hit rate, disk wait and attention time within ~1–2%. **Watch item:** expert-matmul is +3–5% in the same direction in all five ram-110 configs (A 182→188, C 161→164, D 184→189, E 157→165 s) — each within noise, but the consistent sign deserves repeats before dismissing.

## Recommendations

1. **Operating point on a 128 GB M1 Ultra:** `--ram 120`–`125` · `--cap 33` · `COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8` · MTP off. No memory-pressure thrash anywhere in the sweep (GPU-sched canaries flat at 13–16 s).
2. **Beware the v1.4.0 fast-SSD default:** it detects this drive as fast and drops the expert cache to cap=1 — 0.98 vs 1.45 tok/s on the warmup. If a Metal+SSD box regresses against older builds, check `--cap` first.
3. **MTP is a strict loss at 128 GB.** Acceptance was real (63–74%, 1.63–1.74 tok/fw in accidental MTP-active runs) but the +12–15 GB RSS crosses the memory knee into swap thrash (0.07–1.04 tok/s vs 1.47 MTP-off). Do not enable MTP on this machine/RAM class.
4. **To go faster on this hardware class, spend on storage, not GPU.** The drive runs at ~93% of its measured ceiling during decode; a faster expert-read path (or a much larger pin budget) is the only lever with headroom.

## Caveats / untested levers

- **Decode only.** Prefill (~11 s) is excluded from tok/s.
- **Single run per config.** The E-vs-C and 118/120/125 rankings are within noise; `ram120` stopped at 934 tokens (early EOS) — its tok/s is a valid rate, its decode seconds are not comparable. Repeats for confidence intervals (and the matmul watch item above) are the open follow-up.
- **`ram105` is not pin-matched** to the rest of the sweep (pin auto-capped 46.9 → 41.9 GB, v1.4.0 LRU-preservation behavior; net-neutral: same 1.35 tok/s as pre-rebase).
- **fmt=2 quality caveat** — see Setup. The official g64 container was not benchmarked here (Metal fmt=4 path open: #585/#587).
- **Non-determinism:** as in the M5 Max report — parallel expert reductions occasionally flip an argmax at a token boundary; output quality and throughput are unaffected.
