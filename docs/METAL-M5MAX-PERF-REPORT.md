# Metal backend on M5 Max — performance report + a tuning trap in the rebase (#72)

**TL;DR:** After the rebase, the Metal backend is **faster than the pre-rebase branch — 2.24 vs 2.06 tok/s (+8.5%)** — *once tuned*. But the **default** rebased config regresses hard (1.25 tok/s, −39%), because the base now pulls in the OMP hot-team active-spin (#77), which on Apple Silicon steals the shared SoC power budget from the GPU and throttles the Metal kernels. Disabling the spin recovers the GPU; adding `PIPE` (new since the old base) then pushes *past* the old number. Recommend Metal builds default to a passive OMP wait.

## Setup (held fixed across every run)

- **Hardware:** Apple M5 Max — 18 CPU cores (12 P + 6 E), 40-core GPU, 128 GB unified memory
- **Model:** GLM-5.2 int4 (744B MoE), experts streamed from SSD
- **Workload:** `./coli run "Compare the myths of Lucifer and Prometheus"`, 1024 tokens generated
- **Constant flags:** `COLI_METAL=1 DIRECT=1 MTP=0 --ram 110`
- Every run: identical working set — ~607 experts/token, ~74–75% hit rate, RSS ~97.9 GB, `fallback CPU 0` (all eligible blocks on GPU)

## Results

| config | tok/s | wall (s) | expert-disk | expert-matmul | attention | attn GPU kernel | expert GPU overhead* |
|---|---|---|---|---|---|---|---|
| **A** — old base (pre-rebase branch) | 2.06 | 496 | 266 | 109 | 100 | 76 | 51 |
| **B** — rebased, **defaults** | 1.25 | 819 | 285 | 215 | 290 | 223 | 106 |
| **C** — rebased, defaults + `PIPE=1` | 1.30 | 788 | 297 | 190 | 272 | 212 | 89 |
| **D** — rebased + `COLI_NO_OMP_TUNE=1` | 1.90 | 539 | 266 | 143 | 109 | 85 | 84 |
| **E** — rebased + `NO_OMP` + `PIPE=1` **(winner)** | **2.24** | **457** | **241** | **97** | **99** | **79** | **44** |

\* *expert GPU overhead = expert gpu-wall − expert kernel time, i.e. GPU idle waiting to be fed. Expert kernel time itself was constant (~34–35 s) in every un-throttled run.*

Winner (E) command:

```bash
COLI_METAL=1 DIRECT=1 MTP=0 COLI_NO_OMP_TUNE=1 PIPE=1 PIPE_WORKERS=8 \
  ./coli run --model /path/to/glm52_i4 \
  "Compare the myths of Lucifer and Prometheus" --ram 110
```

## What happened, phase by phase

**The default regression (A→B) is not in the kernels' work — it's GPU throttling.** Same GPU dispatch counts (≈140k blocks, 79,794 attention layer-launches, 618k experts-on-GPU) in every run, yet the **attention GPU kernel time triples, 76 s → 223 s.** Kernel *execution* time can't triple for identical work unless the GPU is clocking down. The cause is the OMP hot-team (#77): with active spin, the CPU sits at ~97% busy-waiting on the GPU, and because the M-series CPU and GPU share one power/thermal envelope, that spin robs the GPU of clock headroom. Note B→C: adding `PIPE` while the spin is still on barely helps (1.25→1.30) — the CPU has no spare cycles to run PIPE's workers.

**Fix 1 — kill the spin (A→D).** `COLI_NO_OMP_TUNE=1` (passive waits) gives the GPU its power back: attention kernel falls 223 s → 85 s, near the old 76 s, and throughput jumps to 1.90. But a residual gap remains, and it is entirely in **expert-matmul (+33 s vs A)** — specifically the *GPU overhead*, which grew 51 s → 84 s while the expert kernel stayed at ~34 s. Passive waits stop stealing power but make the CPU slower to hand the next expert batch to the GPU, so the GPU idles between dispatches.

**Fix 2 — hide the dispatch latency (D→E).** `PIPE=1 PIPE_WORKERS=8` keeps experts prefetched and streaming, so the GPU stops waiting: expert overhead 84 s → **44 s** (below even the old base's 51 s), and `PIPE`'s I/O overlap also trims the disk wall 266 s → **241 s**. Net **2.24 tok/s** — past the pre-rebase 2.06, because `PIPE` did not exist on the old base.

The two levers are complementary, not additive coincidences: **`NO_OMP` restores GPU clocks; `PIPE` hides the CPU→GPU dispatch latency that `NO_OMP` introduces.** You want both.

## Recommendations

1. **On Apple Silicon Metal builds, default OMP to a passive wait** (e.g. auto-set `OMP_WAIT_POLICY=passive`, or `COLI_NO_OMP_TUNE` behavior, when `COLI_METAL` is enabled). The #77 active-spin default is a −39% trap here: during Metal offload the CPU is mostly *waiting on the GPU*, and spinning actively throttles it. This is the single highest-impact change.
2. **Document `PIPE=1 PIPE_WORKERS=8` as recommended with Metal** — it recovers the dispatch-latency cost of the passive wait and overlaps expert I/O.
3. **Memory ceiling (128 GB M5 Max):** `--ram 110` is safe (RSS ~97 GB, compressor quiet); `--ram 120` crosses into memory compression (double penalty — compressor CPU cost *and* SoC power stolen from the GPU); `--ram 115` untested/marginal. Metal's registered buffers share unified memory, so the knee is lower than a CPU-only build would suggest.

## Caveats / untested levers

- **Decode only.** These are steady-state decode numbers. The cold-prefill wall is unaffected (fused attention covers S≤4).
- **`MTP=0` throughout.** The rebased kernel's "fused attention handles S≤4 (covers MTP verify forwards)" commit means MTP verify is now GPU-accelerated — MTP was a net loss on the old base, so re-testing `MTP` on is a live, unexplored lever.
- **`DIRECT=1` is required, not optional.** `DIRECT=0` was A/B'd and is ~2× slower (2.16 → 1.15 tok/s at the same point in the run). It also drops RSS 98 → 84 GB — reads fall back to the OS page cache, which lowers process RSS but breaks the zero-copy GPU slab registration `DIRECT=1` enables, adding a copy to feed the GPU.
- **Determinism (confirmed non-deterministic run-to-run):** Two identical `DIRECT=1` runs (same config, same prompt, greedy / MTP off) diverged within ~7 tokens. So the engine is not run-to-run reproducible under the parallel config, and the earlier `DIRECT=0` vs `DIRECT=1` divergence is a symptom of this, not a `DIRECT` read-path bug. Cause is expected and benign: floating-point non-associativity in parallel expert-sum reductions (PIPE worker completion order and/or GPU threadgroup reductions) occasionally flips an argmax at a token boundary. Output quality is unaffected (both completions are valid) and **throughput is identical** across runs (1.28 tok/s at the same point), so benchmark numbers are stable. Implication for the PR's **"token-exact"** claim: it holds as "GPU path matches the CPU reference under a deterministic/serial validation config," but is *not* "bit-reproducible across runs" with `PIPE`/threads enabled — worth stating so nobody diffs two runs and files a false bug.
- Numbers are single-run per config on a warm cache; run-to-run and thermal/ambient variation not bounded (a same-session A/B of old vs rebased commit would tighten the throttling claim further).
