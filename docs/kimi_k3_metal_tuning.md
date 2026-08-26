# Kimi K3 — Metal tuning & performance (Apple Silicon)

Measured on an **M5 Max, 128 GB** (unified memory), model split across two
drives via `K3_DIRS`. Kimi K3 is the most disk-bound engine in the project, so
these results separate **compute** (where the GPU wins) from **decode
wall-clock** (which is governed by routed-expert streaming, not the backend).

## Recommended settings (M5 Max, 128 GB)

Metal:

```
K3_METAL=1 K3_PIPE=1 K3_DIRECT=1 K3_EXPERT_GB=44 K3_LOAD_THREADS=6 OMP_NUM_THREADS=18 \
  ./kimi_k3 <model_dir> "<prompt>" --ngen N
```

CPU (Metal off), for reference / correctness oracle:

```
K3_METAL=0 K3_PIPE=1 K3_DIRECT=1 K3_EXPERT_GB=48 K3_LOAD_THREADS=6 OMP_NUM_THREADS=18 \
  ./kimi_k3 <model_dir> "<prompt>" --ngen N
```

Build with `make -C c kimi_k3 METAL=1`.

## Speed improvement (Metal vs CPU, matched expert hit rate)

The clean, apples-to-apples win is on the compute-bound phases:

| Phase | CPU (`K3_METAL=0`) | Metal (`K3_METAL=1`) | Speedup |
|---|---|---|---|
| Prefill (13-token prompt) | 45.4 s | 27.2 s | **1.7×** |
| Attention (`time: attn`) | 34.5 s | 14.4 s | **2.4×** |

Decode wall-clock is I/O-bound (routed experts stream from disk at these cache
sizes), so it is set by the expert cache and storage, not by the compute
backend — Metal and CPU decode land close once the cache is matched. The
`time: attn / moe / eload` breakdown is printed per run so the split is
checkable.

## Tuning findings

### `OMP_NUM_THREADS` — 18 (all logical cores)

Swept 6→18. **18 gave the best decode throughput (0.28 tok/s).** The built-in
auto-sizer (`omp_tune.h`) caps the OpenMP team at `perflevel0` cores (6 on this
chip), which is correct for P+E layouts but wrong here: the M5 Max's second tier
runs ~0.70× per core — well above the `N_fast/(N_fast+N_slow)=0.33` break-even
for equal-split scheduling — so including all 18 helps rather than drags.
`OMP_NUM_THREADS` overrides the auto-sizer (it yields if the env var is set).

### `K3_EXPERT_GB` — ~40–44

Swept 8→96. The **cache-driven metrics are monotonic**: hit rate 4→53%, bytes
streamed and `eload` fall as the budget grows. But `eload` hits its floor
(~10 s) and hit-rate gains flatten around **44 GB** — beyond that you buy less than 1% hit
for more memory. On the CPU path the equivalent knee is ~48 GB; Metal sits a
touch lower because GPU-wired buffers trim the headroom.

**Memory-pressure ceiling:** at very large budgets (≈96 GB) the machine tips
into the macOS memory compressor and *everything* slows 3–5×, including
pure-compute phases (`attn`, `head`) that never touch the cache — the tell that
it's system pressure, not a cache effect. `attn` is the canary: it stays flat
until pressure begins. Keep total footprint (≈35 GB weights + cache + KV + OS
headroom) comfortably under ~100 GB on a 128 GB box.

### `K3_LOAD_THREADS` — 6

Storage-concurrency for the `K3_PIPE` prefetcher; 6 balanced well against an
18-thread compute team on the CPU path. On the Metal path (compute offloaded)
this can likely go higher — re-check if tuning specifically for GPU decode.

## Caveats on measurement

- Single short (16-token) runs are noisy; thermal state and background load can
  swing the compute timings several-fold across a back-to-back sweep. For
  trustworthy numbers, run each point 2–3× and take the **min**, add a cooldown,
  and use a longer `--ngen` (32–64) so warmup/overhead don't dominate.
- Judge cache size by the monotonic signals (`hit`, `streamed`, `eload`), not by
  a single decode time.

## What transfers CPU → GPU

- **Transfers:** the `K3_EXPERT_GB` hit-rate curve (routing/caching are CPU and
  identical), and `K3_LOAD_THREADS` (storage-bound).
- **Re-discover on GPU:** the `K3_EXPERT_GB` pressure ceiling (GPU wiring lowers
  it) and `OMP_NUM_THREADS` (most compute moves to the GPU, so the CPU team
  optimum shifts — generally lower).
