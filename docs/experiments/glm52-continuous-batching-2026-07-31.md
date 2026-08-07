# GLM-5.2 continuous batching experiment

Date: 2026-07-31

## Question

Can Colibri's existing mux scheduler raise aggregate GLM-5.2 decode throughput
by evaluating one token from several independent KV slots in the same forward?

## Method

- Host: 6 x RTX 5090, dual Xeon Silver 4510, 251 GiB RAM.
- Model: `/data/models/GLM-5.2-colibri-int4`.
- Greedy decode: `DRAFT=0`, temperature 0.
- CUDA dense tensors, attention, routed experts, and async expert groups enabled.
- Frozen placement:
  `/data/test/colibri-simroute-20260726/prune/frozen-general.stats`.
- Full residency: 9,335 VRAM experts + 10,121 RAM experts, zero disk reads.
- Eight KV slots with `CTX=256`; this is sufficient for the 21-token prompts
  and 32-token completions without wasting the RAM needed by expert residency.
- One engine process, batch order `1,2,4,8,4,2,1`.
- Timing starts when the last request emits its first token. This excludes all
  serial prompt prefill and measures only the interval in which every submitted
  request is eligible for batched decode.

The harness is `c/tools/benchmark_continuous_batch.py`.

## Result

| Active sessions | Aggregate tok/s, runs | Median aggregate tok/s | Median per-session tok/s |
|---:|---:|---:|---:|
| 1 | 4.29, 5.39 | **4.84** | **4.84** |
| 2 | 7.19, 5.47 | **6.33** | **3.16** |
| 4 | 8.22, 8.07 | **8.14** | **2.04** |
| 8 | 8.30 | **8.30** | **1.04** |

The experiment does **not** show an aggregate throughput breakthrough.
Throughput rises relative to the mux single-slot path, but saturates at roughly
8-9 tok/s by four sessions. That is approximately the existing non-mux
single-stream baseline (8.90 tok/s on the same frozen placement), not a
multiple of it.

The 8-session median TBT was 0.852 s and p95 TBT was 1.306 s. Batching trades
away individual-session latency without increasing total work completed enough
to justify the trade at this point.

## Profile

A shorter `1,4,8` run with `PROF=1` reproduced the same shape:

| Active sessions | Aggregate tok/s | Forward p50 | CPU expert bandwidth |
|---:|---:|---:|---:|
| 1 | 3.74 | 267 ms | 26.57 GB/s |
| 4 | 7.35 | 522 ms | about 31-35 GB/s |
| 8 | 9.05 | 813 ms | about 33-35 GB/s |

Expert matmul consumed 69-72% of the measured time. At eight sessions the CPU
expert side was 5.6-11.1 times slower than the overlapping GPU critical path.
The ordinary 8.90 tok/s single-stream run reached 66.88 GB/s on CPU experts.

The current multi-row CPU expert path therefore loses roughly half of the
effective RAM bandwidth. It groups rows that select the same expert, but its
multi-row quantized matmul does not turn that grouping into enough weight reuse
or parallel memory bandwidth. Attention was only 15-17% and is not the primary
bottleneck.

## Memory trap found during the experiment

With eight default `CTX=4096` slots and automatic live-history placement, only
9,335 VRAM + 5,321 RAM experts were pinned. The first pass then measured
96.6-99.8% hits and aggregate medians of only 3.64/3.51/5.48/6.00 tok/s for
1/2/4/8 sessions. Those numbers mix batching with disk/LRU warm-up and are
invalid as a compute comparison.

Serving configuration must reserve KV memory before claiming full expert
residency. `KV_SLOTS`, `CTX`, and expert placement are one capacity decision,
not independent tuning knobs.

## Verdict

Continuous batching is already functionally implemented, but **the present
CPU+GPU execution path does not increase aggregate GLM-5.2 decode throughput
beyond the existing single-stream ceiling**. It must not be advertised as a
throughput multiplier yet.

The next bounded experiment is a dedicated multi-row INT4 CPU expert kernel:
decode each expert's weights once into vector registers/cache and accumulate
all rows before advancing through the weight stream. It is worth integrating
only if it restores at least the single-row 60+ GB/s effective bandwidth and
raises the full-resident 4/8-session aggregate above 8.9 tok/s.

## Artifacts

- `/data/test/colibri-contbatch-20260731-IucO7Z/continuous-batch-fullresident-abba.json`
- `/data/test/colibri-contbatch-20260731-IucO7Z/continuous-batch-fullresident-abba.stderr`
- `/data/test/colibri-contbatch-20260731-IucO7Z/continuous-batch-profile.json`
- `/data/test/colibri-contbatch-20260731-IucO7Z/continuous-batch-profile.stderr`
- Invalid memory-pressure control:
  `/data/test/colibri-contbatch-20260731-IucO7Z/continuous-batch-abba.json`
