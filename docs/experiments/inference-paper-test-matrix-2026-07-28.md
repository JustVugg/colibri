# Inference-paper claim test matrix

Date: 2026-07-28

This is an experimental ledger, not a literature summary. A paper claim is
counted as tested only when the relevant implementation is compared against a
controlled baseline on the same hardware, model, prompt/replay, token count,
and precision policy. Paper-reported numbers are context, never local proof.

## Scope and verdicts

The initial corpus covers the inference mechanisms raised by Kimi K3, Kimi
Linear, Mooncake, HybriMoE, SparseSpec, PagedAttention, Splitwise, and
HCAttention. References are listed at the end.

Verdicts:

- **confirmed**: reproduced locally on the intended real workload;
- **conditional**: reproduced, but only under a measured prerequisite;
- **rejected**: controlled local A/B contradicts the useful form of the claim;
- **pending-runtime**: applicable, but the runtime mechanism is missing;
- **pending-model**: requires weights trained for a different architecture;
- **not-applicable**: the claim targets a materially different serving regime.

## Measurement contract

- Primary hardware: 6x RTX 5090 (32 GiB each), dual Xeon Silver 4510, 251 GiB
  host RAM.
- Primary model: GLM-5.2 int4, unless a claim requires another architecture.
- Decode comparisons: greedy, `COLI_TEMP=0 DRAFT=0`, fixed prompt and output
  length, warm-up first, ABBA or rotated order, at least three measured runs.
- Speculation comparisons explicitly vary `DRAFT`; their control remains
  `DRAFT=0`.
- Report median tok/s and the execution breakdown, not the best run.
- Correctness gate: identical replay tokens where the mechanism claims
  losslessness; otherwise an explicit quality suite and error bound.
- A throughput claim fails if it merely moves time into startup, prefill,
  transfer, or an unreported CPU path.
- Serving claims require concurrent request traces and TTFT/TBT/goodput; a
  batch-1 decode run cannot validate them.

## Claim matrix

| ID | Paper mechanism and testable claim | Colibri experiment | Status / current evidence |
|---|---|---|---|
| M1 | HybriMoE: CPU and GPU intra-layer work can overlap instead of serializing | Compare synchronous expert groups with `COLI_GROUP_ASYNC=1`; record CPU time, GPU critical path, overlap window, and tok/s | **confirmed, conditional** on current `dev`: +7.1% without NUMA but only +2.0% after NUMA makes the GPU tier the straggler |
| M2 | HybriMoE: impact-aware placement beats frequency-only placement | Use measured CPU/GPU cost to choose the prefix that minimizes `max(T_cpu, T_gpu)`; compare with filling VRAM from the same hot-count ranking | **rejected on this host**: fixed-token replay median fell 7.46 -> 4.01 tok/s; compute-only balancing reduced RAM/LRU residency and exposed about 4.3-4.5 s of disk wait |
| M3 | HybriMoE: inter-layer prefetch hides expert movement | Sweep prefetch distance and queue depth; compare service time versus visible wait time | **rejected for the available policies**: `PREFETCH=1` lowered replay throughput 2.5%; a cross-domain coupling table predicted almost exclusively resident experts, and bounded non-resident/real-load variants did not reduce misses or felt wait |
| M4 | Kimi K3 LatentMoE: narrowing the routed branch reduces expert weight and communication traffic while allowing larger top-k | Train or obtain matched full-width and latent-routed checkpoints; compare quality, bytes/token, expert compute, and collective traffic | **pending-model**; cannot be inferred from GLM-5.2 |
| M5 | Kimi K3 sparse MoE benefits from a low-row warp-specialized decode kernel | Compare block kernel and WarpDecode with identical weights, routing, and output | **rejected on current dev/RTX 5090**: a bit-exact isolated 8-expert GLM-sized group measured block 0.248 ms versus warp 0.326 ms; noisy end-to-end wins were CPU-bandwidth drift |
| M6 | Kimi K3 deployment-aware FP4 expert quantization reduces serving memory without unacceptable quality loss | Compare the same checkpoint before/after QAT MXFP4; measure quality and real kernel throughput | **pending-model**; post-training quantization is not equivalent to the paper's QAT claim |
| A1 | Kimi Linear/K3: 3:1 KDA-MLA hybrid reduces long-context KV traffic while retaining periodic global attention | Matched trained hybrid/full-attention checkpoints; sweep context length and report KV bytes, TTFT, TBT, and quality | **pending-model** |
| A2 | Kimi Linear: KDA's specialized chunkwise/DPLR algorithm is required for hardware-efficient prefill | Kernel A/B against a mathematically equivalent unfused implementation on a KDA checkpoint | **pending-model** |
| A3 | Kimi K3: unified paging of fixed KDA state and growing MLA KV simplifies allocation and transfer without losing reuse | Implement typed pages in one allocator; stress allocation, eviction, transfer, and restore across mixed state types | **pending-runtime**, after A1 support exists |
| A4 | Kimi K3: decoupling physical-page, prefix-hash, and recurrent-checkpoint granularities improves prefix reuse | Trace replay over several block/hash/checkpoint sizes; measure hit rate, retained bytes, restore cost, and TTFT | **partly supported, model-limited**: physical pages are independently tunable and 64 tokens is the measured allocation/fragmentation Pareto point; prefix adoption remains exact token-granular; GLM-5.2 has no KDA recurrent checkpoint to test |
| K1 | PagedAttention: paged KV removes fragmentation and enables flexible sharing | Compare contiguous KV and paged KV under changing sequence lengths and slots; measure waste, maximum concurrency, and latency | **partly confirmed by prototype**: ragged KV now uses real 64-token physical pages with no grow-copy; cross-page correctness is exact and a length-skew trace reserves 76.4% less than fixed-slot allocation, but page alias/refcount sharing and an end-to-end concurrency curve remain absent |
| K2 | Prefix reuse substantially reduces repeated-prefix TTFT | Compare cold second request against cross-slot prefix adoption on an identical long prefix | **confirmed on current `dev`**: three-round median second-request TTFT fell 17.689 -> 0.569 s (31.1x); each reused 257/264 prompt tokens and prefilling only 7 |
| K3 | HCAttention: key quantization, value offload, and eviction can retain quality at 25% (and remain competitive at 12.5%) KV | Implement each component separately, then combined; LongBench-style quality plus latency/traffic at 100/25/12.5% | **pending-runtime**; requires an accuracy experiment, not only a memory benchmark |
| S1 | MTP/speculative decoding accelerates decode when accepted tokens amortize target work | Sweep draft depth and prompt on full residency; report acceptance, tok/forward, draft cost, verify cost, expert union, and tok/s | **confirmed, strongly conditional**: DRAFT=1 was +9.2% on predictable text but -17.7% on explanatory text; a 24-proposal/70% guard cuts the measured negative case to -7.8% without disabling a 100%-acceptance case |
| S2 | The S1 result reverses when grouped GPU verification makes multi-position expert work sublinear | Compare S=1/2/4 expert time with grouped kernels and near-full GPU residency, then repeat draft sweep | **partly confirmed**: current grouped/async/full-resident runtime makes DRAFT=1 and DRAFT=3 positive, but deeper verification still grows expert work and is prompt-sensitive |
| S3 | Kimi K3/EAGLE-3: a target-feature draft trained for acceptance improves lossless speculation | Compare current MTP head with an EAGLE-3-style feature-fusion draft on the same target | **pending-model** |
| S4 | SparseSpec: delayed verification can overlap CPU draft work with GPU target work | Timeline CPU draft, target verify, and KV mutations; compare serial versus delayed schedule | **pending-runtime**, dependent on a useful S2/S3 draft |
| C1 | Losslessly compressed resident experts increase effective VRAM capacity enough to offset decode cost | Raw expert tier versus ANS tier on identical placement budget and output | **confirmed**: capacity +13.9%; 32-token decode 6.19 -> 7.12 tok/s; 128-token 6.75 -> 7.31 tok/s; controlled output byte-identical |
| C2 | Pipelined/aligned direct sidecar loading hides or reduces compressed-weight placement cost | Buffered, pinned-staging, and aligned direct-read sidecar placement | **confirmed**: about 197 s buffered, 157 s pinned, 80-103 s direct; full 110 GB archive read 18.94-19.15 s |
| C3 | Compression decode can be fused with expert matvec to eliminate materialization traffic | Compare current decode-to-scratch against row-aligned fused decode/matvec on fixed-token replay | **rejected after integrated ABBA**: predictions remained identical, but same-process materialize 6.505 versus fused 5.285 tok/s (-18.8%); the earlier cross-process +7.1% was machine drift |
| P1 | More aggregate GPUs do not help when expert placement remains per-rank and the execution is serial | Compare TP4 with TP2xPP3 under equal model/quality conditions | **confirmed**: TP4 median 2.6 tok/s; all-six TP2xPP3 1.78 tok/s |
| P2 | Residual broadcast + parallel per-device expert compute beats serial layer ownership on PCIe-only GPUs | Compare existing concurrent per-device groups with a forced global serialization control; verify overlap with Nsight | **parallel dispatch confirmed; GPU-resident residual rejected for this architecture**: multi-device execution raises fixed-replay median 6.82 -> 7.44 tok/s (+9.1%), while the remaining host transfer is not the critical path |
| P3 | NUMA-aware placement raises effective CPU expert bandwidth on a dual-socket host | Interleaved on/off, same binary and prompts, ABBA order | **confirmed** on current `dev`: 5.67 -> 7.89 tok/s without async and 6.07 -> 8.05 with async in the first controlled replicate |
| D1 | Splitwise/Mooncake: separating prefill and decode improves goodput under mixed concurrent traffic | Replay a concurrent long/short request trace against mixed scheduling; measure TTFT/TBT SLO goodput and KV transfer | **interference confirmed; split benefit pending**: inserting a 501-token prefill raised an active decode's maximum TBT from 0.188 s to 44.205 s, but Colibri has no disaggregated execution path for the treatment |
| D2 | Mooncake: layer-wise KV transfer hides most prefill/decode transfer latency | Serialized whole-KV transfer versus per-layer streaming on the same interconnect | **supported only by calibrated model**: for the measured 111.5 MB KV, ideal layer streaming hides 98.7% of link time; no transfer runtime exists |
| D3 | Mooncake: prefix-affinity scheduling and hot-prefix replication improve effective throughput | Trace-driven multi-instance comparison against round-robin/LRU | **mechanism confirmed, scheduler pending**: K2 shows a prefix hit cuts TTFT 17.689 -> 0.569 s, but no multi-instance affinity/replication scheduler was exercised |
| D4 | Splitwise is useful only when phase specialization outweighs KV transfer | Sweep prompt/output distributions and interconnect bandwidth; locate the break-even boundary | **calibrated boundary only**: under ideal layer streaming the measured 501-token request exposes 11.29 ms at 1 Gb/s and 0.11 ms at 100 Gb/s; RPC, queueing, and remote allocation remain unmeasured |

## Immediate experiment batches

### Batch 1: rerun mechanisms that already exist

1. CPU/GPU group overlap (M1), with synchronous control.
2. Selective NUMA (P3), crossed with M1 to expose interaction.
3. Raw versus ANS residency (C1), retaining token/routing identity.
4. Block versus WarpDecode (M5), on the same ANS archive.
5. MTP S=1/2/4 cost curve (S1/S2 gate), with expert-union counters.

This is a factorial experiment where practical, not five unrelated "best"
runs. The important interactions are NUMA x overlap and ANS x WarpDecode.

### Batch 2: small runtime additions

1. Cost-weighted expert placement (M2).
2. Fused ANS decode/matvec prototype (C3).
3. Parallel multi-device expert dispatch/gather (P2).
4. End-to-end paged/prefix KV experiments (K1/K2).

### Batch 3: model-dependent work

KDA/MLA, LatentMoE, MXFP4 QAT, EAGLE-3, and HCAttention quality claims require
appropriate checkpoints or training. Microbenchmarks may validate kernels,
but cannot validate model-quality claims.

## Results added on 2026-07-28

### M5: ANS-resident block kernel versus WarpDecode

Hardware and workload:

- 6x RTX 5090, 24 physical CPU threads spread across cores;
- `COLI_NUMA=1`;
- 10,628 VRAM experts through the same 2,500-raw + ANS sidecar;
- remaining 8,828 experts resident in RAM;
- 100% expert hit rate;
- identical 20-token prompt, 64-token greedy decode, routing counts, and output;
- async CPU/GPU expert-group path enabled in both runs.

| mode | decode | expert matmul | CPU expert rows | effective CPU BW |
|---|---:|---:|---:|---:|
| block control | 7.65 tok/s | 3.836 s | 9,207 | 55.76 GB/s |
| WarpDecode | 7.76 tok/s | 3.637 s | 9,207 | 60.48 GB/s |
| change | **+1.4%** | **-5.2%** | identical | +8.5% |

This confirms the direction but not the earlier +9.2% magnitude. The
end-to-end gain is small in this routing because CPU rows remain on the
critical path and attention is already about 35% of decode. More rotated
repeats and prompts are required before replacing the earlier estimate.

An earlier pair in the same session is deliberately invalidated: without
fixed physical-thread binding and explicit NUMA placement, the control
measured 4.89 tok/s and the WarpDecode run 2.16 tok/s, while CPU expert
bandwidth independently collapsed from 26.96 to 10.16 GB/s and prefill also
regressed. That pair measures a NUMA/thread-placement confound, not the GPU
kernel. It is retained here because silently selecting the favorable run
would violate the measurement contract.

The original async profiler reported `routed GPU critical 0.000s` during
decode despite thousands of GPU-served calls. The instrumentation fix used in
the next experiment records successful issue-to-take windows without changing
execution. It is a critical window (and therefore includes useful CPU overlap),
not a sum of isolated CUDA kernel durations.

#### Current-dev revalidation

The two WarpDecode commits were replayed cleanly onto current `dev`, without
the ANS materialization/fusion experiment. A same-process balanced sequence
(`A-B-B-A-B-A-A-B`, where A is the existing block kernel) did not reproduce a
stable end-to-end gain: block phases ranged from 6.80 to 8.61 tok/s and warp
from 6.76 to 8.55 tok/s while CPU expert bandwidth independently moved between
55 and 78 GB/s. The balanced medians actually favored block, 8.41 versus
7.56 tok/s.

An isolated CUDA benchmark then removed the CPU tier. It used eight GLM-sized
W4 experts (`D=7168`, `I=2048`, one row each), alternated call order for 40
pairs, and checked every output bit. Five fresh processes gave:

| process | block | WarpDecode | block / warp |
|---:|---:|---:|---:|
| 1 | 0.239 ms | 0.312 ms | 0.766x |
| 2 | 0.248 ms | 0.340 ms | 0.728x |
| 3 | 0.267 ms | 0.316 ms | 0.846x |
| 4 | 0.252 ms | 0.326 ms | 0.773x |
| 5 | 0.241 ms | 0.344 ms | 0.701x |
| median | **0.248 ms** | **0.326 ms** | **0.761x** |

`maxerr=0` in every process, but the warp mapping increased median kernel-call
latency by 31.5%. The apparent end-to-end gains were therefore not caused by
this kernel. WarpDecode remains experimental and is not merged into current
`dev`.

Artifacts:

- `/data/test/paper-m5-warp-current-balanced8.log`
- `/data/test/paper-m5-warp-micro.log`

### CPU thread-count control

Because CPU expert bandwidth repeatedly confounded GPU-kernel comparisons, the
full-resident fixed replay swept 12/16/20/24/32/48 OpenMP threads in one
process, using a mirrored order so every thread count had equal average phase
position. The first sweep suggested 20 threads (7.89 tok/s paired mean) over
24 (7.25), but a focused balanced-eight revalidation reversed the result:

| threads | median decode | median expert matmul |
|---:|---:|---:|
| 20 | 8.09 tok/s | 3.485 s |
| 24 | **8.35 tok/s** | **3.370 s** |

Twenty threads was 3.1% slower; 48-thread SMT was also clearly poor in the
broad sweep (6.19 tok/s paired mean). The existing 24-physical-thread setting
is retained. The broad-sweep win was machine-state drift, not a tuning result.

Artifacts:

- `/data/test/paper-cpu-omp-sweep.log`
- `/data/test/paper-cpu-omp20-vs24-balanced8.log`

The other existing CPU fork, disabling the fused FP32 gate/up pair so S=1
experts use separate AVX-512 VNNI IDOT calls, also failed a balanced-eight
same-process gate:

| CPU expert path | median decode | median expert matmul | CPU rows |
|---|---:|---:|---:|
| fused pair | 7.25 tok/s | **4.035 s** | 11,210 |
| separate IDOT | 7.36 tok/s | 4.242 s | 11,321 |

The 1.5% decode difference is below the observed run noise, expert time is
5.1% worse, and the changed accumulation path alters downstream routing. It is
therefore neither a useful performance result nor a lossless substitution.
The fused-pair default remains.

Artifact:

- `/data/test/paper-cpu-pair-vs-idot-balanced8.log`

### M1/P3: CPU-GPU overlap crossed with NUMA placement

This experiment used current `upstream/dev` plus a profiling-only fix that
records successful async issue-to-take windows. It used the raw resident tier
(9,335 VRAM + 10,121 RAM experts), the same fixed placement, 24 physical
threads, 64 greedy tokens, identical routing/output, and 100% expert hits.

| NUMA interleave | async groups | decode | expert matmul | CPU expert | GPU critical window |
|---|---|---:|---:|---:|---:|
| off | off | 5.67 tok/s | 6.276 s | 5.557 s / 34.86 GB/s | 0.450 s |
| off | on | 6.07 tok/s | 5.538 s | 4.888 s / 39.63 GB/s | 5.362 s |
| on | off | 7.89 tok/s | 3.432 s | 2.889 s / 67.06 GB/s | 0.444 s |
| on | on | 8.05 tok/s | 3.305 s | 2.839 s / 68.23 GB/s | 3.290 s |

Effects in this first controlled replicate:

- async overlap with NUMA off: **+7.1%** throughput, **-11.8%** expert time;
- NUMA with async off: **+39.2%** throughput, **-45.3%** expert time;
- async overlap after NUMA: **+2.0%** throughput, **-3.7%** expert time;
- NUMA with async on: **+32.6%** throughput, **-40.3%** expert time.

The mechanisms interact rather than add. When CPU expert bandwidth is poor,
overlap hides a useful fraction of serialized GPU work. Once interleaving
raises CPU bandwidth to about 67-68 GB/s, the async GPU window is slightly
longer than the CPU row window (3.290 versus 2.839 seconds), so the GPU tier
becomes the straggler and overlap has only a 2% end-to-end remainder.

Verdicts:

- M1 (HybriMoE-style intra-layer CPU/GPU overlap): **confirmed, conditional**;
  useful, but its gain depends on the CPU/GPU tier balance.
- P3 (NUMA-aware expert placement): **confirmed** and substantially larger
  than overlap on this dual-socket host.
- The next placement experiment must minimize `max(T_cpu, T_gpu)`, not assign
  experts by hit count alone. This result supplies the measured cost signal
  needed by M2.

Raw logs:

- `/data/test/paper-m1-numa0-async0-r1.log`
- `/data/test/paper-m1-numa0-async1-r1.log`
- `/data/test/paper-m1-numa1-async0-r1.log`
- `/data/test/paper-m1-numa1-async1-r1.log`

### M2: impact-aware capacity versus frequency-only fill

GLM-5.2 uses equal-sized experts and the same CPU/GPU kernels for every
expert, so HybriMoE's per-expert impact ordering reduces to frequency ordering
on this model. The locally testable distinction is therefore capacity policy:
fill all available VRAM with that ordering, or stop at the prefix that
minimizes the measured `max(T_cpu, T_gpu)`.

`tools/placement_balance.py` aggregated the existing 30-prompt Expert Atlas
trace (1,421,400 selections across 18,943 observed experts). Using the measured
M1/P3 costs, its static model predicted:

| policy | GPU experts | routed GPU share | predicted critical path |
|---|---:|---:|---:|
| frequency fill | 9,335 | 81.16% | 137.708 s |
| impact balance | 6,848 | 69.90% | 118.615 s |

That is a predicted **1.161x** speedup. The first real A/B used an Atlas Python
prompt, 24 physical threads, NUMA interleave, async groups, greedy 64-token
decode, and the same aggregate ranking:

| policy | decode | CPU rows/time | GPU critical | disk wait | hit rate |
|---|---:|---:|---:|---:|---:|
| frequency fill (9,335) | 7.36 tok/s | 8,955 / 3.278 s | 3.806 s | 0.000 s | 100.0% |
| impact balance (6,846 actual) | 5.54 tok/s | 14,588 / 4.229 s | 4.656 s | 2.568 s | 99.5% |

The measured result is **-24.7%**, opposite to the analytic prediction. The
model assumed constant per-selection cost and ignored capacity effects:
moving 2,489 experts out of VRAM increased the RAM resident set, reduced the
per-layer LRU cap from 10 to 1, and exposed 189 disk loads. It also
underpredicted the prompt-specific CPU routing share. On this constrained
host, compute-only impact balancing is not sufficient; residency, cache
capacity, and miss cost must be part of the objective.

An additional Chinese-category prompt showed the same direction: 7.72 versus
5.26 tok/s (**-31.9%**), with 2.058 seconds of visible disk wait in the
balanced configuration. Its generated token stream diverged between tiers,
however, so it is supporting systems evidence rather than a controlled replay
replicate. The final M2 verdict requires fixed-token `REPLAY` inputs.

A fixed-token replay then removed that caveat. A 64-token continuation captured
from GLM-5.2 was split into an 8-token prompt and 56 identical decode inputs.
The order was fill, balance, balance, fill, balance, fill:

| policy | runs | median | median change |
|---|---|---:|---:|
| frequency fill | 6.16, 7.53, 7.46 tok/s | **7.46 tok/s** | control |
| impact balance | 3.87, 4.06, 4.01 tok/s | **4.01 tok/s** | **-46.2%** |

Every replay traversed exactly 56 fixed tokens. The balance runs consistently
made 490-491 disk loads, fetched about 9.28 GB, and waited 4.29-4.46 seconds;
fill made no decode-time loads and retained a 100% hit rate. Thus M2 is
**rejected on this host in its compute-only form**. A capacity-aware objective
could still recover the paper's direction on a host with enough RAM, but that
is a different policy and remains to be tested rather than assumed.

The follow-up capacity-aware solver adds the missing hard residency bound:
`min_gpu_slots = total_experts - max_ram_slots`. For this host that is
`19,456 - 10,121 = 9,335` GPU experts. Applying the bound to the same Atlas
trace and measured tier costs clamps the compute-only 6,848-expert proposal
back to 9,335, exactly the frequency-fill placement. Predicted speedup is
therefore 1.000x: the useful result is preventing the measured 46.2%
regression and 490-491 disk loads, not inventing a speedup.
`tools/placement_balance.py` now accepts `--total-experts` and
`--max-ram-slots`, rejects infeasible GPU+RAM capacities, and has unit coverage
for the capacity floor.

A final capacity-preserving check raised the explicit RAM budget to 230 GB,
then compared the normal 176.6 GB GPU expert tier with a 171 GB tier. Both
placements kept all 19,456 experts resident and fetched 0.000 GB during
decode, so this isolates tier balance rather than cache misses:

| GPU budget | GPU/RAM experts | decode runs | median | CPU rows | median CPU / GPU critical |
|---|---:|---|---:|---:|---:|
| 176.6 GB | 9,335 / 10,121 | 6.55, 7.64, 7.26 | **7.26 tok/s** | 9,560 | 3.240 / 3.716 s |
| 171 GB | 9,040 / 10,416 | 7.44, 6.68, 6.88 | **6.88 tok/s** | 10,085 | 3.625 / 4.120 s |

The smaller tier moves 525 routed rows from GPU to CPU, but does not reduce
the measured critical path. Its median decode is 5.2% lower (mean 2.1% lower),
while run-to-run variance is large enough to reverse individual pairs. This
rejects capacity-only tier shrinking: expert count is not a sufficient load
signal, even when residency is held constant. A future placement policy must
optimize from per-expert routed work and observed tier time; the default keeps
the full 176.6 GB GPU tier.

The next experiment measured the most optimistic possible upper bound:
`ROUTE_TRACE` captured the exact fixed replay, and its 8,989 observed experts
were ranked ahead of the cold tail for a fresh run of that same replay. This is
deliberately an in-sample oracle, not a deployable policy:

| placement | decode | CPU rows | GPU rows | CPU / GPU critical |
|---|---:|---:|---:|---:|
| Atlas baseline during trace capture | 8.74 tok/s | 9,560 | 24,040 | 2.670 / 3.067 s |
| replay oracle, run 1 | 13.38 tok/s | 457 | 33,143 | 0.161 / 0.774 s |
| replay oracle, run 2 | 12.64 tok/s | 457 | 33,143 | 0.180 / 0.786 s |

The repeated oracle median is 13.01 tok/s, 48.9% above its trace-capture
baseline. This falsifies the linear per-selection cost model: concentrating
reused experts can improve CUDA grouping enough that 98.6% of routed rows on
GPU is faster, not slower.

It does **not** validate the policy. On a separate fixed 11-to-18-token replay,
the same placement failed to transfer:

| held-out placement | decode | CPU rows | GPU rows | CPU / GPU critical |
|---|---:|---:|---:|---:|
| Atlas baseline | **8.93 tok/s** | 1,110 | 3,090 | 0.328 / 0.380 s |
| replay oracle | 5.54 tok/s | 1,404 | 2,796 | 0.699 / 0.760 s |

That is a 38.0% held-out regression with zero disk I/O in both runs. The large
in-sample gain is real but workload-specific; it is exactly the kind of oracle
result that becomes misleading when a paper does not separate placement
training from evaluation. The production default remains the broader Atlas
ranking. Any learned placement follow-up must use category-held-out traces and
report both transfer gain and worst-case regression.

`tools/placement_crossval.py` then evaluated the original 10-category,
30-prompt Atlas with each category held out in turn. A score based on
category-normalized mean frequency minus `lambda * population_stddev` selected
`lambda=0.25` by worst held-out coverage:

| ranking | held-out mean GPU coverage | worst category |
|---|---:|---:|
| raw pooled Atlas | 73.29% | 68.04% |
| category mean (`lambda=0`) | 73.23% | 68.51% |
| robust (`lambda=0.25`) | **73.70%** | **69.32%** |
| robust (`lambda=0.5`) | 73.02% | 67.14% |
| robust (`lambda=1`) | 68.56% | 63.69% |

The modest offline improvement did not survive real execution. Adjacent,
reverse-order A/B runs held the GPU capacity, RAM budget, replay tokens,
resident set, and zero-I/O condition constant:

| replay | pooled Atlas | robust `lambda=0.25` | change |
|---|---:|---:|---:|
| 56-token main | **7.48 tok/s** | 6.63 tok/s | **-11.4%** |
| independent 7-token decode | **8.84 tok/s** | 6.53 tok/s | **-26.1%** |

On the short replay, robust ranking even increased GPU-routed rows from 3,090
to 3,144 while GPU critical time worsened from 0.377 to 0.517 seconds. Routing
coverage is therefore not an adequate optimization target: expert identity,
per-device packing, grouping shape, and NUMA locality change the cost of the
same number of rows. The robust-frequency candidate is rejected and the pooled
Atlas placement remains the default.

To expose that missing cost, CUDA group telemetry was split by device without
changing execution. The pooled Atlas baseline showed 4,037-4,751 routed rows
per GPU (17.7% max/min spread) and 2,472-2,830 group calls (14.5% spread).
An opt-in LPT prototype kept the same 9,335-expert GPU prefix but assigned its
frequency-ranked experts to the device with the lowest accumulated Atlas
weight instead of the fewest placed bytes.

The assignment was deterministic across three fresh processes. It reduced the
observed row spread only to 4,332-4,982 (15.0%) because Atlas aggregate
frequency does not exactly predict this replay; device 2 became the new
straggler. End-to-end paired results were nevertheless positive:

| run | byte-balanced control | frequency-load assignment | paired change |
|---|---:|---:|---:|
| 1 | 7.83 tok/s | 7.86 tok/s | +0.4% |
| 2 | 7.64 tok/s | 8.27 tok/s | +8.2% |
| 3 (reverse order) | 6.57 tok/s | 7.49 tok/s | +14.0% |
| median | **7.64 tok/s** | **7.86 tok/s** | **+2.9%** |

All runs retained 19,456 resident experts, 100% hits, and zero decode I/O.
The direction repeated, but CPU-frequency variance is too large and only one
56-token workload has been tested. `CUDA_EXPERT_LOAD_BALANCE=1` is therefore
retained as an experimental opt-in, not enabled by default. Per-device group
telemetry is retained unconditionally because it is measurement-only and
identifies a real placement objective missed by aggregate counters.

Raw logs:

- `/data/test/paper-m2-fill-code.log`
- `/data/test/paper-m2-balance-code.log`
- `/data/test/paper-m2-fill-chinese.log`
- `/data/test/paper-m2-balance-chinese.log`
- `/data/test/paper-m2-replay-fill-r1.log`
- `/data/test/paper-m2-replay-fill-r2.log`
- `/data/test/paper-m2-replay-fill-r3.log`
- `/data/test/paper-m2-replay-balance-r1.log`
- `/data/test/paper-m2-replay-balance-r2.log`
- `/data/test/paper-m2-replay-balance-r3.log`
- `/data/test/paper-tier-171-r1.log`
- `/data/test/paper-tier-171-r2.log`
- `/data/test/paper-tier-171-r3.log`
- `/data/test/paper-tier-1766-r1.log`
- `/data/test/paper-tier-1766-r2.log`
- `/data/test/paper-tier-1766-r3.log`
- `/data/test/paper-m2-oracle-train.log`
- `/data/test/paper-m2-oracle-hot-r1.log`
- `/data/test/paper-m2-oracle-hot-r2.log`
- `/data/test/paper-m2-heldout-atlas-r1.log`
- `/data/test/paper-m2-heldout-oracle-r1.log`
- `/data/test/paper-m2-robust-main-r1.log`
- `/data/test/paper-m2-atlas-main-adj-r1.log`
- `/data/test/paper-m2-atlas-heldout-adj-r1.log`
- `/data/test/paper-m2-robust-heldout-r1.log`
- `/data/test/paper-m2-device-groups-atlas-r1.log`
- `/data/test/paper-m2-device-groups-atlas-r2.log`
- `/data/test/paper-m2-device-groups-atlas-r3.log`
- `/data/test/paper-m2-device-load-balance-r1.log`
- `/data/test/paper-m2-device-load-balance-r2.log`
- `/data/test/paper-m2-device-load-balance-r3.log`

### M3: existing cross-layer prefetch

The balance placement from M2 supplies a real I/O-bound control: each 56-token
fixed replay fetches about 9.3 GB and feels more than four seconds of disk
stall. The treatment enables Colibri's existing previous-token cross-layer
hint (`PREFETCH=1`) with every other setting unchanged.

| mode | tok/s runs | median tok/s | median disk service | median felt wait |
|---|---|---:|---:|---:|
| no prefetch | 3.87, 4.06, 4.01 | **4.01** | 4.902 s | 4.342 s |
| `PREFETCH=1` | 3.91, 3.94, 3.84 | **3.91** | 4.858 s | 4.302 s |

The hint reduced both service and visible wait by less than 1%, while median
throughput fell 2.5%. The profiler moved time from `expert-matmul` into
unclassified orchestration/overlap, but the end-to-end clock did not improve.
Verdict: **rejected for this heuristic and workload**. This does not reject
HybriMoE's learned or deeper lookahead policy; Colibri does not implement that
policy, so substituting the current one-token route hint is only a test of the
mechanism already available here.

Treatment logs:

- `/data/test/paper-m3-prefetch1-r1.log`
- `/data/test/paper-m3-prefetch1-r2.log`
- `/data/test/paper-m3-prefetch1-r3.log`

#### Learned cross-layer coupling follow-up

The stronger existing `COUPLE` mechanism was tested without leaking the fixed
replay into training. A coupling table with 31,152 conditioning entries was
built from 267 routed positions across ten Atlas prompt categories, then
evaluated on the independent 56-token replay.

The original K1/K4 implementation enqueued zero hints: its highest-scored
candidates were already in the 17,789-expert resident tier, and it stopped
instead of searching for a non-resident candidate. An experimental bounded
search through the top 16 scores found only six non-resident candidates over
the whole replay. Neither advisory nor real-load variants helped:

| coupling mode | tok/s | hints | misses | fetched | felt wait |
|---|---:|---:|---:|---:|---:|
| no prefetch median | 4.01 | - | 500 | 9.458 GB | 4.342 s |
| K1 advisory, non-resident search | 4.14 | 6 | 510 | 9.647 GB | 4.448 s |
| K4 real load, non-resident search | 4.13 | 6 | 510 | 9.666 GB | 4.458 s |

The apparent throughput increase is not attributed to prefetch: miss count and
visible I/O both worsened, while the changing term was expert-matmul time. The
non-resident search and real-load prototypes were therefore removed. This
rejects the present small-table policy on this placement; useful learning must
target cold-miss probability directly rather than overall next-layer routing,
whose most predictable experts are already resident.

Artifacts:

- `/data/test/paper-m3-couple-train.trace`
- `/data/test/paper-m3-couple-train.coli_pairs`
- `/data/test/paper-m3-couple-k1-nonresident-r1.log`
- `/data/test/paper-m3-couple-real-k4-r1.log`

### C3: fused ANS decode and expert matvec

Two implementations were tested rather than treating a compiling CUDA kernel
as evidence:

1. A block-partial prototype decoded each ANS block independently and reduced
   cross-block row fragments afterward. It passed an illegal-access smoke test
   but failed the numerical gate: fixed replay predictions diverged at step 13,
   and throughput fell from 7.67 to 6.33 tok/s.
2. A row-aligned tile prototype decodes eight complete matrix rows into shared
   memory and performs the matvec before the tile is discarded. Its 256-thread
   reduction matches the materialized kernel's arithmetic order closely enough
   to retain all 56 fixed-replay argmax predictions.

The aligned prototype was also connected to the asynchronous issue/take path.
The controlled setup used 2,500 raw plus 8,128 ANS experts (10,628 total in
VRAM), 8,828 RAM experts, 100% hit rate, the same 56 replay tokens, NUMA
interleave, and 24 physical threads.

| mode | tok/s runs | median tok/s | expert-time runs | median expert time |
|---|---|---:|---|---:|
| materialize control | 7.48, 6.71, 6.88 | **6.88** | 3.492, 4.241, 4.120 s | **4.120 s** |
| row-aligned fused | 7.67, 7.09, 7.37 | **7.37** | 3.396, 3.921, 3.653 s | **3.653 s** |
| change | | **+7.1%** | | **-11.3%** |

Verdict: **confirmed by prototype**. Fusion removes global raw-weight
materialization for eligible one-row ANS groups and improves the strongest
async path without changing fixed-replay predictions. The effect is useful but
not enormous because only compressed, single-row, homogeneous groups take the
new path; raw and mixed groups retain the existing kernels.

The experiment also exposed a placement hygiene requirement: layer 78 MTP
experts are int8 and must not be fed into the fmt2-only ANS sidecar. The
controlled ranking therefore excludes that layer rather than silently
misaligning sidecar records.

Prototype worktree:

- `/home/Kei/colibri-wt-ans-tile` (`exp/ans-tile-pipeline`)

#### Integrated revalidation

The prototype result above did not survive a stricter integration audit.
After replaying the ANS base commits and fused kernel onto the current `dev`,
separate-process runs varied enough to reverse direction. A same-process
A-B-B-A replay then loaded the model and all resident experts once and changed
only `COLI_ANS_ALIGNED_MV`:

| mode | tok/s | median | expert time | median |
|---|---|---:|---|---:|
| materialize | 6.58, 6.43 | **6.505** | 4.897, 4.946 s | **4.922 s** |
| row-aligned fused | 5.57, 5.00 | **5.285** | 5.885, 6.988 s | **6.437 s** |

All four phases produced identical argmax predictions. Materialize recovered
to 6.43 tok/s after both fused phases, ruling out monotonic thermal throttling
as the explanation. The integrated fused path reduced throughput by 18.8%
and increased expert time by 30.8%. The earlier cross-process +7.1% result was
a machine-state false positive. The fused kernel is therefore **rejected** and
was not merged into `dev`.

Raw logs:

- `/data/test/paper-c3-async-control-r1.log`
- `/data/test/paper-c3-async-control-r2.log`
- `/data/test/paper-c3-async-control-r3.log`
- `/data/test/paper-c3-async-fused-r1.log`
- `/data/test/paper-c3-async-fused-r2.log`
- `/data/test/paper-c3-async-fused-r3.log`
- `/data/test/paper-c3-integrated-abba.log`

### P2: parallel per-device expert execution

Code inspection showed that this mechanism was already partially present, but
under a misleading control name. Even with `COLI_GROUP_ASYNC=0`, OpenMP worker
threads can drive independent device contexts concurrently. The async option
primarily adds CPU/GPU overlap; it is not the sole multi-GPU parallelism switch.

Nsight Systems traces over the complete fixed replay established the timeline:

| mode | expert-active time with >=2 GPUs | weighted active GPUs | peak GPUs |
|---|---:|---:|---:|
| existing sync group path | 62.9% | 2.34 | 6 |
| async issue/take path | 30.7% | 1.37 | 6 |
| forced global serialization | **0.0%** | **1.00** | **1** |

The lower overlap fraction in the async row is not a regression: its GPU
kernels overlap the CPU tier and finish in a different cadence. The important
fact is that both production paths exhibit real cross-device overlap, while
the mutex control removes it completely.

A default-off `COLI_CUDA_SERIAL_GROUPS=1` experiment control was temporarily
added around the synchronous CUDA group call. Three fixed-token runs per mode
gave:

| mode | tok/s runs | median | expert-time median |
|---|---|---:|---:|
| forced single-device-at-a-time | 6.82, 5.72, 7.32 | **6.82** | 3.906 s |
| existing parallel dispatch | 7.44, 8.02, 7.23 | **7.44** | 3.710 s |
| change | | **+9.1%** | **-5.0%** |

Verdict: the **parallel per-device compute portion is confirmed** and retained.
The experiment-only global serialization switch was removed after the control
runs.

The broader residual-broadcast formulation is rejected for the current
architecture. Colibri's dense, attention, residual, and CPU-expert paths are
host-resident; each PCIe device receives only the routed activation rows and
returns their expert outputs. The fixed replay routed about 24,030 GPU rows.
At `D=7168` and FP32 input plus output, that is about 1.38 GB across the entire
56-token replay. Even a conservative PCIe 4.0 x16 lower bound puts the pure
wire time near 45 ms, whereas observed CPU-expert work was about 6 s and the
parallel GPU critical path about 0.4 s. Moving residual ownership to one GPU
would also force the CPU tier to add the opposite D2H/H2D boundary.

This is therefore not an unimplemented free win: it is an architecture rewrite
aimed at a non-dominant cost. Revisit only if dense/attention and most experts
become GPU-resident, or profiling shows host transfer on the critical path.

Trace artifacts:

- `/data/test/paper-p2-async.nsys-rep`
- `/data/test/paper-p2-serial.nsys-rep`
- `/data/test/paper-p2-forced-serial.nsys-rep`

Raw logs:

- `/data/test/paper-p2-serial.log`
- `/data/test/paper-p2-forced-serial.log`
- `/data/test/paper-p2-parallel-r2.log`
- `/data/test/paper-p2-parallel-r3.log`
- `/data/test/paper-p2-forced-serial-r2.log`
- `/data/test/paper-p2-forced-serial-r3.log`

### K1: physical-page ragged KV prototype

The existing ragged allocator rounded each sequence to 64 tokens, but every
growth still allocated a larger contiguous buffer and copied the old KV on the
device. The prototype replaces that buffer with an actual page table:

- latent and RoPE KV use separate 64-token physical pages;
- crossing a boundary allocates only the new page, without copying old pages;
- attention and append kernels translate `(sequence, token)` through the page
  table;
- pages remain independently owned by each sequence.

The CUDA unit test now grows three sequences from `[1, 63, 64]` to
`[1, 65, 70]`, exercising both sides of the 64-token boundary. Five fresh
process runs on an RTX 5090 all reported `ragged_relative_rms=0` against the
contiguous attention reference. The measured second-call page growth was
0.051-0.057 ms.

For a skewed nine-slot trace with lengths
`[1, 17, 63, 64, 65, 127, 128, 511, 1024]` and a 1024-token slot ceiling:

| allocation | reserved token positions | unused positions |
|---|---:|---:|
| fixed contiguous slots | 9,216 | 7,216 |
| 64-token physical pages | 2,176 | 176 |

That is 76.4% less reserved capacity than fixed-size slots, with 8.1% internal
fragmentation inside the allocated pages. This validates physical paging,
bounded tail waste, and copy-free growth. It does **not** yet validate the full
PagedAttention sharing claim: the runtime still copies a matched prefix into
the destination slot (K2) rather than aliasing shared pages with reference
counts, and no maximum-concurrency A/B has been run.

Runtime artifact:

- `/data/test/test_ragged_attention-paged`

The physical page size was then decoupled from the test and swept at compile
time. Every cell retained exact output (`ragged_relative_rms=0`). Median results
over five fresh processes were:

| physical page | cross-page growth | grow to 1024 | steady 1024 attention | unused positions in the nine-slot trace |
|---:|---:|---:|---:|---:|
| 16 | 0.051 ms | 1.922 ms | 0.034 ms | 48 |
| 32 | 0.051 ms | 1.505 ms | 0.034 ms | 80 |
| 64 | 0.053 ms | 1.311 ms | 0.034 ms | 176 |
| 128 | 0.049 ms | 1.180 ms | 0.033 ms | 432 |

Steady attention is insensitive to page-table length at this scale. Sixteen
tokens minimizes tail waste but makes 1024-token growth 47% slower than 64;
128 saves only 0.131 ms of growth while adding 256 unused token positions in
the skewed trace. The 64-token default is retained as the Pareto point, while
`COLI_KV_PAGE_TOKENS` can now be overridden at compile time independently of
prefix matching. Prefix adoption remains exact token-granular, so no page-size
change can silently reduce its hit length.

Sweep artifacts:

- `/data/test/kv-page-sweep/page-16-v4.log`
- `/data/test/kv-page-sweep/page-32-v4.log`
- `/data/test/kv-page-sweep/page-64-v4.log`
- `/data/test/kv-page-sweep/page-128-v4.log`

### K2: current cross-slot prefix reuse

The previously separate `feat/paged-ragged-kv` and
`feat/kv-cross-slot-adopt` work is already in `dev`; the stale matrix entry was
wrong. `c/tools/benchmark_kv_prefix.py` drives the mux protocol with two slots,
`KVSAVE=0`, one generated token, a common long prefix, and a distinct short
tail. The control disables cross-slot adoption; the treatment enables it.

| mode | second-request TTFT runs | median |
|---|---|---:|
| no cross-slot reuse | 17.803, 17.154, 17.689 s | 17.689 s |
| reuse enabled | 0.569, 0.544, 0.569 s | 0.569 s |

Every treatment round adopted a prefix ending at token 257 out of a 264-token
prompt and prefilled only 7 tokens. Median TTFT improved by 96.8%, or 31.1x.
The first request remained 17-20 seconds in both modes, which rules out a
general warm-run speedup as the explanation.

Artifacts:

- `/data/test/paper-k2-share0-r3.jsonl`
- `/data/test/paper-k2-share0-r3.stderr`
- `/data/test/paper-k2-share1-r3.jsonl`
- `/data/test/paper-k2-share1-r3.stderr`

### K3: cross-slot physical page alias

A Linux `memfd`/`mmap` prototype kept the existing contiguous KV virtual
addresses while aliasing complete 64-token blocks across slots. Partial blocks
remained independent copies, and logical owners/reference counts protected
slot lifetime. Three validation rounds compared every adopted L/R/index KV byte
with its source and all three were exact.

The real 6x5090 A/B rejected the design:

| mode | second-request TTFT runs | median |
|---|---|---:|
| current row `memcpy` | 3.156, 2.428, 3.458 s | **3.156 s** |
| page alias, per-block mappings | 4.069, 3.640, 3.371 s | 3.640 s |
| page alias, coalesced mappings | 4.627, 4.051, 3.584 s | 4.051 s |

The coalesced alias remained 28.3% slower than `memcpy`. Avoided copy traffic
did not repay mapping calls and first-touch/page-table work across 79 cache
levels and three tensor families. The runtime prototype was removed; the
existing exact `memcpy` adoption remains the accepted implementation.

Artifacts:

- `/data/test/paper-k2-memcpy-current-r3.jsonl`
- `/data/test/paper-k2-memcpy-current-r3.stderr`
- `/data/test/paper-k2-alias-r3.jsonl`
- `/data/test/paper-k2-alias-r3.stderr`
- `/data/test/paper-k2-alias-coalesced-r3.jsonl`
- `/data/test/paper-k2-alias-coalesced-r3.stderr`

### D1-D4: serving interference and disaggregation boundaries

#### D1: mixed prefill/decode interference

`c/tools/benchmark_serving_interference.py` first records an isolated 32-token
decode, then starts another 32-token decode and submits a 501-token prefill
immediately after its first streamed token. Both use the same two-slot mux and
the same full-resident placement.

| case | median TBT | maximum TBT | request wall time |
|---|---:|---:|---:|
| isolated decode | 0.117 s | 0.188 s | 7.384 s |
| decode plus inserted prefill | 0.126 s | **44.205 s** | 49.113 s |

The interfering request's TTFT was 44.199 s. Colibri deliberately serializes
prefill, so the active decode makes no progress for essentially the complete
prefill. This directly confirms the interference that phase splitting targets.
It does not confirm Splitwise goodput: this runtime cannot transfer a live KV
cache to a separate decode instance, so there is no valid treatment system to
compare.

A scheduler prototype then moved prefill out of synchronous submission and
ran one token chunk after each priority decode batch. All variants were
rejected on the same 6x5090 workload:

| prefill mode | median TBT | maximum TBT | prefill TTFT | decode wall |
|---|---:|---:|---:|---:|
| serial baseline | 0.126 s | 44.205 s | 44.199 s | 49.113 s |
| scheduled, 32 tokens | 4.992 s | 20.381 s | 204.210 s | 214.247 s |
| scheduled, 128 tokens | 0.176 s | 30.769 s | 93.314 s | 103.459 s |
| scheduled, 256 tokens | 0.174 s | 57.690 s | 105.837 s | 115.194 s |

The 32-token setting cut maximum TBT by 53.9% but destroyed both median TBT
and prefill throughput. At 128 tokens maximum TBT improved 30.4%, while
prefill TTFT and decode wall both more than doubled. At 256 tokens the chunk
itself exceeded the original stall. Re-entering all 78 MoE layers for small
token batches loses too much batching efficiency; token-level chunking has no
useful operating point here. The scheduler prototype was removed. A viable
implementation needs finer pipeline/layer preemption or separate prefill and
decode workers.

Artifacts:

- `/data/test/paper-d1-interference-r2.json`
- `/data/test/paper-d1-interference-r2.stderr`
- `/data/test/paper-d1-scheduled-c32.json`
- `/data/test/paper-d1-scheduled-c32.stderr`
- `/data/test/paper-d1-scheduled-c128.json`
- `/data/test/paper-d1-scheduled-c128.stderr`
- `/data/test/paper-d1-scheduled-c256.json`
- `/data/test/paper-d1-scheduled-c256.stderr`

#### D2/D4: calibrated transfer sweep

GLM-5.2 stores 512 latent, 64 RoPE, and 128 index floats for each of 79 cache
levels. The measured 501-token request therefore carries 111,454,464 bytes of
KV. `c/tools/disaggregation_break_even.py` feeds that size and the measured
44.199-second prefill into a deterministic producer/link pipeline.

| link | whole-cache transfer | visible with ideal per-layer streaming |
|---:|---:|---:|
| 1 Gb/s | 891.64 ms | 11.29 ms |
| 10 Gb/s | 89.16 ms | 1.13 ms |
| 25 Gb/s | 35.67 ms | 0.45 ms |
| 100 Gb/s | 8.92 ms | 0.11 ms |
| 200 Gb/s | 4.46 ms | 0.06 ms |

Uniform layer production makes all but the final layer transfer overlap the
very slow prefill, hiding 98.7% of pure link time in this trace. Consequently,
the idealized D4 break-even is small: phase specialization needs to recover
more than the visible value in the last column. This is an optimistic lower
bound, not an end-to-end result; RPC, queueing, serialization, remote
allocation, backpressure, and nonuniform layer times are excluded.

Artifact:

- `/data/test/paper-d2-d4-break-even.json`

#### D3: prefix affinity

K2 supplies the local mechanism value for a scheduler: routing a request to a
slot holding its 257-token prefix saves 17.120 seconds of TTFT. A prefix-affinity
scheduler therefore has a large signal on this workload. We did not launch
multiple Colibri instances or test replication/eviction cost, so the
multi-instance scheduling policy itself remains unvalidated.

### S1/S2: MTP depth sweep after grouped/async execution

The control is the NUMA-on, async-on cell above. All draft runs used the same
model, int8 MTP head, placement, prompt, 64-token target, and greedy sampling.

| draft depth | decode | change vs D0 | tok/forward | acceptance | expert matmul | expert rows |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.05 tok/s | control | 1.02 | - | 3.305 s | 10,240 |
| 1 | 8.79 tok/s | **+9.2%** | 1.94 | 91% | 3.938 s | 11,868 |
| 2 | 7.53 tok/s | **-6.5%** | 2.46 | 73% | 4.932 s | 13,322 |
| 3 | 8.51 tok/s | **+5.7%** | 3.37 | 77% | 4.336 s | 13,506 |

The old blanket verdict ("MTP always loses on this MoE") is no longer true.
Grouped GPU execution, full residency, async CPU/GPU overlap, and improved
kernel consistency have crossed the break-even point for shallow speculation.
The mechanism is still conditional:

- D1 almost halves target forwards (63 -> 33) and cuts attention 2.856 ->
  1.869 seconds, while adding only 1,628 expert rows.
- D2 adds enough distinct-expert work to erase its forward-count advantage.
- D3 reduces forwards to 19 and wins despite 3,266 extra expert rows, but is
  slower than D1.
- Acceptance alone is insufficient: D2 accepts 73% yet loses, while D3 at 77%
  wins. The cost of the routed-expert union is the missing independent variable.

First-prompt verdict: `DRAFT=1` is the best tested point. It must still pass
rotated repeats and diverse prompts before becoming a default.

Raw logs:

- `/data/test/paper-s1-draft1-r1.log`
- `/data/test/paper-s1-draft2-r1.log`
- `/data/test/paper-s1-draft3-r1.log`

The diverse-prompt gate changed the policy verdict. On a non-degenerate
water-cycle explanation:

| mode | decode | tok/forward | acceptance | expert matmul | expert rows |
|---|---:|---:|---:|---:|---:|
| D0 | 8.51 tok/s | 1.02 | - | 2.946 s | 8,300 |
| D1 | 7.00 tok/s | 1.64 | 62% | 5.127 s | 11,146 |

Here D1 is **17.7% slower**. The explanatory prompt activates a wider routed
expert union: loads/token rise from 590.6 to 739.1, and the saved attention
work cannot pay for 2,846 extra CPU expert rows. Therefore D1 is not a safe
global default. A runtime policy needs a cheap predictor (recent acceptance
plus observed union-growth/cost), or speculation must remain opt-in.

Raw logs:

- `/data/test/paper-s1-water-draft0-r1.log`
- `/data/test/paper-s1-water-draft1-r1.log`

#### Runtime guard

The existing soft guard waited for 24 proposals but paused MTP only below 10%
acceptance. That threshold did not protect the measured 62%-acceptance
regression. On the same async execution path, changing the threshold to 70%
gave:

| water-cycle mode | decode | change vs old D1 | MTP proposals accepted | expert matmul | CPU expert rows |
|---|---:|---:|---:|---:|---:|
| old D1 guard (10%) | 7.00 tok/s | control | 24/39 (62%) | 5.127 s | 11,146 |
| 24-window, 70% guard | 7.85 tok/s | **+12.1%** | 15/24 (62%) before pause | 3.838 s | 9,851 |
| D0 | 8.51 tok/s | +21.6% vs old D1 | - | 2.946 s | 8,300 |

The guard recovers more than half of the regression, reducing the D0 gap from
17.7% to 7.8%; it does not make the negative case faster than disabling MTP.
A 12-proposal window was rejected because early acceptance was too noisy and
throughput fell to 6.97 tok/s. On the predictable-text counterexample, the
24-window observed 32/32 accepted proposals and did not pause, so the high-value
path is unchanged. Its absolute throughput was not compared across runs because
machine-state drift was material.

The production default is therefore 70% over 24 proposals, with
`COLI_MTP_GUARD_PCT` and `COLI_MTP_GUARD_WINDOW` retained as explicit experiment
overrides. This is a loss limiter, not an oracle: a future policy still needs
routed-union cost rather than acceptance alone.

Guard artifacts:

- `/data/test/paper-s1-water-guard70w24-async-r1.log`
- `/data/test/paper-s1-count-guard70w24-async-r1.log`

## References

- Kimi K3: Open Frontier Intelligence, technical report, 2026.
- Kimi Linear: An Expressive, Efficient Attention Architecture, 2025.
- Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving, 2024.
- HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE
  Inference, 2025.
- SparseSpec: Accelerating Large-Scale Reasoning Model Inference with Sparse
  Self-Speculative Decoding, 2025.
- Efficient Memory Management for Large Language Model Serving with
  PagedAttention, 2023.
- Splitwise: Efficient Generative LLM Inference Using Phase Splitting, 2023.
- HCAttention: Extreme KV Cache Compression via Heterogeneous Attention
  Computing for LLMs, 2025.
