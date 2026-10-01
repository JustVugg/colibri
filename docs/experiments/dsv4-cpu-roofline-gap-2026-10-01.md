# Four-GPU CPU offload: the gap to the streaming roof

## Review scope and final result

This change builds on the CPU-offload study in #1816. The integrated implementation adds opt-in CPU-tail cache pooling/preload, wider exact FP8 decoding, SIMD activation QDQ, parallel sparse-attention heads and BF16 routing, grouped CPU output projection, regression tests, and decode-only fixture counters. The branch also carries the intervening CPU-tail packing and offloaded-head work documented in the companion reports.

| Evidence boundary | Decode tokens/s | Interpretation |
| --- | ---: | --- |
| Fixed-task starting baseline | 7.6318 | Includes the earlier CPU-tail optimization |
| Integrated CPU changes with pooling/preload enabled | 15.5838 | Single screening run; startup and locked-parent memory excluded |
| Selected isolated research configuration | 24.1432 / 24.2722 | Includes host-specific NUMA and further CPU/CUDA prototypes absent from the integrated source |
| Highest isolated combined screen | 24.3800 | Extra switches have no established incremental benefit |
| Fixed objective | 26.8 | Not achieved |

The 24-token/s results are **research evidence, not the performance of the integrated/default implementation in this PR**. All 492 ordinary successful requests match reference token IDs; this count spans multiple experimental variants and is not 492 repetitions of the final configuration. Six Nsight requests are validated separately. The numerically divergent attention-dot trial is excluded. The chronology below retains failed and inconclusive experiments so their results are not mistaken for accepted optimizations. No new full-model run was started during PR preparation.

## Scope and target

DeepSeek-V4-Flash-0731, concurrency one, target-only decode, four RTX 5090 GPUs holding layers 0–34, and eight CPU tail layers on two Xeon Silver 4510 processors. Precision, layer placement, prompts and output token IDs remain fixed. The optimization target is 26.8 decode tokens/s, approximately 50% of the estimated streaming roof. **The target has not been reached.**

The earlier estimate is a weight-streaming roof, not a prediction of application throughput:

```text
CPU: 1.895102144 GB / 140.8 GB/s = 13.4595 ms
GPU: 9.322199560 GB / 1792 GB/s = 5.2021 ms
Total: 18.6617 ms/token -> 53.5858 tokens/s
```

The CPU denominator uses four populated 64 GB DIMMs configured at 4400 MT/s. GPU layers execute serially for one request, so the calculation does not multiply a card's bandwidth by four. This simplified model omits quantization/dequantization, arithmetic, activation/KV traffic, scheduling, synchronization, and bandwidth loss from access patterns and NUMA placement. It is not a universal maximum: cache reuse or a different execution algorithm would change its assumptions.

## Revised offload timing model (2026-10-02)

Keep the ideal streaming roof separate from the calibrated execution model. For concurrency-one target-only decode, use actual layer placement rather than GPU count alone:

```text
T_decode(P, context, configuration)
    = sum(CPU-layer elapsed times)
    + sum(GPU-layer elapsed times)
    + remaining non-overlapping execution time
R_decode = 1000 / T_decode_ms
```

Here `P` identifies which layers execute on CPU and on each GPU. A per-layer elapsed time already contains that layer's computation, weight/KV access, quantization and internal waits; do not add those costs again. Charge transfers, scheduling and output-head time only where they are not already included in an enclosing timer. This is a serial single-request accounting model, not a formula for concurrent serving or MTP. Counts of GPUs do not multiply single-request bandwidth across sequential layers.

The fine diagnostic supplies the following non-overlapping calibration for the selected isolated four-GPU prototype, 24 physical CPU cores, ACTIVE waiting and the fixed short-context fixtures:

| Component | ms per decode token |
| --- | ---: |
| CPU attention, eight layers | 15.897127 |
| CPU MoE, eight layers | 11.042019 |
| CPU hyperconnection phases, eight layers | 1.207590 |
| GPU and all remaining execution time | 13.215793 |
| Total | 41.362529 |

Thus `1000 / (28.146736 + 13.215793) = 24.1765 tokens/s`, consistent with the separate uninstrumented 24.1432 and 24.2722 screens. This is a one-point calibration, not independent predictive validation. In particular, **13.215793 ms is a residual, not a directly measured GPU-only duration**, and must not be converted into a GPU bandwidth efficiency. The CPU tail averages 3.518342 ms/layer at this placement; this average is not a transferable cost for every model layer.

The ideal equation remains `13.4595 + 5.2021 = 18.6616 ms/token`, approximately 53.59 tokens/s (the more precise byte-based calculation above is authoritative). Actual CPU block time is about 2.091 times its ideal weight-streaming term, but that ratio includes computation and synchronization and is not a DRAM efficiency measurement. The empirical model explains the gap without redefining the ideal roof or treating implementation overhead as a hardware constant.

At this calibrated point, reaching 26.8 tokens/s requires reducing elapsed time to **37.313433 ms/token**, a **4.049096 ms/token** reduction. If the GPU/other residual stays fixed, CPU block time must fall from 28.146736 to 24.097640 ms, approximately **14.39% less CPU time**. This is a budget calculation, not a promise that the savings are achievable.

There is not yet a calibrated optimized 6-to-0 GPU curve. The earlier sweep used a different implementation/thread policy and reported prefill-inclusive output throughput; those rates cannot be fitted together with these decode-only points. For another placement, the CPU/GPU layer sets, cache fit, NUMA traffic, context-dependent attention and scheduling costs must be recalibrated. Do not extrapolate the tail's average linearly to pure CPU or use one constant efficiency multiplier for all GPU counts.

These coefficients describe the previously measured isolated prototype, not the integrated 15.5838-tokens/s implementation or the subsequently merged CI-fix head. This revision uses existing records only; no new runtime experiment was performed.

## Screening results

All rows below are one-round screening measurements, not confidence intervals or stable multi-run guarantees. Fixture 0 is excluded from each round. The baseline contains the earlier CPU-tail optimization; it is not the original unoptimized engine.

| Variant | Decode tokens/s | End-to-end output tokens/s |
| --- | ---: | ---: |
| Baseline, 24 physical cores | 7.6318 | 5.2457 |
| CPU-tail cache pooling and packing | 8.6956 | 6.5053 |
| Parallel sparse-attention heads only | 8.4957 | 5.7152 |
| Combined cache, attention and FP8 SIMD changes | 10.0697 | 7.3634 |
| Combined, NUMA interleave | 11.0330 | 7.9388 |
| Wider FP8 matrix kernels and F16C decoding | 11.1628 | 7.9826 |
| SIMD activation quantize/dequantize | 13.7825 | 10.6469 |
| Parallel BF16 router | 14.8817 | 11.6702 |
| Grouped attention output projection | 14.9550 | 11.8064 |
| Preload all CPU-tail experts | 15.5838 | 12.3697 |

Except for the two isolated cache/attention controls, variants are cumulative in the order shown. They also differ in NUMA policy where stated. Small differences between adjacent rows are not established speedups. An AVX2 batch-one specialization alone measured 7.4941 tokens/s and did not improve the baseline. Removing OpenMP affinity regressed the QDQ variant from 13.7825 to 9.8301 tokens/s; fixed physical-core affinity is retained.

A trial that grouped CPU experts into two matrix dispatches measured 15.1439 decode tokens/s with exact output IDs, below the 15.5838 preload control. It was not adopted. Its source and results are retained in the raw bundle.

A separate synchronous expert-lookup trial measured 15.8092 decode tokens/s and 12.4825 end-to-end tokens/s. Its approximately 1.4% difference from preload alone is not a demonstrated stable gain. This trial remains isolated in the raw bundle rather than becoming a default code path. Both additional trials preserve all baseline token IDs; 108 requests across the complete campaign pass the independent checker.

Two additional QDQ rounds in a fresh process measured 13.6945 and 15.4977 decode tokens/s. The faster later round demonstrates warm-cache sensitivity; it is not substituted for the first-round comparison.

Preloading packs all 2,048 experts in the eight CPU layers and adds 18.690 seconds to startup. Every measured decode request then has 3,024 expert-store hits and zero misses/bytes loaded through that store. This does not mean zero DRAM traffic: matrix computation still reads the cached weights. Peak engine RSS is about 43.44 GiB and excludes the campaign parent's separate 155.425 GiB locked checkpoint mapping.

## Measured time breakdown

A separate diagnostic build of the QDQ variant measured 72.4778 ms/token (13.7973 tokens/s). These timers predate the parallel router, grouped output projection and CPU-expert preload; they must not be presented as a profile of the latest variant.

| Non-overlapping phase | ms/token |
| --- | ---: |
| CPU attention, eight layers combined | 22.9679 |
| CPU MoE, eight layers combined | 31.1638 |
| CPU hyperconnection phases | 1.5341 |
| Remaining elapsed time | 16.8120 |

Attention includes 10.1619 ms in output projections, 4.5106 ms in query projection and 4.3035 ms in sparse-attention computation. MoE includes 13.3774 ms in routed expert computation, 3.6362 ms in the shared expert and 5.8127 ms waiting for the loader after shared computation. These are subdivisions, not extra costs to add to the table. The remaining elapsed time includes GPU work and other host/transfer/scheduling costs; it is not a direct measurement of GPU execution alone. Loader-worker elapsed counters can overlap and must not be added to request wall time.

The practical gap is therefore not explained by weights failing to reside in RAM. Computation and execution overhead are substantial, and current kernels do not attain the nominal memory-bandwidth roof.

## Follow-up kernel and placement investigation

A current preload profile measures 64.5056 ms/token (15.5025 decode tokens/s): CPU attention 25.9517 ms, CPU MoE 19.9823 ms, and hyperconnection phases 1.7602 ms. Routed experts account for 14.1928 ms and the shared expert for 4.3977 ms. Loader wait is now only 0.3550 ms. All three profile record types contain exactly 2,520 CPU decode calls across fixtures 1–5. The parser accounts for fixture 3 having one extra prompt token.

Two isolated AVX-512 trials preserve the original arithmetic order and pass the remote quantization/DeepSeek unit tests and complete-model token comparison:

| Trial | Decode tokens/s | End-to-end tokens/s |
| --- | ---: | ---: |
| FP4 paired low/high nibble decode | 15.6087 | 12.4919 |
| FP8 four independent row accumulators | 15.5253 | 12.3554 |

Neither shows a meaningful gain over preload alone, so neither is adopted into the working implementation. Their source snapshots and full results remain in the raw bundle.

A separate synthetic FP8 matrix benchmark rotates four matrices per shape, performs four warmup calls and times sixteen calls, under the same 24-core affinity/interleave policy. It is not an end-to-end inference benchmark. All three variants produce identical output hashes for all four shapes.

| Rows x columns | Current kernel ms | Four accumulators ms | Predecoded FP16 storage ms |
| --- | ---: | ---: | ---: |
| 4096 x 16384 | 0.7364 | 0.6761 | 1.3165 |
| 32768 x 1024 | 0.3160 | 0.2898 | 0.5353 |
| 16384 x 2048 | 0.3317 | 0.2942 | 2.4186 |
| 1024 x 4096 | 0.1363 | 0.1717 | 0.0721 |

The first three current-kernel measurements imply approximately 91–106 GB/s of logical weight bytes per elapsed second; these are not hardware-counter DRAM bandwidth readings and can include cache reuse. They do not support a claim that FP8 decoding alone explains the remaining application gap. The predecoded-storage trial doubles weight traffic and does not show a consistent benefit. These are single screening passes, with a conspicuous slow FP16 sample; no general performance guarantee is inferred.

## NUMA placement controls

All four controls below use the same `grouped-preload` executable, four GPUs, 24 physical CPU cores, the same interleave policy for the child, and the same complete expert preload. Each is one screening round. Raw `/proc/<pid>/numa_maps` snapshots are retained for both the parent and runner; reported runner placement is sampled at peak RSS.

| Control | Parent node 0 / node 1 GiB | Runner node 0 / node 1 GiB | Decode tokens/s |
| --- | ---: | ---: | ---: |
| Original locked checkpoint | 106.06 / 49.38 | 7.92 / 35.52 | 15.6708 |
| Unlock and unmap checkpoint before launch | Unmapped | 8.09 / 35.35 | 15.5760 |
| `mbind(MPOL_INTERLEAVE, MOVE_ALL)` | 106.06 / 49.38 | 7.67 / 35.77 | 15.6128 |
| Explicit alternating single-node checkpoint stripes | 78.70 / 76.74 | 26.96 / 16.48 | 16.2691 |

Unlocking the parent's mapping leaves clean file-cache pages resident and did not rebalance runner memory in this experiment. The interleave `mbind` call returned success but likewise did not redistribute already-resident pages between nodes that were both in the allowed mask. Neither result is interpreted as a successful placement change.

The final control first faults and locks each shard, then uses strict `MPOL_BIND` plus `MPOL_MF_MOVE_ALL | MPOL_MF_STRICT` on alternating 64 MiB ranges. That changes actual physical placement while keeping the complete checkpoint locked. Its 3.8% difference from the adjacent original locked control is a single-run result, not a confidence interval. It reaches about 30.4% of the streaming estimate, not the 50% target. The child is less skewed but not exactly balanced; total RSS alone does not identify the placement of individual CPU matrices.

All four controls have six exact token sequences, 3,024 expert-store hits and zero decode misses/bytes per request, and zero sampled swap. The unlocked control is an explicit exception to the locked-checkpoint protocol. Startup migration/load cost is excluded from decode rates. No checkpoint contents are changed. Physical file-cache placement can persist after a run, so future comparisons must record placement rather than assuming the earlier 106/49 GiB state still exists.

## Sparse-attention SIMD screening

A separate AVX-512 trial compacts valid KV indices, transposes their columns, and computes sixteen independent scores in parallel while retaining each score's ascending-column accumulation order. A seven-head, 63-dimensional, 35-index unit case compares it bit-for-bit against single-head scalar fallback, including non-binary-exact input factors, negative indices, invalid indices, and the all-invalid case. Remote native-quant and DeepSeek tests pass.

Full-model screening remains exact but does not establish an improvement:

| Physical cores / OpenMP threads | Decode tokens/s | End-to-end tokens/s |
| --- | ---: | ---: |
| 24, CPUs 0–23 | 16.1584 | 12.7826 |
| 16, CPUs 0–7 and 12–19 | 15.7299 | 12.1757 |
| 12, CPUs 0–5 and 12–17 | 14.8280 | 11.3629 |

All three retain four GPUs, 35 GPU layers, eight CPU layers and complete CPU-expert preload. They run after the checkpoint placement migration. The SIMD trial remains outside the working implementation; 24 threads remain the selected setting. Further investigation uses decode-only CPU cycle sampling, with the sampler disabled during loading, warmup, fixture 0 and prefill.

A fresh SMBIOS check records populated slots DIMM_000, DIMM_060, DIMM_100 and DIMM_160, with channel labels 0 and 6 on each socket, 64-bit data width and configured speed 4400 MT/s. No two-DIMMs-on-one-channel correction to the 140.8 GB/s estimate is indicated by these records. The 26.8 tokens/s optimization target is unchanged.

## Decode cycle sampling and scheduling controls

A separate decode-only `perf` run sampled fixtures 1–5, with sampling disabled during loading, warmup, fixture 0 and prefill. About 64K user-cycle samples were recorded with zero lost samples. The engine accounted for 49.84% of samples and libgomp for 47.91%. CPU-cycle fractions are not request wall-time fractions: waiting OpenMP workers can accrue cycles while GPU work progresses. The sampled run measured 17.9515 tokens/s, an unexplained difference from ordinary launches; it is not adopted as a speedup or as the accepted baseline.

Native-quant static cyclic scheduling measured 15.6123 tokens/s; dynamic scheduling measured 12.4980. Reducing `GOMP_SPINCOUNT` to 1000 measured 14.1839 using the original preload executable. All three retained exact output IDs and failed to improve the selected control. Their source variants remain isolated. A separate FP8 microbenchmark counted zero floating-point assists and zero SSE/AVX mixing assists; its run overlapped compilation, so its elapsed timing is excluded from performance conclusions.

## Exact FP4 BF16 dot-product prototype

An isolated, opt-in `V4_CPU_BF16_DOT=1` prototype decodes packed FP4 pairs directly into BF16 registers and uses AVX-512 BF16 dot instructions. It retains packed weight storage and the existing activation QDQ. The [Intel instruction reference](https://cdrdv2-public.intel.com/774492/325383-sdm-vol-2abcd.pdf) specifies high-half then low-half FP32 accumulation; the prototype reverses the pair layout to retain ascending-column accumulation order.

This is not an unrestricted replacement. It requires round-to-nearest-even with masked FP exceptions, exact BF16-representable QDQ activations with nonzero magnitude between 2^-32 and 2^32, UE8M0 scale codes 64–190, and at most 16,384 columns. Other inputs retain the original kernel. These bounds avoid the instruction's differing subnormal handling and keep products and accumulated sums within normal finite FP32 range.

A standalone 10,000-case differential test compared sixteen rows of 128 ordered terms, with mixed scale exponents and signed zeros, bit for bit against the original arithmetic. An additional 84-case public-API test covers single and dual matrices, all four rounding modes, scale-boundary fallback, and activation-magnitude fallback. Both passed on the AVX-512 BF16 host, as did existing native-quant and DeepSeek unit tests. The prototype is retained in the raw bundle pending performance acceptance. An initial model launch omitted the CUDA build flag and failed before generating any token; it is retained as a failed setup record and excluded from throughput results.

The corrected CUDA build measured **15.3699 decode tokens/s** and **12.1665 end-to-end tokens/s**, with six exact token sequences and zero sampled swap. This does not improve the selected 16.2691 single-round control, so the BF16 FP4 prototype is not adopted. Across the retained successful campaigns, the independent checker now validates 198 requests. This does not make all experimental variants accepted optimizations.

A subsequent read-only checkpoint metadata audit found the five attention projection scale tensors in CPU layer 35 (`wq_a`, `wq_b`, `wo_a`, `wkv`, `wo_b`) use `F8_E8M0`. This makes exact exponent-scaled FP8/BF16 decoding worth investigating next; it does not establish eligibility of every value or of all eight CPU layers. Hyperconnection scales are separate arbitrary F32 values and are not covered by that observation.

## Exact FP8 BF16 projection prototype

A separate `V4_CPU_FP8_BF16=1` prototype targets packed FP8 single-matrix, batch-one, and grouped projections. It combines adjacent columns into BF16 dot-product pairs while retaining ascending-column accumulation. FP8 bytes are decoded directly with integer exponent construction and a small subnormal lookup; weight storage stays FP8. Activation QDQ and model quantization are unchanged.

Eligibility requires the same conservative activation and FP-environment bounds as the FP4 trial, positive exact power-of-two F32 scale values with exponent codes 64–190, compatible rows8 packing and at most 16,384 columns per group. It limits total grouped inputs to 131,072 floats. Non-finite FP8 weight codes cause the computation to fall back to the original kernel. These guards are checked explicitly; the checkpoint's E8M0 metadata alone is not used as permission to change arithmetic for arbitrary input tensors.

The direct-kernel differential test requires the new path to report success for 100 randomized single/grouped cases of 256 rows by 1,024 terms. Results match the original arithmetic bit for bit, including wide scale ranges, finite positive/negative FP8 codes and signed zeros. It separately requires refusals for alternate rounding modes, non-BF16 activations, out-of-range activation magnitudes, non-power-of-two/extreme scales, and both NaN codes. Existing native-quant and DeepSeek tests also pass. The model screening uses the same executable with the switch disabled and enabled in adjacent fresh processes; the disabled control measured 16.3292 decode tokens/s and 12.8642 end-to-end tokens/s, while the enabled trial measured 16.2679 and 12.7528 respectively. All twelve token sequences match the baseline and sampled swap stays zero. The trial is not adopted.

## NUMA row-ownership prototype

A second isolated experiment keeps the original arithmetic kernels and attempts to place private CPU weight pages with the threads owning their output-row tiles. It explicitly requires 24 OpenMP threads bound to CPUs 0–23 in order, and fixes the dense CPU tail to layers 35–42 for this screening only. It computes the uneven `schedule(static)` split, rather than assuming exactly half the rows belong to each socket. Packed FP8/FP4 matrices use 16-row tasks; other dense matrices use individual rows. This host-specific prototype is not a portable/default implementation.

The first placement helper verified the left range before migrating the right range. A whole-buffer test found pages in the left range had subsequently moved to the wrong node. Merely setting `MADV_NOHUGEPAGE` after allocation, or replacing a single boundary page, did not reliably fix it. Existing large-page behavior is a plausible cause; these observations do not establish a general kernel bug.

The revised helper marks only the private malloc payload's interior pages `MADV_NOHUGEPAGE`, preserves their bytes in a temporary buffer, discards and restores those interior pages before binding, and verifies every interior page after both migrations. It leaves allocation-boundary pages untouched. Three fresh 4 MiB tests verified all 1,024 page locations and all original bytes. It therefore changes both page granularity and NUMA placement; either may affect throughput, and a speedup would need separate controls before attribution. No system-wide THP or NUMA setting changes. Checkpoint mappings are not passed to this helper. The first full-model screening measured **19.6650 decode tokens/s** and **15.1117 end-to-end tokens/s**, with all six token sequences exact and zero sampled swap. Startup verified 6,684,672 expert pages (27,380,416,512 bytes) and 305,672 dense pages (1,252,032,512 bytes). CPU expert preload including rebuilding/migration/verification took 38.597 seconds, versus roughly 18 seconds without these steps. Peak engine RSS remained about 43.44 GiB. A fresh-process control using the same executable with placement disabled measured 16.3593 decode tokens/s (13.0523 end-to-end). A second fresh-process placement run measured 19.6078 decode tokens/s (15.0524 end-to-end), again with exact tokens and zero sampled swap. The two placement runs are approximately 19.9–20.2% faster than this control. This is evidence of a repeatable screening gain for the fixed fixtures, not broad serving/long-context acceptance, and it does not separate page-granularity effects from NUMA locality. The host-specific prototype is retained as the next experimental baseline, not installed as a portable default. The 26.8 tokens/s target remains unmet. The complete independent checker now covers 228 successful requests.

## Profile after row-owned placement

A diagnostic run of the row-owned placement baseline measured 51.0380 ms/token (19.5933 decode tokens/s), with exact output IDs. The three profile kinds each contain exactly 2,520 decode records. CPU attention accounts for 18.7210 ms, CPU MoE for 14.0630 ms, and hyperconnection phases for 1.4810 ms; the remaining elapsed time is 16.7729 ms and includes GPU execution and other overhead. Within attention, query projection costs 3.6162 ms, output projections 7.5013 ms and the attention phase 4.0937 ms. Routed experts cost 9.9354 ms and the shared expert 2.8180 ms. Loader wait stays small at 0.3598 ms. These subdivisions must not be added to their parent phases.

## Page-granularity and placement control

An opt-in control rebuilds the same private weight interiors as 4 KiB pages but requests alternating placement across nodes 0 and 1. A first attempt relying on `MPOL_INTERLEAVE` policy failed full-model startup verification: one roughly 32 MiB dense buffer had 6,141 pages on node 0 and 2,050 on node 1. It generated no token and is retained under failed setup records, not throughput results.

The corrected control explicitly moves each page to its alternating target node and subsequently verifies every page. The 4 MiB unit test checks all 1,024 locations and unchanged bytes both for row-owned and alternating placement. Full-model screening measured **19.2945 decode tokens/s** and **14.8751 end-to-end tokens/s**, with six exact token sequences and zero sampled swap. This is close to the 19.6078–19.6650 row-owned results and well above the 16.3593 original same-binary control from the preceding experiment. The full gain cannot be attributed solely to row ownership; fine-grained, balanced physical placement also matters. These are separate executable screenings, not a repeated factorial attribution study. The row-owned variant remains the selected experimental baseline.

## Independent row accumulators with unchanged ownership

A new FP8 prototype keeps the existing sixteen-row OpenMP partition for every worker, then processes four of that worker's adjacent tiles with independent accumulators. It falls back to one tile at worker tails and grouped-input boundaries. Unlike changing the global OpenMP tile size to 64 rows, this preserves the NUMA ownership established at startup. Each accumulator retains its original column order and multiplication order, including the difference between single-matrix and batch/grouped paths.

Expanded tests cover 136, 144, 256, 272 and 4,112 output rows with non-power-of-two scales, comparing batch-one against unchanged batch-two and single matvec against unchanged dual matvec bit for bit. A seven-group, 384-rows-per-group case compares grouped results against unchanged batch-two and exercises group boundaries within uneven 24-thread partitions. Native-quant and DeepSeek tests pass with 24 threads. The full-model FP8 trial measured 19.7571 decode tokens/s and 15.0998 end-to-end tokens/s, with exact output IDs. Its less-than-one-percent difference from the selected placement baseline is not an established gain, so it remains isolated.

An independent FP4 version applies the same unchanged worker ownership to single and dual packed matrices. A 4,112-row, 256-column test compares both outputs against the unchanged row-major arithmetic, using different data and scales for the two matrices; it passes with 24 threads. Full-model screening measured 19.5740 decode tokens/s and 15.0883 end-to-end tokens/s, with exact output IDs. This version is not adopted either.

A separate sparse-attention microbenchmark compared the earlier transposed-KV SIMD trial with the original kernel, using 64 heads, dimension 512 and 64/128 valid keys. Original timings were 0.0767/0.1395 ms per call, versus 0.0904/0.1401 ms for the SIMD trial. All output bytes matched. This narrow warm-cache screening does not establish an application improvement, so that older trial is not promoted based on instruction count alone.

The independent result checker now also requires zero speculative drafted and accepted tokens in every target-mode record. All **252** completed successful requests pass this stronger check and exact token comparison. Inherited environment entries may mention MTP, but the fixture runner selects target-only generation with `no_dspark`; the measured speculative counters are zero. Failed setup attempts remain separate and are not counted.

A next FP4-scale prototype replaces the per-block AVX-512 gather with exact exponent-bit construction, retaining the existing table entries for codes 0 and 255 to preserve special-value and initialization semantics. Its full-model screening measured 19.7916 decode tokens/s and 15.1289 end-to-end tokens/s with exact outputs. The less-than-one-percent difference from the selected baseline does not establish a gain; the prototype remains isolated.

## GPU phase and routed-expert subphase diagnostic

A diagnostic build of the original row-owned baseline records 315 batch-one GPU calls for decode fixtures 1–5. Their mean duration is **15.5950 ms/token**, including the full four-device target call and its transfers/synchronization. This is a direct GPU-phase measurement, unlike the earlier unassigned residual. Read-only NVML snapshots immediately before and after every sixteenth position produce 152 valid clock records: SM clocks range from 2,377 to 2,460 MHz, and all memory clocks report 13,801 MHz. Boundary samples do not establish continuous in-kernel clock behavior, but they do not support a low-clock explanation. No clock setting was changed.

The same run counts exactly 15,120 routed expert calls during the selected decode intervals. Gate/up matrices take 5.3078 ms/token, down matrices 3.0672 ms, activation processing 1.5005 ms, final BF16 rounding 0.0849 ms, allocation 0.0139 ms and free 0.0082 ms. These counters exclude shared experts and are subdivisions of routed-expert time, not additional end-to-end costs. Counter and clock instrumentation affects the diagnostic's throughput; it is not an optimization candidate. CPU block measurements remain approximately 18.7366 ms attention and 14.1493 ms MoE per token.

The independent checker validates **264 successful requests** with exact token IDs, zero speculative tokens and zero sampled swap. A follow-up Nsight Systems capture covers fixture 1 decode after its first output token: 63 intervals, 2,205 target layers, and 63 vocabulary-head calls. All six fixture token sequences remain exact. CUDA kernel timings are instrumented diagnostics, not accepted throughput. The additional six Nsight requests are validated separately because profiler report messages share stdout with the JSON records.

The principal GPU kernel totals per token are 3.4782 ms for target FP8 MMA (split plus unsplit), 2.0176 ms for routed FP4 matvec, 1.9398 ms for shared-expert FP8 matvec, 1.5820 ms for sparse attention, 1.2326 ms for BF16 compressor projections, 1.1698 ms for hyperconnection input normalization, and 0.8823 ms for target normalization. The 0.6303 ms vocabulary head is outside the earlier target-GPU phase. Kernel sums and enclosing wall timers from different diagnostic runs must not be added together. No single kernel explains the entire gap. Reported CUDA memory-copy operation time totals 0.0693 ms/token across this capture; this is not a measurement of all transfer-related host latency.

An isolated GPU scale-decoding candidate replaces constant-memory lookups for ordinary E8M0 codes with bit construction, retaining table reads for special codes 0 and 255. A CUDA differential test verifies all 256 output bit patterns against the original host-initialized table. Full-model screening measured 19.5957 decode tokens/s and 14.9773 end-to-end tokens/s with six exact sequences and zero sampled swap. This does not improve the selected NUMA baseline, so the GPU scale variant is not adopted. The original GPU source is restored for subsequent builds. The ordinary independent checker now covers 270 requests, plus six separately validated Nsight requests. A CPU routed-expert activation candidate applies the existing BF16/SwiGLU/route-weight operations to independent 64-element chunks in one OpenMP dispatch. A 96-case bitwise differential test covers dimensions 1–4,099, the parallel threshold, uneven tails, clipping, signed zero and several route weights; the DeepSeek unit suite also passes with 24 physical-core threads. The first full-model screen reaches 20.0257 decode tokens/s and 15.4775 end-to-end tokens/s, with exact outputs and zero sampled swap. A fresh-process repeat reaches 19.8667 decode tokens/s and 15.4012 end-to-end tokens/s, also exact. Both are modestly above the older placement controls, but no adjacent same-binary disable control has been run. The candidate remains isolated pending stronger acceptance; this is not a claim that 20 tok/s is a stable new baseline.

A separate GPU candidate combines the eight target attention output-projection groups into one quantization launch and one three-dimensional MMA grid. It retains each group's original split-K accumulation order, writes directly to the concatenated output, and enlarges workspace quantization scratch for the full 32,768-column context. Draft and unsupported shapes keep the original path. A CUDA differential test compares all output bits against eight original projection calls at batch sizes 1, 3, 5, 16, 17, 49 and 128; all pass. Full-model screening uses the original CPU placement baseline, so its effect is separated from the activation candidate. Neither candidate changes the fixed CPU/GPU layer placement. The grouped GPU screen reaches **20.1812 decode tokens/s** and **15.3428 end-to-end tokens/s**, with six exact sequences and zero sampled swap. This roughly 3% difference needs a repeat and adjacent control; combined performance with CPU activation parallelism has not been measured. The independent ordinary checker now validates 288 requests, in addition to six separate Nsight requests. Both candidates remain artifact prototypes, not portable defaults. The 26.8 tokens/s goal remains unmet.

## Combined candidates and shared-expert GPU tiling

A fresh grouped-projection run measures 20.1261 decode tokens/s and 15.3445 end-to-end tokens/s, consistent with its first 20.1812 screening. Combining that GPU change with CPU activation parallelism measures **20.6575 decode tokens/s** and **15.8372 end-to-end tokens/s**. This combined result is one screening, not a repeated stable baseline. Every fixture remains bit-exact at the output-token level.

The same combined executable with `DSV4_CUDA_FP8_ROWS=4` measures 19.9256 decode tokens/s; eight rows measure 19.2936. Both regress from the explicit single-row control, despite exact output IDs. These settings are not adopted. The four cases run sequentially with unchanged layer placement and verified zero sampled swap. All 312 ordinary requests pass the independent checker, with six additional Nsight requests validated separately. The 26.8 tokens/s objective remains unmet.

A next isolated CPU expert-group candidate retains each worker's original sixteen-row static partition for every matrix rather than flattening expert and row tasks. It quantizes the common input once, computes all gate/up matrices in one dispatch, processes independent expert activations in parallel, and computes down matrices in another dispatch. Activation sums are still combined in original expert order. Unlike the earlier grouped screen, NUMA row ownership is retained. When explicitly enabled, a refusal fails this prototype rather than silently reverting, so full-model validation cannot accidentally measure the fallback. The 24-thread unit suite passes: 36 grouped matrix outputs match the original matvec bit for bit across shared/separate inputs and 32, 272 and 4,112 rows; the existing 96 activation differential cases also pass. Full-model performance is not yet measured. Sources and tests remain artifacts, not production defaults.

## Ownership-preserving expert-group screening

The first complete CUDA build exposed the candidate toggle declaration in the wrong CPU fallback scope. It was moved into `moe_token_pipeline`, and the complete fixture executable and unit suite were rebuilt successfully. A new differential test additionally compares twelve complete expert forward outputs (six experts, clipping off/on, multiple route weights) against the original per-expert path bit for bit. This covers activation processing and down projection as well as the existing 36 matrix-output comparisons.

A same-executable off/on pair measures **20.6399 / 21.0713 decode tokens/s**, with end-to-end rates of 15.8443 / 16.0938. All twelve requests match reference token IDs, speculative counters remain zero, and sampled swap stays zero. The approximately 2.1% screening gain needs a fresh-process repeat. It is not a stable 21 tokens/s acceptance claim and does not meet the 26.8 target. The independent ordinary checker now covers 324 successful requests, plus six separately checked Nsight requests.

A subsequent isolated GPU candidate hoists the four hyperconnection sigmoid coefficients out of the hidden-dimension loop in `mhc_input_norm_batch`. It preserves reduction order and arithmetic expressions. Eight CUDA differential cases (1, 3, 16, 49 rows; widths 512 and 4,096) match the original kernel bit for bit on disjoint input/output buffers. Full-model performance is not yet measured. A planned paired campaign first repeats the ownership-preserving expert group, then applies the coefficient-hoisting kernel with the same CPU/GPU placement and expert-group setting.

## Expert-group repeat and GPU coefficient hoisting

The ownership-preserving expert-group repeat measures **20.9075 decode tokens/s**, with exact output IDs, versus its first 21.0713 and adjacent disabled control 20.6399. The GPU coefficient-hoisting screen measures **21.1888 decode tokens/s** and **16.1543 end-to-end tokens/s**, also exact. Its roughly 1.3% difference from the adjacent repeat is modest and needs confirmation. No stable 21.19 baseline or completion claim is made. The ordinary independent checker covers 336 requests, in addition to six separately checked Nsight requests; all ordinary cases retain zero speculative tokens and zero sampled swap.

Two further isolated candidates pass differential tests but have not been measured with the full model. A CPU matrix-pair version retains the same NUMA row ownership while carrying two independent matrix accumulators; its tests cover 126 matrix outputs including odd group tails, twelve complete expert outputs, and the existing unit suite. A GPU shared-expert FP8 candidate assigns one warp to each output row, carries eight partial accumulators per lane, and reproduces the original 256-thread reduction tree before the final warp shuffles. Eighteen CUDA differential cases cover output-row tails, column tails, separate grouped inputs and real projection dimensions, all bit-exact. It is opt-in and avoids the rejected multi-row CTA variants. A follow-up campaign will repeat the hoisting control and measure each new candidate independently, preserving fixed placement, quantization and target-only C1 generation.

## Paired CPU matrices and exact GPU warp reduction

The coefficient-hoisting control repeats at **21.2415 decode tokens/s** (16.1755 end-to-end), near its first 21.1888 result. Independent candidates measure **21.7175 decode tokens/s** for CPU paired accumulators and **21.5156** for the GPU shared-expert warp reduction, with end-to-end rates of 16.4510 and 16.2708. All eighteen requests match reference token IDs and retain zero speculative tokens and zero sampled swap. These candidates have not yet been combined or individually repeated. The ordinary checker now covers 354 successful requests, plus six separately validated Nsight requests. The 26.8 target remains unmet.

A further GPU hyperconnection prototype separates input mixing across multiple CTAs from the original ordered normalization reduction. Eight differential cases match the original outputs bit for bit. Full-model performance is pending.

A separate, explicitly opt-in CPU attention dot-product prototype changes FP32 reduction association through OpenMP SIMD, without changing weight quantization. This is not bit-exact internally: a diagnostic over 524,288 BF16 output elements reports 505 differences, maximum absolute error 0.001953125 and RMS error 0.0000116253095. These are observed errors on synthetic BF16 inputs, not universal bounds or acceptance criteria. It has not been accepted, and full-model token equality has not been evaluated. The candidate remains separate from the measured exact-output changes.

Before attributing model attention time to arithmetic throughput alone, a decode-only CPU hardware-counter campaign is prepared for cycles, instructions, FP assists and SSE/AVX mixing assists. The earlier zero-assist finding applied only to an FP8 microbenchmark, not model attention. The model counter run has not been executed. Its monitor resolves the actual engine executable beneath the perf wrapper rather than sampling the wrapper's memory usage.

## Combined exact-output controls and CPU hardware counters

A new same-executable sequence measures 21.3630 decode tokens/s with paired CPU matrices and coefficient hoisting, **21.8514** after enabling the exact GPU warp reduction, and **21.9490** with the two-stage GPU hyperconnection normalization also enabled. End-to-end rates are 16.0520, 16.5281 and 16.5597. All eighteen requests retain reference token IDs, zero speculative counters and zero sampled swap. The approximately 0.4% split-normalization difference is not an established gain. The first control is below the earlier paired-matrix screening, so historical peaks are not substituted for the adjacent control.

A separate decode-only `perf stat` run of the combined split candidate records 830,188,044,858 user cycles and 950,070,594,169 user instructions across the process's measured threads, including spinning workers. Both `assists.fp:u` and `assists.sse_avx_mix:u` report **zero**. The log contains five enable messages and six disable messages (initial disable plus five measured fixtures), covering 315 decode intervals. Startup, prefill and fixture 0 are outside the enabled intervals. The instrumented 21.8017 tokens/s is diagnostic throughput, not an optimization result. These counters do not support FP microcode assists or SSE/AVX transition assists as the current bottleneck; they do not rule out other instruction, memory or synchronization costs.

The ordinary result checker now validates 378 requests, plus six separate Nsight requests. The 26.8 target remains unmet. A new GPU normalization candidate computes squared inputs cooperatively into shared memory while keeping the original sequential addition order and a fallback for widths above 1,024. Twenty-four CUDA differential cases cover weighted/unweighted normalization, row counts 1/7/64, widths 63/512/1,024 and the 1,057 fallback; all output bits match. Full-model performance is pending. An existing exact FP8/BF16 candidate is also prepared on the current NUMA/paired CPU source, without a new performance claim. The reassociated CPU attention candidate remains separate and unaccepted pending full-model token checks.

## Protocol and validation

- Six fixed `long6` fixtures, 48–49 input tokens and 64 generated tokens, one initial 16-token warmup per process, no MTP.
- Per round, fixtures 1–5 contribute 315 decode intervals and 320 output tokens. End-to-end rates include prefill/request setup but exclude model load and warmup.
- OpenMP uses 24 threads, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, physical CPUs 0–23. Interleaved variants additionally use `numactl --interleave=all`.
- Unless explicitly marked unlocked above, all checkpoint shards are locked read-only by the campaign parent; sampled runner swap is zero. Every completed fixture is independently compared to baseline raw token IDs.
- Cache pooling and preload are opt-in through `V4_CPU_TAIL_CACHE=1` and `V4_CPU_TAIL_PRELOAD=1`. Preload requires full GPU expert residency and enough existing expert-cache slots; it does not silently enlarge the slot budget.
- Local CPU and remote AVX-512 unit tests cover quantization rounding, packed-kernel convergence, parallel attention, routing, and grouped output projection. The 22 related Python source/build/oracle tests pass.
- No new 0–6 GPU sweep, concurrent serving, long-context, or MTP performance claim is made.

Raw logs, token sequences, source snapshots, commands/environments, resource samples, and `summarize.py` are retained under `/home/Kei/colibri/result/dsv4-roofline-20261001`. The remote experiment is isolated under `/data/test/colibri-roofline-20261001`; the service checkout is not modified. Further experimental kernels remain subject to full-model validation and repeated target measurements.


## Cooperative normalization and exact CPU BF16 screening

The adjacent old-normalization control measured 21.9664 decode tokens/s; cooperative squared-input loading with the original ordered sum measured 22.0338. The 0.3% difference is not an established gain. Enabling the exact CPU FP8/BF16 path on the same NUMA/paired implementation measured 22.1328 decode tokens/s (16.6236 end-to-end). Trace messages confirm eligible calls with shapes 8192 x 1024 and 1024 x 4096, so this result is not merely a disabled-path control. The small difference does not establish a repeatable improvement. All eighteen requests match baseline IDs, have zero speculative counters, and show zero sampled swap. The ordinary checker now covers 396 requests, plus six separate Nsight requests.

The reassociated attention-dot case is still being measured in a separate result directory. OpenMP default waiting, ACTIVE waiting, and a long finite spin count have been prepared as sequential same-executable controls; no performance claim is made before those runs and output checks finish.


The full-model attention-dot screen has now finished: 22.1775 decode tokens/s, but only five of six reference sequences match. Fixture 5 first differs at output index 29. This numerical candidate is rejected and retained exclusively under `results-numerical/`; it is not promoted into accepted performance results. Zero sampled swap and zero speculative counters do not override the token mismatch. The independent report is `attention-dot-full-model-validation.json`.

A source/runtime check rules out missing CUDA Graph replay in the current build: the forced-disable branch applies only to `COLI_DSV4_DEEPGEMM`, which is absent from the CUDA compiler invocation. Each of the four current stages reports two captures and 389 replays. Enabling an already active mechanism is therefore not a remaining optimization.


## Pending worker-wait and local huge-page controls

The sequential wait-policy campaign is running with the frozen `numa-norm-candidates` executable, CPU BF16 and reassociated attention disabled. Cases are `wait-default24`, `wait-active24` (`OMP_WAIT_POLICY=ACTIVE`), and `wait-spin-long24` (`GOMP_SPINCOUNT=10000000`). The same checkpoint locking/migration, core affinity, placement, model, and fixtures apply. Results are pending independent validation.

An additional isolated prototype, `numa-local-huge.c` with `numa_rows_local_huge_trial.h`, is prepared but not enabled in the running campaign. After private-page placement, it requests synchronous 2 MiB collapse only for fully aligned ranges wholly within one node's ownership, then verifies every base page again. It changes no global THP settings or checkpoint mappings. The unit source `test-numa-local-huge.c` checks all bytes, all 2,048 page locations, and 6 MiB of actual AnonHugePages in an aligned 8 MiB mapping. It compiles locally without warnings; remote execution and model performance are still unverified. Remote compilation/testing must wait until the current timing campaign ends.


The first worker-wait control has finished at 22.1821 decode tokens/s with all six exact sequences and zero sampled swap. This is a fresh default-wait control, not a new optimization. ACTIVE and finite-long-spin cases are still running sequentially.

A GPU compressor candidate is prepared in `backend_cuda_dsv4-bf16-warp.cu` / `backend_cuda_dsv4_target-bf16-warp.inc`. For a single target token, each warp carries the eight original per-thread partial sums and reproduces the existing 256-thread reduction tree. Other batch sizes retain `mm_bf16_batch`. Its proposed benefit is avoiding sixteen-token shared scratch and repeated CTA barriers on the single-token path; no benefit or correctness claim is made yet. Nine CUDA differential cases cover 17/512/1024 rows and 129/1024/4096 columns, but have not yet run. Both this unit and the huge-page unit must wait for the current full-model campaign to release the machine. Follow-up cases have been prepared as `huge-compressor-cases.json`, contingent on both units passing.


ACTIVE waiting has completed at **23.8798 decode tokens/s** (17.6630 end-to-end), compared with the adjacent default-wait result of 22.1821: approximately 7.7% higher. All six sequences remain exact with zero speculative counters and zero sampled swap. This is a promising single-pass result requiring repetition; the fixed 26.8 target remains unmet. The finite-spin case is still running. Follow-up huge-page/compressor controls are prepared with ACTIVE waiting, including an unchanged-kernel control and a combined candidate, subject to unit validation before launch.


The finite-spin run finished at **23.9584 decode tokens/s** (17.5872 end-to-end), also with exact IDs, zero speculative counters and zero sampled swap. This is approximately 8.0% above the adjacent default-wait control and close to ACTIVE, not evidence that finite spinning outperforms ACTIVE. The independent ordinary checker now covers 414 requests; the rejected six-request attention-dot trial remains separate. The target is still unmet.

Both proposed unit gates now pass on Yuesheng. The local-huge unit preserves all 8 MiB of data, verifies all 2,048 base pages on their expected nodes, and observes 6,144 KiB of AnonHugePages. The CUDA compressor test passes all nine bitwise comparisons. These checks establish bounded correctness of the prototypes, not application speed. The combined frozen executable is being built for ACTIVE-wait same-binary control, local huge pages, compressor warp, and both together. Build/test artifacts remain isolated from the portable implementation.

The isolated `numa-huge-compressor` build completed successfully, and the four-case campaign has started. Before launch all six GPUs reported 2 MiB allocated and 0% utilization, and the vLLM watchdog remained inactive. No performance result from this campaign is yet accepted.


The new executable with both new candidates disabled repeats ACTIVE waiting at **23.8568 decode tokens/s** (17.6671 end-to-end), close to the earlier 23.8798. All six sequences are exact with zero speculative counters and zero sampled swap. The local-huge run is now active; compressor-only and combined runs remain queued in the same sequential campaign.

A further CPU attention candidate, `numa-attn-rank4.c`, interleaves four independent key-score accumulators while preserving the ascending column order inside each score and the original rank order for softmax/value accumulation. It is distinct from the rejected reassociated dot reduction. A local build passes 32 bitwise output comparisons (326,600 elements) across BF16/non-BF16 inputs, 63/512 dimensions, 1/7/64/128 keys, missing keys and tail groups; 32 all-invalid cases retain failure behavior. Yuesheng compiler/runtime validation and full-model speed are still pending. No numerical or performance acceptance is inferred from the local check alone.


The strict full-model local-huge attempt failed during startup after 59.156 seconds: `MADV_COLLAPSE` returned `ENOMEM` while packing expert slot 129, before any generation. The campaign stopped, so its queued compressor and combined cases did not run. The failed setup is preserved under `results/failed-setups/local-huge24/` locally and excluded from accepted throughput statistics. The successful 8 MiB unit does not establish that every eligible range can be collapsed with the full checkpoint resident.

The revised helper (`numa_rows_local_huge_best_effort.h`) requests collapse one aligned 2 MiB range at a time. For ENOMEM/EAGAIN it restores NOHUGEPAGE on that range, retains normal pages, and counts failed bytes; all other errors still fail the prototype. Every page is still checked for its original NUMA owner. The ordinary unit observes 6 MiB of actual huge pages, all exact bytes, and all 2,048 correct page locations. An injected single ENOMEM unit observes 4 MiB successful / 2 MiB fallback, again with exact data and placement. Success/fallback counters are distinct; partial collapse is not described as full huge-page residency.

The rank-four attention differential suite now also passes on Yuesheng: 32 exact cases, 326,600 elements, and 32 all-invalid cases. A new isolated `numa-huge-rank4` executable is being built with five same-binary ACTIVE-wait cases: all new options disabled, compressor warp only, rank-four attention only, best-effort huge pages only, and all three combined. No result from this revised campaign is yet accepted.

The revised build completed successfully and `campaign-huge2.py` is now running. The prior strict campaign is terminal, not concurrently active. Remote unit logs are copied into the local raw artifact directory.


## Revised candidate campaign: initial completed cases

The revised same-binary ACTIVE control measures 23.8871 decode tokens/s (17.6770 end-to-end). Enabling only the exact compressor warp measures **24.2722 decode tokens/s** (17.8944 end-to-end), about 1.6% above that adjacent control. All twelve requests match baseline IDs with zero speculative counters and zero sampled swap. This is a single screening difference, not a repeated performance guarantee. Rank-four attention, best-effort huge pages, and the combined candidate remain in the live sequential campaign. The fixed 26.8 target remains unmet.


The revised campaign is complete. Rank-four attention alone measures 23.9567 decode tokens/s, only 0.3% above control, so no useful standalone gain is established. Best-effort huge pages alone measure 23.8791, effectively unchanged. Its helper reports 2,858,418,176 expert bytes plus 859,832,320 dense bytes successfully collapsed and 134,217,728 expert bytes falling back, with all original NUMA page locations verified. The combined candidate measures **24.3800 decode tokens/s** (17.7563 end-to-end), versus compressor-only 24.2722; that extra 0.4% does not establish a benefit from retaining the two other options. All thirty campaign requests match baseline IDs with zero speculative counters and zero sampled swap. The ordinary checker now covers 450 requests, plus six separate Nsight requests; the rejected numerical trial remains excluded.

The next campaign repeats compressor-only with ACTIVE waiting and then collects updated CPU phase timers on that configuration. The diagnostic source changes only attnprof/blockprof output precision from integer to six-decimal milliseconds. Each profiler kind must contain exactly 2,520 decode records for CPU layers 35–42 across fixtures 1–5, with first-token/prefill records excluded. Diagnostic throughput is not claimed as a new optimization. A smaps rollup taken during the combined run's startup is retained remotely, but is not a decode-residency measurement or attribution of huge pages to the selected weights.

The precision-only diagnostic build succeeded. `campaign-active-profile.py` has now started, with the unchanged frozen candidate repeated before the separately instrumented run. The prior five-case campaign is terminal.


## Updated ACTIVE/compressor profile and SMT preparation

The uninstrumented compressor-only repetition measures **24.1432 decode tokens/s** (17.7431 end-to-end), with exact IDs and zero sampled swap. This is below the first 24.2722 screen but still modestly above the approximately 23.86–23.89 ACTIVE controls. The separate diagnostic run remains exact and measures 42.0301 ms/token. Each profiler kind contains exactly 2,520 unique decode records on CPU layers 35–42. CPU attention accounts for 16.2252 ms, MoE 11.2755 ms, and HC phases 1.2652 ms; the remaining 13.2642 ms includes GPU and other costs. Attention subdivisions include output projection 6.4731 ms, sparse-attention phase 3.8098 ms, query projection 2.9681 ms, and compressor 1.5247 ms. Subdivisions must not be added again to parent phases. The ordinary checker now covers 462 requests.

The historical `qdq-free48` case actually used 24 OpenMP threads with a 48-logical-CPU allowed mask; it is not a 48-thread SMT result. A current topology read confirms physical cores 0–23 and sibling logical CPUs 24–47, with node runs 0–11, 12–23, 24–35, 36–47. A new isolated helper supports 24 or 48 workers with explicit logical-CPU places in ascending order. It coalesces twelve-worker runs, derives each range from the actual static OpenMP tile partition, binds it to the corresponding node, then verifies every interior page after all migrations. This keeps small tasks distributed over physical cores before SMT siblings. It changes no arithmetic or global host settings.

The final physical-first unit passes with both 24 and 48 threads: each preserves all 8 MiB and independently checks all 2,048 page owners from the worker that actually receives each statically scheduled task. The 24-thread and 48-thread full-model controls use the same new executable, ACTIVE waiting, compressor warp enabled, and unchanged model/layer placement. Build and full-model performance are pending; SMT is a hypothesis, not an accepted gain.

The isolated SMT build completed successfully and `campaign-smt.py` is now running. The prior compressor/profile campaign has terminated successfully. No SMT throughput result is yet accepted.


## SMT result and finer phase diagnostic

The same-executable physical-first control measures 24.0225 decode tokens/s with 24 threads; 48 threads regress to **16.7246** (12.7491 end-to-end). All twelve sequences match reference IDs, with zero speculative counters, zero sampled swap, and startup NUMA page verification. The SMT candidate is rejected; the selected configuration remains 24 physical cores. This result establishes a regression under the tested ACTIVE policy, not a universal explanation of SMT behavior. The ordinary checker now covers 474 requests.

A finer diagnostic is now running on the previous 24-thread ACTIVE/compressor configuration. It separates CPU sparse-attention call time, inverse RoPE, and the remaining setup inside the attention phase, and adds shared-expert, loader-wait, and routed-expert timers around the grouped CPU path. The build succeeded. All these counters are subdivisions of existing parent phases, not additive new costs. The diagnostic still requires full token validation and exact record-count checks before interpretation.


## Hidden indexer preparation cost

The fine diagnostic completed with exact IDs, zero speculative counters and zero sampled swap. All three profiler kinds contain exactly 2,520 records. It measures 41.3625 ms/token, including CPU attention 15.8971 ms and MoE 11.0420 ms. Within the 3.7468 ms attention subphase, the sparse-attention call costs only **0.8794 ms**, inverse RoPE 0.0447 ms, and remaining setup **2.8226 ms**. Deferred CPU indexer selection falls inside this setup bucket. Its 1,260 decode records attribute 0.3947 ms to query projection, **1.7947 ms to preparation**, 0.2676 ms to scoring, and 0.0049 ms to sorting (these indexer counters have 0.001 ms source precision). MoE subdivisions are shared expert 2.7352 ms, loader wait 0.3387 ms, and grouped routed experts 7.1047 ms. These nested values must not be added to parent phases. The ordinary checker now covers 480 requests.

The indexer currently parallelizes preparation over tokens. For C1 this dispatches a team while one worker serially prepares all 64 independent heads and their head-weight dot products. The isolated `numa-indexer-head-parallel.c` candidate serializes that outer region for a single needed token, then parallelizes independent heads with private FP4-QDQ scratch and unchanged within-head arithmetic. Multi-token preparation retains its original outer parallelism; active nested regions do not create extra teams. It is gated by `V4_CPU_IDX_HEAD_PARALLEL=1`.

A 72-case Yuesheng differential test compares the modified functions against extracted, untouched originals, covering 1/7/64 heads, 128/256/512 head dimensions, 257/4096 hidden widths, positions 0/85, and both disabled/enabled modes inside a serialized OpenMP outer region. Query preparation and head-weight outputs are bitwise identical. The full-model candidate/control build is pending completion; no end-to-end speedup is yet claimed.

The isolated indexer-head build completed successfully and `campaign-indexer-head.py` is now running the same-binary disabled/enabled controls. The fine diagnostic campaign is terminal.


## Final wrap-up at user request

The last indexer-head campaign completed normally. The disabled control measures **24.1979 decode tokens/s**, while head parallelism regresses to **10.2508** (8.9762 end-to-end). Both cases match all six reference sequences with zero speculative counters and zero sampled swap. The candidate is rejected on performance. The timing difference alone does not identify the mechanism of the regression; no further experiment was started. The ordinary result checker now validates **492 exact requests**; separate profiler and rejected numerical artifacts retain their documented boundaries.

The fixed target remains **26.8 decode tokens/s and is not achieved**. The highest single screening result is **24.3800**, approximately 45.5% of the ideal 53.5858 streaming roof and 3.19 times the initial 7.6318 baseline. The simpler selected configuration has observed results around **24.1–24.3**. The additional rank-four attention and huge-page switches do not have an established incremental benefit. Keep the selected 24-physical-core configuration with ACTIVE waiting and exact GPU compressor warp; do not promote the rejected SMT, indexer-head or reassociated-dot variants.

Advanced changes remain isolated research prototypes and frozen runners under `/home/Kei/colibri/result/dsv4-roofline-20261001` and `/data/test/colibri-roofline-20261001`. The integrated changes and this report are prepared on `perf/dsv4-offload-roofline` in `/tmp/colibri-dsv4-roofline`. The advanced research prototypes are not integrated into defaults. The main checkout was not modified by this work.

The files `bf16_compressor_rows16_trial.h` and `test-bf16-compressor-rows16.c` are unfinished standalone prototypes: they have **not been compiled, run or integrated into the model**. They carry no correctness or speed claim. CPU output projection, routed experts, and indexer preparation remain measured areas for possible future work, not commitments to further experiments in this session.

The final SSH campaign exited successfully. No indexer campaign or benchmark process remains; locked checkpoint memory is released (`Mlocked: 0 kB`). All six GPUs report 2 MiB used and 0% utilization. The vLLM watchdog timer remains inactive and no vLLM container is running. The service was not restored. Optimization is paused at the user's request.
