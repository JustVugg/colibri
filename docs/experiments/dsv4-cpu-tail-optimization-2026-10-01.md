# Four-GPU CPU-tail optimization and validation

This follow-up optimizes CPU offload on Yuesheng. It retains 35 target layers on four RTX 5090 GPUs and computes the remaining eight target layers on CPU. MTP is disabled throughout, so drafting does not confound target-layer timing.

## Results

Each primary configuration has three measured rounds: one screening round and two confirmation rounds in a fresh process. Rates include prefill and request setup, exclude model loading and warmup, and exclude fixture 0 separately in each round. The table reports the median of per-round rates, not a median over individual requests.

| Configuration | Output tokens/s | Decode tokens/s | Mean TTFT, median across rounds (s) | Output tokens/s range |
| --- | ---: | ---: | ---: | --- |
| baseline-12 | 4.7598 | 6.1920 | 3.3274 | 4.6569–4.7606 |
| optimized-12 | 5.2327 | 6.9330 | 3.1008 | 5.1844–5.2670 |
| optimized-24 | 5.2461 | 7.6219 | 3.8862 | 5.0246–5.2709 |

optimized-12: **9.94%** higher end-to-end throughput and **11.97%** higher decode throughput than baseline-12.

optimized-24: **10.22%** higher end-to-end throughput and **23.09%** higher decode throughput than baseline-12.

The 12-thread comparison isolates code changes. The 24-thread comparison combines those changes with physical-core affinity and twice the OpenMP thread count; its entire gain must not be attributed to code. More threads improve decode here but increase TTFT. No thread/NUMA defaults are changed. Optimized 24-thread end-to-end throughput is effectively tied with optimized 12-thread throughput in this short workload, while TTFT is longer; prefer 12 threads for this tested latency/throughput balance.

## Implementation

- Share the existing exact AVX2 FP8 decoder with the dual and batched CPU paths, replacing table gathers without changing their multiplication or accumulation order.
- Gate hot-expert rows16 packing by GPU ownership of the specific layer. Previously, any non-null GPU cache disabled packing even for CPU-only tail layers. GPU-owned layers continue to retain uploadable row-major slabs; CPU tail layers use the existing converged packed kernels.
- Preserve `COLI_V4_ROWS16=0` as the existing packing opt-out. No layer-placement or numerical precision change is introduced.

## Profiling and screening

With the original 12-thread binary and `DSV4_ATTN_PROF=1`, measured fixtures 1–5 contain 2,520 CPU decode block calls. Rounded log timers sum to 24.579 s in MoE and 21.001 s in attention, approximately 54%/46% of their combined time. MoE timing includes expert access and computation; these timers do not establish memory bandwidth versus compute saturation. Attention output projections account for 9.578 s. Millisecond-rounded per-call times undercount small phases. Prefill is analyzed separately.

Single-round screening controls (not repeated confidence estimates):

| Configuration | Output tokens/s | Decode tokens/s | TTFT s |
| --- | ---: | ---: | ---: |
| baseline-12 | 4.7598 | 6.1920 | 3.2260 |
| decode-12 | 4.8871 | 6.3552 | 3.1295 |
| decode-node0-12 | 4.4367 | 6.3495 | 4.4567 |
| decode-24 | 5.2057 | 7.5833 | 3.9397 |
| packed-12 | 5.1844 | 6.8479 | 3.1017 |
| packed-24 | 5.2709 | 7.6677 | 3.8763 |

`decode-*` enables only the FP8 decoder change. `packed-*` adds CPU-tail rows16 packing. `decode-node0-12` pins threads to physical CPUs 0–11 and regresses TTFT, so it is not selected. The 24-thread configurations pin physical CPUs 0–23 with `OMP_PROC_BIND=close` and `OMP_PLACES=cores`. These are affinity/thread experiments, not an exhaustive controlled-memory-placement NUMA study.

## Protocol and evidence

- Official DeepSeek-V4-Flash-0731, 48 shards / 155.425 GiB; six available RTX 5090 32 GB cards with only four visible to the runner; 2 × Xeon Silver 4510, 251 GiB RAM.
- Concurrency one, existing `long6` fixtures, 48–49 prompt tokens (fixture 3 has 49; all others have 48) and 64 output tokens per request, one initial 16-token target warmup per process.
- Per-round throughput is 320 outputs / sum of request wall times for fixtures 1–5. Decode rate is 315 intervals / summed recorded decode duration; small generation-end bookkeeping remains in that duration.
- The campaign parent locks all checkpoint shards read-only with `mmap`/`mlock`; `COLI_V4_DIRECT=0`. Locking is outside timing. All sampled benchmark-process swap is zero. Small process physical reads, if any, are retained in `validated-results.json`, rather than claiming zero I/O.
- Configurations run sequentially, without randomization or statistical confidence intervals. Primary medians combine the original screening pass and two confirmation passes; raw round ranges are shown above.
- **78/78 requests** completed with exactly 64 output tokens and zero generation/runner errors. Independent raw-token comparisons match every output to its baseline fixture, including the profiling and screening runs. This is workload-specific correctness evidence, not a general model-quality assessment.
- Default CPU engine and CUDA-linked fixture runner builds pass. Native quant tests cover all 256 FP8 codes, including signed zero and NaN classification; existing DeepSeek-V4 tests, including packed/unpacked convergence, pass. The 22 related Python source/build/oracle tests also pass.
- This campaign does not validate new MTP throughput, other GPU counts, long contexts, concurrent serving, Windows runtime behavior, or full-model sanitizer behavior.

Implementation commit: `78a5ae2c`. Local and remote changed-source hashes match. Runner hashes are retained with the raw evidence:

```text
3431f367efd6505fe5e01752cb2f5a85cc6d8e449284f11edd40a51634a8ecac  /data/test/colibri-cpu-tail-20261001/baseline
6c05a6297e0975515df0bf98aa7a7372e55ff496ea4ad35ef09a681a13c5b00e  /data/test/colibri-cpu-tail-20261001/decode-only
e972e3c8c0808f071fac02ad787dd12d483bc580be5a08ffdfc1e72c5a29590a  /data/test/colibri-cpu-tail-20261001/c/tools/bench_dsv4_fixtures
a550c1a17e4de5c1649835787c7e557662f3678fb37173b53aba4f713decf432  /data/test/colibri-cpu-tail-20261001/c/deepseek_v4.c
25ad08db1d449d0e3391c1e966775b0b6c62fa5eed2627579bfef3f73104ade9  /data/test/colibri-cpu-tail-20261001/c/native_quant.h
b898e0d21be27ebbe962b77de92a8d8a11381eae0f58ac3e20c13c37b4c6e51a  /data/test/colibri-cpu-tail-20261001/c/deepseek_v4_internal.h
```

Raw evidence, exact commands/environments, campaign and aggregation scripts, resource samples, build logs, and SHA-256 manifest are retained in `/home/Kei/colibri/result/dsv4-cpu-tail-20261001`. Remote experiment files are isolated in `/data/test/colibri-cpu-tail-20261001`. The adjacent committed results JSON contains aggregate validation and all primary round measurements; the raw bundle is not included in Git.

After the campaign, all benchmark processes exited, all six GPUs returned to 2 MiB / 0% utilization, and locked memory returned to zero. vLLM remains stopped and its watchdog timer remains disabled/inactive.
