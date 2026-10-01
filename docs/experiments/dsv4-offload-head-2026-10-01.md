# Independent MTP GPU head and resident CPU head scheduling

Follow-up to the CPU offload study: five-GPU MTP spent 20.011 of 24.148 measured seconds drafting, including 5.044 seconds in the head region. Target layers were fully GPU-resident. The previous `V4_MTP_GPU_HEAD=1` setting still depended on `V4_MTP_GPU_DENSE=1` through `g_v4ds_gpu_engine`, so turning off GPU dense execution also disabled the GPU head.

This change lets the explicit GPU head flag use an enabled GPU engine independently of drafter dense placement. The existing GPU head implementation and its failure propagation remain unchanged. GPU head execution also avoids allocating the unused host logits buffer. CPU-only engines still use the CPU path.

For a resident CPU head, the projection now schedules all vocabulary rows in one OpenMP parallel loop rather than starting a loop for each 64-row tile. Nonresident weights retain the bounded 64-row buffered-read path. Each output row retains its original accumulation order.

## Validation

- Full default CPU engine build passed.
- CUDA-linked fixture runner built on Yuesheng and completed the hardware campaign below.
- 22 existing DSpark/build/oracle Python tests passed.
- Extracted before/after projection loops passed 24 bit-exact comparisons: resident and buffered paths, one and four OpenMP threads, and vocabulary sizes 1, 63, 64, 65, 129, and 8193. This checks projection logits, not the complete MTP model or GPU path.
- `git diff --check` passed.

Local validation artifacts are in `/home/Kei/colibri/result/dsv4-offload-head-20261001`.

## Yuesheng hardware validation

Measured implementation: `5892d828`. Five RTX 5090 32 GB cards, all 43 target layers resident on GPU, three drafter stages on CPU; two Xeon Silver 4510 CPUs, 12 OpenMP threads, concurrency one. The sixth GPU remains idle.

Each configuration ran three rounds of the six `long6` fixtures with 64 output tokens and one initial 16-token target warmup. Within each round, fixture 0 is excluded separately in each mode. Each rate is 320 tokens divided by summed request wall time for fixtures 1–5, including prefill and excluding model loading/warmup. The table reports the median of the three rates. Mode order alternates by round. Configurations ran sequentially in the order below; this is not a randomized confidence-interval study.

All 48 checkpoint shards (155.425 GiB) were locked read-only in the parent and `COLI_V4_DIRECT=0`. Sampled benchmark-process swap was zero. Baseline total sampled physical reads were 16.60 MiB; both new configurations recorded zero. No whole-checkpoint SSD rereading occurred.

| Configuration | Target-only median tokens/s | MTP median tokens/s | MTP round range | Mean logged head ms/call |
| --- | ---: | ---: | --- | ---: |
| Baseline, original CPU head | 43.4814 | 13.7520 | 13.1381–14.1084 | 74.89 |
| New CPU head scheduling, GPU head disabled | 43.4347 | 14.0234 | 13.7615–14.3230 | 63.24 |
| New independent GPU head | 43.0830 | 16.6738 | 16.6685–16.9892 | 2.00 |

Independent GPU head improves MTP median throughput by **21.25%** against the fresh baseline. CPU scheduling alone improves the observed median by **1.97%**, with overlapping round ranges, so a robust standalone end-to-end gain is not established. Its measured head-region total falls 15.55%.

Across the three measured rounds, each configuration logged 201 drafter calls. Head-region totals were 15.053 / 12.712 / 0.402 seconds. GPU-head CPU-stage totals still sum to 41.010 seconds; CPU drafting remains the dominant cost. Every measured MTP round accepted 231/333 draft tokens, so the observed gain does not depend on improved acceptance.

All **108/108 requests** completed with 64 tokens, no early EOS, and zero generation return codes. Independent raw-token comparisons matched every request against the corresponding baseline target-only fixture, across all modes/configurations/rounds. This is workload-specific parity, not general numerical equivalence. All three runner exit codes were zero.

Runner SHA-256:

- Baseline: `faeca9dcac26262fa5e2fef2c010728adbaaad33e7d31b7fabcf3f276508e888`
- Updated: `3431f367efd6505fe5e01752cb2f5a85cc6d8e449284f11edd40a51634a8ecac`

Local artifacts contain `campaign.py`, `summarize.py`, `summary.json`, per-configuration command/environment, raw token JSONL, engine logs, resource traces, build log, hashes, and final state. Remote artifacts are isolated in `/data/test/colibri-offload-head-20261001`.

After the campaign all six GPUs returned to 2 MiB / 0% utilization, locked memory returned to zero, the vLLM container remained stopped, and its watchdog timer remained disabled/inactive.

## Remaining limit

Even after this improvement, five-GPU target-only execution (about 43 tokens/s) is substantially faster than CPU-stage MTP (16.67 tokens/s). This change does not address CPU target-layer or drafter-stage costs. No new 4-to-0 GPU performance curve or full CPU model validation is claimed.
