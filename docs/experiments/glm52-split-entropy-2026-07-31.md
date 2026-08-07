# GLM-5.2 lossless VRAM-base / RAM-residual feasibility

Date: 2026-07-31

## Question

Can every routed-expert INT4 weight be represented by a small base kept in
VRAM, with an exact residual fetched from RAM only after routing, then restored
on the GPU without changing model output?

## Real-weight capacity result

`c/tools/analyze_split_entropy.py` scanned 3,840 routed projection tensors
(24.16 GB packed weights) from layers 3, 10, 30, 50, and 76 of
`/data/models/GLM-5.2-colibri-int4`. It also read all shard headers to obtain
the corpus size:

- routed packed weights: 372.052 GB;
- routed F32 scale sidecars: 0.797 GB;
- measured symbol entropy: 2.9147 bits/weight (72.87% of raw INT4);
- per-layer entropy: 2.7698, 2.9433, 2.9495, 2.9505, and 2.9386 bits/weight.

At the measured 176.57 GB expert-VRAM budget, after scales and one 32-bit tile
offset per 4,096 weights, the base may average 1.882 bits/weight. An ideal
entropy residual would be 96.06 GB, but that is not a practical random-access
codec.

The tested GPU-decodable representation uses a global unequal-group codebook:

- 1-bit base: groups of 4/12 symbols, 2.559 residual bits/weight;
- 2-bit base: groups of 1/2/2/11 symbols, 1.258 residual bits/weight;
- 11.8% 1-bit tiles plus 88.2% 2-bit tiles fit the VRAM budget;
- practical RAM residual: **131.31 GB**;
- total representation: 3.294 bits/weight, or 82.34% of raw INT4.

This is lossless symbol coding. It does not claim that 744B model information
has been reduced to the VRAM base alone; the missing information remains in
RAM.

## RTX 5090 decode and transfer microbenchmark

`c/tools/benchmark_split_entropy.cu` generated symbols with the measured GLM
histogram, packed both formats, reconstructed every symbol on one RTX 5090,
and compared the complete output before timing.

| Format | GPU decode only | Pinned H2D + decode | Result |
|---|---:|---:|---|
| 1-bit base + 2.560-bit residual | 184.0 Gweight/s (92.0 raw-INT4 GB/s) | 65.8 Gweight/s (32.9 raw-INT4 GB/s) | exact |
| 2-bit base + 1.259-bit residual | 184.3 Gweight/s (92.1 raw-INT4 GB/s) | 97.8 Gweight/s (48.9 raw-INT4 GB/s) | exact |

The capacity-weighted H2D-plus-decode rate is about 91.9 Gweight/s, equivalent
to 46.0 GB/s of original packed INT4 per GPU. With about 607 routed experts per
token and 19.12 MB of packed weight per expert, a perfectly balanced six-GPU
transfer/decode-only ceiling is roughly **23.8 tok/s**. This is an upper bound,
not model throughput: expert matmul, routing, attention, synchronization, and
contention are excluded.

## Conclusion

The representation is technically feasible. The standalone decoder initially
looked fast enough to continue, but the decisive fused-matvec test rejected the
current representation as a decode-speed path.

The important negative constraint is that all routed experts now require RAM
traffic and GPU compute. The current 8.90 tok/s engine already overlaps hot
GPU experts with cold CPU experts; replacing that path can lose the existing
parallelism. A real claim requires a fused residual-decode + INT4 matvec path,
six-GPU placement, a 131 GB pinned/registered residual store, and an ABBA
end-to-end replay against the 8.90 tok/s baseline. Until then, 23.8 tok/s is
only a transfer/decode roof and must not be reported as achieved throughput.

## Fused matvec follow-up

`c/tools/benchmark_split_matvec.cu` tested the two real GLM projection
geometries (`2048 x 6144` and `6144 x 2048`) against a resident raw-INT4
W4A32 kernel. All paths reconstructed the same weights; the exact global-scan
variant produced bit-identical FP32 outputs. A faster segmented reduction only
changed FP32 accumulation order (maximum relative error below 3.1e-5).

| Geometry | Codec | Raw resident | Fused decode+matvec | Pinned H2D+fused |
|---|---|---:|---:|---:|
| 2048 x 6144 | 1-bit variable | 0.013 ms | 0.054 ms | 0.186 ms |
| 2048 x 6144 | 2-bit variable | 0.013 ms | 0.055 ms | 0.120 ms |
| 2048 x 6144 | 2-bit direct+exceptions | 0.013 ms | 0.049 ms | 0.124 ms |
| 6144 x 2048 | 1-bit variable | 0.014 ms | 0.056 ms | 0.172 ms |
| 6144 x 2048 | 2-bit variable | 0.014 ms | 0.055 ms | 0.098 ms |
| 6144 x 2048 | 2-bit direct+exceptions | 0.014 ms | 0.047 ms | 0.107 ms |

The first address scheme scanned all variable residual widths per row and was
7-10x slower than raw. A segmented warp-prefix implementation reduced that to
about 4x slower. The final direct-bitplane plus per-warp exception-stream
format removed the global prefix dependency at the cost of increasing the
2-bit residual from 1.259 to 1.491 bits/weight; it was still 3.4-3.8x slower
before transfer and 7.6-9.5x slower with transfer.

This is worse than the engine baseline in two additional ways not represented
by the table: Colibri already fuses gate/up projection work, and its CPU and
GPU expert lanes overlap. Therefore full-engine integration is not justified.
The capacity result remains useful, but the proposed lossless split exchanges
RAM capacity for an address-dependent codec that destroys the cheap raw-INT4
inner loop.

Artifacts on `rs-yuesheng-gpu`:

- `/tmp/glm52-split-entropy.json`
- `/tmp/analyze_split_entropy.py`
- `/tmp/benchmark_split_entropy`
- `/tmp/benchmark_split_entropy.cu`
- `/tmp/benchmark_split_matvec`
- `/tmp/benchmark_split_matvec.cu`
