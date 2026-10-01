# DeepSeek V4 device execution

The final opt-in path reaches **92.7322 token/s** (formal C1 MTP median),
up from **16.0957 token/s** at `d60213a4`, a 5.76x improvement.

This experiment uses six RTX 5090 GPUs with the official
DeepSeek-V4-Flash-0731 checkpoint. It follows the execution differences found
in Naruto's retained `pipeline-batch-11` source and benchmark artifacts.

## Implemented paths

- `V4_DEVICE_TARGET=1`: prepared, multirow target layers. Activations, raw
  compressor projections, full-set KV and expert routing remain on device.
  Contiguous device boundaries use peer copies with stream event ordering.
- Retained speculative prefixes update a visibility cursor. Absolute history
  avoids overwriting older window entries; a rejected suffix is overwritten by
  the next execution. Execution failure destroys the prepared state.
- `V4_DEVICE_GRAPH=1`: captures each contiguous device stage by row count.
  Positions are device inputs. Changes to shared scratch capacities invalidate
  graphs both before and after eager execution. Feature taps are device copies.
- `V4_DEVICE_MMA=1`: native FP8 MMA with per-row 128-element activation scales,
  16-byte loads and fixed row tiling. Small-output matrices use deterministic
  split-K. The register layout follows Naruto's `cuda_quantized_mma.cuh`.
  Compressor BF16 projections retain the existing portable kernel.
- `V4_DEVICE_FP4_VEC=1`: vectorized four-byte FP4 loads and one warp per
  output, removing CTA-wide reductions from the routed expert GEMV.
- `V4_MTP_GPU_HEAD=1`: vocabulary head, sequential Markov correction, argmax and
  confidence on GPU; only proposal IDs and confidence are downloaded.
- `V4_MTP_DEVICE=1`: experimental five-row device execution of each MTP stage,
  with a separate resident 256-expert table on each stage's GPU.

These flags default off. The target path currently supports the official
4096/1024/64-head/512-dimension layout and full-set attention up to
`min(2048, 4 * index_topk)` positions. Admission includes a 25-token draft tail.
It disables CPU prefix checkpoint reuse. It does not implement long-context
indexer selection or concurrent sessions. The prepared engine owns one active
prefix. The optional device paths are currently Linux CUDA only.

This is still contiguous layer placement, not Naruto's dense TP2 / expert EP2
execution. Final target head inputs and MTP context features still cross to
host. MTP stages currently exchange their residual through host memory.

## Numerical contract

The device compressor uses the production CPU compressor as its oracle,
including ratio-4 overlap, ratio-128 boundaries, and overwritten suffixes after
retaining 0 through 6 speculative positions. CPU/GPU floating-point elementary
functions can differ near BF16/FP8 rounding boundaries.

The vectorized FP4 kernel changes the floating-point reduction order.
The FP8 MMA mode also introduces activation quantization that the old generic
CUDA dense dispatch bypassed. It is a distinct numerical mode; equivalence to
legacy target token IDs is not promised. Acceptance requires exact IDs between
single-step target and MTP verification within the selected mode, repeated-run
stability, complete outputs, and no early EOS. All performance comparisons must
state this distinction.

## Rejected experiments and fixes

- A BF16 WMMA prototype regressed the first 256-token target fixture from about
  13.5 seconds to 78.3 seconds. It was stopped and removed. Native FP8 MMA is the
  replacement, not a claim that the BF16 prototype accelerated inference.
- Initial graph invalidation checked scratch capacity only before execution.
  A 48-to-49-token prompt transition grew allocations during eager prefill and
  left the old one-row graph with freed addresses. The failing run is retained;
  invalidation now also occurs after execution. The corrected six-fixture
  256-token graph run passed target/MTP token parity.

## Validation and measurements

Results are being collected in `result/dsv4-device-target-20261001`.
The fixture remains long6, C1, 256 output tokens per prompt, including prefill
and excluding model load. A formal run consists of one warmup and three measured
rounds, alternating target/MTP order. Partial runs are not acceptance results.

Initial completed graph-only warmup: target 18.5297 token/s; MTP 23.6315 token/s.
Both produced all 1536 tokens; all six MTP outputs matched target IDs exactly.
Completed formal medians:

| Path | Target-only token/s | MTP token/s |
| --- | ---: | ---: |
| Previous `d60213a4` | 13.1743 | 16.0957 |
| Device target + FP8 MMA + GPU Markov head | 30.3785 | 34.3304 |
| Above + device MTP stages | 30.3815 | 43.3199 |
| Above + vectorized FP4 experts | **45.4714** | **92.7322** |

Each new formal run above contains 48 complete samples with exact within-mode
single-step/MTP token IDs. All six final target outputs differ somewhere from the
legacy `d60213a4` output; the first differences occur at zero-based positions
108, 28, 59, 28, 3 and 241. This is not a bit-exact legacy-kernel speed comparison.
Final formal MTP rounds are **91.7660, 92.8205, 92.7322 token/s**. Each
produced 1536 tokens, drafted 1683, and accepted 1178 draft tokens. All 48 final
samples produced exactly 256 tokens without EOS or errors; all repeated IDs and
all target/MTP comparisons are exact. The first MTP warmup was 66.0534 token/s
and includes lazy loading of the three MTP expert banks; it is excluded from the
formal median.

Naruto's 77.9747 token/s is a retained historical measurement, not a rerun in
this experiment. Draft width and numerical paths differ; this result does not
establish cross-engine output equivalence.

## Checks

- CPU release, CUDA CLI/fixture runner, Windows GPU/session-unit cross-compiles,
  and 22 related Python tests passed.
- Existing attention-prefix trials passed all retained lengths, split chunks,
  ring wrapping and fallback cases.
- GPU 0/5 component tests cover compressor overlap/boundaries and suffix
  overwrite; CPU activation-QDQ equality; FP8 MMA scalar/multirow equality with
  mixed activation and weight scales; sequential Markov IDs and confidence;
  FP4 vector GEMV; workspace admission; and cross-device data/token/position
  transfer. Compute-sanitizer memcheck reports **0 errors** for this suite.
- Existing resident-route scalar/batch and ordinary/hash parity tests passed
  on both devices through batch 128. The existing DSpark attention oracle also
  passed (maximum relative L2 4.4533e-05).
- FP4 GEMV microbenchmark, 36 distinct 2048x4096 matrices on GPU 5: portable
  0.522539 ms versus vectorized 0.106957 ms. This is a kernel-only diagnostic,
  separate from the end-to-end result.

## Reproduction and evidence

Build the generic CUDA tier with:

```sh
make -C c -f Makefile.deepseek-v4 CUDA=1 CUDA_ARCH=sm_120 -j12 tools/bench_dsv4_fixtures
```

Set all experiment variables before opening the engine. The six new switches
are `V4_DEVICE_TARGET=1 V4_DEVICE_GRAPH=1 V4_DEVICE_MMA=1
V4_DEVICE_FP4_VEC=1 V4_MTP_DEVICE=1 V4_MTP_GPU_HEAD=1`. The complete environment,
per-round results, fixture hash and source/binary hashes are in the adjacent
`dsv4-device-execution-2026-10-01-results.json`. Run the fixture tool with
`MODEL_DIR long6.prompts 256 4 td`.

Local evidence is retained under
`/home/Kei/colibri/result/dsv4-device-target-20261001`, including per-stage source
archives, raw token IDs, logs, run scripts and rejected experiments. The final
isolated remote build is `/data/test/colibri-device-vector-20261001`.
Measured fixture-runner SHA-256:
`40137237ff97aa9dfb0ae3e8d2dcdc82819b8b6b69ce897995e48d7eb063a100`.
Its implementation source hashes match the submitted implementation. The final
mixed-scale test extension and main Makefile dependency fixes do not alter the
measured runner.
