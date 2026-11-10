# Exact MXFP4 host unpack: REAP decode A/B/A

Replacing the scalar nibble lookup in `dev_codes_fp4` with byte shuffles reduces
CPU work in the hybrid expert path. The integer codes, scales, dot products,
and accumulation order are unchanged. AVX2 builds use the vector loop; other
builds and the tail retain the scalar implementation.

On this machine, three 128-token decode samples per arm give **54.740% higher
mean throughput** against the combined original controls, or **55.736% higher
median throughput**. This is a controlled short-decode result for one model
and configuration, not a general speed claim for all MXFP4 workloads.

| Arm / run ID | Decode samples (tok/s) | Mean | Median | Short TTFT samples (s) |
| --- | --- | ---: | ---: | --- |
| A1 original / 20261108-123123 | 1.873533, 2.063555, 2.225330 | 2.054139 | 2.063555 | 11.434885, 9.722035, 9.809959 |
| B SIMD / 20261108-124254 | 3.031738, 3.267190, 3.261263 | 3.186730 | 3.261263 | 11.332329, 9.643631, 9.773303 |
| A2 original / 20261108-125349 | 1.833811, 2.124646, 2.235572 | 2.064676 | 2.124646 | 11.363491, 9.683080, 9.837459 |

All three fixed reference answers and all seven benchmark request/response
pairs (one warmup plus six measurements) are byte-identical across A/B/A.
The combined six-sample control median is 2.094100 tok/s; the candidate median
is 3.261263 tok/s. Short TTFT is essentially unchanged.

## Why this change

An engine-PID-only `perf` recording at 199 Hz found 60.79% of CPU samples in
`host_rows`, with 79.15% of those in the inlined scalar FP4 expansion: about
48.12% of all CPU samples. That recording covered a decode request including
its prompt, with no lost samples. These percentages describe CPU samples,
not the fraction of wall time that can be eliminated. See `hotspot-evidence.json`.

The change expands 16 packed bytes into 32 signed int8 codes with two table
shuffles and interleaved stores. It does not add quantization, change expert
placement, or change a GPU shader. All 69 runtime SPIR-V hashes match between
the two measured builds (`runtime-artifacts.json`).

## Controls and limitations

- Model: local SHA256-verified DeepSeek V4 Flash REAP, native MXFP4 experts,
  FP8 dense weights, D=4096, expert I=2048, 43 layers, 132 experts, top 6.
- Hardware: Xeon Gold 6254, guest 32 vCPUs and 247.37 GiB RAM; experiment
  CPUs 16–31, OMP 16; one Arc Pro B60 24 GB SR-IOV VF, PCIe Gen3 x8.
- Runtime: Ubuntu 26.04.1, kernel 7.0.0-42-generic, Mesa 26.2.6,
  GCC `-O3 -march=x86-64-v3`, 64K context, RAM_GB=128, BIG_ROWS=2,
  automatic chain rows. Local ext4 on virtio-scsi; no model reads from CIFS
  during these runs. Full service configuration and image/binary hashes are
  in `provenance.json`.
- A fresh engine for each arm; disk prefix checkpoints disabled. One benchmark
  warmup, then three decode and three short-prefill requests in identical order,
  with identical text, generation caps, temperature 0 and reasoning effort none.
  Reference prompts run before the benchmark and are included in `replay.py`.
- `COLI_VK_TIER_SYNC=1` and `COLI_VK_TIER_BALANCE=0` are fixed in **every arm**.
  Default asynchronous placement changed one answer on an unchanged-engine
  repeat, so it could not establish an exact-output control. The deterministic
  reference is separate and differs from the earlier adaptive reference; these
  settings have not been adopted for serving.
- Exact output parity here is a regression check, not a semantic-quality
  evaluation. It does not prove all prompts or models are unchanged.
- These are Light HTTP benchmarks, without `perf`/PCM tracing. Decode is
  `(completion_tokens - 1) / (last content event - first content event)`.
  Instrumented profiling timings are not used as performance baselines.
- Earlier full-workload original and row-cap-512 runs exhausted GPU memory on
  the second long prompt. This CPU patch does not claim to fix that failure,
  long-prefill latency, or the separate missing cooperative-matrix GEMM path.

## Source identity and reproduction

The end-to-end runtime base is
`e24cd51afd44b480c1e41ba82e20a28bac76c34f` plus `runtime-base.patch.gz` (decompress before applying), the common
Xe2 patches already used by the experiment engine. The trial adds only this
PR's `dev_codes_fp4` change. The publication branch is based on upstream
`f5d65566de468c8e7ac285a5e8c7257bf4d06856`; its complete `vk_tier.c` matches
the measured candidate byte-for-byte, as does the control file. Other source
files are not claimed to be the same between these two base revisions.

Build each runtime source with:

```sh
make -C c deepseek-v4 VK=1 ARCH=x86-64-v3 -j8
```

Start a separate GPU1-only engine with the recorded service environment,
mounts and Mesa version in `provenance.json`, then replay from this directory:

```sh
python3 replay.py --url http://127.0.0.1:8002/v1/chat/completions --output NEW-ARM.json
```

Recreate the engine between A, B and A. The replay preserves the measured HTTP
timing implementation and fixed request order, checks reference/benchmark
outputs, and writes raw timing events. It does not start/stop services or collect
telemetry. The original measurements used the local colibench session runner:

```sh
set -a
. /opt/colibri/colibench/models/deepseek-v4-flash-reap.env
set +a
export COLIBENCH_REQUEST_TAG=f6734185f687174a
python3 -u /opt/colibri/colibench/experiments/model_session.py \
  --stage baseline --reference deterministic-v1 \
  --knob COLI_VK_TIER_SYNC=1 --knob COLI_VK_TIER_BALANCE=0 \
  --workloads decode_short,prefill_short
# B: same command, --stage variant, plus:
# --image colibri-vulkan:xe2-models-fp4-mesa26.2.6 --source /opt/colibri/xe2-models-fp4
```

For the CPU-only microbenchmark, run the same benchmark driver on the original,
candidate and original again. No Vulkan device is required by this mode:

```sh
make -C c tests/bench_host_pairs tests/test_vk_tier VK=1 ARCH=x86-64-v3 -j8
taskset -c 16-31 env OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./c/tests/bench_host_pairs 7 4096 2048 16 160
./c/tests/test_vk_tier c/shaders/qmatmul.spv host-pairs
```

The driver uses 64 synthetic experts, two warmups plus five measured repetitions
at 1, 2 and 8 pairs per call. Output hashes are computed outside the timed region.
The earlier 1-thread and 16-thread A/B/A raw samples are in `runs.txt`; the
publication-base checks and samples are in `current-validation.txt`.

The exhaustive probe extracts the actual function and compares it with an
independent E2M1 formula, including every packed byte, unaligned buffers, vector
boundaries, scalar tails, guard bytes and unchanged input. Each probe checks
147,712 cases and 10,272,768 output values:

```sh
python3 docs/experiments/vulkan-mxfp4-host-unpack-2026-11-08/unpack_probe.py \
  c/vk_tier.c /tmp/mxfp4-probe-v3 --arch x86-64-v3
python3 docs/experiments/vulkan-mxfp4-host-unpack-2026-11-08/unpack_probe.py \
  c/vk_tier.c /tmp/mxfp4-probe-scalar --arch x86-64
env -u OMP_NUM_THREADS -u OMP_PROC_BIND -u OMP_PLACES make check
python3 c/experiment_manifest.py \
  docs/experiments/vulkan-mxfp4-host-unpack-2026-11-08/manifest.json
```

`end-to-end-{A1,B,A2}.json` retain all warmups, samples, generated text and timing
events. `end-to-end-audit.json` records comparisons and hashes of those raw files.
`manifest.json` uses the measured runtime base commit, with explicit patch
provenance, and the combined controls as its baseline. The manifest validates.

The current publication-base validation passed: all four unpack probes, both
host-pair suites, all 63 microbenchmark output hashes (including warmups), and
full `make check` (3,089 Python tests, 397 skips). The full build emitted 60
warnings in unchanged source/test paths; this is not a warning-free build claim.
The earlier-base check initially inherited OMP_NUM_THREADS=16 and failed three
Python environment-default subtests; removing that benchmark-only override
made its full Python rerun pass. Both the failure and rerun are retained in
`runs.txt`; the current full check ran without those OMP overrides.

Compiler versions are in `compiler.json`; the exact local model file identities
are in `local-model-manifest.json`. `perf-report.txt`, `host-rows-annotate.txt`
and `trace-stats.txt` support the sampled CPU finding.

The shared runtime patch is stored as deterministic gzip to preserve its exact
patch-format whitespace. `perf-report.txt` has trailing display whitespace
removed; its numbers and symbols are unchanged.
