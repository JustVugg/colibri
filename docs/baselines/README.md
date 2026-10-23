# Colibri, SGLang and vLLM serving baseline

This is a reproducible collection protocol, **not a measured performance ranking**.
`c/tools/benchmark_baseline.py` reuses the existing HTTP harness. It does not start,
stop, reconfigure, or install engines. Run one engine at a time on an explicitly
allocated host; three servers sharing the same GPU would invalidate isolation.
No model or GPU is needed to plan or summarize an experiment.

## Freeze the experiment

Copy `three-engine.example.json` and `workload.jsonl` into an experiment directory.
Fill every `REPLACE` value before running; the example intentionally fails
validation. Workload paths are relative to the manifest, not the working directory.
Keep the same manifest for every engine/round. Do not store keys in launch commands:
use environment-variable references. Only `api_key_env` names are read by the
collector; their secret values are never included in results.

Record:

- The physical host, CPU/NUMA topology, RAM, GPU count/model/VRAM, storage,
  OS and driver/runtime versions. Use the same allocated hardware for every arm.
- The exact source model revision, tokenizer and rendered chat-template SHA-256s.
  Hash a sorted inventory of weight files for each `artifact_sha256`; retain the
  inventory and conversion commands. Include weight **and KV** precision in
  `quantization`. These are operator declarations, not remote attestations.
- Engine commits or image digests and complete launch commands, including TP/PP/EP,
  context limit, admission limits, memory budget and CPU thread placement.
  Aliases and URLs may differ; they do not identify the checkpoint.
- Reasoning, speculation, prefix-cache and warmup policies. Disable speculation in
  the initial baseline, then create a separate experiment to measure it. Warmup is
  performed before **each concurrency cell**. Caches persist across cells; the
  collector never flushes them. Reset/restart before each round if the declared
  policy calls for it. Cold and warm results belong in separate experiments.
- A fixed quality evaluation dataset/revision, metric and acceptance threshold.
  Run this independently on all three configurations and retain the results.

Choose the comparison explicitly:

| Mode | Enforced identity | Permitted interpretation |
| --- | --- | --- |
| `matched_artifact` | All three weight inventory hashes, formats and quantization descriptions must match | Matched-artifact performance observations, conditional on external quality and configuration checks |
| `deployment` | Shared source model/tokenizer/template and workload; engine artifacts may differ | A comparison of complete deployment configurations, **not an isolated engine speedup** |

A Colibri converted INT4 artifact and a vLLM NVFP4 checkpoint belong in
`deployment`, even if they originated from the same model. Matching labels or
hashes does not prove equal quality or backend numerical behavior.

Use separate manifests/directories for `fully_resident` and `offload`. Verify
residency with engine counters and I/O measurements; do not infer it from model
size or successful startup. Collect RAM/VRAM peaks, disk/PCIe traffic and power in
external telemetry, aligned to each report's UTC `started_at`. The manifest's
hardware data is descriptive; the tool does not measure memory or energy.

## Collect the matrix

From the repository root:

```sh
python3 c/tools/benchmark_baseline.py plan --manifest /path/experiment/manifest.json
```

The default example describes C1/C4/C8/C16, three independent rounds, and 16
measured requests per cell. **The two included prompts are smoke inputs only**;
replace them with representative short, long and shared-prefix workloads and
increase the request count before interpreting P95/P99. Request counts must at
least reach the largest concurrency; this does not guarantee sustained occupancy.
`repeats` repeats requests within a cell; `rounds` repeats complete measurements.

The plan rotates engine order to reduce systematic order bias:

1. Colibri, SGLang, vLLM.
2. SGLang, vLLM, Colibri.
3. vLLM, Colibri, SGLang.

After manually starting the selected engine with the recorded configuration:

```sh
python3 c/tools/benchmark_baseline.py run \
  --manifest /path/experiment/manifest.json \
  --engine colibri --round 1 --results /path/experiment/results
```

Repeat for each plan entry, releasing the previous engine's allocated hardware
before starting the next. The collector only contacts the selected endpoint.
Each cell writes `<engine>-r<round>-c<concurrency>.json`, with the full manifest,
workload and harness hashes, raw per-request records, warmup and summary.
Existing cells are never overwritten. A failed warmup saves evidence and stops
the round; measured failures are retained and make the command exit nonzero.
For a failed/invalid campaign, keep the evidence and use a new results directory
for a replacement campaign. Do not silently substitute the fastest repeat.

## Summarize and interpret

```sh
python3 c/tools/benchmark_baseline.py compare \
  --manifest /path/experiment/manifest.json \
  --results /path/experiment/results > /path/experiment/comparison.json
```

The comparison rejects mismatched manifests/workloads, mixed collector/harness
versions, duplicate or unexpected cells, altered summaries and missing requests.
Missing cells, failed requests, missing usage and empty/filtered output are listed
as issues and return a nonzero exit. No failed cell is silently dropped.

For each engine/concurrency it reports min/median/max **across rounds** for:

- Aggregate successful completion tokens per full batch wall second.
- Client first-output P50/P95/P99 and completion-duration P95.
- Failure rate and latency-SLO request goodput.
- It also reports actual server-reported output-length distribution across rounds.

Round percentiles are not pooled percentiles. Throughput is aggregate, never
per-user decode speed, and includes HTTP, queueing and prefill. First output is
the first nonempty content/reasoning/tool delta, not an instrumented first token.
SSE chunk gaps are **not ITL**, and no token-latency/TPOT estimate is manufactured.
Instrument the engines or use their native benchmarks for token timestamps;
report those separately with their definitions. The underlying HTTP timing,
usage and socket-timeout boundaries are documented in [benchmarking.md](../benchmarking.md).

`protocol_complete` means the matrix completed without those protocol/data gaps.
It does **not** mean quality passed, outputs have equivalent lengths, SLOs were
met, resources were isolated, residency was verified, or memory/power was measured.
`quality: not_assessed` remains explicit. This report deliberately emits neither
winner labels nor speedup ratios. Inspect lengths, reasoning accounting, external
quality and telemetry before publishing any performance conclusion.

The matrix currently uses closed-loop traffic. Rate-driven interference tests
remain available in `benchmark_http_serving.py`; do not mix them into this matrix.
A useful follow-up workload adds a long prefill while short decodes are active,
with a separate report of first-output and ongoing token-latency interference.
