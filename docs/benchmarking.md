# Benchmarking protocol

The [published benchmark table](benchmarks.md) is useful only when every number
carries enough context to reproduce its operating point. These rules exist
because plausible measurements reported in good faith have failed each one.

## A cautionary table

One Windows 11 / Gen4 NVMe host produced four answers to "the disk's random-read
bandwidth":

| result | method | what it measured instead |
|---|---|---|
| 75 MB/s | 4 KiB random, QD1, Python | latency-bound small blocks, not expert-slab traffic |
| 3683 MB/s | 128 KiB, QD32 | warm page cache, not the device |
| 675 MB/s | unbuffered, 4 KiB-aligned lengths | a confounded run whose proposed 64 KiB length effect was later refuted ([#863](https://github.com/JustVugg/colibri/issues/863)) |
| about 2900–3000 MB/s | unbuffered, 64 KiB-multiple, QD4+ | the useful result for that host and workload |

The third row is deliberately retained as a retraction. A controlled rerun did
not reproduce the claimed length effect, so Colibri did not carry the proposed
64 KiB rounding patch. Correcting a mechanism is part of the benchmark record,
not something to hide from it.

## Five rules

1. **State cache state beside every number.** Say how a cold cache was evicted,
   or label the run warm. A warm number without the label is a page-cache
   measurement presented as storage performance.

2. **State request size, alignment, and queue depth.** "4 KiB random" and
   "1 MiB random" can differ by orders of magnitude on one drive. Record
   buffered versus direct/unbuffered I/O too. Alignment is part of the method;
   it is not evidence for a particular slow-path mechanism without a control.

3. **Measure the access pattern the engine will run.** Match expert-slab size,
   queue depth, readahead, and concurrency. A textbook storage benchmark does
   not predict a workload that reads 19 MiB expert slabs at QD2.

4. **Run an independent control before trusting the headline metric.** A second
   engine process, a background build, thermal drift, or predictions written
   under the wrong index can all produce believable throughput. Record a
   sanity metric that must stay stable and stop when it does not.

5. **State the model and expert-cache state.** Include model and quantization,
   cache policy and capacity, and whether the LRU or pinned set was cold, warm,
   or already trained. Hit rate alone is insufficient: a lower hit rate can be
   faster when GPU dense compute pays for the extra expert reads.

## Minimum viable report

Every throughput or latency result should include:

- commit or release, full model ID, and quantization/container format;
- exact command, environment, prompt or corpus, seed, and requested token count;
- cache policy, capacity, state, and the procedure that established that state;
- buffered or direct I/O, request size, alignment, queue depth, and model-volume
  location;
- CPU, RAM total and available, accelerator, storage device, and relevant
  topology;
- throughput, TTFT/latency, expert hit rate, bytes read, and the independent
  control or quality check;
- run count, ordering or interleaving, summary statistic, and spread;
- raw logs, including failed or excluded runs and the reason for exclusion.

Change one variable at a time. For small effects, interleave baseline and
candidate runs in the same session; ordered sweeps can turn cache warming,
frequency drift, or background load into a false win. Publish negative results
and retractions with the same detail as positive ones.

Use `python tools/datapoint.py` for the standard machine, cold/warm, and rotating
prompt record. See [benchmarks.md](benchmarks.md) for current measurements,
[tuning.md](tuning.md) for runtime controls, and [windows.md](windows.md) for
platform-specific I/O constraints.

This protocol originated with the measurements and draft contributed by
[@outtodata in #867](https://github.com/JustVugg/colibri/issues/867), including
their later correction of the #863 explanation.

## Compare HTTP serving with a fixed workload

`c/tools/benchmark_http_serving.py` uses the OpenAI-compatible streaming chat
endpoint and Python's standard library. Run it against Colibri, SGLang, or vLLM
with the same workload and generation settings. This measures the whole HTTP
request path, including server queueing and prefill, unlike a steady-decode-only
engine benchmark. No performance comparison is implied by providing the tool.

Create a UTF-8 JSONL file with one conversation per line. Rows contain only
`messages`; messages contain a `system`, `user`, or `assistant` role and string
`content`:

```jsonl
{"messages":[{"role":"user","content":"Explain how a CPU cache works."}]}
{"messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is a mutex?"}]}
```

From the repository root, with the server already running:

```sh
python3 c/tools/benchmark_http_serving.py \
  --base-url http://127.0.0.1:8000/v1 --model your-served-model \
  --workload prompts.jsonl --concurrency 4 --repeats 3 \
  --max-tokens 128 --temperature 0 --timeout 60 --output colibri-c4.json
```

Repeat with the other server's API root, model alias, and a distinct output file.
Authentication uses `OPENAI_API_KEY`, or the environment variable named by
`--api-key-env`; keys and message contents are not copied into reports. The
endpoint and model alias are recorded. URLs with credentials, queries or
fragments are rejected, and redirects are not followed. The requested endpoint
must support `stream_options.include_usage`; errors are recorded rather than
silently changing the workload or retrying.

The JSON report includes the workload's SHA-256, generation settings, one record
per attempt in input order, HTTP status, failure category, relative start time,
duration, first output time, finish reason, and reported completion tokens.
Failed requests remain in the report, and any failure makes the command exit 1.
Latency summaries use successful requests only, with nearest-rank p50/p95/p99
and sample counts. Failed requests retain their individual durations and any
observed output/usage.

Optional `--warmup-requests N` sends N requests before measurement, cycling
through workload rows with the same generation settings and concurrency limit.
Warmup is closed-loop even when the measured phase uses `--request-rate`.
All warmup requests finish before a fresh measurement clock and arrival schedule
start. Their rows and summary are recorded separately under `warmup`; they do
not contribute to measured latency, tokens, throughput, or SLO goodput. The
default is zero (no warmup). If any warmup request fails, the report has
`status: "warmup_failed"`, `summary: null`, and an empty measured `requests` list;
the command exits 1 without starting measurement. Successful warmup does not
prove stable performance. It can populate prefix caches, so record the same
warmup and cache policy when comparing engines or runs.

Measurement boundaries:

- Without `--request-rate`, concurrency is closed-loop: at most that many
  requests are in flight, and each worker starts its next request after its
  previous stream ends. There is no automatic retry or cache flush. Repeats reuse the
  conversations in file order; prefix caching and scheduling can affect results.
- Request timing begins inside the worker, before HTTP connection setup, and
  excludes waiting for a local worker. Each request uses a new connection.
  `--timeout` limits individual socket operations, **not total request time**;
  a stream that keeps sending data can last longer.
- `first_output_seconds` measures receipt of the first nonempty content,
  reasoning, or tool-function name/arguments delta (`tool_calls` or legacy
  `function_call`). Role-only and empty deltas
  do not count. This is client-visible first output latency, not necessarily
  time to a visible answer or to exactly one token. Empty successful output
  has no first-output sample. SSE chunk gaps are not reported as token latency.
- Success requires a supported finish reason (`stop`, `length`, `tool_calls`,
  `function_call`, or `content_filter`) and `[DONE]`. Error/unknown finish reasons,
  HTTP errors, stream errors, malformed responses, and incomplete streams fail.
  With `n=1`, each nonempty choices array must contain exactly one choice with
  integer index 0. Output text and tool-function fields must be strings or null.
  Once a choice has finished, further choice chunks are rejected; a trailing
  usage chunk with empty choices is accepted.
  This establishes protocol completion, not output correctness; filtered or
  length-limited output can still count as protocol success.
- Token counts come only from `usage.completion_tokens`. Successful completion
  token throughput divides those counts by the **entire batch wall time**,
  including failed attempts. It is null if any successful request lacks usage
  (or no request succeeds). Counts from failed streams are excluded. Inspect
  failure rate and usage coverage alongside throughput; backend tokenizers,
  reasoning-token accounting, and stopping policies may differ.

For a meaningful comparison, record the model weights, quantization, tokenizer,
chat template, reasoning mode, server commands/versions, cache state and hardware
beside the report. Keep requested settings equal, check actual output lengths,
and run an independent quality check: this tool deliberately does not save
response text or assess correctness. Retain the workload file with its hash,
interleave server runs, and report repeated-run spread. Start at concurrency 1,
then increase it to expose queueing and prefill interference.

### Measure throughput within a latency target

Add `--slo-first-output 1 --slo-duration 15` to require first output within one
second and protocol completion within fifteen seconds. Either flag can be used
alone; values are finite positive seconds and the boundary is inclusive.
`summary.latency_slo` reports the thresholds, timing basis, `requests_met`, the fraction of
**all attempts** meeting them, and `goodput_requests_per_second` (qualifying
successful requests divided by the entire batch wall time). Without thresholds,
this field is null. Failures never qualify, even if they emitted output before
failing. When a first-output target is set, empty-output successes also do not
qualify. Missing token usage does not prevent evaluating these latency targets. In
closed-loop mode the targets start at the HTTP request; scheduled-arrival mode uses
the scheduled arrival time and includes client dispatch delay.

This follows the latency-constrained goodput approach used by
[vLLM's serving benchmark](https://docs.vllm.ai/en/latest/api/vllm/benchmarks/serve/),
with the client first-output boundary defined above. It does not assert TPOT or
per-token SLOs, output quality, a minimum response length, or an equivalent vLLM
TTFT definition. SLO misses alone do not change the CLI exit status: exit 1 still
means at least one protocol/transport failure. Record output-length and quality
controls alongside goodput so short or empty answers cannot masquerade as an
improvement.

Keep the load model fixed when comparing reports.
[SGLang's serving benchmark](https://github.com/sgl-project/sglang/blob/main/python/sglang/benchmark/serving.py)
also supports request-rate-driven arrivals and trace timestamps. This tool uses
closed-loop concurrency by default; `--request-rate` selects periodic arrivals
by default, or Poisson arrivals with `--arrival-distribution poisson`,
as described below. None of these modes alone establishes a production SLO guarantee;
warmup/cache policy, workload representativeness and repeated-run controls still
matter.

### Schedule arrivals independently of response time

Add `--request-rate 5` to schedule five arrivals per second. Request `i` is due at
`i / rate` seconds from batch start; the first is due immediately. Absolute
monotonic deadlines prevent accumulated timer drift. This is a deterministic
periodic schedule. Add `--arrival-distribution poisson --seed 42` for
independent exponential intervals with mean `1 / rate`. The first request is
still immediate; subsequent deadlines accumulate sampled intervals. This follows
the arrival model supported by
[vLLM](https://docs.vllm.ai/en/latest/cli/bench/serve/#--burstiness) and SGLang;
it does not reproduce their random-number sequences or implement trace replay.
The default seed is 0. Equal seeds, rates and request counts reproduce planned
arrivals independently of warmup and response time, not actual network timing.
Finite Poisson samples need not realize the configured mean rate. The report
records the distribution and seed; retain per-request scheduled times for exact
schedule comparison. Compare identical schedules across servers, then repeat
with multiple seeds to measure sensitivity to arrival patterns.

Requests are not retried
or dropped, and the finite workload drains before the report is written.

`--concurrency` still caps simultaneous HTTP requests. Arrivals accumulate in
the client's executor queue when all workers are busy. Therefore the configured
rate is **scheduled arrivals, not guaranteed wire or server arrival rate**.
Delayed producer wakeups also contribute to dispatch delay; inspect client load
before attributing all delay to the server. The queue and retained results can
grow to the workload's total request count, so size workloads accordingly.

Scheduled-arrival per-request records add:

- `scheduled_seconds`: planned arrival relative to batch start;
- `dispatch_delay_seconds`: actual worker start minus scheduled arrival;
- `arrival_first_output_seconds`: dispatch delay plus HTTP first-output latency,
  or null if no output was observed;
- `arrival_duration_seconds`: dispatch delay plus HTTP request duration.

`summary.arrival_timing` reports dispatch-delay samples for all attempts and
arrival-based latency samples for successes. Existing HTTP timing fields retain
their original meaning. With a rate set, SLO evaluation uses arrival-based times
and records `timing_basis: scheduled_arrival`; without one it records
`request_start`. This prevents a request waiting two seconds for a worker and
then completing in 100 ms from meeting a one-second total-latency target.
Goodput still divides qualifying completions by the entire batch wall time,
including the drain after the last scheduled arrival. Keep rate, concurrency,
request count, latency targets and timing basis equal across compared reports.
