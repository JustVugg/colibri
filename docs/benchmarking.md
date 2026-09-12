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
