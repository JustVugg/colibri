# Multi-disk streaming

Use extra drives when model reads limit inference. `COLI_MODEL_MIRROR` points
to read-only copies of the primary model's shards; no RAID setup is required.
These examples target the GLM-5.2 engine. Other engines have their own I/O
paths, so check their model guides before applying the same settings.

## Run with two or more copies

Run from `c/` in a source checkout, or the directory containing `coli` in an
unpacked release. Replace the paths with your existing model copies.

Bash, with one primary and two mirrors:

```bash
COLI_MODEL_MIRROR='/mnt/nvme2/glm52_i4;/mnt/nvme3/glm52_i4' \
  python3 ./coli chat --model /mnt/nvme1/glm52_i4
```

PowerShell, with the same three-drive layout:

```powershell
$env:COLI_MODEL_MIRROR = 'E:\models\glm52_i4;F:\models\glm52_i4'
python ./coli chat --model 'D:\models\glm52_i4'
```

For two drives, supply just one mirror directory. Quote the list: an unquoted
semicolon separates shell commands. The primary is supplied by `--model`
(or `COLI_MODEL`); do not repeat it in the mirror list.

With `COLI_DISK_WEIGHTS` unset, the engine probes bandwidth at startup and prints
the measured split. An optional override is a comma-separated list of positive
relative weights, primary first, followed by each usable mirror in list order.
For example, `9,3,3` requests shares of 60%, 20%, and 20% across three drives;
it is not a measured speedup. Check the startup log for skipped mirrors and
probe failures before trusting the split. See the
[environment reference](ENVIRONMENT.md#dual-ssd-streaming) for the variables.

Missing or divergent mirror shards fall back to the primary. For a smaller
drive, use the [partial-mirror planner](../README.md#multiple-ssds-stream-model-copies-from-more-than-one-drive)
to stage a usage-ranked subset. `COLI_MODEL_DIRS` is a different layout: it
spreads distinct shards across drives to combine capacity, rather than adding
replicas of the same shards.

## What has been measured

More storage bandwidth does not translate directly into more tokens per second.
Compute, cache residency, and shared PCIe/controller bandwidth can dominate.
Compare the complete inference workload, not just the drive probe.

- **Two independent NVMe drives helped.** The Threadripper PRO 7965WX report in
  [#1249](https://github.com/JustVugg/colibri/issues/1249), also recorded in
  [the benchmark table](benchmarks.md), measured 0.80 to 1.10 tok/s (+37.5%)
  with `DIRECT=1`; the buffered comparison gained about 16%.
- **Mixed-speed drives exposed a bug that has since been fixed.**
  [#1270](https://github.com/JustVugg/colibri/pull/1270) replaced equal-sized
  stripes with bandwidth-weighted chunks. Before that fix, adding a SATA drive
  to two NVMe drives reduced GLM-5.2 decode from 0.900 to 0.666 tok/s in the
  reported experiment. This is historical evidence, not a current slowdown
  attributed to every SATA mirror.
- **The post-fix third drive was neutral on that host.** The contributor's
  [follow-up on `dev`](https://github.com/JustVugg/colibri/pull/1270#issuecomment-5466324415)
  reported 1.030 tok/s with two NVMe drives and 1.006 with the added SATA drive
  (five interleaved runs per arm, reported standard deviations 0.041 and 0.034).
  The author judged the difference within run-to-run variation. That Windows
  11 / i9-14900K / RTX 3090 result does not promise a gain from a slower disk
  on another machine.

These are community measurements on specific configurations, not new results
from this guide. [#1137](https://github.com/JustVugg/colibri/issues/1137) contains
the multi-disk discussion and an earlier mixed-speed report.

## Compare against one drive

1. Record the commit, model/container, hardware and drive/controller layout.
   Keep the prompt, seed, output-token limit, RAM/VRAM budgets, and I/O mode
   fixed. Keep background disk work idle. Use the same learned usage history
   and cache preparation for each arm, and report cold and warm runs separately.
2. Start a fresh process for each arm. For the primary-only baseline, clear
   `COLI_MODEL_MIRROR`, its legacy alias `SNAP_MIRROR`, and any
   `COLI_DISK_WEIGHTS` override. For the mirror arm, set only
   `COLI_MODEL_MIRROR` to the copies being tested and let the probe choose the
   weights. Keep any `COLI_MODEL_DIRS` layout fixed; if it spans drives, label
   this a split-layout baseline rather than a single-drive baseline.
3. Set `PROF=1` in both arms. Save startup `[MIRROR]` messages, throughput,
   TTFT, expert hit rate, and per-drive `MIRROR:` byte/read counters. The GLM
   serve loop emits cumulative profile counters on clean shutdown; do not
   assume they appear after every chat response.
4. Interleave at least three repetitions per arm. Report the median and
   spread, output/correctness checks, and raw logs, including excluded runs.
   If testing `DIRECT=1`, run it as a separate comparison and verify that the
   platform and filesystem actually use direct I/O. See the
   [benchmarking protocol](benchmarking.md) for cache-state and reporting rules.

To return to the primary-only configuration in your shell:

```bash
unset COLI_MODEL_MIRROR SNAP_MIRROR COLI_DISK_WEIGHTS
```

```powershell
Remove-Item Env:COLI_MODEL_MIRROR, Env:SNAP_MIRROR, Env:COLI_DISK_WEIGHTS -ErrorAction SilentlyContinue
```

PowerShell's `$env:` assignments persist in that session; remove `PROF` too if
you enabled it only for this comparison.
