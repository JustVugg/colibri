# DeepSeek V4.1 Flash: where the expert bytes come from, and whether it matters

Two changes to the V4.1 engine are measured here: reading the routed experts with
`O_DIRECT` instead of through the page cache, and serving them from more than one
drive. Both are about the same fact -- an expert is read once and never wanted again --
and both are ports of mechanisms the V4 and GLM engines already had.

> ## Correction, 2026-09-15: the A/B below ran in the wrong regime
>
> Everything in this record was first measured with the engine's default OpenMP
> team, which on this host is one thread per logical CPU (208). That configuration
> is pathological for an engine whose per-region work is a few thousand rows:
> **93% of cycles land inside libgomp**, and the turn takes 409.9 s instead of
> 43.5 s. See `dsv41-omp-team-2026-09-15.md` and PR #1517.
>
> Measured with the team sized (`OMP_NUM_THREADS=32`), the same `O_DIRECT` change
> is not worth 2.3% -- it is worth **-52% of the disk phase and +63% of decode
> throughput**, and the disk phase is then 16.0 s of a ~34 s turn rather than 5% of
> it. Section ["The same A/B with the team
> sized"](#the-same-ab-with-the-team-sized) has those numbers.
>
> The 208-thread numbers are kept because they are real measurements of a real
> configuration, and because the ratio they produce is exactly the trap: the phase
> breakdown looks entirely reasonable either way. But the conclusion they were used
> for -- that I/O cannot matter for this engine -- was wrong, and the device
> characterisation below is the part that survives unchanged.

The short version, with the correction folded in: **the value of this work depends
entirely on whether the CPU side has been put in order first.** With the shipped
thread count the disk is 5% of the turn and neither change is worth 2.5%; with the
team sized the engine reaches the device's own ceiling and the same changes are worth
tens of percent. That is not a contradiction, it is a precondition, and it is the
reason this record now leads with a correction rather than a table.

## Machine

| | |
|---|---|
| CPU | 2x Intel Xeon Platinum 8555C, 52 cores/socket, 2 threads/core = 208 CPUs, 4 NUMA nodes |
| RAM | 1007 GiB |
| Storage A | Samsung MZQL2960HCJR (`nvme0n1`, 894 GB), ext4, `noatime`, mounted `/mnt/nvme0` |
| Storage B | 2x Samsung MZQLB7T6HMLA (`nvme1n1`, `nvme3n1`, 7.6 TB each) in RAID0 (`md0`), ext4, `/mdr5` |
| Storage C | yrfs network filesystem (`/ky200t`) -- staging only, never measured |
| commit | `8eb197e5` (`feat/dsv41-expert-io`, rebased onto `dev` at `f4aff21`) |
| OS / toolchain | Ubuntu 24.04, kernel 6.8.0-31-generic, gcc 13.3.0, `-O3 -march=native -fopenmp` |

A and B are independent controllers, which is what the two-drive question needs. B is
itself a two-drive RAID0, so the arms below are really *one* NVMe versus *two* versus
*three*: the point is the number of independent storage endpoints, and the device
sweep at the end shows why that distinction turns out not to matter here.

## Container

`deepseek-ai/DeepSeek-V4.1-Flash`, the released checkpoint, no conversion: 48 shards,
**510.3 GB**, 88 files. Both copies used here were verified file-by-file against the
upstream manifest (size, all 88 entries, exact) and one shard was sha256-compared
across them. The engram sidecar was generated once with
`python3 c/tools/prepare_dsv41.py --model <dir>` (2 tables, 99,092 compressed ids,
384,006,168 and 384,016,682 rows).

Access went through `hf-mirror.com`; `huggingface.co` is unreachable from this host
and the two 101 GB engram shards exceed the CLI's non-Xet download size limit, so
they were fetched as ranged requests (12 per shard) and joined locally. Worth
recording because it is the third time this checkpoint's tail has been the hard part.

## Method

The engine's own one-shot oracle mode (`SNAP=<dir> ./deepseek_v41 <cap> <ref.json>`),
the same entry point CI uses. It prints the generated token ids on stdout and the
phase breakdown on stderr, and drafting is idle in it, so the expert reads being timed
are the decoding path's own.

`tools`-side harness: prompt tokenized to 25 ids, 16 tokens generated, `cap=8`.
`ref.json` holds the 16 reference ids produced by a first run, so **every arm is held
token-exact** against the same stream -- routing an expert to a different drive is an
I/O decision and must not be visible in the output.

Protocol, per CONTRIBUTING.md:

- arms interleaved (P, M3, D0, M2) so any drift hits all of them;
- every run starts from an evicted page cache (`posix_fadvise(DONTNEED)` on the
  container's files -- `/proc/sys/vm/drop_caches` would need root per run);
- 3 runs per arm, medians reported;
- one variable changed per comparison;
- all 12 runs token-exact (16/16).

Arms:

| arm | configuration | independent endpoints serving expert bytes |
|---|---|---|
| `P` | primary only, on `nvme0n1` | 1 |
| `M3` | `COLI_MODEL_MIRROR` + `COLI_DISK_WEIGHTS=1,1` | 2 (`nvme0n1` + `md0`) |
| `M2` | `COLI_MODEL_MIRROR` + `COLI_DISK_WEIGHTS=1,1000` | 1 (`md0` only) |
| `D0` | `V41_DIRECT=0` (buffered reads) | 1 |
| `MP` | `COLI_MODEL_MIRROR` only, split left to the startup probe | 2 (`nvme0n1` + `md0`) |

## What the turn is made of

Baseline (`P`, `cap=8`, 25 prompt tokens, 16 generated):

| phase | seconds | share |
|---|---:|---:|
| expert matmul | 189.1 | 46% |
| attention | 111.5 | 27% |
| **expert disk** | **20.3** | **5%** |
| engram | 0.6 | 0.1% |
| prefill / decode | 144.5 / 252.1 | -- |

99.2 GB of experts read in 5,274 misses against 4,956 hits; 178 of 208 threads busy;
peak RSS 24.8 GB. The docs' own measurement of this engine is 10.4 s of disk against
a 7.6 s matmul on a 16-thread box, where the device dominates. On 208 cores the ratio
inverts. Nothing about the engine changed; the host did.

## The two arms

Medians of 3 cold runs, all token-exact:

| arm | expert disk | rate | whole turn | vs `P` |
|---|---:|---:|---:|---:|
| `P` (1 endpoint) | 20.29 s | 4.89 GB/s | 408.8 s | -- |
| `M3` (2 endpoints) | **17.05 s** | **5.82 GB/s** | **405.1 s** | **-0.9%** |
| `M2` (1 endpoint, RAID0 pair) | 20.51 s | 4.83 GB/s | 407.6 s | -0.3% |
| `D0` (buffered) | 32.10 s | 3.09 GB/s | 418.0 s | +2.3% |
| `MP` (probe-chosen split) | 18.83 s | 5.27 GB/s | 403.1 s | -1.4% |

The disk phase is the quantity that moves, and it is reproducible: `P` 20.44 / 20.29 /
20.27 s, `M3` 17.00 / 17.05 / 17.10 s, `M2` 20.51 / 20.53 / 20.50 s, `D0` 32.10 /
31.06 / 32.29 s, `MP` 18.87 / 18.83 / 17.14 s. The whole turn is a *derived* quantity here, and its differences are
at the edge of what the compute side allows: the expert matmul alone ranges 186.9 to
190.6 s across these same runs, a 2.0% spread, and `M2`'s turn comes out 0.3% *faster*
than `P`'s while its disk is 1.1% *slower*. So read the turn column as "consistent in
sign with the disk column, and no more precise than that"; the disk column is the
measurement.

### The startup probe is noisier than the thing it measures

`MP` leaves the split to the engine's own bandwidth probe, which samples about
150 MB per drive at startup, and it lands *worse* than pinning `1,1`: 18.83 s of disk
against 17.05 s. The probe reported `primary 4.38 GB/s | mirror 4.65 GB/s` and picked
a 48/52 split, which is the right answer; the problem is that it picks a slightly
different one each run, and the resulting disk times move with it (18.87 / 18.83 /
17.14 s across three runs, versus 17.00 / 17.05 / 17.10 s for the pinned split). On a
pair of drives that are actually equal the probe adds variance without adding
information, and the shipped default (unset weights) therefore inherits that. A fixed
`COLI_DISK_WEIGHTS` is the better choice on a known-even pair, which is what the
documentation already recommends but does not explain.

The `MIRROR:` line the engine prints confirms the split is real rather than declared:
`M3` reports `primary 49.20 GB (15702 reads) | mirror1 49.95 GB (15942 reads) — 50% of
expert bytes from the mirrors`, and `M2` reports 100% on the replica.

### O_DIRECT is the better of the two, and still small

32.10 s to 20.29 s of disk is -37%, and the whole turn moves 418.0 s to 408.8 s,
-2.3%. It is worth defaulting (`V41_DIRECT=1`, as V4 does) on this class of device,
because buffered reads are strictly wasted work here: the engine *already* drops each
expert's pages after reading it, so a buffered read copies the bytes into the page
cache, copies them out, and then throws the pages away.

### The split helps across controllers, and not within one filesystem

`M3` (-16% disk) versus `M2` (+1.1% disk, i.e. nothing). Both arms read the same
bytes; the difference is what backs them. `M2`'s two drives are a RAID0 pair behind
one `md0` device, and serving all reads from it is no faster than one `nvme0n1`.
Adding a second *filesystem* to read from is what helped.

## The same A/B with the team sized

The arms above were re-run with `OMP_NUM_THREADS=32` -- the team size that puts the CPU
side in order, now the shipped behaviour via PR #1517. Same container, same 25-token
prompt, same 16 reference ids, cold cache, arms interleaved, 3 runs each, and again
**every run token-exact 16/16**.

| arm | expert disk | rate | whole turn | decode | vs `P` |
|---|---:|---:|---:|---:|---:|
| `P` one NVMe | 16.07 s | 6.17 GB/s | 37.2 s | 12.86 s | -- |
| `P0` split ratio set, no mirror (inert) | 16.10 s | 6.16 GB/s | 38.1 s | 13.0 s | +2% |
| `M3` mirror, second controller | **12.66 s** | **7.83 GB/s** | **34.0 s** | **11.00 s** | disk -21%, decode +17% |
| `MP` mirror, startup probe picks the split | 13.44 s | 7.38 GB/s | 36.3 s | 12.19 s | disk -16%, decode +5% |
| `M2` mirror, both copies behind one RAID0 | 16.36 s | 6.06 GB/s | 37.9 s | 12.96 s | +2%, i.e. nothing |
| `D0` buffered reads | 31.14 s | 3.18 GB/s | 52.7 s | 21.01 s | disk +94%, decode -39% |

Per-run disk times: `P` 15.95 / 16.09 / 16.07 s, `M3` 12.55 / 12.66 / 12.76 s,
`M2` 16.36 / 16.38 / 16.32 s, `MP` 12.81 / 13.44 / 13.72 s,
`D0` 31.14 / 32.61 / 26.28 s (the last `D0` run is an outlier; the median is used
throughout). `P0` exists so the mirror can be compared as one variable: it sets the
same `COLI_DISK_WEIGHTS=1,1` as `M3` with no mirror registered, where the ratio is
inert, so `P0` and `M3` differ in exactly `COLI_MODEL_MIRROR`.

Everything the earlier section said changes. `O_DIRECT` is worth **-48% of the disk
phase and +63% of the decode rate**; the split is worth **-21% and +17%**; and **the
engine now reaches the device's own ceiling** -- 6.17 GB/s against the 6.34 GB/s
`iobench` measures one `nvme0n1` at depth 4, which is what the corrected phase
breakdown means: the disk is 16.1 s of a 37.2 s turn, not 5% of it.

`M3` clearing that ceiling at 7.83 GB/s is the other half of the result: 7.83 is above
what one device can deliver, so the split is genuinely using two. And it is still
**two filesystems on independent controllers** that matter, not two drives -- `M2`
serves every byte from the `md0` RAID0 pair and is indistinguishable from one NVMe.
The mechanism the 208-thread section inferred is confirmed here with the CPU side out
of the way, which is the only reason the 208-thread numbers could not show it.

The probe arm (`MP`) lands between the two: it picks a split that drifts a few percent
run to run (52/52/51% of bytes to the mirror, disk 12.81 / 13.44 / 13.72 s) and gets
part of the pinned split's benefit with more variance. On a pair of drives that are
actually equal, pinning `1,1` is the better move -- the docs already recommend that
without saying why, and the 208-thread runs said the opposite for the wrong reason.

## Why: the device was never the limit

`c/iobench`, 5.6 MB blocks (one weight matrix), O_DIRECT, caches evicted, one thread
per read:

| depth | `nvme0n1` (1 NVMe) | `md0` (RAID0 of 2) | buffered `nvme0n1` |
|---:|---:|---:|---:|
| 1 | 5.45 GB/s | 5.03 GB/s | 2.10 GB/s |
| 4 | **6.34** | **6.19** | -- |
| 8 | 6.03 | 5.80 | 4.99 |
| 16 | 5.70 | 5.63 | -- |
| 32 | 5.55 | 5.39 | 4.55 |

Two things fall out of this, and they are the actual findings of the day.

**One NVMe on this host already reaches the platform ceiling (~6.2 GB/s), and adding
a second drive behind the same filesystem does not raise it.** That is why `M2` did
nothing: it is hypothesis #2's premise, tested directly, and on this box the premise
does not hold -- the drives are not the scarce resource.

**The engine gets 4.89 GB/s from a device that answers at 6.03.** So the arm that
did help, `M3`, helped by giving the reader a second *file* to keep requests in
flight against, not by adding spindle bandwidth. The mirror is a concurrency knob
here, not a capacity knob.

Depth 4 beats depth 8 at the device level, which the engine's `V41_READ_DEPTH`
default of 8 does not follow, and the docs' own knee (8, on a striped PM9A1 pair)
was measured on a different device.

## What this does not show

- **It confirms hypothesis #2, and it shows what the hypothesis was missing.** The
  README's premise -- that independent drives can turn into decode speed -- holds here
  (7.83 GB/s against a 6.34 GB/s single-device ceiling, -21% series, +17% decode), but
  *only* once the CPU side is not the bottleneck. The original phrasing is a
  conditional that reads like an unconditional, and the condition is cheap to state
  and easy to get wrong: it is not enough that the host has several drives, the engine
  has to be able to need them.
- **Not a claim about `O_DIRECT` in general.** The project already documents it as
  drive-dependent (QLC/DRAM-less drives behave differently). One device class was
  measured.
- **`M2`'s null is one machine's null.** A host with a slower platform relative to
  its drives would show the two-drive RAID0 pulling ahead.
- The `MIRROR:` split shares are from the engine's counters, which is the mechanism
  being checked, not an independent instrument.

## Still open, and cheap

The engine reaches 4.89 GB/s where the device answers at 6.03, and depth 4 beats
depth 8 in isolation. `V41_READ_DEPTH` is a fixed default that the docs describe as
"the measured knee"; on this host it is not. A device-aware or tuned default is a
small change with a measurable target, and the same question applies to `SNAP`-side
read scheduling in general.

## Reproduce

The container, as assembled here:

```sh
# hf-mirror.com -- huggingface.co is unreachable from this host, and the two 101 GB
# engram shards exceed the CLI's non-Xet size limit, so they were fetched as 12
# ranged requests each and joined locally.
hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir /mnt/nvme0/DeepSeek-V4.1-Flash
python3 c/tools/prepare_dsv41.py --model /mnt/nvme0/DeepSeek-V4.1-Flash
```

The container was then checked against the upstream file manifest, size for size:
88/88 exact, 510.3 GB in each of the two copies. The mirror arms need the same
container a second time on an independent controller -- 510 GB duplicated for the
duration of the runs, which is precisely the cost `COLI_MODEL_DIRS` exists to avoid.

Two thread configurations are reported. The first set of runs used the engine's
default team (208 logical CPUs) because that is what shipped at the time; the second
set sets `OMP_NUM_THREADS=32`, which is what the CPU side needs on this host and what
PR #1517 now does by default (it lands on 104 physical cores; 32 is faster still here
and is why the override is the documented escape hatch). **The two sets disagree, and
the second one is the one to use** -- the first is kept because the disagreement is
the finding.

One turn, one arm:

```sh
make -C c deepseek_v41
SNAP=/mnt/nvme0/DeepSeek-V4.1-Flash ./c/deepseek_v41 8 ref.json
SNAP=/mnt/nvme0/DeepSeek-V4.1-Flash COLI_MODEL_MIRROR=<copy> COLI_DISK_WEIGHTS=1,1 \
    ./c/deepseek_v41 8 ref.json
# and with the CPU side in order:
OMP_NUM_THREADS=32 SNAP=/mnt/nvme0/DeepSeek-V4.1-Flash COLI_MODEL_MIRROR=<copy> \
    COLI_DISK_WEIGHTS=1,1 ./c/deepseek_v41 8 ref.json
```

`ref.json` carries 25 prompt ids and the 16 reference ids; the engine prints the
generated ids on stdout and the phase breakdown on stderr. The arms interleave
P / M3 / D0 / M2, evicting the container's pages with `posix_fadvise(DONTNEED)` before
every run, 3 runs per arm; `MP` was run separately against the same reference. Raw
stderr for every run is attached to the PR.

The device sweep is `c/iobench`, whose arguments are the file, the block size in MB,
the number of reads, the reader count and the direct flag:

```sh
c/iobench <a-shard> 5.6 48 <depth> 1
```
