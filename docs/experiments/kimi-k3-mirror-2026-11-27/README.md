# kimi_k3: routed experts from COLI_MODEL_MIRROR copies (2026-11-27)

The bench for `c/mirror_policy.h` and its use in `c/kimi_k3.c` (#812). Kimi K3
keeps all 896 experts of a layer in one shard, so with one drive only the drive
that holds the shard reads. With copies of the container on more drives,
`kimi_k3` spreads the reads of each batch over the drives.

| arm | engine | COLI_MODEL_MIRROR | decode tok/s (3 runs) | median | prefill tok/s | GB read nvme0 / nvme2 / nvme3 |
|---|---|---|---|---|---|---|
| base-off | v2.2.0 (`e479104a`) | unset | 0.256, 0.256, 0.257 | 0.256 | 0.83 | 1946 / 0 / 0 |
| pr-off | this branch (`523d0e3b`) | unset | 0.256, 0.256, 0.256 | 0.256 | 0.83 | 1946 / 0 / 0 |
| pr-mirror | this branch (`523d0e3b`) | copies on nvme2 and nvme3 | 0.429, 0.430, 0.430 | 0.430 | 1.35 | 687 / 630 / 630 |

All 9 runs generated the same 64 tokens. The primary is on a CPU root port, the
two copies are on chipset root ports that share one uplink to the CPU. With
`COLI_DISK_WEIGHTS=1,1,1`, the decode reads split evenly (35889, 35880 and 35885
reads in the first run of pr-mirror). No drive throttled during the runs: the
thermal management counters in `nvme smart-log` did not change, and the hottest
drive peaked at 66 C.

The kernel names of the drives do not follow the mount order on this host.
`DRIVES` lists `nvme0n1 nvme2n1 nvme3n1 nvme1n1`, the drives of nvme0, nvme1,
nvme2 and nvme3, so the bytes in `summary.md` and in the raw files follow the
mount order.

## Files

- `manifest.json`: pr-off against pr-mirror. The changed variable is
  `COLI_MODEL_MIRROR`.
- `engine-manifest.json`: base-off against pr-off, both on one drive. The changed
  variable is the engine.
- `base-off-raw.txt`, `pr-off-raw.txt`, `pr-mirror-raw.txt`: for each run, the
  engine's summary lines, the bytes each drive read (`/proc/diskstats`) and the
  SHA-256 of the generated text. The local model directories are replaced with
  `<container on nvme0>`, `<copy on nvme2>` and `<copy on nvme3>`.
- `summary.md`: the table above as `summarize.py` writes it.
- `prompt.txt`, `common.env`: the prompt and the environment of every run.
- `bench.sh`, `summarize.py`: the scripts that made the runs and these files.

## Run it again

`bench.sh` drops the page cache before each run, so it needs root. `DRIVES`
names the block devices whose bytes it counts, `RUNS` and `NGEN` set the rounds
and the tokens. A run on one drive keeps that drive busy for minutes. If the
drive throttles, the arms without a copy slow down, so check the drive
temperatures during the runs.

```sh
make -C c kimi_k3 ARCH=native        # in a checkout of each engine
sudo MODEL=/nvme0/kimi DRIVES='nvme0n1 nvme1n1 nvme2n1 nvme3n1' ./bench.sh out \
    base-off@/src/v2.2.0/c/kimi_k3 \
    pr-off@/src/branch/c/kimi_k3 \
    'pr-mirror@/src/branch/c/kimi_k3@COLI_MODEL_MIRROR=/nvme2/kimi;/nvme3/kimi'
python3 summarize.py out result --base-commit <sha> --commit <sha> --mirrors '<value>' \
    --model '<text>' --command '<text>' --cpu '<text>' --ram '<text>' --storage '<text>' --os '<text>'
python3 c/experiment_manifest.py result/manifest.json
```
