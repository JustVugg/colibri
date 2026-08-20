#!/usr/bin/env python3
"""Sweep iobench across block sizes and thread counts; print a datapoint block.

An expert's on-disk size is fixed at conversion time by the model's geometry --
``3 * d_model * moe_intermediate_size * bits/8`` for a SwiGLU expert -- and the engine
reads one expert per request. So the architecture chooses where on the drive's
block-size curve the whole streaming path operates, and that curve is not flat: on a
consumer NVMe it is steep below ~1 MB and saturated above it.

This walks that curve with the engine's own microbenchmark so the numbers are
comparable with every other iobench figure in the tracker.

Stdlib only, like tools/datapoint.py.

Usage:
  python tools/blocksweep.py --shard /path/to/out-00069.safetensors
  python tools/blocksweep.py --shard <file> --threads 1 4 8 --reads 128 --repeats 5

The default block list spans the expert sizes real MoE containers produce, from a
fine-grained 128-wide expert up to GLM-5.2's ~19 MB.
"""

import argparse
import os
import re
import statistics
import subprocess
import sys

GBS = re.compile(r"->\s*([\d.]+)\s*GB/s")

# expert bytes for a SwiGLU expert at int4, in MB, for a few (d_model, ff) shapes
DEFAULT_BLOCKS = [0.1875, 0.375, 0.75, 1.5, 3.0, 6.0, 12.0, 19.0]


def run(iobench, shard, blk, reads, threads, direct):
    p = subprocess.run([iobench, shard, str(blk), str(reads), str(threads), str(direct)],
                       capture_output=True, text=True)
    m = GBS.search(p.stdout)
    if not m:
        sys.stderr.write(f"  ! no GB/s in output for blk={blk} t={threads}: "
                         f"{(p.stdout + p.stderr)[:160]!r}\n")
        return None
    return float(m.group(1))


def human(mb):
    return f"{mb*1024:.0f} KB" if mb < 1 else (f"{mb:.0f} MB" if mb == int(mb)
                                               else f"{mb:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iobench", default="./iobench", help="iobench binary")
    ap.add_argument("--shard", required=True,
                    help="a large file to read; a container shard is ideal")
    ap.add_argument("--blocks", type=float, nargs="+", default=DEFAULT_BLOCKS,
                    help="block sizes in MB (fractions allowed)")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--reads", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--buffered", action="store_true",
                    help="also measure buffered reads (default: O_DIRECT only)")
    a = ap.parse_args()

    if not os.path.exists(a.shard):
        sys.exit(f"no such file: {a.shard}")
    if not (os.path.exists(a.iobench) or os.path.exists(a.iobench + ".exe")):
        sys.exit(f"no iobench at {a.iobench} -- build it first:\n"
                 f"  gcc -D_FILE_OFFSET_BITS=64 -O2 -fopenmp iobench.c -o iobench")

    size_mb = os.path.getsize(a.shard) / (1 << 20)
    blocks = [b for b in a.blocks if b * 2 <= size_mb]
    if len(blocks) < len(a.blocks):
        dropped = [human(b) for b in a.blocks if b not in blocks]
        sys.stderr.write(f"note: {a.shard} is {size_mb:.0f} MB; iobench needs a file at "
                         f"least 2x the block, so dropping {', '.join(dropped)}\n")

    modes = [1, 0] if a.buffered else [1]
    rows = []
    for blk in blocks:
        row = {"blk": blk}
        for direct in modes:
            for t in a.threads:
                vals = [v for v in (run(a.iobench, a.shard, blk, a.reads, t, direct)
                                    for _ in range(a.repeats)) if v is not None]
                if vals:
                    row[(direct, t)] = (statistics.median(vals), min(vals), max(vals))
                print(f"  {human(blk):>9}  {'O_DIRECT' if direct else 'buffered':>8}  "
                      f"{t:>2}T  {row.get((direct,t),(float('nan'),))[0]:.2f} GB/s",
                      flush=True)
        rows.append(row)

    print()
    print(f"**Drive block-size curve** — `{os.path.basename(a.shard)}`, "
          f"{a.reads} reads, median of {a.repeats}, GB/s")
    print()
    for direct in modes:
        if len(modes) > 1:
            print(f"*{'O_DIRECT' if direct else 'buffered'}*")
            print()
        print("| expert size | " + " | ".join(f"{t} thread{'s' if t > 1 else ''}"
                                              for t in a.threads) + " |")
        print("|---" * (len(a.threads) + 1) + "|")
        for r in rows:
            cells = []
            for t in a.threads:
                v = r.get((direct, t))
                cells.append(f"{v[0]:.2f} ± {(v[2]-v[1])/2:.2f}" if v else "—")
            print(f"| {human(r['blk'])} | " + " | ".join(cells) + " |")
        print()
    print("Please also state: drive model, bus (PCIe 3/4/5), DRAM-less/HMB or not, "
          "filesystem, and OS.")


if __name__ == "__main__":
    main()
