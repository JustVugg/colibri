#!/usr/bin/env python3
"""Sample RSS of the serving stack's processes until killed.

Usage: python3 c13_mem_sampler.py <out.jsonl> <pid> [<pid> ...]
Writes {"ts": epoch, "pids": {"<pid>": rss_kb}} every 0.5 s, flushed per line.
Run in the background while a request driver records wall-clock windows; the
join script computes per-window peak/baseline RSS from these lines.
"""
import json
import subprocess
import sys
import time

out_path, pids = sys.argv[1], sys.argv[2:]
subprocess.run(["ps", "-o", "pid=", "-p", ",".join(pids)], check=True,
               stdout=subprocess.DEVNULL)
with open(out_path, "w") as f:
    while True:
        now = time.time()
        sample = {}
        for pid in pids:
            r = subprocess.run(["ps", "-o", "rss=", "-p", pid],
                               capture_output=True, text=True)
            try:
                sample[pid] = int(r.stdout.strip())
            except ValueError:
                sample[pid] = None       # process gone
        f.write(json.dumps({"ts": now, "pids": sample}) + "\n")
        f.flush()
        time.sleep(0.5)
