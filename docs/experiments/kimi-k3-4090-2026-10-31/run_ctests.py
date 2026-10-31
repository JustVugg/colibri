"""run_ctests.py CDIR OUT.json [EXCLUDE...] -- every built tests/test_*.exe of CDIR, run from CDIR
(as tools/run_tests.py does), 8 at a time, 600 s each; their exit codes and last lines."""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path

cdir = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
skip = set(sys.argv[3:])
exes = sorted(p for p in (cdir / "tests").glob("test_*.exe") if p.stem not in skip)


def run(p):
    try:
        r = subprocess.run([str(p)], cwd=cdir, capture_output=True, timeout=600)
        tail = (r.stdout + r.stderr).decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
        return p.stem, r.returncode, tail[0][:200]
    except subprocess.TimeoutExpired:
        return p.stem, "timeout", ""


res = {}
with cf.ThreadPoolExecutor(8) as ex:
    for name, rc, tail in ex.map(run, exes):
        res[name] = {"rc": rc, "tail": tail}
out.write_text(json.dumps(res, indent=1))
bad = {k: v for k, v in res.items() if v["rc"] != 0}
print(f"{len(res)} test binaries, {len(bad)} failed")
for k, v in sorted(bad.items()):
    print(f"  {k}: rc={v['rc']} {v['tail']}")
