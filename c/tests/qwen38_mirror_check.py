"""qwen38: an expert sidecar read from COLI_MODEL_MIRROR copies, split by COLI_DISK_WEIGHTS.

Runs the engine on a converted tiny fixture with copies of its sidecar in
temporary mirror directories, and checks:
  * the tokens and the last logits are the bits of the run without a mirror,
    with one mirror (weights unset and 1,3) and with two (1,1,1), at caps 1
    and 2, through the parallel and the serial expert reads;
  * every replica serves records (the closing "[qwen38 MIRROR] reads:" line);
  * a copy whose layer file differs from the primary's (one header byte) keeps
    that file on the primary; a directory without the sidecar is skipped;
  * COLI_DISK_WEIGHTS with the wrong count, a zero or junk is refused;
  * without the sidecar (Q38_EXPERT_INT4=0) no mirror is opened.
--lowbit NAME runs the same checks on a low-bit sidecar (Q38_EXPERT_LOWBIT=NAME,
experts-NAME/), as make qwen38-tiny-mxfp4-check does for mxfp4.

  python3 tests/qwen38_mirror_check.py --model ./qwen38_tiny_int4 --engine ./qwen38
  python3 tests/qwen38_mirror_check.py --model ./qwen38_tiny_mxfp4 --engine ./qwen38 --lowbit mxfp4 --ref ref.json
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SIDECAR = "experts-int4g64"
ON = {"Q38_EXPERT_INT4": "1"}


def run(engine, model, ref, cap, dump, **env):
    full = dict(os.environ, OMP_NUM_THREADS="2", SNAP=str(model), DUMP=str(dump), **env)
    for key in ("COLI_MODEL_MIRROR", "COLI_DISK_WEIGHTS", "Q38_EXPERT_LOWBIT"):
        if key not in env:
            full.pop(key, None)
    p = subprocess.run([str(engine), str(cap), "8", str(ref)], env=full,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def reads(stderr):
    """[reads per replica] from the closing line, or None."""
    line = re.search(r"^\[qwen38 MIRROR\] reads:(.*)$", stderr, re.M)
    return [int(n) for n in re.findall(r"(?:primary|mirror\d+) (\d+) \(", line.group(1))] if line else None


def main():
    global SIDECAR, ON
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path, help=f"a snapshot with {SIDECAR}/ (qwen38_tiny_int4)")
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--ref", default="ref_int4.json", help="the reference inside --model")
    ap.add_argument("--lowbit", help="a low-bit sidecar instead: Q38_EXPERT_LOWBIT=NAME, experts-NAME/")
    a = ap.parse_args()
    if a.lowbit:
        SIDECAR, ON = f"experts-{a.lowbit}", {"Q38_EXPERT_LOWBIT": a.lowbit}
    model, engine, ref = a.model.resolve(), a.engine.resolve(), a.model.resolve() / a.ref
    failures = 0

    def check(ok, what):
        nonlocal failures
        print(f"  {'ok  ' if ok else 'FAIL'} {what}")
        failures += not ok

    tmp = Path(tempfile.mkdtemp())
    try:
        m1, m2, m3, empty = tmp / "m1", tmp / "m2", tmp / "m3", tmp / "empty"
        for m in (m1, m2, m3):
            shutil.copytree(model / SIDECAR, m / SIDECAR)
        empty.mkdir()
        bad = m3 / SIDECAR / "layer-001.safetensors"   # one header byte (the padding) differs
        data = bytearray(bad.read_bytes())
        data[8 + int.from_bytes(data[:8], "little") - 1] ^= 0x01
        bad.write_bytes(bytes(data))
        cases = [
            ("one mirror, even split", {"COLI_MODEL_MIRROR": str(m1)}, 2),
            ("one mirror, COLI_DISK_WEIGHTS=1,3", {"COLI_MODEL_MIRROR": str(m1), "COLI_DISK_WEIGHTS": "1,3"}, 2),
            ("two mirrors, COLI_DISK_WEIGHTS=1,1,1",
             {"COLI_MODEL_MIRROR": f"{m1};{m2}", "COLI_DISK_WEIGHTS": "1,1,1"}, 3),
            ("serial reads, two mirrors", {"COLI_MODEL_MIRROR": f"{m1},{m2}", "Q38_EXPERT_PARALLEL_READS": "0"}, 3),
        ]
        for cap in (1, 2):
            base = tmp / f"base-{cap}.f32"
            rc, out, err = run(engine, model, ref, cap, base, **ON)
            if rc or "Matching tokens: 8/8" not in out:
                print(err[-2000:])
                check(False, f"cap {cap}: the run without a mirror")
                continue
            for what, env, n in cases:
                dump = tmp / f"mirror-{cap}.f32"
                rc, out2, err = run(engine, model, ref, cap, dump, **ON, **env)
                counts = reads(err)
                same = rc == 0 and out2 == out and dump.read_bytes() == base.read_bytes()
                check(same and counts is not None and len(counts) == n and all(counts),
                      f"cap {cap}, {what}: same tokens and logits, reads per replica {counts}")
                if not same:
                    print(err[-2000:])
        dump = tmp / "skip.f32"
        rc, out, err = run(engine, model, ref, 1, dump, **ON,
                           COLI_MODEL_MIRROR=f"{m3};{empty}")
        check(rc == 0 and "header differs" in err and "no file equal to the primary's -- skipped" in err
              and reads(err) is not None and len(reads(err)) == 2,
              "a differing file and an empty directory are skipped, the rest of the copy serves")
        for spec in ("1", "1,1,1", "1,0", "1,x", "2,1,"):
            rc, _, err = run(engine, model, ref, 1, dump, **ON,
                             COLI_MODEL_MIRROR=str(m1), COLI_DISK_WEIGHTS=spec)
            check(rc != 0 and "COLI_DISK_WEIGHTS" in err, f"COLI_DISK_WEIGHTS={spec} with one mirror is refused")
        rc, out, err = run(engine, model, model / "ref.json", 1, dump, Q38_EXPERT_INT4="0", COLI_MODEL_MIRROR=str(m1))
        check(rc == 0 and "[qwen38 MIRROR]" not in err, "without the sidecar no mirror is opened")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("qwen38_mirror_check: " + ("ok" if not failures else f"{failures} failure(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
