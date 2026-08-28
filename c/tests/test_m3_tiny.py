#!/usr/bin/env python3
"""#601 CI gate for the MiniMax-M3 port: run the engine on the converted tiny
M3 container in teacher-forcing mode and require EVERY prefill and decode
position to match the numpy oracle's argmax.

The fixture chain (make m3-tiny-generate) is tools/make_m3tiny.py ->
tools/convert_fp8_to_int4.py --arch m3 -> tools/oracle_m3.py; this driver only
runs the engine, because the engine's REF/TF gate prints its counts but always
exits 0 -- the pass/fail decision lives here.

IDOT=0 on purpose: int8 activation quantization can flip a borderline argmax
(~0.06 logit margin on the tiny fixture), so the gate compares the exact f32
path -- documented in tools/oracle_m3.py. Everything under __main__ so
`unittest discover` can import this file without running the engine.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./colibri")
    ap.add_argument("--snap", default="m3tiny_i8")
    ap.add_argument("--ref", default="ref_m3.json")
    a = ap.parse_args()

    env = dict(os.environ, SNAP=a.snap, REF=a.ref,
               TF="1", TF_DECODE="1", IDOT="0")
    proc = subprocess.run([a.binary, "8"], env=env, text=True,
                          encoding="utf-8", errors="replace",
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=600)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    prefill = re.search(r"PREFILL \(teacher-forcing\) C vs oracle: (\d+)/(\d+)",
                        proc.stdout)
    decode = re.search(r"DECODE \(incremental\) C vs oracle: (\d+)/(\d+)",
                       proc.stdout)
    problems = []
    if proc.returncode:
        problems.append(f"engine exited {proc.returncode}")
    for label, match in (("prefill", prefill), ("decode", decode)):
        if not match:
            problems.append(f"{label} gate line missing from engine output")
        elif int(match.group(2)) == 0 or match.group(1) != match.group(2):
            problems.append(f"{label} {match.group(1)}/{match.group(2)}")
    if problems:
        print(f"m3-tiny-check: FAIL ({'; '.join(problems)})")
        return 1
    print(f"m3-tiny-check: OK (prefill {prefill.group(0).split(': ')[1]}, "
          f"decode {decode.group(1)}/{decode.group(2)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
