"""OLMoE's CUDA chain (olmoe_cuda_chain.h) against the CPU, on the tiny fixture's oracle.

    python3 tests/cuda_olmoe_compare.py [ENV=VALUE ...]

Needs a CUDA build of olmoe and the fixture (tools/make_olmoe_tiny.py --output olmoe_tiny,
then tools/convert_olmoe_merged.py --model olmoe_tiny --out olmoe_tiny_c, as
tests/vulkan_engines.sh makes it). Runs the oracle (olmoe_tiny/ref_olmoe.json) on the CPU and
with COLI_CUDA=1 plus the pairs given, each with DUMP: the same tokens as the reference, the
last logits within LOGIT_TOL of the CPU's, and the chain's "[CUDA] olmoe chain: F forwards"
line with F > 0. With COLI_CUDA_CHAIN_FAULT=n the device is lost at its n-th frame: the CPU
rebuilds the KV the device held from the token ids, and the run ends with the reference's
tokens. Prints one summary line.
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EXE = ".exe" if os.name == "nt" else ""
LOGIT_TOL = 1e-5


def run(env, dump):
    e = dict(os.environ, **env, DUMP=str(dump), SNAP=str(HERE / "olmoe_tiny_c"),
             OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
    e.pop("COLI_VULKAN", None)
    r = subprocess.run([str(HERE / ("olmoe" + EXE)), "8", "8", str(HERE / "olmoe_tiny" / "ref_olmoe.json")],
                       capture_output=True, text=True, env=e, timeout=600)
    if r.returncode:
        raise SystemExit(f"FAIL: olmoe exited {r.returncode}\n{r.stderr[-3000:]}")
    match = re.search(r"Matching tokens: (\d+)/(\d+)", r.stdout)
    if not match:
        raise SystemExit(f"FAIL: no token count in the output\n{r.stdout[-2000:]}")
    data = dump.read_bytes()
    return (int(match.group(1)), int(match.group(2))), struct.unpack(f"{len(data) // 4}f", data), r.stderr


def main():
    extra = dict(a.split("=", 1) for a in sys.argv[1:])
    base = {k: v for k, v in extra.items() if not k.startswith("COLI_CUDA")}
    with tempfile.TemporaryDirectory() as tmp:
        (mc, nc), cpu, _ = run(dict(base, COLI_CUDA="0"), Path(tmp) / "cpu.f32")
        (md, nd), dev, err = run(dict(extra, COLI_CUDA="1"), Path(tmp) / "dev.f32")
    if mc != nc or md != nd:
        raise SystemExit(f"FAIL: tokens: CPU {mc}/{nc}, CUDA {md}/{nd} of the reference's\n{err[-3000:]}")
    if "[CUDA] olmoe chain: every layer" not in err:
        raise SystemExit(f"FAIL: the chain did not come up\n{err[-3000:]}")
    d = max(abs(p - q) for p, q in zip(cpu, dev))
    if len(cpu) != len(dev) or d > LOGIT_TOL:
        raise SystemExit(f"FAIL: the last logits differ by {d:.2e} (bound {LOGIT_TOL})")
    fw = re.findall(r"^\[CUDA\] olmoe chain: (\d+) forwards", err, re.M)
    if "COLI_CUDA_CHAIN_FAULT" in extra:
        if "the device was lost at position" not in err:
            raise SystemExit(f"FAIL: the device was not lost\n{err[-3000:]}")
        rebuilt = re.search(r"rebuilding the KV of positions (\d+)\.\.(\d+)", err)
        chain = (f"the device lost at frame {extra['COLI_CUDA_CHAIN_FAULT']}, "
                 + (f"positions {rebuilt.group(1)}..{rebuilt.group(2)} rebuilt on the CPU" if rebuilt else "nothing to rebuild"))
    elif not fw or int(fw[-1]) <= 0:
        raise SystemExit(f"FAIL: no forward ran on the device\n{err[-3000:]}")
    else:
        chain = f"{fw[-1]} forwards on the device"
    print(f"OK olmoe {' '.join(sys.argv[1:])}: {md}/{nd} tokens as the reference, the last logits within {d:.1e} "
          f"of the CPU's; {chain}")


if __name__ == "__main__":
    main()
