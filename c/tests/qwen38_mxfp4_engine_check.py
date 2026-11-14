"""qwen38 on the experts-mxfp4/ sidecar (Q38_EXPERT_LOWBIT=mxfp4), the engine end to end.

Run by make qwen38-tiny-mxfp4-check after the engine reproduced ref.json on the
sidecar through every expert path. It checks:
  * the last logits of the MXFP4 run against the FP8 run's on the same fixture
    (its FP8 shards hold the MXFP4 values exactly, so only the kernels' float
    order separates them): cosine >= 0.99999;
  * the refusals, each in its own engine process on a copy of the fixture: an
    index.json for another geometry, an incomplete conversion, a record whose
    tensors are not the layout's, no sidecar at all, and an unknown
    Q38_EXPERT_LOWBIT;
  * under COLI_CUDA=1 the engine says the expert tier does not take MXFP4 yet
    and still reproduces ref.json on the CPU.

  python3 tests/qwen38_mxfp4_engine_check.py --model ./qwen38_tiny_mxfp4 --engine ./qwen38 \\
      --mxfp4 logits/mxfp4.f32 --fp8 logits/fp8.f32
"""
import argparse
import array
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SIDECAR = "experts-mxfp4"


def floats(path):
    a = array.array("f")
    a.frombytes(Path(path).read_bytes())
    return a


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-30)


def run(engine, model, **env):
    full = dict(os.environ, OMP_NUM_THREADS="2", SNAP=str(model))
    for key in ("Q38_EXPERT_LOWBIT", "Q38_EXPERT_INT4", "COLI_CUDA"):
        full.pop(key, None)
    full.update(env)
    p = subprocess.run([str(engine), "2", "8", str(model / "ref.json")], env=full,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--mxfp4", required=True, help="the MXFP4 run's last logits (DUMP)")
    ap.add_argument("--fp8", required=True, help="the FP8 run's last logits (DUMP)")
    a = ap.parse_args()
    model, engine = a.model.resolve(), a.engine.resolve()
    failures = 0

    def check(ok, what, detail=""):
        nonlocal failures
        print(f"  {'ok  ' if ok else 'FAIL'} {what}")
        if not ok and detail:
            print(detail[-2000:])
        failures += not ok

    mx, f8 = floats(a.mxfp4), floats(a.fp8)
    cos = cosine(mx, f8) if len(mx) == len(f8) and len(mx) else 0.0
    check(cos >= 0.99999, f"last logits of the MXFP4 run against the FP8 run's: cosine {cos:.8f} >= 0.99999")

    rc, out, err = run(engine, model, Q38_EXPERT_LOWBIT="mxfp4", COLI_CUDA="1")
    check(rc == 0 and "Matching tokens: 8/8" in out and "the mxfp4 experts stay on the CPU" in err,
          "COLI_CUDA=1: the tier declines the MXFP4 experts, the CPU reproduces ref.json", err)
    rc, _, err = run(engine, model, Q38_EXPERT_LOWBIT="mxfp5")
    check(rc != 0 and "mxfp4" in err, "an unknown Q38_EXPERT_LOWBIT is refused, naming mxfp4", err)

    tmp = Path(tempfile.mkdtemp())
    try:
        def copy():
            dst = tmp / "snap"
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(model, dst, ignore=shutil.ignore_patterns("quark", "logits"))
            return dst

        def edit_index(snap, **fields):
            path = snap / SIDECAR / "index.json"
            index = json.loads(path.read_text(encoding="utf-8"))
            index.update(fields)
            path.write_text(json.dumps(index), encoding="utf-8")
            return index

        snap = copy()
        index = json.loads((snap / SIDECAR / "index.json").read_text(encoding="utf-8"))
        edit_index(snap, hidden_size=index["hidden_size"] + 32)
        rc, _, err = run(engine, snap, Q38_EXPERT_LOWBIT="mxfp4")
        check(rc != 0 and "another geometry" in err, "an index.json for another geometry is refused", err)

        snap = copy()
        edit_index(snap, complete=False, done=[0])
        rc, _, err = run(engine, snap, Q38_EXPERT_LOWBIT="mxfp4")
        check(rc != 0 and "conversion incomplete" in err, "an incomplete conversion is refused", err)

        snap = copy()
        layer = snap / SIDECAR / "layer-001.safetensors"
        data = layer.read_bytes()
        rows, cols = index["moe_intermediate_size"], -(-index["hidden_size"] // 2)
        name = f'"{index["prefix"]}.layers.1.mlp.experts.0.gate_proj.weight":{{"dtype":"U8","shape":'
        old, new = f"{name}[{rows},{cols}]".encode(), f"{name}[{cols},{rows}]".encode()
        check(len(old) == len(new) and data.count(old) == 1, "the fixture's layer file holds the tensor to edit")
        layer.write_bytes(data.replace(old, new))
        rc, _, err = run(engine, snap, Q38_EXPERT_LOWBIT="mxfp4")
        check(rc != 0 and "layer 1 expert 0 is missing or is not one mxfp4 record" in err,
              "a record whose tensors are not the layout's is refused", err)

        snap = copy()
        shutil.rmtree(snap / SIDECAR)
        rc, _, err = run(engine, snap, Q38_EXPERT_LOWBIT="mxfp4")
        check(rc != 0 and "convert_qwen38_experts_mxfp4.py" in err, "no sidecar: refused, naming the converter", err)
        rc, out, err = run(engine, snap)
        check(rc == 0 and "Matching tokens: 8/8" in out, "no sidecar and no Q38_EXPERT_LOWBIT: the FP8 experts", err)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("qwen38_mxfp4_engine_check: " + ("ok" if not failures else f"{failures} failure(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
