"""A decision engine on CUDA against the same engine on the CPU, on its tiny fixture.

    python3 tests/cuda_decide_compare.py laya|gliner [ENV=VALUE ...]

vulkan_decide_compare.py's gates for decide_cuda.h: every case of the fixture's ref.json
through the engine's --records mode on the CPU (COLI_CUDA and COLI_VULKAN unset), then
with COLI_CUDA=1 and the pairs given. The same answer to every question (the decision,
the refusals), every probability within PROB_TOL and every logit within LOGIT_TOL of the
CPU's (relative to the question's largest |logit|, at least 1), and the run's
"[CUDA] <engine> chain: K forwards on the device ..., N on the CPU" with K > 0 and N = 0.
With COLI_CUDA_CHAIN_FAULT=n the device is lost at its n-th frame: forwards on it before,
on the CPU after, the answers still the CPU's. Prints one summary line.
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vulkan_decide_compare import HERE, LOGIT_TOL, PROB_TOL, records_of, run  # noqa: E402

import json  # noqa: E402


def main():
    engine = sys.argv[1]
    if engine not in ("laya", "gliner"):
        raise SystemExit("usage: cuda_decide_compare.py laya|gliner [ENV=VALUE ...]")
    extra = dict(a.split("=", 1) for a in sys.argv[2:])
    fixture = {"laya": "laya_tiny", "gliner": "gliner_decide_tiny"}[engine]
    name = {"laya": "laya", "gliner": "gliner_decide"}[engine]
    cases = json.loads((HERE / fixture / "ref.json").read_text(encoding="utf-8"))["cases"]
    base = {k: v for k, v in os.environ.items() if k not in ("COLI_CUDA", "COLI_VULKAN")}
    cpu_env = dict(base, **{k: v for k, v in extra.items() if not k.startswith("COLI_CUDA")})
    dev_env = dict(base, COLI_CUDA="1", **extra)
    recs = records_of(cases)
    cpu, _ = run(engine, recs, cpu_env)
    dev, err = run(engine, recs, dev_env)
    if len(cpu) != len(cases) or len(dev) != len(cases):
        raise SystemExit(f"FAIL: {len(cpu)} CPU and {len(dev)} CUDA answers for {len(cases)} cases\n{err[-3000:]}")
    worst_p = worst_l = 0.0
    decisions = refusals = 0
    for case, a, b in zip(cases, cpu, dev):
        if "error" in a or "error" in b:
            if a.get("error") != b.get("error"):
                raise SystemExit(f"FAIL: {case['name']}: CPU {a.get('error')!r}, CUDA {b.get('error')!r}")
            refusals += 1
            continue
        for qa, qb in zip(a["answers"], b["answers"]):
            where = f"{case['name']}.{qa['id']}"
            pa, pb = qa["probs"], qb["probs"]
            if max(range(len(pa)), key=lambda i: (pa[i], -i)) != max(range(len(pb)), key=lambda i: (pb[i], -i)):
                raise SystemExit(f"FAIL: {where}: a different decision, CPU {pa} CUDA {pb}")
            dp = max(abs(x - y) for x, y in zip(pa, pb))
            scale = max(1.0, max(abs(x) for x in qa["logits"]))
            dl = max(abs(x - y) for x, y in zip(qa["logits"], qb["logits"])) / scale
            worst_p, worst_l = max(worst_p, dp), max(worst_l, dl)
            if dp > PROB_TOL or dl > LOGIT_TOL:
                raise SystemExit(f"FAIL: {where}: |dp| {dp:.2e}, |dlogit|/scale {dl:.2e}\n"
                                 f"  CPU {qa['logits']}\n  CUDA {qb['logits']}")
            decisions += 1
    fw = re.findall(rf"^\[CUDA\] {name} chain: (\d+) forwards on the device .*?, (\d+) on the CPU", err, re.M)
    if "COLI_CUDA_CHAIN_FAULT" in extra:
        if not re.search(rf"^\[CUDA\] {name}: the device was lost", err, re.M) or not fw or \
                int(fw[-1][0]) <= 0 or int(fw[-1][1]) <= 0:
            raise SystemExit(f"FAIL: the device was not lost in the middle of the run\n{err[-3000:]}")
        chain = f"{fw[-1][0]} forwards on the device and {fw[-1][1]} on the CPU after the loss"
    elif not fw or int(fw[-1][0]) <= 0 or int(fw[-1][1]) != 0:
        raise SystemExit(f"FAIL: the forward did not run on the device\n{err[-3000:]}")
    else:
        chain = f"{fw[-1][0]} forwards on the device"
    print(f"OK {engine} {' '.join(sys.argv[2:])}: {decisions} decisions and {refusals} refusals as on the CPU; "
          f"max |dp| {worst_p:.2e}, max |dlogit| {worst_l:.2e} of the largest; {chain}")


if __name__ == "__main__":
    main()
