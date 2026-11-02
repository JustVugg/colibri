"""A decision engine on Vulkan against the same engine on the CPU, on its tiny fixture.

    python3 tests/vulkan_decide_compare.py laya|gliner|clef [ENV=VALUE ...]

Runs every case of the fixture's ref.json (the cases the tiny oracle tests read) through
the engine's --records mode twice: once with the environment as it is minus COLI_VULKAN
plus the pairs given that are not the device's own (COLI_VK_*), the CPU; once with
COLI_VULKAN=1 and every pair given. Gates:
  - the same answer to every question: the same decision (argmax), the same refusals;
  - every probability within PROB_TOL and every logit within LOGIT_TOL of the CPU's
    (relative to the largest |logit| of the question, at least 1): f32 activations on
    both, so the two differ only in the order of the sums (tests/vulkan_engines.sh decide
    records the measured worst case);
  - the Vulkan run's "[VK] <engine>: N matmuls on the GPU" line has N > 0, and with
    COLI_VK_CHAIN=1 its "chain: K forwards on the device" has K > 0 and none on the CPU
    (laya and gliner; Clef's chain is qwen36's, whose own line counts its forwards).
Prints one summary line with the worst differences.

    python3 tests/vulkan_decide_compare.py --off laya|gliner|clef [ENV=VALUE ...]

The device opened and every Vulkan path of the engine turned off by the pairs given
(COLI_VK_CHAIN=0 COLI_VK_DENSE=0): the answers must be the CPU run's byte for byte (the
elapsed time aside) and no matmul may run on the device.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from openai_server import systemone_decision_record  # noqa: E402

PROB_TOL = 1e-5
LOGIT_TOL = 1e-4
EXE = ".exe" if os.name == "nt" else ""


def records_of(cases, form=None):
    out = []
    for case in cases:
        rec = systemone_decision_record({"state": case["state"], "questions": case["questions"]},
                                        *(() if form is None else (form,)))
        out.append({"payload": json.dumps(rec, ensure_ascii=False)})
    return out


def run(engine, records, env, max_length=None):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "records.json"
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        e = dict(env, OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "2"), COLI_NO_OMP_TUNE="1")
        if engine == "laya":
            argv = [str(HERE / ("laya" + EXE)), "--model", str(HERE / "laya_tiny"), "--records", str(path)]
        elif engine == "gliner":
            ref = json.loads((HERE / "gliner_decide_tiny" / "ref.json").read_text(encoding="utf-8"))
            e["COLI_GLINER_MAX_LEN"] = str(ref["max_len"])
            argv = [str(HERE / ("gliner_decide" + EXE)), "--model", str(HERE / "gliner_decide_tiny"),
                    "--records", str(path)]
        else:
            e.update(SNAP=str(HERE / "clef_tiny_c"), CLEF_RECORDS=str(path),
                     COLI_DENSE_I8=env.get("COLI_DENSE_I8", "0"))
            if max_length:
                e["COLI_CLEF_MAX_LEN"] = str(max_length)
            argv = [str(HERE / ("qwen36" + EXE)), "8", "8"]
        r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", env=e, timeout=1800)
        if r.returncode:
            raise SystemExit(f"FAIL: {engine} exited {r.returncode}\n{r.stderr[-3000:]}")
        return [json.loads(line) for line in r.stdout.splitlines()], r.stderr


def strip_ms(out):
    return [re.sub(r'"[a-z_]*ms": ?[0-9.eE+-]+', '"ms":0', json.dumps(o, sort_keys=True)) for o in out]


def main():
    off = sys.argv[1] == "--off"
    args = sys.argv[2:] if off else sys.argv[1:]
    engine = args[0]
    extra = dict(a.split("=", 1) for a in args[1:])
    fixture = {"laya": "laya_tiny", "gliner": "gliner_decide_tiny", "clef": "clef_tiny"}[engine]
    ref = json.loads((HERE / fixture / "ref.json").read_text(encoding="utf-8"))
    groups = {}
    for case in ref["cases"]:
        groups.setdefault(case.get("max_length") if engine == "clef" else None, []).append(case)
    base = {k: v for k, v in os.environ.items() if k != "COLI_VULKAN"}
    vk_env = dict(base, COLI_VULKAN="1", **extra)
    # the CPU arm takes every pair but the device's own switches: the same weights, and
    # the same activation arithmetic where the engine has a choice (COLI_DENSE_IDOT=0)
    cpu_env = dict(base, **{k: v for k, v in extra.items() if not k.startswith("COLI_VK")})
    name = {"laya": "laya", "gliner": "gliner_decide", "clef": "qwen36"}[engine]
    worst_p = worst_l = 0.0
    decisions = refusals = 0
    logs = []
    for max_length, cases in groups.items():
        recs = records_of(cases, "raw" if engine == "clef" else None)
        cpu, _ = run(engine, recs, cpu_env, max_length)
        vk, err = run(engine, recs, vk_env, max_length)
        logs.append(err)
        if off:
            if strip_ms(cpu) != strip_ms(vk):
                raise SystemExit(f"FAIL: with every Vulkan path off the answers differ from the CPU's\n{err[-3000:]}")
            decisions += sum(len(o.get("answers", [])) for o in vk)
            continue
        if len(cpu) != len(cases) or len(vk) != len(cases):
            raise SystemExit(f"FAIL: {len(cpu)} CPU and {len(vk)} Vulkan answers for {len(cases)} cases\n{err[-3000:]}")
        for case, a, b in zip(cases, cpu, vk):
            if "error" in a or "error" in b:
                if a.get("error") != b.get("error"):
                    raise SystemExit(f"FAIL: {case['name']}: CPU {a.get('error')!r}, Vulkan {b.get('error')!r}")
                refusals += 1
                continue
            for qa, qb in zip(a["answers"], b["answers"]):
                where = f"{case['name']}.{qa['id']}"
                pa, pb = qa["probs"], qb["probs"]
                if max(range(len(pa)), key=lambda i: (pa[i], -i)) != max(range(len(pb)), key=lambda i: (pb[i], -i)):
                    raise SystemExit(f"FAIL: {where}: a different decision, CPU {pa} Vulkan {pb}")
                dp = max(abs(x - y) for x, y in zip(pa, pb))
                scale = max(1.0, max(abs(x) for x in qa["logits"]))
                dl = max(abs(x - y) for x, y in zip(qa["logits"], qb["logits"])) / scale
                worst_p, worst_l = max(worst_p, dp), max(worst_l, dl)
                if dp > PROB_TOL or dl > LOGIT_TOL:
                    raise SystemExit(f"FAIL: {where}: |dp| {dp:.2e}, |dlogit|/scale {dl:.2e}\n"
                                     f"  CPU {qa['logits']}\n  Vulkan {qb['logits']}")
                decisions += 1
    err = "\n".join(logs)
    counts = [int(n) for n in re.findall(rf"^\[VK\] {name}: (\d+) matmuls on the GPU", err, re.M)]
    if off:
        if not re.search(rf"^\[VK\] {name}: device ready", err, re.M) or any(counts):
            raise SystemExit(f"FAIL: the device did not open, or a matmul ran on it\n{err[-3000:]}")
        print(f"OK {engine} {' '.join(args[1:])}: the device open and every path off, {decisions} answers "
              f"byte for byte the CPU's, no matmul on the device")
        return
    chain = ""
    if engine == "clef" and extra.get("COLI_VK_CHAIN", "") not in ("", "0"):
        # qwen36's chain records its matrices in its frames: its line counts the forwards
        fw = [int(n) for n in re.findall(r"^\[VK\] qwen36 chain: (\d+) forwards", err, re.M)]
        if "COLI_VK_CHAIN_FAULT" in extra:
            # A failed prompt can have completed rows without a completed forward.
            # Require the injected submit failure in every process, and prove that
            # the discarded host weights were reloaded before checking its answers.
            for log in logs:
                if "queue submit (COLI_VK_CHAIN_FAULT) failed" not in log:
                    raise SystemExit("FAIL: Clef did not reach the injected device loss")
                if extra.get("COLI_VK_DENSE_HOST") == "0":
                    restored = re.search(r"dense weights at exit: (\d+) matrices.*?, (\d+) read back", log)
                    if not restored or min(map(int, restored.groups())) <= 0:
                        raise SystemExit("FAIL: Clef did not drop and restore its host weights")
            chain = ", failed prompt replayed on the CPU after the injected device loss"
        elif not fw or fw[-1] <= 0:
            raise SystemExit(f"FAIL: qwen36's chain did not run\n{err[-3000:]}")
        else:
            chain = f", {fw[-1]} chain forwards"
            layers = json.loads((HERE / "clef_tiny_c" / "qwen36_meta.json").read_text())["n_layers"]
            # A dense backbone must not put a whole model in one submission:
            # on the real 27B checkpoint that exceeded the driver watchdog.
            # Compare successive reports to exclude one-time allocation frames.
            for log in logs:
                snapshots = [tuple(map(int, x)) for x in re.findall(
                    r"^\[VK\] qwen36 chain: (\d+) forwards, (\d+) frames", log, re.M)]
                for (f0, s0), (f1, s1) in zip(snapshots, snapshots[1:]):
                    if s1 - s0 < (f1 - f0) * layers:
                        raise SystemExit("FAIL: Clef accumulated multiple dense layers in one submission")
        counts = counts or [0]
    elif not counts or min(counts) <= 0:
        raise SystemExit(f"FAIL: no matmul ran on the device\n{err[-3000:]}")
    if engine == "clef":
        # Import follows the actual device policy: default on shared-memory devices,
        # copies on discrete devices. Device-only mode keeps imported pages only on
        # an integrated GPU. An unsupported import (e.g. Dozen) falls back to copies.
        inplace = [float(x) for x in re.findall(r"^\[VK\] qwen36: \d+ matmuls.*; ([0-9.]+) MiB read in place", err, re.M)]
        rows = extra.get("COLI_DENSE_I8", "0") != "0" and extra.get("COLI_DENSE_BITS", "8") in ("8", "16")
        shared = "dense rows shared with the integrated GPU" in err
        requested = "dense rows read in place, not copied" in err
        copy_required = "qwen36: dense weights on the device only" in err and not shared
        want = (rows and requested and not copy_required
                and "cannot import host memory" not in err)
        if want != bool(inplace and inplace[-1] > 0):
            raise SystemExit(f"FAIL: rows read in place: expected {want}, got {inplace[-1:] or 'none'}\n{err[-3000:]}")
        chain += f", {inplace[-1]:.1f} MiB read in place" if want else ", rows copied"
    if engine != "clef" and extra.get("COLI_VK_CHAIN", "") not in ("", "0"):
        fw = re.findall(rf"^\[VK\] {name} chain: (\d+) forwards on the device .*?, (\d+) on the CPU", err, re.M)
        if "COLI_VK_CHAIN_FAULT" in extra:
            # a device lost mid-run: forwards before it on the device, the rest on the CPU
            if not re.search(rf"^\[VK\] {name}: the device was lost", err, re.M) or not fw or \
                    int(fw[-1][0]) <= 0 or int(fw[-1][1]) <= 0:
                raise SystemExit(f"FAIL: the device was not lost in the middle of the run\n{err[-3000:]}")
            chain = f", {fw[-1][0]} forwards on the device and {fw[-1][1]} on the CPU after the loss"
        elif not fw or int(fw[-1][0]) <= 0 or int(fw[-1][1]) != 0:
            raise SystemExit(f"FAIL: the forward did not run on the device\n{err[-3000:]}")
        else:
            chain = f", {fw[-1][0]} forwards on the device"
    print(f"OK {engine} {' '.join(args[1:])}: {decisions} decisions and {refusals} refusals as on the CPU; "
          f"max |dp| {worst_p:.2e}, max |dlogit| {worst_l:.2e} of the largest; {counts[-1]} matmuls on the GPU{chain}")


if __name__ == "__main__":
    main()
