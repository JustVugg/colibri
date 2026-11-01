#!/usr/bin/env python3
"""Checkpoint MTP: Qwen reference logits, exact verification and multi-turn serve.

On the CPU by default. Uses the converted bytes for the independent transformers reference.
Natural and forced accept/reject/mixed/cycle verifies cover every depth 1..7;
stdout and final logits must be byte-identical to ordinary decoding.
--env KEY=VALUE runs everything under other settings (tests/vulkan_engines.sh: the Vulkan
chain), --device requires the head's experts on the device in every run with drafts and
--quick keeps depths 1, 3 and 7. --head-tol is the draft logits' tolerance, against the
reference and after a pin against a fresh session; above the CPU's 1e-6 the two wirings
are not told apart (on the tiny weights a device's roundings are as large as their gap:
the CPU's run checks it).
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import qwen36_mtp_ref as reference
import spec_drafts_harness as spec
import qwen38_mtp_harness as wire

ENV = {"COLI_VULKAN": "0", "COLI_CUDA": "0", "COLI_LOOKUP": "0", "Q36_MTP": "0",
       "OMP_NUM_THREADS": "2", "COLI_NO_OMP_TUNE": "1", "USAGE_SAVE": "0", "NOSTREAM": "1",
       "COLI_DENSE_I8": "0", "COLI_DENSE_IDOT": "0", "QWEN_EXPERT_ACT": "f32", "Q36_MAXT": "256"}
KEYS = tuple(k for k in os.environ if k.startswith(("Q36_MTP", "COLI_", "QWEN_"))) + (
    "Q36_MTP", "Q36_MTP_FORCE", "Q36_MTP_DRAFTS", "Q36_MTP_DUMP", "Q36_MTP_PMIN", "Q36_MTP_VOCAB", "Q36_MTP_VOCAB_IDS")
spec.SPEC_KEYS += KEYS
spec.BASE_ENV.update(ENV)
wire.BASE_ENV = ENV
wire.MTP_KEYS += KEYS


HEAD_TOL = 1e-6
QUICK = False
EXTRA = None
DEVICE = b"[mtp] the head routes and runs its experts on the device"


def on_device(got, err):
    """--device: the run's drafts came from the head on the device"""
    return " (the head on the device)" if DEVICE in err else None


def check_head(eng, ref, tmp):
    ids = json.loads(ref.read_text())["full_ids"]
    want = reference.mtp_logits(eng.fixture, ids).numpy()
    wrong = reference.mtp_logits(eng.fixture, ids, pre_norm=True).numpy()
    seen = set()
    for mode in ("reject", "mixed"):
        dump = tmp / f"head-{mode}.f32"
        r = eng.run(ref, {"Q36_MTP": "1", "Q36_MTP_FORCE": mode, "Q36_MTP_DUMP": str(dump)}, tmp / "last.f32")
        spec.check(r.returncode == 0 and dump.exists(), f"head {mode}: ran")
        if not dump.exists():
            print(r.stderr.decode()[-2000:])
            continue
        rows = wire.read_dump(dump, want.shape[1])
        worst = wrong_error = 0.
        same = True
        for row, token, logits in rows:
            seen.add(row)
            # a device's roundings may flip a near tie of the tiny weights' pick
            same &= token == ids[row+1] and (HEAD_TOL > 1e-6 or logits.argmax() == want[row].argmax())
            worst = max(worst, float(np.abs(logits - want[row]).max()))
            wrong_error = max(wrong_error, float(np.abs(logits - wrong[row]).max()))
        spec.check(bool(rows) and same and worst <= HEAD_TOL and (HEAD_TOL > 1e-6 or wrong_error > 10*worst)
                   and (EXTRA is None or EXTRA(None, r.stderr) is not None),
                   f"head {mode}: {len(rows)} rows, max error {worst:.3g}, pre-norm wiring {wrong_error:.3g}")
    prompt_n = len(json.loads(ref.read_text())["prompt_ids"])
    spec.check(seen >= set(range(prompt_n-1, len(ids)-3)), "head: every eligible backbone row checked")


def check_lossless(eng, ref, stdout, logits, n_new, tmp):
    for depth in ((1, 3, 7) if QUICK else range(1, 8)):
        for mode in ("", "reject", "cycle") if QUICK else ("", "reject", "accept", "mixed", "cycle"):
            settings = {"Q36_MTP": "1", "Q36_MTP_DRAFTS": str(depth), "Q36_MTP_FORCE": mode}
            want = spec.expected(n_new, depth, mode, room_cap=7) if mode else None
            spec.gate(eng, ref, stdout, logits, tmp, f"depth {depth} {mode or 'natural'}", settings, "mtp", want,
                      EXTRA)
    for mode in ("", "cycle"):
        spec.gate(eng, ref, stdout, logits, tmp, f"auto {mode}",
                  {"Q36_MTP": "1", "Q36_MTP_DRAFTS": "auto", "Q36_MTP_FORCE": mode}, "mtp", None, EXTRA)
    vocab = tmp / "ids.txt"
    vocab.write_text("\n".join(map(str, [7, 11, 13, 7, 17])))  # duplicate is counted once
    spec.gate(eng, ref, stdout, logits, tmp, "listed vocabulary + probability stop", {
        "Q36_MTP": "1", "Q36_MTP_DRAFTS": "7", "Q36_MTP_PMIN": "1", "Q36_MTP_VOCAB_IDS": str(vocab)}, "mtp", None, EXTRA)
    if EXTRA is None:
        spec.gate(eng, ref, stdout, logits, tmp, "adaptive vocabulary Q36_MTP_VOCAB", {
            "Q36_MTP": "1", "Q36_MTP_DRAFTS": "7", "Q36_MTP_VOCAB": "32"}, "mtp", None, EXTRA)


def check_serve(engine, model):
    rc, baseline, err = wire.serve_session(engine, model, {})
    spec.check(rc == 0 and sum(f.startswith(b"DONE") for f in baseline) == 5, "serve: five baseline turns")
    for mode in ("", "reject"):
        rc, frames, err = wire.serve_session(engine, model, {"Q36_MTP": "1", "Q36_MTP_DRAFTS": "5", "Q36_MTP_FORCE": mode})
        spec.check(rc == 0 and frames == baseline and b"[qwen36 mtp]" in err and (EXTRA is None or DEVICE in err),
                   f"serve {mode or 'natural'}: prefix reuse, pin, logprobs and sampling identical")
        if frames != baseline:
            print(next(((a, b) for a, b in zip(baseline, frames) if a != b), err[-2000:]))

    # A pin restores the pending pre-norm row as well as the backbone recurrence.
    turns = [(1, b"123 456:", 4, "0", "1", " pin=1"),
             (2, b"123 456: 01", 4, "0", "1", ""),
             (3, b"123 456: 9", 20, "0", "1", "")]
    with tempfile.TemporaryDirectory() as directory:
        dumps = []
        for history in (turns, turns[-1:]):
            dump = Path(directory) / f"{len(dumps)}.f32"
            rc, _, err = wire.serve_session(engine, model, {
                "Q36_MTP": "1", "Q36_MTP_FORCE": "reject", "Q36_MTP_DRAFTS": "1",
                "Q36_MTP_DUMP": str(dump)}, history)
            dumps.append(wire.read_dump(dump, 128) if rc == 0 and dump.exists() else [])
        warm, fresh = dumps
        worst = max((float(np.max(np.abs(a[2] - b[2]))) for a, b in zip(warm[-len(fresh):], fresh)), default=0.)
        same = bool(fresh) and len(warm) > len(fresh) and worst <= HEAD_TOL and all(
            a[0:2] == b[0:2] for a, b in zip(warm[-len(fresh):], fresh))
        spec.check(same, f"serve: head logits after pin restore equal fresh session (max difference {worst:.3g})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--model", required=True, action="append", type=Path)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for every run")
    ap.add_argument("--device", action="store_true", help="every run with drafts runs the head on the device")
    ap.add_argument("--head-tol", type=float, default=1e-6, help="draft logits against the reference")
    ap.add_argument("--quick", action="store_true", help="depths 1, 3 and 7")
    args = ap.parse_args()
    global HEAD_TOL, QUICK, EXTRA
    HEAD_TOL, QUICK, EXTRA = args.head_tol, args.quick, on_device if args.device else None
    for kv in args.env:
        k, v = kv.split("=", 1)
        ENV[k] = v   # the engines' runs, the reference's and serve's (wire.BASE_ENV is ENV)
        spec.BASE_ENV[k] = v
    for model in args.model:
        print(f"container: {model}", flush=True)
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            eng = spec.Engine(args.engine, "qwen36", model, 2, ENV)
            ref, stdout, logits = spec.build_reference(eng, args.source / "ref.json", 24, tmp)
            check_head(eng, ref, tmp)
            check_lossless(eng, ref, stdout, logits, 24, tmp)
            for cap in (1, 8):
                eng.cap = cap
                base = eng.run(ref, {}, tmp / "base.f32")
                spec.gate(eng, ref, base.stdout, (tmp / "base.f32").read_bytes(), tmp, f"cap {cap} depth 7",
                          {"Q36_MTP": "1", "Q36_MTP_DRAFTS": "7", "Q36_MTP_FORCE": "cycle"}, "mtp")
            # Also exercise default activation/dense quantization independently of the f32 oracle.
            eng.extra = dict(ENV, COLI_DENSE_I8="1", COLI_DENSE_IDOT="1", QWEN_EXPERT_ACT="i8")
            ref, stdout, logits = spec.build_reference(eng, args.source / "ref.json", 24, tmp)
            spec.gate(eng, ref, stdout, logits, tmp, "quantized decode depth 7", {
                "Q36_MTP": "1", "Q36_MTP_DRAFTS": "7", "Q36_MTP_FORCE": "cycle"}, "mtp")
        check_serve(args.engine, model)
    if spec.FAILS:
        sys.exit(f"{len(spec.FAILS)} MTP gates failed")
    print("all Qwen3.6 checkpoint MTP gates passed")


if __name__ == "__main__":
    main()
