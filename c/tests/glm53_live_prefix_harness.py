#!/usr/bin/env python3
"""CPU-only generated-model GLM-5.3 live-extension and overflow oracle for #2507.

A real, pinned prefix followed by valid plain-text extensions must use the live
cache rather than rewind to an older pin. Tests exact served bytes vs a cold
engine, predictor-sensitive ECHO scores, and divergent-prefix behavior.
No production service, GPU or network is touched. Exits nonzero on missing
inputs, early EOF, mismatch or timeout, never reports a skipped oracle as PASS.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

P = b"the quick brown"
A = b" fox jumps"
B = b" cat sleeps"
X = b" over it"


class Engine:
    def __init__(self, binary, fixture, root, slots=1):
        self.note_path = Path(root) / "serve.stderr"
        self.note = open(self.note_path, "wb")
        env = dict(os.environ, SERVE="1", SERVE_BATCH="1", SNAP=str(fixture),
                   GLM53_BITS="32", GLM53_VERBOSE="1", CUDA_VISIBLE_DEVICES="-1",
                   OMP_NUM_THREADS="2", KV_SLOTS=str(slots))
        self.p = subprocess.Popen([str(binary)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.note, env=env)
        ready = self.line()
        if "READY" not in ready:
            raise AssertionError(f"expected READY, got {ready!r}")
        while True:
            status = self.line()
            if status.startswith("STAT "):
                break
            if not status.startswith("CAPS "):
                raise AssertionError(f"expected STAT/CAPS, got {status!r}")

    def line(self):
        b = self.p.stdout.readline()
        if not b:
            raise AssertionError("unexpected engine EOF")
        return b.decode("utf-8", "replace").rstrip("\n")

    def request(self, rid, payload, max_tokens=4, logprobs=False, pin=False, slot=0):
        ext = (" logprobs=1" if logprobs else "") + (" pin=1" if pin else "")
        header = f"SUBMIT {rid} {slot} {len(payload)} {max_tokens} 0 1{ext}\n".encode()
        self.p.stdin.write(header + payload + b"\n")
        self.p.stdin.flush()
        chunks, echoes = [], {}
        accepted = False
        while True:
            fields = self.line().split()
            if not fields:
                continue
            typ = fields[0]
            if typ in ("DATA", "ECHO"):
                assert int(fields[1]) == rid, (rid, fields)
                size = int(fields[2])
                raw = self.p.stdout.read(size + 1)
                assert len(raw) == size + 1 and raw[-1:] == b"\n", (fields, raw)
                if typ == "DATA":
                    chunks.append(raw[:size])
                else:
                    assert len(fields) >= 5, fields
                    echoes[int(fields[3])] = fields[4]
            elif typ in ("DONE", "ERROR"):
                assert int(fields[1]) == rid, (rid, fields)
                assert typ == "DONE", fields
                self.note.flush()
                log = self.note_path.read_text(errors="replace")
                matches = [line.split() for line in log.splitlines()
                           if line.startswith(f"REUSE {rid} ")]
                assert len(matches) == 1, (rid, matches, log[-1200:])
                return b"".join(chunks), echoes, {
                    "reused": int(matches[0][2]),
                    "prompt": int(matches[0][3]),
                    "cached": int(matches[0][4]),
                    "common": int(matches[0][5]),
                    "reason": matches[0][6],
                }
            elif typ == "ACCEPT":
                # #2508 adds early acknowledgement before prefill.  It is
                # optional on older engines, but when present must be
                # exactly once, for our request, before any model data.
                assert not accepted and not chunks and not echoes, (rid, fields)
                assert len(fields) == 3 and int(fields[1]) == rid, (rid, fields)
                assert 0 < int(fields[2]) <= len(payload), (rid, fields)
                accepted = True
            else:
                # The serving protocol permits diagnostic frames.
                assert typ in ("HITS", "PROF", "STATS", "PREFILL", "FRAME", "EMAP"), fields

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        self.note.close()


def in_engine(binary, fixture, use, slots=1):
    with tempfile.TemporaryDirectory(prefix="glm2507-") as root:
        eng = Engine(binary, fixture, root, slots=slots)
        try:
            return use(eng)
        finally:
            eng.close()


def check_equal(a, b, label):
    if a != b:
        raise AssertionError(f"{label}: got {a!r}, reference {b!r}")


def run(binary, fixture):
    def cold(payload, budget=4, logprobs=False):
        return in_engine(binary, fixture,
                         lambda e: e.request(1, payload, budget, logprobs))

    # Seed a pin without output, then feed a valid plain-text extension that
    # should not restore it. Older code restarts at P rather than reusing P+A.
    def extension(eng):
        _, p_echo, p = eng.request(1, P, 0, logprobs=True, pin=True)
        assert p["reused"] == 0 and p_echo, p
        _, _, middle = eng.request(2, P + A, 1)
        assert middle["reused"] == len(P), middle
        answer, _, last = eng.request(3, P + A + X, 4)
        assert last["reused"] == len(P + A), (
            "old-pin rewind detected: wanted complete live extension", last)
        check_equal(answer, cold(P + A + X)[0], "strict live-extension answer")

        # Predictor-sensitive request keeps the pin restore/ECHO path.
        _, scores, predictor = eng.request(4, P + A + X, 0, logprobs=True)
        assert predictor["reused"] == len(P), predictor
        cold_scores = cold(P + A + X, 0, logprobs=True)[1]
        tail = [i for i in cold_scores if i >= len(P)]
        assert tail, "no predictor-sensitive tail scores"
        for i in tail:
            check_equal(scores.get(i), cold_scores[i], f"predictor ECHO[{i}]")
        return last, predictor, len(tail)

    extension_last, predictor_last, n_scores = in_engine(
        binary, fixture, extension)

    # Divergence cannot use stale live rows; it may restore P's valid pin.
    def divergence(eng):
        eng.request(11, P, 0, logprobs=True, pin=True)
        eng.request(12, P + A, 1)
        eng.request(13, P + B, 1)
        out, _, reuse = eng.request(14, P + B + X, 4)
        assert reuse["reused"] == len(P + B), reuse
        check_equal(out, cold(P + B + X)[0], "divergent branch fresh state")
        return reuse

    divergent = in_engine(binary, fixture, divergence)
    first_cold = cold(P + A + X)[2]
    assert first_cold["reused"] == 0, first_cold
    print("PASS #2507 actual GLM pinned-live extension:")
    print(f"  strict extension reused={extension_last['reused']} cached={extension_last['cached']} (old pin={len(P)})")
    print(f"  predictor-sensitive request reused={predictor_last['reused']} and {n_scores} exact cold ECHO scores")
    print(f"  divergent branch recovered reused={divergent['reused']} and matched cold output")
    print("  restarted process expected cold reuse=0; no persistence inference")
    return 0



def multislot_run(binary, fixture):
    """Two independent slots must not alias live state or pinned ECHO predictors."""
    Q = b"hello wonderful"

    def cold(payload, *, logprobs=False, budget=4):
        return in_engine(binary, fixture,
                         lambda e: e.request(1000, payload, budget, logprobs),
                         slots=2)

    def check(eng):
        p0 = eng.request(1, P, 0, logprobs=True, pin=True, slot=0)
        p1 = eng.request(2, Q, 0, logprobs=True, pin=True, slot=1)
        assert p0[2]["reused"] == p1[2]["reused"] == 0 and p0[1] and p1[1]

        m0 = eng.request(3, P + A, 1, slot=0)
        m1 = eng.request(4, Q + B, 1, slot=1)
        assert m0[2]["reused"] == len(P), m0[2]
        assert m1[2]["reused"] == len(Q), m1[2]

        out0 = eng.request(5, P + A + X, 4, slot=0)
        out1 = eng.request(6, Q + B + X, 4, slot=1)
        assert out0[2]["reused"] == len(P + A), out0[2]
        assert out1[2]["reused"] == len(Q + B), out1[2]
        check_equal(out0[0], cold(P + A + X)[0], "slot0 exact output")
        check_equal(out1[0], cold(Q + B + X)[0], "slot1 exact output")

        for rid, slot, pin, prompt in ((7,0,P,P+A+X),(8,1,Q,Q+B+X)):
            observed = eng.request(rid, prompt, 0, logprobs=True, slot=slot)
            assert observed[2]["reused"] == len(pin), observed[2]
            reference = cold(prompt, logprobs=True, budget=0)
            tail = [i for i in reference[1] if i >= len(pin)]
            assert tail, "no predictor-sensitive scores"
            for i in tail:
                check_equal(observed[1].get(i), reference[1][i],
                            f"slot{slot} ECHO[{i}]")
        return out0[2]["reused"], out1[2]["reused"]

    a, b = in_engine(binary, fixture, check, slots=2)
    print(f"PASS #2507 independent slots: live reuse={a}/{b}, cold output and ECHO predictors exact")


def overflow_run(binary, fixture):
    previous = os.environ.get("GLM53_MAXT")
    os.environ["GLM53_MAXT"] = "64"
    try:
        with tempfile.TemporaryDirectory(prefix="glm2507-overflow-") as root:
            e = Engine(binary, fixture, root)
            try:
                _, echo, pin = e.request(1, P, 0, logprobs=True, pin=True)
                assert pin["reused"] == 0 and echo, pin
                _, _, middle = e.request(2, P + A, 1)
                assert middle["reused"] == len(P), middle
                oversized = b"z" * 80
                header = f"SUBMIT 3 0 {len(oversized)} 4 0 1\n".encode()
                e.p.stdin.write(header + oversized + b"\n")
                e.p.stdin.flush()
                for _ in range(16):
                    rejected = e.line()
                    if rejected.startswith("ERROR 3 "):
                        break
                    assert rejected.split()[0] in ("EMAP", "PROF", "HITS", "STATS"), rejected
                assert rejected.startswith("ERROR 3 CONTEXT_EXCEEDED "), rejected
                assert "capacity=64" in rejected and "prompt_tokens=64" in rejected
                last, _, reuse = e.request(4, P + A + X, 4)
                assert reuse["reused"] == len(P + A), reuse
                assert reuse["cached"] == len(P + A), reuse
            finally:
                e.close()
        with tempfile.TemporaryDirectory(prefix="glm2507-overflow-cold-") as root:
            fresh = Engine(binary, fixture, root)
            try:
                answer, _, cold = fresh.request(1, P + A + X, 4)
                assert cold["reused"] == 0, cold
            finally:
                fresh.close()
        check_equal(last, answer, "post-overflow exact cold answer")
        print("PASS #2507 overflow recovery: CONTEXT_EXCEEDED capacity=64; valid 25-token cache survived; exact cold output")
    finally:
        if previous is None:
            os.environ.pop("GLM53_MAXT", None)
        else:
            os.environ["GLM53_MAXT"] = previous

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    a = p.parse_args()
    if not a.binary.is_file() or not (a.fixture / "ref.json").is_file():
        print("SKIP: no generated multimodal fixture or engine")
        return 2
    binary,fixture=a.binary.resolve(),a.fixture.resolve()
    run(binary,fixture)
    overflow_run(binary,fixture)
    multislot_run(binary,fixture)
    return 0


if __name__ == "__main__":
    sys.exit(main())
