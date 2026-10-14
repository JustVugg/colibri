#!/usr/bin/env python3
"""Qwen3.8 MTP head and speculative decoding, on the tiny fixtures with a head.

make qwen38-tiny-mtp-check runs it on the BF16, FP8 and int4-g64 fixtures that
tools/make_qwen38_tiny.py --mtp writes (with a byte tokenizer for text and
serve). Gates:

  head      the engine's draft logits (Q38_MTP_DUMP, one record per draft)
            equal tools/qwen38_mtp_ref.py's float32 reference of the head to
            float tolerance, for the default wiring b and the diagnostic a
            (Q38_MTP_WIRING), through the one-row and the two-row (after an
            accepted draft) paths, each closer to its own reference than to
            the other's.
  lossless  with the head drafting, stdout and the last logits (DUMP) are
            byte-identical to Q38_MTP=0 at caps 1/2/8, prefill batching on
            and off, BF16 and int8 trunk, in the normal mode and in the three
            Q38_MTP_FORCE modes: reject (every draft rolled back), accept
            (every draft right) and mixed (alternating); the forced modes'
            acceptance and tokens/forward match the count the loop must give.
  text      a 40-token text-prompt decode, MTP on and off, byte-identical.
  serve     a multi-turn serve session (prefix reuse, a pinned prefix,
            logprobs, a sampled turn) gives the same frames MTP on and off.

    python tests/qwen38_mtp_harness.py --engine ./qwen38 --bf16 ./qwen38_tiny_mtp \\
        --fp8 ./qwen38_tiny_fp8_mtp --int4 ./qwen38_tiny_int4_mtp
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tools"))

FAILS = []
# Q38_MTP_DRAFTS=1: this harness gates the one-draft verify (S=2) and counts its loop;
# deeper verifies and prompt lookup are tests/spec_drafts_harness.py's
BASE_ENV = {"OMP_NUM_THREADS": "2", "COLI_NO_OMP_TUNE": "1", "COLI_CUDA": "0",
            "Q38_TRUNK_GPU": "0", "NOSTREAM": "1", "USAGE_SAVE": "0", "Q38_MTP_DRAFTS": "1",
            # the comparisons are against plain decoding, and the MTP head and prompt lookup
            # are on by default since 1.13.0: a run is plain unless it sets Q38_MTP=1 itself
            "Q38_MTP": "0", "COLI_LOOKUP": "0"}
MTP_KEYS = ("Q38_MTP", "Q38_MTP_WIRING", "Q38_MTP_FORCE", "Q38_MTP_DUMP", "DUMP",
            "Q38_PREFILL_BATCH", "Q38_TRUNK_MIN_KB", "Q38_EXPERT_INT4", "Q38_MTP_DRAFTS", "COLI_LOOKUP")


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILS.append(what)


def run(engine, model, args, extra, stdin=None):
    env = {k: v for k, v in os.environ.items() if k not in MTP_KEYS}
    env.update(BASE_ENV)
    env["SNAP"] = str(model)
    env.update(extra)
    return subprocess.run([str(Path(engine).resolve()), *map(str, args)], env=env, input=stdin,
                          capture_output=True, timeout=600)


SPEC_LINE = re.compile(r"\[qwen38 MTP\] (\S+): ([0-9.]+) tokens/forward \((\d+) forwards per (\d+) tokens\) "
                       r"\| acceptance [0-9.]+% \((\d+)/(\d+) drafts\)")


def spec_counts(stderr):
    """(forwards, tokens, accepted, drafts) of the run's speculation line."""
    found = SPEC_LINE.search(stderr.decode(errors="replace"))
    return tuple(int(found.group(i)) for i in (3, 4, 5, 6)) if found else None


def expected_counts(n_new, mode):
    """What q38_spec_step must count for a CLI run of n_new tokens in a forced
    mode: the prompt's forward gives token 1, then one call per token fed; a
    call drafts when two more tokens may follow, and the next call takes the
    draft's logits with no forward when the draft stands."""
    drafts = accepted = forwards = tokens = 0
    ahead = None
    for s in range(n_new - 1):
        tokens += 1
        if ahead is not None:
            standing, ahead = ahead, None
            if standing:
                accepted += 1
                continue
        if n_new - 1 - s >= 2:
            standing = mode == "accept" or (mode == "mixed" and drafts % 2 == 0)
            drafts += 1
            ahead = standing
        forwards += 1
    return forwards, tokens, accepted, drafts


# ---------- head: draft logits against the torch reference ----------

def read_dump(path, vocab):
    raw = Path(path).read_bytes()
    record = 8 + 4 * vocab
    if len(raw) % record:
        raise ValueError(f"{path}: {len(raw)} bytes is not a whole number of {record}-byte records")
    out = []
    for at in range(0, len(raw), record):
        row, token = np.frombuffer(raw[at:at + 8], np.int32)
        out.append((int(row), int(token), np.frombuffer(raw[at + 8:at + record], np.float32)))
    return out


def check_head(engine, model):
    """The head's numbers. The fixture's four streams end with nearly equal
    norms, so wirings a and b part by only ~1e-5 on its logits; the engine
    meets its own wiring's float32 reference ~100x closer than that, which is
    what tells the wirings apart here."""
    import qwen38_mtp_ref as ref
    data = json.loads((model / "ref.json").read_text(encoding="utf-8"))
    ids, vocab = data["full_ids"], len(data["final_logits"])
    print(f"head: {model}")
    references = {w: ref.mtp_logits(model, ids, w).numpy() for w in "ba"}
    gap = float(np.abs(references["a"] - references["b"]).max())
    check(gap > 2e-6, f"wirings b and a give different logits (max |diff| {gap:.3g})")
    with tempfile.TemporaryDirectory() as tmp:
        # b with Q38_MTP_WIRING unset, as the engine runs by default; a by name
        for wiring, knob in (("b (default)", {}), ("a", {"Q38_MTP_WIRING": "a"})):
            rows_seen = set()
            for mode in ("reject", "mixed"):
                dump = Path(tmp) / f"{wiring[0]}-{mode}.f32"
                # reject drafts at every token (rows past the prompt, one pending
                # row each); mixed reaches the head with two pending rows after
                # each accepted draft. The BF16 trunk: every fixture matrix is
                # under the 1 MiB int8 threshold.
                r = run(engine, model, [2, 8, model / "ref.json"],
                        dict(knob, Q38_MTP="1", Q38_MTP_FORCE=mode,
                             Q38_MTP_DUMP=str(dump), Q38_TRUNK_MIN_KB="1024"))
                if r.returncode:
                    check(False, f"wiring {wiring} {mode}: engine exit {r.returncode}\n"
                                 + r.stderr.decode(errors="replace")[-2000:])
                    continue
                worst, nearest_other, argmax_ok, rows = 0.0, float("inf"), True, 0
                for row, token, logits in read_dump(dump, vocab):
                    if token != ids[row + 1]:
                        check(False, f"wiring {wiring} {mode}: draft row {row} paired token {token}, "
                                     f"the sequence has {ids[row + 1]}")
                        continue
                    own = wiring[0]
                    errors = {w: float(np.abs(logits - references[w][row]).max()) for w in "ba"}
                    worst = max(worst, errors[own])
                    nearest_other = min(nearest_other, *(e for w, e in errors.items() if w != own))
                    argmax_ok &= int(logits.argmax()) == int(references[own][row].argmax())
                    rows += 1
                    rows_seen.add(row)
                check(rows > 0 and worst <= 1e-6 and 20 * worst <= nearest_other and argmax_ok,
                      f"wiring {wiring} {mode}: {rows} draft rows within {worst:.2g} of the reference "
                      f"(the other wiring's is {nearest_other:.2g} away), argmax {'equal' if argmax_ok else 'DIFFERS'}")
            check(rows_seen >= set(range(len(data["prompt_ids"]) - 1, len(ids) - 3)),
                  f"wiring {wiring}: rows {sorted(rows_seen)} compared")


# ---------- lossless: MTP on == MTP off ----------

def check_lossless(engine, model, ref_name, extra=None):
    extra = dict(extra or {})
    data = json.loads((model / ref_name).read_text(encoding="utf-8"))
    n_new = len(data["full_ids"]) - len(data["prompt_ids"])
    print(f"lossless: {model} ({ref_name}{', ' + ' '.join(f'{k}={v}' for k, v in extra.items()) if extra else ''})")
    configs, failed_before = 0, len(FAILS)
    with tempfile.TemporaryDirectory() as tmp:
        for trunk in ("1024", "0"):
            for batch in ("0", "1"):
                for cap in (1, 2, 8):
                    tag = f"trunk={trunk} batch={batch} cap={cap}"
                    env = dict(extra, Q38_TRUNK_MIN_KB=trunk, Q38_PREFILL_BATCH=batch)
                    base_dump = Path(tmp) / "off.f32"
                    off = run(engine, model, [cap, 8, model / ref_name], dict(env, DUMP=str(base_dump)))
                    if off.returncode:
                        check(False, f"{tag}: MTP off fails the oracle\n" + off.stdout.decode(errors="replace"))
                        continue
                    base_logits = base_dump.read_bytes()
                    for mode in ("", "reject", "accept", "mixed"):
                        dump = Path(tmp) / "on.f32"
                        on = run(engine, model, [cap, 8, model / ref_name],
                                 dict(env, DUMP=str(dump), Q38_MTP="1", Q38_MTP_FORCE=mode))
                        name = f"{tag} {mode or 'normal'}"
                        same = on.returncode == 0 and on.stdout == off.stdout and dump.read_bytes() == base_logits
                        counts = spec_counts(on.stderr)
                        if mode:
                            want = expected_counts(n_new, mode)
                            same &= counts == want
                        else:
                            same &= counts is not None and counts[2] <= counts[3] and counts[0] <= counts[1]
                        if not same:
                            check(False, f"{name}: output differs from MTP off or wrong accounting "
                                         f"(counts {counts}, want {expected_counts(n_new, mode) if mode else 'consistent'})\n"
                                         + on.stdout.decode(errors="replace") + on.stderr.decode(errors="replace")[-1500:])
                        configs += 1
        off = run(engine, model, [8, 8, model / ref_name], dict(extra))
        for knob in ({"Q38_MTP_WIRING": "a"}, {"Q38_MTP_CAP": "1"}):
            on = run(engine, model, [8, 8, model / ref_name], dict(extra, Q38_MTP="1", **knob))
            name = " ".join(f"{k}={v}" for k, v in knob.items())
            check(on.returncode == 0 and on.stdout == off.stdout, f"{name}: output equals MTP off")
    want = {m: expected_counts(n_new, m) for m in ("reject", "accept", "mixed")}
    check(len(FAILS) == failed_before,
          f"{configs} runs byte-identical to MTP off (stdout and last logits); forced counts "
          + ", ".join(f"{m} {w[0]} forwards/{w[1]} tokens {w[2]}/{w[3]} accepted" for m, w in want.items()))


# ---------- text: a longer greedy decode ----------

def check_text(engine, model, extra=None):
    print(f"text: {model}")
    with tempfile.TemporaryDirectory() as tmp:
        prompt = Path(tmp) / "prompt.txt"
        prompt.write_bytes(b"123 456: 7")
        outs = {}
        for mode in ("off", "", "reject"):
            dump = Path(tmp) / f"{mode}.f32"
            env = dict(extra or {}, N_NEW="40", DUMP=str(dump), TOK=str(model / "tokenizer.json"))
            if mode != "off":
                env.update(Q38_MTP="1", Q38_MTP_FORCE=mode)
            r = run(engine, model, [2, 8, prompt], env)
            outs[mode] = (r.returncode, r.stdout, re.findall(rb"Text +:.*", r.stderr), dump.read_bytes() if dump.exists() else b"")
            if mode == "":
                counts = spec_counts(r.stderr)
                print(f"       normal mode on 40 tokens: forwards/tokens/accepted/drafts = {counts}")
        check(outs["off"][0] == 0 and outs["off"][2], "40-token text decode runs with MTP off")
        for mode in ("", "reject"):
            check(outs[mode] == outs["off"], f"40-token text decode, MTP {mode or 'normal'}: same text and last logits")


# ---------- serve: several turns through the wire protocol ----------

SERVE_TURNS = [
    (1, b"123 456:", 24, "0", "1", ""),
    (2, b"123 456: 0", 24, "0", "1", " pin=1"),
    (3, b"123 456: 01", 20, "0", "1", " logprobs=2"),
    (4, b"789 012:", 16, "0.9", "0.95", ""),
    (5, b"123 456: 0123", 20, "0", "1", ""),
]


def serve_session(engine, model, extra, turns=SERVE_TURNS):
    """Turns sent one at a time, each when the previous one is DONE (a SUBMIT
    during a turn is refused as busy). Returns the exit code, the frames that
    must not depend on drafting, and stderr."""
    env = {k: v for k, v in os.environ.items() if k not in MTP_KEYS}
    env.update(BASE_ENV)
    env.update(extra, SNAP=str(model), SERVE="1", TOK=str(model / "tokenizer.json"), Q38_MAXT="128")
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen([str(Path(engine).resolve()), "1", "8"], env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=err)
        frames = []

        def frame():
            line = p.stdout.readline()
            if not line:
                raise EOFError("engine closed")
            fields = line.replace(b"\x01", b"").split()
            if fields and fields[0] in (b"DATA", b"ECHO"):
                n = int(fields[2])
                line = line.rstrip(b"\n") + b"|" + p.stdout.read(n)
                p.stdout.read(1)
            return fields, line

        try:
            while frame()[0][:1] != [b"READY"]:
                pass
            for rid, text, count, temp, top_p, ext in turns:
                p.stdin.write(f"SUBMIT {rid} 0 {len(text)} {count} {temp} {top_p}{ext}\n".encode() + text + b"\n")
                p.stdin.flush()
                while True:
                    fields, line = frame()
                    if not fields or fields[0] in (b"STAT", b"PROF", b"EMAP", b"HITS", b"CAPS"):
                        continue    # timings, and the expert residency the draft rows change
                    if fields[0] == b"DONE":    # tokens/s and RSS vary: keep the count and the reason
                        line = b" ".join(fields[:4] + fields[-2:])
                    frames.append(line)
                    if fields[0] in (b"DONE", b"ERROR"):
                        break
            p.stdin.close()
            rc = p.wait(timeout=60)
        except Exception as exc:       # noqa: BLE001 -- reported as a failed gate
            p.kill()
            p.wait()
            frames.append(f"harness: {exc!r}".encode())
            rc = -1
        err.seek(0)
        return rc, frames, err.read()


def check_serve(engine, model, extra=None):
    extra = dict(extra or {})
    print(f"serve: {model}")
    rc, base, _ = serve_session(engine, model, extra)
    done = [f for f in base if f.startswith(b"DONE")]
    check(rc == 0 and len(done) == 5, f"MTP off: five turns served ({len(done)} DONE)")
    for mode in ("", "reject"):
        rc, frames, err = serve_session(engine, model, dict(extra, Q38_MTP="1", Q38_MTP_FORCE=mode))
        turns = len(re.findall(rb"\[qwen38 MTP\] turn \d+:", err))
        check(rc == 0 and frames == base and turns == 5,
              f"MTP {mode or 'normal'}: every frame equals MTP off (prefix reuse, pin, logprobs, "
              f"sampled turn), {turns} per-turn MTP lines")
        if frames != base:
            for a, b in zip(base, frames):
                if a != b:
                    print(f"       first difference:\n         off {a[:160]!r}\n         on  {b[:160]!r}")
                    break
    # The head's place survives a restored prefix: after the prompt cache
    # (turn 2 extends turn 1) and after a pin (turn 8 extends turn 6's pin;
    # turn 7 moved the prompt cache to a longer prompt that turn 8 leaves
    # at position 9, and kept the pinned rows), its drafts are bit for bit
    # those of a session that sees the turn first.
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    vocab = config.get("text_config", config)["vocab_size"]
    turn = {t[0]: t for t in SERVE_TURNS + [(6, b"123 456:", 4, "0", "1", " pin=1"),
                                            (7, b"123 456: 01", 4, "0", "1", ""),
                                            (8, b"123 456: 9", 20, "0", "1", "")]}
    with tempfile.TemporaryDirectory() as tmp:
        for history, last, how in (((1,), 2, "the prompt cache"), ((6, 7), 8, "a pin")):
            dumps = []
            for turns in ([turn[i] for i in history] + [turn[last]], [turn[last]]):
                dump = Path(tmp) / f"{len(dumps)}.f32"
                rc, _, _ = serve_session(engine, model, dict(extra, Q38_MTP="1", Q38_MTP_DUMP=str(dump)), turns)
                dumps.append(read_dump(dump, vocab) if rc == 0 else [])
            warm, fresh = dumps
            same = bool(fresh) and len(warm) > len(fresh) and all(
                a[0] == b[0] and a[1] == b[1] and a[2].tobytes() == b[2].tobytes()
                for a, b in zip(warm[-len(fresh):], fresh))
            check(same, f"turn {last} after {how}: its {len(fresh)} drafts' logits equal a fresh session's")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--bf16", type=Path, help="tools/make_qwen38_tiny.py --mtp")
    ap.add_argument("--fp8", type=Path, help="... --fp8-experts --mtp")
    ap.add_argument("--int4", type=Path, help="... --fp8-experts --int4-experts --expert-gain 3 --mtp")
    ap.add_argument("--skip-head", action="store_true", help="no torch reference (lossless gates only)")
    a = ap.parse_args()
    for model in (a.bf16, a.fp8):
        if model and not a.skip_head:
            check_head(a.engine, model)
    for model in (a.bf16, a.fp8):
        if model:
            check_lossless(a.engine, model, "ref.json")
    if a.int4:
        check_lossless(a.engine, a.int4, "ref_int4.json", {"Q38_EXPERT_INT4": "1"})
    for model in (a.bf16, a.fp8, a.int4):
        if model and (model / "tokenizer.json").exists():
            extra = {"Q38_EXPERT_INT4": "1"} if model == a.int4 else {}
            check_text(a.engine, model, extra)
            check_serve(a.engine, model, extra)
    if FAILS:
        print(f"qwen38 MTP: {len(FAILS)} check(s) failed")
        return 1
    print("qwen38 MTP: head matches the reference for wirings b (default) and a; decoding with drafts is "
          "byte-identical to decoding without them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
