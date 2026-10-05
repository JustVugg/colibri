#!/usr/bin/env python3
"""Speculative decoding with deeper verifies: MTP drafts up to 3 (qwen38) and
prompt-lookup drafts up to 5 (qwen38 and qwen36), against plain decoding.

The gate is the one docs/speculative.md states: with drafts, the tokens and the
last logits are byte-identical to the same run without them, whatever the drafts
were and wherever they were rejected. On a tiny fixture:

  reference  the fixture's prompt with a repeating tail (so the context has
             n-grams to find), decoded plainly for N_NEW tokens; that decode is
             the oracle every forced mode drafts from.
  mtp        (qwen38, a fixture with an MTP head) Q38_MTP_DRAFTS=1, 2, 3 in the
             normal mode and forced: every draft rejected (reject), every one
             right (accept), alternating verifies (mixed), the rejected row
             cycling over the verifies (cycle) and the rejection at each row
             (row1..rowD); the run's forwards, tokens and accepted/proposed
             drafts must be the count the loop gives. Q38_MTP_DRAFTS=0 (the gate
             picks the depth) in the normal and cycle modes.
  lookup     COLI_LOOKUP=1 with the gate off (COLI_SPEC_GATE=0) and forced
             proposals of the oracle's tokens: accept, mixed, cycle, row1..row5
             and a shallower COLI_LOOKUP_DRAFTS, counts checked; natural lookup
             with the gate on and off (off: the repeating prompt must give
             verifies, and accepted drafts), and on qwen38 combined with the MTP
             head at depth 3 and at the gate's depth.
  serve      (fixtures with a tokenizer) a session of turns with drafts against
             the same session without: a repeating prompt, its continuation by
             the reply (prefix reuse of the state a speculating turn left, which
             a turn cut off mid-verify must leave as plain decoding would), a
             pinned prompt and a turn restoring the pin, a sampled turn and a
             logprobs turn. Every DATA and ECHO frame must be the same.

usage: spec_drafts_harness.py --engine ./qwen38 --kind qwen38 --fixture qwen38_tiny_mtp [--ref ref.json]
       [--env K=V ...] [--mtp] [--serve] [--n-new 40] [--cap 2]
The extra environment applies to both arms (COLI_VULKAN=1 COLI_VK_CHAIN=1 runs the
whole gate in the dense chain: with drafts against without, on the device).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

FAILS = []
SANITIZE = False   # --sanitize: every run's stderr is checked for an ASan or UBSan report
REQUIRE_KV_SPLIT = False
SAN_RE = re.compile(rb"ERROR: AddressSanitizer|runtime error:")
BASE_ENV = {"OMP_NUM_THREADS": "2", "COLI_NO_OMP_TUNE": "1", "COLI_CUDA": "0", "Q38_TRUNK_GPU": "0",
            "NOSTREAM": "1", "USAGE_SAVE": "0",
            # plain decoding is the reference, and both draft sources are on by default since
            # 1.13.0: every run starts from them off, and a case turns on what it checks
            "Q38_MTP": "0", "COLI_LOOKUP": "0"}
SPEC_KEYS = ("Q38_MTP", "Q38_MTP_DRAFTS", "Q38_MTP_FORCE", "Q38_MTP_DUMP", "COLI_LOOKUP", "COLI_LOOKUP_DRAFTS",
             "COLI_LOOKUP_FORCE", "COLI_SPEC_GATE", "DUMP", "SERVE", "N_NEW", "TOK")
MAX_ROWS = 6


def sanitizer_check(stderr, what):
    if SANITIZE and SAN_RE.search(stderr if isinstance(stderr, bytes) else stderr.encode()):
        text = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        check(False, f"{what}: sanitizer report\n" + text[-4000:])


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILS.append(what)


class Engine:
    def __init__(self, path, kind, fixture, cap, extra):
        self.path, self.kind, self.fixture, self.cap, self.extra = Path(path).resolve(), kind, Path(fixture), cap, extra

    def env(self, spec):
        env = {k: v for k, v in os.environ.items() if k not in SPEC_KEYS}
        env.update(BASE_ENV)
        env.update(self.extra)
        env["SNAP"] = str(self.fixture)
        env.update(spec)
        return env

    def run(self, ref, spec, dump):
        env = self.env(dict(spec, DUMP=str(dump)))
        r = subprocess.run([str(self.path), str(self.cap), "8", str(ref)], env=env, capture_output=True,
                           timeout=1800)
        sanitizer_check(r.stderr, " ".join(f"{k}={v}" for k, v in spec.items()) or "plain")
        if REQUIRE_KV_SPLIT:
            trace = rb"\[VK\] " + self.kind.encode() + rb" chain: KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part"
            check(re.search(trace, r.stderr) is not None,
                  "split KV executed host attention: " + (" ".join(f"{k}={v}" for k, v in spec.items()) or "plain"))
        return r


SPEC_LINE = r"\[{kind} {src}\] run: [0-9.]+ tokens/forward \((\d+) forwards per (\d+) tokens\) \| acceptance [0-9.]+% \((\d+)/(\d+) drafts"


def counts(kind, src, stderr):
    """(forwards, tokens, accepted, drafts) of the run's line for a draft source."""
    m = re.search(SPEC_LINE.format(kind=kind, src=src), stderr.decode(errors="replace"))
    return tuple(int(m.group(i)) for i in (1, 2, 3, 4)) if m else None


def expected(n_new, depth, mode, row=0, room_cap=MAX_ROWS - 1):
    """What the speculation loop must count for a CLI run of n_new tokens with every
    verify carrying min(depth, more - 1) drafts, forced: the prompt's forward gives
    token 1, then one call per token fed; a pick equal to the next draft is answered
    from the verify, the first that differs ends it."""
    drafts = accepted = forwards = tokens = verifies = 0
    queue = []
    for s in range(n_new - 1):
        tokens += 1
        if queue:
            if queue.pop(0):
                accepted += 1
                continue
            queue = []
        more = n_new - 1 - s
        k = min(depth, more - 1, room_cap)
        if k >= 1:
            if mode == "reject":
                wrong = 1
            elif mode == "accept":
                wrong = 0
            elif mode == "mixed":
                wrong = 1 if verifies % 2 else 0
            elif mode == "cycle":
                wrong = verifies % (k + 1) + 1
            else:
                wrong = row
            queue = [wrong == 0 or j < wrong - 1 for j in range(k)]
            drafts += k
            verifies += 1
        forwards += 1
    return forwards, tokens, accepted, drafts


def engine_tokens(stdout):
    m = re.search(rb"^C engine :([^\n]*?)(?:Text|$)", stdout, re.M)
    return [int(t) for t in m.group(1).split()] if m else None


def check_device_loss(eng, ref, base_out, tmp):
    """A late failed chunk must replay embeddings, not earlier chunks' streams."""
    forced = ({"Q38_MTP": "1", "Q38_MTP_DRAFTS": "3", "Q38_MTP_FORCE": "cycle"}
              if eng.kind == "qwen38" else
              {"COLI_LOOKUP": "1", "COLI_LOOKUP_DRAFTS": "5", "COLI_SPEC_GATE": "0", "COLI_LOOKUP_FORCE": "cycle"})
    probe = eng.run(ref, forced, tmp / "loss-probe.f32")
    frames = re.search(rb"\[VK\] " + eng.kind.encode() + rb" chain: [0-9]+ forwards, ([0-9]+) frames", probe.stderr)
    if not frames or int(frames[1]) <= 15:
        check(False, "device-loss probe: no sufficiently long chain run")
        print(probe.stderr.decode(errors="replace")[-4000:])
        return
    check(probe.returncode == 0 and engine_tokens(probe.stdout) == engine_tokens(base_out),
          "device-loss probe: speculative tokens = plain")
    fault = int(frames[1]) - 15
    lost = eng.run(ref, dict(forced, COLI_VK_CHAIN_FAULT=str(fault)), tmp / "loss.f32")
    trace = lost.stderr.decode(errors="replace")
    released = re.search(r": ([1-9][0-9]*) dense matrices on the device only", trace)
    reloaded = re.search(r"\), ([1-9][0-9]*) read back from disk for the CPU", trace)
    rebuilt = re.search(r"rebuilding the state of ([1-9][0-9]*) positions", trace)
    check(bool(released and reloaded and rebuilt), "device loss: host matrices released, restored and recurrent state rebuilt")
    same = lost.returncode == 0 and engine_tokens(lost.stdout) == engine_tokens(base_out)
    check(same, f"device loss at frame {fault}: every token = plain decoding after CPU replay")
    if not same:
        print(lost.stdout.decode(errors="replace"))
        print(trace[-4000:])


def build_reference(eng, base_ref, n_new, tmp):
    """The fixture's prompt with a repeating tail, decoded plainly: the oracle."""
    data = json.loads(Path(base_ref).read_text(encoding="utf-8"))
    vocab = len(data.get("final_logits") or []) or 64
    prompt = list(data["prompt_ids"])
    tail = [t % vocab for t in (7, 11, 13, 17)]
    prompt += tail * 3 + tail[:2]
    probe = tmp / "probe.json"
    probe.write_text(json.dumps({"schema_version": 1, "prompt_ids": prompt, "full_ids": prompt + [0] * n_new}))
    r = eng.run(probe, {}, tmp / "probe.f32")
    gen = engine_tokens(r.stdout)
    if not gen or len(gen) != n_new:
        sys.exit("could not decode the reference:\n" + r.stdout.decode(errors="replace")
                 + r.stderr.decode(errors="replace")[-3000:])
    ref = tmp / "ref_spec.json"
    ref.write_text(json.dumps({"schema_version": 1, "prompt_ids": prompt, "full_ids": prompt + gen}))
    base = eng.run(ref, {}, tmp / "base.f32")
    if base.returncode:
        sys.exit("the plain decode does not reproduce itself:\n" + base.stdout.decode(errors="replace"))
    return ref, base.stdout, (tmp / "base.f32").read_bytes()


def gate(eng, ref, base_out, base_logits, tmp, name, spec, src=None, want=None, extra_check=None):
    dump = tmp / "spec.f32"
    if dump.exists():
        dump.unlink()
    r = eng.run(ref, spec, dump)
    same = r.returncode == 0 and r.stdout == base_out and dump.exists() and dump.read_bytes() == base_logits
    got = counts(eng.kind, src, r.stderr) if src else None
    if want is not None:
        same &= got == want
    elif src:
        same &= got is not None and got[2] <= got[3] and got[0] <= got[1]
    extra = extra_check(got, r.stderr) if extra_check else ""
    if extra is None:
        same = False
        extra = ""
    detail = f" {src} forwards/tokens/accepted/drafts {got}" if src else ""
    check(same, f"{name}: tokens and last logits = plain decoding{detail}{extra}")
    if not same:
        print(r.stdout.decode(errors="replace")[-1500:] + r.stderr.decode(errors="replace")[-2500:])
        if want is not None:
            print(f"       want {want}")
    return r


def check_mtp(eng, ref, base_out, base_logits, n_new, tmp, quick):
    print(f"mtp: {eng.fixture}")
    for depth in ((3,) if quick else (1, 2, 3)):
        modes = ["reject", "cycle", "row2"] if quick else \
            ["", "reject", "accept", "mixed", "cycle"] + [f"row{r}" for r in range(1, depth + 1)]
        for mode in modes:
            spec = {"Q38_MTP": "1", "Q38_MTP_DRAFTS": str(depth), "Q38_MTP_FORCE": mode}
            want = None
            if mode:
                m, row = (mode, 0) if not mode.startswith("row") else ("row", int(mode[3:]))
                want = expected(n_new, depth, m, row, 3)
            gate(eng, ref, base_out, base_logits, tmp, f"MTP depth {depth} {mode or 'normal'}", spec, "MTP", want)
    for mode in (() if quick else ("", "cycle")):
        gate(eng, ref, base_out, base_logits, tmp, f"MTP depth by the gate {mode or 'normal'}",
             {"Q38_MTP": "1", "Q38_MTP_DRAFTS": "auto", "Q38_MTP_FORCE": mode}, "MTP")


def check_lookup(eng, ref, base_out, base_logits, n_new, tmp, mtp, quick):
    print(f"lookup: {eng.fixture}")
    for drafts in ((5,) if quick else (5, 2)):
        modes = ["cycle", "row3"] if quick else ["accept", "mixed", "cycle"] + [f"row{r}" for r in range(1, drafts + 1)]
        for mode in modes:
            m, row = (mode, 0) if not mode.startswith("row") else ("row", int(mode[3:]))
            spec = {"COLI_LOOKUP": "1", "COLI_SPEC_GATE": "0", "COLI_LOOKUP_FORCE": mode,
                    "COLI_LOOKUP_DRAFTS": str(drafts)}
            gate(eng, ref, base_out, base_logits, tmp, f"lookup forced {mode}, {drafts} drafts", spec, "lookup",
                 expected(n_new, drafts, m, row, drafts))

    def proposed(got, _err):
        return f" (proposals on the repeating prompt: {got[3] if got else 0})" if got and got[3] > 0 else None

    gate(eng, ref, base_out, base_logits, tmp, "lookup, gate off", {"COLI_LOOKUP": "1", "COLI_SPEC_GATE": "0"},
         "lookup", extra_check=proposed)
    if quick:
        return
    gate(eng, ref, base_out, base_logits, tmp, "lookup, gate on", {"COLI_LOOKUP": "1"}, "lookup")
    if mtp:
        gate(eng, ref, base_out, base_logits, tmp, "MTP depth 3 + lookup",
             {"Q38_MTP": "1", "Q38_MTP_DRAFTS": "3", "COLI_LOOKUP": "1"}, "lookup")
        gate(eng, ref, base_out, base_logits, tmp, "MTP depth by the gate + lookup",
             {"Q38_MTP": "1", "Q38_MTP_DRAFTS": "auto", "COLI_LOOKUP": "1"}, "lookup")
        gate(eng, ref, base_out, base_logits, tmp, "MTP depth 2 + lookup, gate off",
             {"Q38_MTP": "1", "Q38_MTP_DRAFTS": "2", "COLI_LOOKUP": "1", "COLI_SPEC_GATE": "0"}, "lookup")


# ---------- serve ----------

READY = b"\x01\x01READY\x01\x01"


def serve_session(eng, spec, alphabet):
    """Turns sent one at a time. Returns (frames, stderr). A turn's prompt may be
    built from an earlier turn's reply (the transcript a chat client resends), so
    both arms run the same procedure and get the same prompts when they agree."""
    env = eng.env(dict(spec, SERVE="1", TOK=str(eng.fixture / "tokenizer.json"), COLI_PREFIX_LOG="1",
                       Q38_PREFIX_LOG="1", Q38_MAXT="128", Q36_MAXT="512"))
    p = subprocess.Popen([str(eng.path), str(eng.cap), "8"], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    err = []
    threading.Thread(target=lambda: [err.append(x) for x in iter(p.stderr.readline, b"")], daemon=True).start()
    frames, replies = [], {}
    a = alphabet
    rep = (a[:5] * 6)                      # a prompt built to repeat
    turns = [
        ("t1", lambda: rep, 24, ""),
        ("t2", lambda: rep + replies["t1"] + a[:3], 20, ""),             # the transcript, continued
        ("t3", lambda: rep + replies["t1"] + a[:3] + replies["t2"], 9, "pin=1"),
        ("t4", lambda: rep + replies["t1"] + a[:3] + replies["t2"] + a[5:8], 16, ""),
        ("t5", lambda: rep + a[1:4] * 3, 16, "logprobs=2"),
        ("t6", lambda: rep + a[1:4] * 3 + replies["t5"], 12, ""),
        ("t7", lambda: rep, 14, "", "0.9", "0.95"),
        ("t8", lambda: rep + replies["t1"] + a[:3] + replies["t2"] + a[2:4], 18, ""),
        ("t9", lambda: rep + replies["t1"] + a[:3] + replies["t2"] + a[2:4] + replies["t8"] + a[4:6], 10, ""),
    ]
    try:
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited before READY")
            if READY in line:
                break
        for t in turns:
            rid, build, max_tok, ext = t[:4]
            temp, top_p = (t[4], t[5]) if len(t) > 4 else ("0", "1")
            prompt = build()
            p.stdin.write(f"SUBMIT {rid} 0 {len(prompt)} {max_tok} {temp} {top_p}{' ' + ext if ext else ''}\n".encode()
                          + prompt + b"\n")
            p.stdin.flush()
            reply = []
            while True:
                line = p.stdout.readline()
                if not line:
                    raise RuntimeError("engine died")
                kind = line.split(b" ", 1)[0]
                if kind in (b"DATA", b"ECHO"):
                    body = p.stdout.read(int(line.split()[2]))
                    p.stdout.readline()
                    frames.append(line.rstrip() + b"|" + body)
                    if kind == b"DATA":
                        reply.append(body)
                elif kind == b"DONE":
                    frames.append(b" ".join(line.split()[:4]))
                    break
                elif kind == b"ERROR":
                    raise RuntimeError(f"engine error: {line!r}")
            replies[rid] = b"".join(reply)
        p.stdin.close()
        p.wait(timeout=120)
    except Exception as exc:   # noqa: BLE001 -- a failed gate, reported
        p.kill()
        p.wait()
        frames.append(f"harness: {exc!r}".encode())
    text = b"".join(err).decode(errors="replace")
    sanitizer_check(text, "serve " + (" ".join(f"{k}={v}" for k, v in spec.items()) or "plain"))
    return frames, text


def check_serve(eng, mtp, quick):
    print(f"serve: {eng.fixture}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prefix_serve_harness import ensure_byte_tokenizer
    ensure_byte_tokenizer(eng.fixture)
    alphabet = b"0123456789" if eng.kind == "qwen38" else b"abcdefghij"
    base, berr = serve_session(eng, {}, alphabet)
    breused = reuse_count(berr)
    check(len([f for f in base if f.startswith(b"DONE")]) == 9, f"plain session: 9 turns ({len(base)} frames)")
    # (name, environment, what the session must have done for the gate to mean something)
    arms = [("lookup, gate off", {"COLI_LOOKUP": "1", "COLI_SPEC_GATE": "0"}, "verified"),
            ("lookup, gate on", {"COLI_LOOKUP": "1"}, "")]
    if mtp:
        arms += [("MTP depth 3 + lookup, gate off", {"Q38_MTP": "1", "Q38_MTP_DRAFTS": "3", "COLI_LOOKUP": "1",
                                                     "COLI_SPEC_GATE": "0"}, "cut"),
                 ("MTP depth by the gate", {"Q38_MTP": "1", "Q38_MTP_DRAFTS": "auto"}, ""),
                 ("MTP depth 2, every draft rejected", {"Q38_MTP": "1", "Q38_MTP_DRAFTS": "2",
                                                        "Q38_MTP_FORCE": "reject"}, "")]
    if quick:
        arms = arms[:1] + arms[2:3]
    if eng.kind == "qwen36":
        check_serve_eos_mid_verify(eng, base, alphabet)
    for name, spec, need in arms:
        frames, err = serve_session(eng, spec, alphabet)
        verifies = sum(int(x) for x in re.findall(r"acceptance [0-9.]+% \(\d+/\d+ drafts in (\d+) verifies\)", err))
        mtp_drafts = sum(int(x) for x in re.findall(r"\[qwen38 MTP\] turn \S+: .*?\(\d+/(\d+) drafts\)", err))
        cut = len(re.findall(r"ended mid-verify", err))
        reused = reuse_count(err)
        meant = reused > 0 and reused == breused and (need != "verified" or verifies > 0) and (need != "cut" or cut > 0)
        check(frames == base and meant,
              f"{name}: every frame = plain ({verifies} lookup verifies, {mtp_drafts} MTP drafts, {cut} turns "
              f"ended mid-verify, {reused} turns reused a prefix, plain {breused})")
        if frames != base:
            for x, y in zip(base, frames):
                if x != y:
                    print(f"       first difference:\n         plain {x[:160]!r}\n         spec  {y[:160]!r}")
                    break
            print(err[-3000:])


def reuse_count(err):
    """Turns that reused a prefix: live state (qwen36) or a snapshot (qwen38). A
    wrong state after a speculating turn can hide behind a reuse that is refused
    (the record no longer matches, so the turn prefills from scratch and answers
    right): the sessions with drafts must reuse as often as the one without."""
    return len(re.findall(r"\[PREFIX\] reusing|\[qwen38 prefix\] request=\S+ reused=[1-9]", err))


def check_serve_eos_mid_verify(eng, base_frames, alphabet):
    """A turn that ends inside a verify (the sampled token is an end of sequence while
    drafts are pending) must leave the state of the tokens fed, which the next turn
    reuses live (qwen36 keeps the state and reuses it when the transcript extends it).
    The end-of-sequence id is picked from the first turn's reply: the first of its
    tokens that, made the end of sequence (Q36_EOS), ends a turn mid-verify."""
    reply = b"".join(f.split(b"|", 1)[1] for f in base_frames if f.startswith(b"DATA "))
    spec = {"COLI_LOOKUP": "1", "COLI_SPEC_GATE": "0"}
    tried = []
    for b in dict.fromkeys(reply):
        if b >= 128 or len(tried) >= 16:
            continue
        tried.append(b)
        frames, err = serve_session(eng, dict(spec, Q36_EOS=str(b)), alphabet)
        # a turn that ended mid-verify, and the next turn reusing its live state
        lines = [x for x in err.splitlines() if x.startswith("[PREFIX]") or "lookup] turn" in x]
        if not any("ended mid-verify" in x and nxt.startswith("[PREFIX] reusing")
                   for x, nxt in zip(lines, lines[1:])):
            continue
        plain, perr = serve_session(eng, {"Q36_EOS": str(b)}, alphabet)
        reused, preused = reuse_count(err), reuse_count(perr)
        check(frames == plain and reused > 0 and reused == preused,
              f"lookup, end of sequence {b} inside a verify, the next turn reusing the state it left: every "
              f"frame = plain, {reused} turns reused the live state (plain: {preused})")
        if frames != plain:
            for x, y in zip(plain, frames):
                if x != y:
                    print(f"       first difference:\n         plain {x[:160]!r}\n         spec  {y[:160]!r}")
                    break
        return
    check(False, f"no end-of-sequence id among {tried} ends a turn inside a verify before a turn that reuses it")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--kind", required=True, choices=("qwen38", "qwen36"))
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--ref", help="the fixture's reference (default: <fixture>/ref.json, ref_full.json for qwen36)")
    ap.add_argument("--cap", type=int, default=2)
    ap.add_argument("--n-new", type=int, default=40)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for both arms")
    ap.add_argument("--mtp", action="store_true", help="the fixture has an MTP head (qwen38)")
    ap.add_argument("--serve", action="store_true", help="the serve session too (needs a tokenizer)")
    ap.add_argument("--sanitize", action="store_true", help="fail on an ASan or UBSan report in any run's stderr")
    ap.add_argument("--require-kv-split", action="store_true", help="require every CLI run to execute attention over host KV rows")
    ap.add_argument("--device-loss", action="store_true", help="only the late device-loss gate with host copies released")
    ap.add_argument("--quick", action="store_true",
                    help="a subset (sanitizer runs): depth 3 rejected at row 1, at row 2 and cycling; lookup cycling "
                         "and at row 3; one serve arm per source")
    a = ap.parse_args()
    global SANITIZE, REQUIRE_KV_SPLIT
    SANITIZE = a.sanitize
    REQUIRE_KV_SPLIT = a.require_kv_split
    extra = dict(kv.split("=", 1) for kv in a.env)
    fixture = Path(a.fixture)
    ref = a.ref or str(fixture / ("ref_full.json" if a.kind == "qwen36" else "ref.json"))
    eng = Engine(a.engine, a.kind, fixture, a.cap, extra)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        if "COLI_USAGE" not in extra:
            eng.extra["COLI_USAGE"] = str(tmp / "usage")
        oracle, base_out, base_logits = build_reference(eng, ref, a.n_new, tmp)
        print(f"reference: {a.n_new} tokens decoded plainly{(' with ' + ' '.join(a.env)) if a.env else ''}")
        if a.device_loss:
            check_device_loss(eng, oracle, base_out, tmp)
        else:
            if a.mtp:
                check_mtp(eng, oracle, base_out, base_logits, a.n_new, tmp, a.quick)
            check_lookup(eng, oracle, base_out, base_logits, a.n_new, tmp, a.mtp, a.quick)
            if a.serve:
                check_serve(eng, a.mtp, a.quick)
    if FAILS:
        print(f"spec drafts: {len(FAILS)} check(s) failed")
        return 1
    print("spec drafts: device-loss replay preserves plain decoding's tokens" if a.device_loss else
          "spec drafts: every configuration gives plain decoding's tokens and last logits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
