#!/usr/bin/env python3
"""The qwen38 Metal chain in serve mode (tests/metal_qwen38.sh runs it; METAL=1 build).

Each case runs the chain (no off line, its report with forwards > 0) and compares the
engine's served steps (COLI_LOGIT_DUMP=1: every step's top five id:logit) with a fresh
engine's on the same text: the same ids, the logits within 1e-4 of the largest. Its
replies alone would not tell: the tiny fixture's greedy replies barely move with the
state. Served K/V are rounded to f16 (q38_kv16), on both sides.
serve two-turn: the second turn resends the first prompt and its reply, so the engine
  continues from its live state on the GPU and reads only the new text; it must answer as
  a fresh engine given the whole conversation.
serve regenerate: the first prompt again: the prompt-end snapshot (the state copied down,
  q38m_sync_host) goes back on the host (q38m_host_wrote), not the GPU's live state.
serve park: a short request between two turns parks the conversation (its rows and state
  copied down) and brings it back (q38m_rollback, q38m_host_wrote); the next turn answers as
  a fresh engine, and the short request as it would alone.
kv slots: KV_SLOTS=2 in serve mode turns the chain off with its one line.

--segments <dir> (a fixture with a tokenizer and 2048 positions; metal_qwen38.sh gives it
  qwen38_decay's weights) runs only this case: a 1,020-token prompt read in segments of 300
  rows (COLI_PROMPT_SEGMENT=300, the serve's own prompt segments), then 8 tokens decoded
  across position 1024. The device caches grow with valid rows twice (512 -> 1024 at the
  second segment, 300 rows; 1024 -> 2048 in decode, 1024 rows), each growth dropping the
  device's rows and uploading them again from the host; the report's `kv:` line must count
  both. Every served step against the CPU engine's (COLI_METAL=0), both in reference
  precision (Ruling P2-4) with the same segments: the same ids, logits within 1e-4.

Prints `ok   [...]` / `FAIL [...]` lines like the shell test; exit 1 on a failure.
"""
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import ServeEngine  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / "qwen38"
FIXTURE = HERE / "qwen38_tiny"
OFF = "[qwen38-metal] the Metal chain is off: "
REPORT = re.compile(r"\[metal\] qwen38 chain: (\d+) forwards, (\d+) declined")
LOGITS = re.compile(r"^\[LOGITS\] (.*)$", re.M)
OPENING = b"12+34=46; 7*8=56; 9-3="
QUESTION = b"; 81/9="
SIDE = b"5*5="

failed = 0


def say(ok, what, detail=""):
    global failed
    print(("ok   [%s] %s" if ok else "FAIL [%s] %s") % (what, detail))
    if not ok:
        failed = 1


class Engine:
    """A served engine whose steps' logits are split by request."""

    def __init__(self, tok, **extra):
        env = {"SNAP": str(FIXTURE), "SERVE": "1", "TOK": str(tok), "OMP_NUM_THREADS": "2",
               "COLI_NO_OMP_TUNE": "1", "COLI_CUDA": "0", "Q38_MAXT": "128", "COLI_LOGIT_DUMP": "1"}
        env.update(extra)
        self.e = ServeEngine(ENGINE, ["1", "8"], env)
        self.steps = []   # the [LOGITS] lines of each request

    def ask(self, rid, prompt, max_tok=8, temp=None):
        before = len(LOGITS.findall("".join(self.e._err)))
        reply = self.e.ask(rid, prompt, max_tok=max_tok) if temp is None else self.ask_temp(rid, prompt, max_tok, temp)
        # stderr is drained by a thread: wait until its step lines stop coming
        n, still = -1, 0
        while still < 5:
            time.sleep(0.05)
            m = len(LOGITS.findall("".join(self.e._err)))
            still = still + 1 if m == n else 0
            n = m
        self.steps.append(LOGITS.findall("".join(self.e._err))[before:])
        return reply

    def ask_temp(self, rid, prompt, max_tok, temp):
        """ServeEngine.ask at a temperature: a greedy request (temperature 0) takes the step's
        argmax without the dump, so only its first step's logits would be seen; logprobs
        would read the prompt in one call (no segments). A temperature of 1e-4 samples the
        argmax (rand() unseeded: the same draws in both engines) through serve_sample, which
        dumps every step."""
        p = self.e.p
        p.stdin.write(f"SUBMIT {rid} 0 {len(prompt)} {max_tok} {temp} 1\n".encode() + prompt + b"\n")
        p.stdin.flush()
        chunks = []
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError("engine closed mid-request:\n" + "".join(self.e._err)[-2000:])
            text = line.decode("latin-1").rstrip("\n")
            kind = text.split(" ", 1)[0]
            if kind == "DATA":
                chunks.append(p.stdout.read(int(text.split()[2])))
                p.stdout.readline()
            elif kind in ("DONE", "END"):
                return b"".join(chunks)
            elif kind == "ERROR":
                raise RuntimeError(f"engine error: {text}")

    def close(self):
        return self.e.close()


def parse(line):
    return [(int(i), float(v)) for i, v in (p.split(":") for p in line.split())]


def same_steps(a, b):
    """The same ids, logits within 1e-4 of the largest: (ok, worst)."""
    if len(a) != len(b) or not a:
        return False, "%d steps vs %d" % (len(a), len(b))
    worst = 0.0
    for la, lb in zip(a, b):
        pa, pb = parse(la), parse(lb)
        if [i for i, _ in pa] != [i for i, _ in pb]:
            return False, "ids %s vs %s" % (la, lb)
        scale = max(abs(v) for _, v in pb) or 1.0
        worst = max(worst, max(abs(x - y) for (_, x), (_, y) in zip(pa, pb)) / scale)
    return worst <= 1e-4, "worst %.3g" % worst


def chain_ran(what, log):
    runs = [int(m.group(1)) for m in REPORT.finditer(log)]
    ok = OFF not in log and runs and runs[-1] > 0
    if not ok:
        say(False, what + " chain on", "report %s" % (runs[-1] if runs else "none"))
        print("\n".join(line for line in log.splitlines() if "metal" in line))
    return ok


def fresh(tok, prompt):
    e = Engine(tok, COLI_KV_PREFIX="0")
    reply = e.ask("1", prompt)
    log = e.close()
    return reply, e.steps[0], log


SEG_ENV = {"GQ_EXACT": "1", "Q38_TRUNK_CPU_INT8": "0", "Q38_NATIVE_BF16": "1", "COLI_PROMPT_SEGMENT": "300",
           "Q38_MAXT": "2048"}
KV = re.compile(r"\[metal\] qwen38 kv: device caches grown (\d+) times to (\d+) rows, (\d+) of them with valid rows "
                r"\((\d+) rows uploaded again from the host\)")


def segments(fixture):
    global FIXTURE
    FIXTURE = Path(fixture)
    x, prompt = 4711, bytearray()
    for _ in range(1020):   # bytes 33..63: one token each in the fixture's byte vocabulary
        x = (x * 1103515245 + 12345) % 2**31
        prompt.append(33 + x % 31)
    prompt = bytes(prompt)
    runs = {}
    for name, extra in (("cpu", {"COLI_METAL": "0"}), ("metal", {"QWEN38_METAL_EXACT": "1"})):
        e = Engine(FIXTURE / "tokenizer.json", **SEG_ENV, **extra)
        reply = e.ask("1", prompt, temp="0.0001")
        runs[name] = (reply, e.steps[0], e.close())
    log = runs["metal"][2]
    if not chain_ran("serve segments", log):
        return failed
    kv = KV.findall(log)
    grown = [int(v) for v in kv[-1]] if kv else None
    ok, detail = same_steps(runs["metal"][1], runs["cpu"][1])
    regrown = grown is not None and grown[2] >= 2 and grown[3] >= 1024 + 300 and grown[1] >= 2048
    fw = [int(m.group(1)) for m in REPORT.finditer(log)]
    regrown = regrown and fw and fw[-1] >= 4 + 7   # four segments, seven decode steps
    say(ok and regrown and runs["metal"][0] == runs["cpu"][0] and len(runs["metal"][1]) == 8,
        "serve segments 1020 tokens, kv growth with valid rows",
        "reply %r vs the CPU's %r, %d steps %s, %s forwards; kv grown %s" % (runs["metal"][0], runs["cpu"][0], len(runs["metal"][1]), detail,
                                                                        fw[-1] if fw else "no",
                                                               "%d times to %d rows, %d with valid rows (%d rows uploaded again)"
                                                               % tuple(grown) if grown else "no kv line"))
    if not regrown:
        print("\n".join(line for line in log.splitlines() if "metal" in line or "segment" in line))
    return failed


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--segments":
        return segments(sys.argv[2])
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, str(HERE / "tools" / "make_edge_tiny_tokenizer.py"), "--vocab-size", "64", d],
                       check=True, stdout=subprocess.DEVNULL)
        tok = Path(d) / "tokenizer.json"

        # two turns: the live state
        warm = Engine(tok)
        reply = warm.ask("1", OPENING)
        second = OPENING + reply + QUESTION
        got = warm.ask("2", second)
        log = warm.close()
        want, want_steps, cold_log = fresh(tok, second)
        if chain_ran("serve two-turn", log) and chain_ran("serve two-turn fresh", cold_log):
            ok, detail = same_steps(warm.steps[1], want_steps)
            say(ok and got == want and "[PREFIX] reusing" in log, "serve two-turn",
                "turn 2 %r, fresh %r, steps %s" % (got, want, detail))
        cpu = Engine(tok, COLI_METAL="0")
        cpu_reply = cpu.ask("1", second)
        cpu.close()
        ok, detail = same_steps(cpu.steps[0], want_steps)
        print("info [serve two-turn] the CPU engine (default mode) on the conversation: %r, steps %s"
              % (cpu_reply, detail))

        # the same prompt again: the prompt-end snapshot
        warm = Engine(tok)
        first = warm.ask("1", OPENING)
        again = warm.ask("2", OPENING)
        log = warm.close()
        want, want_steps, cold_log = fresh(tok, OPENING)
        if chain_ran("serve regenerate", log):
            ok, detail = same_steps(warm.steps[1], want_steps)
            say(ok and again == want == first and "the prompt-end snapshot" in log, "serve regenerate",
                "%r, fresh %r, steps %s" % (again, want, detail))

        # a short request between two turns: the conversation parked and back
        warm = Engine(tok, COLI_PARK_MIN="16")
        reply = warm.ask("1", OPENING)
        side = warm.ask("2", SIDE)
        second = OPENING + reply + QUESTION
        got = warm.ask("3", second)
        log = warm.close()
        if chain_ran("serve park", log):
            want, want_steps, _ = fresh(tok, second)
            side_want, side_steps, _ = fresh(tok, SIDE)
            ok, detail = same_steps(warm.steps[2], want_steps)
            ok2, detail2 = same_steps(warm.steps[1], side_steps)
            parked = "conversation parked for a %d-token request" % len(SIDE) in log and "conversation is back" in log
            say(parked and ok and ok2 and got == want and side == side_want, "serve park",
                "turn 2 steps %s, side request steps %s%s" % (detail, detail2, "" if parked else ", not parked"))

        slots = ServeEngine(ENGINE, ["1", "8"], {"SNAP": str(FIXTURE), "SERVE": "1", "TOK": str(tok),
                                                 "OMP_NUM_THREADS": "2", "COLI_NO_OMP_TUNE": "1", "KV_SLOTS": "2"})
        slots.ask("1", OPENING, max_tok=2)
        slog = slots.close()
        line = OFF + "KV_SLOTS>1 (one conversation on the GPU); the CPU engine runs the model"
        n = slog.count(line)
        say(n == 1 and slog.count(OFF) == 1, "serve KV_SLOTS=2", "off line %d time(s)" % n)
    return failed


if __name__ == "__main__":
    sys.exit(main())
