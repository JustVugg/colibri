#!/usr/bin/env python3
"""Several conversations at once (KV_SLOTS=n): every request's frames against the same
request served alone.

The multiplexed session gets each wave of requests in one write, every request on its
own cache slot, so the engine decodes them as one batch of rows at different positions
that grows as requests come in and shrinks as they end. The second wave goes back to
each slot: one turn extends its conversation, one goes back inside it, one starts over;
one request of it is stopped and one cancelled after their first token. The reference
serves each slot's requests in a session of its own with KV_SLOTS=1, one after the
other, so each turn finds the prompt cache its conversation's turns left there. Greedy sampling throughout: every DATA frame, its text and its
logprobs, must be the reference's, byte for byte (the CPU kernels give a row the same
bits in a batch of any size), and the engine's report must show decode steps of more
than one row.

usage: serve_mux_check.py <engine> <snapshot> <slots> [KEY=VALUE ...]   (extra environment)
MUX_LONG=k: prompts k times longer. MUX_TOL: compare logprobs within this bound instead of byte for byte (an engine whose
batched kernels sum in another order). MUX_IDS=numeric: request ids 1, 2, ... instead of
r0, r1, ...; MUX_GBYTES=1: the grammar-bytes field (0) before the options (colibri).
"""
import os
import re
import subprocess
import sys
import threading

READY = b"\x01\x01READY\x01\x01"


def waves(n):
    # MUX_LONG=k: the prompts k times longer (an engine whose state changes shape past
    # a few dozen positions: compression groups, index top-k, windows that wrap)
    k = int(os.environ.get("MUX_LONG", "1"))
    base = bytes(range(65, 65 + 23)) * 3 * k
    one = [(base[:(7 + 3 * i) * k] + bytes([97 + i]) * (i + 1), 4 + 2 * (i % 3), "logprobs=2" if i % 2 else "")
           for i in range(n)]
    two = []
    for i, (prompt, _, _) in enumerate(one):
        if i % 3 == 0:
            two.append((prompt + b"xyz", 6, ""))                 # after the reply: the prompt and a new tail
        elif i % 3 == 1:
            two.append((prompt[:6] + b"QRS", 5, "logprobs=1"))   # back inside the prompt
        else:
            two.append((b"zz" + prompt, 5, ""))                  # a new conversation on the slot
    return [one, two]


class Session:
    def __init__(self, engine, snap, slots, extra):
        env = dict(os.environ, SNAP=snap, SERVE="1", SERVE_BATCH="1", KV_SLOTS=str(slots), **extra)
        self.p = subprocess.Popen([engine, "8", "8"], env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.err = []
        threading.Thread(target=lambda: [self.err.append(l) for l in iter(self.p.stderr.readline, b"")],
                         daemon=True).start()
        while True:
            line = self.p.stdout.readline()
            if not line:
                self.die("engine exited before READY")
            if READY in line:
                break
        self.n = 0

    def die(self, why):
        sys.exit(why + ":\n" + b"".join(self.err).decode(errors="replace")[-4000:])

    def rid(self):
        self.n += 1
        return str(self.n) if os.environ.get("MUX_IDS") == "numeric" else f"r{self.n}"

    def header(self, rid, slot, prompt, max_tok, ext):
        return (f"SUBMIT {rid} {slot} {len(prompt)} {max_tok} 0 1" +
                (" 0" if os.environ.get("MUX_GBYTES") == "1" else "") +
                (f" {ext}" if ext else "") + "\n").encode() + prompt + b"\n"

    def collect(self, ids, out, controls=None):
        """Frames until every request in ids is DONE or ERROR. controls: {id: b"STOP"
        or b"CANCEL"}, sent after that request's first DATA frame."""
        open_ids, sent = set(ids), set()
        while open_ids:
            line = self.p.stdout.readline()
            if not line:
                self.die("engine died")
            parts = line.split()
            kind = parts[0] if parts else b""
            if kind in (b"DATA", b"ECHO"):
                rid = parts[1].decode()
                body = self.p.stdout.read(int(parts[2]))
                self.p.stdout.readline()
                out.setdefault(rid, []).append(line.rstrip() + b" " + body)
                if controls and rid in controls and rid not in sent:
                    sent.add(rid)
                    self.p.stdin.write(controls[rid] + b" " + rid.encode() + b"\n")
                    self.p.stdin.flush()
            elif kind == b"DONE":
                rid = parts[1].decode()
                out.setdefault(rid, []).append(b"DONE")
                open_ids.discard(rid)
            elif kind == b"ERROR":
                rid = parts[1].decode()
                out.setdefault(rid, []).append(b"ERROR " + b" ".join(parts[2:]))
                open_ids.discard(rid)

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=600)
        return b"".join(self.err).decode(errors="replace")


def main():
    engine, snap, slots = sys.argv[1], sys.argv[2], int(sys.argv[3])
    extra = dict(a.split("=", 1) for a in sys.argv[4:])
    plan = waves(slots)
    # the reference: each slot's conversation alone, in a session of its own, so its
    # turns find the prompt cache its turns left (what a slot keeps between turns)
    want = [None] * sum(len(w) for w in plan)
    for slot in range(slots):
        ref = Session(engine, snap, 1, extra)
        for w, wave in enumerate(plan):
            prompt, max_tok, ext = wave[slot]
            rid = ref.rid()
            ref.p.stdin.write(ref.header(rid, 0, prompt, max_tok, ext))
            ref.p.stdin.flush()
            got = {}
            ref.collect([rid], got)
            want[w * slots + slot] = got[rid]
        ref.close()
    # the multiplexed session: a wave in one write
    mux = Session(engine, snap, slots, extra)
    got_all, order, controls = {}, [], {}
    for w, wave in enumerate(plan):
        blob, ids = b"", []
        for slot, (prompt, max_tok, ext) in enumerate(wave):
            rid = mux.rid()
            ids.append(rid)
            blob += mux.header(rid, slot, prompt, max_tok, ext)
        order += ids
        if w == 1 and len(ids) >= 3:   # the second wave: one stopped, one cancelled after their first token
            controls = {ids[0]: b"STOP", ids[2]: b"CANCEL"}
        mux.p.stdin.write(blob)
        mux.p.stdin.flush()
        mux.collect(ids, got_all, controls)
    err = mux.close()
    tol = float(os.environ.get("MUX_TOL", "0"))
    bad = []
    def strip(frames):   # the request id differs between the sessions
        return [f.split(b" ", 2)[2] if f.startswith((b"DATA", b"ECHO")) else f for f in frames]
    for k, rid in enumerate(order):
        a, b = strip(want[k]), strip(got_all.get(rid, []))
        if rid in controls:
            # stopped or cancelled after its first token: what it sent is the start of the
            # reference's, and it has ended, unless it was through before the command arrived
            body, end = b[:-1], b[-1:]
            full = b == a
            # the bytes it sent, not the frames: an engine flushes the partial UTF-8 it
            # held when a request ends early, a frame the reference split otherwise
            def text(frames):
                return b"".join(f.split(b" ", 1)[1] if b" " in f else b"" for f in frames
                                if not f.startswith((b"DONE", b"ERROR")))
            if not full and body and text(body) == text(a)[:len(text(body))]:
                body = a[:len(body)]
            # it ends as the engine ends one alone: DONE or ERROR CANCELLED (qwen36 treats a
            # STOP as a cancel, OLMoE ends a cancel with DONE)
            ends = [[b"DONE"], [b"ERROR CANCELLED"]]
            ok = len(body) >= 1 and body == a[:len(body)] and (full or end in ends)
            if not ok:
                bad.append(f"{rid}: {controls[rid].decode()} after the first token gave {b}, the reference {a}")
            continue
        if tol and len(a) == len(b):
            same = all(x == y or close(x, y, tol) for x, y in zip(a, b))
        else:
            same = a == b
        if not same:
            bad.append(f"{rid} (request {k}): {b} != {a}")
    if bad:
        print("\n".join(bad))
        print(err[-3000:])
        sys.exit(f"FAIL: {len(bad)} of {len(order)} requests differ from the same request served alone")
    m = re.search(r"(\d+) decode steps, (\d+) rows", err)
    if not m or int(m.group(2)) <= int(m.group(1)):
        print(err[-3000:])
        sys.exit("FAIL: no decode step of more than one row (the report says " +
                 (m.group(0) if m else "nothing") + ")")
    print(f"OK {engine} KV_SLOTS={slots}: {len(order)} requests as they are alone, {m.group(0)}")


def close(x, y, tol):
    """Two DATA frames that differ only in the last digits of their logprobs."""
    num = re.compile(rb"-?\d+\.\d+")
    if num.sub(b"#", x) != num.sub(b"#", y):
        return False
    return all(abs(float(p) - float(q)) <= tol for p, q in zip(num.findall(x), num.findall(y)))


if __name__ == "__main__":
    main()
