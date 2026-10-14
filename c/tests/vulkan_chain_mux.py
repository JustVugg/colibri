#!/usr/bin/env python3
"""The dense chain's multiplexed decode (KV_SLOTS, glm_chain.h's glmc_forward_rows):
several conversations at once on the CPU and with the chain, every DATA and ECHO frame
compared.

Every request of a wave goes in one write, each on its own KV slot: the engine admits
one a loop and decodes the active ones as one batch, so the batches mix rows of
different conversations at different positions, grow as requests come in and shrink as
they finish. The second wave goes back to each slot: one turn extends its conversation,
one goes back in it, one starts over. Token ids and texts must be the CPU's exactly,
the logprobs printed with them within CHAIN_SERVE_TOL (default 1e-4), and the chain's
report must show multiplexed steps of more than one row (CHAIN_MUX_STEPS=0: none, the
chain declining them; the chain's other forwards still there). CHAIN_MUX_EXPECT, a
regular expression the chain session's stderr must match; CHAIN_MUX_LOG, a file to keep
that stderr in.

usage: vulkan_chain_mux.py <engine> <snapshot> <slots> [KEY=VALUE ...]   (extra environment)
"""
import os
import re
import subprocess
import sys
import threading

READY = b"\x01\x01READY\x01\x01"


def waves(n):
    base = bytes(range(65, 65 + 23))
    one = [(base[:9 + 4 * i] + bytes([97 + i]) * (i + 1), 5 + 2 * i, "logprobs=2" if i % 2 else "") for i in range(n)]
    two = []
    for i, (prompt, _, _) in enumerate(one):
        if i % 3 == 0:
            two.append((prompt + b"xyz", 6, ""))          # after the reply: the prompt and a new tail
        elif i % 3 == 1:
            two.append((prompt[:6] + b"QRS", 5, "logprobs=1"))   # back inside the prompt
        else:
            two.append((b"zz" + prompt, 4, ""))           # a new conversation on the slot
    return [one, two]


def session(engine, snap, slots, extra, chain):
    env = dict(os.environ, SNAP=snap, SERVE="1", KV_SLOTS=str(slots), **extra)
    if chain:
        env.update(COLI_VULKAN="1", COLI_VK_CHAIN=os.environ.get("COLI_VK_CHAIN", "1"), COLI_VK_TIER_SYNC="1",
                   COLI_USAGE=os.path.abspath("chain-mux.usage"))
        try:
            os.remove(env["COLI_USAGE"])
        except FileNotFoundError:
            pass
    p = subprocess.Popen([engine, "8", "8"], env=env, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = []
    threading.Thread(target=lambda: [err.append(l) for l in iter(p.stderr.readline, b"")], daemon=True).start()
    while True:
        line = p.stdout.readline()
        if not line:
            sys.exit("engine exited before READY:\n" + b"".join(err).decode(errors="replace")[-3000:])
        if READY in line:
            break
    frames = {}
    rid = 0
    for wave in waves(slots):
        batch, ids = b"", []
        for slot, (prompt, max_tok, ext) in enumerate(wave):
            rid += 1
            ids.append(rid)
            frames[rid] = []
            batch += (f"SUBMIT {rid} {slot} {len(prompt)} {max_tok} 0 1 0" + (f" {ext}" if ext else "") +
                      "\n").encode() + prompt + b"\n"
        p.stdin.write(batch)
        p.stdin.flush()
        left = set(ids)
        while left:
            line = p.stdout.readline()
            if not line:
                sys.exit("engine died:\n" + b"".join(err).decode(errors="replace")[-3000:])
            f = line.split()
            kind = f[0] if f else b""
            if kind in (b"DATA", b"ECHO"):
                body = p.stdout.read(int(f[2]))
                p.stdout.readline()
                frames[int(f[1])].append(line.rstrip() + b" " + body)
            elif kind == b"DONE":
                frames[int(f[1])].append(b"DONE")
                left.discard(int(f[1]))
            elif kind == b"ERROR":
                sys.exit(f"engine error: {line!r}")
    p.stdin.close()
    p.wait(timeout=120)
    return frames, b"".join(err).decode(errors="replace")


TOL = float(os.environ.get("CHAIN_SERVE_TOL", "1e-4"))
WORST = [0.0]


def close(a, b):
    if a == b:
        return True
    fa, fb = a.split(b" "), b.split(b" ")
    if len(fa) != len(fb):
        return False
    for x, y in zip(fa, fb):
        if x == y:
            continue
        try:
            if b"." not in x or abs(float(x) - float(y)) > TOL:
                return False
            WORST[0] = max(WORST[0], abs(float(x) - float(y)))
        except ValueError:
            return False
    return True


def main():
    engine, snap, slots = sys.argv[1], sys.argv[2], int(sys.argv[3])
    extra = dict(a.split("=", 1) for a in sys.argv[4:])
    cpu, err_cpu = session(engine, snap, slots, extra, False)
    dev, err = session(engine, snap, slots, extra, True)
    for text in (err_cpu, err):   # a sanitized build reports into the engine's stderr
        if "ERROR: AddressSanitizer" in text or "runtime error:" in text:
            sys.exit("FAIL: sanitizer diagnostic\n" + text[-4000:])
    if os.environ.get("CHAIN_MUX_LOG"):
        with open(os.environ["CHAIN_MUX_LOG"], "w") as f:
            f.write(err)
    mux = re.findall(r"chain: (\d+) multiplexed steps \((\d+) rows\)", err)
    none = os.environ.get("CHAIN_MUX_STEPS") == "0"
    if none and (mux or not re.search(r"chain: [1-9]\d* forwards", err)):
        sys.exit("FAIL: the chain ran multiplexed steps, or nothing at all\n" + err[-3000:])
    if not none and (not mux or not any(int(r) > int(s) for s, r in mux)):
        sys.exit("FAIL: the chain ran no multiplexed step of several rows\n" + err[-3000:])
    expect = os.environ.get("CHAIN_MUX_EXPECT")
    if expect and not re.search(expect, err):
        sys.exit(f"FAIL: the chain session's stderr has no {expect!r}\n" + err[-3000:])
    n = 0
    for rid in sorted(cpu):
        a, b = cpu[rid], dev.get(rid, [])
        if len(a) != len(b) or not all(close(x, y) for x, y in zip(a, b)):
            for x, y in zip(a, b):
                if not close(x, y):
                    print(f"  request {rid} cpu: {x[:200]!r}\n  request {rid} vk : {y[:200]!r}")
                    break
            sys.exit(f"FAIL: request {rid}'s frames differ from the CPU's ({len(a)} vs {len(b)})")
        n += len(a)
    line = [l for l in err.splitlines() if "multiplexed steps" in l or (none and " forwards" in l)][-1]
    print(f"OK mux {os.path.basename(snap)} slots={slots} {' '.join(sys.argv[4:])}: {len(cpu)} requests, "
          f"{n} frames = CPU (numbers within {WORST[0]:.1e}); {line.split('] ', 1)[1]}")


if __name__ == "__main__":
    main()
