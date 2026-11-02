#!/usr/bin/env python3
"""The dense chain in serve mode (COLI_VK_CHAIN=1, vk_chain.c): one session of turns
on the CPU and the same session with the chain, every DATA and ECHO frame compared.

The turns exercise what moves the chain's device state around: a pinned snapshot
(SUBMIT pin=1) and turns that restore it, the prompt cache a following turn extends,
a prompt that diverges and one that starts over, a cache that grows between turns,
and the prefill read-out (logprobs=k, ECHO frames). Token ids and texts must be the
CPU's exactly; the logprobs printed with them may differ in their last digits (the
device sums in another order), so numbers compare within 1e-4 (CHAIN_SERVE_TOL in the
environment sets another bound: a fixture whose logits run to the hundreds, as MiMo's
does, prints logprobs whose rounding is larger in absolute terms; the OK line gives the
largest difference seen). CHAIN_SERVE_EXPECT, a regular expression the chain
session's stderr must match (a lost device's rebuild, say, with COLI_VK_CHAIN_FAULT).

usage: vulkan_chain_serve.py <engine> <snapshot> [KEY=VALUE ...]   (extra environment)
COLI_VK_CHAIN in the caller's environment picks the chain's mode (default 1; 2 runs
prompts on the device and decode on the CPU, the state moving between them).
CHAIN_SERVE_PAIRS=1 with CHAIN_SERVE_SLOTS=n sends the turns n at a time, one a slot,
together: their decode rows run in the same steps (several conversations in one forward).
CHAIN_SERVE_LONG=n (with CHAIN_SERVE_PAIRS) puts n bytes before each group's first prompt.
CHAIN_SERVE_TEXT=1 compares each frame's kind, id and text only (a device lost mid-session:
the state rebuilt on the CPU sums in another order, the numbers move in their last digits).
CHAIN_SERVE_BACKEND=cuda runs the chain session on the CUDA chain instead (COLI_CUDA=1).
CHAIN_SERVE_DIALECT picks the SUBMIT header: unset, ids r0, r1, ... and the options
straight after top_p (what the qwen engines have always been sent); "numeric", ids 1,
2, ... (glm53); "colibri", numeric ids and the grammar-bytes field (0) before the
options (colibri's multiplexed serve). CHAIN_SERVE_SLOTS=n sends request i to KV slot
i % n (the engine needs KV_SLOTS=n): the turns move between sessions, and with them
whatever state the chain keeps on the device. CHAIN_SERVE_INTERLEAVE=1 with it keeps the
turns on slot 0 and runs another conversation's short turn on the other slots between two
of them: a turn then reuses its own conversation's prompt after the device held another's. CHAIN_SERVE_CONTINUE=1 adds a last turn
that continues the one before it exactly (its prompt, what it wrote, two bytes more) and
asks for enough tokens to grow the cache: the state is reused while the KV mirrors are
made again from the host's rows, which then have to hold what the device wrote.
The KV cache stays float32 here (Q36_KV=f32, Q38_KV=f32, KV16=0): the engines' default float16 rounds a
value the CPU and the device computed a few bits apart into two halves an ulp apart, about
1e-3 in a logprob. CHAIN_SERVE_KV16=1 keeps the float16 cache (the device rounds its rows as
the host does) and compares the logprobs within 5e-3; token ids and texts stay exact.
"""
import os
import re
import subprocess
import sys
import threading

READY = b"\x01\x01READY\x01\x01"


def session(engine, snap, extra, chain):
    kv = {} if KV16 else {"Q36_KV": "f32", "Q38_KV": "f32", "KV16": "0"}   # float16 only when asked (the docstring says why)
    env = dict(os.environ, SNAP=snap, SERVE="1", COLI_PREFIX_LOG="1", Q38_PREFIX_LOG="1", **kv, **extra)
    if chain and os.environ.get("CHAIN_SERVE_BACKEND") == "cuda":   # the CUDA chain (a CUDA=1 build)
        env.update(COLI_VULKAN="0", COLI_CUDA="1", COLI_USAGE=os.path.abspath("chain-serve.usage"))
        try:
            os.remove(env["COLI_USAGE"])
        except FileNotFoundError:
            pass
    elif chain:
        env.update(COLI_VULKAN="1", COLI_VK_CHAIN=os.environ.get("COLI_VK_CHAIN", "1"), COLI_VK_TIER_SYNC="1",
                   COLI_USAGE=os.path.abspath("chain-serve.usage"))
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
    base = bytes(range(65, 65 + 23))
    turns = [(base, 6, "pin=1 logprobs=3"), (base + b"xyz", 6, "logprobs=2"), (base + b"QRS", 6, ""),
             (base + b"QRS" + b"tuvw", 6, ""), (b"zzz" + base, 5, "logprobs=2"), (base + b"QRS", 4, "pin=1"),
             (base + b"QRSk", 4, "")]
    if os.environ.get("CHAIN_SERVE_CONTINUE") == "1":
        turns.append(None)
    nslots = int(os.environ.get("CHAIN_SERVE_SLOTS", "1"))
    if os.environ.get("CHAIN_SERVE_PAIRS") == "1" and nslots > 1:
        return pairs(p, err, turns, nslots)
    interleave = os.environ.get("CHAIN_SERVE_INTERLEAVE") == "1" and nslots > 1
    sched = []   # (turn index, slot); -1: a foreign turn on another slot between two of slot 0's
    for i in range(len(turns)):
        sched.append((i, 0 if interleave else i % nslots))
        if interleave and i + 1 < len(turns):
            sched.append((-1, 1 + i % (nslots - 1)))
    out = []
    written = last = b""
    for j, (i, slot) in enumerate(sched):
        if i < 0:   # another conversation's turn: the device's state moves away and back
            prompt, max_tok, ext = b"abc" + bytes([97 + j % 26]) * (5 + j % 7), 3, ""
        else:
            if turns[i] is None:   # the turn before, what it wrote and two bytes more; a cache that grows
                turns[i] = (turns[i - 1][0] + last + b"mn", 12, "")
            prompt, max_tok, ext = turns[i]
        written = b""
        dialect = os.environ.get("CHAIN_SERVE_DIALECT", "")
        rid = f"{j + 1}" if dialect in ("numeric", "colibri") else f"r{j}"
        hdr = (f"SUBMIT {rid} {slot} {len(prompt)} {max_tok} 0 1" + (" 0" if dialect == "colibri" else "") +
               (f" {ext}" if ext else "") + "\n")
        p.stdin.write(hdr.encode() + prompt + b"\n")
        p.stdin.flush()
        while True:
            line = p.stdout.readline()
            if not line:
                sys.exit("engine died:\n" + b"".join(err).decode(errors="replace")[-3000:])
            kind = line.split(b" ", 1)[0]
            if kind in (b"DATA", b"ECHO"):
                body = p.stdout.read(int(line.split()[2]))
                p.stdout.readline()
                out.append(line.rstrip() + b" " + body)
                if kind == b"DATA":
                    written += body
            elif kind == b"DONE":
                out.append(b"DONE " + line.split()[1])
                if i >= 0:
                    last = written
                break
            elif kind == b"ERROR":
                sys.exit(f"engine error: {line!r}")
    p.stdin.close()
    p.wait(timeout=120)
    return out, b"".join(err).decode(errors="replace")


def pairs(p, err, turns, nslots):
    """CHAIN_SERVE_PAIRS=1: the turns go nslots at a time, one a slot, written together, so
    the engine prefills them one after the other and decodes their rows in the same steps
    (several conversations in one forward). The frames come in the engine's order, which
    the CPU's session and the chain's share."""
    turns = [t for t in turns if t is not None]
    out = []
    for k in range(0, len(turns), nslots):
        group = turns[k:k + nslots]
        wire = b""
        for j, (prompt, max_tok, ext) in enumerate(group):
            if j == 0 and LONG:   # a conversation past the places' first mirrors (they grow)
                prompt = bytes(97 + (t * 7) % 26 for t in range(LONG)) + prompt
            hdr = f"SUBMIT r{k + j} {j} {len(prompt)} {max_tok + 8 * j} 0 1" + (f" {ext}" if ext else "") + "\n"
            wire += hdr.encode() + prompt + b"\n"
        p.stdin.write(wire)
        p.stdin.flush()
        done = 0
        while done < len(group):
            line = p.stdout.readline()
            if not line:
                sys.exit("engine died:\n" + b"".join(err).decode(errors="replace")[-3000:])
            kind = line.split(b" ", 1)[0]
            if kind in (b"DATA", b"ECHO"):
                body = p.stdout.read(int(line.split()[2]))
                p.stdout.readline()
                out.append(line.rstrip() + b" " + body)
            elif kind == b"DONE":
                out.append(b"DONE " + line.split()[1])
                done += 1
            elif kind == b"ERROR":
                sys.exit(f"engine error: {line!r}")
    p.stdin.close()
    p.wait(timeout=120)
    return out, b"".join(err).decode(errors="replace")


KV16 = os.environ.get("CHAIN_SERVE_KV16") == "1"
TOL = float(os.environ.get("CHAIN_SERVE_TOL", "5e-3" if KV16 else "1e-4"))
WORST = [0.0]


TEXT = os.environ.get("CHAIN_SERVE_TEXT") == "1"
LONG = int(os.environ.get("CHAIN_SERVE_LONG", "0"))


def close(a, b):
    if a == b:
        return True
    fa, fb = a.split(b" "), b.split(b" ")
    if TEXT:   # the frame's kind, id and text; its numbers may differ (a state rebuilt on the CPU)
        return len(fa) == len(fb) and fa[:2] == fb[:2] and fa[-1] == fb[-1]
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
    engine, snap = sys.argv[1], sys.argv[2]
    extra = dict(a.split("=", 1) for a in sys.argv[3:])
    cpu, err_cpu = session(engine, snap, extra, False)
    dev, err = session(engine, snap, extra, True)
    for text in (err_cpu, err):   # a sanitized build reports into the engine's stderr
        if "ERROR: AddressSanitizer" in text or "runtime error:" in text:
            sys.exit("FAIL: sanitizer diagnostic\n" + text[-4000:])
    forwards = [l for l in err.splitlines() if (" chain: " in l or l.startswith("[chain]")) and "forwards" in l]
    reuse = [l for l in err.splitlines() if "[PREFIX] reusing" in l or ("prefix] request" in l and "reused=0/" not in l)]
    pins = err.count("[PIN]")
    if not forwards:
        sys.exit("FAIL: the chain never ran\n" + err[-3000:])
    expect = os.environ.get("CHAIN_SERVE_EXPECT")
    if expect and not re.search(expect, err):
        sys.exit(f"FAIL: the chain session's stderr has no {expect!r}\n" + err[-3000:])
    if len(cpu) != len(dev) or not all(close(a, b) for a, b in zip(cpu, dev)):
        for a, b in zip(cpu, dev):
            if not close(a, b):
                print(f"  cpu: {a[:200]!r}\n  vk : {b[:200]!r}")
                break
        sys.exit(f"FAIL: the chain's frames differ from the CPU's ({len(cpu)} vs {len(dev)})")
    print(f"OK serve {os.path.basename(snap)} {' '.join(sys.argv[3:])}: {len(cpu)} frames = CPU "
          f"(numbers within {WORST[0]:.1e}), {pins} pin lines, {len(reuse)} turns reusing state, "
          f"{forwards[-1].split('] ', 1)[1][:60]}")


if __name__ == "__main__":
    main()
