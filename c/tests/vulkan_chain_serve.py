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
CHAIN_SERVE_DIALECT picks the SUBMIT header: unset, ids r0, r1, ... and the options
straight after top_p (what the qwen engines have always been sent); "numeric", ids 1,
2, ... (glm53); "colibri", numeric ids and the grammar-bytes field (0) before the
options (colibri's multiplexed serve). CHAIN_SERVE_SLOTS=n sends request i to KV slot
i % n (the engine needs KV_SLOTS=n): the turns move between sessions, and with them
whatever state the chain keeps on the device.
"""
import os
import re
import subprocess
import sys
import threading

READY = b"\x01\x01READY\x01\x01"


def session(engine, snap, extra, chain):
    env = dict(os.environ, SNAP=snap, SERVE="1", COLI_PREFIX_LOG="1", Q38_PREFIX_LOG="1", **extra)
    if chain:
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
    out = []
    for i, (prompt, max_tok, ext) in enumerate(turns):
        dialect = os.environ.get("CHAIN_SERVE_DIALECT", "")
        rid = f"{i + 1}" if dialect in ("numeric", "colibri") else f"r{i}"
        slot = i % int(os.environ.get("CHAIN_SERVE_SLOTS", "1"))
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
            elif kind == b"DONE":
                out.append(b"DONE " + line.split()[1])
                break
            elif kind == b"ERROR":
                sys.exit(f"engine error: {line!r}")
    p.stdin.close()
    p.wait(timeout=120)
    return out, b"".join(err).decode(errors="replace")


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
    engine, snap = sys.argv[1], sys.argv[2]
    extra = dict(a.split("=", 1) for a in sys.argv[3:])
    cpu, err_cpu = session(engine, snap, extra, False)
    dev, err = session(engine, snap, extra, True)
    for text in (err_cpu, err):   # a sanitized build reports into the engine's stderr
        if "ERROR: AddressSanitizer" in text or "runtime error:" in text:
            sys.exit("FAIL: sanitizer diagnostic\n" + text[-4000:])
    forwards = [l for l in err.splitlines() if " chain: " in l and "forwards" in l]
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
