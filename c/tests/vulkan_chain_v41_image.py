#!/usr/bin/env python3
"""DeepSeek V4.1's dense chain with an image on the wire: one serve session on the CPU and
the same session with the chain (COLI_VK_CHAIN=1), every DATA frame compared.

The turns: an IMAGE frame and the prompt with its placeholder span (the vision tower's
rows replace those embeddings and go through the chain's layers), a text turn after it
(an image taints the prefix record: no reuse), and a second image. Texts must be the
CPU's exactly.

usage: vulkan_chain_v41_image.py <engine> <snapshot> [KEY=VALUE ...]   (extra environment)
"""
import json
import os
import struct
import subprocess
import sys
import threading

READY = b"\x01\x01READY\x01\x01"


def session(engine, snap, extra, chain):
    cfg = json.load(open(os.path.join(snap, "config.json")))
    vis = cfg["vision_config"]
    patch_in = 3 * vis["patch_size"] ** 2
    ratio = vis.get("downsample_ratio", 1)
    env = dict(os.environ, SNAP=snap, SERVE="1", **extra)
    if chain:
        env.update(COLI_VULKAN="1", COLI_VK_CHAIN="1", COLI_VK_TIER_SYNC="1",
                   COLI_USAGE=os.path.abspath("chain-image.usage"))
        try:
            os.remove(env["COLI_USAGE"])
        except FileNotFoundError:
            pass
    p = subprocess.Popen([engine, "8"], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = []
    threading.Thread(target=lambda: [err.append(l) for l in iter(p.stderr.readline, b"")], daemon=True).start()
    while True:
        line = p.stdout.readline()
        if not line:
            sys.exit("engine exited before READY:\n" + b"".join(err).decode(errors="replace")[-3000:])
        if READY in line:
            break
    out = []

    def image(rid, gh, gw, seed):
        vals = [((i * 7919 + seed * 104729) % 2001 - 1000) / 1000.0 for i in range(gh * gw * patch_in)]
        payload = struct.pack(f"<{len(vals)}f", *vals)
        p.stdin.write(f"IMAGE {rid} {len(payload)} {gh} {gw}\n".encode() + payload + b"\n")
        h, w = (gh + ratio - 1) // ratio, (gw + ratio - 1) // ratio
        return 1 + (w + 1) * h + 1

    def submit(rid, prompt, max_tok):
        p.stdin.write(f"SUBMIT {rid} 0 {len(prompt)} {max_tok} 0 1\n".encode() + prompt + b"\n")
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

    tok = json.load(open(os.path.join(snap, "tokenizer.json")))
    pid = cfg.get("image_token_id", 255)
    pad = next(a["content"] for a in tok.get("added_tokens", []) if a["id"] == pid).encode()   # the placeholder's text
    span = image("i1", 6, 6, 1)
    submit("i1", b"AB" + pad * span + b"CD", 5)
    submit("t1", b"ABCDEFG", 4)
    span = image("i2", 5, 7, 2)
    submit("i2", b"xy" + pad * span + b"zw", 5)
    p.stdin.close()
    p.wait(timeout=300)
    return out, b"".join(err).decode(errors="replace")


def main():
    engine, snap = sys.argv[1], sys.argv[2]
    extra = dict(a.split("=", 1) for a in sys.argv[3:])
    cpu, err_cpu = session(engine, snap, extra, False)
    dev, err = session(engine, snap, extra, True)
    for text in (err_cpu, err):
        if "ERROR: AddressSanitizer" in text or "runtime error:" in text:
            sys.exit("FAIL: sanitizer diagnostic\n" + text[-4000:])
    if "answering without the image" in err_cpu or "patches ->" not in err_cpu:
        sys.exit("FAIL: the image did not reach the model\n" + err_cpu[-3000:])
    forwards = [l for l in err.splitlines() if " chain: " in l and "forwards" in l]
    if not forwards:
        sys.exit("FAIL: the chain never ran\n" + err[-3000:])
    if cpu != dev:
        for a, b in zip(cpu, dev):
            if a != b:
                print(f"  cpu: {a[:200]!r}\n  vk : {b[:200]!r}")
                break
        sys.exit(f"FAIL: the chain's frames differ from the CPU's ({len(cpu)} vs {len(dev)})")
    print(f"OK serve images {os.path.basename(snap)} {' '.join(sys.argv[3:])}: {len(cpu)} frames = CPU, "
          f"{err_cpu.count('patches ->')} images, {forwards[-1].split('] ', 1)[1][:60]}")


if __name__ == "__main__":
    main()
