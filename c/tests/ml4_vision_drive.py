#!/usr/bin/env python3
"""Mistral-Large-4's picture path, engine side, without Pillow: a synthetic picture drawn at a size
whose sides are whole merged tokens (no resize), its patches cut by tools/mistral_image.patchify,
the prompt rendered by the gateway's own render_chat_mistral_large4 with the [IMG] rows, then the
IMAGE and SUBMIT frames to SERVE=1 SERVE_BATCH=1 ./colibri, the answer read back.

  python3 tests/ml4_vision_drive.py SNAP [max_tokens]     (numpy needed; run from c/)
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "tools"))
import mistral_image as mi            # noqa: E402
import openai_server as gw            # noqa: E402


def picture(w, h):
    """a red disc on the left, a blue square in the middle, a green triangle on the right, white ground"""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    img[(xx - w * 0.2) ** 2 + (yy - h * 0.5) ** 2 < (h * 0.18) ** 2] = (220, 30, 30)
    sq = (abs(xx - w * 0.5) < h * 0.16) & (abs(yy - h * 0.5) < h * 0.16)
    img[sq] = (30, 60, 220)
    tri = (yy > h * 0.32) & (yy < h * 0.68) & (abs(xx - w * 0.8) < (yy - h * 0.32) * 0.55)
    img[tri] = (30, 170, 60)
    return img


def main():
    snap = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    gh, gw_ = 8, 12                                   # merged tokens: 224 x 336 pixels
    unit = mi.PATCH * mi.MERGE
    img = picture(gw_ * unit, gh * unit).astype(np.float32) / 255.0
    px = (img - np.array(mi.MEAN, dtype=np.float32)) / np.array(mi.STD, dtype=np.float32)
    patches = mi.patchify(np.ascontiguousarray(px.transpose(2, 0, 1)), np)
    text = "What shapes are in this picture, and what colour is each? Answer in one sentence."
    msgs = [{"role": "user", "content": [{"type": "text", "text": mi.placeholder(gh, gw_)},
                                         {"type": "text", "text": text}]}]
    prompt = gw.render_chat_mistral_large4(msgs, enable_thinking=False)
    env = dict(os.environ, SNAP=snap, SERVE="1", SERVE_BATCH="1", COLI_TEMP="0")
    p = subprocess.Popen(["./colibri", "8"], cwd=HERE, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = p.stdout
    def line():
        return out.readline().decode("utf-8", "replace").rstrip("\n")
    while True:                                       # READY, CAPS, STAT
        l = line()
        if not l: sys.exit("the engine ended before READY")
        if l.startswith("CAPS"): print(l)
        if l.startswith("STAT"): break
    blob = patches.tobytes()
    payload = prompt.encode()
    p.stdin.write(f"IMAGE 1 {len(blob)} {gh * mi.MERGE} {gw_ * mi.MERGE}\n".encode() + blob + b"\n")
    p.stdin.write(f"SUBMIT 1 0 {len(payload)} {max_tokens} 0 1\n".encode() + payload + b"\n")
    p.stdin.flush()
    answer = b""
    while True:
        l = line()
        if not l and p.poll() is not None: break
        if l.startswith("DATA "):
            n = int(l.split()[2]); answer += out.read(n); out.read(1)
        elif l.startswith("DONE") or l.startswith("ERROR"):
            print(l); break
        elif l.startswith("ACCEPT"):
            print(l)
    print("ANSWER:", answer.decode("utf-8", "replace"))
    p.stdin.close(); p.wait(timeout=120)


if __name__ == "__main__":
    main()
