#!/usr/bin/env python3
"""A stand-in for the qwenimage engine: same command line, same serve protocol.

The front ends (coli chat/run/serve, the web UI) were written before the C
engine could draw anything, and they are tested without it. This script speaks
the protocol byte for byte and draws a synthetic picture that depends on the
prompt, the seed and the size, so a rendered image is recognisable and two
requests that differ in one of them visibly differ. It is not a model.

It is deliberately self-contained (stdlib only, its own PNG writer): it plays
the part of a separate binary, and sharing code with the client it is used to
test would let a bug on one side hide the same bug on the other.

    qwenimage_stub.py --model DIR --serve
    qwenimage_stub.py --model DIR --prompt "..." [--width 768] [--height 512]
                      [--steps 8] [--seed 42] --out out.png

Serve protocol (one request at a time, logs on stderr):
    engine -> READY {...}
    client -> GEN {"id","prompt","width","height","steps","seed","preview"}
    engine -> PROGRESS {"id","stage","step","steps","elapsed"}   step = steps COMPLETED
    engine -> PREVIEW {"id","step","width","height","channels":3,"bytes":N} + N bytes + "\\n"
    engine -> IMAGE {"id","width","height","channels":4,"bytes":N,"seed","steps","timings"}
              + N bytes + "\\n"
    engine -> ERROR {"id","message"}
    client -> CANCEL {"id"}        answered with ERROR "cancelled"
    stdin EOF: clean exit

Knobs (environment), all optional:
    QWENIMAGE_STUB_DELAY       seconds per denoising step (default 0.15)
    QWENIMAGE_STUB_LOAD_DELAY  seconds before READY (default 0)
    QWENIMAGE_STUB_PREVIEW     0 never sends PREVIEW even when asked (default 1)
    QWENIMAGE_STUB_MULTIPLE    the side multiple READY advertises (default 32)
Prompts containing __stub_error__ get an ERROR, __stub_crash__ kills the
process in the middle of denoising, __stub_garbage__ writes a stray stdout line
first: the three failure shapes a front end has to survive.
"""
import argparse
import hashlib
import json
import math
import os
import queue
import struct
import sys
import threading
import time
import zlib

# The contract's READY line, verbatim. 32 and not 16: the pipeline snaps both
# sides to vae_scale_factor*2 because the image tokens come in 2x2 groups, so
# 16:9 is 1024x576 or 512x288 and never 768x432.
DEFAULTS = {"model": "qwen-image-2.1", "default_width": 768, "default_height": 512,
            "default_steps": 8, "min_side": 256, "max_side": 2048, "multiple": 32}
MAX_STEPS = 200


def log(message):
    sys.stderr.write(f"[qwenimage-stub] {message}\n")
    sys.stderr.flush()


def env_float(name, default):
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except ValueError:
        return default


# ---------------------------------------------------------------- the picture
def _palette(prompt, seed):
    """Colours from the prompt, layout from the seed: changing either shows."""
    words = hashlib.sha256(prompt.encode("utf-8")).digest()
    layout = hashlib.sha256(f"{prompt}\0{seed}".encode("utf-8")).digest()
    return words, layout


def render(prompt, seed, width, height):
    """RGBA8 bytes, row-major, top row first.

    Resolution independent on purpose: every coordinate is a fraction of the
    size, so the preview drawn at 1/16 is the same picture as the final one.
    Rows are assembled with bytes.translate and slice assignment, which keeps a
    768x512 image well under a second in pure Python."""
    words, layout = _palette(prompt, seed)
    top = words[0:3]
    bottom = words[3:6]
    right = words[6:9]
    transparent = "transparent" in prompt.lower()
    # Horizontal ramp per channel, shifted per row by a vertical ramp.
    planes = []
    for c in range(3):
        ramp = bytes(int(top[c] + (right[c] - top[c]) * x / max(1, width - 1)) & 255
                     for x in range(width))
        planes.append(ramp)
    tables = {}

    def table(c, shift):
        key = (c, shift)
        found = tables.get(key)
        if found is None:
            found = bytes(max(0, min(255, v + shift)) for v in range(256))
            tables[key] = found
        return found

    image = bytearray(width * height * 4)
    # A prompt asking for transparency gets a clear background under opaque
    # shapes, the RGBA cut-out the real model can produce.
    alpha_row = bytes([0 if transparent else 255]) * width
    for y in range(height):
        row = bytearray(width * 4)
        for c in range(3):
            shift = int((bottom[c] - top[c]) * y / max(1, height - 1)) // 2
            row[c::4] = planes[c].translate(table(c, shift))
        row[3::4] = alpha_row
        image[y * width * 4:(y + 1) * width * 4] = row

    def span(y, x0, x1, rgba):
        x0 = max(0, x0)
        x1 = min(width, x1)
        if y < 0 or y >= height or x1 <= x0:
            return
        base = (y * width + x0) * 4
        image[base:base + (x1 - x0) * 4] = rgba * (x1 - x0)

    shapes = 3 + layout[0] % 3
    unit = min(width, height)
    for i in range(shapes):
        b = layout[1 + i * 5:6 + i * 5]
        colour = bytes((b[0], b[1], b[2], 255))
        cx = int(width * (0.1 + 0.8 * b[3] / 255))
        cy = int(height * (0.1 + 0.8 * b[4] / 255))
        size = int(unit * (0.08 + 0.18 * ((b[0] ^ b[4]) / 255)))
        if i % 2 == 0:                                   # filled circle
            for y in range(cy - size, cy + size + 1):
                dy = y - cy
                dx = int(math.sqrt(max(0, size * size - dy * dy)))
                span(y, cx - dx, cx + dx + 1, colour)
        else:                                            # rectangle with a frame
            frame = bytes((255 - b[0], 255 - b[1], 255 - b[2], 255))
            border = max(1, unit // 128)
            for y in range(cy - size, cy + size + 1):
                if abs(y - cy) > size - border:
                    span(y, cx - size, cx + size + 1, frame)
                else:
                    span(y, cx - size, cx + size + 1, colour)
                    span(y, cx - size, cx - size + border, frame)
                    span(y, cx + size + 1 - border, cx + size + 1, frame)
    # A diagonal band whose slope comes from the seed.
    slope = (layout[30] / 255 - 0.5) * 2
    band = max(2, unit // 40)
    stripe = bytes((255, 255, 255, 255))
    for y in range(height):
        x = int(width / 2 + slope * (y - height / 2))
        span(y, x - band // 2, x + band // 2, stripe)
    return bytes(image)


def preview(prompt, seed, width, height, step, steps):
    """RGB8 thumbnail at 1/16, fading from grey noise to the picture."""
    pw, ph = max(1, width // 16), max(1, height // 16)
    final = render(prompt, seed, pw, ph)
    mix = step / max(1, steps)
    noise = hashlib.sha256(f"noise\0{seed}\0{step}".encode("utf-8")).digest()
    out = bytearray(pw * ph * 3)
    for i in range(pw * ph):
        n = noise[i % len(noise)]
        for c in range(3):
            v = final[i * 4 + c]
            out[i * 3 + c] = int(v * mix + n * (1 - mix)) & 255
    return pw, ph, bytes(out)


def png(width, height, rgba):
    raw = bytearray()
    stride = width * 4
    for y in range(height):
        raw.append(0)
        raw += rgba[y * stride:(y + 1) * stride]

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)

    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


# ---------------------------------------------------------------- validation
def check(request, info):
    """The engine's own validation. The gateway validates too; this is the
    last line, and it must answer ERROR rather than draw a wrong picture."""
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return "prompt must be a non-empty string"
    for key in ("width", "height"):
        value = request.get(key, info[f"default_{key}"])
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{key} must be an integer"
        if value % info["multiple"] or not info["min_side"] <= value <= info["max_side"]:
            return (f"{key} {value} is not a multiple of {info['multiple']} in "
                    f"[{info['min_side']}, {info['max_side']}]")
    steps = request.get("steps", info["default_steps"])
    if isinstance(steps, bool) or not isinstance(steps, int) or not 2 <= steps <= MAX_STEPS:
        return f"steps must be an integer in [2, {MAX_STEPS}]"
    seed = request.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        return "seed must be a non-negative integer"
    return None


# ---------------------------------------------------------------- serve loop
class Wire:
    def __init__(self):
        self.out = sys.stdout.buffer
        self.lock = threading.Lock()

    def line(self, keyword, obj, payload=None):
        text = f"{keyword} {json.dumps(obj, separators=(',', ':'))}\n".encode("utf-8")
        with self.lock:
            self.out.write(text)
            if payload is not None:
                self.out.write(payload)
                self.out.write(b"\n")
            self.out.flush()


def serve(args):
    info = dict(DEFAULTS)
    try:
        info["multiple"] = max(1, int(os.environ.get("QWENIMAGE_STUB_MULTIPLE", info["multiple"])))
    except ValueError:
        pass
    load_delay = env_float("QWENIMAGE_STUB_LOAD_DELAY", 0.0)
    log(f"loading {args.model} (stub: no weights are read)")
    if load_delay:
        time.sleep(load_delay)
    wire = Wire()
    wire.line("READY", info)
    inbox = queue.Queue()

    def reader():
        for raw in sys.stdin.buffer:
            inbox.put(raw.decode("utf-8", "replace").rstrip("\r\n"))
        inbox.put(None)

    threading.Thread(target=reader, daemon=True).start()
    pending = []

    def next_line(timeout=None):
        if pending:
            return pending.pop(0)
        return inbox.get(timeout=timeout)

    while True:
        line = next_line()
        if line is None:
            log("stdin closed, exiting")
            return 0
        keyword, _, rest = line.partition(" ")
        try:
            body = json.loads(rest) if rest else {}
        except ValueError:
            log(f"unparseable line ignored: {line[:80]!r}")
            continue
        if keyword == "CANCEL":
            continue                          # nothing running: nothing to cancel
        if keyword != "GEN":
            log(f"unknown command {keyword!r} ignored")
            continue
        rid = body.get("id", "")
        error = check(body, info)
        if error:
            wire.line("ERROR", {"id": rid, "message": error})
            continue
        prompt = body["prompt"]
        width = body.get("width", info["default_width"])
        height = body.get("height", info["default_height"])
        steps = body.get("steps", info["default_steps"])
        seed = body.get("seed", 0)
        want_preview = bool(body.get("preview")) and os.environ.get("QWENIMAGE_STUB_PREVIEW", "1") != "0"
        delay = env_float("QWENIMAGE_STUB_DELAY", 0.15)
        if "__stub_garbage__" in prompt:
            with wire.lock:
                wire.out.write(b"this is not a protocol line\n")
                wire.out.flush()
        if "__stub_error__" in prompt:
            wire.line("ERROR", {"id": rid, "message": "stub failure requested by the prompt"})
            continue
        t0 = time.monotonic()
        wire.line("PROGRESS", {"id": rid, "stage": "encode", "step": 0, "steps": steps,
                               "elapsed": 0.0})
        time.sleep(delay)
        t_encode = time.monotonic() - t0
        cancelled = eof = False
        # `step` counts COMPLETED denoising steps: 0 as the first one starts,
        # k after the k-th, `steps` again at decode.
        wire.line("PROGRESS", {"id": rid, "stage": "denoise", "step": 0, "steps": steps,
                               "elapsed": round(time.monotonic() - t0, 3)})
        for step in range(1, steps + 1):
            # CANCEL is honoured between steps, like the real engine.
            while True:
                try:
                    waiting = inbox.get_nowait()
                except queue.Empty:
                    break
                if waiting is None:
                    eof = True
                    break
                if waiting.startswith("CANCEL"):
                    try:
                        target = json.loads(waiting.partition(" ")[2]).get("id")
                    except ValueError:
                        target = None
                    if target == rid:
                        cancelled = True
                else:
                    pending.append(waiting)
            if cancelled or eof:
                break
            if "__stub_crash__" in prompt and step == max(1, steps // 2):
                log("crashing on purpose (__stub_crash__)")
                os._exit(3)
            time.sleep(delay)
            wire.line("PROGRESS", {"id": rid, "stage": "denoise", "step": step, "steps": steps,
                                   "elapsed": round(time.monotonic() - t0, 3)})
            if want_preview:
                pw, ph, data = preview(prompt, seed, width, height, step, steps)
                wire.line("PREVIEW", {"id": rid, "step": step, "width": pw, "height": ph,
                                      "channels": 3, "bytes": len(data)}, data)
        if eof:
            log("stdin closed during generation, exiting")
            return 0
        if cancelled:
            wire.line("ERROR", {"id": rid, "message": "cancelled"})
            continue
        t_denoise = time.monotonic() - t0 - t_encode
        wire.line("PROGRESS", {"id": rid, "stage": "decode", "step": steps, "steps": steps,
                               "elapsed": round(time.monotonic() - t0, 3)})
        t1 = time.monotonic()
        rgba = render(prompt, seed, width, height)
        t_decode = time.monotonic() - t1
        wire.line("IMAGE", {"id": rid, "width": width, "height": height, "channels": 4,
                            "bytes": len(rgba), "seed": seed, "steps": steps,
                            "timings": {"encode": round(t_encode, 3),
                                        "denoise": round(t_denoise, 3),
                                        "decode": round(t_decode, 3)}}, rgba)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--prompt")
    parser.add_argument("--width", type=int, default=DEFAULTS["default_width"])
    parser.add_argument("--height", type=int, default=DEFAULTS["default_height"])
    parser.add_argument("--steps", type=int, default=DEFAULTS["default_steps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out")
    args = parser.parse_args()
    if not os.path.isdir(args.model):
        log(f"model directory not found: {args.model}")
        return 2
    if args.serve:
        return serve(args)
    if not args.prompt or not args.out:
        parser.error("one-shot mode needs --prompt and --out (or use --serve)")
    error = check({"prompt": args.prompt, "width": args.width, "height": args.height,
                   "steps": args.steps, "seed": args.seed}, DEFAULTS)
    if error:
        log(error)
        return 2
    with open(args.out, "wb") as handle:
        handle.write(png(args.width, args.height,
                         render(args.prompt, args.seed, args.width, args.height)))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
