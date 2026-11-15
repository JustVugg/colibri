#!/usr/bin/env python3
"""Qwen3.8: AMD Quark's MXFP4 routed experts, as a sidecar next to the release.

AMD publishes Qwen3.8 with the routed experts in MXFP4 (OCP microscaling FP4,
quantized with Quark): per row, e2m1 values two to a byte and one E8M0
exponent per 32 inputs, 4.25 bits a weight. This tool copies those bytes,
unchanged, into <model>/experts-mxfp4/ next to the FP8 release qwen38 runs
(--model): one record per expert, so a cache miss is one read. Nothing is
quantized again; the shared expert, attention, DeltaNet, PLE, vision and MTP
tensors stay the FP8 snapshot's and are not read.

Values, as quant.h's matmul_mxfp4 reads them: w = e2m1(nibble) * 2^(s - 127),
the low nibble the even column, bit 3 the sign, bits 0-2 index
{0, .5, 1, 1.5, 2, 3, 4, 6}; s = 255 (E8M0's NaN) is refused.

Layout: one safetensors file per layer, layer-NNN.safetensors, and index.json.
Every expert is one record starting on a 4096-byte boundary:

  gate scales U8 [I, H/32] | down scales U8 [H, I/32] | up scales U8 [I, H/32] |
  gate e2m1 U8 [I, H/2] | down e2m1 U8 [H, I/2] | up e2m1 U8 [I, H/2] | pad

(w1 = gate, w2 = down, w3 = up), named like the Quark tensors
(<prefix>.layers.L.mlp.experts.E.gate_proj.weight_scale and .weight), with
the release's prefix; <...>.experts.E.record_pad (U8) fills the record to the
next 4096 bytes, so the file has no hole. A width that is not a multiple of
32 keeps a short last group (H/32 and H/2 round up). On the release (H 2560,
I 640) a record is 2,611,200 bytes, 2,613,248 with its pad: 53% of the FP8
expert, 94.5% of the int4-g64 sidecar's.

Checks before a layer is written: every expert's six tensors are there, U8
nibbles and U8 (or F8_E8M0) scales of exactly the config's shapes; and
--check-experts experts of the layer, dequantized, against the same expert of
the FP8 snapshot (cosine per matrix, --min-cos): a nibble order, a scale axis
or a name mix-up gives a cosine near 0.

index.json carries the format, the geometry, the record size and alignment,
the layers written and "complete", which is true only when all of them are
there; the engine refuses anything else. A layer goes to a .part file renamed
when whole, so an interrupted run resumes at the first layer not listed; a run
against a snapshot or a Quark checkpoint whose files changed is refused.

Usage:
  python3 tools/convert_qwen38_experts_mxfp4.py --model /path/to/qwen38-fp8 --quark /path/to/qwen38-mxfp4 --plan
  python3 tools/convert_qwen38_experts_mxfp4.py --model /path/to/qwen38-fp8 --quark /path/to/qwen38-mxfp4
"""
import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_qwen38_experts_int4 as int4  # noqa: E402  the snapshot reader, safetensors header, index writer

SIDECAR_DIR = "experts-mxfp4"
FORMAT = "colibri.qwen38.experts-mxfp4"
VERSION = 1
GS = 32
ALIGN = 4096
CODES = "e2m1-lo-even"
SCALES = "ue8m0-per-32"
RECORD = ("gate_proj", "down_proj", "up_proj")      # w1, w2, w3: scales of all three, then their nibbles
SCALE_DTYPES = ("U8", "F8_E8M0", "F8_E8M0FNU")
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def align_up(n):
    return -(-n // ALIGN) * ALIGN


class Geometry:
    def __init__(self, layers, experts, hidden, inter):
        self.layers, self.experts, self.hidden, self.inter = layers, experts, hidden, inter
        self.shape = {"gate_proj": (inter, hidden), "up_proj": (inter, hidden), "down_proj": (hidden, inter)}
        self.parts = []   # (projection, tensor suffix, shape) in record order
        for suffix in (".weight_scale", ".weight"):
            for proj in RECORD:
                rows, cols = self.shape[proj]
                width = -(-cols // GS) if suffix == ".weight_scale" else -(-cols // 2)
                self.parts.append((proj, suffix, [rows, width]))
        self.record_bytes = sum(r * c for _, _, (r, c) in self.parts)
        self.stride = align_up(self.record_bytes)

    def tensors(self, prefix, layer, expert):
        """(name, shape, offset in the record, bytes) in record order, the pad last if there is one."""
        base = f"{prefix}.layers.{layer}.mlp.experts.{expert}"
        out, off = [], 0
        for proj, suffix, (rows, width) in self.parts:
            out.append((f"{base}.{proj}{suffix}", [rows, width], off, rows * width))
            off += rows * width
        if self.stride > off:
            out.append((f"{base}.record_pad", [self.stride - off], off, self.stride - off))
        return out


def mxfp4_values(nibbles, scales, rows, cols):
    """U8 nibbles [rows, ceil(cols/2)] and U8 scales [rows, ceil(cols/32)] -> f32 [rows, cols]."""
    w = np.frombuffer(nibbles, np.uint8).reshape(rows, -1)
    v = np.stack([E2M1[w & 15], E2M1[w >> 4]], -1).reshape(rows, -1)[:, :cols]
    s = np.frombuffer(scales, np.uint8).reshape(rows, -1).astype(np.float32)
    return v * np.repeat(np.exp2(s - 127), GS, axis=1)[:, :cols]


class Quark:
    """The routed experts of an AMD Quark MXFP4 checkpoint, located by name in its shards."""

    def __init__(self, quark_dir):
        self.dir = quark_dir
        index_path = os.path.join(quark_dir, "model.safetensors.index.json")
        weight_map = None
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                weight_map = json.load(f)["weight_map"]
            files = sorted(set(weight_map.values()))
        else:
            files = sorted(f for f in os.listdir(quark_dir) if f.endswith(".safetensors"))
        self.tensors = {}
        for name in files:
            path = os.path.join(quark_dir, name)
            header, base = int4.read_header(path)
            for key, meta in header.items():
                if key == "__metadata__" or ".mlp.experts." not in key or (weight_map and weight_map.get(key) != name):
                    continue
                begin, end = meta["data_offsets"]
                self.tensors[key] = (path, base + begin, end - begin, meta["dtype"], meta["shape"])
        prefixes = [p for p in ("model.language_model", "model")
                    if any(k.startswith(p + ".layers.") for k in self.tensors)]
        if not prefixes:
            sys.exit(f"{quark_dir}: no routed expert tensors (<prefix>.layers.L.mlp.experts.E.*)")
        self.prefix = prefixes[0]
        self.config_bytes = b""
        if os.path.exists(os.path.join(quark_dir, "config.json")):
            with open(os.path.join(quark_dir, "config.json"), "rb") as f:
                self.config_bytes = f.read()
        self._fds = {}

    def fingerprint(self):
        shards = sorted({t[0] for t in self.tensors.values()})
        return {"config_sha256": hashlib.sha256(self.config_bytes).hexdigest(),
                "shards": {os.path.basename(p): os.path.getsize(p) for p in shards}}

    def read(self, name):
        path, offset, nbytes, _, _ = self.tensors[name]
        fd = self._fds.get(path)
        if fd is None:
            fd = self._fds[path] = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        out = bytearray()
        while len(out) < nbytes:
            chunk = int4._pread(fd, nbytes - len(out), offset + len(out))
            if not chunk:
                raise IOError(f"{path}: short read at {offset + len(out)}")
            out += chunk
        return bytes(out)

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def expert(self, geometry, layer, expert):
        """The six tensors' bytes in record order, after checking dtype, shape and scale bytes."""
        out = []
        for proj, suffix, shape in geometry.parts:
            name = f"{self.prefix}.layers.{layer}.mlp.experts.{expert}.{proj}{suffix}"
            entry = self.tensors.get(name)
            if entry is None:
                sys.exit(f"{self.dir}: {name} is missing (is the checkpoint complete?)")
            _, _, nbytes, dtype, have = entry
            want = SCALE_DTYPES if suffix == ".weight_scale" else ("U8",)
            if dtype not in want or list(have) != shape or nbytes != shape[0] * shape[1]:
                sys.exit(f"{self.dir}: {name} is {dtype} {have}, expected {' or '.join(want)} {shape}")
            raw = self.read(name)
            if suffix == ".weight_scale" and b"\xff" in raw:
                sys.exit(f"{self.dir}: {name} holds the E8M0 NaN (255)")
            out.append(raw)
        return out


def check_layer(src, quark, geometry, layer, picks, min_cos):
    """Dequantize the picked experts and compare them with the snapshot's; returns the worst cosine."""
    worst = 1.0
    for expert in picks:
        parts = quark.expert(geometry, layer, expert)
        for k, proj in enumerate(int4.PROJ):
            rows, cols = geometry.shape[proj]
            m = RECORD.index(proj)
            q = mxfp4_values(parts[3 + m], parts[m], rows, cols).astype(np.float64)
            r = src.matrix(layer, expert, k).astype(np.float64)
            cos = float((q * r).sum() / (np.linalg.norm(q) * np.linalg.norm(r) + 1e-30))
            worst = min(worst, cos)
            if not cos >= min_cos:
                sys.exit(f"layer {layer} expert {expert} {proj}: MXFP4 against the snapshot's expert has cosine "
                         f"{cos:.4f} < {min_cos} (nibble order, scale axis or tensor names wrong, or another model?)")
    return worst


def layer_header(geometry, prefix, layer):
    """The layer file's header, padded so the data region starts on ALIGN."""
    header = {}
    for expert in range(geometry.experts):
        base = expert * geometry.stride
        for name, shape, off, n in geometry.tensors(prefix, layer, expert):
            header[name] = {"dtype": "U8", "shape": shape, "data_offsets": [base + off, base + off + n]}
    header["__metadata__"] = {"format": FORMAT, "version": str(VERSION), "layer": str(layer)}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-(8 + len(raw))) % ALIGN)
    return struct.pack("<Q", len(raw)) + raw


def layer_file_bytes(geometry, prefix, layer):
    return len(layer_header(geometry, prefix, layer)) + geometry.experts * geometry.stride


def new_index(src, quark, geometry):
    return {
        "format": FORMAT, "version": VERSION, "complete": False,
        "group_size": GS, "codes": CODES, "scales": SCALES,
        "layers": geometry.layers, "experts": geometry.experts,
        "hidden_size": geometry.hidden, "moe_intermediate_size": geometry.inter,
        "prefix": src.prefix, "record_bytes": geometry.record_bytes, "record_align": ALIGN,
        "record": ["gate scales", "down scales", "up scales", "gate e2m1", "down e2m1", "up e2m1"],
        "quantizer": "AMD Quark MXFP4 (OCP), bytes copied unchanged",
        "files": [int4.layer_file(i) for i in range(geometry.layers)],
        "done": [],
        "source": {"model": src.fingerprint(), "quark": quark.fingerprint()},
    }


def resume_index(out_dir, src, quark, geometry):
    """The index of a previous run on the same inputs, or a fresh one."""
    fresh = new_index(src, quark, geometry)
    path = os.path.join(out_dir, "index.json")
    if not os.path.exists(path):
        return fresh
    with open(path, encoding="utf-8") as f:
        index = json.load(f)
    keys = ("format", "version", "group_size", "codes", "scales", "layers", "experts", "hidden_size",
            "moe_intermediate_size", "prefix", "record_bytes", "record_align", "source")
    stale = [k for k in keys if index.get(k) != fresh[k]]
    if stale:
        sys.exit(f"{path} was written for something else ({', '.join(stale)} differ); "
                 f"remove {out_dir} to convert from scratch")
    for layer in index.get("done", []):
        file = os.path.join(out_dir, int4.layer_file(layer))
        if os.path.exists(file) and os.path.getsize(file) == layer_file_bytes(geometry, src.prefix, layer):
            fresh["done"].append(layer)
        else:
            print(f"layer {layer}: listed as done but {file} is missing or the wrong size; converting it again")
    fresh["done"].sort()
    fresh["complete"] = len(fresh["done"]) == geometry.layers
    return fresh


def write_layer(src, quark, geometry, out_dir, layer, check_experts, min_cos):
    picks = sorted({round(i * (geometry.experts - 1) / max(check_experts - 1, 1)) for i in range(check_experts)})
    worst = check_layer(src, quark, geometry, layer, picks, min_cos) if check_experts else None
    final = os.path.join(out_dir, int4.layer_file(layer))
    part = final + ".part"
    with open(part, "wb") as f:   # one expert in memory at a time
        f.write(layer_header(geometry, src.prefix, layer))
        for expert in range(geometry.experts):
            record = b"".join(quark.expert(geometry, layer, expert))
            f.write(record + bytes(geometry.stride - len(record)))
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, final)
    return worst


def convert(model_dir, quark_dir, out_dir=None, layers=None, plan=False, check_experts=4, min_cos=0.95):
    """Write (or finish) model_dir's MXFP4 sidecar from quark_dir; returns its directory."""
    src = int4.Source(model_dir)
    quark = Quark(quark_dir)
    try:
        g0 = src.geometry
        geometry = Geometry(g0.layers, g0.experts, g0.hidden, g0.inter)
        if quark.config_bytes:
            text = json.loads(quark.config_bytes)
            text = text.get("text_config", text)
            theirs = tuple(text.get(k) for k in ("num_hidden_layers", "num_experts", "hidden_size",
                                                "moe_intermediate_size"))
            if theirs != (geometry.layers, geometry.experts, geometry.hidden, geometry.inter):
                sys.exit(f"{quark_dir}/config.json describes another geometry {theirs} than {model_dir}'s")
        out_dir = out_dir or os.path.join(model_dir, SIDECAR_DIR)
        fp8 = sum(r * c for r, c in geometry.shape.values())
        total = sum(layer_file_bytes(geometry, src.prefix, i) for i in range(geometry.layers))
        print(f"{model_dir}: {geometry.layers} layers x {geometry.experts} experts, hidden {geometry.hidden}, "
              f"moe_intermediate {geometry.inter}, prefix {src.prefix}; Quark prefix {quark.prefix}")
        print(f"  per expert: FP8 {fp8:,} bytes -> MXFP4 {geometry.record_bytes:,} bytes, "
              f"{geometry.stride:,} with its pad ({100 * geometry.stride / fp8:.1f}%)")
        print(f"  sidecar {out_dir}: {total:,} bytes ({total / 2**30:.2f} GiB)")
        if plan:
            return out_dir
        os.makedirs(out_dir, exist_ok=True)
        index = resume_index(out_dir, src, quark, geometry)
        wanted = int4.parse_layers(layers, geometry.layers)
        todo = [i for i in wanted if i not in index["done"]]
        if len(wanted) > len(todo):
            print(f"  {len(wanted) - len(todo)} layer(s) already converted, skipped")
        int4.write_index(out_dir, index)
        for layer in todo:
            worst = write_layer(src, quark, geometry, out_dir, layer, check_experts, min_cos)
            index["done"] = sorted(index["done"] + [layer])
            index["complete"] = len(index["done"]) == geometry.layers
            int4.write_index(out_dir, index)
            print(f"  layer {layer:3d}: {geometry.experts} records"
                  + (f", worst cosine against the snapshot {worst:.5f}" if worst is not None else ""), flush=True)
        state = "complete" if index["complete"] else f"{len(index['done'])}/{geometry.layers} layers"
        print(f"{out_dir}: {state}")
        return out_dir
    finally:
        quark.close()
        for fd in src._fds.values():
            os.close(fd)
        src._fds.clear()


def main():
    ap = argparse.ArgumentParser(description="Qwen3.8 routed experts: AMD Quark MXFP4 -> experts-mxfp4/ sidecar")
    ap.add_argument("--model", required=True, help="the Qwen3.8 FP8 snapshot qwen38 runs (config.json + shards)")
    ap.add_argument("--quark", required=True, help="the AMD Quark MXFP4 checkpoint (its safetensors shards)")
    ap.add_argument("--out", help=f"sidecar directory (default: <model>/{SIDECAR_DIR}, where the engine looks)")
    ap.add_argument("--layers", help="only these layers, e.g. 0-3,7 (the index stays incomplete)")
    ap.add_argument("--plan", action="store_true", help="print sizes and exit without writing")
    ap.add_argument("--check-experts", type=int, default=4,
                    help="experts per layer compared with the snapshot's (default 4; 0 skips the check)")
    ap.add_argument("--min-cos", type=float, default=0.95,
                    help="the smallest cosine against the snapshot's expert accepted (default 0.95)")
    a = ap.parse_args()
    if a.check_experts < 0:
        sys.exit("--check-experts must be 0 or more")
    convert(a.model, a.quark, a.out, a.layers, a.plan, a.check_experts, a.min_cos)


if __name__ == "__main__":
    main()
