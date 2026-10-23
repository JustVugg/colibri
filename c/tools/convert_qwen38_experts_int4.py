#!/usr/bin/env python3
"""Qwen3.8: the routed experts as int4-g64, in a sidecar next to the release.

The release stores every routed expert as FP8 e4m3 with one scale per 128x128
block, 4.9 MB per expert on Qwen3.8-Flash-Next. Decode waits on those bytes:
an expert that is not in the RAM cache is read from disk, and how many fit in
the cache decides how often that happens. This converter writes the same
experts as int4 with one f32 scale per 64 inputs, 2.76 MB per expert, into
<model>/experts-int4g64/. The FP8 shards are only read: the snapshot stays
usable as it is, and qwen38 picks the sidecar up when it is there
(Q38_EXPERT_INT4=0 ignores it).

Only the routed experts change. The shared expert, attention, DeltaNet, PLE,
vision and MTP tensors are not read at all.

Numerics, so the engine and every tier read the bytes they already know:
  dequantize  e4m3 byte -> quant.h's E4M3_LUT value, times the block's
              weight_scale_inv (BF16 or F32, widened to f32), one f32 multiply:
              exactly the expanded FP8 path in qwen38_core.h. BF16/F16/F32
              experts (the tiny fixture, fused or per expert) are widened.
  quantize    tools/convert_qwen36.py's quantize_row_grouped with --ebits 4
              --gs 64: per row, groups of 64 inputs (a short last group is
              zero-padded), scale = max(absmax, 1e-12) / 7 in f32, code =
              round-half-even(w / scale) clamped to [-8, 7].
  codes       stored as v+8 in expert_ffn.h's planar layout: a row is blocks
              of 64 elements, 32 bytes each, byte k holding element k in its
              low nibble and element k+32 in its high nibble. A width that is
              not a multiple of 64 keeps its tail in pairs (byte j = element
              2j low, 2j+1 high), as idot.h's planarize_i4 leaves it. A row is
              ceil(cols/2) bytes.
  scales      f32, [rows][ceil(cols/64)].

Layout: one safetensors file per layer, layer-NNN.safetensors, and index.json.
Every expert is one contiguous record, so a cache miss is one read:

  gate codes U8 [I, H/2] | up codes U8 [I, H/2] | down codes U8 [H, I/2] |
  gate scales F32 [I, H/64] | up scales F32 [I, H/64] | down scales F32 [H, I/64]

named like the release's tensors (<prefix>.layers.L.mlp.experts.E.gate_proj.weight
and the same name + ".qs" for its scales, the .qs convention of the other
int4 containers). Records follow each other with no gap; the data region of
a file starts on a 4096-byte boundary, so on the release's geometry (record
2,764,800 bytes = 675 pages) every record is page aligned.

index.json carries the format name and version, the geometry, the group size,
the record size and the layers already written. It is rewritten (atomically)
after every layer and says "complete": true only when all of them are there;
the engine refuses anything else. A layer is written to a .part file and
renamed when it is whole, so an interrupted run resumes at the first layer
the index does not list.

Usage:
  python3 tools/convert_qwen38_experts_int4.py --model /path/to/qwen38-fp8 --plan
  python3 tools/convert_qwen38_experts_int4.py --model /path/to/qwen38-fp8 [--workers N]

Peak RAM does not grow with the model: each worker holds one expert in flight
(about 0.11 GB with numpy), and the writer at most the records of one layer
(1.4 GB) when the disk lags behind the workers. Measured on a 12-thread
i7-1355U with a synthetic snapshot of the release's expert geometry: 22 s of
compute per layer with 12 workers.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(errors="backslashreplace")
except AttributeError:
    pass

SIDECAR_DIR = "experts-int4g64"
FORMAT = "colibri.qwen38.experts-int4g64"
VERSION = 1
GS = 64
CODES = "planar64-v8"
SCALES = "f32-per-64"
DATA_ALIGN = 4096
PROJ = ("gate_proj", "up_proj", "down_proj")
FP8_BLOCK = 128

_FLOAT_DTYPES = {"BF16": 2, "F16": 2, "F32": 4}
_FP8_DTYPES = ("F8_E4M3", "F8_E4M3FN", "float8_e4m3fn")


def e4m3_table():
    """quant.h's E4M3_LUT (torch.float8_e4m3fn): bias 7, subnormals at
    exponent 0, only 0x7F/0xFF are NaN (no infinities)."""
    table = np.empty(256, np.float32)
    for byte in range(256):
        exponent, mantissa = (byte >> 3) & 0xF, byte & 7
        if exponent == 0xF and mantissa == 7:
            value = math.nan
        elif exponent == 0:
            value = mantissa / 8.0 * 2.0 ** -6
        else:
            value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        table[byte] = -value if byte & 0x80 else value
    return table


E4M3 = e4m3_table()


# ---------- int4-g64: quantize, pack, and their inverses ----------

def groups(cols):
    return (cols + GS - 1) // GS


def row_bytes(cols):
    return (cols + 1) // 2


def quantize_g64(w):
    """w f32 [R, C] -> (codes u8 [R, C] holding v+8, scales f32 [R, ceil(C/64)]).
    tools/convert_qwen36.py quantize_row_grouped, bits=4, in numpy."""
    rows, cols = w.shape
    ng = groups(cols)
    pad = ng * GS - cols
    padded = np.pad(w, ((0, 0), (0, pad))) if pad else w
    g = padded.reshape(rows, ng, GS)
    amax = np.abs(g).max(axis=2, keepdims=True)
    scales = np.maximum(amax, np.float32(1e-12)) / np.float32(7.0)
    codes = np.clip(np.rint(g / scales), -8, 7).astype(np.int16) + 8
    return codes.reshape(rows, ng * GS)[:, :cols].astype(np.uint8), scales.reshape(rows, ng)


def pack_planar(codes):
    """u8 codes [R, C] (0..15) -> u8 [R, ceil(C/2)]: planar blocks of 64, tail in pairs."""
    rows, cols = codes.shape
    body = cols // GS * GS
    parts = []
    if body:
        blocks = codes[:, :body].reshape(rows, body // GS, 2, GS // 2)
        parts.append((blocks[:, :, 0, :] | (blocks[:, :, 1, :] << 4)).reshape(rows, body // 2))
    if body < cols:
        tail = codes[:, body:]
        if tail.shape[1] & 1:
            tail = np.concatenate([tail, np.full((rows, 1), 8, np.uint8)], axis=1)
        parts.append(tail[:, 0::2] | (tail[:, 1::2] << 4))
    return np.ascontiguousarray(np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0])


def unpack_planar(packed, cols):
    """Inverse of pack_planar: u8 [R, ceil(C/2)] -> codes u8 [R, C]."""
    rows = packed.shape[0]
    body = cols // GS * GS
    out = np.empty((rows, cols), np.uint8)
    if body:
        blocks = packed[:, :body // 2].reshape(rows, body // GS, GS // 2)
        out[:, :body] = np.stack([blocks & 15, blocks >> 4], axis=2).reshape(rows, body)
    if body < cols:
        tail = packed[:, body // 2:]
        pairs = np.stack([tail & 15, tail >> 4], axis=2).reshape(rows, -1)
        out[:, body:] = pairs[:, :cols - body]
    return out


def dequantize(codes, scales):
    """codes u8 [R, C], scales f32 [R, ceil(C/64)] -> f32 [R, C]."""
    cols = codes.shape[1]
    return (codes.astype(np.float32) - 8.0) * np.repeat(scales, GS, axis=1)[:, :cols]


def rel_l2(reference, approx):
    """||reference - approx|| / ||reference||, in float64. Plain reductions, not
    np.linalg.norm: that is a BLAS call, and every worker's BLAS starting a
    thread per core made a 12-worker layer 2.4x slower (46 s against 19 s)."""
    ref = reference.astype(np.float64)
    norm = float(np.sum(np.square(ref)))
    diff = float(np.sum(np.square(ref - approx.astype(np.float64))))
    return math.sqrt(diff / norm) if norm > 0 else 0.0


# ---------- the geometry of one record ----------

class Geometry:
    def __init__(self, layers, experts, hidden, inter):
        self.layers, self.experts, self.hidden, self.inter = layers, experts, hidden, inter
        self.shape = ((inter, hidden), (inter, hidden), (hidden, inter))
        self.code_bytes = [r * row_bytes(c) for r, c in self.shape]
        self.scale_count = [r * groups(c) for r, c in self.shape]
        self.scale_off = sum(self.code_bytes)
        self.record_bytes = self.scale_off + 4 * sum(self.scale_count)
        if self.scale_off % 4:
            raise ValueError(f"hidden={hidden}, moe_intermediate_size={inter}: the codes of an "
                             f"expert are {self.scale_off} bytes, not a multiple of 4, so its f32 "
                             f"scales could not follow them aligned; this geometry is not supported")

    def tensors(self, prefix, layer, expert):
        """(name, dtype, shape, offset in the record, bytes) in record order."""
        base = f"{prefix}.layers.{layer}.mlp.experts.{expert}"
        out, off = [], 0
        for k, (r, c) in enumerate(self.shape):
            out.append((f"{base}.{PROJ[k]}.weight", "U8", [r, row_bytes(c)], off, self.code_bytes[k]))
            off += self.code_bytes[k]
        for k, (r, c) in enumerate(self.shape):
            n = 4 * self.scale_count[k]
            out.append((f"{base}.{PROJ[k]}.weight.qs", "F32", [r, groups(c)], off, n))
            off += n
        assert off == self.record_bytes
        return out

    def fp8_expert_bytes(self):
        return sum(r * c for r, c in self.shape)


# ---------- the release, read by offset ----------

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 512 << 20:
            raise ValueError(f"{path}: safetensors header of {n} bytes")
        return json.loads(f.read(n)), 8 + n


class Source:
    """The routed experts of a Qwen3.8 snapshot, located by name in its shards."""

    def __init__(self, model_dir):
        self.dir = model_dir
        with open(os.path.join(model_dir, "config.json"), "rb") as f:
            self.config_bytes = f.read()
        config = json.loads(self.config_bytes)
        text = config.get("text_config", config)
        if text.get("model_type") != "qwen4_exp_text":
            sys.exit(f"{model_dir}: text model_type {text.get('model_type')!r} is not qwen4_exp_text")
        self.geometry = Geometry(int(text["num_hidden_layers"]), int(text["num_experts"]),
                                 int(text["hidden_size"]), int(text["moe_intermediate_size"]))
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                weight_map = json.load(f)["weight_map"]
            files = sorted(set(weight_map.values()))
        else:
            weight_map = None
            files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
        if not files:
            sys.exit(f"{model_dir}: no safetensors shards")
        self.tensors = {}
        for name in files:
            path = os.path.join(model_dir, name)
            header, base = read_header(path)
            for key, meta in header.items():
                if key == "__metadata__" or (weight_map and weight_map.get(key) != name):
                    continue
                if ".mlp.experts." not in key:
                    continue
                begin, end = meta["data_offsets"]
                self.tensors[key] = (path, base + begin, end - begin, meta["dtype"], meta["shape"])
        if os.path.exists(index_path):
            prefixes = [p for p in ("model.language_model", "model")
                        if p + ".embed_tokens.weight" in weight_map]
        else:
            prefixes = []
            for name in files:
                header, _ = read_header(os.path.join(model_dir, name))
                prefixes += [p for p in ("model.language_model", "model")
                             if p + ".embed_tokens.weight" in header]
        if not prefixes:
            sys.exit(f"{model_dir}: no Qwen4-Exp text embedding; is this a Qwen3.8 snapshot?")
        self.prefix = prefixes[0]
        self._fds = {}

    def fingerprint(self):
        """What a resumed run must find unchanged: the config and the shards
        that hold routed experts (name and size)."""
        shards = sorted({t[0] for t in self.tensors.values()})
        return {"config_sha256": hashlib.sha256(self.config_bytes).hexdigest(),
                "shards": {os.path.basename(p): os.path.getsize(p) for p in shards}}

    def _read(self, path, offset, nbytes):
        fd = self._fds.get(path)
        if fd is None:
            fd = self._fds[path] = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        out = bytearray()
        while len(out) < nbytes:
            chunk = os.pread(fd, nbytes - len(out), offset + len(out))
            if not chunk:
                raise IOError(f"{path}: short read at {offset + len(out)}")
            out += chunk
        return bytes(out)

    def _float(self, entry, element_offset, count):
        path, offset, nbytes, dtype, _ = entry
        size = _FLOAT_DTYPES[dtype]
        raw = self._read(path, offset + element_offset * size, count * size)
        if dtype == "BF16":
            return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        if dtype == "F16":
            return np.frombuffer(raw, np.float16).astype(np.float32)
        return np.frombuffer(raw, np.float32).copy()

    def matrix(self, layer, expert, k):
        """Expert `expert`'s projection k (gate, up, down) as the engine sees it, f32."""
        rows, cols = self.geometry.shape[k]
        name = f"{self.prefix}.layers.{layer}.mlp.experts.{expert}.{PROJ[k]}.weight"
        entry = self.tensors.get(name)
        if entry is not None:
            path, offset, nbytes, dtype, shape = entry
            if list(shape) != [rows, cols]:
                raise ValueError(f"{name}: shape {shape}, expected {[rows, cols]}")
            if dtype in _FP8_DTYPES:
                if nbytes != rows * cols:
                    raise ValueError(f"{name}: {nbytes} bytes for an e4m3 [{rows}, {cols}]")
                scale = self.tensors.get(name + "_scale_inv")
                nb_r, nb_c = -(-rows // FP8_BLOCK), -(-cols // FP8_BLOCK)
                if scale is None or list(scale[4]) != [nb_r, nb_c] or scale[3] not in _FLOAT_DTYPES:
                    raise ValueError(f"{name}_scale_inv: missing, or not a [{nb_r}, {nb_c}] float tensor")
                block = self._float(scale, 0, nb_r * nb_c).reshape(nb_r, nb_c)
                codes = np.frombuffer(self._read(path, offset, nbytes), np.uint8).reshape(rows, cols)
                expand = np.repeat(np.repeat(block, FP8_BLOCK, axis=0)[:rows], FP8_BLOCK, axis=1)[:, :cols]
                w = E4M3[codes] * expand
            elif dtype in _FLOAT_DTYPES:
                w = self._float(entry, 0, rows * cols).reshape(rows, cols)
            else:
                raise ValueError(f"{name}: unsupported expert dtype {dtype}")
        else:
            fused = "gate_up_proj" if k < 2 else "down_proj"
            fname = f"{self.prefix}.layers.{layer}.mlp.experts.{fused}"
            entry = self.tensors.get(fname)
            g = self.geometry
            want = [g.experts, 2 * g.inter, g.hidden] if k < 2 else [g.experts, g.hidden, g.inter]
            if entry is None:
                raise ValueError(f"layer {layer} expert {expert}: neither {name} nor {fname}")
            if list(entry[4]) != want or entry[3] not in _FLOAT_DTYPES:
                raise ValueError(f"{fname}: expected a float {want}, found {entry[3]} {entry[4]}")
            first = expert * want[1] * want[2] + (rows * cols if k == 1 else 0)
            w = self._float(entry, first, rows * cols).reshape(rows, cols)
        if not np.isfinite(w).all():
            raise ValueError(f"{name}: non-finite weights; refusing to quantize them")
        return w


# ---------- one expert, one layer ----------

_SOURCE = None


def _init_worker(model_dir):
    global _SOURCE
    _SOURCE = Source(model_dir)


def convert_expert(task):
    """(layer, expert) -> (the record's bytes, relative L2 error of gate, up,
    down, this process's peak RSS in MB)."""
    layer, expert = task
    codes, scales, errors = [], [], []
    for k in range(3):
        w = _SOURCE.matrix(layer, expert, k)
        q, s = quantize_g64(w)
        errors.append(rel_l2(w, dequantize(q, s)))
        codes.append(pack_planar(q).tobytes())
        scales.append(s.astype(np.float32).tobytes())
    return b"".join(codes) + b"".join(scales), errors, peak_rss_mb()


def layer_file(layer):
    return f"layer-{layer:03d}.safetensors"


def layer_header(geometry, prefix, layer):
    """The layer file's header, padded so the data region starts on DATA_ALIGN."""
    header, meta = {}, {"format": FORMAT, "version": str(VERSION), "layer": str(layer)}
    for expert in range(geometry.experts):
        base = expert * geometry.record_bytes
        for name, dtype, shape, off, n in geometry.tensors(prefix, layer, expert):
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [base + off, base + off + n]}
    header["__metadata__"] = meta
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-(8 + len(raw))) % DATA_ALIGN)
    return struct.pack("<Q", len(raw)) + raw


def layer_file_bytes(geometry, prefix, layer):
    return len(layer_header(geometry, prefix, layer)) + geometry.experts * geometry.record_bytes


def write_layer(src, out_dir, layer, pool):
    g = src.geometry
    final = os.path.join(out_dir, layer_file(layer))
    part = final + ".part"
    tasks = [(layer, e) for e in range(g.experts)]
    errors, worker_mb = [], 0.0
    with open(part, "wb") as f:
        f.write(layer_header(g, src.prefix, layer))
        results = pool.imap(convert_expert, tasks, chunksize=4) if pool else map(convert_expert, tasks)
        for record, err, rss in results:
            if len(record) != g.record_bytes:
                raise RuntimeError(f"layer {layer}: record of {len(record)} bytes, expected {g.record_bytes}")
            f.write(record)
            errors.append(err)
            worker_mb = max(worker_mb, rss)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, final)
    return np.array(errors), worker_mb


def write_index(out_dir, index):
    tmp = os.path.join(out_dir, "index.json.part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(out_dir, "index.json"))


def new_index(src):
    g = src.geometry
    return {
        "format": FORMAT, "version": VERSION, "complete": False,
        "group_size": GS, "codes": CODES, "scales": SCALES,
        "layers": g.layers, "experts": g.experts,
        "hidden_size": g.hidden, "moe_intermediate_size": g.inter,
        "prefix": src.prefix, "record_bytes": g.record_bytes,
        "record": ["gate codes", "up codes", "down codes", "gate scales", "up scales", "down scales"],
        "quantizer": "per 64 inputs: scale = max(absmax, 1e-12)/7 (f32), code = rint(w/scale) "
                     "clamped to [-8,7], stored v+8 (tools/convert_qwen36.py quantize_row_grouped)",
        "files": [layer_file(i) for i in range(g.layers)],
        "done": [],
        "source": src.fingerprint(),
    }


def resume_index(out_dir, src):
    """The index of a previous run on the same snapshot, or a fresh one."""
    path = os.path.join(out_dir, "index.json")
    fresh = new_index(src)
    if not os.path.exists(path):
        return fresh
    with open(path, encoding="utf-8") as f:
        index = json.load(f)
    keys = ("format", "version", "group_size", "codes", "scales", "layers", "experts",
            "hidden_size", "moe_intermediate_size", "prefix", "record_bytes", "source")
    stale = [k for k in keys if index.get(k) != fresh[k]]
    if stale:
        sys.exit(f"{path} was written for something else ({', '.join(stale)} differ); "
                 f"remove {out_dir} to convert this snapshot from scratch")
    done = []
    for layer in index.get("done", []):
        path = os.path.join(out_dir, layer_file(layer))
        if os.path.exists(path) and os.path.getsize(path) == layer_file_bytes(src.geometry, src.prefix, layer):
            done.append(layer)
        else:
            print(f"layer {layer}: listed as done but {path} is missing or the wrong size; converting it again")
    fresh["done"] = sorted(done)
    fresh["complete"] = len(done) == src.geometry.layers
    return fresh


def parse_layers(spec, count):
    if not spec:
        return list(range(count))
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b) if b else int(a)
        if not 0 <= lo <= hi < count:
            sys.exit(f"--layers {spec}: {part} is outside 0..{count - 1}")
        out.update(range(lo, hi + 1))
    return sorted(out)


def peak_rss_mb():
    """This process's peak resident set in MB (0 where getrusage is missing)."""
    try:
        import resource
    except ImportError:
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 << 20) if sys.platform == "darwin" else peak / 1024


def convert(model_dir, out_dir=None, workers=1, layers=None, plan=False):
    """Write (or finish) model_dir's sidecar; returns its directory."""
    src = Source(model_dir)
    g = src.geometry
    out_dir = out_dir or os.path.join(model_dir, SIDECAR_DIR)
    total = sum(layer_file_bytes(g, src.prefix, i) for i in range(g.layers))
    fp8 = g.fp8_expert_bytes() * g.experts * g.layers
    print(f"{model_dir}: {g.layers} layers x {g.experts} experts, hidden {g.hidden}, "
          f"moe_intermediate {g.inter}, prefix {src.prefix}")
    print(f"  per expert: FP8 {g.fp8_expert_bytes():,} bytes -> int4-g64 {g.record_bytes:,} bytes "
          f"({100 * g.record_bytes / g.fp8_expert_bytes():.1f}%)")
    print(f"  sidecar {out_dir}: {total:,} bytes ({total / 2**30:.2f} GiB) "
          f"for {fp8:,} bytes of FP8 experts")
    if plan:
        return out_dir

    os.makedirs(out_dir, exist_ok=True)
    index = resume_index(out_dir, src)
    wanted = parse_layers(layers, g.layers)
    todo = [i for i in wanted if i not in index["done"]]
    if len(wanted) > len(todo):
        print(f"  {len(wanted) - len(todo)} layer(s) already converted, skipped")
    write_index(out_dir, index)

    pool = None
    if workers > 1 and todo:
        from multiprocessing import get_context
        pool = get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(model_dir,))
    else:
        _init_worker(model_dir)
    worst = 0.0
    try:
        for layer in todo:
            started = time.time()
            errors, worker_mb = write_layer(src, out_dir, layer, pool)
            index["done"] = sorted(index["done"] + [layer])
            index["complete"] = len(index["done"]) == g.layers
            write_index(out_dir, index)
            worst = max(worst, float(errors.max()))
            print(f"  layer {layer:3d}: {g.experts} experts in {time.time() - started:6.1f}s  "
                  f"rel L2 gate {errors[:, 0].mean():.4f} up {errors[:, 1].mean():.4f} "
                  f"down {errors[:, 2].mean():.4f} (max {errors.max():.4f})  "
                  f"peak RSS {peak_rss_mb():.0f} MB"
                  + (f" + {workers} workers x {worker_mb:.0f} MB" if pool else ""), flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    state = "complete" if index["complete"] else f"{len(index['done'])}/{g.layers} layers"
    print(f"{out_dir}: {state}" + (f", worst expert matrix rel L2 {worst:.4f}" if todo else ""))
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Qwen3.8 routed experts FP8 -> int4-g64 sidecar")
    ap.add_argument("--model", required=True, help="the Qwen3.8 snapshot (config.json + shards)")
    ap.add_argument("--out", help=f"sidecar directory (default: <model>/{SIDECAR_DIR}, where the engine looks)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="processes converting experts in parallel (default: every CPU)")
    ap.add_argument("--layers", help="only these layers, e.g. 0-3,7 (the index stays incomplete)")
    ap.add_argument("--plan", action="store_true", help="print sizes and exit without writing")
    a = ap.parse_args()
    convert(a.model, a.out, a.workers, a.layers, a.plan)


if __name__ == "__main__":
    main()
