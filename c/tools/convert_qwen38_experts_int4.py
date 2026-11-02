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

--int2 writes int2_g32.h's format into <model>/experts-int2g32/ instead
(qwen38 reads it with Q38_EXPERT_INT2=1): per row, groups of 32 inputs on the
fixed levels -10, -3, 3, 10 times the group's scale, the scales stored as one
u8 per group under one f32 per row; 1.40 MB per expert, 2.25 bits per weight.
A group's scale and codes are fitted in turn (nearest level, then the least
squares scale for those codes) against the activation-weighted error when
--act-stats is given, then the u8 scales are refitted for the codes they
end up with. The record:

  gate codes U8 [I, H/4] | up codes | down codes U8 [H, I/4] |
  gate q U8 [I, H/32] | up q | down q U8 [H, I/32] |
  gate d F32 [I, 1] | up d | down d F32 [H, 1]

named <...>.weight, <...>.weight.q8 and <...>.weight.d.

--lowbit vq2g32 writes vq2_g32.h's format into <model>/experts-vq2g32/: each 8 weights of
a row are one vector A[a] + B[b] of two 256 x 8 int8 codebooks (an index byte each, 2 bits
per weight) times a 4-bit scale per 32 inputs under one f32 per row; 2.125 bits per weight.
The codebooks are fitted first (k-means on 8-vectors of sample experts over their group's
RMS, then on what the first book leaves), scaled to int8 so any a + b stays an int8, and
written into index.json, where the engine reads them. The record is the low-bit one with
scale bytes [rows, cols/64].

Usage:
  python3 tools/convert_qwen38_experts_int4.py --model /path/to/qwen38-fp8 --plan
  python3 tools/convert_qwen38_experts_int4.py --model /path/to/qwen38-fp8 [--workers N]
  python3 tools/convert_qwen38_experts_int4.py --model /path/to/qwen38-fp8 --int2 --act-stats act.bin

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
import threading
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
SIDECAR_DIR2 = "experts-int2g32"
FORMAT2 = "colibri.qwen38.experts-int2g32"
GS2 = 32
CODES2 = "planar128-2bit-lv10"
SCALES2 = "u8-per-32-f32-per-row"
LEVELS2 = np.array([-10.0, -3.0, 3.0, 10.0])


class Lowbit:
    """A low-bit expert format: `bits`-bit codes on fixed integer `levels` (the Lloyd-Max
    grid of a bell-shaped weight; levels fitted per layer on Qwen3.8 came out the same), one
    scale per `gs` inputs stored as an integer up to `smax` (255: a byte each, 15: a nibble,
    two to a byte) under one f32 per row. Codes in planar blocks of 128: the low two bits as
    int2_g32.h lays them (32 bytes), and for 3 bits a plane of the high bits after them
    (16 bytes: byte k holds elements k, k+16, ..., k+112 in bits 0..7)."""
    def __init__(self, name, bits, levels, gs, smax, codes):
        self.name, self.bits, self.gs, self.smax, self.codes = name, bits, gs, smax, codes
        self.levels = np.array(levels, np.float64)
        self.edges = (self.levels[1:] + self.levels[:-1]) / 2
        self.rms = math.sqrt(float(np.mean(self.levels ** 2)))
        self.sidecar = "experts-" + name
        self.format = "colibri.qwen38.experts-" + name
        self.scales = ("u8" if smax == 255 else "u4") + f"-per-{gs}-f32-per-row"
        self.q_suffix = ".q8" if smax == 255 else ".q4"

    def code_of(self, v):
        c = np.zeros(np.shape(v), np.uint8)
        for e in self.edges:
            c += v > e
        return c

    def row_code_bytes(self, cols):
        return cols * self.bits // 8

    def row_scale_bytes(self, cols):
        return cols // self.gs if self.smax == 255 else cols // self.gs // 2

    vq = False


VQ_BEAM = 4


class VqFormat(Lowbit):
    """Vector codes: a row's `vdim`-vectors as a[i] + b[j] of two `book` x `vdim` int8
    codebooks, a 4-bit scale per `gs` inputs (two to a byte) under one f32 per row.
    vq2g32 (vq2_g32.h): 8-vectors, 256-entry books, an index byte each (a0 b0 a1 b1 ...,
    cols/4 bytes a row), a scale per 32. vq1g64 (vq1_g64.h): 16-vectors, 1024-entry books,
    10-bit indices, the eight of four vectors (a0 b0 .. a3 b3) in 10 bytes, LSB first
    (cols/64 x 10 bytes a row), a scale per 64. `books` (set_books) holds the codebooks as
    float64 integers and the factor K they were scaled by: a weight is d * q * (a + b), d =
    the row scale the quantizer fits over K."""
    vq = True

    def __init__(self, name, vdim, book, gs, codes):
        super().__init__(name, 2, (-1.0, 1.0), gs, 15, codes)
        self.vdim, self.book = vdim, book
        self.ibits = int(book - 1).bit_length()
        self.books = None

    def row_code_bytes(self, cols):
        return cols // self.vdim * 2 * self.ibits // 8

    def set_books(self, a8, b8, k):
        shape = (self.book, self.vdim)
        a8, b8 = np.asarray(a8, np.int64).reshape(shape), np.asarray(b8, np.int64).reshape(shape)
        if int((np.abs(a8).max(axis=0) + np.abs(b8).max(axis=0)).max()) > 127:
            raise ValueError(f"{self.name} codebooks: some a + b leaves int8")
        self.books = (a8.astype(np.float64), b8.astype(np.float64), float(k))


INT2G32 = Lowbit("int2g32", 2, (-10, -3, 3, 10), 32, 255, CODES2)
INT2G16 = Lowbit("int2g16", 2, (-10, -3, 3, 10), 16, 15, CODES2)
INT3G16 = Lowbit("int3g16", 3, (-26, -16, -9, -3, 3, 9, 16, 26), 16, 15, "planar128-3bit-lv26")
VQ2G32 = VqFormat("vq2g32", 8, 256, 32, "vq2-2x256x8-int8")
VQ1G64 = VqFormat("vq1g64", 16, 1024, 64, "vq1-2x1024x16-int8")
LOWBIT = {f.name: f for f in (INT2G32, INT2G16, INT3G16, VQ2G32, VQ1G64)}
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


ALPHAS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)


def quantize_g64_weighted(w, imp, alphas=ALPHAS):
    """quantize_g64 with the scale of every group of 64 chosen among absmax/7 * alpha
    (alphas holds 1.0, so it is never worse than quantize_g64 by this measure) to
    minimise sum_i imp_i * (w_i - s * code_i)^2: imp is the mean square of each input
    channel's activations over calibration text (Q38_ACT_STATS), so an error counts as
    much as the product it spoils. Same codes and scales layout as quantize_g64."""
    rows, cols = w.shape
    ng = groups(cols)
    pad = ng * GS - cols
    padded = np.pad(w, ((0, 0), (0, pad))) if pad else w
    g = padded.reshape(rows, ng, GS)
    wimp = (np.pad(imp, (0, pad)) if pad else imp).reshape(1, ng, GS).astype(np.float32)
    base = np.maximum(np.abs(g).max(axis=2, keepdims=True), np.float32(1e-12)) / np.float32(7.0)
    best_err = best_q = best_s = None
    for a in alphas:
        s = (base * np.float32(a)).astype(np.float32)
        q = np.clip(np.rint(g / s), -8, 7)
        err = (np.square(g - q * s) * wimp).sum(axis=2, keepdims=True)
        if best_err is None:
            best_err, best_q, best_s = err, q, s
        else:
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_q = np.where(better, q, best_q)
            best_s = np.where(better, s, best_s)
    codes = (best_q.astype(np.int16) + 8).reshape(rows, ng * GS)[:, :cols].astype(np.uint8)
    return codes, best_s.reshape(rows, ng).astype(np.float32)


class ActStats:
    """The engine's Q38_ACT_STATS file: per layer and expert the rows seen and the sum
    of squares of each gate/up input channel (hidden) and down input channel (inter)."""
    def __init__(self, path):
        with open(path, "rb") as f:
            if f.read(8) != b"Q38ACT1\0":
                raise ValueError(f"{path}: not a Q38_ACT_STATS file")
            self.layers, self.experts, self.hidden, self.inter = struct.unpack("<4i", f.read(16))
        cells = self.layers * self.experts
        self.n = np.memmap(path, np.int64, "r", offset=24, shape=(cells,))
        self.sums = np.memmap(path, np.float64, "r", offset=24 + 8 * cells, shape=(cells, self.hidden + self.inter))
        self.path, self.rows = path, int(np.asarray(self.n).sum())

    def importance(self, layer, expert, k):
        """mean square per input channel for matrix k (0 gate, 1 up: hidden; 2 down: inter);
        uniform where calibration never routed the expert"""
        i = layer * self.experts + expert
        n = int(self.n[i])
        width = self.hidden if k < 2 else self.inter
        if n == 0:
            return np.ones(width, np.float32)
        part = self.sums[i, :self.hidden] if k < 2 else self.sums[i, self.hidden:]
        return (np.asarray(part) / n).astype(np.float32) + np.float32(1e-12)


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


def quantize_lowbit(w, imp=None, fmt=INT2G32, iters=6):
    """w f32 [R, C] (C % 128 == 0) -> (codes u8 [R, C], q u8 [R, C/gs], d f32 [R]):
    w ~ d[r] * q[r, g] * levels[code]. Per group the scale and the codes are fitted in turn
    against sum_i imp_i * (w_i - s * level_i)^2 (imp uniform without statistics), then the
    scales are stored as integers up to fmt.smax under the row's largest and refitted, with
    the codes, for what those integers can hold."""
    if fmt.vq:   # no correlations: GPTQ under H = diag(imp) is the importance-weighted search, no carry
        a = np.ones(w.shape[1]) if imp is None else np.maximum(imp.astype(np.float64), 1e-30)
        return quantize_vq_gptq(w, np.diag(1.0 / np.sqrt(a)), fmt)
    rows, cols = w.shape
    gs, top, L = fmt.gs, float(fmt.smax), fmt.levels
    ng = cols // gs
    g = w.reshape(rows, ng, gs).astype(np.float64)
    a = np.ones((1, ng, gs)) if imp is None else imp.astype(np.float64).reshape(1, ng, gs)
    tiny = 1e-30

    def refit(scale):
        lv = L[fmt.code_of(g / np.maximum(scale, tiny)[..., None])]
        num, den = (a * g * lv).sum(axis=2), (a * lv * lv).sum(axis=2)
        return np.where(den > 0, num / np.maximum(den, tiny), scale)

    s = np.sqrt((a * g * g).sum(axis=2) / a.sum(axis=2)) / fmt.rms
    for _ in range(iters):
        s = refit(s)
    d = np.maximum(s.max(axis=1), tiny) / top
    d = d.astype(np.float32).astype(np.float64)          # the f32 the file stores
    q = np.clip(np.rint(s / d[:, None]), 1, top)
    for _ in range(2):
        q = np.clip(np.rint(refit(d[:, None] * q) / d[:, None]), 0, top)
    code = fmt.code_of(g / np.maximum(d[:, None] * q, tiny)[..., None])
    return code.reshape(rows, cols), q.astype(np.uint8), d.astype(np.float32)


def quantize_lowbit_gptq(w, imp, U, fmt=INT2G32, iters=6, batch=128, margin=1.4):
    """quantize_lowbit with the error of every quantized column carried into the columns
    still to come (GPTQ's update, written from the paper): U is the upper Cholesky factor
    of H^-1 for this matrix's inputs, so quantizing column i moves the rest of its row by
    -(w_i - q_i) / U[i,i] * U[i, i+1:], which is the least-squares answer under H. Columns
    go in order; a group's scale is fitted when its first column comes up, on the weights
    as the earlier groups' errors left them. The row scale d is fixed first, from the
    groups' weighted RMS with `margin` of room for what the updates add. Works on the
    transpose, so a column is contiguous; a code is one comparison per level boundary."""
    if fmt.vq:
        return quantize_vq_gptq(w, U, fmt)
    rows, cols = w.shape
    gs, top, L = fmt.gs, float(fmt.smax), fmt.levels
    ng = cols // gs
    WT = np.ascontiguousarray(w.astype(np.float64).T)          # [cols, rows]
    a = np.ones(cols) if imp is None else imp.astype(np.float64)
    ga_all = a.reshape(ng, gs, 1)
    s0 = np.sqrt((ga_all * WT.reshape(ng, gs, rows) ** 2).sum(axis=1) / ga_all.sum(axis=1)) / fmt.rms
    d = np.maximum(s0.max(axis=0) * margin / top, 1e-30).astype(np.float32).astype(np.float64)
    codes = np.zeros((cols, rows), np.uint8)
    qs = np.zeros((ng, rows), np.uint8)
    tiny = 1e-30
    s = inv_s = np.ones(rows)
    for b0 in range(0, cols, batch):
        b1 = min(b0 + batch, cols)
        W1 = WT[b0:b1].copy()
        E1 = np.empty_like(W1)
        U1 = U[b0:b1, b0:b1]
        for i in range(b1 - b0):
            col = b0 + i
            if col % gs == 0:
                g = W1[i:i + gs]
                ga = a[col:col + gs][:, None]
                s = np.sqrt((ga * g * g).sum(axis=0) / ga.sum()) / fmt.rms
                for _ in range(iters):
                    lv = L[fmt.code_of(g / np.maximum(s, tiny))]
                    num, den = (ga * g * lv).sum(axis=0), (ga * lv * lv).sum(axis=0)
                    s = np.where(den > 0, num / np.maximum(den, tiny), s)
                u = np.clip(np.rint(s / d), 1, top)
                qs[col // gs] = u
                s = d * u
                inv_s = 1.0 / s
            wi = W1[i]
            c = fmt.code_of(wi * inv_s)
            codes[col] = c
            err = (wi - L[c] * s) / U1[i, i]
            W1[i + 1:] -= np.outer(U1[i, i + 1:], err)
            E1[i] = err
        WT[b1:] -= U[b0:b1, b1:].T @ E1
    return np.ascontiguousarray(codes.T), np.ascontiguousarray(qs.T), d.astype(np.float32)


def _vq_search(xw, aw, bw, beam):
    """xw [R, D] targets and aw, bw [K, D] the codebooks, all in whitened coordinates: the
    pair (i, j) with the least |xw - aw[i] - bw[j]|^2 among the `beam` nearest aw[i]. In
    float32: only the choice comes out of here, the caller rebuilds the vectors exactly."""
    xw, aw, bw = xw.astype(np.float32), aw.astype(np.float32), bw.astype(np.float32)
    d0 = (aw * aw).sum(1)[None, :] - 2 * xw @ aw.T
    top = np.argpartition(d0, beam, axis=1)[:, :beam] if beam < len(aw) else np.broadcast_to(np.arange(len(aw)), d0.shape)
    res = xw[:, None, :] - aw[top]                                           # [R, beam, D]
    d1 = (bw * bw).sum(1)[None, None, :] - 2 * res @ bw.T                     # + |res|^2, per (row, candidate)
    d1 += (res * res).sum(2)[:, :, None]
    flat = d1.reshape(len(xw), -1).argmin(1)
    k, j = flat // len(bw), flat % len(bw)
    return top[np.arange(len(xw)), k], j


def quantize_vq_gptq(w, U, fmt=VQ2G32, beam=VQ_BEAM, batch=128):
    """w f32 [R, C] (C % gs == 0) -> (codes u16 [R, C/D, 2], q u8 [R, C/gs], d f32 [R]) with
    w ~ d[r] * q[r, g] * (A[a] + B[b]): GPTQ over blocks of D columns (fmt.vdim). U is the upper
    Cholesky factor of H^-1; a block's vectors are searched in the coordinates U_BB^-1
    whitens (|(w - s c) U_BB^-1|, the block's cost once the later columns absorb its error),
    and the error is carried on as W[:, after] -= (delta U_BB^-1) U[B, after], lazily per
    `batch` columns as quantize_lowbit_gptq does. A group's scale q (1..15) is its current
    weights' RMS over d K, fixed when its first block comes up; d = the row's largest group
    RMS / (15 K), K the codebooks' scale (fmt.books)."""
    A, B, K = fmt.books
    D, G = fmt.vdim, fmt.gs
    W = w.astype(np.float64).copy()
    rows, cols = W.shape
    top = float(fmt.smax)
    rms = np.sqrt((W.reshape(rows, cols // G, G) ** 2).mean(axis=2))
    d = np.maximum(rms.max(axis=1) / (top * K), 1e-30).astype(np.float32).astype(np.float64)   # the stored f32
    codes = np.zeros((rows, cols // D, 2), np.uint16)
    qs = np.zeros((rows, cols // G), np.uint8)
    s = np.ones((rows, 1))
    for b0 in range(0, cols, batch):
        b1 = min(b0 + batch, cols)
        W1 = W[:, b0:b1].copy()
        E1 = np.zeros_like(W1)
        for c0 in range(b0, b1, D):
            i = c0 - b0
            if c0 % G == 0:
                q = np.clip(np.rint(np.sqrt((W1[:, i:i + G] ** 2).mean(axis=1)) / (d * K)), 1, top)
                qs[:, c0 // G] = q
                s = (d * q)[:, None]
            ubb = U[c0:c0 + D, c0:c0 + D]
            m = np.linalg.inv(ubb)
            ia, ib = _vq_search((W1[:, i:i + D] / s) @ m, A @ m, B @ m, beam)
            codes[:, c0 // D, 0], codes[:, c0 // D, 1] = ia, ib
            e = (W1[:, i:i + D] - (A[ia] + B[ib]) * s) @ m
            E1[:, i:i + D] = e
            if i + D < b1 - b0:
                W1[:, i + D:] -= e @ U[c0:c0 + D, c0 + D:b1]
        if b1 < cols:
            W[:, b1:] -= E1 @ U[b0:b1, b1:]
    return codes, qs, d.astype(np.float32)


def dequantize_vq(codes, q, d, fmt=VQ2G32):
    A, B, _ = fmt.books
    rows = codes.shape[0]
    v = (A[codes[:, :, 0]] + B[codes[:, :, 1]]).reshape(rows, -1, fmt.gs)
    return (v * (d[:, None].astype(np.float64) * q.astype(np.float64))[..., None]).reshape(rows, -1)


def _vq_vectors(w, fmt=VQ2G32):
    """a matrix's D-vectors over the RMS of their scale group (what the codebooks code)"""
    rows, cols = w.shape
    g = w.reshape(rows, cols // fmt.gs, fmt.gs).astype(np.float64)
    return (g / (np.sqrt((g * g).mean(axis=2, keepdims=True)) + 1e-12)).reshape(-1, fmt.vdim)


def _vq_nearest(x, c):
    out = np.empty(len(x), np.int64)
    cc = (c * c).sum(1)[None, :]
    for a in range(0, len(x), 65536):
        out[a:a + 65536] = (cc - 2 * x[a:a + 65536] @ c.T).argmin(1)
    return out


def _vq_kmeans(x, k, rng, iters=25):
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        idx = _vq_nearest(x, c)
        n = np.bincount(idx, minlength=k)
        for j in range(x.shape[1]):
            c[:, j] = np.where(n > 0, np.bincount(idx, weights=x[:, j], minlength=k) / np.maximum(n, 1), c[:, j])
        empty = np.nonzero(n == 0)[0]
        if len(empty):
            c[empty] = x[rng.integers(len(x), size=len(empty))]
    return c


def train_vq_books(sample, seed=38, iters=25, book=256):
    """sample [N, D] normalized vectors -> (a8, b8 int64 [book, D], K): k-means for A, then
    for what A leaves; both times K = 127 / max over coordinates of (max|A| + max|B|), rounded,
    so any a + b is an int8 (K lowered until the rounding keeps that too)."""
    rng = np.random.default_rng(seed)
    a = _vq_kmeans(sample, book, rng, iters)
    b = _vq_kmeans(sample - a[_vq_nearest(sample, a)], book, rng, iters)
    k = 127.0 / float((np.abs(a).max(axis=0) + np.abs(b).max(axis=0)).max())
    while True:
        a8, b8 = np.rint(a * k).astype(np.int64), np.rint(b * k).astype(np.int64)
        if int((np.abs(a8).max(axis=0) + np.abs(b8).max(axis=0)).max()) <= 127:
            return a8, b8, k
        k *= 0.999


def vq_book_sample(src, experts_per_layer=4, layer_step=4, vectors=400000, seed=38, fmt=VQ2G32):
    """the codebooks' training vectors: `experts_per_layer` experts of every `layer_step`-th
    layer, chosen with `seed`, their three matrices, an equal share of vectors from each"""
    rng = np.random.default_rng(seed)
    g = src.geometry
    picks = [(layer, int(e)) for layer in range(0, g.layers, layer_step)
             for e in rng.choice(g.experts, min(experts_per_layer, g.experts), replace=False)]
    share = max(1, vectors // (3 * len(picks)))
    out = []
    for layer, expert in picks:
        for k in range(3):
            v = _vq_vectors(src.matrix(layer, expert, k), fmt)
            out.append(v[rng.choice(len(v), min(share, len(v)), replace=False)])
    return np.concatenate(out), picks


def quantize_int2_g32(w, imp=None, iters=6):
    return quantize_lowbit(w, imp, INT2G32, iters)


def quantize_int2_g32_gptq(w, imp, U, iters=6, batch=128, margin=1.4):
    return quantize_lowbit_gptq(w, imp, U, INT2G32, iters, batch, margin)


class CovStats:
    """The engine's Q38_ACT_COV file: per layer sum_t x x^T of the routed experts' input.
    upper_inv(layer, imp) is the upper Cholesky factor of H^-1 for one expert, with
    H = D^1/2 C D^1/2: C the layer's correlations (damped by `damp` on the diagonal, as
    GPTQ damps H), D the expert's own mean squares (Q38_ACT_STATS); the factor of C^-1 is
    computed once per layer and an expert only rescales its columns."""
    def __init__(self, path, damp=0.01):
        with open(path, "rb") as f:
            if f.read(8) != b"Q38COV1\0":
                raise ValueError(f"{path}: not a Q38_ACT_COV file")
            self.layers, self.hidden = struct.unpack("<2i", f.read(8))
            self.n = np.frombuffer(f.read(8 * self.layers), np.int64).copy()
        self.mat = np.memmap(path, np.float64, "r", offset=16 + 8 * self.layers,
                             shape=(self.layers, self.hidden, self.hidden))
        self.damp, self._layer, self._u = damp, None, None

    def _factor(self, layer):
        if self._layer != layer:
            m = np.array(self.mat[layer])
            dg = np.sqrt(np.maximum(np.diag(m), 1e-30))
            c = m / dg[:, None] / dg[None, :] + self.damp * np.eye(self.hidden)
            self._u = np.linalg.cholesky(np.linalg.inv(c)).T     # C^-1 = U^T U, U upper
            self._layer = layer
        return self._u

    def upper_inv(self, layer, imp):
        return self._factor(layer) / np.sqrt(np.maximum(imp.astype(np.float64), 1e-30))[None, :]


class Dump:
    """The engine's Q38_ACT_DUMP directory: per layer, every calibration token's routed-expert
    input with its route (K expert ids, K gates). rows(layer, expert) are the inputs that expert
    really received and the gates they were weighed with; the layer's full matrix of inputs is
    kept for the prior of experts with few rows."""
    def __init__(self, path, max_tokens=None):
        self.dir, self.max_tokens, self._layer = path, max_tokens, None

    def _load(self, layer):
        if self._layer == layer:
            return
        recs = []   # several dumps (comma-separated directories, e.g. two calibration sets): their records in turn
        for d in self.dir.split(","):
            raw = np.fromfile(os.path.join(d, f"layer-{layer:03d}.bin"), np.uint8)
            if raw[:8].tobytes() != b"Q38DMP1\0":
                raise ValueError(f"{d}: layer {layer} is not a Q38_ACT_DUMP file")
            h, k = struct.unpack("<2i", raw[8:16].tobytes())
            width = 8 * k + 4 * h
            if recs and recs[0].shape[1] != width:
                raise ValueError(f"{d}: layer {layer} has another geometry than {self.dir.split(',')[0]}'s")
            recs.append(raw[16:16 + (len(raw) - 16) // width * width].reshape(-1, width))   # whole records only
        rec = recs[0] if len(recs) == 1 else np.concatenate(recs)
        if self.max_tokens:
            rec = rec[:self.max_tokens]
        self.ids = np.ascontiguousarray(rec[:, :4 * k]).view(np.int32)
        self.gates = np.ascontiguousarray(rec[:, 4 * k:8 * k]).view(np.float32)
        self.x = np.ascontiguousarray(rec[:, 8 * k:]).view(np.float32)
        self.pool = None
        self._layer = layer

    def rows(self, layer, expert):
        x, gate, _ = self.rows_idx(layer, expert)
        return x, gate

    def rows_idx(self, layer, expert):
        """rows(), and the token indices they are at (to pair them with another dump of the
        same calibration blocks)"""
        self._load(layer)
        hit = self.ids == expert
        rows = hit.any(axis=1)
        gate = (self.gates * hit).sum(axis=1)       # a route lists an expert once; summed all the same
        return self.x[rows].astype(np.float64), gate[rows].astype(np.float64), np.nonzero(rows)[0]

    def tokens(self, layer):
        self._load(layer)
        return len(self.x)

    def x_at(self, layer, idx):
        self._load(layer)
        return self.x[idx].astype(np.float64)

    def layer_moment(self, layer):
        """sum_t x x^T / tokens over every token of the layer (the prior)."""
        self._load(layer)
        if self.pool is None:
            x = self.x.astype(np.float64)
            self.pool = x.T @ x / max(len(x), 1)
        return self.pool


def _upper_inv(h):
    """The upper Cholesky factor U of h^-1 (h^-1 = U^T U), for GPTQ, without forming h^-1:
    with P the reversal, P h P = L L^T gives h^-1 = (P L^-T P)(P L^-1 P), so U = P L^-1 P,
    one Cholesky and one triangular inverse (half of inv + Cholesky). scipy for the
    triangular inverse where it is installed, numpy's general inverse otherwise."""
    lower = np.linalg.cholesky(np.ascontiguousarray(h[::-1, ::-1]))
    try:
        from scipy.linalg import lapack
        inv, info = lapack.dtrtri(lower, lower=1)
        if info:
            raise np.linalg.LinAlgError(f"dtrtri info {info}")
    except ImportError:
        inv = np.linalg.inv(lower)
    return np.ascontiguousarray(np.tril(inv)[::-1, ::-1])


def _hessian(x, wts, prior, alpha, damp):
    """sum_t w_t x_t x_t^T, plus alpha times the prior scaled to its trace, plus damp times
    its mean diagonal: rows are few next to the width, so the prior fills the directions
    they never visit."""
    h = (x * wts[:, None]).T @ x if len(x) else np.zeros((prior.shape[0],) * 2)
    if prior is not None:
        tp = float(np.trace(prior))
        th = float(np.trace(h))
        h = h + (alpha * th / tp if th > 0 and tp > 0 else 1.0) * prior
    h = h + damp * float(np.mean(np.diag(h))) * np.eye(h.shape[0])
    return h


def _target_weights(w, h, data, cross):
    """The weights GPTQ should quantize when the rows' targets are not w's own outputs on
    them: with inputs X~ (data = sum g^2 x~ x~^T, h = data + R, R the prior and damping
    added to it) and targets w x (cross = sum g^2 x x~^T over the same rows), the
    regularized least-squares map is W' = w (cross + R) h^-1, and
    sum g^2 |w x - q x~|^2 + |w - q|_R^2 = const + (W' - q) h (W' - q)^T, so GPTQ under h on
    W' minimizes it. cross = data (targets on the same inputs) gives W' = w."""
    m = np.linalg.solve(h, (cross + (h - data)).T).T
    return (w.astype(np.float64) @ m).astype(np.float32)


def quantize_expert_lowbit_exact(w3, x, gates, layer_prior, alpha=0.1, damp=0.01, gate_weight=True, fmt=INT2G32,
                                 x_orig=None):
    """One expert from the rows it really received: gate and up with GPTQ under
    H = sum_t g_t^2 x_t x_t^T (+ the layer's moment as a prior), then down with GPTQ under the
    SwiGLU outputs of the QUANTIZED gate and up on the same rows (what down will really see).
    w3 = (gate, up, down) f32; returns [(codes, q, d)] * 3.
    x_orig: the same tokens' inputs in the original model when x comes from a model whose
    earlier layers are already quantized (the sequential calibration): the targets are then
    the original expert's outputs on x_orig, the error upstream left in x compensated where
    a linear map can (_target_weights), down's against the original expert's output."""
    wts = gates ** 2 if gate_weight else np.ones(len(x))
    h = _hessian(x, wts, layer_prior, alpha, damp)
    u = _upper_inv(h)
    imp = np.diag(h).astype(np.float32)
    # gate and up share H: one pass over their stacked rows (rows are independent in GPTQ)
    rows = w3[0].shape[0]
    wgu = np.vstack([w3[0], w3[1]])
    if x_orig is not None and len(x):
        data = (x * wts[:, None]).T @ x
        wgu = _target_weights(wgu, h, data, (x_orig * wts[:, None]).T @ x)
    c, q, d = quantize_lowbit_gptq(wgu, imp, u, fmt)
    out = [(c[:rows], q[:rows], d[:rows]), (c[rows:], q[rows:], d[rows:])]
    if len(x):
        g = x @ dequantize_lowbit(*out[0], fmt).T
        hid = g / (1.0 + np.exp(-g)) * (x @ dequantize_lowbit(*out[1], fmt).T)
        data_d = (hid * wts[:, None]).T @ hid
        hd = data_d + alpha * np.diag(np.diag(data_d)) + damp * float(np.mean(np.diag(data_d)) or 1.0) * np.eye(data_d.shape[0])
        wd = w3[2]
        if x_orig is not None:   # down's target: the original expert's output on the original input
            go = x_orig @ w3[0].T.astype(np.float64)
            ho = go / (1.0 + np.exp(-go)) * (x_orig @ w3[1].T.astype(np.float64))
            wd = _target_weights(wd, hd, data_d, (ho * wts[:, None]).T @ hid)
        out.append(quantize_lowbit_gptq(wd, np.diag(hd).astype(np.float32), _upper_inv(hd), fmt))
    else:
        out.append(quantize_lowbit(w3[2], None, fmt))
    return out


def quantize_expert_int2_exact(w3, x, gates, layer_prior, alpha=0.1, damp=0.01, gate_weight=True):
    return quantize_expert_lowbit_exact(w3, x, gates, layer_prior, alpha, damp, gate_weight, INT2G32)


def pack_int2(codes):
    """u8 codes [R, C] (0..3) -> u8 [R, C/4]: blocks of 128, byte k holding elements
    k, k+32, k+64, k+96 in bits 0-1, 2-3, 4-5, 6-7 (int2_g32.h)."""
    rows, cols = codes.shape
    c = codes.reshape(rows, cols // 128, 4, 32)
    return (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).reshape(rows, cols // 4)


def pack_lowbit(codes, fmt):
    """u8 codes [R, C] -> the rows as fmt lays them: per block of 128 the low two bits as
    pack_int2 does (32 bytes), then for 3 bits the high bits (16 bytes: byte k holds
    elements k, k+16, ..., k+112 in bits 0..7, lowbit_g16.h)."""
    rows, cols = codes.shape[0], codes.shape[1]
    if fmt.vq:   # a0 b0 a1 b1 ...: a byte each, or 10 bits each, eight to 10 bytes, LSB first
        flat = codes.reshape(rows, -1)
        if fmt.ibits == 8:
            return np.ascontiguousarray(flat.astype(np.uint8))
        e = flat.reshape(rows, -1, 8).astype(np.uint64)
        lo = sum(e[:, :, k] << np.uint64(10 * k) for k in range(6)) | ((e[:, :, 6] & np.uint64(15)) << np.uint64(60))
        hi = ((e[:, :, 6] >> np.uint64(4)) | (e[:, :, 7] << np.uint64(6))).astype(np.uint16)
        out = np.concatenate([lo.astype("<u8").view(np.uint8).reshape(rows, -1, 8),
                              hi.astype("<u2").view(np.uint8).reshape(rows, -1, 2)], axis=2)
        return np.ascontiguousarray(out.reshape(rows, -1))
    low = pack_int2(codes & 3)
    if fmt.bits == 2:
        return low
    hi = ((codes >> 2) & 1).astype(np.uint8).reshape(rows, cols // 128, 8, 16)
    plane = np.packbits(hi, axis=2, bitorder="little").reshape(rows, cols // 128, 16)
    return np.concatenate([low.reshape(rows, cols // 128, 32), plane], axis=2).reshape(rows, fmt.row_code_bytes(cols))


def pack_scales(q, fmt):
    """u8 group scales [R, C/gs] -> as stored: a byte each, or two nibbles to a byte
    (group 2k low, 2k+1 high)."""
    if fmt.smax == 255:
        return np.ascontiguousarray(q.astype(np.uint8))
    q = q.astype(np.uint8)
    return np.ascontiguousarray(q[:, 0::2] | (q[:, 1::2] << 4))


def dequantize_lowbit(codes, q, d, fmt):
    if fmt.vq:
        return dequantize_vq(codes, q, d, fmt)
    rows, cols = codes.shape
    return (fmt.levels[codes].reshape(rows, cols // fmt.gs, fmt.gs) *
            (d[:, None].astype(np.float64) * q.astype(np.float64))[..., None]).reshape(rows, cols)


def unpack_lowbit(packed, cols, fmt):
    """pack_lowbit's inverse: u8 [R, row_code_bytes] -> codes [R, C] ([R, C/8, 2] for vq)"""
    if fmt.vq:
        rows = packed.shape[0]
        if fmt.ibits == 8:
            return packed.reshape(rows, cols // fmt.vdim, 2).astype(np.uint16)
        g = packed.reshape(rows, -1, 10)
        lo = np.ascontiguousarray(g[:, :, :8]).view("<u8")[..., 0]
        hi = np.ascontiguousarray(g[:, :, 8:]).view("<u2")[..., 0].astype(np.uint64)
        idx = [(lo >> np.uint64(10 * k)) & np.uint64(1023) for k in range(6)]
        idx.append(((lo >> np.uint64(60)) | (hi << np.uint64(4))) & np.uint64(1023))
        idx.append((hi >> np.uint64(6)) & np.uint64(1023))
        return np.stack(idx, axis=2).reshape(rows, cols // fmt.vdim, 2).astype(np.uint16)
    rows, nb = packed.shape[0], cols // 128
    blk = packed.reshape(rows, nb, 48 if fmt.bits == 3 else 32)
    low = blk[:, :, :32]
    c = np.stack([(low >> (2 * h)) & 3 for h in range(4)], axis=2).reshape(rows, nb, 128)
    if fmt.bits == 3:   # byte k of the plane: elements k, k+16, ..., k+112 in bits 0..7
        bits = np.unpackbits(blk[:, :, 32:], axis=2, bitorder="little").reshape(rows, nb, 16, 8)
        c = c | (bits.transpose(0, 1, 3, 2).reshape(rows, nb, 128) << 2)
    return c.reshape(rows, cols).astype(np.uint8)


def unpack_scales(packed, fmt):
    """pack_scales' inverse"""
    if fmt.smax == 255:
        return packed.astype(np.uint8)
    q = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    q[:, 0::2] = packed & 15
    q[:, 1::2] = packed >> 4
    return q


def output_rescale(w3, wq, x, gates, lo=0.8, hi=1.25):
    """The factor that brings a quantized expert's output back to the original's scale on
    its calibration rows: <y, yq> / <yq, yq>, each row weighted by its squared gate (the
    weight it has in the layer's sum), clipped to [lo, hi]; 1 without rows. Measured on 23
    Qwen3.8 experts, rows never seen: least-squares low-bit experts come out ~4% smaller
    than the original at every token (best rescale 1.039 int2-g16, 1.035 int3-g16; ISTA's
    Q2_0 1.005), an error with the same sign in every layer; this factor takes it to 1.008."""
    if len(x) == 0:
        return 1.0
    x = np.asarray(x, dtype=np.float64)
    g2 = np.asarray(gates, dtype=np.float64) ** 2

    def out(w):
        g = x @ w[0].T.astype(np.float64)
        return (g / (1 + np.exp(-g)) * (x @ w[1].T.astype(np.float64))) @ w[2].T.astype(np.float64)

    y, yq = out(w3), out(wq)
    num, den = float((g2[:, None] * y * yq).sum()), float((g2[:, None] * yq * yq).sum())
    return float(np.clip(num / den, lo, hi)) if den > 0 else 1.0


def dequantize_int2(codes, q, d):
    return dequantize_lowbit(codes, q, d, INT2G32)


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
    def __init__(self, layers, experts, hidden, inter, bits=4, fmt=None):
        self.layers, self.experts, self.hidden, self.inter = layers, experts, hidden, inter
        self.fmt = fmt if fmt is not None else (INT2G32 if bits == 2 else None)
        self.bits = self.fmt.bits if self.fmt else 4
        self.shape = ((inter, hidden), (inter, hidden), (hidden, inter))
        if self.fmt:
            if hidden % 128 or inter % 128:
                raise ValueError(f"hidden={hidden}, moe_intermediate_size={inter}: {self.fmt.name} needs "
                                 f"widths that are multiples of 128")
            self.code_bytes = [r * self.fmt.row_code_bytes(c) for r, c in self.shape]
            self.q_count = [r * self.fmt.row_scale_bytes(c) for r, c in self.shape]
            self.scale_count = [r for r, c in self.shape]
            self.scale_off = sum(self.code_bytes) + sum(self.q_count)
            self.record_bytes = self.scale_off + 4 * sum(self.scale_count)
        else:
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
        if self.fmt:
            for k, (r, c) in enumerate(self.shape):
                out.append((f"{base}.{PROJ[k]}.weight", "U8", [r, self.fmt.row_code_bytes(c)], off, self.code_bytes[k]))
                off += self.code_bytes[k]
            for k, (r, c) in enumerate(self.shape):
                out.append((f"{base}.{PROJ[k]}.weight{self.fmt.q_suffix}", "U8", [r, self.fmt.row_scale_bytes(c)],
                            off, self.q_count[k]))
                off += self.q_count[k]
            for k, (r, c) in enumerate(self.shape):
                out.append((f"{base}.{PROJ[k]}.weight.d", "F32", [r, 1], off, 4 * r))
                off += 4 * r
            assert off == self.record_bytes
            return out
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

_pread_lock = threading.Lock()


def _pread(fd, nbytes, offset):
    """os.pread where the platform has it. Windows has none (Python 3.14 included):
    there a seek and a read, under a lock, since they move the descriptor's offset."""
    pread = getattr(os, "pread", None)
    if pread is not None:
        return pread(fd, nbytes, offset)
    with _pread_lock:
        os.lseek(fd, offset, os.SEEK_SET)
        return os.read(fd, nbytes)


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
            chunk = _pread(fd, nbytes - len(out), offset + len(out))
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
_ACT = None
_BITS = 4
_COV = None
_DUMP = None
_FMT = None
_ALPHA = 0.1


def _init_worker(model_dir, act_path=None, bits=4, cov_path=None, dump_path=None, fmt_name=None, alpha=0.1,
                 rescale=False, target_path=None, dump_tokens=None, books=None):
    global _SOURCE, _ACT, _BITS, _COV, _DUMP, _FMT, _ALPHA, _RESCALE, _TARGET
    if books is not None:
        LOWBIT[fmt_name].set_books(*books)
    _SOURCE = Source(model_dir)
    _ACT = ActStats(act_path) if act_path else None
    _BITS = bits
    _COV = CovStats(cov_path) if cov_path else None
    _DUMP = Dump(dump_path, max_tokens=dump_tokens) if dump_path else None
    _FMT = LOWBIT[fmt_name] if fmt_name else (INT2G32 if bits == 2 else None)
    _ALPHA = alpha
    _RESCALE = rescale
    _TARGET = Dump(target_path, max_tokens=dump_tokens) if target_path else None


def convert_expert_int2(layer, expert):
    """One expert in a low-bit format (_FMT): codes, then group scales, then row scales."""
    fmt = _FMT
    codes, qs, ds, errors = [], [], [], []
    w3 = [_SOURCE.matrix(layer, expert, k) for k in range(3)]
    if _DUMP is not None:
        x, gates, idx = _DUMP.rows_idx(layer, expert)
        x_orig = None
        if _TARGET is not None:   # the same tokens in the original model (the sequential calibration)
            if _TARGET.tokens(layer) != _DUMP.tokens(layer):
                raise RuntimeError(f"layer {layer}: the --dump has {_DUMP.tokens(layer)} tokens, the --dump-target "
                                   f"{_TARGET.tokens(layer)}: not the same calibration blocks")
            x_orig = _TARGET.x_at(layer, idx)
        out = quantize_expert_lowbit_exact(w3, x, gates, _DUMP.layer_moment(layer), alpha=_ALPHA, fmt=fmt, x_orig=x_orig)
        if _RESCALE:   # down's row scales carry the factor
            a = output_rescale(w3, [dequantize_lowbit(*o, fmt) for o in out], x, gates)
            out[2] = (out[2][0], out[2][1], (out[2][2].astype(np.float64) * a).astype(np.float32))
    else:
        out = []
        for k in range(3):
            imp = _ACT.importance(layer, expert, k) if _ACT else None
            if _COV is not None and k < 2:   # gate and up read the layer's hidden row; down an expert's own
                out.append(quantize_lowbit_gptq(w3[k], imp, _COV.upper_inv(
                    layer, imp if imp is not None else np.ones(w3[k].shape[1])), fmt))
            else:
                out.append(quantize_lowbit(w3[k], imp, fmt))
    for k, (c, q, d) in enumerate(out):
        errors.append(rel_l2(w3[k], dequantize_lowbit(c, q, d, fmt)))
        codes.append(pack_lowbit(c, fmt).tobytes())
        qs.append(pack_scales(q, fmt).tobytes())
        ds.append(d.astype(np.float32).tobytes())
    return b"".join(codes) + b"".join(qs) + b"".join(ds), errors, peak_rss_mb()


def convert_expert(task):
    """(layer, expert) -> (the record's bytes, relative L2 error of gate, up,
    down, this process's peak RSS in MB)."""
    layer, expert = task
    if _BITS == 2:
        return convert_expert_int2(layer, expert)
    codes, scales, errors = [], [], []
    for k in range(3):
        w = _SOURCE.matrix(layer, expert, k)
        q, s = quantize_g64_weighted(w, _ACT.importance(layer, expert, k)) if _ACT else quantize_g64(w)
        errors.append(rel_l2(w, dequantize(q, s)))
        codes.append(pack_planar(q).tobytes())
        scales.append(s.astype(np.float32).tobytes())
    return b"".join(codes) + b"".join(scales), errors, peak_rss_mb()


def layer_file(layer):
    return f"layer-{layer:03d}.safetensors"


def layer_header(geometry, prefix, layer):
    """The layer file's header, padded so the data region starts on DATA_ALIGN."""
    header, meta = {}, {"format": geometry.fmt.format if geometry.fmt else FORMAT, "version": str(VERSION), "layer": str(layer)}
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
    if g.fmt:
        out = {
            "format": g.fmt.format, "version": VERSION, "complete": False,
            "group_size": g.fmt.gs, "codes": g.fmt.codes, "scales": g.fmt.scales,
            "layers": g.layers, "experts": g.experts,
            "hidden_size": g.hidden, "moe_intermediate_size": g.inter,
            "prefix": src.prefix, "record_bytes": g.record_bytes,
            "record": ["gate codes", "up codes", "down codes", "gate q", "up q", "down q",
                       "gate d", "up d", "down d"],
            "quantizer": f"per {g.fmt.gs} inputs: levels {' '.join(str(int(v)) for v in g.fmt.levels)}, scale and "
                         f"codes fitted in turn (uniform weights), scales as integers to {g.fmt.smax} under the "
                         f"row's largest and refitted",
            "files": [layer_file(i) for i in range(g.layers)],
            "done": [],
            "source": src.fingerprint(),
        }
        if g.fmt.vq:
            f = g.fmt
            out["quantizer"] = (f"per {f.vdim} inputs one vector of two {f.book} x {f.vdim} int8 codebooks, the pair searched "
                                f"with a beam of {VQ_BEAM}; a 4-bit scale per {f.gs} (the group's RMS) under the row's largest "
                                f"(uniform weights)")
        return out
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
    for key in ("quantizer", "output_rescale", "output_rescale_factors", "codebooks"):   # what the earlier run recorded
        if key in index:
            fresh[key] = index[key]
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


def _rescale_expert(task):
    """(layer, expert, path, offset of its record) -> (expert, factor, down's new row scales)"""
    layer, expert, path, off = task
    fmt, geo = _FMT, _SOURCE.geometry
    g = Geometry(geo.layers, geo.experts, geo.hidden, geo.inter, fmt=fmt)
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(g.record_bytes)
    if len(raw) != g.record_bytes:
        raise RuntimeError(f"{path}: short record of expert {expert} ({len(raw)} bytes)")
    rec = np.frombuffer(raw, dtype=np.uint8)
    wq, pos = [], [0, 0, 0]
    o = 0
    for k in range(3):
        pos[k] = o; o += g.code_bytes[k]
    qo = o
    for k, (r, c) in enumerate(g.shape):
        codes = unpack_lowbit(rec[pos[k]:pos[k] + g.code_bytes[k]].reshape(r, fmt.row_code_bytes(c)), c, fmt)
        qoff = qo + sum(g.q_count[:k])
        q = unpack_scales(rec[qoff:qoff + g.q_count[k]].reshape(r, fmt.row_scale_bytes(c)), fmt)
        doff = g.scale_off + 4 * sum(g.scale_count[:k])
        d = rec[doff:doff + 4 * r].view(np.float32)
        wq.append((codes, q, d))
    x, gates = _DUMP.rows(layer, expert)
    w3 = [_SOURCE.matrix(layer, expert, k) for k in range(3)]
    a = output_rescale(w3, [dequantize_lowbit(*t, fmt) for t in wq], x, gates)
    return expert, a, (wq[2][2].astype(np.float64) * a).astype(np.float32).tobytes()


def rescale_sidecar(model_dir, out_dir, dump_path, fmt_name, workers=1, layers=None):
    """--rescale-only: an existing low-bit sidecar's down row scales times each expert's
    output_rescale factor, in place; the index lists the layers done (never scaled twice)
    and each one's factors (dividing by them undoes it)."""
    src = Source(model_dir)
    geo = src.geometry
    src.geometry = Geometry(geo.layers, geo.experts, geo.hidden, geo.inter, fmt=LOWBIT[fmt_name])
    g = src.geometry
    with open(os.path.join(out_dir, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    if index.get("format") != g.fmt.format or index.get("record_bytes") != g.record_bytes or not index.get("complete"):
        sys.exit(f"{out_dir}: not a complete {fmt_name} sidecar of this snapshot")
    books = index_books(index, g.fmt) if g.fmt.vq else None
    done = set(index.get("output_rescale", []))
    todo = [i for i in parse_layers(layers, g.layers) if i not in done]
    pool = None
    if workers > 1 and todo:
        from multiprocessing import get_context
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        pool = get_context("spawn").Pool(workers, initializer=_init_worker,
                                         initargs=(model_dir, None, 2, None, dump_path, fmt_name, 0.1, False, None,
                                                   None, books))
    else:
        _init_worker(model_dir, None, 2, None, dump_path, fmt_name, 0.1, books=books)
    down_off = g.scale_off + 4 * (g.scale_count[0] + g.scale_count[1])
    try:
        for layer in todo:
            started = time.time()
            path = os.path.join(out_dir, layer_file(layer))
            head = len(layer_header(g, src.prefix, layer))
            tasks = [(layer, e, path, head + e * g.record_bytes) for e in range(g.experts)]
            res = pool.imap(_rescale_expert, tasks, chunksize=4) if pool else map(_rescale_expert, tasks)
            factors = []
            with open(path, "r+b") as f:
                for expert, a, d in res:   # one writer: seek and write (no pwrite on Windows)
                    f.seek(head + expert * g.record_bytes + down_off)
                    f.write(d)
                    factors.append(a)
                f.flush()
                os.fsync(f.fileno())
            done.add(layer)
            index["output_rescale"] = sorted(done)
            index.setdefault("output_rescale_factors", {})[str(layer)] = [round(float(v), 6) for v in factors]
            write_index(out_dir, index)
            fa = np.array(factors)
            print(f"  layer {layer:3d}: {len(fa)} experts in {time.time() - started:6.1f}s, output rescale mean "
                  f"{fa.mean():.4f} (min {fa.min():.4f} max {fa.max():.4f}, {int((fa == 1.0).sum())} without rows)", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    print(f"{out_dir}: output rescale on {len(done)}/{g.layers} layers")


def index_books(index, fmt=VQ2G32):
    """(a8, b8, K) from a vq sidecar's index.json"""
    cb = index.get("codebooks") or {}
    shape = (fmt.book, fmt.vdim)
    try:
        return (np.array(cb["a"], np.int64).reshape(shape), np.array(cb["b"], np.int64).reshape(shape), float(cb["scale"]))
    except (KeyError, ValueError, TypeError):
        sys.exit(f"index.json: no vq codebooks (a, b: {fmt.book} x {fmt.vdim}, scale)")


def convert(model_dir, out_dir=None, workers=1, layers=None, plan=False, act_path=None, bits=4, cov_path=None,
            dump_path=None, fmt_name=None, alpha=0.1, rescale=False, target_path=None, dump_tokens=None):
    """Write (or finish) model_dir's sidecar; returns its directory."""
    src = Source(model_dir)
    if fmt_name or bits == 2:
        fmt_name = fmt_name or INT2G32.name
        bits = 2
        geo = src.geometry
        try:
            src.geometry = Geometry(geo.layers, geo.experts, geo.hidden, geo.inter, fmt=LOWBIT[fmt_name])
        except ValueError as exc:
            sys.exit(f"{model_dir}: {exc}")
    act = ActStats(act_path) if act_path else None
    if act and (act.layers, act.experts, act.hidden, act.inter) != (src.geometry.layers, src.geometry.experts,
                                                                  src.geometry.hidden, src.geometry.inter):
        sys.exit(f"{act_path}: statistics of another geometry")
    g = src.geometry
    out_dir = out_dir or os.path.join(model_dir, g.fmt.sidecar if g.fmt else SIDECAR_DIR)
    label = g.fmt.name if g.fmt else "int4-g64"
    total = sum(layer_file_bytes(g, src.prefix, i) for i in range(g.layers))
    fp8 = g.fp8_expert_bytes() * g.experts * g.layers
    print(f"{model_dir}: {g.layers} layers x {g.experts} experts, hidden {g.hidden}, "
          f"moe_intermediate {g.inter}, prefix {src.prefix}")
    print(f"  per expert: FP8 {g.fp8_expert_bytes():,} bytes -> {label} {g.record_bytes:,} bytes "
          f"({100 * g.record_bytes / g.fp8_expert_bytes():.1f}%)")
    print(f"  sidecar {out_dir}: {total:,} bytes ({total / 2**30:.2f} GiB) "
          f"for {fp8:,} bytes of FP8 experts")
    if plan:
        return out_dir

    os.makedirs(out_dir, exist_ok=True)
    index = resume_index(out_dir, src)
    if (cov_path or dump_path) and bits != 2:
        sys.exit("--cov and --dump are for --int2 and --lowbit")
    if dump_path and (cov_path or act_path):
        sys.exit("--dump carries the statistics itself: no --act-stats or --cov with it")
    cov = CovStats(cov_path) if cov_path else None
    if cov and (cov.layers, cov.hidden) != (g.layers, g.hidden):
        sys.exit(f"{cov_path}: matrices of another geometry")
    if bits == 2:
        quantizer = (f"weighted: per 32 inputs levels -10 -3 3 10, scale and codes fitted in turn against the "
                     f"activation-weighted squared error (Q38_ACT_STATS {os.path.basename(act_path)}, "
                     f"{act.rows} expert rows), scales as u8 under the row's largest and refitted") if act else None
        if dump_path:
            quantizer = (f"{g.fmt.name}, every expert fitted to the rows it received (Q38_ACT_DUMP "
                         f"{os.path.basename(dump_path.rstrip('/'))}): gate and up with GPTQ under H = sum g^2 x x^T + "
                         f"{alpha} x the layer's moment, down with GPTQ under the quantized gate/up's SwiGLU outputs"
                         + ("; down's row scales times the output rescale on those rows" if rescale else "")
                         + ("; inputs from a model with the earlier layers quantized, targets the original model's "
                            "outputs on the same tokens (--dump-target, the sequential calibration)" if target_path else ""))
        if cov:
            quantizer = (quantizer or "uniform weights") + (
                f"; gate and up with the error carried column by column (GPTQ) under H = D^1/2 C D^1/2 "
                f"(Q38_ACT_COV {os.path.basename(cov_path)}, {int(cov.n[0])} tokens, damp {cov.damp})")
    else:
        quantizer = (f"weighted: per 64 inputs the scale among absmax/7 * {list(ALPHAS)} with the least "
                     f"activation-weighted squared error (Q38_ACT_STATS {os.path.basename(act_path)}, "
                     f"{act.rows} expert rows)") if act else None
    if quantizer and g.fmt and g.fmt.vq:
        quantizer = quantizer.replace(f"{g.fmt.name}, ", f"{g.fmt.name} ({g.fmt.vdim}-vectors of two int8 codebooks, beam "
                                                         f"{VQ_BEAM}, GPTQ over blocks of {g.fmt.vdim} columns), ", 1)
    if quantizer and index["done"] and index.get("quantizer") != quantizer:
        sys.exit(f"{out_dir}: started with another quantizer; give a new --out")
    if quantizer:
        index["quantizer"] = quantizer
    books = None
    if g.fmt and g.fmt.vq:
        if "codebooks" in index:   # a resumed run keeps the books its layers were coded with
            books = index_books(index, g.fmt)
        else:
            started = time.time()
            sample, picks = vq_book_sample(src, fmt=g.fmt)
            books = train_vq_books(sample, book=g.fmt.book)
            index["codebooks"] = {"a": books[0].ravel().tolist(), "b": books[1].ravel().tolist(), "scale": books[2],
                                  "fitted_on": f"{len(sample)} {g.fmt.vdim}-vectors of {len(picks)} experts "
                                               f"(layers {picks[0][0]}..{picks[-1][0]})"}
            print(f"  vq codebooks: k-means on {len(sample)} vectors of {len(picks)} experts in "
                  f"{time.time() - started:.0f}s, int8 scale {books[2]:.2f}", flush=True)
        g.fmt.set_books(*books)
    wanted = parse_layers(layers, g.layers)
    todo = [i for i in wanted if i not in index["done"]]
    if len(wanted) > len(todo):
        print(f"  {len(wanted) - len(todo)} layer(s) already converted, skipped")
    write_index(out_dir, index)

    pool = None
    if workers > 1 and todo:
        from multiprocessing import get_context
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")   # one BLAS thread per worker: the workers are the parallelism
        pool = get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(model_dir, act_path, bits, cov_path, dump_path, fmt_name, alpha, rescale, target_path, dump_tokens, books))
    else:
        _init_worker(model_dir, act_path, bits, cov_path, dump_path, fmt_name, alpha, rescale, target_path, dump_tokens, books)
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
    ap.add_argument("--act-stats", help="the engine's Q38_ACT_STATS file: each group's scale chosen against "
                                        "activation-weighted error instead of absmax/7")
    ap.add_argument("--int2", action="store_true", help=f"write int2-g32 (int2_g32.h) into <model>/{SIDECAR_DIR2}/ "
                                                        "instead (qwen38: Q38_EXPERT_INT2=1)")
    ap.add_argument("--cov", help="the engine's Q38_ACT_COV file (with --int2): gate and up quantized with the "
                                  "error carried column by column under the layer's input correlations")
    ap.add_argument("--dump", help="the engine's Q38_ACT_DUMP directory, or several comma-separated (with --int2): every expert fitted with GPTQ "
                                   "to the inputs it really received, down to the SwiGLU outputs of its quantized gate/up")
    ap.add_argument("--lowbit", choices=sorted(LOWBIT), help="a low-bit format: int2g32 (= --int2), int2g16 or int3g16 "
                                                            "(groups of 16 with 4-bit scales), vq2g32 (8-vectors of two "
                                                            "int8 codebooks), into <model>/experts-<name>/")
    ap.add_argument("--alpha", type=float, default=0.1, help="with --dump: the weight of the layer's moment in each "
                                                             "expert's H (default 0.1)")
    ap.add_argument("--rescale", action="store_true", help="with --dump: down's row scales times each expert's output "
                                                           "rescale on its rows (the least-squares fit's ~4%% shrink undone)")
    ap.add_argument("--rescale-only", action="store_true", help="with --dump and --lowbit: apply --rescale to the "
                                                                "existing sidecar --out in place (layers done are kept in "
                                                                "its index, never scaled twice)")
    ap.add_argument("--dump-target", help="with --dump: the original model's Q38_ACT_DUMP over the same calibration "
                                          "blocks, when --dump comes from a model whose earlier layers are already "
                                          "quantized (Q38_EXPERT_PARTIAL): the targets are the original outputs")
    ap.add_argument("--dump-tokens", type=int, help="with --dump: only each layer's first N tokens (the same calibration "
                                                     "blocks in dumps of different lengths)")
    a = ap.parse_args()
    if a.dump_target and not a.dump:
        sys.exit("--dump-target needs --dump")
    if a.rescale_only:
        if not (a.dump and a.lowbit):
            sys.exit("--rescale-only needs --dump and --lowbit")
        fmt = LOWBIT[a.lowbit]
        rescale_sidecar(a.model, a.out or os.path.join(a.model, fmt.sidecar), a.dump, a.lowbit, a.workers, a.layers)
        return
    if a.rescale and not a.dump:
        sys.exit("--rescale needs --dump")
    convert(a.model, a.out, a.workers, a.layers, a.plan, a.act_stats, 2 if (a.int2 or a.lowbit) else 4, a.cov, a.dump,
            a.lowbit or ("int2g32" if a.int2 else None), a.alpha, a.rescale, a.dump_target, a.dump_tokens)


if __name__ == "__main__":
    main()
