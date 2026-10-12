#!/usr/bin/env python3
"""Reference data for tests/test_kolibri1_kernels.c (kernel components).

NumPy float64 only. Nothing here imports the engine, the reference implementation or any colibri code; the
reference of every kernel is the plain mathematical definition evaluated in
double precision on the same float32 inputs the C test reads:

  fp8     y = x @ W^T, W = e4m3(code) * scale[o/128, i/128] (exact in float64)
  bf16    y = x @ W^T with bf16 widened exactly
  router  the same, for the router logits (bf16 gate)
  norm    w * x / sqrt(mean(x^2) + eps)
  rope    cos/sin of (position * theta^(-2i/d)) in float64, rounded once to float32,
          and rotate-half on random head vectors
  route   top-k of float32(z + bias), ties to the lowest id; weights = sigmoid(z[id])
  sigmoid the weight formula alone, on 100,000 logits

The input classes: random, wide dynamic range, and ADVERSARIAL cancelling inputs
(adjacent column pairs carry the same weight code and x of opposite sign and large
magnitude, so terms of size |w| B cancel to |w| d), plus one heavier cancelling
DIAG case that is reported but not gated.

cases.txt is the manifest the C test reads.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SEED = 20261006


def e4m3(codes):
    c = codes.astype(np.int64)
    s, e, m = c >> 7, (c >> 3) & 15, c & 7
    v = np.where(e == 0, m * 2.0 ** -9, (8 + m) * np.exp2((e - 10).astype(np.float64)))
    v = np.where(s == 1, -v, v)
    return np.where((e == 15) & (m == 7), np.nan, v)          # float64


def bf16_bits(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return (((bits >> 16) & 1) + np.uint32(0x7FFF) + bits >> 16).astype(np.uint16)


def bf16_to_f64(b):
    return (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


class Out:
    def __init__(self, d):
        self.d = Path(d)
        self.d.mkdir(parents=True, exist_ok=True)
        self.lines = []

    def put(self, name, arr):
        np.ascontiguousarray(arr).tofile(self.d / name)

    def line(self, *f):
        self.lines.append(" ".join(str(x) for x in f))

    def close(self):
        (self.d / "cases.txt").write_text("\n".join(self.lines) + "\n")


def fp8_case(out, rng, name, O, I, S, kind="random", gate=1):
    gr, gc = -(-O // 128), -(-I // 128)
    codes = rng.integers(0, 256, (O, I), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x38                      # no NaN codes: a gate needs finite output
    grid = np.exp2(rng.uniform(-9, -4, (gr, gc))).astype(np.float32)
    if kind in ("cancel", "cancel_heavy"):
        codes[:, 1::2][:, : I // 2] = codes[:, 0::2][:, : I // 2]   # pairs share the weight code (and the block)
        big = (rng.uniform(1, 8, (S, I // 2)) if kind == "cancel" else rng.uniform(300, 3000, (S, I // 2)))
        d = rng.normal(0, 1, (S, I))
        x = np.zeros((S, I))
        x[:, 0::2][:, : I // 2] = big + d[:, 0::2][:, : I // 2]
        x[:, 1::2][:, : I // 2] = -big + d[:, 1::2][:, : I // 2]
        x = x.astype(np.float32)
    elif kind == "range":
        x = (rng.choice([-1, 1], (S, I)) * np.exp2(rng.uniform(-8, 8, (S, I)))).astype(np.float32)
    else:
        x = rng.normal(0, 1, (S, I)).astype(np.float32)
    w = e4m3(codes) * np.repeat(np.repeat(grid.astype(np.float64), 128, 0), 128, 1)[:O, :I]
    y = x.astype(np.float64) @ w.T
    out.put(f"{name}.codes.u8", codes); out.put(f"{name}.grid.f32", grid)
    out.put(f"{name}.x.f32", x); out.put(f"{name}.y.f64", y)
    out.line("fp8", name, O, I, S, gate)


def bf16_case(out, rng, name, O, I, S, kind, gate=1):
    w = rng.normal(0, 0.05, (O, I)).astype(np.float32)
    wb = bf16_bits(w)
    if kind == "cancel":
        wb[:, 1::2][:, : I // 2] = wb[:, 0::2][:, : I // 2]
        big = rng.uniform(1, 8, (S, I // 2))
        x = np.zeros((S, I)); d = rng.normal(0, 1, (S, I))
        x[:, 0::2][:, : I // 2] = big + d[:, 0::2][:, : I // 2]
        x[:, 1::2][:, : I // 2] = -big + d[:, 1::2][:, : I // 2]
        x = x.astype(np.float32)
    else:
        x = rng.normal(0, 1, (S, I)).astype(np.float32)
    y = x.astype(np.float64) @ bf16_to_f64(wb).T
    out.put(f"{name}.w.u16", wb); out.put(f"{name}.x.f32", x); out.put(f"{name}.y.f64", y)
    return wb


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Out(args.out)
    rng = np.random.default_rng(SEED)

    # ---- FP8 matmul ---------------------------------------------------------------
    fp8_case(out, rng, "fp8_small", 7, 192, 3)
    fp8_case(out, rng, "fp8_partial", 130, 300, 9)                 # partial row block, column tail of 4
    fp8_case(out, rng, "fp8_gate", 512, 2560, 17)                  # the real expert gate/up shape
    fp8_case(out, rng, "fp8_down", 2560, 512, 5)                   # the real expert down shape
    fp8_case(out, rng, "fp8_range", 256, 640, 6, "range")
    fp8_case(out, rng, "fp8_cancel", 256, 1024, 8, "cancel")
    fp8_case(out, rng, "fp8_cancel_heavy", 64, 2560, 4, "cancel_heavy", gate=0)   # DIAG, reported not gated

    # ---- bf16 matmul and router logits ------------------------------------------------
    for name, O, I, S, kind in (("bf16_head", 640, 2560, 7, "random"), ("bf16_odd", 33, 203, 3, "random"),
                                ("bf16_cancel", 128, 1024, 5, "cancel")):
        bf16_case(out, rng, name, O, I, S, kind)
        out.line("bf16", name, O, I, S, 1)
    bf16_case(out, rng, "router", 384, 2560, 6, "random")
    out.line("router", "router", 384, 2560, 6, 1)

    # ---- rmsnorm ----------------------------------------------------------------------
    eps = np.float32(1e-6)
    for name, n, scale in (("norm_2560", 2560, 1.0), ("norm_128", 128, 1.0), ("norm_odd", 203, 1.0),
                           ("norm_big", 2560, 1e3), ("norm_small", 2560, 1e-3)):
        rows = 8
        x = (rng.normal(0, 1, (rows, n)) * scale).astype(np.float32)
        w = (1.0 + rng.normal(0, 0.1, n)).astype(np.float32)
        xd = x.astype(np.float64)
        y = w.astype(np.float64) * xd / np.sqrt((xd ** 2).mean(axis=1, keepdims=True) + float(eps))
        out.put(f"{name}.x.f32", x); out.put(f"{name}.w.f32", w); out.put(f"{name}.y.f64", y)
        out.line("norm", name, n, rows, 1)

    # ---- rope tables and rotation --------------------------------------------------------
    hd, theta = 128, 10000.0
    fixed = [0, 1, 2, 511, 512, 513, 1026, 1027, 4096, 16383, 16384, 16385, 100000, 262142, 262143]
    rest = np.setdiff1d(rng.choice(262144, 1100, replace=False), fixed)[: 1000 - len(fixed)]
    pos = np.concatenate([fixed, rest]).astype(np.int64)
    assert len(np.unique(pos)) == 1000
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    ang = pos[:, None].astype(np.float64) * inv[None, :]
    out.put("rope.pos.i64", pos)
    out.put("rope.cos.f32", np.cos(ang).astype(np.float32)); out.put("rope.sin.f32", np.sin(ang).astype(np.float32))
    nv = 64
    v = rng.normal(0, 1, (nv, hd)).astype(np.float32)
    pv = pos[:nv]
    c32 = np.cos(pv[:, None] * inv[None, :]).astype(np.float32); s32 = np.sin(pv[:, None] * inv[None, :]).astype(np.float32)
    c, s = c32.astype(np.float64), s32.astype(np.float64)
    x1, x2 = v[:, :64].astype(np.float64), v[:, 64:].astype(np.float64)
    yv = np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=1)
    out.put("rope.v.f32", v); out.put("rope.vy.f64", yv)
    out.line("rope", "rope", len(pos), hd, nv, int(theta))

    # ---- routing -----------------------------------------------------------------------------
    E, K = 384, 6

    def route_ref(z, bias):
        b = (z.astype(np.float32) + bias.astype(np.float32)).astype(np.float32)
        ids = np.stack([np.lexsort((np.arange(E), -b[r]))[:K] for r in range(len(z))]).astype(np.int32)
        zz = np.take_along_axis(z.astype(np.float64), ids, 1)
        w = np.where(zz >= 0, 1 / (1 + np.exp(-np.abs(zz))), np.exp(-np.abs(zz)) / (1 + np.exp(-np.abs(zz))))
        # near-tie bookkeeping: gap between the 6th and 7th biased score, in float32 ulps of the 6th
        srt = -np.sort(-b, axis=1)
        gap = (srt[:, K - 1].astype(np.float64) - srt[:, K].astype(np.float64))
        return ids, w, gap

    R = 4096
    z = (rng.normal(0, 1, (R, E)) * rng.choice([0.5, 1.0, 3.0], (R, 1))).astype(np.float32)
    bias = rng.uniform(-1, 1, E).astype(np.float32)
    bias = (bf16_to_f64(bf16_bits(bias))).astype(np.float32)
    ids, w, gap = route_ref(z, bias)
    out.put("route_a.z.f32", z); out.put("route_a.bias.f32", bias); out.put("route_a.ids.i32", ids)
    out.put("route_a.w.f64", w); out.put("route_a.gap.f64", gap)
    out.line("route", "route_a", R, E, K)
    Rt = 512                                                       # heavy exact ties: z and bias on a coarse grid
    zt = rng.integers(-3, 4, (Rt, E)).astype(np.float32) * 0.5
    bt = rng.integers(-2, 3, E).astype(np.float32) * 0.5
    ids, w, gap = route_ref(zt, bt)
    out.put("route_b.z.f32", zt); out.put("route_b.bias.f32", bt); out.put("route_b.ids.i32", ids)
    out.put("route_b.w.f64", w); out.put("route_b.gap.f64", gap)
    out.line("route", "route_b", Rt, E, K)

    # ---- the weight formula on 100,000 logits ------------------------------------------------
    n = 100000
    xs = np.concatenate([rng.normal(0, 3, 50000), rng.uniform(-30, 30, 30000), rng.uniform(-100, 100, 15000),
                         np.array([0.0, -0.0, 1e-30, -1e-30, 88.0, 89.0, 104.0, -88.0, -89.0, -104.0, 1e30, -1e30,
                                   3.4e38, -3.4e38, 20.0, -20.0, 0.5, -0.5, 17.3, -17.3]),
                         rng.uniform(-1, 1, 5000 - 20)]).astype(np.float32)
    assert len(xs) == n, len(xs)
    xd = xs.astype(np.float64)
    with np.errstate(over="ignore"):
        wsig = np.where(xd >= 0, 1 / (1 + np.exp(-np.abs(xd))), np.exp(-np.abs(xd)) / (1 + np.exp(-np.abs(xd))))
    out.put("sigmoid.x.f32", xs); out.put("sigmoid.w.f64", wsig)
    out.line("sigmoid", "sigmoid", n)

    out.close()
    print(f"kernel reference written to {args.out} ({len(out.lines)} cases), numpy {np.__version__}, seed {SEED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
