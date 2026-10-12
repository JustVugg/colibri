#!/usr/bin/env python3
"""Reference data for tests/test_kolibri1_dequant.c (FP8 dequantisation contract).

Writes, into --out, everything the C test compares the engine's dequantisation
against. The values come from torch / NumPy only, never from the engine, the reference implementation or any
shared colibri code:

  scales.f32        4096 random finite FP32 scales (seeded): normal range, negative,
                    subnormal range, huge (products overflow), tiny normal (products
                    land in the subnormal range), special values (+-0, +-1, min and
                    max normal, powers of two)
  products.u32      256 x 4096 bit patterns of torch `code.view(float8_e4m3fn)
                    .to(float32) * scale` (float32 multiply), code-major
  decode.u32        256 bit patterns of torch's e4m3fn -> float32 decode
  <O>x<I>.{codes.u8,grid.f32,w.f32}   a matrix of random e4m3 codes, its FP32 grid
                    [ceil(O/128), ceil(I/128)] and the dequantised matrix as torch
                    computes it (grid expanded to blocks, one float32 multiply)

It also cross-checks torch's decode against a NumPy integer-arithmetic decode of
the format (so the reference itself is not a single point of trust) and writes
meta.json with the library versions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SEED = 20261006
N_SCALES = 4096
SHAPES = [(256, 384), (192, 128), (2560, 6144)]


def numpy_e4m3(codes):
    """e4m3fn by integer arithmetic: sign, 4-bit exponent (bias 7), 3-bit mantissa,
    subnormals at exponent 0, no infinity, NaN at exponent 15 mantissa 7."""
    c = codes.astype(np.int64)
    s, e, m = c >> 7, (c >> 3) & 15, c & 7
    v = np.where(e == 0, m * 2.0 ** -9, (8 + m) * np.exp2((e - 10).astype(np.float64)))
    v = np.where(s == 1, -v, v)
    v = np.where((e == 15) & (m == 7), np.nan, v)
    return v.astype(np.float32)


def make_scales(rng):
    parts = []
    parts.append(np.exp2(rng.uniform(-30, 10, 1024)).astype(np.float32))                 # normal range
    typical = rng.uniform(0.004, 0.012, 512).astype(np.float32)                           # the fixture's range
    parts += [typical, -typical]
    sub = rng.integers(1, 1 << 23, 512, dtype=np.uint32).view(np.float32)                 # positive subnormals
    parts += [sub, -sub[:256]]
    parts.append(np.exp2(rng.uniform(60, 127, 512)).astype(np.float32))                   # products overflow
    parts.append(np.exp2(rng.uniform(-126, -100, 256)).astype(np.float32))                # products go subnormal
    special = np.array([0.0, -0.0, 1.0, -1.0, 2.0 ** -126, -2.0 ** -126, 2.0 ** -149, np.finfo(np.float32).max,
                        -np.finfo(np.float32).max, 3.0, 0.1, 1.0 / 3.0] + [2.0 ** k for k in range(-20, 21, 2)] +
                       [-(2.0 ** k) for k in range(-20, 21, 5)], dtype=np.float32)
    parts.append(special)
    base = np.concatenate(parts)
    extra = N_SCALES - len(base)
    assert extra >= 0, len(base)
    bits = (rng.integers(0, 2, extra, dtype=np.uint32) << 31) | (rng.integers(1, 255, extra, dtype=np.uint32) << 23) \
        | rng.integers(0, 1 << 23, extra, dtype=np.uint32)
    scales = np.concatenate([base, bits.view(np.float32)]).astype(np.float32)
    assert scales.shape == (N_SCALES,) and np.isfinite(scales).all()
    return scales


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    codes = np.arange(256, dtype=np.uint8)
    f8 = torch.from_numpy(codes.copy()).view(torch.float8_e4m3fn)
    dec = f8.to(torch.float32).numpy()
    ref = numpy_e4m3(codes)
    same = (dec.view(np.uint32) == ref.view(np.uint32)) | (np.isnan(dec) & np.isnan(ref))
    assert same.all(), np.nonzero(~same)
    assert sorted(np.nonzero(np.isnan(dec))[0].tolist()) == [0x7F, 0xFF]
    dec.view(np.uint32).tofile(out / "decode.u32")

    scales = make_scales(rng)
    scales.tofile(out / "scales.f32")
    prod = (dec[:, None] * scales[None, :]).astype(np.float32)           # float32 x float32, rounded once
    torch_prod = (f8.to(torch.float32)[:, None] * torch.from_numpy(scales)[None, :]).numpy()
    assert (prod.view(np.uint32) == torch_prod.view(np.uint32))[~np.isnan(prod)].all()
    prod.view(np.uint32).tofile(out / "products.u32")

    for O, I in SHAPES:
        r2 = np.random.default_rng(SEED + O + I)
        gr, gc = -(-O // 128), -(-I // 128)
        # codes without the NaN encodings: a matrix is meant to compare bitwise everywhere
        cm = r2.integers(0, 256, (O, I), dtype=np.uint8)
        cm[(cm & 0x7F) == 0x7F] = 0x3C
        grid = (np.exp2(r2.uniform(-12, -4, (gr, gc)))).astype(np.float32)
        w = torch.from_numpy(cm).view(torch.float8_e4m3fn).to(torch.float32)
        g = torch.from_numpy(grid).repeat_interleave(128, 0).repeat_interleave(128, 1)[:O, :I]
        wd = (w * g).numpy()
        tag = f"{O}x{I}"
        cm.tofile(out / f"{tag}.codes.u8")
        grid.tofile(out / f"{tag}.grid.f32")
        wd.tofile(out / f"{tag}.w.f32")
        print(f"{tag}: grid [{gr},{gc}]")

    (out / "meta.json").write_text(json.dumps({
        "seed": SEED, "n_scales": N_SCALES, "shapes": SHAPES,
        "torch": torch.__version__, "numpy": np.__version__,
        "numpy_decode_equals_torch": True,
        "n_product_bits_nan": int(np.isnan(prod).sum()),
        "n_product_inf": int(np.isinf(prod).sum()),
        "n_product_subnormal": int(((np.abs(prod) < np.finfo(np.float32).tiny) & (prod != 0)).sum()),
    }, indent=1) + "\n")
    print((out / "meta.json").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
