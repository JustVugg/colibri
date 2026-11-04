#!/usr/bin/env python3
"""fmt=6 (E8/IQ3 lattice container) index codec — #452 ladder step 2.

Note: fmt=5 is taken by the int3 dual-plane container (#132); this lattice
container is fmt=6.

The ablation (#453) proved the SCHEME: an IQ3_XXS-style codebook plus rotation
matches our simulated E8 ball (51.5% vs 51.5% on OLMoE). That code quantizes to
lattice points and keeps floats. This module produces the DEPLOYABLE bytes and
reads them back, so the container, the converter and the decode kernels all
agree on one layout.

Layout - one 256-weight super-block, 98 bytes, 3.0625 bpw:

    [0  .. 63]  uint8  grid index per 4-dim magnitude block   (64 blocks)
    [64 .. 95]  uint32 x8, one per 32-weight sub-block:
                  bits  0..20  three 7-bit sign words (8 weights each,
                               bit i set => weight i negative; the 8th sign
                               is implied by odd parity)
                  bits 21..27  the fourth 7-bit sign word
                  bits 28..31  4-bit sub-scale code
    [96 .. 97]  fp16   super-scale d

    value(w) = d * (0.5 + code) * 0.5 * grid[idx][j] * 0.5 * sign

The last 0.5 is the half-unit convention of the published grid (magnitudes are
stored doubled: 4,12,...,62 mean 2.0,6.0,...,31.0).

Odd-parity signs: llama.cpp stores 7 of every 8 signs and derives the 8th so the
product of the eight is +1. The encoder therefore flips the smallest-magnitude
weight of any block whose true signs violate that — the same cost the ablation
priced in, now applied for real.
"""
import json
import os
import numpy as np

QK = 256                      # weights per super-block
SUB = 32                      # weights per sub-block (one uint32 of signs+scale)
BLOCK_BYTES = QK // 4 + (QK // SUB) * 4 + 2      # 64 + 32 + 2 = 98
ROW_CHUNK = int(os.environ.get("IQ3_ROW_CHUNK", "128"))   # rows per encode pass (measured optimum)

_GRID = None


def grid():
    """[256,4] float32 magnitudes in weight units (published table is doubled)."""
    global _GRID
    if _GRID is None:
        path = os.path.join(os.path.dirname(__file__), "iq3xxs_grid.json")
        _GRID = np.asarray(json.load(open(path)), dtype=np.float32) * 0.5
    return _GRID


def _nearest(mag4):
    """[N,4] magnitudes -> [N] grid indices, argmin ||m-g||^2 without cdist."""
    g = grid()
    g2 = (g * g).sum(1)
    out = np.empty(len(mag4), dtype=np.uint8)
    for i in range(0, len(mag4), 1 << 16):          # bounded working set
        c = mag4[i:i + (1 << 16)]
        out[i:i + len(c)] = np.argmin(g2 - 2.0 * (c @ g.T), axis=1).astype(np.uint8)
    return out


_LIB = False        # False = not tried yet, None = unavailable

def _native():
    """iq3_encode.c built as a shared library, if it is next to this file.

    Optional by design: without it everything still works through numpy, just
    ~25x slower. IQ3_NATIVE=0 forces the numpy path (used to A/B the two).
    """
    global _LIB
    if _LIB is not False:
        return _LIB
    _LIB = None
    want = os.environ.get("IQ3_NATIVE", "1") != "0"
    if want:
        import ctypes, ctypes.util
        for name in ("libiq3.so", "libiq3.dylib", "iq3.dll"):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
            if not os.path.exists(path):
                continue
            try:
                lib = ctypes.CDLL(path)
                if lib.iq3_encode_abi() != 1:
                    break                       # stale build: fall back rather than corrupt
                lib.iq3_encode.restype = None
                lib.iq3_encode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                           ctypes.c_void_p, ctypes.c_void_p]
                _LIB = lib
            except (OSError, AttributeError):
                pass
            break
    if _LIB is None and want:
        # Once per process, and only when an E8 encode is actually about to run:
        # the numpy path is correct but ~15x slower, which on a large model is the
        # difference between hours and days. Silence here would just cost the user
        # that time without telling them why.
        import sys as _sys
        print("[iq3] native encoder not built - using the numpy path (~15x slower). "
              "Build it with:  make iq3", file=_sys.stderr, flush=True)
    return _LIB


def encode(x):
    """float32 [..., K] (K % 256 == 0) -> packed uint8 [..., K//256 * 98].

    Uses the native encoder when available (see _native); otherwise falls back to
    the numpy path, which processes rows in cache-sized blocks — the search keeps
    a [rows*8, 256] score array live per sub-block, and letting that grow to a
    whole expert tensor turns the argmin memory-bound.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    K = x.shape[-1]
    if K % QK:
        raise ValueError(f"fmt=5 needs K % {QK} == 0, got {K}")
    rows = x.reshape(-1, K)
    nsb = K // QK
    out = np.empty((len(rows), nsb * BLOCK_BYTES), dtype=np.uint8)
    lib = _native()
    if lib is not None:
        g = np.ascontiguousarray(grid(), dtype=np.float32)
        lib.iq3_encode(rows.ctypes.data, rows.shape[0], K, g.ctypes.data, out.ctypes.data)
        return out.reshape(*x.shape[:-1], nsb * BLOCK_BYTES)
    rc = max(1, ROW_CHUNK)
    for r0 in range(0, len(rows), rc):
        _encode_rows(rows[r0:r0 + rc], out[r0:r0 + rc], nsb)
    return out.reshape(*x.shape[:-1], nsb * BLOCK_BYTES)


def _encode_rows(rows, out, nsb):
    for sb in range(nsb):
        blk = rows[:, sb * QK:(sb + 1) * QK]                     # [R,256]
        sign = np.where(blk < 0, -1.0, 1.0).astype(np.float32)
        mag = np.abs(blk)

        # parity fix: flip the smallest magnitude of every 8 whose product is -1
        s8 = sign.reshape(len(rows), QK // 8, 8)
        m8 = mag.reshape(len(rows), QK // 8, 8)
        viol = s8.prod(-1) < 0                                   # [R,32]
        amin = m8.argmin(-1)
        r, b = np.nonzero(viol)
        s8[r, b, amin[r, b]] *= -1.0
        sign = s8.reshape(len(rows), QK)

        base = sb * BLOCK_BYTES
        # super-scale: RMS anchor, same statistic the ablation searches around
        d = np.sqrt((mag * mag).mean(-1, keepdims=True)) / 20.0 + 1e-12
        out[:, base + 96:base + 98] = d.astype(np.float16).view(np.uint8)
        d = d.astype(np.float16).astype(np.float32)              # encode what we store

        g = grid()
        G2 = (g * g).sum(1)                                      # [256] ||g||^2
        R = len(rows)
        for ib in range(QK // SUB):
            m = mag[:, ib * SUB:(ib + 1) * SUB]                  # [R,32]
            # The sub-scale search is 16 candidates for the SAME magnitudes, and the
            # only code-dependent quantity is the scalar db. Writing the nearest-grid
            # test as
            #     argmin_g ||m/db - g||^2 = argmin_g [ db*||g||^2 - 2*(m.g) ]     (db>0)
            # takes m.g outside the loop: one [R*8,4]x[4,256] product serves all 16
            # codes instead of one each. The squared error follows from the same
            # product — sum (db*g - m)^2 = db^2*||g||^2 - 2*db*(m.g) + ||m||^2 — and
            # ||m||^2 is the same for every code, so it drops out of the comparison.
            # Measured 4.6x on the GLM-5.2 expert shapes; the encoding is unchanged
            # except where two codes tie to within float rounding.
            m4 = m.reshape(-1, 4)                                # [R*8,4]
            A = m4 @ g.T                                         # [R*8,256] = m.g
            A2 = -2.0 * A
            nrow = np.arange(len(m4))
            best_err = None; best_idx = None; best_code = None
            for code in range(16):
                db = np.maximum(d * (0.5 + code) * 0.5, 1e-20)   # [R,1]
                dbN = np.repeat(db, SUB // 4, axis=0)[:, 0]      # [R*8]
                idx = np.argmin(dbN[:, None] * G2[None, :] + A2, axis=1)
                gi = G2[idx]; ai = A[nrow, idx]
                err = ((dbN * dbN * gi - 2.0 * dbN * ai)
                       .reshape(R, SUB // 4).sum(1, keepdims=True))
                iu = idx.astype(np.uint8).reshape(R, SUB // 4)
                if best_err is None:
                    best_err, best_idx, best_code = err, iu, np.zeros(R, np.int64)
                else:
                    take = (err < best_err)[:, 0]
                    if take.any():
                        best_idx = np.where(take[:, None], iu, best_idx)
                        best_code = np.where(take, code, best_code)
                        best_err = np.where(take[:, None], err, best_err)
            bidx, bcode = best_idx, best_code
            out[:, base + ib * 8:base + (ib + 1) * 8] = bidx.astype(np.uint8)

            # signs: four 7-bit words for this sub-block + the 4-bit code
            s = sign[:, ib * SUB:(ib + 1) * SUB].reshape(len(rows), 4, 8)
            neg = (s < 0).astype(np.uint32)
            word = np.zeros(len(rows), dtype=np.uint32)
            for l in range(4):
                seven = np.zeros(len(rows), dtype=np.uint32)
                for j in range(7):
                    seven |= neg[:, l, j] << j
                word |= seven << (7 * l)
            word |= (bcode.astype(np.uint32) & 0xF) << 28
            off = base + QK // 4 + ib * 4
            out[:, off:off + 4] = word.view(np.uint8).reshape(len(rows), 4) if False else \
                np.ascontiguousarray(word).view(np.uint8).reshape(len(rows), 4)


def decode(packed, K):
    """packed uint8 [..., K//256*98] -> float32 [..., K]. The kernels' reference."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    nsb = K // QK
    rows = packed.reshape(-1, nsb * BLOCK_BYTES)
    out = np.empty((len(rows), K), dtype=np.float32)
    g = grid()
    for sb in range(nsb):
        base = sb * BLOCK_BYTES
        d = rows[:, base + 96:base + 98].copy().view(np.float16).astype(np.float32)
        for ib in range(QK // SUB):
            idx = rows[:, base + ib * 8:base + (ib + 1) * 8]                  # [R,8]
            off = base + QK // 4 + ib * 4
            word = np.ascontiguousarray(rows[:, off:off + 4]).view(np.uint32).reshape(-1)
            code = (word >> 28) & 0xF
            db = d[:, 0] * (0.5 + code) * 0.5                                 # [R]
            mag = g[idx].reshape(len(rows), SUB)                              # [R,32]
            sgn = np.ones((len(rows), 4, 8), dtype=np.float32)
            for l in range(4):
                seven = (word >> (7 * l)) & 0x7F
                par = 0
                for j in range(7):
                    bit = (seven >> j) & 1
                    sgn[:, l, j] = np.where(bit == 1, -1.0, 1.0)
                    par ^= bit
                sgn[:, l, 7] = np.where(par == 1, -1.0, 1.0)   # odd parity closes the block
            out[:, sb * QK + ib * SUB:sb * QK + (ib + 1) * SUB] = \
                mag * sgn.reshape(len(rows), SUB) * db[:, None]
    return out.reshape(*packed.shape[:-1], K)


def bpw():
    return BLOCK_BYTES * 8 / QK


# ---- rotation (converter side of quant.h e8_signs / e8_rot_rows) -------------
#
# Q = D @ H / sqrt(n) per power-of-two block; W@Q on weight rows here equals
# Q^T x on activations in the engine (sign-flip then FWHT — one routine both
# sides). The sign diagonal D is REGENERATED from the block size, never stored:
# both sides draw the same xorshift64* stream seeded 417+n, and the kernel
# fixture (make_e8_fixture.py) pins the agreement.

def signs(n):
    """[n] float32 in {+1,-1} — bit-exact mirror of quant.h e8_signs()."""
    s = np.uint64(417 + n)
    out = np.empty((n + 7) // 8, dtype=np.uint8)
    with np.errstate(over="ignore"):
        for i in range(len(out)):
            s ^= s >> np.uint64(12)
            s ^= (s << np.uint64(25)) & np.uint64(0xFFFFFFFFFFFFFFFF)
            s ^= s >> np.uint64(27)
            out[i] = np.uint8(((s * np.uint64(2685821657736338717)) &
                               np.uint64(0xFFFFFFFFFFFFFFFF)) >> np.uint64(56))
    bits = np.unpackbits(out, bitorder="little")[:n]
    return np.where(bits == 1, -1.0, 1.0).astype(np.float32)


def rot_blocks(dim):
    """Block tiling: each block is the largest power of two dividing the
    remainder (its lowest set bit) — 6144 -> [2048, 4096], 1536 -> [512, 1024].
    Blocks over 32768 halve, mirroring the engine's sign-buffer cap."""
    out, rem = [], dim
    while rem:
        b = rem & (-rem)
        while b > 32768:
            b >>= 1
        out.append(b)
        rem -= b
    return out


def _fwht_rows(blk, b):
    """In-place-style FWHT over rows of [..., b] (b a power of two)."""
    h = 1
    while h < b:
        blk = blk.reshape(-1, b // (2 * h), 2, h)
        u, v = blk[:, :, 0, :].copy(), blk[:, :, 1, :].copy()
        blk[:, :, 0, :] = u + v
        blk[:, :, 1, :] = u - v
        blk = blk.reshape(-1, b)
        h <<= 1
    return blk / np.sqrt(b, dtype=np.float32)


def rotate_rows(x):
    """Apply the rotation to rows of float32 [..., dim] (weights W@Q or
    activations Q^T x — the transform is the same). Returns a new array."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    dim = x.shape[-1]
    rows = x.reshape(-1, dim).copy()
    off = 0
    for b in rot_blocks(dim):
        rows[:, off:off + b] = _fwht_rows(rows[:, off:off + b] * signs(b)[None, :], b)
        off += b
    return rows.reshape(x.shape)


def unrotate_rows(x):
    """Inverse of rotate_rows (Q v back to v): FWHT first, then the sign flip.
    Eval-side only — the engine always works in the rotated space."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    dim = x.shape[-1]
    rows = x.reshape(-1, dim).copy()
    off = 0
    for b in rot_blocks(dim):
        rows[:, off:off + b] = _fwht_rows(rows[:, off:off + b].copy(), b) * signs(b)[None, :]
        off += b
    return rows.reshape(x.shape)
