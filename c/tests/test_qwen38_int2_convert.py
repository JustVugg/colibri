"""tools/convert_qwen38_experts_int4.py --int2: the int2-g32 sidecar's quantizer and layout.

numpy only:
  * pack_int2 puts element i's code where int2_g32.h reads it (byte (i/128)*32 + i%32,
    bits 2*((i%128)/32)), checked with an independent decoder;
  * quantize_int2_g32 gives codes 0..3, u8 group scales and an f32 per row, and on
    Gaussian weights the error of the four-level Lloyd-Max grid; activation weights
    move the error away from the heavy channels;
  * the record: nine tensors back to back, the f32 row scales aligned, 1,397,760 bytes
    on Qwen3.8-Flash-Next's geometry;
  * CovStats: the factor it hands out is upper triangular and inverts the damped H;
  * the error-carrying quantizer (GPTQ) lowers the output error under a correlated H.
"""
import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:   # the CI's Python tests run without numpy
    raise unittest.SkipTest("numpy not installed: %s" % exc)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import convert_qwen38_experts_int4 as cv  # noqa: E402


def decode_codes(packed, cols):
    rows = packed.shape[0]
    out = np.zeros((rows, cols), np.uint8)
    for i in range(cols):
        byte = packed[:, (i // 128) * 32 + i % 32]
        out[:, i] = (byte >> (2 * ((i % 128) // 32))) & 3
    return out


def out_err(w, q, h):
    dw = w - q
    return math.sqrt(np.einsum("ij,jk,ik->", dw, h, dw) / np.einsum("ij,jk,ik->", w, h, w))


class Int2Convert(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_pack_layout(self):
        codes = self.rng.integers(0, 4, (5, 384), dtype=np.uint8)
        packed = cv.pack_int2(codes)
        self.assertEqual(packed.shape, (5, 96))
        np.testing.assert_array_equal(decode_codes(packed, 384), codes)

    def test_quantizer_ranges_and_error(self):
        w = (self.rng.standard_normal((64, 512)) * 0.02).astype(np.float32)
        codes, q, d = cv.quantize_int2_g32(w)
        self.assertEqual((codes.dtype, q.dtype, d.dtype), (np.uint8, np.uint8, np.float32))
        self.assertEqual((codes.shape, q.shape, d.shape), ((64, 512), (64, 16), (64,)))
        self.assertLessEqual(int(codes.max()), 3)
        self.assertEqual(int(q.max(axis=1).min()), 255)          # the row scale is its largest group's
        err = cv.rel_l2(w, cv.dequantize_int2(codes, q, d))
        # 4-level Lloyd-Max on a Gaussian: sqrt(0.1175) = 0.343; a scale per 32 does a bit better
        self.assertTrue(0.30 < err < 0.35, err)

    def test_weights_protect_heavy_channels(self):
        w = (self.rng.standard_normal((64, 512)) * 0.02).astype(np.float32)
        imp = np.ones(512, np.float32)
        imp[::32] = 100.0                                          # one heavy channel per group
        plain = cv.dequantize_int2(*cv.quantize_int2_g32(w))
        weighted = cv.dequantize_int2(*cv.quantize_int2_g32(w, imp))
        heavy = slice(0, 512, 32)
        self.assertLess(np.abs(w[:, heavy] - weighted[:, heavy]).mean(), np.abs(w[:, heavy] - plain[:, heavy]).mean())
        self.assertLess((imp * (w - weighted) ** 2).sum(), (imp * (w - plain) ** 2).sum())

    def test_record_geometry(self):
        g = cv.Geometry(48, 512, 2560, 640, bits=2)
        self.assertEqual(g.record_bytes, 1397760)
        off = 0
        names = []
        for name, dtype, shape, start, n in g.tensors("model.language_model", 3, 7):
            self.assertEqual(start, off)
            self.assertEqual(n, int(np.prod(shape)) * (4 if dtype == "F32" else 1))
            if dtype == "F32":
                self.assertEqual(start % 4, 0)
            off += n
            names.append(name.rsplit(".", 2)[-2:])
        self.assertEqual(off, g.record_bytes)
        self.assertEqual([n[-1] for n in names[:3]], ["weight"] * 3)
        self.assertEqual([n[-1] for n in names[3:6]], ["q8"] * 3)
        self.assertEqual([n[-1] for n in names[6:]], ["d"] * 3)
        with self.assertRaises(ValueError):
            cv.Geometry(2, 4, 192, 64, bits=2)

    def _cov_file(self, tmp, mats):
        path = os.path.join(tmp, "cov.bin")
        layers, hidden = len(mats), mats[0].shape[0]
        with open(path, "wb") as f:
            f.write(b"Q38COV1\0" + struct.pack("<2i", layers, hidden) + np.full(layers, 1000, np.int64).tobytes())
            for m in mats:
                f.write(m.astype(np.float64).tobytes())
        return path

    def test_cov_factor(self):
        a = self.rng.standard_normal((256, 64))
        m = a @ a.T + np.eye(256)
        with tempfile.TemporaryDirectory() as tmp:
            cov = cv.CovStats(self._cov_file(tmp, [m, 2 * m]))
            imp = self.rng.uniform(0.5, 2.0, 256)
            u = cov.upper_inv(0, imp)
            self.assertTrue(np.allclose(np.tril(u, -1), 0))
            dg = np.sqrt(np.diag(m))
            c = m / dg[:, None] / dg[None, :] + cov.damp * np.eye(256)
            h = np.sqrt(imp)[:, None] * c * np.sqrt(imp)[None, :]
            np.testing.assert_allclose(u.T @ u @ h, np.eye(256), atol=1e-6)
            np.testing.assert_allclose(cov.upper_inv(1, imp), u)   # the scale of a layer's matrix does not matter

    def test_gptq_lowers_output_error(self):
        a = self.rng.standard_normal((256, 16))
        m = a @ a.T + 0.5 * np.eye(256)
        w = (self.rng.standard_normal((32, 256)) * 0.02).astype(np.float32)
        imp = np.diag(m).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            cov = cv.CovStats(self._cov_file(tmp, [m]))
            u = cov.upper_inv(0, imp)
            dg = np.sqrt(np.diag(m))
            h = (m / dg[:, None] / dg[None, :] + cov.damp * np.eye(256)) * np.sqrt(imp)[:, None] * np.sqrt(imp)[None, :]
            plain = cv.dequantize_int2(*cv.quantize_int2_g32(w, imp))
            gptq = cv.dequantize_int2(*cv.quantize_int2_g32_gptq(w, imp, u))
            self.assertLess(out_err(w.astype(np.float64), gptq, h), 0.8 * out_err(w.astype(np.float64), plain, h))

    def test_dump_rows_and_exact_quantizer(self):
        h, k, n, experts = 256, 4, 600, 6
        ids = np.stack([self.rng.permutation(experts)[:k] for _ in range(n)]).astype(np.int32)
        gates = self.rng.dirichlet(np.ones(k), n).astype(np.float32)
        basis = self.rng.standard_normal((16, h))
        x = (self.rng.standard_normal((n, 16)) @ basis + 0.1 * self.rng.standard_normal((n, h))).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "layer-000.bin"), "wb") as f:
                f.write(b"Q38DMP1\0" + struct.pack("<2i", h, k))
                f.write(np.concatenate([ids.view(np.uint8).reshape(n, -1), gates.view(np.uint8).reshape(n, -1),
                                        x.view(np.uint8).reshape(n, -1)], axis=1).tobytes())
            dump = cv.Dump(tmp)
            xe, ge = dump.rows(0, 2)
            want = (ids == 2).any(axis=1)
            np.testing.assert_array_equal(xe, x[want].astype(np.float64))
            np.testing.assert_allclose(ge, (gates * (ids == 2)).sum(axis=1)[want])
            np.testing.assert_allclose(dump.layer_moment(0), x.astype(np.float64).T @ x / n, rtol=1e-12)
            w3 = [(self.rng.standard_normal(shape) * 0.05).astype(np.float32) for shape in ((128, h), (128, h), (h, 128))]

            def out(ws, rows):
                g = rows @ ws[0].T
                return (g / (1 + np.exp(-g)) * (rows @ ws[1].T)) @ ws[2].T
            y = out([w.astype(np.float64) for w in w3], xe)
            exact = [cv.dequantize_int2(*o) for o in cv.quantize_expert_int2_exact(w3, xe, ge, dump.layer_moment(0))]
            plain = [cv.dequantize_int2(*cv.quantize_int2_g32(w)) for w in w3]
            e_exact = np.linalg.norm(y - out(exact, xe)) / np.linalg.norm(y)
            e_plain = np.linalg.norm(y - out(plain, xe)) / np.linalg.norm(y)
            self.assertLess(e_exact, 0.7 * e_plain)
            # an expert the calibration never routed: the layer's moment alone, down plain
            none = cv.quantize_expert_int2_exact(w3, np.zeros((0, h)), np.zeros(0), dump.layer_moment(0))
            self.assertEqual([o[0].shape for o in none], [(128, h), (128, h), (h, 128)])

    def test_lowbit_g16_layouts(self):
        # lowbit_g16.h: low two bits as int2_g32.h, then for 3 bits 16 bytes per block of 128
        # where byte k holds elements k, k+16, ..., k+112 in bits 0..7; scales two to a byte
        for name, bits in (("int2g16", 2), ("int3g16", 3)):
            fmt = cv.LOWBIT[name]
            codes = self.rng.integers(0, 2 ** bits, (3, 256), dtype=np.uint8)
            packed = cv.pack_lowbit(codes, fmt)
            self.assertEqual(packed.shape, (3, 256 * bits // 8))
            back = np.zeros_like(codes)
            for i in range(256):
                blk = packed[:, (i // 128) * (16 * bits):]
                k = i % 128
                c = (blk[:, k % 32] >> (2 * (k // 32))) & 3
                if bits == 3:
                    c |= ((blk[:, 32 + k % 16] >> (k // 16)) & 1) << 2
                back[:, i] = c
            np.testing.assert_array_equal(back, codes)
            q = self.rng.integers(0, 16, (3, 16), dtype=np.uint8)
            ps = cv.pack_scales(q, fmt)
            self.assertEqual(ps.shape, (3, 8))
            np.testing.assert_array_equal(np.stack([ps & 15, ps >> 4], axis=2).reshape(3, 16), q)
            w = (self.rng.standard_normal((32, 256)) * 0.02).astype(np.float32)
            c, qq, d = cv.quantize_lowbit(w, None, fmt)
            self.assertLessEqual(int(qq.max()), 15)
            self.assertLessEqual(int(c.max()), 2 ** bits - 1)
            err = cv.rel_l2(w, cv.dequantize_lowbit(c, qq, d, fmt))
            self.assertTrue((0.28 < err < 0.34) if bits == 2 else (0.14 < err < 0.18), (name, err))
        # the release's records: int2g16 the size of int2g32, int3g16 2.01 MB
        self.assertEqual(cv.Geometry(48, 512, 2560, 640, fmt=cv.INT2G16).record_bytes, 1397760)
        g3 = cv.Geometry(48, 512, 2560, 640, fmt=cv.INT3G16)
        self.assertEqual(g3.record_bytes, 2012160)
        self.assertEqual([t[0].rsplit(".", 1)[-1] for t in g3.tensors("m", 0, 0)][3:6], ["q4"] * 3)

    def test_sequential_targets(self):
        # --dump-target: the inputs x~ of a model with its earlier layers quantized, the
        # targets the original expert on the original inputs x of the same tokens
        fmt = cv.INT2G16
        w3 = [(self.rng.standard_normal(s) * 0.05).astype(np.float32) for s in ((128, 256), (128, 256), (256, 128))]
        x = self.rng.standard_normal((300, 256)); g = self.rng.uniform(0.05, 0.5, 300)
        prior = x.T @ x / len(x)

        def out(w, xi):
            a = xi @ w[0].T.astype(np.float64)
            return (a / (1 + np.exp(-a)) * (xi @ w[1].T.astype(np.float64))) @ w[2].T.astype(np.float64)

        def deq(o):
            return [cv.dequantize_lowbit(*t, fmt) for t in o]
        y = out(w3, x)

        def err(w, xi):
            return float(np.sqrt((((out(w, xi) - y) ** 2) * (g ** 2)[:, None]).sum() / ((y ** 2) * (g ** 2)[:, None]).sum()))
        # the original inputs as their own targets: gate and up the plain fit's but for a rounding
        # flip here and there (W' = W to rounding), down aimed at the expert's true output instead of
        # its own output on the quantized hidden rows, so no worse on them
        po = cv.quantize_expert_lowbit_exact(w3, x, g, prior, alpha=2.0, fmt=fmt)
        so = cv.quantize_expert_lowbit_exact(w3, x, g, prior, alpha=2.0, fmt=fmt, x_orig=x)
        for k in range(2):
            self.assertGreater(float((po[k][0] == so[k][0]).mean()), 0.999)
        self.assertLessEqual(err(deq(so), x), err(deq(po), x) + 1e-4)
        # upstream error that shrinks the inputs: the targeted fit maps x~ back to the original output
        xt = 0.85 * x + 0.02 * self.rng.standard_normal(x.shape)
        blind = err(deq(cv.quantize_expert_lowbit_exact(w3, xt, g, prior, alpha=2.0, fmt=fmt)), xt)
        aimed = err(deq(cv.quantize_expert_lowbit_exact(w3, xt, g, prior, alpha=2.0, fmt=fmt, x_orig=x)), xt)
        self.assertLess(aimed, 0.9 * blind, (aimed, blind))

    def test_unpack_and_output_rescale(self):
        # --rescale-only reads the records back: unpack_lowbit / unpack_scales invert the packers
        for fmt in (cv.INT2G32, cv.INT2G16, cv.INT3G16):
            codes = self.rng.integers(0, 2 ** fmt.bits, (6, 384)).astype(np.uint8)
            back = cv.unpack_lowbit(cv.pack_lowbit(codes, fmt), 384, fmt)
            self.assertTrue(np.array_equal(back, codes), fmt.name)
            q = self.rng.integers(0, fmt.smax + 1, (6, 384 // fmt.gs)).astype(np.uint8)
            self.assertTrue(np.array_equal(cv.unpack_scales(cv.pack_scales(q, fmt), fmt), q), fmt.name)
        # an expert whose down came out 10% small: the factor finds it back (1/0.9), and it
        # is clipped, and 1 without rows
        w3 = [(self.rng.standard_normal(s) * 0.05).astype(np.float32) for s in ((64, 128), (64, 128), (128, 64))]
        x = self.rng.standard_normal((40, 128)); g = self.rng.uniform(0.05, 0.5, 40)
        self.assertAlmostEqual(cv.output_rescale(w3, w3[:2] + [w3[2] * 0.9], x, g), 1 / 0.9, places=6)
        self.assertEqual(cv.output_rescale(w3, w3[:2] + [w3[2] * 0.5], x, g), 1.25)
        self.assertEqual(cv.output_rescale(w3, w3, x[:0], g[:0]), 1.0)

    def test_vq2(self):
        # vq2_g32.h: codebooks whose sums stay int8, codes a0 b0 a1 b1 ... per row, scales a
        # nibble per 32, d * q * (a + b) as the kernel reads it; GPTQ over blocks of 8 lowers the
        # output error under a correlated H; the release's record 1,320,960 bytes
        fmt = cv.VQ2G32
        a8, b8, k = cv.train_vq_books(self.rng.standard_normal((20000, 8)), iters=8)
        self.assertLessEqual(int((np.abs(a8).max(axis=0) + np.abs(b8).max(axis=0)).max()), 127)
        self.assertGreater(int((np.abs(a8).max(axis=0) + np.abs(b8).max(axis=0)).max()), 120)
        fmt.set_books(a8, b8, k)
        w = (self.rng.standard_normal((48, 256)) * 0.02).astype(np.float32)
        c, q, d = cv.quantize_lowbit(w, None, fmt)
        self.assertEqual((c.shape, q.shape, d.shape), ((48, 32, 2), (48, 8), (48,)))
        self.assertEqual((c.dtype, q.dtype, d.dtype), (np.uint16, np.uint8, np.float32))
        self.assertTrue(1 <= int(q.min()) and int(q.max()) == 15)
        deq = cv.dequantize_lowbit(c, q, d, fmt)
        err = cv.rel_l2(w, deq)
        self.assertTrue(0.25 < err < 0.33, err)   # 2.125 bits a weight on a Gaussian
        packed, ps = cv.pack_lowbit(c, fmt), cv.pack_scales(q, fmt)
        self.assertEqual((packed.shape, ps.shape), ((48, 64), (48, 4)))
        self.assertTrue(np.array_equal(cv.unpack_lowbit(packed, 256, fmt), c))
        self.assertTrue(np.array_equal(cv.unpack_scales(ps, fmt), q))
        # the kernel's reading, element by element (vq2_value, vq2_scale)
        r, i = 5, np.arange(256)
        v = a8[packed[r, (i // 8) * 2], i % 8] + b8[packed[r, (i // 8) * 2 + 1], i % 8]
        sc = (ps[r, i // 64] >> (4 * ((i // 32) % 2))) & 15
        np.testing.assert_allclose(deq[r], np.float64(d[r]) * sc * v, rtol=1e-12)
        # GPTQ: correlated inputs, the output error lower than the plain search's
        a = self.rng.standard_normal((256, 16))
        h = a @ a.T + 0.5 * np.eye(256)
        g = cv.quantize_lowbit_gptq(w, np.diag(h).astype(np.float32), cv._upper_inv(h), fmt)
        plain = cv.dequantize_lowbit(*cv.quantize_lowbit(w, np.diag(h).astype(np.float32), fmt), fmt)
        self.assertLess(out_err(w.astype(np.float64), cv.dequantize_lowbit(*g, fmt), h),
                        0.8 * out_err(w.astype(np.float64), plain, h))
        geo = cv.Geometry(48, 512, 2560, 640, fmt=fmt)
        self.assertEqual(geo.record_bytes, 1320960)
        self.assertEqual([t[0].rsplit(".", 1)[-1] for t in geo.tensors("m", 0, 0)][3:6], ["q4"] * 3)
        self.assertEqual(geo.tensors("m", 0, 0)[3][2], [640, 40])

    def test_vq1(self):
        # vq1_g64.h: 16-vectors of two 1024-entry int8 books, 10-bit indices (a0 b0 .. a3 b3 of four
        # vectors in 10 bytes, LSB first), a nibble scale per 64; the release's record 821,760 bytes
        fmt = cv.VQ1G64
        a8, b8, k = cv.train_vq_books(self.rng.standard_normal((30000, 16)), iters=6, book=1024)
        self.assertEqual(a8.shape, (1024, 16))
        self.assertLessEqual(int((np.abs(a8).max(axis=0) + np.abs(b8).max(axis=0)).max()), 127)
        fmt.set_books(a8, b8, k)
        w = (self.rng.standard_normal((32, 256)) * 0.02).astype(np.float32)
        c, q, d = cv.quantize_lowbit(w, None, fmt)
        self.assertEqual((c.shape, q.shape, d.shape), ((32, 16, 2), (32, 4), (32,)))
        self.assertLess(int(c.max()), 1024)
        deq = cv.dequantize_lowbit(c, q, d, fmt)
        err = cv.rel_l2(w, deq)
        self.assertTrue(0.35 < err < 0.6, err)   # 1.31 bits a weight on a Gaussian
        packed, ps = cv.pack_lowbit(c, fmt), cv.pack_scales(q, fmt)
        self.assertEqual((packed.shape, ps.shape), ((32, 40), (32, 2)))
        self.assertTrue(np.array_equal(cv.unpack_lowbit(packed, 256, fmt), c))
        self.assertTrue(np.array_equal(cv.unpack_scales(ps, fmt), q))
        # the kernel's reading, element by element: group of 64 = 10 bytes, index k at bits 10k
        r = 7
        bits = [int.from_bytes(bytes(packed[r, 10 * g:10 * g + 10]), "little") for g in range(4)]
        idx = lambda v, k: (bits[v // 4] >> (10 * (2 * (v % 4) + k))) & 1023
        vals = [a8[idx(i // 16, 0), i % 16] + b8[idx(i // 16, 1), i % 16] for i in range(256)]
        sc = [(ps[r, i // 128] >> (4 * ((i // 64) % 2))) & 15 for i in range(256)]
        np.testing.assert_allclose(deq[r], np.float64(d[r]) * np.array(sc) * np.array(vals), rtol=1e-12)
        a = self.rng.standard_normal((256, 16))
        h = a @ a.T + 0.5 * np.eye(256)
        g = cv.quantize_lowbit_gptq(w, np.diag(h).astype(np.float32), cv._upper_inv(h), fmt)
        plain = cv.dequantize_lowbit(*cv.quantize_lowbit(w, np.diag(h).astype(np.float32), fmt), fmt)
        self.assertLess(out_err(w.astype(np.float64), cv.dequantize_lowbit(*g, fmt), h),
                        0.8 * out_err(w.astype(np.float64), plain, h))
        self.assertEqual(cv.Geometry(48, 512, 2560, 640, fmt=fmt).record_bytes, 821760)


if __name__ == "__main__":
    unittest.main()
