#!/usr/bin/env python3
"""The FP64 residual rule (residual policy kolibri1-residual-fp64-c1): an INDEPENDENT reference of the mixed-precision residual rule, against the engine's probes.

  KOLIBRI1_BIN=./kolibri1 python3 tests/test_kolibri1_residual_c.py        (needs numpy; K1_REQUIRE_FORWARD=1 forbids skipping)

The rule (from the engine's design):  h_attn64 = h64 + FP64(a32);  x2_32 = FP32(RMSNorm_FP64(h_attn64));  h_next64 = h_attn64 + FP64(f32);  head_in32 = FP32(RMSNorm_FP64(h_final64)).
The reference below shares no code with the engine: Python `fractions` / `decimal` (exact rational sums, a 60-digit square root), explicit round-to-nearest-even
to FP64 / FP32 by integer arithmetic.  It is bit-compared with the engine on
  * the K1_PROBE=ops operators on declared inputs that carry information below one FP32 ulp (x_prev = fp32 value + 2^-29 |value|), a cancelling addition
    (2^20 + 2^-10 against -2^20: exact in FP64, lost by any FP32 stream), a massive coordinate row and a plain random row, for steps 1, 8 (+ its norm), 15;
  * the production K1_PROBE=layer path: the engine's own a32 / f32 branch outputs satisfy the rule exactly, and the outputs are byte-identical for K1_CHUNK 1 / 3 / 64
    (prefill/decode row-count invariance of the residual path);
  * the tails: final_in (FP64) -> final_norm.
Control: the same fixtures run through the fp32-stream emulation of candidate A (narrow the stream, add in FP32) DIFFER from the engine, i.e. the fixtures can tell the rules apart.
An .f32 residual file given to the engine is refused (it is never read as FP64)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "make_kolibri1_tiny.py"
BIN = Path(os.environ.get("KOLIBRI1_BIN", ROOT / "kolibri1")).resolve()
T = 6
getcontext().prec = 80


def rne(fr, mant, emin, emax):
    """round the Fraction to a binary float format with `mant` explicit+implicit mantissa bits (53 / 24), round-to-nearest-even, subnormals down to 2^emin; -> Fraction"""
    if fr == 0:
        return Fraction(0)
    s = -1 if fr < 0 else 1
    a = abs(fr)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    e = max(e, emin)
    q = a / Fraction(2) ** (e - mant + 1)                 # a in units of the last place
    n = q.numerator // q.denominator
    rem = q - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2 == 1):
        n += 1
    r = Fraction(n) * Fraction(2) ** (e - mant + 1)
    return s * r


def r64(fr):
    return rne(fr, 53, -1022, 1023)


def r32(fr):
    return rne(fr, 24, -126, 127)


def rms_ref(x64, w32, eps32, n):
    """exact RMSNorm of a vector of doubles: y_i = w_i * x_i / sqrt(sum x^2 / n + eps), rounded ONCE to FP32; eps is the engine's fp32 config value taken as a double"""
    xs = [Fraction(float(v)) for v in x64]
    ss = sum(v * v for v in xs)
    m = ss / n + Fraction(float(np.float32(eps32)))
    d = Decimal(m.numerator) / Decimal(m.denominator)
    root = d.sqrt()
    out = []
    for v, w in zip(xs, w32):
        wv = Decimal(w.numerator) / Decimal(w.denominator)
        xv = Decimal(v.numerator) / Decimal(v.denominator)
        out.append(wv * xv / root)
    res = np.empty(len(out), dtype=np.float32)
    for i, yd in enumerate(out):
        fr = Fraction(yd)                                  # the 80-digit decimal as an exact rational: far below any fp32 rounding boundary scale
        res[i] = np.float32(float(r32(fr)))                # r32 is exactly representable in float64 (24 bits)
    return res


def st_tensors(path):
    """a minimal safetensors reader: name -> float64 numpy array (F32 / BF16 / F16 / F64)"""
    raw = path.read_bytes()
    hl = int.from_bytes(raw[:8], "little")
    hdr = json.loads(raw[8:8 + hl])
    base = 8 + hl
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        b, e = v["data_offsets"]
        buf = raw[base + b:base + e]
        dt = v["dtype"]
        if dt == "F32":
            a = np.frombuffer(buf, dtype="<f4").astype(np.float64)
        elif dt == "F64":
            a = np.frombuffer(buf, dtype="<f8").copy()
        elif dt == "BF16":
            a = (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        elif dt == "F16":
            a = np.frombuffer(buf, dtype="<f2").astype(np.float64)
        else:
            continue
        out[k] = a.reshape(v["shape"])
    return out


class ResidualRuleC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [w for w, ok in (("numpy", HAVE_NUMPY), (f"kolibri1 {BIN}", BIN.exists())) if not ok]
        if missing:
            msg = "the residual-rule checks need " + ", ".join(missing)
            if os.environ.get("K1_REQUIRE_FORWARD") == "1":
                raise AssertionError(msg + " -- K1_REQUIRE_FORWARD=1 forbids skipping")
            raise unittest.SkipTest(msg)
        cls.tmp = tempfile.TemporaryDirectory(); cls.t = Path(cls.tmp.name)
        cls.fx = cls.t / "fx"
        subprocess.run([sys.executable, str(TOOL), "--out", str(cls.fx), "--layout", "release", "--force"], check=True, capture_output=True)
        cfg = json.loads((cls.fx / "config.json").read_text())
        cls.cfg = cfg
        cls.H, cls.E, cls.K = cfg["hidden_size"], cfg["num_experts"], cfg["num_experts_per_tok"]
        cls.qw, cls.kw = cfg["num_attention_heads"] * cfg["head_dim"], cfg["num_key_value_heads"] * cfg["head_dim"]
        cls.eps = cfg["rms_norm_eps"]
        W = {}
        for f in sorted(cls.fx.glob("*.safetensors")):
            W.update(st_tensors(f))
        cls.W = W
        cls.L = 1                                           # the probed layer

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ------------------------------------------------------------------ helpers ----
    @classmethod
    def env(cls, extra):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("K1_", "COLI_"))}
        e["OMP_NUM_THREADS"] = "2"
        e.update(extra)
        return e

    @classmethod
    def go(cls, extra):
        return subprocess.run([str(BIN), str(cls.fx)], capture_output=True, text=True, env=cls.env(extra), timeout=600)

    def probe(self, mode, indir, outdir, **extra):
        outdir.mkdir(exist_ok=True)
        return self.go({"K1_PROBE": mode, "K1_PROBE_IN": str(indir), "K1_PROBE_OUT": str(outdir), "K1_PROBE_T": str(T), "K1_PROBE_LAYER": str(self.L), **{k: str(v) for k, v in extra.items()}})

    def wvec(self, name):
        return [Fraction(float(v)) for v in self.W[name]]

    def ref_norm(self, x64, wname):
        return rms_ref(x64, self.wvec(wname), self.eps, self.H)

    def crafted(self, mild=False):
        """rows of FP64 residual inputs: 0 plain, 1 sub-FP32-ulp information, 2 cancelling, 3 massive coordinate, 4 plain, 5 mixture"""
        rng = np.random.default_rng(20261008)
        H = self.H
        base = rng.normal(size=(T, H)).astype(np.float32).astype(np.float64)
        x = base.copy()
        x[1] = base[1] * (1.0 + 2.0 ** -29)                  # strictly between two FP32 values for (almost) every element: information below one fp32 ulp
        big = 2.0 ** 20 + 2.0 ** -10
        if mild:                                              # the production layer probe: values the tiny model can digest (finite attention / experts)
            x[3, H // 3] = 50.0 * (1.0 + 2.0 ** -31)
            x[5] = base[5] * (1.0 + 2.0 ** -30)
            return x, None
        x[2, : H // 2] = big
        x[3, H // 3] = 1.0e4 * (1.0 + 2.0 ** -31)
        x[5] = np.where(np.arange(H) % 2, base[5] * (1.0 + 2.0 ** -30), big)
        pan = rng.normal(size=(T, H)).astype(np.float32)
        pan[2, : H // 2] = -(2.0 ** 20)                      # cancels the large part exactly: the result 2^-10 lives only in the FP64 stream
        pan[5, 0::2] = -(2.0 ** 20)
        return x, pan

    # ------------------------------------------------------------------ ops probe ----
    def test_ops_probe_steps_1_8_14_15_against_the_exact_reference(self):
        H, E, K, qw, kw = self.H, self.E, self.K, self.qw, self.kw
        rng = np.random.default_rng(5)
        ind = self.t / "ops_in"; ind.mkdir()
        x, pan = self.crafted()
        ffn_i = x[::-1].copy()                               # a second, different FP64 residual for step 15
        pf = pan[::-1].copy()
        widths = {"xin": H, "qkv": qw + 2 * kw, "q_n": qw, "k_n": kw, "q_a": qw, "k_a": kw, "v": kw, "attn_pre_o": qw, "attn_o": H, "x2": H,
                  "router_logits": E, "sel_w": K, "routed": H, "shared": H, "moe_sum": H}
        for nm, w in widths.items():
            rng.normal(size=(T, w)).astype(np.float32).tofile(ind / f"L{self.L}.{nm}.f32")
        pan.tofile(ind / f"L{self.L}.post_attn_norm.f32"); pf.tofile(ind / f"L{self.L}.post_ffn.f32")
        x.astype("<f8").tofile(ind / f"L{self.L}.x_prev.f64"); ffn_i.astype("<f8").tofile(ind / f"L{self.L}.ffn_inp.f64")
        np.stack([rng.permutation(E)[:K] for _ in range(T)]).astype(np.int32).tofile(ind / f"L{self.L}.sel_ids.i32")
        out = self.t / "ops_out"
        r = self.probe("ops", ind, out)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        got_ffn = np.fromfile(out / f"L{self.L}.ffn_inp.f64", dtype="<f8").reshape(T, H)
        got_xo = np.fromfile(out / f"L{self.L}.x_out.f64", dtype="<f8").reshape(T, H)
        got_xin = np.fromfile(out / f"L{self.L}.xin.f32", dtype="<f4").reshape(T, H)
        got_x2 = np.fromfile(out / f"L{self.L}.x2.f32", dtype="<f4").reshape(T, H)
        lw = f"model.layers.{self.L}."
        a_ffn = np.empty_like(x); a_xo = np.empty_like(x)
        for t in range(T):
            for j in range(H):
                a_ffn[t, j] = float(r64(Fraction(float(x[t, j])) + Fraction(float(pan[t, j]))))
                a_xo[t, j] = float(r64(Fraction(float(ffn_i[t, j])) + Fraction(float(pf[t, j]))))
            # step 8 (+ the pre-FFN norm of the FP64 sum), step 15
            self.assertTrue(np.array_equal(got_ffn[t], a_ffn[t]), f"row {t}: ffn_inp (FP64 add) differs from the exact FP64-rounded sum")
            self.assertTrue(np.array_equal(got_xo[t], a_xo[t]), f"row {t}: x_out (FP64 add) differs")
            self.assertTrue(np.array_equal(got_xin[t], self.ref_norm(x[t], lw + "input_layernorm.weight")), f"row {t}: xin = FP32(RMSNorm_FP64(x_prev)) differs")
            self.assertTrue(np.array_equal(got_x2[t], self.ref_norm(a_ffn[t], lw + "post_attention_layernorm.weight")), f"row {t}: x2 = FP32(RMSNorm_FP64(ffn_inp)) differs")
        # scalar == AVX2 for the residual path: the same steps with the scalar FP8 kernel selected (K1_FP8_KERNEL=scalar) give the same bits (these steps contain no FP8 GEMM)
        out_s = self.t / "ops_out_scalar"
        r = self.probe("ops", ind, out_s, K1_FP8_KERNEL="scalar")
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        for nm in ("ffn_inp.f64", "x_out.f64", "xin.f32", "x2.f32"):
            self.assertEqual((out / f"L{self.L}.{nm}").read_bytes(), (out_s / f"L{self.L}.{nm}").read_bytes(), f"scalar vs vector kernel: {nm}")
        # the row-1 / row-2 / row-5 FP64 sums carry information an FP32 stream cannot hold
        self.assertTrue(np.any(got_ffn[1] != got_ffn[1].astype(np.float32).astype(np.float64)), "row 1 should hold sub-fp32-ulp information")
        self.assertTrue(np.all(got_ffn[2, : H // 2] == 2.0 ** -10), "the cancellation 2^20 + 2^-10 - 2^20 = 2^-10 must survive")
        # CONTROL: candidate A's stream (narrow to FP32, add in FP32) gives other bits
        a_emul = (x.astype(np.float32) + pan).astype(np.float64)
        self.assertTrue(np.any(a_emul[1] != got_ffn[1]), "control: the fp32 stream must differ on the sub-ulp row")
        self.assertTrue(np.any(a_emul[2, : H // 2] != got_ffn[2, : H // 2]), "control: the fp32 stream loses the cancellation result")
        self.assertTrue(np.any(a_emul[5] != got_ffn[5]), "control: the fp32 stream differs on the mixed row")
        a_x2 = np.stack([self.ref_norm(a_emul[t], lw + "post_attention_layernorm.weight") for t in range(T)])
        self.assertFalse(np.array_equal(a_x2[2], got_x2[2]), "control: the fp32 stream also changes the normalised activation of the cancelling row")

    # ------------------------------------------------------------------ layer probe (production path) ----
    def test_layer_probe_obeys_the_rule_and_is_chunk_invariant(self):
        ind = self.t / "layer_in"; ind.mkdir()
        x, _ = self.crafted(mild=True)
        x.astype("<f8").tofile(ind / f"L{self.L}.x_prev.f64")
        outs = {}
        for chunk in (1, 3, 64):
            o = self.t / f"layer_out_{chunk}"
            r = self.probe("layer", ind, o, K1_CHUNK=chunk)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            outs[chunk] = o
        names = sorted(p.name for p in outs[64].iterdir())
        self.assertIn(f"L{self.L}.x_out.f64", names); self.assertIn(f"L{self.L}.ffn_inp.f64", names)
        self.assertNotIn(f"L{self.L}.x_out.f32", names)
        for chunk in (1, 3):
            self.assertEqual(sorted(p.name for p in outs[chunk].iterdir()), names)
            for n in names:
                self.assertEqual((outs[chunk] / n).read_bytes(), (outs[64] / n).read_bytes(), f"chunk {chunk} vs 64: {n}")
        o_s = self.t / "layer_out_scalar"                      # the production layer with the scalar FP8 kernel: other GEMM bits (A's two kernels differ), the SAME residual rule
        r = self.probe("layer", ind, o_s, K1_FP8_KERNEL="scalar")
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        for oo in (outs[64], o_s):
            self._check_rule(oo, x)
    def _check_rule(self, o, x):
        H = self.H
        a32 = np.fromfile(o / f"L{self.L}.post_attn_norm.f32", dtype="<f4").reshape(T, H)
        f32 = np.fromfile(o / f"L{self.L}.post_ffn.f32", dtype="<f4").reshape(T, H)
        ffn = np.fromfile(o / f"L{self.L}.ffn_inp.f64", dtype="<f8").reshape(T, H)
        xo = np.fromfile(o / f"L{self.L}.x_out.f64", dtype="<f8").reshape(T, H)
        x2 = np.fromfile(o / f"L{self.L}.x2.f32", dtype="<f4").reshape(T, H)
        lw = f"model.layers.{self.L}."
        for t in range(T):
            e1 = np.array([float(r64(Fraction(float(x[t, j])) + Fraction(float(a32[t, j])))) for j in range(H)])
            e2 = np.array([float(r64(Fraction(float(ffn[t, j])) + Fraction(float(f32[t, j])))) for j in range(H)])
            self.assertTrue(np.array_equal(ffn[t], e1), f"row {t}: ffn_inp != h + FP64(a32)")
            self.assertTrue(np.array_equal(xo[t], e2), f"row {t}: x_out != h_attn + FP64(f32)")
            self.assertTrue(np.array_equal(x2[t], self.ref_norm(ffn[t], lw + "post_attention_layernorm.weight")), f"row {t}: x2 != FP32(RMSNorm_FP64(ffn_inp))")

    # ------------------------------------------------------------------ tails ----
    def test_tails_final_norm_reads_the_fp64_stream(self):
        H = self.H
        rng = np.random.default_rng(9)
        x, _ = self.crafted()
        ind = self.t / "tails_in"; ind.mkdir()
        x.astype("<f8").tofile(ind / "final_in.f64")
        rng.integers(0, 10, size=T).astype(np.int32).tofile(ind / "ids.i32")
        rng.normal(size=(T, H)).astype("<f4").tofile(ind / "final_norm.f32")
        out = self.t / "tails_out"; out.mkdir()
        r = self.go({"K1_PROBE": "ops", "K1_PROBE_TAILS": "1", "K1_PROBE_IN": str(ind), "K1_PROBE_OUT": str(out), "K1_PROBE_T": str(T)})
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        fn = np.fromfile(out / "final_norm.f32", dtype="<f4").reshape(T, H)
        for t in range(T):
            self.assertTrue(np.array_equal(fn[t], self.ref_norm(x[t], "model.norm.weight")), f"row {t}: final_norm differs from FP32(RMSNorm_FP64(final_in))")

    # ------------------------------------------------------------------ refusals ----
    def test_an_fp32_residual_file_is_refused_never_converted(self):
        H = self.H
        ind = self.t / "bad_in"; ind.mkdir()
        np.zeros((T, H), dtype="<f4").tofile(ind / f"L{self.L}.x_prev.f32")
        out = self.t / "bad_out"
        r = self.probe("layer", ind, out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("x_prev.f64 is missing", r.stderr)
        ind2 = self.t / "short_in"; ind2.mkdir()
        np.zeros((T, H), dtype="<f4").tofile(ind2 / f"L{self.L}.x_prev.f64")            # half the bytes of an FP64 file of T rows
        r = self.probe("layer", ind2, self.t / "short_out")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("shorter", r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
