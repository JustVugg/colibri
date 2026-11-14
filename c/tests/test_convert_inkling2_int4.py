"""tools/convert_inkling2_int4.py: Inkling-2's NVFP4 and MXFP8 experts to int4 g64.

A small plain snapshot under the official names is quantized as the official checkpoint lays
it out (--fabricate: NVFP4 experts in the first layer, MXFP8 in the second, two shards with a
layer across them), then converted. The container must hold every tensor of each layer, the
experts as expert_ffn.h's planar int4 with a scale per 64 (gate rows, then up rows), near the
source weights; a shard is removed only once every output that reads it is written, and a
second run finds nothing left to do.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:   # pragma: no cover
    raise unittest.SkipTest(f"numpy not installed: {exc}")

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
import convert_inkling2_int4 as cv   # noqa: E402

E, D, I = 4, 128, 64


def run(*args):
    return subprocess.run([sys.executable, str(TOOLS / "convert_inkling2_int4.py"), *args],
                          capture_output=True, text=True, timeout=300)


class ConvertInkling2Test(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        rng = np.random.default_rng(3)
        self.w = {}
        tens = {}
        for L in range(2):
            p = f"model.llm.layers.{L}."
            w13 = (rng.standard_normal((E, 2 * I, D)) * 0.02).astype(np.float32)
            w2 = (rng.standard_normal((E, D, I)) * 0.02).astype(np.float32)
            self.w[L] = (w13, w2)
            tens[p + "mlp.experts.w13_weight"] = ("F32", w13.shape, w13.tobytes())
            tens[p + "mlp.experts.w2_weight"] = ("F32", w2.shape, w2.tobytes())
            g = rng.standard_normal((E, D)).astype(np.float32)
            tens[p + "mlp.gate.weight"] = ("F32", g.shape, g.tobytes())
            bf = (rng.standard_normal(D * 8).astype(np.float32).view(np.uint32) >> 16).astype("<u2")
            tens[p + "attn.wq_du.weight"] = ("BF16", (8, D), bf.tobytes())
        emb = rng.standard_normal((16, D)).astype(np.float32)
        tens["model.llm.embed.weight"] = ("F32", emb.shape, emb.tobytes())
        self.snap = self.root / "snap"
        self.snap.mkdir()
        cv.write_safetensors(str(self.snap / "model.safetensors"), tens)
        (self.snap / "config.json").write_text(json.dumps({
            "model_type": "inkling2", "quantization_config": {"quant_method": "modelopt"},
            "text_config": {"hidden_size": D, "moe_intermediate_size": I, "n_routed_experts": E}}))
        self.fab = self.root / "fab"
        r = run("--fabricate", str(self.snap), str(self.fab), "--nvfp4-layers", "1")
        self.assertEqual(r.returncode, 0, r.stderr)

    def tensors(self, d):
        out = {}
        for p in sorted(Path(d).glob("*.safetensors")):
            for k, (dt, shape, off, nb) in cv.read_header(str(p))[0].items():
                out[k] = (dt, shape, cv.read_bytes(str(p), off, nb))
        return out

    def test_fabricated_source_has_the_official_layout(self):
        t = self.tensors(self.fab)
        self.assertEqual(t["model.llm.layers.0.mlp.experts.w13_weight"][:2], ("U8", (E, 2 * I, D // 2)))
        self.assertEqual(t["model.llm.layers.0.mlp.experts.w13_weight_scale"][:2], ("F8_E4M3", (E, 2 * I, D // 16)))
        self.assertEqual(t["model.llm.layers.1.mlp.experts.w13_weight"][:2], ("F8_E4M3", (E, 2 * I, D)))
        self.assertEqual(t["model.llm.layers.1.mlp.experts.w2_weight_scale"][:2], ("U8", (E, D, I // 32)))
        wmap = json.loads((self.fab / "model.safetensors.index.json").read_text())["weight_map"]
        self.assertEqual(len(set(wmap.values())), 2)
        straddle = [L for L in range(2) if len({s for k, s in wmap.items() if f"layers.{L}." in k}) == 2]
        self.assertTrue(straddle, "no layer across the two shards")

    def test_convert_delete_src_and_resume(self):
        out = self.root / "out"
        r = run("--indir", str(self.fab), "--outdir", str(out), "--jobs", "2", "--delete-src",
                "--min-free-gb", "0")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("conversion complete", r.stdout)
        self.assertEqual(sorted(p.name for p in self.fab.glob("*.safetensors")), [])
        t = self.tensors(out)
        for L, fmt_err in ((0, 0.16), (1, 0.13)):
            p = f"model.llm.layers.{L}.mlp.experts."
            self.assertNotIn(p + "w13_weight", t)
            self.assertNotIn(p + "w13_weight_scale", t)
            dt, shape, raw = t[p + "gate_up_proj"]
            self.assertEqual((dt, shape), ("U8", (E, 2 * I, D // 2)))
            q = np.frombuffer(raw, np.uint8).reshape(shape)
            dt, sshape, sraw = t[p + "gate_up_proj.qs"]
            self.assertEqual((dt, sshape), ("F32", (E, 2 * I, D // 64)))
            s = np.frombuffer(sraw, "<f4").reshape(sshape)
            w13, w2 = self.w[L]
            for e in range(E):
                got = cv.unpack_planar(q[e], s[e])
                want = np.concatenate([w13[e][0::2], w13[e][1::2]])   # gate rows, then up rows
                rel = np.linalg.norm(got - want) / np.linalg.norm(want)
                self.assertLess(rel, fmt_err, f"layer {L} expert {e} gate_up {rel:.3f}")
            dt, shape, raw = t[p + "down_proj"]
            self.assertEqual((dt, shape), ("U8", (E, D, I // 2)))
            q2 = np.frombuffer(raw, np.uint8).reshape(shape)
            s2 = np.frombuffer(t[p + "down_proj.qs"][2], "<f4").reshape(E, D, I // 64)
            rel = np.linalg.norm(cv.unpack_planar(q2[1], s2[1]) - w2[1]) / np.linalg.norm(w2[1])
            self.assertLess(rel, fmt_err)
            self.assertEqual(t[f"model.llm.layers.{L}.attn.wq_du.weight"][:2], ("BF16", (8, D)))
            self.assertEqual(t[f"model.llm.layers.{L}.mlp.gate.weight"][:2], ("F32", (E, D)))
        self.assertIn("model.llm.embed.weight", t)
        cfg = json.loads((out / "config.json").read_text())
        self.assertNotIn("quantization_config", cfg)
        self.assertEqual(cfg["colibri_experts"], "int4-g64")
        # every tensor sits on its element size; the file is exactly what the header indexes
        for p in out.glob("*.safetensors"):
            hdr, _, size = cv.read_header(str(p))
            self.assertEqual(os.path.getsize(p), size)
            for dt, shape, off, nb in hdr.values():
                self.assertEqual(off % cv.DT_SIZE[dt], 0)
        again = run("--indir", str(self.fab), "--outdir", str(out), "--jobs", "1", "--min-free-gb", "0")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertNotIn("layer 0", again.stdout)

    def test_missing_shard_is_reported(self):
        os.remove(self.fab / "model-00002-of-00002.safetensors")
        r = run("--indir", str(self.fab), "--outdir", str(self.root / "out2"), "--jobs", "1",
                "--min-free-gb", "0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("model-00002-of-00002.safetensors", r.stderr)

    def test_disk_floor_stops_the_run(self):
        out = self.root / "out3"
        r = run("--indir", str(self.fab), "--outdir", str(out), "--jobs", "1", "--delete-src",
                "--min-free-gb", "1000000000")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not enough disk", r.stderr)
        self.assertEqual(sorted(p.name for p in out.glob("*.safetensors")), [])
        self.assertEqual(len(list(self.fab.glob("*.safetensors"))), 2)   # nothing removed

    def test_selftest(self):
        r = run("--selftest")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("SELFTEST OK", r.stdout)


if __name__ == "__main__":
    unittest.main()
