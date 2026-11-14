"""tools/convert_qwen38_experts_mxfp4.py: AMD Quark's MXFP4 routed experts as Qwen3.8's experts-mxfp4/ sidecar.

As a unittest (numpy only, part of `make test`) it converts a synthetic FP8-side
snapshot and Quark checkpoint (widths with a short last group of 32, records
that need a pad) and reads the sidecar back with its own parser:
  * the layout the engine reads: data on a 4096-byte boundary, every expert one
    record of six U8 tensors back to back in index.json's order, then its pad,
    so every record starts on a 4096-byte boundary; index.json's fields;
  * the bytes: every record is the Quark tensors' bytes, unchanged;
  * resume: a layer subset stays "complete": false, a second run rewrites
    nothing, a removed layer comes back with the same bytes, a changed
    checkpoint is refused;
  * refusals: swapped nibbles (the check against the snapshot's experts), a
    wrong dtype, a wrong shape, a missing expert, the E8M0 NaN.

As a script it verifies a converted snapshot: every record equals the Quark
bytes and every expert, dequantized, equals the snapshot's expert (--exact:
to the bit, as the tiny fixture's FP8 shards hold the MXFP4 values):

  python3 tests/test_qwen38_mxfp4_convert.py --model ./qwen38_tiny_mxfp4 --exact
"""
import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
sys.path.insert(0, str(HERE))
if HAVE_NUMPY:
    from test_qwen38_int4_convert import Snapshot, read_bytes, read_safetensors, write_safetensors

PROJ = ("gate_proj", "up_proj", "down_proj")
RECORD = [("gate_proj", ".weight_scale"), ("down_proj", ".weight_scale"), ("up_proj", ".weight_scale"),
          ("gate_proj", ".weight"), ("down_proj", ".weight"), ("up_proj", ".weight")]
E2M1 = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]


def shape_of(proj, suffix, hidden, inter):
    rows, cols = (hidden, inter) if proj == "down_proj" else (inter, hidden)
    return [rows, -(-cols // 32) if suffix == ".weight_scale" else -(-cols // 2)], cols


def dequantize(nibbles, scales, rows, cols):
    w = np.frombuffer(nibbles, np.uint8).reshape(rows, -1)
    lut = np.array(E2M1, np.float32)
    v = np.stack([lut[w & 15], lut[w >> 4]], -1).reshape(rows, -1)[:, :cols]
    s = np.frombuffer(scales, np.uint8).reshape(rows, -1).astype(np.float32)
    return v * np.repeat(np.exp2(s - 127), 32, axis=1)[:, :cols]


class Quark:
    """Every tensor of a Quark checkpoint by name: (dtype, shape, bytes)."""

    def __init__(self, quark_dir):
        self.tensors = {}
        for shard in sorted(Path(quark_dir).glob("*.safetensors")):
            tensors, _, _ = read_safetensors(shard)
            for name, (dtype, shape, off, n) in tensors.items():
                self.tensors[name] = (dtype, shape, read_bytes(shard, off, n))
        self.prefix = next(p for p in ("model.language_model", "model")
                           if any(k.startswith(p + ".layers.") for k in self.tensors))


def verify(model_dir, quark_dir=None, exact=False, min_cos=0.95):
    """Every record of <model>/experts-mxfp4/: its layout, its bytes against the Quark
    checkpoint's, its values against the snapshot's experts. Returns the worst cosine."""
    model_dir = Path(model_dir)
    side = model_dir / "experts-mxfp4"
    index = json.loads((side / "index.json").read_text(encoding="utf-8"))
    snap = Snapshot(model_dir)
    quark = Quark(quark_dir or model_dir / "quark")
    assert (index["format"], index["version"], index["group_size"]) == ("colibri.qwen38.experts-mxfp4", 1, 32), index
    assert (index["codes"], index["scales"], index["record_align"]) == ("e2m1-lo-even", "ue8m0-per-32", 4096)
    assert index["record"] == ["gate scales", "down scales", "up scales", "gate e2m1", "down e2m1", "up e2m1"]
    assert (index["layers"], index["experts"], index["hidden_size"], index["moe_intermediate_size"]) == \
        (snap.layers, snap.experts, snap.hidden, snap.inter)
    assert index["prefix"] == snap.prefix
    assert index["complete"] is True and index["done"] == list(range(snap.layers)), "incomplete sidecar"
    worst = 1.0
    for layer in range(snap.layers):
        path = side / f"layer-{layer:03d}.safetensors"
        tensors, data_start, meta = read_safetensors(path)
        assert data_start % 4096 == 0, f"layer {layer}: data starts at {data_start}"
        assert meta == {"format": index["format"], "version": "1", "layer": str(layer)}, meta
        end = data_start
        for expert in range(snap.experts):
            base = f"{snap.prefix}.layers.{layer}.mlp.experts.{expert}"
            at = start = end
            parts = {}
            assert start % 4096 == 0, f"layer {layer} expert {expert}: record at {start}"
            for proj, suffix in RECORD:
                dtype, shape, off, n = tensors[f"{base}.{proj}{suffix}"]
                want, _ = shape_of(proj, suffix, snap.hidden, snap.inter)
                assert (dtype, shape, off) == ("U8", want, at), (base, proj, suffix, dtype, shape, off - start)
                raw = read_bytes(path, off, n)
                qname = f"{quark.prefix}.layers.{layer}.mlp.experts.{expert}.{proj}{suffix}"
                assert raw == quark.tensors[qname][2], f"{qname}: the record's bytes are not Quark's"
                parts[(proj, suffix)] = raw
                at += n
            assert at - start == index["record_bytes"], (at - start, index["record_bytes"])
            end = start + -(-index["record_bytes"] // 4096) * 4096
            if end > at:
                dtype, shape, off, n = tensors[f"{base}.record_pad"]
                assert (dtype, shape, off, off + n) == ("U8", [end - at], at, end), (base, "pad")
                assert read_bytes(path, off, n) == bytes(n)
            for k, proj in enumerate(PROJ):
                (rows, _), cols = shape_of(proj, ".weight", snap.hidden, snap.inter)
                q = dequantize(parts[(proj, ".weight")], parts[(proj, ".weight_scale")], rows, cols)
                r = snap.matrix(layer, expert, k)
                if exact and not np.array_equal(q, r):
                    raise AssertionError(f"layer {layer} expert {expert} {proj}: the MXFP4 values differ from "
                                         f"the snapshot's (max {np.abs(q - r).max()})")
                q, r = q.astype(np.float64), r.astype(np.float64)
                cos = float((q * r).sum() / (np.linalg.norm(q) * np.linalg.norm(r) + 1e-30))
                assert cos >= min_cos, (layer, expert, proj, cos)
                worst = min(worst, cos)
        assert os.path.getsize(path) == end, f"layer {layer}: {os.path.getsize(path)} bytes, records end at {end}"
    return worst


# ---------- a synthetic snapshot and Quark checkpoint ----------

def bf16(values):
    """f32 values exact in BF16 (MXFP4 values are) -> BF16 bytes."""
    return (np.ascontiguousarray(values, np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()


def make_model(root, layers=2, experts=3, hidden=64, inter=40, seed=11):
    """A snapshot with BF16 experts holding the MXFP4 values, and quark/ with the MXFP4 bytes
    in two shards plus their index (gate/up scales as U8, down's as F8_E8M0). hidden 64 is two
    groups of 32; inter 40 is one and a short one of 8; a record is 4,128 bytes, 8,192 with its pad."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    (root / "quark").mkdir()
    prefix = "model.language_model"
    snap = {f"{prefix}.embed_tokens.weight": ("F32", [4, hidden], np.zeros((4, hidden), np.float32).tobytes())}
    shards = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    for layer in range(layers):
        for expert in range(experts):
            for proj in PROJ:
                (rows, nb), cols = shape_of(proj, ".weight", hidden, inter)
                nibbles = rng.integers(0, 256, size=(rows, nb), dtype=np.uint8)
                scales = rng.integers(118, 128, size=(rows, -(-cols // 32)), dtype=np.uint8)
                name = f"{prefix}.layers.{layer}.mlp.experts.{expert}.{proj}"
                shard = "model-00002-of-00002.safetensors" if proj == "down_proj" else "model-00001-of-00002.safetensors"
                shards[shard][name + ".weight"] = ("U8", [rows, nb], nibbles.tobytes())
                shards[shard][name + ".weight_scale"] = ("F8_E8M0" if proj == "down_proj" else "U8",
                                                         list(scales.shape), scales.tobytes())
                snap[name + ".weight"] = ("BF16", [rows, cols],
                                          bf16(dequantize(nibbles.tobytes(), scales.tobytes(), rows, cols)))
    write_safetensors(root / "model.safetensors", snap)
    weight_map = {}
    for fname, tensors in shards.items():
        write_safetensors(root / "quark" / fname, tensors)
        weight_map.update({name: fname for name in tensors})
    (root / "quark" / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (root / "config.json").write_text(json.dumps({"text_config": {
        "model_type": "qwen4_exp_text", "num_hidden_layers": layers, "num_experts": experts,
        "hidden_size": hidden, "moe_intermediate_size": inter}}))


def edit_quark(root, name, dtype=None, shape=None, raw=None, drop=False):
    """Rewrite the Quark shard holding `name` with that tensor changed (or dropped)."""
    index = json.loads((root / "quark" / "model.safetensors.index.json").read_text())
    path = root / "quark" / index["weight_map"][name]
    tensors, _, _ = read_safetensors(path)
    out = {}
    for key, (d, s, off, n) in tensors.items():
        if key == name and drop:
            continue
        body = read_bytes(path, off, n)
        if key == name:
            d, s, body = dtype or d, shape or s, body if raw is None else raw(body)
        out[key] = (d, s, body)
    write_safetensors(path, out)


@unittest.skipUnless(HAVE_NUMPY, "numpy is required")
class ConvertQwen38ExpertsMxfp4(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "snap"
        make_model(self.root)
        sys.path.insert(0, str(TOOLS))
        import convert_qwen38_experts_mxfp4 as converter
        self.converter = converter
        self.side = self.root / "experts-mxfp4"

    def tearDown(self):
        self.tmp.cleanup()

    def convert(self, **kw):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.converter.convert(str(self.root), str(self.root / "quark"), **kw)
        return buf.getvalue()

    def test_layout_and_bytes(self):
        self.convert()
        index = json.loads((self.side / "index.json").read_text())
        self.assertEqual((index["record_bytes"], index["complete"]), (4128, True))
        self.assertGreater(verify(self.root, exact=True), 0.999999)
        self.assertEqual(sorted(p.name for p in self.side.iterdir()),
                         ["index.json", "layer-000.safetensors", "layer-001.safetensors"])

    def test_release_record(self):
        g = self.converter.Geometry(48, 256, 2560, 640)
        self.assertEqual((g.record_bytes, g.stride), (2611200, 2613248))

    def test_resume(self):
        self.convert(layers="0")
        index = json.loads((self.side / "index.json").read_text())
        self.assertEqual((index["done"], index["complete"]), ([0], False))
        self.assertFalse((self.side / "layer-001.safetensors").exists())
        first = (self.side / "layer-000.safetensors").read_bytes()
        mtime = (self.side / "layer-000.safetensors").stat().st_mtime_ns
        self.assertIn("1 layer(s) already converted", self.convert())
        self.assertEqual((self.side / "layer-000.safetensors").stat().st_mtime_ns, mtime)
        self.assertTrue(json.loads((self.side / "index.json").read_text())["complete"])
        (self.side / "layer-000.safetensors").unlink()
        self.assertIn("converting it again", self.convert())
        self.assertEqual((self.side / "layer-000.safetensors").read_bytes(), first)
        with open(self.root / "quark" / "model-00002-of-00002.safetensors", "ab") as f:
            f.write(b"\0")   # another checkpoint: the sidecar is not resumed from it
        with self.assertRaises(SystemExit):
            self.convert()

    def refused(self, needle, **edit):
        name = "model.language_model.layers.1.mlp.experts.2." + edit.pop("tensor")
        edit_quark(self.root, name, **edit)
        buf = io.StringIO()
        with self.assertRaises(SystemExit) as caught, redirect_stdout(buf):
            self.converter.convert(str(self.root), str(self.root / "quark"))
        self.assertIn(needle, str(caught.exception.code))
        index = json.loads((self.side / "index.json").read_text())
        self.assertEqual((index["done"], index["complete"]), ([0], False))
        self.assertFalse((self.side / "layer-001.safetensors").exists())

    def test_swapped_nibbles(self):
        self.refused("cosine", tensor="up_proj.weight",
                     raw=lambda b: bytes(((x & 15) << 4) | (x >> 4) for x in b))

    def test_wrong_dtype(self):
        self.refused("expected U8", tensor="down_proj.weight", dtype="I8")

    def test_wrong_shape(self):
        self.refused("expected U8", tensor="gate_proj.weight_scale", shape=[2, 40])

    def test_missing_expert(self):
        self.refused("is missing", tensor="up_proj.weight_scale", drop=True)

    def test_e8m0_nan(self):
        self.refused("NaN", tensor="gate_proj.weight_scale", raw=lambda b: b"\xff" + b[1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="a snapshot with experts-mxfp4/")
    ap.add_argument("--quark", help="the Quark checkpoint it was converted from (default: <model>/quark)")
    ap.add_argument("--exact", action="store_true", help="the snapshot's experts must hold the MXFP4 values exactly")
    a = ap.parse_args()
    worst = verify(a.model, a.quark, a.exact)
    print(f"test_qwen38_mxfp4_convert: ok, {a.model}/experts-mxfp4 holds the Quark bytes in the documented layout; "
          + ("its values are the snapshot's to the bit" if a.exact else f"worst cosine against the snapshot {worst:.5f}"))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    unittest.main()
