"""tools/convert_qwen38_experts_int4.py: the int4-g64 sidecar of Qwen3.8's routed experts.

As a unittest (numpy only, part of `make test`) it converts a synthetic FP8
snapshot whose widths have partial FP8 blocks and an int4 tail, then reads the
sidecar back with its own parser and checks, against the source dequantized
with quant.h's own E4M3_LUT (parsed from the C source):
  * the layout the engine validates: data on a 4096-byte boundary, every
    expert one record of six tensors back to back, index.json complete;
  * the quantizer: every scale is its group's absmax / 7 in f32, every value
    within half a scale of the source, relative L2 error that of int4-g64;
  * resume: a second run rewrites nothing, a removed layer is converted again
    to the same bytes, a changed snapshot is refused.

As a script it verifies an existing sidecar against its snapshot and prints the
relative L2 error of every expert matrix (sampled with --layers/--experts on
the real model), and with --engine it checks that qwen38 refuses a sidecar that
disagrees with its index and falls back from an incomplete one:

  python3 tests/test_qwen38_int4_convert.py --model ./qwen38_tiny_int4 --engine ./qwen38
"""
import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
QUANT_H = HERE.parent / "quant.h"
PROJ = ("gate_proj", "up_proj", "down_proj")


# ---------- independent readers ----------

def engine_e4m3_table():
    """quant.h's E4M3_LUT, read from the C source the engine compiles."""
    text = QUANT_H.read_text(encoding="utf-8")
    body = text[text.index("E4M3_LUT[256] = {") + len("E4M3_LUT[256] = {"):]
    body = body[:body.index("};")]
    values = []
    for item in body.replace("\n", " ").split(","):
        item = item.strip().rstrip("f")
        if not item:
            continue
        values.append(math.nan if item.upper() == "NAN" else float.fromhex(item))
    assert len(values) == 256, len(values)
    return np.array(values, np.float32)


def read_safetensors(path):
    """{name: (dtype, shape, absolute offset, nbytes)} and the data start."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    tensors = {k: (v["dtype"], v["shape"], 8 + n + v["data_offsets"][0],
                   v["data_offsets"][1] - v["data_offsets"][0])
               for k, v in header.items() if k != "__metadata__"}
    return tensors, 8 + n, header.get("__metadata__", {})


def read_bytes(path, offset, nbytes):
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(nbytes)


def floats(raw, dtype):
    if dtype == "BF16":
        return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(raw, np.float16).astype(np.float32)
    return np.frombuffer(raw, np.float32)


class Snapshot:
    """The routed experts of a Qwen3.8 snapshot, dequantized as the engine's
    expanded FP8 path does (byte -> E4M3_LUT, times the f32 block scale)."""

    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        config = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        text = config.get("text_config", config)
        self.layers, self.experts = text["num_hidden_layers"], text["num_experts"]
        self.hidden, self.inter = text["hidden_size"], text["moe_intermediate_size"]
        self.where = {}
        for shard in sorted(self.dir.glob("*.safetensors")):
            tensors, _, _ = read_safetensors(shard)
            for name, info in tensors.items():
                self.where[name] = (shard, info)
        self.prefix = next(p for p in ("model.language_model", "model")
                           if p + ".embed_tokens.weight" in self.where)
        self.lut = engine_e4m3_table()

    def matrix(self, layer, expert, k):
        name = f"{self.prefix}.layers.{layer}.mlp.experts.{expert}.{PROJ[k]}.weight"
        shard, (dtype, shape, off, n) = self.where[name]
        rows, cols = shape
        raw = read_bytes(shard, off, n)
        if dtype.startswith("F8") or dtype == "float8_e4m3fn":
            sshard, (sdtype, sshape, soff, sn) = self.where[name + "_scale_inv"]
            block = floats(read_bytes(sshard, soff, sn), sdtype).reshape(sshape)
            out = np.empty((rows, cols), np.float32)
            codes = np.frombuffer(raw, np.uint8).reshape(rows, cols)
            for r in range(rows):
                for c0 in range(0, cols, 128):
                    out[r, c0:c0 + 128] = self.lut[codes[r, c0:c0 + 128]] * block[r // 128, c0 // 128]
            return out
        return floats(raw, dtype).reshape(rows, cols)


def planar_codes(packed_row, cols):
    """One int4-g64 row's codes (0..15), decoded element by element."""
    out = np.empty(cols, np.int32)
    body = cols // 64 * 64
    for i in range(cols):
        if i < body:
            block, k = packed_row[(i // 64) * 32:(i // 64) * 32 + 32], i % 64
            out[i] = block[k] & 15 if k < 32 else block[k - 32] >> 4
        else:
            byte = packed_row[i >> 1]
            out[i] = byte >> 4 if i & 1 else byte & 15
    return out


class Sidecar:
    def __init__(self, sidecar_dir):
        self.dir = Path(sidecar_dir)
        self.index = json.loads((self.dir / "index.json").read_text(encoding="utf-8"))
        self.files = {}

    def layer(self, layer):
        if layer not in self.files:
            self.files[layer] = read_safetensors(self.dir / f"layer-{layer:03d}.safetensors")
        return self.files[layer]

    def expert(self, prefix, layer, expert, shapes):
        """[(codes [R, C] 0..15, scales [R, G])] for gate, up, down, after checking
        that the six tensors are one record in record order."""
        tensors, data_start, _ = self.layer(layer)
        path = self.dir / f"layer-{layer:03d}.safetensors"
        base = f"{prefix}.layers.{layer}.mlp.experts.{expert}"
        names = [f"{base}.{p}.weight" for p in PROJ] + [f"{base}.{p}.weight.qs" for p in PROJ]
        start, at, parts = tensors[names[0]][2], 0, []
        for i, name in enumerate(names):
            dtype, shape, off, n = tensors[name]
            rows, cols = shapes[i % 3]
            want = ("U8", [rows, (cols + 1) // 2]) if i < 3 else ("F32", [rows, (cols + 63) // 64])
            if (dtype, shape) != want or off != start + at:
                raise AssertionError(f"{name}: {dtype} {shape} at +{off - start}, "
                                     f"expected {want[0]} {want[1]} at +{at}")
            parts.append(read_bytes(path, off, n))
            at += n
        if at != self.index["record_bytes"] or start % 4:
            raise AssertionError(f"layer {layer} expert {expert}: record of {at} bytes at {start}")
        out = []
        for k in range(3):
            rows, cols = shapes[k]
            packed = np.frombuffer(parts[k], np.uint8).reshape(rows, (cols + 1) // 2)
            codes = np.stack([planar_codes(packed[r], cols) for r in range(rows)])
            scales = np.frombuffer(parts[3 + k], np.float32).reshape(rows, (cols + 63) // 64)
            out.append((codes, scales))
        return out


def verify(model_dir, layers=None, experts=None, quiet=False):
    """Every sampled expert matrix: the quantizer's invariants hold and the
    relative L2 error against the source. Returns the list of errors."""
    snap = Snapshot(model_dir)
    side = Sidecar(snap.dir / "experts-int4g64")
    idx = side.index
    assert idx["format"] == "colibri.qwen38.experts-int4g64" and idx["version"] == 1, idx["format"]
    assert idx["group_size"] == 64 and idx["codes"] == "planar64-v8" and idx["scales"] == "f32-per-64"
    assert (idx["layers"], idx["experts"], idx["hidden_size"], idx["moe_intermediate_size"]) == \
        (snap.layers, snap.experts, snap.hidden, snap.inter)
    assert idx["complete"] is True and idx["done"] == list(range(snap.layers)), "incomplete sidecar"
    shapes = ((snap.inter, snap.hidden), (snap.inter, snap.hidden), (snap.hidden, snap.inter))
    errors = []
    for layer in layers if layers is not None else range(snap.layers):
        _, data_start, meta = side.layer(layer)
        assert data_start % 4096 == 0, f"layer {layer}: data starts at {data_start}"
        assert meta.get("layer") == str(layer), meta
        for expert in experts if experts is not None else range(snap.experts):
            stored = side.expert(snap.prefix, layer, expert, shapes)
            for k, (codes, scales) in enumerate(stored):
                w = snap.matrix(layer, expert, k)
                rows, cols = w.shape
                g = (cols + 63) // 64
                padded = np.zeros((rows, g * 64), np.float32)
                padded[:, :cols] = w
                amax = np.abs(padded.reshape(rows, g, 64)).max(axis=2)
                expect = np.maximum(amax, np.float32(1e-12)) / np.float32(7.0)
                if not np.array_equal(scales, expect):
                    raise AssertionError(f"layer {layer} expert {expert} {PROJ[k]}: scales are not absmax/7")
                values = (codes.astype(np.float32) - 8.0) * np.repeat(scales, 64, axis=1)[:, :cols]
                slack = np.repeat(scales, 64, axis=1)[:, :cols] * np.float32(0.5 + 1e-5)
                if not (np.abs(values - w) <= slack).all():
                    raise AssertionError(f"layer {layer} expert {expert} {PROJ[k]}: a value is "
                                         f"further than half a scale from its source")
                err = float(np.linalg.norm((values - w).astype(np.float64)) /
                            max(np.linalg.norm(w.astype(np.float64)), 1e-30))
                errors.append(err)
                if not quiet:
                    print(f"  layer {layer:3d} expert {expert:3d} {PROJ[k]:9s} rel L2 {err:.4f}")
    return errors


# ---------- a synthetic snapshot ----------

def write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for blob in blobs:
            f.write(blob)


def make_snapshot(root, layers=2, experts=3, hidden=192, inter=72, seed=7):
    """Per-expert FP8 experts with BF16 scale_inv, gate/up and down in two shards
    plus the index, as the release ships them. hidden 192 is one and a half FP8
    blocks and three int4 groups; inter 72 is one group plus a tail of 8."""
    rng = np.random.default_rng(seed)
    prefix = "model.language_model"
    shards = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    shards["model-00001-of-00002.safetensors"][f"{prefix}.embed_tokens.weight"] = \
        ("F32", [4, hidden], np.zeros((4, hidden), np.float32).tobytes())
    for layer in range(layers):
        for expert in range(experts):
            for k, (rows, cols) in enumerate(((inter, hidden), (inter, hidden), (hidden, inter))):
                codes = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)
                codes[(codes & 0x7F) == 0x7F] = 0x3C        # no NaN encodings
                scale = (rng.random((-(-rows // 128), -(-cols // 128))) * 0.02 + 0.001).astype(np.float32)
                bf16 = (scale.view(np.uint32) >> 16).astype(np.uint16)
                name = f"{prefix}.layers.{layer}.mlp.experts.{expert}.{PROJ[k]}.weight"
                shard = "model-00002-of-00002.safetensors" if k == 2 else "model-00001-of-00002.safetensors"
                shards[shard][name] = ("F8_E4M3", [rows, cols], codes.tobytes())
                shards[shard][name + "_scale_inv"] = ("BF16", list(scale.shape), bf16.tobytes())
    weight_map = {}
    for fname, tensors in shards.items():
        write_safetensors(root / fname, tensors)
        weight_map.update({name: fname for name in tensors})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (root / "config.json").write_text(json.dumps({"text_config": {
        "model_type": "qwen4_exp_text", "num_hidden_layers": layers, "num_experts": experts,
        "hidden_size": hidden, "moe_intermediate_size": inter}}))


def import_converter():
    sys.path.insert(0, str(TOOLS))
    import convert_qwen38_experts_int4 as converter
    return converter


@unittest.skipUnless(HAVE_NUMPY, "numpy is required")
class ConvertQwen38ExpertsInt4(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_snapshot(self.root)
        self.converter = import_converter()

    def tearDown(self):
        self.tmp.cleanup()

    def convert(self, **kw):
        from contextlib import redirect_stdout
        import io
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.converter.convert(str(self.root), workers=1, **kw)
        return buf.getvalue()

    def test_layout_quantizer_and_error(self):
        self.convert()
        index = json.loads((self.root / "experts-int4g64" / "index.json").read_text())
        self.assertTrue(index["complete"])
        self.assertEqual(index["record_bytes"], 2 * 72 * 96 + 192 * 36 + 4 * (2 * 72 * 3 + 192 * 2))
        errors = verify(self.root, quiet=True)
        self.assertEqual(len(errors), 2 * 3 * 3)
        mean = sum(errors) / len(errors)
        self.assertTrue(0.05 < mean < 0.15 and max(errors) < 0.2, (mean, max(errors)))

    def test_matches_the_engine_table(self):
        self.assertTrue(np.array_equal(self.converter.E4M3, engine_e4m3_table(), equal_nan=True))

    def test_resume(self):
        side = self.root / "experts-int4g64"
        self.convert(layers="0")
        index = json.loads((side / "index.json").read_text())
        self.assertEqual((index["done"], index["complete"]), ([0], False))
        first = (side / "layer-000.safetensors").read_bytes()
        mtime = (side / "layer-000.safetensors").stat().st_mtime_ns
        out = self.convert()
        self.assertIn("1 layer(s) already converted", out)
        self.assertEqual((side / "layer-000.safetensors").stat().st_mtime_ns, mtime)
        self.assertTrue(json.loads((side / "index.json").read_text())["complete"])
        # a layer whose file is gone is converted again, to the same bytes
        (side / "layer-000.safetensors").unlink()
        out = self.convert()
        self.assertIn("converting it again", out)
        self.assertEqual((side / "layer-000.safetensors").read_bytes(), first)
        # a sidecar started from another snapshot is not resumed
        config = json.loads((self.root / "config.json").read_text())
        config["note"] = "changed"
        (self.root / "config.json").write_text(json.dumps(config))
        with self.assertRaises(SystemExit):
            self.convert()


# ---------- engine checks on a converted fixture ----------

def engine_refusals(model_dir, engine):
    """qwen38 against edited copies of a converted snapshot."""
    def run(snap, **env):
        full = dict(os.environ, OMP_NUM_THREADS="2", SNAP=str(snap), **env)
        p = subprocess.run([engine, "2", "8", str(snap / "ref.json")], env=full,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return p.returncode, p.stderr

    def edited(edit_index=None, truncate=False):
        tmp = Path(tempfile.mkdtemp())
        snap = tmp / "snap"
        shutil.copytree(model_dir, snap)
        index_path = snap / "experts-int4g64" / "index.json"
        if edit_index:
            index = json.loads(index_path.read_text())
            edit_index(index)
            index_path.write_text(json.dumps(index))
        if truncate:
            layer = snap / "experts-int4g64" / "layer-001.safetensors"
            data = layer.read_bytes()
            with open(layer, "wb") as f:
                header = struct.unpack("<Q", data[:8])[0]
                f.write(data[:8 + header + 64])
        return tmp, snap

    failures = 0
    cases = [
        ("an index for another geometry is refused", dict(edit_index=lambda i: i.update(hidden_size=i["hidden_size"] * 2)), {}, "refusing", True),
        ("an index of another format version is refused", dict(edit_index=lambda i: i.update(version=2)), {}, "refusing", True),
        ("a layer file shorter than its header says is refused", dict(truncate=True), {}, "out of file bounds", True),
        ("an incomplete sidecar falls back to the snapshot's experts", dict(edit_index=lambda i: i.update(complete=False)), {}, "conversion incomplete", False),
        ("Q38_EXPERT_INT4=1 refuses an incomplete sidecar", dict(edit_index=lambda i: i.update(complete=False)), {"Q38_EXPERT_INT4": "1"}, "refuses to fall back", True),
    ]
    for what, edit, env, needle, must_fail in cases:
        tmp, snap = edited(**edit)
        try:
            rc, err = run(snap, **env)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        ok = needle in err and ((rc != 0) if must_fail else (rc == 0 and "native FP8" in err))
        print(f"  {'ok  ' if ok else 'FAIL'} {what}")
        failures += not ok
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="a snapshot with experts-int4g64/")
    ap.add_argument("--layers", help="comma list of layers to check (default: all)")
    ap.add_argument("--experts", help="comma list of experts to check (default: all)")
    ap.add_argument("--engine", help="qwen38 binary: also check its refusals on edited copies")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")] if a.layers else None
    experts = [int(x) for x in a.experts.split(",")] if a.experts else None
    print(f"{a.model}/experts-int4g64 against the snapshot's experts:")
    errors = verify(a.model, layers, experts)
    mean = sum(errors) / len(errors)
    print(f"  {len(errors)} matrices: rel L2 mean {mean:.4f}, max {max(errors):.4f} "
          f"(every scale absmax/7, every value within half a scale)")
    failures = 0 if 0.02 < mean < 0.2 and max(errors) < 0.25 else 1
    if failures:
        print("  FAIL relative L2 error outside what int4-g64 gives (mean 2-20%, max 25%)")
    if a.engine:
        print("qwen38 on edited copies:")
        failures += engine_refusals(Path(a.model), a.engine)
    print("test_qwen38_int4_convert: " + ("ok" if not failures else f"{failures} failure(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    unittest.main()
