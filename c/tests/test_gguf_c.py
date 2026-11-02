"""gguf.h reads GGUF files exactly as the reference reader does, and refuses bad ones.

A GGUF is written with gguf-py (the reference implementation of the format):
metadata of every scalar kind, string and numeric arrays, tensors in several
ggml types including pre-quantized blocks, and a two-part split. The C reader
must report the same keys, values, tensor shapes and bytes; reading a tensor
back through gguf_read_f32 must match gguf-py's dequantization bit for bit.
Truncated and corrupted files must be refused with a message, never crash.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:   # the CI's Python tests run without numpy
    raise unittest.SkipTest("numpy not installed: %s" % exc)

HERE = Path(__file__).resolve().parent.parent
# GQ_TEST_BIN_DIR / GQ_TEST_WRAP run the same checks on another build, e.g. an
# aarch64 one under qemu: GQ_TEST_BIN_DIR=/tmp/arm GQ_TEST_WRAP="qemu-aarch64 -cpu max"
def _bin(name):
    d = os.environ.get("GQ_TEST_BIN_DIR")
    return (Path(d) if d else HERE / "tests") / name
def _wrap(cmd):
    w = os.environ.get("GQ_TEST_WRAP")
    return (w.split() if w else []) + [str(c) for c in cmd]
BIN = _bin("test_gguf_c.exe" if sys.platform == "win32" else "test_gguf_c")
sys.path.insert(0, str(HERE / "tests"))

try:
    import gguf
    from gguf import quants as gq
    from gguf.constants import GGMLQuantizationType as QT
    from test_ggml_quants import make_blocks, TYPES
except ImportError:  # pragma: no cover
    gguf = None


def write_gguf(path, rng, tensors_only=None):
    w = gguf.GGUFWriter(str(path), "testarch")
    w.add_uint32("t.u32", 7); w.add_int32("t.i32", -5); w.add_float32("t.f32", 0.25)
    w.add_uint64("t.u64", 1 << 40); w.add_int64("t.i64", -(1 << 33)); w.add_float64("t.f64", 1e-9)
    w.add_bool("t.bool", True); w.add_uint8("t.u8", 200); w.add_int8("t.i8", -3)
    w.add_uint16("t.u16", 65000); w.add_int16("t.i16", -1234)
    w.add_string("t.str", "colibri ñ")
    w.add_array("t.strs", ["a", "Ġb", "<|im_end|>", "zz"])
    w.add_array("t.ints", [1, 2, 3, 4, 5])
    w.add_array("t.floats", [0.5, 1.5])
    out = {}
    specs = [("f32", "F32", (3, 64)), ("f16", "F16", (2, 32)), ("bf16", "BF16", (2, 32)),
             ("q8", "Q8_0", (4, 64)), ("q4k", "Q4_K", (2, 512)), ("iq2", "IQ2_XXS", (3, 256)),
             ("mx", "MXFP4", (2, 64)), ("exps", "Q5_K", (4, 2, 256))]
    for name, tname, shape in specs:
        if tensors_only is not None and name not in tensors_only:
            continue
        qt = QT[tname]
        if tname == "F32":
            a = rng.standard_normal(shape).astype(np.float32); w.add_tensor(name, a); ref = a
        elif tname == "F16":
            a = rng.standard_normal(shape).astype(np.float16); w.add_tensor(name, a); ref = a.astype(np.float32)
        elif tname == "BF16":
            raw = (rng.standard_normal(shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            w.add_tensor(name, raw.view(np.uint8).reshape(*shape[:-1], -1), raw_dtype=qt)
            ref = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            blck, size, _ = TYPES[tname]
            rows = int(np.prod(shape[:-1])); nb = shape[-1] // blck
            raw = make_blocks(tname, rows * nb, rng)
            w.add_tensor(name, raw.reshape(*shape[:-1], nb * size), raw_dtype=qt)
            ref = gq.dequantize(raw.reshape(-1), qt).astype(np.float32).reshape(shape)
        out[name] = (tname, shape, ref)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return out


def run(*args):
    return subprocess.run(_wrap([BIN, *args]), stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@unittest.skipUnless(BIN.is_file(), "tests/test_gguf_c is not built")
class GgufPathJoin(unittest.TestCase):
    """gguf.json's path: kept when absolute in either system's form, else under the model
    directory (#2189: a Windows import's "D:\\..." was joined to the directory)."""
    def test_absolute_paths_are_kept(self):
        d = "D:\\colibri\\models\\flashnext"
        for path in ("D:\\models\\m-00001-of-00003.gguf", "d:/models/m.gguf", "\\\\host\\share\\m.gguf",
                     "\\models\\m.gguf", "/srv/models/m.gguf"):
            with self.subTest(path=path):
                p = run("--join", d, path)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertEqual(p.stdout.decode().strip(), path)

    def test_relative_paths_join_the_directory(self):
        for path in ("m.gguf", "src\\m.gguf", "src/m.gguf"):
            with self.subTest(path=path):
                p = run("--join", "/models/x", path)
                self.assertEqual(p.stdout.decode().strip(), "/models/x/" + path)


@unittest.skipIf(gguf is None, "gguf-py (pip install gguf) is not installed")
@unittest.skipUnless(BIN.is_file(), "tests/test_gguf_c is not built")
class GgufReader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_metadata_and_tensors_match_the_reference(self):
        rng = np.random.default_rng(5)
        path = self.dir / "m.gguf"
        tensors = write_gguf(path, rng)
        p = run(path); self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        lines = p.stdout.decode().splitlines()
        self.assertTrue(lines[0].startswith("OK files=1"))
        kv = {l.split(" ")[1]: l.split(" ", 3)[3] for l in lines if l.startswith("KV ")}
        self.assertEqual(kv["t.u32"], "i:7"); self.assertEqual(kv["t.i32"], "i:-5")
        self.assertEqual(kv["t.u64"], f"i:{1 << 40}"); self.assertEqual(kv["t.i64"], f"i:{-(1 << 33)}")
        self.assertEqual(kv["t.f32"], "f:0.25"); self.assertEqual(kv["t.bool"], "i:1")
        self.assertEqual(kv["t.u8"], "i:200"); self.assertEqual(kv["t.i8"], "i:-3")
        self.assertEqual(kv["t.u16"], "i:65000"); self.assertEqual(kv["t.i16"], "i:-1234")
        self.assertEqual(kv["t.str"], "s:colibri ñ")
        self.assertEqual(kv["t.strs"], "a:8:4:a:zz")
        self.assertTrue(kv["t.ints"].startswith("a:") and kv["t.ints"].endswith(":5"))
        ref = gguf.GGUFReader(str(path))
        tl = {l.split(" ")[1]: l.split(" ")[2:] for l in lines if l.startswith("T ")}
        for t in ref.tensors:
            self.assertIn(t.name, tl)
            f = tl[t.name]
            self.assertEqual(f[0], t.tensor_type.name)
            self.assertEqual([int(x) for x in f[1:1 + len(t.shape)]], [int(x) for x in t.shape])
            self.assertEqual(int(f[-1].split("=")[1]), int(t.n_bytes))
        for name, (tname, shape, want) in tensors.items():
            with self.subTest(tensor=name, type=tname):
                p = run(path, name); self.assertEqual(p.returncode, 0, p.stderr)
                got = np.frombuffer(p.stdout, dtype=np.float32)
                self.assertTrue(np.array_equal(got.view(np.uint32), want.reshape(-1).view(np.uint32)))

    def test_split_model_opens_every_part(self):
        rng = np.random.default_rng(6)
        a = write_gguf(self.dir / "m-00001-of-00002.gguf", rng, tensors_only={"f32", "q8"})
        b = write_gguf(self.dir / "m-00002-of-00002.gguf", rng, tensors_only={"q4k", "exps"})
        p = run(self.dir / "m-00001-of-00002.gguf"); self.assertEqual(p.returncode, 0, p.stdout)
        out = p.stdout.decode()
        self.assertIn("files=2", out); self.assertIn("tensors=4", out)
        self.assertIn("T exps Q5_K 256 2 4 1 part=1", out)
        for name, (_, _, want) in list(a.items()) + list(b.items()):
            got = np.frombuffer(run(self.dir / "m-00001-of-00002.gguf", name).stdout, dtype=np.float32)
            self.assertTrue(np.array_equal(got.view(np.uint32), want.reshape(-1).view(np.uint32)), name)

    def test_malformed_files_are_refused(self):
        rng = np.random.default_rng(8)
        good = self.dir / "g.gguf"; write_gguf(good, rng)
        data = good.read_bytes()
        cases = {
            "not gguf": b"GGUX" + data[4:],
            "truncated header": data[:200],
            "truncated data": data[:-100],
            "huge tensor count": data[:8] + (1 << 50).to_bytes(8, "little") + data[16:],
            "huge kv count": data[:16] + (1 << 40).to_bytes(8, "little") + data[24:],
            "bad version": data[:4] + (9).to_bytes(4, "little") + data[8:],
        }
        for label, blob in cases.items():
            with self.subTest(case=label):
                f = self.dir / "bad.gguf"; f.write_bytes(blob)
                p = run(f)
                self.assertNotEqual(p.returncode, 0, label)
                self.assertTrue(p.stdout.startswith(b"ERR "), (label, p.stdout[:200]))
                self.assertGreaterEqual(p.returncode, 0, f"{label}: killed by a signal")
        # a random-byte flip anywhere in the header must never crash
        for seed in range(40):
            r = np.random.default_rng(seed)
            blob = bytearray(data)
            for _ in range(8):
                blob[int(r.integers(8, 4000))] = int(r.integers(0, 256))
            f = self.dir / "fuzz.gguf"; f.write_bytes(bytes(blob))
            p = run(f)
            self.assertGreaterEqual(p.returncode, 0, f"seed {seed}: crashed")


if __name__ == "__main__":
    unittest.main()
