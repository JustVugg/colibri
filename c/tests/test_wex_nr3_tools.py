"""tools/build_nr3.py and tools/wex_nr3_tools.c (the WNR3 encoder) on a synthetic WEX1 container.

The library is built with the C compiler found ($CC, else cc or gcc; the test is skipped without
one). A WEX1 file of 2 layers x 3 MXFP4 experts (dim 256, moe_intermediate 64: 26,112-byte
records) goes through `build_nr3.py ... container`, plain and with --cz, and through `bins` with
--save-model. Checked: the WEX3 header the tool writes, every record WNR3 and smaller than its
RAW page, every record decoded by the library's CPU decoder back to the source (its canonical-
zero form for --cz), a changed byte refused, and the saved model the one the container names."""
import ctypes
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.dirname(HERE)
TOOL = os.path.join(C_DIR, 'tools', 'build_nr3.py')
SOURCE = os.path.join(C_DIR, 'tools', 'wex_nr3_tools.c')
HEADER = struct.Struct('<IIIIQQII')
ENTRY = struct.Struct('<QIII6H6H3I3II')
ALIGN, MODEL_BYTES = 4096, 59408
LAYERS, EXPERTS, DIM, INTER = 2, 3, 256, 64


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def read(path):
    with open(path, 'rb') as f:
        return f.read()


def compiler():
    for name in (os.environ.get('CC'), 'cc', 'gcc'):
        if name and shutil.which(name):
            return shutil.which(name)
    return None


def geometry():
    """(shapes, offsets) of one expert in the slab: w1 [I][D], w2 [D][I], w3 [I][D] (gate, down, up)."""
    rows, cols = (INTER, DIM, INTER), (DIM, INTER, DIM)
    scale = [rows[m] * cols[m] // 32 for m in range(3)]
    weight = [rows[m] * cols[m] // 2 for m in range(3)]
    s_off = [sum(scale[:m]) for m in range(3)]
    w_off = [sum(scale) + sum(weight[:m]) for m in range(3)]
    return rows, cols, scale, weight, s_off, w_off


def expert(seed, negzero):
    """An MXFP4 expert's bytes: scales near 2^-7, codes mostly small; negzero: some -0 codes."""
    rng = random.Random(seed)
    rows, cols, scale, weight, s_off, w_off = geometry()
    out = bytearray(sum(scale) + sum(weight))
    for i in range(sum(scale)):
        out[i] = 118 + rng.randrange(5)
    magnitude = (0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7)
    for i in range(sum(scale), len(out)):
        lo, hi = magnitude[rng.randrange(16)], magnitude[rng.randrange(16)]
        lo |= rng.choice((0, 8)) if lo else (8 if negzero and rng.randrange(50) == 0 else 0)
        hi |= rng.choice((0, 8)) if hi else 0
        out[i] = lo | hi << 4
    return bytes(out)


def write_wex1(path, records):
    rows, cols, scale, weight, s_off, w_off = geometry()
    raw = sum(scale) + sum(weight)
    table = align(LAYERS * EXPERTS * ENTRY.size)
    at = ALIGN + table
    entries, data = [], []
    for rec in records:
        shapes = []
        for m in range(3):
            shapes += [rows[m], cols[m] // 32]
        wshapes = []
        for m in range(3):
            wshapes += [rows[m], cols[m] // 2]
        entries.append(ENTRY.pack(at, align(raw), sum(scale), sum(weight), *shapes, *wshapes, *s_off, *w_off, 0))
        data.append(rec.ljust(align(raw), b'\0'))
        at += align(raw)
    with open(path, 'wb') as f:
        f.write(HEADER.pack(0x57455831, 1, LAYERS, EXPERTS, align(raw), ALIGN, 0, 0).ljust(ALIGN, b'\0'))
        f.write(b''.join(entries).ljust(table, b'\0'))
        f.write(b''.join(data))


@unittest.skipUnless(compiler(), 'no C compiler (CC, cc or gcc) to build tools/wex_nr3_tools.c')
class WexNr3ToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix='wex-nr3-tools-')
        cls.lib = os.path.join(cls.dir, 'wex_nr3_tools.dll' if os.name == 'nt' else 'libwex_nr3_tools.so')
        subprocess.run([compiler(), '-O2', '-shared', '-fPIC', SOURCE, '-o', cls.lib], check=True)
        cls.records = [expert(20261116 + i, negzero=(i == 4)) for i in range(LAYERS * EXPERTS)]
        cls.wex1 = os.path.join(cls.dir, 'experts.wex')
        write_wex1(cls.wex1, cls.records)
        cls.tools = ctypes.CDLL(cls.lib)
        P = ctypes.c_void_p
        cls.tools.wex_nr3_verify.argtypes = [P, P, P, ctypes.c_uint32, ctypes.c_uint32, P]
        cls.tools.wex_nr3_canonicalize.argtypes = [P, P]
        cls.tools.wex_nr3_canonicalize.restype = ctypes.c_uint64
        cls.tools.wex_nr3_model_id.argtypes = [P, P]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def run_tool(self, *args):
        done = subprocess.run([sys.executable, '-I', TOOL, self.lib, self.wex1, *args],
                              capture_output=True, text=True, cwd=self.dir)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout

    def check_container(self, cz):
        out = os.path.join(self.dir, 'experts%s.nr3.wex' % ('-cz' if cz else ''))
        self.run_tool('container', out, '--layers', 'all', '--train', '0:0,1:2', *(['--cz'] if cz else []))
        blob = read(out)
        magic, version, layers, experts, slab, table_offset, flags, _ = HEADER.unpack_from(blob)
        self.assertEqual((magic, version, layers, experts), (0x57455833, 3, LAYERS, EXPERTS))
        self.assertEqual(flags, 1 | 32 | (64 if cz else 0))   # scale tiles, WNR3, canonical zero
        model = blob[ALIGN:ALIGN + MODEL_BYTES]
        changed = 0
        for cell in range(LAYERS * EXPERTS):
            fields = list(ENTRY.unpack_from(blob, table_offset + cell * ENTRY.size))
            offset, total, raw = fields[0], fields[1], fields[2] + fields[3]
            self.assertEqual(fields[-1], 5, f'cell {cell}: not a WNR3 record')
            self.assertLess(total, align(raw))
            expected = ctypes.create_string_buffer(self.records[cell], raw)
            entry = ENTRY.pack(*fields)
            if cz:
                changed += self.tools.wex_nr3_canonicalize(entry, expected)
            record = blob[offset:offset + total]
            layer, e = divmod(cell, EXPERTS)
            self.assertEqual(self.tools.wex_nr3_verify(model, entry, record, layer, e, expected), 0,
                             f'cell {cell}: does not decode to the source')
            self.assertNotEqual(self.tools.wex_nr3_verify(model, entry, record, layer, (e + 1) % EXPERTS, expected), 0,
                                f'cell {cell}: accepted under another expert')
            bad = bytearray(record)
            bad[200] ^= 0x10
            self.assertNotEqual(self.tools.wex_nr3_verify(model, entry, bytes(bad), layer, e, expected), 0,
                                f'cell {cell}: a changed byte decoded')
        self.assertEqual(changed > 0, cz)   # expert 4 has -0 codes: --cz stored them as +0
        return model

    def test_container_plain(self):
        self.check_container(False)

    def test_container_canonical_zero(self):
        self.check_container(True)

    def test_bins_and_saved_model(self):
        saved = os.path.join(self.dir, 'model.wnm2')
        out = os.path.join(self.dir, 'records.bin')
        self.run_tool('bins', out, '--experts', '0:1,1:0', '--train', '0:0,1:2', '--save-model', saved)
        model = read(saved)
        self.assertEqual(len(model), MODEL_BYTES)
        mid = ctypes.c_uint32()
        self.assertEqual(self.tools.wex_nr3_model_id(model, ctypes.byref(mid)), 0)
        self.assertEqual(read(out)[4:4 + MODEL_BYTES], model)
        self.assertEqual(self.check_container(False), model)   # the same training picks: the same model


class BuildNr3PicksTest(unittest.TestCase):
    """--train-lab80 picks: unchanged on V4.1 (40 x 192), every layer on Qwen3.8 (48 x 512)."""
    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location('build_nr3', TOOL)
        cls.build_nr3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.build_nr3)

    def test_v41_picks_unchanged(self):
        rng = random.Random(20261107)
        old = [(L, e) for L in range(40) for e in sorted(rng.sample(range(192), 4))]
        self.assertEqual(self.build_nr3.lab80(40, 192), [old[i] for i in range(0, 160, 2)])

    def test_qwen38_picks_cover_every_layer(self):
        q = self.build_nr3.lab80(48, 512)
        self.assertEqual(len(set(q)), 80, q)
        self.assertEqual({L for L, _ in q}, set(range(48)), q)


if __name__ == '__main__':
    unittest.main()
