"""int4-rans256-g0 offline-tool e2e: repack_rans.py + rans_verify.py against
synthetic per-row-int4 fixtures.

Covers the tool-level contract the C oracle (test_rans.c) cannot see:
  - end-to-end lossless round trip through real shard files (original weight
    bytes reproduced exactly; .qs sidecars byte-identical pass-through);
  - writer determinism (two runs diff clean, including a C-codec run vs a
    forced pure-Python run — the two implementations must emit identical
    bytes, not just equivalent ones);
  - the MANDATORY stamp + per-shard table metadata shape;
  - the verifier's named-refusal battery: every corruption class produces
    its class, enumerated not sampled;
  - writer-side geometry refusals (g64-signature and missing-scales inputs
    are refused by name, never silently entropy-coded).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

np = None
rf = None
try:
    import numpy as np

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import rans_format as rf
except ImportError:  # pragma: no cover - dependency-free CI skips these
    pass

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def run_tool(script, *args, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable, str(TOOLS / script), *args],
                          capture_output=True, text=True, env=env)


@unittest.skipIf(rf is None, "numpy not available (offline-tooling test)")
class TestRansRepack(unittest.TestCase):
    O, I = 8, 256           # per-row ratio I/2 = 128, above every grouped
                            # signature (32 = g64, 64 = g128/ambiguous)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.indir = self.root / "int4"
        self.indir.mkdir()
        rng = np.random.default_rng(42)
        self.weights = {}
        tensors_a, tensors_b = [], []
        wb = self.O * self.I // 2
        for e in range(3):
            for proj in ("gate", "up", "down"):
                name = f"model.layers.0.mlp.experts.{e}.{proj}_proj.weight"
                # skewed 15-symbol nibbles (value 0 never used, like the corpus)
                nib = rng.choice(
                    np.arange(1, 16, dtype=np.uint8), size=wb * 2,
                    p=np.array([30, 5, 4, 3, 3, 2, 2, 20, 10, 6, 5, 4, 3, 2, 1],
                               dtype=np.float64) / 100.0)
                raw = rf.pack_nibbles(nib)
                qs = rng.standard_normal(self.O).astype(np.float32).tobytes()
                self.weights[name] = (raw, qs)
                tensors_a.append((name, "U8", [wb], raw))
                # one expert's sidecars live in a SECOND shard: the tool must
                # find them through its corpus-wide index
                (tensors_b if e == 2 else tensors_a).append(
                    (name + ".qs", "F32", [self.O], qs))
        tensors_a.append(("model.norm.weight", "F32", [4],
                          np.ones(4, dtype=np.float32).tobytes()))
        rf.write_shard(str(self.indir / "out-00000.safetensors"), tensors_a, {})
        rf.write_shard(str(self.indir / "out-00001.safetensors"), tensors_b, {})

    def tearDown(self):
        self.tmp.cleanup()

    def repack(self, outdir, **env):
        r = run_tool("repack_rans.py", "--indir", str(self.indir),
                     "--outdir", str(outdir), env_extra=env or None)
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        return sorted(Path(outdir).glob("*.safetensors"))

    def test_roundtrip_stamp_and_determinism(self):
        out1 = self.repack(self.root / "out1")
        self.assertEqual(len(out1), 1)      # only shard A holds weight tensors
        header, data_start = rf.read_header(str(out1[0]))
        meta = header["__metadata__"]

        # MANDATORY stamp + table metadata
        fmt_map = json.loads(meta[rf.METADATA_KEY])
        self.assertEqual(sorted(fmt_map), sorted(self.weights))
        self.assertTrue(all(v == rf.FORMAT_NAME for v in fmt_map.values()))
        table = rf.parse_table_blob(meta[rf.TABLE_KEY])
        self.assertEqual(table["table_id"], rf.TABLE_ID)
        self.assertEqual(int(table["freq"][0]), 0)   # 15-symbol corpus shape

        # byte-exact round trip + raw .qs pass-through
        for name, (raw, qs) in self.weights.items():
            blob, _ = rf.read_tensor_bytes(str(out1[0]), header, data_start, name)
            self.assertEqual(rf.decode_record(blob, table), raw, name)
            qblob, _ = rf.read_tensor_bytes(str(out1[0]), header, data_start,
                                            name + ".qs")
            self.assertEqual(qblob, qs, name + ".qs")

        # determinism: an independent second run is byte-identical, manifest
        # (the completion marker) included
        out2 = self.repack(self.root / "out2")
        self.assertEqual(out1[0].read_bytes(), out2[0].read_bytes())
        self.assertEqual(
            (out1[0].parent / "repack-manifest.json").read_bytes(),
            (out2[0].parent / "repack-manifest.json").read_bytes())

        # implementation identity: a forced pure-Python run emits the SAME
        # bytes as the C-bridge run. VACUOUS unless the bridge is actually
        # loaded, so that is asserted — `make test-python` builds the bridge;
        # a direct unittest invocation without it skips LOUDLY instead of
        # green-lighting a pin it never checked.
        if rf.LIB is None:
            self.skipTest("tools/librans_c not built (run `make rans`): "
                          "the C-vs-Python byte-identity pin was NOT checked")
        out3 = self.repack(self.root / "out3", RANS_FORCE_PYTHON="1")
        self.assertEqual(out1[0].read_bytes(), out3[0].read_bytes())

        # bridge-presence: the default run must actually have used the C
        # codec (not fallen back to Python with the lib sitting there)
        r = run_tool("repack_rans.py", "--indir", str(self.indir),
                     "--outdir", str(self.root / "outdry"), "--dry-run")
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        self.assertIn("codec: C (librans_c)", r.stdout)

        # the validator trusts it
        r = run_tool("rans_verify.py", str(out1[0]))
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        self.assertIn("RESULT: TRUST", r.stdout)

    def _reshard(self, shard, meta_mutator=None, tensor_mutator=None):
        """Read a shard fully, apply mutators, rewrite it deterministically.
        Drops the manifest sitting next to it: the battery tests the
        PER-RECORD layer, which must hold even when an attacker removes the
        whole-artifact manifest (digest tampering has its own test)."""
        manifest = shard.parent / "repack-manifest.json"
        if manifest.exists():
            manifest.unlink()
        header, data_start = rf.read_header(str(shard))
        meta = dict(header["__metadata__"])
        tensors = []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            blob, _ = rf.read_tensor_bytes(str(shard), header, data_start, name)
            if tensor_mutator:
                blob = tensor_mutator(name, bytearray(blob))
            tensors.append((name, info["dtype"], info["shape"], bytes(blob)))
        if meta_mutator:
            meta = meta_mutator(meta)
        rf.write_shard(str(shard), tensors, meta)

    _corrupt_seq = 0

    def _corrupt_reshard_and_expect(self, expected_code, meta_mutator=None,
                                    tensor_mutator=None):
        TestRansRepack._corrupt_seq += 1
        outdir = self.root / f"corrupt_{TestRansRepack._corrupt_seq}_{expected_code}"
        shard = self.repack(outdir)[0]
        self._reshard(shard, meta_mutator, tensor_mutator)
        r = run_tool("rans_verify.py", str(shard))
        self.assertEqual(r.returncode, 1,
                         msg=f"{expected_code}: verifier did not refuse\n"
                             f"{r.stdout}{r.stderr}")
        self.assertIn(expected_code, r.stdout,
                      msg=f"wanted {expected_code} in:\n{r.stdout}")

    def test_refusal_battery(self):
        target = "model.layers.0.mlp.experts.0.gate_proj.weight"

        def flip_payload(name, blob):
            if name == target:
                blob[rf.record_header_bytes() + 5] ^= 0x10   # payload corruption
            return blob

        def flip_count(name, blob):
            if name == target:
                blob[8] ^= 1                                  # packed_bytes field
            return blob

        def break_offsets(name, blob):
            if name == target:
                blob[16 + 8 * 4] ^= 0x80                      # offsets landmine
            return blob

        def truncate(name, blob):
            if name == target:
                del blob[len(blob) - 16:]                     # drop final 16 bytes
            return blob

        def pad_dirt(name, blob):
            if name == target:
                blob[16 + (rf.N_STREAMS + 1) * 4] = 0x77      # header pad byte
            return blob

        def bomb_header(name, blob):
            if name == target:
                # n_symbols = 2^40 with a consistent packed_bytes: the
                # "oversize fields" class the spec names — must refuse as
                # E_OVERSIZE, never an allocator traceback
                blob[0:8] = (1 << 40).to_bytes(8, "little")
                blob[8:16] = (1 << 39).to_bytes(8, "little")
            return blob

        def drop_stamp(meta):
            del meta[rf.METADATA_KEY]
            return meta

        def drop_table(meta):
            del meta[rf.TABLE_KEY]
            return meta

        def break_crc(meta):
            blob = json.loads(meta[rf.TABLE_KEY])
            blob["table_crc32"] = "00000000"
            meta[rf.TABLE_KEY] = json.dumps(blob, sort_keys=True)
            return meta

        def break_freq(meta):
            blob = json.loads(meta[rf.TABLE_KEY])
            blob["freq"][15] += 1
            meta[rf.TABLE_KEY] = json.dumps(blob, sort_keys=True)
            return meta

        def dangling_stamp(meta):
            fmt = json.loads(meta[rf.METADATA_KEY])
            fmt["model.layers.0.mlp.experts.99.gate_proj.weight"] = rf.FORMAT_NAME
            meta[rf.METADATA_KEY] = json.dumps(fmt, sort_keys=True)
            return meta

        # payload corruption trips a stream invariant or the re-encode pin;
        # both are refusals — assert on the shared prefix
        self._corrupt_reshard_and_expect("E_", tensor_mutator=flip_payload)
        self._corrupt_reshard_and_expect("E_OVERSIZE", tensor_mutator=bomb_header)
        self._corrupt_reshard_and_expect("E_COUNT_MISMATCH",
                                         tensor_mutator=flip_count)
        self._corrupt_reshard_and_expect("E_", tensor_mutator=break_offsets)
        self._corrupt_reshard_and_expect("E_", tensor_mutator=truncate)
        self._corrupt_reshard_and_expect("E_PAD_NONZERO", tensor_mutator=pad_dirt)
        self._corrupt_reshard_and_expect("E_STAMP_MISSING", meta_mutator=drop_stamp)
        self._corrupt_reshard_and_expect("E_TABLE_MISSING", meta_mutator=drop_table)
        self._corrupt_reshard_and_expect("E_TABLE_CRC", meta_mutator=break_crc)
        self._corrupt_reshard_and_expect("E_TABLE_FREQ_SUM", meta_mutator=break_freq)
        self._corrupt_reshard_and_expect("E_STAMP_DANGLING",
                                         meta_mutator=dangling_stamp)

    def test_geometry_refusals(self):
        # g64 signature: weight/scale ratio exactly 32 -> named refusal
        g64dir = self.root / "g64"
        g64dir.mkdir()
        wb = 2048
        rows = wb // 32
        rf.write_shard(str(g64dir / "out-00000.safetensors"), [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", "U8", [wb],
             bytes(wb)),
            ("model.layers.0.mlp.experts.0.gate_proj.weight.qs", "F32", [rows],
             bytes(rows * 4)),
        ], {})
        r = run_tool("repack_rans.py", "--indir", str(g64dir),
                     "--outdir", str(self.root / "g64out"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("E_GEOMETRY_G64", r.stderr)

        # gs=128 signature (upstream grouped-int4's actual group size):
        # ratio exactly 64, byte-indistinguishable from per-row I=128 ->
        # its own named refusal, never silently accepted as per-row
        g128dir = self.root / "g128"
        g128dir.mkdir()
        rows128 = wb // 64
        rf.write_shard(str(g128dir / "out-00000.safetensors"), [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", "U8", [wb],
             bytes(wb)),
            ("model.layers.0.mlp.experts.0.gate_proj.weight.qs", "F32",
             [rows128], bytes(rows128 * 4)),
        ], {})
        r = run_tool("repack_rans.py", "--indir", str(g128dir),
                     "--outdir", str(self.root / "g128out"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("E_GEOMETRY_AMBIGUOUS", r.stderr)

        # 2-D weight shape: per-row geometry is verified POSITIVELY from the
        # shape fields — accepted when qs rows == O, refused by name when not
        rng = np.random.default_rng(3)
        O, rb2 = 8, 64
        nib = rng.choice(np.arange(1, 16, dtype=np.uint8), size=O * rb2 * 2)
        raw = rf.pack_nibbles(nib)
        for rows2d, expect_ok in ((O, True), (O * 2, False)):
            d2 = self.root / f"twod_{rows2d}"
            d2.mkdir()
            rf.write_shard(str(d2 / "out-00000.safetensors"), [
                ("model.layers.0.mlp.experts.0.gate_proj.weight", "U8",
                 [O, rb2], raw),
                ("model.layers.0.mlp.experts.0.gate_proj.weight.qs", "F32",
                 [rows2d], bytes(rows2d * 4)),
            ], {})
            r = run_tool("repack_rans.py", "--indir", str(d2),
                         "--outdir", str(self.root / f"twod_out_{rows2d}"))
            if expect_ok:
                self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
            else:
                self.assertEqual(r.returncode, 1)
                self.assertIn("E_GEOMETRY_NOT_PER_ROW", r.stderr)

        # missing .qs sidecar -> named refusal
        nsdir = self.root / "noscales"
        nsdir.mkdir()
        rf.write_shard(str(nsdir / "out-00000.safetensors"), [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", "U8", [64],
             bytes(64)),
        ], {})
        r = run_tool("repack_rans.py", "--indir", str(nsdir),
                     "--outdir", str(self.root / "nsout"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("E_GEOMETRY_NO_SCALES", r.stderr)

        # nothing selected -> loud error, no silent empty container
        emptydir = self.root / "empty"
        emptydir.mkdir()
        rf.write_shard(str(emptydir / "out-00000.safetensors"), [
            ("model.norm.weight", "F32", [4], bytes(16)),
        ], {})
        r = run_tool("repack_rans.py", "--indir", str(emptydir),
                     "--outdir", str(self.root / "emptyout"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("refusing to emit an empty container", r.stderr)

    def test_parity_c_python_refusal_classes(self):
        """Same bytes into the C parser (via the bridge) and the Python
        parser must produce the same named class — parity is a tested
        property, not a documented intention. Covers the u64-wrap and
        decompression-bomb header battery."""
        if rf.LIB is None:
            self.skipTest("tools/librans_c not built (run `make rans`): "
                          "C/Python refusal parity was NOT checked")
        import ctypes

        def c_parse(blob):
            buf = np.zeros(len(blob) + rf.SLACK, dtype=np.uint8)
            buf[:len(blob)] = np.frombuffer(blob, dtype=np.uint8)
            ns = ctypes.c_uint64()
            pb = ctypes.c_uint64()
            rc = rf.LIB.rc_record_parse(rf._p8(buf), len(blob), rf.N_STREAMS,
                                        ctypes.byref(ns), ctypes.byref(pb))
            return "OK" if rc == 0 else rf.LIB.rc_err_name(rc).decode()

        def py_parse(blob):
            try:
                rf.parse_record(blob)
                return "OK"
            except rf.RansRefusal as exc:
                return exc.code

        rng = np.random.default_rng(11)
        n = 4096
        nib = rng.choice(np.arange(1, 16, dtype=np.uint8), size=n)
        hist = np.bincount(nib, minlength=16)
        freq = rf.quantize_freq(hist)
        start, slot = rf.build_table(freq)
        base = bytearray(rf.build_record(nib, freq, start, slot))
        u64max = (1 << 64) - 1

        def hdr(ns, pb):
            b = bytearray(base)
            b[0:8] = ns.to_bytes(8, "little")
            b[8:16] = pb.to_bytes(8, "little")
            return bytes(b)

        cases = [bytes(base)]                          # the valid control
        for ns, pb in ((u64max, 0), (u64max, 1 << 63), (0, 0),
                       (1 << 40, 1 << 39), (1 << 32, 1 << 31),
                       (n + 1, (n + 1) // 2), (n, n)):
            cases.append(hdr(ns, pb))
        cases.append(bytes(base[:64]))                 # truncated header
        cases.append(bytes(base[:len(base) - 16]))     # truncated payload
        cases.append(bytes(base) + b"\x00" * 16)       # trailing pad-shaped
        cases.append(bytes(base) + b"\x77" * 16)       # trailing garbage
        mono = bytearray(base)
        mono[16 + 8 * 4] ^= 0x80                       # offsets landmine
        cases.append(bytes(mono))
        dirty = bytearray(base)
        dirty[16 + (rf.N_STREAMS + 1) * 4] = 0x55      # header pad byte
        cases.append(bytes(dirty))

        for i, blob in enumerate(cases):
            c_cls, py_cls = c_parse(blob), py_parse(blob)
            self.assertEqual(c_cls, py_cls,
                             msg=f"case {i}: C={c_cls} Python={py_cls}")
        self.assertEqual(c_parse(cases[0]), "OK")      # control really is OK

    def test_verifier_never_crashes(self):
        """Hostile safetensors headers (the review round's traceback
        classes): zero tracebacks, nonzero exit, a named REFUSE line each."""
        name = "model.layers.0.mlp.experts.0.gate_proj.weight"
        ok_entry = {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}
        stamp = json.dumps({name: rf.FORMAT_NAME})
        cases = {
            "meta_not_string": {"__metadata__": {rf.METADATA_KEY: 5},
                                name: dict(ok_entry)},
            "header_array": [1, 2, 3],
            "no_data_offsets": {"__metadata__": {rf.METADATA_KEY: stamp,
                                                 rf.TABLE_KEY: "{}"},
                                name: {"dtype": "U8", "shape": [4]}},
            "entry_not_dict": {"__metadata__": {rf.METADATA_KEY: stamp,
                                                rf.TABLE_KEY: "{}"},
                               name: "hello"},
            "fmt_value_list": {"__metadata__":
                               {rf.METADATA_KEY: json.dumps({name: ["a"]})},
                               name: dict(ok_entry)},
            "reversed_offsets": {"__metadata__": {rf.METADATA_KEY: stamp,
                                                  rf.TABLE_KEY: "{}"},
                                 name: {"dtype": "U8", "shape": [4],
                                        "data_offsets": [8, 2]}},
            "offsets_past_eof": {"__metadata__": {rf.METADATA_KEY: stamp,
                                                  rf.TABLE_KEY: "{}"},
                                 name: {"dtype": "U8", "shape": [4],
                                        "data_offsets": [0, 4096]}},
            "offsets_not_pair": {"__metadata__": {rf.METADATA_KEY: stamp,
                                                  rf.TABLE_KEY: "{}"},
                                 name: {"dtype": "U8", "shape": [4],
                                        "data_offsets": [1, 2, 3]}},
        }
        hostile = self.root / "hostile"
        hostile.mkdir()
        for label, hdr in cases.items():
            p = hostile / f"h_{label}.safetensors"
            hj = json.dumps(hdr, separators=(",", ":")).encode()
            with open(p, "wb") as f:
                f.write(len(hj).to_bytes(8, "little"))
                f.write(hj)
                f.write(b"\x00" * 32)
            r = run_tool("rans_verify.py", str(p))
            self.assertNotIn("Traceback", r.stderr,
                             msg=f"{label}: crashed\n{r.stderr}")
            self.assertEqual(r.returncode, 1, msg=f"{label}: {r.stdout}")
            self.assertIn("REFUSE", r.stdout, msg=f"{label}: no REFUSE line")

    def test_partial_output_refusal(self):
        """An outdir already holding repack output is refused (crash
        evidence: an interrupted set must not be papered over); --force
        redoes it; a completed run leaves repack-manifest.json."""
        out = self.root / "partial"
        shards = self.repack(out)
        manifest = out / "repack-manifest.json"
        self.assertTrue(manifest.exists())
        man = json.loads(manifest.read_text())
        self.assertEqual(man["n_shards"], len(shards))

        # rerun over completed output: refused
        r = run_tool("repack_rans.py", "--indir", str(self.indir),
                     "--outdir", str(out))
        self.assertEqual(r.returncode, 1)
        self.assertIn("refusing to overwrite", r.stderr)

        # simulate a crashed run: shards present, no completion manifest
        manifest.unlink()
        r = run_tool("repack_rans.py", "--indir", str(self.indir),
                     "--outdir", str(out))
        self.assertEqual(r.returncode, 1)
        self.assertIn("NO manifest", r.stderr)

        # --force redoes from scratch, byte-identical to a fresh run
        r = run_tool("repack_rans.py", "--indir", str(self.indir),
                     "--outdir", str(out), "--force")
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        self.assertTrue(manifest.exists())
        fresh = self.repack(self.root / "partial_fresh")
        self.assertEqual(shards[0].read_bytes(), fresh[0].read_bytes())

    def test_manifest_digests(self):
        """Whole-artifact digest layer: mint-time digests verify green; a
        tampered record bites as E_DIGEST_MISMATCH; retro --manifest-only
        reproduces the mint manifest byte-for-byte; a digest-less manifest
        (older mint) means note-and-proceed, never an error."""
        import hashlib
        out = self.root / "digests"
        shard = self.repack(out)[0]
        manifest = out / "repack-manifest.json"
        man = json.loads(manifest.read_text())

        # every weight record has a digest, and it matches the bytes on disk;
        # the whole-file sha256 matches the complete shard bytes
        header, data_start = rf.read_header(str(shard))
        recs = man["shards"][0]["records"]
        self.assertEqual(sorted(recs), sorted(self.weights))
        for name, want in recs.items():
            blob, _ = rf.read_tensor_bytes(str(shard), header, data_start, name)
            self.assertEqual(hashlib.sha256(blob).hexdigest(), want, name)
        self.assertEqual(man["shards"][0]["sha256"],
                         hashlib.sha256(shard.read_bytes()).hexdigest())

        # verify green, and it says both digest layers were actually checked
        r = run_tool("rans_verify.py", str(shard))
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        self.assertIn("file digest checked", r.stdout)
        self.assertIn("record digests checked", r.stdout)

        # retro equivalence: --manifest-only over the minted dir must
        # reproduce the manifest byte-for-byte (fields preserved, digests
        # recomputed identical)
        mint_bytes = manifest.read_bytes()
        r = run_tool("repack_rans.py", "--manifest-only", str(out))
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        self.assertEqual(manifest.read_bytes(), mint_bytes)

        # retro from scratch (manifest deleted): digests must equal the
        # mint-time ones; provenance-only fields (source) are unknowable
        manifest.unlink()
        r = run_tool("repack_rans.py", "--manifest-only", str(out))
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        # provenance loss is visible in output, not just by field absence
        self.assertIn("provenance is unknowable in retro mode", r.stderr)
        man2 = json.loads(manifest.read_text())
        self.assertEqual(man2["shards"][0]["records"], recs)
        self.assertEqual(man2["shards"][0]["sha256"], man["shards"][0]["sha256"])
        self.assertEqual(man2["table_crc32"], man["table_crc32"])
        r = run_tool("rans_verify.py", str(shard))
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)

        # tamper bite A: flip one payload byte of a digested record IN THE
        # SHARD FILE (manifest untouched) -> the whole-file gate fires AND
        # the per-record layer localizes it to the tensor
        target = sorted(recs)[0]
        b, e = header[target]["data_offsets"]
        raw = bytearray(shard.read_bytes())
        raw[data_start + b + rf.record_header_bytes() + 3] ^= 0x40
        shard.write_bytes(bytes(raw))
        r = run_tool("rans_verify.py", str(shard))
        self.assertEqual(r.returncode, 1, msg=r.stdout)
        self.assertIn("E_SHARD_DIGEST_MISMATCH", r.stdout)
        self.assertIn("E_DIGEST_MISMATCH", r.stdout)

        # tamper bite B — the exact gap the whole-file hash closes: flip a
        # byte inside a .qs SIDECAR (raw passthrough; not covered by any
        # record digest or record invariant). Only the file-level gate can
        # see it: E_SHARD_DIGEST_MISMATCH fires, and NO record-level digest
        # refusal appears (every record is intact)
        raw[data_start + b + rf.record_header_bytes() + 3] ^= 0x40  # restore
        qs_b, qs_e = header[target + ".qs"]["data_offsets"]
        raw[data_start + qs_b + 1] ^= 0x01
        shard.write_bytes(bytes(raw))
        r = run_tool("rans_verify.py", str(shard))
        self.assertEqual(r.returncode, 1, msg=r.stdout)
        self.assertIn("E_SHARD_DIGEST_MISMATCH", r.stdout)
        self.assertNotIn("E_DIGEST_MISMATCH:", r.stdout.replace(
            "E_SHARD_DIGEST_MISMATCH:", ""))
        raw[data_start + qs_b + 1] ^= 0x01                          # restore
        shard.write_bytes(bytes(raw))

        # backward compat: a manifest WITHOUT digest evidence (older mint)
        # is note-and-proceed
        for entry in man["shards"]:
            del entry["records"]
            del entry["sha256"]
        manifest.write_text(json.dumps(man, indent=1, sort_keys=True))
        r = run_tool("rans_verify.py", str(shard))
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        self.assertIn("carries no digests", r.stdout)
        self.assertIn("file digest unavailable", r.stdout)
        self.assertIn("record digests unavailable", r.stdout)

        # degraded mid-state: file sha present, records absent -> the file
        # gate still runs (green here), record layer notes-and-skips
        for entry, src in zip(man["shards"], man2["shards"]):
            entry["sha256"] = src["sha256"]
        manifest.write_text(json.dumps(man, indent=1, sort_keys=True))
        r = run_tool("rans_verify.py", str(shard))
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        self.assertIn("file digest checked", r.stdout)
        self.assertIn("no record digests", r.stdout)

        # a corrupt manifest is refused by name, not skipped
        manifest.write_text("{not json")
        r = run_tool("rans_verify.py", str(shard))
        self.assertEqual(r.returncode, 1, msg=r.stdout)
        self.assertIn("E_MANIFEST_MALFORMED", r.stdout)
        self.assertNotIn("Traceback", r.stderr)

        # malformed-typed digest fields must be malformation even when the
        # malformation removes ALL digest evidence — an entry whose only
        # digest fields are wrong-typed must never be classed as a
        # digest-less older mint (note + TRUST)
        for corpse in (
            {"shards": [{"file": shard.name, "sha256": 12345}]},
            {"shards": [{"file": shard.name, "records": ["a"]}]},
        ):
            manifest.write_text(json.dumps(corpse, sort_keys=True))
            r = run_tool("rans_verify.py", str(shard))
            self.assertEqual(r.returncode, 1, msg=f"{corpse}\n{r.stdout}")
            self.assertIn("E_MANIFEST_MALFORMED", r.stdout, msg=str(corpse))
            self.assertNotIn("RESULT: TRUST", r.stdout, msg=str(corpse))
            self.assertNotIn("Traceback", r.stderr, msg=str(corpse))

        # digests present but this shard missing from the set -> refusal
        manifest.write_text(json.dumps(
            {"shards": [{"file": "out-rans-99999.safetensors",
                         "records": {"x": "0" * 64}}]}, sort_keys=True))
        r = run_tool("rans_verify.py", str(shard))
        self.assertEqual(r.returncode, 1, msg=r.stdout)
        self.assertIn("E_DIGEST_MISSING", r.stdout)

    def test_census_layout_accounting(self):
        """--census (#594): every record's bytes fully attributed — the framed
        extent is exactly the round16 header plus the round16-padded payload,
        offsets point at real record starts, and the per-shard sums close."""
        out = self.root / "census"
        shard = self.repack(out)[0]
        r = run_tool("repack_rans.py", "--census", str(out))
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        cen = json.loads((out / "rans-census.json").read_text())
        self.assertEqual(cen["format"], rf.FORMAT_NAME)
        sh = cen["shards"][0]
        rows = {row["record_id"]: row for row in sh["records"]}
        self.assertEqual(sorted(rows), sorted(self.weights))

        header, data_start = rf.read_header(str(shard))
        head_bytes = rf.record_header_bytes()
        for name, (raw, _) in self.weights.items():
            row = rows[name]
            # logical length is the byte-exact int4 content
            self.assertEqual(row["logical_bytes"], len(raw), name)
            # full attribution: extent == header + payload + its round16 pad
            pad = (row["physical_extent_bytes"] - head_bytes
                   - row["encoded_payload_bytes"])
            self.assertGreaterEqual(pad, 0, name)
            self.assertLessEqual(pad, 15, name)
            self.assertEqual(row["framing_bytes"],
                             row["physical_extent_bytes"]
                             - row["encoded_payload_bytes"], name)
            self.assertNotIn("UNATTRIBUTED_BYTES", row, name)
            # physical_offset really points at this record's bytes
            blob, info = rf.read_tensor_bytes(str(shard), header, data_start, name)
            self.assertEqual(row["physical_offset"],
                             data_start + info["data_offsets"][0], name)
            self.assertEqual(row["physical_extent_bytes"], len(blob), name)
            # no per-record physical alignment in this layout, stated not omitted
            self.assertEqual(row["alignment_bytes"], 0, name)
        self.assertEqual(sh["sum_physical_extent_bytes"],
                         sum(r_["physical_extent_bytes"] for r_ in sh["records"]))
        self.assertEqual(sh["file_bytes"], shard.stat().st_size)

    def test_quantize_freq_ties_and_termination(self):
        """Exact fractional-remainder ties must break identically everywhere
        (stable argsort -> lowest symbol index wins), and an impossible table
        size must raise instead of looping forever."""
        hist = np.zeros(16, dtype=np.int64)
        hist[1] = hist[2] = hist[3] = 1        # remainders tie exactly at 1/3
        freq = rf.quantize_freq(hist)
        self.assertEqual(freq.tolist(),
                         [0, 5462, 5461, 5461, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(int(freq.sum()), rf.M)
        with self.assertRaises(ValueError):
            rf.quantize_freq(np.ones(16, dtype=np.int64), m=8)


if __name__ == "__main__":
    unittest.main()
