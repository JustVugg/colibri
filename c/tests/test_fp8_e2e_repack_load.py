"""End-to-end: REAL tools/repack_fp8_passthrough.py output fed into the REAL
C loader (st_init/qt_from_disk in colibri.c) via a small C harness.

Neither test_fp8_load.c (hand-built wire-format C fixtures -- proves the C
loader's OWN logic, but the container bytes are hand-authored, not
tool-produced) nor test_fp8_repack.py (proves the Python tool's OUTPUT
shape, but never invokes the C loader at all) covers this round trip. This
test does, end to end:

  1. build a synthetic FP8 checkpoint with tools/glm_fp8_emit.py (the same
     fixture-generation helper test_fp8_repack.py uses -- not reimplemented
     here, imported directly);
  2. repack it with the REAL tools/repack_fp8_passthrough.py, invoked as a
     subprocess exactly as a user would run it from the command line (not
     imported and called as a library, so the CLI entry point is exercised
     too, not just the importable functions);
  3. compile tests/test_fp8_e2e_loader.c -- production flags mirroring the
     Makefile's own CFLAGS, asserted to produce zero warnings -- into a
     temp binary;
  4. run that binary against the REAL repacked output directory, asserting
     every selected tensor loads fmt=8 through the real qt_from_disk with
     finite dequantized values.

Three cases the round trip has to carry, all through the same real loader:

  * a RESIDENT tensor at the COLLISION shape (nblkO*nblkI==O). Its stamp is the
    only thing standing between fmt=8 and a silent int8-row misread, and this is
    the first test in the tree that proves the stamp does that job on a
    tool-produced container rather than a hand-authored one. Its proof-of-bite
    is a second run over the SAME container with the `colibri.fmt` key deleted
    from the header: the loader must then resolve the identical bytes to fmt=1.
  * a ROUTED EXPERT, unstamped by design (colibri.c's routed-expert load sites
    pass stamped_name=NULL), resolving to fmt=8 by byte arithmetic alone.
  * the container-level `colibri.container.formats` declaration, which no engine
    reads yet: an unknown `__metadata__` key must be IGNORED, not refused. That
    is invariant I-1's half of this PR -- an engine that does not know the key
    loads such a container exactly as it does today.

Hermetic: everything (checkpoint, repacked output, compiled harness binary)
lives under one TemporaryDirectory; nothing is left behind.
"""
import glob, json, os, shutil, struct, subprocess, sys, tempfile, unittest

try:
    import torch
except ImportError as e:
    raise unittest.SkipTest(f"torch not installed: {e}")

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(C_DIR, "tools"))
from glm_fp8_emit import save_fp8_safetensors  # reuse: real fp8 block-quantize, not reimplemented


def _cc_flags():
    """Mirror the Makefile's CFLAGS closely enough to compile colibri.c
    cleanly: -O3 + the same warning flags, plus libomp on macOS if present
    (falls back to single-threaded -- exactly like the Makefile's own
    OMPDIR probe -- if it's not, rather than failing the build). Returns
    (cc, cflags, ldflags) or (None, None, None) if no compiler is found."""
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if not cc:
        return None, None, None
    cflags = ["-O3", "-Wall", "-Wextra", "-Wno-unused-parameter",
              "-Wno-misleading-indentation", "-Wno-unused-function"]
    ldflags = ["-lm"]
    if sys.platform not in ("darwin", "win32"):
        # -fopenmp, like the Makefile. Without it every '#pragma omp' in colibri.c
        # becomes a -Wunknown-pragmas warning (-Wall enables it), and the assertion
        # below requires an empty stderr -- so on Linux this test failed for a
        # reason that had nothing to do with what it is testing. macOS gets the
        # libomp probe just below; Windows/MinGW is left alone.
        cflags += ["-fopenmp"]
        ldflags += ["-fopenmp"]
    if sys.platform == "darwin":
        try:
            prefix = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True,
                                    text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
            prefix = ""
        inc, lib = os.path.join(prefix, "include"), os.path.join(prefix, "lib")
        if prefix and os.path.exists(os.path.join(inc, "omp.h")):
            cflags += ["-Xclang", "-fopenmp", "-I", inc]
            ldflags += ["-L", lib, "-lomp"]
    return cc, cflags, ldflags


def _read_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        return hlen, json.loads(f.read(hlen))


def _rewrite_metadata(path, drop_key=None, set_key=None, value=None):
    """Edit `__metadata__` IN PLACE, keeping every tensor's data offset valid by
    re-padding the header JSON to its original byte length with spaces (the
    safetensors header is length-prefixed and may carry trailing whitespace;
    every JSON parser in the chain, including c/json.h's, skips it). Used to
    build the proof-of-bite containers: identical weight bytes, identical
    offsets, one metadata key changed."""
    hlen, hdr = _read_header(path)
    if drop_key is not None:
        del hdr["__metadata__"][drop_key]
    if set_key is not None:
        hdr["__metadata__"][set_key] = value
    new = json.dumps(hdr, separators=(",", ":")).encode()
    assert len(new) <= hlen, "rewritten header must fit the original slot"
    new += b" " * (hlen - len(new))
    with open(path, "r+b") as f:
        f.seek(8)
        f.write(new)


class Fp8RepackLoadE2ETest(unittest.TestCase):
    """The real tools/repack_fp8_passthrough.py -> the real C loader."""

    # colibri.c is a big translation unit; build the harness once for the class
    # rather than once per test.
    _harness_dir = None
    _harness_bin = None
    _build_stderr = None

    @classmethod
    def setUpClass(cls):
        cc, cflags, ldflags = _cc_flags()
        if not cc:
            raise unittest.SkipTest("no C compiler found on PATH")
        cls._harness_dir = tempfile.TemporaryDirectory()
        harness_src = os.path.join(HERE, "test_fp8_e2e_loader.c")
        harness_bin = os.path.join(cls._harness_dir.name, "test_fp8_e2e_loader")
        build = subprocess.run([cc] + cflags + [harness_src, "-o", harness_bin] + ldflags,
                               capture_output=True, text=True, cwd=C_DIR)
        if build.returncode != 0:
            raise AssertionError(f"e2e harness build failed:\n{build.stderr}")
        cls._harness_bin = harness_bin
        cls._build_stderr = build.stderr

    @classmethod
    def tearDownClass(cls):
        if cls._harness_dir:
            cls._harness_dir.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.indir = os.path.join(self.tmp.name, "fp8src")
        os.makedirs(self.indir)
        self.outdir = os.path.join(self.tmp.name, "out")
        self.shard = os.path.join(self.indir, "model-00001-of-00001.safetensors")
        # FULL-FAMILY EMISSION (M1 audit): config.json AND tokenizer.json are
        # now MANDATORY in --indir -- the real repack tool refuses to run
        # without either. This harness only exercises the C loader's
        # tensor-decode path (qt_from_disk via test_fp8_e2e_loader.c), never
        # load_cfg or tok_load, so a minimal placeholder is enough for both.
        with open(os.path.join(self.indir, "config.json"), "w") as fh:
            json.dump({"model_type": "glm_moe_dsa_test_fixture"}, fh)
        with open(os.path.join(self.indir, "tokenizer.json"), "w") as fh:
            json.dump({"model": {"vocab": {"<|endoftext|>": 0}, "merges": []}}, fh)

    def tearDown(self):
        self.tmp.cleanup()

    def _repack(self):
        tool = os.path.join(C_DIR, "tools", "repack_fp8_passthrough.py")
        rc = subprocess.run([sys.executable, tool, "--indir", self.indir,
                            "--outdir", self.outdir, "--n-layers", "5"],
                           capture_output=True, text=True)
        self.assertEqual(rc.returncode, 0,
                         f"real repack tool failed:\nSTDOUT:\n{rc.stdout}\nSTDERR:\n{rc.stderr}")
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        self.assertEqual(len(outs), 1, "expected exactly one repacked output shard")
        return outs[0]

    def _load(self, shapes):
        args = [self._harness_bin, self.outdir]
        for name, (O, I) in shapes.items():
            args += [name, str(O), str(I)]
        return subprocess.run(args, capture_output=True, text=True)

    def test_harness_builds_without_warnings(self):
        """Production flags require zero warnings (pr-cycle mechanical gate)."""
        self.assertEqual(self._build_stderr.strip(), "",
                         f"e2e harness build produced warnings:\n{self._build_stderr}")

    def test_real_repack_output_loads_through_real_c_loader(self):
        D, I_ = 256, 384   # non-degenerate shapes: nblkO*nblkI never coincides with O (no
                            # THE DESIGN LANDMINE ambiguity here -- that boundary is
                            # test_fp8_load.c's job and this file's collision test below)
        torch.manual_seed(7)
        sd = {
            "model.layers.0.self_attn.o_proj.weight": torch.randn(D, D) * 0.02,
            "model.layers.0.self_attn.q_a_proj.weight": torch.randn(D, D) * 0.02,
            "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(D, I_) * 0.02,
            "model.layers.0.mlp.gate_proj.weight": torch.randn(D, I_) * 0.02,
            # routed expert: selected and byte-preserved, but UNSTAMPED -- it must
            # resolve to fmt=8 by byte arithmetic alone, because colibri.c's
            # routed-expert load sites pass stamped_name=NULL and would never see a
            # stamp even if one were written.
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(I_, D) * 0.02,
            "model.layers.0.input_layernorm.weight": torch.randn(D),   # f32: must NOT be selected
        }
        # (O, I) exactly as the engine/repack tool see them -- known here because this
        # test authored the checkpoint; the real C loader never gets told this by us,
        # only by the container it reads (st_init parses shape from the file itself;
        # O/I are qt_from_disk's own [O,I] contract, same as every real model load).
        shapes = {
            "model.layers.0.self_attn.o_proj.weight": (D, D),
            "model.layers.0.self_attn.q_a_proj.weight": (D, D),
            "model.layers.0.mlp.shared_experts.gate_proj.weight": (D, I_),
            "model.layers.0.mlp.gate_proj.weight": (D, I_),
            "model.layers.0.mlp.experts.0.gate_proj.weight": (I_, D),
        }
        save_fp8_safetensors(sd, self.shard)
        out = self._repack()

        # Sanity-check the real tool's own selection/stamp BEFORE asking the C harness
        # to load it: if this fails, the bug is in the tool, not the loader, and running
        # the harness anyway would only muddy which side broke.
        _, hdr = _read_header(out)
        self.assertIn("__metadata__", hdr)
        self.assertIn("colibri.fmt", hdr["__metadata__"])
        stamp = json.loads(hdr["__metadata__"]["colibri.fmt"])
        self.assertEqual(set(stamp.keys()),
                         set(shapes) - {"model.layers.0.mlp.experts.0.gate_proj.weight"},
                         "stamp map must cover the residents and NOT the routed expert")
        self.assertIn("model.layers.0.mlp.experts.0.gate_proj.weight", hdr,
                      "routed experts are repacked into the container")
        # FULL-FAMILY EMISSION (M1 audit): input_layernorm is now emitted too
        # -- as a raw F32 pass-through (is_passthrough_target), not through
        # the fp8-repack path this harness's qt_from_disk-specific decode
        # check below exercises (that's a different loader function, `ld`'s
        # st_read_f32 fallback -- out of this C harness's scope, and out of
        # this tool's scope per this task: no engine/loader changes here).
        # What matters for THIS test is that it stays out of the fp8 stamp
        # map and carries no `.qs` -- test_fp8_repack.py and
        # test_fp8_repack_full_family.py cover the pass-through copy itself.
        self.assertIn("model.layers.0.input_layernorm.weight", hdr)
        self.assertEqual(hdr["model.layers.0.input_layernorm.weight"]["dtype"], "F32")
        self.assertNotIn("model.layers.0.input_layernorm.weight.qs", hdr)
        self.assertNotIn("model.layers.0.input_layernorm.weight", stamp)
        # I-1: the container-level declaration is an UNKNOWN key to this engine.
        self.assertIn("colibri.container.formats", hdr["__metadata__"])

        run = self._load(shapes)
        self.assertEqual(run.returncode, 0,
                         f"e2e harness (real loader) failed:\nSTDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}")
        self.assertNotIn("FAIL", run.stdout)
        for name in shapes:
            self.assertIn(f"ok {name}:", run.stdout)
        self.assertIn("fp8 e2e repack->load: ok", run.stdout)
        # ...and the unknown container-level key produced no diagnostic at all.
        self.assertEqual(run.stderr.strip(), "",
                         f"unknown __metadata__ key must be ignored silently:\n{run.stderr}")

    def test_container_declaration_stays_inert_even_when_malformed(self):
        """I-1's sharp edge. The well-formed declaration being ignored is checked
        above; this checks the KEY is inert, not just this VALUE. An engine that
        does not know `colibri.container.formats` must ignore whatever is under
        it -- if it ever started validating the value, a container written by a
        newer tool would stop loading on an older engine, which is exactly the
        compatibility break the split between PR-A's writer half and the later
        engine half is arranged to avoid. Contrast `colibri.fmt`, which IS
        parsed and DOES refuse loudly on a malformed value (st.h's
        st_fmt_stamp_ingest) -- that asymmetry is deliberate and is what this
        test pins.

        Non-square, non-degenerate shape on purpose: resolution here must come
        from byte arithmetic, so nothing about this test depends on the stamp
        path that the collision test exercises."""
        torch.manual_seed(13)
        name = "model.layers.0.self_attn.q_a_proj.weight"
        save_fp8_safetensors({name: torch.randn(256, 384) * 0.02}, self.shard)
        out = self._repack()
        # Each replacement must fit the original value's serialized slot, so the
        # tensor data offsets the header declares stay valid (see
        # _rewrite_metadata): not-JSON, a JSON object where an array is
        # expected, a JSON array of the wrong element type, and empty.
        for bogus in ('not json', '{"a":1}', '[42]', ''):
            _rewrite_metadata(out, set_key="colibri.container.formats", value=bogus)
            run = self._load({name: (256, 384)})
            self.assertEqual(run.returncode, 0,
                             f"unknown container key with value {bogus!r} must not "
                             f"break the load:\n{run.stdout}\n{run.stderr}")
            self.assertIn(f"ok {name}:", run.stdout)
            self.assertEqual(run.stderr.strip(), "",
                             f"value {bogus!r} produced a diagnostic:\n{run.stderr}")

    def test_collision_shape_resident_loads_fmt8_and_the_stamp_is_what_does_it(self):
        """I-4 end to end, both polarities, on a container the REAL tool produced.

        [2,256] is the collision shape (nblkO=1, nblkI=2, product=2==O): the
        block-scale array and a per-row int8 scale array are the same 8 bytes, so
        byte arithmetic alone cannot tell fmt=8 from fmt=1 and qt_resolve_fmt's
        INVERSION resolves an unstamped one to fmt=1. Stamped, the loader must
        return fmt=8. PROOF-OF-BITE, on the SAME container with the SAME weight
        bytes: delete `colibri.fmt` from the header and the loader must return
        fmt=1 -- so this test would notice if the stamp stopped being consulted,
        rather than passing on byte arithmetic that never needed it."""
        torch.manual_seed(9)
        name = "model.layers.0.self_attn.o_proj.weight"
        # glm_fp8_emit's block-quantizer is the fixture generator everywhere else in
        # this suite; [2,256] goes through it unchanged (one 128x128 block row, two
        # block columns).
        save_fp8_safetensors({name: torch.randn(2, 256) * 0.02}, self.shard)
        out = self._repack()

        _, hdr = _read_header(out)
        self.assertEqual(json.loads(hdr["__metadata__"]["colibri.fmt"]),
                         {name: "fp8-e4m3-b128"})
        self.assertEqual(hdr[name + ".qs"]["shape"], [2])
        self.assertEqual(hdr[name]["shape"], [2, 256])

        run = self._load({name: (2, 256)})
        self.assertEqual(run.returncode, 0,
                         f"stamped collision-shape tensor must load fmt=8:\n{run.stdout}\n{run.stderr}")
        self.assertIn(f"ok {name}:", run.stdout)

        _rewrite_metadata(out, drop_key="colibri.fmt")
        bite = self._load({name: (2, 256)})
        self.assertNotEqual(bite.returncode, 0,
                            "unstamped, the identical bytes must NOT resolve to fmt=8")
        self.assertIn("fmt=1, expected 8", bite.stdout,
                      f"expected the INVERSION's int8-row default:\n{bite.stdout}")


if __name__ == "__main__":
    unittest.main()
