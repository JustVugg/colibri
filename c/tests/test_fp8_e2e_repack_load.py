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
     every stamped tensor loads fmt=8 through the real qt_from_disk with
     finite dequantized values.

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
    # Mirror the Makefile's compiler choice: Apple Clang on Darwin, GCC on
    # every other supported target. A generic `cc` may be Clang without
    # libomp even when the production GCC toolchain is installed.
    if sys.platform == "darwin":
        cc = shutil.which("clang") or shutil.which("cc")
    else:
        cc = shutil.which("gcc")
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


class Fp8RepackLoadE2ETest(unittest.TestCase):
    """The real tools/repack_fp8_passthrough.py -> the real C loader, once."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.indir = os.path.join(self.tmp.name, "fp8src")
        os.makedirs(self.indir)
        self.outdir = os.path.join(self.tmp.name, "out")
        self.shard = os.path.join(self.indir, "model-00001-of-00001.safetensors")

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_repack_output_loads_through_real_c_loader(self):
        D, I_ = 256, 384   # non-degenerate shapes: nblkO*nblkI never coincides with O (no
                            # THE DESIGN LANDMINE ambiguity here -- that boundary is
                            # test_fp8_load.c's job, this test's job is the round trip)
        torch.manual_seed(7)
        sd = {
            "model.layers.0.self_attn.o_proj.weight": torch.randn(D, D) * 0.02,
            "model.layers.0.self_attn.q_a_proj.weight": torch.randn(D, D) * 0.02,
            "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(D, I_) * 0.02,
            "model.layers.0.mlp.gate_proj.weight": torch.randn(D, I_) * 0.02,
            # routed expert: must NOT be selected/stamped (stays int4-g64 path) -- included
            # as a negative control so this test also proves the real tool's selection
            # logic held on the real round trip, not just in test_fp8_repack.py's own suite.
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
        }
        save_fp8_safetensors(sd, self.shard)

        tool = os.path.join(C_DIR, "tools", "repack_fp8_passthrough.py")
        rc = subprocess.run([sys.executable, tool, "--indir", self.indir,
                            "--outdir", self.outdir, "--n-layers", "5"],
                           capture_output=True, text=True)
        self.assertEqual(rc.returncode, 0, f"real repack tool failed:\nSTDOUT:\n{rc.stdout}\nSTDERR:\n{rc.stderr}")

        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        self.assertEqual(len(outs), 1, "expected exactly one repacked output shard")

        # Sanity-check the real tool's own selection/stamp BEFORE asking the C harness
        # to load it: if this fails, the bug is in the tool, not the loader, and running
        # the harness anyway would only muddy which side broke.
        with open(outs[0], "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
        self.assertIn("__metadata__", hdr)
        self.assertIn("colibri.fmt", hdr["__metadata__"])
        stamp = json.loads(hdr["__metadata__"]["colibri.fmt"])
        self.assertEqual(set(stamp.keys()), set(shapes.keys()),
                         "real tool's selection must match exactly the resident-kind tensors")
        self.assertNotIn("model.layers.0.mlp.experts.0.gate_proj.weight", hdr)
        self.assertNotIn("model.layers.0.input_layernorm.weight", hdr)

        cc, cflags, ldflags = _cc_flags()
        if not cc:
            raise unittest.SkipTest("no C compiler found on PATH")

        harness_src = os.path.join(HERE, "test_fp8_e2e_loader.c")
        harness_bin = os.path.join(self.tmp.name, "test_fp8_e2e_loader")
        build = subprocess.run([cc] + cflags + [harness_src, "-o", harness_bin] + ldflags,
                               capture_output=True, text=True, cwd=C_DIR)
        self.assertEqual(build.returncode, 0, f"e2e harness build failed:\n{build.stderr}")
        self.assertEqual(build.stderr.strip(), "",
                         f"e2e harness build produced warnings (production flags require zero):\n{build.stderr}")

        args = [harness_bin, self.outdir]
        for name, (O, I) in shapes.items():
            args += [name, str(O), str(I)]
        run = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0,
                         f"e2e harness (real loader) failed:\nSTDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}")
        self.assertNotIn("FAIL", run.stdout)
        for name in shapes:
            self.assertIn(f"ok {name}:", run.stdout)
        self.assertIn("fp8 e2e repack->load: ok", run.stdout)


if __name__ == "__main__":
    unittest.main()
