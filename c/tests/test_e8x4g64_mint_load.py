"""wq-v0-class regression case: REAL tools/convert_fp8_to_int4.py output fed
into the REAL C loader (st_init/qt_from_disk in colibri.c) at toy scale.

The wq-v0 production container class is "e8x4g64": int8 per-row spine
(attention / shared expert / dense MLP / embed / lm_head -> fmt=1) with
grouped-int4 g64 routed experts (fmt=4, gs=64). This test mints that class
each cycle at toy scale with GLM-shaped tensor names and the load-bearing
real dimensions (kv_lora_rank-width kv_b contraction, group-64-divisible
expert dims) through the real converter CLI -- not a reimplementation of its
quantizers -- and then proves the REAL C loader resolves every tensor to the
format the recipe promises. Regression coverage, not new-feature proof: the
full-scale wq-v0 container is already banked and audited; what this pins is
that the TOOLING and the LOADER still agree on the class after engine
changes (the mint half of the mint->load->run regression bar; the run half
executes at full scale against the banked container, outside this suite).

Also carried here, because it belongs to exactly this container class: the
D-2 duplicate-tensor-name pair on a TOOL-PRODUCED container.
  * positive: a copy of the minted container with one shard duplicated must
    refuse at st_init with the exact D-2 message (naming both shards), before
    any tensor is read;
  * negative control: the untouched mint loads clean through the same
    binary -- the guard is duplicate-targeted, not a blanket gate.
(tests/test_dup_name_refusal.c pins the same guard on hand-built fixtures;
this is the tool-produced end of it.)

Hermetic: checkpoint, minted output, duplicated copy, and the compiled
harness all live under temporary directories; nothing is left behind.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
except ImportError as e:
    raise unittest.SkipTest(f"torch not installed: {e}")

try:
    import numpy as np
except ImportError as e:
    raise unittest.SkipTest(f"numpy not installed: {e}")

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(C_DIR, "tools"))
# reuse: the real fp8 block-quantize fixture helpers, not reimplemented. The
# quantize/dequantize pair anchors the numeric reference below to EXACTLY what
# the converter ingests (its dequant() applies the same repeat_interleave
# formula to the same saved fp8 bytes); the converter's own int8/int4
# quantizers are NOT imported anywhere in this file -- their output is decoded
# and bounded by independent code below.
from glm_fp8_emit import save_fp8_safetensors, fp8_block_quantize, fp8_block_dequantize


def _cc_flags():
    """Mirror the Makefile's CFLAGS closely enough to compile colibri.c
    cleanly (same arrangement as tests/test_fp8_e2e_repack_load.py)."""
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if not cc:
        return None, None, None
    cflags = ["-O3", "-Wall", "-Wextra", "-Wno-unused-parameter",
              "-Wno-misleading-indentation", "-Wno-unused-function"]
    ldflags = ["-lm"]
    if sys.platform not in ("darwin", "win32"):
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


# Toy scale, real geometry where it is load-bearing:
#   * kv_b_proj keeps the REAL contraction width I=512 (GLM-5.2's
#     kv_lora_rank) with a toy head count -> O=160;
#   * routed-expert dims stay multiples of 64 (real containers are), so the
#     g64 grouping has no synthetic tail the production mint never sees --
#     while D_H (o_proj/q_a rows, expert contraction) is NOT a multiple of
#     128, keeping the fp8 input's block scales on the partial-tail path;
#   * no tensor sits on the fmt=1-vs-fmt=8 collision boundary
#     (O == ceil(O/128)*ceil(I/128)), so byte-arithmetic resolution is
#     unambiguous, as it is at full scale.
D_H   = 320    # toy hidden size (multiple of 64, not of 128)
KV_B_O, KV_B_I = 160, 512
E_M   = 192    # toy expert intermediate (multiple of 64)
VOCAB = 512


class E8x4g64MintLoadTest(unittest.TestCase):
    """The real e8x4g64 mint recipe -> the real C loader."""

    _harness_dir = None
    _harness_bin = None
    _build_stderr = None

    @classmethod
    def setUpClass(cls):
        cc, cflags, ldflags = _cc_flags()
        if not cc:
            raise unittest.SkipTest("no C compiler found on PATH")
        cls._harness_dir = tempfile.TemporaryDirectory()
        harness_src = os.path.join(HERE, "test_e8x4g64_loader.c")
        harness_bin = os.path.join(cls._harness_dir.name, "test_e8x4g64_loader")
        build = subprocess.run([cc] + cflags + [harness_src, "-o", harness_bin] + ldflags,
                               capture_output=True, text=True, cwd=C_DIR)
        if build.returncode != 0:
            raise AssertionError(f"e8x4g64 harness build failed:\n{build.stderr}")
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

    def tearDown(self):
        self.tmp.cleanup()

    def _emit_checkpoint(self):
        """A GLM-shaped fp8 source checkpoint: every name below is one the real
        converter's classify() routes exactly like the production checkpoint's
        (kvb/attn/o -> ebits, sh/dmlp -> ebits, io -> io_bits, experts -> xbits,
        norms -> f32 passthrough)."""
        torch.manual_seed(11)
        L = "model.layers.0"
        sd = {
            f"{L}.self_attn.kv_b_proj.weight": torch.randn(KV_B_O, KV_B_I) * 0.02,
            f"{L}.self_attn.q_a_proj.weight": torch.randn(D_H, D_H) * 0.02,
            f"{L}.self_attn.o_proj.weight": torch.randn(D_H, D_H) * 0.02,
            f"{L}.mlp.shared_experts.gate_proj.weight": torch.randn(E_M, D_H) * 0.02,
            f"{L}.mlp.experts.0.gate_proj.weight": torch.randn(E_M, D_H) * 0.02,
            f"{L}.mlp.experts.0.up_proj.weight": torch.randn(E_M, D_H) * 0.02,
            f"{L}.mlp.experts.0.down_proj.weight": torch.randn(D_H, E_M) * 0.02,
            "model.embed_tokens.weight": torch.randn(VOCAB, D_H) * 0.02,
            f"{L}.input_layernorm.weight": torch.randn(D_H),  # f32 passthrough control
        }
        self.sd = sd            # kept: the numeric round-trip test derives its reference from it
        save_fp8_safetensors(sd, self.shard)

    def _mint(self):
        """The REAL converter CLI at the wq-v0 e8x4g64 recipe: int8 spine
        (--ebits 8, --io-bits 8), grouped-int4 g64 routed experts (--xbits 4,
        --group-size 64)."""
        tool = os.path.join(C_DIR, "tools", "convert_fp8_to_int4.py")
        rc = subprocess.run([sys.executable, tool, "--indir", self.indir,
                             "--outdir", self.outdir, "--n-layers", "1",
                             "--ebits", "8", "--xbits", "4", "--io-bits", "8",
                             "--group-size", "64"],
                            capture_output=True, text=True)
        self.assertEqual(rc.returncode, 0,
                         f"real converter failed:\nSTDOUT:\n{rc.stdout}\nSTDERR:\n{rc.stderr}")
        outs = glob.glob(os.path.join(self.outdir, "out-*.safetensors"))
        self.assertEqual(len(outs), 1, f"expected exactly one minted shard, got {outs}")
        return outs[0]

    # 5-tuples for the C harness: (name, O, I, wantfmt, wantgs)
    _EXPECT = [
        ("model.layers.0.self_attn.kv_b_proj.weight", KV_B_O, KV_B_I, 1, 0),
        ("model.layers.0.self_attn.q_a_proj.weight", D_H, D_H, 1, 0),
        ("model.layers.0.self_attn.o_proj.weight", D_H, D_H, 1, 0),
        ("model.layers.0.mlp.shared_experts.gate_proj.weight", E_M, D_H, 1, 0),
        ("model.embed_tokens.weight", VOCAB, D_H, 1, 0),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", E_M, D_H, 4, 64),
        ("model.layers.0.mlp.experts.0.up_proj.weight", E_M, D_H, 4, 64),
        ("model.layers.0.mlp.experts.0.down_proj.weight", D_H, E_M, 4, 64),
    ]

    def _load(self, container_dir):
        args = [self._harness_bin, container_dir]
        for name, n_out, n_in, fmt, gs in self._EXPECT:
            args += [name, str(n_out), str(n_in), str(fmt), str(gs)]
        return subprocess.run(args, capture_output=True, text=True)

    def test_harness_builds_without_warnings(self):
        """Production flags require zero warnings (pr-cycle mechanical gate)."""
        self.assertEqual(self._build_stderr.strip(), "",
                         f"e8x4g64 harness build produced warnings:\n{self._build_stderr}")

    def test_e8x4g64_mint_loads_through_real_c_loader(self):
        self._emit_checkpoint()
        self._mint()
        rc = self._load(self.outdir)
        self.assertEqual(rc.returncode, 0,
                         f"loader harness failed:\nSTDOUT:\n{rc.stdout}\nSTDERR:\n{rc.stderr}")
        for name, _, _, fmt, _ in self._EXPECT:
            self.assertIn(f"ok {name}: fmt={fmt}", rc.stdout)
        # negative D-2 control rides along: the clean mint produced NO
        # duplicate-name refusal anywhere in the load
        self.assertNotIn("duplicate tensor name", rc.stderr)

    def test_e8x4g64_numeric_round_trip(self):
        """The minted VALUES, not just the format class (deep-audit MAJOR-1:
        the fmt/finite checks alone passed a 37x scale corruption of
        quant_int8 -- format class and finiteness both survive numeric
        corruption). This closes DR-9(a)'s literal words, "mint ROUND-TRIP at
        toy scale": dequantize the minted bytes with independent numpy code
        (the converter's quantizers are never imported here) and require
        agreement with the fp8-round-tripped source within the schemes' own
        exact bound -- symmetric absmax rint quantization can never miss by
        more than half a quantization step, so the tolerance is 0.5 steps
        (+1e-3 for f32 arithmetic slop), derived from a scale RECOMPUTED from
        the source, never from the minted .qs (a corrupted stored scale must
        widen the error, not the bound)."""
        self._emit_checkpoint()
        minted = self._mint()
        from safetensors.numpy import load_file
        out = load_file(minted)
        for name, n_out, n_in, fmt, gs in self._EXPECT:
            # Reference: exactly what the converter ingested -- the fp8
            # block-quantized source, dequantized (bit-identical formula to
            # the converter's dequant(); see the import comment above).
            w_fp8, scale = fp8_block_quantize(self.sd[name].float())
            w_ref = fp8_block_dequantize(w_fp8, scale).numpy().astype(np.float32)
            self.assertEqual(w_ref.shape, (n_out, n_in))
            q, qs = out[name], out[name + ".qs"]
            if fmt == 1:
                # int8 per-row: one scale per output row, q in [-127,127]
                # (amax maps to qmax=127, so -128 is unreachable).
                deq = q.view(np.int8).reshape(n_out, n_in).astype(np.float32) * qs.reshape(n_out, 1)
                s_true = np.maximum(np.abs(w_ref).max(axis=1, keepdims=True) / 127.0, 1e-8)
                step = np.broadcast_to(s_true, (n_out, n_in))
            else:
                # int4 grouped g64: packed nibbles (low = even column), one
                # scale per (row, 64-column group), q in [-7,7] likewise.
                rb, ng = (n_in + 1) // 2, (n_in + gs - 1) // gs
                b = q.reshape(n_out, rb)
                deq = np.empty((n_out, n_in), np.float32)
                deq[:, 0::2] = (b & 0xF).astype(np.float32)[:, : (n_in + 1) // 2] - 8.0
                deq[:, 1::2] = (b >> 4).astype(np.float32)[:, : n_in // 2] - 8.0
                deq *= np.repeat(qs.reshape(n_out, ng), gs, axis=1)[:, :n_in]
                pad = np.zeros((n_out, ng * gs - n_in), np.float32)
                grp = np.concatenate([np.abs(w_ref), pad], axis=1).reshape(n_out, ng, gs)
                s_true = np.maximum(grp.max(axis=2) / 7.0, 1e-8)     # [n_out, ng]
                step = np.repeat(s_true, gs, axis=1)[:, :n_in]
            worst = float((np.abs(deq - w_ref) / step).max())
            self.assertLessEqual(
                worst, 0.5 + 1e-3,
                f"{name}: minted values diverge from the source by {worst:.4g} "
                f"quantization steps (bound: half a step) -- the converter's "
                f"numeric output regressed even though the format class may "
                f"still be intact")

    def test_d2_duplicate_shard_refuses_by_name(self):
        """D-2 positive on the tool-produced container: duplicating the minted
        shard under a second indexed name must refuse at st_init with the
        exact duplicate-tensor-name message, before any tensor check runs."""
        self._emit_checkpoint()
        minted = self._mint()
        dupdir = os.path.join(self.tmp.name, "dup")
        shutil.copytree(self.outdir, dupdir)
        shutil.copy(minted, os.path.join(dupdir, "out-99999.safetensors"))
        rc = self._load(dupdir)
        self.assertNotEqual(rc.returncode, 0, "duplicated container must refuse to load")
        self.assertIn("duplicate tensor name across indexed shards, refusing", rc.stderr)
        self.assertNotIn("ok model.", rc.stdout,
                         "no tensor may load from a container st_init refused")


if __name__ == "__main__":
    unittest.main()
