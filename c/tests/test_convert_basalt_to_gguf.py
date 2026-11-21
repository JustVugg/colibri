"""tools/convert_basalt_to_gguf.py on a tiny qwen35 pack in Basalt's form (tests/basalt_fixture.py: the QUASAR
pack's tensor types, MTP layer and DFlash2 drafter).

Checked here: the container is refused when a section's SHA-256, the magic or the version is wrong; every tensor
of the GGUF against the pack's (NVFP4_PLANES -> ggml NVFP4 bit for bit, decoded as ggml_quants.h's gq_blk_nvfp4
decodes; the `.scale` companions; the F8_E4M3 head as Q8_0 within half a Q8_0 step of its values, or F16 / BF16
rounded once; everything else byte for byte); the MTP layer as llama.cpp's nextn layer; the drafter as the
directory COLI_DFLASH reads (its Q8_0 matrices come back to the pack's bytes through df_upload_q8's rounding).

With the qwen36 engine built (BASALT_CONVERT_REQUIRED=1, `make basalt-convert-check`, refuses to skip): the
imported GGUF passes GGUF_CHECK=1, decodes with and without Q36_MTP=1 to the same tokens, and with GQ_EXACT=1 its
residuals equal, to float rounding, those of the same model written with every NVFP4 and FP8 matrix as F32 of its
values (the engine reads the converted blocks and scales as the pack's values).
"""
import json
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
except ImportError as exc:
    # make basalt-convert-check sets BASALT_CONVERT_REQUIRED=1: there a missing numpy is a failure, not a skip
    if os.environ.get("BASALT_CONVERT_REQUIRED") == "1":
        raise
    raise unittest.SkipTest("numpy not installed: %s" % exc)

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tests"))
sys.path.insert(0, str(HERE / "tools"))
import basalt_fixture as B  # noqa: E402
from gguf_reader import GGUFReader  # noqa: E402

TOOL = HERE / "tools" / "convert_basalt_to_gguf.py"
ENGINE = HERE / ("qwen36.exe" if sys.platform == "win32" else "qwen36")
REQUIRED = os.environ.get("BASALT_CONVERT_REQUIRED") == "1"

# ggml_quants.h: GGML_KVALUES_MXFP4 (e2m1 doubled) and gq_ue4m3_half (0 and 0x7F decode to 0)
KVALUES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float64)


def ue4m3_half(x):
    x = x.astype(np.int64)
    e, m = (x >> 3) & 15, x & 7
    v = np.where(e == 0, m * 2.0 ** -10, (8 + m) * 2.0 ** (e - 11))
    return np.where((x == 0) | (x == 0x7F), 0.0, v)


def nvfp4_decode(raw, rows, K):
    """ggml NVFP4 blocks -> values [rows, K], as gq_blk_nvfp4"""
    b = np.frombuffer(raw, np.uint8).reshape(rows, K // 64, 36)
    d = ue4m3_half(b[:, :, :4])                                   # [rows, nb, 4]
    qs = b[:, :, 4:].reshape(rows, K // 64, 4, 8)
    y = np.empty((rows, K // 64, 4, 16))
    y[..., :8], y[..., 8:] = KVALUES[qs & 15], KVALUES[qs >> 4]
    return (y * d[..., None]).reshape(rows, K)


def run(*args):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True)


class Gguf:
    def __init__(self, path):
        self.r = GGUFReader(str(path))
        self.meta = self.r.metadata
        self.t = {t["name"]: t for t in self.r.tensors}

    def raw(self, name):
        return self.r.read_tensor(self.t[name])

    def close(self):
        self.r.close()


class ConvertBasaltTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="basalt_conv_"))
        cls.pack = cls.tmp / "tiny.basalt"
        cls.fx = B.write_pack(cls.pack)
        cls.out = {}
        for tag, extra in (("q8", []), ("f16", ["--f8", "f16"]), ("bf16", ["--f8", "bf16"]), ("nomtp", ["--no-mtp"]),
                           ("df", ["--dflash-dir", cls.tmp / "dflash"])):
            path = cls.tmp / f"{tag}.gguf"
            r = run(cls.pack, path, *extra)
            if r.returncode:
                raise RuntimeError(r.stdout + r.stderr)
            cls.out[tag] = Gguf(path)
        cls.g = cls.out["q8"]

    @classmethod
    def tearDownClass(cls):
        for g in cls.out.values():
            g.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---- the container
    def test_refuses_a_damaged_or_foreign_file(self):
        bad = self.tmp / "bad.basalt"
        data = bytearray(self.pack.read_bytes())
        off = struct.unpack_from("<Q", data, 64 + 128 + 64)[0]   # the second section: gguf
        data[off + 10] ^= 1
        bad.write_bytes(data)
        r = run(bad, self.tmp / "bad.gguf")
        self.assertEqual(r.returncode, 1)
        self.assertIn("section gguf does not match its SHA-256", r.stderr)
        self.assertFalse((self.tmp / "bad.gguf").exists())
        self.assertEqual(run(bad, self.tmp / "bad.gguf", "--no-verify").returncode, 0)   # unchecked: it converts
        v1 = bytearray(self.pack.read_bytes())
        struct.pack_into("<I", v1, 8, 1)
        bad.write_bytes(v1)
        r = run(bad, self.tmp / "v1.gguf")
        self.assertIn("version 1", r.stderr)
        bad.write_bytes(b"GGUF" + bytes(70000))
        self.assertIn("not a .basalt model file", run(bad, self.tmp / "x.gguf").stderr)

    # ---- metadata
    def test_metadata(self):
        m = self.g.meta
        self.assertEqual(m["general.architecture"], "qwen35")
        self.assertEqual(m["qwen35.block_count"], 3)              # the nextn layer counted, as llama.cpp does
        self.assertEqual(m["qwen35.nextn_predict_layers"], 1)
        self.assertEqual(m["qwen35.attention.recurrent_layers"], [True, False])
        self.assertEqual(m["qwen35.rope.dimension_sections"], [11, 11, 10, 0])
        self.assertAlmostEqual(m["qwen35.rope.freq_base"], 1e7)
        self.assertEqual(m["tokenizer.ggml.tokens"], B.gguf_kv()["tokenizer.ggml.tokens"])
        self.assertEqual(m["tokenizer.ggml.token_type"], B.gguf_kv()["tokenizer.ggml.token_type"])
        self.assertEqual(m["tokenizer.ggml.eos_token_id"], B.VOCAB - 1)
        self.assertEqual(m["tokenizer.chat_template"], "{{ messages }}")
        self.assertFalse([k for k in m if ".dflash." in k])
        n = self.out["nomtp"].meta
        self.assertEqual((n["qwen35.block_count"], n["qwen35.nextn_predict_layers"]), (2, 0))
        self.assertFalse([t for t in self.out["nomtp"].t if t.startswith("blk.2.")])

    # ---- tensors
    def test_every_tensor_round_trips(self):
        T = self.fx.tensors
        seen = set()
        for name, (ty, ne, data) in T.items():
            if name.startswith("dflash.") or name == "output.scale":
                continue
            gname = name
            if name.startswith("mtp."):
                gname = "blk.2." + {"mtp.fc.weight": "nextn.eh_proj.weight", "mtp.norm.weight": "nextn.shared_head_norm.weight",
                                    "mtp.pre_fc_norm_embedding.weight": "nextn.enorm.weight",
                                    "mtp.pre_fc_norm_hidden.weight": "nextn.hnorm.weight"}.get(name, name[4:])
            t = self.g.t[gname]
            seen.add(gname)
            self.assertEqual(t["dims"], ne, name)
            raw = self.g.raw(gname)
            if ty == "NVFP4_PLANES":
                self.assertEqual(t["type_name"], "NVFP4", name)
                codes, sc, gs = self.fx.planes[name]
                rows, K = ne[1], ne[0]
                self.assertEqual(len(raw), len(data))
                blk = np.frombuffer(raw, np.uint8).reshape(rows, K // 64, 36)
                np.testing.assert_array_equal(blk[:, :, :4].reshape(rows, K // 16), sc, name)   # the scale bytes moved
                want = self.fx.values[name] / np.float64(gs)     # e2m1 x ue4m3, exact
                got = nvfp4_decode(raw, rows, K)
                self.assertTrue(np.array_equal(got, want), f"{name}: {np.count_nonzero(got != want)} values differ")
                s = np.frombuffer(self.g.raw(gname[:-len('.weight')] + ".scale"), "<f4")
                self.assertEqual(s.tolist(), [float(gs)], name)
            elif ty == "F8_E4M3":
                continue   # test_f8_head
            else:
                self.assertEqual(t["type_name"], ty if ty != "NVFP4_PLANES" else "NVFP4", name)
                self.assertEqual(raw, data, name)
        self.assertEqual(set(self.g.t), seen)   # nothing else written

    def test_f8_head(self):
        codes, rs = self.fx.f8["output.weight"]
        x = (B.E4M3[codes].astype(np.float32) * rs[:, None]).astype(np.float64)   # the FP8 values
        self.assertNotIn("output.scale", self.g.t)                               # folded in
        t = self.g.t["output.weight"]
        self.assertEqual((t["type_name"], t["dims"]), ("Q8_0", [B.H, B.VOCAB]))
        raw = np.frombuffer(self.g.raw("output.weight"), np.uint8).reshape(-1, 34)
        d = np.ascontiguousarray(raw[:, :2]).view("<f2").astype(np.float64).reshape(-1)
        q = raw[:, 2:].view(np.int8).astype(np.float64)
        err = np.abs(q * d[:, None] - x.reshape(-1, 32))
        amax = np.abs(x.reshape(-1, 32)).max(1)
        # ggml's rounding: half a step of amax / 127, the step itself an f16 (2^-11 off): 0.5625 steps at most
        self.assertTrue((err.max(1) <= amax / 127 * 0.5625 + 1e-12).all())
        self.assertLess(np.sqrt((err ** 2).mean() / (x ** 2).mean()), 0.01)
        f16 = np.frombuffer(self.out["f16"].raw("output.weight"), "<f2").astype(np.float64).reshape(x.shape)
        self.assertTrue((np.abs(f16 - x) <= np.abs(x) * 2.0 ** -11 + 2.0 ** -25).all())
        u = np.frombuffer(self.out["bf16"].raw("output.weight"), "<u2").astype(np.uint32) << 16
        bf = u.view(np.float32).astype(np.float64).reshape(x.shape)
        self.assertTrue((np.abs(bf - x) <= np.abs(x) * 2.0 ** -8).all())
        self.assertEqual(self.out["f16"].t["output.weight"]["type_name"], "F16")
        self.assertEqual(self.out["bf16"].t["output.weight"]["type_name"], "BF16")

    # ---- the drafter
    def test_dflash_directory(self):
        d = self.tmp / "dflash"
        cfg = json.loads((d / "config.json").read_text())
        self.assertEqual((cfg["hidden_size"], cfg["vocab_size"], cfg["num_hidden_layers"]), (B.H, B.VOCAB, B.DF_L))
        self.assertEqual(cfg["dflash_config"]["target_layer_ids"], B.DF_TARGETS)
        self.assertEqual(cfg["dflash_config"]["mask_token_id"], B.VOCAB - 3)
        self.assertEqual((cfg["head_dim"], cfg["intermediate_size"], cfg["dflash_config"]["selector_rank"]), (B.DF_HD, B.DF_I, B.DF_RANK))
        blob = (d / "model.safetensors").read_bytes()
        n = struct.unpack_from("<Q", blob)[0]
        head = json.loads(blob[8:8 + n])
        base = 8 + n

        def tensor(name):
            h = head[name]
            a, b = h["data_offsets"]
            return h, blob[base + a:base + b]

        T = self.fx.tensors
        # fc: the target layers' blocks side by side, F32 of their Q8_0 values; df_upload_q8 rounds them back
        h, raw = tensor("fc.weight")
        self.assertEqual((h["dtype"], h["shape"]), ("F32", [B.H, B.H * len(B.DF_TARGETS)]))
        x = np.frombuffer(raw, "<f4").reshape(B.H, -1)
        want = np.concatenate([self.fx.values[f"dflash.fc.{j}.weight"] for j in range(len(B.DF_TARGETS))], axis=1)
        np.testing.assert_array_equal(x, want)
        for j in range(len(B.DF_TARGETS)):
            self.assertEqual(df_upload_q8(x[:, j * B.H:(j + 1) * B.H]), T[f"dflash.fc.{j}.weight"][2])
        pairs = {"layers.0.self_attn.q_proj.weight": "dflash.blk.0.attn_q.weight",
                 "layers.0.mlp.down_proj.weight": "dflash.blk.0.ffn_down.weight",
                 "layers.0.mlp_conv.kernel_projection.weight": "dflash.blk.0.ffn_conv.weight"}
        for hf, src in pairs.items():
            h, raw = tensor(hf)
            x = np.frombuffer(raw, "<f4").reshape(h["shape"])
            self.assertEqual(df_upload_q8(x), T[src][2], hf)
        for hf, src in (("candidate_selector.predecessor_codebook", "dflash.selector.pred"), ("norm.weight", "dflash.norm.weight"),
                        ("layers.0.attention_conv.base_kernel", "dflash.blk.0.attn_conv.base")):
            h, raw = tensor(hf)
            self.assertEqual(raw, T[src][2], hf)
            self.assertEqual(h["shape"], list(reversed(T[src][1])), hf)
        self.assertEqual(head["candidate_selector.predecessor_codebook"]["dtype"], "BF16")
        self.assertEqual(len([k for k in head if k != "__metadata__"]), 6 + 15 * B.DF_L)   # fc, two norms, the selector's three; 15 a layer


def df_upload_q8(x):
    """dflash_cuda.h df_upload_q8's rounding: d = f16(amax / 127), q = rint(x / d) (by 1/d in float32)"""
    b = np.ascontiguousarray(x, np.float32).reshape(-1, 32)
    am = np.abs(b).max(1)
    dh = (am / np.float32(127)).astype(np.float16)
    dd = dh.astype(np.float32)
    inv = np.where(dd > 0, np.float32(1) / np.where(dd > 0, dd, np.float32(1)), np.float32(0)).astype(np.float32)
    q = np.clip(np.rint(b * inv[:, None]), -127, 127).astype(np.int8)
    out = np.empty((b.shape[0], 34), np.uint8)
    out[:, :2] = dh.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


ENV = dict(OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1", USAGE_SAVE="0", COLI_CUDA="0", COLI_VULKAN="0", COLI_LOOKUP="0")
PROMPT = [5, 6, 7, 8, 9, 10]


@unittest.skipUnless(ENGINE.exists() or REQUIRED, "qwen36 is not built")
class ConvertedRunsTest(unittest.TestCase):
    """The converted file on the qwen36 engine (CPU)."""

    @classmethod
    def setUpClass(cls):
        if not ENGINE.exists():
            raise RuntimeError(f"BASALT_CONVERT_REQUIRED=1 but {ENGINE} is not built")
        cls.tmp = Path(tempfile.mkdtemp(prefix="basalt_run_"))
        cls.models = {}
        for tag, ref in (("pack", False), ("ref", True)):
            src, gg = cls.tmp / f"{tag}.basalt", cls.tmp / f"{tag}.gguf"
            B.write_pack(src, ref=ref)
            r = run(src, gg)
            if r.returncode:
                raise RuntimeError(r.stdout + r.stderr)
            out = cls.tmp / tag
            r = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(gg), str(out)], capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(r.stdout + r.stderr)
            cls.models[tag] = out

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def engine(self, model, n_new, **env):
        (model / "ref.json").write_text(json.dumps({"prompt_ids": PROMPT, "full_ids": PROMPT + [0] * n_new}))
        e = dict(os.environ, SNAP=str(model), Q36_MTP="0", **ENV)
        e.update(env)
        r = subprocess.run([str(ENGINE), "2", "8", "ref.json"], cwd=model, env=e, capture_output=True, text=True,
                           errors="replace", timeout=300)
        log = r.stdout + r.stderr
        line = [l for l in log.splitlines() if l.startswith("C engine :")]
        self.assertTrue(line, log[-2000:])
        return line[-1].split(":", 1)[1].split("Text")[0].split(), log

    def test_gguf_check(self):
        for tag, m in self.models.items():
            r = subprocess.run([str(ENGINE)], env=dict(os.environ, SNAP=str(m), GGUF_CHECK="1", **ENV), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr[-2000:])
            self.assertIn("GGUF CHECK OK", r.stdout)

    def test_mtp_head_drafts_the_same_tokens(self):
        m = self.models["pack"]
        plain, _ = self.engine(m, 8)
        self.assertEqual(len(plain), 8)
        drafted, log = self.engine(m, 8, Q36_MTP="1")
        self.assertIn("[qwen36 mtp]", log)
        self.assertEqual(drafted, plain)

    def test_residuals_are_the_packs_values(self):
        dumps = {}
        for tag, m in self.models.items():
            path = m / "layers.bin"
            self.engine(m, 1, GQ_EXACT="1", DUMP_LAYERS=str(path))
            dumps[tag] = np.fromfile(path, "<f4")
        a, b = dumps["pack"], dumps["ref"]
        self.assertEqual(a.size, b.size)
        self.assertGreater(a.size, 0)
        rel = np.linalg.norm(a - b) / np.linalg.norm(b)
        # the scale applied to the product instead of the weights: float rounding apart (measured 2.3e-7)
        self.assertLess(rel, 1e-5)


if __name__ == "__main__":
    unittest.main()
