"""Qwen-Image-2.1 from GGUFs, on a tiny model: the engine's GGUF path against the checkpoint path.

tools/make_qwenimage_gguf_tiny.py writes a random denoiser as unsloth ships Qwen-Image-2.1's
(no metadata, "model.diffusion_model." names, gate and proj fused in img_mlp.gate_up, K-quant
types), a random Qwen3-VL text encoder as llama.cpp writes it, and the same weights decoded to
f32 in the checkpoint's layout. tools/gguf_import.py makes the model folder from the two GGUFs.

  - with every matrix decoded to f32 at load (COLI_IMG_BITS=32), the GGUF folder gives the
    checkpoint's picture: the names, the split of gate_up, the text encoder's map, the
    tokenizer from the GGUF's metadata and the decoders are the checkpoint's. Measured on a
    Zen 4: byte for byte; held to 80 dB, as numpy's decoders and C's (fused multiply-adds where
    the compiler contracts) may round apart by an ulp elsewhere. gate and proj swapped: 47 dB.
  - the matrices kept in the file's blocks (QI_GG, the default) give the bytes of the same
    matrices decoded at load (f32 activations); and where the CPU has VNNI, the blocks' int8 x
    int8 product gives the bytes of the int8 rows COLI_IMG_NATIVE=0 makes (a denoiser whose
    other matrices are int8 rows either way, the checkpoint's text encoder for both)
  - --turbo: the reference folder carries Turbo's schedule; the same holds.

Needs QWENIMAGE_TINY (make_qwenimage_tiny.py's output), numpy and the engine built (make
qwenimage); skipped otherwise.
  QWENIMAGE_TINY=/path/tiny python3 -m unittest -v tests.test_qwenimage_gguf_tiny"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwenimage.exe" if sys.platform == "win32" else "qwenimage")
TINY = os.environ.get("QWENIMAGE_TINY")
sys.path.insert(0, str(HERE))
try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
import image_engine  # noqa: E402


def psnr(a, b):
    wa, ha, ca, pa = image_engine.decode_png(Path(a).read_bytes())
    wb, hb, cb, pb = image_engine.decode_png(Path(b).read_bytes())
    assert (wa, ha, ca) == (wb, hb, cb)
    se = n = 0
    for i in range(len(pa)):
        if ca == 4 and i % 4 == 3:
            continue
        se += (pa[i] - pb[i]) ** 2
        n += 1
    return 99.0 if se == 0 else 10 * math.log10(255 * 255 * n / se)


@unittest.skipUnless(TINY and HAVE_NUMPY and ENGINE.exists(), "needs QWENIMAGE_TINY, numpy and a built qwenimage")
class QwenImageGgufTiny(unittest.TestCase):
    TURBO = False

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cmd = [sys.executable, str(HERE / "tools" / "make_qwenimage_gguf_tiny.py"), TINY, str(d / "src")]
        subprocess.run(cmd + (["--turbo"] if cls.TURBO else []), check=True, capture_output=True)
        imp = [sys.executable, str(HERE / "tools" / "gguf_import.py"), str(d / "src" / "dit.gguf"), str(d / "gg"),
               "--text-encoder", str(d / "src" / "te.gguf"), "--vae", TINY] + (["--variant", "turbo"] if cls.TURBO else [])
        p = subprocess.run(imp, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        cls.import_log = p.stdout
        # the denoiser whose txt_in and modulation are BF16 (int8 rows either way), with the
        # reference's text encoder (an official folder's, unchanged by COLI_IMG_NATIVE)
        p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(d / "src" / "dit_io16.gguf"),
                            str(d / "gg_te"), "--text-encoder", str(d / "src" / "ref"), "--vae", TINY]
                           + (["--variant", "turbo"] if cls.TURBO else []), capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        cls.d = d

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def image(self, model, name, steps=None, **env):
        out = self.d / f"{name}.png"
        steps = steps or (8 if self.TURBO else 3)
        p = subprocess.run([str(ENGINE), "--model", str(self.d / model), "--prompt", "a red fox sitting in the snow",
                            "--width", "256", "--height", "256", "--steps", str(steps), "--seed", "7", "--threads", "2",
                            "--out", str(out)], env=dict(os.environ, **env), capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr[-3000:])
        return out, p.stderr

    def test_folder(self):
        index = json.loads((self.d / "gg" / "model_index.json").read_text())
        self.assertEqual(index["_class_name"], "QwenImage21Pipeline")
        self.assertEqual("sample_sigmas" in index, self.TURBO)
        cfg = json.loads((self.d / "gg" / "transformer" / "config.json").read_text())
        self.assertEqual((cfg["num_layers"], cfg["num_attention_heads"], cfg["mlp_ratio"], cfg["in_channels"],
                          cfg["context_in_dim"]), (2, 2, 3, 16, 256))
        self.assertTrue(image_engine.is_image_model_dir(self.d / "gg"))
        plan = image_engine.plan_image_model(self.d / "gg", 8e9)
        comps = {c["name"]: c for c in plan["components"]}
        self.assertEqual(comps["transformer"]["precision"], "gguf")
        self.assertEqual(comps["text_encoder"]["precision"], "gguf")
        self.assertTrue(plan["model"]["research_license"])
        self.assertFalse(plan["warnings"], plan["warnings"])

    def test_checkpoint_picture(self):
        ref, _ = self.image("src/ref", "ref", COLI_IMG_BITS="32")
        gg, err = self.image("gg", "gg32", COLI_IMG_BITS="32")
        self.assertIn("qwen_image, 1 part(s)", err)
        self.assertIn("qwen3vl, 1 part(s)", err)
        self.assertGreater(psnr(ref, gg), 80.0)

    def test_blocks_are_the_decoded_matrices(self):
        f32, _ = self.image("gg", "gg32b", COLI_IMG_BITS="32")
        native, err = self.image("gg", "native", COLI_IMG_ACT8="0")
        self.assertIn("matrices in the GGUF's blocks", err)
        self.assertEqual(f32.read_bytes(), native.read_bytes())

    def test_int8_products_match_int8_rows(self):
        a, err = self.image("gg_te", "act8_native", COLI_IMG_ACT8="1")
        if "needs VNNI" in err:
            self.skipTest("no VNNI: the int8 x int8 product is not built")
        b, err2 = self.image("gg_te", "act8_rows", COLI_IMG_ACT8="1", COLI_IMG_NATIVE="0")
        self.assertIn("matrices in the GGUF's blocks", err)
        self.assertNotIn("matrices in the GGUF's blocks", err2)
        self.assertEqual(a.read_bytes(), b.read_bytes())


class QwenImageGgufTinyTurbo(QwenImageGgufTiny):
    TURBO = True


if __name__ == "__main__":
    unittest.main()
