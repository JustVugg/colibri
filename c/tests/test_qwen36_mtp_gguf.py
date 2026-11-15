"""The MTP head of a dense model's GGUF (Qwen3.8-27B): llama.cpp writes it as one more layer,
blk.<n_layers> with nextn.* tensors, and Q36_MTP=1 drafts with it.

A small qwen35 dense GGUF with that layer runs on the CPU without the head, then with it: the
engine names the dense head, and every run's tokens are plain decoding's, the head's own drafts
(depths 1 and 3) and the forced verifies (every draft accepted, every one rejected, mixed) alike.
The same file without the layer runs with Q36_MTP=1 as without it. On the real 27B the head's
drafts were 97.5 % and 80.7 % accepted at one draft a verify (code, prose; UD-Q3_K_XL, CPU).

unsloth's Dynamic v3.1 files ship the head apart (MTP/mtp-*.gguf beside the model): the importer
links it, and the engine drafts with it exactly as with the head inside, the same drafts and the
same tokens. On the real 27B (UD-IQ2_XXS v3.1 with MTP/mtp-Qwen3.8-27B-Q4_0.gguf, CUDA chain) three
drafts a verify took decoding from 30.5 to 55.8 tok/s (code) and 41.2 (prose), the texts unchanged.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen36.exe" if sys.platform == "win32" else "qwen36")
sys.path.insert(0, str(HERE / "tests"))
sys.path.insert(0, str(HERE))
try:
    import numpy as np
    import gguf  # noqa: F401
    from test_gguf_import import write_model
except ImportError:  # pragma: no cover
    write_model = None

HEAD = "[mtp] the checkpoint's head: a dense MLP"
PROMPT = [5, 6, 7, 8, 9, 10]
GEN = 14


@unittest.skipUnless(ENGINE.exists(), "qwen36 is not built")
@unittest.skipUnless(write_model is not None, "the gguf package is not installed")
class Qwen36MtpGgufTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="q36_mtp_gguf_"))
        cls.models = {}
        for name, mtp in (("head", True), ("plain", False), ("apart", True)):
            gg = cls.tmp / name / f"{name}.gguf"
            gg.parent.mkdir()
            apart = None
            if name == "apart":   # unsloth's layout: MTP/mtp-<model>-Q8_0.gguf beside the model
                (gg.parent / "MTP").mkdir()
                apart = gg.parent / "MTP" / "mtp-apart-Q8_0.gguf"
            write_model(gg, np.random.default_rng(29), dense=True, mtp=mtp, mtp_file=apart)
            out = cls.tmp / f"{name}.colibri"
            r = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(gg), str(out)],
                               capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError("import failed:\n" + r.stdout + r.stderr)
            cls.models[name] = out

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_engine(self, model, ref, **env):
        """(tokens, the log) of one run on ref's prompt"""
        e = dict(os.environ, SNAP=str(model), OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1", USAGE_SAVE="0",
                 COLI_CUDA="0", COLI_VULKAN="0", COLI_LOOKUP="0", Q36_MTP="0")
        e.update(env)
        r = subprocess.run([str(ENGINE), "2", "8", str(ref)], env=e, capture_output=True, text=True,
                           errors="replace", timeout=300)
        log = r.stdout + r.stderr   # the exit status says whether the tokens matched ref.json's
        self.assertNotIn("Traceback", log)
        tokens = re.findall(r"^C engine : (.*?)(?: +Text|$)", log, re.M)
        self.assertTrue(tokens, "no token line:\n" + log[-2000:])
        return tokens[-1].split(), log

    def ref(self, model, full):
        path = model / "ref.json"
        path.write_text(json.dumps({"prompt_ids": PROMPT, "full_ids": PROMPT + full}))
        return path

    def test_the_head_drafts_and_every_token_is_plain_decodings(self):
        model = self.models["head"]
        plain, log = self.run_engine(model, self.ref(model, [0] * GEN))
        self.assertNotIn(HEAD, log)
        self.assertEqual(len(plain), GEN)
        # the plain run's tokens are the oracle the forced verifies draft from
        ref = self.ref(model, [int(t) for t in plain])
        runs = [("natural, one draft", {}), ("natural, three drafts", {"Q36_MTP_DRAFTS": "3"})]
        runs += [(f"forced {f}, three drafts", {"Q36_MTP_DRAFTS": "3", "Q36_MTP_FORCE": f}) for f in ("accept", "reject", "mixed")]
        for what, env in runs:
            with self.subTest(what):
                got, log = self.run_engine(model, ref, Q36_MTP="1", **env)
                self.assertIn(HEAD, log, log[-2000:])
                self.assertRegex(log, r"\[qwen36 mtp\] .*acceptance", log[-2000:])
                self.assertEqual(got, plain, f"{what}: the tokens moved")
                if env.get("Q36_MTP_FORCE") == "accept":   # every draft stood: fewer forwards than tokens
                    m = re.search(r"\[qwen36 mtp\] [^\n]*?\(\d+/(\d+) drafts\)", log)
                    self.assertTrue(m and int(m.group(1)) > 0, log[-2000:])

    def test_a_head_in_its_own_file_drafts_as_the_head_inside(self):
        model, inside = self.models["apart"], self.models["head"]
        self.assertIn("mtp", json.loads((model / "gguf.json").read_text()))
        plain, _ = self.run_engine(model, self.ref(model, [0] * GEN))
        ref = self.ref(model, [int(t) for t in plain])
        self.ref(inside, [int(t) for t in plain])
        for what, env in (("one draft", {}), ("three drafts", {"Q36_MTP_DRAFTS": "3"})):
            with self.subTest(what):
                got, log = self.run_engine(model, ref, Q36_MTP="1", **env)
                self.assertIn("[gguf] MTP head", log, log[-2000:])
                self.assertIn(HEAD, log, log[-2000:])
                self.assertEqual(got, plain, f"{what}: the tokens moved")
                want, wlog = self.run_engine(inside, inside / "ref.json", Q36_MTP="1", **env)
                self.assertEqual(got, want)
                stat = lambda t: re.search(r"\[qwen36 mtp\] [^\n]*?\(\d+/\d+ drafts\)", t).group(0)
                self.assertEqual(stat(log), stat(wlog), "the drafts differ from the head inside")

    def test_a_file_without_the_layer_runs_as_before(self):
        model = self.models["plain"]
        ref = self.ref(model, [0] * GEN)
        off, _ = self.run_engine(model, ref)
        on, log = self.run_engine(model, ref, Q36_MTP="1")
        self.assertNotIn(HEAD, log)
        self.assertEqual(on, off)


if __name__ == "__main__":
    unittest.main()
