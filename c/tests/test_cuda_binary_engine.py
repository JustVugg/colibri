"""#1533: cuda_binary() must inspect the engine that will run, not always the
GLM binary. On Windows a CUDA_DLL build is recognised by a marker string in
the executable plus coli_cuda.dll next to it; a qwen36 build that carries
both must pass even when colibri.exe is CPU-only or absent."""
import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

CLI = Path(__file__).resolve().parent.parent / "coli"


def load_cli():
    loader = importlib.machinery.SourceFileLoader("coli_cuda_binary_t", str(CLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = ["coli"]
    try:
        loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


class CudaBinaryEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_cli()

    def test_windows_probe_reads_the_given_engine(self):
        with tempfile.TemporaryDirectory() as d:
            qwen = Path(d) / "qwen36.exe"
            qwen.write_bytes(b"MZ...[CUDA] mode: routed experts (qwen36 VRAM tier)...")
            (Path(d) / "coli_cuda.dll").write_bytes(b"MZ")
            glm = Path(d) / "colibri.exe"
            glm.write_bytes(b"MZ cpu only")
            with mock.patch.object(self.m.sys, "platform", "win32"), \
                 mock.patch.object(self.m, "GLM", str(glm)):
                self.assertTrue(self.m.cuda_binary(str(qwen)))
                self.assertFalse(self.m.cuda_binary())          # the default is still the GLM binary
                self.assertFalse(self.m.cuda_binary(str(glm)))
            (Path(d) / "coli_cuda.dll").unlink()
            with mock.patch.object(self.m.sys, "platform", "win32"):
                self.assertFalse(self.m.cuda_binary(str(qwen)), "no DLL next to the engine")

    def test_missing_engine_is_false(self):
        self.assertFalse(self.m.cuda_binary("/no/such/engine"))


if __name__ == "__main__":
    unittest.main()
