"""#1533: cuda_binary() must inspect the engine that will run, not always the
GLM binary. On Windows a CUDA_DLL/HIP_DLL host is recognised by the backend
basename compiled into the loader, plus that file next to the engine. The
GLM/Qwen banner is not required: Kimi K3 CUDA_DLL builds link the same
loader without printing it, and a HIP host loads coli_hip.dll."""
import importlib.machinery
import importlib.util
import json
import os
import sys
import tempfile
import types
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

    def test_windows_kimi_cuda_dll_without_glm_banner(self):
        """Kimi K3 CUDA_DLL=1 links backend_loader.o and prints [K3-CUDA], not
        the GLM routed-experts banner. --gpu used to refuse that install."""
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d) / "kimi_k3.exe"
            eng.write_bytes(b"MZ [K3-CUDA] MXFP4 routed experts coli_cuda.dll")
            (Path(d) / "coli_cuda.dll").write_bytes(b"MZ")
            with mock.patch.object(self.m.sys, "platform", "win32"):
                self.assertTrue(self.m.cuda_binary(str(eng)))
            (Path(d) / "coli_cuda.dll").unlink()
            with mock.patch.object(self.m.sys, "platform", "win32"):
                self.assertFalse(self.m.cuda_binary(str(eng)))

    def test_windows_hip_host_accepts_coli_hip_dll(self):
        """backend_loader.c under HIP_DLL=1 loads coli_hip.dll. Requiring
        coli_cuda.dll refused a working HIP host that doctor already accepted."""
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d) / "colibri.exe"
            eng.write_bytes(b"MZ [CUDA] mode: routed experts coli_hip.dll")
            (Path(d) / "coli_hip.dll").write_bytes(b"MZ")
            with mock.patch.object(self.m.sys, "platform", "win32"):
                self.assertTrue(self.m.cuda_binary(str(eng)))
            (Path(d) / "coli_cuda.dll").write_bytes(b"MZ")
            (Path(d) / "coli_hip.dll").unlink()
            with mock.patch.object(self.m.sys, "platform", "win32"):
                self.assertFalse(self.m.cuda_binary(str(eng)),
                                 "a stray CUDA DLL must not satisfy a HIP host")

    def test_missing_engine_is_false(self):
        self.assertFalse(self.m.cuda_binary("/no/such/engine"))

    def test_kimi_gpu_flag_sets_k3_cuda(self):
        """--gpu writes COLI_CUDA=1. Kimi reads K3_CUDA, so the flag used to
        pass the build check and still leave experts on the CPU."""
        with tempfile.TemporaryDirectory() as d:
            model = Path(d)
            (model / "config.json").write_text(
                json.dumps({"model_type": "kimi_linear"}), encoding="utf-8")
            a = types.SimpleNamespace(
                model=str(model), gpu="0", vram=None, ram=0, ctx=None,
                ngen=None, temp=None, cap=None, auto_tier=False)
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(self.m, "cuda_binary", return_value=True):
                env = self.m.env_for_engine(a, "kimi")
            self.assertEqual(env["COLI_CUDA"], "1")
            self.assertEqual(env["K3_CUDA"], "1")

    def test_kimi_gpu_none_clears_k3_cuda(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d)
            (model / "config.json").write_text(
                json.dumps({"model_type": "kimi_linear"}), encoding="utf-8")
            a = types.SimpleNamespace(
                model=str(model), gpu="none", vram=None, ram=0, ctx=None,
                ngen=None, temp=None, cap=None, auto_tier=False)
            with mock.patch.dict(os.environ, {"K3_CUDA": "1"}, clear=True):
                env = self.m.env_for_engine(a, "kimi")
            self.assertEqual(env["COLI_CUDA"], "0")
            self.assertEqual(env["K3_CUDA"], "0")


if __name__ == "__main__":
    unittest.main()
