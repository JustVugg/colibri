"""COLI_DSV4_DLL as a path, and where the backend's CUDA runtime comes from (Windows, real GPU).

dsv4_load_one used to glue COLI_DSV4_DLL onto the engine's folder, so an
absolute path never loaded as given; the by-name fallback then found the DLL but
searched its CUDA runtime only in the engine's folder and System32, not PATH:
error 126, "continuing-CPU", a run ~3x slower with no error. This copies a real
coli_cuda_dsv4*.dll to a folder that is not the engine's, with the CUDA runtime
only on PATH, and checks that
  (a) COLI_DSV4_DLL=<that file> loads it, spelled with '/' and with '\\', and
      the log names the path it loaded;
  (b) with the CUDA runtime off PATH, the failure says a DLL it needs is missing
      and names the path that was tried;
  (c) a path longer than MAX_PATH says so;
  (d) the default (no COLI_DSV4_DLL) still loads the DLL beside the engine, and a
      fake CUDA runtime in the CWD is not loaded over the one on PATH;
  (e) a backend DLL planted in the CWD is not loaded for a bare name.

Needs DSV4_TEST_DLL (a built coli_cuda_dsv4_dg.dll / coli_cuda_dsv4.dll, only
read), an NVIDIA GPU, the CUDA runtime on PATH, the engine built and the
committed tiny fixture (copied); `make dsv4-dll-path-check DSV4_DLL=...` runs
it. Otherwise (test-python discovery) it skips.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BINARY = Path(os.environ.get("DSV4_TEST_BINARY", HERE / "deepseek_v4.exe")).resolve()
FIXTURE = Path(os.environ.get("DSV4_TEST_TINY", HERE / "deepseek_v4_tiny")).resolve()
DLL = os.environ.get("DSV4_TEST_DLL", "")
SYSTEM32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
RUNTIME_GLOBS = ("cudart64_*.dll", "cublasLt64_*.dll")


def runtime_files(folder):
    return [f for g in RUNTIME_GLOBS for f in Path(folder).glob(g)]


def runtime_dirs():
    """PATH entries that hold the CUDA runtime the backend imports."""
    return [d for d in os.environ.get("PATH", "").split(os.pathsep) if d and runtime_files(d)]


@unittest.skipUnless(sys.platform == "win32" and DLL and BINARY.is_file()
                     and (FIXTURE / "config.json").is_file(),
                     "set DSV4_TEST_DLL (make dsv4-dll-path-check)")
class Dsv4DllPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="dsv4-dll-path-")
        root = Path(cls.tmp.name)
        cls.dll = root / "backend" / Path(DLL).name
        cls.dll.parent.mkdir()
        shutil.copyfile(DLL, cls.dll)
        # an engine with the backend beside it: the default install layout
        cls.installed = root / "install" / BINARY.name
        cls.installed.parent.mkdir()
        shutil.copyfile(BINARY, cls.installed)
        shutil.copyfile(DLL, cls.installed.parent / Path(DLL).name)
        # a copy: the engine writes its expert-usage history into the model folder
        cls.fixture = root / "tiny"
        shutil.copytree(FIXTURE, cls.fixture)
        cls.cwd = root / "cwd"
        cls.cwd.mkdir()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_engine(self, dll=None, binary=None, path=None, cwd=None):
        # this test's own DSV4_TEST_* would be "[env] unknown variable" lines
        env = {k: v for k, v in os.environ.items()
               if k.upper() not in ("COLI_DSV4_DLL", "DSV4_CUDA") and not k.upper().startswith("DSV4_TEST_")}
        if dll is not None:
            env["COLI_DSV4_DLL"] = str(dll)
        if path is not None:
            env["PATH"] = path
        args = [str(binary or BINARY), str(self.fixture), "<t001><t002><t003>", "--raw-prompt", "--max-tokens", "2"]
        result = subprocess.run(args, env=env, cwd=cwd or self.cwd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=300, stdin=subprocess.DEVNULL)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stderr

    def test_runtime_only_on_path(self):
        # the precondition that made the old loader fail: no CUDA runtime beside the engine
        # or the DLL, nor in System32
        for folder in (BINARY.parent, self.dll.parent, SYSTEM32):
            self.assertFalse(runtime_files(folder), folder)
        self.assertTrue(runtime_dirs(), "no CUDA runtime (cudart64_*.dll) on PATH")

    def test_absolute_path_loads(self):
        for spelling in (self.dll.as_posix(), str(self.dll)):
            with self.subTest(spelling=spelling):
                log = self.run_engine(spelling)
                self.assertIn(f"backend={self.dll.name} (", log)
                self.assertIn(f"path={self.dll}", log)
                self.assertNotIn("backend-unavailable", log)

    def test_missing_runtime_is_named(self):
        cuda = {os.path.normcase(d) for d in runtime_dirs()}
        path = os.pathsep.join(d for d in os.environ["PATH"].split(os.pathsep)
                               if os.path.normcase(d) not in cuda)
        log = self.run_engine(self.dll.as_posix(), path=path)
        self.assertIn("present, but a DLL it needs is missing (the CUDA runtime cudart64_*.dll", log)
        self.assertIn(f"tried {self.dll}, Windows error 126", log)
        self.assertIn("v4_gpu warning=backend-unavailable; continuing-CPU", log)

    def test_too_long_path_is_named(self):
        long_dll = "C:/" + "d" * 300 + "/" + self.dll.name
        log = self.run_engine(long_dll)
        self.assertIn("path too long or invalid", log)
        self.assertIn("v4_gpu warning=backend-unavailable; continuing-CPU", log)

    def test_default_loads_beside_engine_not_cwd_runtime(self):
        # LOAD_WITH_ALTERED_SEARCH_PATH searches the CWD before PATH: a planted runtime there
        # (System32's version.dll under the runtime's names: loads, lacks every export) must lose
        cwd = Path(self.tmp.name) / "cwd-runtime"
        cwd.mkdir(exist_ok=True)
        for runtime in runtime_files(runtime_dirs()[0]):
            shutil.copyfile(SYSTEM32 / "version.dll", cwd / runtime.name)
        log = self.run_engine(binary=self.installed, cwd=cwd)
        self.assertIn(f"path={self.installed.parent / self.dll.name}", log)
        self.assertNotIn("backend-unavailable", log)

    def test_bare_name_ignores_cwd_dll(self):
        cwd = Path(self.tmp.name) / "cwd-backend"
        cwd.mkdir(exist_ok=True)
        self.assertFalse((BINARY.parent / self.dll.name).exists(), "the engine's folder has the DLL")
        shutil.copyfile(self.dll, cwd / self.dll.name)
        log = self.run_engine(self.dll.name, cwd=cwd)
        self.assertIn(f"tried {BINARY.parent / self.dll.name}", log)
        self.assertIn("v4_gpu warning=backend-unavailable; continuing-CPU", log)

    def test_relative_path_stays_under_engine(self):
        # a relative path resolves against the engine's folder, as before, never the CWD
        sub = self.installed.parent / "sub"
        sub.mkdir(exist_ok=True)
        shutil.copyfile(self.dll, sub / self.dll.name)
        log = self.run_engine("sub\\" + self.dll.name, binary=self.installed)
        self.assertIn(f"path={sub / self.dll.name}", log)
        self.assertNotIn("backend-unavailable", log)


if __name__ == "__main__":
    unittest.main()
