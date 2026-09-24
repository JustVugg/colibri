"""backend_cuda_dsv4.o must be rebuilt when the nvcc command line changes.

The object is named after its source, but its contents come from
$(NVCC) $(V4_NVCCFLAGS): ``-arch=$(CUDA_ARCH)`` (or the -gencode preset), the
-DCOLI_DSV4_NO_TC guard and the DeepGEMM defines. Timestamps cannot tell those
apart, so without a record of the command make reports "is up to date" and
reuses the previous object (#1702 is the C-side case; #1707 fixed the units,
this is the nvcc side):

    make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=sm_86
    make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=sm_80
    -> no output at all, still the sm_86 engine

The parent Makefile records CUDA_ARCH in .build-config for this reason (#306);
the standalone build keeps its own stamp, deepseek_v4.cudaflags.

The nvcc here is a stand-in that only creates its -o target and appends its own
command line to a log, so the test needs make and sh but no CUDA toolkit and no
GPU, and it checks exactly what make decides to recompile. The sources are
copied, not linked, so their timestamps can be pinned without touching the tree.
"""
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
MAKE = shutil.which("make")
OBJ = "backend_cuda_dsv4.o"
STAMP = "deepseek_v4.cudaflags"

FAKE_NVCC = """#!/bin/sh
# Stand-in nvcc: records the command line, creates the -o target, compiles nothing.
printf '%s\\n' "$*" >> "$FAKE_NVCC_LOG"
while [ "$#" -gt 0 ]; do
    if [ "$1" = "-o" ]; then : > "$2"; exit 0; fi
    shift
done
exit 1
"""


@unittest.skipUnless(MAKE and os.name == "posix", "make and a POSIX shell are required")
class DeepseekV4CudaArchFlagsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        for pattern in ("Makefile.deepseek-v4*", "deepseek_v4.c", "*.h", "*.inc", "*.cu"):
            for src in C_DIR.glob(pattern):
                shutil.copy(src, self.dir / src.name)
        self.nvcc = self.dir / "fake-nvcc.sh"
        self.nvcc.write_text(FAKE_NVCC, encoding="utf-8")
        self.nvcc.chmod(0o755)
        self.log = self.dir / "nvcc.log"
        # Sources well in the past: every object built below is newer than all of
        # them, so the only thing left that can make the object stale is a flag.
        past = time.time() - 1000
        for f in self.dir.iterdir():
            os.utime(f, (past, past))
        probe = self.make("print-v4-objs")
        if probe.returncode != 0:
            self.skipTest("Makefile.deepseek-v4 does not support this host: "
                          + probe.stderr.strip()[-200:])

    def tearDown(self):
        self._tmp.cleanup()

    def make(self, *args):
        env = dict(os.environ, FAKE_NVCC_LOG=str(self.log))
        return subprocess.run(
            [MAKE, "-f", "Makefile.deepseek-v4", f"NVCC={self.nvcc}", *args],
            cwd=self.dir, text=True, capture_output=True, check=False,
            timeout=120, env=env)

    def nvcc_calls(self):
        return len(self.log.read_text(encoding="utf-8").split("\n")) - 1 if self.log.exists() else 0

    def build(self, *variables):
        """Build the object, then age it so only a flags change can rebuild it.

        Leaves the object and the stamp older than now but newer than the
        sources, so a later build recompiles only if make finds a real reason,
        never because two writes landed in the same clock tick.
        """
        self.log.write_text("", encoding="utf-8")
        result = self.make(OBJ, *variables)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.nvcc_calls()
        aged = time.time() - 500
        for f in self.dir.iterdir():
            if f.stat().st_mtime > aged + 400:
                os.utime(f, (aged, aged))
        return calls

    def test_same_arch_does_not_rebuild(self):
        """The guard: a fix that always recompiles would pass the tests below."""
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_86"), 1)
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_86"), 0,
                         f"{OBJ} was recompiled with an unchanged nvcc command line")

    def test_changed_arch_rebuilds(self):
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_86"), 1)
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_80"), 1,
                         f"{OBJ} built for sm_86 was reused by an sm_80 build: the "
                         "engine links an object for the wrong architecture")

    def test_changed_gencode_preset_rebuilds(self):
        """The presets change -gencode just as -arch does."""
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_86"), 1)
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=portable"), 1,
                         f"{OBJ} was reused after the gencode preset changed")

    def test_changed_no_tc_rebuilds(self):
        """NO_TC adds -DCOLI_DSV4_NO_TC to the nvcc line, same object name."""
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_86", "NO_TC=0"), 1)
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_86", "NO_TC=1"), 1,
                         f"{OBJ} was reused after NO_TC changed")

    def test_a_dry_run_writes_nothing(self):
        """make -n must not create build state. The stamp is written by a recipe,
        not while the Makefile is read, so a dry run only prints it."""
        dry = self.make("-n", OBJ, "CUDA=1", "CUDA_ARCH=sm_86")
        self.assertEqual(dry.returncode, 0, dry.stdout + dry.stderr)
        self.assertFalse((self.dir / STAMP).exists(),
                         f"make -n created {STAMP}")
        self.assertEqual(self.nvcc_calls(), 0, "make -n ran nvcc")
        clean = self.make("-n", "deepseek-v4-clean")
        self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)
        self.assertFalse((self.dir / STAMP).exists(),
                         f"make -n deepseek-v4-clean created {STAMP}")

    def test_clean_writes_nothing(self):
        """A clean recipe that removes the stamp must not recreate it first."""
        self.build("CUDA=1", "CUDA_ARCH=sm_86")
        result = self.make("deepseek-v4-clean")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.dir / STAMP).exists(),
                         f"{STAMP} survived deepseek-v4-clean")

    def test_clean_removes_the_stamp(self):
        """A stale stamp after `make clean` would make the next build a no-op."""
        self.build("CUDA=1", "CUDA_ARCH=sm_86")
        subprocess.run([MAKE, "-f", "Makefile.deepseek-v4", "deepseek-v4-clean"],
                       cwd=self.dir, text=True, capture_output=True, check=False,
                       timeout=120)
        self.assertFalse((self.dir / STAMP).exists(),
                         f"{STAMP} survived deepseek-v4-clean")


if __name__ == "__main__":
    unittest.main()
