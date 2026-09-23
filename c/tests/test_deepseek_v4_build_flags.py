"""Makefile.deepseek-v4 must rebuild its objects when the build flags change.

The unit objects are named after the unit (COLI_V4_UNIT_MATH.o), not after the
flags they were compiled with, and three builds share those names in c/: the
engine (`CUDA=1` adds -DCOLI_V4_GPU_TIER), the parent's `make test-c` (no GPU
tier) and `make test-asan` (EXTRA_CFLAGS). Timestamps cannot tell them apart,
so without a record of the flags make reports "is up to date" and links the
other build's objects (#1702):

    make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1
    make test-c
    -> undefined reference to `coli_v4_gpu_matvec_grouped'

and, the silent direction, `make test-c` followed by the CUDA=1 engine build
links CPU-only units into an engine that was asked for the GPU tier.

The compiler here is a stand-in that only creates its -o target, so the test
needs make and sh but no toolchain, and checks exactly what make decides to
rebuild. The sources are copied, not linked, so their timestamps can be pinned
without touching the tree.
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
UNIT = "COLI_V4_UNIT_MATH.o"

FAKE_CC = """#!/bin/sh
# Stand-in compiler: creates the -o target and compiles nothing.
while [ "$#" -gt 0 ]; do
    if [ "$1" = -o ]; then : > "$2"; exit 0; fi
    shift
done
exit 1
"""


@unittest.skipUnless(MAKE and os.name == "posix", "make and a POSIX shell are required")
class DeepseekV4BuildFlagsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        for pattern in ("Makefile.deepseek-v4*", "deepseek_v4.c", "*.h", "*.inc"):
            for src in C_DIR.glob(pattern):
                shutil.copy(src, self.dir / src.name)
        fake = self.dir / "fakecc.sh"
        fake.write_text(FAKE_CC, encoding="utf-8")
        self.cc = f"sh {fake}"
        # Sources well in the past: every object built below is newer than all
        # of them, so the only thing left that can make it stale is the flags.
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
        return subprocess.run(
            [MAKE, "-f", "Makefile.deepseek-v4", f"CC={self.cc}", *args],
            cwd=self.dir, text=True, capture_output=True, check=False, timeout=120)

    def build(self, *variables):
        """Build the unit, then age everything it produced by the same amount.

        Leaves every file this build wrote at one timestamp, older than now
        and newer than the sources: a later build only recompiles if make
        finds a real reason to, never because two writes landed in the same
        clock tick.
        """
        before = {f: f.stat().st_mtime for f in self.dir.iterdir()}
        result = self.make(UNIT, *variables)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        aged = time.time() - 500
        for f in self.dir.iterdir():
            if before.get(f) != f.stat().st_mtime:
                os.utime(f, (aged, aged))
        return "-c deepseek_v4.c" in result.stdout

    def test_same_flags_do_not_rebuild(self):
        """The guard: a fix that always recompiles would pass the tests below."""
        self.assertTrue(self.build("EXTRA_CFLAGS=-DCOLI_TEST_FLAGS_A"))
        self.assertFalse(self.build("EXTRA_CFLAGS=-DCOLI_TEST_FLAGS_A"),
                         f"{UNIT} was recompiled with unchanged flags")

    def test_changed_flags_rebuild(self):
        """EXTRA_CFLAGS is how the parent's test-asan reaches this Makefile."""
        self.assertTrue(self.build("EXTRA_CFLAGS=-DCOLI_TEST_FLAGS_A"))
        self.assertTrue(self.build("EXTRA_CFLAGS=-DCOLI_TEST_FLAGS_B"),
                        f"{UNIT} was reused after EXTRA_CFLAGS changed: the "
                        "objects of one build configuration leak into another")

    def test_cuda_tier_objects_are_not_reused_by_a_cpu_build(self):
        """The reported case, in both directions (#1702). Building one unit
        object needs no nvcc: only backend_cuda_dsv4.o does."""
        self.assertTrue(self.build("CUDA=1"))
        self.assertTrue(self.build(),
                        f"{UNIT} built with CUDA=1 was reused by a CPU build")
        self.assertTrue(self.build("CUDA=1"),
                        f"{UNIT} built without the GPU tier was reused by CUDA=1")


if __name__ == "__main__":
    unittest.main()
