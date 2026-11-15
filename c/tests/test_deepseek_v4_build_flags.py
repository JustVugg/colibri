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
import sys
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
# GNU make: the BSDs' own make cannot read this Makefile, and they install GNU make as gmake.
MAKE = shutil.which("gmake" if sys.platform.startswith(("freebsd", "openbsd", "netbsd", "dragonfly")) else "make")
UNIT = "COLI_V4_UNIT_MATH.o"

FAKE_CC = """#!/bin/sh
output=
depfile=
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o) output="$2"; shift ;;
        -MF) depfile="$2"; shift ;;
    esac
    shift
done
[ -n "$output" ] || exit 1
if [ -n "$depfile" ]; then
    header=deepseek_v4.h
    case "$output" in
        COLI_V4_UNIT_MATH.o) header=hyper_connections.h ;;
        backend_vulkan_v4.o) header=vk_memtype.h ;;
    esac
    printf '%s: %s\\n%s:\\n' "$output" "$header" "$header" > "$depfile"
fi
: > "$output"
"""


@unittest.skipUnless(MAKE and os.name == "posix", "make and a POSIX shell are required")
class DeepseekV4BuildFlagsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        for pattern in ("Makefile.deepseek-v4*", "*.c", "*.h", "*.inc"):
            for src in C_DIR.glob(pattern):
                shutil.copy(src, self.dir / src.name)
        shutil.copytree(C_DIR / "shaders", self.dir / "shaders")
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
        result = self.make_aged(UNIT, *variables)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return "-c deepseek_v4.c" in result.stdout

    def make_aged(self, *args):
        """make, then age every file it wrote, as build() needs. A make whose
        clock has whole seconds (macOS's 3.81) sees a stamp written by the next
        call in the same second as no newer than what this call linked."""
        before = {f: f.stat().st_mtime for f in self.dir.iterdir()}
        result = self.make(*args)
        aged = time.time() - 500
        for f in self.dir.iterdir():
            if before.get(f) != f.stat().st_mtime:
                os.utime(f, (aged, aged))
        return result

    def test_math_header_rebuilds(self):
        self.assertTrue(self.build())
        os.utime(self.dir / "hyper_connections.h", (time.time() - 400,) * 2)
        self.assertTrue(self.build())
        os.utime(self.dir / "hyper_connections.h", (time.time() - 1000,) * 2)
        self.assertFalse(self.build())

    def test_missing_depfile_rebuilds(self):
        self.assertTrue(self.build())
        depfile = self.dir / UNIT.replace(".o", ".d")
        depfile.unlink(missing_ok=True)
        self.assertTrue(self.build())
        self.assertTrue(depfile.exists())
        self.assertFalse(self.build())

    def test_vulkan_header_rebuilds(self):
        target = "backend_vulkan_v4.o"
        first = self.make(target)
        self.assertEqual(first.returncode, 0, first.stderr)
        past = time.time() - 100
        for name in (target, "deepseek_v4.cflags", "backend_vulkan_v4.d"):
            path = self.dir / name
            if path.exists():
                os.utime(path, (past, past))
        os.utime(self.dir / "vk_memtype.h", None)
        result = self.make(target)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-c backend_vulkan.c", result.stdout)

    def test_link_flags_relink_without_recompiling(self):
        first = self.make_aged("deepseek-v4")
        self.assertEqual(first.returncode, 0, first.stderr)
        result = self.make_aged("deepseek-v4", "EXTRA_LDFLAGS=-Wl,--as-needed")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-Wl,--as-needed", result.stdout)
        self.assertNotIn(" -c ", result.stdout)
        unchanged = self.make("deepseek-v4", "EXTRA_LDFLAGS=-Wl,--as-needed")
        self.assertEqual(unchanged.returncode, 0, unchanged.stderr)
        self.assertNotIn(" -o ", unchanged.stdout)

    def test_clean_removes_disabled_backend_artifacts(self):
        artifacts = ("backend_cuda_dsv4.o", "backend_cuda_dsv4.d",
                     "backend_loader_dsv4.o", "backend_loader_dsv4.d",
                     "backend_vulkan_v4.o", "backend_vulkan_v4.d",
                     "deepseek_v4.ldflags", "deepseek_v4.exe")
        for name in artifacts:
            (self.dir / name).touch()
        result = self.make("deepseek-v4-clean")
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in artifacts:
            self.assertFalse((self.dir / name).exists(), name)

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
