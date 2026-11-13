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
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
MAKE = shutil.which("make")
OBJ = "backend_cuda_dsv4.o"
STAMP = "deepseek_v4.cudaflags"

FAKE_NVCC = """#!/bin/sh
if [ "$1" = "--version" ]; then
    echo "Cuda compilation tools, release ${FAKE_NVCC_RELEASE:-12.8}, V0.0.0"
    exit 0
fi
# Stand-in nvcc: records the command line, creates the -o target, compiles nothing.
printf '%s\\n' "$*" >> "$FAKE_NVCC_LOG"
depfile=
output=
while [ "$#" -gt 0 ]; do
    case "$1" in
        -MF) depfile="$2"; shift ;;
        -o) output="$2"; shift ;;
    esac
    shift
done
[ -n "$output" ] || exit 1
: > "$output"
if [ -n "$depfile" ]; then printf '%s: backend_cuda_dsv4.cu\\n' "$output" > "$depfile"; fi
exit 0
"""


@unittest.skipUnless(MAKE and os.name == "posix", "make and a POSIX shell are required")
class DeepseekV4CudaArchFlagsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        for pattern in ("Makefile.deepseek-v4*", "deepseek_v4.c", "*.h", "*.inc", "*.cu", "*.cuh"):
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
        env = dict(os.environ, FAKE_NVCC_LOG=str(self.log),
                   FAKE_NVCC_RELEASE=getattr(self, "release", "12.8"))
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

    def test_transitive_header_dependency_rebuilds(self):
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_86"), 1)
        self.assertIn("-MMD -MP -MF backend_cuda_dsv4.d", self.log.read_text())
        # Simulate nvcc's depfile for a transitively included library header.
        header = self.dir / "library_detail.cuh"
        header.write_text("/* library implementation */\n")
        (self.dir / "backend_cuda_dsv4.d").write_text(
            OBJ + ": library_detail.cuh\nlibrary_detail.cuh:\n")
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_86"), 1)
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_86"), 0)
        clean = self.make("deepseek-v4-clean")
        self.assertEqual(clean.returncode, 0, clean.stderr)
        self.assertFalse((self.dir / "backend_cuda_dsv4.d").exists())

    def test_missing_depfile_rebuilds_existing_object(self):
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_86"), 1)
        depfile = self.dir / "backend_cuda_dsv4.d"
        depfile.unlink()
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_86"), 1)
        self.assertTrue(depfile.exists())
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_86"), 0)

    def test_old_or_unknown_toolkit_rejected_before_compilation(self):
        for release in ("11.8", "12.7", "unknown"):
            with self.subTest(release=release):
                self.release = release
                result = self.make(OBJ, "CUDA=1", "CUDA_ARCH=sm_86")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires CUDA toolkit >= 12.8", result.stderr)
                self.assertIn("NO_TC=1", result.stderr)
                self.assertFalse((self.dir / OBJ).exists())
                self.assertEqual(self.nvcc_calls(), 0)

    def test_new_toolkit_accepted(self):
        self.release = "13.0"
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_86"), 1)

    def test_no_tc_skips_toolkit_requirement(self):
        self.release = "unknown"
        self.assertEqual(self.build("CUDA=1", "CUDA_ARCH=sm_86", "NO_TC=1"), 1)
        self.assertIn("-DCOLI_DSV4_NO_TC", self.log.read_text())

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


class PreAmperePresetToolkitTest(unittest.TestCase):
    """CUDA 13 dropped compute_61, the oldest target of CUDA_ARCH=portable-pre-ampere,
    so nvcc 13 stops at once with "Unsupported gpu architecture 'compute_61'". Every
    place that sends Pascal and Turing cards to the preset names the last toolkit
    that still builds it. Reads files only: no make, no CUDA."""

    def test_every_hint_names_the_last_toolkit_that_builds_the_preset(self):
        sys.path.insert(0, str(C_DIR))
        import resource_plan
        import setup_flow
        makefile = (C_DIR / "Makefile.deepseek-v4").read_text(encoding="utf-8")
        preset = makefile.split("else ifeq ($(CUDA_ARCH),portable-pre-ampere)", 1)[1].split("\nelse\n", 1)[0]
        oldest = min(int(sm) for sm in re.findall(r"arch=compute_(\d+)", preset))
        # setup_flow's measured table of what each nvcc release lists
        builds = [first for first, low, high in setup_flow.CUDA_RELEASES if low <= oldest <= high]
        self.assertNotEqual(builds[-1], setup_flow.CUDA_RELEASES[-1][0],
                            f"compute_{oldest} is still built by the newest CUDA release")
        limit = "CUDA <= %d.%d" % builds[-1]
        hints = {"Makefile.deepseek-v4 preset": [preset],
                 "resource_plan.CUDA_TIER_FLOOR": [resource_plan.CUDA_TIER_FLOOR["deepseek_v4"][1]]}
        for name in ("backend_loader_dsv4.c", "backend_cuda_dsv4.cu", "../docs/deepseek-v4.md"):
            text = " ".join((C_DIR / name).read_text(encoding="utf-8").split())
            hints[name] = re.findall(r"portable-pre-ampere NO_TC=1.{0,40}", text)
        for where, found in hints.items():
            with self.subTest(where=where):
                self.assertTrue(found)
                missing = [hint[:80] for hint in found if limit not in hint]
                self.assertEqual(missing, [], f"recommends the preset without '{limit}'")


if __name__ == "__main__":
    unittest.main()
