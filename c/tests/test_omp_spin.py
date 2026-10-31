"""omp_spin.c serves the OpenMP regions of an engine linked with it, as libgomp would.

The Makefile links mimo on MinGW with omp_spin.c and --wrap for the calls listed in
OMP_SPIN_WRAP. This builds tests/omp_spin_check.c the same way and runs it with the team
spinning between regions, with the team sleeping after every region (the wake-up path),
and with libgomp's team (COLI_OMP_SPIN=0): every loop's iterations once, barriers that
hold, regions inside tasks run as teams of one, a second thread's regions beside the
first's. Then it compiles mimo.c as each build does (a VK=1 build, the Windows release's,
also links the Vulkan objects) and checks that every libgomp entry point they call is
one the file serves or one that is safe without a team (a call to single, sections,
ordered, tasks or another schedule would run on every thread of a region here).
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
# libgomp's, safe from any thread: they read or set the thread count and the nesting level
UNWRAPPED_OK = {"omp_get_max_threads", "omp_set_num_threads", "omp_get_level"}


def wrapped():
    text = (C_DIR / "Makefile").read_text(encoding="utf-8").replace("\\\n", " ")
    names = re.search(r"^OMP_SPIN_WRAP\s*=\s*(.+)$", text, re.M).group(1).split()
    assert "GOMP_parallel" in names, names
    return names


def gnu_gcc():
    gcc = shutil.which("gcc")
    if not gcc:
        return None
    version = subprocess.run([gcc, "--version"], capture_output=True, text=True).stdout
    return gcc if "clang" not in version.lower() else None


class OmpSpin(unittest.TestCase):
    def setUp(self):
        self.gcc = gnu_gcc()
        if not self.gcc:
            self.skipTest("GNU gcc not on PATH")

    def test_regions(self):
        flags = [f"-Wl,--wrap={name}" for name in wrapped()]
        # libgomp linked in as the engine links it: a libgomp-1.dll build hangs now and then
        # with two threads opening regions at once, with or without this file
        static = ["-static"] if os.name == "nt" else []
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "omp_spin_check.exe"
            subprocess.run([self.gcc, "-O2", "-fopenmp", "-Wall", str(C_DIR / "tests" / "omp_spin_check.c"),
                            str(C_DIR / "omp_spin.c"), "-o", str(exe), *static, "-lpthread", *flags], check=True)
            for extra in ({}, {"COLI_OMP_SPIN_US": "0"}, {"COLI_OMP_SPIN": "0"}):
                with self.subTest(**extra):
                    env = {k: v for k, v in os.environ.items() if not k.startswith("COLI_OMP_SPIN")}
                    env.update(extra, OMP_NUM_THREADS="8")
                    # a second thread's regions: the spinning team's fallback to libgomp
                    two = [] if "COLI_OMP_SPIN" in extra else ["--two"]
                    run = subprocess.run([str(exe), *two], capture_output=True, text=True, timeout=600, env=env)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    self.assertEqual(run.stdout.splitlines()[-1], "ok", run.stdout)
                    self.assertEqual("spinning team" in run.stderr, "COLI_OMP_SPIN" not in extra, run.stderr)

    def test_mimo_calls_only_what_is_served(self):
        nm = subprocess.run([self.gcc, "-print-prog-name=nm"], capture_output=True, text=True).stdout.strip()
        nm = shutil.which(nm) or shutil.which("nm")
        if not nm:
            self.skipTest("nm not found beside gcc")
        served = set(wrapped()) | UNWRAPPED_OK
        # mimo.c and kimi_k3.c in the builds that link omp_spin.o (plain and CUDA). A VK=1
        # build links libgomp's own team (the Makefile: vk_tier.c's schedule(runtime) loops),
        # so there the sources only have to compile (backend_vulkan.c and vk_chain.c need the
        # Vulkan headers to compile).
        vk = ["-DCOLI_VULKAN"]
        objects = [("mimo.c", []), ("mimo.c", ["-DCOLI_MIMO_CUDA"]), ("mimo.c", vk),
                   ("kimi_k3.c", []), ("kimi_k3.c", ["-DCOLI_CUDA"]), ("kimi_k3.c", vk),
                   ("vk_tier.c", vk), ("backend_vulkan.c", vk), ("vk_chain.c", vk)]
        with tempfile.TemporaryDirectory() as tmp:
            obj = Path(tmp) / "object.o"
            for src, defs in objects:
                with self.subTest(src=src, defs=defs):
                    build = subprocess.run([self.gcc, "-O1", "-fopenmp", "-D_FILE_OFFSET_BITS=64", *defs, "-c",
                                            str(C_DIR / src), "-o", str(obj)], capture_output=True, text=True, cwd=C_DIR)
                    if build.returncode and "vulkan/vulkan.h" in build.stderr:
                        self.skipTest(f"{src}: no Vulkan headers")
                    self.assertEqual(build.returncode, 0, build.stderr)
                    undefined = subprocess.run([nm, "-u", str(obj)], capture_output=True, text=True, check=True).stdout
                    calls = {line.split()[-1].lstrip("_") for line in undefined.splitlines()
                             if re.search(r"\b_?(GOMP_|omp_)", line)}
                    calls = {c for c in calls if c.startswith(("GOMP_", "omp_"))}
                    if src in ("mimo.c", "kimi_k3.c"):
                        self.assertIn("GOMP_parallel", calls)
                    if "-DCOLI_VULKAN" in defs:
                        continue
                    self.assertEqual(calls - served, set(), f"{src} {defs} calls libgomp entry points omp_spin.c does not serve")


if __name__ == "__main__":
    unittest.main()
