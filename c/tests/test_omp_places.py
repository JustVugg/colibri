"""The engines that re-exec with OMP_PROC_BIND=close also set OMP_PLACES=cores.

Without OMP_PLACES libgomp makes one place per hardware thread. Where a core's SMT
siblings are numbered side by side (Intel's hybrid CPUs on Linux: CPUs 0 and 1 are
P-core 0) a close-bound team of the physical-core count sat on half the cores.
"""
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

C = Path(__file__).resolve().parent.parent


class OmpPlacesTest(unittest.TestCase):
    def test_every_close_binding_comes_with_core_places(self):
        seen = 0
        for src in sorted(C.glob("*.c")) + [C / "omp_tune.h"]:
            text = src.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r'setenv\("OMP_PROC_BIND"\s*,\s*"close"\s*,\s*0\)', text):
                seen += 1
                block = text[m.start():m.start() + 1200]
                self.assertRegex(block, r'setenv\("OMP_PLACES"\s*,\s*"cores"\s*,\s*0\)',
                                 "%s: OMP_PROC_BIND=close without OMP_PLACES=cores" % src.name)
        self.assertGreaterEqual(seen, 3)   # colibri.c, inkling.c and omp_tune.h's hot team


@unittest.skipUnless(sys.platform.startswith("linux") and (C / "olmoe").exists(), "a Linux build of olmoe")
class HotTeamTest(unittest.TestCase):
    """olmoe runs itself again once with the hot team (omp_tune.h), not with CUDA on, not under
    COLI_NO_OMP_TUNE; the help it prints without SNAP is enough to see it."""

    def run_olmoe(self, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "GOMP_", "KMP_", "COLI_", "SNAP"))}
        env.update(extra)
        return subprocess.run([str(C / "olmoe")], env=env, capture_output=True, text=True, timeout=120).stderr

    def test_once(self):
        self.assertEqual(self.run_olmoe().count("hot-thread tuning, re-exec once"), 1)

    def test_not_with_cuda_or_the_kill_switch(self):
        self.assertNotIn("hot-thread tuning", self.run_olmoe(COLI_CUDA="1"))
        self.assertNotIn("hot-thread tuning", self.run_olmoe(COLI_NO_OMP_TUNE="1"))


if __name__ == "__main__":
    unittest.main()
