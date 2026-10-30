"""The engines that re-exec with OMP_PROC_BIND=close also set OMP_PLACES=cores.

Without OMP_PLACES libgomp makes one place per hardware thread. Where a core's SMT
siblings are numbered side by side (Intel's hybrid CPUs on Linux: CPUs 0 and 1 are
P-core 0) a close-bound team of the physical-core count sat on half the cores.
"""
import re
import unittest
from pathlib import Path

C = Path(__file__).resolve().parent.parent


class OmpPlacesTest(unittest.TestCase):
    def test_every_close_binding_comes_with_core_places(self):
        seen = 0
        for src in sorted(C.glob("*.c")):
            text = src.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r'setenv\("OMP_PROC_BIND"\s*,\s*"close"\s*,\s*0\)', text):
                seen += 1
                block = text[m.start():m.start() + 1200]
                self.assertRegex(block, r'setenv\("OMP_PLACES"\s*,\s*"cores"\s*,\s*0\)',
                                 "%s: OMP_PROC_BIND=close without OMP_PLACES=cores" % src.name)
        self.assertGreaterEqual(seen, 2)   # colibri.c and inkling.c


if __name__ == "__main__":
    unittest.main()
