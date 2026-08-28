"""The K3_VALIDATE_OUT fallback must not be rooted at /tmp.

kimi_k3.c defaulted K3_VALIDATE_LAYER's dump target to "/tmp/k3_val" when
K3_VALIDATE_OUT was unset. Same rule the repo already states twice in its own
source: tests/test_stops.c and tests/test_pipe_block.c both say a path must be
relative to the CWD and not rooted at /tmp, because the windows job builds
native .exe files that resolve Windows paths, /tmp is not one, and the whole
`make check` goes red on the windows job and only there. setup.sh was moved off
its own /tmp probe for the same reason.

fopen(..., "w") creates the file but not the directory, so the old default
worked wherever /tmp already existed and failed where it did not.

test_reports_the_path_it_chose is the one that makes the behaviour change safe
rather than merely correct: the default moved, so anyone who relied on the old
one has to be told where the dump went. Both branches print it, so they are.

The source is read as text because building kimi_k3.c needs the full toolchain
and running this path needs a K3 checkpoint, neither of which the python job
has. Same approach as test_cuda_test_makefile's setup.sh probe test.
"""
import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "kimi_k3.c"


class K3ValidateOutPathTest(unittest.TestCase):
    def setUp(self):
        self.src = SRC.read_text(encoding="utf-8")
        # Scope to the K3-VAL init block so an unrelated /tmp elsewhere in this
        # 1000+ line file does not silently pass or fail this test.
        start = self.src.index('getenv("K3_VALIDATE_OUT")')
        self.block = self.src[start:start + 1200]

    def test_fallback_is_not_rooted_at_tmp(self):
        hits = re.findall(r'ofn\s*=\s*"(/tmp/[^"]*)"', self.block)
        self.assertEqual(hits, [], f"K3_VALIDATE_OUT falls back to a /tmp path: {hits}")

    def test_fallback_is_cwd_relative(self):
        m = re.search(r'if\s*\(\s*!\s*ofn\s*\)\s*ofn\s*=\s*"([^"]*)"', self.block)
        self.assertIsNotNone(m, "no K3_VALIDATE_OUT fallback assignment found")
        path = m.group(1)
        self.assertFalse(path.startswith("/"), f"fallback {path!r} is absolute, not CWD-relative")
        self.assertFalse(re.match(r"^[A-Za-z]:", path), f"fallback {path!r} is a Windows drive path")
        self.assertTrue(path, "fallback is an empty path")

    def test_reports_the_path_it_chose(self):
        """The default moved, so the user must be told where the dump landed.
        Both the success and the failure branch print it."""
        self.assertIn("cannot open %s", self.block)
        self.assertIn("output %s", self.block)


if __name__ == "__main__":
    unittest.main()
