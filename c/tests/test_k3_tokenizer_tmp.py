"""k3_tokenizer.py --ctest must not write to a hardcoded /tmp path.

Same bug the repo already fixed twice elsewhere, surviving in a third file.

tests/test_stops.c and tests/test_pipe_block.c both carry the rule in the
source: 'Relative to the CWD [...] NOT "/tmp/..."', because the windows job
builds native .exe files that resolve Windows paths, "/tmp" is not one, and
"the whole `make check` goes red on the windows job (and only there)".
setup.sh was moved off its own /tmp probe to OMP_PROBE=".colibri-omp-probe-$$"
for the same reason, and test_cuda_test_makefile.py's
test_setup_openmp_probe_does_not_require_tmp pins it there. This file is that
same pin for tools/k3_tokenizer.py.

Note the failure is conditional rather than universal: fopen(..., "w") creates
the FILE but not the directory, so on a box where something has already created
C:/tmp (Git Bash does) the old code writes fine, and on a clean runner it does
not. Measured both ways rather than assumed.

A fixed FILENAME is a second, platform-independent bug: two concurrent runs
collide on it, and nothing ever deleted it.

The source is read as text rather than executed because --ctest needs tiktoken
and a compiled C binary, neither of which CI has here.
"""
import re
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent / "tools" / "k3_tokenizer.py"


class K3TokenizerTempPathTest(unittest.TestCase):
    def setUp(self):
        self.src = TOOL.read_text(encoding="utf-8")

    def test_no_hardcoded_tmp_path(self):
        hits = re.findall(r'["\']/tmp/[^"\']*["\']', self.src)
        self.assertEqual(hits, [], f"hardcoded /tmp path(s) back in k3_tokenizer.py: {hits}")

    def test_uses_tempfile_for_the_ctest_cases(self):
        # assertTrue, not assertIn: assertIn dumps the whole 8 KB source into
        # the failure message, which buries the actual reason in CI output.
        self.assertTrue("import tempfile" in self.src, "tempfile is not imported")
        self.assertTrue("tempfile.mkstemp" in self.src,
                        "the --ctest cases file is not created via tempfile.mkstemp")

    def test_the_cases_file_is_cleaned_up(self):
        """os.unlink must sit in a finally, so a ctest that raises still tidies
        up rather than leaving the file behind."""
        self.assertTrue("os.unlink(cases)" in self.src,
                        "the --ctest cases file is never removed")
        body = self.src[self.src.index("tempfile.mkstemp"):]
        finally_at = body.find("finally:")
        unlink_at = body.find("os.unlink(cases)")
        self.assertNotEqual(finally_at, -1, "no finally: guarding the cleanup")
        self.assertLess(finally_at, unlink_at, "os.unlink is not inside the finally block")


if __name__ == "__main__":
    unittest.main()
