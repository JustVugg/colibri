"""hyper_connections.h: the results do not depend on the OpenMP team.

coli_hc_pre_team / coli_hc_post_team run the same loops on the default team,
on a small team or on the calling thread. That they give the same bits is
checked with the flags the engines are built with rather than argued: a
differently shaped loop is free to be vectorized and contracted differently
(on FMA targets it is), so the claim holds only as long as the loops stay the
same code, and this test is what says so.
"""
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).with_suffix(".c")


class HyperConnectionTeamTests(unittest.TestCase):
    def test_results_do_not_depend_on_the_team(self):
        cc = shlex.split(os.environ.get("CC", "cc"))
        if not shutil.which(cc[0]):
            self.skipTest("C compiler unavailable")
        variants = [["-O3"], ["-O3", "-fopenmp"], ["-O2", "-fopenmp"]]
        if platform.machine().lower() in ("x86_64", "amd64"):
            variants += [["-O3", "-march=x86-64-v3", "-fopenmp"],
                         ["-O3", "-march=native", "-fopenmp"]]
        passed = 0
        with tempfile.TemporaryDirectory() as tmp:
            for index, flags in enumerate(variants):
                with self.subTest(flags=" ".join(flags)):
                    binary = str(Path(tmp) / f"hc-team-{index}")
                    compiled = subprocess.run(
                        cc + ["-D_GNU_SOURCE"] + flags + [str(SOURCE), "-lm", "-o", binary],
                        capture_output=True, text=True)
                    if compiled.returncode != 0:
                        if flags == ["-O3"]:
                            self.fail(compiled.stderr)
                        continue  # a compiler without OpenMP or without the target
                    result = subprocess.run([binary], capture_output=True, text=True)
                    if result.returncode != 0 and not result.stdout and \
                            "-march=x86-64-v3" in flags:
                        continue  # a host without AVX2 cannot start this binary
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("test_hc_team: ok", result.stdout)
                    passed += 1
        self.assertGreater(passed, 0)


if __name__ == "__main__":
    unittest.main()
