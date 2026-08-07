"""kimi_k3's command line must be discoverable, and its refusals must be useful.

`--help` used to fall through as the model directory: the engine went looking
for `--help/config.json` and a user could only learn the argument order by
reading 1,800 lines of C. Reported on #783 by someone who had already got the
build right and was stopped by the interface.

`no prompt and no --ids` is accurate and useless on its own — an error should
say what to do, not only what went wrong.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("kimi_k3.exe" if sys.platform == "win32" else "kimi_k3")


def run(*args, **env):
    return subprocess.run([str(ENGINE), *args], cwd=HERE, text=True,
                          capture_output=True, timeout=120,
                          env=dict(os.environ, **env))


@unittest.skipUnless(ENGINE.exists(), "kimi_k3 is not built")
class KimiUsageTest(unittest.TestCase):
    def assertUsage(self, out, ctx):
        for fragment in ("<model_dir>", "--ngen", "--ids", "coli chat --model"):
            self.assertIn(fragment, out, f"{ctx}: usage is missing {fragment!r}")

    def test_help_flags_print_usage_and_succeed(self):
        for flag in ("--help", "-h", "help"):
            with self.subTest(flag=flag):
                r = run(flag)
                out = r.stdout + r.stderr
                self.assertUsage(out, flag)
                self.assertEqual(r.returncode, 0,
                                 f"{flag} must succeed; asking for help is not an error")
                self.assertNotIn("config.json", out,
                                 f"{flag} was treated as a model directory")

    def test_no_arguments_prints_usage_and_fails(self):
        r = run()
        self.assertUsage(r.stdout + r.stderr, "no args")
        self.assertNotEqual(r.returncode, 0, "a missing model dir is a failure")

    def test_usage_names_the_launcher(self):
        """Most people reaching for the raw engine want `coli chat`. It runs this
        engine for them and does not need the GLM binary built — which is what
        the #783 reporter had assumed was required."""
        self.assertIn("coli chat --model", run("--help").stdout + run("--help").stderr)

    def test_usage_documents_autopin_and_the_gpu_path(self):
        """Both are things a user cannot guess and will otherwise never find:
        Kimi's NVIDIA path is Vulkan, not CUDA (#783), and AUTOPIN=0 is the only
        way to opt out of pin seeding once a history exists."""
        out = run("--help").stdout + run("--help").stderr
        self.assertIn("COLI_VULKAN=1", out)
        self.assertIn("AUTOPIN=0", out)


if __name__ == "__main__":
    unittest.main()
