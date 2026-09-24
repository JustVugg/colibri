"""c/Makefile records the build-affecting flags in .build-config (#306).

The write used $(file ...), which GNU Make 3.81 -- /usr/bin/make on macOS --
does not have: the stamp was never written and every build relinked (#1732).
Make 3.x now writes it through printf. This drives the parse with
MAKE_VERSION forced to 3.81 and to the running make's own version, and checks
that both write the stamp and that an unchanged configuration leaves it alone.
The previous .build-config is restored afterwards.
"""
import os
import shutil
import subprocess
import time
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
MAKE = shutil.which("make")
STAMP = C_DIR / ".build-config"


@unittest.skipUnless(MAKE and os.name == "posix", "make and a POSIX shell are required")
class BuildConfigStampTest(unittest.TestCase):
    def setUp(self):
        self.saved = STAMP.read_bytes() if STAMP.exists() else None

    def tearDown(self):
        if self.saved is None:
            STAMP.unlink(missing_ok=True)
        else:
            STAMP.write_bytes(self.saved)

    def parse(self, *variables):
        result = subprocess.run([MAKE, "-s", *variables, ".build-config"], cwd=C_DIR,
                                text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)

    def check(self, *version):
        marker = f"-DCOLI_STAMP_TEST_{time.monotonic_ns()}"
        self.parse(*version, f"EXTRA_CFLAGS={marker}")
        self.assertTrue(STAMP.exists(), "the stamp was not written")
        text = STAMP.read_text()
        self.assertIn(marker, text)
        before = STAMP.stat().st_mtime_ns
        time.sleep(0.05)
        self.parse(*version, f"EXTRA_CFLAGS={marker}")
        self.assertEqual(STAMP.stat().st_mtime_ns, before,
                         "an unchanged configuration rewrote the stamp")
        self.assertEqual(STAMP.read_text(), text)

    def test_make_381_writes_the_stamp(self):
        self.check("MAKE_VERSION=3.81")

    def test_current_make_writes_the_stamp(self):
        self.check()


if __name__ == "__main__":
    unittest.main()
