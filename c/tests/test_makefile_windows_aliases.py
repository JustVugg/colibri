"""Every engine the installer builds has a bare `make <engine>` on Windows (#1945, #1900).

With EXE=.exe the real rules are `<engine>.exe:`; a bare `make qwen38` without an alias
falls to make's built-in %: %.c rule, which compiles qwen38.c alone (no CUDA loader, no
Vulkan objects) and fails at the link, or keeps a CPU-only binary in silence. The dry
run below is what make would do for the installer's own target names.
"""
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
MAKE = shutil.which("make")
sys.path.insert(0, str(C_DIR))
import family_registry  # noqa: E402


@unittest.skipUnless(MAKE, "make is required")
class MakefileWindowsAliasTests(unittest.TestCase):
    def test_every_installer_target_takes_its_exe_rule(self):
        targets = sorted({f.build_target for f in family_registry.all_families()})
        self.assertIn("qwen38", targets)
        for target in targets:
            if target == "deepseek-v4":   # its own phony target in Makefile.deepseek-v4
                continue
            with self.subTest(target=target):
                out = subprocess.run(
                    [MAKE, "--no-print-directory", "-B", "-n", target, "TRIPLET=x86_64-w64-mingw32"],
                    cwd=C_DIR, text=True, capture_output=True, check=True,
                ).stdout
                self.assertIn(f"-o {target}.exe", out)
                self.assertNotRegex(out, rf"\b{target}\.c\s+-o {target}\s*$")


if __name__ == "__main__":
    unittest.main()
