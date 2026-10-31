#!/usr/bin/env python3
"""k3_exact_tables.h is what tools/k3_exact_tables.c writes today: it builds the tool, which
checks k3_exact.h's double arithmetic against exp_f and tanhf on all 2^32 floats, and compares
the tables it would write with the file's (--check). MinGW (msvcrt) x86-64 only: the tables
hold MinGW's bits, and elsewhere exp_f is the C library's expf."""
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

C = Path(__file__).resolve().parents[1]


class K3ExactTables(unittest.TestCase):
    def test_tables_are_current(self):
        gcc = shutil.which(os.environ.get("CC", "gcc"))
        if not gcc or platform.machine().lower() not in ("amd64", "x86_64"):
            self.skipTest("needs gcc on x86-64")
        version = subprocess.run([gcc, "-v"], capture_output=True, text=True).stderr
        if sys.platform != "win32" or "mingw" not in version.lower() or "msvcrt" not in version.lower():
            self.skipTest("the tables hold MinGW (msvcrt) exp_f and tanhf bits")
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "k3_exact_tables.exe"
            subprocess.run([gcc, "-O2", "-march=native", "-fopenmp", "-ffp-contract=off",
                            str(C / "tools" / "k3_exact_tables.c"), "-o", str(exe), "-lm"],
                           check=True, capture_output=True)
            run = subprocess.run([str(exe), "--check", str(C / "k3_exact_tables.h")],
                                 capture_output=True, text=True, timeout=900)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("0 differ from exp_f", run.stdout)
            self.assertIn("0 differ from tanhf", run.stdout)
            self.assertIn("the same tables", run.stdout)


if __name__ == "__main__":
    unittest.main()
