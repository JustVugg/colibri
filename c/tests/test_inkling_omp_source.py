import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class InklingOpenMPWiringTest(unittest.TestCase):
    def test_direct_engine_uses_shared_physical_core_sizing(self):
        source = (ROOT / "inkling.c").read_text(encoding="utf-8")
        self.assertTrue('#include "omp_tune.h"' in source,
                        "inkling.c does not include the shared OpenMP sizing helper")
        self.assertTrue('coli_omp_tune_threads("inkling")' in source,
                        "Inkling direct execution does not size its OpenMP team")


if __name__ == "__main__":
    unittest.main()
