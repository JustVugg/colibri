"""The CUDA chain's element-wise op: every CC_EW_* code cuda_chain.h defines is one cc_ew accepts.

cc_ew (cuda_chain.cu) validates its buffers in a switch whose default refuses, ahead of a
kernel that implements every code. CC_EW_SIGMUL and CC_EW_RESGATE were added to the kernel
for Muse Spark and never to the switch: cc_ew returned 0 for them, and every forward of
Muse on the CUDA chain fell back to the host's step.
A code the switch does not name is refused at run time without a word, so this checks the
two lists against each other."""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class CudaEwOpsSource(unittest.TestCase):
    def test_every_code_is_accepted(self):
        header = (ROOT / "cuda_chain.h").read_text(encoding="utf-8")
        codes = re.findall(r"^#define\s+(CC_EW_[A-Z0-9_]+)\s+\d+", header, re.M)
        self.assertGreater(len(codes), 5)
        source = (ROOT / "cuda_chain.cu").read_text(encoding="utf-8")
        start = source.index("static int cc_ew(")
        body = source[start:source.index("coli_launch(cc_ew_k", start)]   # the kernel launch after the switch
        accepted = set(re.findall(r"case\s+(CC_EW_[A-Z0-9_]+)\s*:", body))
        missing = [c for c in codes if c not in accepted]
        self.assertEqual(missing, [], "CC_EW codes cc_ew refuses: " + ", ".join(missing))


if __name__ == "__main__":
    unittest.main()
