import unittest

from tools.benchmark_cuda_fixture import parse_output


SAMPLE = """
REPLAY decode: 4 tokens | 12.34 tok/s
PROFILE: expert-disk 1.25s | expert-matmul 2.50s | attention 0.75s | lm_head 0.10s | other -0.05s
"""


class ParseOutputTest(unittest.TestCase):
    def test_extracts_speed_and_profile(self):
        speed, profile = parse_output(SAMPLE)
        self.assertEqual(speed, 12.34)
        self.assertEqual(profile, [1.25, 2.5, 0.75, 0.1, -0.05])

    def test_rejects_incomplete_output(self):
        with self.assertRaisesRegex(RuntimeError, "benchmark output missing"):
            parse_output("REPLAY decode: 4 tokens | 12.34 tok/s", "engine failed")


if __name__ == "__main__":
    unittest.main()
