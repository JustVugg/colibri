import unittest

from tools.benchmark_cuda_fixture import parse_output, parse_p0


SAMPLE = """
REPLAY decode: 4 tokens | 12.34 tok/s
PROFILE: expert-disk 1.25s | expert-matmul 2.50s | attention 0.75s | lm_head 0.10s | other -0.05s
P0-EXEC: routed CPU 1.200s / 123.40 GB/s (456 row) | routed GPU critical 0.150s | router 0.200s | residual P2P 0.030s / 75 hop | orchestration 0.100s
"""


class ParseOutputTest(unittest.TestCase):
    def test_extracts_speed_and_profile(self):
        speed, profile = parse_output(SAMPLE)
        self.assertEqual(speed, 12.34)
        self.assertEqual(profile, [1.25, 2.5, 0.75, 0.1, -0.05])

    def test_rejects_incomplete_output(self):
        with self.assertRaisesRegex(RuntimeError, "benchmark output missing"):
            parse_output("REPLAY decode: 4 tokens | 12.34 tok/s", "engine failed")

    def test_extracts_p0_profile(self):
        self.assertEqual(parse_p0(SAMPLE), [1.2, 123.4, 456.0, 0.15, 0.2, 0.03, 75.0, 0.1])


if __name__ == "__main__":
    unittest.main()
