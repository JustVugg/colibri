"""Every way out of the dense chain's setup (vk_chain.c, vkc_init) says why.

An engine whose vkc_init fails takes its per-matrix path. On an Arc Pro B60 (Mesa ANV
26.0.8) the chain stayed off at its first device buffer without a line (its memory type one
the buffers do not take), and decode ran at a third of the chain's speed with nothing in the
log to say so. The other exits cannot be made to happen on the CI's devices, so this checks
the source: no bare `return 0;` in vkc_init, each failure through init_off, which prints
`[VK] chain: <reason>, the chain stays off`."""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def vkc_init_body(src):
    start = src.index("\nint vkc_init(void) {")
    return src[start:src.index("\n}\n", start)]


class VkcInitSaysWhy(unittest.TestCase):
    def test_every_exit_says_why(self):
        src = (ROOT / "vk_chain.c").read_text(encoding="utf-8")
        body = vkc_init_body(src)
        self.assertIsNone(re.search(r"\breturn\s+0\s*;", body),
                          "a way out of vkc_init without its reason: return init_off(...) instead")
        self.assertGreater(len(re.findall(r"\breturn init_off\(", body)), 0)
        self.assertIn('"[VK] %schain: %s, the chain stays off\\n"', src)


if __name__ == "__main__":
    unittest.main()
