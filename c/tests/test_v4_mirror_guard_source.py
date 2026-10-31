"""The decode mirror cache's growth guard counts what it grew between probes.

v4_gpu_expert_attach_cached_ex re-reads free VRAM only every 64 misses and
reuses the last answer in between. Measured against that stale answer alone,
the cache grew up to 64 mirrors (816 MiB) past an explicit
DSV4_CUDA_VRAM_RESERVE_MB or the 600 MiB default, past the card's free VRAM.
On the full DeepSeek V4 Flash at 24 GB the compressed-KV buffer then could not
grow ("kv comp allocation: out of memory", hundreds per request), its layers'
attention ran on the CPU, and greedy texts changed from run to run. No fixture
reaches a full card, so this pins the guard at source level:

- the mirrors appended since the last probe (grown_bytes) count against the
  reserve, the probe resets them, and a failed upload clears them;
- only an appended slot adds to them (a recycled one reuses its VRAM);
- the bank-sized default keeps its 64-mirror margin (v4_gpu_mirror_reserve_mb
  already carries it), so its decisions do not change; an explicit reserve and
  the headroom-only default carry none.
"""
import re
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "deepseek_v4.c"


def find(text, needle, start=0, where="deepseek_v4.c"):
    at = text.find(needle, start)
    if at < 0:
        raise AssertionError(f"{where}: no {needle!r}")
    return at


class V4MirrorGuardSourceTest(unittest.TestCase):
    def setUp(self):
        text = SOURCE.read_text(encoding="utf-8")
        decl = find(text, "static int v4_gpu_expert_attach_cached_ex(")   # the forward declaration
        start = find(text, "static int v4_gpu_expert_attach_cached_ex(", decl + 1)
        self.body = text[start:find(text, "\n}\n", start)]

    def test_probe_resets_the_grown_count(self):
        probe = find(self.body, "last_free_mb = dsv4_cuda_mem_free_mb(cache->device);")
        close = find(self.body, "}", probe)
        self.assertIn("grown_bytes = 0;", self.body[probe:close])

    def test_grown_mirrors_count_against_the_reserve(self):
        self.assertRegex(self.body, r"grown_mb = \(grown_bytes \+ mirror_bytes")
        self.assertRegex(
            self.body,
            r"cache->count >= 8 && last_free_mb - grown_mb < reserve_mb - margin_mb")

    def test_only_appended_slots_add_and_failures_clear(self):
        self.assertIn("found = cache->count++;\n            appended = 1;", self.body)
        self.assertIn("if (appended) grown_bytes += mirror_bytes;", self.body)
        fail = find(self.body, "upload_failed = 1;")
        self.assertIn("grown_bytes = 0;", self.body[fail:fail + 80])

    def test_bank_default_keeps_its_margin(self):
        self.assertIn("long long margin_mb = v4_gpu_mirror_reserve_margin_mb(cache->count);",
                      self.body)
        text = SOURCE.read_text(encoding="utf-8")
        start = find(text, "static long long v4_gpu_mirror_reserve_margin_mb(int live) {")
        margin = text[start:find(text, "\n}\n", start)]
        self.assertIn("v4_gpu_mirror_reserve_explicit_mb() >= 0", margin)
        self.assertIn("v4_gpu_mirror_reserve_mb(live) <= COLI_V4_RESERVE_HEADROOM_MB", margin)
        self.assertIn("COLI_V4_MIRROR_PROBE_EVERY * v4_reserve_mirror_bytes) >> 20", margin)

if __name__ == "__main__":
    unittest.main()
