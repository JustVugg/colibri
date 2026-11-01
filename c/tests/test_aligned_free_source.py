#!/usr/bin/env python3
"""posix_memalign is _aligned_malloc on Windows (compat.h), and a block from it must go
back through compat_aligned_free: free() on it corrupts the heap and the engine dies
with no message. Kimi K3's warm start freed its expert buffers with free() and every
start with an expert history crashed on Windows (#2311); MiMo's warm start and DeepSeek
V4.1's slots given to the Vulkan tier's host pool did the same.

The rule checked here, over every engine source: a struct field that only
posix_memalign fills (no malloc, calloc or realloc writes it) is never handed to free().
"""
from pathlib import Path
import re
import unittest

C = Path(__file__).resolve().parents[1]
ALIGNED = re.compile(r"posix_memalign\s*\(\s*(?:\(\s*void\s*\*\*\s*\)\s*)?&\s*([\w.\->\[\]]+?)\s*,")
FREE = re.compile(r"(?<![\w])free\s*\(\s*([^;]*?)\)\s*;")
FIELD = re.compile(r"(?:\.|->)(\w+)(?:\[[^\]]*\])?$")


def field_of(expr):
    m = FIELD.search(expr.strip())
    return m.group(1) if m else None


class AlignedFreeSourceTests(unittest.TestCase):
    def test_no_free_of_a_posix_memalign_field(self):
        bad = []
        for path in sorted(C.glob("*.c")):
            src = path.read_text(errors="replace")
            fields = {field_of(m.group(1)) for m in ALIGNED.finditer(src)} - {None}
            for f in sorted(fields):
                # another allocator writes it too: the engine tracks which (deepseek_v4's aligned_slab)
                if re.search(r"(?:\.|->)%s(?:\[[^\]]*\])?\s*=\s*\(?[\w\s\*]*\)?\s*(?:malloc|calloc|realloc|falloc|xzalloc)\s*\(" % f, src):
                    fields.discard(f)
            for no, line in enumerate(src.split("\n"), 1):
                for m in FREE.finditer(line):
                    if field_of(m.group(1)) in fields:
                        bad.append(f"{path.name}:{no}: free({m.group(1)})")
        self.assertEqual(bad, [], "posix_memalign'd memory freed with free() (use compat_aligned_free)")

    def test_the_slots_the_tier_takes_back(self):
        # DeepSeek V4.1's slot windows come from a local posix_memalign, so the rule above
        # does not see them: the host pool's ram_given hook gives them back
        src = (C / "deepseek_v41.c").read_text()
        self.assertIn("compat_aligned_free(lc->slot[k].base[part])", src)
        self.assertNotIn("free(lc->slot[k].base[part])", src.replace("compat_aligned_free(lc->slot[k].base[part])", ""))


if __name__ == "__main__":
    unittest.main()
