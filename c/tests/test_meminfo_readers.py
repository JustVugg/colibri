"""kimi_k3 and deepseek_v4 sized their caches from their own /proc/meminfo reads.

compat_meminfo_gb bounds MemAvailable by a container's cgroup (its limit less
its working set); an engine that reads the file itself sees the host's RAM in
a container and sizes a cache the cgroup then kills. The behaviour of the
shared probe is tested in test_compat_cgroup.c; here, that these two budgets go
through it, as inkling's and colibri's already do.
"""
import re
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _c_function(source, signature):
    match = re.search(re.escape(signature) + r"\s*\{", source)
    if not match:
        raise ValueError("no definition for " + signature)
    depth = 0
    for index, char in enumerate(source[match.end() - 1:], start=match.end() - 1):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[match.start():index + 1]
    raise ValueError("unbalanced braces in " + signature)


def _linux_branch(body):
    """The #else branch after #ifdef _WIN32 / #elif defined(__APPLE__), the one Linux compiles."""
    return body.split("#else", 1)[1].split("#endif", 1)[0] if "#else" in body else body


class MeminfoReadersTest(unittest.TestCase):
    def check(self, engine, signature):
        body = _c_function((HERE.parent / engine).read_text(encoding="utf-8"), signature)
        linux = _linux_branch(body)
        self.assertIn("compat_mem_available_gb", linux)
        self.assertNotIn("/proc/meminfo", body)

    def test_kimi_k3_budget_reads_the_shared_probe(self):
        self.check("kimi_k3.c", "static double k3_mem_avail(void)")

    def test_deepseek_v4_budget_reads_the_shared_probe(self):
        self.check("deepseek_v4.c", "uint64_t coli_v4_os_available_memory(void)")


if __name__ == "__main__":
    unittest.main()
