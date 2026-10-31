"""Windows: the RAM a budget is sized from must be bounded by the commit still grantable.

GlobalMemoryStatusEx's ullAvailPhys is the physical memory free now; what decides whether
the next malloc succeeds is ullAvailPageFile, the commit left (RAM + page file, less what
every process has committed). #1375 paid it in the planner and compat_meminfo_gb; kimi_k3
(through compat_meminfo) and deepseek_v4 still read ullAvailPhys alone, and on a machine
whose commit is nearly gone (a small page file, a VM that reserved its RAM) they sized
caches Windows then refused. Every C reader of ullAvailPhys reads ullAvailPageFile beside
it."""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class WindowsCommitAvailSource(unittest.TestCase):
    def test_every_reader_bounds_by_the_commit(self):
        offenders = []
        for path in sorted(list(ROOT.glob("*.c")) + list(ROOT.glob("*.h"))):
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for i, line in enumerate(lines):
                code = line.split("//")[0]
                if not re.search(r"(\.|->)ullAvailPhys\b", code):
                    continue
                window = "\n".join(lines[max(0, i - 6):i + 7])
                if "ullAvailPageFile" not in window:
                    offenders.append(f"{path.name}:{i + 1}: {line.strip()}")
        self.assertEqual(offenders, [], "ullAvailPhys read without the commit bound:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
