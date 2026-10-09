import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).parents[1] / "tools" / "expert_atlas"
HEADER = "-1 92 256\n-2 1 3815245270\n"


def write_stats(root, header):
    root.mkdir()
    for c, cat in enumerate(("python", "poetry", "sql")):
        scale = 5 if cat == "poetry" else 1
        for run in range(3):
            lines = []
            for layer in range(3, 6):
                for expert in range(8):
                    n = (1 + (layer + expert + run) % 3) * scale
                    if expert == c:
                        n += 40 * scale
                    lines.append(f"{layer} {expert} {n}\n")
            (root / f"{cat}_{run}.txt").write_text((HEADER if header else "") + "".join(lines))


def run(script, *args):
    out = subprocess.run([sys.executable, str(TOOLS / script), *map(str, args)],
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if not line.startswith("wrote ")]


class ExpertAtlasHeaderTest(unittest.TestCase):
    def test_header_records_do_not_change_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_stats(tmp / "plain", header=False)
            write_stats(tmp / "header", header=True)
            plain = run("analyze.py", "--stats", tmp / "plain", "--out", tmp / "plain.json")
            header = run("analyze.py", "--stats", tmp / "header", "--out", tmp / "header.json")
            self.assertEqual(plain, header)
            experts = json.loads((tmp / "header.json").read_text())["experts"]
            self.assertFalse([k for k in experts if str(k).startswith("-")])
            self.assertEqual(json.loads((tmp / "plain.json").read_text()), json.loads((tmp / "header.json").read_text()))
            self.assertEqual(run("validate.py", tmp / "plain", 3), run("validate.py", tmp / "header", 3))


if __name__ == "__main__":
    unittest.main()
