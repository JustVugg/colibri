"""A held-out prompt must not contribute to any category's learned lift."""
import contextlib
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

VALIDATOR = Path(__file__).resolve().parents[1] / "tools/expert_atlas/validate.py"


def load_validator(directory):
    with patch.object(sys, "argv", [str(VALIDATOR), str(directory), "1"]), \
            contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(VALIDATOR))


class ExpertAtlasHoldoutTest(unittest.TestCase):
    def test_heldout_run_is_absent_from_all_learned_sets(self):
        # A's held-out run changes which B expert looks specialized if it is
        # mistakenly left inside the cross-category lift denominator.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {("a", "0"): "0 0 1000\n0 1 1\n",
                    ("a", "1"): "0 0 1\n0 1 1\n",
                    ("b", "0"): "0 0 6\n0 1 4\n",
                    ("b", "1"): "0 0 6\n0 1 4\n"}
            for (category, prompt), text in runs.items():
                (root / f"{category}_{prompt}.txt").write_text(text)
            state = load_validator(root)
            sets = {category: state["specialists"](category, ("a", "0"))
                    for category in ("a", "b")}
            # Hand-calculated training shares after dropping a_0 globally:
            # A=(.5,.5), B=(.6,.4). Thus expert1 favors A; expert0 favors B.
            self.assertEqual(sets, {"a": {(0, 1)}, "b": {(0, 0)}})
            (root / "a_0.txt").write_text("0 0 1\n0 1 1000\n")
            changed = load_validator(root)
            for category in ("a", "b"):
                self.assertEqual(changed["specialists"](category, ("a", "0")),
                                 sets[category])

    def test_public_validation_uses_the_uncontaminated_competing_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, text in {
                "a_0.txt": "0 0 1000\n0 1 1\n",
                "a_1.txt": "0 0 1\n0 1 1\n",
                "b_0.txt": "0 0 6\n0 1 4\n",
                "b_1.txt": "0 0 6\n0 1 4\n",
            }.items():
                (root / name).write_text(text)
            output = io.StringIO()
            with patch.object(sys, "argv", [str(VALIDATOR), str(root), "1"]), \
                    contextlib.redirect_stdout(output):
                runpy.run_path(str(VALIDATOR))
            line = next(line for line in output.getvalue().splitlines()
                        if line.strip().startswith("a ") and "prompt 0" in line)
            # Expert0 is B's independently calculated training specialist.
            # Its coverage on a_0 is 1000/1001, or 99.90 percent.
            self.assertIn("best-other 99.90%", line)


if __name__ == "__main__":
    unittest.main()
