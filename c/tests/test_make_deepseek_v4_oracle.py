import importlib.util
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "tools" / "make_deepseek_v4_oracle.py"
SPEC = importlib.util.spec_from_file_location("make_deepseek_v4_oracle", SCRIPT)
ORACLE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ORACLE)


class TargetOracleTests(unittest.TestCase):
    def test_make_target_uses_supported_validation_flags(self):
        makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(
            encoding="utf-8"
        )
        target = makefile.split("deepseek-v4-oracle: deepseek-v4", 1)[1]
        target = target.split("\nelse\n", 1)[0]
        self.assertIn("--validate", target)
        self.assertIn("--teacher-forcing", target)
        self.assertNotIn("--check-dspark", target)
        self.assertNotIn("--continuation", target)

    def run_validation(self, returncode: int) -> int:
        process = mock.Mock(returncode=returncode)
        with mock.patch.object(ORACLE, "run_c", return_value=process):
            return ORACLE.validate(
                Path("deepseek_v4"),
                "/model",
                Path("oracle.json"),
                teacher_forcing=2,
                greedy=2,
                memory_gb=None,
            )

    def test_validation_success(self):
        self.assertEqual(self.run_validation(0), 0)

    def test_validation_failure_is_propagated(self):
        self.assertEqual(self.run_validation(3), 3)


if __name__ == "__main__":
    unittest.main()
