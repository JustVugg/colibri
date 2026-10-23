"""Every tests/test_*.py that `make test-python` collects holds tests.

`make test-python` is `unittest discover -p 'test_*.py'`. A standalone argparse
harness that matches the glob is imported, contributes zero tests and still
lets the run report OK, so whatever it checks silently stops being checked
(#1700: seven GLM-5.3 harnesses, the whole chat template, serve, streaming,
vision and Vulkan coverage). A harness belongs under another name, as
tests/glm53_*_harness.py and tests/prefix_serve_harness.py do, with a unittest
wrapper where it can run unattended (tests/test_glm53_oracles.py).
"""
import importlib
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Harnesses that still match the glob. Most run by path from the Makefile, so
# they are exercised there and only count as empty modules here. The list only
# shrinks: rename a harness out of the glob and drop it from here.
KNOWN_HARNESSES = {
    "test_deepseek_v4_brio.py",
    "test_deepseek_v4_prefix.py",
    "test_deepseek_v4_tiny.py",
    "test_efficiency_report.py",
    "test_glm52_chat_template.py",
    "test_kimi_k3_ckpt.py",
    "test_kimi_k3_dashboard.py",
    "test_kimi_k3_tiny.py",
    "test_olmoe_chat_template.py",
    "test_qwen36_chat_template.py",
    "test_qwen38_chat_template.py",
    "test_qwen38_image.py",
    "test_qwen38_vision_serve.py",
}


class PythonDiscoveryTest(unittest.TestCase):
    def test_every_collected_module_holds_tests(self):
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        loader = unittest.TestLoader()
        empty = []
        for path in sorted(HERE.glob("test_*.py")):
            if path.name in KNOWN_HARNESSES:
                continue
            try:
                module = importlib.import_module(path.stem)
            except Exception:
                # discover reports an import error, or a SkipTest raised at
                # import, as a result of its own: that module is not silent.
                continue
            if loader.loadTestsFromModule(module).countTestCases() == 0:
                empty.append(path.name)
        self.assertEqual(empty, [], "these modules match test_*.py but hold no "
                         "tests, so `make test-python` counts them as passed: "
                         "rename them out of the glob or add a TestCase")

    def test_known_harnesses_still_exist(self):
        """A stale entry would let a new file of the same name slip through."""
        self.assertEqual(sorted(name for name in KNOWN_HARNESSES
                                if not (HERE / name).exists()), [])


if __name__ == "__main__":
    unittest.main()
