"""The GLM-5.3 harnesses, seen by unittest.

tests/glm53_*_harness.py are argparse programs, not unittest modules. Under
their old test_*.py names `make test-python` imported them, found no TestCase
and counted nothing, so the suite stayed green while the chat template, serve,
streaming, vision and Vulkan paths were never exercised (#1700).

Two things are checked here. Everywhere, with the standard library only: a
harness that cannot find what it needs exits 2, never 0, so a skip cannot pass
for a verification. And the two stdlib-only oracles run against their fixtures
when GLM53_TINY (tools/make_glm53_tiny.py) and GLM53_MM_TINY
(tools/make_glm53_multimodal_tiny.py) point at them, as the GLM-5.3 CI job
does; without them those two tests are skipped with the reason on the record.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINARY = next((HERE.parent / name for name in ("glm53", "glm53.exe")
               if (HERE.parent / name).exists()), None)


def harness(name, *args):
    return subprocess.run(
        [sys.executable, str(HERE / f"glm53_{name}_harness.py"), *args],
        capture_output=True, text=True, timeout=600)


class Glm53HarnessSkipTest(unittest.TestCase):
    def test_missing_input_exits_2(self):
        """Every harness, pointed at a fixture that is not there, says SKIP
        and exits 2. The binary is never reached, so none is needed."""
        with tempfile.TemporaryDirectory() as empty:
            missing = str(Path(empty, "missing"))
            cases = {
                "chat_template": ["--template", missing],
                "tiny": ["--binary", "glm53", "--fixture", missing],
                "multimodal_tiny": ["--binary", "glm53", "--fixture", missing],
                "pin_branch": ["--binary", "glm53", "--fixture", missing],
                "serve": ["--binary", "glm53", "--fixture", missing],
                "streaming": ["--binary", "glm53", "--quantized", missing,
                              "--dequantized", missing],
                "vision_serve": ["--binary", "glm53", "--fixture", missing],
                "vulkan": ["--binary", "glm53", "--fixture", missing],
            }
            self.assertEqual(sorted(cases), sorted(
                path.name[len("glm53_"):-len("_harness.py")]
                for path in HERE.glob("glm53_*_harness.py")),
                "a harness was added or renamed: give it a case here")
            for name, args in cases.items():
                with self.subTest(harness=name):
                    result = harness(name, *args)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("SKIP", result.stdout)


class Glm53TinyOracleTest(unittest.TestCase):
    def run_oracle(self, name, variable):
        fixture = os.environ.get(variable)
        if not fixture:
            self.skipTest(f"{variable} not set to a fixture")
        # Set but unusable is a failure, not a skip: whoever set it expected a run.
        self.assertIsNotNone(BINARY, f"{variable} is set but glm53 is not built")
        result = harness(name, "--binary", str(BINARY), "--fixture", fixture)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_text_oracle(self):
        """KDA, DSA, mHC, dense FFN and routed MoE, token-exact against the
        transformers reference in f32."""
        self.run_oracle("tiny", "GLM53_TINY")

    def test_multimodal_oracle(self):
        """The vision tower and the image tokens in the prompt, token-exact."""
        self.run_oracle("multimodal_tiny", "GLM53_MM_TINY")

    def test_serve_harness(self):
        """REUSE and CANCEL over the serve protocol, including the reason
        REUSE gives for reusing the slot or not."""
        self.run_oracle("serve", "GLM53_MM_TINY")

    def test_pin_branch_harness(self):
        """A pin is not restored over rows another branch rewrote (#1750)."""
        self.run_oracle("pin_branch", "GLM53_MM_TINY")


if __name__ == "__main__":
    unittest.main()
