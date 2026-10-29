"""olmoe's command line reads the oracle (ref.json) before it loads a model: a file that is not
one is refused with a line and exit status 1. A text file or an object without the arrays read
through a null pointer and crashed. Lengths and id ranges are checked once the model's vocabulary
is known (test_olmoe_reference_bounds.py)."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EXE = HERE / ("olmoe.exe" if os.name == "nt" else "olmoe")


@unittest.skipUnless(EXE.exists(), "olmoe is not built")
class OlmoeRefInput(unittest.TestCase):
    def run_ref(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.json"
            ref.write_text(text, encoding="utf-8")
            env = dict(os.environ, SNAP=str(Path(tmp) / "no_model"))
            env.pop("SERVE", None)
            env.pop("CHAT", None)
            return subprocess.run([str(EXE), "8", "8", str(ref)], capture_output=True, text=True, env=env, timeout=120)

    def test_a_text_file(self):
        r = self.run_ref("Read this code and say OK.\n\nint main(void) { return 0; }\n")
        self.assertEqual(r.returncode, 1, r.stderr[-1000:])
        self.assertIn('ref.json: missing array "prompt_ids"', r.stderr)

    def test_an_object_without_the_arrays(self):
        r = self.run_ref('{"prompt": [1, 2, 3]}')
        self.assertEqual(r.returncode, 1, r.stderr[-1000:])
        self.assertIn('ref.json: missing array "prompt_ids"', r.stderr)

    def test_a_value_that_is_no_integer(self):
        for bad in ('2.5', '"2"', '1e300', 'null'):
            with self.subTest(bad=bad):
                r = self.run_ref('{"prompt_ids": [1, %s], "full_ids": [1, 2, 3]}' % bad)
                self.assertEqual(r.returncode, 1, r.stderr[-1000:])
                self.assertIn("ref.json: prompt_ids[1] is not an integer", r.stderr)


if __name__ == "__main__":
    unittest.main()
