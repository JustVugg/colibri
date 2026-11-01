"""The fixed greedy UX experiment overrides inherited sampling settings."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "bench_ux.sh"


# bench_ux.sh and its fixture engine (a #!/bin/sh file it executes) are POSIX: on the Windows
# runner the script cannot run the engine from its temporary directory (exit 1 there).
@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "a POSIX shell is required")
class BenchUxGreedyTest(unittest.TestCase):
    def test_primary_temperature_is_greedy_for_every_scenario(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "engine"
            output = root / "temperatures"
            engine.write_text(
                '#!/bin/sh\nprintf "%s,%s,%s\\n" "$COLI_TEMP" "$TEMP" "$DRAFT" >> "$RECORD"\n'
                'echo "prefill 1 decode 2 tok/s"\n', encoding="utf-8")
            engine.chmod(0o755)
            env = dict(os.environ, REPS="2", COLI_TEMP="0.8", TEMP="0.7", DRAFT="3", RECORD=str(output))
            result = subprocess.run(["bash", str(SCRIPT), str(engine)], env=env,
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_text().splitlines(), ["0,0,0"] * 4)


if __name__ == "__main__":
    unittest.main()
