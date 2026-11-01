"""A failed benchmark engine must not look like a successful experiment."""
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
class BenchUxFailureTest(unittest.TestCase):
    def test_engine_failure_stops_battery_and_preserves_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "engine"
            engine.write_text("#!/bin/sh\necho 'fixture engine failed' >&2\nexit 7\n", encoding="utf-8")
            engine.chmod(0o755)
            env = dict(os.environ, REPS="1", TMPDIR=temporary)
            result = subprocess.run(["bash", str(SCRIPT), str(engine)], env=env,
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 7)
            self.assertIn("fixture engine failed", result.stderr)
            self.assertNotIn("bench_ux done", result.stdout)
            self.assertEqual([p.name for p in root.iterdir()], ["engine"])


if __name__ == "__main__":
    unittest.main()
