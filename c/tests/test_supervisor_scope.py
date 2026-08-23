import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
SUPERVISOR = C_DIR / "scripts" / "supervisor.sh"


@unittest.skipUnless(
    sys.platform.startswith("linux") and shutil.which("flock") and Path("/proc").is_dir(),
    "the conversion supervisor targets Linux/WSL",
)
class SupervisorScopeTests(unittest.TestCase):
    def test_completion_stops_only_its_own_converter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "convert_fp8_to_int4.py"
            script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
            own_dir = root / "own model"
            other_dir = root / "other model"
            own_dir.mkdir()
            other_dir.mkdir()

            own = subprocess.Popen([
                sys.executable, str(script), "--repo", "repo", "--outdir", str(own_dir)
            ])
            other = subprocess.Popen([
                sys.executable, str(script), "--repo", "repo", "--outdir", str(other_dir)
            ])
            try:
                for _ in range(50):
                    cmdline = Path(f"/proc/{own.pid}/cmdline")
                    if cmdline.exists() and b"convert_fp8_to_int4.py" in cmdline.read_bytes():
                        break
                    time.sleep(0.02)
                env = os.environ.copy()
                env.update({"COLI_MODEL": str(own_dir), "TOTAL_SHARDS": "0"})
                result = subprocess.run(
                    ["bash", str(SUPERVISOR)], cwd=C_DIR, env=env,
                    text=True, capture_output=True, timeout=10, check=False,
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                own.wait(timeout=5)
                self.assertIsNone(other.poll(),
                                  "the supervisor stopped another model's converter")
            finally:
                for process in (own, other):
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
