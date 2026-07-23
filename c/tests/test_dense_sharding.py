"""Token-exact correctness gate for the dense activation shard."""

import os
import re
import socket
import subprocess
import time
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
ENGINE = C_DIR / "colibri"
TINY = C_DIR / "glm_tiny"


def _available() -> bool:
    return ENGINE.exists() and (TINY / "config.json").exists()


def _skip_reason() -> str:
    if not ENGINE.exists():
        return "colibri is not built (run: make colibri)"
    return "glm_tiny fixture absent (run tools/make_glm_oracle.py)"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _tf_signature(result: subprocess.CompletedProcess[str]):
    match = re.search(
        r"PREFILL \(teacher-forcing\).*:\s+(\d+)/(\d+) positions",
        result.stdout,
    )
    mismatches = tuple(
        line for line in result.stderr.splitlines() if line.startswith("[ORACLE] mismatch")
    )
    return (match.groups() if match else None, mismatches)


@unittest.skipUnless(_available(), _skip_reason())
class DenseShardingCorrectnessTest(unittest.TestCase):
    """The remote dense path must preserve every teacher-forced token."""

    def test_dense_shard_matches_local_cpu(self):
        port = _free_port()
        worker_env = {
            **os.environ,
            "SNAP": str(TINY),
            "DENSE_WORKER": "1",
            "DENSE_WORKER_PORT": str(port),
            "DENSE_WORKER_FIRST": "0",
            "DENSE_WORKER_LAST": "2",
            "COLI_NO_OMP_TUNE": "1",
        }
        worker = subprocess.Popen(
            [str(ENGINE), "64", "8", "8"],
            cwd=C_DIR,
            env=worker_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if worker.poll() is not None:
                    stderr = worker.stderr.read() if worker.stderr else ""
                    self.fail(f"dense worker exited early: {stderr}")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                self.fail("dense worker did not start listening")

            common_env = {
                **os.environ,
                "SNAP": str(TINY),
                "TF": "1",
                "TEMP": "0",
                "DRAFT": "0",
                "COLI_NO_OMP_TUNE": "1",
            }
            baseline = subprocess.run(
                [str(ENGINE), "64", "8", "8"],
                cwd=C_DIR,
                env=common_env,
                capture_output=True,
                text=True,
                check=False,
            )
            delegated_env = {
                **common_env,
                "DENSE_SHARDS": f"127.0.0.1:{port}:0:2",
            }
            delegated = subprocess.run(
                [str(ENGINE), "64", "8", "8"],
                cwd=C_DIR,
                env=delegated_env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            self.assertEqual(delegated.returncode, 0, delegated.stderr)
            self.assertEqual(
                _tf_signature(baseline),
                _tf_signature(delegated),
                "dense sharding changed teacher-forced token predictions",
            )
        finally:
            worker.terminate()
            try:
                worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()


if __name__ == "__main__":
    unittest.main()
