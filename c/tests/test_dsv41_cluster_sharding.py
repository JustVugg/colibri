"""Token-exact parity gate: local vs cluster-delegated routed experts, DeepSeek V4.1.

test_cluster_sharding.py pins this for colibri.c (GLM). deepseek_v41.c carries the
same COLIEX01 expert-worker protocol, so it gets the same gate: the local CPU run
must reproduce tools/make_dsv41_tiny.py's reference token-exactly, and delegating
the backbone's routed experts to a worker must not perturb a single position.

What moves to the worker is exactly what moe_run_at hands to expert_ffn_rows for
`kind == "layers"`: the fp4 routed experts of the backbone. Routing, the shared
expert, attention, the DSA indexer, the engram tables, the vision tower and the
DSpark draft stages ("mtp", their own expert set) stay on the coordinator in both
runs, so they are not part of the shard under test.

Three things are asserted, and they are different things:
  1. both runs exit 0 with no `[mismatch]` line -- each is token-exact on its own;
  2. the two token streams are equal -- delegation changed nothing;
  3. the delegated run actually delegated: the coordinator reports the worker it
     connected to, and reads fewer expert bytes than the baseline (the backbone's
     are now the worker's to read). Without this the gate would pass on a
     coordinator that ignored CLUSTER_WORKERS.

cap=1 is the cache the one-at-a-time path runs on (cap < topk); cap=8 holds every
expert of the fixture. Delegation bypasses the cache on the coordinator either
way, so both are compared against it.

Fixture (gitignored, regenerated -- the ci.yml dsv41-tiny-check job writes it):
    python3 tools/make_dsv41_tiny.py --out dsv41_tiny --emit-ref dsv41_tiny/ref.json

Determinism: the worker and both coordinator runs are the SAME c/deepseek_v41
binary with the same env block, so the comparison is meaningful rather than
accidental.
"""

import os
import re
import socket
import subprocess
import time
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
ENGINE = C_DIR / "deepseek_v41"
FIXTURE = C_DIR / "dsv41_tiny"
REF = FIXTURE / "ref.json"

# Applied to the worker and to both coordinator runs. The worker is always CPU
# (it returns from main() before the device opens), and every run is the same
# binary, so the numeric path matches by construction; this only keeps the
# host's thread tuning and the opt-in prefix cache out of the picture.
_ENV = {
    "COLI_NO_OMP_TUNE": "1",
    "COLI_VULKAN": "0",
    "COLI_KV_PREFIX": "0",
}


def _fixture_ok() -> bool:
    return (
        (FIXTURE / "config.json").exists()
        and (FIXTURE / "model.safetensors").exists()
        and REF.exists()
    )


def _available() -> bool:
    return ENGINE.exists() and _fixture_ok()


def _skip_reason() -> str:
    if not ENGINE.exists():
        return "deepseek_v41 is not built (run: make deepseek_v41)"
    return ("dsv41_tiny fixture absent (run: python3 tools/make_dsv41_tiny.py "
            "--out dsv41_tiny --emit-ref dsv41_tiny/ref.json)")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _signature(result: subprocess.CompletedProcess[str]):
    """(token stream, 'matched/expected', mismatch lines) of one oracle run."""
    tokens = tuple(result.stdout.split())
    match = re.search(r"Matching tokens:\s+(\d+)/(\d+)", result.stderr)
    mismatches = tuple(
        line for line in result.stderr.splitlines() if line.startswith("[mismatch]")
    )
    return (tokens, match.groups() if match else None, mismatches)


def _expert_mb(result: subprocess.CompletedProcess[str]):
    """Expert bytes the coordinator read itself, from its own summary line."""
    match = re.search(
        r"\[v41\] experts: \d+ hits, \d+ misses, ([0-9.]+) MB read", result.stderr
    )
    return float(match.group(1)) if match else None


@unittest.skipUnless(_available(), _skip_reason())
class DsV41ClusterShardingParityTest(unittest.TestCase):
    """Local CPU must equal cluster-delegated routed experts, token-exact."""

    def _run_parity(self, cap: str):
        port = _free_port()
        worker_env = {
            **os.environ,
            "SNAP": str(FIXTURE),
            "EXPERT_WORKER": "1",
            "CLUSTER_WORKER_PORT": str(port),
            **_ENV,
        }
        # The worker returns from main() before argv is read; "1" is what
        # `coli cluster worker` passes as the cache size, and is ignored here.
        worker = subprocess.Popen(
            [str(ENGINE), "1"],
            cwd=C_DIR,
            env=worker_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if worker.poll() is not None:
                    stderr = worker.stderr.read() if worker.stderr else ""
                    self.fail(f"cluster worker exited early: {stderr}")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                self.fail("cluster worker did not start listening")

            common_env = {**os.environ, "SNAP": str(FIXTURE), **_ENV}
            baseline = subprocess.run(
                [str(ENGINE), cap, str(REF)],
                cwd=C_DIR,
                env=common_env,
                capture_output=True,
                text=True,
                check=False,
            )
            delegated = subprocess.run(
                [str(ENGINE), cap, str(REF)],
                cwd=C_DIR,
                env={**common_env, "CLUSTER_WORKERS": f"127.0.0.1:{port}"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            self.assertEqual(delegated.returncode, 0, delegated.stderr)

            base_sig = _signature(baseline)
            del_sig = _signature(delegated)
            self.assertIsNotNone(
                base_sig[1],
                f"baseline produced no 'Matching tokens' summary:\n"
                f"stdout={baseline.stdout}\nstderr={baseline.stderr}",
            )
            self.assertTrue(len(base_sig[0]) > 0, "baseline emitted no tokens")
            self.assertEqual(base_sig[1][0], base_sig[1][1],
                             f"baseline is not token-exact: {base_sig[1]}")
            self.assertEqual(
                base_sig[2], (),
                f"baseline mismatched the reference ({base_sig[1]}): {base_sig[2]}",
            )
            self.assertEqual(
                del_sig[2], (),
                f"delegated mismatched the reference ({del_sig[1]}): {del_sig[2]}",
            )
            self.assertEqual(
                base_sig, del_sig,
                "cluster delegation changed the decoded tokens",
            )

            # The run must have gone through the worker, not past it.
            self.assertIn(
                "[CLUSTER] coordinator connected to 1 expert worker(s)",
                delegated.stderr,
                "the delegated run never connected to the worker",
            )
            base_mb, del_mb = _expert_mb(baseline), _expert_mb(delegated)
            self.assertIsNotNone(base_mb, baseline.stderr)
            self.assertIsNotNone(del_mb, delegated.stderr)
            self.assertLess(
                del_mb, base_mb,
                "the coordinator read as many expert bytes as the baseline: "
                "the backbone's routed experts were not delegated",
            )
        finally:
            worker.terminate()
            try:
                worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            for stream in (worker.stdout, worker.stderr):
                if stream is not None:
                    stream.close()

    def test_cap1_parity(self):
        """cap=1 (cap < topk, the one-at-a-time local path) against the worker."""
        self._run_parity("1")

    def test_cap8_parity(self):
        """cap=8 (every fixture expert resident locally) against the worker."""
        self._run_parity("8")


if __name__ == "__main__":
    unittest.main()
