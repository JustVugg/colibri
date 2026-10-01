"""GLM-5.3 routed experts served by expert workers (CLUSTER_WORKERS).

The coordinator keeps attention, the router and the shared expert, and sends
each worker the token rows of the experts it owns. It adds the replies back in
the order the local loop would, so on one machine, with one binary, a cluster
run must print exactly what the local run prints: the same teacher-forcing
tokens, the same greedy tokens and the same last-position logits, byte for
byte. Anything short of that means an expert was computed on the wrong rows,
with the wrong gate, or summed in a different order.

The engine checks run on the streaming fixture (experts in the int4
container, the only kind a worker serves) when COLI_GLM53_FIXTURE points at
it, as the GLM-5.3 CI job does:

    python tools/make_glm53_multimodal_tiny.py --output /tmp/glm53_mm
    python tools/make_glm53_streaming_pair.py --fixture /tmp/glm53_mm --output /tmp/glm53_stream
    COLI_GLM53_FIXTURE=/tmp/glm53_stream-i4 python -m unittest -v tests.test_glm53_cluster

The routing reference runs everywhere with the standard library only.
"""
import json
import os
import re
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
SRC = (C_DIR / "glm53.c").read_text()
BINARY = next((C_DIR / name for name in ("glm53", "glm53.exe")
               if (C_DIR / name).exists()), None)
FIXTURE = os.environ.get("COLI_GLM53_FIXTURE")

# Numeric outputs only; the timing lines legitimately differ.
COMPARED = ("teacher_forcing", "last_logits", "greedy")


def owner(layer, eid, cuts):
    """Reference for glm53_cluster_owner()."""
    h = ((layer * 0x85EBCA6B) & 0xFFFFFFFF) ^ ((eid * 0xC2B2AE35) & 0xFFFFFFFF)
    h ^= h >> 16
    h = (h * 0x45D9F3B) & 0xFFFFFFFF
    h ^= h >> 16
    hv, w = h & 255, 0
    while w < len(cuts) - 1 and hv >= cuts[w]:
        w += 1
    return w


def cuts_for(weights):
    """Reference for glm53_cluster_set_cuts()."""
    total, accum, out = sum(weights), 0, []
    for w in weights:
        accum += w
        out.append(min(256, (256 * accum + total // 2) // total))
    out[-1] = 256
    return out


class Glm53ClusterRoutingTest(unittest.TestCase):
    def test_reference_matches_source(self):
        """The Python reference hashes with the constants the engine uses."""
        body = SRC[SRC.index("static int glm53_cluster_owner("):]
        body = body[:body.index("\n}\n")]
        for constant in ("0x85EBCA6Bu", "0xC2B2AE35u", "0x45d9f3bu"):
            self.assertIn(constant, body)

    def test_weights_set_the_shares(self):
        """A 3:2:1 split over the real model's 42 MoE layers x 288 experts
        lands near 50/33/17: the faster disk carries more of every layer."""
        cuts = cuts_for([3, 2, 1])
        counts = [0, 0, 0]
        for layer in range(3, 45):
            for eid in range(288):
                counts[owner(layer, eid, cuts)] += 1
        total = sum(counts)
        for got, want in zip(counts, (3 / 6, 2 / 6, 1 / 6)):
            self.assertAlmostEqual(got / total, want, delta=0.02)

    def test_owner_is_not_the_mirror_split(self):
        """A worker's own mirror split must not see only one side of its
        hash: the cluster hash is independent of glm53_expert_replica()."""
        def mirror(layer, eid):
            h = ((layer * 2654435761) & 0xFFFFFFFF) ^ ((eid * 0x9E3779B9) & 0xFFFFFFFF)
            h ^= h >> 16
            h = (h * 0x45D9F3B) & 0xFFFFFFFF
            h ^= h >> 16
            return (h & 255) < 128
        cuts = cuts_for([1, 1])
        mine = [(l, e) for l in range(3, 45) for e in range(288) if owner(l, e, cuts) == 0]
        share = sum(mirror(l, e) for l, e in mine) / len(mine)
        self.assertAlmostEqual(share, 0.5, delta=0.03)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def parse(stdout):
    out = {}
    for line in stdout.splitlines():
        if line.strip():
            out[line.split()[0]] = line.split()[1:]
    return out


@unittest.skipUnless(FIXTURE, "COLI_GLM53_FIXTURE not set to the int4 streaming fixture")
class Glm53ClusterEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert BINARY, "COLI_GLM53_FIXTURE is set but glm53 is not built"
        fixture = Path(FIXTURE)
        ref = json.loads((fixture / "ref.json").read_text())
        cls.args = [str(BINARY), "--model", str(fixture),
                    "--ids", ",".join(str(t) for t in ref["prompt"]),
                    "--greedy", "6", "--logits"]
        patches = fixture / "patches.f32"
        if patches.exists():
            grid = ref.get("grid", (0, 0))
            cls.args += ["--patches", str(patches), "--grid", f"{grid[0]}x{grid[1]}"]
        cls.env = {**os.environ, "GLM53_BITS": "32"}
        for name in ("CLUSTER_WORKERS", "COLI_CLUSTER_WEIGHTS", "EXPERT_WORKER"):
            cls.env.pop(name, None)
        cls.local = cls.run_engine({})

    @classmethod
    def run_engine(cls, extra, check=True):
        result = subprocess.run(cls.args, capture_output=True, text=True, timeout=300,
                                env={**cls.env, **extra})
        if check and result.returncode != 0:
            raise AssertionError(result.stderr)
        return result

    def start_workers(self, n, extra=None):
        workers, ports = [], []
        for _ in range(n):
            port = free_port()
            env = {**self.env, "EXPERT_WORKER": "1", "SNAP": FIXTURE,
                   "CLUSTER_WORKER_PORT": str(port), "CLUSTER_WORKER_BIND": "127.0.0.1",
                   **(extra or {})}
            proc = subprocess.Popen([str(BINARY)], env=env, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, text=True)
            self.addCleanup(self.stop, proc)
            workers.append(proc)
            ports.append(port)
        for proc, port in zip(workers, ports):
            deadline = time.time() + 30
            while True:
                if proc.poll() is not None:
                    self.fail(f"worker on {port} exited: {proc.stderr.read()}")
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=1).close()
                    break
                except OSError:
                    if time.time() > deadline:
                        self.fail(f"worker on {port} never listened")
                    time.sleep(0.05)
        return workers, ",".join(f"127.0.0.1:{p}" for p in ports)

    @staticmethod
    def stop(proc):
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if proc.stderr:
            proc.stderr.close()

    def assert_same_as_local(self, result):
        want, got = parse(self.local.stdout), parse(result.stdout)
        for key in COMPARED:
            self.assertIn(key, got, result.stderr)
            self.assertEqual(got[key], want[key], f"{key} differs from the local run")

    def test_cluster_is_bit_identical_to_local(self):
        """One, two and three workers print exactly what the local run prints,
        and with more than one worker the experts really are split."""
        for n in (1, 2, 3):
            with self.subTest(workers=n):
                _, spec = self.start_workers(n)
                result = self.run_engine({"CLUSTER_WORKERS": spec, "GLM53_VERBOSE": "1"})
                self.assert_same_as_local(result)
                served = re.findall(r"\[CLUSTER\] \S+: (\d+) requests, (\d+) rows", result.stderr)
                self.assertEqual(len(served), n, result.stderr)
                if n > 1:
                    self.assertGreater(sum(int(r) > 0 for r, _ in served), 1,
                                       "every expert went to one worker")

    def test_weights_override_and_route_everything_to_one_worker(self):
        """COLI_CLUSTER_WEIGHTS wins over the probes; a lopsided split still
        answers exactly, with the idle worker never asked."""
        _, spec = self.start_workers(2)
        result = self.run_engine({"CLUSTER_WORKERS": spec, "COLI_CLUSTER_WEIGHTS": "1000000,1",
                                  "GLM53_VERBOSE": "1"})
        self.assert_same_as_local(result)
        self.assertIn("(COLI_CLUSTER_WEIGHTS)", result.stderr)
        self.assertRegex(result.stderr, r"routing \S+ 100% / \S+ 0%")

    def test_worker_loss_is_an_error_not_a_wrong_answer(self):
        """A worker that dies mid-run stops the coordinator with a named
        error; it never lets a layer finish without that worker's experts."""
        workers, spec = self.start_workers(1)
        env = {**self.env, "CLUSTER_WORKERS": spec}
        coordinator = subprocess.Popen(self.args, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.stop, coordinator)
        time.sleep(0.2)
        workers[0].kill()
        out, err = coordinator.communicate(timeout=120)
        if coordinator.returncode == 0:
            # the tiny run can finish before the kill lands; then it must be right
            self.assert_same_as_local(subprocess.CompletedProcess(self.args, 0, out, err))
        else:
            self.assertRegex(err, r"\[CLUSTER\] (expert worker \S+ failed during layer|"
                                  r"cannot connect|no expert workers)")

    def test_no_reachable_worker_is_fatal(self):
        result = self.run_engine({"CLUSTER_WORKERS": f"127.0.0.1:{free_port()}"}, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no expert workers reachable", result.stderr)

    def test_stranger_is_dropped(self):
        """A client that does not speak the protocol is disconnected and the
        worker keeps serving the real coordinator."""
        _, spec = self.start_workers(1)
        host, port = spec.rsplit(":", 1)
        with socket.create_connection((host, int(port)), timeout=5) as s:
            s.sendall(b"GET / HTTP/1.0\r\n\r\n" + b"\0" * 32)
            s.settimeout(5)
            try:
                self.assertEqual(s.recv(64), b"")
            except ConnectionResetError:
                pass            # closed with our junk unread: the kernel sends RST
        self.assert_same_as_local(self.run_engine({"CLUSTER_WORKERS": spec}))


if __name__ == "__main__":
    unittest.main()
