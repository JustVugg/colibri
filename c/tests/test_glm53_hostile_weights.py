"""glm53 on a container whose router weights were chosen by someone else.

An all-NaN router row: NaN > best is false for every expert, so the top-k pick
stayed at -1, and the gate weight was read as score[-1] before the id reached the
expert cache. GHSA-5xpg-vw35-2687 fixed this shape in inkling, kimi_k3 and olmoe
through rt_router_pick() in route_trace.h, which glm53 already includes.

Runs in the GLM-5.3 oracle job of check.yml, on the streaming fixture it builds
(the same one test_glm53_context_exceeded uses):

  COLI_GLM53_FIXTURE=/tmp/glm53_stream-i4 python -m unittest -v tests.test_glm53_hostile_weights
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from safetensors_edit import copy_fixture, fill_nan  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
BINARY = next((path for path in (HERE / "glm53.exe", HERE / "glm53") if path.exists()), None)
FIXTURE = Path(os.environ.get("COLI_GLM53_FIXTURE", ""))
ROUTER = "model.language_model.layers.3.mlp.gate.weight"   # the fixture's one MoE layer
SANITIZER = ("ERROR: AddressSanitizer", "runtime error:")


def serve_turn(snapshot, prompt=b"abc"):
    """One SERVE turn; return (the line that ends it, the engine's stderr)."""
    environment = {**os.environ, "SNAP": str(snapshot), "SERVE": "1",
                   "GLM53_BITS": "32", "GLM53_MAXT": "64"}
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([str(BINARY)], env=environment, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=errors, bufsize=0)
        ending = None
        try:
            while b"READY" not in (line := process.stdout.readline()):
                if not line:
                    break
            else:
                process.stdin.write(f"SUBMIT 7 0 {len(prompt)} 4 0 1\n".encode() + prompt + b"\n")
                process.stdin.flush()
                for _ in range(400):
                    line = process.stdout.readline()
                    if not line:
                        break
                    text = line.decode("latin-1").rstrip("\n")
                    if text.startswith("DATA "):
                        process.stdout.read(int(text.split()[2]))
                        process.stdout.readline()
                    elif text.startswith(("ERROR ", "DONE ")):
                        ending = text
                        break
        finally:
            process.stdin.close()
            try:
                process.wait(60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()
        errors.seek(0)
        return ending, errors.read().decode("utf-8", "replace")


@unittest.skipUnless(BINARY, "glm53 engine is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "config.json").is_file(),
                     "COLI_GLM53_FIXTURE not set to a GLM-5.3-Flash container")
class Glm53HostileWeightsTest(unittest.TestCase):
    def test_an_all_nan_router_degrades_instead_of_indexing_minus_one(self):
        with tempfile.TemporaryDirectory() as scratch:
            snapshot = copy_fixture(FIXTURE, Path(scratch) / "model")
            fill_nan(snapshot, ROUTER)
            ending, stderr = serve_turn(snapshot)
        for marker in SANITIZER:
            self.assertNotIn(marker, stderr, stderr[-4000:])
        self.assertIsNotNone(ending, "the engine died mid-turn:\n" + stderr[-4000:])
        self.assertTrue(ending.startswith("DONE 7 "), ending)
        self.assertIn("[router] non-finite logits", stderr)


if __name__ == "__main__":
    unittest.main()
