"""GLM-5.3 prefill batching must not change a single bit.

A prefill block runs every dense matrix once over all of its tokens instead of
once per token. The kernels compute each (token, output) pair exactly as the
one-row call does, so the block size may change the speed and nothing else:
GLM53_PREFILL_CHUNK=1 (every matrix called one row at a time) and larger
chunks must print the same teacher-forcing tokens, greedy tokens and
last-position logits, byte for byte, at every dense precision.

Runs on whichever fixtures are set, as the GLM-5.3 CI job provides them:
GLM53_TINY (tools/make_glm53_tiny.py, resident f32 experts) and
COLI_GLM53_FIXTURE (the int4 streaming fixture, which exercises the routed
experts' batched CPU loop).
"""
import json
import os
import subprocess
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parents[1]
BINARY = next((C_DIR / name for name in ("glm53", "glm53.exe")
               if (C_DIR / name).exists()), None)
COMPARED = ("teacher_forcing", "last_logits", "greedy")


def fixtures():
    found = []
    for variable in ("GLM53_TINY", "COLI_GLM53_FIXTURE"):
        value = os.environ.get(variable)
        if value:
            found.append(Path(value))
    return found


def command(fixture):
    ref = json.loads((fixture / "ref.json").read_text())
    patches = fixture / "patches.f32"
    if patches.exists():
        grid = ref.get("grid", (0, 0))
        ids = ref["prompt"]
        extra = ["--patches", str(patches), "--grid", f"{grid[0]}x{grid[1]}"]
    else:
        ids = ref.get("prompt_ids") or ref["prompt"]
        extra = []
    return [str(BINARY), "--model", str(fixture), "--ids", ",".join(map(str, ids)),
            *extra, "--greedy", "4", "--logits"]


def run(fixture, bits, chunk):
    env = {**os.environ, "GLM53_BITS": str(bits), "GLM53_PREFILL_CHUNK": str(chunk)}
    for name in ("CLUSTER_WORKERS", "EXPERT_WORKER"):
        env.pop(name, None)
    result = subprocess.run(command(fixture), capture_output=True, text=True,
                            timeout=300, env=env)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    lines = {}
    for line in result.stdout.splitlines():
        if line.strip():
            lines[line.split()[0]] = line.split()[1:]
    return lines


@unittest.skipUnless(fixtures(), "neither GLM53_TINY nor COLI_GLM53_FIXTURE is set")
class Glm53PrefillBatchTest(unittest.TestCase):
    def test_chunk_size_never_changes_the_output(self):
        self.assertIsNotNone(BINARY, "a fixture is set but glm53 is not built")
        for fixture in fixtures():
            for bits in (32, 8, 4):
                one = run(fixture, bits, 1)
                for key in COMPARED:          # a refused run must not pass as "equal"
                    self.assertTrue(one.get(key), f"{fixture.name}: no {key} at chunk 1")
                for chunk in (3, 7, 128):
                    with self.subTest(fixture=fixture.name, bits=bits, chunk=chunk):
                        got = run(fixture, bits, chunk)
                        for key in COMPARED:
                            self.assertEqual(got.get(key), one[key],
                                             f"{key} at chunk {chunk} differs from chunk 1")


if __name__ == "__main__":
    unittest.main()
