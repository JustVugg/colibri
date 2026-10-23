"""A GLM-5.3 prompt longer than GLM53_MAXT is a 400 context_length_exceeded, not a 500.

The gateway maps one engine frame to the OpenAI error clients act on:
`ERROR <id> CONTEXT_EXCEEDED ...` (#506, both spellings since #1381). glm53
refused a prompt that does not fit its session with `ERROR <id> BAD_REQUEST`,
which the gateway can only treat as an engine fault: HTTP 500 "The colibri
engine failed to process the request.", streaming or not. A client that
compacts its conversation on context_length_exceeded never gets the chance to.
olmoe and inkling had the same gap (#1483).

What is asserted is the frame the real engine writes on a real container, and
what the gateway's own mapping makes of that frame. Needs a GLM-5.3-Flash
container and a built `glm53`, the ones the generated GLM-5.3 oracle job builds.
"""
import os
import subprocess
import unittest
from pathlib import Path

from openai_server import APIError, _engine_error

HERE = Path(__file__).resolve().parent.parent
# Found the way test_glm53_dashboard finds it: both run in the GLM-5.3 oracle
# job of check.yml, which builds glm53 and the fixture; the Python job of
# ci.yml cannot generate the fixture (it needs PyTorch and Transformers).
BINARY = next((path for path in (HERE / "glm53.exe", HERE / "glm53") if path.exists()), None)
FIXTURE = Path(os.environ.get("COLI_GLM53_FIXTURE", ""))
MAXT = 64                               # the smallest session glm53 accepts
MAX_TOKENS = 4


def serve_turn(prompt):
    """Submit one prompt in serve mode; return the line that ends the turn."""
    environment = {**os.environ, "SNAP": str(FIXTURE), "SERVE": "1",
                   "GLM53_BITS": "32", "GLM53_MAXT": str(MAXT)}
    process = subprocess.Popen([str(BINARY)], env=environment,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, bufsize=0)
    try:
        while b"READY" not in (line := process.stdout.readline()):
            if not line:
                raise AssertionError("the engine exited before READY")
        process.stdin.write(f"SUBMIT 7 0 {len(prompt)} {MAX_TOKENS} 0 1\n".encode()
                            + prompt + b"\n")
        process.stdin.flush()
        for _ in range(400):
            line = process.stdout.readline()
            if not line:
                raise AssertionError("the engine closed the stream mid-turn")
            text = line.decode("latin-1").rstrip("\n")
            if text.startswith("DATA "):
                process.stdout.read(int(text.split()[2]))
                process.stdout.readline()
            elif text.startswith(("ERROR ", "DONE ")):
                return text
        raise AssertionError("no ERROR or DONE within 400 lines")
    finally:
        process.stdin.close()
        try:
            process.wait(60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()


@unittest.skipUnless(BINARY, "glm53 engine is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "config.json").is_file(),
                     "COLI_GLM53_FIXTURE not set to a GLM-5.3-Flash container")
class Glm53ContextExceededTest(unittest.TestCase):
    OVER_LONG = b"abc" * 40             # 120 bytes, more tokens than MAXT holds

    def test_an_over_long_prompt_is_refused_with_the_context_frame(self):
        fields = serve_turn(self.OVER_LONG).split()
        self.assertEqual(fields[:3], ["ERROR", "7", "CONTEXT_EXCEEDED"], fields)
        numbers = dict(field.split("=", 1) for field in fields[3:])
        self.assertEqual(set(numbers), {"prompt_tokens", "requested", "capacity"})
        # The encoder stops at the session size, so the count is a lower bound.
        self.assertGreaterEqual(int(numbers["prompt_tokens"]), MAXT)
        self.assertEqual(numbers["requested"], str(MAX_TOKENS))
        self.assertEqual(numbers["capacity"], str(MAXT))

    def test_the_gateway_answers_it_with_a_400(self):
        fields = serve_turn(self.OVER_LONG).split()
        error = _engine_error(fields[2:], " ".join(fields[2:]))
        self.assertIsInstance(error, APIError,
                              "the gateway reads this frame as an engine fault (HTTP 500)")
        self.assertEqual((error.status, error.code), (400, "context_length_exceeded"))
        self.assertIn(f"maximum context length is {MAXT} tokens", error.message)

    def test_a_prompt_that_fits_is_still_served(self):
        self.assertTrue(serve_turn(b"abc").startswith("DONE 7 "))


if __name__ == "__main__":
    unittest.main()
