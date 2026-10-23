"""An OLMoE prompt longer than CTX is a 400 context_length_exceeded, not a 500.

The gateway maps one engine frame to the OpenAI error clients act on:
`ERROR <id> CONTEXT_EXCEEDED ...` (#506, both spellings since #1381). OLMoE
refused an over-long prompt with free text instead,
`ERROR <id> context exceeds CTX (30 + 4 > 16)`, which the gateway can only
treat as an engine fault: HTTP 500 "The colibri engine failed to process the
request.", streaming or not. A client that compacts its conversation on
context_length_exceeded never gets the chance to.

What is asserted is the frame the real engine writes on a real container, and
what the gateway's own mapping makes of that frame. Needs a converted tiny
OLMoE container and a built engine, the ones the OLMoE tiny oracle job builds.
"""
import os
import subprocess
import unittest
from pathlib import Path

from openai_server import APIError, _engine_error

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("olmoe.exe" if os.name == "nt" else "olmoe")
FIXTURE = Path(os.environ.get("COLI_OLMOE_FIXTURE", ""))
CTX = 16
MAX_TOKENS = 4


def serve_turn(prompt):
    """Submit one prompt in serve mode; return the line that ends the turn."""
    environment = {**os.environ, "SNAP": str(FIXTURE), "SERVE": "1", "CTX": str(CTX)}
    process = subprocess.Popen([str(ENGINE), "4", "8"], env=environment,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, bufsize=0)
    try:
        while b"READY" not in (line := process.stdout.readline()):
            if not line:
                raise AssertionError("the engine exited before READY")
        process.stdin.write(f"SUBMIT 7 0 {len(prompt)} {MAX_TOKENS} 0 1\n".encode()
                            + prompt + b"\n")
        process.stdin.flush()
        for _ in range(200):
            line = process.stdout.readline()
            if not line:
                raise AssertionError("the engine closed the stream mid-turn")
            text = line.decode("latin-1").rstrip("\n")
            if text.startswith("DATA "):
                process.stdout.read(int(text.split()[2]))
                process.stdout.readline()
            elif text.startswith(("ERROR ", "DONE ")):
                return text
        raise AssertionError("no ERROR or DONE within 200 lines")
    finally:
        process.stdin.close()
        try:
            process.wait(30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()


@unittest.skipUnless(ENGINE.exists(), "olmoe engine is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "config.json").is_file(),
                     "COLI_OLMOE_FIXTURE not set to a converted OLMoE container")
class OlmoeContextExceededTest(unittest.TestCase):
    OVER_LONG = b"abc" * 10          # 30 byte-level tokens, 30 + 4 > CTX

    def test_an_over_long_prompt_is_refused_with_the_context_frame(self):
        fields = serve_turn(self.OVER_LONG).split()
        self.assertEqual(fields[:3], ["ERROR", "7", "CONTEXT_EXCEEDED"], fields)
        numbers = dict(field.split("=", 1) for field in fields[3:])
        self.assertEqual(numbers, {"prompt_tokens": "30",
                                   "requested": str(MAX_TOKENS),
                                   "capacity": str(CTX)})

    def test_the_gateway_answers_it_with_a_400(self):
        fields = serve_turn(self.OVER_LONG).split()
        error = _engine_error(fields[2:], " ".join(fields[2:]))
        self.assertIsInstance(error, APIError,
                              "the gateway reads this frame as an engine fault (HTTP 500)")
        self.assertEqual((error.status, error.code), (400, "context_length_exceeded"))
        self.assertIn(f"maximum context length is {CTX} tokens", error.message)
        self.assertIn("at least 30 tokens", error.message)

    def test_a_prompt_that_fits_is_still_served(self):
        self.assertTrue(serve_turn(b"abc").startswith("DONE 7 "))


if __name__ == "__main__":
    unittest.main()
