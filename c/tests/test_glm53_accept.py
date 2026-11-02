"""GLM-5.3 acknowledges a valid native request before its first payload.

Uses the same optional fixture as the existing GLM-5.3 oracle jobs. The source
ordering check also runs in the dependency-free suite, without model weights.
"""
import os
from pathlib import Path
import subprocess
import threading
import queue
import unittest

HERE = Path(__file__).resolve().parent.parent
BINARY = next((p for p in (HERE / "glm53.exe", HERE / "glm53") if p.is_file()), None)
# Reuse the generated-fixture variables already set by upstream oracle CI.
# An explicit standalone override takes precedence and must not silently skip.
def configured_fixture(environment):
    for variable in ("COLI_GLM53_FIXTURE", "GLM53_MM_TINY", "GLM53_TINY"):
        value = environment.get(variable)
        if value:
            return Path(value)
    return None

FIXTURE = configured_fixture(os.environ)

class Glm53FixtureSelectionTest(unittest.TestCase):
    def test_unconfigured_is_optional(self):
        self.assertIsNone(configured_fixture({}))

    def test_generated_ci_variables_are_used(self):
        self.assertEqual(configured_fixture({"GLM53_MM_TINY": "mm"}), Path("mm"))
        self.assertEqual(configured_fixture({"GLM53_TINY": "text"}), Path("text"))

    def test_precedence_preserves_explicit_override(self):
        self.assertEqual(configured_fixture({"COLI_GLM53_FIXTURE": "chosen",
            "GLM53_MM_TINY": "mm", "GLM53_TINY": "text"}), Path("chosen"))
        self.assertEqual(configured_fixture({"GLM53_MM_TINY": "mm",
            "GLM53_TINY": "text"}), Path("mm"))

class Glm53AcceptOrderTest(unittest.TestCase):
    def test_accept_is_after_validation_and_before_restore_and_prefill(self):
        source = (HERE / "glm53.c").read_text(encoding="utf-8")
        begin = source.index("static int serve_one(")
        body = source[begin:source.index("\nstatic ", begin + 1)]
        accepted = body.index('serve_line("ACCEPT ')
        self.assertGreater(accepted, body.index("kv_keys_build("))
        self.assertGreater(accepted, body.index("CONTEXT_EXCEEDED"))
        self.assertLess(accepted, body.index("int pinned ="))
        self.assertLess(accepted, body.index("forward_prefill("))
        self.assertEqual(body.count('serve_line("ACCEPT '), 1)

@unittest.skipUnless(FIXTURE is not None, "requires a configured GLM-5.3 fixture")
class Glm53AcceptWireTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A configured but unusable fixture is a failed gate, never a skip.
        if BINARY is None:
            raise AssertionError("GLM-5.3 fixture configured but glm53 is not built")
        for name in ("config.json", "tokenizer.json"):
            if not (FIXTURE / name).is_file():
                raise AssertionError(f"Configured GLM-5.3 fixture is missing {name}")

    def first_request_frame(self, prompt, slot=0, raw_frame=None):
        env = dict(os.environ, SNAP=str(FIXTURE), SERVE="1", SERVE_BATCH="1",
                   GLM53_BITS="32", GLM53_MAXT="64", OMP_NUM_THREADS="1")
        p = subprocess.Popen([str(BINARY)], env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        lines = queue.Queue()
        def read_lines():
            for line in p.stdout: lines.put(line)
            lines.put(b"")
        reader = threading.Thread(target=read_lines, daemon=True); reader.start()
        try:
            for _ in range(100):
                line = lines.get(timeout=30)
                if b"READY" in line: break
                if not line: self.fail("engine exited before READY")
            else: self.fail("no READY frame")
            if raw_frame is None:
                raw_frame = (f"SUBMIT 7 {slot} {len(prompt)} 1 0 1\n".encode()
                             + prompt + b"\n")
            p.stdin.write(raw_frame)
            p.stdin.flush()
            for _ in range(100):
                line = lines.get(timeout=30).decode("utf-8", "replace").strip()
                if line.startswith(("ACCEPT ", "DATA ", "ECHO ", "DONE ", "ERROR ")):
                    return line.split()
            self.fail("no request frame")
        finally:
            p.stdin.close()
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()  # Only this disposable fixture subprocess.
            reader.join(timeout=2); p.stdout.close()

    def test_valid_request_is_acknowledged_before_payload(self):
        fields = self.first_request_frame(b"xxx")
        self.assertEqual(fields[:2], ["ACCEPT", "7"])
        self.assertEqual(len(fields), 3); self.assertGreater(int(fields[2]), 0)

    def test_invalid_slot_is_not_accepted(self):
        self.assertEqual(self.first_request_frame(b"xxx", slot=999)[:3],
                         ["ERROR", "7", "BAD_REQUEST"])

    def test_overlong_prompt_is_not_accepted(self):
        self.assertEqual(self.first_request_frame(b"x" * 120)[:3],
                         ["ERROR", "7", "CONTEXT_EXCEEDED"])


    def test_malformed_header_is_rejected_before_accept(self):
        # Unparseable byte count must be BAD_FRAME, never ACCEPT.
        frame = b"SUBMIT 7 0 nonsense 1 0 1\n"
        self.assertEqual(self.first_request_frame(b"", raw_frame=frame)[:3],
                         ["ERROR", "7", "BAD_FRAME"])

    def test_empty_prompt_is_rejected_before_accept(self):
        self.assertEqual(self.first_request_frame(b"")[:3],
                         ["ERROR", "7", "EMPTY_PROMPT"])


if __name__ == "__main__": unittest.main()
