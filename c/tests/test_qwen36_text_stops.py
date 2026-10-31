import os
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from openai_server import StopFilter, stop_policy
from tests.prefix_serve_harness import ServeEngine, ensure_byte_tokenizer

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen36.exe" if sys.platform == "win32" else "qwen36")
FIXTURE = Path(os.environ.get("QWEN36_TINY", HERE / "qwen36_tiny_c"))


class QwenTextStopTest(unittest.TestCase):
    def test_spelled_role_markers_do_not_leak_across_chunks(self):
        with patch("openai_server.ARCH", "qwen36"):
            stops, ignore = stop_policy({}, True)
        for marker in ("<|im_end|>", "<|im_start|>"):
            for split in range(1, len(marker)):
                with self.subTest(marker=marker, split=split):
                    output = []
                    filt = StopFilter(stops, output.append, ignore)
                    filt.feed("Answer." + marker[:split])
                    filt.feed(marker[split:] + "user\nInvented next turn")
                    filt.finish()
                    self.assertEqual("".join(output), "Answer.")
                    self.assertTrue(filt.stopped())

    def test_explicit_stops_and_raw_completions_keep_their_policy(self):
        with patch("openai_server.ARCH", "qwen36"):
            self.assertEqual(stop_policy({"stop": "END"}, True), (("END",), False))
            self.assertEqual(stop_policy({}, False), ((), False))
        with patch("openai_server.ARCH", "qwen38"):
            self.assertEqual(stop_policy({}, True), ((), False))

    def test_explicit_ignore_leading_stop_is_honored(self):
        with patch("openai_server.ARCH", "qwen36"):
            stops, ignore = stop_policy({"x_colibri_ignore_leading_stop": True}, True)
        output = []
        filt = StopFilter(stops, output.append, ignore)
        filt.feed("<|im_")
        filt.feed("start|>Answer.<|im_end|>next turn")
        filt.finish()
        self.assertEqual("".join(output), "Answer.")
        self.assertTrue(filt.stopped())


@unittest.skipUnless(ENGINE.exists() and (FIXTURE / "model-globals.safetensors").exists(),
                     "build qwen36 and generate the tiny fixture first")
class QwenStopProtocolTest(unittest.TestCase):
    def test_stop_succeeds_and_cancel_remains_an_error(self):
        ensure_byte_tokenizer(FIXTURE)
        for slots in (1, 2):
            for command in ("STOP", "CANCEL"):
                with self.subTest(slots=slots, command=command):
                    engine = ServeEngine(ENGINE, ["8", "8"], {
                        "SNAP": str(FIXTURE), "SERVE": "1", "COLI_DENSE_I8": "0",
                        "KV_SLOTS": str(slots), "Q36_MAXT": "1024"})
                    timer = threading.Timer(30, engine.p.kill)
                    timer.daemon = True
                    timer.start()
                    try:
                        prompt = b"test normal stop"
                        engine.p.stdin.write(f"SUBMIT control 0 {len(prompt)} 128 0 1\n".encode()
                                             + prompt + b"\n")
                        engine.p.stdin.flush()
                        sent = False
                        while True:
                            line = engine.p.stdout.readline()
                            self.assertTrue(line, "engine exited before terminal frame")
                            text = line.decode("latin-1").strip()
                            if text.startswith("DATA control "):
                                engine.p.stdout.read(int(text.split()[2]))
                                engine.p.stdout.readline()
                                if not sent:
                                    engine.p.stdin.write(f"{command} control\n".encode())
                                    engine.p.stdin.flush()
                                    sent = True
                            if text.startswith(("DONE control ", "ERROR control ")):
                                self.assertTrue(sent)
                                if command == "STOP":
                                    self.assertTrue(text.startswith("DONE control STAT "), text)
                                    self.assertEqual(text.split()[-1], "0", "STOP is not length-limited")
                                else:
                                    self.assertEqual(text, "ERROR control CANCELLED")
                                break
                        engine.ask("after-control", b"still responsive", max_tok=1)
                    finally:
                        engine.close()
                        timer.cancel()


if __name__ == "__main__":
    unittest.main()
