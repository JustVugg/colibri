"""A request whose engine went before it started runs on the engine keep_engine starts next.

An engine exits between two requests when it lost its GPU for real (vkc_restart_if_lost: the
device does not come back in its process); the gateway's keep_engine starts it again. The request
the gateway had already written to the old one, and those sent while the new one loads, used to
fail ("colibri engine exited unexpectedly"). Now APIServer.generate waits for the new engine and
runs such a request there once; a request that had produced output, or that fails again, fails.
"""
import os
import sys
import threading
import time
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openai_server as srv  # noqa: E402


class Gone:
    """an engine whose process ended: generate fails as the dispatcher's error says"""
    alive = False

    def __init__(self, message="colibri engine exited unexpectedly (exit status 75)", emit=""):
        self.message, self.emit = message, emit

    def generate(self, prompt, max_tokens, temperature, top_p, on_text, *args, **kwargs):
        if self.emit:
            on_text(self.emit)
        raise RuntimeError(self.message)


class Fresh:
    alive = True

    def generate(self, prompt, max_tokens, temperature, top_p, on_text, *args, **kwargs):
        on_text("hello")
        return {"prompt_tokens": 3, "completion_tokens": 1}


def server(engine):
    s = types.SimpleNamespace(engine=engine, ENGINE_RETRY_WAIT_S=5,
                              scheduler=types.SimpleNamespace(observe_timing=lambda *a: None),
                              note_prompt_size=lambda *a, **k: None)
    s.await_fresh_engine = lambda old, timeout=None: srv.APIServer.await_fresh_engine(s, old, timeout)
    return s


def swap_later(s, engine, after=0.3):
    def go():
        time.sleep(after)
        s.engine = engine
    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


class EngineRetryTest(unittest.TestCase):
    def test_a_request_not_started_runs_on_the_next_engine(self):
        s = server(Gone())
        swap_later(s, Fresh())
        out = []
        stats = srv.APIServer.generate(s, "hi", 8, 0.0, 1.0, out.append)
        self.assertEqual(out, ["hello"])
        self.assertEqual(stats["completion_tokens"], 1)

    def test_a_request_written_to_a_gone_engine_too(self):
        s = server(Gone("colibri engine dispatcher stopped: colibri engine exited unexpectedly (exit status 75)"))
        swap_later(s, Fresh())
        self.assertEqual(srv.APIServer.generate(s, "hi", 8, 0.0, 1.0, lambda t: None)["prompt_tokens"], 3)

    def test_a_request_that_had_output_fails(self):
        s = server(Gone(emit="partial"))
        swap_later(s, Fresh())
        with self.assertRaises(RuntimeError):
            srv.APIServer.generate(s, "hi", 8, 0.0, 1.0, lambda t: None)

    def test_another_error_is_the_requests(self):
        s = server(Gone("prompt too long"))
        swap_later(s, Fresh())
        with self.assertRaises(RuntimeError):
            srv.APIServer.generate(s, "hi", 8, 0.0, 1.0, lambda t: None)

    def test_no_engine_comes_back(self):
        s = server(Gone())
        s.ENGINE_RETRY_WAIT_S = 0.5
        with self.assertRaises(RuntimeError):
            srv.APIServer.generate(s, "hi", 8, 0.0, 1.0, lambda t: None)

    def test_the_error_names(self):
        self.assertTrue(srv.engine_gone_error(RuntimeError("colibri engine exited unexpectedly")))
        self.assertTrue(srv.engine_gone_error(RuntimeError("colibri engine dispatcher stopped: x")))
        self.assertFalse(srv.engine_gone_error(RuntimeError("engine_error")))


if __name__ == "__main__":
    unittest.main()
