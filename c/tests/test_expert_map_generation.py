"""Actual dispatcher and HTTP tests, without a model or engine subprocess."""
import collections
import json
import queue
import threading
import types
import unittest
from urllib.request import Request, urlopen

from openai_server import Engine
from test_openai_server import FakeEngine, _spawn_test_server


class ExpertMapGenerationTest(unittest.TestCase):
    def dispatch(self, frames, initial=None):
        engine = Engine.__new__(Engine)
        engine.emap = initial
        engine.hits = None
        engine.hits_seq = 0
        engine.closed = False
        engine.dispatcher_error = None
        # Current upstream DONE also clears the independent live-turn map.
        engine.live_lock = threading.Lock()
        engine.live = {}
        engine.pending_lock = threading.Lock()
        pending = types.SimpleNamespace(events=queue.Queue(), failed=None)
        engine.pending = {"7": pending}
        snapshots = []
        iterator = iter(frames)

        class Stream:
            def readline(self):
                snapshots.append((engine.emap, engine.hits_seq, "7" in engine.pending))
                try:
                    return (next(iterator) + "\n").encode()
                except StopIteration:
                    engine.closed = True  # Normal end of this finite mock stream.
                    return b""

        engine.process = types.SimpleNamespace(stdout=Stream())
        engine._dispatch_stdout()
        self.assertIsNone(engine.dispatcher_error)
        return engine, snapshots, pending

    def test_initial_map_has_independent_generation(self):
        engine, _, _ = self.dispatch(["EMAP 1 2 4040"])
        self.assertEqual(engine.emap, {"rows": 1, "cols": 2, "map": "4040",
                                     "map_seq": 1, "map_hits_seq": 0})

    def test_hits_do_not_advance_map_generation(self):
        engine, _, _ = self.dispatch(["EMAP 1 2 4040", "HITS 1 2 f"])
        self.assertEqual(engine.hits_seq, 1)
        self.assertEqual(engine.emap["map_seq"], 1)
        self.assertEqual(engine.emap["map_hits_seq"], 0)

    def test_done_does_not_falsely_refresh_the_old_map(self):
        engine, _, pending = self.dispatch(["EMAP 1 2 4040", "HITS 1 2 f",
                                           "DONE 7 STAT 1 1.0 100 0.1 3 0"])
        self.assertEqual(pending.events.get_nowait()[0], "done")
        self.assertNotIn("7", engine.pending)
        self.assertLess(engine.emap["map_hits_seq"], engine.hits_seq)

    def test_delayed_map_publishes_correlated_generation(self):
        engine, snapshots, _ = self.dispatch(["EMAP 1 2 4040", "HITS 1 2 f",
                                              "DONE 7 STAT 1 1.0 100 0.1 3 0",
                                              "EMAP 1 2 8080"])
        old = snapshots[3][0]
        self.assertFalse(snapshots[3][2])  # DONE was already dispatched.
        self.assertEqual(old["map_seq"], 1)
        self.assertEqual(old["map_hits_seq"], 0)
        self.assertEqual(old["map"], "4040")
        self.assertIsNot(old, engine.emap)
        self.assertEqual(engine.emap["map_seq"], 2)
        self.assertEqual(engine.emap["map_hits_seq"], 1)
        self.assertEqual(engine.emap["map"], "8080")

    def test_repeated_placement_still_has_a_new_snapshot(self):
        engine, _, _ = self.dispatch(["EMAP 1 2 4040", "EMAP 1 2 4040"])
        self.assertEqual(engine.emap["map_seq"], 2)
        self.assertEqual(engine.emap["map_hits_seq"], 0)

    def test_map_records_latest_of_multiple_hits_generations(self):
        engine, _, _ = self.dispatch(["HITS 1 2 a", "HITS 1 2 b", "EMAP 1 2 4040"])
        self.assertEqual(engine.emap["map_hits_seq"], 2)
        self.assertEqual(engine.hits, "b")

    def test_legacy_initial_map_is_not_mutated(self):
        legacy = {"rows": 1, "cols": 1, "map": "40"}
        engine, _, _ = self.dispatch(["EMAP 1 2 4040"], legacy)
        self.assertEqual(engine.emap["map_seq"], 1)
        self.assertNotIn("map_seq", legacy)


class ExpertMapHttpTest(unittest.TestCase):
    def endpoint(self, emap):
        engine = FakeEngine()
        engine.emap = emap
        engine.hits = "f"
        engine.hits_seq = 2
        return engine, _spawn_test_server(self, engine)

    def get(self, base, authorized=True, query=""):
        headers = {"Authorization": "Bearer secret"} if authorized else {}
        with urlopen(Request(base + "/experts" + query, headers=headers), timeout=5) as response:
            self.assertEqual(response.status, 200)
            return json.loads(response.read())

    def test_authenticated_response_exposes_both_generations(self):
        snapshot = {"rows": 1, "cols": 2, "map": "4040", "map_seq": 4, "map_hits_seq": 1}
        engine, base = self.endpoint(snapshot)
        body = self.get(base)
        self.assertEqual(body, dict(snapshot, hits="f", seq=2, live_seq=0))
        self.assertIs(engine.emap, snapshot)
        self.assertNotIn("hits", snapshot)

    def test_no_map_has_explicit_zero_generations(self):
        _, base = self.endpoint(None)
        body = self.get(base)
        self.assertEqual((body["map_seq"], body["map_hits_seq"], body["seq"]), (0, 0, 0))
        self.assertEqual(body["map"], "")

    def test_unauthenticated_response_does_not_leak_generations(self):
        _, base = self.endpoint({"rows": 1, "cols": 2, "map": "8080", "map_seq": 55, "map_hits_seq": 2})
        body = self.get(base, authorized=False)
        self.assertEqual(body, {"rows": 0, "cols": 0, "map": "", "hits": "", "seq": 0,
                               "map_seq": 0, "map_hits_seq": 0})



    def test_since_uses_compact_live_shape_without_regression(self):
        snapshot = {"rows": 1, "cols": 2, "map": "4040",
                    "map_seq": 4, "map_hits_seq": 1}
        engine, base = self.endpoint(snapshot)
        engine.hits_live_seq = 5
        engine.hits_live = collections.deque([(3, "0010"), (5, "0020")])
        body = self.get(base, query="?since=3")
        self.assertEqual(body, {"rows": 1, "cols": 2,
                                "live_seq": 5, "steps": ["0020"]})

    def test_since_unauthorized_redacts_live_and_map_generations(self):
        snapshot = {"rows": 1, "cols": 2, "map": "4040",
                    "map_seq": 4, "map_hits_seq": 1}
        engine, base = self.endpoint(snapshot)
        engine.hits_live_seq = 99
        engine.hits_live = collections.deque([(99, "0020")])
        body = self.get(base, authorized=False, query="?since=0")
        self.assertEqual(body, {"rows": 0, "cols": 0, "map": "", "hits": "",
                                "seq": 0, "map_seq": 0, "map_hits_seq": 0})


if __name__ == "__main__":
    unittest.main()
