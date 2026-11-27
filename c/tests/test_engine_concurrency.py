"""An engine that runs fewer requests at once than it has slots says so (CAPS concurrent=N), and the server
keeps the others in its queue.

glm53 serves one request at a time over its KV_SLOTS conversations and answers SLOT_BUSY to a second SUBMIT.
With --kv-slots 4 the scheduler sent it four requests at once: three failed with HTTP 500. Now the engine's
CAPS line carries concurrent=1 and the scheduler admits one request at a time, the slots still choosing the
cached conversation each one goes to.
"""
import os
import sys
import threading
import time
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openai_server as srv  # noqa: E402


class ConcurrencyTest(unittest.TestCase):
    def test_caps_or_the_slots(self):
        e = lambda caps: types.SimpleNamespace(caps=caps)
        self.assertEqual(srv.engine_concurrency(e({"concurrent": "1"}), 4), 1)
        self.assertEqual(srv.engine_concurrency(e({}), 4), 4)
        self.assertEqual(srv.engine_concurrency(e({"concurrent": "x"}), 4), 4)
        self.assertEqual(srv.engine_concurrency(e({"concurrent": "9"}), 4), 4)
        self.assertEqual(srv.engine_concurrency(e({"concurrent": "0"}), 4), 1)
        self.assertEqual(srv.engine_concurrency(types.SimpleNamespace(), 2), 2)

    def test_one_at_a_time_over_four_slots(self):
        sch = srv.GenerationScheduler(max_queue=8, queue_timeout=30, capacity=4)
        sch.set_concurrency(1)
        running, most, slots, lock = [0], [0], [], threading.Lock()

        def one():
            with sch.admit() as (_, slot):
                with lock:
                    running[0] += 1
                    most[0] = max(most[0], running[0])
                    slots.append(slot)
                time.sleep(0.15)
                with lock:
                    running[0] -= 1

        th = [threading.Thread(target=one) for _ in range(4)]
        for t in th:
            t.start()
        for t in th:
            t.join(10)
        self.assertEqual(most[0], 1)
        self.assertEqual(len(slots), 4)

    def test_the_slots_at_once_without_caps(self):
        sch = srv.GenerationScheduler(max_queue=8, queue_timeout=30, capacity=3)
        started, release = threading.Barrier(4), threading.Event()

        def one():
            with sch.admit():
                started.wait(5)
                release.wait(5)

        th = [threading.Thread(target=one) for _ in range(3)]
        for t in th:
            t.start()
        started.wait(5)   # all three run together
        release.set()
        for t in th:
            t.join(10)


class ServeWiringTest(unittest.TestCase):
    def test_serve_sets_it_from_the_engine_it_started(self):
        """serve() binds the port first (APIServer with no engine), then starts the engine: the concurrency must
        come from the engine's CAPS there and at each restart, not at the server's creation"""
        import inspect
        src = inspect.getsource(srv.serve)
        i = src.index("server.engine = runtime")
        self.assertIn("set_concurrency(engine_concurrency(runtime", src[i:i + 200])
        self.assertIn("set_concurrency(engine_concurrency(server.engine", inspect.getsource(srv.keep_engine))


if __name__ == "__main__":
    unittest.main()
