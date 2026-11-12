"""Invalid compaction KV targets are request errors, never queued inference work."""
import json
import sys
from pathlib import Path
import threading
import unittest
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openai_server import APIServer, GenerationScheduler
from test_openai_server import ScriptedEngine, _post_chat


class CompactionSlotAdmissionTest(unittest.TestCase):
    def test_invalid_slot_is_rejected_before_scheduling(self):
        for slot in (-1, 1, [], {}, False, 0.0):
            with self.subTest(slot=slot):
                engine = ScriptedEngine()
                server = APIServer(('127.0.0.1', 0), engine, 'test-model', 'secret',
                                   max_tokens=16, kv_slots=1, queue_timeout=0.02)
                server.context_limit = 1000
                server.tokens_per_char = 0.25
                worker = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
                worker.start()
                try:
                    with self.assertRaises(HTTPError) as caught:
                        _post_chat(f'http://127.0.0.1:{server.server_port}', {
                            'model': 'test-model', 'context_compaction': 'auto',
                            'cache_slot': slot, 'messages': [
                                {'role': 'user', 'content': 'old ' * 2000},
                                {'role': 'assistant', 'content': 'answer ' * 80},
                                {'role': 'user', 'content': 'Latest question'}]})
                    with caught.exception as response:
                        body = json.load(response)
                        self.assertEqual(response.code, 400)
                        self.assertEqual(body['error']['param'], 'cache_slot')
                    self.assertEqual(engine.calls, [])
                    snapshot = server.scheduler.snapshot()
                    self.assertEqual(snapshot['queued'], 0)
                    self.assertEqual(snapshot['admitted'], 0)
                    self.assertEqual(snapshot['timed_out'], 0)
                finally:
                    server.scheduler.close()
                    server.shutdown()
                    server.server_close()
                    worker.join(1)

    def test_scheduler_still_accepts_every_real_slot_and_any_slot(self):
        scheduler = GenerationScheduler(capacity=2)
        for requested in (None, 0, 1):
            with scheduler.admit(slot=requested) as (_, actual):
                self.assertIn(actual, (0, 1))
                if requested is not None:
                    self.assertEqual(actual, requested)
        self.assertEqual(scheduler.snapshot()['completed'], 3)


if __name__ == '__main__':
    unittest.main()
