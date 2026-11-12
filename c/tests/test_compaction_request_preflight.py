"""Compaction must validate a request before spending inference on its summary."""
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openai_server as gateway
from test_openai_server import ScriptedEngine, _post_chat


class CompactionRequestPreflightTest(unittest.TestCase):
    def request(self, extra, arch='glm'):
        engine = ScriptedEngine()
        server = gateway.APIServer(('127.0.0.1', 0), engine, 'test-model', 'secret', 16)
        server.context_limit = 1000
        server.tokens_per_char = 0.25
        worker = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
        worker.start()
        body = {'model': 'test-model', 'context_compaction': 'auto', 'messages': [
            {'role': 'user', 'content': 'old ' * 2000},
            {'role': 'assistant', 'content': 'answer ' * 80},
            {'role': 'user', 'content': 'Latest question'}]}
        body.update(extra)
        try:
            with patch.object(gateway, 'ARCH', arch):
                try:
                    response = _post_chat(f'http://127.0.0.1:{server.server_port}', body)
                except HTTPError as response:
                    with response:
                        result = json.load(response)
                        return response.code, result, engine.calls
                with response:
                    result = json.load(response)
                    return response.status, result, engine.calls
        finally:
            server.scheduler.close()
            server.shutdown()
            server.server_close()
            worker.join(1)

    def test_invalid_request_is_rejected_without_summary_inference(self):
        cases = [({'stream': 'yes'}, 'stream', 'glm'),
                 ({'logprobs': 'yes'}, 'logprobs', 'glm'),
                 ({'echo': True}, 'echo', 'glm'),
                 ({'stream': True, 'logprobs': True}, 'logprobs', 'glm'),
                 ({'stream': True, 'stream_options': {'include_usage': 'yes'}},
                  'stream_options.include_usage', 'glm'),
                 ({'response_format': {'type': 'json_object'}}, 'response_format', 'qwen36')]
        for options, param, arch in cases:
            with self.subTest(options=options, arch=arch):
                status, body, calls = self.request(options, arch)
                self.assertEqual(status, 400)
                self.assertEqual(body['error']['param'], param)
                self.assertEqual(calls, [], 'an invalid request spent inference on its summary')

    def test_valid_compaction_request_still_summarizes_and_answers(self):
        status, body, calls = self.request({})
        self.assertEqual(status, 200)
        self.assertEqual(body['choices'][0]['message']['content'], 'ok')
        self.assertTrue(any('Summarize our conversation so far' in call[0] for call in calls))
        self.assertGreaterEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
