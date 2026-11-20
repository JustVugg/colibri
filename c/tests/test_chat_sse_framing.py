"""SSE data fields form one event; a space after the colon is optional."""
import contextlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest import mock
from tests.test_coli_chat_documents import coli


class QuietSpinner:
    def __init__(self, *_args): pass
    def start(self): pass
    def stop(self): pass


class ChatSSEFrames(unittest.TestCase):
    def test_answer_survives_valid_event_framings(self):
        payload = '{"choices":\n[{"delta":{"content":"ANSWER"},"finish_reason":"stop"}]}'
        for wire in [
            ('data:' + payload.replace('\n', '') + '\n\ndata:[DONE]\n\n').encode(),
            ('data: ' + payload.replace('\n', '\ndata: ') + '\n\ndata: [DONE]\n\n').encode(),
            (': heartbeat\r\nevent: message\r\ndata: ' + payload.replace('\n', '\r\ndata: ') + '\r\n\r\ndata: [DONE]\r\n\r\n').encode(),
        ]:
            with self.subTest(wire=wire):
                prompts = iter(['first', 'second', ':q'])
                seen = []
                def response(req):
                    seen.append(json.loads(req.data)); return io.BytesIO(wire)
                output = io.StringIO()
                args = SimpleNamespace(api_key=None, ngen=8, think=None, stats='off', file=None)
                with mock.patch.object(coli, 'TTY', False), \
                     mock.patch.object(coli, 'Spinner', QuietSpinner), \
                     mock.patch.object(coli, 'install_chat_completer'), \
                     mock.patch.object(coli, 'server_arch', return_value='glm'), \
                     mock.patch('builtins.input', side_effect=lambda: next(prompts)), \
                     mock.patch('urllib.request.urlopen', side_effect=response), \
                     contextlib.redirect_stdout(output):
                    coli.chat_attached(args, 'http://localhost:8000', 'fixture')
                self.assertIn('ANSWER', output.getvalue())
                self.assertEqual(seen[1]['messages'][1], {'role': 'assistant', 'content': 'ANSWER'})


if __name__ == '__main__': unittest.main()
