"""A started SSE reply can fail without changing its HTTP200 status."""
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


class StreamStatus(unittest.TestCase):
    def run_chat(self, wire):
        prompts = iter(['question', ':q'])
        output = io.StringIO()
        args = SimpleNamespace(api_key=None, ngen=8, think=None, stats='off', file=None)
        with mock.patch.object(coli, 'TTY', False), \
             mock.patch.object(coli, 'Spinner', QuietSpinner), \
             mock.patch.object(coli, 'install_chat_completer'), \
             mock.patch.object(coli, 'server_arch', return_value='glm'), \
             mock.patch('builtins.input', side_effect=lambda: next(prompts)), \
             mock.patch('urllib.request.urlopen', return_value=io.BytesIO(wire)), \
             contextlib.redirect_stdout(output):
            coli.chat_attached(args, 'http://localhost:8000', 'fixture')
        return output.getvalue()

    def test_stream_error_explanation_is_visible(self):
        event = {'error': {'message': 'synthetic engine failure'}}
        output = self.run_chat(('data: ' + json.dumps(event) + '\n\n').encode())
        self.assertIn('synthetic engine failure', output)
        self.assertNotIn('server unreachable', output)

    def test_early_eof_reports_incomplete_answer(self):
        event = {'choices': [{'delta': {'content': 'Partial answer'}}]}
        wire = ('data: ' + json.dumps(event) + '\n\n').encode()
        output = self.run_chat(wire)
        self.assertIn('Partial answer', output)
        self.assertIn('stream closed before the response finished', output)
        completed = self.run_chat(wire + b'data: [DONE]\n\n')
        self.assertNotIn('stream closed before the response finished', completed)


if __name__ == '__main__': unittest.main()
