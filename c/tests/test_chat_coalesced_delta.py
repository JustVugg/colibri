"""Reasoning visibility must never discard answer bytes in the same SSE delta."""
import contextlib
import io
import json
import os
from types import SimpleNamespace
import unittest
from unittest import mock
from tests.test_coli_chat_documents import coli


class QuietSpinner:
    def __init__(self, *_args): pass
    def start(self): pass
    def stop(self): pass


class CoalescedDelta(unittest.TestCase):
    def test_answer_is_kept_with_visible_or_hidden_reasoning(self):
        for show in ('0', '1'):
            with self.subTest(show=show):
                requests = []
                event = {'choices': [{'delta': {'reasoning_content': 'SCRATCHPAD',
                                                 'content': 'ANSWER'}, 'finish_reason': 'stop'}]}
                wire = ('data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n').encode()
                def response(request):
                    requests.append(json.loads(request.data))
                    return io.BytesIO(wire)
                prompts = iter(['first', 'second', ':q'])
                output = io.StringIO()
                args = SimpleNamespace(api_key=None, ngen=8, think=None, stats='off', file=None)
                with mock.patch.object(coli, 'TTY', False), \
                     mock.patch.object(coli, 'Spinner', QuietSpinner), \
                     mock.patch.object(coli, 'install_chat_completer'), \
                     mock.patch.object(coli, 'server_arch', return_value='glm'), \
                     mock.patch('builtins.input', side_effect=lambda: next(prompts)), \
                     mock.patch('urllib.request.urlopen', side_effect=response), \
                     mock.patch.dict(os.environ, {'COLI_SHOW_THINK': show}), \
                     contextlib.redirect_stdout(output):
                    coli.chat_attached(args, 'http://localhost:8000', 'fixture')
                self.assertIn('ANSWER', output.getvalue())
                self.assertEqual('SCRATCHPAD' in output.getvalue(), show == '1')
                self.assertEqual(requests[1]['messages'][1], {'role': 'assistant', 'content': 'ANSWER'})
                self.assertNotIn('SCRATCHPAD', json.dumps(requests[1]['messages']))


if __name__ == '__main__': unittest.main()
