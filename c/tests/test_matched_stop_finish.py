"""A matched API stop sequence is a stop, even in the last allowed engine token."""
import json
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_openai_server import ScriptedEngine, _spawn_test_server, _post_json


class MatchedStopFinishTest(unittest.TestCase):
    def response(self, chat, stream, stop, tools=False):
        # The final emitted token both exhausts the engine budget and contains the
        # API stop sequence. Engine length_limited describes the raw generation;
        # the gateway strips STOP and must report its own observed stopping reason.
        engine = ScriptedEngine(chunks=('AnswerSTOP',), length_limited=True)
        base = _spawn_test_server(self, engine, max_tokens=1)
        request = {'model': 'test-model', 'max_tokens': 1, 'stream': stream}
        if chat:
            request['messages'] = [{'role': 'user', 'content': 'hi'}]
            if tools:
                request['tools'] = [{'type': 'function', 'function': {'name': 'lookup'}}]
        else:
            request['prompt'] = 'hi'
        if stop:
            request['stop'] = 'STOP'
        route = '/v1/chat/completions' if chat else '/v1/completions'
        with _post_json(base, route, request) as response:
            if not stream:
                choice = json.load(response)['choices'][0]
                return choice['finish_reason'], (choice['message']['content'] if chat else choice['text'])
            text, finish = [], None
            for line in response:
                if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                    for choice in json.loads(line[6:])['choices']:
                        text.append(choice['delta'].get('content', '') if chat else choice['text'])
                        finish = choice['finish_reason'] or finish
            return finish, ''.join(text)

    def test_matched_stop_takes_precedence_on_every_openai_response_shape(self):
        for chat in (False, True):
            for stream in (False, True):
                for tools in ((False, True) if chat else (False,)):
                    with self.subTest(chat=chat, stream=stream, tools=tools):
                        self.assertEqual(self.response(chat, stream, True, tools), ('stop', 'Answer'))

    def test_budget_exhaustion_without_matched_stop_remains_length(self):
        for chat in (False, True):
            for stream in (False, True):
                for tools in ((False, True) if chat else (False,)):
                    with self.subTest(chat=chat, stream=stream, tools=tools):
                        self.assertEqual(self.response(chat, stream, False, tools), ('length', 'AnswerSTOP'))


if __name__ == '__main__':
    unittest.main()
