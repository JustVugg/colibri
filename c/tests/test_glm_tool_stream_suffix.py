"""Streamed GLM tool replies preserve the answer outside complete tool-call boxes."""
import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openai_server as gateway
from test_openai_server import ScriptedEngine, _spawn_test_server, _post_chat


class GlmToolSuffixTest(unittest.TestCase):
    TOOLS = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {
        'type': 'object', 'properties': {'q': {'type': 'string'}}}}}]

    def response(self, chunks, stream):
        engine = ScriptedEngine(chunks=chunks)
        base = _spawn_test_server(self, engine)
        with _post_chat(base, {'model': 'test-model', 'stream': stream,
                              'messages': [{'role': 'user', 'content': 'Look it up'}],
                              'tools': self.TOOLS}) as response:
            if not stream:
                message = json.load(response)['choices'][0]['message']
                return message['content'], message['tool_calls']
            text, calls = [], []
            for line in response:
                if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                    continue
                event = json.loads(line[6:])
                for choice in event['choices']:
                    delta = choice['delta']
                    text.append(delta.get('content', ''))
                    calls.extend(delta.get('tool_calls', []))
            return ''.join(text), calls

    def test_prose_outside_complete_calls_survives_streaming(self):
        cases = [
            (('Before ', '<tool_call>lookup<arg_key>q</arg_key>',
              '<arg_value>one</arg_value></tool_call>', ' Between ',
              '<tool_call>lookup<arg_key>q</arg_key><arg_value>two</arg_value>',
              '</tool_call>', ' After'), ['Before', 'Between', 'After']),
            (('  Ωmega ', '<tool', '_call>lookup<arg_key>q</arg_key>',
              '<arg_value>one</arg_value></tool_call>', ' 終  '), ['Ωmega', '終']),
            (('<tool_call>lookup<arg_key>q</arg_key><arg_value>one</arg_value>',
              '</tool_call>', ' Answer'), ['Answer']),
        ]
        for arch in ('glm', 'glm53'):
            for chunks, expected in cases:
                with self.subTest(arch=arch, expected=expected), patch.object(gateway, 'ARCH', arch):
                    # GLM-5.3 starts inside thinking; close it before the answer.
                    if arch == 'glm53':
                        chunks = ('</think>',) + chunks
                    plain, plain_calls = self.response(chunks, False)
                    streamed, stream_calls = self.response(chunks, True)
                    self.assertEqual(plain.split(), expected)
                    self.assertEqual(streamed.split(), expected)
                    self.assertEqual([c['function'] for c in stream_calls],
                                     [c['function'] for c in plain_calls])
                    self.assertNotIn('<tool_call>', streamed)


if __name__ == '__main__':
    unittest.main()
