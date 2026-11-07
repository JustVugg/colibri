"""Qualified V4.1 function names survive the OpenAI response/history boundary."""
import json
import unittest
from unittest.mock import patch

import openai_server as srv
import v41_dsml
from test_openai_server import ScriptedEngine, _spawn_test_server, _post_chat


class QualifiedToolNamesTest(unittest.TestCase):
    # Literal checkpoint syntax: two separate namespaces expose the same bare name.
    REPLY = ('\n\n<｜DSML｜ calls>\n'
             '<｜DSML｜ invoke name="browser::search">\n'
             '<｜DSML｜ parameter name="q" string="true">weather</｜DSML｜ parameter>\n'
             '</｜DSML｜ invoke>\n'
             '<｜DSML｜ invoke name="files::search">\n'
             '<｜DSML｜ parameter name="q" string="true">notes</｜DSML｜ parameter>\n'
             '</｜DSML｜ invoke>\n'
             '</｜DSML｜ calls>')
    TOOLS = [{"type": "function", "function": {
        "name": name, "parameters": {"type": "object", "properties": {
            "q": {"type": "string"}}}}} for name in ("browser::search", "files::search")]

    def test_reference_decoder_retains_namespace_identity(self):
        # The vendored strict decoder is an independent boundary: it returns namespace
        # separately from the bare name. The gateway must preserve both on its wire.
        _, _, decoded = v41_dsml._parse_tool_calls_strict(
            len(v41_dsml.TOOL_CALLS_BLOCK_START), self.REPLY)
        self.assertEqual([(c['namespace'], c['name']) for c in decoded],
                         [('browser', 'search'), ('files', 'search')])

    def test_qualified_names_survive_both_http_response_shapes_and_history(self):
        with patch.object(srv, "ARCH", "deepseek_v41"):
            for stream in (False, True):
                with self.subTest(stream=stream):
                    engine = ScriptedEngine(chunks=(self.REPLY,))
                    base = _spawn_test_server(self, engine)
                    with _post_chat(base, {"model": "test-model", "messages": [
                        {"role": "user", "content": "Search both places"}],
                        "tools": self.TOOLS, "stream": stream}) as response:
                        if stream:
                            calls = []
                            for line in response:
                                if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                                    event = json.loads(line[6:])
                                    for choice in event.get('choices', []):
                                        calls.extend(choice.get('delta', {}).get('tool_calls', []))
                        else:
                            calls = json.load(response)['choices'][0]['message']['tool_calls']
                    self.assertEqual([c['function']['name'] for c in calls],
                                     ['browser::search', 'files::search'])
                    self.assertEqual([json.loads(c['function']['arguments']) for c in calls],
                                     [{'q': 'weather'}, {'q': 'notes'}])
                    prompt = srv.render_chat_dsv41([
                        {"role": "assistant", "content": "", "tool_calls": calls},
                        {"role": "tool", "tool_call_id": calls[0]['id'], "content": "sunny"},
                        {"role": "tool", "tool_call_id": calls[1]['id'], "content": "found"}],
                        tools=self.TOOLS)
                    self.assertIn('name="browser::search"', prompt)
                    self.assertIn('name="files::search"', prompt)

    def test_unqualified_name_is_unchanged(self):
        _, calls = v41_dsml.parse_completion_text(self.REPLY.replace('browser::', '').replace('files::', ''))
        self.assertEqual([c['function']['name'] for c in calls], ['search', 'search'])


if __name__ == '__main__':
    unittest.main()
