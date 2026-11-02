"""MCP responses stay UTF-8 encodable even when a request contains a surrogate escape."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

C_DIR = Path(__file__).resolve().parent.parent


class StdioEncoding(unittest.TestCase):
    def exchange(self, first):
        messages = [first, {"jsonrpc": "2.0", "id": 2, "method": "ping"}]
        wire = ''.join(json.dumps(message) + '\n' for message in messages).encode('utf-8')
        child = subprocess.run([sys.executable, str(C_DIR / 'coli'), 'mcp'], input=wire,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        self.assertEqual(child.returncode, 0, child.stderr.decode('utf-8', errors='replace'))
        replies = [json.loads(line) for line in child.stdout.decode('utf-8').splitlines()]
        self.assertEqual({reply['id'] for reply in replies}, {1, 2}, child.stderr)
        self.assertEqual(next(reply for reply in replies if reply['id'] == 2)['result'], {})
        return next(reply for reply in replies if reply['id'] == 1)

    def test_surrogate_unknown_method_returns_error_and_preserves_session(self):
        reply = self.exchange({"jsonrpc": "2.0", "id": 1, "method": "unknown-\ud800"})
        self.assertEqual(reply['error']['code'], -32601)

    def test_surrogate_tool_name_returns_error_and_preserves_session(self):
        reply = self.exchange({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                               "params": {"name": "unknown-\ud800", "arguments": {}}})
        self.assertEqual(reply['error']['code'], -32602)

    def test_valid_unicode_error_text_is_preserved(self):
        method = 'unknown-☃-😀'
        reply = self.exchange({"jsonrpc": "2.0", "id": 1, "method": method})
        self.assertEqual(reply['error']['code'], -32601)
        self.assertEqual(reply['error']['message'], 'method not found: ' + method)


if __name__ == '__main__':
    unittest.main()
