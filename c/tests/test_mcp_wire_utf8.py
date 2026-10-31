"""A malformed UTF-8 frame must not terminate the MCP stdio session."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

C_DIR = Path(__file__).resolve().parents[1]


class WireEncodingTest(unittest.TestCase):
    def test_invalid_utf8_is_one_parse_error_then_next_frame_works(self):
        ping = json.dumps({'jsonrpc': '2.0', 'id': 2, 'method': 'ping'}).encode() + b'\n'
        for malformed in (b'\xff\n', b'{"jsonrpc":"2.0","id":1,"method":"bad-\xff"}\n',
                          b'\xc3\n'):
            with self.subTest(frame=malformed):
                child = subprocess.run([sys.executable, str(C_DIR / 'coli'), 'mcp'],
                                       input=malformed + ping, capture_output=True, timeout=5)
                self.assertEqual(child.returncode, 0, child.stderr.decode('utf-8', 'replace'))
                replies = [json.loads(line) for line in child.stdout.splitlines()]
                self.assertEqual(len(replies), 2)
                self.assertEqual(replies[0]['error']['code'], -32700)
                self.assertIsNone(replies[0]['id'])
                self.assertEqual(replies[1], {'jsonrpc': '2.0', 'id': 2, 'result': {}})

    def test_utf8_frame_preserves_non_ascii_text(self):
        method = 'unknown-☃-😀'
        wire = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': method},
                          ensure_ascii=False).encode('utf-8') + b'\n'
        child = subprocess.run([sys.executable, str(C_DIR / 'coli'), 'mcp'],
                               input=wire, capture_output=True, timeout=5)
        self.assertEqual(child.returncode, 0, child.stderr.decode('utf-8', 'replace'))
        reply = json.loads(child.stdout)
        self.assertEqual(reply['error']['message'], 'method not found: ' + method)


if __name__ == '__main__':
    unittest.main()
