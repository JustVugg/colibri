"""Malformed JSON constants cannot dispatch tools or leak into MCP replies."""
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))
import mcp_server


def reject_constant(value):
    raise ValueError('non-JSON constant: ' + value)


class JsonConstantTest(unittest.TestCase):
    def test_malformed_json_cannot_dispatch_install(self):
        for constant in ('NaN', 'Infinity', '-Infinity'):
            with self.subTest(constant=constant):
                server = mcp_server.Server(io.StringIO())
                effects = []
                with mock.patch.dict(mcp_server.HANDLERS, {
                        'install': lambda args, notify: effects.append(args) or {'fixture': True}}):
                    server.handle_line('{"jsonrpc":"2.0","id":1,"method":"tools/call",'
                                       '"params":{"name":"install","_meta":{"progressToken":'
                                       + constant + '}}}')
                    for worker in server.threads:
                        worker.join(timeout=2)
                        self.assertFalse(worker.is_alive())
                self.assertEqual(effects, [], 'invalid JSON dispatched a mutating tool')
                reply = json.loads(server.out.getvalue(), parse_constant=reject_constant)
                self.assertIsNone(reply['id'])
                self.assertEqual(reply['error']['code'], -32700)

    def test_non_json_id_gets_parse_error_then_next_request_works(self):
        for constant in ('NaN', 'Infinity', '-Infinity'):
            with self.subTest(constant=constant):
                wire = ('{"jsonrpc":"2.0","id":' + constant + ',"method":"ping"}\n'
                        '{"jsonrpc":"2.0","id":2,"method":"ping"}\n').encode()
                child = subprocess.run([sys.executable, str(C_DIR / 'coli'), 'mcp'],
                                       input=wire, capture_output=True, timeout=5)
                self.assertEqual(child.returncode, 0, child.stderr.decode('utf-8', 'replace'))
                replies = [json.loads(line, parse_constant=reject_constant)
                           for line in child.stdout.splitlines()]
                self.assertEqual(len(replies), 2)
                self.assertIsNone(replies[0]['id'])
                self.assertEqual(replies[0]['error']['code'], -32700)
                self.assertEqual(replies[1]['result'], {})
                self.assertEqual(replies[1]['id'], 2)


if __name__ == '__main__':
    unittest.main()
