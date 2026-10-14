"""Malformed explicit arguments must not execute mutating tool defaults."""
import io
import json
import unittest
from unittest.mock import patch

import mcp_server


class ToolArgumentObjectTest(unittest.TestCase):
    def exchange(self, params):
        server = mcp_server.Server(out=io.StringIO())
        effects = []

        def install(args, notify):
            effects.append(args)
            return {"installed": "default"}

        with patch.dict(mcp_server.HANDLERS, {"install": install}):
            server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params})
            for thread in server.threads:
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
        return json.loads(server.out.getvalue()), effects

    def test_falsey_nonobjects_are_invalid_without_install_effects(self):
        for value in ([], False, 0, "", None):
            with self.subTest(value=value):
                reply, effects = self.exchange({"name": "install", "arguments": value})
                self.assertEqual(effects, [], "malformed arguments executed install defaults")
                self.assertEqual(reply["error"]["code"], mcp_server.INVALID_PARAMS)

    def test_omitted_and_empty_object_arguments_still_execute(self):
        for params in ({"name": "install"}, {"name": "install", "arguments": {}}):
            with self.subTest(params=params):
                reply, effects = self.exchange(params)
                self.assertEqual(effects, [{}])
                self.assertFalse(reply["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
