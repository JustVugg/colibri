"""The MCP server: JSON-RPC 2.0 over stdio, as an AI coding assistant drives it.

Round trips through the Server class (initialize, tools/list, tools/call and
the error codes), progress notifications, and the same exchange through a real
child process on stdin/stdout, which is how a client runs it. Tools that would
build or download are exercised with their effects mocked; nothing here
fetches anything.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))
import mcp_server  # noqa: E402
import setup_flow  # noqa: E402


class Exchange:
    """A Server writing into a buffer, with helpers to read the replies."""

    def __init__(self):
        self.out = io.StringIO()
        self.server = mcp_server.Server(out=self.out)

    def send(self, message):
        self.server.handle_line(json.dumps(message))

    def wait_threads(self):
        for thread in self.server.threads:
            thread.join(timeout=30)

    def messages(self):
        return [json.loads(line) for line in self.out.getvalue().splitlines() if line.strip()]

    def reply(self, msg_id):
        self.wait_threads()
        for message in self.messages():
            if message.get("id") == msg_id and ("result" in message or "error" in message):
                return message
        raise AssertionError(f"no reply to {msg_id}: {self.out.getvalue()}")


def request(msg_id, method, params=None):
    message = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


class Protocol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {"COLI_SETUP_HOME": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.x = Exchange()

    def test_initialize_negotiates_the_version(self):
        self.x.send(request(1, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                              "clientInfo": {"name": "t", "version": "1"}}))
        result = self.x.reply(1)["result"]
        self.assertEqual(result["protocolVersion"], "2025-03-26")
        self.assertEqual(result["serverInfo"]["name"], "colibri")
        self.assertIn("tools", result["capabilities"])
        self.assertIn("install", result["instructions"])
        self.x.send(request(2, "initialize", {"protocolVersion": "1999-01-01"}))
        self.assertEqual(self.x.reply(2)["result"]["protocolVersion"], mcp_server.SUPPORTED_PROTOCOLS[0])

    def test_notifications_get_no_reply(self):
        self.x.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.x.send({"jsonrpc": "2.0", "method": "notifications/cancelled",
                     "params": {"requestId": 3}})
        self.assertEqual(self.x.messages(), [])

    def test_ping_and_unknown_method(self):
        self.x.send(request(1, "ping"))
        self.assertEqual(self.x.reply(1)["result"], {})
        self.x.send(request(2, "sampling/whatever"))
        self.assertEqual(self.x.reply(2)["error"]["code"], mcp_server.METHOD_NOT_FOUND)

    def test_parse_error_and_invalid_request(self):
        self.x.server.handle_line("{not json")
        self.assertEqual(self.x.messages()[-1]["error"]["code"], mcp_server.PARSE_ERROR)
        self.x.send({"id": 5, "method": "ping"})                       # no jsonrpc member
        self.assertEqual(self.x.messages()[-1]["error"]["code"], mcp_server.INVALID_REQUEST)

    def test_batch(self):
        self.x.server.handle_line(json.dumps([request(1, "ping"), request(2, "ping")]))
        self.assertEqual({m["id"] for m in self.x.messages()}, {1, 2})

    def test_tools_list(self):
        self.x.send(request(1, "tools/list"))
        tools = self.x.reply(1)["result"]["tools"]
        names = [t["name"] for t in tools]
        self.assertEqual(names, ["detect_hardware", "recommend_models", "install", "start", "stop",
                                 "status", "logs"])
        for tool in tools:
            self.assertTrue(tool["description"])
            self.assertEqual(tool["inputSchema"]["type"], "object")
            json.dumps(tool)

    def test_status_before_setup(self):
        self.x.send(request(1, "tools/call", {"name": "status", "arguments": {}}))
        result = self.x.reply(1)["result"]
        self.assertFalse(result["isError"])
        self.assertEqual(result["structuredContent"]["configured"], False)
        self.assertEqual(json.loads(result["content"][0]["text"])["configured"], False)

    def test_tool_errors_are_results_not_protocol_errors(self):
        self.x.send(request(1, "tools/call", {"name": "start", "arguments": {}}))
        result = self.x.reply(1)["result"]
        self.assertTrue(result["isError"])
        self.assertIn("install", result["content"][0]["text"])

    def test_argument_validation(self):
        cases = [("nope", {}), ("logs", {"lines": "ten"}), ("logs", {"which": "kernel"}),
                 ("logs", {"lines": 0}), ("status", {"extra": 1}), ("install", {"port": True})]
        for number, (name, args) in enumerate(cases, 1):
            with self.subTest(name=name, args=args):
                self.x.send(request(number, "tools/call", {"name": name, "arguments": args}))
                self.assertEqual(self.x.reply(number)["error"]["code"], mcp_server.INVALID_PARAMS)

    def test_recommend_models(self):
        fake_hw = {"os": {"machine": "x86_64"}, "memory": {"total": 27 * 10**9}}
        with mock.patch.object(mcp_server.setup_hw, "detect", return_value=fake_hw), \
             mock.patch.object(mcp_server.setup_hw, "disk_free", return_value=800 * 10**9), \
             mock.patch.object(setup_flow.setup_hw, "disk_free", return_value=800 * 10**9):
            self.x.send(request(1, "tools/call", {"name": "recommend_models", "arguments": {}}))
            data = self.x.reply(1)["result"]["structuredContent"]
        self.assertEqual(data["recommended"], "qwen36-35b")
        self.assertTrue(all(m["fits"] for m in data["models"]))
        self.assertIn("dense part", data["fits_means"])

    def test_install_starts_a_background_job(self):
        class FakeProcess:
            pid = 4242

            def poll(self):
                return None

        with mock.patch.object(mcp_server.setup_flow, "spawn_detached",
                               return_value=FakeProcess()) as spawn:
            self.x.send(request(1, "tools/call", {"name": "install", "arguments": {
                "model": "qwen3-coder-30b", "backend": "cpu", "port": 8100}}))
            result = self.x.reply(1)["result"]
        self.assertFalse(result["isError"], result)
        command = spawn.call_args[0][0]
        self.assertEqual(command[1:4], [str(C_DIR / "coli"), "setup", "--yes"])
        for piece in ("--model", "qwen3-coder-30b", "--backend", "cpu", "--port", "8100",
                      "--background", "--no-browser"):
            self.assertIn(piece, command)
        self.assertEqual(result["structuredContent"]["job"]["pid"], 4242)
        self.assertEqual(setup_flow.read_state()["pid"], 4242)

    def test_install_refuses_unknown_models_and_a_second_job(self):
        self.x.send(request(1, "tools/call", {"name": "install", "arguments": {"model": "gpt-9"}}))
        self.assertTrue(self.x.reply(1)["result"]["isError"])
        setup_flow._write_json(setup_flow.state_path(), {"phase": "download", "pid": os.getpid()})
        self.x.send(request(2, "tools/call", {"name": "install", "arguments": {}}))
        result = self.x.reply(2)["result"]
        self.assertTrue(result["isError"])
        self.assertIn("already running", result["content"][0]["text"])

    def test_progress_notifications_increase(self):
        def handler(args, notify):
            notify(None, None, "build")
            notify(500, 1000, "half")
            notify(400, 1000, "a smaller byte count must not go backwards")
            notify(None, None, "ready")
            return {"ok": True}

        with mock.patch.dict(mcp_server.HANDLERS, {"status": handler}):
            self.x.send(request(9, "tools/call", {"name": "status", "arguments": {},
                                                  "_meta": {"progressToken": "tok"}}))
            self.x.reply(9)
        progress = [m["params"] for m in self.x.messages()
                    if m.get("method") == "notifications/progress"]
        values = [p["progress"] for p in progress]
        self.assertEqual(len(values), 4)
        self.assertEqual(values, sorted(set(values)))
        self.assertTrue(all(p["progressToken"] == "tok" for p in progress))
        self.assertEqual(progress[1]["total"], 1000)

    def test_no_token_no_notifications(self):
        def handler(args, notify):
            notify(1, 2, "x")
            return {}

        with mock.patch.dict(mcp_server.HANDLERS, {"status": handler}):
            self.x.send(request(1, "tools/call", {"name": "status", "arguments": {}}))
            self.x.reply(1)
        self.assertFalse(any(m.get("method") for m in self.x.messages()))

    def test_logs(self):
        log = setup_flow.log_path("serve")
        os.makedirs(os.path.dirname(log))
        Path(log).write_text("\n".join(f"line {i}" for i in range(50)))
        self.x.send(request(1, "tools/call", {"name": "logs", "arguments": {"lines": 3}}))
        data = self.x.reply(1)["result"]["structuredContent"]
        self.assertEqual(data["text"], "line 47\nline 48\nline 49")

    def test_a_crashing_tool_does_not_take_the_server_down(self):
        with mock.patch.dict(mcp_server.HANDLERS, {"status": lambda a, n: 1 / 0}), \
             mock.patch.object(mcp_server, "_log") as logged:
            self.x.send(request(1, "tools/call", {"name": "status", "arguments": {}}))
            result = self.x.reply(1)["result"]
        self.assertTrue(result["isError"])
        self.assertIn("ZeroDivisionError", result["content"][0]["text"])
        self.assertIn("ZeroDivisionError", logged.call_args[0][0])     # the traceback goes to stderr
        self.x.send(request(2, "ping"))
        self.assertEqual(self.x.reply(2)["result"], {})


class StdioRoundTrip(unittest.TestCase):
    """The way a client runs it: a child process, one JSON message per line."""

    def test_over_stdin_and_stdout(self):
        with tempfile.TemporaryDirectory() as home:
            env = dict(os.environ, COLI_SETUP_HOME=home)
            process = subprocess.Popen([sys.executable, str(C_DIR / "coli"), "mcp"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, env=env)
            lines = [request(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                               "clientInfo": {"name": "test", "version": "0"}}),
                     {"jsonrpc": "2.0", "method": "notifications/initialized"},
                     request(2, "tools/list"),
                     request(3, "tools/call", {"name": "status", "arguments": {}})]
            out, err = process.communicate("\n".join(json.dumps(m) for m in lines) + "\n",
                                           timeout=60)
        replies = [json.loads(line) for line in out.splitlines()]
        self.assertEqual([r["id"] for r in replies], [1, 2, 3])     # stdout carries JSON only
        self.assertEqual(replies[0]["result"]["serverInfo"]["name"], "colibri")
        self.assertEqual(len(replies[1]["result"]["tools"]), 7)
        self.assertFalse(replies[2]["result"]["isError"])
        self.assertIn("colibri MCP server", err)                     # diagnostics on stderr


if __name__ == "__main__":
    unittest.main()
