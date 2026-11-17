"""Parallel search uses the same result contract and keeps provider selection local."""
import io
import json
import sys
import unittest
from email.message import Message
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ui"))
import web   # noqa: E402


def response(status, content_type="application/json", body=b"", session=None):
    r = mock.MagicMock()
    r.__enter__.return_value = r
    r.status = status
    r.headers = Message()
    r.headers["Content-Type"] = content_type
    r.getheader.return_value = session
    stream = io.BytesIO(body)
    r.read.side_effect = stream.read
    r.readline.side_effect = stream.readline
    return r


def reply(id, result):
    return json.dumps({"jsonrpc": "2.0", "id": id, "result": result}).encode()


class ParallelSearchTest(unittest.TestCase):
    def test_session_and_native_results(self):
        rows = [
            {"url": "javascript:alert(1)"},
            {"url": "http://127.0.0.1/"},
            {"url": "https://example.org/moon", "title": "Moon &amp; eclipses",
             "publish_date": "2024-01-02T00:00:00Z", "excerpts": ["One.", "Two."]},
            {"url": "https://example.org/moon", "title": "duplicate"},
            {"url": "https://example.org/another", "excerpts": ["x" * 400]},
        ]
        event = b": comment\r\n\r\ndata: " + reply(99, {}) + b"\r\n\r\n"
        event += b"data: " + reply(2, {"structuredContent": {"results": rows}}) + b"\r\n\r\n"
        responses = [
            response(200, body=reply(1, {"protocolVersion": web.MCP_VERSION}), session="test-session"),
            response(202),
            response(200, "text/event-stream", event),
            response(200),
        ]
        requests = []

        def send(method, path, body=None, headers=None):
            requests.append((method, path, json.loads(body) if body else None, dict(headers)))

        conn = mock.Mock()
        conn.request.side_effect = send
        conn.getresponse.side_effect = responses

        def resolve(host, port, **kwargs):
            address = "127.0.0.1" if host == "127.0.0.1" else "93.184.216.34"
            return [(None, None, None, None, (address, port))]

        with mock.patch.object(web, "PARALLEL_SEARCH", True), \
             mock.patch.object(web, "SEARX", "https://unused.example"), \
             mock.patch.object(web.socket, "getaddrinfo", side_effect=resolve), \
             mock.patch.object(web.http.client, "HTTPSConnection", return_value=conn) as https, \
             mock.patch.object(web.urllib.request, "urlopen") as other:
            self.assertEqual(web.status(), {"search": True, "engine": "parallel"})
            self.assertEqual(requests, [])
            data = web.search("  lunar eclipse  ", 2)
            other.assert_not_called()

        self.assertEqual([r[0] for r in requests], ["POST", "POST", "POST", "DELETE"])
        self.assertEqual([r[2]["method"] for r in requests[:3]],
                         ["initialize", "notifications/initialized", "tools/call"])
        self.assertNotIn("Mcp-Session-Id", requests[0][3])
        self.assertNotIn("id", requests[1][2])
        for method, path, body, headers in requests[1:]:
            self.assertEqual(path, "/mcp")
            self.assertEqual(headers["Mcp-Session-Id"], "test-session")
            self.assertEqual(headers["MCP-Protocol-Version"], web.MCP_VERSION)
        for request in requests:
            self.assertEqual(request[3]["User-Agent"], "colibri")
            self.assertNotIn("Authorization", request[3])
            self.assertNotIn("x-api-key", request[3])
        self.assertTrue(all(call.args == ("mcp.parallel.ai",) for call in https.call_args_list))
        self.assertEqual(requests[2][2]["params"], {"name": "web_search", "arguments": {
            "objective": "lunar eclipse", "search_queries": ["lunar eclipse"]}})
        self.assertEqual(data, {"q": "lunar eclipse", "answers": [], "results": [
            {"title": "Moon & eclipses", "url": "https://example.org/moon", "snippet": "One. Two.",
             "engine": "parallel", "date": "2024-01-02"},
            {"title": "https://example.org/another", "url": "https://example.org/another",
             "snippet": "x" * 320, "engine": "parallel", "date": None},
        ]})

    def test_explicit_search_failure_does_not_fall_back(self):
        for tool_result in ({"isError": True}, {"structuredContent": {"results": None}}):
            with self.subTest(tool_result=tool_result):
                conn = mock.Mock()
                conn.getresponse.side_effect = [
                    response(200, body=reply(1, {"protocolVersion": web.MCP_VERSION})),
                    response(202), response(200, body=reply(2, tool_result)),
                ]
                with mock.patch.object(web, "PARALLEL_SEARCH", True), \
                     mock.patch.object(web.http.client, "HTTPSConnection", return_value=conn), \
                     mock.patch.object(web, "_searx") as searx, \
                     mock.patch.object(web, "_duckduckgo") as ddg:
                    with self.assertRaisesRegex(ValueError, "empty query"):
                        web.search(" ")
                    conn.request.assert_not_called()
                    with self.assertRaisesRegex(RuntimeError, "Parallel MCP"):
                        web.search("lunar eclipse")
                    searx.assert_not_called()
                    ddg.assert_not_called()

    def test_existing_provider_selection(self):
        health = mock.MagicMock()
        health.__enter__.return_value.status = 200
        with mock.patch.object(web, "SEARX", ""), \
             mock.patch.object(web.urllib.request, "urlopen", return_value=health) as healthz, \
             mock.patch.object(web, "_searx", return_value=([], [])) as searx, \
             mock.patch.object(web, "_duckduckgo", return_value=([], [])) as ddg, \
             mock.patch.object(web, "_parallel") as parallel:
            web.search("lunar eclipse")
            ddg.assert_called_once_with("lunar eclipse", 6)
            healthz.assert_not_called()
            with mock.patch.object(web, "SEARX", "https://searx.example"):
                web.search("lunar eclipse", 20)
                searx.assert_called_once_with("lunar eclipse", 10)
                healthz.side_effect = OSError("offline")
                web.search("lunar eclipse")
            self.assertEqual(ddg.call_count, 2)
            parallel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
