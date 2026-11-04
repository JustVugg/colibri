"""The dashboard's file tools on the gateway (file_tools.py, POST /fs/*).

- Absent unless COLI_FILE_TOOLS=1: a 404, and /health says file_tools false.
- With it: write, read, list and run in the working folder the page names.
- Only this server's own page: another Origin, or a body that is not JSON, is a 403.
- The API key still applies.
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import file_tools                       # noqa: E402
from openai_server import APIServer     # noqa: E402
from test_openai_server import FakeEngine   # noqa: E402


class FileToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = APIServer(("127.0.0.1", 0), FakeEngine(), "test-model", "secret", 16)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.scheduler.close()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.addCleanup(setattr, file_tools, "ENABLED", file_tools.ENABLED)
        file_tools.ENABLED = True
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def fs(self, action, headers=None, key="secret", **body):
        h = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        h.update(headers or {})
        req = Request(f"{self.base}/fs/{action}", data=json.dumps({"folder": self.folder.name, **body}).encode(),
                      method="POST", headers=h)
        with urlopen(req, timeout=10) as r:
            return json.load(r)

    def status(self, action, **kw):
        with self.assertRaises(HTTPError) as caught:
            self.fs(action, **kw)
        self.addCleanup(caught.exception.close)
        return caught.exception.code

    def health(self):
        req = Request(self.base + "/health", headers={"Authorization": "Bearer secret"})
        with urlopen(req, timeout=5) as r:
            return json.load(r)

    def test_write_read_list_run(self):
        d = self.folder.name
        self.assertEqual(self.fs("write", path="sub/a.txt", content="hello\n")["path"], os.path.join(d, "sub", "a.txt"))
        self.assertEqual(self.fs("read", path="sub/a.txt")["text"], "hello\n")
        self.assertEqual(self.fs("list", path=".")["entries"], ["sub/"])
        r = self.fs("run", command="cat sub/a.txt; exit 3")
        self.assertEqual((r["exit"], r["output"], r["cwd"]), (3, "hello\n", os.path.abspath(d)))
        self.assertTrue(self.health()["file_tools"])

    def test_edit_replaces_one_exact_block(self):
        self.fs("write", path="a.py", content="x = 1\ny = 2\n")
        self.assertEqual(self.fs("edit", path="a.py", old_text="y = 2", new_text="y = 3\nz = 4")["added"], 2)
        self.assertEqual(self.fs("read", path="a.py")["text"], "x = 1\ny = 3\nz = 4\n")

    def test_edit_refuses_a_missing_or_repeated_block(self):
        self.fs("write", path="a.py", content="v = 1\nv = 1\n")
        self.assertEqual(self.status("edit", path="a.py", old_text="v = 1", new_text="v = 2"), 400)
        self.assertEqual(self.status("edit", path="a.py", old_text="nope", new_text="v = 2"), 400)
        self.assertEqual(self.status("edit", path="a.py", old_text="", new_text="v = 2"), 400)
        self.assertEqual(self.fs("read", path="a.py")["text"], "v = 1\nv = 1\n")

    def test_off_by_default(self):
        file_tools.ENABLED = False
        self.assertEqual(self.status("read", path="a.txt"), 404)
        self.assertFalse(self.health()["file_tools"])

    def test_a_missing_file_is_a_400(self):
        self.assertEqual(self.status("read", path="nope.txt"), 400)

    def test_an_unknown_action_is_a_404(self):
        self.assertEqual(self.status("delete", path="a.txt"), 404)

    def test_only_this_servers_own_page(self):
        self.assertEqual(self.status("run", command="true", headers={"Origin": "http://evil.example"}), 403)
        self.assertEqual(self.status("run", command="true", headers={"Content-Type": "text/plain"}), 403)
        same = {"Origin": self.base}
        self.assertEqual(self.fs("run", command="true", headers=same)["exit"], 0)

    def test_the_api_key_still_applies(self):
        self.assertEqual(self.status("run", command="true", key="wrong"), 401)


if __name__ == "__main__":
    unittest.main()
