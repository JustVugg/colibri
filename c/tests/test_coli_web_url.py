import importlib.util
import types
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock


CLI = Path(__file__).resolve().parent.parent / "coli"


class WebUrlTest(unittest.TestCase):
    """`coli web` serves the API and starts the chat UI (ui/server.py) in front of it.
    `--host 0.0.0.0` binds every interface, but 0.0.0.0 is not an address a client can
    connect to on Windows: the readiness poll, the UI's backend and the browser go to
    loopback, as `coli start` already does (setup_flow.urls)."""

    @classmethod
    def setUpClass(cls):
        loader = SourceFileLoader("coli_web_under_test", str(CLI))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def _run(self, host, port=8123, ui_port=None, free=True, server="/x/ui/server.py"):
        coli = self.coli
        polled, opened, started = [], [], []

        class Inline:
            def __init__(self, target, **_kw):
                self.target = target

            def start(self):
                self.target()

        class UI:
            def __init__(self, argv, **kw):
                started.append((argv, kw))
                self.terminated = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

        args = types.SimpleNamespace(model="/m", host=host, port=port, ui_port=ui_port, no_browser=False)
        with mock.patch.object(coli, "need_model"), \
                mock.patch.object(coli, "cmd_serve") as serve, \
                mock.patch.object(coli, "ui_server", return_value=server), \
                mock.patch("setup_flow.port_free", return_value=free), \
                mock.patch("subprocess.Popen", UI), \
                mock.patch("threading.Thread", Inline), \
                mock.patch("time.sleep"), \
                mock.patch("urllib.request.urlopen", side_effect=lambda url, **_kw: polled.append(url)), \
                mock.patch("webbrowser.open", side_effect=opened.append), \
                mock.patch("builtins.print"):
            coli.cmd_web(args)
        serve.assert_called_once_with(args)
        return polled, opened, started

    def test_wildcard_bind_polls_and_opens_loopback(self):
        polled, opened, started = self._run("0.0.0.0")
        self.assertEqual(polled, ["http://127.0.0.1:8123/health"])
        self.assertEqual(opened, ["http://127.0.0.1:8913/"])
        argv, kw = started[0]
        self.assertEqual(argv[1:], ["/x/ui/server.py", "--host", "0.0.0.0", "--port", "8913",
                                    "--backend", "http://127.0.0.1:8123", "--exit-with-stdin"])
        self.assertIsNotNone(kw.get("stdin"))   # a pipe: the UI ends with this process

    def test_named_host_is_kept(self):
        polled, opened, started = self._run("192.168.1.20")
        self.assertEqual(polled, ["http://192.168.1.20:8123/health"])
        self.assertEqual(opened, ["http://192.168.1.20:8913/"])
        self.assertIn("http://192.168.1.20:8123", started[0][0])

    def test_default_port_gives_8790_and_ui_port_wins(self):
        self.assertEqual(self._run("127.0.0.1", port=8000)[1], ["http://127.0.0.1:8790/"])
        self.assertEqual(self._run("127.0.0.1", port=8000, ui_port=9000)[1], ["http://127.0.0.1:9000/"])

    def test_taken_ui_port_serves_the_api_alone(self):
        polled, opened, started = self._run("127.0.0.1", free=False)
        self.assertEqual((polled, opened, started), ([], [], []))

    def test_no_ui_in_the_tree_serves_the_api_alone(self):
        polled, opened, started = self._run("127.0.0.1", server=None)
        self.assertEqual((polled, opened, started), ([], [], []))


if __name__ == "__main__":
    unittest.main()
