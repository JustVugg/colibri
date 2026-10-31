import importlib.util
import types
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock


CLI = Path(__file__).resolve().parent.parent / "coli"


class WebDashboardUrlTest(unittest.TestCase):
    """`coli web --host 0.0.0.0` binds every interface, but 0.0.0.0 is not an
    address a client can connect to on Windows: the readiness poll and the
    browser must go to loopback, as `coli start` already does (setup_flow.urls)."""

    @classmethod
    def setUpClass(cls):
        loader = SourceFileLoader("coli_web_under_test", str(CLI))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def _run(self, host):
        coli = self.coli
        polled, opened = [], []

        class Inline:
            def __init__(self, target, **_kw):
                self.target = target

            def start(self):
                self.target()

        args = types.SimpleNamespace(model="/m", host=host, port=8123, no_browser=False)
        with mock.patch.object(coli, "need_model"), \
                mock.patch.object(coli, "cmd_serve") as serve, \
                mock.patch("threading.Thread", Inline), \
                mock.patch("time.sleep"), \
                mock.patch("urllib.request.urlopen", side_effect=lambda url, **_kw: polled.append(url)), \
                mock.patch("webbrowser.open", side_effect=opened.append), \
                mock.patch("builtins.print"):
            coli.cmd_web(args)
        serve.assert_called_once_with(args)
        return polled, opened

    def test_wildcard_bind_polls_and_opens_loopback(self):
        polled, opened = self._run("0.0.0.0")
        self.assertEqual(polled, ["http://127.0.0.1:8123/health"])
        self.assertEqual(opened, ["http://127.0.0.1:8123/"])

    def test_named_host_is_kept(self):
        polled, opened = self._run("192.168.1.20")
        self.assertEqual(polled, ["http://192.168.1.20:8123/health"])
        self.assertEqual(opened, ["http://192.168.1.20:8123/"])


if __name__ == "__main__":
    unittest.main()
