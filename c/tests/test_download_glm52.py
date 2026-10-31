"""Readonly GLM preflight with the complete helper and actual optional Hub SDK.

The owned loopback server supplies tiny metadata only; no weights are served.
Default CPU checks do not require the tool's optional Hugging Face dependency.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.parse import urlsplit


class DownloadGlm52DiskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import huggingface_hub
            import httpx
        except ImportError:
            raise unittest.SkipTest("install optional huggingface_hub/httpx to run real-SDK preflight checks")
        if not hasattr(huggingface_hub, "set_client_factory"):
            raise unittest.SkipTest("real-SDK integration checks need huggingface_hub's HTTP client factory")
        from huggingface_hub import constants
        from huggingface_hub.utils import _http
        cls.hub = huggingface_hub
        cls.old_factory = _http._GLOBAL_CLIENT_FACTORY
        cls.temp = tempfile.TemporaryDirectory(prefix="glm52-sdk-metadata-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.requests = []
        metadata = {"id": "zai-org/GLM-5.2-FP8", "sha": "a" * 40, "private": False,
                    "gated": False, "siblings": [{"rfilename": "fixture.safetensors", "size": 24},
                                                {"rfilename": "config.json", "size": 12}]}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *unused):
                pass

            def do_GET(self):
                if (urlsplit(self.path).path != "/api/models/zai-org/GLM-5.2-FP8/revision/main"
                        or self.headers.get("Authorization")):
                    self.send_error(400)
                    return
                cls.requests.append(self.path)
                body = json.dumps(metadata).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      kwargs={"poll_interval": .01}, daemon=True)
        cls.thread.start()
        cls.addClassCleanup(cls.thread.join, timeout=2)
        cls.addClassCleanup(cls.server.server_close)
        cls.addClassCleanup(cls.server.shutdown)
        endpoint = f"http://127.0.0.1:{cls.server.server_port}"
        home = Path(cls.temp.name)
        settings = patch.multiple(constants, ENDPOINT=endpoint, HF_HOME=str(home),
                                  HF_TOKEN_PATH=str(home / "token"),
                                  HF_STORED_TOKENS_PATH=str(home / "stored_tokens"),
                                  HF_HUB_DISABLE_IMPLICIT_TOKEN=True, HF_HUB_DISABLE_TELEMETRY=True)
        settings.start()
        cls.addClassCleanup(settings.stop)
        tokens = patch.dict(os.environ, {"HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": ""})
        tokens.start()
        cls.addClassCleanup(tokens.stop)

        class OwnedMetadataTransport(httpx.BaseTransport):
            def __init__(self):
                self.actual = httpx.HTTPTransport()

            def handle_request(self, request):
                if (request.url.scheme != "http" or request.url.host != "127.0.0.1"
                        or request.url.port != cls.server.server_port or request.method != "GET"
                        or request.url.path != "/api/models/zai-org/GLM-5.2-FP8/revision/main"
                        or "authorization" in request.headers):
                    raise RuntimeError("preflight test permits only owned metadata requests")
                return self.actual.handle_request(request)

            def close(self):
                self.actual.close()

        cls.hub.set_client_factory(lambda: httpx.Client(transport=OwnedMetadataTransport(),
                                                      trust_env=False, follow_redirects=False, timeout=3))
        cls.addClassCleanup(cls.hub.set_client_factory, cls.old_factory)
        path = Path(__file__).resolve().parents[1] / "tools/download_glm52.py"
        spec = importlib.util.spec_from_file_location("actual_glm52_downloader", path)
        cls.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tool)
        cls.tool.REVISION = "main"
        if cls.tool.HfApi is not cls.hub.HfApi:
            raise AssertionError("test must use the complete helper's real SDK class")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="glm52-selected-destination-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.original_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.original_cwd)

    def check_path(self, destination, expected):
        probes = []
        original = self.tool.shutil.disk_usage
        previous_requests = len(self.requests)
        existed = Path(destination).exists()

        def usage(path):
            probes.append(os.path.abspath(os.fspath(path)))
            return original(path)

        with patch.object(self.tool, "DEST", str(destination)), \
                patch.object(self.tool.shutil, "disk_usage", usage), contextlib.redirect_stdout(io.StringIO()):
            self.tool.check()
        self.assertEqual(probes, [os.path.abspath(expected)])
        self.assertEqual(Path(destination).exists(), existed, "--check must not create destinations")
        self.assertEqual(len(self.requests), previous_requests + 1)
        self.assertIn("blobs=true", self.requests[-1])

    def test_new_one_level_destination_uses_existing_parent(self):
        self.check_path(self.root / "model", self.root)

    def test_new_nested_destination_uses_ancestor_without_creating_parents(self):
        self.check_path(self.root / "new-parent/deeper/model", self.root)
        self.assertFalse((self.root / "new-parent").exists())

    def test_existing_destination_uses_its_own_filesystem(self):
        destination = self.root / "existing"
        destination.mkdir()
        self.check_path(destination, destination)

    def test_bare_relative_new_destination_uses_working_directory(self):
        self.check_path("model", self.root)

    def test_bare_relative_existing_destination_uses_target(self):
        destination = self.root / "existing"
        destination.mkdir()
        self.check_path("existing", destination)

    def test_existing_symlink_destination_uses_its_target_filesystem(self):
        destination = self.root / "existing"
        destination.mkdir()
        link = self.root / "link"
        try:
            link.symlink_to(destination, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("host does not permit an owned directory symlink")
        self.check_path(link, link)


if __name__ == "__main__":
    unittest.main()
