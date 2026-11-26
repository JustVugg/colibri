import os
import http.server
import shutil
import threading
import sys
import tempfile
import types
from urllib.parse import quote, parse_qs, urlsplit
import unittest
from unittest import mock

import download_fp8


class DownloadExitStatusTests(unittest.TestCase):
    META_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
                  "generation_config.json", "model.safetensors.index.json")

    def run_main(self, dest, *args):
        with mock.patch.object(download_fp8, "DEST", dest), \
             mock.patch.object(sys, "argv", ["download_fp8.py", *args]):
            return download_fp8.main()

    def test_huggingface_listing_uses_download_revision(self):
        for revision in (None, "refs/pr/12"):
            api = mock.Mock()
            api.repo_info.return_value = types.SimpleNamespace(siblings=[])
            hub = types.SimpleNamespace(HfApi=mock.Mock(return_value=api))
            with self.subTest(revision=revision), mock.patch.dict(sys.modules, {"huggingface_hub": hub}), \
                    mock.patch.dict(os.environ, {}, clear=True):
                if revision:
                    os.environ["GLM_HF_REVISION"] = revision
                download_fp8.get_shard_list_hf()
                self.assertEqual(api.repo_info.call_args.kwargs.get("revision", "main"), revision or "main")

    def test_modelscope_listing_uses_download_revision(self):
        for revision in (None, "release/tag"):
            requests = types.SimpleNamespace(get=mock.Mock())
            requests.get.return_value.json.return_value = {"Data": {"Files": []}}
            with self.subTest(revision=revision), mock.patch.dict(sys.modules, {"requests": requests}), \
                    mock.patch.dict(os.environ, {}, clear=True):
                if revision:
                    os.environ["GLM_MS_REVISION"] = revision
                download_fp8.get_shard_list_ms()
                call = requests.get.call_args
                query = parse_qs(urlsplit(call.args[0]).query, keep_blank_values=True)
                query.update({key: [value] for key, value in call.kwargs.get("params", {}).items()})
                self.assertEqual(query["Revision"], [revision or "master"])

    def test_curl_fallback_uses_the_download_revision(self):
        name = "model-00001.safetensors"
        for source, variable in (("hf", "GLM_HF_REVISION"), ("ms", "GLM_MS_REVISION")):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as dest, \
                    mock.patch.dict(os.environ, {variable: "release/tag"}), \
                    mock.patch.object(download_fp8, f"get_shard_list_{source}", return_value=([name], {name: 4})), \
                    mock.patch.object(download_fp8, f"download_file_{source}"), \
                    mock.patch.object(download_fp8.time, "sleep"):
                for metadata in self.META_FILES:
                    open(os.path.join(dest, metadata), "wb").close()
                def fallback(fn, url, expected):
                    self.assertIn(quote("release/tag", safe=""), url)
                    if source == "ms":
                        self.assertEqual(parse_qs(urlsplit(url).query)["FilePath"], [name])
                    else:
                        self.assertTrue(urlsplit(url).path.endswith("/" + name))
                    with open(os.path.join(dest, fn), "wb") as out:
                        out.write(b"data")
                    return True
                with mock.patch.object(download_fp8, "download_file_curl", side_effect=fallback):
                    self.assertEqual(self.run_main(dest, "--source", source), 0)

    @unittest.skipUnless(shutil.which("curl"), "curl is required for the fallback")
    def test_curl_fallback_requests_the_exact_remote_filename(self):
        # Check the actual request sent by curl against the two service contracts:
        # ModelScope uses FilePath as a query value; HF uses a resolve path.
        name = "nested/model#1&part.safetensors"
        revision = "release/tag"
        requests_seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                requests_seen.append(self.path)
                self.send_response(200)
                self.send_header("Content-Length", "4")
                self.end_headers()
                self.wfile.write(b"data")

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        real_run = download_fp8.subprocess.run
        try:
            for source, variable in (("ms", "GLM_MS_REVISION"), ("hf", "GLM_HF_REVISION")):
                def local_run(command, **kwargs):
                    command = list(command)
                    remote = urlsplit(command[-1])
                    command[-1] = (f"http://127.0.0.1:{server.server_port}"
                                   f"{remote.path}?{remote.query}")
                    # The transport is real; only its origin is replaced.
                    return real_run(command, **kwargs)

                with self.subTest(source=source), tempfile.TemporaryDirectory() as dest, \
                        mock.patch.dict(os.environ, {variable: revision}), \
                        mock.patch.object(download_fp8, f"get_shard_list_{source}",
                                          return_value=([name], {name: 4})), \
                        mock.patch.object(download_fp8, f"download_file_{source}"), \
                        mock.patch.object(download_fp8.subprocess, "run", side_effect=local_run):
                    os.makedirs(os.path.join(dest, "nested"))
                    for metadata in self.META_FILES:
                        open(os.path.join(dest, metadata), "wb").close()
                    self.assertEqual(self.run_main(dest, "--source", source, "--parallel", "1"), 0)
                    request = urlsplit(requests_seen[-1])
                    if source == "ms":
                        query = parse_qs(request.query)
                        self.assertEqual(query["Revision"], [revision])
                        self.assertEqual(query["FilePath"], [name])
                    else:
                        self.assertEqual(request.path,
                                         f"/{download_fp8.REPO_HF}/resolve/"
                                         f"{quote(revision, safe='')}/{quote(name, safe='/')}")
                        self.assertEqual(request.query, "")
                    with open(os.path.join(dest, name), "rb") as shard:
                        self.assertEqual(shard.read(), b"data")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_wrong_size_sdk_shard_reaches_curl_fallback(self):
        name = "model-00001.safetensors"
        for source in ("hf", "ms"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as dest:
                for metadata in self.META_FILES:
                    open(os.path.join(dest, metadata), "wb").close()
                def sdk_download(fn):
                    with open(os.path.join(dest, fn), "wb") as out:
                        out.write(b"x")
                def fallback(fn, base, expected):
                    self.assertEqual(expected, 4)
                    with open(os.path.join(dest, fn), "wb") as out:
                        out.write(b"data")
                    return True
                with mock.patch.object(download_fp8, f"get_shard_list_{source}",
                                       return_value=([name], {name: 4})), \
                     mock.patch.object(download_fp8, f"download_file_{source}",
                                       side_effect=sdk_download), \
                     mock.patch.object(download_fp8, "download_file_curl",
                                       side_effect=fallback) as curl:
                    self.assertEqual(self.run_main(dest, "--source", source, "--parallel", "1"), 0)
                    curl.assert_called_once()
                with open(os.path.join(dest, name), "rb") as shard:
                    self.assertEqual(shard.read(), b"data")

    def test_failed_shard_returns_nonzero(self):
        manifest = (["model-00001.safetensors"],
                    {"model-00001.safetensors": 4})
        with tempfile.TemporaryDirectory() as dest, \
             mock.patch.object(download_fp8, "get_shard_list_hf",
                               return_value=manifest), \
             mock.patch.object(download_fp8, "download_file_hf",
                               side_effect=RuntimeError("offline")), \
             mock.patch.object(download_fp8.time, "sleep"):
            self.assertEqual(self.run_main(dest, "--source", "hf"), 1)

    def test_explicit_modelscope_failure_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as dest, \
             mock.patch.object(download_fp8, "get_shard_list_ms",
                               side_effect=RuntimeError("offline")):
            self.assertEqual(self.run_main(dest, "--source", "ms"), 1)

    def test_explicit_modelscope_empty_manifest_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as dest, \
             mock.patch.object(download_fp8, "get_shard_list_ms",
                               return_value=([], {})), \
             mock.patch.object(download_fp8, "get_shard_list_hf") as hf:
            self.assertEqual(self.run_main(dest, "--source", "ms"), 1)
            hf.assert_not_called()

    def test_auto_empty_modelscope_manifest_uses_huggingface(self):
        name = "model-00001.safetensors"
        manifest = ([name], {name: 4})
        with tempfile.TemporaryDirectory() as dest, \
             mock.patch.object(download_fp8, "get_shard_list_ms",
                               return_value=([], {})), \
             mock.patch.object(download_fp8, "get_shard_list_hf",
                               return_value=manifest), \
             mock.patch.object(download_fp8, "download_file_ms") as ms:
            def download(fn):
                with open(os.path.join(dest, fn), "wb") as out:
                    out.write(b"data" if fn == name else b"")
            with mock.patch.object(download_fp8, "download_file_hf",
                                   side_effect=download):
                self.assertEqual(self.run_main(dest), 0)
            ms.assert_not_called()

    def test_complete_manifest_returns_zero(self):
        name = "model-00001.safetensors"
        manifest = ([name], {name: 4})
        with tempfile.TemporaryDirectory() as dest, \
             open(os.path.join(dest, name), "wb") as shard:
            shard.write(b"data")
            shard.flush()
            for metadata in self.META_FILES:
                open(os.path.join(dest, metadata), "wb").close()
            with mock.patch.object(download_fp8, "get_shard_list_hf",
                                   return_value=manifest), \
                 mock.patch.object(download_fp8, "download_file_hf"):
                self.assertEqual(self.run_main(dest, "--source", "hf"), 0)

    def test_missing_metadata_returns_nonzero(self):
        name = "model-00001.safetensors"
        with tempfile.TemporaryDirectory() as dest:
            with open(os.path.join(dest, name), "wb") as shard:
                shard.write(b"data")
            manifest = ([name], {name: 4})
            with mock.patch.object(download_fp8, "get_shard_list_hf",
                                   return_value=manifest), \
                 mock.patch.object(download_fp8, "download_file_hf"):
                self.assertEqual(self.run_main(dest, "--source", "hf"), 1)

    def test_unknown_shard_sizes_do_not_terminate_worker(self):
        names = ["model-00001.safetensors", "model-00002.safetensors"]
        manifest = (names, {name: 0 for name in names})
        with tempfile.TemporaryDirectory() as dest, \
             mock.patch.object(download_fp8, "get_shard_list_ms", return_value=manifest), \
             mock.patch.object(download_fp8.threading, "excepthook") as failed_worker:
            def download(name):
                with open(os.path.join(dest, name), "wb") as output:
                    output.write(b"data")
            with mock.patch.object(download_fp8, "download_file_ms", side_effect=download):
                self.assertEqual(self.run_main(dest, "--source", "ms", "--parallel", "1"), 0)
            failed_worker.assert_not_called()
            for name in names:
                with open(os.path.join(dest, name), "rb") as shard:
                    self.assertEqual(shard.read(), b"data")

    def test_empty_manifest_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as dest, \
             mock.patch.object(download_fp8, "get_shard_list_hf",
                               return_value=([], {})):
            self.assertEqual(self.run_main(dest, "--source", "hf"), 1)


if __name__ == "__main__":
    unittest.main()
