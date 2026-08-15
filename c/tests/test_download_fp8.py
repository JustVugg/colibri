import os
import sys
import tempfile
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

    def test_empty_manifest_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as dest, \
             mock.patch.object(download_fp8, "get_shard_list_hf",
                               return_value=([], {})):
            self.assertEqual(self.run_main(dest, "--source", "hf"), 1)


if __name__ == "__main__":
    unittest.main()
