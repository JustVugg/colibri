"""Resumable downloads, against a local fake of the Hugging Face hub.

Nothing here touches the network: tests/fake_hub.py serves a repository from a
temporary folder with the hub's listing API, a redirect to a "CDN" path, Range
support and injected faults. The contract under test: a dropped connection
resumes from the bytes on disk (Range, then append), a server that ignores
Range restarts the file instead of appending a second copy, corrupt bytes never
become a file without `.part`, and the token never follows the redirect.
"""
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import setup_download  # noqa: E402
from fake_hub import FakeHub  # noqa: E402

REPO = "tester/tiny-model"


def make_repo(root):
    files = {
        "config.json": b'{"model_type": "qwen3_5_moe_text"}',
        "tokenizer.json": b'{"model": {}}' * 10,
        "model-00000.safetensors": os.urandom(300_000),
        "model-00001.safetensors": os.urandom(200_000),
        "sub/extra.bin": os.urandom(5_000),
        "README.md": b"# readme\n",
    }
    for name, data in files.items():
        path = Path(root, *name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return files


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = os.path.join(self.tmp.name, "src")
        self.dst = os.path.join(self.tmp.name, "dst")
        self.files = make_repo(self.src)
        self.hub = FakeHub({REPO: self.src})
        self.hub.__enter__()
        self.addCleanup(self.hub.__exit__)
        # Retries without the real backoff sleeps.
        patcher = mock.patch.object(setup_download.time, "sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def listing(self, **kw):
        return setup_download.list_repo_files(REPO, "main", base=self.hub.base, **kw)

    def spec(self, name):
        return next(f for f in self.listing() if f["path"] == name)

    def url(self, name):
        return setup_download.file_url(REPO, "main", name, self.hub.base)

    def ranges(self, name):
        return [r["headers"].get("Range") for r in self.hub.requests
                if r["path"].endswith("/" + name) and "/cdn/" in r["path"]]

    def test_listing_sizes_hashes_and_filters(self):
        files = {f["path"]: f for f in self.listing()}
        self.assertEqual(set(files), set(self.files))
        shard = files["model-00000.safetensors"]
        self.assertEqual(shard["size"], 300_000)
        self.assertEqual(shard["sha256"], hashlib.sha256(self.files["model-00000.safetensors"]).hexdigest())
        self.assertIsNone(files["config.json"]["sha256"])
        self.assertIsNotNone(files["config.json"]["git_oid"])
        kept = {f["path"] for f in self.listing(exclude=("sub/*", "*.md"))}
        self.assertEqual(kept, {"config.json", "tokenizer.json", "model-00000.safetensors",
                                "model-00001.safetensors"})
        only = {f["path"] for f in self.listing(include=("*.safetensors",))}
        self.assertEqual(only, {"model-00000.safetensors", "model-00001.safetensors"})

    def test_listing_follows_pagination(self):
        self.hub.page_size = 2
        self.assertEqual({f["path"] for f in self.listing()}, set(self.files))

    def test_missing_repo_is_a_clear_error(self):
        with self.assertRaises(setup_download.DownloadError) as caught:
            setup_download.list_repo_files("nobody/nothing", base=self.hub.base)
        self.assertIn("not found", str(caught.exception))

    def test_whole_repo_and_manifest(self):
        files = self.listing()
        events = []
        setup_download.download_repo(REPO, "main", self.dst, files, base=self.hub.base,
                                     progress=events.append)
        for name, data in self.files.items():
            self.assertEqual(Path(self.dst, *name.split("/")).read_bytes(), data)
        self.assertTrue(setup_download.is_complete(self.dst))
        self.assertEqual(events[-1]["done"], events[-1]["total"])
        self.assertEqual(events[0]["file"], "README.md")        # small files first
        # A rerun fetches nothing.
        before = len(self.hub.requests)
        setup_download.download_repo(REPO, "main", self.dst, files, base=self.hub.base)
        self.assertEqual(len(self.hub.requests), before)

    def test_empty_repository_file_is_completed_and_manifest_is_reusable(self):
        Path(self.src, "empty.txt").write_bytes(b"")
        files = self.listing(include=("empty.txt",))
        self.assertEqual(files[0]["git_oid"], hashlib.sha1(b"blob 0\0").hexdigest())
        setup_download.download_repo(REPO, "main", self.dst, files, base=self.hub.base)
        self.assertEqual(Path(self.dst, "empty.txt").read_bytes(), b"")
        self.assertFalse(Path(self.dst, "empty.txt.part").exists())
        self.assertTrue(setup_download.is_complete(self.dst))
        before = len(self.hub.requests)
        setup_download.download_repo(REPO, "main", self.dst, files, base=self.hub.base)
        self.assertEqual(len(self.hub.requests), before)

    def test_interrupted_download_resumes_with_range(self):
        name = "model-00000.safetensors"
        spec = self.spec(name)
        dest = os.path.join(self.dst, name)
        self.hub.cut_after[name] = 120_000
        with self.assertRaises(setup_download.DownloadError) as caught:
            setup_download.download_file(self.url(name), dest, spec, retries=0)
        self.assertIn("rerun to continue", str(caught.exception))
        self.assertFalse(os.path.exists(dest))
        self.assertEqual(os.path.getsize(dest + ".part"), 120_000)
        # The rerun: asks for the rest only, and the file checks out.
        result = setup_download.download_file(self.url(name), dest, spec, retries=0)
        self.assertEqual(result, "downloaded")
        self.assertEqual(Path(dest).read_bytes(), self.files[name])
        self.assertEqual(self.ranges(name)[-1], "bytes=120000-")
        self.assertFalse(os.path.exists(dest + ".part"))

    def test_dropped_connection_retries_inside_one_call(self):
        name = "model-00001.safetensors"
        self.hub.cut_after[name] = 50_000
        dest = os.path.join(self.dst, name)
        setup_download.download_file(self.url(name), dest, self.spec(name), retries=2)
        self.assertEqual(Path(dest).read_bytes(), self.files[name])
        self.assertEqual(self.ranges(name), [None, "bytes=50000-"])

    def test_server_ignoring_range_restarts_instead_of_appending(self):
        name = "model-00001.safetensors"
        dest = os.path.join(self.dst, name)
        os.makedirs(self.dst, exist_ok=True)
        Path(dest + ".part").write_bytes(self.files[name][:70_000])
        self.hub.ignore_range = True
        setup_download.download_file(self.url(name), dest, self.spec(name), retries=0)
        self.assertEqual(Path(dest).read_bytes(), self.files[name])

    def test_corrupt_partial_is_removed_not_kept(self):
        name = "model-00000.safetensors"
        dest = os.path.join(self.dst, name)
        os.makedirs(self.dst, exist_ok=True)
        Path(dest + ".part").write_bytes(b"\0" * 100_000)       # wrong bytes, right offset
        with self.assertRaises(setup_download.DownloadError) as caught:
            setup_download.download_file(self.url(name), dest, self.spec(name), retries=0)
        self.assertIn("checksum", str(caught.exception))
        self.assertFalse(os.path.exists(dest))
        self.assertFalse(os.path.exists(dest + ".part"))
        # and the rerun after that starts clean and succeeds
        setup_download.download_file(self.url(name), dest, self.spec(name), retries=0)
        self.assertEqual(Path(dest).read_bytes(), self.files[name])

    def test_oversized_partial_starts_over(self):
        name = "config.json"
        dest = os.path.join(self.dst, name)
        os.makedirs(self.dst, exist_ok=True)
        Path(dest + ".part").write_bytes(b"x" * 10_000)
        setup_download.download_file(self.url(name), dest, self.spec(name), retries=0)
        self.assertEqual(Path(dest).read_bytes(), self.files[name])

    def test_token_reaches_the_hub_and_not_the_cdn(self):
        name = "config.json"
        setup_download.download_file(self.url(name), os.path.join(self.dst, name),
                                     self.spec(name), token="hf_secret", retries=0)
        hub_side = [r for r in self.hub.requests if "/resolve/" in r["path"]]
        cdn_side = [r for r in self.hub.requests if "/cdn/" in r["path"]]
        self.assertEqual(hub_side[-1]["headers"].get("Authorization"), "Bearer hf_secret")
        self.assertIsNone(cdn_side[-1]["headers"].get("Authorization"))

    def test_bytes_present_counts_partials(self):
        files = self.listing()
        os.makedirs(self.dst, exist_ok=True)
        Path(self.dst, "config.json").write_bytes(self.files["config.json"])
        Path(self.dst, "model-00000.safetensors.part").write_bytes(b"x" * 1000)
        self.assertEqual(setup_download.bytes_present(self.dst, files),
                         len(self.files["config.json"]) + 1000)
        self.assertFalse(setup_download.is_complete(self.dst))

    def test_repository_download_refuses_existing_symlinks_outside_model_dir(self):
        os.makedirs(self.dst)
        for index, (linked_path, name) in enumerate(((Path(self.dst, "sub"), "sub/extra.bin"),
                                  (Path(self.dst, "model-00000.safetensors.part"), "model-00000.safetensors"))):
            with self.subTest(path=name):
                outside = Path(self.tmp.name, f"outside-{index}")
                outside.mkdir()
                target = outside if name.startswith("sub/") else outside / "part.bin"
                if not name.startswith("sub/"):
                    target.write_bytes(b"")
                try:
                    linked_path.symlink_to(target, target_is_directory=name.startswith("sub/"))
                except NotImplementedError as error:
                    self.skipTest(f"symlinks unavailable: {error}")
                except OSError as error:
                    if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                        self.skipTest(f"symlink privilege unavailable: {error}")
                    raise
                with self.assertRaises(setup_download.DownloadError):
                    setup_download.download_repo(REPO, "main", self.dst, [self.spec(name)], base=self.hub.base)
                self.assertFalse((outside / "extra.bin").exists())
                if not name.startswith("sub/"):
                    self.assertEqual(target.read_bytes(), b"")

    def test_repository_manifest_refuses_existing_symlinks_outside_model_dir(self):
        os.makedirs(self.dst)
        for index, name in enumerate((setup_download.MANIFEST, setup_download.MANIFEST + ".tmp")):
            with self.subTest(path=name):
                outside = Path(self.tmp.name, f"outside-manifest-{index}.json")
                outside.write_bytes(b"keep me")
                linked = Path(self.dst, name)
                try:
                    linked.symlink_to(outside)
                except NotImplementedError as error:
                    self.skipTest(f"symlinks unavailable: {error}")
                except OSError as error:
                    if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                        self.skipTest(f"symlink privilege unavailable: {error}")
                    raise
                try:
                    with self.assertRaises(setup_download.DownloadError):
                        setup_download.download_repo(REPO, "main", self.dst, [self.spec("README.md")], base=self.hub.base)
                    self.assertEqual(outside.read_bytes(), b"keep me")
                finally:
                    linked.unlink()

    def test_unsafe_paths_are_refused(self):
        for bad in ("../escape.bin", "a/../../b", "C:/x", ""):
            with self.subTest(path=bad), self.assertRaises(setup_download.DownloadError):
                setup_download.safe_join(self.dst, bad)

    def test_throughput_probe(self):
        rate = setup_download.probe_throughput(self.url("model-00000.safetensors"),
                                               max_bytes=100_000, max_seconds=5)
        self.assertGreater(rate, 0)
        self.assertEqual(self.ranges("model-00000.safetensors")[-1], "bytes=0-99999")

    def test_curl_speed_parsing(self):
        self.assertEqual(setup_download.parse_curl_speed("3456789.000"), 3456789.0)
        self.assertEqual(setup_download.parse_curl_speed("3456789,500"), 3456789.5)
        self.assertIsNone(setup_download.parse_curl_speed(""))

    def test_verify_file(self):
        name = "model-00001.safetensors"
        path = Path(self.dst, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[name])
        self.assertTrue(setup_download.verify_file(str(path), self.spec(name)))
        path.write_bytes(b"\1" * len(self.files[name]))
        self.assertFalse(setup_download.verify_file(str(path), self.spec(name)))

    def stand_in_curl(self):
        """A curl.exe that does what the transport asks of it: fetch the last
        argument into the file after -o. A script with a shebang on POSIX, a
        .cmd around the same script on Windows."""
        script = Path(self.tmp.name, "curl_stand_in.py")
        script.write_text(
            "import sys, urllib.request\n"
            "args = sys.argv[1:]\n"
            "out = args[args.index('-o') + 1]\n"
            "data = urllib.request.urlopen(args[-1]).read()\n"
            "open(out, 'wb').write(data)\n")
        if os.name == "nt":
            wrapper = Path(self.tmp.name, "curl.cmd")
            wrapper.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
            return str(wrapper)
        wrapper = Path(self.tmp.name, "curl.exe")
        wrapper.write_text("#!" + sys.executable + "\n" + script.read_text())
        wrapper.chmod(0o755)
        return str(wrapper)

    def test_windows_curl_transport(self):
        """WSL's slow-network path: Windows' curl.exe writes the .part file, then
        the same size and hash checks decide whether it becomes the file."""
        fake_curl = self.stand_in_curl()
        name = "model-00000.safetensors"
        dest = os.path.join(self.dst, name)
        seen = []
        with mock.patch.object(setup_download, "wsl_windows_path", side_effect=lambda p: p):
            result = setup_download.download_file_windows(
                self.url(name), dest, self.spec(name), fake_curl,
                progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(result, "downloaded")
        self.assertEqual(Path(dest).read_bytes(), self.files[name])
        self.assertEqual(seen[-1], (300_000, 300_000))
        # A transfer that ends with the wrong bytes is not kept.
        other = "model-00001.safetensors"
        bad = dict(self.spec(other), sha256="0" * 64)
        with mock.patch.object(setup_download, "wsl_windows_path", side_effect=lambda p: p), \
                self.assertRaises(setup_download.DownloadError):
            setup_download.download_file_windows(self.url(other), os.path.join(self.dst, other),
                                                 bad, fake_curl)
        self.assertFalse(os.path.exists(os.path.join(self.dst, other)))

    def test_token_from_file(self):
        with tempfile.TemporaryDirectory() as hf_home:
            Path(hf_home, "token").write_text("hf_from_file\n")
            env = {k: v for k, v in os.environ.items()
                   if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
            env["HF_HOME"] = hf_home
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(setup_download.hf_token(), "hf_from_file")


if __name__ == "__main__":
    unittest.main()
