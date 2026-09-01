import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.request import Request

from tools.qpack_http_install import (
    HASHES_NAME,
    QpackHTTPError,
    SafeRedirectHandler,
    install_mirror_qpack,
    parse_hashes_index,
)
from tools.qpack_install_policy import QpackInstallError


class Headers:
    def __init__(self, values=()):
        self.values = list(values)

    def get_all(self, name, default=None):
        matches = [value for key, value in self.values
                   if key.lower() == name.lower()]
        return matches if matches else (default or [])

    def items(self):
        return iter(self.values)


class Response:
    def __init__(self, status, body=b"", headers=()):
        self.status = status
        self.body = body
        self.headers = Headers(headers)
        self.offset = 0

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self.body) - self.offset
        chunk = self.body[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk

    def close(self):
        pass


class ScriptedTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, headers, timeout):
        call = {
            "method": method,
            "url": url,
            "headers": dict(headers),
            "timeout": timeout,
        }
        self.calls.append(call)
        if not self.responses:
            raise AssertionError(f"unexpected request: {method} {url}")
        response = self.responses.pop(0)
        return response(call) if callable(response) else response

    def assert_empty(self):
        if self.responses:
            raise AssertionError(f"{len(self.responses)} scripted responses unused")


class QpackMirrorInstallTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "tiny.qpack"
        self.base_url = "https://mirror.example/models/tiny"
        self.config = b'{"model_type":"qwen3_next"}'
        self.files = {
            "model.safetensors": b"weights",
            "packed_experts/layout.json": b"layout",
        }
        self.manifest = json.dumps({
            "magic": "QPACK",
            "version": 1,
            "modelName": "tiny-qwen",
            "sourceCheckpoint": "owner/model@static",
            "quantBits": 4,
            "quantGroupSize": 32,
            "files": {name: len(value) for name, value in self.files.items()},
        }, sort_keys=True, separators=(",", ":")).encode()

    @staticmethod
    def response(body, *, status=200, etag=None, content_length=None,
                 content_range=None, checksum=None):
        headers = [("Content-Length", str(
            len(body) if content_length is None else content_length))]
        if etag is not None:
            headers.append(("ETag", etag))
        if content_range is not None:
            headers.append(("Content-Range", content_range))
        if checksum is not None:
            headers.append(("X-Checksum-Sha256", checksum))
        return Response(status, body, headers)

    def hash_map(self, *, extra=None, omit=(), replacements=None):
        values = {
            "manifest.json": self.manifest,
            "config.json": self.config,
            **self.files,
            **(extra or {}),
        }
        hashes = {
            name: hashlib.sha256(value).hexdigest()
            for name, value in values.items() if name not in omit
        }
        hashes.update(replacements or {})
        return hashes

    def hashes_raw(self, **kwargs):
        return json.dumps(
            {"files": self.hash_map(**kwargs)},
            sort_keys=True, separators=(",", ":")).encode()

    def artifact_response(self, name, *, offset=0, status=None, etag=None,
                          body=None):
        value = self.files[name]
        body = value[offset:] if body is None else body
        status = status if status is not None else (206 if offset else 200)
        content_range = None
        if status == 206:
            content_range = f"bytes {offset}-{len(value) - 1}/{len(value)}"
        return self.response(
            body, status=status, etag=etag,
            content_length=len(body), content_range=content_range)

    def success_responses(self, *, hashes_raw=None, optional=()):
        hashes_raw = hashes_raw or self.hashes_raw(
            extra=dict(optional))
        responses = [
            self.response(hashes_raw, etag='"hashes"'),
            self.response(self.manifest, etag='"manifest"'),
            self.response(self.config, etag='"config"'),
        ]
        responses.extend(self.response(value) for _name, value in optional)
        responses.extend(self.artifact_response(name) for name in sorted(self.files))
        return responses

    def install(self, transport, *, root=None, **kwargs):
        return install_mirror_qpack(
            self.base_url, root or self.root, transport=transport,
            retries=kwargs.pop("retries", 0), timeout=7, chunk_size=3,
            sleep=lambda _seconds: None, log=lambda _message: None, **kwargs)

    def test_installs_legacy_index_and_persists_exact_raw_hashes(self):
        hashes_raw = self.hashes_raw()
        transport = ScriptedTransport(*self.success_responses(hashes_raw=hashes_raw))

        with mock.patch.dict(os.environ, {"HF_TOKEN": "must-not-leak"}):
            result = self.install(transport)

        self.assertIsNone(result.commit)
        self.assertEqual(
            result.source_id,
            f"mirror:{self.base_url}@sha256:"
            f"{hashlib.sha256(hashes_raw).hexdigest()}")
        self.assertEqual(result.downloaded_files, 3)
        self.assertEqual((self.root / "manifest.json").read_bytes(), self.manifest)
        self.assertEqual((self.root / "config.json").read_bytes(), self.config)
        self.assertEqual((self.root / HASHES_NAME).read_bytes(), hashes_raw)
        self.assertTrue(all("Authorization" not in call["headers"]
                            for call in transport.calls))
        self.assertEqual(
            [call["url"].rsplit("/", 1)[-1] for call in transport.calls[:3]],
            ["hashes.json", "manifest.json", "config.json"])
        self.assertEqual(transport.calls[0]["headers"]["Cache-Control"],
                         "no-cache")
        transport.assert_empty()

    def test_v1_and_legacy_have_the_same_semantic_identity(self):
        hashes = self.hash_map()
        legacy = json.dumps({"files": hashes}, indent=2).encode()
        v1 = json.dumps({
            "files": {
                name: {"sha256": digest, "size": len({
                    "manifest.json": self.manifest,
                    "config.json": self.config,
                    **self.files,
                }[name])}
                for name, digest in reversed(list(hashes.items()))
            },
            "schema": "qpack.hashes.v1",
        }, separators=(",", ":")).encode()

        legacy_entries, legacy_id = parse_hashes_index(legacy)
        v1_entries, v1_id = parse_hashes_index(v1)

        self.assertEqual(legacy_id, v1_id)
        self.assertEqual(
            {name: entry.sha256 for name, entry in legacy_entries.items()},
            {name: entry.sha256 for name, entry in v1_entries.items()})

    def test_v1_size_must_match_the_manifest_before_install_starts(self):
        values = {
            "manifest.json": self.manifest,
            "config.json": self.config,
            **self.files,
        }
        index = {
            name: {
                "size": len(value) + int(name == "model.safetensors"),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
            for name, value in values.items()
        }
        hashes_raw = json.dumps({
            "schema": "qpack.hashes.v1",
            "files": index,
        }).encode()
        transport = ScriptedTransport(
            self.response(hashes_raw), self.response(self.manifest))

        with self.assertRaisesRegex(QpackHTTPError, "size mismatch"):
            self.install(transport)

        self.assertFalse(self.root.exists())

    def test_raw_hash_pin_and_manifest_hash_fail_before_destination_creation(self):
        hashes_raw = self.hashes_raw()
        with self.assertRaisesRegex(QpackHTTPError, "pinned SHA-256"):
            self.install(
                ScriptedTransport(self.response(hashes_raw)),
                hashes_sha256="f" * 64)
        self.assertFalse(self.root.exists())

        wrong = self.hashes_raw(replacements={"manifest.json": "f" * 64})
        with self.assertRaisesRegex(QpackHTTPError, "SHA-256 mismatch"):
            self.install(ScriptedTransport(
                self.response(wrong), self.response(self.manifest)))
        self.assertFalse(self.root.exists())

    def test_missing_hash_entry_stops_before_large_transfers(self):
        hashes_raw = self.hashes_raw(omit=("config.json",))
        transport = ScriptedTransport(
            self.response(hashes_raw), self.response(self.manifest))

        with self.assertRaisesRegex(QpackHTTPError, "does not cover"):
            self.install(transport)

        self.assertEqual(len(transport.calls), 2)
        self.assertFalse(self.root.exists())

    def test_extra_hashed_sidecars_are_validated_but_not_downloaded(self):
        hashes_raw = self.hashes_raw(extra={
            "README.md": b"notes",
            "licenses/NOTICE.txt": b"notice",
        })
        transport = ScriptedTransport(
            *self.success_responses(hashes_raw=hashes_raw))

        self.install(transport)

        requested = {call["url"].removeprefix(self.base_url + "/")
                     for call in transport.calls}
        self.assertNotIn("README.md", requested)
        self.assertNotIn("licenses/NOTICE.txt", requested)

    def test_resume_without_etag_uses_exact_range_and_full_hash(self):
        value = self.files["model.safetensors"]
        first = self.artifact_response(
            "model.safetensors", body=value[:3])
        first.headers = Headers([("Content-Length", str(len(value)))])
        with self.assertRaisesRegex(QpackHTTPError, "short HTTP body"):
            self.install(ScriptedTransport(
                self.response(self.hashes_raw()), self.response(self.manifest),
                self.response(self.config), first))

        transport = ScriptedTransport(
            self.response(self.hashes_raw()), self.response(self.manifest),
            self.artifact_response("model.safetensors", offset=3),
            self.artifact_response("packed_experts/layout.json"))
        result = self.install(transport)

        model_call = transport.calls[2]
        self.assertEqual(model_call["headers"]["Range"], "bytes=3-6")
        self.assertNotIn("If-Match", model_call["headers"])
        self.assertEqual(result.resumed_files, 1)
        self.assertEqual((self.root / "model.safetensors").read_bytes(), value)

    def test_failed_if_match_restarts_without_appending(self):
        value = self.files["model.safetensors"]
        first = self.artifact_response(
            "model.safetensors", body=value[:3], etag='"old"')
        first.headers = Headers([
            ("Content-Length", str(len(value))),
            ("ETag", '"old"'),
        ])
        with self.assertRaises(QpackHTTPError):
            self.install(ScriptedTransport(
                self.response(self.hashes_raw()), self.response(self.manifest),
                self.response(self.config), first))

        transport = ScriptedTransport(
            self.response(self.hashes_raw()), self.response(self.manifest),
            self.response(b"", status=412),
            self.artifact_response("model.safetensors", etag='"new"'),
            self.artifact_response("packed_experts/layout.json"))
        self.install(transport)

        self.assertEqual(transport.calls[2]["headers"]["If-Match"], '"old"')
        self.assertNotIn("If-Match", transport.calls[3]["headers"])
        self.assertEqual(transport.calls[3]["headers"]["Range"], "bytes=0-6")
        self.assertEqual((self.root / "model.safetensors").read_bytes(), value)

    def test_changed_snapshot_cannot_resume_existing_journal(self):
        value = self.files["model.safetensors"]
        first = self.artifact_response("model.safetensors", body=value[:3])
        first.headers = Headers([("Content-Length", str(len(value)))])
        with self.assertRaises(QpackHTTPError):
            self.install(ScriptedTransport(
                self.response(self.hashes_raw()), self.response(self.manifest),
                self.response(self.config), first))
        old_hashes = (self.root / HASHES_NAME).read_bytes()

        changed = self.hashes_raw(replacements={
            "model.safetensors": hashlib.sha256(b"changed").hexdigest(),
        })
        transport = ScriptedTransport(
            self.response(changed), self.response(self.manifest))
        with self.assertRaisesRegex(QpackInstallError, "another source"):
            self.install(transport)
        self.assertEqual((self.root / HASHES_NAME).read_bytes(), old_hashes)
        self.assertEqual(len(transport.calls), 2)

    def test_file_hash_mismatch_truncates_and_never_publishes_manifest(self):
        wrong = self.hashes_raw(replacements={
            "model.safetensors": hashlib.sha256(b"different").hexdigest(),
        })
        transport = ScriptedTransport(
            self.response(wrong), self.response(self.manifest),
            self.response(self.config), self.artifact_response("model.safetensors"))

        with self.assertRaisesRegex(QpackHTTPError, "SHA-256 mismatch"):
            self.install(transport)

        self.assertEqual((self.root / "model.safetensors.part").stat().st_size, 0)
        self.assertFalse((self.root / "manifest.json").exists())

    def test_listed_optional_auxiliary_is_required_and_verified(self):
        tokenizer = b'{"version":"1.0"}'
        hashes_raw = self.hashes_raw(extra={"tokenizer.json": tokenizer})
        transport = ScriptedTransport(*self.success_responses(
            hashes_raw=hashes_raw, optional=(("tokenizer.json", tokenizer),)))

        self.install(transport)

        self.assertEqual((self.root / "tokenizer.json").read_bytes(), tokenizer)
        self.assertTrue(transport.calls[3]["url"].endswith("/tokenizer.json"))

        root = Path(self.temporary.name) / "missing-optional.qpack"
        transport = ScriptedTransport(
            self.response(hashes_raw), self.response(self.manifest),
            self.response(self.config), self.response(b"", status=404))
        with self.assertRaisesRegex(QpackHTTPError, "HTTP 404"):
            self.install(transport, root=root)

    def test_completed_mirror_install_is_rehashed_without_object_requests(self):
        self.install(ScriptedTransport(*self.success_responses()))
        transport = ScriptedTransport(
            self.response(self.hashes_raw()), self.response(self.manifest))

        result = self.install(transport)

        self.assertTrue(result.already_complete)
        self.assertEqual(result.downloaded_files, 0)
        self.assertEqual(len(transport.calls), 2)

        (self.root / "model.safetensors").write_bytes(b"WEIGHTS")
        with self.assertRaisesRegex(QpackHTTPError, "artifact changed"):
            self.install(ScriptedTransport(
                self.response(self.hashes_raw()), self.response(self.manifest)))

    def test_hashes_parser_and_base_url_fail_closed(self):
        digest = "a" * 64
        invalid_indexes = [
            b'{"files":{"manifest.json":"' + digest.encode()
            + b'","manifest.json":"' + digest.encode() + b'"}}',
            json.dumps({"files": {"../manifest.json": digest}}).encode(),
            json.dumps({"files": {"hashes.json": digest}}).encode(),
            json.dumps({"files": {"manifest.json": "bad"}}).encode(),
            json.dumps({"schema": "future", "files": {"manifest.json": digest}}).encode(),
        ]
        for raw in invalid_indexes:
            with self.subTest(raw=raw), self.assertRaises(QpackHTTPError):
                parse_hashes_index(raw)

        for base_url in (
                "http://mirror.example/qpack",
                "https://user@mirror.example/qpack",
                "https://mirror.example/a/../qpack",
                "https://mirror.example/qpack?token=secret"):
            with self.subTest(base_url=base_url), self.assertRaises(QpackHTTPError):
                install_mirror_qpack(
                    base_url, self.root, transport=ScriptedTransport())

    def test_cross_origin_redirect_strips_all_explicit_secrets(self):
        handler = SafeRedirectHandler()
        request = Request("https://mirror.example/object", headers={
            "Authorization": "Bearer secret",
            "Cookie": "session=secret",
            "X-Api-Key": "secret",
        })
        redirected = handler.redirect_request(
            request, None, 302, "Found", {}, "https://cdn.example/object")
        for name in ("Authorization", "Cookie", "X-Api-Key"):
            self.assertIsNone(redirected.get_header(name))


if __name__ == "__main__":
    unittest.main()
