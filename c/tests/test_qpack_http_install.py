import hashlib
import http.client
import json
import tempfile
import unittest
from pathlib import Path
from urllib.request import Request

from tools.qpack_http_install import (
    STATE_NAME,
    QpackHTTPError,
    SafeRedirectHandler,
    _check_adapter_namespace,
    _declared_auxiliary_names,
    install_hf_qpack,
)
from tools.qpack_install_policy import parse_manifest


COMMIT = "0123456789abcdef0123456789abcdef01234567"


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
        self.closed = False

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self.body) - self.offset
        chunk = self.body[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk

    def close(self):
        self.closed = True


class IncompleteResponse(Response):
    def read(self, _size=-1):
        if self.offset == 0:
            self.offset = len(self.body)
            raise http.client.IncompleteRead(self.body)
        return b""


class ScriptedTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, headers, timeout):
        self.calls.append({
            "method": method,
            "url": url,
            "headers": dict(headers),
            "timeout": timeout,
        })
        if not self.responses:
            raise AssertionError(f"unexpected request: {method} {url}")
        response = self.responses.pop(0)
        if callable(response):
            return response(self.calls[-1])
        return response

    def assert_empty(self):
        if self.responses:
            raise AssertionError(f"{len(self.responses)} scripted responses unused")


class QpackHTTPInstallTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "tiny.qpack"
        self.files = {
            "model.safetensors": b"weights",
            "packed_experts/layout.json": b"layout",
        }
        self.config = b'{"model_type":"qwen3_next"}'
        self.manifest = self.make_manifest(
            {name: len(value) for name, value in self.files.items()})

    @staticmethod
    def make_manifest(files):
        return json.dumps({
            "magic": "QPACK",
            "version": 1,
            "modelName": "tiny-qwen",
            "sourceCheckpoint": f"owner/model@{COMMIT}",
            "quantBits": 4,
            "quantGroupSize": 32,
            "files": files,
        }, sort_keys=True, separators=(",", ":")).encode()

    @staticmethod
    def response(body, *, status=200, etag=None, checksum=False,
                 content_range=None, extra=()):
        headers = [("Content-Length", str(len(body))),
                   ("X-Repo-Commit", COMMIT)]
        if etag is not None:
            headers.append(("ETag", etag))
        if checksum:
            headers.append(("X-Checksum-Sha256",
                            hashlib.sha256(body).hexdigest()))
        if content_range is not None:
            headers.append(("Content-Range", content_range))
        headers.extend(extra)
        return Response(status, body, headers)

    def manifest_response(self):
        return self.response(self.manifest, etag='"manifest"', checksum=True)

    def config_response(self, body=None, status=200):
        body = self.config if body is None else body
        return self.response(body, status=status, etag='"config"', checksum=True)

    def artifact_response(self, name, *, offset=0, status=None, etag=None,
                          body=None, extra=()):
        value = self.files[name]
        body = value[offset:] if body is None else body
        status = status if status is not None else (206 if offset else 200)
        content_range = None
        if status == 206:
            content_range = f"bytes {offset}-{len(value) - 1}/{len(value)}"
        checksum = hashlib.sha256(value).hexdigest()
        headers = [("Content-Length", str(len(body))),
                   ("X-Repo-Commit", COMMIT),
                   ("ETag", etag or f'"{name}"'),
                   ("X-Checksum-Sha256", checksum)]
        if content_range is not None:
            headers.append(("Content-Range", content_range))
        headers.extend(extra)
        return Response(status, body, headers)

    def success_responses(self, include_config=True):
        responses = [self.manifest_response()]
        if include_config:
            responses.append(self.config_response())
        responses.extend(self.artifact_response(name)
                         for name in sorted(self.files))
        return responses

    def install(self, transport, *, root=None, revision=COMMIT, retries=0,
                token=None, optional_aux=()):
        return install_hf_qpack(
            "owner/model", revision, root or self.root,
            endpoint="https://hub.example", token=token,
            transport=transport, retries=retries, timeout=7,
            chunk_size=3, optional_aux=optional_aux,
            sleep=lambda _seconds: None,
            log=lambda _message: None)

    def test_resolves_revision_then_uses_only_commit_pinned_urls(self):
        api = Response(200, json.dumps({"sha": COMMIT}).encode(),
                       [("Content-Length", str(len(json.dumps({"sha": COMMIT}))))])
        transport = ScriptedTransport(api, *self.success_responses())

        result = self.install(transport, revision="main", token="secret-token")

        self.assertEqual(result.commit, COMMIT)
        self.assertEqual(result.downloaded_files, 3)
        self.assertEqual(result.downloaded_bytes,
                         len(self.config) + sum(map(len, self.files.values())))
        self.assertTrue((self.root / "manifest.json").is_file())
        self.assertEqual((self.root / "config.json").read_bytes(), self.config)
        for name, value in self.files.items():
            self.assertEqual((self.root / name).read_bytes(), value)
        self.assertIn("/api/models/owner/model/revision/main",
                      transport.calls[0]["url"])
        for call in transport.calls[1:]:
            self.assertIn(f"/resolve/{COMMIT}/", call["url"])
            self.assertEqual(call["headers"]["Authorization"],
                             "Bearer secret-token")
        state = (self.root / STATE_NAME).read_text(encoding="utf-8")
        self.assertIn(f"owner/model@{COMMIT}", state)
        self.assertNotIn("secret-token", state)
        transport.assert_empty()

    def test_short_body_retries_with_range_and_if_range(self):
        first = self.artifact_response(
            "model.safetensors", body=self.files["model.safetensors"][:3])
        # The declared response length is the full object, but EOF arrives early.
        first.headers = Headers([
            ("Content-Length", str(len(self.files["model.safetensors"]))),
            ("X-Repo-Commit", COMMIT),
            ("ETag", '"model.safetensors"'),
            ("X-Checksum-Sha256",
             hashlib.sha256(self.files["model.safetensors"]).hexdigest()),
        ])
        transport = ScriptedTransport(
            self.manifest_response(), self.config_response(), first,
            self.artifact_response("model.safetensors", offset=3),
            self.artifact_response("packed_experts/layout.json"))

        result = self.install(transport, retries=1)

        model_calls = [call for call in transport.calls
                       if call["url"].endswith("/model.safetensors")]
        self.assertEqual(model_calls[0]["headers"]["Range"], "bytes=0-6")
        self.assertNotIn("If-Range", model_calls[0]["headers"])
        self.assertEqual(model_calls[1]["headers"]["Range"], "bytes=3-6")
        self.assertEqual(model_calls[1]["headers"]["If-Range"],
                         '"model.safetensors"')
        self.assertEqual(result.resumed_files, 1)
        self.assertEqual((self.root / "model.safetensors").read_bytes(),
                         self.files["model.safetensors"])
        transport.assert_empty()

    def test_incomplete_read_partial_bytes_are_resumed(self):
        value = self.files["model.safetensors"]
        first = IncompleteResponse(200, value[:3], [
            ("Content-Length", str(len(value))),
            ("ETag", '"model.safetensors"'),
            ("X-Checksum-Sha256", hashlib.sha256(value).hexdigest()),
        ])
        transport = ScriptedTransport(
            self.manifest_response(), self.config_response(), first,
            self.artifact_response("model.safetensors", offset=3),
            self.artifact_response("packed_experts/layout.json"))

        self.install(transport, retries=1)

        model_calls = [call for call in transport.calls
                       if call["url"].endswith("/model.safetensors")]
        self.assertEqual(model_calls[1]["headers"]["Range"], "bytes=3-6")
        self.assertEqual((self.root / "model.safetensors").read_bytes(), value)

    def test_resume_rejects_http_200_and_keeps_existing_prefix(self):
        first = self.artifact_response(
            "model.safetensors", body=self.files["model.safetensors"][:3])
        first.headers = Headers([
            ("Content-Length", str(len(self.files["model.safetensors"]))),
            ("ETag", '"model.safetensors"'),
        ])
        transport = ScriptedTransport(
            self.manifest_response(), self.config_response(), first)
        with self.assertRaisesRegex(QpackHTTPError, "short HTTP body"):
            self.install(transport)
        state = json.loads((self.root / STATE_NAME).read_text())
        self.assertEqual(state["files"]["model.safetensors"]["etag"],
                         '"model.safetensors"')

        resumed = self.artifact_response(
            "model.safetensors", offset=3, status=200)
        transport = ScriptedTransport(self.manifest_response(), resumed)
        with self.assertRaisesRegex(QpackHTTPError, "did not return HTTP 206"):
            self.install(transport)
        self.assertEqual((self.root / "model.safetensors.part").read_bytes(),
                         self.files["model.safetensors"][:3])

    def test_resume_rejects_changed_etag_before_appending(self):
        first = self.artifact_response(
            "model.safetensors", body=self.files["model.safetensors"][:3])
        first.headers = Headers([
            ("Content-Length", str(len(self.files["model.safetensors"]))),
            ("ETag", '"original"'),
        ])
        with self.assertRaises(QpackHTTPError):
            self.install(ScriptedTransport(
                self.manifest_response(), self.config_response(), first))

        changed = self.artifact_response(
            "model.safetensors", offset=3, etag='"changed"')
        with self.assertRaisesRegex(QpackHTTPError, "ETag changed"):
            self.install(ScriptedTransport(self.manifest_response(), changed))
        self.assertEqual((self.root / "model.safetensors.part").read_bytes(),
                         self.files["model.safetensors"][:3])

    def test_checksum_mismatch_resets_partial_and_never_publishes_manifest(self):
        response = self.artifact_response("model.safetensors")
        response.headers = Headers([
            ("Content-Length", str(len(self.files["model.safetensors"]))),
            ("X-Checksum-Sha256", "f" * 64),
        ])
        transport = ScriptedTransport(
            self.manifest_response(), self.config_response(), response)

        with self.assertRaisesRegex(QpackHTTPError, "SHA-256 mismatch"):
            self.install(transport)

        self.assertEqual((self.root / "model.safetensors.part").stat().st_size, 0)
        self.assertFalse((self.root / "manifest.json").exists())

    def test_range_and_encoding_protocol_errors_fail_closed(self):
        cases = {
            "missing-range": Response(206, self.files["model.safetensors"], [
                ("Content-Length", "7"),
            ]),
            "wrong-range": Response(206, self.files["model.safetensors"], [
                ("Content-Length", "7"), ("Content-Range", "bytes 1-6/7"),
            ]),
            "range-on-200": Response(200, self.files["model.safetensors"], [
                ("Content-Length", "7"), ("Content-Range", "bytes 0-6/7"),
            ]),
            "duplicate-length": Response(200, self.files["model.safetensors"], [
                ("Content-Length", "7"), ("Content-Length", "7"),
            ]),
            "compressed": Response(200, self.files["model.safetensors"], [
                ("Content-Length", "7"), ("Content-Encoding", "gzip"),
            ]),
        }
        for label, response in cases.items():
            with self.subTest(label=label):
                root = Path(self.temporary.name) / f"{label}.qpack"
                transport = ScriptedTransport(
                    self.manifest_response(), self.config_response(), response)
                with self.assertRaises(QpackHTTPError):
                    self.install(transport, root=root)
                self.assertFalse((root / "manifest.json").exists())

    def test_required_config_must_exist_and_be_valid_before_weights(self):
        cases = [
            self.config_response(body=b"", status=404),
            self.config_response(body=b"not-json"),
            self.config_response(body=b"[]"),
        ]
        for index, response in enumerate(cases):
            with self.subTest(index=index):
                root = Path(self.temporary.name) / f"config-{index}.qpack"
                transport = ScriptedTransport(self.manifest_response(), response)
                with self.assertRaises(QpackHTTPError):
                    self.install(transport, root=root)
                self.assertEqual(len(transport.calls), 2)
                self.assertFalse((root / "manifest.json").exists())

    def test_manifest_declared_auxiliaries_use_the_resumable_artifact_path(self):
        tokenizer = b'{"version":"1.0"}'
        files = dict(self.files)
        files["config.json"] = self.config
        files["tokenizer.json"] = tokenizer
        raw = self.make_manifest({name: len(value) for name, value in files.items()})
        responses = [self.response(raw)]
        for name in ["config.json", "tokenizer.json", "model.safetensors",
                     "packed_experts/layout.json"]:
            value = files[name]
            responses.append(Response(200, value, [
                ("Content-Length", str(len(value))),
                ("ETag", f'"{name}"'),
                ("X-Checksum-Sha256", hashlib.sha256(value).hexdigest()),
            ]))
        transport = ScriptedTransport(*responses)

        self.install(transport)

        self.assertEqual((self.root / "config.json").read_bytes(), self.config)
        self.assertEqual((self.root / "tokenizer.json").read_bytes(), tokenizer)
        artifact_calls = [call["url"].rsplit("/", 1)[-1]
                          for call in transport.calls[1:]]
        self.assertEqual(artifact_calls[0], "config.json")
        transport.assert_empty()

    def test_optional_auxiliary_generator_is_not_consumed_by_validation(self):
        tokenizer = b'{"version":"1.0"}'
        transport = ScriptedTransport(
            self.manifest_response(), self.config_response(),
            self.response(tokenizer, checksum=True),
            *(self.artifact_response(name) for name in sorted(self.files)))

        self.install(
            transport,
            optional_aux=(name for name in ("tokenizer.json",)))

        self.assertEqual((self.root / "tokenizer.json").read_bytes(), tokenizer)
        self.assertTrue(transport.calls[2]["url"].endswith("/tokenizer.json"))
        transport.assert_empty()

    def test_real_swiftlet_manifest_declared_auxiliaries_are_accepted(self):
        fixture = Path(__file__).parent / "fixtures" / "swiftlet_qpack_manifest.json"
        plan = parse_manifest(fixture.read_bytes(), f"hf:owner/model@{COMMIT}")
        _check_adapter_namespace(plan)
        self.assertEqual(
            _declared_auxiliary_names(plan),
            {"chat_template.jinja", "config.json", "tokenizer.json",
             "tokenizer_config.json", "vocab.json"})
        self.assertEqual(len(plan.files), 47)

    def test_hugging_face_manifest_may_declare_hashes_json_as_an_artifact(self):
        files = dict(self.files)
        files["hashes.json"] = b'{"files":{}}'
        raw = self.make_manifest({name: len(value) for name, value in files.items()})
        responses = [self.response(raw), self.config_response()]
        for name in sorted(files):
            value = files[name]
            responses.append(Response(200, value, [
                ("Content-Length", str(len(value))),
                ("ETag", f'"{name}"'),
                ("X-Checksum-Sha256", hashlib.sha256(value).hexdigest()),
            ]))

        self.install(ScriptedTransport(*responses))

        self.assertEqual((self.root / "hashes.json").read_bytes(), files["hashes.json"])

    def test_manifest_cannot_claim_adapter_state_or_sidecar_namespace(self):
        for name in (STATE_NAME, "Config.json", "tokenizer.json/child",
                     ".tokenizer.json.part"):
            with self.subTest(name=name):
                files = {"packed_experts/layout.json": 6, name: 1}
                raw = self.make_manifest(files)
                transport = ScriptedTransport(self.response(raw))
                with self.assertRaisesRegex(QpackHTTPError, "collides"):
                    self.install(transport,
                                 root=Path(self.temporary.name) / name.replace("/", "_"))
                self.assertEqual(len(transport.calls), 1)

    def test_completed_install_is_offline_checked_against_validator_state(self):
        transport = ScriptedTransport(*self.success_responses())
        self.install(transport)

        transport = ScriptedTransport(self.manifest_response())
        result = self.install(transport)
        self.assertTrue(result.already_complete)
        self.assertEqual(result.downloaded_files, 0)
        self.assertEqual(len(transport.calls), 1)

        (self.root / "model.safetensors").write_bytes(b"WEIGHTS")
        with self.assertRaisesRegex(QpackHTTPError, "artifact changed"):
            self.install(ScriptedTransport(self.manifest_response()))

    def test_generic_digest_on_resumed_range_is_not_a_full_object_checksum(self):
        value = self.files["model.safetensors"]
        first = self.artifact_response("model.safetensors", body=value[:3])
        first.headers = Headers([
            ("Content-Length", str(len(value))),
            ("ETag", '"model.safetensors"'),
            ("Digest", "sha-256=:range-one:"),
        ])
        second = self.artifact_response("model.safetensors", offset=3)
        second.headers = Headers([
            ("Content-Length", str(len(value) - 3)),
            ("Content-Range", f"bytes 3-{len(value) - 1}/{len(value)}"),
            ("ETag", '"model.safetensors"'),
            ("Digest", "sha-256=:range-two:"),
        ])
        transport = ScriptedTransport(
            self.manifest_response(), self.config_response(), first, second,
            self.artifact_response("packed_experts/layout.json"))
        self.install(transport, retries=1)
        state = json.loads((self.root / STATE_NAME).read_text())
        self.assertIsNone(state["files"]["model.safetensors"]["sha256"])
        self.assertEqual(state["files"]["model.safetensors"]["local_sha256"],
                         hashlib.sha256(value).hexdigest())

    def test_lost_http_state_rejects_an_already_committed_artifact(self):
        short_layout = self.artifact_response(
            "packed_experts/layout.json",
            body=self.files["packed_experts/layout.json"][:2])
        short_layout.headers = Headers([
            ("Content-Length", str(len(self.files["packed_experts/layout.json"]))),
            ("ETag", '"layout"'),
        ])
        with self.assertRaisesRegex(QpackHTTPError, "short HTTP body"):
            self.install(ScriptedTransport(
                self.manifest_response(), self.config_response(),
                self.artifact_response("model.safetensors"), short_layout))
        self.assertTrue((self.root / "model.safetensors").is_file())
        (self.root / STATE_NAME).unlink()

        with self.assertRaisesRegex(QpackHTTPError, "no validator"):
            self.install(ScriptedTransport(
                self.manifest_response(), self.config_response()))
        self.assertFalse((self.root / "manifest.json").exists())

    def test_revision_resolution_retries_transient_status(self):
        api_body = json.dumps({"sha": COMMIT}).encode()
        retry = Response(503, b"", [("Content-Length", "0")])
        api = Response(200, api_body,
                       [("Content-Length", str(len(api_body)))])
        transport = ScriptedTransport(retry, api, *self.success_responses())
        self.install(transport, revision="main", retries=1)
        self.assertEqual(len([call for call in transport.calls
                             if "/api/models/" in call["url"]]), 2)

    def test_endpoint_must_be_an_https_origin(self):
        for endpoint in ("http://hub.example", "https://hub.example/path",
                         "https://user@hub.example"):
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(
                    QpackHTTPError, "HTTPS origin"):
                install_hf_qpack(
                    "owner/model", COMMIT, self.root, endpoint=endpoint,
                    transport=ScriptedTransport(), optional_aux=())


class SafeRedirectHandlerTest(unittest.TestCase):
    def setUp(self):
        self.handler = SafeRedirectHandler()

    @staticmethod
    def request():
        return Request("https://hub.example/object",
                       headers={"Authorization": "Bearer secret"})

    def test_cross_origin_redirect_strips_authorization(self):
        redirected = self.handler.redirect_request(
            self.request(), None, 302, "Found", {},
            "https://cdn.example/object")
        self.assertIsNone(redirected.get_header("Authorization"))

    def test_same_origin_redirect_keeps_authorization(self):
        redirected = self.handler.redirect_request(
            self.request(), None, 302, "Found", {},
            "https://hub.example/elsewhere")
        self.assertEqual(redirected.get_header("Authorization"), "Bearer secret")

    def test_redirect_rejects_downgrade_and_userinfo(self):
        for url in ("http://hub.example/object",
                    "https://user@cdn.example/object"):
            with self.subTest(url=url), self.assertRaisesRegex(
                    QpackHTTPError, "unsafe redirect"):
                self.handler.redirect_request(
                    self.request(), None, 302, "Found", {}, url)


if __name__ == "__main__":
    unittest.main()
