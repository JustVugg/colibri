#!/usr/bin/env python3
"""Install a commit-pinned Hugging Face qpack without staging raw weights."""

from __future__ import annotations

import argparse
import errno
import hashlib
import http.client
import json
import os
import re
import stat
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

try:
    from tools.qpack_install_policy import (
        MAX_FILE_SIZE,
        MAX_FILES,
        MAX_MANIFEST,
        QpackFile,
        QpackInstallError,
        _decode_json,
        _validate_name,
        begin_install,
        commit_file,
        parse_manifest,
        publish_manifest,
        resume_decision,
        validate_delivery,
    )
except ModuleNotFoundError:  # Direct execution from c/tools.
    from qpack_install_policy import (
        MAX_FILE_SIZE,
        MAX_FILES,
        MAX_MANIFEST,
        QpackFile,
        QpackInstallError,
        _decode_json,
        _validate_name,
        begin_install,
        commit_file,
        parse_manifest,
        publish_manifest,
        resume_decision,
        validate_delivery,
    )


STATE_SCHEMA = "colibri.qpack-http.v1"
STATE_NAME = ".qpack-http.json"
STATE_TEMP_NAME = "..qpack-http.json.part"
HASHES_NAME = "hashes.json"
CHUNK_SIZE = 4 * 1024 * 1024
MAX_AUX_SIZE = 128 * 1024 * 1024
MAX_API_SIZE = 1024 * 1024
MAX_HASHES_SIZE = 16 * 1024 * 1024
MAX_HASH_ENTRIES = MAX_FILES + 16
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
CONTENT_RANGE_RE = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

REQUIRED_AUX = ("config.json",)
OPTIONAL_AUX = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


class QpackHTTPError(RuntimeError):
    pass


class RetryableTransferError(QpackHTTPError):
    pass


class HTTPStatusFailure(QpackHTTPError):
    def __init__(self, url: str, status: int):
        super().__init__(f"HTTP {status} for {_safe_url(url)}")
        self.url = url
        self.status = status


@dataclass(frozen=True)
class ResolvedSource:
    repo: str
    commit: str | None
    endpoint: str
    base_url: str
    source_id: str
    headers: tuple[tuple[str, str], ...]
    repo_commit: str | None = None
    resume_header: str = "If-Range"
    restart_changed_resume: bool = False

    def url(self, name: str) -> str:
        encoded = "/".join(quote(component, safe="") for component in name.split("/"))
        return f"{self.base_url}/{encoded}"


@dataclass(frozen=True)
class InstallResult:
    output_dir: Path
    commit: str | None
    downloaded_files: int
    downloaded_bytes: int
    resumed_files: int
    already_complete: bool
    source_id: str = ""


@dataclass(frozen=True)
class HashEntry:
    sha256: str
    size: int | None = None


def _safe_url(url: str) -> str:
    split = urlsplit(url)
    host = split.hostname or ""
    port = f":{split.port}" if split.port else ""
    return f"{split.scheme}://{host}{port}{split.path}"


def _origin(url: str):
    split = urlsplit(url)
    port = split.port
    if port is None:
        port = 443 if split.scheme == "https" else 80
    return split.scheme.lower(), (split.hostname or "").lower(), port


class SafeRedirectHandler(HTTPRedirectHandler):
    """Keep credentials on-origin and never follow a downgrade or userinfo URL."""

    SENSITIVE_HEADERS = (
        "Authorization",
        "Cookie",
        "Proxy-Authorization",
        "X-Api-Key",
    )

    def redirect_request(self, request, fp, code, msg, headers, new_url):
        target = urljoin(request.full_url, new_url)
        split = urlsplit(target)
        if split.scheme.lower() != "https" or split.username or split.password:
            raise QpackHTTPError(f"unsafe redirect to {_safe_url(target)}")
        redirected = super().redirect_request(request, fp, code, msg, headers, target)
        if redirected is not None and _origin(request.full_url) != _origin(target):
            for name in self.SENSITIVE_HEADERS:
                redirected.remove_header(name)
                for existing in list(redirected.unredirected_hdrs):
                    if existing.lower() == name.lower():
                        redirected.unredirected_hdrs.pop(existing, None)
        return redirected


class URLLibTransport:
    def __init__(self):
        self.opener = build_opener(SafeRedirectHandler())

    def request(self, method: str, url: str, headers, timeout: float):
        request = Request(url, headers=dict(headers), method=method)
        try:
            return self.opener.open(request, timeout=timeout)
        except HTTPError as error:
            return error
        except URLError as error:
            raise RetryableTransferError(
                f"request failed for {_safe_url(url)}: {error.reason}") from error


def _header_values(headers, name: str):
    if hasattr(headers, "get_all"):
        return list(headers.get_all(name, []))
    values = []
    for key, value in headers.items():
        if key.lower() != name.lower():
            continue
        if isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return values


def _single_header(headers, name: str, required=False):
    values = _header_values(headers, name)
    if len(values) > 1:
        raise QpackHTTPError(f"duplicate {name} response header")
    if required and not values:
        raise QpackHTTPError(f"missing {name} response header")
    return values[0].strip() if values else None


def _status(response) -> int:
    value = getattr(response, "status", None)
    if value is None and hasattr(response, "getcode"):
        value = response.getcode()
    return int(value or 0)


def _strong_etag(headers):
    value = _single_header(headers, "ETag")
    if not value or value[:2].lower() == "w/":
        return None
    if len(value) < 2 or value[0] != '"' or value[-1] != '"':
        return None
    return value


def _checksum_sha256(headers):
    direct = _single_header(headers, "X-Checksum-Sha256")
    if not direct:
        return None
    direct = direct.strip('"')
    if not SHA256_RE.fullmatch(direct):
        raise QpackHTTPError("invalid X-Checksum-Sha256 response header")
    return direct.lower()


def _validate_mirror_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url or len(base_url) > 8192:
        raise QpackHTTPError("mirror base URL is invalid")
    try:
        split = urlsplit(base_url)
        _ = split.port
    except ValueError as error:
        raise QpackHTTPError("mirror base URL is invalid") from error
    if (split.scheme.lower() != "https" or not split.hostname
            or split.username or split.password or split.query or split.fragment):
        raise QpackHTTPError(
            "mirror base URL must be HTTPS without userinfo, query, or fragment")
    try:
        decoded_path = unquote(split.path, errors="strict")
    except UnicodeError as error:
        raise QpackHTTPError("mirror base URL path is invalid") from error
    components = decoded_path.split("/")
    if ("\\" in decoded_path or any(ord(character) < 32 for character in decoded_path)
            or any(part in (".", "..") for part in components)
            or any(part == "" for part in components[1:-1])):
        raise QpackHTTPError("mirror base URL path is unsafe")
    return base_url.rstrip("/")


def _hash_entry_name(name: str) -> str:
    if name == "manifest.json":
        return name
    try:
        return _validate_name(name)
    except QpackInstallError as error:
        raise QpackHTTPError(str(error)) from error


def _semantic_hashes_digest(entries: dict[str, HashEntry]) -> str:
    digest = hashlib.sha256()
    for name in sorted(entries):
        encoded = name.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(entries[name].sha256))
    return digest.hexdigest()


def parse_hashes_index(raw: bytes) -> tuple[dict[str, HashEntry], str]:
    """Strictly parse Swiftlet's legacy index or qpack.hashes.v1."""
    try:
        value = _decode_json(raw, HASHES_NAME, MAX_HASHES_SIZE)
    except QpackInstallError as error:
        raise QpackHTTPError(str(error)) from error
    if not isinstance(value, dict):
        raise QpackHTTPError("hashes.json must be an object")
    schema = value.get("schema")
    if schema is None:
        if set(value) != {"files"}:
            raise QpackHTTPError("legacy hashes.json has unsupported fields")
    elif schema == "qpack.hashes.v1":
        if set(value) != {"schema", "files"}:
            raise QpackHTTPError("qpack.hashes.v1 has unsupported fields")
    else:
        raise QpackHTTPError("unsupported hashes.json schema")
    files = value.get("files")
    if (not isinstance(files, dict) or not files
            or len(files) > MAX_HASH_ENTRIES):
        raise QpackHTTPError("hashes.json files table is invalid")

    entries = {}
    normalized_names = set()
    for raw_name, raw_entry in files.items():
        name = _hash_entry_name(raw_name)
        if name == HASHES_NAME:
            raise QpackHTTPError("hashes.json cannot list itself")
        normalized = name.casefold()
        if normalized in normalized_names:
            raise QpackHTTPError(f"case-colliding hashes.json path: {name}")
        normalized_names.add(normalized)
        if schema is None:
            sha256 = raw_entry
            size = None
        else:
            if not isinstance(raw_entry, dict) or set(raw_entry) != {"size", "sha256"}:
                raise QpackHTTPError(f"invalid qpack.hashes.v1 entry: {name}")
            sha256 = raw_entry.get("sha256")
            size = raw_entry.get("size")
            if (isinstance(size, bool) or not isinstance(size, int)
                    or size < 0 or size > MAX_FILE_SIZE):
                raise QpackHTTPError(f"invalid hashes.json size for {name}")
        if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256):
            raise QpackHTTPError(f"invalid hashes.json SHA-256 for {name}")
        entries[name] = HashEntry(sha256.lower(), size)

    sorted_names = sorted(normalized_names)
    occupied = set(sorted_names)
    for name in sorted_names:
        components = name.split("/")
        for count in range(1, len(components)):
            if "/".join(components[:count]) in occupied:
                raise QpackHTTPError("hashes.json has a file/parent path collision")
    return entries, _semantic_hashes_digest(entries)


def _read_bounded(response, maximum: int, description: str) -> bytes:
    data = response.read(maximum + 1)
    if len(data) > maximum:
        raise QpackHTTPError(f"{description} exceeds {maximum} bytes")
    return data


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in (errno.EBADF, errno.EINVAL, errno.ENOTSUP):
                raise
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.part")
    if path.is_symlink() or temporary.is_symlink():
        raise QpackHTTPError(f"refusing symlink auxiliary path: {path}")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload) -> None:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    _atomic_bytes(path, raw)


def _state_path(root: Path) -> Path:
    return root / STATE_NAME


def _new_state(session, manifest_etag):
    return {
        "schema": STATE_SCHEMA,
        "source_id": session.plan.source_id,
        "manifest_sha256": session.plan.manifest_sha256,
        "manifest_etag": manifest_etag,
        "files": {},
        "aux": {},
    }


def _load_state(session, manifest_etag, require_existing=False):
    path = _state_path(session.root)
    if path.is_symlink():
        raise QpackHTTPError("qpack HTTP state is a symlink")
    if not path.exists():
        if require_existing:
            raise QpackHTTPError("completed HTTP install has no validator state")
        state = _new_state(session, manifest_etag)
        _atomic_json(path, state)
        return state
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QpackHTTPError(f"invalid qpack HTTP state: {error}") from error
    expected = (STATE_SCHEMA, session.plan.source_id, session.plan.manifest_sha256)
    actual = (state.get("schema"), state.get("source_id"),
              state.get("manifest_sha256")) if isinstance(state, dict) else ()
    if actual != expected or not isinstance(state.get("files"), dict) \
            or not isinstance(state.get("aux"), dict):
        raise QpackHTTPError("qpack HTTP state belongs to another source or manifest")
    stored_etag = state.get("manifest_etag")
    if stored_etag and manifest_etag and stored_etag != manifest_etag:
        raise QpackHTTPError("qpack manifest ETag changed for an immutable source")
    if stored_etag is None and manifest_etag is not None:
        state["manifest_etag"] = manifest_etag
        _atomic_json(path, state)
    return state


def _save_state(session, state) -> None:
    _atomic_json(_state_path(session.root), state)


def _declared_auxiliary_names(plan):
    declared = {file.name for file in plan.files}
    return {name for name in (*REQUIRED_AUX, *OPTIONAL_AUX) if name in declared}


def _check_adapter_namespace(plan, reserve_hashes=False) -> None:
    reserved = {
        STATE_NAME.casefold(),
        STATE_TEMP_NAME.casefold(),
    }
    if reserve_hashes:
        reserved.update((
            HASHES_NAME.casefold(),
            f".{HASHES_NAME}.part".casefold(),
        ))
    declared_aux = _declared_auxiliary_names(plan)
    for name in (*REQUIRED_AUX, *OPTIONAL_AUX):
        if name in declared_aux:
            continue
        reserved.add(name.casefold())
        reserved.add((f".{name}.part").casefold())
    for file in plan.files:
        if file.name in declared_aux and file.size > MAX_AUX_SIZE:
            raise QpackHTTPError(
                f"qpack auxiliary file is too large: {file.name}")
        name = file.name.casefold()
        if (name in reserved
                or any(name.startswith(path + "/") for path in reserved)
                or any(path.startswith(name + "/") for path in reserved)):
            raise QpackHTTPError(
                f"qpack manifest collides with HTTP installer path: {file.name}")


def _request_headers(source: ResolvedSource, extra=()):
    values = dict(source.headers)
    values["Accept-Encoding"] = "identity"
    values["User-Agent"] = "colibri-qpack-install/1.0"
    values.update(dict(extra))
    return values


def _response_metadata(response, source: ResolvedSource, expected_size=None):
    headers = response.headers
    encoding = _single_header(headers, "Content-Encoding")
    if encoding and encoding.lower() != "identity":
        raise QpackHTTPError(f"unsupported content encoding: {encoding}")
    repo_commit = _single_header(headers, "X-Repo-Commit")
    if (repo_commit and source.repo_commit is not None
            and repo_commit.lower() != source.repo_commit):
        raise QpackHTTPError("response belongs to a different repository commit")
    linked_size = _single_header(headers, "X-Linked-Size")
    if linked_size is not None and expected_size is not None:
        try:
            linked_size_value = int(linked_size)
        except ValueError as error:
            raise QpackHTTPError("invalid X-Linked-Size response header") from error
        if linked_size_value != expected_size:
            raise QpackHTTPError("response linked size differs from qpack manifest")
    return _strong_etag(headers), _checksum_sha256(headers)


def _resolve_source(repo: str, revision: str, endpoint: str, token: str | None,
                    transport, timeout: float, retries: int, sleep) -> ResolvedSource:
    if (not REPO_RE.fullmatch(repo)
            or any(part in (".", "..") for part in repo.split("/"))):
        raise QpackHTTPError("repository must be OWNER/REPO")
    if not isinstance(revision, str) or not revision or len(revision) > 4096:
        raise QpackHTTPError("revision must be a non-empty commit or ref")
    endpoint = endpoint.rstrip("/")
    split = urlsplit(endpoint)
    if split.scheme != "https" or not split.hostname or split.username or split.password \
            or split.path not in ("", "/") or split.query or split.fragment:
        raise QpackHTTPError("Hugging Face endpoint must be an HTTPS origin")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    headers["Accept-Encoding"] = "identity"
    headers["User-Agent"] = "colibri-qpack-install/1.0"
    if COMMIT_RE.fullmatch(revision):
        commit = revision.lower()
    else:
        api = (f"{endpoint}/api/models/{quote(repo, safe='/')}/revision/"
               f"{quote(revision, safe='')}?expand=sha")
        for attempt in range(retries + 1):
            try:
                response = transport.request("GET", api, headers, timeout)
                with closing(response):
                    status = _status(response)
                    if status in RETRYABLE_STATUS:
                        raise HTTPStatusFailure(api, status)
                    if status != 200:
                        raise HTTPStatusFailure(api, status)
                    try:
                        raw = _read_bounded(
                            response, MAX_API_SIZE,
                            "Hugging Face revision metadata")
                    except (OSError, http.client.IncompleteRead) as error:
                        raise RetryableTransferError(
                            "interrupted Hugging Face revision metadata") from error
                break
            except (RetryableTransferError, HTTPStatusFailure) as error:
                retryable = (isinstance(error, RetryableTransferError)
                             or error.status in RETRYABLE_STATUS)
                if not retryable or attempt >= retries:
                    raise
                sleep(min(30, 2 ** attempt))
        try:
            value = json.loads(raw)
            commit = value["sha"].lower()
        except (UnicodeError, json.JSONDecodeError, KeyError, AttributeError) as error:
            raise QpackHTTPError("Hugging Face revision metadata has no commit SHA") from error
        if not COMMIT_RE.fullmatch(commit):
            raise QpackHTTPError("Hugging Face returned an invalid commit SHA")
    encoded_repo = "/".join(quote(part, safe="") for part in repo.split("/"))
    base_url = f"{endpoint}/{encoded_repo}/resolve/{commit}"
    source_id = f"hf:{endpoint}/models/{repo}@{commit}"
    return ResolvedSource(repo, commit, endpoint, base_url, source_id,
                          tuple(headers.items()), repo_commit=commit)


def _mirror_source(base_url: str, source_id: str,
                   authorization: str | None) -> ResolvedSource:
    base_url = _validate_mirror_base_url(base_url)
    headers = ()
    if authorization is not None:
        if (not isinstance(authorization, str) or not authorization.strip()
                or "\r" in authorization or "\n" in authorization):
            raise QpackHTTPError("mirror authorization value is invalid")
        headers = (("Authorization", authorization),)
    split = urlsplit(base_url)
    origin = f"{split.scheme}://{split.netloc}"
    return ResolvedSource(
        repo="", commit=None, endpoint=origin, base_url=base_url,
        source_id=source_id, headers=headers, resume_header="If-Match",
        restart_changed_resume=True)


def _fetch_small(source, name, transport, timeout, maximum, required,
                 retries, sleep, expected_checksum=None, expected_size=None,
                 request_headers=()):
    url = source.url(name)
    for attempt in range(retries + 1):
        try:
            response = transport.request(
                "GET", url, _request_headers(source, request_headers), timeout)
            with closing(response):
                status = _status(response)
                if status == 404 and not required:
                    return None, None
                if status in RETRYABLE_STATUS:
                    raise HTTPStatusFailure(url, status)
                if status != 200:
                    raise HTTPStatusFailure(url, status)
                etag, checksum = _response_metadata(response, source)
                if (expected_checksum is not None and checksum is not None
                        and checksum != expected_checksum):
                    raise QpackHTTPError(
                        f"server SHA-256 disagrees with hashes.json for {name}")
                try:
                    raw = _read_bounded(response, maximum, name)
                except (OSError, http.client.IncompleteRead) as error:
                    raise RetryableTransferError(
                        f"interrupted HTTP body for {name}") from error
                content_length = _single_header(response.headers, "Content-Length")
                if content_length is not None:
                    try:
                        expected = int(content_length)
                    except ValueError as error:
                        raise QpackHTTPError(
                            "invalid Content-Length response header") from error
                    if expected != len(raw):
                        raise RetryableTransferError(f"short HTTP body for {name}")
                if expected_size is not None and len(raw) != expected_size:
                    raise QpackHTTPError(f"hashes.json size mismatch for {name}")
                digest = hashlib.sha256(raw).hexdigest()
                if expected_checksum is not None and digest != expected_checksum:
                    raise QpackHTTPError(f"SHA-256 mismatch for {name}")
                if checksum and digest != checksum:
                    raise QpackHTTPError(f"SHA-256 mismatch for {name}")
                return raw, etag
        except (RetryableTransferError, HTTPStatusFailure) as error:
            retryable = (isinstance(error, RetryableTransferError)
                         or error.status in RETRYABLE_STATUS)
            if not retryable or attempt >= retries:
                raise
            sleep(min(30, 2 ** attempt))
    raise QpackHTTPError(f"retry budget exhausted for {name}")


def _verify_or_fetch_aux(session, state, source, name, required, transport,
                         timeout, retries, sleep, completed,
                         expected_checksum=None, expected_size=None):
    target = session.root / name
    recorded = state["aux"].get(name)
    if recorded is not None:
        if (not isinstance(recorded, dict)
                or not isinstance(recorded.get("sha256"), str)
                or not SHA256_RE.fullmatch(recorded["sha256"])):
            raise QpackHTTPError(f"invalid validator state for {name}")
        if (expected_checksum is not None
                and recorded["sha256"].lower() != expected_checksum):
            raise QpackHTTPError(f"validator state disagrees with hashes.json for {name}")
        if target.is_symlink() or not target.is_file():
            raise QpackHTTPError(f"validated auxiliary file is missing: {name}")
        raw = target.read_bytes()
        if expected_size is not None and len(raw) != expected_size:
            raise QpackHTTPError(f"validated auxiliary file has wrong size: {name}")
        if hashlib.sha256(raw).hexdigest() != recorded["sha256"].lower():
            raise QpackHTTPError(f"validated auxiliary file changed: {name}")
        return False, 0
    if completed:
        if required:
            raise QpackHTTPError(f"completed qpack has no validator for {name}")
        return False, 0
    raw, etag = _fetch_small(source, name, transport, timeout, MAX_AUX_SIZE,
                             required, retries, sleep, expected_checksum,
                             expected_size)
    if raw is None:
        return False, 0
    if name == "config.json":
        try:
            config = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise QpackHTTPError("config.json is not valid JSON") from error
        if not isinstance(config, dict) or not config:
            raise QpackHTTPError("config.json must be a non-empty object")
    _atomic_bytes(target, raw)
    state["aux"][name] = {
        "sha256": expected_checksum or hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "etag": etag,
    }
    _save_state(session, state)
    return True, len(raw)


def _verify_or_store_prefetched_aux(session, state, name, raw, completed):
    target = session.root / name
    digest = hashlib.sha256(raw).hexdigest()
    recorded = state["aux"].get(name)
    if recorded is not None:
        if (not isinstance(recorded, dict) or recorded.get("sha256") != digest
                or recorded.get("size") != len(raw)):
            raise QpackHTTPError(f"validator state changed for {name}")
        if target.is_symlink() or not target.is_file() \
                or target.read_bytes() != raw:
            raise QpackHTTPError(f"validated auxiliary file changed: {name}")
        return
    if completed:
        raise QpackHTTPError(f"completed qpack has no validator for {name}")
    _atomic_bytes(target, raw)
    state["aux"][name] = {
        "sha256": digest,
        "size": len(raw),
        "etag": None,
    }
    _save_state(session, state)


def _validate_config(path: Path) -> None:
    try:
        raw = path.read_bytes()
        if not raw or len(raw) > MAX_AUX_SIZE:
            raise QpackHTTPError("config.json has an invalid size")
        config = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QpackHTTPError("config.json is not valid JSON") from error
    if not isinstance(config, dict) or not config:
        raise QpackHTTPError("config.json must be a non-empty object")


def _open_partial(decision, file: QpackFile):
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if decision.action == "restart":
        flags |= os.O_CREAT | os.O_TRUNC
    descriptor = os.open(decision.partial_path, flags, 0o644)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise QpackHTTPError(f"qpack partial is not a regular file: {file.name}")
    expected = 0 if decision.action == "restart" else decision.offset
    if metadata.st_size != expected:
        os.close(descriptor)
        raise QpackHTTPError(f"qpack partial changed while opening: {file.name}")
    return descriptor


def _write_all(descriptor, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise QpackHTTPError("short write to qpack partial")
        offset += written


def _bind_response_validator(session, state, file, etag, checksum,
                             expected_checksum=None):
    if (expected_checksum is not None and checksum is not None
            and checksum != expected_checksum):
        raise QpackHTTPError(
            f"server SHA-256 disagrees with hashes.json for {file.name}")
    authoritative_checksum = expected_checksum or checksum
    record = state["files"].get(file.name)
    if record is None:
        record = {
            "size": file.size,
            "etag": etag,
            "sha256": authoritative_checksum,
            "local_sha256": None,
        }
        state["files"][file.name] = record
        _save_state(session, state)
        return record
    if not isinstance(record, dict) or record.get("size") != file.size:
        raise QpackHTTPError(f"invalid HTTP validator state for {file.name}")
    stored_etag = record.get("etag")
    stored_checksum = record.get("sha256")
    if stored_etag is not None and etag != stored_etag:
        raise QpackHTTPError(f"strong ETag changed or disappeared for {file.name}")
    if expected_checksum is not None:
        if stored_checksum != expected_checksum:
            raise QpackHTTPError(
                f"validator state disagrees with hashes.json for {file.name}")
    elif stored_checksum is not None and checksum != stored_checksum:
        raise QpackHTTPError(
            f"SHA-256 validator changed or disappeared for {file.name}")
    changed = False
    if stored_etag is None and etag is not None:
        record["etag"] = etag
        changed = True
    if stored_checksum is None and authoritative_checksum is not None:
        record["sha256"] = authoritative_checksum
        changed = True
    if changed:
        _save_state(session, state)
    return record


def _verify_completed_file_state(session, state, file, chunk_size,
                                 expected_checksum=None) -> None:
    record = state["files"].get(file.name)
    if not isinstance(record, dict) or record.get("size") != file.size:
        raise QpackHTTPError(f"completed qpack has no validator for {file.name}")
    etag = record.get("etag")
    checksum = record.get("sha256")
    local_checksum = record.get("local_sha256")
    if etag is not None and (
            not isinstance(etag, str) or _strong_etag({"ETag": etag}) != etag):
        raise QpackHTTPError(f"invalid stored ETag for {file.name}")
    if checksum is not None and (
            not isinstance(checksum, str) or not SHA256_RE.fullmatch(checksum)):
        raise QpackHTTPError(f"invalid stored SHA-256 for {file.name}")
    if expected_checksum is not None and checksum != expected_checksum:
        raise QpackHTTPError(
            f"validator state disagrees with hashes.json for {file.name}")
    if (not isinstance(local_checksum, str)
            or not SHA256_RE.fullmatch(local_checksum)):
        raise QpackHTTPError(f"completed qpack has no local digest for {file.name}")
    digest = hashlib.sha256()
    with (session.root / file.name).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    actual = digest.hexdigest()
    if checksum is not None and actual != checksum.lower():
        raise QpackHTTPError(f"validated qpack artifact changed: {file.name}")
    if actual != local_checksum.lower():
        raise QpackHTTPError(f"validated qpack artifact changed: {file.name}")


def _validate_range_headers(response, file, offset):
    status = _status(response)
    remaining = file.size - offset
    if offset > 0 and status != 206:
        raise QpackHTTPError(f"resumed request for {file.name} did not return HTTP 206")
    if offset == 0 and status not in (200, 206):
        raise HTTPStatusFailure(file.name, status)
    content_length = _single_header(response.headers, "Content-Length", required=True)
    try:
        content_length = int(content_length)
    except ValueError as error:
        raise QpackHTTPError("invalid Content-Length response header") from error
    if content_length != remaining:
        raise QpackHTTPError(f"Content-Length mismatch for {file.name}")
    content_range = _single_header(response.headers, "Content-Range",
                                   required=status == 206)
    if status == 200:
        if content_range is not None:
            raise QpackHTTPError("HTTP 200 response unexpectedly has Content-Range")
        return remaining
    match = CONTENT_RANGE_RE.fullmatch(content_range or "")
    if match is None:
        raise QpackHTTPError(f"invalid Content-Range for {file.name}")
    start, end, total = map(int, match.groups())
    if start != offset or end != file.size - 1 or total != file.size:
        raise QpackHTTPError(f"Content-Range mismatch for {file.name}")
    if end - start + 1 != remaining:
        raise QpackHTTPError(f"Content-Range length mismatch for {file.name}")
    return remaining


def _download_file(session, state, source, file, transport, timeout,
                   retries, sleep, chunk_size, expected_checksum=None):
    decision = resume_decision(session, file)
    if decision.action == "complete":
        _verify_completed_file_state(
            session, state, file, chunk_size, expected_checksum)
        return False, 0, False
    resumed = decision.action == "resume" and decision.offset > 0
    restarted_from_zero = False
    descriptor = _open_partial(decision, file)
    try:
        cursor = decision.offset if decision.action == "resume" else 0
        attempts = 0
        while cursor < file.size:
            record = state["files"].get(file.name)
            if record is not None and not isinstance(record, dict):
                raise QpackHTTPError(
                    f"invalid HTTP validator state for {file.name}")
            record = record or {}
            headers = {"Range": f"bytes={cursor}-{file.size - 1}"}
            if cursor > 0 and record.get("etag"):
                headers[source.resume_header] = record["etag"]
            url = source.url(file.name)
            try:
                response = transport.request(
                    "GET", url, _request_headers(source, headers.items()), timeout)
                with closing(response):
                    status = _status(response)
                    if status in RETRYABLE_STATUS:
                        raise HTTPStatusFailure(url, status)
                    if (source.restart_changed_resume and cursor > 0
                            and status in (200, 412)):
                        os.ftruncate(descriptor, 0)
                        os.fsync(descriptor)
                        record["etag"] = None
                        record["local_sha256"] = None
                        _save_state(session, state)
                        cursor = 0
                        restarted_from_zero = True
                        continue
                    response_bytes = _validate_range_headers(response, file, cursor)
                    etag, checksum = _response_metadata(response, source, file.size)
                    if (source.restart_changed_resume and cursor > 0
                            and record.get("etag") is not None
                            and etag != record["etag"]):
                        os.ftruncate(descriptor, 0)
                        os.fsync(descriptor)
                        record["etag"] = None
                        record["local_sha256"] = None
                        _save_state(session, state)
                        cursor = 0
                        restarted_from_zero = True
                        continue
                    _bind_response_validator(
                        session, state, file, etag, checksum,
                        expected_checksum)
                    os.lseek(descriptor, cursor, os.SEEK_SET)
                    delivered = 0
                    request_offset = cursor
                    while True:
                        try:
                            chunk = response.read(chunk_size)
                        except http.client.IncompleteRead as error:
                            chunk = error.partial
                            if chunk:
                                if delivered + len(chunk) > response_bytes:
                                    os.ftruncate(descriptor, request_offset)
                                    os.fsync(descriptor)
                                    raise QpackHTTPError(
                                        f"HTTP body overrun for {file.name}")
                                _write_all(descriptor, chunk)
                                delivered += len(chunk)
                            raise RetryableTransferError(
                                f"interrupted HTTP body for {file.name}") from error
                        except OSError as error:
                            raise RetryableTransferError(
                                f"interrupted HTTP body for {file.name}") from error
                        if not chunk:
                            break
                        if delivered + len(chunk) > response_bytes:
                            os.ftruncate(descriptor, request_offset)
                            os.fsync(descriptor)
                            raise QpackHTTPError(f"HTTP body overrun for {file.name}")
                        validate_delivery(file, cursor + delivered,
                                          cursor + delivered, len(chunk))
                        _write_all(descriptor, chunk)
                        delivered += len(chunk)
                    os.fsync(descriptor)
                    if delivered != response_bytes:
                        cursor += delivered
                        raise RetryableTransferError(
                            f"short HTTP body for {file.name}")
                    cursor += delivered
                    validate_delivery(file, request_offset, request_offset,
                                      delivered, complete=cursor == file.size)
                    attempts = 0
            except (RetryableTransferError, HTTPStatusFailure) as error:
                retryable = (isinstance(error, RetryableTransferError)
                             or error.status in RETRYABLE_STATUS)
                if not retryable or attempts >= retries:
                    raise
                attempts += 1
                sleep(min(30, 2 ** (attempts - 1)))
                cursor = os.fstat(descriptor).st_size
                resumed = resumed or cursor > 0
        record = state["files"].get(file.name)
        if not isinstance(record, dict):
            raise QpackHTTPError(f"invalid HTTP validator state for {file.name}")
        checksum = record.get("sha256")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, chunk_size)
            if not chunk:
                break
            digest.update(chunk)
        local_checksum = digest.hexdigest()
        if checksum and local_checksum != checksum:
            os.ftruncate(descriptor, 0)
            os.fsync(descriptor)
            raise QpackHTTPError(f"SHA-256 mismatch for {file.name}")
        stored_local = record.get("local_sha256")
        if stored_local is not None and stored_local != local_checksum:
            os.ftruncate(descriptor, 0)
            os.fsync(descriptor)
            raise QpackHTTPError(f"local SHA-256 changed for {file.name}")
        if stored_local is None:
            record["local_sha256"] = local_checksum
            _save_state(session, state)
    finally:
        os.close(descriptor)
    commit_file(session, file)
    downloaded = file.size if restarted_from_zero else file.size - decision.offset
    return True, downloaded, resumed


def _validate_hash_coverage(plan, manifest_raw, entries) -> None:
    required = {"manifest.json", "config.json", *(file.name for file in plan.files)}
    missing = sorted(required - entries.keys())
    if missing:
        raise QpackHTTPError(
            f"hashes.json does not cover required file: {missing[0]}")
    manifest_size = entries["manifest.json"].size
    if manifest_size is not None and manifest_size != len(manifest_raw):
        raise QpackHTTPError("hashes.json size mismatch for manifest.json")
    for file in plan.files:
        indexed_size = entries[file.name].size
        if indexed_size is not None and indexed_size != file.size:
            raise QpackHTTPError(f"hashes.json size mismatch for {file.name}")


def _install_resolved_qpack(source, output_dir, manifest_raw, manifest_etag,
                            transport, retries, timeout, chunk_size,
                            optional_aux, sleep, log, expected_hashes=None,
                            persisted_hashes=None) -> InstallResult:
    try:
        plan = parse_manifest(manifest_raw, source.source_id)
    except QpackInstallError as error:
        raise QpackHTTPError(str(error)) from error
    _check_adapter_namespace(plan, reserve_hashes=persisted_hashes is not None)
    optional_aux = tuple(optional_aux)
    for name in optional_aux:
        if name not in OPTIONAL_AUX:
            raise QpackHTTPError(f"unsupported optional auxiliary path: {name}")
    if expected_hashes is not None:
        _validate_hash_coverage(plan, manifest_raw, expected_hashes)
    declared_aux = _declared_auxiliary_names(plan)
    files_by_name = {file.name: file for file in plan.files}

    downloaded_files = 0
    downloaded_bytes = 0
    resumed_files = 0
    with begin_install(output_dir, plan) as session:
        state = _load_state(session, manifest_etag,
                            require_existing=session.complete)
        if persisted_hashes is not None:
            _verify_or_store_prefetched_aux(
                session, state, HASHES_NAME, persisted_hashes,
                session.complete)
        if "config.json" in declared_aux and not session.complete:
            changed, count, resumed = _download_file(
                session, state, source, files_by_name["config.json"],
                transport, timeout, retries, sleep, chunk_size,
                expected_hashes["config.json"].sha256
                if expected_hashes is not None else None)
            downloaded_files += int(changed)
            downloaded_bytes += count
            resumed_files += int(resumed)
        elif "config.json" not in declared_aux:
            changed, count = _verify_or_fetch_aux(
                session, state, source, "config.json", True, transport, timeout,
                retries, sleep, session.complete,
                expected_hashes["config.json"].sha256
                if expected_hashes is not None else None,
                expected_hashes["config.json"].size
                if expected_hashes is not None else None)
            downloaded_files += int(changed)
            downloaded_bytes += count
        _validate_config(session.root / "config.json")
        if not session.complete:
            for name in sorted(declared_aux - {"config.json"}):
                changed, count, resumed = _download_file(
                    session, state, source, files_by_name[name], transport,
                    timeout, retries, sleep, chunk_size,
                    expected_hashes[name].sha256
                    if expected_hashes is not None else None)
                downloaded_files += int(changed)
                downloaded_bytes += count
                resumed_files += int(resumed)
        for name in optional_aux:
            if name in declared_aux:
                continue
            changed, count = _verify_or_fetch_aux(
                session, state, source, name, expected_hashes is not None,
                transport, timeout, retries, sleep, session.complete,
                expected_hashes[name].sha256
                if expected_hashes is not None else None,
                expected_hashes[name].size
                if expected_hashes is not None else None)
            downloaded_files += int(changed)
            downloaded_bytes += count
        if session.complete:
            for file in plan.files:
                _verify_completed_file_state(
                    session, state, file, chunk_size,
                    expected_hashes[file.name].sha256
                    if expected_hashes is not None else None)
            log(f"qpack already complete at {session.root}")
            return InstallResult(session.root, source.commit, downloaded_files,
                                 downloaded_bytes, 0, True, source.source_id)

        for file in plan.files:
            if file.name in declared_aux:
                continue
            changed, count, resumed = _download_file(
                session, state, source, file, transport, timeout,
                retries, sleep, chunk_size,
                expected_hashes[file.name].sha256
                if expected_hashes is not None else None)
            downloaded_files += int(changed)
            downloaded_bytes += count
            resumed_files += int(resumed)
            if changed:
                log(f"installed {file.name} ({file.size} bytes)")
        publish_manifest(session, manifest_raw)
        log(f"qpack complete at {session.root}")
        return InstallResult(session.root, source.commit, downloaded_files,
                             downloaded_bytes, resumed_files, False,
                             source.source_id)


def install_hf_qpack(repo: str, revision: str, output_dir,
                     *, endpoint="https://huggingface.co", token=None,
                     transport=None, retries=5, timeout=120.0,
                     chunk_size=CHUNK_SIZE, optional_aux=OPTIONAL_AUX,
                     sleep=time.sleep, log=print) -> InstallResult:
    if retries < 0 or timeout <= 0 or chunk_size <= 0:
        raise QpackHTTPError("retry, timeout, and chunk-size values are invalid")
    transport = transport or URLLibTransport()
    source = _resolve_source(repo, revision, endpoint, token,
                             transport, timeout, retries, sleep)
    manifest_raw, manifest_etag = _fetch_small(
        source, "manifest.json", transport, timeout, MAX_MANIFEST,
        True, retries, sleep)
    return _install_resolved_qpack(
        source, output_dir, manifest_raw, manifest_etag, transport, retries,
        timeout, chunk_size, optional_aux, sleep, log)


def install_mirror_qpack(base_url: str, output_dir, *, hashes_sha256=None,
                         authorization=None, transport=None, retries=5,
                         timeout=120.0, chunk_size=CHUNK_SIZE,
                         sleep=time.sleep, log=print) -> InstallResult:
    """Install a static HTTPS qpack whose hashes.json is the completion marker."""
    if retries < 0 or timeout <= 0 or chunk_size <= 0:
        raise QpackHTTPError("retry, timeout, and chunk-size values are invalid")
    if hashes_sha256 is not None:
        if not isinstance(hashes_sha256, str) or not SHA256_RE.fullmatch(hashes_sha256):
            raise QpackHTTPError("hashes_sha256 must be a 64-digit SHA-256")
        hashes_sha256 = hashes_sha256.lower()
    transport = transport or URLLibTransport()
    bootstrap = _mirror_source(base_url, "mirror:bootstrap", authorization)
    hashes_raw, _ = _fetch_small(
        bootstrap, HASHES_NAME, transport, timeout, MAX_HASHES_SIZE, True,
        retries, sleep, request_headers=(("Cache-Control", "no-cache"),))
    raw_digest = hashlib.sha256(hashes_raw).hexdigest()
    if hashes_sha256 is not None and raw_digest != hashes_sha256:
        raise QpackHTTPError("hashes.json does not match the pinned SHA-256")
    expected_hashes, _ = parse_hashes_index(hashes_raw)
    source = _mirror_source(
        bootstrap.base_url,
        f"mirror:{bootstrap.base_url}@sha256:{raw_digest}", authorization)
    manifest_entry = expected_hashes.get("manifest.json")
    if manifest_entry is None:
        raise QpackHTTPError("hashes.json does not cover required file: manifest.json")
    manifest_raw, _ = _fetch_small(
        source, "manifest.json", transport, timeout, MAX_MANIFEST, True,
        retries, sleep, manifest_entry.sha256, manifest_entry.size,
        (("Cache-Control", "no-cache"),))
    mirror_aux = tuple(
        name for name in OPTIONAL_AUX if name in expected_hashes)
    return _install_resolved_qpack(
        source, output_dir, manifest_raw, None, transport, retries, timeout,
        chunk_size, mirror_aux, sleep, log, expected_hashes,
        hashes_raw)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Hugging Face repository as OWNER/REPO")
    parser.add_argument("--revision", required=True,
                        help="commit SHA or revision resolved to a commit before download")
    parser.add_argument("--output", required=True, type=Path,
                        help="destination .qpack directory")
    parser.add_argument("--endpoint", default="https://huggingface.co",
                        help=argparse.SUPPRESS)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    try:
        result = install_hf_qpack(
            args.repo, args.revision, args.output, endpoint=args.endpoint,
            token=os.environ.get("HF_TOKEN"), retries=args.retries,
            timeout=args.timeout)
    except (OSError, QpackInstallError, QpackHTTPError) as error:
        print(f"qpack install failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "output": str(result.output_dir),
        "commit": result.commit,
        "downloaded_files": result.downloaded_files,
        "downloaded_bytes": result.downloaded_bytes,
        "resumed_files": result.resumed_files,
        "already_complete": result.already_complete,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
