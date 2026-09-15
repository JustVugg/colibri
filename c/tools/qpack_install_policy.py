#!/usr/bin/env python3
"""Resumable qpack policy with POSIX-durable manifest-last publication.

Qpack v1 records file sizes but no content digests. Callers must use an
immutable ``source_id`` and validate any ETag or checksum supplied by the
transport; this module prevents cross-revision resume, not silent media damage.

The manifest files table is the authority for every artifact it declares, and
nothing else. Swiftlet's own manifests vary in what they declare: the
Qwen3.6-35B container sizes ``packed_experts/layout.json`` and its tokenizer
files, while the production Qwen3-Next-80B container sizes only
``model.safetensors`` and the 48 layer blobs. Swiftlet's reader never consults
the files table for ``layout.json``, so a declared entry is verified and an
undeclared one is not required here; frontends that need the file fetch it as
a required auxiliary, the way they already fetch ``config.json``.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal


SCHEMA = "colibri.qpack-install.v1"
JOURNAL = ".qpack-install.json"
LOCK = ".qpack-install.lock"
MANIFEST = "manifest.json"
MAX_MANIFEST = 16 * 1024 * 1024
MAX_JOURNAL = 32 * 1024 * 1024
MAX_FILE_SIZE = (1 << 53) - 1
MAX_FILES = 8192
WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


class QpackInstallError(RuntimeError):
    pass


@dataclass(frozen=True, order=True)
class QpackFile:
    name: str
    size: int


@dataclass(frozen=True)
class QpackInstallPlan:
    source_id: str
    manifest_sha256: str
    files: tuple[QpackFile, ...]


@dataclass(frozen=True)
class ResumeDecision:
    action: Literal["complete", "resume", "restart"]
    offset: int
    partial_path: Path


@dataclass
class QpackInstallSession:
    root: Path
    plan: QpackInstallPlan
    _lock_stream: object
    complete: bool = False
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        _unlock(self._lock_stream)
        self._lock_stream.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()


def _object_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise QpackInstallError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value):
    raise QpackInstallError(f"non-finite JSON number: {value}")


def _has_nul(value) -> bool:
    if isinstance(value, str):
        return "\x00" in value
    if isinstance(value, list):
        return any(_has_nul(item) for item in value)
    if isinstance(value, dict):
        return any(_has_nul(key) or _has_nul(item) for key, item in value.items())
    return False


def _decode_json(raw: bytes, description: str, maximum=MAX_MANIFEST):
    if not isinstance(raw, (bytes, bytearray)) or not raw or len(raw) > maximum:
        raise QpackInstallError(f"{description} has an invalid size")
    try:
        text = raw.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_object_without_duplicates,
                           parse_constant=_reject_constant)
        if _has_nul(value):
            raise QpackInstallError(f"{description} contains an embedded NUL")
        return value
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise QpackInstallError(f"invalid {description}: {error}") from error


def _validate_name(name: str) -> str:
    if (not isinstance(name, str) or not name or not name.isascii()
            or "\\" in name or "\x00" in name
            or len(name.encode("utf-8")) > 4096):
        raise QpackInstallError("qpack file name is invalid")
    components = name.split("/")
    path = PurePosixPath(name)
    if (path.is_absolute() or any(part in ("", ".", "..") for part in components)
            or path.as_posix() != name):
        raise QpackInstallError(f"unsafe qpack file name: {name}")
    for component in components:
        if (component[-1] in (" ", ".")
                or len(component.encode("utf-8")) > 255
                or any(ord(character) < 32 or character in '<>:"|?*'
                       for character in component)
                or component.split(".", 1)[0].casefold() in WINDOWS_RESERVED):
            raise QpackInstallError(f"Windows-unsafe qpack file name: {name}")
    reserved = {
        MANIFEST, JOURNAL, LOCK, f".{MANIFEST}.part",
        f".{JOURNAL}.part",
    }
    if name.casefold() in {value.casefold() for value in reserved}:
        raise QpackInstallError(f"reserved qpack file name: {name}")
    return name


def _validate_file(file: QpackFile) -> None:
    _validate_name(file.name)
    if (isinstance(file.size, bool) or not isinstance(file.size, int)
            or file.size <= 0 or file.size > MAX_FILE_SIZE):
        raise QpackInstallError(f"invalid qpack file size for {file.name}")


def parse_manifest(raw: bytes, source_id: str) -> QpackInstallPlan:
    """Parse a qpack v1 manifest and bind it to an immutable source identity.

    Every declared file must carry a positive size, which the install session
    then enforces on commit and before publication. ``packed_experts/layout.json``
    is verified when declared and not required: the production Qwen3-Next-80B
    manifest omits it (see the module docstring).
    """
    if (not isinstance(source_id, str) or not source_id.strip()
            or "\x00" in source_id or len(source_id.encode("utf-8")) > 4096):
        raise QpackInstallError("source_id must identify an immutable source revision")
    manifest = _decode_json(raw, "qpack manifest")
    if not isinstance(manifest, dict):
        raise QpackInstallError("qpack manifest must be an object")
    version = manifest.get("version")
    if (manifest.get("magic") != "QPACK" or isinstance(version, bool)
            or version != 1
            or not isinstance(manifest.get("modelName"), str)
            or not manifest["modelName"]
            or not isinstance(manifest.get("sourceCheckpoint"), str)
            or not manifest["sourceCheckpoint"]):
        raise QpackInstallError("manifest is not a supported QPACK v1 container")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > MAX_FILES:
        raise QpackInstallError("qpack manifest files table is invalid")
    for field in ("quantBits", "quantGroupSize"):
        value = manifest.get(field)
        if (value is not None and (isinstance(value, bool)
                                   or not isinstance(value, int)
                                   or value < 0 or value > (1 << 31) - 1)):
            raise QpackInstallError(f"manifest {field} is invalid")

    planned = []
    for name, size in files.items():
        name = _validate_name(name)
        file = QpackFile(name, size)
        _validate_file(file)
        planned.append(file)
    _validate_namespace(planned)
    return QpackInstallPlan(
        source_id=source_id,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        files=tuple(sorted(planned)),
    )


def _validate_namespace(files) -> None:
    paths = [MANIFEST, JOURNAL, LOCK, f".{MANIFEST}.part", f".{JOURNAL}.part"]
    for file in files:
        paths.extend((file.name, file.name + ".part"))
    normalized = sorted(path.casefold() for path in paths)
    for previous, path in zip(normalized, normalized[1:]):
        if path == previous:
            raise QpackInstallError(f"qpack file namespace collision: {path}")
    occupied = set(normalized)
    for path in normalized:
        components = path.split("/")
        for count in range(1, len(components)):
            parent = "/".join(components[:count])
            if parent in occupied:
                raise QpackInstallError(
                    f"qpack file/parent namespace collision: {parent}")


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
    if temporary.is_symlink():
        raise QpackInstallError(f"refusing symlink temporary path: {temporary}")
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
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8") + b"\n"
    if len(raw) > MAX_JOURNAL:
        raise QpackInstallError("qpack install journal exceeds its size limit")
    _atomic_bytes(path, raw)


def _journal_payload(plan: QpackInstallPlan):
    return {
        "schema": SCHEMA,
        "source_id": plan.source_id,
        "manifest_sha256": plan.manifest_sha256,
        "files": [{"name": file.name, "size": file.size} for file in plan.files],
    }


def _read_json_file(path: Path, description: str):
    try:
        if path.is_symlink():
            raise QpackInstallError(f"refusing symlink {description}: {path}")
        return _decode_json(path.read_bytes(), description, MAX_JOURNAL)
    except OSError as error:
        raise QpackInstallError(f"cannot read {description} {path}: {error}") from error


def _root(path) -> Path:
    root = Path(path).expanduser().absolute()
    if root.is_symlink():
        raise QpackInstallError(f"qpack install root is a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise QpackInstallError(f"qpack install root is not a directory: {root}")
    return root


def _lock(root: Path):
    path = root / LOCK
    if path.is_symlink():
        raise QpackInstallError(f"qpack install lock is a symlink: {path}")
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return stream
    except (OSError, BlockingIOError) as error:
        stream.close()
        raise QpackInstallError(f"qpack install is locked: {root}") from error


def _unlock(stream) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _paths(root: Path, file: QpackFile, create_parent: bool = False):
    _validate_file(file)
    parent = root
    components = file.name.split("/")
    for component in components[:-1]:
        parent = parent / component
        if parent.is_symlink():
            raise QpackInstallError(f"qpack path crosses a symlink: {parent}")
        if parent.exists() and not parent.is_dir():
            raise QpackInstallError(f"qpack parent is not a directory: {parent}")
        if create_parent and not parent.exists():
            parent.mkdir()
            _fsync_directory(parent.parent)
    target = parent / components[-1]
    partial = target.with_name(target.name + ".part")
    if target.is_symlink() or partial.is_symlink():
        raise QpackInstallError(f"qpack artifact path is a symlink: {target}")
    return target, partial


def _matching_journal(root: Path, plan: QpackInstallPlan | None = None):
    path = root / JOURNAL
    if not path.is_file():
        raise QpackInstallError("qpack install journal is missing")
    journal = _read_json_file(path, "qpack install journal")
    if not isinstance(journal, dict) or journal.get("schema") != SCHEMA:
        raise QpackInstallError("qpack install journal is invalid")
    if plan is not None and journal != _journal_payload(plan):
        raise QpackInstallError("qpack install journal belongs to another source or manifest")
    return journal


def _journal_declares(root: Path, file: QpackFile) -> None:
    journal = _matching_journal(root)
    entry = {"name": file.name, "size": file.size}
    if entry not in journal.get("files", []):
        raise QpackInstallError(f"qpack install journal does not declare {file.name}")


def _all_files_exact(root: Path, plan: QpackInstallPlan) -> bool:
    for file in plan.files:
        target, partial = _paths(root, file)
        if (partial.exists() or not target.is_file()
                or target.stat().st_size != file.size):
            return False
    return True


def begin_install(root, plan: QpackInstallPlan) -> QpackInstallSession:
    """Lock the root and create or verify its source-bound install journal."""
    _validate_plan(plan)
    root = _root(root)
    lock_stream = _lock(root)
    session = QpackInstallSession(root, plan, lock_stream)
    manifest_path = root / MANIFEST
    journal_path = root / JOURNAL
    try:
        if manifest_path.is_symlink():
            raise QpackInstallError("existing qpack manifest is a symlink")
        if manifest_path.exists():
            if not manifest_path.is_file():
                raise QpackInstallError("existing qpack manifest is not a regular file")
            digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if digest != plan.manifest_sha256 or not _all_files_exact(root, plan):
                raise QpackInstallError("install root contains a different or incomplete qpack")
            journal_path.unlink(missing_ok=True)
            _fsync_directory(root)
            session.complete = True
            return session

        if journal_path.is_symlink():
            raise QpackInstallError("qpack install journal is a symlink")
        if journal_path.exists():
            _matching_journal(root, plan)
            return session

        for file in plan.files:
            target, partial = _paths(root, file)
            if target.exists() or partial.exists():
                raise QpackInstallError(
                    f"unbound qpack artifact exists before journal creation: {file.name}")
        _atomic_json(journal_path, _journal_payload(plan))
        return session
    except Exception:
        session.close()
        raise


def _validate_plan(plan: QpackInstallPlan) -> None:
    if (not isinstance(plan, QpackInstallPlan)
            or not isinstance(plan.source_id, str) or not plan.source_id.strip()
            or "\x00" in plan.source_id
            or len(plan.source_id.encode("utf-8")) > 4096
            or not isinstance(plan.manifest_sha256, str)
            or len(plan.manifest_sha256) != 64):
        raise QpackInstallError("qpack install plan identity is invalid")
    try:
        int(plan.manifest_sha256, 16)
    except ValueError as error:
        raise QpackInstallError("qpack manifest digest is invalid") from error
    if (not isinstance(plan.files, tuple) or not plan.files
            or len(plan.files) > MAX_FILES):
        raise QpackInstallError("qpack install plan files are invalid")
    for file in plan.files:
        _validate_file(file)
    _validate_namespace(plan.files)


def _require_session(session: QpackInstallSession) -> None:
    if not isinstance(session, QpackInstallSession) or session.closed:
        raise QpackInstallError("qpack install session is closed or invalid")
    if session.complete:
        raise QpackInstallError("qpack install is already complete")
    _matching_journal(session.root, session.plan)


def resume_decision(session: QpackInstallSession,
                    file: QpackFile) -> ResumeDecision:
    _require_session(session)
    _journal_declares(session.root, file)
    target, partial = _paths(session.root, file, create_parent=True)
    if target.exists():
        if not target.is_file() or target.stat().st_size != file.size:
            raise QpackInstallError(f"committed qpack artifact has wrong size: {file.name}")
        if partial.exists():
            raise QpackInstallError(f"committed and partial qpack artifacts both exist: {file.name}")
        return ResumeDecision("complete", file.size, partial)
    if not partial.exists():
        return ResumeDecision("restart", 0, partial)
    if not partial.is_file():
        raise QpackInstallError(f"qpack partial is not a regular file: {partial}")
    size = partial.stat().st_size
    if size > file.size:
        return ResumeDecision("restart", 0, partial)
    return ResumeDecision("resume", size, partial)


def validate_delivery(file: QpackFile, requested_offset: int,
                      delivered_offset: int, delivered_size: int,
                      *, complete: bool = False) -> int:
    """Validate the byte range a transport claims it delivered."""
    _validate_file(file)
    values = (requested_offset, delivered_offset, delivered_size)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise QpackInstallError("delivery offsets and size must be integers")
    if requested_offset < 0 or requested_offset > file.size:
        raise QpackInstallError("requested qpack offset is out of bounds")
    if delivered_offset != requested_offset:
        raise QpackInstallError("transport did not honor the requested qpack range")
    if delivered_size < 0 or delivered_offset + delivered_size > file.size:
        raise QpackInstallError("delivered qpack range is out of bounds")
    end = delivered_offset + delivered_size
    if complete and end != file.size:
        raise QpackInstallError("transport reported a short final qpack delivery")
    return end


def commit_file(session: QpackInstallSession, file: QpackFile) -> None:
    """Durably publish one exact-size partial artifact."""
    _require_session(session)
    _journal_declares(session.root, file)
    target, partial = _paths(session.root, file, create_parent=True)
    if target.exists():
        if target.is_file() and target.stat().st_size == file.size and not partial.exists():
            return
        raise QpackInstallError(f"cannot commit over existing qpack artifact: {file.name}")
    if not partial.is_file() or partial.stat().st_size != file.size:
        raise QpackInstallError(f"qpack partial is not complete: {file.name}")
    # Windows implements fsync as _commit, which needs a writable handle; a
    # read-only descriptor fails with EBADF there. POSIX flushes either way.
    flags = (os.O_RDWR if os.name == "nt" else os.O_RDONLY) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(partial, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(partial, target)
    _fsync_directory(target.parent)


def publish_manifest(session: QpackInstallSession, raw: bytes) -> None:
    """Publish manifest.json last, after every declared artifact is durable."""
    _require_session(session)
    root = session.root
    plan = session.plan
    parsed = parse_manifest(raw, session.plan.source_id)
    if parsed != session.plan:
        raise QpackInstallError("manifest bytes do not match the qpack install plan")
    if not _all_files_exact(root, plan):
        raise QpackInstallError("cannot publish qpack manifest before all files are complete")
    manifest_path = root / MANIFEST
    if manifest_path.is_symlink():
        raise QpackInstallError("refusing symlink qpack manifest")
    if manifest_path.exists():
        if (not manifest_path.is_file()
                or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                != plan.manifest_sha256):
            raise QpackInstallError("refusing to replace a different qpack manifest")
    else:
        _atomic_bytes(manifest_path, raw)
    (root / JOURNAL).unlink(missing_ok=True)
    _fsync_directory(root)
    session.complete = True
