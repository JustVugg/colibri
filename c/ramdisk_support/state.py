"""Durable RAM-disk state, manifests, locking, and usage recovery."""

from __future__ import print_function

import contextlib
import json
import os
import posixpath
import re
import secrets
import stat
import tempfile
import threading

from .common import (
    DEFAULT_MOUNT_ROOT,
    MANIFEST_VERSION,
    PROFILE_LINE_RE,
    USAGE_MERGE_RE,
    RamdiskError,
    _path_is_below,
    _path_without_symlinks,
    _positive_int,
    _validated_usage_header,
    _utc_now,
    _usage_engine_id,
    _usage_engine_name,
)
from .platform_ops import current_uid, get_platform_ops
from .accelerator import _managed_accelerator_contract


try:
    import fcntl
except ImportError:
    fcntl = None


_lifecycle_local = threading.local()
_fallback_usage_lock = threading.RLock()


def _valid_usage_snapshot(value):
    if not isinstance(value, dict):
        return False
    for key, count in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(count, int)
            or isinstance(count, bool)
        ):
            return False
        if re.fullmatch(r"-[12]:[1-9]\d*", key):
            if count <= 0:
                return False
        elif re.fullmatch(r"\d+:\d+", key):
            if count < 0:
                return False
        else:
            return False
    return True


def _state_root():
    base = os.environ.get("XDG_STATE_HOME")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".local", "state")
    base = os.path.expanduser(base)
    if not os.path.isabs(base):
        raise RamdiskError("XDG_STATE_HOME must be an absolute durable path")
    return os.path.normpath(os.path.join(base, "colibri", "ramdisk"))


def _manifest_path():
    override = os.environ.get("COLI_RAMDISK_MANIFEST")
    if override:
        override = os.path.expanduser(override)
        if not os.path.isabs(override):
            raise RamdiskError("COLI_RAMDISK_MANIFEST must be an absolute durable path")
        return os.path.normpath(override)
    return os.path.join(_state_root(), "manifest.json")


def _benchmarks_path():
    return os.path.join(_state_root(), "benchmarks.json")


def _ensure_private_dir(path):
    path = os.path.normpath(path)
    if not _path_without_symlinks(path):
        raise RamdiskError("private state path contains a symlink: %s" % path)
    os.makedirs(path, mode=0o700, exist_ok=True)
    if not _path_without_symlinks(path):
        raise RamdiskError("private state path changed through a symlink: %s" % path)
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RamdiskError("private state path is not a real directory: %s" % path)
    os.chmod(path, 0o700)


def _assert_durable_state_dir(
    path,
    plan=None,
    *,
    filesystem_for_path=None,
):
    """Revalidate a derived engine/benchmark state directory before use."""
    path = os.path.normpath(path)
    if not _path_without_symlinks(path):
        raise RamdiskError("managed state path contains a symlink: %s" % path)
    if filesystem_for_path is None:
        raise RamdiskError("managed state filesystem validation is unavailable")
    if filesystem_for_path(path) in ("tmpfs", "ramfs"):
        raise RamdiskError("managed state path is on a volatile filesystem: %s" % path)
    if plan is not None:
        for mount in plan.get("mounts", []):
            weight_path = mount.get("path")
            if isinstance(weight_path, str) and _path_is_below(
                os.path.realpath(path),
                os.path.realpath(weight_path),
                allow_equal=True,
            ):
                raise RamdiskError(
                    "managed state path overlaps volatile weights: %s" % path
                )
    return path


def _ensure_atomic_parent(path):
    """Create a missing atomic-write parent without mutating an existing one."""
    if os.path.lexists(path):
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RamdiskError(
                "atomic-state parent is not a real directory: %s" % path
            )
        return
    try:
        os.makedirs(path, mode=0o700)
    except FileExistsError:
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RamdiskError(
                "atomic-state parent is not a real directory: %s" % path
            )
        return
    os.chmod(path, 0o700)


@contextlib.contextmanager
def _lifecycle_lock():
    """Serialize all manifest-changing operations for this invoking user."""
    depth = getattr(_lifecycle_local, "depth", 0)
    if depth:
        _lifecycle_local.depth = depth + 1
        try:
            yield
        finally:
            _lifecycle_local.depth -= 1
        return
    if fcntl is None or not get_platform_ops().is_linux:
        raise RamdiskError("RAM-disk lifecycle locking is supported only on Linux")
    root = _state_root()
    _ensure_private_dir(root)
    lock_path = os.path.join(root, "lifecycle.lock")
    with open(lock_path, "a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RamdiskError("another `coli ramdisk` lifecycle operation is active")
        _lifecycle_local.depth = 1
        try:
            yield
        finally:
            _lifecycle_local.depth = 0
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _atomic_json(path, value, mode=0o600):
    parent = os.path.dirname(path)
    _ensure_atomic_parent(parent)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        _fsync_directory(parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path, required=False):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError:
        if required:
            raise RamdiskError("RAM-disk manifest not found; run `coli ramdisk prepare`")
        return None
    except (OSError, ValueError) as exc:
        raise RamdiskError("cannot read %s: %s" % (path, exc))


def _manifest_mount_layout(plan):
    """Validate and return the exact v1 mount layout encoded by a plan."""
    topology = plan.get("topology")
    root = plan.get("mount_root")
    planned = plan.get("mounts")
    model = plan.get("model", {}).get("path")
    if topology not in ("interleaved", "per-node") or not isinstance(root, str):
        raise RamdiskError("RAM-disk manifest has an invalid topology")
    root = posixpath.normpath(root)
    if (
        not posixpath.isabs(root)
        or posixpath.commonpath([root, "/mnt"]) != "/mnt"
        or root == "/mnt"
        or not _path_without_symlinks(root)
        or root in ("/mnt", DEFAULT_MOUNT_ROOT + "/..")
    ):
        raise RamdiskError("RAM-disk manifest has an unsafe mount root")
    try:
        normalized_model = str(model).replace("\\", "/")
        if re.match(r"^[A-Za-z]:/", normalized_model):
            common = None
        else:
            normalized_model = posixpath.normpath(normalized_model)
            common = posixpath.commonpath([root, normalized_model])
        if common in (root, normalized_model):
            raise RamdiskError(
                "RAM-disk manifest mount root overlaps its canonical model"
            )
    except (TypeError, ValueError):
        raise RamdiskError(
            "RAM-disk manifest has incompatible model and mount paths"
        )
    if not isinstance(planned, list) or not planned:
        raise RamdiskError("RAM-disk manifest has no planned mounts")
    planned_nodes = plan.get("placement", {}).get(
        "memory_nodes",
        plan.get("hardware", {}).get("online_nodes"),
    )
    if topology == "interleaved":
        expected = [(None, root)]
    else:
        if (
            not isinstance(planned_nodes, list)
            or not planned_nodes
            or any(
                not isinstance(node, int)
                or isinstance(node, bool)
                or node < 0
                for node in planned_nodes
            )
            or len(set(planned_nodes)) != len(planned_nodes)
        ):
            raise RamdiskError("RAM-disk manifest has an invalid NUMA node set")
        expected = [
            (node, posixpath.join(root, "node%d" % node))
            for node in planned_nodes
        ]
    observed = []
    for record in planned:
        if not isinstance(record, dict):
            raise RamdiskError(
                "RAM-disk manifest has invalid planned mount paths"
            )
        path = record.get("path")
        node = record.get("node")
        if (
            not isinstance(path, str)
            or not posixpath.isabs(path)
            or posixpath.normpath(path) != path
            or not _path_without_symlinks(path)
        ):
            raise RamdiskError(
                "RAM-disk manifest has invalid planned mount paths"
            )
        observed.append((node, path))
    if observed != expected:
        raise RamdiskError(
            "RAM-disk manifest mounts do not match its topology"
        )
    return root, expected


def _load_manifest(
    required=False,
    *,
    filesystem_for_path=None,
    read_json=None,
    manifest_path=None,
    state_root=None,
    benchmarks_path=None,
    assert_durable_state_dir=None,
    uid_provider=None,
):
    """Read and minimally validate lifecycle state before it can drive actions."""
    read_json = _read_json if read_json is None else read_json
    manifest_path = _manifest_path if manifest_path is None else manifest_path
    state_root = _state_root if state_root is None else state_root
    benchmarks_path = _benchmarks_path if benchmarks_path is None else benchmarks_path
    uid_provider = current_uid if uid_provider is None else uid_provider
    if assert_durable_state_dir is None:
        def assert_durable_state_dir(path, plan=None):
            return _assert_durable_state_dir(
                path,
                plan=plan,
                filesystem_for_path=filesystem_for_path,
            )

    manifest = read_json(manifest_path(), required=required)
    if manifest is None:
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != MANIFEST_VERSION
    ):
        raise RamdiskError("unsupported or malformed RAM-disk manifest")
    base_port = manifest.get("base_port")
    if base_port is not None and (
        isinstance(base_port, bool)
        or not isinstance(base_port, int)
        or not 1 <= base_port <= 65535
    ):
        raise RamdiskError("RAM-disk manifest has an invalid managed base port")
    deployment_id = manifest.get("deployment_id")
    if deployment_id is not None and (
        not isinstance(deployment_id, str)
        or not re.fullmatch(r"[0-9a-f]{32}", deployment_id)
    ):
        raise RamdiskError("RAM-disk manifest has an invalid deployment identity")
    plan = manifest.get("plan")
    mounts = manifest.get("mounts")
    processes = manifest.get("processes", [])
    if (
        not isinstance(plan, dict)
        or not isinstance(mounts, list)
        or not isinstance(processes, list)
    ):
        raise RamdiskError("RAM-disk manifest is missing lifecycle records")
    if (
        not isinstance(plan.get("model"), dict)
        or not isinstance(plan.get("mounts"), list)
    ):
        raise RamdiskError("RAM-disk manifest has an invalid plan")
    fingerprint = manifest.get("model_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint)
        or plan["model"].get("fingerprint") != fingerprint
        or not isinstance(plan["model"].get("path"), str)
        or not os.path.isabs(plan["model"]["path"])
        or not plan["mounts"]
    ):
        raise RamdiskError("RAM-disk manifest has an invalid model identity")
    _managed_accelerator_contract(plan)
    mount_root, _ = _manifest_mount_layout(plan)
    planned_paths = {record["path"] for record in plan["mounts"]}
    if len(planned_paths) != len(plan["mounts"]):
        raise RamdiskError("RAM-disk manifest has invalid planned mount paths")
    mount_by_path = {record["path"]: record for record in plan["mounts"]}
    mount_ownership = {}
    for record in mounts:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not os.path.isabs(record["path"])
        ):
            raise RamdiskError(
                "RAM-disk manifest contains an unsafe mount record"
            )
        if record["path"] not in planned_paths:
            raise RamdiskError(
                "RAM-disk manifest mount does not belong to its plan"
            )
        if record.get("node") != mount_by_path[record["path"]].get("node"):
            raise RamdiskError(
                "RAM-disk manifest mount has the wrong NUMA node"
            )
        identity = record.get("identity")
        identity_is_exact = (
            isinstance(identity, dict)
            and _positive_int(identity.get("mount_id"))
            and isinstance(identity.get("device"), str)
            and bool(identity["device"])
        )
        ownership = record.get("ownership")
        if ownership is None and identity_is_exact:
            # Manifests written before ownership transitions were explicit
            # contain only exact identities and are managed by definition.
            ownership = "managed"
        if ownership not in ("pending", "identified", "managed"):
            raise RamdiskError(
                "RAM-disk manifest mount has an invalid ownership state"
            )
        if ownership == "pending" and identity is not None:
            raise RamdiskError(
                "RAM-disk pending mount unexpectedly has an identity"
            )
        if ownership != "pending" and not identity_is_exact:
            raise RamdiskError(
                "RAM-disk manifest mount is missing its identity"
            )
        if record["path"] in mount_ownership:
            raise RamdiskError(
                "RAM-disk manifest contains duplicate mount records"
            )
        mount_ownership[record["path"]] = ownership
    state = manifest.get("state")
    if state not in (
        "preparing",
        "ready",
        "starting",
        "running",
        "stopped",
        "error",
    ):
        raise RamdiskError("RAM-disk manifest has an invalid lifecycle state")
    recovery = manifest.get("recovery")
    if recovery is not None and not isinstance(recovery, dict):
        raise RamdiskError("RAM-disk manifest has invalid recovery metadata")
    retained_processes = (
        recovery.get("retained_processes", [])
        if isinstance(recovery, dict)
        else []
    )
    if not isinstance(retained_processes, list):
        raise RamdiskError(
            "RAM-disk manifest has invalid retained process recovery"
        )
    if retained_processes and state != "error":
        raise RamdiskError(
            "RAM-disk retained process recovery requires the error state"
        )
    pending_launches = manifest.get("pending_launches", [])
    if not isinstance(pending_launches, list):
        raise RamdiskError("RAM-disk manifest has invalid pending launches")
    if pending_launches and state not in ("starting", "error"):
        raise RamdiskError(
            "RAM-disk pending launches require starting or error state"
        )
    for pending in pending_launches:
        operation_id = (
            pending.get("operation_id") if isinstance(pending, dict) else None
        )
        nonce = pending.get("nonce") if isinstance(pending, dict) else None
        port = pending.get("port") if isinstance(pending, dict) else None
        node = pending.get("node") if isinstance(pending, dict) else None
        state_dir = (
            pending.get("state_dir") if isinstance(pending, dict) else None
        )
        baseline = (
            pending.get("usage_baseline")
            if isinstance(pending, dict)
            else None
        )
        merge_id = (
            pending.get("usage_merge_id")
            if isinstance(pending, dict)
            else None
        )
        if (
            not isinstance(operation_id, str)
            or not re.fullmatch(r"start:[0-9a-f]{32}", operation_id)
            or not isinstance(nonce, str)
            or not re.fullmatch(r"[0-9a-f]{48}", nonce)
            or not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65535
            or node not in {record.get("node") for record in plan["mounts"]}
            or not isinstance(state_dir, str)
            or not os.path.isabs(state_dir)
            or not _valid_usage_snapshot(baseline)
            or not isinstance(merge_id, str)
            or not re.fullmatch(r"[0-9a-f]{32}", merge_id)
        ):
            raise RamdiskError(
                "RAM-disk manifest has unsafe pending launch recovery"
            )
    for retained in retained_processes:
        pid = retained.get("pid") if isinstance(retained, dict) else None
        pgid = retained.get("pgid") if isinstance(retained, dict) else None
        node = retained.get("node") if isinstance(retained, dict) else None
        state_dir = (
            retained.get("state_dir") if isinstance(retained, dict) else None
        )
        baseline = (
            retained.get("usage_baseline")
            if isinstance(retained, dict)
            else None
        )
        merge_id = (
            retained.get("usage_merge_id")
            if isinstance(retained, dict)
            else None
        )
        if (
            not _positive_int(pid)
            or pgid != pid
            or node not in {record.get("node") for record in plan["mounts"]}
            or not isinstance(state_dir, str)
            or not os.path.isabs(state_dir)
            or not _valid_usage_snapshot(baseline)
            or not isinstance(merge_id, str)
            or not re.fullmatch(r"[0-9a-f]{32}", merge_id)
        ):
            raise RamdiskError(
                "RAM-disk manifest has unsafe retained process recovery"
            )
    recorded_paths = {record["path"] for record in mounts}
    non_managed = sorted(
        path
        for path, ownership in mount_ownership.items()
        if ownership != "managed"
    )
    if state not in ("preparing", "error") and non_managed:
        raise RamdiskError(
            "RAM-disk manifest contains pending mount ownership outside "
            "preparation/recovery: %s" % ", ".join(non_managed)
        )
    if (
        state in ("ready", "starting", "running", "stopped")
        and recorded_paths != planned_paths
    ):
        raise RamdiskError(
            "ready RAM-disk manifest does not contain every planned mount"
        )
    if state == "running" and len(processes) != len(mounts):
        raise RamdiskError(
            "running RAM-disk manifest has an incomplete process set"
        )
    durable = plan.get("durable_state")
    expected_durable = {
        "root": state_root(),
        "manifest": manifest_path(),
        "benchmarks": benchmarks_path(),
    }
    if durable != expected_durable or any(
        not _path_without_symlinks(path)
        for path in expected_durable.values()
    ):
        raise RamdiskError(
            "RAM-disk manifest has an invalid durable-state identity"
        )
    if filesystem_for_path is None:
        raise RamdiskError("durable-state filesystem validation is unavailable")
    if any(
        filesystem_for_path(path) in ("tmpfs", "ramfs")
        for path in expected_durable.values()
    ):
        raise RamdiskError(
            "RAM-disk durable state is on a volatile filesystem"
        )
    if any(
        _path_is_below(path, mount_root, allow_equal=True)
        for path in expected_durable.values()
    ):
        raise RamdiskError(
            "RAM-disk durable state overlaps the volatile mount"
        )
    source_shards = plan.get("source_shards")
    if not isinstance(source_shards, list) or not source_shards:
        raise RamdiskError(
            "RAM-disk manifest is missing canonical shard identities"
        )
    fingerprint_dir = fingerprint.split(":", 1)[1]
    expected_state_root = os.path.join(
        state_root(),
        "engines",
        fingerprint_dir,
    )
    recovery_state_dirs = set()
    for recovery_record in pending_launches + retained_processes:
        node = recovery_record.get("node")
        label = "interleaved" if node is None else "node-%d" % node
        expected_recovery_state_dir = os.path.join(
            expected_state_root,
            label,
        )
        state_dir = recovery_record["state_dir"]
        if (
            state_dir != expected_recovery_state_dir
            or state_dir in recovery_state_dirs
        ):
            raise RamdiskError(
                "RAM-disk recovery has an unsafe or duplicate state directory"
            )
        assert_durable_state_dir(state_dir, plan=plan)
        recovery_state_dirs.add(state_dir)
    process_keys = {
        "pid": set(),
        "pgid": set(),
        "port": set(),
        "node": set(),
        "state_dir": set(recovery_state_dirs),
        "weights_dir": set(),
    }
    invoking_uid = uid_provider()
    for record in processes:
        if not isinstance(record, dict):
            raise RamdiskError(
                "RAM-disk manifest contains an invalid process record"
            )
        pid, pgid = record.get("pid"), record.get("pgid")
        uid, starttime = record.get("uid"), record.get("starttime")
        nonce, port = record.get("nonce"), record.get("port")
        node, weights_dir = record.get("node"), record.get("weights_dir")
        state_dir, command = record.get("state_dir"), record.get("command")
        usage_baseline = record.get("usage_baseline")
        usage_merge_id = record.get("usage_merge_id")
        mount = next(
            (
                item
                for item in plan["mounts"]
                if item.get("node") == node
            ),
            None,
        )
        label = (
            "interleaved"
            if node is None
            else "node-%d" % node
            if isinstance(node, int)
            else ""
        )
        expected_state_dir = os.path.join(expected_state_root, label)
        valid_command = (
            isinstance(command, list)
            and command
            and all(isinstance(item, str) and item for item in command)
        )
        try:
            model_at = command.index("--model") if valid_command else -1
            port_at = command.index("--port") if valid_command else -1
            serves = "serve" in command
            command_matches = (
                serves
                and command[model_at + 1] == plan["model"]["path"]
                and int(command[port_at + 1]) == port
            )
        except (ValueError, IndexError, TypeError):
            command_matches = False
        if (
            not _positive_int(pid)
            or not _positive_int(pgid)
            or pid != pgid
            or uid != invoking_uid
            or not _positive_int(starttime)
            or not isinstance(nonce, str)
            or not re.fullmatch(r"[0-9a-f]{48}", nonce)
            or not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65535
            or mount is None
            or weights_dir != mount["path"]
            or state_dir != expected_state_dir
            or not command_matches
            or not _valid_usage_snapshot(usage_baseline)
            or (
                usage_merge_id is not None
                and (
                    not isinstance(usage_merge_id, str)
                    or not re.fullmatch(r"[0-9a-f]{32}", usage_merge_id)
                )
            )
        ):
            raise RamdiskError(
                "RAM-disk manifest contains an unsafe managed process record"
            )
        assert_durable_state_dir(state_dir, plan=plan)
        for key, value in (
            ("pid", pid),
            ("pgid", pgid),
            ("port", port),
            ("node", node),
            ("state_dir", state_dir),
            ("weights_dir", weights_dir),
        ):
            if value in process_keys[key]:
                raise RamdiskError(
                    "RAM-disk manifest contains duplicate managed process records"
                )
            process_keys[key].add(value)
    return manifest


def _save_manifest(
    manifest,
    *,
    atomic_json=None,
    manifest_path=None,
):
    atomic_json = _atomic_json if atomic_json is None else atomic_json
    manifest_path = _manifest_path if manifest_path is None else manifest_path
    manifest["updated_at"] = _utc_now()
    atomic_json(manifest_path(), manifest)


def _read_optional_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as stream:
            return stream.read()
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeError) as exc:
        raise RamdiskError("cannot read usage state %s: %s" % (path, exc))


def _usage_header(counts, source="usage history"):
    if not isinstance(counts, dict):
        raise RamdiskError("%s counts must contain an object" % source)
    records = []
    for key, value in counts.items():
        if not isinstance(key, str):
            continue
        match = re.fullmatch(r"(-?\d+):(\d+)", key)
        if not match:
            continue
        layer, second = (int(item) for item in match.groups())
        if layer in (-1, -2):
            try:
                records.append((layer, second, int(value)))
            except (TypeError, ValueError):
                raise RamdiskError("%s has a malformed usage header" % source)
    return _validated_usage_header(records, source=source)


def _validate_usage_for_plan(counts, plan, source="usage history"):
    """Validate identified usage metadata against the managed model."""
    header = _usage_header(counts, source=source)
    if header is None:
        return None
    try:
        model_path = plan["model"]["path"]
        with open(
            os.path.join(model_path, "config.json"),
            "r",
            encoding="utf-8",
        ) as stream:
            config = json.load(stream)
        if not isinstance(config, dict):
            raise ValueError("config root is not an object")
        dimensions = (
            int(config["num_hidden_layers"]),
            int(config["n_routed_experts"]),
        )
        engine_id = _usage_engine_id(
            _usage_engine_name(config.get("model_type"))
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise RamdiskError(
            "cannot validate %s against the managed model config: %s"
            % (source, exc)
        )
    return _validated_usage_header(
        [
            (-1, header["n_layers"], header["n_experts"]),
            (
                -2,
                header["format_version"],
                header["engine_id"],
            ),
        ],
        source=source,
        expected_dimensions=dimensions,
        expected_engine_id=engine_id,
    )


def _usage_header_counts(header):
    if not header:
        return {}
    return {
        "-1:%d" % header["n_layers"]: header["n_experts"],
        "-2:%d" % header["format_version"]: header["engine_id"],
    }


def _usage_data_counts(counts):
    result = {}
    for key, value in counts.items():
        if not isinstance(key, str):
            continue
        match = re.fullmatch(r"(\d+):(\d+)", key)
        if match:
            result[key] = int(value)
    return result


def _compatible_usage_header(*histories):
    reference = None
    reference_source = None
    for source, counts in histories:
        header = _usage_header(counts, source=source)
        if header is None:
            continue
        if reference is None:
            reference = header
            reference_source = source
            continue
        if (
            header["n_layers"],
            header["n_experts"],
        ) != (
            reference["n_layers"],
            reference["n_experts"],
        ):
            raise RamdiskError(
                "%s history dimensions do not match %s"
                % (source, reference_source)
            )
        if header["format_version"] != reference["format_version"]:
            raise RamdiskError(
                "%s usage format version does not match %s"
                % (source, reference_source)
            )
        if header["engine_id"] != reference["engine_id"]:
            raise RamdiskError(
                "%s engine identity does not match %s"
                % (source, reference_source)
            )
    return reference


def _usage_read(path):
    counts = {}
    header_records = []
    for line in _read_optional_text(path).splitlines():
        match = PROFILE_LINE_RE.match(line)
        if not match:
            if re.match(r"^\s*-(?:1|2)(?:\s|$)", line):
                raise RamdiskError("%s has a malformed usage header" % path)
            continue
        layer, expert, count = (int(value) for value in match.groups())
        if layer in (-1, -2):
            header_records.append((layer, expert, count))
        elif layer >= 0:
            counts["%d:%d" % (layer, expert)] = count
    header = _validated_usage_header(header_records, source=path)
    counts.update(_usage_header_counts(header))
    return counts


def _usage_merge_ids(path):
    result = set()
    for line in _read_optional_text(path).splitlines():
        match = USAGE_MERGE_RE.match(line.strip())
        if match:
            result.add(match.group(1))
    return result


def _fsync_directory(path):
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _durable_unlink(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    _fsync_directory(os.path.dirname(path))


def _usage_write(path, counts, merge_id=None, merge_ids=None):
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        raise RamdiskError(
            "usage-state parent directory is absent: %s" % parent
        )
    markers = set(merge_ids or ())
    if merge_id:
        markers.add(merge_id)
    header = _usage_header(counts, source=path)
    data_counts = _usage_data_counts(counts)
    fd, tmp = tempfile.mkstemp(prefix=".usage-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            # route_trace.h deliberately keeps an all-zero history zero-byte,
            # so emit identifying records only when there are data records.
            if header and data_counts:
                stream.write(
                    "-1 %d %d\n"
                    % (header["n_layers"], header["n_experts"])
                )
                stream.write(
                    "-2 %d %d\n"
                    % (header["format_version"], header["engine_id"])
                )
            for key in sorted(
                data_counts,
                key=lambda item: tuple(
                    int(value)
                    for value in item.split(":")
                ),
            ):
                layer, expert = key.split(":")
                stream.write(
                    "%s %s %d\n" % (layer, expert, data_counts[key])
                )
            for marker in sorted(markers):
                stream.write("# coli-ramdisk-merge %s\n" % marker)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        _fsync_directory(parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _assert_canonical_usage_target(
    canonical_path,
    plan=None,
    *,
    source_still_matches=None,
):
    parent = os.path.dirname(canonical_path)
    if not os.path.isdir(parent):
        raise RamdiskError(
            "canonical model directory is absent; usage delta remains journaled"
        )
    if plan is not None:
        if os.path.realpath(parent) != os.path.realpath(plan["model"]["path"]):
            raise RamdiskError(
                "canonical usage target no longer matches the managed model"
            )
        if source_still_matches is None:
            raise RamdiskError(
                "canonical source identity validation is unavailable"
            )
        source_still_matches(plan)


@contextlib.contextmanager
def _usage_lock(lock):
    if fcntl is None:
        with _fallback_usage_lock:
            yield
        return
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _recover_delta(
    state_dir,
    canonical_path,
    plan=None,
    *,
    filesystem_for_path=None,
    source_still_matches=None,
):
    _assert_durable_state_dir(
        state_dir,
        plan=plan,
        filesystem_for_path=filesystem_for_path,
    )
    delta_path = os.path.join(state_dir, ".coli_usage.delta.json")
    payload = _read_json(delta_path)
    if not payload:
        return
    if not isinstance(payload, dict):
        raise RamdiskError("usage delta journal must contain a JSON object")
    delta = payload.get("delta", {})
    if not isinstance(delta, dict):
        raise RamdiskError("usage delta journal has invalid counts")
    headers = payload.get("headers", {})
    if not isinstance(headers, dict):
        raise RamdiskError("usage delta journal has invalid headers")
    _compatible_usage_header(
        ("usage delta journal", headers),
    )
    merge_id = payload.get("id")
    if not merge_id:
        raise RamdiskError(
            "usage delta journal is missing its transaction id"
        )
    if (
        not isinstance(merge_id, str)
        or not re.fullmatch(r"[0-9a-f]{32}", merge_id)
    ):
        raise RamdiskError(
            "usage delta journal has an invalid transaction id"
        )
    _assert_canonical_usage_target(
        canonical_path,
        plan=plan,
        source_still_matches=source_still_matches,
    )
    lock_path = os.path.join(_state_root(), "usage.lock")
    _ensure_private_dir(os.path.dirname(lock_path))
    with open(lock_path, "a+", encoding="utf-8") as lock:
        with _usage_lock(lock):
            applied = _usage_merge_ids(canonical_path)
            if merge_id not in applied:
                canonical = _usage_read(canonical_path)
                merged_header = _compatible_usage_header(
                    ("usage delta journal", headers),
                    ("canonical usage history", canonical),
                )
                for key, value in delta.items():
                    if not re.fullmatch(r"\d+:\d+", str(key)):
                        raise RamdiskError(
                            "usage delta journal has invalid counts"
                        )
                    canonical[key] = canonical.get(key, 0) + int(value)
                canonical.update(_usage_header_counts(merged_header))
                applied.add(merge_id)
                _usage_write(
                    canonical_path,
                    canonical,
                    merge_ids=applied,
                )
    _durable_unlink(delta_path)


def _merge_usage(
    record,
    canonical_path,
    plan=None,
    keep_journal=False,
    *,
    filesystem_for_path=None,
    source_still_matches=None,
):
    _assert_durable_state_dir(
        record["state_dir"],
        plan=plan,
        filesystem_for_path=filesystem_for_path,
    )
    state_usage = os.path.join(record["state_dir"], ".coli_usage")
    current = _usage_read(state_usage)
    baseline = record.get("usage_baseline")
    if not _valid_usage_snapshot(baseline):
        raise RamdiskError(
            "managed usage recovery is missing a valid exact baseline"
        )
    persisted_merge_id = record.get("usage_merge_id")
    if persisted_merge_id is not None and (
        not isinstance(persisted_merge_id, str)
        or not re.fullmatch(r"[0-9a-f]{32}", persisted_merge_id)
    ):
        raise RamdiskError(
            "managed usage recovery has an invalid transaction id"
        )
    source_header = _compatible_usage_header(
        ("managed usage history", current),
        ("usage baseline", baseline),
    )
    current_counts = _usage_data_counts(current)
    baseline_counts = _usage_data_counts(baseline)
    delta = {
        key: value - baseline_counts.get(key, 0)
        for key, value in current_counts.items()
        if value > baseline_counts.get(key, 0)
    }
    delta_path = os.path.join(
        record["state_dir"],
        ".coli_usage.delta.json",
    )
    if os.path.exists(delta_path):
        _recover_delta(
            record["state_dir"],
            canonical_path,
            plan=plan,
            filesystem_for_path=filesystem_for_path,
            source_still_matches=source_still_matches,
        )
        return
    merge_id = record.get("usage_merge_id") or secrets.token_hex(16)
    record["usage_merge_id"] = merge_id
    if merge_id in _usage_merge_ids(canonical_path):
        return
    if not delta:
        # A current engine upgrades a legacy headerless seed when it saves.
        # Preserve that newly known identity even when no counters changed.
        if source_header is not None:
            _assert_canonical_usage_target(
                canonical_path,
                plan=plan,
                source_still_matches=source_still_matches,
            )
            lock_path = os.path.join(_state_root(), "usage.lock")
            _ensure_private_dir(os.path.dirname(lock_path))
            with open(lock_path, "a+", encoding="utf-8") as lock:
                with _usage_lock(lock):
                    canonical = _usage_read(canonical_path)
                    canonical_header = _compatible_usage_header(
                        ("managed usage history", current),
                        ("usage baseline", baseline),
                        ("canonical usage history", canonical),
                    )
                    if _usage_header(canonical) is None:
                        canonical.update(
                            _usage_header_counts(canonical_header)
                        )
                        _usage_write(
                            canonical_path,
                            canonical,
                            merge_ids=_usage_merge_ids(canonical_path),
                        )
        try:
            _durable_unlink(delta_path)
        except OSError:
            pass
        return
    _atomic_json(
        delta_path,
        {
            "version": 1,
            "id": merge_id,
            "delta": delta,
            "headers": _usage_header_counts(source_header),
            "created_at": _utc_now(),
        },
    )
    _assert_canonical_usage_target(
        canonical_path,
        plan=plan,
        source_still_matches=source_still_matches,
    )
    lock_path = os.path.join(_state_root(), "usage.lock")
    _ensure_private_dir(os.path.dirname(lock_path))
    with open(lock_path, "a+", encoding="utf-8") as lock:
        with _usage_lock(lock):
            applied = _usage_merge_ids(canonical_path)
            if merge_id not in applied:
                canonical = _usage_read(canonical_path)
                merged_header = _compatible_usage_header(
                    ("usage delta journal", _usage_header_counts(source_header)),
                    ("canonical usage history", canonical),
                )
                for key, value in delta.items():
                    canonical[key] = canonical.get(key, 0) + value
                canonical.update(_usage_header_counts(merged_header))
                applied.add(merge_id)
                _usage_write(
                    canonical_path,
                    canonical,
                    merge_ids=applied,
                )
    if not keep_journal:
        _durable_unlink(delta_path)
