"""NUMA-aware tmpfs staging and managed-engine lifecycle for ``coli ramdisk``.

The module intentionally uses only the Python standard library.  Planning and
status are unprivileged; the only privileged subprocesses are the exact mount
and unmount commands issued by :func:`prepare` and :func:`destroy`.
"""

from __future__ import print_function

import argparse
import concurrent.futures
import contextlib
import copy
import datetime
import errno
import functools
import hashlib
import json
import math
import mmap
import os
import platform
import queue
import re
import secrets
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.request
from ramdisk_ui import (
    ActionPolicy,
    DeploymentHealth,
    HealthLevel,
    PlacementContract,
    ReviewIdentity,
)

try:
    import fcntl
except ImportError:  # The command parser must remain importable on Windows.
    fcntl = None


MANIFEST_VERSION = 1
PLAN_SCHEMA = "colibri.ramdisk.plan.v1"
STATUS_SCHEMA = "colibri.ramdisk.status.v1"
BENCHMARK_SCHEMA = "colibri.ramdisk.benchmark.v1"
DEFAULT_MOUNT_ROOT = "/mnt/colibri-ram"
GIB = 1 << 30
MIB = 1 << 20
MAX_ST_HEADER = 512 * MIB
TMPFS_MAGIC = 0x01021994
EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight(\.qs)?$"
)
PROFILE_LINE_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$")
USAGE_MERGE_RE = re.compile(r"^# coli-ramdisk-merge ([0-9a-f]{32})$")


class RamdiskError(RuntimeError):
    """An expected, user-actionable lifecycle failure."""


class _OperationCancelled(RamdiskError):
    """A cooperative cancellation that reached a clean lifecycle checkpoint."""


class _EngineCleanupError(RamdiskError):
    """A benchmark engine may still be live, so no later variant may launch."""


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


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


def _path_without_symlinks(path):
    """True when no existing component redirects the reviewed absolute path."""
    return os.path.isabs(path) and os.path.realpath(path) == os.path.normpath(path)


def _path_is_below(path, parent, allow_equal=False):
    try:
        normalized = os.path.normpath(os.path.abspath(path))
        root = os.path.normpath(os.path.abspath(parent))
        return os.path.commonpath([normalized, root]) == root and (allow_equal or normalized != root)
    except ValueError:
        return False


def _reusable_empty_mountpoint(path):
    """Recognize an empty root-owned leaf left by X-mount.mkdir=0755."""
    try:
        info = os.stat(path, follow_symlinks=False)
        return (
            stat.S_ISDIR(info.st_mode)
            and info.st_uid == os.stat("/").st_uid
            and not (info.st_mode & 0o022)
            and not os.listdir(path)
        )
    except OSError:
        return False


def _ensure_private_dir(path):
    path = os.path.normpath(path)
    # os.makedirs(..., exist_ok=True) and chmod both follow an existing
    # symlink. Reject redirected components before either operation, then
    # verify the completed path again so manager-owned state cannot silently
    # land on a volatile or attacker-selected target.
    if not _path_without_symlinks(path):
        raise RamdiskError("private state path contains a symlink: %s" % path)
    os.makedirs(path, mode=0o700, exist_ok=True)
    if not _path_without_symlinks(path):
        raise RamdiskError("private state path changed through a symlink: %s" % path)
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RamdiskError("private state path is not a real directory: %s" % path)
    os.chmod(path, 0o700)


def _assert_durable_state_dir(path, plan=None):
    """Revalidate a derived engine/benchmark state directory before use."""
    path = os.path.normpath(path)
    if not _path_without_symlinks(path):
        raise RamdiskError("managed state path contains a symlink: %s" % path)
    if _filesystem_for_path(path) in ("tmpfs", "ramfs"):
        raise RamdiskError("managed state path is on a volatile filesystem: %s" % path)
    if plan is not None:
        for mount in plan.get("mounts", []):
            weight_path = mount.get("path")
            if isinstance(weight_path, str) and _path_is_below(
                os.path.realpath(path), os.path.realpath(weight_path), allow_equal=True
            ):
                raise RamdiskError(
                    "managed state path overlaps volatile weights: %s" % path
                )
    return path


def _ensure_atomic_parent(path):
    """Create a missing atomic-write parent without mutating an existing one.

    Atomic JSON is also used for an explicitly overridden manifest and for
    recovery journals.  Those parents are not necessarily manager-owned, so
    changing their mode would be both surprising and dangerous (for example,
    an override directly below /tmp).  Manager-owned state roots are tightened
    separately through ``_ensure_private_dir``.
    """
    if os.path.lexists(path):
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RamdiskError("atomic-state parent is not a real directory: %s" % path)
        return
    try:
        os.makedirs(path, mode=0o700)
    except FileExistsError:
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RamdiskError("atomic-state parent is not a real directory: %s" % path)
        return
    os.chmod(path, 0o700)


_lifecycle_local = threading.local()


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
    if fcntl is None:
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


def _exclusive_lifecycle(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with _lifecycle_lock():
            return function(*args, **kwargs)

    return wrapped


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
        try:
            dfd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
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


def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _manifest_mount_layout(plan):
    """Validate and return the exact v1 mount layout encoded by a plan."""
    topology = plan.get("topology")
    root = plan.get("mount_root")
    planned = plan.get("mounts")
    model = plan.get("model", {}).get("path")
    if topology not in ("interleaved", "per-node") or not isinstance(root, str):
        raise RamdiskError("RAM-disk manifest has an invalid topology")
    root = os.path.normpath(root)
    if (
        not os.path.isabs(root)
        or not _path_is_below(root, "/mnt")
        or not _path_without_symlinks(root)
        or root in ("/mnt", DEFAULT_MOUNT_ROOT + "/..")
    ):
        raise RamdiskError("RAM-disk manifest has an unsafe mount root")
    try:
        if os.path.commonpath([root, os.path.normpath(model)]) in (root, os.path.normpath(model)):
            raise RamdiskError("RAM-disk manifest mount root overlaps its canonical model")
    except (TypeError, ValueError):
        raise RamdiskError("RAM-disk manifest has incompatible model and mount paths")
    if not isinstance(planned, list) or not planned:
        raise RamdiskError("RAM-disk manifest has no planned mounts")
    planned_nodes = plan.get("placement", {}).get(
        "memory_nodes", plan.get("hardware", {}).get("online_nodes")
    )
    if topology == "interleaved":
        expected = [(None, root)]
    else:
        if (
            not isinstance(planned_nodes, list)
            or not planned_nodes
            or any(
                not isinstance(node, int) or isinstance(node, bool) or node < 0
                for node in planned_nodes
            )
            or len(set(planned_nodes)) != len(planned_nodes)
        ):
            raise RamdiskError("RAM-disk manifest has an invalid NUMA node set")
        expected = [
            (node, os.path.join(root, "node%d" % node)) for node in planned_nodes
        ]
    observed = []
    for record in planned:
        if not isinstance(record, dict):
            raise RamdiskError("RAM-disk manifest has invalid planned mount paths")
        path = record.get("path")
        node = record.get("node")
        if (
            not isinstance(path, str)
            or not os.path.isabs(path)
            or os.path.normpath(path) != path
            or not _path_without_symlinks(path)
        ):
            raise RamdiskError("RAM-disk manifest has invalid planned mount paths")
        observed.append((node, path))
    if observed != expected:
        raise RamdiskError("RAM-disk manifest mounts do not match its topology")
    return root, expected


def _load_manifest(required=False):
    """Read and minimally validate lifecycle state before it can drive actions."""
    manifest = _read_json(_manifest_path(), required=required)
    if manifest is None:
        return None
    if not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION:
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
    if not isinstance(plan, dict) or not isinstance(mounts, list) or not isinstance(processes, list):
        raise RamdiskError("RAM-disk manifest is missing lifecycle records")
    if not isinstance(plan.get("model"), dict) or not isinstance(plan.get("mounts"), list):
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
    mount_root, expected_layout = _manifest_mount_layout(plan)
    planned_paths = {record["path"] for record in plan["mounts"]}
    if len(planned_paths) != len(plan["mounts"]):
        raise RamdiskError("RAM-disk manifest has invalid planned mount paths")
    mount_by_path = {record["path"]: record for record in plan["mounts"]}
    for record in mounts:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not os.path.isabs(record["path"])
        ):
            raise RamdiskError("RAM-disk manifest contains an unsafe mount record")
        if record["path"] not in planned_paths:
            raise RamdiskError("RAM-disk manifest mount does not belong to its plan")
        if record.get("node") != mount_by_path[record["path"]].get("node"):
            raise RamdiskError("RAM-disk manifest mount has the wrong NUMA node")
        identity = record.get("identity")
        if not isinstance(identity, dict) or not _positive_int(identity.get("mount_id")) or not isinstance(identity.get("device"), str):
            raise RamdiskError("RAM-disk manifest mount is missing its identity")
    state = manifest.get("state")
    if state not in ("preparing", "ready", "starting", "running", "stopped", "error"):
        raise RamdiskError("RAM-disk manifest has an invalid lifecycle state")
    recorded_paths = {record["path"] for record in mounts}
    if state in ("ready", "starting", "running", "stopped") and recorded_paths != planned_paths:
        raise RamdiskError("ready RAM-disk manifest does not contain every planned mount")
    if state == "running" and len(processes) != len(mounts):
        raise RamdiskError("running RAM-disk manifest has an incomplete process set")
    durable = plan.get("durable_state")
    expected_durable = {
        "root": _state_root(),
        "manifest": _manifest_path(),
        "benchmarks": _benchmarks_path(),
    }
    if durable != expected_durable or any(
        not _path_without_symlinks(path) for path in expected_durable.values()
    ):
        raise RamdiskError("RAM-disk manifest has an invalid durable-state identity")
    if any(_filesystem_for_path(path) in ("tmpfs", "ramfs") for path in expected_durable.values()):
        raise RamdiskError("RAM-disk durable state is on a volatile filesystem")
    if any(_path_is_below(path, mount_root, allow_equal=True) for path in expected_durable.values()):
        raise RamdiskError("RAM-disk durable state overlaps the volatile mount")
    source_shards = plan.get("source_shards")
    if not isinstance(source_shards, list) or not source_shards:
        raise RamdiskError("RAM-disk manifest is missing canonical shard identities")
    fingerprint_dir = fingerprint.split(":", 1)[1]
    expected_state_root = os.path.join(_state_root(), "engines", fingerprint_dir)
    process_keys = {"pid": set(), "pgid": set(), "port": set(), "node": set(), "state_dir": set(), "weights_dir": set()}
    for record in processes:
        if not isinstance(record, dict):
            raise RamdiskError("RAM-disk manifest contains an invalid process record")
        pid, pgid = record.get("pid"), record.get("pgid")
        uid, starttime = record.get("uid"), record.get("starttime")
        nonce, port = record.get("nonce"), record.get("port")
        node, weights_dir = record.get("node"), record.get("weights_dir")
        state_dir, command = record.get("state_dir"), record.get("command")
        mount = next((item for item in plan["mounts"] if item.get("node") == node), None)
        label = "interleaved" if node is None else "node-%d" % node if isinstance(node, int) else ""
        expected_state_dir = os.path.join(expected_state_root, label)
        valid_command = isinstance(command, list) and command and all(isinstance(item, str) and item for item in command)
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
            or uid != os.getuid()
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
        ):
            raise RamdiskError("RAM-disk manifest contains an unsafe managed process record")
        _assert_durable_state_dir(state_dir, plan=plan)
        for key, value in (
            ("pid", pid), ("pgid", pgid), ("port", port), ("node", node),
            ("state_dir", state_dir), ("weights_dir", weights_dir),
        ):
            if value in process_keys[key]:
                raise RamdiskError("RAM-disk manifest contains duplicate managed process records")
            process_keys[key].add(value)
    return manifest


def _save_manifest(manifest):
    manifest["updated_at"] = _utc_now()
    _atomic_json(_manifest_path(), manifest)


def _read_text(path, default=""):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            return stream.read()
    except OSError:
        return default


def _parse_range_list(value):
    result = []
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise ValueError("descending range")
            result.extend(range(start, end + 1))
        else:
            result.append(int(item))
    return sorted(set(result))


def _format_range_list(values):
    values = sorted(set(int(value) for value in values))
    groups = []
    for value in values:
        if not groups or value != groups[-1][1] + 1:
            groups.append([value, value])
        else:
            groups[-1][1] = value
    return ",".join(
        str(start) if start == end else "%d-%d" % (start, end)
        for start, end in groups
    )


def _status_allowed_list(field, fallback):
    """Read the kernel's effective task mask from ``/proc/self/status``."""
    status = _read_text("/proc/self/status")
    match = re.search(r"^%s:\s*(.*?)\s*$" % re.escape(field), status, re.MULTILINE)
    if match:
        try:
            return _parse_range_list(match.group(1))
        except (TypeError, ValueError):
            pass
    return sorted(set(int(value) for value in fallback))


def _thread_sibling_groups(cpus):
    """Return physical-core sibling groups clipped to the supplied CPU mask."""
    remaining = set(int(cpu) for cpu in cpus)
    groups = []
    while remaining:
        cpu = min(remaining)
        siblings_text = _read_text(
            "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % cpu,
            str(cpu),
        )
        try:
            siblings = set(_parse_range_list(siblings_text)) & set(cpus)
        except ValueError:
            siblings = {cpu}
        if not siblings:
            siblings = {cpu}
        groups.append(sorted(siblings))
        remaining.difference_update(siblings)
    return groups


def _meminfo(path="/proc/meminfo"):
    values = {}
    for line in _read_text(path).splitlines():
        match = re.match(r"^([^:]+):\s*(\d+)(?:\s+kB)?", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    return values


def _mountinfo_unescape(value):
    """Decode the octal escapes used for paths in ``/proc/*/mountinfo``."""
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _cgroup_mounts(mountinfo):
    """Return normalized cgroup mount records from one mountinfo snapshot."""
    records = []
    for line in mountinfo.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
            root = _mountinfo_unescape(fields[3])
            mountpoint = _mountinfo_unescape(fields[4])
            mount_options = fields[5].split(",")
            filesystem = fields[separator + 1]
            source = fields[separator + 2]
            super_options = fields[separator + 3].split(",")
        except (IndexError, ValueError):
            continue
        if filesystem not in ("cgroup", "cgroup2"):
            continue
        records.append(
            {
                "filesystem": filesystem,
                "root": os.path.normpath(root),
                "mountpoint": os.path.normpath(mountpoint),
                "source": source,
                "mount_options": mount_options,
                "optional_fields": fields[6:separator],
                "super_options": super_options,
            }
        )
    return records


def _cgroup_memberships(cgroup_text):
    """Parse v1 controller paths and the v2 unified path."""
    memberships = {"v1": {}, "v2": None}
    for line in cgroup_text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, path = fields
        if not path.startswith("/"):
            continue
        normalized = os.path.normpath(path)
        if not controllers:
            memberships["v2"] = normalized
        else:
            for controller in controllers.split(","):
                if controller:
                    memberships["v1"][controller] = normalized
    return memberships


def _resolve_cgroup_directory(membership, mounts, filesystem, controller=None):
    """Map a membership path to the most-specific visible cgroup mount."""
    candidates = []
    for mount in mounts:
        if mount["filesystem"] != filesystem:
            continue
        controller_options = set(mount["super_options"]) | set(
            mount["mount_options"]
        ) | set(mount["source"].split(",")) | set(mount["optional_fields"])
        if controller is not None and controller not in controller_options:
            continue
        root = mount["root"]
        explicit_root = membership == root or membership.startswith(
            root.rstrip("/") + "/"
        )
        # A cgroup namespace reports /proc/self/cgroup relative to its namespace
        # root, while mountinfo may retain the underlying cgroupfs mount root.
        # In that case `/` maps directly to the visible mountpoint.
        relative = (
            os.path.relpath(membership, root)
            if explicit_root
            else membership.lstrip("/") or "."
        )
        resolved = os.path.normpath(os.path.join(mount["mountpoint"], relative))
        try:
            contained = os.path.commonpath([resolved, mount["mountpoint"]]) == mount["mountpoint"]
        except ValueError:
            contained = False
        if contained:
            candidates.append((int(explicit_root), len(root), mount, resolved))
    if not candidates:
        return None, None
    _, _, mount, resolved = max(candidates, key=lambda item: item[:2])
    return mount, resolved


def _cgroup_ancestors(path, mountpoint):
    """Yield a cgroup and every visible ancestor through its mount root."""
    current = os.path.normpath(path)
    root = os.path.normpath(mountpoint)
    while True:
        try:
            if os.path.commonpath([current, root]) != root:
                raise ValueError("cgroup path escaped its mount")
        except ValueError:
            raise RamdiskError("resolved cgroup path is outside its controller mount")
        yield current
        if current == root:
            break
        parent = os.path.dirname(current)
        if parent == current:
            raise RamdiskError("cgroup ancestry did not reach its controller mount")
        current = parent


def _read_cgroup_value(path):
    """Read one controller file, distinguishing absence from access failure."""
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as stream:
            return stream.read().strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RamdiskError("cannot read cgroup controller file %s: %s" % (path, exc))


def _read_cgroup_contract(path):
    """Read a procfs cgroup contract without treating denial as absence."""
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as stream:
            return stream.read()
    except (OSError, UnicodeError) as exc:
        raise RamdiskError("cannot read cgroup contract %s: %s" % (path, exc))


def _parse_cgroup_bytes(value, path, unlimited_word=False, v1_unlimited=False):
    if value is None:
        return None
    if unlimited_word and value == "max":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise RamdiskError("invalid cgroup memory value in %s" % path)
    if parsed < 0:
        raise RamdiskError("negative cgroup memory value in %s" % path)
    # cgroup v1 represents an unlimited limit with a page-aligned value just
    # below signed LONG_MAX (commonly 9223372036854771712).
    if v1_unlimited and parsed >= (1 << 60):
        return None
    return parsed


def _discover_cgroup_memory(cgroup_text=None, mountinfo_text=None):
    """Return hard/high headroom across every limiting cgroup ancestor.

    A child limit is not the only relevant boundary: an ancestor can have less
    remaining headroom because sibling workloads are charged to it.  Compare
    ``limit - current`` at every level and retain the tightest value.
    """
    result = {
        "version": None,
        "status": "none",
        "path": None,
        "mountpoint": None,
        "limit_bytes": None,
        "current_bytes": None,
        "available_bytes": None,
        "limiting_path": None,
        "high_bytes": None,
        "high_available_bytes": None,
        "high_limiting_path": None,
        "error": None,
    }
    if not sys.platform.startswith("linux"):
        return result
    try:
        if cgroup_text is None:
            cgroup_text = _read_cgroup_contract("/proc/self/cgroup")
        if mountinfo_text is None:
            mountinfo_text = _read_cgroup_contract("/proc/self/mountinfo")
    except RamdiskError as exc:
        result.update({"status": "unavailable", "error": str(exc)})
        return result
    memberships = _cgroup_memberships(cgroup_text)
    mounts = _cgroup_mounts(mountinfo_text)
    version = None
    membership = None
    mount = resolved = None
    if memberships["v2"] is not None:
        version = 2
        membership = memberships["v2"]
        mount, resolved = _resolve_cgroup_directory(
            membership, mounts, "cgroup2"
        )
        # A hybrid host can expose an empty/systemd-only v2 hierarchy while
        # the memory controller remains on v1. Prefer v1 when the resolved v2
        # ancestry has no memory controller files at all.
        v2_memory_visible = (
            mount is not None
            and resolved is not None
            and any(
                os.path.exists(os.path.join(ancestor, leaf))
                for ancestor in _cgroup_ancestors(resolved, mount["mountpoint"])
                for leaf in ("memory.current", "memory.max", "memory.high")
            )
        )
        if "memory" in memberships["v1"] and not v2_memory_visible:
            v1_membership = memberships["v1"]["memory"]
            v1_mount, v1_resolved = _resolve_cgroup_directory(
                v1_membership, mounts, "cgroup", controller="memory"
            )
            if v1_mount is not None and v1_resolved is not None:
                version = 1
                membership = v1_membership
                mount, resolved = v1_mount, v1_resolved
    elif "memory" in memberships["v1"]:
        version = 1
        membership = memberships["v1"]["memory"]
        mount, resolved = _resolve_cgroup_directory(
            membership, mounts, "cgroup", controller="memory"
        )
    if version is None:
        return result
    result.update({"version": version, "path": membership})
    if mount is None or resolved is None:
        result.update(
            {
                "status": "unavailable",
                "error": "memory cgroup membership has no visible controller mount",
            }
        )
        return result
    result["mountpoint"] = mount["mountpoint"]
    try:
        for ancestor in _cgroup_ancestors(resolved, mount["mountpoint"]):
            if version == 2:
                limit_path = os.path.join(ancestor, "memory.max")
                current_path = os.path.join(ancestor, "memory.current")
                high_path = os.path.join(ancestor, "memory.high")
                limit = _parse_cgroup_bytes(
                    _read_cgroup_value(limit_path), limit_path, unlimited_word=True
                )
                high = _parse_cgroup_bytes(
                    _read_cgroup_value(high_path), high_path, unlimited_word=True
                )
            else:
                limit_path = os.path.join(ancestor, "memory.limit_in_bytes")
                current_path = os.path.join(ancestor, "memory.usage_in_bytes")
                limit = _parse_cgroup_bytes(
                    _read_cgroup_value(limit_path),
                    limit_path,
                    v1_unlimited=True,
                )
                high = None
                high_path = None
            if limit is None and high is None:
                continue
            current = _parse_cgroup_bytes(
                _read_cgroup_value(current_path), current_path
            )
            if current is None:
                raise RamdiskError(
                    "cgroup memory limit is visible but usage is unavailable at %s"
                    % ancestor
                )
            if limit is not None:
                available = max(0, limit - current)
                if (
                    result["available_bytes"] is None
                    or available < result["available_bytes"]
                ):
                    result.update(
                        {
                            "limit_bytes": limit,
                            "current_bytes": current,
                            "available_bytes": available,
                            "limiting_path": ancestor,
                        }
                    )
            if high is not None:
                high_available = max(0, high - current)
                if (
                    result["high_available_bytes"] is None
                    or high_available < result["high_available_bytes"]
                ):
                    result.update(
                        {
                            "high_bytes": high,
                            "high_available_bytes": high_available,
                            "high_limiting_path": ancestor,
                        }
                    )
        result["status"] = (
            "limited"
            if result["available_bytes"] is not None
            or result["high_available_bytes"] is not None
            else "unlimited"
        )
    except RamdiskError as exc:
        result.update({"status": "unavailable", "error": str(exc)})
    return result


def _cgroup_available_memory():
    """Read current hard cgroup headroom, failing closed on a broken contract."""
    cgroup = _discover_cgroup_memory()
    if cgroup.get("error"):
        raise RamdiskError(
            "cannot validate cgroup memory headroom: %s" % cgroup["error"]
        )
    return cgroup.get("available_bytes")


def _node_meminfo(node):
    values = {}
    path = "/sys/devices/system/node/node%d/meminfo" % node
    for line in _read_text(path).splitlines():
        match = re.search(r"Node\s+\d+\s+([^:]+):\s*(\d+)\s+kB", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    return values


def _physical_cores(cpus):
    cores = set()
    for cpu in cpus:
        base = "/sys/devices/system/cpu/cpu%d/topology" % cpu
        package = _read_text(os.path.join(base, "physical_package_id"), "0").strip()
        core = _read_text(os.path.join(base, "core_id"), str(cpu)).strip()
        cores.add((package, core))
    return max(1, len(cores))


def _kernel_at_least(major, minor):
    match = re.match(r"^(\d+)\.(\d+)", platform.release())
    return bool(match and (int(match.group(1)), int(match.group(2))) >= (major, minor))


def discover_hardware():
    """Return dependency-free Linux, NUMA, tmpfs, THP, and swap discovery."""
    linux = sys.platform.startswith("linux")
    online_text = _read_text("/sys/devices/system/node/online", "0")
    try:
        online = _parse_range_list(online_text)
    except ValueError:
        online = [0]
    if not online:
        online = [0]
    nodes = []
    all_cpus = []
    for node in online:
        cpus_text = _read_text(
            "/sys/devices/system/node/node%d/cpulist" % node,
            _read_text("/sys/devices/system/cpu/online", "0"),
        )
        try:
            cpus = _parse_range_list(cpus_text)
        except ValueError:
            cpus = []
        all_cpus.extend(cpus)
        memory = _node_meminfo(node)
        distance = []
        for word in _read_text(
            "/sys/devices/system/node/node%d/distance" % node
        ).split():
            try:
                distance.append(int(word))
            except ValueError:
                pass
        nodes.append(
            {
                "id": node,
                "cpus": cpus,
                "cpu_list": cpus_text.strip(),
                "physical_cores": _physical_cores(cpus),
                "memory_total_bytes": memory.get("MemTotal", 0),
                "memory_available_bytes": memory.get(
                    "MemFree", memory.get("MemAvailable", 0)
                ),
                "distance": distance,
            }
        )
    all_cpus = sorted(set(all_cpus))
    try:
        affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = _status_allowed_list("Cpus_allowed_list", all_cpus)
    effective_cpus = sorted(set(affinity) & set(all_cpus))
    effective_nodes = sorted(
        set(_status_allowed_list("Mems_allowed_list", online)) & set(online)
    )
    core_groups = _thread_sibling_groups(effective_cpus)
    for node in nodes:
        node["effective_cpus"] = sorted(
            set(node["cpus"]) & set(effective_cpus)
        )
        node["effective_cpu_list"] = _format_range_list(node["effective_cpus"])
    memory = _meminfo()
    swaps = []
    swap_text = _read_text("/proc/swaps")
    for line in swap_text.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 5:
            swaps.append(
                {
                    "path": fields[0],
                    "type": fields[1],
                    "size_bytes": int(fields[2]) * 1024,
                    "used_bytes": int(fields[3]) * 1024,
                }
            )
    shmem_enabled = _read_text(
        "/sys/kernel/mm/transparent_hugepage/shmem_enabled"
    ).strip()
    thp_modes = re.findall(r"\[?([A-Za-z_]+)\]?", shmem_enabled)
    filesystems = _read_text("/proc/filesystems")
    cgroup_memory = _discover_cgroup_memory()
    return {
        "linux": linux,
        "kernel_release": platform.release(),
        "online_nodes": online,
        "effective_nodes": effective_nodes,
        "effective_cpus": effective_cpus,
        "effective_cpu_list": _format_range_list(effective_cpus),
        "effective_mask_source": "kernel-task-status",
        "core_groups": core_groups,
        "nodes": nodes,
        "physical_cores": _physical_cores(all_cpus),
        "effective_physical_cores": len(core_groups),
        "memory": {
            "total_bytes": memory.get("MemTotal", 0),
            "available_bytes": memory.get("MemAvailable", memory.get("MemFree", 0)),
        },
        "cgroup_memory": cgroup_memory,
        "swap": {
            "configured": swaps,
            "used_bytes": sum(item["used_bytes"] for item in swaps),
        },
        "tmpfs": {
            "supported": any(line.strip().endswith("tmpfs") for line in filesystems.splitlines()),
            # tmpfs noswap was merged in Linux 6.4.  Preparation still treats a
            # mount(8) rejection as authoritative and never changes global swap.
            "noswap_supported": linux and _kernel_at_least(6, 4),
        },
        "thp": {
            "shmem_enabled": shmem_enabled,
            "modes": sorted(set(thp_modes)),
            "within_size_supported": "within_size" in thp_modes,
            "advise_supported": "advise" in thp_modes or bool(shmem_enabled),
        },
        "numactl": shutil.which("numactl"),
        "mount": shutil.which("mount"),
        "umount": shutil.which("umount"),
        "sudo": shutil.which("sudo"),
        "hugetlb": {
            "total_pages": memory.get("HugePages_Total", 0) // 1024,
            "free_pages": memory.get("HugePages_Free", 0) // 1024,
            "page_size_bytes": memory.get("Hugepagesize", 0),
        },
    }


def _read_safetensors_header(path):
    size = os.path.getsize(path)
    with open(path, "rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise RamdiskError("truncated safetensors file: %s" % path)
        header_size = struct.unpack("<Q", raw)[0]
        if header_size <= 1 or header_size > MAX_ST_HEADER or header_size > size - 8:
            raise RamdiskError("invalid safetensors header length in %s" % path)
        raw_header = stream.read(header_size)
        if len(raw_header) != header_size:
            raise RamdiskError("truncated safetensors header: %s" % path)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RamdiskError("invalid safetensors header in %s: %s" % (path, exc))
    if not isinstance(header, dict):
        raise RamdiskError("safetensors header is not an object: %s" % path)
    data_start = 8 + header_size
    tensors = {}
    for name, record in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(record, dict) or "data_offsets" not in record:
            raise RamdiskError("invalid tensor record %r in %s" % (name, path))
        offsets = record["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or data_start + offsets[1] > size
        ):
            raise RamdiskError("invalid tensor offsets for %s in %s" % (name, path))
        tensors[name] = {
            "dtype": record.get("dtype"),
            "shape": record.get("shape"),
            "offset": data_start + offsets[0],
            "bytes": offsets[1] - offsets[0],
        }
    return raw_header, tensors


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(8 * MIB)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _shape_numel(shape):
    if not isinstance(shape, list) or not shape or not all(isinstance(value, int) and value >= 0 for value in shape):
        return None
    result = 1
    for value in shape:
        result *= value
    return result


def _direct_tensor_set_eligible(entry, config):
    hidden = int(config["hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    prefix = "model.layers.%d.mlp.experts.%d." % (entry["layer"], entry["expert"])
    for projection, rows, columns in (
        ("gate_proj", intermediate, hidden),
        ("up_proj", intermediate, hidden),
        ("down_proj", hidden, intermediate),
    ):
        weight = entry["tensors"][prefix + projection + ".weight"]
        scale = entry["tensors"][prefix + projection + ".weight.qs"]
        if (
            weight["dtype"] not in ("U8", "I8")
            or scale["dtype"] != "F32"
            or weight["offset"] % 4
            or scale["offset"] % 4
            or _shape_numel(weight["shape"]) != weight["bytes"]
            or _shape_numel(scale["shape"]) != scale["bytes"] // 4
            or scale["bytes"] % 4
        ):
            return False
        weight_bytes = weight["bytes"]
        scale_values = scale["bytes"] // 4
        int8_bytes = rows * columns
        int4_bytes = rows * ((columns + 1) // 2)
        int2_bytes = rows * ((columns + 3) // 4)
        if weight_bytes == int8_bytes or weight_bytes == int2_bytes:
            if scale_values != rows:
                return False
        elif weight_bytes == int4_bytes:
            valid_scales = {rows}
            for group_size in (16, 32, 48, 64, 96, 128, 192, 256):
                if group_size <= columns:
                    valid_scales.add(rows * ((columns + group_size - 1) // group_size))
            if scale_values not in valid_scales:
                return False
        else:
            return False
    return True


def scan_model(model_dir):
    """Index shards and each expert's complete six-tensor direct-map closure."""
    model_dir = os.path.realpath(os.path.abspath(os.path.expanduser(model_dir)))
    if not os.path.isdir(model_dir):
        raise RamdiskError("model directory not found: %s" % model_dir)
    names = sorted(name for name in os.listdir(model_dir) if name.endswith(".safetensors"))
    if not names:
        raise RamdiskError("no .safetensors shards found in %s" % model_dir)
    fingerprint = hashlib.sha256()
    identity_files = {}
    required_metadata = ("config.json", "tokenizer.json")
    optional_metadata = ("generation_config.json", "tokenizer_config.json")
    for name in required_metadata + optional_metadata:
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            if name in required_metadata:
                raise RamdiskError("required model metadata is missing: %s" % path)
            continue
        digest = _sha256_file(path)
        size = os.path.getsize(path)
        identity_files[name] = {"size_bytes": size, "sha256": digest}
        fingerprint.update(("metadata\0%s\0%d\0%s\n" % (name, size, digest)).encode("utf-8"))
    config_path = os.path.join(model_dir, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, ValueError) as exc:
        raise RamdiskError("cannot parse %s: %s" % (config_path, exc))
    if not isinstance(config, dict):
        raise RamdiskError("config.json must contain a JSON object")
    required_positive = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "n_routed_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "intermediate_size",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "n_shared_experts",
        "vocab_size",
    )
    missing = [name for name in required_positive if not isinstance(config.get(name), int) or config[name] <= 0]
    if missing:
        raise RamdiskError("config.json is missing positive engine fields: %s" % ", ".join(missing))
    shards = []
    experts = {}
    tensor_bytes = 0
    expert_tensor_bytes = 0
    for name in names:
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            raise RamdiskError("shard is not a regular file: %s" % path)
        st = os.stat(path, follow_symlinks=True)
        raw_header, tensors = _read_safetensors_header(path)
        header_digest = hashlib.sha256(raw_header).hexdigest()
        identity = "%s\0%d\0%d\0%d\0%s\n" % (
            name,
            st.st_size,
            st.st_mtime_ns,
            st.st_ino,
            header_digest,
        )
        fingerprint.update(identity.encode("utf-8"))
        shards.append(
            {
                "name": name,
                "path": path,
                "size_bytes": st.st_size,
                "device": st.st_dev,
                "inode": st.st_ino,
                "mtime_ns": st.st_mtime_ns,
                "header_sha256": header_digest,
                "tensor_count": len(tensors),
            }
        )
        for tensor_name, tensor in tensors.items():
            tensor_bytes += tensor["bytes"]
            match = EXPERT_RE.match(tensor_name)
            if not match:
                continue
            layer, expert = int(match.group(1)), int(match.group(2))
            key = "%d:%d" % (layer, expert)
            entry = experts.setdefault(
                key,
                {
                    "layer": layer,
                    "expert": expert,
                    "tensors": {},
                    "shards": set(),
                    "tensor_bytes": 0,
                },
            )
            entry["tensors"][tensor_name] = {
                "shard": name,
                "bytes": tensor["bytes"],
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "offset": tensor["offset"],
            }
            entry["shards"].add(name)
            entry["tensor_bytes"] += tensor["bytes"]
            expert_tensor_bytes += tensor["bytes"]
    complete = {}
    for key, entry in experts.items():
        prefix = "model.layers.%d.mlp.experts.%d." % (entry["layer"], entry["expert"])
        expected = set()
        for projection in ("gate_proj", "up_proj", "down_proj"):
            weight = prefix + projection + ".weight"
            expected.add(weight)
            expected.add(weight + ".qs")
        if expected == set(entry["tensors"]):
            entry["shards"] = sorted(entry["shards"])
            entry["direct_map_eligible"] = _direct_tensor_set_eligible(entry, config)
            complete[key] = entry
    total_bytes = sum(shard["size_bytes"] for shard in shards)
    return {
        "path": model_dir,
        "fingerprint": "sha256:" + fingerprint.hexdigest(),
        "fingerprint_algorithm": "metadata content plus sorted shard name,size,mtime,inode,header-sha256",
        "identity_files": identity_files,
        "shards": shards,
        "shard_names": names,
        "total_shard_bytes": total_bytes,
        "tensor_bytes": tensor_bytes,
        "dense_tensor_bytes": max(0, tensor_bytes - expert_tensor_bytes),
        "experts": complete,
        "complete_experts": len(complete),
        "config": config,
    }


def _load_profile(path, model):
    if not path:
        path = os.path.join(model["path"], ".coli_usage")
    if not os.path.isfile(path):
        raise RamdiskError(
            "partial staging requires .coli_usage or an explicit compatible --profile"
        )
    counts = {}
    fingerprint = None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            text = stream.read()
        if path.endswith(".json") or text.lstrip().startswith("{"):
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise RamdiskError("profile JSON must contain an object")
            fingerprint = payload.get("model_fingerprint")
            rows = payload.get("counts", [])
            if not isinstance(rows, list):
                raise RamdiskError("profile JSON counts must be a list")
            for row in rows:
                if isinstance(row, dict):
                    layer, expert, count = row.get("layer"), row.get("expert"), row.get("count")
                else:
                    layer, expert, count = row
                counts["%d:%d" % (int(layer), int(expert))] = int(count)
        else:
            for number, line in enumerate(text.splitlines(), 1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                match = PROFILE_LINE_RE.match(line)
                if not match:
                    raise RamdiskError("invalid profile line %d in %s" % (number, path))
                layer, expert, count = (int(value) for value in match.groups())
                counts["%d:%d" % (layer, expert)] = count
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, RamdiskError):
            raise
        raise RamdiskError("cannot parse profile %s: %s" % (path, exc))
    if fingerprint and fingerprint != model["fingerprint"]:
        raise RamdiskError("profile model fingerprint does not match the selected model")
    compatible = {key: count for key, count in counts.items() if key in model["experts"] and count > 0}
    if not compatible:
        raise RamdiskError("profile contains no experts compatible with this model")
    return os.path.realpath(path), compatible


def _select_partial(model, counts, budget_bytes):
    shard_sizes = {item["name"]: item["size_bytes"] for item in model["shards"]}
    # Experts commonly share the same one- or two-shard closure.  Grouping those
    # closures makes the greedy score both exact and cheap: each candidate gets
    # credit for every newly completed profiled expert, not just the expert that
    # happened to nominate the shard set.
    closure_groups = {}
    for key in sorted(counts):
        closure = frozenset(model["experts"][key]["shards"])
        group = closure_groups.setdefault(closure, {"keys": [], "benefit": 0})
        group["keys"].append(key)
        group["benefit"] += counts[key] * model["experts"][key]["tensor_bytes"]
    selected = set()
    covered_closures = set()
    while True:
        candidates = []
        for closure in sorted(closure_groups, key=lambda value: tuple(sorted(value))):
            if closure in covered_closures:
                continue
            trial = selected | set(closure)
            added = trial - selected
            cost = sum(shard_sizes[name] for name in added)
            if not added or sum(shard_sizes[name] for name in selected) + cost > budget_bytes:
                continue
            newly_covered = [
                other
                for other in closure_groups
                if other not in covered_closures and other.issubset(trial)
            ]
            benefit = sum(closure_groups[item]["benefit"] for item in newly_covered)
            ratio = float(benefit) / float(cost)
            # The sorted closure tuple is the final deterministic tie-breaker.
            candidates.append((ratio, benefit, -cost, tuple(sorted(closure)), trial))
        if not candidates:
            break
        _, _, _, _, selected = max(candidates)
        covered_closures = {
            closure for closure in closure_groups if closure.issubset(selected)
        }
    staged_experts = sorted(
        key
        for key, expert in model["experts"].items()
        if set(expert["shards"]).issubset(selected) and expert["direct_map_eligible"]
    )
    return sorted(selected), staged_experts


def _runtime_reserve(model, ctx, direct_experts, cache_cap=8, kv_slots=1):
    config = model.get("config") or {}
    layers = int(config.get("num_hidden_layers", 0) or 0)
    kv_lora = int(config.get("kv_lora_rank", 0) or 0)
    rope = int(config.get("qk_rope_head_dim", 0) or 0)
    index_dim = int(config.get("index_head_dim", 0) or 0)
    qk_nope = int(config.get("qk_nope_head_dim", 0) or 0)
    v_head = int(config.get("v_head_dim", 0) or 0)
    heads = int(config.get("num_attention_heads", 0) or 0)
    kv_bytes = (layers + 1) * max(1, ctx) * (kv_lora + rope) * 4 * kv_slots
    index_bytes = layers * max(1, ctx) * index_dim * 4 * kv_slots
    attention_scratch = max(1, ctx) * heads * (qk_nope + v_head) * 4
    dense = model["dense_tensor_bytes"]
    direct = set(direct_experts)
    fallback_by_layer = {}
    for key, expert in model["experts"].items():
        if key not in direct:
            fallback_by_layer.setdefault(expert["layer"], []).append(expert["tensor_bytes"])
    fallback_cache = sum(
        min(cache_cap, len(sizes)) * max(sizes)
        for sizes in fallback_by_layer.values()
        if sizes
    )
    max_expert = max((entry["tensor_bytes"] for entry in model["experts"].values()), default=0)
    working_set = min(64, max((len(sizes) for sizes in fallback_by_layer.values()), default=0)) * max_expert
    engine_overhead = max(int(1.2e9), dense // 100)
    return {
        "dense_bytes": dense,
        "kv_bytes": kv_bytes,
        "index_bytes": index_bytes,
        "attention_scratch_bytes": attention_scratch,
        "fallback_cache_bytes": fallback_cache,
        "working_set_bytes": working_set,
        "engine_overhead_bytes": engine_overhead,
    }


def _requested_ids(value, label, allowed, default):
    """Normalize an operator range list without ever widening its effective mask."""
    allowed = sorted(set(int(item) for item in allowed))
    if value is None or value == "":
        selected = sorted(set(int(item) for item in default))
    elif isinstance(value, str):
        if len(value) > 4096:
            raise RamdiskError("%s range list is unreasonably long" % label)
        for token in value.split(","):
            token = token.strip()
            match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
            if not match:
                raise RamdiskError("%s must be a CPU/NUMA range list such as 0-3,8" % label)
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if end < start:
                raise RamdiskError("%s contains a descending range" % label)
            if allowed and (start > allowed[-1] or end > allowed[-1]):
                raise RamdiskError("%s requests IDs outside the effective host mask" % label)
        selected = _parse_range_list(value)
    elif isinstance(value, (list, tuple, set)):
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        ):
            raise RamdiskError("%s must contain non-negative integer IDs" % label)
        selected = sorted(set(value))
    else:
        raise RamdiskError("%s must be a CPU/NUMA range list" % label)
    if not selected:
        raise RamdiskError("%s resolves to an empty effective mask" % label)
    outside = sorted(set(selected) - set(allowed))
    if outside:
        raise RamdiskError(
            "%s requests IDs outside the effective host mask: %s"
            % (label, _format_range_list(outside))
        )
    return selected


def _build_placement(args, hardware, topology):
    """Resolve selected memory nodes and whole-core CPU masks for one plan."""
    online_nodes = sorted(set(int(node) for node in hardware.get("online_nodes", [])))
    node_rows = {
        int(node["id"]): node
        for node in hardware.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), int)
    }
    all_cpus = sorted(
        {
            int(cpu)
            for node in node_rows.values()
            for cpu in node.get("cpus", [])
            if isinstance(cpu, int) and not isinstance(cpu, bool) and cpu >= 0
        }
    )
    effective_nodes = sorted(
        set(int(node) for node in hardware.get("effective_nodes", online_nodes))
        & set(online_nodes)
    )
    effective_cpus = sorted(
        set(int(cpu) for cpu in hardware.get("effective_cpus", all_cpus))
        & set(all_cpus)
    )
    if not effective_nodes:
        raise RamdiskError("the effective cpuset exposes no NUMA memory nodes")
    if not effective_cpus:
        raise RamdiskError("the effective cpuset exposes no CPUs")

    default_nodes = [
        node
        for node in effective_nodes
        if set(node_rows.get(node, {}).get("cpus", [])) & set(effective_cpus)
    ] or effective_nodes
    memory_nodes = _requested_ids(
        getattr(args, "memory_nodes", None),
        "--memory-nodes",
        effective_nodes,
        default_nodes,
    )
    missing_rows = sorted(set(memory_nodes) - set(node_rows))
    if missing_rows:
        raise RamdiskError(
            "hardware discovery has no details for selected NUMA node(s): %s"
            % _format_range_list(missing_rows)
        )
    default_cpus = sorted(
        set(effective_cpus)
        & {
            int(cpu)
            for node in memory_nodes
            for cpu in node_rows.get(node, {}).get("cpus", [])
        }
    ) or effective_cpus
    cpus = _requested_ids(
        getattr(args, "cpu_list", None),
        "--cpu-list",
        effective_cpus,
        default_cpus,
    )
    memory_node_cpus = {
        int(cpu)
        for node in memory_nodes
        for cpu in node_rows.get(node, {}).get("cpus", [])
    }
    remote_cpus = sorted(set(cpus) - memory_node_cpus)
    if topology == "per-node" and remote_cpus:
        raise RamdiskError(
            "per-node --cpu-list includes CPUs outside the selected replica nodes: %s"
            % _format_range_list(remote_cpus)
        )

    raw_groups = hardware.get("core_groups") or [[cpu] for cpu in effective_cpus]
    core_groups = []
    covered = set()
    for raw_group in raw_groups:
        group = sorted(set(int(cpu) for cpu in raw_group) & set(effective_cpus))
        if group and not (set(group) & covered):
            core_groups.append(group)
            covered.update(group)
    core_groups.extend([[cpu] for cpu in effective_cpus if cpu not in covered])
    selected = set(cpus)
    split_groups = [
        group
        for group in core_groups
        if selected.intersection(group) and not set(group).issubset(selected)
    ]
    if split_groups:
        raise RamdiskError(
            "--cpu-list must select whole effective physical cores; split sibling group(s): %s"
            % ", ".join(_format_range_list(group) for group in split_groups)
        )

    engine_cpu_sets = []
    if topology == "interleaved":
        targets = [(None, cpus)]
    else:
        targets = [
            (
                node,
                sorted(set(cpus) & set(node_rows.get(node, {}).get("cpus", []))),
            )
            for node in memory_nodes
        ]
    for node, engine_cpus in targets:
        physical_cores = sum(
            1 for group in core_groups if set(group).issubset(set(engine_cpus))
        )
        engine_cpu_sets.append(
            {
                "node": node,
                "cpus": engine_cpus,
                "cpu_list": _format_range_list(engine_cpus),
                "physical_cores": physical_cores,
            }
        )
    return {
        "memory_nodes": memory_nodes,
        "memory_node_list": _format_range_list(memory_nodes),
        "cpus": cpus,
        "cpu_list": _format_range_list(cpus),
        "engine_cpu_sets": engine_cpu_sets,
        "effective_nodes": effective_nodes,
        "effective_node_list": _format_range_list(effective_nodes),
        "effective_cpus": effective_cpus,
        "effective_cpu_list": _format_range_list(effective_cpus),
        "remote_cpus": remote_cpus,
        "remote_cpu_list": _format_range_list(remote_cpus),
        "memory_policy": (
            "equal-interleave"
            if topology == "interleaved" and len(memory_nodes) > 1
            else "strict-bind"
        ),
        "dimm_control": "informational-only",
    }


def build_plan(args, hardware=None, model=None):
    hardware = copy.deepcopy(hardware or discover_hardware())
    model = model or scan_model(args.model)
    mode = getattr(args, "mode", "full")
    topology = getattr(args, "topology", "interleaved")
    capacity_gb = getattr(args, "capacity_gb", None)
    if mode not in ("full", "partial") or topology not in ("interleaved", "per-node"):
        raise RamdiskError("invalid RAM-disk mode or topology")
    placement = _build_placement(args, hardware, topology)
    if capacity_gb is not None and (
        isinstance(capacity_gb, bool)
        or not isinstance(capacity_gb, (int, float))
        or not math.isfinite(capacity_gb)
        or capacity_gb <= 0
    ):
        raise RamdiskError("--capacity-gb must be a finite positive number")
    raw_ctx = getattr(args, "ctx", 0)
    if isinstance(raw_ctx, bool) or not isinstance(raw_ctx, int) or raw_ctx < 0:
        raise RamdiskError("--ctx must be zero (default) or a positive integer")
    raw_parallel = getattr(args, "parallel", 2)
    if (
        isinstance(raw_parallel, bool)
        or not isinstance(raw_parallel, int)
        or not 1 <= raw_parallel <= 64
    ):
        raise RamdiskError("--parallel must be an integer between 1 and 64")
    capacity_bytes = int(capacity_gb * GIB) if capacity_gb is not None else model["total_shard_bytes"]
    profile_path = None
    counts = None
    if mode == "full":
        selected = list(model["shard_names"])
        direct_experts = sorted(
            key for key, entry in model["experts"].items() if entry["direct_map_eligible"]
        )
    else:
        if not capacity_gb or capacity_gb <= 0:
            raise RamdiskError("partial staging requires a positive --capacity-gb budget")
        profile_path, counts = _load_profile(getattr(args, "profile", None), model)
        selected, direct_experts = _select_partial(model, counts, capacity_bytes)
        if not selected:
            raise RamdiskError("no complete shard closure fits the partial staging budget")
    resident_experts = sorted(
        key
        for key, expert in model["experts"].items()
        if set(expert["shards"]).issubset(selected)
    )
    shard_sizes = {item["name"]: item["size_bytes"] for item in model["shards"]}
    staged_bytes = sum(shard_sizes[name] for name in selected)
    managed_ctx = int(raw_ctx or 4096)
    managed_cache_cap = 8
    managed_kv_slots = 1
    managed_reserve = _runtime_reserve(
        model,
        managed_ctx,
        direct_experts,
        cache_cap=managed_cache_cap,
        kv_slots=managed_kv_slots,
    )
    # The benchmark contract includes SSD and tmpfs-through-slab baselines even
    # for a fully staged model. Those paths need the ordinary cap/LRU working
    # set in addition to the resident tmpfs copy, so admission uses the larger
    # of managed-direct and non-RAMMAP benchmark runtime projections.
    benchmark_reserve = _runtime_reserve(
        model,
        managed_ctx,
        [],
        cache_cap=managed_cache_cap,
        kv_slots=managed_kv_slots,
    )
    managed_runtime_bytes = sum(managed_reserve.values())
    benchmark_runtime_bytes = sum(benchmark_reserve.values())
    runtime_bytes = max(managed_runtime_bytes, benchmark_runtime_bytes)
    memory = hardware["memory"]
    selected_nodes = list(placement["memory_nodes"])
    selected_node_rows = [
        node for node in hardware.get("nodes", []) if node.get("id") in selected_nodes
    ]
    selected_total = sum(
        int(node.get("memory_total_bytes", 0)) for node in selected_node_rows
    ) or int(memory["total_bytes"])
    selected_available = sum(
        int(node.get("memory_available_bytes", 0)) for node in selected_node_rows
    ) or int(memory["available_bytes"])
    cgroup_memory = hardware.get("cgroup_memory") or {}
    cgroup_available = cgroup_memory.get("available_bytes")
    if (
        not isinstance(cgroup_available, int)
        or isinstance(cgroup_available, bool)
        or cgroup_available < 0
    ):
        cgroup_available = None
    cgroup_high_available = cgroup_memory.get("high_available_bytes")
    if (
        not isinstance(cgroup_high_available, int)
        or isinstance(cgroup_high_available, bool)
        or cgroup_high_available < 0
    ):
        cgroup_high_available = None
    effective_available = (
        min(selected_available, cgroup_available)
        if cgroup_available is not None
        else selected_available
    )
    global_margin = max(selected_total // 10, 16 * GIB)
    page_tables = int(math.ceil(float(staged_bytes + runtime_bytes) / 512.0))
    required_global = staged_bytes + runtime_bytes + page_tables + global_margin
    blockers = []
    warnings = []
    if cgroup_memory.get("error"):
        blockers.append(
            "cannot validate cgroup memory headroom: %s"
            % cgroup_memory["error"]
        )
    if not hardware["linux"]:
        blockers.append("coli ramdisk is supported only on Linux")
    if not hardware["tmpfs"]["supported"]:
        blockers.append("tmpfs is not available in /proc/filesystems")
    allow_swappable = bool(getattr(args, "allow_swappable", False))
    if not hardware["tmpfs"]["noswap_supported"] and not allow_swappable:
        blockers.append("this kernel does not advertise tmpfs noswap; use --allow-swappable only if accepted")
    if hardware["swap"]["used_bytes"]:
        warnings.append("swap is already in use; managed commands never run swapoff")
    if topology == "per-node" and not hardware.get("numactl"):
        blockers.append("per-node topology requires numactl")
    if topology == "per-node":
        engine_cpu_sets = {
            item["node"]: item for item in placement["engine_cpu_sets"]
        }
        for node in selected_node_rows:
            if not node.get("cpus"):
                blockers.append(
                    "NUMA node %d has no online CPUs and cannot host a node-local engine"
                    % node["id"]
                )
            elif not engine_cpu_sets[node["id"]]["cpus"]:
                blockers.append(
                    "NUMA node %d has no selected whole-core CPUs for its replica"
                    % node["id"]
                )
    if capacity_bytes < staged_bytes:
        blockers.append("selected shard closures exceed the staging budget")
    if topology == "interleaved":
        if selected_available < required_global:
            blockers.append(
                "selected NUMA nodes would breach the runtime/OS reserve"
            )
        if placement["remote_cpus"]:
            warnings.append(
                "selected engine CPUs outside the memory-node mask will perform "
                "intentional remote NUMA access: %s" % placement["remote_cpu_list"]
            )
        if len(selected_nodes) > 1:
            warnings.append(
                "Linux interleave may fall back outside the selected nodes under severe "
                "memory pressure; Colibri reserves headroom and verifies initial page placement"
            )
    else:
        for node in selected_node_rows:
            margin = max(node["memory_total_bytes"] // 10, 8 * GIB)
            node_page_tables = int(math.ceil(float(staged_bytes + runtime_bytes) / 512.0))
            required = staged_bytes + runtime_bytes + node_page_tables + margin
            node["required_bytes"] = required
            node["reserve_bytes"] = margin
            if node["memory_available_bytes"] < required:
                blockers.append("NUMA node %d cannot hold its replica and reserve" % node["id"])
    if mode == "partial":
        total_count = sum(counts.values())
        covered_profile = [
            key
            for key in counts
            if set(model["experts"][key]["shards"]).issubset(selected)
        ]
        staged_count = sum(counts[key] for key in covered_profile)
        coverage = float(staged_count) / total_count if total_count else 0.0
        predicted_avoided = sum(
            counts[key] * model["experts"][key]["tensor_bytes"]
            for key in covered_profile
        )
        pin_selected = []
        pin_bytes = 0
        for key in sorted(counts, key=lambda item: (-counts[item], item)):
            expert_bytes = model["experts"][key]["tensor_bytes"]
            if pin_bytes + expert_bytes <= capacity_bytes:
                pin_selected.append(key)
                pin_bytes += expert_bytes
        pin_count = sum(counts[key] for key in pin_selected)
        pin_comparison = {
            "budget_bytes": capacity_bytes,
            "selected_experts": pin_selected,
            "resident_expert_bytes": pin_bytes,
            "coverage": float(pin_count) / total_count if total_count else 0.0,
        }
    else:
        covered_profile = []
        coverage = 1.0
        predicted_avoided = sum(
            model["experts"][key]["tensor_bytes"] for key in resident_experts
        )
        pin_comparison = None
    direct_bytes = sum(model["experts"][key]["tensor_bytes"] for key in direct_experts)
    resident_expert_bytes = sum(
        model["experts"][key]["tensor_bytes"] for key in resident_experts
    )
    efficiency = float(resident_expert_bytes) / staged_bytes if staged_bytes else 0.0
    mount_root = os.path.abspath(os.path.expanduser(getattr(args, "mount_root", DEFAULT_MOUNT_ROOT)))
    mount_root_preexisting = os.path.isdir(mount_root) and not os.path.islink(mount_root)
    forbidden_roots = {
        "/",
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/home",
        "/lib",
        "/lib64",
        "/mnt",
        "/opt",
        "/proc",
        "/root",
        "/run",
        "/sbin",
        "/srv",
        "/sys",
        "/tmp",
        "/usr",
        "/var",
        os.path.normpath(os.path.expanduser("~")),
    }
    if os.path.normpath(mount_root) in forbidden_roots:
        blockers.append("mount root is a protected broad directory")
    try:
        under_mnt = os.path.commonpath([mount_root, "/mnt"]) == "/mnt" and mount_root != "/mnt"
    except ValueError:
        under_mnt = False
    if not under_mnt:
        blockers.append("v1 managed mount roots must be below /mnt")
    if os.path.realpath(mount_root) != mount_root:
        blockers.append("mount root path must not traverse symbolic links")
    if os.path.lexists(mount_root):
        if os.path.islink(mount_root) or not os.path.isdir(mount_root):
            blockers.append("mount root exists but is not a real directory")
        elif os.stat(mount_root).st_mode & 0o022:
            blockers.append("existing mount root must not be group/world writable")
        elif os.geteuid() != 0 and os.access(mount_root, os.W_OK):
            blockers.append("existing mount root must not be writable by the invoking user")
        elif topology == "interleaved":
            try:
                entries = os.listdir(mount_root)
                reusable = bool(entries) and all(
                    re.fullmatch(r"node\d+", name)
                    and _reusable_empty_mountpoint(os.path.join(mount_root, name))
                    for name in entries
                )
                if entries and not reusable:
                    blockers.append("interleaved mount root must be absent or empty")
                elif reusable:
                    warnings.append(
                        "interleaved mount will temporarily cover verified empty node mountpoints from an earlier topology"
                    )
            except OSError as exc:
                blockers.append("cannot inspect mount root: %s" % exc)
    # Every existing parent below /mnt must be non-writable by this user, so a
    # second process cannot exchange a directory for a symlink between review
    # and the privileged mount(8) call. X-mount.mkdir creates absent parents.
    if under_mnt:
        parent = os.path.dirname(mount_root)
        while parent.startswith("/mnt"):
            if os.path.lexists(parent):
                if os.path.islink(parent) or not os.path.isdir(parent):
                    blockers.append("mount root has an unsafe parent: %s" % parent)
                    break
                if os.geteuid() != 0 and os.access(parent, os.W_OK):
                    blockers.append("mount root parent is writable by the invoking user: %s" % parent)
                    break
            if parent == "/mnt":
                break
            parent = os.path.dirname(parent)
    model_path = os.path.normpath(model["path"])
    try:
        if os.path.commonpath([mount_root, model_path]) in (mount_root, model_path):
            blockers.append("mount root must not contain or be contained by the canonical model")
    except ValueError:
        blockers.append("mount root and model path are on incompatible path roots")
    nodes = selected_nodes
    requested_thp = getattr(args, "thp", "auto") or "auto"
    thp = (
        "within_size" if hardware["thp"]["within_size_supported"] else "advise"
    ) if requested_thp == "auto" else requested_thp
    if thp == "within_size" and not hardware["thp"]["within_size_supported"]:
        warnings.append("THP within_size is not advertised; mount will fall back to advise if rejected")
    if thp == "advise" and not hardware["thp"]["advise_supported"]:
        blockers.append("tmpfs THP advise mode is not available")
    replicas = [None] if topology == "interleaved" else nodes
    replica_count = len(replicas)
    mounts = []
    for node in replicas:
        path = mount_root if node is None else os.path.join(mount_root, "node%d" % node)
        # A contiguous range avoids an option-separator comma in mount(8)'s
        # ``-o`` string on the overwhelmingly common 0..N online-node layout.
        node_list = _format_range_list(nodes)
        if "," in node_list:
            node_list = node_list.replace(",", "\\,")
        # Prevent ordinal remapping while reviewed nodes remain allowed.
        # Without ``static``, Linux maps the policy's ordinal nodes into a new
        # cpuset; Start/Benchmark separately refuse every effective-mask drift.
        policy = (
            "interleave=static:" + node_list
            if node is None and len(nodes) > 1
            else "bind=static:%d" % (nodes[0] if node is None else node)
        )
        mounts.append(
            {
                "node": node,
                "path": path,
                "path_preexisting": os.path.isdir(path) and not os.path.islink(path),
                "policy": policy,
                "size_bytes": max(staged_bytes + max(64 * MIB, staged_bytes // 100), 64 * MIB),
            }
        )
        if os.path.lexists(path):
            if os.path.islink(path) or not os.path.isdir(path):
                blockers.append("managed mount path exists but is not a real directory: %s" % path)
            elif os.geteuid() != 0 and os.access(path, os.W_OK):
                blockers.append("managed mount path is writable by the invoking user: %s" % path)
            else:
                try:
                    if os.listdir(path):
                        blockers.append("managed mount path is not empty: %s" % path)
                except OSError as exc:
                    blockers.append("cannot inspect managed mount path %s: %s" % (path, exc))
    durable_state = {}
    try:
        durable_state = {
            "root": _state_root(),
            "manifest": _manifest_path(),
            "benchmarks": _benchmarks_path(),
        }
        for label, durable_path in durable_state.items():
            if not _path_without_symlinks(durable_path):
                blockers.append("durable %s path must not traverse symbolic links" % label)
            if _path_is_below(durable_path, mount_root, allow_equal=True):
                blockers.append("durable %s path must be outside every volatile mount" % label)
            filesystem = _filesystem_for_path(durable_path)
            if filesystem in ("tmpfs", "ramfs"):
                blockers.append(
                    "durable %s path is on volatile %s; use an SSD-backed XDG state directory"
                    % (label, filesystem)
                )
    except (RamdiskError, OSError, subprocess.SubprocessError) as exc:
        blockers.append(str(exc))
    total_staged_bytes = staged_bytes * replica_count
    total_runtime_bytes = runtime_bytes * replica_count
    total_page_table_bytes = page_tables * replica_count
    if topology == "per-node":
        total_os_margin = sum(
            int(node.get("reserve_bytes", 0)) for node in selected_node_rows
        )
        total_required = sum(
            int(node.get("required_bytes", 0)) for node in selected_node_rows
        )
        if selected_available < total_required:
            blockers.append("available memory cannot hold all per-node replicas and reserves")
    else:
        total_os_margin = global_margin
        total_required = required_global
    if cgroup_available is not None and cgroup_available < total_required:
        blockers.append(
            "cgroup memory hard-limit headroom cannot hold the staged copies, "
            "managed runtime, and reserve"
        )
    if (
        cgroup_high_available is not None
        and cgroup_high_available < total_required
    ):
        warnings.append(
            "cgroup memory.high headroom is below the projected deployment; "
            "staging or runtime may be heavily reclaimed/throttled"
        )
    return {
        "schema": PLAN_SCHEMA,
        "version": MANIFEST_VERSION,
        "created_at": _utc_now(),
        "mode": mode,
        "topology": topology,
        "mount_root": mount_root,
        "capacity_bytes": capacity_bytes,
        "model": {
            "path": model["path"],
            "fingerprint": model["fingerprint"],
            "fingerprint_algorithm": model["fingerprint_algorithm"],
            "shard_count": len(model["shards"]),
            "total_shard_bytes": model["total_shard_bytes"],
            "complete_experts": model["complete_experts"],
        },
        "profile": {
            "path": profile_path,
            "coverage": coverage,
            "staging_efficiency": efficiency,
            "predicted_expert_bytes_avoided": predicted_avoided,
            "predicted_expert_bytes_avoided_per_staged_byte": (
                float(predicted_avoided) / staged_bytes if staged_bytes else 0.0
            ),
            "covered_experts": covered_profile,
            "pin_comparison": pin_comparison,
        },
        "staging": {
            "selected_shards": selected,
            "linked_shards": sorted(set(model["shard_names"]) - set(selected)),
            "staged_bytes": staged_bytes,
            "staged_experts": resident_experts,
            "staged_expert_count": len(resident_experts),
            "staged_expert_bytes": resident_expert_bytes,
            "direct_mapped_experts": direct_experts,
            "direct_mapped_expert_count": len(direct_experts),
            "direct_mapped_bytes": direct_bytes,
            "replica_count": replica_count,
            "total_staged_bytes": total_staged_bytes,
        },
        "reserve": {
            "runtime": managed_reserve,
            "benchmark_runtime": benchmark_reserve,
            "managed_runtime_bytes": managed_runtime_bytes,
            "benchmark_runtime_bytes": benchmark_runtime_bytes,
            "runtime_bytes": runtime_bytes,
            "page_table_bytes": page_tables,
            "os_margin_bytes": global_margin,
            "required_global_bytes": required_global,
            "available_bytes": effective_available,
            "host_available_bytes": selected_available,
            "cgroup_available_bytes": cgroup_available,
            "cgroup_high_available_bytes": cgroup_high_available,
            "total_runtime_bytes": total_runtime_bytes,
            "total_page_table_bytes": total_page_table_bytes,
            "total_os_margin_bytes": total_os_margin,
            "total_required_bytes": total_required,
        },
        "placement": placement,
        "hardware": hardware,
        "mounts": mounts,
        "mount_root_preexisting": mount_root_preexisting,
        "mount_options": {
            "noswap": hardware["tmpfs"]["noswap_supported"],
            "allow_swappable": allow_swappable,
            "thp": thp,
            "fixed": ["noatime", "nodev", "nosuid", "noexec", "mode=0700"],
        },
        "prefault": int(
            getattr(args, "prefault", None)
            if getattr(args, "prefault", None) is not None
            else mode == "full"
        ),
        "parallel": raw_parallel,
        "managed_runtime": {
            "ctx": managed_ctx,
            "kv_slots": managed_kv_slots,
            "cache_cap": managed_cache_cap,
            "autopin": 0,
            "cap_raise": 0,
        },
        "blockers": sorted(set(blockers)),
        "warnings": warnings,
        "durable_state": durable_state,
        # Internal source identities are retained in a plan used by prepare but
        # omitted by the compact human renderer only.
        "source_shards": model["shards"],
    }


def _add_lifecycle_options(parser, suppress=False):
    default = argparse.SUPPRESS if suppress else None
    # ``coli`` already supplies --model/--ctx on the outer ramdisk parser via
    # its common parent.  Add them only when this module is used standalone;
    # the action-local parser always receives suppressing copies so the same
    # options can also appear after ``plan``/``prepare`` without overwriting a
    # value parsed before the action.
    if "--model" not in parser._option_string_actions:
        parser.add_argument("--model", default=default, help="canonical model directory on durable storage")
    parser.add_argument(
        "--mode",
        choices=("full", "partial"),
        default=argparse.SUPPRESS if suppress else "full",
        help="stage the full model or profile-selected shard closures",
    )
    parser.add_argument(
        "--topology",
        choices=("interleaved", "per-node"),
        default=argparse.SUPPRESS if suppress else "interleaved",
        help=(
            "interleaved = one shared model copy and one engine; per-node = one complete "
            "copy and independent engine per NUMA node (replication, not sharding)"
        ),
    )
    parser.add_argument(
        "--memory-nodes",
        default=default,
        metavar="NODELIST",
        help=(
            "effective NUMA memory nodes (for example 0-3,8); defaults to allowed "
            "CPU-bearing nodes"
        ),
    )
    parser.add_argument(
        "--cpu-list",
        default=default,
        metavar="CPULIST",
        help=(
            "whole-core managed-engine CPUs (for example 0-15,32-47); defaults "
            "to allowed CPUs on the selected memory nodes"
        ),
    )
    parser.add_argument(
        "--capacity-gb",
        type=float,
        default=default,
        help="per-copy staging budget; required for partial mode",
    )
    parser.add_argument("--profile", default=default, help="compatible .coli_usage text or JSON profile")
    parser.add_argument(
        "--mount-root",
        default=argparse.SUPPRESS if suppress else DEFAULT_MOUNT_ROOT,
        help="managed tmpfs root below /mnt",
    )
    parser.add_argument(
        "--thp",
        choices=("auto", "within_size", "advise"),
        default=argparse.SUPPRESS if suppress else "auto",
        help="transparent huge-page policy for tmpfs",
    )
    parser.add_argument(
        "--allow-swappable",
        action="store_true",
        default=argparse.SUPPRESS if suppress else False,
        help="allow tmpfs without noswap support",
    )
    parser.add_argument(
        "--prefault",
        type=int,
        choices=(0, 1),
        default=default,
        help="touch direct mappings at engine startup",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=argparse.SUPPRESS if suppress else 2,
        help="concurrent shard-copy workers (does not create replicas)",
    )
    if "--ctx" not in parser._option_string_actions:
        parser.add_argument(
            "--ctx",
            type=int,
            default=argparse.SUPPRESS if suppress else 0,
            help="managed engine context length (0 = 4096)",
        )


def configure_parser(parser, common_parent=None):
    """Attach scriptable subcommands; options work before or after the action."""
    _add_lifecycle_options(parser, suppress=False)
    after = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    _add_lifecycle_options(after, suppress=True)
    actions = parser.add_subparsers(dest="ramdisk_action", metavar="ACTION")
    plan = actions.add_parser("plan", parents=[after], help="show an exact staging and reserve plan")
    plan.add_argument("--json", action="store_true")
    prepare_parser = actions.add_parser("prepare", parents=[after], help="mount, stage, and validate weights")
    prepare_parser.add_argument("--yes", action="store_true", help="accept the reviewed plan non-interactively")
    status_parser = actions.add_parser("status", parents=[after], help="show mounts and managed processes")
    status_parser.add_argument("--json", action="store_true")
    benchmark_parser = actions.add_parser("benchmark", parents=[after], help="run equal RAM/SSD scorecards")
    benchmark_parser.add_argument("--json", action="store_true")
    start_parser = actions.add_parser("start", parents=[after], help="start managed engine process(es)")
    start_parser.add_argument(
        "--base-port",
        type=int,
        default=None,
        help="managed base port (defaults to the prepared deployment's last value)",
    )
    actions.add_parser("stop", parents=[after], help="stop only verified managed processes")
    destroy_parser = actions.add_parser("destroy", parents=[after], help="unmount volatile weights safely")
    destroy_parser.add_argument("--yes", action="store_true")


def _json_print(value):
    print(json.dumps(value, indent=2, sort_keys=True))


def _human_plan(plan):
    placement = _placement_summary(plan)
    print("RAM-disk plan: %s" % placement["title"])
    print("  model: %s" % plan["model"]["path"])
    print("  placement: %s" % placement["cost"])
    print("  endpoints after start: %s" % placement["endpoints"])
    print(
        "  NUMA memory nodes: %s; managed engine CPUs: %s"
        % (
            plan.get("placement", {}).get("memory_node_list", "all"),
            plan.get("placement", {}).get("cpu_list", "all"),
        )
    )
    print("  DIMM/channel placement: informational only; Linux allocates by NUMA node")
    print("  %s" % placement["explanation"])
    print(
        "  staged set: %d shard(s); %d direct expert(s)"
        % (
            len(plan["staging"]["selected_shards"]),
            plan["staging"]["direct_mapped_expert_count"],
        )
    )
    print("  total staged + OS/runtime projection: %.2f GiB; available: %.2f GiB" % (
        plan["reserve"]["total_required_bytes"] / float(GIB),
        plan["reserve"]["available_bytes"] / float(GIB),
    ))
    if plan["mode"] == "partial":
        print("  profile coverage: %.1f%%; staging efficiency: %.1f%%" % (
            plan["profile"]["coverage"] * 100,
            plan["profile"]["staging_efficiency"] * 100,
        ))
        pin = plan["profile"]["pin_comparison"]
        print(
            "  same-budget hot PIN comparison: %.1f%% profile coverage with %d expert(s)"
            % (pin["coverage"] * 100, len(pin["selected_experts"]))
        )
    for warning in plan["warnings"]:
        print("  warning: %s" % warning)
    for blocker in plan["blockers"]:
        print("  BLOCKED: %s" % blocker)


def _unescape_mount(value):
    return re.sub(
        r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value
    )


def _split_mount_options(value):
    """Split mountinfo options while preserving comma-bearing mpol masks."""
    result = []
    for token in value.split(","):
        if (
            result
            and result[-1].startswith("mpol=")
            and re.fullmatch(r"\d+(?:-\d+)?", token)
        ):
            result[-1] += "," + token
        else:
            result.append(token)
    return result


def _mount_table(path="/proc/self/mountinfo"):
    result = []
    for line in _read_text(path).splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
            result.append(
                {
                    "mount_id": int(fields[0]),
                    "parent_id": int(fields[1]),
                    "device": fields[2],
                    "root": _unescape_mount(fields[3]),
                    "path": _unescape_mount(fields[4]),
                    "options": sorted(set(_split_mount_options(fields[5]))),
                    "optional": fields[6:separator],
                    "filesystem": fields[separator + 1],
                    "source": _unescape_mount(fields[separator + 2]),
                    "super_options": sorted(set(_split_mount_options(fields[separator + 3]))),
                }
            )
        except (ValueError, IndexError):
            continue
    return result


def _mount_at(path):
    path = os.path.normpath(os.path.abspath(path))
    matches = [
        mount
        for mount in _mount_table()
        if os.path.normpath(mount["path"]) == path
    ]
    if len(matches) > 1:
        raise RamdiskError(
            "refusing ambiguous stacked mounts at %s (mount ids %s)"
            % (path, ", ".join(str(item["mount_id"]) for item in matches))
        )
    return matches[0] if matches else None


def _filesystem_for_path(path):
    """Return the filesystem of the longest mountpoint containing ``path``."""
    normalized = os.path.normpath(os.path.abspath(path))
    matches = []
    for mount in _mount_table():
        root = os.path.normpath(mount["path"])
        if _path_is_below(normalized, root, allow_equal=True):
            matches.append((len(root), mount["filesystem"]))
    return max(matches)[1] if matches else None


def _run(command, **kwargs):
    kwargs.setdefault("text", True)
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("check", False)
    return subprocess.run(command, **kwargs)


def _trusted_system_binary(name):
    """Resolve a fixed system executable safe to place after ``sudo --``."""
    candidates = [
        os.path.join(prefix, name)
        for prefix in (
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
            "/run/current-system/sw/bin",
            "/run/wrappers/bin",
            "/nix/var/nix/profiles/default/bin",
        )
    ]
    discovered = shutil.which(name)
    if discovered:
        candidates.append(discovered)
    system_uid = os.stat("/").st_uid  # uid 0 outside user namespaces
    groups = set(os.getgroups())
    rejected = []
    for candidate in candidates:
        path = os.path.realpath(candidate)
        if os.path.basename(path) != name:
            rejected.append(candidate)
            continue
        try:
            info = os.stat(path)
        except OSError:
            continue
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_mode & 0o022
            or info.st_uid != system_uid
        ):
            rejected.append(path)
            continue
        parent = os.path.dirname(path)
        unsafe_parent = False
        while True:
            parent_info = os.stat(parent)
            group_writable_by_us = bool(parent_info.st_mode & stat.S_IWGRP and parent_info.st_gid in groups)
            if (
                parent_info.st_uid != system_uid
                or parent_info.st_mode & stat.S_IWOTH
                or group_writable_by_us
                or (os.geteuid() != 0 and os.access(parent, os.W_OK))
            ):
                unsafe_parent = True
                break
            next_parent = os.path.dirname(parent)
            if next_parent == parent:
                break
            parent = next_parent
        if unsafe_parent:
            rejected.append(path)
            continue
        return path
    detail = " (rejected writable candidates: %s)" % ", ".join(rejected) if rejected else ""
    raise RamdiskError("trusted %s executable was not found%s" % (name, detail))


def _fresh_user_binary(name):
    """Resolve an unprivileged helper now, never from serialized manifest data."""
    path = shutil.which(name)
    if not path or os.path.basename(path) != name or not os.access(path, os.X_OK):
        raise RamdiskError("%s was not found on PATH" % name)
    return os.path.realpath(path)


_privilege_local = threading.local()


def _validate_noninteractive_sudo(sudo):
    """Confirm the foreground authorization can be reused without a prompt."""
    return subprocess.run(
        [sudo, "-n", "-v"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _sudo_ticket_keepalive(
    stop_event,
    sudo,
    interval=1.0,
    failure_event=None,
    cancel_event=None,
):
    """Refresh an already-authorized sudo timestamp during a long operation.

    Refresh immediately, then frequently enough for deliberately short sudo
    timestamp policies.  A failed non-interactive refresh is terminal for this
    authorization window: signal cancellable staging work so it reaches its
    rollback path instead of continuing until cleanup authority is certainly
    gone.
    """
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                [sudo, "-n", "-v"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is None or result.returncode:
            if failure_event is not None:
                failure_event.set()
            if cancel_event is not None:
                cancel_event.set()
            return
        if stop_event.wait(interval):
            return


@contextlib.contextmanager
def _noninteractive_privilege(keepalive=False, cancel_event=None):
    """Keep a validated sudo ticket alive without prompting from a TUI worker."""
    previous = getattr(_privilege_local, "noninteractive", False)
    _privilege_local.noninteractive = True
    keepalive_stop = None
    keepalive_failure = None
    keepalive_thread = None
    try:
        if keepalive and not previous and os.geteuid() != 0:
            sudo = _trusted_system_binary("sudo")
            keepalive_stop = threading.Event()
            keepalive_failure = threading.Event()
            keepalive_thread = threading.Thread(
                target=_sudo_ticket_keepalive,
                args=(
                    keepalive_stop,
                    sudo,
                    1.0,
                    keepalive_failure,
                    cancel_event,
                ),
                name="coli-sudo-ticket-keepalive",
                daemon=True,
            )
            keepalive_thread.start()
        yield
    finally:
        if keepalive_stop is not None:
            keepalive_stop.set()
        if keepalive_thread is not None:
            keepalive_thread.join(timeout=6.0)
        _privilege_local.noninteractive = previous


def _privileged(command, hardware):
    if os.geteuid() == 0:
        return command
    sudo = _trusted_system_binary("sudo")
    options = ["-n"] if getattr(_privilege_local, "noninteractive", False) else []
    return [sudo] + options + ["--"] + command


def _mount_option_list(plan, mount, thp=None, include_noswap=None):
    thp = thp or plan["mount_options"]["thp"]
    if include_noswap is None:
        include_noswap = plan["mount_options"]["noswap"]
    options = [
        "size=%d" % mount["size_bytes"],
        "huge=%s" % thp,
        "noatime",
        "nodev",
        "nosuid",
        "noexec",
        "mode=0700",
        "uid=%d" % os.getuid(),
        "gid=%d" % os.getgid(),
        "mpol=%s" % mount["policy"],
        # util-linux handles creation as part of mount(8), keeping sudo scoped
        # to the exact mount command rather than a separate mkdir/chown.
        # The underlying /mnt directory may remain after unmount because the
        # invoking user cannot remove a root-owned child of /mnt. Keep that
        # empty mountpoint traversable/readable for safe prepare→destroy→prepare
        # reuse; the mounted tmpfs itself remains private via mode=0700 above.
        "X-mount.mkdir=0755",
    ]
    if include_noswap:
        options.insert(1, "noswap")
    return options


def _mount_tmpfs(plan, mount):
    hardware = plan["hardware"]
    # Never execute a path recovered from the user-editable manifest under
    # sudo. Resolve and verify the exact system utility at the point of use.
    mount_bin = _trusted_system_binary("mount")
    attempts = []
    thp = plan["mount_options"]["thp"]
    noswap = plan["mount_options"]["noswap"]
    attempts.append((thp, noswap))
    if thp == "within_size":
        attempts.append(("advise", noswap))
    if plan["mount_options"]["allow_swappable"] and noswap:
        # If an older kernel rejects only ``noswap``, preserve the requested
        # THP policy. ``advise`` is a fallback for unsupported within_size,
        # not an accidental side effect of accepting swappable tmpfs.
        attempts.append((thp, False))
        if thp == "within_size":
            attempts.append(("advise", False))
    seen = set()
    errors = []
    for try_thp, try_noswap in attempts:
        if (try_thp, try_noswap) in seen:
            continue
        seen.add((try_thp, try_noswap))
        options = _mount_option_list(plan, mount, try_thp, try_noswap)
        command = [mount_bin, "-t", "tmpfs", "-o", ",".join(options), "tmpfs", mount["path"]]
        try:
            result = _run(_privileged(command, hardware))
        except BaseException as interrupted:
            _rollback_interrupted_mount(
                plan,
                mount,
                try_thp,
                try_noswap,
                interrupted,
            )
            raise
        if result.returncode == 0:
            mount["effective_thp"] = try_thp
            mount["effective_noswap"] = try_noswap
            return
        errors.append(result.stderr.strip() or result.stdout.strip() or "mount failed")
        # Only option-recognition failures justify the documented fallback.
        message = (result.stderr + result.stdout).lower()
        if not any(word in message for word in ("invalid argument", "unknown", "not supported", "wrong fs")):
            break
    raise RamdiskError("cannot mount tmpfs at %s: %s" % (mount["path"], "; ".join(errors)))


def _umount_path(path, hardware):
    umount = _trusted_system_binary("umount")
    result = _run(_privileged([umount, "--", path], hardware))
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip() or "umount failed"
        raise RamdiskError("cannot unmount %s: %s" % (path, message))


def _rollback_interrupted_mount(plan, mount, effective_thp, effective_noswap, cause):
    """Remove a mount that completed just as its helper was interrupted."""
    actual = _mount_at(mount["path"])
    if actual is None:
        return
    attempted = dict(mount)
    attempted["effective_thp"] = effective_thp
    attempted["effective_noswap"] = effective_noswap
    try:
        _validate_mount(attempted, plan)
    except Exception as verification_error:
        raise RamdiskError(
            "mount helper was interrupted and a mount now exists at %s, "
            "but it cannot be attributed safely: %s"
            % (mount["path"], verification_error)
        ) from cause
    try:
        _umount_path(mount["path"], plan["hardware"])
    except Exception as cleanup_error:
        raise RamdiskError(
            "mount helper was interrupted after tmpfs appeared at %s; "
            "immediate rollback failed: %s"
            % (mount["path"], cleanup_error)
        ) from cause


def _option_present(options, name):
    return any(option == name or option.startswith(name + "=") for option in options)


def _validate_mount(mount, plan):
    actual = _mount_at(mount["path"])
    if not actual:
        raise RamdiskError("expected mount is absent: %s" % mount["path"])
    if actual["filesystem"] != "tmpfs" or actual["source"] != "tmpfs":
        raise RamdiskError("refusing foreign mount at %s" % mount["path"])
    options = set(actual["options"] + actual["super_options"])
    required = ("noatime", "nodev", "nosuid", "noexec")
    missing = [name for name in required if not _option_present(options, name)]
    if mount.get("effective_noswap", plan["mount_options"]["noswap"]):
        if not _option_present(options, "noswap"):
            missing.append("noswap")
    if missing:
        raise RamdiskError("tmpfs at %s is missing options: %s" % (mount["path"], ", ".join(missing)))
    mode_ok = any(option in ("mode=700", "mode=0700") for option in options)
    huge = mount.get("effective_thp", plan["mount_options"]["thp"])
    policy = mount["policy"].replace("\\,", ",")
    normalized_options = {option.replace("\\,", ",") for option in options}
    if not mode_ok or not _option_present(normalized_options, "huge"):
        raise RamdiskError("tmpfs at %s is missing managed mode/THP options" % mount["path"])
    if "huge=%s" % huge not in normalized_options:
        raise RamdiskError("tmpfs at %s has an unexpected THP policy" % mount["path"])
    if "mpol=%s" % policy not in normalized_options:
        raise RamdiskError("tmpfs at %s has an unexpected NUMA policy" % mount["path"])
    actual["all_options"] = sorted(options)
    return actual


def _available_memory():
    values = _meminfo()
    available = values.get("MemAvailable", values.get("MemFree", 0))
    cgroup_available = _cgroup_available_memory()
    return (
        min(available, cgroup_available)
        if cgroup_available is not None
        else available
    )


def _host_available_for_mount(mount, plan=None):
    """Return host/NUMA availability without reusing shared cgroup headroom."""
    if mount.get("node") is None:
        nodes = (plan or {}).get("placement", {}).get("memory_nodes")
        if nodes:
            available = 0
            for node in nodes:
                values = _node_meminfo(int(node))
                available += values.get("MemFree", values.get("MemAvailable", 0))
            return available
        values = _meminfo()
        return values.get("MemAvailable", values.get("MemFree", 0))
    values = _node_meminfo(int(mount["node"]))
    return values.get("MemFree", values.get("MemAvailable", 0))


def _available_for_mount(mount, plan=None):
    available = _host_available_for_mount(mount, plan=plan)
    cgroup_available = _cgroup_available_memory()
    return (
        min(available, cgroup_available)
        if cgroup_available is not None
        else available
    )


def _raise_if_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise _OperationCancelled("operation cancelled by user at a safe checkpoint")


def _copy_stream(src, tmp, expected_size, cancel_event=None):
    source_fd = os.open(src, os.O_RDONLY)
    try:
        destination_fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            copied = 0
            while copied < expected_size:
                _raise_if_cancelled(cancel_event)
                data = os.read(source_fd, min(8 * MIB, expected_size - copied))
                if not data:
                    raise RamdiskError("source shard was truncated while copying: %s" % src)
                view = memoryview(data)
                while view:
                    written = os.write(destination_fd, view)
                    if written <= 0:
                        raise RamdiskError("short write while staging %s" % src)
                    view = view[written:]
                copied += len(data)
            if os.read(source_fd, 1):
                raise RamdiskError("source shard grew while copying: %s" % src)
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(source_fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    finally:
        os.close(source_fd)


def _copy_one(
    src,
    destination,
    expected_size,
    reserve_floor,
    progress=None,
    available=None,
    cancel_event=None,
):
    available = available or _available_memory
    _raise_if_cancelled(cancel_event)
    if available() < reserve_floor:
        raise RamdiskError("available memory reached the protected reserve before %s" % os.path.basename(src))
    tmp = destination + ".coli-copy-%d-%s" % (os.getpid(), secrets.token_hex(4))
    started = time.monotonic()
    try:
        _copy_stream(src, tmp, expected_size, cancel_event=cancel_event)
        os.chmod(tmp, 0o400)
        if os.path.getsize(tmp) != expected_size:
            raise RamdiskError("staged size mismatch for %s" % os.path.basename(src))
        _read_safetensors_header(tmp)
        os.replace(tmp, destination)
        if progress:
            progress(os.path.basename(src), expected_size, time.monotonic() - started)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _copy_worker_main(src, tmp, expected_size):
    _copy_stream(src, tmp, int(expected_size))
    os.chmod(tmp, 0o400)
    return 0


def _copy_one_affined(
    src,
    destination,
    expected_size,
    node,
    numactl,
    cpu_list,
    reserve_floor,
    progress=None,
    available=None,
    cancel_event=None,
):
    available = available or _available_memory
    _raise_if_cancelled(cancel_event)
    if available() < reserve_floor:
        raise RamdiskError("available memory reached the protected reserve before replica copy")
    tmp = destination + ".coli-copy-%d-%s" % (os.getpid(), secrets.token_hex(4))
    started = time.monotonic()
    command = [
        numactl,
        "--physcpubind=%s" % cpu_list,
        "--membind=%d" % node,
        sys.executable,
        os.path.abspath(__file__),
        "--copy-worker",
        src,
        tmp,
        str(expected_size),
    ]
    try:
        if cancel_event is None:
            result = _run(command)
        else:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                while process.poll() is None:
                    if cancel_event.wait(0.1):
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        _raise_if_cancelled(cancel_event)
                stdout, stderr = process.communicate()
            except BaseException:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                raise
            result = subprocess.CompletedProcess(
                command,
                process.returncode,
                stdout,
                stderr,
            )
        if result.returncode:
            raise RamdiskError("node-affined replica copy failed: %s" % (result.stderr.strip() or result.stdout.strip()))
        if os.path.getsize(tmp) != expected_size:
            raise RamdiskError("replica size mismatch for %s" % os.path.basename(src))
        _read_safetensors_header(tmp)
        os.replace(tmp, destination)
        if progress:
            progress(os.path.basename(src), expected_size, time.monotonic() - started)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _populate_mount(plan, mount, source_root=None, progress=None, cancel_event=None):
    source_root = source_root or plan["model"]["path"]
    selected = plan["staging"]["selected_shards"]
    linked = plan["staging"]["linked_shards"]
    identities = {item["name"]: item for item in plan["source_shards"]}
    if mount.get("node") is None:
        reserve_floor = (
            plan["reserve"]["runtime_bytes"]
            + plan["reserve"]["page_table_bytes"]
            + plan["reserve"]["os_margin_bytes"]
        )
    else:
        node_info = next(item for item in plan["hardware"]["nodes"] if item["id"] == mount["node"])
        reserve_floor = (
            plan["reserve"]["runtime_bytes"]
            + plan["reserve"]["page_table_bytes"]
            + node_info.get("reserve_bytes", 8 * GIB)
        )
    def available():
        return _available_for_mount(mount, plan=plan)
    workers = max(1, min(plan["parallel"], len(selected) or 1))
    admission_lock = threading.Lock()
    inflight = [0]

    def copy_name(name):
        _raise_if_cancelled(cancel_event)
        source = os.path.join(source_root, name)
        destination = os.path.join(mount["path"], name)
        expected = identities[name]["size_bytes"]
        with admission_lock:
            observed = available()
            if observed - inflight[0] - expected < reserve_floor:
                raise RamdiskError("projected shard copies would breach the protected memory reserve")
            inflight[0] += expected
        try:
            if source_root != plan["model"]["path"] and mount["node"] is not None:
                return _copy_one_affined(
                    source,
                    destination,
                    expected,
                    mount["node"],
                    plan["hardware"]["numactl"],
                    _engine_cpu_list(plan, node=mount["node"]),
                    reserve_floor,
                    progress,
                    available,
                    cancel_event,
                )
            return _copy_one(
                source,
                destination,
                expected,
                reserve_floor,
                progress,
                available,
                cancel_event,
            )
        finally:
            with admission_lock:
                inflight[0] -= expected

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(copy_name, name) for name in selected]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    _raise_if_cancelled(cancel_event)
    for name in linked:
        _raise_if_cancelled(cancel_event)
        target = os.path.join(plan["model"]["path"], name)
        destination = os.path.join(mount["path"], name)
        os.symlink(target, destination)


def _sample_page_indices(total_pages, sample_pages, node_count):
    sample_pages = max(1, min(sample_pages, total_pages))
    if sample_pages == total_pages:
        return list(range(total_pages))
    step = max(1, total_pages // sample_pages)
    # A stride sharing a divisor with the node count can observe only one
    # residue of an actually round-robin interleaved mapping. Advance to a
    # coprime stride, then walk modulo the file page count deterministically.
    while math.gcd(step, max(1, node_count)) != 1:
        step += 1
    indices = []
    seen = set()
    value = 0
    while len(indices) < sample_pages:
        if value not in seen:
            indices.append(value)
            seen.add(value)
        value = (value + step) % total_pages
        if value in seen and len(indices) < sample_pages:
            value = (max(seen) + 1) % total_pages
    return indices


def _sample_numa_allocation(path, max_pages=1024, node_count=1):
    """Touch a bounded page sample and report its /proc/self/numa_maps nodes."""
    counts = {}
    size = os.path.getsize(path)
    if size <= 0:
        return counts
    with open(path, "rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
            total_pages = max(1, (size + 4095) // 4096)
            pages = max(1, min(max_pages, total_pages))
            for page in _sample_page_indices(total_pages, pages, node_count):
                mapping[min(size - 1, page * 4096)]
            needle = "file=" + path.replace(" ", "\\040")
            for line in _read_text("/proc/self/numa_maps").splitlines():
                if needle not in line and path not in line:
                    continue
                for node, value in re.findall(r"\bN(\d+)=(\d+)\b", line):
                    counts[node] = counts.get(node, 0) + int(value)
    return counts


def _validate_namespace(plan, mount, sample_numa=True):
    expected_names = sorted(item["name"] for item in plan["source_shards"])
    actual_names = sorted(name for name in os.listdir(mount["path"]) if name.endswith(".safetensors"))
    if actual_names != expected_names:
        raise RamdiskError("staged namespace filenames do not match the canonical model")
    identities = {item["name"]: item for item in plan["source_shards"]}
    selected = set(plan["staging"]["selected_shards"])
    linked = set(plan["staging"]["linked_shards"])
    for name in expected_names:
        path = os.path.join(mount["path"], name)
        if name in selected:
            if os.path.islink(path) or not stat.S_ISREG(os.stat(path).st_mode):
                raise RamdiskError("staged shard is not a regular tmpfs file: %s" % name)
            if os.path.getsize(path) != identities[name]["size_bytes"]:
                raise RamdiskError("staged shard size mismatch: %s" % name)
            if os.stat(path).st_mode & 0o222:
                raise RamdiskError("staged shard is writable: %s" % name)
            raw, _ = _read_safetensors_header(path)
            if hashlib.sha256(raw).hexdigest() != identities[name]["header_sha256"]:
                raise RamdiskError("staged shard header mismatch: %s" % name)
        elif name in linked:
            if not os.path.islink(path):
                raise RamdiskError("unstaged shard is not an SSD fallback symlink: %s" % name)
            canonical = os.path.join(plan["model"]["path"], name)
            if os.path.realpath(path) != os.path.realpath(canonical):
                raise RamdiskError("fallback symlink does not target the canonical shard: %s" % name)
    allocation = {}
    if not sample_numa:
        return allocation
    selected_names = plan["staging"]["selected_shards"]
    placement_nodes = plan.get("placement", {}).get(
        "memory_nodes", plan["hardware"]["online_nodes"]
    )
    pages_per_shard = max(32, min(1024, 4096 // max(1, len(selected_names))))
    for name in selected_names:
        path = os.path.join(mount["path"], name)
        for node, count in _sample_numa_allocation(
            path, pages_per_shard, node_count=len(placement_nodes)
        ).items():
            allocation[node] = allocation.get(node, 0) + count
    online_nodes = plan.get("hardware", {}).get("online_nodes", placement_nodes)
    verify_numa = len(placement_nodes) > 1 or len(online_nodes) > 1
    if verify_numa:
        total = sum(allocation.values())
        if not total:
            raise RamdiskError("could not verify actual NUMA allocation for staged shards")
        outside = sum(
            count for node, count in allocation.items() if int(node) not in placement_nodes
        )
        if float(outside) / total > 0.01:
            raise RamdiskError(
                "tmpfs sample escaped the reviewed NUMA memory-node mask"
            )
        if mount["node"] is not None:
            local = allocation.get(str(mount["node"]), 0)
            if float(local) / total < 0.95:
                raise RamdiskError("node-local tmpfs sample is below 95%% local allocation")
        elif len(placement_nodes) > 1:
            ideal = float(total) / len(placement_nodes)
            if any(
                abs(allocation.get(str(node), 0) - ideal) / ideal > 0.15
                for node in placement_nodes
            ):
                raise RamdiskError("interleaved tmpfs sample differs by more than 15%% across nodes")
    return allocation


def _source_still_matches(plan):
    current = scan_model(plan["model"]["path"])
    if current["fingerprint"] != plan["model"]["fingerprint"]:
        raise RamdiskError("canonical model changed while staging; refusing to publish the manifest")


def _confirm(message, accepted=False):
    if accepted:
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise RamdiskError(message + "; rerun with --yes after reviewing `ramdisk plan`")
    answer = input(message + " [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        raise RamdiskError("cancelled")


@_exclusive_lifecycle
def prepare(
    args,
    progress=None,
    display_plan=True,
    expected_plan_token=None,
    cancel_event=None,
):
    if _load_manifest(required=False) is not None:
        raise RamdiskError("a RAM-disk manifest already exists; stop/destroy it before preparing another")
    plan = build_plan(args)
    try:
        base_port = int(getattr(args, "base_port", 8000))
    except (TypeError, ValueError):
        raise RamdiskError("managed base port must be an integer")
    planned_ports = _managed_ports_for_plan(plan, base_port)
    if (
        not 1 <= base_port <= 65535
        or len(set(planned_ports)) != len(planned_ports)
        or any(port < 1 or port > 65535 for port in planned_ports)
    ):
        raise RamdiskError("managed base port produces invalid or duplicate replica ports")
    _raise_if_cancelled(cancel_event)
    if expected_plan_token is not None and _plan_confirmation_token(plan) != expected_plan_token:
        raise RamdiskError(
            "RAM-disk plan changed since review; inspect the updated plan and confirm again"
        )
    if plan["blockers"]:
        raise RamdiskError("preparation blocked: " + "; ".join(plan["blockers"]))
    if display_plan:
        _human_plan(plan)
    _confirm("Mount tmpfs and stage the reviewed bytes?", bool(getattr(args, "yes", False)))
    if progress is None:
        progress_lock = threading.Lock()

        def progress(name, size, elapsed):
            with progress_lock:
                rate = size / elapsed / MIB if elapsed > 0 else 0.0
                print("  staged %-36s %8.1f MiB/s" % (name, rate), flush=True)
    manifest = {
        "version": MANIFEST_VERSION,
        "deployment_id": secrets.token_hex(16),
        "base_port": base_port,
        "state": "preparing",
        "created_at": _utc_now(),
        "plan": plan,
        "model_fingerprint": plan["model"]["fingerprint"],
        "mounts": [],
        "processes": [],
        "ports": [],
        "benchmark_results": [],
        "initial_swap_used_bytes": plan["hardware"]["swap"]["used_bytes"],
    }
    _save_manifest(manifest)
    mounted = []
    try:
        for mount in plan["mounts"]:
            _raise_if_cancelled(cancel_event)
            if _mount_at(mount["path"]):
                raise RamdiskError("refusing already-mounted path: %s" % mount["path"])
            _mount_tmpfs(plan, mount)
            mounted_actual = _mount_at(mount["path"])
            if not mounted_actual:
                # The mount command succeeded against a path proven empty
                # immediately above, but no identity can be recorded. Roll it
                # back now; never carry an identity-less mount into the general
                # cleanup path where it could be confused with a foreign one.
                try:
                    _umount_path(mount["path"], plan["hardware"])
                except Exception as cleanup_exc:
                    raise RamdiskError(
                        "mounted %s but could not identify or roll it back: %s"
                        % (mount["path"], cleanup_exc)
                    )
                raise RamdiskError(
                    "mounted %s but could not read its mount identity; rolled it back"
                    % mount["path"]
                )
            mounted.append((mount, mounted_actual))
            record = dict(mount)
            record["identity"] = mounted_actual
            record["validated"] = False
            manifest["mounts"].append(record)
            _save_manifest(manifest)
            identity = _validate_mount(mount, plan)
            record["identity"] = identity
            record["validated"] = True
            _save_manifest(manifest)
        seed = None
        for index, mount in enumerate(plan["mounts"]):
            _raise_if_cancelled(cancel_event)
            source = seed if index and plan["topology"] == "per-node" else None
            _populate_mount(
                plan,
                mount,
                source_root=source,
                progress=progress,
                cancel_event=cancel_event,
            )
            if seed is None:
                seed = mount["path"]
            manifest["mounts"][index]["numa_allocation"] = _validate_namespace(plan, mount)
            _save_manifest(manifest)
        _raise_if_cancelled(cancel_event)
        _source_still_matches(plan)
        manifest["state"] = "ready"
        manifest["ready_at"] = _utc_now()
        _save_manifest(manifest)
        return manifest
    except BaseException as exc:
        manifest["state"] = "error"
        manifest["error"] = str(exc)
        cleanup_errors = []
        try:
            _save_manifest(manifest)
        except Exception as save_exc:
            cleanup_errors.append("could not persist preparation error: %s" % save_exc)
        for mount, expected in reversed(mounted):
            try:
                actual = _mount_at(mount["path"])
                if actual:
                    if (
                        expected
                        and actual["filesystem"] == "tmpfs"
                        and actual["source"] == "tmpfs"
                        and actual["mount_id"] == expected.get("mount_id")
                        and actual["device"] == expected.get("device")
                    ):
                        _umount_path(mount["path"], plan["hardware"])
                    else:
                        cleanup_errors.append(
                            "refusing changed mount during preparation rollback: %s"
                            % mount["path"]
                        )
            except Exception as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
        if isinstance(exc, _OperationCancelled) and not cleanup_errors:
            try:
                _durable_unlink(_manifest_path())
            except Exception as cleanup_exc:
                cleanup_errors.append(
                    "could not remove cancelled preparation manifest: %s"
                    % cleanup_exc
                )
            if not cleanup_errors:
                raise
        if cleanup_errors:
            manifest["cleanup_errors"] = cleanup_errors
            try:
                _save_manifest(manifest)
            except Exception as save_exc:
                cleanup_errors.append("could not persist cleanup result: %s" % save_exc)
            raise RamdiskError(
                "%s; preparation rollback/reporting errors: %s"
                % (exc, "; ".join(cleanup_errors))
            ) from exc
        raise


def _proc_identity(pid):
    try:
        raw_stat = _read_text("/proc/%d/stat" % pid)
        close = raw_stat.rfind(")")
        fields = raw_stat[close + 2 :].split()
        starttime = int(fields[19])
        cmdline = open("/proc/%d/cmdline" % pid, "rb").read().split(b"\0")
        environ = open("/proc/%d/environ" % pid, "rb").read().split(b"\0")
        env = {}
        for item in environ:
            if b"=" in item:
                key, value = item.split(b"=", 1)
                env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
        return {
            "pid": pid,
            "uid": os.stat("/proc/%d" % pid).st_uid,
            "starttime": starttime,
            "cmdline": [value.decode("utf-8", "replace") for value in cmdline if value],
            "nonce": env.get("COLI_MANAGED_NONCE"),
            "pgid": os.getpgid(pid),
        }
    except (OSError, ValueError, IndexError):
        return None


def _process_group_members(pgid):
    """Return readable identities in a process group, plus unreadable member PIDs."""
    members, unreadable = [], []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return members, unreadable
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        raw = _read_text("/proc/%d/stat" % pid)
        close = raw.rfind(")")
        try:
            fields = raw[close + 2 :].split()
            member_pgid = int(fields[2])
        except (ValueError, IndexError):
            continue
        if member_pgid != pgid:
            continue
        identity = _proc_identity(pid)
        if identity:
            members.append(identity)
        else:
            unreadable.append(pid)
    return members, unreadable


def _process_matches(record):
    pid = int(record["pid"])
    expected_pgid = int(record.get("pgid", pid))
    actual = _proc_identity(pid)
    if not actual:
        # The Python serve wrapper can die before its engine child.  A managed
        # session keeps the original PGID, so validate every surviving member's
        # inherited UID+nonce before treating the group as signalable.
        members, unreadable = _process_group_members(expected_pgid)
        if not members and not unreadable:
            return False, "not-running", None
        if unreadable:
            return False, "unverified-process-group", {"pgid": expected_pgid, "members": unreadable}
        if any(
            member["uid"] != record.get("uid")
            or member["nonce"] != record.get("nonce")
            for member in members
        ):
            return False, "foreign-process-group", {"pgid": expected_pgid, "members": members}
        return True, "running-group", {"pid": pid, "pgid": expected_pgid, "members": members}
    if actual["uid"] != record.get("uid"):
        return False, "foreign-uid", actual
    if actual["starttime"] != record.get("starttime"):
        return False, "reused-pid", actual
    if actual["nonce"] != record.get("nonce"):
        return False, "foreign-nonce", actual
    if actual["pgid"] != expected_pgid:
        return False, "foreign-process-group", actual
    return True, "running", actual


def _process_tree_alive(record, actual):
    expected_pgid = int(record.get("pgid", record["pid"]))
    if actual and actual.get("pgid") == expected_pgid:
        try:
            os.killpg(expected_pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    return bool(_proc_identity(int(record["pid"])))


def _usage_read(path):
    counts = {}
    for line in _read_text(path).splitlines():
        match = PROFILE_LINE_RE.match(line)
        if match:
            layer, expert, count = (int(value) for value in match.groups())
            counts["%d:%d" % (layer, expert)] = count
    return counts


def _usage_merge_ids(path):
    result = set()
    for line in _read_text(path).splitlines():
        match = USAGE_MERGE_RE.match(line.strip())
        if match:
            result.add(match.group(1))
    return result


def _fsync_directory(path):
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
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
    # Never recreate a missing canonical model directory during stop/recovery:
    # doing so would turn a moved model into an empty directory whose only file
    # is .coli_usage and falsely mark the durable delta as merged. Manager-owned
    # state directories are created explicitly before this writer is called.
    if not os.path.isdir(parent):
        raise RamdiskError("usage-state parent directory is absent: %s" % parent)
    markers = set(merge_ids or ())
    if merge_id:
        markers.add(merge_id)
    fd, tmp = tempfile.mkstemp(prefix=".usage-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for key in sorted(counts, key=lambda item: tuple(int(value) for value in item.split(":"))):
                layer, expert = key.split(":")
                stream.write("%s %s %d\n" % (layer, expert, counts[key]))
            for marker in sorted(markers):
                # glm's fscanf loop consumes the numeric rows then stops at this
                # trailing comment. Preserve every still-relevant transaction,
                # not only the latest node's marker: otherwise a crash between
                # two node merges can replay the first node's delta.
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


def _assert_canonical_usage_target(canonical_path, plan=None):
    parent = os.path.dirname(canonical_path)
    if not os.path.isdir(parent):
        raise RamdiskError("canonical model directory is absent; usage delta remains journaled")
    if plan is not None:
        if os.path.realpath(parent) != os.path.realpath(plan["model"]["path"]):
            raise RamdiskError("canonical usage target no longer matches the managed model")
        _source_still_matches(plan)


def _merge_usage(record, canonical_path, plan=None, keep_journal=False):
    _assert_durable_state_dir(record["state_dir"], plan=plan)
    state_usage = os.path.join(record["state_dir"], ".coli_usage")
    current = _usage_read(state_usage)
    baseline = record.get("usage_baseline", {})
    delta = {key: value - baseline.get(key, 0) for key, value in current.items() if value > baseline.get(key, 0)}
    delta_path = os.path.join(record["state_dir"], ".coli_usage.delta.json")
    if os.path.exists(delta_path):
        _recover_delta(record["state_dir"], canonical_path, plan=plan)
        return
    merge_id = record.get("usage_merge_id") or secrets.token_hex(16)
    record["usage_merge_id"] = merge_id
    if merge_id in _usage_merge_ids(canonical_path):
        return
    if not delta:
        try:
            _durable_unlink(delta_path)
        except OSError:
            pass
        return
    _atomic_json(delta_path, {"version": 1, "id": merge_id, "delta": delta, "created_at": _utc_now()})
    _assert_canonical_usage_target(canonical_path, plan=plan)
    lock_path = os.path.join(_state_root(), "usage.lock")
    _ensure_private_dir(os.path.dirname(lock_path))
    with open(lock_path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        applied = _usage_merge_ids(canonical_path)
        if merge_id not in applied:
            canonical = _usage_read(canonical_path)
            for key, value in delta.items():
                canonical[key] = canonical.get(key, 0) + value
            applied.add(merge_id)
            _usage_write(canonical_path, canonical, merge_ids=applied)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    if not keep_journal:
        _durable_unlink(delta_path)


def _recover_delta(state_dir, canonical_path, plan=None):
    _assert_durable_state_dir(state_dir, plan=plan)
    delta_path = os.path.join(state_dir, ".coli_usage.delta.json")
    payload = _read_json(delta_path)
    if not payload:
        return
    if not isinstance(payload, dict):
        raise RamdiskError("usage delta journal must contain a JSON object")
    delta = payload.get("delta", {})
    if not isinstance(delta, dict):
        raise RamdiskError("usage delta journal has invalid counts")
    merge_id = payload.get("id")
    if not merge_id:
        raise RamdiskError("usage delta journal is missing its transaction id")
    if not isinstance(merge_id, str) or not re.fullmatch(r"[0-9a-f]{32}", merge_id):
        raise RamdiskError("usage delta journal has an invalid transaction id")
    _assert_canonical_usage_target(canonical_path, plan=plan)
    lock_path = os.path.join(_state_root(), "usage.lock")
    _ensure_private_dir(os.path.dirname(lock_path))
    with open(lock_path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        applied = _usage_merge_ids(canonical_path)
        if merge_id not in applied:
            canonical = _usage_read(canonical_path)
            for key, value in delta.items():
                canonical[key] = canonical.get(key, 0) + int(value)
            applied.add(merge_id)
            _usage_write(canonical_path, canonical, merge_ids=applied)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    _durable_unlink(delta_path)


def _assert_ready_mounts(manifest):
    plan = manifest["plan"]
    _source_still_matches(plan)
    for record in manifest.get("mounts", []):
        actual = _validate_mount(record, plan)
        expected = record.get("identity", {})
        if expected and (actual["mount_id"] != expected.get("mount_id") or actual["device"] != expected.get("device")):
            raise RamdiskError("mount identity changed at %s" % record["path"])
        _validate_namespace(plan, record)


def _runtime_admission_requirement(plan, mount, benchmark=False):
    """Return the runtime, page-table, and protected host floor for one engine."""
    reserve = plan["reserve"]
    runtime_bytes = int(
        reserve.get("benchmark_runtime_bytes" if benchmark else "managed_runtime_bytes")
        or reserve["runtime_bytes"]
    )
    page_tables = int(reserve["page_table_bytes"])
    if mount.get("node") is None:
        margin = int(reserve["os_margin_bytes"])
    else:
        node = next(
            item for item in plan["hardware"]["nodes"] if item["id"] == mount["node"]
        )
        margin = int(node.get("reserve_bytes", max(node["memory_total_bytes"] // 10, 8 * GIB)))
    return runtime_bytes + page_tables + margin


def _admit_runtime(plan, mount, benchmark=False):
    """Recheck the reviewed post-staging floor immediately before launch."""
    required = _runtime_admission_requirement(plan, mount, benchmark=benchmark)
    available = _available_for_mount(mount, plan=plan)
    if available < required:
        label = "global memory" if mount.get("node") is None else "NUMA node %d" % mount["node"]
        raise RamdiskError(
            "%s has %d bytes available; launch would breach the %d-byte runtime/OS floor"
            % (label, available, required)
        )
    return {"available_bytes": available, "required_bytes": required}


def _admit_concurrent_runtimes(plan, mounts, benchmark=False):
    """Admit a replica set against one shared cgroup-headroom snapshot."""
    mounts = list(mounts)
    if not mounts:
        raise RamdiskError("concurrent runtime admission requires at least one mount")
    admissions = []
    for mount in mounts:
        required = _runtime_admission_requirement(
            plan, mount, benchmark=benchmark
        )
        host_available = _host_available_for_mount(mount, plan=plan)
        if host_available < required:
            label = (
                "global memory"
                if mount.get("node") is None
                else "NUMA node %d" % mount["node"]
            )
            raise RamdiskError(
                "%s has %d bytes available; launch would breach the "
                "%d-byte runtime/OS floor"
                % (label, host_available, required)
            )
        admissions.append(
            {
                "mount": mount,
                "host_available_bytes": host_available,
                "required_bytes": required,
            }
        )
    cgroup_available = _cgroup_available_memory()
    aggregate_required = sum(item["required_bytes"] for item in admissions)
    if (
        cgroup_available is not None
        and cgroup_available < aggregate_required
    ):
        raise RamdiskError(
            "cgroup memory has %d bytes available; concurrent launch would "
            "breach the %d-byte aggregate runtime/OS floor"
            % (cgroup_available, aggregate_required)
        )
    return {
        "mounts": admissions,
        "cgroup_available_bytes": cgroup_available,
        "required_bytes": aggregate_required,
    }


def _assert_effective_masks_unchanged(plan):
    """Refuse a managed launch after its cgroup/cpuset placement contract drifts."""
    hardware = plan.get("hardware", {})
    placement = plan.get("placement")
    if (
        not placement
        or hardware.get("effective_mask_source") != "kernel-task-status"
    ):
        return
    current = discover_hardware()
    expected_nodes = list(placement.get("effective_nodes", []))
    expected_cpus = list(placement.get("effective_cpus", []))
    if (
        list(current.get("effective_nodes", [])) != expected_nodes
        or list(current.get("effective_cpus", [])) != expected_cpus
    ):
        raise RamdiskError(
            "effective CPU/NUMA mask changed since preparation; destroy and review a fresh plan"
        )


def _group_alive(pgid):
    try:
        os.killpg(int(pgid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


_managed_children_lock = threading.Lock()
_managed_children = {}


def _track_managed_child(process):
    """Retain local Popen handles so a long-lived TUI can reap engine zombies."""
    with _managed_children_lock:
        _managed_children[int(process.pid)] = process


def _poll_managed_child(pid):
    with _managed_children_lock:
        process = _managed_children.get(int(pid))
    if process is None:
        return None
    try:
        returncode = process.poll()
    except (ChildProcessError, OSError):
        returncode = getattr(process, "returncode", None)
    if returncode is not None:
        with _managed_children_lock:
            if _managed_children.get(int(pid)) is process:
                _managed_children.pop(int(pid), None)
    return returncode


def _forget_managed_child(pid):
    with _managed_children_lock:
        _managed_children.pop(int(pid), None)


def _terminate_direct_child(process, term_seconds=10.0, kill_seconds=3.0):
    """Terminate an unrecorded direct child by PID, never an unverified PGID."""
    if process.poll() is not None:
        return None
    try:
        process.terminate()
    except ProcessLookupError:
        return None
    try:
        process.wait(timeout=term_seconds)
        return None
    except ChildProcessError:
        return None
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
    except ProcessLookupError:
        return None
    try:
        process.wait(timeout=kill_seconds)
        return None
    except ChildProcessError:
        return None
    except subprocess.TimeoutExpired:
        return "direct child PID %s survived SIGKILL" % process.pid


def _terminate_group(pgid, term_seconds=10.0, kill_seconds=3.0):
    """Terminate one already-verified or directly-created process group."""
    try:
        os.killpg(int(pgid), signal.SIGTERM)
    except ProcessLookupError:
        return None
    deadline = time.monotonic() + term_seconds
    while time.monotonic() < deadline and _group_alive(pgid):
        time.sleep(0.1)
    if not _group_alive(pgid):
        return None
    try:
        os.killpg(int(pgid), signal.SIGKILL)
    except ProcessLookupError:
        return None
    deadline = time.monotonic() + kill_seconds
    while time.monotonic() < deadline and _group_alive(pgid):
        time.sleep(0.05)
    return "process group %s survived SIGKILL" % pgid if _group_alive(pgid) else None


def _terminate_verified_group(record, term_seconds=10.0, kill_seconds=3.0):
    """Signal only while the persisted process identity still owns its PGID."""
    expected_pgid = int(record.get("pgid", record["pid"]))

    def revalidate(stage):
        # When Start ran inside the still-live TUI process, poll its retained
        # Popen handle first so an exited group leader cannot remain a zombie
        # that falsely appears to have survived SIGTERM/SIGKILL.
        _poll_managed_child(record["pid"])
        matches, reason, actual = _process_matches(record)
        if matches and actual and int(actual.get("pgid", -1)) == expected_pgid:
            return True, None
        if reason == "not-running":
            return False, None
        return (
            False,
            "PID/PGID %s identity changed %s (%s); refusing another signal"
            % (expected_pgid, stage, reason),
        )

    alive, failure = revalidate("before SIGTERM")
    if failure or not alive:
        return failure
    try:
        os.killpg(expected_pgid, signal.SIGTERM)
    except ProcessLookupError:
        return None
    deadline = time.monotonic() + term_seconds
    while time.monotonic() < deadline:
        alive, failure = revalidate("after SIGTERM")
        if failure or not alive:
            return failure
        time.sleep(0.1)

    alive, failure = revalidate("before SIGKILL")
    if failure or not alive:
        return failure
    try:
        os.killpg(expected_pgid, signal.SIGKILL)
    except ProcessLookupError:
        return None
    deadline = time.monotonic() + kill_seconds
    while time.monotonic() < deadline:
        alive, failure = revalidate("after SIGKILL")
        if failure or not alive:
            return failure
        time.sleep(0.05)
    alive, failure = revalidate("after SIGKILL")
    if failure or not alive:
        return failure
    return "process group %s survived SIGKILL" % expected_pgid


def _wait_managed_ready(record, timeout, api_key=None, cancel_event=None):
    deadline = time.monotonic() + timeout
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    last_error = "listener not ready"
    while time.monotonic() < deadline:
        _raise_if_cancelled(cancel_event)
        matches, reason, _ = _process_matches(record)
        if not matches:
            raise RamdiskError(
                "managed engine PID %s exited before readiness (%s); see %s"
                % (record["pid"], reason, record["log"])
            )
        try:
            request = urllib.request.Request(
                "http://127.0.0.1:%d/health" % record["port"], headers=headers
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "ok":
                record["ready_at"] = _utc_now()
                return
            last_error = "health response was not ready"
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = str(exc)
        if cancel_event is None:
            time.sleep(0.5)
        elif cancel_event.wait(0.5):
            _raise_if_cancelled(cancel_event)
    raise RamdiskError(
        "managed engine on port %s did not become ready within %.0fs (%s); see %s"
        % (record["port"], timeout, last_error, record["log"])
    )


@_exclusive_lifecycle
def start(args, cli_path=None, engine_path=None, cancel_event=None):
    manifest = _load_manifest(required=True)
    _raise_if_cancelled(cancel_event)
    if manifest.get("state") not in ("ready", "stopped"):
        raise RamdiskError("manifest state is %s, not ready" % manifest.get("state"))
    _assert_effective_masks_unchanged(manifest["plan"])
    _assert_ready_mounts(manifest)
    plan = manifest["plan"]
    cli_path = cli_path or os.path.join(os.path.dirname(__file__), "coli")
    model = plan["model"]["path"]
    canonical_usage = os.path.join(model, ".coli_usage")
    foreign = []
    recovered = False
    for record in manifest.get("processes", []):
        _raise_if_cancelled(cancel_event)
        if record.get("stopped_at"):
            if not record.get("usage_merged_at"):
                record.setdefault("usage_merge_id", secrets.token_hex(16))
                _save_manifest(manifest)
                _merge_usage(record, canonical_usage, plan=plan)
                record["usage_merged_at"] = _utc_now()
                record.pop("usage_merge_error", None)
                recovered = True
                _save_manifest(manifest)
            continue
        matches, reason, _ = _process_matches(record)
        if matches:
            raise RamdiskError("managed engine is already running on port %s" % record.get("port"))
        if reason == "not-running":
            # Crash recovery must merge the node's post-baseline counts before
            # its stable state directory is seeded for a replacement process.
            record.setdefault("usage_merge_id", secrets.token_hex(16))
            _save_manifest(manifest)
            _merge_usage(record, canonical_usage, plan=plan)
            record["usage_merged_at"] = _utc_now()
            record["stopped_at"] = _utc_now()
            record["crash_recovered_at"] = _utc_now()
            record.pop("usage_merge_error", None)
            recovered = True
            _save_manifest(manifest)
        else:
            foreign.append("PID %s (%s)" % (record.get("pid"), reason))
    if foreign:
        raise RamdiskError("refusing stale foreign process records: " + ", ".join(foreign))
    if recovered:
        _save_manifest(manifest)
    requested_base_port = getattr(args, "base_port", None)
    if requested_base_port is None:
        base_port = _persisted_base_port(manifest)
    else:
        if isinstance(requested_base_port, bool):
            raise RamdiskError("managed base port must be an integer")
        try:
            base_port = int(requested_base_port)
        except (TypeError, ValueError):
            raise RamdiskError("managed base port must be an integer")
    ports = [base_port + (0 if record.get("node") is None else int(record["node"])) for record in manifest["mounts"]]
    if len(set(ports)) != len(ports) or any(port < 1 or port > 65535 for port in ports):
        raise RamdiskError("managed ports are invalid or duplicated")
    for port in ports:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RamdiskError("port %d is unavailable: %s" % (port, exc))
        finally:
            probe.close()
    previous_state = manifest["state"]
    previous_processes = copy.deepcopy(manifest.get("processes", []))
    previous_ports = list(manifest.get("ports", []))
    previous_base_port = _persisted_base_port(manifest)
    manifest["base_port"] = base_port
    nonce = secrets.token_hex(24)
    managed_numactl = (
        _fresh_user_binary("numactl") if plan["topology"] == "per-node" else None
    )
    records = []
    runtime = plan.get("managed_runtime", {})
    saved_runtime_knobs = dict(
        manifest.get("best_runtime", {}).get(plan["topology"], {}).get("knobs") or {}
    )
    # Thread counts are node-relative and the managed topology contract always
    # uses every physical core. Retain other measured knobs for this topology.
    saved_runtime_knobs.pop("OMP_NUM_THREADS", None)
    managed_ctx = int(runtime.get("ctx", 4096))
    managed_slots = int(runtime.get("kv_slots", 1))
    managed_cap = int(runtime.get("cache_cap", 8))
    fingerprint_dir = manifest["model_fingerprint"].split(":", 1)[-1]
    try:
        startup_timeout = float(os.environ.get("COLI_RAMDISK_START_TIMEOUT", "7200"))
    except ValueError:
        raise RamdiskError("COLI_RAMDISK_START_TIMEOUT must be numeric")
    if not math.isfinite(startup_timeout) or not 1 <= startup_timeout <= 86400:
        raise RamdiskError("COLI_RAMDISK_START_TIMEOUT must be between 1 and 86400 seconds")
    # Replica processes load concurrently after their wrappers are spawned.
    # Admit the complete set from one cgroup snapshot before the first child,
    # rather than allowing every node to reuse the same uncharged headroom.
    _admit_concurrent_runtimes(plan, manifest["mounts"], benchmark=False)
    spawned = []
    launch_contexts = []
    manifest["processes"] = []
    manifest["ports"] = []
    manifest["state"] = "starting"
    _save_manifest(manifest)
    try:
        for index, mount in enumerate(manifest["mounts"]):
            _raise_if_cancelled(cancel_event)
            node = mount.get("node")
            label = "interleaved" if node is None else "node-%d" % node
            state_dir = os.path.join(_state_root(), "engines", fingerprint_dir, label)
            _ensure_private_dir(state_dir)
            _assert_durable_state_dir(state_dir, plan=plan)
            _recover_delta(state_dir, canonical_usage, plan=plan)
            baseline = _usage_read(canonical_usage)
            _usage_write(os.path.join(state_dir, ".coli_usage"), baseline)
            context = {"state_dir": state_dir, "usage_baseline": baseline, "record": None}
            launch_contexts.append(context)
            port = base_port + (0 if node is None else int(node))
            environment = os.environ.copy()
            environment.update(
                {
                    "COLI_WEIGHTS_DIR": mount["path"],
                    "COLI_STATE_DIR": state_dir,
                    "COLI_RAMMAP": "1",
                    "COLI_RAM_PREFAULT": str(plan["prefault"]),
                    "COLI_MANAGED_NONCE": nonce,
                    "COLI_NUMA": "1" if _managed_numa_enabled(plan, node) else "0",
                    "COLI_NUMA_NODES": _memory_node_list(plan, node=node),
                    "COLI_CPU_AFFINITY": _engine_cpu_list(plan, node=node),
                    "OMP_NUM_THREADS": str(_node_core_count(plan, node)),
                    "OMP_PROC_BIND": "close",
                    "OMP_PLACES": "cores",
                    "CTX": str(managed_ctx),
                    "KV_SLOTS": str(managed_slots),
                    "COLI_KV_SLOTS": str(managed_slots),
                    # Managed engines deliberately keep durable, node-specific
                    # KV state under COLI_STATE_DIR.  Do not inherit a shell
                    # override that silently disables that lifecycle promise.
                    "KVSAVE": "1",
                    "CAP_RAISE": "0",
                    "AUTOPIN": "0",
                    "PROF": "1",
                }
            )
            for inherited in (
                "COLI_MMAP",
                "PIN",
                "PIN_GB",
                "PIN_FILL",
                "RAM_GB",
                "COLI_RAM_OVERCOMMIT",
                "CUDA_EXPERT_GB",
                "CUDA_DENSE",
                "COLI_GPUS",
                "COLI_GPU",
                "COLI_CUDA",
                "COLI_METAL",
                "COLI_NO_OMP_TUNE",
                "COLI_OMP_TUNED",
                "DIRECT",
                "PIPE",
                "PIPE_WORKERS",
                "URING",
            ):
                environment.pop(inherited, None)
            applied_runtime_knobs = _normalized_runtime_knobs(
                plan, saved_runtime_knobs, node=node
            )
            for key, value in applied_runtime_knobs.items():
                environment[key] = str(value)
            command = [
                cli_path,
                "serve",
                "--model",
                model,
                "--port",
                str(port),
                "--cap",
                str(managed_cap),
                "--ctx",
                str(managed_ctx),
                "--kv-slots",
                str(managed_slots),
            ]
            if not os.access(cli_path, os.X_OK):
                command.insert(0, sys.executable)
            if node is not None:
                command = [
                    managed_numactl,
                    "--physcpubind=%s" % _engine_cpu_list(plan, node=node),
                    "--membind=%d" % node,
                ] + command
            log_path = os.path.join(state_dir, "engine.log")
            log = open(log_path, "ab", buffering=0)
            try:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            finally:
                log.close()
            spawned.append(process)
            identity = None
            identity_deadline = time.monotonic() + 1.0
            while time.monotonic() < identity_deadline and process.poll() is None:
                _raise_if_cancelled(cancel_event)
                identity = _proc_identity(process.pid)
                if identity and identity.get("pgid") == process.pid and identity.get("nonce") == nonce:
                    break
                time.sleep(0.01)
            if (
                process.poll() is not None
                or not identity
                or identity.get("pgid") != process.pid
                or identity.get("nonce") != nonce
            ):
                raise RamdiskError("managed engine exited during launch; see %s" % log_path)
            record = {
                "pid": process.pid,
                "pgid": identity["pgid"],
                "uid": identity["uid"],
                "starttime": identity["starttime"],
                "nonce": nonce,
                "port": port,
                "node": node,
                "command": command,
                "state_dir": state_dir,
                "weights_dir": mount["path"],
                "usage_baseline": baseline,
                "started_at": _utc_now(),
                "log": log_path,
                "runtime_knobs": applied_runtime_knobs,
            }
            context["record"] = record
            records.append(record)
            manifest["processes"] = records
            manifest["ports"] = [item["port"] for item in records]
            manifest["state"] = "starting"
            _save_manifest(manifest)
        # Every node is launched first, so replicated model loading proceeds in
        # parallel. Commit `running` only after each verified wrapper exposes a
        # healthy API (which occurs after its engine/model has fully loaded).
        for record in records:
            _wait_managed_ready(
                record,
                startup_timeout,
                api_key=os.environ.get("COLI_API_KEY"),
                cancel_event=cancel_event,
            )
            _save_manifest(manifest)
        _raise_if_cancelled(cancel_event)
        manifest["state"] = "running"
        _save_manifest(manifest)
        for process in spawned:
            _track_managed_child(process)
        return manifest
    except BaseException as launch_error:
        cleanup_failures = []
        surviving_groups = set()

        def rollback_save(label):
            try:
                _save_manifest(manifest)
                return True
            except Exception as save_exc:
                cleanup_failures.append("%s: %s" % (label, save_exc))
                return False

        records_by_pid = {
            int(record["pid"]): record
            for record in records
        }
        # Published children use the same immediate identity+nonce revalidation
        # as Stop. A child interrupted before its record was published is
        # signaled only through its exact Popen PID, never through a PGID that
        # could have been recycled.
        for process in reversed(spawned):
            record = records_by_pid.get(int(process.pid))
            if record is not None:
                _track_managed_child(process)
                failure = _terminate_verified_group(record)
            else:
                failure = _terminate_direct_child(process)
            try:
                process.wait(timeout=1)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
            _forget_managed_child(process.pid)
            if record is not None:
                still_alive = _process_matches(record)[0]
                surviving_pgid = int(record["pgid"])
            else:
                still_alive = process.poll() is None
                surviving_pgid = int(process.pid)
            if failure and still_alive:
                cleanup_failures.append(failure)
                surviving_groups.add(surviving_pgid)
        for context in launch_contexts:
            record = context.get("record") or {
                "state_dir": context["state_dir"],
                "usage_baseline": context["usage_baseline"],
            }
            if context.get("record") and record["pgid"] in surviving_groups:
                record["stop_error"] = "launch rollback could not terminate process group"
                continue
            record.setdefault("usage_merge_id", secrets.token_hex(16))
            transaction_persisted = bool(context.get("record")) and rollback_save(
                "could not persist usage transaction for %s" % context["state_dir"]
            )
            try:
                # If the transaction id could not be serialized, retain the
                # durable delta journal after applying it. A future start then
                # recognizes the canonical marker before removing the journal,
                # rather than inventing a new id and replaying the delta.
                _merge_usage(
                    record,
                    canonical_usage,
                    plan=plan,
                    keep_journal=not transaction_persisted,
                )
                if context.get("record"):
                    record["usage_merged_at"] = _utc_now()
                    record["stopped_at"] = _utc_now()
                    rollback_save(
                        "could not persist usage recovery for %s" % context["state_dir"]
                    )
            except Exception as exc:
                if context.get("record"):
                    record["usage_merge_error"] = str(exc)
                cleanup_failures.append("usage recovery for %s: %s" % (context["state_dir"], exc))
        if isinstance(launch_error, _OperationCancelled) and not cleanup_failures:
            manifest["state"] = previous_state
            manifest["processes"] = previous_processes
            manifest["ports"] = previous_ports
            manifest["base_port"] = previous_base_port
            manifest.pop("launch_error", None)
            manifest.pop("cleanup_errors", None)
            try:
                _save_manifest(manifest)
            except Exception as save_exc:
                cleanup_failures.append(
                    "could not persist clean launch cancellation: %s" % save_exc
                )
            if not cleanup_failures:
                raise

        manifest["state"] = "error"
        manifest["launch_error"] = str(launch_error)
        manifest["cleanup_errors"] = cleanup_failures
        rollback_save("could not persist launch rollback")
        if cleanup_failures:
            raise RamdiskError(
                "%s; launch rollback/reporting errors: %s"
                % (launch_error, "; ".join(cleanup_failures))
            ) from launch_error
        raise


@_exclusive_lifecycle
def stop(args=None):
    manifest = _load_manifest(required=True)
    plan = manifest["plan"]
    canonical_usage = os.path.join(manifest["plan"]["model"]["path"], ".coli_usage")
    refusals = []
    identities = []
    for record in manifest.get("processes", []):
        if record.get("stopped_at"):
            identities.append((record, False, "already-stopped", None))
            continue
        matches, reason, actual = _process_matches(record)
        if not matches and reason != "not-running":
            refusals.append("PID %s is %s" % (record.get("pid"), reason))
        identities.append((record, matches, reason, actual))
    # Validate the complete set before signaling any process.  A stale/foreign
    # record must make stop all-or-nothing, not leave a partially stopped
    # topology whose remaining process still owns the mounts.
    if refusals:
        raise RamdiskError("refusing to signal unverified processes: " + "; ".join(refusals))
    for record, _, _, _ in identities:
        if not record.get("usage_merged_at"):
            record.setdefault("usage_merge_id", secrets.token_hex(16))
    # Persist transaction ids before signaling. Recovery can then recognize a
    # canonical marker even if this manager dies between the atomic usage-file
    # replacement and the final manifest update.
    _save_manifest(manifest)
    failures = []
    for record, matches, reason, actual in identities:
        if matches:
            pgid = int(record.get("pgid", record["pid"]))
            failure = _terminate_verified_group(record)
            if failure:
                record["stop_error"] = failure
                failures.append("PID/PGID %s survived SIGKILL" % pgid)
                continue
        if not record.get("usage_merged_at"):
            try:
                _merge_usage(record, canonical_usage, plan=plan)
                record["usage_merged_at"] = _utc_now()
                record.pop("usage_merge_error", None)
                # Each node is its own committed transaction. Persist success
                # before advancing so a crash cannot replay an earlier node.
                _save_manifest(manifest)
            except Exception as exc:
                record["usage_merge_error"] = str(exc)
                failures.append("PID %s usage delta was not merged: %s" % (record.get("pid"), exc))
        record.setdefault("stopped_at", _utc_now())
        record.pop("stop_error", None)
    planned_paths = {record["path"] for record in plan["mounts"]}
    recorded_paths = {record["path"] for record in manifest.get("mounts", [])}
    incomplete_mount_layout = recorded_paths != planned_paths
    manifest["state"] = "error" if failures or incomplete_mount_layout or any(
        record.get("stop_error") or record.get("usage_merge_error")
        for record in manifest.get("processes", [])
    ) else "stopped"
    _save_manifest(manifest)
    if failures:
        raise RamdiskError("engines were signaled but cleanup is incomplete: " + "; ".join(failures))
    return manifest


def _busy_mount_references(path):
    path = os.path.normpath(path) + os.sep
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        for leaf in ("cwd", "root"):
            try:
                target = os.path.realpath("/proc/%s/%s" % (entry, leaf)) + os.sep
                if target.startswith(path):
                    found.append(int(entry))
                    break
            except OSError:
                pass
        if found and found[-1] == int(entry):
            continue
        # A process can close the shard fd after mmap(); the mapping remains a
        # live mount reference and appears only in /proc/<pid>/maps.
        for line in _read_text("/proc/%s/maps" % entry).splitlines():
            fields = line.split(None, 5)
            if len(fields) < 6 or not fields[5].startswith("/"):
                continue
            mapped = fields[5]
            if mapped.endswith(" (deleted)"):
                mapped = mapped[: -len(" (deleted)")]
            target = os.path.realpath(mapped) + os.sep
            if target.startswith(path):
                found.append(int(entry))
                break
        if found and found[-1] == int(entry):
            continue
        fd_dir = "/proc/%s/fd" % entry
        try:
            descriptors = os.listdir(fd_dir)
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.path.realpath(os.path.join(fd_dir, descriptor)) + os.sep
                if target.startswith(path):
                    found.append(int(entry))
                    break
            except OSError:
                pass
    return sorted(set(found))


def _managed_path(path, mount_root):
    normalized = os.path.normpath(os.path.abspath(path))
    root = os.path.normpath(os.path.abspath(mount_root))
    if root in ("/", "", os.path.expanduser("~")):
        return False
    return normalized == root or os.path.commonpath([normalized, root]) == root


@_exclusive_lifecycle
def destroy(args, expected_manifest_token=None):
    manifest = _load_manifest(required=True)
    if (
        expected_manifest_token is not None
        and _manifest_confirmation_token(manifest) != expected_manifest_token
    ):
        raise RamdiskError(
            "RAM workspace changed since review; inspect the active deployment and confirm Destroy again"
        )
    _confirm("Stop engines and unmount all volatile RAM-disk weights?", bool(getattr(args, "yes", False)))
    if manifest.get("processes"):
        manifest = stop(args)
    root = manifest["plan"]["mount_root"]
    preserved_mountpoints = []
    all_mounts_verified_here = True
    verified_mounts = []
    # Preflight every replica before changing any mount. A foreign or busy last
    # node must not leave the earlier nodes already unmounted.
    planned_mounts = manifest["plan"]["mounts"]
    managed_paths = [record["path"] for record in planned_mounts]
    recorded_by_path = {
        record["path"]: record for record in manifest.get("mounts", [])
    }
    nested_mounts = sorted(
        mount["path"]
        for mount in _mount_table()
        if any(_path_is_below(mount["path"], path) for path in managed_paths)
    )
    if nested_mounts:
        raise RamdiskError(
            "refusing managed mount(s) with nested child mounts: %s"
            % ", ".join(nested_mounts)
        )
    for planned in planned_mounts:
        path = planned["path"]
        if not _managed_path(path, root):
            raise RamdiskError("refusing unsafe managed path: %s" % path)
        actual = _mount_at(path)
        record = recorded_by_path.get(path)
        if actual and record is None:
            # A prepare error may have occurred after mount(8) succeeded but
            # before an identity could be captured. Without a recorded
            # mount-id/device pair this could now be foreign, so retain the
            # recovery manifest and require an operator to resolve it.
            raise RamdiskError(
                "refusing unverified surviving mount at planned path: %s" % path
            )
        if record is None:
            all_mounts_verified_here = False
            preserved_mountpoints.append(path)
            continue
        expected = record.get("identity", {})
        if actual:
            if (
                actual.get("filesystem") != "tmpfs"
                or actual.get("source") != "tmpfs"
                or actual.get("mount_id") != expected.get("mount_id")
                or actual.get("device") != expected.get("device")
            ):
                raise RamdiskError("refusing foreign or replaced mount: %s" % path)
            try:
                validated = _validate_mount(record, manifest["plan"])
            except RamdiskError as exc:
                raise RamdiskError("refusing foreign or altered mount at %s: %s" % (path, exc))
            if (
                validated["mount_id"] != expected.get("mount_id")
                or validated["device"] != expected.get("device")
            ):
                raise RamdiskError("refusing foreign or replaced mount: %s" % path)
            if manifest.get("state") in ("ready", "stopped"):
                _validate_namespace(manifest["plan"], record, sample_numa=False)
            busy = _busy_mount_references(path)
            if busy:
                raise RamdiskError("mount %s is busy in PID(s): %s" % (path, ",".join(str(pid) for pid in busy)))
            verified_mounts.append(record)
        else:
            # An externally unmounted path no longer has an identity we can
            # prove. Never remove it based only on serialized metadata.
            all_mounts_verified_here = False
            preserved_mountpoints.append(path)
            continue
    for record in reversed(verified_mounts):
        path = record["path"]
        _umount_path(path, manifest["plan"]["hardware"])
        if record.get("path_preexisting"):
            preserved_mountpoints.append(path)
            continue
        try:
            os.rmdir(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EPERM):
                # X-mount.mkdir creates /mnt targets as root.  Sudo remains
                # deliberately limited to umount; leaving a verified empty
                # mountpoint is safer than broadening privilege to rmdir.
                preserved_mountpoints.append(path)
                continue
            if exc.errno not in (errno.ENOTEMPTY, errno.EBUSY):
                raise
            raise RamdiskError("refusing to remove non-empty mount path: %s" % path)
    if (
        manifest["plan"]["topology"] == "per-node"
        and all_mounts_verified_here
        and not manifest["plan"].get("mount_root_preexisting", True)
    ):
        try:
            os.rmdir(root)
        except FileNotFoundError:
            pass
        except OSError:
            pass
    _durable_unlink(_manifest_path())
    return {
        "destroyed": True,
        "durable_state_preserved": True,
        "benchmark_history_preserved": True,
        "empty_mountpoints_preserved": sorted(set(preserved_mountpoints)),
    }


def status(deep=True):
    """Return lifecycle status, optionally skipping shard/header revalidation.

    Scriptable ``status`` always uses the deep default.  The curses dashboard
    polls the cheap form and exposes an explicit refresh for a new deep model
    scan, avoiding repeated reads of every safetensors header on large models.
    """
    manifest = _load_manifest(required=False)
    result = {
        "schema": STATUS_SCHEMA,
        "version": MANIFEST_VERSION,
        "manifest_path": _manifest_path(),
        "present": bool(manifest),
        "state": "absent" if not manifest else manifest.get("state", "unknown"),
        "deep_validation": bool(deep),
        "mounts": [],
        "processes": [],
    }
    if not manifest:
        return result
    source_verified = None if not deep else True
    source_error = None
    if deep:
        try:
            _source_still_matches(manifest["plan"])
        except RamdiskError as exc:
            source_verified = False
            source_error = str(exc)
    for record in manifest.get("mounts", []):
        actual = _mount_at(record["path"])
        expected = record.get("identity", {})
        identity_verified = bool(
            actual
            and actual["filesystem"] == "tmpfs"
            and actual["source"] == "tmpfs"
            and actual["mount_id"] == expected.get("mount_id")
            and actual["device"] == expected.get("device")
        )
        options_verified = False
        namespace_verified = None if not deep else False
        option_error = namespace_error = None
        if identity_verified:
            try:
                _validate_mount(record, manifest["plan"])
                options_verified = True
            except RamdiskError as exc:
                option_error = str(exc)
            if deep and options_verified and source_verified:
                try:
                    _validate_namespace(manifest["plan"], record, sample_numa=False)
                    namespace_verified = True
                except (OSError, RamdiskError) as exc:
                    namespace_error = str(exc)
        result["mounts"].append(
            {
                "path": record["path"],
                "node": record.get("node"),
                "mounted": bool(actual),
                "verified": identity_verified and options_verified and (
                    namespace_verified if deep else True
                ),
                "identity_verified": identity_verified,
                "options_verified": options_verified,
                "namespace_verified": namespace_verified,
                "option_error": option_error,
                "namespace_error": namespace_error,
                "filesystem": actual.get("filesystem") if actual else None,
                "numa_allocation": record.get("numa_allocation", {}),
            }
        )
    for record in manifest.get("processes", []):
        if record.get("stopped_at"):
            matches, reason = False, "stopped"
        else:
            matches, reason, _ = _process_matches(record)
        result["processes"].append(
            {
                "pid": record.get("pid"),
                "port": record.get("port"),
                "node": record.get("node"),
                "running": matches,
                "verified": matches,
                "reason": reason,
                "state_dir": record.get("state_dir"),
                "log": record.get("log"),
            }
        )
    result["model_fingerprint"] = manifest.get("model_fingerprint")
    result["mode"] = manifest["plan"].get("mode")
    result["topology"] = manifest["plan"].get("topology")
    result["ports"] = manifest.get("ports", [])
    result["source_fingerprint_verified"] = source_verified
    result["source_fingerprint_error"] = source_error
    return result


BENCHMARK_PROMPT = "Explain in two sentences why deterministic validation matters."


def _source_build_identity():
    """Best-effort revision metadata for reproducible benchmark reports."""
    explicit = os.environ.get("COLI_BUILD_COMMIT")
    if explicit:
        return {"revision": explicit, "working_tree_modified": None}
    git = shutil.which("git")
    if not git:
        return {"revision": None, "working_tree_modified": None}
    source_dir = os.path.dirname(os.path.abspath(__file__))
    revision = _run([git, "-C", source_dir, "rev-parse", "HEAD"])
    if revision.returncode:
        return {"revision": None, "working_tree_modified": None}
    status = _run([git, "-C", source_dir, "status", "--porcelain"])
    return {
        "revision": revision.stdout.strip() or None,
        "working_tree_modified": None if status.returncode else bool(status.stdout.strip()),
    }


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(percentile * len(ordered))) - 1))
    return ordered[index]


def _parse_profiler(text, elapsed):
    rates = [float(value) for value in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*tok(?:en)?s?/s", text, re.I)]
    forward_p50 = None
    forward_p99 = None
    match = re.search(r"forward[^\n]*p50[=: ]+([0-9.]+)\s*ms[^\n]*p99[=: ]+([0-9.]+)\s*ms", text, re.I)
    if match:
        forward_p50, forward_p99 = float(match.group(1)), float(match.group(2))
    ram_experts = ram_bytes = None
    match = re.search(r"RAM map:\s*(\d+) experts / ([0-9.]+) GB", text)
    if match:
        ram_experts, ram_bytes = int(match.group(1)), float(match.group(2)) * 1e9
    io_bytes = None
    match = re.search(r"(?:physical SSD|disk I/O|expert I/O)[^\n]*?([0-9.]+)\s*(GB|MB|bytes)", text, re.I)
    if match:
        scale = {"gb": 1e9, "mb": 1e6, "bytes": 1}[match.group(2).lower()]
        io_bytes = float(match.group(1)) * scale
    prefault = None
    match = re.search(r"prefaulted in ([0-9.]+)s", text)
    if match:
        prefault = float(match.group(1))
    ttft_ms = None
    match = re.search(r"TTFT\s+([0-9.]+)s", text, re.I)
    if match:
        ttft_ms = float(match.group(1)) * 1000.0
    rss_bytes = None
    match = re.search(r"\bRSS\s+([0-9.]+)\s+GB", text, re.I)
    if match:
        rss_bytes = float(match.group(1)) * 1e9
    return {
        "elapsed_seconds": elapsed,
        "tokens_per_second": rates[-1] if rates else (32.0 / elapsed if elapsed else None),
        "forward_p50_ms": forward_p50,
        "forward_p99_ms": forward_p99,
        "rammap_experts": ram_experts,
        "rammap_bytes": ram_bytes,
        "physical_ssd_bytes": io_bytes,
        "prefault_seconds": prefault,
        "ttft_ms": ttft_ms,
        "rss_bytes": rss_bytes,
    }


def _resolve_engine_path(cli_path, engine_path=None):
    candidates = []
    if engine_path:
        candidates.append(engine_path)
    here = os.path.dirname(os.path.abspath(cli_path))
    suffix = ".exe" if os.name == "nt" else ""
    candidates.extend(
        [
            os.path.join(here, "colibri" + suffix),
            os.path.join(
                os.path.dirname(here), "libexec", "colibri", "colibri" + suffix
            ),
            os.path.join(here, "glm" + suffix),
            os.path.join(
                os.path.dirname(here), "libexec", "colibri", "glm" + suffix
            ),
        ]
    )
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.realpath(candidate)
    raise RamdiskError("cannot locate the executable Colibri engine for persistent benchmarking")


def _node_core_count(plan, node=None):
    for target in plan.get("placement", {}).get("engine_cpu_sets", []):
        if target.get("node") == node:
            return max(1, int(target.get("physical_cores", 0)))
    if node is None:
        return max(1, int(plan["hardware"]["physical_cores"]))
    try:
        return max(
            1,
            int(
                next(
                    item["physical_cores"]
                    for item in plan["hardware"]["nodes"]
                    if int(item["id"]) == int(node)
                )
            ),
        )
    except StopIteration:
        raise RamdiskError("NUMA node %s is absent from the recorded hardware plan" % node)


def _engine_cpu_list(plan, node=None):
    for target in plan.get("placement", {}).get("engine_cpu_sets", []):
        if target.get("node") == node:
            value = target.get("cpu_list")
            if isinstance(value, str) and value:
                return value
            cpus = target.get("cpus")
            if isinstance(cpus, list) and cpus:
                return _format_range_list(cpus)
    if node is None:
        cpus = plan.get("hardware", {}).get("effective_cpus")
        if not cpus:
            cpus = [
                cpu
                for row in plan.get("hardware", {}).get("nodes", [])
                for cpu in row.get("cpus", [])
            ]
    else:
        cpus = next(
            (
                row.get("cpus", [])
                for row in plan.get("hardware", {}).get("nodes", [])
                if row.get("id") == node
            ),
            [],
        )
    if not cpus:
        raise RamdiskError("managed engine CPU mask is empty")
    return _format_range_list(cpus)


def _memory_node_list(plan, node=None):
    if node is not None:
        return str(int(node))
    nodes = plan.get("placement", {}).get(
        "memory_nodes", plan.get("hardware", {}).get("online_nodes", [])
    )
    if not nodes:
        raise RamdiskError("managed memory-node mask is empty")
    return _format_range_list(nodes)


def _managed_numa_enabled(plan, node=None):
    """Use the engine policy for every shared plan, including one-node binds."""
    if node is not None:
        return False
    nodes = plan.get("placement", {}).get(
        "memory_nodes", plan.get("hardware", {}).get("online_nodes", [])
    )
    return bool(nodes)


def _normalized_runtime_knobs(plan, knobs, node=None):
    """Validate the small benchmark-knob vocabulary before placing it in env."""
    result = {}
    thread_limit = _node_core_count(plan, node)
    for key, value in (knobs or {}).items():
        if key in ("PIPE", "DIRECT", "URING"):
            parsed = int(value)
            if parsed not in (0, 1):
                raise RamdiskError("%s benchmark knob must be 0 or 1" % key)
            result[key] = parsed
        elif key == "PIPE_WORKERS":
            parsed = int(value)
            if not 1 <= parsed <= max(64, thread_limit):
                raise RamdiskError("PIPE_WORKERS benchmark knob is outside its safe range")
            result[key] = parsed
        elif key == "OMP_NUM_THREADS":
            parsed = int(value)
            if not 1 <= parsed <= thread_limit:
                raise RamdiskError(
                    "OMP_NUM_THREADS=%s exceeds the %s-core benchmark target"
                    % (parsed, thread_limit)
                )
            result[key] = parsed
        elif key == "OMP_PROC_BIND":
            if value not in ("close", "spread"):
                raise RamdiskError("OMP_PROC_BIND benchmark knob must be close or spread")
            result[key] = value
        else:
            raise RamdiskError("unsupported managed benchmark knob: %s" % key)
    return result


def _benchmark_environment(manifest, weights_dir, state_dir, rammap, node=None, knobs=None):
    plan = manifest["plan"]
    runtime = plan.get("managed_runtime", {})
    environment = os.environ.copy()
    for inherited in (
        "COLI_MMAP",
        "PIN",
        "PIN_GB",
        "PIN_FILL",
        "RAM_GB",
        "COLI_RAM_OVERCOMMIT",
        "CUDA_EXPERT_GB",
        "CUDA_DENSE",
        "COLI_GPUS",
        "COLI_GPU",
        "COLI_CUDA",
        "COLI_METAL",
        "COLI_NO_OMP_TUNE",
        "COLI_OMP_TUNED",
        "DIRECT",
        "PIPE",
        "PIPE_WORKERS",
        "URING",
        "DSA_TOPK",
        "GRAMMAR",
    ):
        environment.pop(inherited, None)
    environment.update(
        {
            "COLI_WEIGHTS_DIR": weights_dir,
            "COLI_STATE_DIR": state_dir,
            "COLI_RAMMAP": "1" if rammap else "0",
            "COLI_RAM_PREFAULT": str(plan["prefault"] if rammap else 0),
            "COLI_NUMA": "1" if _managed_numa_enabled(plan, node) else "0",
            "COLI_NUMA_NODES": _memory_node_list(plan, node=node),
            "COLI_CPU_AFFINITY": _engine_cpu_list(plan, node=node),
            "OMP_NUM_THREADS": str(_node_core_count(plan, node)),
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "TEMP": "0",
            "DRAFT": "0",
            "KVSAVE": "0",
            "AUTOPIN": "0",
            "REPIN": "0",
            "CACHE_ROUTE": "0",
            "TOPK": "0",
            "TOPP": "0",
            "EXPERT_BUDGET": "0",
            "PREFETCH": "0",
            "PILOT": "0",
            "PILOT_REAL": "0",
            "CAP_RAISE": "0",
            "COLI_POLICY": "quality",
            "CTX": str(int(runtime.get("ctx", 4096))),
            "KV_SLOTS": "1",
            "COLI_KV_SLOTS": "1",
            "PROF": "1",
        }
    )
    for key, value in _normalized_runtime_knobs(plan, knobs, node=node).items():
        environment[key] = str(value)
    return environment


def _cancellable_engine_type(
    engine_type,
    read_engine_turn,
    ready_marker,
    cancel_event,
):
    """Adapt benchmark engine startup without changing the shared Engine API."""
    if cancel_event is None:
        return engine_type

    class CancellableEngine(engine_type):
        @classmethod
        def _wait_until_ready(cls, process, timeout):
            outcome = queue.Queue(maxsize=1)

            def read_ready():
                try:
                    read_engine_turn(process.stdout, ready_marker, lambda _: None)
                except BaseException as error:
                    outcome.put(error)
                else:
                    outcome.put(None)

            reader = threading.Thread(
                target=read_ready,
                name="colibri-benchmark-ready",
                daemon=True,
            )
            reader.start()
            deadline = time.monotonic() + timeout
            try:
                while True:
                    _raise_if_cancelled(cancel_event)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(
                            "colibri engine did not become ready within %.3g seconds"
                            % timeout
                        )
                    try:
                        error = outcome.get(timeout=min(0.2, remaining))
                    except queue.Empty:
                        continue
                    if error is not None:
                        raise error
                    break
            except BaseException:
                cls._terminate_process(process)
                reader.join(timeout=5)
                raise
            reader.join()

    return CancellableEngine


def _benchmark_generate(
    engine,
    prompt,
    on_text,
    cancel_event,
    client_cancelled_type,
):
    """Run one benchmark turn and interrupt even before the first token."""
    if cancel_event is None:
        return engine.generate(prompt, 32, 0.0, 1.0, on_text, cache_slot=0)

    done = threading.Event()
    close_errors = []

    def cancel_watch():
        while not done.wait(0.1):
            if not cancel_event.is_set():
                continue
            try:
                # generate() waits on its response queue before the first token,
                # so its callback alone cannot cancel a long TTFT. Closing the
                # benchmark-only engine wakes that queue and terminates the child.
                engine.close()
            except BaseException as exc:
                close_errors.append(exc)
            return

    watcher = threading.Thread(
        target=cancel_watch,
        name="colibri-benchmark-cancel",
        daemon=True,
    )
    watcher.start()
    try:
        try:
            result = engine.generate(
                prompt,
                32,
                0.0,
                1.0,
                on_text,
                cache_slot=0,
                cancelled=cancel_event.is_set,
            )
        except client_cancelled_type:
            watcher.join()
            if close_errors:
                raise _EngineCleanupError(
                    "benchmark cancellation could not close its engine: %s"
                    % close_errors[0]
                )
            _raise_if_cancelled(cancel_event)
            raise
        except BaseException as exc:
            if cancel_event.is_set():
                watcher.join()
                if close_errors:
                    raise _EngineCleanupError(
                        "benchmark cancellation could not close its engine: %s"
                        % close_errors[0]
                    ) from exc
                raise _OperationCancelled(
                    "benchmark cancelled by user at a safe checkpoint"
                ) from exc
            raise
        if cancel_event.is_set():
            watcher.join()
            if close_errors:
                raise _EngineCleanupError(
                    "benchmark cancellation could not close its engine: %s"
                    % close_errors[0]
                )
            _raise_if_cancelled(cancel_event)
        return result
    finally:
        done.set()
        watcher.join(timeout=1)


def _score_variant(
    engine_path,
    manifest,
    name,
    weights_dir,
    rammap,
    knobs,
    cancel_event=None,
):
    # Import the existing stdlib-only engine protocol client lazily. This keeps
    # plan/status usable even in minimal packaging probes while ensuring one
    # persistent process receives the warm-up and all three measured turns.
    from openai_server import (
        READY,
        ClientCancelled,
        Engine as BaseEngine,
        read_engine_turn,
        render_chat,
    )

    Engine = _cancellable_engine_type(
        BaseEngine,
        read_engine_turn,
        READY,
        cancel_event,
    )

    runtime = manifest["plan"].get("managed_runtime", {})
    fingerprint_dir = manifest["model_fingerprint"].split(":", 1)[-1]
    safe_name = re.sub(r"[^a-z0-9_.-]", "-", name.lower())
    state_dir = os.path.join(_state_root(), "benchmark-state", fingerprint_dir, safe_name)
    target_mount = manifest["mounts"][0]
    command_prefix = []
    node = None
    if manifest["plan"]["topology"] == "per-node":
        node = int(target_mount["node"])
        command_prefix = [
            _fresh_user_binary("numactl"),
            "--physcpubind=%s" % _engine_cpu_list(
                manifest["plan"], node=node
            ),
            "--membind=%d" % node,
        ]
    environment = _benchmark_environment(
        manifest, weights_dir, state_dir, rammap, node=node, knobs=knobs
    )
    _ensure_private_dir(state_dir)
    _assert_durable_state_dir(state_dir, plan=manifest["plan"])
    _admit_runtime(manifest["plan"], target_mount, benchmark=not rammap)
    prompt = render_chat(
        [{"role": "user", "content": BENCHMARK_PROMPT}],
        False,
        None,
        None,
        None,
    )
    log_path = os.path.join(state_dir, "benchmark.log")
    log = open(log_path, "ab", buffering=0)
    engine = None
    try:
        _raise_if_cancelled(cancel_event)
        engine = Engine(
            engine_path,
            manifest["plan"]["model"]["path"],
            cap=int(runtime.get("cache_cap", 8)),
            max_tokens=32,
            env=environment,
            kv_slots=1,
            command_prefix=command_prefix,
            stderr=log,
        )

        def run_once():
            _raise_if_cancelled(cancel_event)
            parts = []
            profile_seq = engine.profile_seq
            started = time.monotonic()
            stats = _benchmark_generate(
                engine,
                prompt,
                parts.append,
                cancel_event,
                ClientCancelled,
            )
            _raise_if_cancelled(cancel_event)
            elapsed = time.monotonic() - started
            if stats.get("completion_tokens") != 32:
                raise RamdiskError(
                    "benchmark produced %s tokens instead of the required 32"
                    % stats.get("completion_tokens")
                )
            if engine.profile_seq <= profile_seq or not engine.profile:
                raise RamdiskError("engine did not emit the required benchmark telemetry")
            profile = dict(engine.profile[-1])
            output = "".join(parts)
            return {
                "elapsed_seconds": elapsed,
                "tokens_per_second": stats.get("tokens_per_second"),
                "forward_p50_ms": profile.get("forward_p50_ms"),
                "forward_p99_ms": profile.get("forward_p99_ms"),
                "rammap_experts": profile.get("rammap_experts"),
                "rammap_bytes": profile.get("rammap_bytes"),
                "physical_ssd_bytes": profile.get("physical_ssd_bytes"),
                "physical_ssd_valid": profile.get("physical_ssd_valid"),
                "prefault_seconds": profile.get("prefault_seconds"),
                "ttft_ms": profile.get("ttft_ms"),
                "rss_bytes": float(stats.get("rss_gb", 0.0)) * 1e9,
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            }

        run_once()  # warm-up on this exact process/LRU
        runs = [run_once() for _ in range(3)]
    finally:
        cleanup_failures = []
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                cleanup_failures.append("engine: %s" % exc)
        try:
            log.close()
        except Exception as exc:
            cleanup_failures.append("log: %s" % exc)
        if cleanup_failures:
            raise _EngineCleanupError(
                "benchmark variant cleanup failed: %s"
                % "; ".join(cleanup_failures)
            )
    outputs = {run["output_sha256"] for run in runs}
    if len(outputs) != 1:
        raise RamdiskError("greedy benchmark output changed across deterministic runs")
    observed_experts = {run.get("rammap_experts") for run in runs}
    observed_bytes = {run.get("rammap_bytes") for run in runs}
    expected_experts = manifest["plan"]["staging"]["direct_mapped_expert_count"] if rammap else 0
    expected_bytes = manifest["plan"]["staging"]["direct_mapped_bytes"] if rammap else 0
    if observed_experts != {expected_experts} or observed_bytes != {expected_bytes}:
        raise RamdiskError(
            "RAM-map telemetry mismatch: expected %d experts/%d bytes, observed %s/%s"
            % (expected_experts, expected_bytes, sorted(observed_experts, key=str), sorted(observed_bytes, key=str))
        )
    if rammap and manifest["plan"]["mode"] == "full":
        if any(run.get("physical_ssd_valid") is not True for run in runs):
            raise RamdiskError(
                "full direct RAM-map benchmark could not verify physical SSD reads"
            )
        if any(run.get("physical_ssd_bytes") != 0 for run in runs):
            raise RamdiskError(
                "full direct RAM-map benchmark performed physical SSD expert reads"
            )
    rates = [run["tokens_per_second"] for run in runs if run["tokens_per_second"] is not None]
    forwards50 = [run["forward_p50_ms"] for run in runs if run["forward_p50_ms"] is not None]
    forwards99 = [run["forward_p99_ms"] for run in runs if run["forward_p99_ms"] is not None]
    ssd = [run["physical_ssd_bytes"] for run in runs if run["physical_ssd_bytes"] is not None]
    ttfts = [run["ttft_ms"] for run in runs if run["ttft_ms"] is not None]
    direct_total = manifest["plan"]["model"]["complete_experts"]
    direct_count = next(iter(observed_experts))
    return {
        "name": name,
        "status": "ok",
        "knobs": knobs,
        "runs": runs,
        "output_sha256": next(iter(outputs)),
        "persistent_engine": True,
        "log": log_path,
        "interactive": {
            "ttft_ms": _percentile(ttfts, 0.50),
            "p50_tokens_per_second": _percentile(rates, 0.50),
            "p95_tokens_per_second": _percentile(rates, 0.95),
            "forward_p50_ms": _percentile(forwards50, 0.50),
            "forward_p99_ms": _percentile(forwards99, 0.99),
            "ram_map_coverage": float(direct_count) / direct_total if direct_total else 0.0,
            "ssd_bytes_per_token": (sum(ssd) / len(ssd) / 32.0) if ssd else None,
        },
    }


def _aggregate_score(manifest, engine_path=None, knobs=None, cancel_event=None):
    """Benchmark all node-local replicas under one controlled environment."""
    if manifest["plan"]["topology"] != "per-node":
        return {
            "status": "not-run",
            "reason": "aggregate score applies only to per-node topology",
            "per_node_tokens_per_second": [],
            "slowest_node_tokens_per_second": None,
            "total_tokens_per_second": None,
        }
    if manifest.get("state") == "running":
        return {
            "status": "not-run",
            "reason": "stop managed engines before the fixed-environment aggregate benchmark",
            "per_node_tokens_per_second": [],
            "slowest_node_tokens_per_second": None,
            "total_tokens_per_second": None,
        }
    if not engine_path:
        return {
            "status": "error",
            "reason": "aggregate benchmark engine path was not resolved",
            "per_node_tokens_per_second": [],
            "slowest_node_tokens_per_second": None,
            "total_tokens_per_second": None,
        }

    from openai_server import (
        READY,
        ClientCancelled,
        Engine as BaseEngine,
        read_engine_turn,
        render_chat,
    )

    Engine = _cancellable_engine_type(
        BaseEngine,
        read_engine_turn,
        READY,
        cancel_event,
    )

    plan = manifest["plan"]
    runtime = plan.get("managed_runtime", {})
    fingerprint_dir = manifest["model_fingerprint"].split(":", 1)[-1]
    prompt = render_chat(
        [{"role": "user", "content": BENCHMARK_PROMPT}], False, None, None, None
    )
    launched = []
    launched_lock = threading.Lock()
    normalized_knobs = None
    try:
        _raise_if_cancelled(cancel_event)
        numactl = _fresh_user_binary("numactl")
        mounts = list(manifest.get("mounts", []))
        if not mounts:
            raise RamdiskError("per-node aggregate benchmark has no planned replicas")
        for mount in mounts:
            node_knobs = _normalized_runtime_knobs(
                plan, knobs or {"PIPE": 0}, mount["node"]
            )
            if normalized_knobs is None:
                normalized_knobs = node_knobs
            elif node_knobs != normalized_knobs:
                raise RamdiskError(
                    "aggregate runtime knobs are not valid uniformly across nodes"
                )
        _admit_concurrent_runtimes(plan, mounts, benchmark=False)

        def launch(mount):
            _raise_if_cancelled(cancel_event)
            node = int(mount["node"])
            state_dir = os.path.join(
                _state_root(),
                "benchmark-state",
                fingerprint_dir,
                "aggregate-node-%d" % node,
            )
            _ensure_private_dir(state_dir)
            _assert_durable_state_dir(state_dir, plan=plan)
            environment = _benchmark_environment(
                manifest,
                mount["path"],
                state_dir,
                True,
                node=node,
                knobs=normalized_knobs,
            )
            log_path = os.path.join(state_dir, "benchmark.log")
            log = open(log_path, "ab", buffering=0)
            try:
                engine = Engine(
                    engine_path,
                    plan["model"]["path"],
                    cap=int(runtime.get("cache_cap", 8)),
                    max_tokens=32,
                    env=environment,
                    kv_slots=1,
                    command_prefix=[
                        numactl,
                        "--physcpubind=%s" % _engine_cpu_list(plan, node=node),
                        "--membind=%d" % node,
                    ],
                    stderr=log,
                )
            except BaseException:
                log.close()
                raise
            entry = {
                "node": node,
                "engine": engine,
                "log_stream": log,
                "log": log_path,
            }
            with launched_lock:
                launched.append(entry)
            return entry

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(mounts)) as executor:
            entries = list(executor.map(launch, mounts))
        entries.sort(key=lambda entry: entry["node"])

        def request(entry):
            _raise_if_cancelled(cancel_event)
            parts = []
            engine = entry["engine"]
            profile_seq = engine.profile_seq
            started = time.monotonic()
            stats = _benchmark_generate(
                engine,
                prompt,
                parts.append,
                cancel_event,
                ClientCancelled,
            )
            _raise_if_cancelled(cancel_event)
            elapsed = time.monotonic() - started
            tokens = int(stats.get("completion_tokens", 0) or 0)
            if tokens != 32:
                raise RamdiskError(
                    "node %s produced %d tokens instead of 32" % (entry["node"], tokens)
                )
            if engine.profile_seq <= profile_seq or not engine.profile:
                raise RamdiskError(
                    "node %s did not emit required aggregate telemetry" % entry["node"]
                )
            profile = dict(engine.profile[-1])
            return {
                "node": entry["node"],
                "tokens_per_second": tokens / elapsed if elapsed else None,
                "elapsed_seconds": elapsed,
                "rammap_experts": profile.get("rammap_experts"),
                "rammap_bytes": profile.get("rammap_bytes"),
                "physical_ssd_bytes": profile.get("physical_ssd_bytes"),
                "physical_ssd_valid": profile.get("physical_ssd_valid"),
                "prefault_seconds": profile.get("prefault_seconds"),
                "rss_bytes": float(stats.get("rss_gb", 0.0)) * 1e9,
                "output_sha256": hashlib.sha256(
                    "".join(parts).encode("utf-8")
                ).hexdigest(),
            }

        def concurrent_round():
            _raise_if_cancelled(cancel_event)
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(entries)) as executor:
                rows = list(executor.map(request, entries))
            wall = time.monotonic() - started
            return {
                "rows": rows,
                "wall_seconds": wall,
                "total_tokens_per_second": (32.0 * len(entries) / wall) if wall else None,
            }

        concurrent_round()
        rounds = [concurrent_round() for _ in range(3)]
        hashes = {
            row["output_sha256"]
            for round_result in rounds
            for row in round_result["rows"]
        }
        if len(hashes) != 1:
            raise RamdiskError(
                "deterministic aggregate outputs differed across replicas or runs"
            )
        expected_experts = plan["staging"]["direct_mapped_expert_count"]
        expected_bytes = plan["staging"]["direct_mapped_bytes"]
        for round_result in rounds:
            for row in round_result["rows"]:
                if (
                    row["rammap_experts"] != expected_experts
                    or row["rammap_bytes"] != expected_bytes
                ):
                    raise RamdiskError(
                        "node %s aggregate RAM-map telemetry does not match the staging plan"
                        % row["node"]
                    )
                if plan["mode"] == "full":
                    if row.get("physical_ssd_valid") is not True:
                        raise RamdiskError(
                            "node %s could not verify physical SSD reads"
                            % row["node"]
                        )
                    if row["physical_ssd_bytes"] != 0:
                        raise RamdiskError(
                            "node %s full aggregate run performed physical SSD expert reads"
                            % row["node"]
                        )
        by_node = {}
        for round_result in rounds:
            for row in round_result["rows"]:
                by_node.setdefault(row["node"], []).append(row["tokens_per_second"])
        summary = [
            {
                "node": node,
                "p50_tokens_per_second": _percentile(values, 0.50),
                "p95_tokens_per_second": _percentile(values, 0.95),
            }
            for node, values in sorted(by_node.items())
        ]
        p50_values = [row["p50_tokens_per_second"] for row in summary]
        total_values = [item["total_tokens_per_second"] for item in rounds]
        return {
            "status": "ok",
            "warmups": 1,
            "measured_rounds": 3,
            "persistent_engines": True,
            "fixed_environment": {
                "TEMP": 0,
                "DRAFT": 0,
                "KVSAVE": 0,
                "AUTOPIN": 0,
                "PROF": 1,
            },
            "runtime_knobs": normalized_knobs,
            "per_node_tokens_per_second": summary,
            "slowest_node_tokens_per_second": min(p50_values),
            "total_tokens_per_second": _percentile(total_values, 0.50),
            "output_sha256": next(iter(hashes)),
            "rounds": rounds,
            "logs": [entry["log"] for entry in entries],
        }
    except _OperationCancelled:
        raise
    except (RamdiskError, RuntimeError, OSError, subprocess.SubprocessError, ValueError) as exc:
        if cancel_event is not None and cancel_event.is_set():
            raise
        return {
            "status": "error",
            "error": str(exc),
            "per_node_tokens_per_second": [],
            "slowest_node_tokens_per_second": None,
            "total_tokens_per_second": None,
        }
    finally:
        cleanup_failures = []
        for entry in launched:
            try:
                entry["engine"].close()
            except Exception as exc:
                cleanup_failures.append(
                    "node %s engine: %s" % (entry.get("node"), exc)
                )
            try:
                entry["log_stream"].close()
            except Exception as exc:
                cleanup_failures.append(
                    "node %s log: %s" % (entry.get("node"), exc)
                )
        if cleanup_failures:
            raise _EngineCleanupError(
                "aggregate benchmark cleanup failed: %s"
                % "; ".join(cleanup_failures)
            )


def _system_score(manifest, variants, swap_before, swap_after, aggregate=None):
    memory = _meminfo()
    prefaults = [
        run["prefault_seconds"]
        for variant in variants
        if variant.get("status") == "ok"
        for run in variant.get("runs", [])
        if run.get("prefault_seconds") is not None
    ]
    rss_values = [
        run["rss_bytes"]
        for variant in variants
        if variant.get("status") == "ok"
        for run in variant.get("runs", [])
        if run.get("rss_bytes") is not None
    ]
    aggregate_rows = []
    aggregate_rss_totals = []
    aggregate_prefaults = []
    if aggregate and aggregate.get("status") == "ok":
        aggregate_rows = [
            row
            for round_result in aggregate.get("rounds", [])
            for row in round_result.get("rows", [])
        ]
        aggregate_prefaults = [
            row["prefault_seconds"]
            for row in aggregate_rows
            if row.get("prefault_seconds") is not None
        ]
        rss_values.extend(
            row["rss_bytes"]
            for row in aggregate_rows
            if row.get("rss_bytes") is not None
        )
        aggregate_rss_totals = [
            sum(
                row["rss_bytes"]
                for row in round_result.get("rows", [])
                if row.get("rss_bytes") is not None
            )
            for round_result in aggregate.get("rounds", [])
        ]
    created = manifest.get("created_at")
    ready = manifest.get("ready_at")
    stage_seconds = None
    try:
        if created and ready:
            stage_seconds = (
                datetime.datetime.fromisoformat(ready) - datetime.datetime.fromisoformat(created)
            ).total_seconds()
    except ValueError:
        pass
    shmem = memory.get("Shmem", 0)
    huge = memory.get("ShmemPmdMapped", 0)
    placement = []
    mount_shmem = 0
    for record in manifest.get("mounts", []):
        counts = record.get("numa_allocation", {})
        total = sum(int(value) for value in counts.values())
        node = record.get("node")
        local = int(counts.get(str(node), 0)) if node is not None else None
        allocated_bytes = None
        try:
            filesystem = os.statvfs(record["path"])
            allocated_bytes = (
                filesystem.f_blocks - filesystem.f_bfree
            ) * filesystem.f_frsize
            mount_shmem += allocated_bytes
        except OSError:
            pass
        placement.append(
            {
                "path": record["path"],
                "target_node": node,
                "sampled_pages": total,
                "local_pages": local,
                "remote_pages": total - local if local is not None else None,
                "by_node": counts,
                "allocated_bytes": allocated_bytes,
            }
        )
    return {
        "stage_seconds": stage_seconds,
        "prefault_seconds": (
            max(aggregate_prefaults)
            if aggregate_prefaults
            else _percentile(prefaults, 0.50)
        ),
        "rss_bytes": (
            max(aggregate_rss_totals)
            if aggregate_rss_totals
            else max(rss_values) if rss_values else None
        ),
        "per_process_peak_rss_bytes": max(rss_values) if rss_values else None,
        "aggregate_rss_bytes": max(aggregate_rss_totals) if aggregate_rss_totals else None,
        "shmem_bytes": mount_shmem,
        "host_shmem_bytes": shmem,
        "swap_before_bytes": swap_before,
        "swap_after_bytes": swap_after,
        "swap_delta_bytes": max(0, swap_after - swap_before),
        "huge_page_coverage": float(huge) / shmem if shmem else 0.0,
        "huge_page_coverage_scope": "host-global ShmemPmdMapped/Shmem; per-mount THP accounting is not exposed by tmpfs",
        "numa_allocation": [record.get("numa_allocation", {}) for record in manifest.get("mounts", [])],
        "numa_page_placement": placement,
        "numa_traffic_note": "dependency-free v1 reports sampled page placement; PMU traffic counters require external perf privileges",
    }


@_exclusive_lifecycle
def benchmark(args, cli_path=None, engine_path=None, cancel_event=None):
    manifest = _load_manifest(required=True)
    _raise_if_cancelled(cancel_event)
    if manifest.get("state") not in ("ready", "running", "stopped"):
        raise RamdiskError("benchmark requires a ready RAM-disk manifest")
    _assert_effective_masks_unchanged(manifest["plan"])
    _assert_ready_mounts(manifest)
    cli_path = cli_path or os.path.join(os.path.dirname(__file__), "coli")
    engine_path = _resolve_engine_path(cli_path, engine_path)
    model = manifest["plan"]["model"]["path"]
    mount = manifest["mounts"][0]["path"]
    running = manifest.get("state") == "running"
    if running:
        raise RamdiskError(
            "stop managed engines before benchmarking so every score uses the fixed environment"
        )
    specs = [
        ("ssd_baseline", model, False, {"PIPE": 0, "DIRECT": 0, "URING": 0}),
        ("tmpfs_pread_slabs", mount, False, {"PIPE": 0, "DIRECT": 0, "URING": 0}),
    ]
    if manifest["plan"]["mode"] == "full":
        benchmark_node = (
            manifest["mounts"][0].get("node")
            if manifest["plan"]["topology"] == "per-node"
            else None
        )
        cores = _node_core_count(manifest["plan"], benchmark_node)
        specs.extend(
            [
                ("full_direct_half_threads", mount, True, {"PIPE": 0, "OMP_NUM_THREADS": max(1, cores // 2), "OMP_PROC_BIND": "close"}),
                ("full_direct_pipe0", mount, True, {"PIPE": 0, "OMP_NUM_THREADS": cores, "OMP_PROC_BIND": "close"}),
                ("full_direct_pipe1", mount, True, {"PIPE": 1, "OMP_NUM_THREADS": cores, "OMP_PROC_BIND": "spread"}),
            ]
        )
        skipped = {"name": "partial_direct_ssd_fallback", "status": "not-applicable", "reason": "manifest is full mode"}
    else:
        specs.extend(
            [
                ("partial_direct_buffered", mount, True, {"PIPE": 1, "DIRECT": 0, "PIPE_WORKERS": 4, "URING": 0}),
                ("partial_direct_ssd", mount, True, {"PIPE": 1, "DIRECT": 1, "PIPE_WORKERS": 8, "URING": 0}),
                ("partial_direct_uring", mount, True, {"PIPE": 1, "DIRECT": 1, "PIPE_WORKERS": 8, "URING": 1}),
            ]
        )
        skipped = {"name": "full_direct", "status": "not-applicable", "reason": "manifest is partial mode"}
    variants = []
    swap_before = discover_hardware()["swap"]["used_bytes"]
    for name, weights, rammap, knobs in specs:
        _raise_if_cancelled(cancel_event)
        try:
            if cancel_event is None:
                score = _score_variant(engine_path, manifest, name, weights, rammap, knobs)
            else:
                score = _score_variant(
                    engine_path,
                    manifest,
                    name,
                    weights,
                    rammap,
                    knobs,
                    cancel_event=cancel_event,
                )
            variants.append(score)
        except (_OperationCancelled, _EngineCleanupError):
            raise
        except (RamdiskError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            if cancel_event is not None and cancel_event.is_set():
                raise
            variants.append({"name": name, "status": "error", "error": str(exc), "knobs": knobs})
    variants.append(skipped)
    baseline = next(
        (variant for variant in variants if variant.get("name") == "ssd_baseline" and variant.get("status") == "ok"),
        None,
    )
    token_mismatches = []
    if baseline:
        for variant in variants:
            if variant.get("status") != "ok":
                continue
            equivalent = variant.get("output_sha256") == baseline.get("output_sha256")
            variant["greedy_output_matches_ssd"] = equivalent
            if not equivalent:
                token_mismatches.append(variant["name"])
                variant["status"] = "error"
                variant["error"] = "greedy output differs from the SSD baseline"
    successful = [
        variant
        for variant in variants
        if variant.get("status") == "ok"
        and variant.get("greedy_output_matches_ssd") is True
        and variant["name"].startswith("%s_direct" % manifest["plan"]["mode"])
    ]
    best = max(
        successful,
        key=lambda variant: variant["interactive"]["p50_tokens_per_second"] or 0,
    ) if successful else None
    previous_best = manifest.get("best_runtime", {}).get(manifest["plan"]["topology"], {})
    aggregate_knobs = dict(
        best.get("knobs") if best else previous_best.get("knobs", {"PIPE": 0})
    )
    # The interactive sweep targets the first replica.  Thread counts are
    # node-relative, so each aggregate engine must derive its own local count
    # instead of reusing node 0's absolute value on asymmetric machines.
    aggregate_knobs.pop("OMP_NUM_THREADS", None)
    if cancel_event is None:
        aggregate = _aggregate_score(
            manifest, engine_path=engine_path, knobs=aggregate_knobs
        )
    else:
        aggregate = _aggregate_score(
            manifest,
            engine_path=engine_path,
            knobs=aggregate_knobs,
            cancel_event=cancel_event,
        )
    if baseline and aggregate.get("status") == "ok":
        aggregate_matches = aggregate.get("output_sha256") == baseline.get("output_sha256")
        aggregate["greedy_output_matches_ssd"] = aggregate_matches
        if not aggregate_matches:
            token_mismatches.append("aggregate_per_node")
            aggregate["status"] = "error"
            aggregate["error"] = "aggregate greedy output differs from the SSD baseline"
    tmpfs_ok = any(
        variant.get("name") == "tmpfs_pread_slabs"
        and variant.get("status") == "ok"
        and variant.get("greedy_output_matches_ssd") is True
        for variant in variants
    )
    aggregate_ok = manifest["plan"]["topology"] != "per-node" or (
        aggregate.get("status") == "ok"
        and aggregate.get("greedy_output_matches_ssd") is True
    )
    required_paths_ok = bool(baseline) and tmpfs_ok and bool(successful) and aggregate_ok
    full_zero_verified = None
    if manifest["plan"]["mode"] == "full":
        full_zero_verified = bool(successful) and all(
            len(variant.get("runs", [])) == 3
            and all(
                run.get("physical_ssd_valid") is True
                and run.get("physical_ssd_bytes") == 0
                for run in variant["runs"]
            )
            for variant in successful
        )
    swap_after = discover_hardware()["swap"]["used_bytes"]
    result = {
        "schema": BENCHMARK_SCHEMA,
        "version": MANIFEST_VERSION,
        "created_at": _utc_now(),
        "model_fingerprint": manifest["model_fingerprint"],
        "topology": manifest["plan"]["topology"],
        "mode": manifest["plan"]["mode"],
        "prompt": BENCHMARK_PROMPT,
        "source": _source_build_identity(),
        "command": list(sys.argv),
        "hardware": {
            "kernel_release": manifest["plan"]["hardware"].get("kernel_release"),
            "physical_cores": manifest["plan"]["hardware"].get("physical_cores"),
            "nodes": manifest["plan"]["hardware"].get("nodes"),
            "memory": manifest["plan"]["hardware"].get("memory"),
        },
        "storage": {
            "canonical_model_filesystem": _filesystem_for_path(model),
            "mounts": [
                {
                    "path": record["path"],
                    "node": record.get("node"),
                    "filesystem": record.get("identity", {}).get("filesystem"),
                    "options": record.get("identity", {}).get("all_options", []),
                }
                for record in manifest.get("mounts", [])
            ],
        },
        "warmups": 1,
        "measured_runs": 3,
        "tokens_per_run": 32,
        "variants": variants,
        "aggregate": aggregate,
        "system": _system_score(
            manifest, variants, swap_before, swap_after, aggregate=aggregate
        ),
        "acceptance": {
            "all_required_paths_succeeded": required_paths_ok,
            "greedy_outputs_identical": required_paths_ok and not token_mismatches,
            "output_mismatches": token_mismatches,
            "full_zero_physical_ssd_reads_verified": full_zero_verified,
            "no_swap_growth": swap_after <= swap_before,
            "staging_within_budget": manifest["plan"]["staging"]["staged_bytes"]
            <= manifest["plan"]["capacity_bytes"],
        },
        "best_runtime_knobs": best.get("knobs") if best else previous_best.get("knobs"),
        "best_variant": best.get("name") if best else previous_best.get("variant"),
    }
    _raise_if_cancelled(cancel_event)
    history = _read_json(_benchmarks_path()) or {"version": 1, "results": []}
    if (
        not isinstance(history, dict)
        or history.get("version") != 1
        or not isinstance(history.get("results"), list)
    ):
        raise RamdiskError("benchmark history is malformed or unsupported")
    history.setdefault("results", []).append(result)
    _atomic_json(_benchmarks_path(), history)
    manifest.setdefault("benchmark_results", []).append(result)
    if best:
        manifest.setdefault("best_runtime", {})[manifest["plan"]["topology"]] = {
            "variant": result["best_variant"],
            "knobs": result["best_runtime_knobs"],
        }
    _save_manifest(manifest)
    return result


def _human_status(report):
    print("RAM-disk state: %s" % report["state"])
    if not report["present"]:
        return
    for mount in report["mounts"]:
        print("  %s: %s" % (mount["path"], "verified tmpfs" if mount["verified"] else "missing/unverified"))
    for process in report["processes"]:
        print("  port %s PID %s: %s" % (process["port"], process["pid"], process["reason"]))


def _human_benchmark(result):
    print("RAM-disk benchmark (%s / %s)" % (result["mode"], result["topology"]))
    for variant in result["variants"]:
        if variant.get("status") != "ok":
            print("  %-30s %s" % (variant["name"], variant.get("status")))
            continue
        interactive = variant["interactive"]
        print("  %-30s TTFT %s ms  tok/s p50 %s p95 %s  RAM %.1f%%" % (
            variant["name"],
            "%.1f" % interactive["ttft_ms"] if interactive["ttft_ms"] is not None else "n/a",
            "%.3f" % interactive["p50_tokens_per_second"] if interactive["p50_tokens_per_second"] is not None else "n/a",
            "%.3f" % interactive["p95_tokens_per_second"] if interactive["p95_tokens_per_second"] is not None else "n/a",
            interactive["ram_map_coverage"] * 100,
        ))
        print("    forward p50/p99 %s/%s ms  SSD %s bytes/token" % (
            "%.1f" % interactive["forward_p50_ms"] if interactive["forward_p50_ms"] is not None else "n/a",
            "%.1f" % interactive["forward_p99_ms"] if interactive["forward_p99_ms"] is not None else "n/a",
            "%.0f" % interactive["ssd_bytes_per_token"] if interactive["ssd_bytes_per_token"] is not None else "n/a",
        ))
    aggregate = result["aggregate"]
    print("  aggregate: %s  slowest %s tok/s  total %s tok/s" % (
        aggregate.get("status"),
        "%.3f" % aggregate["slowest_node_tokens_per_second"] if aggregate.get("slowest_node_tokens_per_second") is not None else "n/a",
        "%.3f" % aggregate["total_tokens_per_second"] if aggregate.get("total_tokens_per_second") is not None else "n/a",
    ))
    system = result["system"]
    print("  system: stage %s s  prefault %s s  RSS %s GiB  mount shmem %.2f GiB" % (
        "%.1f" % system["stage_seconds"] if system.get("stage_seconds") is not None else "n/a",
        "%.2f" % system["prefault_seconds"] if system.get("prefault_seconds") is not None else "n/a",
        "%.2f" % (system["rss_bytes"] / GIB) if system.get("rss_bytes") is not None else "n/a",
        system["shmem_bytes"] / float(GIB),
    ))
    print("    swap +%.3f GiB  host huge-page coverage %.1f%%  NUMA %s" % (
        system["swap_delta_bytes"] / float(GIB),
        system["huge_page_coverage"] * 100,
        system["numa_page_placement"],
    ))
    print("  acceptance: paths=%s outputs=%s no-swap-growth=%s within-budget=%s" % (
        result["acceptance"].get("all_required_paths_succeeded"),
        result["acceptance"]["greedy_outputs_identical"],
        result["acceptance"]["no_swap_growth"],
        result["acceptance"]["staging_within_budget"],
    ))
    if result["acceptance"].get("full_zero_physical_ssd_reads_verified") is not None:
        print(
            "    full direct physical SSD reads measured zero: %s"
            % result["acceptance"]["full_zero_physical_ssd_reads_verified"]
        )
    print("  best knobs for this topology: %s" % result["best_runtime_knobs"])


def _managed_process_metrics(record):
    rss_bytes = None
    rss_processes = 0
    matches, _, _ = _process_matches(record)
    if matches:
        members, unreadable = _process_group_members(
            int(record.get("pgid", record["pid"]))
        )
        verified = [
            member
            for member in members
            if member.get("uid") == record.get("uid")
            and member.get("nonce") == record.get("nonce")
        ]
        if not unreadable and len(verified) == len(members) and verified:
            rss_bytes = 0
            for member in verified:
                for line in _read_text(
                    "/proc/%s/status" % member["pid"]
                ).splitlines():
                    if line.startswith("VmRSS:"):
                        try:
                            rss_bytes += int(line.split()[1]) * 1024
                            rss_processes += 1
                        except (ValueError, IndexError):
                            pass
                        break
    tail = ""
    log_path = record.get("log")
    if log_path:
        try:
            with open(log_path, "rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 65536))
                tail = stream.read().decode("utf-8", "replace")
        except OSError:
            pass
    ram_experts = ram_bytes = ssd_bytes = None
    matches = re.findall(r"RAM map:\s*(\d+) experts / ([0-9.]+) GB", tail)
    if matches:
        ram_experts, ram_bytes = int(matches[-1][0]), float(matches[-1][1]) * 1e9
    else:
        matches = re.findall(
            r"\[RAMMAP\]\s*(\d+) direct tmpfs experts,\s*([0-9.]+) GB mapped",
            tail,
        )
        if matches:
            ram_experts, ram_bytes = int(matches[-1][0]), float(matches[-1][1]) * 1e9
    matches = re.findall(r"physical SSD reads:\s*([0-9.]+) GB", tail, re.I)
    if matches:
        ssd_bytes = float(matches[-1]) * 1e9
    return {
        "rss_bytes": rss_bytes,
        "rss_processes": rss_processes,
        "rammap_experts": ram_experts,
        "rammap_bytes": ram_bytes,
        "latest_ssd_bytes": ssd_bytes,
    }


def _managed_ports_for_plan(plan, base_port=8000):
    return [
        int(base_port) + (0 if mount.get("node") is None else int(mount["node"]))
        for mount in plan["mounts"]
    ]


def _persisted_base_port(manifest):
    """Recover the last base port, including manifests predating that field."""
    explicit = manifest.get("base_port")
    if (
        isinstance(explicit, int)
        and not isinstance(explicit, bool)
        and 1 <= explicit <= 65535
    ):
        return explicit

    candidates = []
    for process in manifest.get("processes", []):
        port = process.get("port")
        node = process.get("node")
        if isinstance(port, int) and not isinstance(port, bool):
            candidates.append(port - (0 if node is None else int(node)))
    if not candidates:
        for mount, port in zip(
            manifest.get("mounts", []), manifest.get("ports", [])
        ):
            if isinstance(port, int) and not isinstance(port, bool):
                node = mount.get("node")
                candidates.append(port - (0 if node is None else int(node)))
    if (
        candidates
        and len(set(candidates)) == 1
        and 1 <= candidates[0] <= 65535
    ):
        return candidates[0]
    return 8000


def _placement_summary(plan, base_port=8000):
    """Describe placement in user terms instead of implementation terms.

    ``interleaved`` and ``per-node`` are precise mount-policy names, but they do
    not tell an operator how many complete copies and services will exist.  The
    TUI and confirmation prompt share this description so the expensive choice
    cannot be hidden behind different wording at action time.
    """
    contract = PlacementContract.from_plan(plan, base_port)
    copies = contract.copy_count
    engines = contract.engine_count
    nodes = list(contract.numa_nodes)
    ports = list(contract.ports)
    each_gib = contract.staged_bytes_per_copy / float(GIB)
    total_gib = contract.total_staged_bytes / float(GIB)
    full = contract.mode == "full"
    copy_name = "complete model" if full else "selected shard set"
    copy_word = "copy" if copies == 1 else "copies"
    engine_word = "engine" if engines == 1 else "independent engines"
    port_word = "port" if len(ports) == 1 else "ports"
    endpoints = "%s %s" % (port_word, ", ".join(str(port) for port in ports))
    node_labels = ["N%s" % node for node in nodes]
    selected_cpus = plan.get("placement", {}).get("cpu_list")
    cpu_clause = (
        " Selected engine CPUs: %s." % selected_cpus if selected_cpus else ""
    )

    if contract.is_shared:
        title = "Single shared model (recommended)"
        cost = "%d %s %s (%.2f GiB) · %d %s" % (
            copies,
            copy_name,
            copy_word,
            total_gib,
            engines,
            engine_word,
        )
        explanation = (
            "Stored once; RAM pages are spread across %d NUMA %s selected for this plan and one engine serves one endpoint.%s"
            % (
                len(nodes),
                "node" if len(nodes) == 1 else "nodes",
                cpu_clause,
            )
        )
        rail = "MODEL x1  ->  RAM [%s]  ->  ENGINE x1" % (
            " | ".join(node_labels) if node_labels else "host"
        )
    else:
        title = (
            "Independent full-model replicas (advanced)"
            if full
            else "Independent staged-set replicas (advanced)"
        )
        cost = "%d %s %s (%d x %.2f GiB = %.2f GiB) · %d %s" % (
            copies,
            copy_name,
            copy_word,
            copies,
            each_gib,
            total_gib,
            engines,
            engine_word,
        )
        explanation = (
            "This is replication, not model sharding: every NUMA node stores the entire staged set "
            "and serves a separate endpoint.%s" % cpu_clause
        )
        rail = "MODEL x%d  ->  %s  ->  ENGINES x%d" % (
            copies,
            "  ".join("[%s]" % label for label in node_labels) or "[host]",
            engines,
        )
    return {
        "title": title,
        "cost": cost,
        "explanation": explanation,
        "rail": rail,
        "endpoints": endpoints,
        "copy_count": copies,
        "engine_count": engines,
        "ports": ports,
    }


def _plan_confirmation_token(plan):
    """Stable identity for exactly the plan a user reviewed in the TUI."""
    reviewed = {
        "schema": plan.get("schema"),
        "version": plan.get("version"),
        "model_fingerprint": plan.get("model", {}).get("fingerprint"),
        "mode": plan.get("mode"),
        "topology": plan.get("topology"),
        "placement": plan.get("placement"),
        "mount_root": plan.get("mount_root"),
        "capacity_bytes": plan.get("capacity_bytes"),
        "selected_shards": plan.get("staging", {}).get("selected_shards"),
        "linked_shards": plan.get("staging", {}).get("linked_shards"),
        "total_staged_bytes": plan.get("staging", {}).get("total_staged_bytes"),
        "total_required_bytes": plan.get("reserve", {}).get("total_required_bytes"),
        "mounts": plan.get("mounts"),
        "mount_options": plan.get("mount_options"),
        "prefault": plan.get("prefault"),
        "parallel": plan.get("parallel"),
        "managed_runtime": plan.get("managed_runtime"),
    }
    payload = json.dumps(reviewed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_confirmation_token(manifest):
    """Bind a destructive confirmation to one prepared deployment.

    New manifests carry a random deployment id.  The remaining fields keep
    confirmations safe for manifests created by older releases as well.
    Runtime counters are deliberately excluded so ordinary status collection
    cannot invalidate a confirmation.
    """
    mounts = []
    for record in manifest.get("mounts", []):
        identity = record.get("identity", {})
        mounts.append(
            {
                "path": record.get("path"),
                "node": record.get("node"),
                "mount_id": identity.get("mount_id"),
                "device": identity.get("device"),
            }
        )
    processes = []
    for record in manifest.get("processes", []):
        processes.append(
            {
                "pid": record.get("pid"),
                "pgid": record.get("pgid"),
                "uid": record.get("uid"),
                "starttime": record.get("starttime"),
                "nonce": record.get("nonce"),
                "port": record.get("port"),
                "node": record.get("node"),
            }
        )
    reviewed = {
        "version": manifest.get("version"),
        "deployment_id": manifest.get("deployment_id"),
        "created_at": manifest.get("created_at"),
        "state": manifest.get("state"),
        "base_port": _persisted_base_port(manifest),
        "model_fingerprint": manifest.get("model_fingerprint"),
        "plan_token": _plan_confirmation_token(manifest.get("plan", {})),
        "mounts": mounts,
        "processes": processes,
    }
    payload = json.dumps(reviewed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prepare_confirmation(plan, base_port=8000):
    contract = PlacementContract.from_plan(plan, base_port)
    placement = _placement_summary(plan, base_port)
    copies = contract.copy_count
    each_gib = contract.staged_bytes_per_copy / float(GIB)
    total_gib = contract.total_staged_bytes / float(GIB)
    copy_name = "complete model" if contract.mode == "full" else "selected shard set"
    if contract.is_shared:
        nodes = len(contract.numa_nodes)
        return (
            "CONFIRM SHARED PLAN: stage %d %s copy (%.2f GiB) at %s, spread across %d NUMA %s. "
            "Memory nodes %s; engine CPUs %s. Start will launch 1 engine on %s. "
            "tmpfs size is a cap, THP is requested rather than guaranteed, and copy workers do not create replicas. "
            "Press p again within 10s."
            % (
                copies,
                copy_name,
                total_gib,
                plan["mount_root"],
                nodes,
                "node" if nodes == 1 else "nodes",
                plan.get("placement", {}).get("memory_node_list", "all"),
                plan.get("placement", {}).get("cpu_list", "all"),
                placement["endpoints"],
            )
        )
    return (
        "CONFIRM REPLICA PLAN: stage %d %s copies (%d x %.2f GiB = %.2f GiB) at %s. "
        "Memory nodes %s; selected CPUs %s. Start will launch %d independent engines on %s. "
        "This is replication, not sharding, and does not accelerate one request. "
        "Press p again within 10s."
        % (
            copies,
            copy_name,
            copies,
            each_gib,
            total_gib,
            plan["mount_root"],
            plan.get("placement", {}).get("memory_node_list", "all"),
            plan.get("placement", {}).get("cpu_list", "all"),
            placement["engine_count"],
            placement["endpoints"],
        )
    )


def _prepare_confirmation_rows(plan, base_port=8000):
    """Put the irreversible topology facts in the first three TUI rows."""
    contract = PlacementContract.from_plan(plan, base_port)
    placement = _placement_summary(plan, base_port)
    copies = contract.copy_count
    engines = contract.engine_count
    if contract.mode == "full":
        copies_text = "%d complete model %s" % (
            copies,
            "copy" if copies == 1 else "copies",
        )
    else:
        copies_text = "%d selected shard-set %s" % (
            copies,
            "copy" if copies == 1 else "copies",
        )
    if contract.is_replication:
        placement_text = "DANGER · replication, not sharding"
        engines_text = "%d independent engines" % engines
    else:
        nodes = len(contract.numa_nodes)
        placement_text = "SHARED · pages span %d NUMA %s" % (
            nodes,
            "node" if nodes == 1 else "nodes",
        )
        engines_text = "%d engine on %s" % (engines, placement["endpoints"])
    return [
        ("warn", "REVIEW · %s" % copies_text),
        ("warn", "START · %s" % engines_text),
        ("bad" if contract.is_replication else "accent", placement_text),
    ]


@contextlib.contextmanager
def _cli_termination_guard(cancelable):
    """Translate service/SSH termination into lifecycle-safe checkpoints.

    Prepare, Start, and Benchmark receive a cooperative cancellation event.
    Stop and Destroy deliberately finish their verified cleanup transaction
    before the CLI reports the deferred signal exit code.
    """
    state = {
        "cancel_event": threading.Event(),
        "signum": None,
    }
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGHUP", "SIGTERM"):
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                previous[signum] = signal.getsignal(signum)
            except (OSError, ValueError):
                continue

        def request_termination(signum, _frame):
            if state["signum"] is None:
                state["signum"] = int(signum)
            if cancelable:
                state["cancel_event"].set()

        for signum in tuple(previous):
            try:
                signal.signal(signum, request_termination)
            except (OSError, ValueError):
                previous.pop(signum, None)
    try:
        yield state
    finally:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                pass


def _cli_exit_after_signal(termination, normal_code):
    if termination is not None and termination.get("signum") is not None:
        return 128 + int(termination["signum"])
    return normal_code


def dispatch(args, cli_path=None, engine_path=None, system=None):
    action = getattr(args, "ramdisk_action", None)
    termination = None
    try:
        if action == "plan":
            value = build_plan(args)
            if getattr(args, "json", False):
                _json_print(value)
            else:
                _human_plan(value)
            return 2 if value["blockers"] else 0
        if action == "prepare":
            with _cli_termination_guard(True) as termination:
                value = prepare(
                    args,
                    cancel_event=termination["cancel_event"],
                )
            print("RAM-disk ready: %s" % ", ".join(record["path"] for record in value["mounts"]))
            return _cli_exit_after_signal(termination, 0)
        if action == "status":
            value = status()
            if getattr(args, "json", False):
                _json_print(value)
            else:
                _human_status(value)
            return 0
        if action == "benchmark":
            with _cli_termination_guard(True) as termination:
                value = benchmark(
                    args,
                    cli_path=cli_path,
                    engine_path=engine_path,
                    cancel_event=termination["cancel_event"],
                )
            if getattr(args, "json", False):
                _json_print(value)
            else:
                _human_benchmark(value)
            return _cli_exit_after_signal(termination, 0)
        if action == "start":
            with _cli_termination_guard(True) as termination:
                value = start(
                    args,
                    cli_path=cli_path,
                    engine_path=engine_path,
                    cancel_event=termination["cancel_event"],
                )
            print("managed engine ports: %s" % ", ".join(str(port) for port in value["ports"]))
            return _cli_exit_after_signal(termination, 0)
        if action == "stop":
            with _cli_termination_guard(False) as termination:
                value = stop(args)
            if value.get("state") == "error":
                print(
                    "managed engine cleanup completed, but the RAM workspace is incomplete; "
                    "review `coli ramdisk status`, then run destroy",
                    file=sys.stderr,
                )
                return _cli_exit_after_signal(termination, 2)
            print("managed engines stopped; usage deltas merged")
            return _cli_exit_after_signal(termination, 0)
        if action == "destroy":
            with _cli_termination_guard(False) as termination:
                value = destroy(args)
            print("RAM-disk destroyed; durable state and benchmark history preserved")
            return _cli_exit_after_signal(termination, 0)
        raise RamdiskError("choose a ramdisk action or run the interactive TUI")
    except (RamdiskError, OSError, subprocess.SubprocessError) as exc:
        if getattr(args, "json", False):
            _json_print({"schema": "colibri.ramdisk.error.v1", "version": 1, "error": str(exc)})
        else:
            print("coli ramdisk: %s" % exc, file=sys.stderr)
        return _cli_exit_after_signal(termination, 2)


_TUI_SCREENS = ("Plan", "Hardware", "Activity", "Benchmarks", "Settings")
_tui_worker_guard = threading.Lock()
_tui_worker = None


def _tui_review_scroll(pending_action, requested_scroll):
    """Keep prepare-review facts visible until the review is accepted or cancelled."""
    if pending_action == "prepare":
        return 0
    return max(0, requested_scroll)


def _tui_plan_rows(plan, report, active=False, base_port=8000, confirmation=None):
    placement = _placement_summary(plan, base_port)
    rows = []
    if confirmation:
        # Keep these three facts pinned at the top.  They fit in the minimum
        # supported viewport, so the second confirmation key can never be
        # accepted while copy/engine topology is scrolled offscreen.
        rows.extend(_prepare_confirmation_rows(plan, base_port))
        rows.append(("normal", ""))
    rows.extend([
        ("dim", "ACTIVE DEPLOYMENT" if active else "DRAFT PLAN · nothing has been changed yet"),
        ("warn" if plan["topology"] == "per-node" else "accent", placement["title"]),
        ("accent", placement["rail"]),
        ("normal", placement["cost"]),
        ("normal", "After Start: %s" % placement["endpoints"]),
        ("warn" if plan["topology"] == "per-node" else "dim", placement["explanation"]),
        ("normal", ""),
        (
            "heading",
            "STAGING · %s" % ("full model" if plan["mode"] == "full" else "profile-selected shard set"),
        ),
        (
            "normal",
            "%d of %d shards in RAM · %d direct-mapped experts · prefault %s"
            % (
                len(plan["staging"]["selected_shards"]),
                plan["model"]["shard_count"],
                plan["staging"]["direct_mapped_expert_count"],
                "on" if plan["prefault"] else "off",
            ),
        ),
        (
            "normal",
            "%s host memory %.2f GiB · %s available %.2f GiB"
            % (
                "Planned" if active else "Projected",
                plan["reserve"]["total_required_bytes"] / float(GIB),
                "at preparation" if active else "currently",
                plan["reserve"]["available_bytes"] / float(GIB),
            ),
        ),
    ])
    if plan["mode"] == "partial":
        rows.append(
            (
                "normal",
                "Profile coverage %.1f%% · staging efficiency %.1f%%"
                % (
                    plan["profile"]["coverage"] * 100,
                    plan["profile"]["staging_efficiency"] * 100,
                ),
            )
        )
    if confirmation:
        rows.extend(
            [
                ("normal", ""),
                ("warn", "FULL PREPARATION DETAIL"),
                ("warn", confirmation),
            ]
        )
    if active:
        health = DeploymentHealth.from_report(plan, report)
        if health.level is HealthLevel.VERIFIED:
            health_style = "good"
            health_title = "DEPLOYMENT VERIFIED"
            health_detail = "Persisted settings are locked. Activity shows current mount and engine health."
        elif health.level is HealthLevel.FAST_CHECK:
            health_style = "warn"
            health_title = "FAST CHECK PASSED · DEEP VERIFICATION PENDING"
            health_detail = "Press R for source and NUMA verification; Start also revalidates before launch."
        else:
            health_style = "bad"
            health_title = "DEPLOYMENT NEEDS ATTENTION"
            health_detail = "Open Activity and press R before Start; Destroy revalidates every exact mount."
        rows.extend(
            [
                ("normal", ""),
                (health_style, health_title),
                ("dim" if health.fast_check_ok else "bad", health_detail),
            ]
        )
    elif plan["blockers"]:
        rows.extend([("normal", ""), ("bad", "NOT READY")])
        rows.extend(("bad", blocker) for blocker in plan["blockers"])
    else:
        rows.extend([("normal", ""), ("good", "READY")])
        if confirmation:
            rows.append(("warn", "Press p again before the confirmation expires, or any change cancels it."))
        else:
            rows.append(("dim", "Review the copy count and memory total above, then press p to prepare."))
    rows.extend(("warn", warning) for warning in plan["warnings"])
    if report.get("present"):
        rows.append(("dim", "Lifecycle state: %s" % report.get("state", "unknown")))
    return rows


def _tui_hardware_rows(hardware):
    nodes = hardware.get("nodes", [])
    rows = [
        ("heading", "HOST MEMORY TOPOLOGY"),
        (
            "normal",
            "%.1f GiB available / %.1f GiB total · %d physical cores · %d NUMA %s"
            % (
                hardware["memory"]["available_bytes"] / float(GIB),
                hardware["memory"]["total_bytes"] / float(GIB),
                hardware["physical_cores"],
                len(nodes),
                "node" if len(nodes) == 1 else "nodes",
            ),
        ),
        (
            "dim",
            "NUMA nodes determine RAM placement. CPU cores do not create model copies.",
        ),
        (
            "normal",
            "Kernel %s · tmpfs %s · noswap %s · THP %s"
            % (
                hardware["kernel_release"],
                "available" if hardware["tmpfs"]["supported"] else "missing",
                "available" if hardware["tmpfs"]["noswap_supported"] else "missing",
                hardware["thp"]["shmem_enabled"] or "unknown",
            ),
        ),
        (
            "warn" if hardware["swap"]["used_bytes"] else "dim",
            "Swap in use %.2f GiB" % (hardware["swap"]["used_bytes"] / float(GIB)),
        ),
        ("normal", ""),
    ]
    for node in nodes:
        rows.extend(
            [
                ("accent", "NUMA %s · CPUs %s · %d physical cores" % (node["id"], node["cpu_list"], node["physical_cores"])),
                (
                    "normal",
                    "  %.1f GiB available / %.1f GiB total · distance %s"
                    % (
                        node["memory_available_bytes"] / float(GIB),
                        node["memory_total_bytes"] / float(GIB),
                        node["distance"],
                    ),
                ),
            ]
        )
    return rows


def _tui_activity_rows(report, hardware, process_metrics=None):
    rows = [("heading", "LIFECYCLE · %s" % report.get("state", "unknown").upper())]
    if not report.get("present"):
        rows.extend(
            [
                ("dim", "No RAM workspace exists yet."),
                ("normal", "Review the Plan page, then prepare it with p."),
            ]
        )
        return rows
    rows.append(
        (
            "dim",
            "%s validation · manifest %s"
            % ("deep" if report.get("deep_validation") else "fast", report.get("manifest_path")),
        )
    )
    rows.extend([("normal", ""), ("heading", "RAM MOUNTS")])
    for mount in report.get("mounts", []):
        rows.append(
            (
                "good" if mount.get("verified") else "bad",
                "%s · %s · NUMA pages %s"
                % (
                    mount["path"],
                    "verified" if mount.get("verified") else "missing or unverified",
                    mount.get("numa_allocation") or "not sampled",
                ),
            )
        )
    rows.extend([("normal", ""), ("heading", "MANAGED ENGINES")])
    processes = report.get("processes", [])
    if not processes:
        rows.append(("dim", "No engine is running. Prepared weights stay resident until Destroy."))
    for process in processes:
        rows.append(
            (
                "good" if process.get("running") else "dim",
                "port %s · PID %s · node %s · %s"
                % (process.get("port"), process.get("pid"), process.get("node"), process.get("reason")),
            )
        )
        metrics = (process_metrics or {}).get(process.get("pid"), {})
        if metrics.get("rss_bytes") is not None:
            rows.append(
                (
                    "dim",
                    "  RSS %.2f GiB across %d processes · RAM map %s experts / %s GiB"
                    % (
                        metrics["rss_bytes"] / float(GIB),
                        metrics["rss_processes"],
                        metrics["rammap_experts"] if metrics["rammap_experts"] is not None else "n/a",
                        "%.2f" % (metrics["rammap_bytes"] / GIB)
                        if metrics["rammap_bytes"] is not None
                        else "n/a",
                    ),
                )
            )
    mem = _meminfo()
    rows.extend(
        [
            ("normal", ""),
            (
                "dim",
                "Host shared memory %.2f GiB · swap %.3f GiB"
                % (
                    mem.get("Shmem", 0) / float(GIB),
                    hardware["swap"]["used_bytes"] / float(GIB),
                ),
            ),
        ]
    )
    return rows


def _tui_benchmark_rows(history):
    rows = [("heading", "PERSISTENT PATH SCORECARD")]
    results = (history or {}).get("results", [])
    if not results:
        rows.extend(
            [
                ("dim", "No benchmark history yet."),
                ("normal", "Prepare the workspace, stop managed engines, then press b here."),
            ]
        )
        return rows
    latest = results[-1]
    rows.append(("accent", "Latest %s · best %s" % (latest.get("created_at"), latest.get("best_variant"))))
    for variant in latest.get("variants", []):
        if variant.get("status") != "ok":
            rows.append(("warn", "%s · %s" % (variant.get("name"), variant.get("status"))))
            continue
        score = variant.get("interactive", {})
        rows.append(
            (
                "normal",
                "%s · TTFT %s ms · %.2f tok/s p50 · RAM %.1f%% · SSD %s B/token"
                % (
                    variant.get("name"),
                    "%.1f" % score["ttft_ms"] if score.get("ttft_ms") is not None else "n/a",
                    score.get("p50_tokens_per_second") or 0.0,
                    (score.get("ram_map_coverage") or 0.0) * 100,
                    "%.0f" % score["ssd_bytes_per_token"]
                    if score.get("ssd_bytes_per_token") is not None
                    else "n/a",
                ),
            )
        )
    aggregate = latest.get("aggregate", {})
    rows.append(
        (
            "dim",
            "Aggregate %s · slowest %s tok/s · total %s tok/s"
            % (
                aggregate.get("status", "n/a"),
                aggregate.get("slowest_node_tokens_per_second", "n/a"),
                aggregate.get("total_tokens_per_second", "n/a"),
            ),
        )
    )
    return rows


def _tui_settings_rows(args, plan, report, base_port=8000):
    rows = [("heading", "WORKSPACE SETTINGS")]
    if report.get("present"):
        placement = _placement_summary(plan, base_port)
        can_change_port = report.get("state") in ("ready", "stopped")
        rows.extend(
            [
                ("warn", "LOCKED BY ACTIVE DEPLOYMENT"),
                ("normal", placement["title"]),
                ("normal", placement["cost"]),
                (
                    "normal" if can_change_port else "dim",
                    ("[P] Next Start base port %s" if can_change_port else "Current base port        %s")
                    % base_port,
                ),
                (
                    "dim",
                    "Start uses the persisted weights plan shown here. Stop before changing its next endpoint; Destroy before changing placement or staging.",
                ),
            ]
        )
        return rows
    placement = _placement_summary(plan, base_port)
    rows.extend(
        [
            ("warn" if plan["topology"] == "per-node" else "accent", "Placement · %s" % placement["title"]),
            ("normal", placement["cost"]),
            ("dim", placement["explanation"]),
        ]
    )
    if plan["topology"] == "per-node":
        rows.append(("good", "[i] Return to one shared copy"))
    else:
        rows.append(
            (
                "dim",
                "Replica mode is deliberately CLI-only: pass --topology per-node after reviewing its explicit help.",
            )
        )
    rows.extend(
        [
            ("normal", ""),
            ("normal", "[m] Staging mode        %s" % args.mode),
            ("normal", "[c] Per-copy budget    %s" % ("%.1f GiB" % args.capacity_gb if args.capacity_gb else "full model size")),
            ("normal", "[r] Usage profile      %s" % (args.profile or "<model>/.coli_usage")),
            ("normal", "[o] Mount root         %s" % args.mount_root),
            ("normal", "[P] Base port          %s" % args.base_port),
            ("normal", "[w] Copy workers       %s (copy concurrency only)" % args.parallel),
            ("normal", "[H] Huge pages         %s" % args.thp),
            ("normal", "[f] Prefault           %s" % ("on" if plan["prefault"] else "off")),
            ("normal", "[y] Swappable tmpfs    %s" % ("allowed" if args.allow_swappable else "refused")),
            ("normal", ""),
            ("dim", "Full mode always stages the full model; capacity changes only apply to partial mode."),
        ]
    )
    return rows


def _tui_help_rows():
    return [
        ("heading", "HOW THIS WORKS"),
        ("normal", "1. Plan shows exactly how many model copies, engines, ports, and GiB will be created."),
        ("normal", "2. Prepare mounts tmpfs and copies weights. It does not start an engine."),
        ("normal", "3. Start launches the persisted deployment; Stop keeps RAM weights; Destroy unmounts them."),
        ("normal", ""),
        ("accent", "Shared placement is the normal path: one model copy and one engine across all NUMA nodes."),
        ("warn", "Per-node means independent full replicas, not a model split. It is never enabled by a TUI toggle."),
        ("normal", ""),
        ("heading", "KEYS"),
        ("normal", "Left/Right or h/l · change page"),
        ("normal", "Up/Down or j/k · scroll"),
        ("normal", "p · review/prepare     s · start     x · stop     d · destroy"),
        ("normal", "b · benchmark          R · deep refresh          ? · close help"),
        ("normal", "c · cancel prepare/start/benchmark at a safe cleanup checkpoint"),
        ("normal", "Settings page · edit draft settings before preparation"),
        (
            "normal",
            "q or Esc · quit; long operations cancel safely first, cleanup finishes before exit",
        ),
    ]


def _tui_idle_action_hint(screen, plan, report):
    """Return only actions the shared lifecycle policy currently permits."""
    policy = ActionPolicy.from_state(plan, report)
    if screen == 0 and report and not report.get("present") and policy.prepare.enabled:
        return "[p] review / prepare"
    if screen == 3 and policy.benchmark.enabled:
        return "[b] benchmark"
    if policy.start.enabled:
        return "[s] start  [d] destroy"
    if policy.stop.enabled:
        return "[x] stop  [d] destroy"
    if policy.destroy.enabled:
        return "[d] destroy"
    return "[R] refresh"


def _tui_wrap_rows(rows, width):
    wrapped = []
    usable = max(20, int(width))
    for style, raw in rows:
        text = str(raw)
        if not text:
            wrapped.append((style, ""))
            continue
        leading = text[: len(text) - len(text.lstrip())]
        parts = textwrap.wrap(
            text.strip(),
            width=max(8, usable - len(leading)),
            initial_indent=leading,
            subsequent_indent=leading + ("  " if not leading else ""),
            break_long_words=True,
            break_on_hyphens=False,
        )
        wrapped.extend((style, part) for part in (parts or [leading]))
    return wrapped


def _tui(stdscr, initial, cli_path, engine_path):
    import curses
    global _tui_worker

    args = argparse.Namespace(**vars(initial))
    for name, value in (
        ("mode", "full"),
        ("topology", "interleaved"),
        ("mount_root", DEFAULT_MOUNT_ROOT),
        ("capacity_gb", None),
        ("profile", None),
        ("allow_swappable", False),
        ("thp", "auto"),
        ("prefault", None),
        ("parallel", 2),
        ("ctx", 0),
        ("base_port", 8000),
    ):
        if not hasattr(args, name):
            setattr(args, name, value)

    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.timeout(200)
    attrs = {
        "normal": curses.A_NORMAL,
        "heading": curses.A_BOLD,
        "accent": curses.A_BOLD,
        "good": curses.A_BOLD,
        "warn": curses.A_BOLD,
        "bad": curses.A_BOLD,
        "dim": curses.A_DIM,
    }
    try:
        curses.start_color()
        curses.use_default_colors()
        for pair, color in enumerate(
            (curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED),
            1,
        ):
            curses.init_pair(pair, color, -1)
        attrs.update(
            {
                "accent": curses.color_pair(1) | curses.A_BOLD,
                "good": curses.color_pair(2) | curses.A_BOLD,
                "warn": curses.color_pair(3) | curses.A_BOLD,
                "bad": curses.color_pair(4) | curses.A_BOLD,
            }
        )
    except curses.error:
        pass

    screen = 0
    scroll = 0
    help_open = False
    message = "Shared placement is selected: one model copy, one engine. Press ? for help."
    pending_action = None
    pending_review = None
    pending_deadline = 0.0
    hardware_cache = None
    model_cache = None
    plan_cache = None
    plan_key_cache = None
    report_cache = None
    active_manifest_cache = None
    active_deployment_identity = None
    history_cache = None
    metrics_cache = {}
    hardware_checked = report_checked = metrics_checked = 0.0
    deep_status_refresh = True
    operation = None
    quit_when_idle = False
    quit_exit_code = 0
    operation_lock = threading.Lock()

    def safe_add(row, column, value, limit, attribute=0):
        height, width = stdscr.getmaxyx()
        if row < 0 or row >= height or column < 0 or column >= width or limit <= 0:
            return
        try:
            stdscr.addnstr(row, column, str(value), min(limit, width - column), attribute)
        except curses.error:
            pass

    def invalidate(deep=False, model=False):
        nonlocal hardware_cache, model_cache, plan_cache, plan_key_cache
        nonlocal report_cache, active_manifest_cache, history_cache
        nonlocal deep_status_refresh
        plan_cache = plan_key_cache = None
        report_cache = active_manifest_cache = history_cache = None
        if model:
            model_cache = None
        if deep:
            hardware_cache = None
            deep_status_refresh = True

    def cancel_confirmation():
        nonlocal pending_action, pending_review, pending_deadline
        pending_action = pending_review = None
        pending_deadline = 0.0

    def begin_operation(action, label, target, cancelable=False):
        nonlocal operation
        global _tui_worker
        op = {
            "action": action,
            "label": label,
            "started": time.monotonic(),
            "detail": "Starting…",
            "done": False,
            "result": None,
            "error": None,
            "cancelable": bool(cancelable),
            "cancel_event": threading.Event(),
        }

        def runner():
            try:
                with _noninteractive_privilege(
                    keepalive=action in ("prepare", "destroy"),
                    cancel_event=op["cancel_event"] if cancelable else None,
                ):
                    result = target(op)
                with operation_lock:
                    op["result"] = result
            except BaseException as exc:
                with operation_lock:
                    op["error"] = exc
            finally:
                with operation_lock:
                    op["done"] = True

        operation = op
        thread = threading.Thread(target=runner, name="coli-ramdisk-%s" % action)
        op["thread"] = thread
        with _tui_worker_guard:
            _tui_worker = op
        try:
            thread.start()
        except BaseException:
            with _tui_worker_guard:
                _tui_worker = None
            operation = None
            raise

    def update_operation(op, detail):
        with operation_lock:
            op["detail"] = detail

    def prompt_value(label, current):
        nonlocal message
        height, width = stdscr.getmaxyx()
        if height < 4 or width < 20:
            message = "Terminal is too small for input."
            return None
        current_text = current if current not in (None, "") else "<auto>"
        prompt = "> "
        try:
            stdscr.timeout(-1)
            curses.echo()
            curses.curs_set(1)
            stdscr.move(height - 3, 0)
            stdscr.clrtoeol()
            safe_add(
                height - 3,
                0,
                "%s · current %s · Enter keeps it" % (label, current_text),
                width - 1,
                attrs["dim"],
            )
            stdscr.move(height - 2, 0)
            stdscr.clrtoeol()
            safe_add(height - 2, 0, prompt, width - 1, attrs["accent"])
            stdscr.refresh()
            start = len(prompt)
            value = stdscr.getstr(height - 2, start, max(1, width - start - 1))
            value = value.decode("utf-8").strip()
            return str(current) if not value else value
        except (curses.error, UnicodeDecodeError):
            message = "Input could not be read."
            return None
        finally:
            try:
                curses.noecho()
                curses.curs_set(0)
            except curses.error:
                pass
            stdscr.timeout(200)

    def authorize_privileged_mounts():
        """Obtain sudo credentials before a worker can need the controlling TTY."""
        nonlocal message
        if os.geteuid() == 0:
            return True
        try:
            sudo = _trusted_system_binary("sudo")
        except RamdiskError as exc:
            message = "Cannot authorize mount operations: %s" % exc
            return False
        try:
            try:
                curses.def_prog_mode()
                curses.endwin()
            except curses.error:
                pass
            try:
                result = subprocess.run([sudo, "-v"], check=False)
            except OSError as exc:
                message = "Sudo authorization failed: %s" % exc
                return False
        finally:
            try:
                curses.reset_prog_mode()
                stdscr.refresh()
            except curses.error:
                pass
        if result.returncode:
            message = "Sudo authorization was cancelled; no mount operation started."
            return False
        try:
            reusable = _validate_noninteractive_sudo(sudo)
        except OSError as exc:
            message = "Sudo continuation check failed: %s" % exc
            return False
        if reusable.returncode:
            message = (
                "Sudo policy cannot reuse authorization without prompting; "
                "no mount operation started."
            )
            return False
        return True

    while True:
        now = time.monotonic()
        if operation is not None:
            with operation_lock:
                finished = operation["done"]
                operation_error = operation["error"]
                operation_result = operation["result"]
                operation_action = operation["action"]
            if finished:
                operation["thread"].join(timeout=0.2)
                cancel_requested = operation["cancel_event"].is_set()
                cancelled_cleanly = isinstance(operation_error, _OperationCancelled)
                quit_after_cleanup = quit_when_idle and (
                    operation_error is None or cancelled_cleanly
                )
                if operation_error is not None:
                    if cancelled_cleanly:
                        message = "%s cancelled safely: %s" % (
                            operation_action.capitalize(),
                            operation_error,
                        )
                    elif cancel_requested:
                        message = "%s cleanup failed after cancellation: %s · review Activity before quitting" % (
                            operation_action.capitalize(),
                            operation_error,
                        )
                        quit_when_idle = False
                    else:
                        message = "%s failed: %s" % (
                            operation_action.capitalize(),
                            operation_error,
                        )
                elif operation_action == "prepare":
                    message = "RAM workspace is ready. Open Activity or press s to start."
                elif operation_action == "start":
                    message = "Engine ready on %s." % _placement_summary(
                        operation_result["plan"], args.base_port
                    )["endpoints"]
                elif operation_action == "stop":
                    if operation_result.get("state") == "error":
                        message = (
                            "Engine cleanup finished, but the workspace is incomplete. "
                            "Review Activity, then Destroy."
                        )
                    else:
                        message = "Managed engines stopped; RAM weights remain prepared."
                elif operation_action == "destroy":
                    message = "RAM workspace removed; durable KV and benchmark state preserved."
                elif operation_action == "benchmark":
                    message = "Benchmark complete; best path: %s." % operation_result.get("best_variant")
                operation = None
                with _tui_worker_guard:
                    _tui_worker = None
                cancel_confirmation()
                invalidate(deep=False, model=False)
                if quit_after_cleanup:
                    return quit_exit_code

        plan = report = hardware = None
        rows = []
        try:
            if hardware_cache is None or now - hardware_checked >= 30.0:
                hardware_cache = discover_hardware()
                hardware_checked = now
                plan_key_cache = None
            hardware = hardware_cache
            if report_cache is None or now - report_checked >= 2.0:
                report_cache = status(deep=deep_status_refresh)
                deep_status_refresh = False
                report_checked = now
                active_manifest_cache = None
            report = report_cache
            active = bool(report.get("present"))
            if active:
                if active_manifest_cache is None:
                    active_manifest_cache = _load_manifest(required=True)
                plan = active_manifest_cache["plan"]
                processes = report.get("processes", [])
                deployment_identity = (
                    active_manifest_cache.get("deployment_id"),
                    active_manifest_cache.get("created_at"),
                )
                if deployment_identity != active_deployment_identity:
                    args.base_port = _persisted_base_port(active_manifest_cache)
                    active_deployment_identity = deployment_identity
                if report.get("state") in ("running", "starting") and report.get("ports") and processes:
                    first = processes[0]
                    args.base_port = int(first["port"]) - int(first.get("node") or 0)
            else:
                active_deployment_identity = None
                if model_cache is None:
                    model_cache = scan_model(args.model)
                plan_key = (
                    args.mode,
                    args.topology,
                    args.mount_root,
                    args.capacity_gb,
                    args.profile,
                    args.allow_swappable,
                    args.thp,
                    args.prefault,
                    args.parallel,
                    args.ctx,
                    hardware_checked,
                )
                if plan_cache is None or plan_key != plan_key_cache:
                    plan_cache = build_plan(args, hardware=hardware, model=model_cache)
                    plan_key_cache = plan_key
                plan = plan_cache

            if pending_action == "prepare":
                current_token = _plan_confirmation_token(plan)
                current_review = ReviewIdentity.for_prepare(
                    current_token, plan, args.base_port
                )
                if (
                    active
                    or now > pending_deadline
                    or current_review != pending_review
                ):
                    cancel_confirmation()
            confirmation = (
                _prepare_confirmation(plan, args.base_port)
                if pending_action == "prepare"
                else None
            )
            if help_open:
                rows = _tui_help_rows()
            elif screen == 0:
                rows = _tui_plan_rows(plan, report, active, args.base_port, confirmation)
            elif screen == 1:
                rows = _tui_hardware_rows(hardware)
            elif screen == 2:
                if now - metrics_checked >= 2.0:
                    metrics_cache = {
                        process.get("pid"): _managed_process_metrics(process)
                        for process in report.get("processes", [])
                        if process.get("pid") is not None
                    }
                    metrics_checked = now
                rows = _tui_activity_rows(report, hardware, metrics_cache)
            elif screen == 3:
                if history_cache is None:
                    history_cache = _read_json(_benchmarks_path()) or {"results": []}
                rows = _tui_benchmark_rows(history_cache)
            else:
                rows = _tui_settings_rows(args, plan, report, args.base_port)
        except Exception as exc:
            rows = [
                ("bad", "THIS PAGE COULD NOT BE RENDERED"),
                ("bad", str(exc)),
                ("dim", "Press R to retry a deep refresh. No lifecycle action was taken."),
            ]

        if operation is not None:
            with operation_lock:
                op_label = operation["label"]
                op_detail = operation["detail"]
                op_started = operation["started"]
            spinner = "|/-\\"[int((now - op_started) * 5) % 4]
            rows = [
                ("warn", "%s %s · %.1fs" % (spinner, op_label, now - op_started)),
                ("normal", op_detail),
                ("normal", ""),
            ] + rows

        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 8 or width < 38:
            safe_add(0, 0, "COLIBRÍ · RAM WORKSPACE", max(1, width - 1), attrs["accent"])
            safe_add(2, 0, "Resize to at least 38 x 8.", max(1, width - 1), attrs["warn"])
            stdscr.refresh()
            try:
                key = stdscr.getch()
            except KeyboardInterrupt:
                key = 3
            if key in (ord("q"), 27, 3):
                if operation is not None:
                    quit_when_idle = True
                    quit_exit_code = 130 if key == 3 else 0
                    if operation["cancelable"]:
                        operation["cancel_event"].set()
                else:
                    return 130 if key == 3 else 0
            continue

        state_text = report.get("state", "unknown") if report else "loading"
        safe_add(0, 0, "COLIBRÍ · RAM WORKSPACE", width - 1, attrs["accent"])
        right = "state · %s" % state_text
        safe_add(0, max(0, width - len(right) - 1), right, len(right), attrs["dim"])
        tab_text = "  ".join(
            ("[%s]" % name.upper()) if index == screen and not help_open else name
            for index, name in enumerate(_TUI_SCREENS)
        )
        if help_open:
            tab_text = "[HELP]  " + tab_text
        safe_add(1, 0, tab_text, width - 1, attrs["heading"])
        safe_add(2, 0, "─" * max(1, width - 1), width - 1, attrs["dim"])

        rendered = _tui_wrap_rows(rows, width - 3)
        content_height = max(1, height - 5)
        max_scroll = max(0, len(rendered) - content_height)
        scroll = min(_tui_review_scroll(pending_action, scroll), max_scroll)
        for offset, (style, line) in enumerate(rendered[scroll : scroll + content_height]):
            safe_add(3 + offset, 1, line, width - 3, attrs.get(style, attrs["normal"]))
        if max_scroll:
            indicator = "%d–%d / %d" % (
                scroll + 1,
                min(len(rendered), scroll + content_height),
                len(rendered),
            )
            safe_add(2, max(0, width - len(indicator) - 1), indicator, len(indicator), attrs["dim"])

        if operation is not None:
            action_hint = (
                "Operation in progress · [c] cancel · navigation remains available"
                if operation["cancelable"]
                else "Operation in progress · navigation remains available"
            )
        elif help_open:
            action_hint = "[?] close help"
        else:
            action_hint = _tui_idle_action_hint(screen, plan, report)
        footer = "%s · ←/→ pages · ↑/↓ scroll · [?] help · [q] quit" % action_hint
        safe_add(height - 2, 0, str(message).ljust(width - 1), width - 1, attrs["dim"])
        safe_add(height - 1, 0, footer.ljust(width - 1), width - 1, curses.A_REVERSE)
        stdscr.refresh()
        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            key = 3

        if key in (ord("q"), 27, 3):
            if operation is not None:
                quit_when_idle = True
                quit_exit_code = 130 if key == 3 else 0
                if operation["cancelable"]:
                    operation["cancel_event"].set()
                    message = "Cancelling safely; Colibri will quit after rollback/cleanup finishes."
                else:
                    message = "This cleanup step cannot be interrupted safely; Colibri will quit when it finishes."
            else:
                return 130 if key == 3 else 0
            continue
        if key == ord("c") and operation is not None and operation["cancelable"]:
            operation["cancel_event"].set()
            message = "Cancellation requested; waiting for rollback/cleanup checkpoints."
            continue
        if key == ord("?"):
            if pending_action == "prepare":
                cancel_confirmation()
                message = "Prepare review cancelled when help was opened."
            help_open = not help_open
            scroll = 0
            continue
        if key in (ord("l"), curses.KEY_RIGHT):
            if pending_action == "prepare":
                cancel_confirmation()
                message = "Prepare review cancelled when leaving the Plan page."
            screen = (screen + 1) % len(_TUI_SCREENS)
            help_open = False
            scroll = 0
            continue
        if key in (ord("h"), curses.KEY_LEFT):
            if pending_action == "prepare":
                cancel_confirmation()
                message = "Prepare review cancelled when leaving the Plan page."
            screen = (screen - 1) % len(_TUI_SCREENS)
            help_open = False
            scroll = 0
            continue
        if key in (ord("j"), curses.KEY_DOWN):
            scroll = _tui_review_scroll(pending_action, scroll + 1)
            continue
        if key in (ord("k"), curses.KEY_UP):
            scroll = _tui_review_scroll(pending_action, scroll - 1)
            continue
        if key == curses.KEY_NPAGE:
            scroll = _tui_review_scroll(pending_action, scroll + content_height)
            continue
        if key == curses.KEY_PPAGE:
            scroll = _tui_review_scroll(pending_action, scroll - content_height)
            continue
        if key == ord("R"):
            invalidate(deep=True, model=True)
            cancel_confirmation()
            message = "Refreshing hardware, model plan, and lifecycle validation…"
            continue
        if help_open:
            continue
        if operation is not None:
            message = "A lifecycle operation is already running."
            continue

        if key == ord("p") and screen == 0:
            action_policy = ActionPolicy.from_state(plan, report)
            if not action_policy.prepare.enabled:
                message = action_policy.prepare.reason
            else:
                token = _plan_confirmation_token(plan)
                review = ReviewIdentity.for_prepare(token, plan, args.base_port)
                if (
                    pending_action != "prepare"
                    or pending_review != review
                    or now > pending_deadline
                ):
                    pending_action = "prepare"
                    pending_review = review
                    pending_deadline = now + 10.0
                    message = "TOTAL RAM %.2f GiB · press p again only if the three facts above are correct." % (
                        plan["staging"]["total_staged_bytes"] / float(GIB)
                    )
                    scroll = 0
                else:
                    if not authorize_privileged_mounts():
                        cancel_confirmation()
                        continue
                    reviewed_token = pending_review.token
                    prepared_args = argparse.Namespace(**dict(vars(args), yes=True))
                    expected_copies = (
                        len(plan["staging"]["selected_shards"])
                        * plan["staging"]["replica_count"]
                    )
                    copied = [0]
                    copied_lock = threading.Lock()

                    def run_prepare(op):
                        def progress(name, size, elapsed):
                            with copied_lock:
                                copied[0] += 1
                                update_operation(
                                    op,
                                    "Copy %d/%d · %s · %.1f MiB/s"
                                    % (
                                        copied[0],
                                        expected_copies,
                                        name,
                                        size / elapsed / MIB if elapsed > 0 else 0.0,
                                    ),
                                )

                        return prepare(
                            prepared_args,
                            progress=progress,
                            display_plan=False,
                            expected_plan_token=reviewed_token,
                            cancel_event=op["cancel_event"],
                        )

                    cancel_confirmation()
                    begin_operation(
                        "prepare",
                        "Preparing RAM workspace",
                        run_prepare,
                        cancelable=True,
                    )
                    message = "Preparation started; progress stays visible and navigation remains available."
            continue

        if key == ord("s"):
            action_policy = ActionPolicy.from_state(plan, report)
            if not action_policy.start.enabled:
                message = action_policy.start.reason
            else:
                start_args = argparse.Namespace(base_port=args.base_port)
                begin_operation(
                    "start",
                    "Loading managed engine",
                    lambda op: start(
                        start_args,
                        cli_path=cli_path,
                        engine_path=engine_path,
                        cancel_event=op["cancel_event"],
                    ),
                    cancelable=True,
                )
                message = "Engine startup runs in the background; this can take a while for a large model."
            continue

        if key == ord("x"):
            action_policy = ActionPolicy.from_state(plan, report)
            if not action_policy.stop.enabled:
                message = action_policy.stop.reason
            else:
                begin_operation("stop", "Stopping managed engines", lambda op: stop(argparse.Namespace()))
                message = "Stopping only verified managed process groups."
            continue

        if key == ord("d"):
            action_policy = ActionPolicy.from_state(plan, report)
            if not action_policy.destroy.enabled:
                message = action_policy.destroy.reason
            else:
                try:
                    current_manifest = _load_manifest(required=True)
                    current_token = _manifest_confirmation_token(current_manifest)
                except (RamdiskError, OSError) as exc:
                    cancel_confirmation()
                    invalidate(deep=True)
                    message = "Destroy review failed: %s" % exc
                    continue
                current_review = ReviewIdentity.for_destroy(
                    current_token,
                    current_manifest["plan"],
                    current_manifest.get("mounts", ()),
                    _persisted_base_port(current_manifest),
                )
                if pending_action != "destroy" or now > pending_deadline:
                    pending_action = "destroy"
                    pending_review = current_review
                    pending_deadline = now + 10.0
                    message = (
                        "CONFIRM DESTROY: engines are stopped; unmount volatile "
                        "weights now by pressing d again within 10s."
                    )
                elif pending_review != current_review:
                    cancel_confirmation()
                    invalidate(deep=True)
                    message = "The active deployment changed; review Destroy again."
                else:
                    reviewed_token = pending_review.token
                    cancel_confirmation()
                    if not authorize_privileged_mounts():
                        continue
                    begin_operation(
                        "destroy",
                        "Destroying RAM workspace",
                        lambda op: destroy(
                            argparse.Namespace(yes=True),
                            expected_manifest_token=reviewed_token,
                        ),
                    )
                    message = "Destroy started; durable state will be preserved."
            continue

        if key == ord("b") and screen == 3:
            action_policy = ActionPolicy.from_state(plan, report)
            if not action_policy.benchmark.enabled:
                message = action_policy.benchmark.reason
            else:
                begin_operation(
                    "benchmark",
                    "Running deterministic path benchmark",
                    lambda op: benchmark(
                        argparse.Namespace(),
                        cli_path=cli_path,
                        engine_path=engine_path,
                        cancel_event=op["cancel_event"],
                    ),
                    cancelable=True,
                )
                message = "Benchmark is running; results will be saved to the scorecard."
            continue

        if screen != 4:
            continue
        action_policy = ActionPolicy.from_state(plan, report)
        if not action_policy.edit_weights.enabled and (
            key != ord("P") or not action_policy.edit_base_port.enabled
        ):
            if key != -1:
                message = (
                    action_policy.edit_base_port.reason
                    if key == ord("P")
                    else action_policy.edit_weights.reason
                )
            continue

        changed = False
        try:
            if key == ord("i") and args.topology == "per-node":
                args.topology = "interleaved"
                changed = True
                message = "Placement changed to one shared model copy and one engine."
            elif key == ord("m"):
                args.mode = "partial" if args.mode == "full" else "full"
                args.capacity_gb = 16.0 if args.mode == "partial" else None
                changed = True
                message = "Staging mode changed to %s." % args.mode
            elif key == ord("H"):
                choices = ("auto", "within_size", "advise")
                args.thp = choices[(choices.index(args.thp) + 1) % len(choices)]
                changed = True
                message = "Huge-page policy changed to %s." % args.thp
            elif key == ord("f"):
                effective = args.prefault if args.prefault is not None else args.mode == "full"
                args.prefault = 0 if effective else 1
                changed = True
                message = "Prefault %s." % ("enabled" if args.prefault else "disabled")
            elif key == ord("y"):
                args.allow_swappable = not args.allow_swappable
                changed = True
                message = "Swappable tmpfs %s." % ("allowed" if args.allow_swappable else "refused")
            elif key in (ord("c"), ord("r"), ord("o"), ord("P"), ord("w")):
                label, current = {
                    ord("c"): ("Per-copy budget in GiB", args.capacity_gb or 16.0),
                    ord("r"): ("Usage profile path ('-' = model default)", args.profile or "<model default>"),
                    ord("o"): ("Mount root", args.mount_root),
                    ord("P"): ("Start base port", args.base_port),
                    ord("w"): ("Concurrent copy workers", args.parallel),
                }[key]
                value = prompt_value(label, current)
                if value is not None:
                    if key == ord("c"):
                        value = float(value)
                        if not math.isfinite(value) or value <= 0:
                            raise ValueError("budget must be positive")
                        args.capacity_gb = value
                    elif key == ord("r"):
                        args.profile = None if value in ("-", "<model default>") else value
                    elif key == ord("o"):
                        args.mount_root = value
                    elif key == ord("P"):
                        value = int(value)
                        ports = _managed_ports_for_plan(plan, value)
                        if (
                            not 1 <= value <= 65535
                            or len(set(ports)) != len(ports)
                            or any(port < 1 or port > 65535 for port in ports)
                        ):
                            raise ValueError(
                                "base port produces invalid or duplicate replica ports"
                            )
                        args.base_port = value
                    elif key == ord("w"):
                        value = int(value)
                        if not 1 <= value <= 64:
                            raise ValueError("workers must be between 1 and 64")
                        args.parallel = value
                    changed = True
                    message = "%s updated." % label
        except (TypeError, ValueError) as exc:
            message = "Invalid setting: %s" % exc
        if changed:
            cancel_confirmation()
            plan_cache = plan_key_cache = None
            scroll = 0


def _load_textual_frontend():
    """Import the optional frontend without making scriptable commands depend on it."""
    import ramdisk_textual

    return ramdisk_textual


def _textual_dependency_missing(error):
    missing = getattr(error, "name", "") or ""
    return missing == "textual" or missing.startswith("textual.")


class _TuiTerminationSignal(BaseException):
    def __init__(self, signum):
        super().__init__("terminal signal %s" % signum)
        self.signum = int(signum)


@contextlib.contextmanager
def _curses_termination_guard():
    """Make legacy-curses SIGHUP/SIGTERM follow its existing cleanup path."""
    previous = {}
    first_signal = {"signum": None}

    def terminate(received, _frame):
        if first_signal["signum"] is not None:
            # Cleanup is already running under this guard. Repeated service
            # manager/SSH signals must not restore the default disposition and
            # kill rollback halfway through.
            return
        first_signal["signum"] = int(received)
        raise _TuiTerminationSignal(received)

    if threading.current_thread() is threading.main_thread():
        for name in ("SIGHUP", "SIGTERM"):
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                previous[signum] = signal.getsignal(signum)
                signal.signal(signum, terminate)
            except (OSError, ValueError):
                previous.pop(signum, None)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                pass


def _join_tui_worker(active):
    """Wait through repeated terminal interrupts until cleanup reaches a checkpoint."""
    while True:
        try:
            active["thread"].join()
            return
        except (KeyboardInterrupt, _TuiTerminationSignal):
            if active.get("cancelable"):
                active["cancel_event"].set()


def _run_tui_frontend(callback):
    """Run one frontend while preserving active lifecycle cleanup on every exit."""
    try:
        return callback()
    except (KeyboardInterrupt, _TuiTerminationSignal) as interruption:
        signum = getattr(interruption, "signum", None)
        exit_code = 128 + signum if signum is not None else 130
        with _tui_worker_guard:
            active = _tui_worker
        if active is not None:
            if active.get("cancelable"):
                active["cancel_event"].set()
                print(
                    "coli ramdisk: termination received; waiting for safe rollback/cleanup",
                    file=sys.stderr,
                )
            else:
                print(
                    "coli ramdisk: termination received during non-interruptible cleanup; waiting",
                    file=sys.stderr,
                )
            _join_tui_worker(active)
            error = active.get("error")
            if error is not None and not isinstance(error, _OperationCancelled):
                print(
                    "coli ramdisk: cleanup failed after interrupt: %s" % error,
                    file=sys.stderr,
                )
                return 2
        return exit_code
    except BaseException as interface_error:
        with _tui_worker_guard:
            active = _tui_worker
        if active is not None:
            if active.get("cancelable"):
                active["cancel_event"].set()
            print(
                "coli ramdisk: interface exited; waiting for active cleanup",
                file=sys.stderr,
            )
            _join_tui_worker(active)
            operation_error = active.get("error")
            if operation_error is not None and not isinstance(
                operation_error, _OperationCancelled
            ):
                print(
                    "coli ramdisk: active operation/cleanup also failed: %s"
                    % operation_error,
                    file=sys.stderr,
                )
                raise RamdiskError(
                    "interface failed while active operation cleanup also failed: %s"
                    % operation_error
                ) from interface_error
        raise


def launch_tui(args, cli_path=None, engine_path=None, system=None):
    if not sys.platform.startswith("linux"):
        print("coli ramdisk: the TUI is supported only on Linux", file=sys.stderr)
        return 2
    global _tui_worker
    requested_ui = os.environ.get("COLI_RAMDISK_UI", "auto").strip().lower()
    if requested_ui not in ("auto", "textual", "curses"):
        print(
            "coli ramdisk: COLI_RAMDISK_UI must be auto, textual, or curses",
            file=sys.stderr,
        )
        return 2

    textual_frontend = None
    if requested_ui in ("auto", "textual"):
        try:
            textual_frontend = _load_textual_frontend()
        except ModuleNotFoundError as exc:
            if not _textual_dependency_missing(exc):
                raise
            if requested_ui == "textual":
                print(
                    "coli ramdisk: Textual UI requested but Textual is not installed; "
                    "install the TUI dependency or set COLI_RAMDISK_UI=curses",
                    file=sys.stderr,
                )
                return 2

    try:
        if textual_frontend is not None:
            return _run_tui_frontend(
                lambda: textual_frontend.launch_tui(
                    args,
                    cli_path=cli_path,
                    engine_path=engine_path,
                    lifecycle=sys.modules[__name__],
                )
            )
        import curses

        with _curses_termination_guard():
            return _run_tui_frontend(
                lambda: curses.wrapper(_tui, args, cli_path, engine_path)
            )
    finally:
        with _tui_worker_guard:
            if _tui_worker is not None and not _tui_worker["thread"].is_alive():
                _tui_worker = None


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--copy-worker":
        try:
            sys.exit(_copy_worker_main(sys.argv[2], sys.argv[3], sys.argv[4]))
        except Exception as error:
            print(error, file=sys.stderr)
            sys.exit(1)
    print("ramdisk.py is a support module; run `coli ramdisk`", file=sys.stderr)
    sys.exit(2)
