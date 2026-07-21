"""NUMA-aware tmpfs staging and managed-engine lifecycle for ``coli ramdisk``.

The module intentionally uses only the Python standard library.  Planning and
status are unprivileged; the only privileged subprocesses are the exact mount
and unmount commands issued by :func:`prepare` and :func:`destroy`.
"""

from __future__ import print_function

import argparse
import concurrent.futures
import contextlib
import datetime
import errno
import functools
import hashlib
import json
import math
import mmap
import os
import platform
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
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

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
    online = plan.get("hardware", {}).get("online_nodes")
    if topology == "interleaved":
        expected = [(None, root)]
    else:
        if (
            not isinstance(online, list)
            or not online
            or any(not isinstance(node, int) or isinstance(node, bool) or node < 0 for node in online)
            or len(set(online)) != len(online)
        ):
            raise RamdiskError("RAM-disk manifest has an invalid NUMA node set")
        expected = [(node, os.path.join(root, "node%d" % node)) for node in online]
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


def _meminfo(path="/proc/meminfo"):
    values = {}
    for line in _read_text(path).splitlines():
        match = re.match(r"^([^:]+):\s*(\d+)(?:\s+kB)?", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    return values


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
    return {
        "linux": linux,
        "kernel_release": platform.release(),
        "online_nodes": online,
        "nodes": nodes,
        "physical_cores": _physical_cores(sorted(set(all_cpus))),
        "memory": {
            "total_bytes": memory.get("MemTotal", 0),
            "available_bytes": memory.get("MemAvailable", memory.get("MemFree", 0)),
        },
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


def build_plan(args, hardware=None, model=None):
    hardware = hardware or discover_hardware()
    model = model or scan_model(args.model)
    mode = getattr(args, "mode", "full")
    topology = getattr(args, "topology", "interleaved")
    capacity_gb = getattr(args, "capacity_gb", None)
    if mode not in ("full", "partial") or topology not in ("interleaved", "per-node"):
        raise RamdiskError("invalid RAM-disk mode or topology")
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
    global_margin = max(memory["total_bytes"] // 10, 16 * GIB)
    page_tables = int(math.ceil(float(staged_bytes + runtime_bytes) / 512.0))
    required_global = staged_bytes + runtime_bytes + page_tables + global_margin
    blockers = []
    warnings = []
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
        for node in hardware.get("nodes", []):
            if not node.get("cpus"):
                blockers.append(
                    "NUMA node %d has no online CPUs and cannot host a node-local engine"
                    % node["id"]
                )
    if capacity_bytes < staged_bytes:
        blockers.append("selected shard closures exceed the staging budget")
    if topology == "interleaved":
        if memory["available_bytes"] < required_global:
            blockers.append("available memory would breach the global runtime/OS reserve")
    else:
        for node in hardware["nodes"]:
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
    nodes = hardware["online_nodes"]
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
        policy = "interleave:" + node_list if node is None else "bind:%d" % node
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
        total_os_margin = sum(int(node.get("reserve_bytes", 0)) for node in hardware["nodes"])
        total_required = sum(int(node.get("required_bytes", 0)) for node in hardware["nodes"])
        if memory["available_bytes"] < total_required:
            blockers.append("available memory cannot hold all per-node replicas and reserves")
    else:
        total_os_margin = global_margin
        total_required = required_global
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
            "available_bytes": memory["available_bytes"],
            "total_runtime_bytes": total_runtime_bytes,
            "total_page_table_bytes": total_page_table_bytes,
            "total_os_margin_bytes": total_os_margin,
            "total_required_bytes": total_required,
        },
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
        parser.add_argument("--model", default=default)
    parser.add_argument("--mode", choices=("full", "partial"), default=argparse.SUPPRESS if suppress else "full")
    parser.add_argument(
        "--topology", choices=("interleaved", "per-node"), default=argparse.SUPPRESS if suppress else "interleaved"
    )
    parser.add_argument("--capacity-gb", type=float, default=default)
    parser.add_argument("--profile", default=default)
    parser.add_argument("--mount-root", default=argparse.SUPPRESS if suppress else DEFAULT_MOUNT_ROOT)
    parser.add_argument("--thp", choices=("auto", "within_size", "advise"), default=argparse.SUPPRESS if suppress else "auto")
    parser.add_argument("--allow-swappable", action="store_true", default=argparse.SUPPRESS if suppress else False)
    parser.add_argument("--prefault", type=int, choices=(0, 1), default=default)
    parser.add_argument("--parallel", type=int, default=argparse.SUPPRESS if suppress else 2)
    if "--ctx" not in parser._option_string_actions:
        parser.add_argument("--ctx", type=int, default=argparse.SUPPRESS if suppress else 0)


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
    start_parser.add_argument("--base-port", type=int, default=8000)
    actions.add_parser("stop", parents=[after], help="stop only verified managed processes")
    destroy_parser = actions.add_parser("destroy", parents=[after], help="unmount volatile weights safely")
    destroy_parser.add_argument("--yes", action="store_true")


def _json_print(value):
    print(json.dumps(value, indent=2, sort_keys=True))


def _human_plan(plan):
    print("RAM-disk plan: %s / %s" % (plan["mode"], plan["topology"]))
    print("  model: %s" % plan["model"]["path"])
    print("  staged: %.2f GiB total (%d replica(s), %.2f GiB each) in %d shard(s); %d direct expert(s)" % (
        plan["staging"]["total_staged_bytes"] / float(GIB),
        plan["staging"]["replica_count"],
        plan["staging"]["staged_bytes"] / float(GIB),
        len(plan["staging"]["selected_shards"]),
        plan["staging"]["direct_mapped_expert_count"],
    ))
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


def _privileged(command, hardware):
    if os.geteuid() == 0:
        return command
    sudo = _trusted_system_binary("sudo")
    return [sudo, "--"] + command


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
        result = _run(_privileged(command, hardware))
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
    return values.get("MemAvailable", values.get("MemFree", 0))


def _available_for_mount(mount):
    if mount.get("node") is None:
        return _available_memory()
    values = _node_meminfo(int(mount["node"]))
    return values.get("MemFree", values.get("MemAvailable", 0))


def _copy_stream(src, tmp, expected_size):
    source_fd = os.open(src, os.O_RDONLY)
    try:
        destination_fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            copied = 0
            while copied < expected_size:
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


def _copy_one(src, destination, expected_size, reserve_floor, progress=None, available=None):
    available = available or _available_memory
    if available() < reserve_floor:
        raise RamdiskError("available memory reached the protected reserve before %s" % os.path.basename(src))
    tmp = destination + ".coli-copy-%d-%s" % (os.getpid(), secrets.token_hex(4))
    started = time.monotonic()
    try:
        _copy_stream(src, tmp, expected_size)
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


def _copy_one_affined(src, destination, expected_size, node, numactl, reserve_floor, progress=None, available=None):
    available = available or _available_memory
    if available() < reserve_floor:
        raise RamdiskError("available memory reached the protected reserve before replica copy")
    tmp = destination + ".coli-copy-%d-%s" % (os.getpid(), secrets.token_hex(4))
    started = time.monotonic()
    command = [
        numactl,
        "--cpunodebind=%d" % node,
        "--membind=%d" % node,
        sys.executable,
        os.path.abspath(__file__),
        "--copy-worker",
        src,
        tmp,
        str(expected_size),
    ]
    try:
        result = _run(command)
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


def _populate_mount(plan, mount, source_root=None, progress=None):
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
    available = lambda: _available_for_mount(mount)
    workers = max(1, min(plan["parallel"], len(selected) or 1))
    admission_lock = threading.Lock()
    inflight = [0]

    def copy_name(name):
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
                    reserve_floor,
                    progress,
                    available,
                )
            return _copy_one(source, destination, expected, reserve_floor, progress, available)
        finally:
            with admission_lock:
                inflight[0] -= expected

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(copy_name, name) for name in selected]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    for name in linked:
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
    pages_per_shard = max(32, min(1024, 4096 // max(1, len(selected_names))))
    for name in selected_names:
        path = os.path.join(mount["path"], name)
        for node, count in _sample_numa_allocation(
            path, pages_per_shard, node_count=len(plan["hardware"]["online_nodes"])
        ).items():
            allocation[node] = allocation.get(node, 0) + count
    if len(plan["hardware"]["online_nodes"]) > 1:
        total = sum(allocation.values())
        if not total:
            raise RamdiskError("could not verify actual NUMA allocation for staged shards")
        if mount["node"] is not None:
            local = allocation.get(str(mount["node"]), 0)
            if float(local) / total < 0.95:
                raise RamdiskError("node-local tmpfs sample is below 95%% local allocation")
        else:
            ideal = float(total) / len(plan["hardware"]["online_nodes"])
            if any(abs(allocation.get(str(node), 0) - ideal) / ideal > 0.15 for node in plan["hardware"]["online_nodes"]):
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
def prepare(args, progress=None, display_plan=True):
    if _load_manifest(required=False) is not None:
        raise RamdiskError("a RAM-disk manifest already exists; stop/destroy it before preparing another")
    plan = build_plan(args)
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
            source = seed if index and plan["topology"] == "per-node" else None
            _populate_mount(plan, mount, source_root=source, progress=progress)
            if seed is None:
                seed = mount["path"]
            manifest["mounts"][index]["numa_allocation"] = _validate_namespace(plan, mount)
            _save_manifest(manifest)
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
                if (
                    actual
                    and expected
                    and actual["filesystem"] == "tmpfs"
                    and actual["source"] == "tmpfs"
                    and actual["mount_id"] == expected.get("mount_id")
                    and actual["device"] == expected.get("device")
                ):
                    _umount_path(mount["path"], plan["hardware"])
            except Exception as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
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


def _admit_runtime(plan, mount, benchmark=False):
    """Recheck the reviewed post-staging floor immediately before launch."""
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
    required = runtime_bytes + page_tables + margin
    available = _available_for_mount(mount)
    if available < required:
        label = "global memory" if mount.get("node") is None else "NUMA node %d" % mount["node"]
        raise RamdiskError(
            "%s has %d bytes available; launch would breach the %d-byte runtime/OS floor"
            % (label, available, required)
        )
    return {"available_bytes": available, "required_bytes": required}


def _group_alive(pgid):
    try:
        os.killpg(int(pgid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


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


def _wait_managed_ready(record, timeout, api_key=None):
    deadline = time.monotonic() + timeout
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    last_error = "listener not ready"
    while time.monotonic() < deadline:
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
        time.sleep(0.5)
    raise RamdiskError(
        "managed engine on port %s did not become ready within %.0fs (%s); see %s"
        % (record["port"], timeout, last_error, record["log"])
    )


@_exclusive_lifecycle
def start(args, cli_path=None, engine_path=None):
    manifest = _load_manifest(required=True)
    if manifest.get("state") not in ("ready", "stopped"):
        raise RamdiskError("manifest state is %s, not ready" % manifest.get("state"))
    _assert_ready_mounts(manifest)
    plan = manifest["plan"]
    cli_path = cli_path or os.path.join(os.path.dirname(__file__), "coli")
    model = plan["model"]["path"]
    canonical_usage = os.path.join(model, ".coli_usage")
    foreign = []
    recovered = False
    for record in manifest.get("processes", []):
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
    base_port = int(getattr(args, "base_port", 8000))
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
    spawned = []
    launch_contexts = []
    manifest["processes"] = []
    manifest["ports"] = []
    manifest["state"] = "starting"
    _save_manifest(manifest)
    try:
        for index, mount in enumerate(manifest["mounts"]):
            _admit_runtime(plan, mount, benchmark=False)
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
                    "COLI_NUMA": "1" if node is None else "0",
                    "OMP_NUM_THREADS": str(
                        plan["hardware"]["physical_cores"]
                        if node is None
                        else next(item["physical_cores"] for item in plan["hardware"]["nodes"] if item["id"] == node)
                    ),
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
                    "--cpunodebind=%d" % node,
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
            )
            _save_manifest(manifest)
        manifest["state"] = "running"
        _save_manifest(manifest)
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

        # These are direct children created in this function, so their group ids
        # are trusted independently of serialized state. Include the child that
        # may have failed before a manifest record could be published.
        for process in reversed(spawned):
            failure = _terminate_group(process.pid)
            try:
                process.wait(timeout=1)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
            if failure and _group_alive(process.pid):
                cleanup_failures.append(failure)
                surviving_groups.add(process.pid)
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
        manifest["state"] = "error"
        manifest["launch_error"] = str(launch_error)
        if cleanup_failures:
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
            failure = _terminate_group(pgid)
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
    manifest["state"] = "error" if failures or any(
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
def destroy(args):
    manifest = _load_manifest(required=True)
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
            "COLI_NUMA": "1" if node is None else "0",
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


def _score_variant(engine_path, manifest, name, weights_dir, rammap, knobs):
    # Import the existing stdlib-only engine protocol client lazily. This keeps
    # plan/status usable even in minimal packaging probes while ensuring one
    # persistent process receives the warm-up and all three measured turns.
    from openai_server import Engine, render_chat

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
            "--cpunodebind=%d" % node,
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
            parts = []
            profile_seq = engine.profile_seq
            started = time.monotonic()
            stats = engine.generate(prompt, 32, 0.0, 1.0, parts.append, cache_slot=0)
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
        if engine is not None:
            engine.close()
        log.close()
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


def _aggregate_score(manifest, engine_path=None, knobs=None):
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

    from openai_server import Engine, render_chat

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

        def launch(mount):
            node = int(mount["node"])
            _admit_runtime(plan, mount, benchmark=False)
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
                        "--cpunodebind=%d" % node,
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
            parts = []
            engine = entry["engine"]
            profile_seq = engine.profile_seq
            started = time.monotonic()
            stats = engine.generate(prompt, 32, 0.0, 1.0, parts.append, cache_slot=0)
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
    except (RamdiskError, RuntimeError, OSError, subprocess.SubprocessError, ValueError) as exc:
        return {
            "status": "error",
            "error": str(exc),
            "per_node_tokens_per_second": [],
            "slowest_node_tokens_per_second": None,
            "total_tokens_per_second": None,
        }
    finally:
        for entry in launched:
            try:
                entry["engine"].close()
            except Exception:
                pass
            try:
                entry["log_stream"].close()
            except Exception:
                pass


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
def benchmark(args, cli_path=None, engine_path=None):
    manifest = _load_manifest(required=True)
    if manifest.get("state") not in ("ready", "running", "stopped"):
        raise RamdiskError("benchmark requires a ready RAM-disk manifest")
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
        try:
            variants.append(_score_variant(engine_path, manifest, name, weights, rammap, knobs))
        except (RamdiskError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
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
    aggregate = _aggregate_score(manifest, engine_path=engine_path, knobs=aggregate_knobs)
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


def dispatch(args, cli_path=None, engine_path=None, system=None):
    action = getattr(args, "ramdisk_action", None)
    try:
        if action == "plan":
            value = build_plan(args)
            if getattr(args, "json", False):
                _json_print(value)
            else:
                _human_plan(value)
            return 2 if value["blockers"] else 0
        if action == "prepare":
            value = prepare(args)
            print("RAM-disk ready: %s" % ", ".join(record["path"] for record in value["mounts"]))
            return 0
        if action == "status":
            value = status()
            if getattr(args, "json", False):
                _json_print(value)
            else:
                _human_status(value)
            return 0
        if action == "benchmark":
            value = benchmark(args, cli_path=cli_path, engine_path=engine_path)
            if getattr(args, "json", False):
                _json_print(value)
            else:
                _human_benchmark(value)
            return 0
        if action == "start":
            value = start(args, cli_path=cli_path, engine_path=engine_path)
            print("managed engine ports: %s" % ", ".join(str(port) for port in value["ports"]))
            return 0
        if action == "stop":
            stop(args)
            print("managed engines stopped; usage deltas merged")
            return 0
        if action == "destroy":
            value = destroy(args)
            print("RAM-disk destroyed; durable state and benchmark history preserved")
            return 0
        raise RamdiskError("choose a ramdisk action or run the interactive TUI")
    except (RamdiskError, OSError, subprocess.SubprocessError) as exc:
        if getattr(args, "json", False):
            _json_print({"schema": "colibri.ramdisk.error.v1", "version": 1, "error": str(exc)})
        else:
            print("coli ramdisk: %s" % exc, file=sys.stderr)
        return 2


def _tui(stdscr, initial, cli_path, engine_path):
    import curses

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
    curses.curs_set(0)
    stdscr.timeout(250)
    screen = 0
    pending_action = None
    pending_deadline = 0.0
    message = "h/l screens · m mode · t topology · H THP · +/- capacity · r/o/P config · f/y toggles · R refresh · p prepare · s/x start/stop · b bench · d destroy · q quit"
    screens = ("Hardware / NUMA", "Capacity and staging", "Copy / validation", "Benchmark scorecard", "Running engines / cleanup")
    hardware_cache = None
    model_cache = None
    plan_cache = None
    plan_key_cache = None
    report_cache = None
    hardware_checked = report_checked = 0.0
    deep_status_refresh = True
    while True:
        plan = None
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        stdscr.addnstr(0, 0, "colibri RAM-disk — %s" % screens[screen], width - 1, curses.A_BOLD)
        try:
            now = time.monotonic()
            if hardware_cache is None or now - hardware_checked >= 30.0:
                hardware_cache = discover_hardware()
                hardware_checked = now
                plan_key_cache = None
            hardware = hardware_cache
            if model_cache is None:
                model_cache = scan_model(args.model)
            if report_cache is None or now - report_checked >= 2.0:
                report_cache = status(deep=deep_status_refresh)
                deep_status_refresh = False
                report_checked = now
            report = report_cache
            plan_key = (
                args.mode, args.topology, args.mount_root, args.capacity_gb,
                args.profile, args.allow_swappable, args.thp, args.prefault,
                args.parallel, args.ctx, hardware_checked,
            )
            if plan_cache is None or plan_key != plan_key_cache:
                plan_cache = build_plan(args, hardware=hardware, model=model_cache)
                plan_key_cache = plan_key
            plan = plan_cache
            lines = []
            if screen == 0:
                lines.extend(
                    [
                        "Kernel: %s  tmpfs=%s  noswap=%s  THP=%s" % (
                            hardware["kernel_release"], hardware["tmpfs"]["supported"],
                            hardware["tmpfs"]["noswap_supported"], hardware["thp"]["shmem_enabled"],
                        ),
                        "RAM available/total: %.1f / %.1f GiB  swap used %.2f GiB" % (
                            hardware["memory"]["available_bytes"] / float(GIB),
                            hardware["memory"]["total_bytes"] / float(GIB),
                            hardware["swap"]["used_bytes"] / float(GIB),
                        ),
                    ]
                )
                for node in hardware["nodes"]:
                    lines.append("Node %d CPUs %s (%d physical) memory %.1f/%.1f GiB distances %s" % (
                        node["id"], node["cpu_list"], node["physical_cores"],
                        node["memory_available_bytes"] / float(GIB), node["memory_total_bytes"] / float(GIB), node["distance"],
                    ))
            elif screen == 1:
                lines.extend(
                    [
                        "Mode: %s  topology: %s  mount: %s" % (args.mode, args.topology, args.mount_root),
                        "Budget: %s  profile: %s  THP: %s  prefault: %s  base port: %s" % (args.capacity_gb, args.profile or "auto .coli_usage", plan["mount_options"]["thp"], plan["prefault"], args.base_port),
                        "Stage %.2f GiB total (%d x %.2f) / %d shards; %d direct experts; profile coverage %.1f%%" % (
                            plan["staging"]["total_staged_bytes"] / float(GIB),
                            plan["staging"]["replica_count"],
                            plan["staging"]["staged_bytes"] / float(GIB), len(plan["staging"]["selected_shards"]),
                            plan["staging"]["direct_mapped_expert_count"], plan["profile"]["coverage"] * 100,
                        ),
                        "Total runtime %.2f GiB; OS margins %.2f GiB; page tables %.2f GiB; projection %.2f GiB" % (
                            plan["reserve"]["total_runtime_bytes"] / float(GIB), plan["reserve"]["total_os_margin_bytes"] / float(GIB),
                            plan["reserve"]["total_page_table_bytes"] / float(GIB),
                            plan["reserve"]["total_required_bytes"] / float(GIB),
                        ),
                    ]
                )
                if args.mode == "partial":
                    pin = plan["profile"]["pin_comparison"]
                    lines.append(
                        "Same-budget hot PIN: %d experts, %.1f%% profile coverage (tmpfs shard closures %.1f%%)"
                        % (
                            len(pin["selected_experts"]),
                            pin["coverage"] * 100,
                            plan["profile"]["coverage"] * 100,
                        )
                    )
                lines.extend("BLOCKED: " + value for value in plan["blockers"])
                lines.extend("Warning: " + value for value in plan["warnings"])
            elif screen == 2:
                lines.append(
                    "Lifecycle state: %s (%s validation)"
                    % (report["state"], "deep" if report.get("deep_validation") else "fast")
                )
                lines.extend("%s: %s, NUMA %s" % (item["path"], "verified" if item["verified"] else "not verified", item["numa_allocation"]) for item in report["mounts"])
                lines.append("Preparation uses atomic .coli-copy files, bounded workers, DONTNEED, header/size/fingerprint validation.")
            elif screen == 3:
                history = _read_json(_benchmarks_path()) or {"results": []}
                if history["results"]:
                    latest = history["results"][-1]
                    lines.append("Latest %s: best %s, knobs %s" % (latest["created_at"], latest["best_variant"], latest["best_runtime_knobs"]))
                    for variant in latest["variants"]:
                        if variant.get("status") != "ok":
                            lines.append("%s: %s" % (variant["name"], variant.get("status")))
                            continue
                        score = variant["interactive"]
                        lines.append(
                            "%s: TTFT %s ms | tok/s p50/p95 %s/%s | fwd p50/p99 %s/%s ms | RAM %.1f%% | SSD %s B/tok"
                            % (
                                variant["name"],
                                "%.1f" % score["ttft_ms"] if score["ttft_ms"] is not None else "n/a",
                                "%.2f" % score["p50_tokens_per_second"] if score["p50_tokens_per_second"] is not None else "n/a",
                                "%.2f" % score["p95_tokens_per_second"] if score["p95_tokens_per_second"] is not None else "n/a",
                                "%.1f" % score["forward_p50_ms"] if score["forward_p50_ms"] is not None else "n/a",
                                "%.1f" % score["forward_p99_ms"] if score["forward_p99_ms"] is not None else "n/a",
                                score["ram_map_coverage"] * 100,
                                "%.0f" % score["ssd_bytes_per_token"] if score["ssd_bytes_per_token"] is not None else "n/a",
                            )
                        )
                    aggregate = latest["aggregate"]
                    lines.append("Aggregate %s: slowest %s tok/s | total %s tok/s" % (
                        aggregate.get("status"), aggregate.get("slowest_node_tokens_per_second"), aggregate.get("total_tokens_per_second")
                    ))
                    system_score = latest["system"]
                    lines.append("System: stage %s s | prefault %s s | RSS %s | mount shmem %.2f GiB | swap +%.3f GiB | host huge %.1f%%" % (
                        system_score.get("stage_seconds"), system_score.get("prefault_seconds"),
                        "%.2f GiB" % (system_score["rss_bytes"] / GIB) if system_score.get("rss_bytes") is not None else "n/a",
                        system_score["shmem_bytes"] / float(GIB), system_score["swap_delta_bytes"] / float(GIB),
                        system_score["huge_page_coverage"] * 100,
                    ))
                    lines.append("NUMA placement: %s" % system_score.get("numa_page_placement"))
                else:
                    lines.append("No benchmark history. Press b after prepare.")
            else:
                lines.append("State: %s  ports: %s" % (report["state"], report.get("ports", [])))
                memory_now = _meminfo()
                lines.append("Host: shmem %.2f GiB | swap %.3f GiB | huge %.1f%%" % (
                    memory_now.get("Shmem", 0) / float(GIB), hardware["swap"]["used_bytes"] / float(GIB),
                    100.0 * memory_now.get("ShmemPmdMapped", 0) / max(1, memory_now.get("Shmem", 0)),
                ))
                for item in report["processes"]:
                    metrics = _managed_process_metrics(item)
                    lines.append("PID %s port %s node %s: %s | RSS %s | RAM map %s/%s GiB | latest SSD %s GiB" % (
                        item["pid"], item["port"], item["node"], item["reason"],
                        "%.2f GiB (%d processes)" % (metrics["rss_bytes"] / float(GIB), metrics["rss_processes"])
                        if metrics["rss_bytes"] is not None else "n/a",
                        metrics["rammap_experts"] if metrics["rammap_experts"] is not None else "n/a",
                        "%.2f" % (metrics["rammap_bytes"] / GIB) if metrics["rammap_bytes"] is not None else "n/a",
                        "%.3f" % (metrics["latest_ssd_bytes"] / GIB) if metrics["latest_ssd_bytes"] is not None else "n/a",
                    ))
                lines.append("Stop merges only post-baseline usage deltas. Destroy preserves KV files and benchmarks.")
        except Exception as exc:
            lines = ["Cannot render this screen: %s" % exc]
        for row, line in enumerate(lines[: max(0, height - 4)], 2):
            stdscr.addnstr(row, 0, str(line), width - 1)
        stdscr.addnstr(height - 1, 0, message, width - 1, curses.A_REVERSE)
        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), 27):
            return 0
        if key in (ord("l"), curses.KEY_RIGHT):
            screen = (screen + 1) % len(screens)
        elif key in (ord("h"), curses.KEY_LEFT):
            screen = (screen - 1) % len(screens)
        elif key == ord("m"):
            args.mode = "partial" if args.mode == "full" else "full"
            if args.mode == "partial" and not args.capacity_gb:
                args.capacity_gb = 16.0
        elif key == ord("t"):
            args.topology = "per-node" if args.topology == "interleaved" else "interleaved"
        elif key == ord("H"):
            choices = ("auto", "within_size", "advise")
            args.thp = choices[(choices.index(args.thp) + 1) % len(choices)]
        elif key == ord("R"):
            hardware_cache = report_cache = plan_cache = model_cache = None
            plan_key_cache = None
            deep_status_refresh = True
            message = "hardware, model plan, and lifecycle status refreshed"
        elif key in (ord("+"), ord("=")):
            args.capacity_gb = (args.capacity_gb or 0) + 1
        elif key == ord("-"):
            args.capacity_gb = max(1, (args.capacity_gb or 1) - 1)
        elif key in (ord("r"), ord("o"), ord("P")):
            label, current = {
                ord("r"): ("Profile path (empty = model .coli_usage)", args.profile or ""),
                ord("o"): ("Mount root", args.mount_root),
                ord("P"): ("Base port", str(args.base_port)),
            }[key]
            curses.echo(); curses.curs_set(1)
            stdscr.move(height - 2, 0); stdscr.clrtoeol(); stdscr.addnstr(height - 2, 0, label + ": ", width - 1)
            stdscr.refresh()
            try:
                value = stdscr.getstr(height - 2, min(width - 1, len(label) + 2), max(1, width - len(label) - 3)).decode("utf-8").strip()
                if key == ord("r"):
                    args.profile = value or None
                elif key == ord("o") and value:
                    args.mount_root = value
                elif key == ord("P") and value:
                    args.base_port = int(value)
            except (ValueError, UnicodeDecodeError):
                message = "invalid value"
            finally:
                curses.noecho(); curses.curs_set(0)
        elif key == ord("f"):
            effective = getattr(args, "prefault", None)
            if effective is None:
                effective = args.mode == "full"
            args.prefault = 0 if effective else 1
        elif key == ord("y"):
            args.allow_swappable = not args.allow_swappable
        elif key == ord("p"):
            if not plan or plan.get("blockers"):
                message = "preparation is blocked; review the capacity screen"
                continue
            now = time.monotonic()
            if pending_action != "prepare" or now > pending_deadline:
                pending_action, pending_deadline = "prepare", now + 10.0
                message = (
                    "CONFIRM: stage %.2f GiB at %s with %.2f GiB protected runtime/OS reserve; press p again within 10s"
                    % (
                        plan["staging"]["total_staged_bytes"] / float(GIB),
                        args.mount_root,
                        (
                            plan["reserve"]["total_runtime_bytes"]
                            + plan["reserve"]["total_page_table_bytes"]
                            + plan["reserve"]["total_os_margin_bytes"]
                        )
                        / float(GIB),
                    )
                )
            else:
                pending_action = None
                copied = [0]
                draw_lock = threading.Lock()

                def tui_progress(name, size, elapsed):
                    with draw_lock:
                        copied[0] += 1
                        rate = size / elapsed / MIB if elapsed > 0 else 0.0
                        progress_text = "Copy %d/%d: %s — %.1f MiB/s" % (
                            copied[0],
                            len(plan["staging"]["selected_shards"])
                            * plan["staging"]["replica_count"],
                            name,
                            rate,
                        )
                        stdscr.addnstr(max(1, height - 2), 0, progress_text.ljust(max(1, width - 1)), width - 1)
                        stdscr.refresh()

                try:
                    prepared = prepare(
                        argparse.Namespace(**dict(vars(args), yes=True)),
                        progress=tui_progress,
                        display_plan=False,
                    )
                    message = "ready: " + ", ".join(item["path"] for item in prepared["mounts"])
                    report_cache = None
                    plan_key_cache = None
                except (RamdiskError, OSError, subprocess.SubprocessError) as exc:
                    message = "prepare failed: %s" % exc
        elif key == ord("s"):
            rc = dispatch(argparse.Namespace(**dict(vars(args), ramdisk_action="start", base_port=args.base_port)), cli_path, engine_path)
            message = "start returned %d" % rc
            report_cache = None
        elif key == ord("x"):
            rc = dispatch(argparse.Namespace(**dict(vars(args), ramdisk_action="stop")), cli_path, engine_path)
            message = "stop returned %d" % rc
            report_cache = None
        elif key == ord("d"):
            now = time.monotonic()
            if pending_action != "destroy" or now > pending_deadline:
                pending_action, pending_deadline = "destroy", now + 10.0
                message = "CONFIRM: stop verified engines and unmount volatile weights; press d again within 10s"
            else:
                pending_action = None
                try:
                    result = destroy(argparse.Namespace(yes=True))
                    message = "destroyed; durable state preserved%s" % (
                        " (mountpoints preserved)" if result["empty_mountpoints_preserved"] else ""
                    )
                    report_cache = None
                    plan_key_cache = None
                except (RamdiskError, OSError, subprocess.SubprocessError) as exc:
                    message = "destroy failed: %s" % exc
        elif key == ord("b"):
            message = "benchmark running: one warm-up + three deterministic 32-token runs per variant"
            stdscr.addnstr(height - 1, 0, message.ljust(max(1, width - 1)), width - 1, curses.A_REVERSE)
            stdscr.refresh()
            try:
                result = benchmark(
                    argparse.Namespace(), cli_path=cli_path, engine_path=engine_path
                )
                message = "benchmark complete; best %s" % result.get("best_variant")
                report_cache = None
            except (RamdiskError, OSError, subprocess.SubprocessError) as exc:
                message = "benchmark failed: %s" % exc


def launch_tui(args, cli_path=None, engine_path=None, system=None):
    if not sys.platform.startswith("linux"):
        print("coli ramdisk: the TUI is supported only on Linux", file=sys.stderr)
        return 2
    import curses

    return curses.wrapper(_tui, args, cli_path, engine_path)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--copy-worker":
        try:
            sys.exit(_copy_worker_main(sys.argv[2], sys.argv[3], sys.argv[4]))
        except Exception as error:
            print(error, file=sys.stderr)
            sys.exit(1)
    print("ramdisk.py is a support module; run `coli ramdisk`", file=sys.stderr)
    sys.exit(2)
