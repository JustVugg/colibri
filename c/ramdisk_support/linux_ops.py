"""Linux filesystem and kernel operations used by RAM-disk discovery."""

from __future__ import print_function

import contextlib
import os
import platform
import posixpath
import re
import shutil
import stat
import subprocess
import threading

from .common import RamdiskError, _parse_range_list
from .platform_ops import (
    UNSUPPORTED_PLATFORM_REASON,
    current_euid,
    current_uid,
    get_platform_ops,
)


def _read_text(path, default=""):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            return stream.read()
    except OSError:
        return default


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


def _require_linux():
    if not get_platform_ops().is_linux:
        raise RamdiskError(UNSUPPORTED_PLATFORM_REASON)


def _unescape_mount(value):
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
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
    if path == "/proc/self/mountinfo":
        _require_linux()
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
                    "options": sorted(
                        set(_split_mount_options(fields[5]))
                    ),
                    "optional": fields[6:separator],
                    "filesystem": fields[separator + 1],
                    "source": _unescape_mount(fields[separator + 2]),
                    "super_options": sorted(
                        set(_split_mount_options(fields[separator + 3]))
                    ),
                }
            )
        except (ValueError, IndexError):
            continue
    return result


def _mount_at(path, *, mount_table=None):
    mount_table = _mount_table if mount_table is None else mount_table
    path = posixpath.normpath(posixpath.abspath(path))
    matches = [
        mount
        for mount in mount_table()
        if posixpath.normpath(mount["path"]) == path
    ]
    if len(matches) > 1:
        raise RamdiskError(
            "refusing ambiguous stacked mounts at %s (mount ids %s)"
            % (
                path,
                ", ".join(str(item["mount_id"]) for item in matches),
            )
        )
    return matches[0] if matches else None


def _filesystem_for_path(path, *, mount_table=None):
    """Return the filesystem of the longest mountpoint containing ``path``."""
    mount_table = _mount_table if mount_table is None else mount_table
    normalized = posixpath.normpath(posixpath.abspath(path))
    matches = []
    for mount in mount_table():
        root = posixpath.normpath(mount["path"])
        try:
            contained = posixpath.commonpath([normalized, root]) == root
        except ValueError:
            contained = False
        if contained:
            matches.append((len(root), mount["filesystem"]))
    return max(matches)[1] if matches else None


def _run(command, **kwargs):
    kwargs.setdefault("text", True)
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("check", False)
    return subprocess.run(command, **kwargs)


def _current_gid():
    getgid = getattr(os, "getgid", None)
    return int(getgid()) if getgid is not None else current_uid()


def _trusted_system_binary(name):
    """Resolve a fixed system executable safe to place after ``sudo --``."""
    _require_linux()
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
    system_uid = os.stat("/").st_uid
    getgroups = getattr(os, "getgroups", None)
    groups = set(getgroups() if getgroups is not None else ())
    groups.add(_current_gid())
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
            group_writable_by_us = bool(
                parent_info.st_mode & stat.S_IWGRP
                and parent_info.st_gid in groups
            )
            if (
                parent_info.st_uid != system_uid
                or parent_info.st_mode & stat.S_IWOTH
                or group_writable_by_us
                or (
                    current_euid() != 0
                    and os.access(parent, os.W_OK)
                )
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
    detail = (
        " (rejected writable candidates: %s)" % ", ".join(rejected)
        if rejected
        else ""
    )
    raise RamdiskError(
        "trusted %s executable was not found%s" % (name, detail)
    )


def _fresh_user_binary(name):
    """Resolve an unprivileged helper now, never from serialized manifest data."""
    path = shutil.which(name)
    if (
        not path
        or os.path.basename(path) != name
        or not os.access(path, os.X_OK)
    ):
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
def _noninteractive_privilege(
    keepalive=False,
    cancel_event=None,
    *,
    trusted_system_binary=None,
    sudo_ticket_keepalive=None,
):
    trusted_system_binary = (
        _trusted_system_binary
        if trusted_system_binary is None
        else trusted_system_binary
    )
    sudo_ticket_keepalive = (
        _sudo_ticket_keepalive
        if sudo_ticket_keepalive is None
        else sudo_ticket_keepalive
    )
    previous = getattr(_privilege_local, "noninteractive", False)
    _privilege_local.noninteractive = True
    keepalive_stop = None
    keepalive_thread = None
    try:
        if keepalive and not previous and current_euid() != 0:
            sudo = trusted_system_binary("sudo")
            keepalive_stop = threading.Event()
            keepalive_failure = threading.Event()
            keepalive_thread = threading.Thread(
                target=sudo_ticket_keepalive,
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


def _privileged(command, hardware, *, trusted_system_binary=None):
    del hardware
    if current_euid() == 0:
        return command
    trusted_system_binary = (
        _trusted_system_binary
        if trusted_system_binary is None
        else trusted_system_binary
    )
    sudo = trusted_system_binary("sudo")
    options = (
        ["-n"]
        if getattr(_privilege_local, "noninteractive", False)
        else []
    )
    return [sudo] + options + ["--"] + command


def _process_identity(pid):
    """Read one process identity from procfs without trusting its command."""
    _require_linux()
    getpgid = getattr(os, "getpgid", None)
    if getpgid is None:
        raise RamdiskError(UNSUPPORTED_PLATFORM_REASON)
    pid = int(pid)
    try:
        raw_stat = _read_text("/proc/%d/stat" % pid)
        close = raw_stat.rfind(")")
        fields = raw_stat[close + 2 :].split()
        starttime = int(fields[19])
        with open("/proc/%d/cmdline" % pid, "rb") as stream:
            cmdline = stream.read().split(b"\0")
        with open("/proc/%d/environ" % pid, "rb") as stream:
            environ = stream.read().split(b"\0")
        env = {}
        for item in environ:
            if b"=" in item:
                key, value = item.split(b"=", 1)
                env[key.decode("utf-8", "replace")] = value.decode(
                    "utf-8",
                    "replace",
                )
        return {
            "pid": pid,
            "uid": os.stat("/proc/%d" % pid).st_uid,
            "starttime": starttime,
            "cmdline": [
                value.decode("utf-8", "replace")
                for value in cmdline
                if value
            ],
            "nonce": env.get("COLI_MANAGED_NONCE"),
            "pgid": getpgid(pid),
        }
    except (OSError, ValueError, IndexError):
        return None


def _process_group_member_pids(pgid):
    """Return procfs PIDs whose stat record reports the requested PGID."""
    _require_linux()
    pgid = int(pgid)
    members = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return members
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
        if member_pgid == pgid:
            members.append(pid)
    return members


def _process_group_alive(pgid):
    _require_linux()
    killpg = getattr(os, "killpg", None)
    if killpg is None:
        raise RamdiskError(UNSUPPORTED_PLATFORM_REASON)
    try:
        killpg(int(pgid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_process_group(pgid, signum):
    _require_linux()
    killpg = getattr(os, "killpg", None)
    if killpg is None:
        raise RamdiskError(UNSUPPORTED_PLATFORM_REASON)
    killpg(int(pgid), signum)


def _process_status(pid, *, read_text=None):
    _require_linux()
    read_text = _read_text if read_text is None else read_text
    return read_text("/proc/%d/status" % int(pid))


def _busy_mount_references(path):
    """Return PIDs holding cwd, root, mappings, or descriptors below ``path``."""
    _require_linux()
    path = os.path.normpath(path) + os.sep
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        for leaf in ("cwd", "root"):
            try:
                target = os.path.realpath(
                    "/proc/%s/%s" % (entry, leaf)
                ) + os.sep
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
                target = os.path.realpath(
                    os.path.join(fd_dir, descriptor)
                ) + os.sep
                if target.startswith(path):
                    found.append(int(entry))
                    break
            except OSError:
                pass
    return sorted(set(found))


class LinuxPlatformOps:
    """Narrow Linux discovery operations with no import-time probes."""

    is_linux = True

    def __init__(self, platform_name="linux"):
        self.platform_name = platform_name

    def capabilities(self):
        return {
            "platform": self.platform_name,
            "hardware_discovery": True,
            "cgroup_memory": True,
            "numa": True,
            "ramdisk_lifecycle": True,
            "reason": None,
        }

    read_text = staticmethod(_read_text)
    status_allowed_list = staticmethod(_status_allowed_list)
    thread_sibling_groups = staticmethod(_thread_sibling_groups)
    meminfo = staticmethod(_meminfo)
    read_cgroup_value = staticmethod(_read_cgroup_value)
    read_cgroup_contract = staticmethod(_read_cgroup_contract)
    node_meminfo = staticmethod(_node_meminfo)
    physical_cores = staticmethod(_physical_cores)
    process_identity = staticmethod(_process_identity)
    process_group_member_pids = staticmethod(_process_group_member_pids)
    process_group_alive = staticmethod(_process_group_alive)
    signal_process_group = staticmethod(_signal_process_group)
    process_status = staticmethod(_process_status)
    busy_mount_references = staticmethod(_busy_mount_references)

    @staticmethod
    def path_exists(path):
        return os.path.exists(path)

    @staticmethod
    def cpu_affinity():
        get_affinity = getattr(os, "sched_getaffinity", None)
        if get_affinity is None:
            return None
        try:
            return sorted(int(cpu) for cpu in get_affinity(0))
        except OSError:
            return None

    @staticmethod
    def kernel_release():
        return platform.release()

    @staticmethod
    def kernel_at_least(major, minor):
        return _kernel_at_least(major, minor)

    @staticmethod
    def executable_path(name):
        return shutil.which(name)
