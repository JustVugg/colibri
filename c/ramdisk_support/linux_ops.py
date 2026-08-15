"""Linux filesystem and kernel operations used by RAM-disk discovery."""

from __future__ import print_function

import contextlib
import errno
import os
import platform
import posixpath
import re
import signal
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


def _read_proc_stat(path):
    """Read a task stat record without losing arbitrary ``comm`` bytes."""
    with open(
        path,
        "r",
        encoding="utf-8",
        errors="surrogateescape",
        newline="",
    ) as stream:
        return stream.read()


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
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as stream:
            mountinfo = stream.read()
    except (OSError, UnicodeError) as exc:
        raise RamdiskError(
            "cannot read Linux mount table %s: %s" % (path, exc)
        ) from exc
    result = []
    for line_number, line in enumerate(mountinfo.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            separator = fields.index("-")
            if separator < 6 or len(fields) <= separator + 3:
                raise ValueError("incomplete mountinfo record")
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
        except (ValueError, IndexError) as exc:
            raise RamdiskError(
                "cannot parse Linux mount table %s line %d: %s"
                % (path, line_number, exc)
            ) from exc
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
            matches.append((len(root), root, mount))
    if not matches:
        return None
    longest = max(item[0] for item in matches)
    nearest = [item for item in matches if item[0] == longest]
    if len(nearest) > 1:
        raise RamdiskError(
            "refusing ambiguous stacked mounts at %s (mount ids %s)"
            % (
                nearest[0][1],
                ", ".join(
                    str(item[2]["mount_id"])
                    for item in nearest
                ),
            )
        )
    return nearest[0][2]["filesystem"]


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
        child_info = info
        unsafe_parent = False
        while True:
            parent_info = os.stat(parent)
            sticky_protects_child = bool(
                parent_info.st_uid == system_uid
                and parent_info.st_mode & stat.S_ISVTX
                and child_info.st_uid == system_uid
                and not child_info.st_mode & 0o022
            )
            if (
                not stat.S_ISDIR(parent_info.st_mode)
                or parent_info.st_uid != system_uid
                or parent_info.st_mode & stat.S_IWOTH
                or (
                    parent_info.st_mode & stat.S_IWGRP
                    and not sticky_protects_child
                )
                or (
                    current_euid() != 0
                    and os.access(parent, os.W_OK)
                    and not sticky_protects_child
                )
            ):
                unsafe_parent = True
                break
            next_parent = os.path.dirname(parent)
            if next_parent == parent:
                break
            child_info = parent_info
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


def _process_start_boundary():
    """Return a conservative current process start time in boot ticks."""
    _require_linux()
    try:
        ticks_per_second = os.sysconf("SC_CLK_TCK")
    except (OSError, TypeError, ValueError) as exc:
        raise RamdiskError(
            "cannot read Linux process clock tick rate: %s" % exc
        ) from exc
    if (
        not isinstance(ticks_per_second, int)
        or isinstance(ticks_per_second, bool)
        or ticks_per_second <= 0
    ):
        raise RamdiskError("Linux process clock tick rate is invalid")

    uptime_path = "/proc/uptime"
    try:
        with open(
            uptime_path,
            "r",
            encoding="ascii",
            errors="strict",
            newline="",
        ) as stream:
            raw = stream.read()
    except (OSError, UnicodeError) as exc:
        raise RamdiskError(
            "cannot read Linux boot uptime %s: %s" % (uptime_path, exc)
        ) from exc

    fields = raw.split() if isinstance(raw, str) else []
    match = (
        re.fullmatch(r"([0-9]+)(?:\.([0-9]+))?", fields[0])
        if len(fields) == 2
        else None
    )
    if match is None:
        raise RamdiskError("cannot parse Linux boot uptime %s" % uptime_path)
    whole, fraction = match.groups()
    fraction = fraction or ""
    scale = 10 ** len(fraction)
    uptime_units = int(whole, 10) * scale + int(fraction or "0", 10)
    # Floor rather than round: a process starting in the current partially
    # observed tick remains at/after this boundary and therefore receives the
    # strict attribution checks.
    return uptime_units * ticks_per_second // scale


def _strict_process_identity(pid):
    """Read one complete process identity or report why it is unreadable."""
    _require_linux()
    getpgid = getattr(os, "getpgid", None)
    if getpgid is None:
        raise RamdiskError(UNSUPPORTED_PLATFORM_REASON)
    pid = int(pid)
    proc_path = "/proc/%d" % pid
    stat_path = "%s/stat" % proc_path
    try:
        before_uid = os.stat(proc_path).st_uid
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise RamdiskError(
            "cannot read Linux process owner %s: %s; verified cleanup "
            "requires complete process-table visibility" % (proc_path, exc)
        ) from exc
    before = _strict_proc_stat_identity(pid, stat_path)
    if before is None:
        return None
    if before.get("state") in _INERT_PROCESS_STATES:
        inert = _stable_inert_process_identity(
            pid,
            proc_path,
            before_uid,
            before,
        )
        if inert is None:
            return None
        return {
            "pid": pid,
            "uid": before_uid,
            "state": inert["state"],
            "inert": True,
            "starttime": inert["starttime"],
            "cmdline": [],
            "nonce": None,
            "pgid": inert["pgid"],
            "sid": inert["sid"],
            "state_dir": None,
            "weights_dir": None,
        }
    cmdline_path = "%s/cmdline" % proc_path
    cmdline_raw = _read_proc_binary(pid, cmdline_path)
    if cmdline_raw is None:
        return None
    environ_path = "%s/environ" % proc_path
    environ = _read_proc_binary(pid, environ_path)
    if environ is None:
        return None
    env = _managed_environment_fields(pid, environ_path, environ)
    try:
        observed_pgid = getpgid(pid)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise RamdiskError(
            "cannot read Linux process group for PID %d: %s; verified "
            "cleanup requires complete process-table visibility" % (pid, exc)
        ) from exc
    after = _strict_proc_stat_identity(pid, stat_path)
    try:
        after_uid = os.stat(proc_path).st_uid
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise RamdiskError(
            "cannot recheck Linux process owner %s: %s; verified cleanup "
            "requires complete process-table visibility" % (proc_path, exc)
        ) from exc
    if (
        after is None
        or before != after
        or before_uid != after_uid
        or observed_pgid != after["pgid"]
    ):
        return None
    return {
        "pid": pid,
        "uid": after_uid,
        "state": after["state"],
        "inert": False,
        "starttime": after["starttime"],
        "cmdline": [
            value.decode("utf-8", "replace")
            for value in cmdline_raw.split(b"\0")
            if value
        ],
        "nonce": env["COLI_MANAGED_NONCE"],
        "pgid": after["pgid"],
        "sid": after["sid"],
        "state_dir": env["COLI_STATE_DIR"],
        "weights_dir": env["COLI_WEIGHTS_DIR"],
    }


def _process_identity(pid):
    """Best-effort identity read used by non-destructive status paths."""
    try:
        return _strict_process_identity(pid)
    except (OSError, ValueError, IndexError, RamdiskError):
        return None


def _proc_pid_snapshot():
    """Return one unambiguous snapshot of every numeric procfs PID entry."""
    try:
        entries = os.listdir("/proc")
    except OSError as exc:
        raise RamdiskError(
            "cannot enumerate Linux process table /proc: %s; pending-launch "
            "recovery requires complete process-table visibility" % exc
        ) from exc

    snapshot = {}
    for entry in entries:
        if not isinstance(entry, str) or re.fullmatch(r"[0-9]+", entry) is None:
            continue
        pid = int(entry)
        if pid <= 0 or entry != str(pid) or pid in snapshot:
            raise RamdiskError(
                "cannot trust ambiguous Linux process-table entry %r; "
                "pending-launch recovery requires complete process-table "
                "visibility" % entry
            )
        snapshot[pid] = "/proc/%s" % entry
    return snapshot


def _proc_pid_disappeared(pid, endpoint, missing_error):
    """Accept endpoint ENOENT only when the corresponding PID is now absent."""
    proc_path = "/proc/%d" % pid
    try:
        os.stat(proc_path)
    except (FileNotFoundError, ProcessLookupError):
        return True
    except OSError as exc:
        raise RamdiskError(
            "cannot verify Linux process identity %s after %s disappeared: "
            "%s; pending-launch recovery requires complete process-table "
            "visibility" % (proc_path, endpoint, exc)
        ) from exc
    raise RamdiskError(
        "cannot read Linux process identity %s while PID %d remains; "
        "pending-launch recovery requires complete process-table visibility"
        % (endpoint, pid)
    ) from missing_error


def _strict_proc_stat_identity(pid, path):
    """Read stable process-group fields used by nonce-based recovery."""
    try:
        raw = _read_proc_stat(path)
    except (FileNotFoundError, ProcessLookupError) as exc:
        if _proc_pid_disappeared(pid, path, exc):
            return None
    except (OSError, UnicodeError) as exc:
        raise RamdiskError(
            "cannot read Linux process identity %s: %s; pending-launch "
            "recovery requires complete process-table visibility" % (path, exc)
        ) from exc

    opening = raw.find("(") if isinstance(raw, str) else -1
    closing = raw.rfind(")") if isinstance(raw, str) else -1
    fields = raw[closing + 2 :].split() if closing >= 0 else []
    try:
        declared_pid = int(raw[:opening].strip())
        state = fields[0]
        pgid = int(fields[2], 10)
        session = int(fields[3], 10)
        num_threads = int(fields[17], 10)
        starttime = int(fields[19], 10)
    except (IndexError, TypeError, ValueError) as exc:
        raise RamdiskError(
            "cannot parse Linux process identity %s; pending-launch recovery "
            "requires complete process-table visibility" % path
        ) from exc
    if (
        opening <= 0
        or closing <= opening
        or declared_pid != pid
        or len(state) != 1
        or pgid < 0
        or session < 0
        or num_threads <= 0
        or starttime < 0
    ):
        raise RamdiskError(
            "cannot parse Linux process identity %s; pending-launch recovery "
            "requires complete process-table visibility" % path
        )
    return {
        "state": state,
        "starttime": starttime,
        "pgid": pgid,
        "sid": session,
        "num_threads": num_threads,
    }


def _read_proc_binary(pid, path):
    try:
        with open(path, "rb") as stream:
            value = stream.read()
    except (FileNotFoundError, ProcessLookupError) as exc:
        if _proc_pid_disappeared(pid, path, exc):
            return None
    except OSError as exc:
        raise RamdiskError(
            "cannot read Linux process identity %s: %s; pending-launch "
            "recovery requires complete process-table visibility" % (path, exc)
        ) from exc
    if not isinstance(value, bytes):
        raise RamdiskError(
            "cannot parse Linux process identity %s; pending-launch recovery "
            "requires complete process-table visibility" % path
        )
    return value


def _managed_environment_fields(pid, path, raw):
    """Extract launch attribution without rejecting unrelated environment data."""
    fields = {
        b"COLI_MANAGED_NONCE": [],
        b"COLI_STATE_DIR": [],
        b"COLI_WEIGHTS_DIR": [],
    }
    malformed = set()
    for item in raw.split(b"\0"):
        if not item:
            continue
        if b"=" not in item:
            if item in fields:
                malformed.add(item)
            # Linux permits arbitrary strings in ``environ``.  Firefox and
            # other ordinary processes use entries without ``=``; those say
            # nothing about a Colibri launch and must not wedge recovery.
            continue
        key, value = item.split(b"=", 1)
        if key in fields:
            fields[key].append(value)

    result = {}
    candidates = {}
    ambiguous = set()
    for key, values in fields.items():
        label = key.decode("ascii")
        decoded = []
        invalid_value = False
        for value in values:
            try:
                decoded.append(value.decode("utf-8", "strict"))
            except UnicodeError:
                invalid_value = True
        candidates[label] = tuple(decoded)
        if key in malformed or len(values) != 1 or invalid_value:
            ambiguous.add(label)
            result[label] = None
        else:
            result[label] = decoded[0]
    result["_candidates"] = candidates
    result["_ambiguous"] = tuple(sorted(ambiguous))
    return result


def _recheck_managed_launch_identity(pid, proc_path, expected_uid, before):
    """Recheck one owner/stat pair before trusting its age or contents."""
    stat_path = "%s/stat" % proc_path
    after = _strict_proc_stat_identity(pid, stat_path)
    if after is None:
        return None
    try:
        after_uid = os.stat(proc_path).st_uid
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise RamdiskError(
            "cannot recheck Linux process owner %s: %s; pending-launch "
            "recovery requires complete process-table visibility"
            % (proc_path, exc)
        ) from exc
    if after_uid != expected_uid or after != before:
        raise RamdiskError(
            "Linux process identity %s changed during pending-launch "
            "recovery; refusing an unstable process-table view" % proc_path
        )
    return after


_INERT_PROCESS_STATES = frozenset(("Z", "X", "x"))


def _stable_inert_process_identity(pid, proc_path, expected_uid, before):
    """Prove a dead process identity has no live sibling task."""
    if before.get("state") not in _INERT_PROCESS_STATES:
        raise RamdiskError(
            "Linux process identity %s is not an inert dead task" % proc_path
        )
    task_path = "%s/task" % proc_path
    try:
        entries = os.listdir(task_path)
    except (FileNotFoundError, ProcessLookupError):
        after = _recheck_managed_launch_identity(
            pid,
            proc_path,
            expected_uid,
            before,
        )
        if after is None:
            return None
        raise RamdiskError(
            "cannot enumerate Linux dead process task group %s while PID %d "
            "remains" % (task_path, pid)
        )
    except OSError as exc:
        raise RamdiskError(
            "cannot enumerate Linux dead process task group %s: %s; "
            "pending-launch recovery requires complete process-table visibility"
            % (task_path, exc)
        ) from exc
    if any(not entry.isdigit() for entry in entries):
        raise RamdiskError(
            "cannot parse Linux dead process task group %s" % task_path
        )
    task_ids = [int(entry) for entry in entries]
    if (
        before.get("num_threads") != 1
        or len(task_ids) != 1
        or set(task_ids) != {pid}
    ):
        raise RamdiskError(
            "Linux dead process PID %d still has an incomplete or live task "
            "group; refusing to treat it as inert" % pid
        )
    return _recheck_managed_launch_identity(
        pid,
        proc_path,
        expected_uid,
        before,
    )


def _inspect_managed_launch_pid(pid, expected_uid, not_before_starttime):
    """Return a stable same-UID PID observation, or ``None`` if it exited."""
    proc_path = "/proc/%d" % pid
    try:
        owner = os.stat(proc_path)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise RamdiskError(
            "cannot read Linux process owner %s: %s; pending-launch recovery "
            "requires complete process-table visibility" % (proc_path, exc)
        ) from exc
    owner_uid = getattr(owner, "st_uid", None)
    if (
        not isinstance(owner_uid, int)
        or isinstance(owner_uid, bool)
        or owner_uid < 0
    ):
        raise RamdiskError(
            "cannot parse Linux process owner %s; pending-launch recovery "
            "requires complete process-table visibility" % proc_path
        )
    if owner_uid != expected_uid:
        return {"kind": "foreign", "uid": owner_uid}

    stat_path = "%s/stat" % proc_path
    before = _strict_proc_stat_identity(pid, stat_path)
    if before is None:
        return None
    if before["starttime"] < not_before_starttime:
        stable_old = _recheck_managed_launch_identity(
            pid,
            proc_path,
            expected_uid,
            before,
        )
        if stable_old is None:
            return None
        return {
            "kind": "before-boundary",
            "pid": pid,
            "uid": expected_uid,
            "starttime": stable_old["starttime"],
            "pgid": stable_old["pgid"],
            "sid": stable_old["sid"],
        }
    if before.get("state") in _INERT_PROCESS_STATES:
        inert = _stable_inert_process_identity(
            pid,
            proc_path,
            expected_uid,
            before,
        )
        if inert is None:
            return None
        return {
            "kind": "inert-dead",
            "pid": pid,
            "uid": expected_uid,
            "state": inert["state"],
            "starttime": inert["starttime"],
            "pgid": inert["pgid"],
            "sid": inert["sid"],
        }
    cmdline_path = "%s/cmdline" % proc_path
    cmdline_raw = _read_proc_binary(pid, cmdline_path)
    if cmdline_raw is None:
        return None
    environ_path = "%s/environ" % proc_path
    environ_raw = _read_proc_binary(pid, environ_path)
    if environ_raw is None:
        return None
    environment = _managed_environment_fields(pid, environ_path, environ_raw)
    after = _recheck_managed_launch_identity(
        pid,
        proc_path,
        expected_uid,
        before,
    )
    if after is None:
        return None
    return {
        "kind": "same-uid",
        "pid": pid,
        "uid": expected_uid,
        "starttime": after["starttime"],
        "nonce": environment["COLI_MANAGED_NONCE"],
        "pgid": after["pgid"],
        "sid": after["sid"],
        "cmdline": [
            item.decode("utf-8", "replace")
            for item in cmdline_raw.split(b"\0")
            if item
        ],
        "state_dir": environment["COLI_STATE_DIR"],
        "weights_dir": environment["COLI_WEIGHTS_DIR"],
        "environment_candidates": environment["_candidates"],
        "environment_ambiguities": environment["_ambiguous"],
    }


def _managed_launch_processes(
    nonce,
    uid,
    *,
    state_dir,
    weights_dir,
    not_before_starttime,
    launcher_pid,
    launcher_starttime,
    launcher_cmdline,
    expected_command,
):
    """Discover every stable process attributable to one pending launch.

    Foreign-UID processes are skipped after a trustworthy procfs ownership
    read. Same-UID identities are read repeatedly across independent
    process-table snapshots so PID reuse, exec transitions, and new uninspected
    PIDs fail closed instead of turning an incomplete scan into a false absence
    proof.
    """
    _require_linux()
    if not isinstance(nonce, str) or not nonce or "\0" in nonce:
        raise RamdiskError("managed launch nonce must be a nonempty string")
    if not isinstance(uid, int) or isinstance(uid, bool) or uid < 0:
        raise RamdiskError("managed launch UID must be a nonnegative integer")
    if (
        not isinstance(not_before_starttime, int)
        or isinstance(not_before_starttime, bool)
        or not_before_starttime < 0
    ):
        raise RamdiskError(
            "managed launch process-start boundary must be a nonnegative "
            "integer"
        )
    if (
        not isinstance(launcher_pid, int)
        or isinstance(launcher_pid, bool)
        or launcher_pid <= 0
    ):
        raise RamdiskError(
            "managed launch launcher PID must be a positive integer"
        )
    if (
        not isinstance(launcher_starttime, int)
        or isinstance(launcher_starttime, bool)
        or launcher_starttime <= 0
    ):
        raise RamdiskError(
            "managed launch launcher start time must be a positive integer"
        )
    for label, value in (
        ("state directory", state_dir),
        ("weights directory", weights_dir),
    ):
        if not isinstance(value, str) or not value or "\0" in value:
            raise RamdiskError("managed launch %s must be a nonempty string" % label)
    for label, value in (
        ("launcher command", launcher_cmdline),
        ("expected command", expected_command),
    ):
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise RamdiskError(
                "managed launch %s must be a nonempty string list" % label
            )

    initial = _proc_pid_snapshot()
    first = {}
    for pid in sorted(initial):
        observation = _inspect_managed_launch_pid(
            pid,
            uid,
            not_before_starttime,
        )
        if observation is not None:
            first[pid] = observation

    middle = _proc_pid_snapshot()
    unexpected = sorted(set(middle) - set(first))
    if unexpected:
        raise RamdiskError(
            "Linux process table changed during pending-launch recovery; "
            "uninspected PID(s): %s" % ", ".join(str(pid) for pid in unexpected)
        )

    stable = {}
    for pid in sorted(set(first) & set(middle)):
        observation = _inspect_managed_launch_pid(
            pid,
            uid,
            not_before_starttime,
        )
        if observation is None:
            continue
        if observation != first[pid]:
            raise RamdiskError(
                "Linux process identity /proc/%d changed during "
                "pending-launch recovery; refusing an unstable process-table "
                "view" % pid
            )
        stable[pid] = observation

    settled = _proc_pid_snapshot()
    unexpected = sorted(set(settled) - set(stable))
    if unexpected:
        raise RamdiskError(
            "Linux process table changed during pending-launch recovery; "
            "uninspected PID(s): %s" % ", ".join(str(pid) for pid in unexpected)
        )

    final = {}
    for pid in sorted(set(stable) & set(settled)):
        observation = _inspect_managed_launch_pid(
            pid,
            uid,
            not_before_starttime,
        )
        if observation is None:
            continue
        if observation != stable[pid]:
            raise RamdiskError(
                "Linux process identity /proc/%d changed during final "
                "pending-launch recovery verification" % pid
            )
        final[pid] = observation

    verified = _proc_pid_snapshot()
    unexpected = sorted(set(verified) - set(final))
    if unexpected:
        raise RamdiskError(
            "Linux process table changed after final pending-launch reads; "
            "unverified PID(s): %s" % ", ".join(str(pid) for pid in unexpected)
        )

    checked = {}
    for pid in sorted(set(final) & set(verified)):
        observation = _inspect_managed_launch_pid(
            pid,
            uid,
            not_before_starttime,
        )
        if observation is None:
            continue
        if observation != final[pid]:
            raise RamdiskError(
                "Linux process identity /proc/%d changed after the final "
                "pending-launch snapshot" % pid
            )
        checked[pid] = observation

    confirmation_snapshot = _proc_pid_snapshot()
    unexpected = sorted(set(confirmation_snapshot) - set(checked))
    if unexpected:
        raise RamdiskError(
            "Linux process table changed during final pending-launch identity "
            "confirmation; unverified PID(s): %s"
            % ", ".join(str(pid) for pid in unexpected)
        )

    # The terminal confirmation must carry identities, not only PID names.
    # Otherwise a process can exit and the same numeric PID can be reused after
    # ``checked`` without changing the final key set.
    confirmed = {}
    for pid in sorted(set(checked) & set(confirmation_snapshot)):
        observation = _inspect_managed_launch_pid(
            pid,
            uid,
            not_before_starttime,
        )
        if observation is None:
            continue
        if observation != checked[pid]:
            raise RamdiskError(
                "Linux process identity /proc/%d changed during final "
                "pending-launch identity confirmation" % pid
            )
        confirmed[pid] = observation

    matches = []
    for pid in sorted(confirmed):
        observation = confirmed[pid]
        if observation.get("kind") in {
            "foreign",
            "before-boundary",
            "inert-dead",
        }:
            continue
        if (
            pid == launcher_pid
            and observation.get("starttime") == launcher_starttime
        ):
            if observation.get("cmdline") != launcher_cmdline:
                raise RamdiskError(
                    "managed launch launcher identity changed command for PID %d"
                    % pid
                )
            continue
        candidates = observation.get("environment_candidates") or {}
        target_nonce_present = (
            nonce in candidates.get("COLI_MANAGED_NONCE", ())
        )
        target_path_present = (
            state_dir in candidates.get("COLI_STATE_DIR", ())
            or weights_dir in candidates.get("COLI_WEIGHTS_DIR", ())
        )
        if pid == os.getpid() and not (
            target_nonce_present or target_path_present
        ):
            # The process executing recovery cannot also be an orphaned child
            # from the earlier Popen attempt.  This narrowly excludes a new
            # controller process whose argv matches the persisted launcher.
            continue
        associated = (
            observation.get("cmdline") in (launcher_cmdline, expected_command)
            or target_nonce_present
            or target_path_present
        )
        if not associated:
            continue
        if observation.get("environment_ambiguities"):
            raise RamdiskError(
                "same-UID PID %d has ambiguous managed launch attribution"
                % pid
            )
        if observation.get("nonce") != nonce:
            raise RamdiskError(
                "same-UID PID %d has missing or mismatched nonce attribution"
                % pid
            )
        actual_state_dir = observation.get("state_dir")
        actual_weights_dir = observation.get("weights_dir")
        if (
            not actual_state_dir
            or not actual_weights_dir
            or actual_state_dir != state_dir
            or actual_weights_dir != weights_dir
        ):
            raise RamdiskError(
                "managed launch PID %d has mismatched state or weights "
                "attribution" % pid
            )
        if (
            observation["starttime"] <= 0
            or observation["pgid"] <= 0
            or observation["sid"] <= 0
            or observation["pgid"] != observation["sid"]
        ):
            raise RamdiskError(
                "managed launch PID %d violates the new-session process-group "
                "identity" % pid
            )
        matches.append(
            {
                key: observation[key]
                for key in (
                    "pid",
                    "uid",
                    "starttime",
                    "nonce",
                    "pgid",
                    "sid",
                    "cmdline",
                    "state_dir",
                    "weights_dir",
                )
            }
        )
    return matches


def _process_group_member_pids(pgid):
    """Return procfs PIDs whose stat record reports the requested PGID."""
    _require_linux()
    pgid = int(pgid)
    members = []
    try:
        entries = os.listdir("/proc")
    except OSError as exc:
        raise RamdiskError(
            "cannot enumerate Linux process table /proc: %s; managed "
            "cleanup requires complete process-table visibility" % exc
        ) from exc
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        stat_path = "/proc/%d/stat" % pid
        try:
            raw = _read_proc_stat(stat_path)
        except (FileNotFoundError, ProcessLookupError):
            # Exiting between listdir() and open() is ordinary procfs churn.
            continue
        except (OSError, UnicodeError) as exc:
            raise RamdiskError(
                "cannot read Linux process identity %s: %s"
                "; managed cleanup requires complete process-table visibility"
                % (stat_path, exc)
            ) from exc
        close = raw.rfind(")")
        try:
            if close < 0:
                raise ValueError("missing process-name terminator")
            fields = raw[close + 2 :].split()
            member_pgid = int(fields[2])
        except (ValueError, IndexError) as exc:
            raise RamdiskError(
                "cannot parse Linux process identity %s: %s"
                % (stat_path, exc)
            ) from exc
        if member_pgid == pgid:
            members.append(pid)
    return sorted(members)


def _process_group_alive(pgid):
    _require_linux()
    killpg = getattr(os, "killpg", None)
    if killpg is None:
        raise RamdiskError(UNSUPPORTED_PLATFORM_REASON)
    try:
        killpg(int(pgid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass

    # ``killpg(..., 0)`` reports a group containing only zombies as alive.
    # Such tasks cannot run, retain files, or mutate usage. Prove every member
    # is a stable lone-thread dead identity before treating that group as inert.
    member_pids = _process_group_member_pids(pgid)
    if not member_pids:
        try:
            killpg(int(pgid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    inert = {}
    for pid in member_pids:
        proc_path = "/proc/%d" % pid
        try:
            owner_uid = os.stat(proc_path).st_uid
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as exc:
            raise RamdiskError(
                "cannot read Linux process owner %s while checking process "
                "group liveness: %s" % (proc_path, exc)
            ) from exc
        identity = _strict_proc_stat_identity(
            pid,
            "%s/stat" % proc_path,
        )
        if identity is None:
            continue
        if identity.get("state") not in _INERT_PROCESS_STATES:
            return True
        stable = _stable_inert_process_identity(
            pid,
            proc_path,
            owner_uid,
            identity,
        )
        if stable is not None:
            inert[pid] = (owner_uid, stable)

    settled = _process_group_member_pids(pgid)
    if set(settled) - set(inert):
        return True
    for pid in sorted(set(settled) & set(inert)):
        proc_path = "/proc/%d" % pid
        try:
            owner_uid = os.stat(proc_path).st_uid
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as exc:
            raise RamdiskError(
                "cannot recheck Linux process owner %s while checking process "
                "group liveness: %s" % (proc_path, exc)
            ) from exc
        identity = _strict_proc_stat_identity(
            pid,
            "%s/stat" % proc_path,
        )
        if identity is None:
            continue
        if (owner_uid, identity) != inert[pid]:
            return True
    return False


_PIDFD_REQUIRED_REASON = (
    "verified managed-process cleanup requires Linux pidfd_open and "
    "pidfd_send_signal support; use a newer Python runtime or a libc/kernel "
    "that provides both pidfd operations"
)


def _pidfd_api(pidfd_open=None, pidfd_send_signal=None):
    """Resolve pidfd operations lazily, preferring the Python stdlib."""
    open_operation = (
        getattr(os, "pidfd_open", None)
        if pidfd_open is None
        else pidfd_open
    )
    send_operation = (
        getattr(signal, "pidfd_send_signal", None)
        if pidfd_send_signal is None
        else pidfd_send_signal
    )
    if callable(open_operation) and callable(send_operation):
        return open_operation, send_operation

    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
    except (ImportError, OSError):
        libc = None

    if not callable(open_operation) and libc is not None:
        libc_open = getattr(libc, "pidfd_open", None)
        if libc_open is not None:
            libc_open.argtypes = (ctypes.c_int, ctypes.c_uint)
            libc_open.restype = ctypes.c_int

            def open_operation(pid, flags):
                result = libc_open(int(pid), int(flags))
                if result < 0:
                    error_number = ctypes.get_errno()
                    raise OSError(
                        error_number,
                        os.strerror(error_number),
                    )
                return result

    if not callable(send_operation) and libc is not None:
        libc_send = getattr(libc, "pidfd_send_signal", None)
        if libc_send is not None:
            libc_send.argtypes = (
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint,
            )
            libc_send.restype = ctypes.c_int

            def send_operation(pidfd, signum, siginfo, flags):
                if siginfo is not None:
                    raise ValueError(
                        "verified cleanup does not accept siginfo payloads"
                    )
                result = libc_send(
                    int(pidfd),
                    int(signum),
                    None,
                    int(flags),
                )
                if result < 0:
                    error_number = ctypes.get_errno()
                    raise OSError(
                        error_number,
                        os.strerror(error_number),
                    )

    if not callable(open_operation) or not callable(send_operation):
        raise RamdiskError(_PIDFD_REQUIRED_REASON)
    return open_operation, send_operation


def _pidfd_process_control_supported():
    """Probe that this runtime and kernel can bind and signal a pidfd."""
    descriptor = None
    try:
        open_operation, send_operation = _pidfd_api()
        descriptor = open_operation(os.getpid(), 0)
        send_operation(descriptor, 0, None, 0)
        return True
    except (OSError, RamdiskError, TypeError, ValueError):
        return False
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _pidfd_exited(pidfd):
    """Return whether one pidfd has become readable because its task exited."""
    try:
        import select

        readable, _, _ = select.select([int(pidfd)], [], [], 0)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise RamdiskError(
            "cannot poll a pinned Linux process identity: %s; verified "
            "cleanup requires working pidfd polling" % exc
        ) from exc
    return bool(readable)


def _verified_group_member_mismatch(record, identity, pid, expected_pgid):
    """Return the persisted-attribution mismatch for one pinned task."""
    if not isinstance(identity, dict) or identity.get("pid") != pid:
        return "unreadable-process-identity"
    if (
        not isinstance(identity.get("starttime"), int)
        or isinstance(identity.get("starttime"), bool)
        or identity["starttime"] <= 0
    ):
        return "unreadable-process-identity"
    if identity.get("uid") != record.get("uid"):
        return "foreign-uid"
    if pid == int(record["pid"]) and (
        identity.get("starttime") != record.get("starttime")
    ):
        return "reused-pid"
    if identity.get("pgid") != expected_pgid:
        return "foreign-process-group"
    if identity.get("sid") != expected_pgid:
        return "foreign-session"
    if identity.get("inert") is True:
        return None
    if identity.get("inert") is not False:
        return "unreadable-process-identity"
    if identity.get("nonce") != record.get("nonce"):
        return "foreign-nonce"
    if identity.get("state_dir") != record.get("state_dir"):
        return "foreign-state-directory"
    if identity.get("weights_dir") != record.get("weights_dir"):
        return "foreign-weights-directory"
    return None


def _signal_verified_process_group(
    record,
    signum,
    *,
    process_group_member_pids=None,
    process_identity=None,
    process_group_alive=None,
    pidfd_open=None,
    pidfd_send_signal=None,
    pidfd_exited=None,
    close_fd=None,
):
    """Pin, validate, then signal each exact member without numeric PGID use.

    The result status is one of ``signaled``, ``absent``, ``inconclusive``,
    or ``foreign``.  Inconclusive membership churn is deliberately distinct
    from absence so callers can retry it only inside a bounded deadline.
    """
    _require_linux()
    if not isinstance(record, dict):
        raise RamdiskError("verified cleanup requires a process record")
    try:
        expected_pid = int(record["pid"])
        expected_pgid = int(record.get("pgid", expected_pid))
    except (KeyError, TypeError, ValueError) as exc:
        raise RamdiskError(
            "verified cleanup process identity is incomplete"
        ) from exc
    if expected_pid <= 0 or expected_pgid <= 0:
        raise RamdiskError("verified cleanup process identity is invalid")

    member_pids = (
        _process_group_member_pids
        if process_group_member_pids is None
        else process_group_member_pids
    )
    identity_reader = (
        _strict_process_identity
        if process_identity is None
        else process_identity
    )
    group_alive = (
        _process_group_alive
        if process_group_alive is None
        else process_group_alive
    )
    poll_exited = _pidfd_exited if pidfd_exited is None else pidfd_exited
    close_operation = os.close if close_fd is None else close_fd

    def inconclusive(reason, members=None):
        return {
            "status": "inconclusive",
            "reason": reason,
            "members": [] if members is None else list(members),
        }

    try:
        initial = list(member_pids(expected_pgid))
    except (OSError, RamdiskError) as exc:
        return inconclusive("process-group-enumeration-failed: %s" % exc)
    if initial != sorted(set(initial)) or any(pid <= 0 for pid in initial):
        return {
            "status": "inconclusive",
            "reason": "ambiguous-membership",
            "members": initial,
        }
    if not initial:
        try:
            alive_before = group_alive(expected_pgid)
            confirmed = list(member_pids(expected_pgid))
            alive_after = group_alive(expected_pgid)
        except (OSError, RamdiskError) as exc:
            return inconclusive(
                "empty-group-confirmation-failed: %s" % exc
            )
        if not confirmed and not alive_before and not alive_after:
            return {"status": "absent", "members": []}
        return {
            "status": "inconclusive",
            "reason": "empty-live-process-group",
            "members": confirmed,
        }

    open_operation, send_operation = _pidfd_api(
        pidfd_open=pidfd_open,
        pidfd_send_signal=pidfd_send_signal,
    )

    pidfds = {}
    readiness_before = {}
    identities = {}
    try:
        # Bind the entire candidate set before reading any numeric PID state.
        # A later PID/PGID reuse can therefore never redirect a signal.
        for pid in initial:
            try:
                pidfds[pid] = open_operation(pid, 0)
            except ProcessLookupError:
                return {
                    "status": "inconclusive",
                    "reason": "member-exited-before-pidfd-open",
                    "members": initial,
                }
            except OSError as exc:
                if exc.errno == errno.ESRCH:
                    return {
                        "status": "inconclusive",
                        "reason": "member-exited-before-pidfd-open",
                        "members": initial,
                    }
                if exc.errno in (errno.ENOSYS, errno.EINVAL):
                    raise RamdiskError(
                        "%s (pidfd_open failed: %s)"
                        % (_PIDFD_REQUIRED_REASON, exc)
                    ) from exc
                raise RamdiskError(
                    "cannot pin Linux PID %d for verified cleanup: %s"
                    % (pid, exc)
                ) from exc

        try:
            readiness_before = {
                pid: bool(poll_exited(pidfd))
                for pid, pidfd in pidfds.items()
            }
        except (OSError, RamdiskError) as exc:
            return inconclusive(
                "pidfd-poll-failed: %s" % exc,
                initial,
            )
        for pid in initial:
            try:
                identity = identity_reader(pid)
            except (OSError, RamdiskError) as exc:
                return inconclusive(
                    "process-identity-read-failed for PID %d: %s"
                    % (pid, exc),
                    initial,
                )
            if identity is None:
                return {
                    "status": "inconclusive",
                    "reason": "unreadable-process-identity",
                    "members": initial,
                }
            identities[pid] = identity

        try:
            readiness_after_identity = {
                pid: bool(poll_exited(pidfd))
                for pid, pidfd in pidfds.items()
            }
            settled = list(member_pids(expected_pgid))
        except (OSError, RamdiskError) as exc:
            return inconclusive(
                "process-group-revalidation-failed: %s" % exc,
                initial,
            )
        if settled != initial:
            return {
                "status": "inconclusive",
                "reason": "membership-changed",
                "members": sorted(set(initial) | set(settled)),
            }

        for pid in initial:
            mismatch = _verified_group_member_mismatch(
                record,
                identities[pid],
                pid,
                expected_pgid,
            )
            if mismatch is not None:
                return {
                    "status": "foreign",
                    "reason": mismatch,
                    "members": initial,
                }
            if identities[pid].get("inert") is not True and (
                readiness_before[pid] or readiness_after_identity[pid]
            ):
                return {
                    "status": "inconclusive",
                    "reason": "pinned-member-exited-during-validation",
                    "members": initial,
                }

        try:
            confirmed = list(member_pids(expected_pgid))
        except (OSError, RamdiskError) as exc:
            return inconclusive(
                "process-group-confirmation-failed: %s" % exc,
                initial,
            )
        if confirmed != initial:
            return {
                "status": "inconclusive",
                "reason": "membership-changed",
                "members": sorted(set(initial) | set(confirmed)),
            }
        try:
            readiness_before_signal = {
                pid: bool(poll_exited(pidfd))
                for pid, pidfd in pidfds.items()
            }
        except (OSError, RamdiskError) as exc:
            return inconclusive(
                "final-pidfd-poll-failed: %s" % exc,
                initial,
            )
        if any(
            identities[pid].get("inert") is not True
            and readiness_before_signal[pid]
            for pid in initial
        ):
            return {
                "status": "inconclusive",
                "reason": "pinned-member-exited-before-signal",
                "members": initial,
            }

        live = [
            pid
            for pid in initial
            if identities[pid].get("inert") is not True
        ]
        if not live:
            try:
                alive_before = group_alive(expected_pgid)
                inert_confirmation = list(member_pids(expected_pgid))
                alive_after = group_alive(expected_pgid)
            except (OSError, RamdiskError) as exc:
                return inconclusive(
                    "inert-group-confirmation-failed: %s" % exc,
                    initial,
                )
            if (
                inert_confirmation == initial
                and not alive_before
                and not alive_after
            ):
                return {"status": "absent", "members": initial}
            return {
                "status": "inconclusive",
                "reason": "inert-membership-not-stable",
                "members": sorted(set(initial) | set(inert_confirmation)),
            }

        signaled = []
        exited = []
        for pid in live:
            try:
                send_operation(pidfds[pid], signum, None, 0)
                signaled.append(pid)
            except ProcessLookupError:
                exited.append(pid)
            except OSError as exc:
                if exc.errno == errno.ESRCH:
                    exited.append(pid)
                    continue
                raise RamdiskError(
                    "pidfd signal %s failed for verified PID %d after "
                    "signaling PID(s) %s: %s; cleanup authority remains "
                    "persisted and numeric fallback is forbidden"
                    % (
                        signum,
                        pid,
                        ", ".join(str(value) for value in signaled) or "none",
                        exc,
                    )
                ) from exc
        return {
            "status": "signaled",
            "members": initial,
            "signaled": signaled,
            "exited": exited,
        }
    finally:
        for pidfd in pidfds.values():
            try:
                close_operation(pidfd)
            except OSError:
                pass


def _process_status(pid, *, read_text=None):
    _require_linux()
    read_text = _read_text if read_text is None else read_text
    return read_text("/proc/%d/status" % int(pid))


def _busy_mount_references_proc(path):
    """Strict root-only procfs scan for references below ``path``."""
    _require_linux()
    path = os.path.normpath(path) + os.sep
    found = []

    def visibility_error(action, proc_path, error):
        raise RamdiskError(
            "cannot %s %s: %s; managed cleanup requires complete /proc "
            "visibility (hidepid or a security policy may deny it)"
            % (action, proc_path, error)
        ) from error

    def reference_below(target):
        if target.endswith(" (deleted)"):
            target = target[: -len(" (deleted)")]
        if not os.path.isabs(target):
            return False
        return (os.path.normpath(target) + os.sep).startswith(path)

    def missing_endpoint_is_inert(entry, endpoint, missing_error):
        """Corroborate endpoint ENOENT without overlooking a live task."""
        stat_path = "/proc/%s/stat" % entry
        try:
            process_stat = _read_proc_stat(stat_path)
        except (FileNotFoundError, ProcessLookupError):
            # The PID itself is now absent, so it cannot retain the mount.
            return True
        except (OSError, UnicodeError) as exc:
            visibility_error("verify process identity", stat_path, exc)

        close = process_stat.rfind(")")
        fields = process_stat[close + 2 :].split() if close >= 0 else []
        try:
            state = fields[0]
            flags = int(fields[6], 10)
        except (IndexError, TypeError, ValueError) as exc:
            raise RamdiskError(
                "cannot parse process identity %s after missing endpoint %s; "
                "managed cleanup requires complete /proc visibility"
                % (stat_path, endpoint)
            ) from exc
        if len(state) != 1:
            raise RamdiskError(
                "cannot parse process identity %s after missing endpoint %s; "
                "managed cleanup requires complete /proc visibility"
                % (stat_path, endpoint)
            )

        # PF_KTHREAD tasks have no userspace mm, files, or cwd.
        if flags & 0x00200000:
            return True
        if state in ("Z", "X", "x"):
            # A multithreaded process can retain a zombie group leader while
            # live siblings still share its mm/files/fs. Only a complete task
            # snapshot with no nonleader TID proves this dead leader inert.
            task_dir = "/proc/%s/task" % entry
            try:
                task_entries = os.listdir(task_dir)
            except (FileNotFoundError, ProcessLookupError):
                try:
                    _read_proc_stat(stat_path)
                except (FileNotFoundError, ProcessLookupError):
                    return True
                except (OSError, UnicodeError) as exc:
                    visibility_error("recheck process identity", stat_path, exc)
                raise RamdiskError(
                    "cannot enumerate task group %s while PID %s remains; "
                    "managed cleanup requires complete /proc visibility"
                    % (task_dir, entry)
                ) from missing_error
            except OSError as exc:
                visibility_error("enumerate process task group", task_dir, exc)
            invalid_tasks = [
                task for task in task_entries if not task.isdigit()
            ]
            if invalid_tasks:
                raise RamdiskError(
                    "cannot parse process task group %s; managed cleanup "
                    "requires complete /proc visibility" % task_dir
                )
            try:
                num_threads = int(fields[17], 10)
            except (IndexError, TypeError, ValueError) as exc:
                raise RamdiskError(
                    "cannot parse process thread count %s; managed cleanup "
                    "requires complete /proc visibility" % stat_path
                ) from exc
            task_ids = [int(task) for task in task_entries]
            unique_tasks = set(task_ids)
            if (
                num_threads <= 0
                or len(task_ids) != num_threads
                or len(unique_tasks) != num_threads
                or int(entry) not in unique_tasks
            ):
                raise RamdiskError(
                    "incomplete process task snapshot %s: stat declares %d "
                    "threads but task entries are %s; managed cleanup "
                    "requires complete /proc visibility"
                    % (
                        task_dir,
                        num_threads,
                        ",".join(str(task) for task in sorted(unique_tasks))
                        or "none",
                    )
                )
            live_siblings = sorted(
                task for task in unique_tasks if task != int(entry)
            )
            if not live_siblings:
                return True
            raise RamdiskError(
                "cannot trust missing process endpoint %s: zombie/dead "
                "leader PID %s still has sibling tasks %s; managed cleanup "
                "requires complete /proc visibility"
                % (
                    endpoint,
                    entry,
                    ",".join(str(task) for task in live_siblings),
                )
            ) from missing_error
        raise RamdiskError(
            "cannot trust missing process endpoint %s while PID %s remains "
            "a live userspace task; managed cleanup requires complete /proc "
            "visibility" % (endpoint, entry)
        ) from missing_error

    try:
        entries = os.listdir("/proc")
    except OSError as exc:
        visibility_error("enumerate", "/proc", exc)

    maps_line = re.compile(
        r"^[0-9A-Fa-f]+-[0-9A-Fa-f]+\s+"
        r"[r-][w-][x-][ps]\s+[0-9A-Fa-f]+\s+"
        r"[0-9A-Fa-f]+:[0-9A-Fa-f]+\s+\d+"
        r"(?:\s+(.*))?$"
    )

    def task_group_snapshot(entry):
        """Return one complete task-membership snapshot for a live TGID."""
        stat_path = "/proc/%s/stat" % entry
        try:
            process_stat = _read_proc_stat(stat_path)
        except (FileNotFoundError, ProcessLookupError):
            return None
        except (OSError, UnicodeError) as exc:
            visibility_error("read process identity", stat_path, exc)

        close = process_stat.rfind(")")
        fields = process_stat[close + 2 :].split() if close >= 0 else []
        try:
            state = fields[0]
            flags = int(fields[6], 10)
            num_threads = int(fields[17], 10)
            start_time = int(fields[19], 10)
        except (IndexError, TypeError, ValueError) as exc:
            raise RamdiskError(
                "cannot parse process task identity %s; managed cleanup "
                "requires complete /proc visibility" % stat_path
            ) from exc
        if len(state) != 1 or num_threads <= 0 or start_time < 0:
            raise RamdiskError(
                "cannot parse process task identity %s; managed cleanup "
                "requires complete /proc visibility" % stat_path
            )

        task_dir = "/proc/%s/task" % entry
        try:
            task_entries = os.listdir(task_dir)
        except (FileNotFoundError, ProcessLookupError) as exc:
            try:
                _read_proc_stat(stat_path)
            except (FileNotFoundError, ProcessLookupError):
                return None
            except (OSError, UnicodeError) as recheck_exc:
                visibility_error(
                    "recheck process identity",
                    stat_path,
                    recheck_exc,
                )
            raise RamdiskError(
                "cannot enumerate task group %s while PID %s remains; "
                "managed cleanup requires complete /proc visibility"
                % (task_dir, entry)
            ) from exc
        except OSError as exc:
            visibility_error("enumerate process task group", task_dir, exc)

        if any(not task.isdigit() for task in task_entries):
            raise RamdiskError(
                "cannot parse process task group %s; managed cleanup "
                "requires complete /proc visibility" % task_dir
            )
        task_ids = [int(task) for task in task_entries]
        unique_tasks = set(task_ids)
        if (
            len(task_ids) != num_threads
            or len(unique_tasks) != num_threads
            or int(entry) not in unique_tasks
        ):
            raise RamdiskError(
                "incomplete process task snapshot %s: stat declares %d "
                "threads but task entries are %s; managed cleanup requires "
                "complete /proc visibility"
                % (
                    task_dir,
                    num_threads,
                    ",".join(str(task) for task in sorted(unique_tasks))
                    or "none",
                )
            )
        return {
            "flags": flags,
            "start_time": start_time,
            "tasks": tuple(sorted(unique_tasks)),
        }

    def missing_task_endpoint_is_inert(entry, task, endpoint, missing_error):
        """Reject a partial live-task view; tolerate only proven inert tasks."""
        task_stat_path = "/proc/%s/task/%s/stat" % (entry, task)
        try:
            task_stat = _read_proc_stat(task_stat_path)
        except (FileNotFoundError, ProcessLookupError):
            leader_stat_path = "/proc/%s/stat" % entry
            try:
                _read_proc_stat(leader_stat_path)
            except (FileNotFoundError, ProcessLookupError):
                return True
            except (OSError, UnicodeError) as exc:
                visibility_error(
                    "recheck process identity",
                    leader_stat_path,
                    exc,
                )
            raise RamdiskError(
                "incomplete process task snapshot: task %s disappeared at %s "
                "while PID %s remains; managed cleanup requires complete "
                "/proc visibility" % (task, endpoint, entry)
            ) from missing_error
        except (OSError, UnicodeError) as exc:
            visibility_error("verify process task identity", task_stat_path, exc)

        close = task_stat.rfind(")")
        fields = task_stat[close + 2 :].split() if close >= 0 else []
        try:
            state = fields[0]
            flags = int(fields[6], 10)
        except (IndexError, TypeError, ValueError) as exc:
            raise RamdiskError(
                "cannot parse process task identity %s after missing endpoint "
                "%s; managed cleanup requires complete /proc visibility"
                % (task_stat_path, endpoint)
            ) from exc
        if flags & 0x00200000 or state in ("Z", "X", "x"):
            return True
        raise RamdiskError(
            "cannot trust missing task endpoint %s while task %s in PID %s "
            "remains live; managed cleanup requires complete /proc visibility"
            % (endpoint, task, entry)
        ) from missing_error

    def scan_task_references(entry, pid, endpoint_root, missing_is_inert):
        """Return ``(busy, inert)`` for one leader or nonleader task."""
        for leaf in ("cwd", "root", "exe"):
            proc_path = "%s/%s" % (endpoint_root, leaf)
            try:
                target = os.readlink(proc_path)
                if reference_below(target):
                    return True, False
            except (FileNotFoundError, ProcessLookupError) as exc:
                if missing_is_inert(proc_path, exc):
                    return False, True
            except OSError as exc:
                visibility_error("read process reference", proc_path, exc)

        maps_path = "%s/maps" % endpoint_root
        try:
            with open(
                maps_path,
                "r",
                encoding="utf-8",
                errors="surrogateescape",
            ) as stream:
                mappings = stream.read()
        except (FileNotFoundError, ProcessLookupError) as exc:
            if missing_is_inert(maps_path, exc):
                return False, True
        except OSError as exc:
            visibility_error("read process mappings", maps_path, exc)
        for line_number, line in enumerate(mappings.splitlines(), 1):
            match = maps_line.fullmatch(line)
            if match is None:
                raise RamdiskError(
                    "cannot parse process mappings %s line %d; managed cleanup "
                    "requires complete /proc visibility"
                    % (maps_path, line_number)
                )
            mapped = match.group(1)
            if not mapped or not mapped.startswith("/"):
                continue
            if reference_below(_unescape_mount(mapped)):
                return True, False

        fd_dir = "%s/fd" % endpoint_root
        try:
            descriptors = os.listdir(fd_dir)
        except (FileNotFoundError, ProcessLookupError) as exc:
            if missing_is_inert(fd_dir, exc):
                return False, True
        except OSError as exc:
            visibility_error("enumerate process descriptors", fd_dir, exc)
        for descriptor in descriptors:
            descriptor_path = os.path.join(fd_dir, descriptor)
            try:
                target = os.readlink(descriptor_path)
                if reference_below(target):
                    return True, False
            except (FileNotFoundError, ProcessLookupError):
                # Descriptor closure after listdir() releases that reference.
                continue
            except OSError as exc:
                visibility_error(
                    "read process descriptor",
                    descriptor_path,
                    exc,
                )
        return False, False

    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        leader_root = "/proc/%s" % entry
        busy, inert = scan_task_references(
            entry,
            pid,
            leader_root,
            lambda endpoint, error: missing_endpoint_is_inert(
                entry,
                endpoint,
                error,
            ),
        )
        if inert:
            continue
        if busy:
            found.append(pid)
            continue

        initial = task_group_snapshot(entry)
        if initial is None or initial["flags"] & 0x00200000:
            continue

        for task in initial["tasks"]:
            if task == pid:
                continue
            task_root = "/proc/%s/task/%s" % (entry, task)
            busy, _ = scan_task_references(
                entry,
                pid,
                task_root,
                lambda endpoint, error, task=task: (
                    missing_task_endpoint_is_inert(
                        entry,
                        task,
                        endpoint,
                        error,
                    )
                ),
            )
            if busy:
                found.append(pid)
                break
        if found and found[-1] == pid:
            continue

        final = task_group_snapshot(entry)
        if final is None:
            continue
        if (
            final["start_time"] != initial["start_time"]
            or final["tasks"] != initial["tasks"]
        ):
            raise RamdiskError(
                "incomplete process task snapshot /proc/%s/task changed while "
                "it was inspected; managed cleanup requires complete /proc "
                "visibility" % entry
            )
    return sorted(set(found))


def _fuser_failure(path, result):
    detail = " ".join(
        value.strip()
        for value in (
            getattr(result, "stdout", "") or "",
            getattr(result, "stderr", "") or "",
        )
        if value.strip()
    )
    if len(detail) > 1000:
        detail = detail[:997] + "..."
    suffix = ": %s" % detail if detail else ""
    return RamdiskError(
        "trusted fuser could not inspect managed mount %s (exit %s)%s"
        % (path, getattr(result, "returncode", "unknown"), suffix)
    )


def _parse_fuser_mount_references(path, result):
    """Parse PSmisc fuser's intentionally split stdout/stderr contract."""
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    returncode = getattr(result, "returncode", None)
    if returncode == 0:
        tokens = stdout.split()
        if not tokens or any(
            re.fullmatch(r"[0-9]+", token) is None or int(token) <= 0
            for token in tokens
        ):
            raise RamdiskError(
                "trusted fuser returned an invalid PID list for %s" % path
            )
        # Without --verbose, PSmisc writes exactly the requested mount name
        # followed by the per-PID access letters c/e/f/F/r/m to stderr. Match
        # the exact path independently of its whitespace or regex syntax;
        # arbitrary diagnostics would mean the scan may be incomplete.
        annotation = re.fullmatch(
            re.escape(path) + r":[ \tcefFrm]*(?:\r?\n)?",
            stderr,
        )
        if stderr and annotation is None:
            raise _fuser_failure(path, result)
        return sorted(set(int(token) for token in tokens))
    if returncode == 1 and not stdout.strip() and not stderr.strip():
        return []
    raise _fuser_failure(path, result)


def _trusted_fuser_binary(trusted_system_binary):
    """Resolve PSmisc fuser with install guidance for unprivileged cleanup."""
    try:
        return trusted_system_binary("fuser")
    except RamdiskError as exc:
        raise RamdiskError(
            "unprivileged managed cleanup requires trusted PSmisc fuser: "
            "%s; install the psmisc package and retry" % exc
        ) from exc


def _busy_mount_references(
    path,
    hardware=None,
    *,
    run=None,
    trusted_system_binary=None,
    privileged=None,
):
    """Return a complete busy set via root procfs or trusted privileged fuser."""
    _require_linux()
    path = os.path.normpath(os.path.abspath(os.fspath(path)))
    if current_euid() == 0:
        return _busy_mount_references_proc(path)
    run = _run if run is None else run
    trusted_system_binary = (
        _trusted_system_binary
        if trusted_system_binary is None
        else trusted_system_binary
    )
    fuser = _trusted_fuser_binary(trusted_system_binary)
    command = [fuser, "-mM", path]
    if privileged is None:
        command = _privileged(
            command,
            hardware,
            trusted_system_binary=trusted_system_binary,
        )
    else:
        command = privileged(command, hardware)
    try:
        result = run(command, timeout=10.0)
    except Exception as exc:
        raise RamdiskError(
            "trusted fuser could not inspect managed mount %s: %s"
            % (path, exc)
        ) from exc
    return _parse_fuser_mount_references(path, result)


def _ensure_busy_mount_scan_available(
    path,
    hardware=None,
    *,
    trusted_system_binary=None,
    run=None,
):
    """Prove cleanup discovery and unmount exist before mount mutation."""
    _require_linux()
    trusted_system_binary = (
        _trusted_system_binary
        if trusted_system_binary is None
        else trusted_system_binary
    )
    run = _run if run is None else run
    if current_euid() == 0:
        _busy_mount_references_proc(path)
    else:
        # The planned path is not a mountpoint yet, so ``fuser -M`` cannot
        # probe it. Root is always a mountpoint and exercises the exact trusted
        # PSmisc binary, privileged command shape, option support, and current
        # sudo authorization that cleanup will require later.
        _busy_mount_references(
            os.path.sep,
            hardware=hardware,
            run=run,
            trusted_system_binary=trusted_system_binary,
        )
    umount = trusted_system_binary("umount")
    help_command = [umount, "--help"]
    if current_euid() != 0:
        help_command = _privileged(
            help_command,
            hardware,
            trusted_system_binary=trusted_system_binary,
        )
    try:
        help_result = run(help_command, timeout=5.0)
    except Exception as exc:
        raise RamdiskError(
            "trusted umount helper could not be verified: %s; install the "
            "util-linux package" % exc
        ) from exc
    help_output = "%s\n%s" % (
        getattr(help_result, "stdout", "") or "",
        getattr(help_result, "stderr", "") or "",
    )
    if (
        getattr(help_result, "returncode", None) != 0
        or "--no-canonicalize" not in help_output
    ):
        raise RamdiskError(
            "trusted privileged umount helper is incompatible or "
            "unauthorized: cleanup requires the util-linux "
            "--no-canonicalize option"
        )


class LinuxPlatformOps:
    """Narrow Linux discovery operations with no import-time probes."""

    is_linux = True

    def __init__(self, platform_name="linux"):
        self.platform_name = platform_name

    @property
    def process_control_supported(self):
        """Whether managed tasks can be pinned and safely signalled."""
        return (
            callable(getattr(os, "getpgid", None))
            and callable(getattr(os, "killpg", None))
            and getattr(signal, "SIGTERM", None) is not None
            and getattr(signal, "SIGKILL", None) is not None
            and _pidfd_process_control_supported()
        )

    @property
    def process_control_reason(self):
        return _PIDFD_REQUIRED_REASON

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
    process_start_boundary = staticmethod(_process_start_boundary)
    process_identity = staticmethod(_process_identity)
    managed_launch_processes = staticmethod(_managed_launch_processes)
    process_group_member_pids = staticmethod(_process_group_member_pids)
    process_group_alive = staticmethod(_process_group_alive)
    signal_verified_process_group = staticmethod(
        _signal_verified_process_group
    )
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
