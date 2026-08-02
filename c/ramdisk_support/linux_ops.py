"""Linux filesystem and kernel operations used by RAM-disk discovery."""

from __future__ import print_function

import contextlib
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


def _process_identity(pid):
    """Read one process identity from procfs without trusting its command."""
    _require_linux()
    getpgid = getattr(os, "getpgid", None)
    if getpgid is None:
        raise RamdiskError(UNSUPPORTED_PLATFORM_REASON)
    pid = int(pid)
    try:
        raw_stat = _read_proc_stat("/proc/%d/stat" % pid)
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
        for leaf in ("cwd", "root"):
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
        """Whether managed process groups can be identified and signalled."""
        return (
            callable(getattr(os, "getpgid", None))
            and callable(getattr(os, "killpg", None))
            and getattr(signal, "SIGTERM", None) is not None
            and getattr(signal, "SIGKILL", None) is not None
        )

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
