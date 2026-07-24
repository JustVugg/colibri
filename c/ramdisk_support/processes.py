"""Managed process identity, admission, readiness, and cleanup helpers."""

from __future__ import print_function

import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

from .common import (
    GIB,
    RamdiskError,
    _raise_if_cancelled,
    _utc_now,
)
from .platform_ops import get_platform_ops


def _proc_identity(pid):
    return get_platform_ops().process_identity(pid)


def _process_group_members(
    pgid,
    *,
    ops=None,
    proc_identity=None,
):
    """Return readable identities and unreadable PIDs in one process group."""
    ops = get_platform_ops() if ops is None else ops
    proc_identity = (
        _proc_identity
        if proc_identity is None
        else proc_identity
    )
    members = []
    unreadable = []
    for pid in ops.process_group_member_pids(pgid):
        identity = proc_identity(pid)
        if identity:
            members.append(identity)
        else:
            unreadable.append(pid)
    return members, unreadable


def _process_matches(
    record,
    *,
    proc_identity=None,
    process_group_members=None,
):
    proc_identity = (
        _proc_identity
        if proc_identity is None
        else proc_identity
    )
    process_group_members = (
        _process_group_members
        if process_group_members is None
        else process_group_members
    )
    pid = int(record["pid"])
    expected_pgid = int(record.get("pgid", pid))
    actual = proc_identity(pid)
    if not actual:
        # The Python serve wrapper can die before its engine child. A managed
        # session keeps the original PGID, so validate every surviving
        # member's inherited UID+nonce before treating the group as signalable.
        members, unreadable = process_group_members(expected_pgid)
        if not members and not unreadable:
            return False, "not-running", None
        if unreadable:
            return (
                False,
                "unverified-process-group",
                {"pgid": expected_pgid, "members": unreadable},
            )
        if any(
            member["uid"] != record.get("uid")
            or member["nonce"] != record.get("nonce")
            for member in members
        ):
            return (
                False,
                "foreign-process-group",
                {"pgid": expected_pgid, "members": members},
            )
        return (
            True,
            "running-group",
            {"pid": pid, "pgid": expected_pgid, "members": members},
        )
    if actual["uid"] != record.get("uid"):
        return False, "foreign-uid", actual
    if actual["starttime"] != record.get("starttime"):
        return False, "reused-pid", actual
    if actual["nonce"] != record.get("nonce"):
        return False, "foreign-nonce", actual
    if actual["pgid"] != expected_pgid:
        return False, "foreign-process-group", actual
    return True, "running", actual


def _process_tree_alive(
    record,
    actual,
    *,
    group_alive=None,
    proc_identity=None,
):
    group_alive = _group_alive if group_alive is None else group_alive
    proc_identity = (
        _proc_identity
        if proc_identity is None
        else proc_identity
    )
    expected_pgid = int(record.get("pgid", record["pid"]))
    if actual and actual.get("pgid") == expected_pgid:
        return group_alive(expected_pgid)
    return bool(proc_identity(int(record["pid"])))


def _runtime_admission_requirement(plan, mount, benchmark=False):
    """Return the runtime, page-table, and protected host floor."""
    reserve = plan["reserve"]
    runtime_bytes = int(
        reserve.get(
            "benchmark_runtime_bytes"
            if benchmark
            else "managed_runtime_bytes"
        )
        or reserve["runtime_bytes"]
    )
    page_tables = int(reserve["page_table_bytes"])
    if mount.get("node") is None:
        margin = int(reserve["os_margin_bytes"])
    else:
        node = next(
            item
            for item in plan["hardware"]["nodes"]
            if item["id"] == mount["node"]
        )
        margin = int(
            node.get(
                "reserve_bytes",
                max(node["memory_total_bytes"] // 10, 8 * GIB),
            )
        )
    return runtime_bytes + page_tables + margin


def _admit_runtime(
    plan,
    mount,
    benchmark=False,
    *,
    available_for_mount=None,
):
    """Recheck the reviewed post-staging floor immediately before launch."""
    if available_for_mount is None:
        raise RamdiskError("runtime memory availability is unavailable")
    required = _runtime_admission_requirement(
        plan,
        mount,
        benchmark=benchmark,
    )
    available = available_for_mount(mount, plan=plan)
    if available < required:
        label = (
            "global memory"
            if mount.get("node") is None
            else "NUMA node %d" % mount["node"]
        )
        raise RamdiskError(
            "%s has %d bytes available; launch would breach the "
            "%d-byte runtime/OS floor"
            % (label, available, required)
        )
    return {
        "available_bytes": available,
        "required_bytes": required,
    }


def _admit_concurrent_runtimes(
    plan,
    mounts,
    benchmark=False,
    *,
    host_available_for_mount=None,
    cgroup_available_memory=None,
):
    """Admit replicas against one shared cgroup-headroom snapshot."""
    if host_available_for_mount is None:
        raise RamdiskError("host memory availability is unavailable")
    if cgroup_available_memory is None:
        raise RamdiskError("cgroup memory availability is unavailable")
    mounts = list(mounts)
    if not mounts:
        raise RamdiskError(
            "concurrent runtime admission requires at least one mount"
        )
    admissions = []
    for mount in mounts:
        required = _runtime_admission_requirement(
            plan,
            mount,
            benchmark=benchmark,
        )
        host_available = host_available_for_mount(mount, plan=plan)
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
    cgroup_available = cgroup_available_memory()
    aggregate_required = sum(
        item["required_bytes"]
        for item in admissions
    )
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


def _assert_effective_masks_unchanged(
    plan,
    *,
    discover_hardware=None,
):
    """Refuse launch after the reviewed cgroup/cpuset contract drifts."""
    hardware = plan.get("hardware", {})
    placement = plan.get("placement")
    if (
        not placement
        or hardware.get("effective_mask_source")
        != "kernel-task-status"
    ):
        return
    if discover_hardware is None:
        raise RamdiskError("hardware discovery is unavailable")
    current = discover_hardware()
    expected_nodes = list(placement.get("effective_nodes", []))
    expected_cpus = list(placement.get("effective_cpus", []))
    if (
        list(current.get("effective_nodes", [])) != expected_nodes
        or list(current.get("effective_cpus", [])) != expected_cpus
    ):
        raise RamdiskError(
            "effective CPU/NUMA mask changed since preparation; "
            "destroy and review a fresh plan"
        )


def _group_alive(pgid):
    return get_platform_ops().process_group_alive(pgid)


_managed_children_lock = threading.Lock()
_managed_children = {}


def _track_managed_child(process):
    """Retain Popen handles so a long-lived TUI can reap engine zombies."""
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


def _terminate_direct_child(
    process,
    term_seconds=10.0,
    kill_seconds=3.0,
):
    """Terminate an unrecorded child by PID, never an unverified PGID."""
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


def _terminate_group(
    pgid,
    term_seconds=10.0,
    kill_seconds=3.0,
    *,
    group_alive=None,
    ops=None,
):
    """Terminate one already-verified or directly-created process group."""
    group_alive = _group_alive if group_alive is None else group_alive
    ops = get_platform_ops() if ops is None else ops
    try:
        ops.signal_process_group(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return None
    deadline = time.monotonic() + term_seconds
    while time.monotonic() < deadline and group_alive(pgid):
        time.sleep(0.1)
    if not group_alive(pgid):
        return None
    try:
        ops.signal_process_group(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return None
    deadline = time.monotonic() + kill_seconds
    while time.monotonic() < deadline and group_alive(pgid):
        time.sleep(0.05)
    return (
        "process group %s survived SIGKILL" % pgid
        if group_alive(pgid)
        else None
    )


def _terminate_verified_group(
    record,
    term_seconds=10.0,
    kill_seconds=3.0,
    *,
    poll_managed_child=None,
    process_matches=None,
    ops=None,
):
    """Signal only while the persisted identity still owns its PGID."""
    poll_managed_child = (
        _poll_managed_child
        if poll_managed_child is None
        else poll_managed_child
    )
    process_matches = (
        _process_matches
        if process_matches is None
        else process_matches
    )
    ops = get_platform_ops() if ops is None else ops
    expected_pgid = int(record.get("pgid", record["pid"]))

    def revalidate(stage):
        # Poll retained Popen handles first so an exited group leader cannot
        # remain a zombie that falsely appears to have survived a signal.
        poll_managed_child(record["pid"])
        matches, reason, actual = process_matches(record)
        if (
            matches
            and actual
            and int(actual.get("pgid", -1)) == expected_pgid
        ):
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
        ops.signal_process_group(expected_pgid, signal.SIGTERM)
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
        ops.signal_process_group(expected_pgid, signal.SIGKILL)
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


def _wait_managed_ready(
    record,
    timeout,
    api_key=None,
    cancel_event=None,
    *,
    process_matches=None,
    urlopen=None,
):
    process_matches = (
        _process_matches
        if process_matches is None
        else process_matches
    )
    urlopen = urllib.request.urlopen if urlopen is None else urlopen
    deadline = time.monotonic() + timeout
    headers = (
        {"Authorization": "Bearer " + api_key}
        if api_key
        else {}
    )
    last_error = "listener not ready"
    while time.monotonic() < deadline:
        _raise_if_cancelled(cancel_event)
        matches, reason, _ = process_matches(record)
        if not matches:
            raise RamdiskError(
                "managed engine PID %s exited before readiness (%s); see %s"
                % (record["pid"], reason, record["log"])
            )
        try:
            request = urllib.request.Request(
                "http://127.0.0.1:%d/health" % record["port"],
                headers=headers,
            )
            with urlopen(request, timeout=2) as response:
                payload = json.loads(
                    response.read().decode("utf-8")
                )
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
        "managed engine on port %s did not become ready within %.0fs "
        "(%s); see %s"
        % (record["port"], timeout, last_error, record["log"])
    )


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
                os.path.dirname(here),
                "libexec",
                "colibri",
                "colibri" + suffix,
            ),
            os.path.join(here, "glm" + suffix),
            os.path.join(
                os.path.dirname(here),
                "libexec",
                "colibri",
                "glm" + suffix,
            ),
        ]
    )
    for candidate in candidates:
        if (
            candidate
            and os.path.isfile(candidate)
            and os.access(candidate, os.X_OK)
        ):
            return os.path.realpath(candidate)
    raise RamdiskError(
        "cannot locate the executable Colibri engine for "
        "persistent benchmarking"
    )


def _managed_process_metrics(
    record,
    *,
    process_matches=None,
    process_group_members=None,
    process_status=None,
):
    process_matches = (
        _process_matches
        if process_matches is None
        else process_matches
    )
    process_group_members = (
        _process_group_members
        if process_group_members is None
        else process_group_members
    )
    if process_status is None:
        process_status = get_platform_ops().process_status
    rss_bytes = None
    rss_processes = 0
    matches, _, _ = process_matches(record)
    if matches:
        members, unreadable = process_group_members(
            int(record.get("pgid", record["pid"]))
        )
        verified = [
            member
            for member in members
            if member.get("uid") == record.get("uid")
            and member.get("nonce") == record.get("nonce")
        ]
        if (
            not unreadable
            and len(verified) == len(members)
            and verified
        ):
            rss_bytes = 0
            for member in verified:
                for line in process_status(
                    member["pid"]
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
    ram_experts = None
    ram_bytes = None
    ssd_bytes = None
    matches = re.findall(
        r"RAM map:\s*(\d+) experts / ([0-9.]+) GB",
        tail,
    )
    if matches:
        ram_experts = int(matches[-1][0])
        ram_bytes = float(matches[-1][1]) * 1e9
    else:
        matches = re.findall(
            r"\[RAMMAP\]\s*(\d+) direct tmpfs experts,\s*"
            r"([0-9.]+) GB mapped",
            tail,
        )
        if matches:
            ram_experts = int(matches[-1][0])
            ram_bytes = float(matches[-1][1]) * 1e9
    matches = re.findall(
        r"physical SSD reads:\s*([0-9.]+) GB",
        tail,
        re.I,
    )
    if matches:
        ssd_bytes = float(matches[-1]) * 1e9
    return {
        "rss_bytes": rss_bytes,
        "rss_processes": rss_processes,
        "rammap_experts": ram_experts,
        "rammap_bytes": ram_bytes,
        "latest_ssd_bytes": ssd_bytes,
    }
