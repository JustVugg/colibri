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

from .accelerator import _same_gpu_identity
from .common import (
    GIB,
    RamdiskError,
    _positive_int,
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
    """Return identities only from a bounded stable membership view."""
    ops = get_platform_ops() if ops is None else ops
    proc_identity = (
        _proc_identity
        if proc_identity is None
        else proc_identity
    )
    pgid = int(pgid)

    def identity_fingerprint(identity):
        # Scheduler state (R/S/D) is expected to change between reads. Only
        # compare the stable identity and persisted-attribution fields.
        return tuple(
            identity.get(key)
            for key in (
                "pid",
                "uid",
                "inert",
                "starttime",
                "nonce",
                "pgid",
                "sid",
                "state_dir",
                "weights_dir",
            )
        )

    def scan():
        before = list(ops.process_group_member_pids(pgid))
        if before != sorted(set(before)):
            return [], sorted(set(before) or {pgid}), None
        identities = []
        unreadable = []
        for pid in before:
            identity = proc_identity(pid)
            if (
                isinstance(identity, dict)
                and identity.get("pid") == pid
            ):
                identities.append(identity)
            else:
                unreadable.append(pid)
        after = list(ops.process_group_member_pids(pgid))
        if after != before:
            unreadable.extend(set(before) | set(after))
        if unreadable:
            return identities, sorted(set(unreadable)), None
        return identities, [], (
            before,
            [identity_fingerprint(identity) for identity in identities],
        )

    first_members, first_unreadable, first_view = scan()
    if first_unreadable:
        return first_members, first_unreadable
    second_members, second_unreadable, second_view = scan()
    if second_unreadable:
        return second_members, second_unreadable
    if first_view != second_view:
        pids = {
            member.get("pid")
            for member in first_members + second_members
            if isinstance(member, dict)
            and isinstance(member.get("pid"), int)
        }
        return second_members, sorted(pids or {pgid})

    # Empty and all-inert snapshots authorize irreversible state cleanup, so
    # couple them to kernel liveness and one final complete identity scan.
    if not second_members or all(
        _proven_inert_group_member(member)
        for member in second_members
    ):
        alive_before = ops.process_group_alive(pgid)
        final_members, final_unreadable, final_view = scan()
        alive_after = ops.process_group_alive(pgid)
        if final_unreadable:
            return final_members, final_unreadable
        if final_view != second_view:
            pids = {
                member.get("pid")
                for member in second_members + final_members
                if isinstance(member, dict)
                and isinstance(member.get("pid"), int)
            }
            return final_members, sorted(pids or {pgid})
        if alive_before or alive_after:
            return final_members, [pgid]
        return final_members, []
    return second_members, []


def _proven_inert_group_member(identity):
    """Return whether procfs proved one stable process identity inert."""
    return (
        isinstance(identity, dict)
        and identity.get("inert") is True
        and _positive_int(identity.get("pid"))
        and _positive_int(identity.get("starttime"))
    )


def _inert_group_member_matches(record, identity, expected_pgid):
    """Validate one proven-dead member within a mixed live group."""
    if (
        not _proven_inert_group_member(identity)
        or identity.get("uid") != record.get("uid")
        or identity.get("pgid") != expected_pgid
        or identity.get("sid") != expected_pgid
    ):
        return False
    if identity["pid"] == int(record["pid"]):
        return identity["starttime"] == record.get("starttime")
    return True


def _live_group_member_matches(record, identity, expected_pgid):
    """Require complete persisted attribution for every runnable member."""
    return (
        isinstance(identity, dict)
        and identity.get("inert") is False
        and identity.get("uid") == record.get("uid")
        and identity.get("nonce") == record.get("nonce")
        and identity.get("pgid") == expected_pgid
        and identity.get("sid") == expected_pgid
        and identity.get("state_dir") == record.get("state_dir")
        and identity.get("weights_dir") == record.get("weights_dir")
    )


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

    def attribution_matches(identity):
        if isinstance(identity, dict) and identity.get("inert") is True:
            # Stable lone-thread zombies cannot run, retain files, or mutate
            # usage. Every runnable member still needs full attribution.
            return _inert_group_member_matches(
                record,
                identity,
                expected_pgid,
            )
        return _live_group_member_matches(record, identity, expected_pgid)

    actual = proc_identity(pid)
    if (
        _proven_inert_group_member(actual)
        and actual.get("pid") == pid
    ):
        actual = None
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
        if all(_proven_inert_group_member(member) for member in members):
            if any(
                not _inert_group_member_matches(
                    record,
                    member,
                    expected_pgid,
                )
                for member in members
            ):
                return (
                    False,
                    "foreign-process-group",
                    {"pgid": expected_pgid, "members": members},
                )
            return False, "not-running", None
        if any(not attribution_matches(member) for member in members):
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
    if actual.get("sid") != expected_pgid:
        return False, "foreign-session", actual
    if any(
        actual.get(key) != record.get(key)
        for key in ("state_dir", "weights_dir")
    ):
        return False, "foreign-path-attribution", actual
    members, unreadable = process_group_members(expected_pgid)
    if unreadable or not members:
        return (
            False,
            "unverified-process-group",
            {"pgid": expected_pgid, "members": unreadable},
        )
    if all(_proven_inert_group_member(member) for member in members):
        if any(
            not _inert_group_member_matches(
                record,
                member,
                expected_pgid,
            )
            for member in members
        ):
            return (
                False,
                "foreign-process-group",
                {"pgid": expected_pgid, "members": members},
            )
        return False, "not-running", None
    if any(not attribution_matches(member) for member in members):
        return (
            False,
            "foreign-process-group",
            {"pgid": expected_pgid, "members": members},
        )
    running = dict(actual)
    running["members"] = members
    return True, "running", running


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
    accelerator = plan.get("managed_accelerator") or {}
    check_masks = bool(
        placement
        and hardware.get("effective_mask_source")
        == "kernel-task-status"
    )
    check_gpus = accelerator.get("mode") == "cuda"
    if not check_masks and not check_gpus:
        return
    if discover_hardware is None:
        raise RamdiskError("hardware discovery is unavailable")
    current = discover_hardware()
    if check_masks:
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
    if check_gpus:
        observed = {
            int(device["index"]): device
            for device in current.get("gpus", [])
            if isinstance(device, dict)
            and isinstance(device.get("index"), int)
            and not isinstance(device.get("index"), bool)
        }
        effective_nodes = set(current.get("effective_nodes") or [])
        for expected in accelerator.get("devices") or []:
            index = expected.get("index")
            device = observed.get(index)
            if (
                device is None
                or not _same_gpu_identity(expected, device)
                or device.get("numa_node") != expected.get("numa_node")
                or device.get("numa_node") not in effective_nodes
            ):
                raise RamdiskError(
                    "managed GPU/NUMA identity changed since preparation; "
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


def _managed_child_liveness(pid):
    """Return True/False for a retained child, or None without a handle."""
    with _managed_children_lock:
        process = _managed_children.get(int(pid))
    if process is None:
        return None
    try:
        returncode = process.poll()
    except (ChildProcessError, OSError):
        returncode = getattr(process, "returncode", None)
    if returncode is None:
        return True
    with _managed_children_lock:
        if _managed_children.get(int(pid)) is process:
            _managed_children.pop(int(pid), None)
    return False


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


def _terminate_verified_group(
    record,
    term_seconds=10.0,
    kill_seconds=3.0,
    *,
    managed_child_liveness=None,
    process_matches=None,
    ops=None,
):
    """Terminate persisted members only through freshly verified pidfds."""
    managed_child_liveness = (
        _managed_child_liveness
        if managed_child_liveness is None
        else managed_child_liveness
    )
    # ``process_matches`` remains an accepted injection for facade/API
    # compatibility. Lifecycle preflight uses it before this function, but it
    # cannot safely authorize a later signal because PID/PGID reuse may occur
    # between those two operations.
    del process_matches
    ops = get_platform_ops() if ops is None else ops
    expected_pgid = int(record.get("pgid", record["pid"]))

    def signal_round(signum, stage):
        result = ops.signal_verified_process_group(record, signum)
        if not isinstance(result, dict):
            return (
                "failed",
                "PID/PGID %s verified cleanup returned an invalid result %s"
                % (expected_pgid, stage),
            )
        status = result.get("status")
        reason = result.get("reason", "unspecified")
        if status == "absent":
            # Polling also reaps a retained local group leader. A live Popen
            # handle contradicts procfs absence and must retain authority.
            if managed_child_liveness(record["pid"]) is True:
                return (
                    "failed",
                    "PID/PGID %s retained managed child is still live %s; "
                    "refusing to accept process-group absence"
                    % (expected_pgid, stage),
                )
            return "stopped", None
        if status == "foreign":
            return (
                "failed",
                "PID/PGID %s identity changed %s (%s); refusing any "
                "further signal"
                % (expected_pgid, stage, reason),
            )
        if status == "inconclusive":
            return "inconclusive", reason
        if status == "signaled":
            return "running", None
        return (
            "failed",
            "PID/PGID %s verified cleanup returned unknown status %r %s"
            % (expected_pgid, status, stage),
        )

    def signal_window(signum, duration, interval, label):
        deadline = time.monotonic() + max(0.0, float(duration))
        last_inconclusive = None
        first = True
        while first or time.monotonic() < deadline:
            first = False
            state, detail = signal_round(signum, "during %s" % label)
            if state == "stopped":
                return "stopped", None
            if state == "failed":
                return "failed", detail
            if state == "inconclusive":
                last_inconclusive = detail
            else:
                last_inconclusive = None
            if time.monotonic() >= deadline:
                break
            time.sleep(interval)
        return "deadline", last_inconclusive

    state, failure = signal_window(
        signal.SIGTERM,
        term_seconds,
        0.1,
        "SIGTERM grace period",
    )
    if state == "stopped":
        return None
    if state == "failed":
        return failure

    state, failure = signal_window(
        signal.SIGKILL,
        kill_seconds,
        0.05,
        "SIGKILL grace period",
    )
    if state == "stopped":
        return None
    if state == "failed":
        return failure
    if failure:
        return (
            "process group %s remained unverified after SIGKILL (%s)"
            % (expected_pgid, failure)
        )
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
        expected_pgid = int(record.get("pgid", record["pid"]))
        members, unreadable = process_group_members(
            expected_pgid
        )
        inert_members = [
            member
            for member in members
            if isinstance(member, dict)
            and member.get("inert") is True
        ]
        live_members = [
            member
            for member in members
            if not isinstance(member, dict)
            or member.get("inert") is not True
        ]
        verified_live = [
            member
            for member in live_members
            if _live_group_member_matches(record, member, expected_pgid)
        ]
        if (
            not unreadable
            and all(
                _inert_group_member_matches(
                    record,
                    member,
                    expected_pgid,
                )
                for member in inert_members
            )
            and len(verified_live) == len(live_members)
            and verified_live
        ):
            rss_bytes = 0
            for member in verified_live:
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
