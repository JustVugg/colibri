"""tmpfs mounting, shard staging, and NUMA namespace validation."""

from __future__ import print_function

import concurrent.futures
import hashlib
import mmap
import os
import re
import secrets
import stat
import subprocess
import sys
import threading
import time

from .common import (
    GIB,
    MIB,
    RamdiskError,
    _raise_if_cancelled,
)
from .discovery import _discover_cgroup_memory
from .linux_ops import (
    _current_gid,
    _meminfo,
    _mount_at,
    _node_meminfo,
    _privileged,
    _run,
    _trusted_system_binary,
)
from .model import _read_safetensors_header, scan_model
from .platform_ops import (
    UNSUPPORTED_PLATFORM_REASON,
    current_uid,
    get_platform_ops,
)


def _busy_mount_references(path, *, ops=None):
    """Return processes that keep a managed mount busy."""
    ops = get_platform_ops() if ops is None else ops
    return ops.busy_mount_references(path)


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
        "uid=%d" % current_uid(),
        "gid=%d" % _current_gid(),
        "mpol=%s" % mount["policy"],
        "X-mount.mkdir=0755",
    ]
    if include_noswap:
        options.insert(1, "noswap")
    return options


def _mount_tmpfs(
    plan,
    mount,
    *,
    trusted_system_binary=None,
    run=None,
    privileged=None,
    rollback_interrupted_mount=None,
):
    hardware = plan["hardware"]
    trusted_system_binary = (
        _trusted_system_binary
        if trusted_system_binary is None
        else trusted_system_binary
    )
    run = _run if run is None else run
    privileged = _privileged if privileged is None else privileged
    rollback_interrupted_mount = (
        _rollback_interrupted_mount
        if rollback_interrupted_mount is None
        else rollback_interrupted_mount
    )
    mount_bin = trusted_system_binary("mount")
    attempts = []
    thp = plan["mount_options"]["thp"]
    noswap = plan["mount_options"]["noswap"]
    attempts.append((thp, noswap))
    if thp == "within_size":
        attempts.append(("advise", noswap))
    if plan["mount_options"]["allow_swappable"] and noswap:
        attempts.append((thp, False))
        if thp == "within_size":
            attempts.append(("advise", False))
    seen = set()
    errors = []
    for try_thp, try_noswap in attempts:
        if (try_thp, try_noswap) in seen:
            continue
        seen.add((try_thp, try_noswap))
        options = _mount_option_list(
            plan,
            mount,
            try_thp,
            try_noswap,
        )
        command = [
            mount_bin,
            "-t",
            "tmpfs",
            "-o",
            ",".join(options),
            "tmpfs",
            mount["path"],
        ]
        try:
            result = run(privileged(command, hardware))
        except BaseException as interrupted:
            rollback_interrupted_mount(
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
        errors.append(
            result.stderr.strip()
            or result.stdout.strip()
            or "mount failed"
        )
        message = (result.stderr + result.stdout).lower()
        if not any(
            word in message
            for word in (
                "invalid argument",
                "unknown",
                "not supported",
                "wrong fs",
            )
        ):
            break
    raise RamdiskError(
        "cannot mount tmpfs at %s: %s"
        % (mount["path"], "; ".join(errors))
    )


def _umount_path(
    path,
    hardware,
    *,
    trusted_system_binary=None,
    run=None,
    privileged=None,
):
    trusted_system_binary = (
        _trusted_system_binary
        if trusted_system_binary is None
        else trusted_system_binary
    )
    run = _run if run is None else run
    privileged = _privileged if privileged is None else privileged
    umount = trusted_system_binary("umount")
    result = run(privileged([umount, "--", path], hardware))
    if result.returncode:
        message = (
            result.stderr.strip()
            or result.stdout.strip()
            or "umount failed"
        )
        raise RamdiskError("cannot unmount %s: %s" % (path, message))


def _rollback_interrupted_mount(
    plan,
    mount,
    effective_thp,
    effective_noswap,
    cause,
    *,
    mount_at=None,
    validate_mount=None,
    umount_path=None,
):
    """Remove a mount that completed just as its helper was interrupted."""
    mount_at = _mount_at if mount_at is None else mount_at
    validate_mount = (
        _validate_mount
        if validate_mount is None
        else validate_mount
    )
    umount_path = _umount_path if umount_path is None else umount_path
    actual = mount_at(mount["path"])
    if actual is None:
        return
    attempted = dict(mount)
    attempted["effective_thp"] = effective_thp
    attempted["effective_noswap"] = effective_noswap
    try:
        validate_mount(attempted, plan)
    except Exception as verification_error:
        raise RamdiskError(
            "mount helper was interrupted and a mount now exists at %s, "
            "but it cannot be attributed safely: %s"
            % (mount["path"], verification_error)
        ) from cause
    try:
        umount_path(mount["path"], plan["hardware"])
    except Exception as cleanup_error:
        raise RamdiskError(
            "mount helper was interrupted after tmpfs appeared at %s; "
            "immediate rollback failed: %s"
            % (mount["path"], cleanup_error)
        ) from cause


def _option_present(options, name):
    return any(
        option == name or option.startswith(name + "=")
        for option in options
    )


def _validate_mount(mount, plan, *, mount_at=None):
    mount_at = _mount_at if mount_at is None else mount_at
    actual = mount_at(mount["path"])
    if not actual:
        raise RamdiskError("expected mount is absent: %s" % mount["path"])
    if actual["filesystem"] != "tmpfs" or actual["source"] != "tmpfs":
        raise RamdiskError("refusing foreign mount at %s" % mount["path"])
    options = set(actual["options"] + actual["super_options"])
    required = ("noatime", "nodev", "nosuid", "noexec")
    missing = [
        name
        for name in required
        if not _option_present(options, name)
    ]
    if mount.get(
        "effective_noswap",
        plan["mount_options"]["noswap"],
    ) and not _option_present(options, "noswap"):
        missing.append("noswap")
    if missing:
        raise RamdiskError(
            "tmpfs at %s is missing options: %s"
            % (mount["path"], ", ".join(missing))
        )
    mode_ok = any(
        option in ("mode=700", "mode=0700")
        for option in options
    )
    huge = mount.get(
        "effective_thp",
        plan["mount_options"]["thp"],
    )
    policy = mount["policy"].replace("\\,", ",")
    normalized_options = {
        option.replace("\\,", ",")
        for option in options
    }
    if not mode_ok or not _option_present(
        normalized_options,
        "huge",
    ):
        raise RamdiskError(
            "tmpfs at %s is missing managed mode/THP options"
            % mount["path"]
        )
    if "huge=%s" % huge not in normalized_options:
        raise RamdiskError(
            "tmpfs at %s has an unexpected THP policy"
            % mount["path"]
        )
    if "mpol=%s" % policy not in normalized_options:
        raise RamdiskError(
            "tmpfs at %s has an unexpected NUMA policy"
            % mount["path"]
        )
    actual["all_options"] = sorted(options)
    return actual


def _default_cgroup_available_memory(*, discover_cgroup_memory=None):
    discover_cgroup_memory = (
        _discover_cgroup_memory
        if discover_cgroup_memory is None
        else discover_cgroup_memory
    )
    cgroup = discover_cgroup_memory()
    if cgroup.get("error"):
        raise RamdiskError(
            "cannot validate cgroup memory headroom: %s"
            % cgroup["error"]
        )
    return cgroup.get("available_bytes")


def _available_memory(*, meminfo=None, cgroup_available_memory=None):
    meminfo = _meminfo if meminfo is None else meminfo
    cgroup_available_memory = (
        _default_cgroup_available_memory
        if cgroup_available_memory is None
        else cgroup_available_memory
    )
    values = meminfo()
    available = values.get(
        "MemAvailable",
        values.get("MemFree", 0),
    )
    cgroup_available = cgroup_available_memory()
    return (
        min(available, cgroup_available)
        if cgroup_available is not None
        else available
    )


def _host_available_for_mount(
    mount,
    plan=None,
    *,
    meminfo=None,
    node_meminfo=None,
):
    """Return host/NUMA availability without shared cgroup headroom."""
    meminfo = _meminfo if meminfo is None else meminfo
    node_meminfo = (
        _node_meminfo
        if node_meminfo is None
        else node_meminfo
    )
    if mount.get("node") is None:
        nodes = (plan or {}).get("placement", {}).get(
            "memory_nodes"
        )
        if nodes:
            available = 0
            for node in nodes:
                values = node_meminfo(int(node))
                available += values.get(
                    "MemFree",
                    values.get("MemAvailable", 0),
                )
            return available
        values = meminfo()
        return values.get(
            "MemAvailable",
            values.get("MemFree", 0),
        )
    values = node_meminfo(int(mount["node"]))
    return values.get(
        "MemFree",
        values.get("MemAvailable", 0),
    )


def _available_for_mount(
    mount,
    plan=None,
    *,
    host_available_for_mount=None,
    cgroup_available_memory=None,
):
    host_available_for_mount = (
        _host_available_for_mount
        if host_available_for_mount is None
        else host_available_for_mount
    )
    cgroup_available_memory = (
        _default_cgroup_available_memory
        if cgroup_available_memory is None
        else cgroup_available_memory
    )
    available = host_available_for_mount(mount, plan=plan)
    cgroup_available = cgroup_available_memory()
    return (
        min(available, cgroup_available)
        if cgroup_available is not None
        else available
    )


def _copy_stream(src, tmp, expected_size, cancel_event=None):
    source_fd = os.open(src, os.O_RDONLY)
    try:
        destination_fd = os.open(
            tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o400,
        )
        try:
            copied = 0
            while copied < expected_size:
                _raise_if_cancelled(cancel_event)
                data = os.read(
                    source_fd,
                    min(8 * MIB, expected_size - copied),
                )
                if not data:
                    raise RamdiskError(
                        "source shard was truncated while copying: %s"
                        % src
                    )
                view = memoryview(data)
                while view:
                    written = os.write(destination_fd, view)
                    if written <= 0:
                        raise RamdiskError(
                            "short write while staging %s" % src
                        )
                    view = view[written:]
                copied += len(data)
            if os.read(source_fd, 1):
                raise RamdiskError(
                    "source shard grew while copying: %s" % src
                )
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
        if hasattr(os, "posix_fadvise") and hasattr(
            os,
            "POSIX_FADV_DONTNEED",
        ):
            try:
                os.posix_fadvise(
                    source_fd,
                    0,
                    0,
                    os.POSIX_FADV_DONTNEED,
                )
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
        raise RamdiskError(
            "available memory reached the protected reserve before %s"
            % os.path.basename(src)
        )
    tmp = destination + ".coli-copy-%d-%s" % (
        os.getpid(),
        secrets.token_hex(4),
    )
    started = time.monotonic()
    try:
        _copy_stream(
            src,
            tmp,
            expected_size,
            cancel_event=cancel_event,
        )
        os.chmod(tmp, 0o400)
        if os.path.getsize(tmp) != expected_size:
            raise RamdiskError(
                "staged size mismatch for %s"
                % os.path.basename(src)
            )
        _read_safetensors_header(tmp)
        os.replace(tmp, destination)
        if progress:
            progress(
                os.path.basename(src),
                expected_size,
                time.monotonic() - started,
            )
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
    *,
    run=None,
    worker_entrypoint=None,
):
    if not get_platform_ops().is_linux:
        raise RamdiskError(UNSUPPORTED_PLATFORM_REASON)
    run = _run if run is None else run
    if worker_entrypoint is None:
        raise RamdiskError("copy worker entrypoint is unavailable")
    available = available or _available_memory
    _raise_if_cancelled(cancel_event)
    if available() < reserve_floor:
        raise RamdiskError(
            "available memory reached the protected reserve before replica copy"
        )
    tmp = destination + ".coli-copy-%d-%s" % (
        os.getpid(),
        secrets.token_hex(4),
    )
    started = time.monotonic()
    command = [
        numactl,
        "--physcpubind=%s" % cpu_list,
        "--membind=%d" % node,
        sys.executable,
        worker_entrypoint,
        "--copy-worker",
        src,
        tmp,
        str(expected_size),
    ]
    try:
        if cancel_event is None:
            result = run(command)
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
            raise RamdiskError(
                "node-affined replica copy failed: %s"
                % (
                    result.stderr.strip()
                    or result.stdout.strip()
                )
            )
        if os.path.getsize(tmp) != expected_size:
            raise RamdiskError(
                "replica size mismatch for %s"
                % os.path.basename(src)
            )
        _read_safetensors_header(tmp)
        os.replace(tmp, destination)
        if progress:
            progress(
                os.path.basename(src),
                expected_size,
                time.monotonic() - started,
            )
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _populate_mount(
    plan,
    mount,
    source_root=None,
    progress=None,
    cancel_event=None,
    *,
    available_for_mount=None,
    copy_one=None,
    copy_one_affined=None,
    engine_cpu_list=None,
):
    available_for_mount = (
        _available_for_mount
        if available_for_mount is None
        else available_for_mount
    )
    copy_one = _copy_one if copy_one is None else copy_one
    copy_one_affined = (
        _copy_one_affined
        if copy_one_affined is None
        else copy_one_affined
    )
    source_root = source_root or plan["model"]["path"]
    selected = plan["staging"]["selected_shards"]
    linked = plan["staging"]["linked_shards"]
    identities = {
        item["name"]: item
        for item in plan["source_shards"]
    }
    if mount.get("node") is None:
        reserve_floor = (
            plan["reserve"]["runtime_bytes"]
            + plan["reserve"]["page_table_bytes"]
            + plan["reserve"]["os_margin_bytes"]
        )
    else:
        node_info = next(
            item
            for item in plan["hardware"]["nodes"]
            if item["id"] == mount["node"]
        )
        reserve_floor = (
            plan["reserve"]["runtime_bytes"]
            + plan["reserve"]["page_table_bytes"]
            + node_info.get("reserve_bytes", 8 * GIB)
        )

    def available():
        return available_for_mount(mount, plan=plan)

    workers = max(
        1,
        min(plan["parallel"], len(selected) or 1),
    )
    admission_lock = threading.Lock()
    inflight = [0]

    def copy_name(name):
        _raise_if_cancelled(cancel_event)
        source = os.path.join(source_root, name)
        destination = os.path.join(mount["path"], name)
        expected = identities[name]["size_bytes"]
        with admission_lock:
            observed = available()
            if (
                observed - inflight[0] - expected
                < reserve_floor
            ):
                raise RamdiskError(
                    "projected shard copies would breach the protected "
                    "memory reserve"
                )
            inflight[0] += expected
        try:
            if (
                source_root != plan["model"]["path"]
                and mount["node"] is not None
            ):
                if engine_cpu_list is None:
                    raise RamdiskError(
                        "node-affined copy CPU selection is unavailable"
                    )
                return copy_one_affined(
                    source,
                    destination,
                    expected,
                    mount["node"],
                    plan["hardware"]["numactl"],
                    engine_cpu_list(plan, node=mount["node"]),
                    reserve_floor,
                    progress,
                    available,
                    cancel_event,
                )
            return copy_one(
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

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        futures = [
            executor.submit(copy_name, name)
            for name in selected
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    _raise_if_cancelled(cancel_event)
    for name in linked:
        _raise_if_cancelled(cancel_event)
        target = os.path.join(plan["model"]["path"], name)
        destination = os.path.join(mount["path"], name)
        os.symlink(target, destination)


def _mix_sample_value(value):
    """Return a stable, well-distributed unsigned 64-bit value."""
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = (
        (value ^ (value >> 30))
        * 0xBF58476D1CE4E5B9
    ) & mask
    value = (
        (value ^ (value >> 27))
        * 0x94D049BB133111EB
    ) & mask
    return value ^ (value >> 31)


def _sample_page_indices(total_pages, sample_pages, node_count):
    sample_pages = max(1, min(sample_pages, total_pages))
    if sample_pages == total_pages:
        return list(range(total_pages))
    node_count = max(1, node_count)
    if node_count == 1:
        return [
            ((2 * sample + 1) * total_pages)
            // (2 * sample_pages)
            for sample in range(sample_pages)
        ]

    eligible_orders = [
        order
        for order in range(10)
        if (
            total_pages + (1 << order) - 1
        ) // (1 << order) >= 7 * node_count
    ]
    residue_counts = {
        order: [0] * node_count
        for order in eligible_orders
    }
    seed = _mix_sample_value(
        total_pages
        ^ (sample_pages << 32)
        ^ node_count
    )
    indices = []
    for sample in range(sample_pages):
        lower = sample * total_pages // sample_pages
        upper = (sample + 1) * total_pages // sample_pages
        target = float(sample + 1) / node_count
        best = None
        for salt in range(32):
            value = lower + (
                _mix_sample_value(
                    seed + sample * 32 + salt
                )
                % (upper - lower)
            )
            maximum_distance = 0.0
            squared_distance = 0.0
            for order in eligible_orders:
                residue = (value >> order) % node_count
                for node, count in enumerate(
                    residue_counts[order]
                ):
                    projected = count + (
                        1 if node == residue else 0
                    )
                    distance = abs(projected - target)
                    maximum_distance = max(
                        maximum_distance,
                        distance,
                    )
                    squared_distance += distance * distance
            score = (
                maximum_distance,
                squared_distance,
                salt,
            )
            if best is None or score < best[0]:
                best = (score, value)
        value = best[1]
        indices.append(value)
        for order in eligible_orders:
            residue_counts[order][
                (value >> order) % node_count
            ] += 1
    return indices


def _sample_numa_allocation(
    path,
    max_pages=1024,
    node_count=1,
):
    """Touch a bounded page sample and report its Linux NUMA nodes."""
    if not get_platform_ops().is_linux:
        raise RamdiskError(UNSUPPORTED_PLATFORM_REASON)
    counts = {}
    size = os.path.getsize(path)
    if size <= 0:
        return counts
    with open(path, "rb") as stream:
        with mmap.mmap(
            stream.fileno(),
            0,
            access=mmap.ACCESS_READ,
        ) as mapping:
            total_pages = max(1, (size + 4095) // 4096)
            pages = max(1, min(max_pages, total_pages))
            for page in _sample_page_indices(
                total_pages,
                pages,
                node_count,
            ):
                mapping[min(size - 1, page * 4096)]
            needle = "file=" + path.replace(" ", "\\040")
            from .linux_ops import _read_text

            for line in _read_text(
                "/proc/self/numa_maps"
            ).splitlines():
                if needle not in line and path not in line:
                    continue
                for node, value in re.findall(
                    r"\bN(\d+)=(\d+)\b",
                    line,
                ):
                    counts[node] = (
                        counts.get(node, 0) + int(value)
                    )
    return counts


def _validate_namespace(
    plan,
    mount,
    sample_numa=True,
    *,
    sample_numa_allocation=None,
):
    sample_numa_allocation = (
        _sample_numa_allocation
        if sample_numa_allocation is None
        else sample_numa_allocation
    )
    expected_names = sorted(
        item["name"]
        for item in plan["source_shards"]
    )
    actual_names = sorted(
        name
        for name in os.listdir(mount["path"])
        if name.endswith(".safetensors")
    )
    if actual_names != expected_names:
        raise RamdiskError(
            "staged namespace filenames do not match the canonical model"
        )
    identities = {
        item["name"]: item
        for item in plan["source_shards"]
    }
    selected = set(plan["staging"]["selected_shards"])
    linked = set(plan["staging"]["linked_shards"])
    for name in expected_names:
        path = os.path.join(mount["path"], name)
        if name in selected:
            if (
                os.path.islink(path)
                or not stat.S_ISREG(os.stat(path).st_mode)
            ):
                raise RamdiskError(
                    "staged shard is not a regular tmpfs file: %s"
                    % name
                )
            if os.path.getsize(path) != identities[name]["size_bytes"]:
                raise RamdiskError(
                    "staged shard size mismatch: %s" % name
                )
            if os.stat(path).st_mode & 0o222:
                raise RamdiskError(
                    "staged shard is writable: %s" % name
                )
            raw, _ = _read_safetensors_header(path)
            if (
                hashlib.sha256(raw).hexdigest()
                != identities[name]["header_sha256"]
            ):
                raise RamdiskError(
                    "staged shard header mismatch: %s" % name
                )
        elif name in linked:
            if not os.path.islink(path):
                raise RamdiskError(
                    "unstaged shard is not an SSD fallback symlink: %s"
                    % name
                )
            canonical = os.path.join(
                plan["model"]["path"],
                name,
            )
            if os.path.realpath(path) != os.path.realpath(canonical):
                raise RamdiskError(
                    "fallback symlink does not target the canonical "
                    "shard: %s" % name
                )
    allocation = {}
    if not sample_numa:
        return allocation
    selected_names = plan["staging"]["selected_shards"]
    placement_nodes = plan.get("placement", {}).get(
        "memory_nodes",
        plan["hardware"]["online_nodes"],
    )
    pages_per_shard = max(
        32,
        min(1024, 4096 // max(1, len(selected_names))),
    )
    for name in selected_names:
        path = os.path.join(mount["path"], name)
        for node, count in sample_numa_allocation(
            path,
            pages_per_shard,
            node_count=len(placement_nodes),
        ).items():
            allocation[node] = allocation.get(node, 0) + count
    online_nodes = plan.get("hardware", {}).get(
        "online_nodes",
        placement_nodes,
    )
    verify_numa = (
        len(placement_nodes) > 1
        or len(online_nodes) > 1
    )
    if verify_numa:
        total = sum(allocation.values())
        if not total:
            raise RamdiskError(
                "could not verify actual NUMA allocation for staged shards"
            )
        outside = sum(
            count
            for node, count in allocation.items()
            if int(node) not in placement_nodes
        )
        if float(outside) / total > 0.01:
            raise RamdiskError(
                "tmpfs sample escaped the reviewed NUMA "
                "memory-node mask"
            )
        if mount["node"] is not None:
            local = allocation.get(str(mount["node"]), 0)
            if float(local) / total < 0.95:
                raise RamdiskError(
                    "node-local tmpfs sample is below 95% local allocation"
                )
        elif len(placement_nodes) > 1:
            ideal = float(total) / len(placement_nodes)
            deviations = [
                abs(allocation.get(str(node), 0) - ideal)
                / ideal
                for node in placement_nodes
            ]
            maximum_deviation = max(deviations)
            if maximum_deviation > 0.15:
                node_pages = ", ".join(
                    "%s=%d"
                    % (
                        node,
                        allocation.get(str(node), 0),
                    )
                    for node in placement_nodes
                )
                raise RamdiskError(
                    "interleaved tmpfs sample is imbalanced: "
                    "node pages %s; maximum deviation %.1f%% "
                    "exceeds 15%%"
                    % (
                        node_pages,
                        maximum_deviation * 100.0,
                    )
                )
    return allocation


def _source_still_matches(plan, *, scan_model_fn=None):
    scan_model_fn = scan_model if scan_model_fn is None else scan_model_fn
    current = scan_model_fn(plan["model"]["path"])
    if current["fingerprint"] != plan["model"]["fingerprint"]:
        raise RamdiskError(
            "canonical model changed while staging; refusing to "
            "publish the manifest"
        )
