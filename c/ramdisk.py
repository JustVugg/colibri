"""NUMA-aware tmpfs staging and managed-engine lifecycle for ``coli ramdisk``.

The module intentionally uses only the Python standard library.  Planning and
status are unprivileged; the only privileged subprocesses are the exact mount
and unmount commands issued by :func:`prepare` and :func:`destroy`.
"""

from __future__ import print_function

import argparse
import contextlib
import copy
import functools
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from ramdisk_support.common import (
    BENCHMARK_SCHEMA,
    DEFAULT_MOUNT_ROOT,
    GIB,
    MANIFEST_VERSION,
    MIB,
    PLAN_SCHEMA,
    PROFILE_LINE_RE,
    STATUS_SCHEMA,
    TMPFS_MAGIC,
    USAGE_MERGE_RE,
    RamdiskError,
    _EngineCleanupError,
    _OperationCancelled,
    _format_range_list,
    _parse_range_list,
    _path_is_below,
    _path_without_symlinks,
    _percentile,
    _positive_int,
    _raise_if_cancelled,
    _utc_now,
)
from ramdisk_support.benchmark import (
    BENCHMARK_PROMPT,
    _aggregate_score as _benchmark_aggregate_score,
    _benchmark_environment as _benchmark_make_environment,
    _benchmark_generate as _benchmark_generate_turn,
    _cancellable_engine_type as _benchmark_cancellable_engine_type,
    _parse_profiler as _benchmark_parse_profiler,
    _score_variant as _benchmark_score_variant,
    _source_build_identity as _benchmark_source_build_identity,
    _system_score as _benchmark_system_score,
    run_benchmark as _run_benchmark,
)
from ramdisk_support.discovery import (
    _cgroup_ancestors,
    _cgroup_memberships,
    _cgroup_mounts,
    _discover_cgroup_memory,
    _mountinfo_unescape,
    _parse_cgroup_bytes,
    _resolve_cgroup_directory,
    discover_hardware as _discover_hardware,
)
from ramdisk_support.linux_ops import (
    _filesystem_for_path as _linux_filesystem_for_path,
    _fresh_user_binary,
    _kernel_at_least,
    _meminfo,
    _mount_at as _linux_mount_at,
    _mount_table,
    _node_meminfo,
    _noninteractive_privilege as _linux_noninteractive_privilege,
    _physical_cores,
    _privileged as _linux_privileged,
    _process_status as _linux_process_status,
    _read_cgroup_contract,
    _read_cgroup_value,
    _read_text,
    _run,
    _split_mount_options,
    _status_allowed_list,
    _sudo_ticket_keepalive,
    _thread_sibling_groups,
    _trusted_system_binary,
    _unescape_mount,
    _validate_noninteractive_sudo,
)
from ramdisk_support.lifecycle import (
    _assert_ready_mounts as _lifecycle_assert_ready_mounts,
    _managed_path as _lifecycle_managed_path,
    _managed_ports_for_plan as _lifecycle_managed_ports_for_plan,
    _persisted_base_port as _lifecycle_persisted_base_port,
    destroy as _lifecycle_destroy,
    prepare as _lifecycle_prepare,
    start as _lifecycle_start,
    status as _lifecycle_status,
    stop as _lifecycle_stop,
)
from ramdisk_support.mounts import (
    _available_for_mount as _mounts_available_for_mount,
    _available_memory as _mounts_available_memory,
    _busy_mount_references as _mounts_busy_mount_references,
    _copy_one as _mounts_copy_one,
    _copy_one_affined as _mounts_copy_one_affined,
    _copy_stream,
    _copy_worker_main,
    _host_available_for_mount as _mounts_host_available_for_mount,
    _mount_option_list,
    _mount_tmpfs as _mounts_mount_tmpfs,
    _option_present,
    _populate_mount as _mounts_populate_mount,
    _reusable_empty_mountpoint,
    _rollback_interrupted_mount as _mounts_rollback_interrupted_mount,
    _sample_numa_allocation,
    _sample_page_indices,
    _source_still_matches as _mounts_source_still_matches,
    _umount_path as _mounts_umount_path,
    _validate_mount as _mounts_validate_mount,
    _validate_namespace as _mounts_validate_namespace,
)
from ramdisk_support.model import (
    EXPERT_RE,
    MAX_ST_HEADER,
    _direct_tensor_set_eligible,
    _sha256_file,
    _shape_numel,
    scan_model,
)
from ramdisk_support.planning import (
    _build_placement,
    _load_profile,
    _requested_ids,
    _runtime_reserve,
    _select_partial,
)
from ramdisk_support.processes import (
    _admit_concurrent_runtimes as _processes_admit_concurrent_runtimes,
    _admit_runtime as _processes_admit_runtime,
    _assert_effective_masks_unchanged as _processes_assert_effective_masks_unchanged,
    _forget_managed_child,
    _group_alive,
    _managed_children,
    _managed_children_lock,
    _managed_process_metrics as _processes_managed_process_metrics,
    _poll_managed_child,
    _proc_identity,
    _process_group_members as _processes_process_group_members,
    _process_matches as _processes_process_matches,
    _process_tree_alive as _processes_process_tree_alive,
    _resolve_engine_path,
    _runtime_admission_requirement,
    _terminate_direct_child,
    _terminate_group as _processes_terminate_group,
    _terminate_verified_group as _processes_terminate_verified_group,
    _track_managed_child,
    _wait_managed_ready as _processes_wait_managed_ready,
)
from ramdisk_support.platform_ops import current_euid, get_platform_ops
from ramdisk_support.state import (
    _assert_canonical_usage_target as _state_assert_canonical_usage_target,
    _assert_durable_state_dir as _state_assert_durable_state_dir,
    _atomic_json,
    _benchmarks_path,
    _durable_unlink,
    _ensure_atomic_parent,
    _ensure_private_dir,
    _fsync_directory,
    _lifecycle_lock,
    _load_manifest as _state_load_manifest,
    _manifest_mount_layout,
    _manifest_path,
    _merge_usage as _state_merge_usage,
    _read_json,
    _recover_delta as _state_recover_delta,
    _save_manifest as _state_save_manifest,
    _state_root,
    _usage_merge_ids,
    _usage_read,
    _usage_write,
)
from ramdisk_ui import (
    ActionPolicy,
    DeploymentHealth,
    HealthLevel,
    PlacementContract,
    ReviewIdentity,
)


def _exclusive_lifecycle(function):
    """Keep decorated facade calls patchable while state owns the lock."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with _lifecycle_lock():
            return function(*args, **kwargs)

    return wrapped


def _assert_durable_state_dir(path, plan=None):
    return _state_assert_durable_state_dir(
        path,
        plan=plan,
        filesystem_for_path=_filesystem_for_path,
    )


def _load_manifest(required=False):
    return _state_load_manifest(
        required=required,
        filesystem_for_path=_filesystem_for_path,
        read_json=_read_json,
        manifest_path=_manifest_path,
        state_root=_state_root,
        benchmarks_path=_benchmarks_path,
        assert_durable_state_dir=_assert_durable_state_dir,
    )


def _save_manifest(manifest):
    return _state_save_manifest(
        manifest,
        atomic_json=_atomic_json,
        manifest_path=_manifest_path,
    )


def _cgroup_available_memory():
    """Read current hard cgroup headroom, failing closed on a broken contract."""
    cgroup = _discover_cgroup_memory()
    if cgroup.get("error"):
        raise RamdiskError(
            "cannot validate cgroup memory headroom: %s" % cgroup["error"]
        )
    return cgroup.get("available_bytes")


def discover_hardware():
    return _discover_hardware()












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
    invoking_euid = current_euid()
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
        elif invoking_euid != 0 and os.access(mount_root, os.W_OK):
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
                if invoking_euid != 0 and os.access(parent, os.W_OK):
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
            elif invoking_euid != 0 and os.access(path, os.W_OK):
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
            if hardware["linux"] and get_platform_ops().is_linux:
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


def _mount_at(path):
    return _linux_mount_at(path, mount_table=_mount_table)


def _filesystem_for_path(path):
    return _linux_filesystem_for_path(
        path,
        mount_table=_mount_table,
    )


@contextlib.contextmanager
def _noninteractive_privilege(keepalive=False, cancel_event=None):
    with _linux_noninteractive_privilege(
        keepalive=keepalive,
        cancel_event=cancel_event,
        trusted_system_binary=_trusted_system_binary,
        sudo_ticket_keepalive=_sudo_ticket_keepalive,
    ):
        yield


def _privileged(command, hardware):
    return _linux_privileged(
        command,
        hardware,
        trusted_system_binary=_trusted_system_binary,
    )


def _mount_tmpfs(plan, mount):
    return _mounts_mount_tmpfs(
        plan,
        mount,
        trusted_system_binary=_trusted_system_binary,
        run=_run,
        privileged=_privileged,
        rollback_interrupted_mount=_rollback_interrupted_mount,
    )


def _umount_path(path, hardware):
    return _mounts_umount_path(
        path,
        hardware,
        trusted_system_binary=_trusted_system_binary,
        run=_run,
        privileged=_privileged,
    )


def _rollback_interrupted_mount(plan, mount, effective_thp, effective_noswap, cause):
    return _mounts_rollback_interrupted_mount(
        plan,
        mount,
        effective_thp,
        effective_noswap,
        cause,
        mount_at=_mount_at,
        validate_mount=_validate_mount,
        umount_path=_umount_path,
    )


def _validate_mount(mount, plan):
    return _mounts_validate_mount(
        mount,
        plan,
        mount_at=_mount_at,
    )


def _available_memory():
    return _mounts_available_memory(
        meminfo=_meminfo,
        cgroup_available_memory=_cgroup_available_memory,
    )


def _host_available_for_mount(mount, plan=None):
    return _mounts_host_available_for_mount(
        mount,
        plan=plan,
        meminfo=_meminfo,
        node_meminfo=_node_meminfo,
    )


def _available_for_mount(mount, plan=None):
    return _mounts_available_for_mount(
        mount,
        plan=plan,
        host_available_for_mount=_host_available_for_mount,
        cgroup_available_memory=_cgroup_available_memory,
    )


def _copy_one(
    src,
    destination,
    expected_size,
    reserve_floor,
    progress=None,
    available=None,
    cancel_event=None,
):
    return _mounts_copy_one(
        src,
        destination,
        expected_size,
        reserve_floor,
        progress,
        _available_memory if available is None else available,
        cancel_event,
    )


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
    available = _available_memory if available is None else available
    return _mounts_copy_one_affined(
        src,
        destination,
        expected_size,
        node,
        numactl,
        cpu_list,
        reserve_floor,
        progress,
        available,
        cancel_event,
        run=_run,
        worker_entrypoint=os.path.abspath(__file__),
    )


def _populate_mount(plan, mount, source_root=None, progress=None, cancel_event=None):
    return _mounts_populate_mount(
        plan,
        mount,
        source_root=source_root,
        progress=progress,
        cancel_event=cancel_event,
        available_for_mount=_available_for_mount,
        copy_one=_copy_one,
        copy_one_affined=_copy_one_affined,
        engine_cpu_list=_engine_cpu_list,
    )


def _validate_namespace(plan, mount, sample_numa=True):
    return _mounts_validate_namespace(
        plan,
        mount,
        sample_numa=sample_numa,
        sample_numa_allocation=_sample_numa_allocation,
    )


def _source_still_matches(plan):
    return _mounts_source_still_matches(
        plan,
        scan_model_fn=scan_model,
    )


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
    return _lifecycle_prepare(
        args,
        progress=progress,
        display_plan=display_plan,
        expected_plan_token=expected_plan_token,
        cancel_event=cancel_event,
        load_manifest=_load_manifest,
        build_plan=build_plan,
        managed_ports_for_plan=_managed_ports_for_plan,
        plan_confirmation_token=_plan_confirmation_token,
        render_plan=_human_plan,
        confirm=_confirm,
        save_manifest=_save_manifest,
        mount_at=_mount_at,
        mount_tmpfs=_mount_tmpfs,
        umount_path=_umount_path,
        validate_mount=_validate_mount,
        populate_mount=_populate_mount,
        validate_namespace=_validate_namespace,
        source_still_matches=_source_still_matches,
        durable_unlink=_durable_unlink,
        manifest_path=_manifest_path,
    )


def _process_group_members(pgid):
    return _processes_process_group_members(
        pgid,
        proc_identity=_proc_identity,
    )


def _process_matches(record):
    return _processes_process_matches(
        record,
        proc_identity=_proc_identity,
        process_group_members=_process_group_members,
    )


def _process_tree_alive(record, actual):
    return _processes_process_tree_alive(
        record,
        actual,
        group_alive=_group_alive,
        proc_identity=_proc_identity,
    )


def _assert_canonical_usage_target(canonical_path, plan=None):
    return _state_assert_canonical_usage_target(
        canonical_path,
        plan=plan,
        source_still_matches=_source_still_matches,
    )


def _merge_usage(record, canonical_path, plan=None, keep_journal=False):
    return _state_merge_usage(
        record,
        canonical_path,
        plan=plan,
        keep_journal=keep_journal,
        filesystem_for_path=_filesystem_for_path,
        source_still_matches=_source_still_matches,
    )


def _recover_delta(state_dir, canonical_path, plan=None):
    return _state_recover_delta(
        state_dir,
        canonical_path,
        plan=plan,
        filesystem_for_path=_filesystem_for_path,
        source_still_matches=_source_still_matches,
    )


def _assert_ready_mounts(manifest):
    return _lifecycle_assert_ready_mounts(
        manifest,
        source_still_matches=_source_still_matches,
        validate_mount=_validate_mount,
        validate_namespace=_validate_namespace,
    )


def _admit_runtime(plan, mount, benchmark=False):
    return _processes_admit_runtime(
        plan,
        mount,
        benchmark=benchmark,
        available_for_mount=_available_for_mount,
    )


def _admit_concurrent_runtimes(plan, mounts, benchmark=False):
    return _processes_admit_concurrent_runtimes(
        plan,
        mounts,
        benchmark=benchmark,
        host_available_for_mount=_host_available_for_mount,
        cgroup_available_memory=_cgroup_available_memory,
    )


def _assert_effective_masks_unchanged(plan):
    return _processes_assert_effective_masks_unchanged(
        plan,
        discover_hardware=discover_hardware,
    )


def _terminate_group(pgid, term_seconds=10.0, kill_seconds=3.0):
    return _processes_terminate_group(
        pgid,
        term_seconds=term_seconds,
        kill_seconds=kill_seconds,
        group_alive=_group_alive,
    )


def _terminate_verified_group(record, term_seconds=10.0, kill_seconds=3.0):
    return _processes_terminate_verified_group(
        record,
        term_seconds=term_seconds,
        kill_seconds=kill_seconds,
        poll_managed_child=_poll_managed_child,
        process_matches=_process_matches,
    )


def _wait_managed_ready(record, timeout, api_key=None, cancel_event=None):
    return _processes_wait_managed_ready(
        record,
        timeout,
        api_key=api_key,
        cancel_event=cancel_event,
        process_matches=_process_matches,
        urlopen=urllib.request.urlopen,
    )


@_exclusive_lifecycle
def start(args, cli_path=None, engine_path=None, cancel_event=None):
    return _lifecycle_start(
        args,
        cli_path=cli_path,
        engine_path=engine_path,
        cancel_event=cancel_event,
        default_cli_path=os.path.join(os.path.dirname(__file__), "coli"),
        load_manifest=_load_manifest,
        assert_effective_masks_unchanged=_assert_effective_masks_unchanged,
        assert_ready_mounts=_assert_ready_mounts,
        process_matches=_process_matches,
        save_manifest=_save_manifest,
        merge_usage=_merge_usage,
        persisted_base_port=_persisted_base_port,
        fresh_user_binary=_fresh_user_binary,
        admit_concurrent_runtimes=_admit_concurrent_runtimes,
        state_root=_state_root,
        ensure_private_dir=_ensure_private_dir,
        assert_durable_state_dir=_assert_durable_state_dir,
        recover_delta=_recover_delta,
        usage_read=_usage_read,
        usage_write=_usage_write,
        managed_numa_enabled=_managed_numa_enabled,
        memory_node_list=_memory_node_list,
        engine_cpu_list=_engine_cpu_list,
        node_core_count=_node_core_count,
        normalized_runtime_knobs=_normalized_runtime_knobs,
        proc_identity=_proc_identity,
        wait_managed_ready=_wait_managed_ready,
        track_managed_child=_track_managed_child,
        terminate_verified_group=_terminate_verified_group,
        terminate_direct_child=_terminate_direct_child,
        forget_managed_child=_forget_managed_child,
    )


@_exclusive_lifecycle
def stop(args=None):
    return _lifecycle_stop(
        args,
        load_manifest=_load_manifest,
        process_matches=_process_matches,
        save_manifest=_save_manifest,
        terminate_verified_group=_terminate_verified_group,
        merge_usage=_merge_usage,
    )


def _busy_mount_references(path):
    return _mounts_busy_mount_references(path)


def _managed_path(path, mount_root):
    return _lifecycle_managed_path(path, mount_root)


@_exclusive_lifecycle
def destroy(args, expected_manifest_token=None):
    return _lifecycle_destroy(
        args,
        expected_manifest_token=expected_manifest_token,
        load_manifest=_load_manifest,
        manifest_confirmation_token=_manifest_confirmation_token,
        confirm=_confirm,
        stop_action=stop,
        mount_table=_mount_table,
        path_is_below=_path_is_below,
        managed_path=_managed_path,
        mount_at=_mount_at,
        validate_mount=_validate_mount,
        validate_namespace=_validate_namespace,
        busy_mount_references=_busy_mount_references,
        umount_path=_umount_path,
        durable_unlink=_durable_unlink,
        manifest_path=_manifest_path,
    )


def status(deep=True):
    """Return lifecycle status, optionally skipping shard/header revalidation.

    Scriptable ``status`` always uses the deep default.  The curses dashboard
    polls the cheap form and exposes an explicit refresh for a new deep model
    scan, avoiding repeated reads of every safetensors header on large models.
    """
    return _lifecycle_status(
        deep=deep,
        load_manifest=_load_manifest,
        manifest_path=_manifest_path,
        source_still_matches=_source_still_matches,
        mount_at=_mount_at,
        validate_mount=_validate_mount,
        validate_namespace=_validate_namespace,
        process_matches=_process_matches,
    )


def _source_build_identity():
    return _benchmark_source_build_identity(
        __file__,
        environ=os.environ,
        which=shutil.which,
        run=_run,
    )


def _parse_profiler(text, elapsed):
    return _benchmark_parse_profiler(text, elapsed)


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
    return _benchmark_make_environment(
        manifest,
        weights_dir,
        state_dir,
        rammap,
        node=node,
        knobs=knobs,
        environ=os.environ,
        node_core_count=_node_core_count,
        engine_cpu_list=_engine_cpu_list,
        memory_node_list=_memory_node_list,
        managed_numa_enabled=_managed_numa_enabled,
        normalized_runtime_knobs=_normalized_runtime_knobs,
    )


def _cancellable_engine_type(
    engine_type,
    read_engine_turn,
    ready_marker,
    cancel_event,
):
    return _benchmark_cancellable_engine_type(
        engine_type,
        read_engine_turn,
        ready_marker,
        cancel_event,
    )


def _benchmark_generate(
    engine,
    prompt,
    on_text,
    cancel_event,
    client_cancelled_type,
):
    return _benchmark_generate_turn(
        engine,
        prompt,
        on_text,
        cancel_event,
        client_cancelled_type,
    )


def _score_variant(
    engine_path,
    manifest,
    name,
    weights_dir,
    rammap,
    knobs,
    cancel_event=None,
):
    return _benchmark_score_variant(
        engine_path,
        manifest,
        name,
        weights_dir,
        rammap,
        knobs,
        cancel_event=cancel_event,
        state_root=_state_root,
        ensure_private_dir=_ensure_private_dir,
        assert_durable_state_dir=_assert_durable_state_dir,
        admit_runtime=_admit_runtime,
        fresh_user_binary=_fresh_user_binary,
        engine_cpu_list=_engine_cpu_list,
        benchmark_environment=_benchmark_environment,
        cancellable_engine_type=_cancellable_engine_type,
        benchmark_generate=_benchmark_generate,
    )


def _aggregate_score(manifest, engine_path=None, knobs=None, cancel_event=None):
    return _benchmark_aggregate_score(
        manifest,
        engine_path=engine_path,
        knobs=knobs,
        cancel_event=cancel_event,
        state_root=_state_root,
        ensure_private_dir=_ensure_private_dir,
        assert_durable_state_dir=_assert_durable_state_dir,
        admit_concurrent_runtimes=_admit_concurrent_runtimes,
        fresh_user_binary=_fresh_user_binary,
        engine_cpu_list=_engine_cpu_list,
        normalized_runtime_knobs=_normalized_runtime_knobs,
        benchmark_environment=_benchmark_environment,
        cancellable_engine_type=_cancellable_engine_type,
        benchmark_generate=_benchmark_generate,
    )


def _system_score(manifest, variants, swap_before, swap_after, aggregate=None):
    return _benchmark_system_score(
        manifest,
        variants,
        swap_before,
        swap_after,
        aggregate=aggregate,
        meminfo=_meminfo,
        statvfs=getattr(os, "statvfs", None),
    )


@_exclusive_lifecycle
def benchmark(args, cli_path=None, engine_path=None, cancel_event=None):
    cli_path = cli_path or os.path.join(os.path.dirname(__file__), "coli")
    return _run_benchmark(
        args,
        cli_path,
        engine_path=engine_path,
        cancel_event=cancel_event,
        load_manifest=_load_manifest,
        assert_effective_masks_unchanged=_assert_effective_masks_unchanged,
        assert_ready_mounts=_assert_ready_mounts,
        resolve_engine_path=_resolve_engine_path,
        node_core_count=_node_core_count,
        score_variant=_score_variant,
        discover_hardware=discover_hardware,
        aggregate_score=_aggregate_score,
        system_score=_system_score,
        filesystem_for_path=_filesystem_for_path,
        source_build_identity=_source_build_identity,
        read_json=_read_json,
        benchmarks_path=_benchmarks_path,
        atomic_json=_atomic_json,
        save_manifest=_save_manifest,
        argv=sys.argv,
    )


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
    return _processes_managed_process_metrics(
        record,
        process_matches=_process_matches,
        process_group_members=_process_group_members,
        process_status=lambda pid: _linux_process_status(
            pid,
            read_text=_read_text,
        ),
    )


def _managed_ports_for_plan(plan, base_port=8000):
    return _lifecycle_managed_ports_for_plan(
        plan,
        base_port=base_port,
    )


def _persisted_base_port(manifest):
    """Recover the last base port, including manifests predating that field."""
    return _lifecycle_persisted_base_port(manifest)


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
        if current_euid() == 0:
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
