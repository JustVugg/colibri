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
import importlib
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
import threading
from ramdisk_support.common import (
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
    _TerminationSignal,
    _format_range_list,
    _normalized_runtime_knobs as _common_normalized_runtime_knobs,
    _parse_range_list,
    _path_is_below,
    _path_without_symlinks,
    _percentile,
    _positive_int,
    _raise_if_cancelled,
    _utc_now,
)
from ramdisk_support.cli import (
    _add_lifecycle_options as _cli_add_lifecycle_options,
    _cli_exit_after_signal,
    _cli_termination_guard,
    _confirm as _cli_confirm,
    _json_print as _cli_json_print,
    configure_parser as _cli_configure_parser,
    dispatch as _cli_dispatch,
)
from ramdisk_support.accelerator import (
    ACCELERATOR_ENVIRONMENT_KEYS,
    GPU_LAYOUT_CHOICES,
    GPU_LAYOUT_DENSE_ATTENTION,
    GPU_LAYOUT_DENSE_ATTENTION_SHARDED,
    GPU_LAYOUT_EXPERTS_ONLY,
    GPU_VRAM_RESERVE_BYTES,
    _apply_managed_accelerator_environment,
    _managed_accelerator_contract,
    _managed_accelerator_environment,
    apply_gpu_selection,
    eligible_gpu_devices,
    gpu_device_eligibility,
)
from ramdisk_support.discovery import (
    _cgroup_ancestors,
    _cgroup_memberships,
    _cgroup_mounts,
    _discover_cgroup_memory,
    _discover_gpus,
    _mountinfo_unescape,
    _parse_cgroup_bytes,
    _resolve_cgroup_directory,
    discover_hardware as _discover_hardware,
)
from ramdisk_support.linux_ops import (
    _ensure_busy_mount_scan_available as _linux_ensure_busy_mount_scan_available,
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
    _default_cgroup_available_memory as _mounts_default_cgroup_available_memory,
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
    _engine_cpu_list as _planning_engine_cpu_list,
    _load_profile,
    _managed_numa_enabled as _planning_managed_numa_enabled,
    _memory_node_list as _planning_memory_node_list,
    _node_core_count as _planning_node_core_count,
    _requested_ids,
    _runtime_reserve,
    _select_partial,
    build_plan as _planning_build_plan,
)
from ramdisk_support.presets import (
    PRESET_CHOICES,
    PRESET_GPU_FASTEST,
    PRESET_MINIMAL,
    PRESET_REPLICAS,
    PRESET_SINGLE,
    _engine_cuda_capable,
    mark_preset_custom,
    resolve_preset as _resolve_preset,
)
from ramdisk_support.platform_ops import (
    UNSUPPORTED_PLATFORM_REASON,
    current_euid,
    current_uid,
    get_platform_ops,
)
from ramdisk_support.state import (
    _assert_canonical_usage_target as _state_assert_canonical_usage_target,
    _assert_durable_state_dir as _state_assert_durable_state_dir,
    _atomic_json,
    _benchmarks_path,
    _bind_usage_transaction as _state_bind_usage_transaction,
    _canonical_usage_read as _state_canonical_usage_read,
    _durable_unlink,
    _ensure_atomic_parent,
    _ensure_private_dir,
    _fsync_directory,
    _lifecycle_lock,
    _load_manifest as _state_load_manifest,
    _manifest_mount_layout,
    _manifest_path,
    _managed_usage_write as _state_managed_usage_write,
    _merge_usage as _state_merge_usage,
    _read_json,
    _recover_delta as _state_recover_delta,
    _save_manifest as _state_save_manifest,
    _state_root,
    _usage_merge_ids,
    _usage_journal_transaction_id as _state_usage_journal_transaction_id,
    _usage_read as _state_usage_read,
    _usage_write as _state_usage_write,
    _validate_usage_for_plan,
)
from ramdisk_support.tokens import (
    deployment_token as _deployment_token,
    plan_token as _plan_token,
    validate_token as _validate_token,
)


def _urllib_module():
    # Importing urllib.request initializes ssl and the HTTP stack.  Keep that
    # work behind the historical ``ramdisk.urllib.request`` compatibility seam
    # instead of imposing it on every control-plane import.
    importlib.import_module("urllib.request")
    return importlib.import_module("urllib")


def _presentation_module():
    # presentation is a core headless module and is always present; it must
    # never be routed through the optional-module seam.
    return importlib.import_module("ramdisk_support.presentation")


def _processes_module():
    return importlib.import_module("ramdisk_support.processes")


def _benchmark_module():
    return importlib.import_module("ramdisk_support.benchmark")


def _benchmark_workspace_manager():
    """Build the durable PR3 scratch-workspace owner from facade seams."""
    return _benchmark_module().DurableWorkspaceManager(
        load_manifest=_load_manifest,
        save_manifest=_save_manifest,
        state_root=_state_root,
        ensure_private_dir=_ensure_private_dir,
        assert_durable_state_dir=_assert_durable_state_dir,
        mount_at=_mount_at,
        mount_table=_mount_table,
        path_is_below=_path_is_below,
        busy_mount_references=_busy_mount_references,
        mount_tmpfs=_mount_tmpfs,
        validate_mount=_validate_mount,
        populate_mount=_populate_mount,
        validate_namespace=_validate_namespace,
        source_still_matches=_source_still_matches,
        umount_path=_umount_path,
        available_for_mount=_available_for_mount,
    )


_LAZY_ATTRIBUTES = {
    "urllib": (_urllib_module, None),
    "_managed_children": (_processes_module, "_managed_children"),
    "_managed_children_lock": (
        _processes_module,
        "_managed_children_lock",
    ),
    "_runtime_admission_requirement": (
        _processes_module,
        "_runtime_admission_requirement",
    ),
    "BENCHMARK_SCHEMA": (_benchmark_module, "BENCHMARK_SCHEMA"),
    "BENCHMARK_PROMPT": (_benchmark_module, "BENCHMARK_PROMPT"),
    "_benchmark_environment": (_benchmark_module, "_benchmark_environment"),
    "_benchmark_generate": (_benchmark_module, "_benchmark_generate"),
    "_cancellable_engine_type": (
        _benchmark_module,
        "_cancellable_engine_type",
    ),
    "_parse_profiler": (_benchmark_module, "_parse_profiler"),
    "_source_build_identity": (
        _benchmark_module,
        "_source_build_identity",
    ),
}


def __getattr__(name):
    try:
        loader, attribute = _LAZY_ATTRIBUTES[name]
    except KeyError:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    loaded = loader()
    value = loaded if attribute is None else getattr(loaded, attribute)
    globals()[name] = value
    return value


def _proc_identity(pid, *, ops=None):
    if ops is not None:
        return ops.process_identity(pid)
    return _processes_module()._proc_identity(pid)


def _group_alive(pgid, *, ops=None):
    if ops is not None:
        return ops.process_group_alive(pgid)
    return _processes_module()._group_alive(pgid)


def _managed_launch_processes(
    nonce,
    uid,
    *,
    state_dir=None,
    weights_dir=None,
    not_before_starttime=None,
    launcher_pid=None,
    launcher_starttime=None,
    launcher_cmdline=None,
    expected_command=None,
):
    return get_platform_ops().managed_launch_processes(
        nonce,
        uid,
        state_dir=state_dir,
        weights_dir=weights_dir,
        not_before_starttime=not_before_starttime,
        launcher_pid=launcher_pid,
        launcher_starttime=launcher_starttime,
        launcher_cmdline=launcher_cmdline,
        expected_command=expected_command,
    )


def _process_start_boundary():
    return get_platform_ops().process_start_boundary()


def _current_process_identity():
    return get_platform_ops().process_identity(os.getpid())


def _poll_managed_child(pid):
    return _processes_module()._poll_managed_child(pid)


def _managed_child_liveness(pid):
    return _processes_module()._managed_child_liveness(pid)


def _track_managed_child(process):
    return _processes_module()._track_managed_child(process)


def _forget_managed_child(pid):
    return _processes_module()._forget_managed_child(pid)


def _terminate_direct_child(*args, **kwargs):
    return _processes_module()._terminate_direct_child(*args, **kwargs)


def _resolve_engine_path(*args, **kwargs):
    return _processes_module()._resolve_engine_path(*args, **kwargs)


def _exclusive_lifecycle(function=None, *, require_process_control=False):
    """Keep decorated facade calls patchable while state owns the lock."""
    if function is None:
        return functools.partial(
            _exclusive_lifecycle,
            require_process_control=require_process_control,
        )

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        ops = get_platform_ops()
        if not getattr(ops, "is_linux", False):
            raise RamdiskError(UNSUPPORTED_PLATFORM_REASON)
        if require_process_control and not getattr(
            ops,
            "process_control_supported",
            False,
        ):
            reason = getattr(
                ops,
                "process_control_reason",
                UNSUPPORTED_PLATFORM_REASON,
            )
            if not isinstance(reason, str) or not reason:
                reason = UNSUPPORTED_PLATFORM_REASON
            raise RamdiskError(reason)
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
    return _mounts_default_cgroup_available_memory(
        discover_cgroup_memory=_discover_cgroup_memory,
    )


def discover_hardware():
    return _discover_hardware()












def build_plan(args, hardware=None, model=None):
    return _planning_build_plan(
        args,
        hardware=hardware,
        model=model,
        discover_hardware=discover_hardware,
        scan_model=scan_model,
        load_profile=_load_profile,
        select_partial=_select_partial,
        runtime_reserve=_runtime_reserve,
        build_placement=_build_placement,
        reusable_empty_mountpoint=_reusable_empty_mountpoint,
        filesystem_for_path=_filesystem_for_path,
        state_root=_state_root,
        manifest_path=_manifest_path,
        benchmarks_path=_benchmarks_path,
        current_euid=current_euid,
        get_platform_ops=get_platform_ops,
    )


def resolve_preset(
    preset_id,
    args,
    hardware=None,
    model=None,
    cli_path=None,
    engine_path=None,
):
    hardware = hardware or discover_hardware()
    model = model or scan_model(args.model)
    cuda_capable = None
    if preset_id == PRESET_GPU_FASTEST:
        try:
            resolved_engine = _resolve_engine_path(
                cli_path or os.path.join(os.path.dirname(__file__), "coli"),
                engine_path=engine_path,
            )
        except RamdiskError:
            resolved_engine = engine_path
        cuda_capable = _engine_cuda_capable(resolved_engine)
    return _resolve_preset(
        preset_id,
        args,
        hardware=hardware,
        model=model,
        build_plan=build_plan,
        load_profile=_load_profile,
        cuda_capable=cuda_capable,
    )


def _add_lifecycle_options(parser, suppress=False):
    return _cli_add_lifecycle_options(
        parser,
        suppress=suppress,
    )


def configure_parser(parser, common_parent=None):
    return _cli_configure_parser(
        parser,
        common_parent=common_parent,
    )


def _json_print(value):
    return _cli_json_print(value)


def _human_plan(plan):
    return _presentation_module()._human_plan(plan)


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
    return _cli_confirm(
        message,
        accepted=accepted,
    )


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
        ensure_busy_mount_scan_available=(
            _ensure_busy_mount_scan_available
        ),
        durable_unlink=_durable_unlink,
        manifest_path=_manifest_path,
        mount_table=_mount_table,
        path_is_below=_path_is_below,
        busy_mount_references=_busy_mount_references,
    )


def stage(
    args,
    progress=None,
    display_plan=True,
    expected_plan_token=None,
    cancel_event=None,
):
    """Preflight and stage exactly the plan identified by a public token."""
    token = _validate_token(expected_plan_token, "plan token")
    reviewed_plan = build_plan(args)
    if _plan_confirmation_token(reviewed_plan) != token:
        raise RamdiskError(
            "RAM-disk plan changed since review; inspect the updated plan "
            "and confirm again"
        )
    # ``prepare`` acquires the lifecycle lock and rebuilds the plan.  Passing
    # the same expected token keeps the donor's under-lock TOCTOU check.
    return prepare(
        args,
        progress=progress,
        display_plan=display_plan,
        expected_plan_token=token,
        cancel_event=cancel_event,
    )


def _process_group_members(pgid):
    return _processes_module()._process_group_members(
        pgid,
        proc_identity=_proc_identity,
    )


def _process_matches(record):
    return _processes_module()._process_matches(
        record,
        proc_identity=_proc_identity,
        process_group_members=_process_group_members,
    )


def _process_tree_alive(record, actual):
    return _processes_module()._process_tree_alive(
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


def _usage_read(path, plan=None):
    if plan is None:
        return _state_usage_read(path)
    return _state_canonical_usage_read(
        path,
        plan=plan,
        source_still_matches=_source_still_matches,
    )


def _usage_write(
    path,
    counts,
    merge_id=None,
    merge_ids=None,
    *,
    plan=None,
    expected_snapshot=None,
    validator=None,
    require_native=False,
):
    if plan is not None:
        if (
            merge_id is not None
            or merge_ids is not None
            or expected_snapshot is not None
            or validator is not None
            or require_native
        ):
            raise RamdiskError(
                "managed usage seed does not accept generic write authority"
            )
        return _state_managed_usage_write(
            path,
            counts,
            plan=plan,
            filesystem_for_path=_filesystem_for_path,
        )
    return _state_usage_write(
        path,
        counts,
        merge_id=merge_id,
        merge_ids=merge_ids,
        expected_snapshot=expected_snapshot,
        validator=validator,
        require_native=require_native,
    )


def _bind_usage_transaction(record, plan=None, *, reserved_ids=None):
    return _state_bind_usage_transaction(
        record,
        plan=plan,
        filesystem_for_path=_filesystem_for_path,
        reserved_ids=reserved_ids,
    )


def _usage_journal_transaction_id(state_dir, plan=None):
    return _state_usage_journal_transaction_id(
        state_dir,
        plan=plan,
        filesystem_for_path=_filesystem_for_path,
    )


def _recover_delta(
    state_dir,
    canonical_path,
    plan=None,
    expected_merge_id=None,
):
    return _state_recover_delta(
        state_dir,
        canonical_path,
        plan=plan,
        expected_merge_id=expected_merge_id,
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
    return _processes_module()._admit_runtime(
        plan,
        mount,
        benchmark=benchmark,
        available_for_mount=_available_for_mount,
    )


def _admit_concurrent_runtimes(plan, mounts, benchmark=False):
    return _processes_module()._admit_concurrent_runtimes(
        plan,
        mounts,
        benchmark=benchmark,
        host_available_for_mount=_host_available_for_mount,
        cgroup_available_memory=_cgroup_available_memory,
    )


def _assert_effective_masks_unchanged(plan):
    return _processes_module()._assert_effective_masks_unchanged(
        plan,
        discover_hardware=discover_hardware,
    )


def _terminate_verified_group(record, term_seconds=10.0, kill_seconds=3.0):
    return _processes_module()._terminate_verified_group(
        record,
        term_seconds=term_seconds,
        kill_seconds=kill_seconds,
        managed_child_liveness=_managed_child_liveness,
        process_matches=_process_matches,
    )


def _wait_managed_ready(
    record,
    timeout,
    api_key=None,
    cancel_event=None,
    *,
    urlopen=None,
):
    if urlopen is None:
        urlopen = _urllib_module().request.urlopen
    return _processes_module()._wait_managed_ready(
        record,
        timeout,
        api_key=api_key,
        cancel_event=cancel_event,
        process_matches=_process_matches,
        urlopen=urlopen,
    )


@_exclusive_lifecycle(require_process_control=True)
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
        group_alive=_group_alive,
        managed_child_liveness=_managed_child_liveness,
        save_manifest=_save_manifest,
        merge_usage=_merge_usage,
        bind_usage_transaction=_bind_usage_transaction,
        persisted_base_port=_persisted_base_port,
        fresh_user_binary=_fresh_user_binary,
        admit_concurrent_runtimes=_admit_concurrent_runtimes,
        state_root=_state_root,
        ensure_private_dir=_ensure_private_dir,
        assert_durable_state_dir=_assert_durable_state_dir,
        usage_journal_transaction_id=_usage_journal_transaction_id,
        recover_delta=_recover_delta,
        usage_read=_usage_read,
        usage_write=_usage_write,
        validate_usage_for_plan=_validate_usage_for_plan,
        managed_numa_enabled=_managed_numa_enabled,
        memory_node_list=_memory_node_list,
        engine_cpu_list=_engine_cpu_list,
        node_core_count=_node_core_count,
        normalized_runtime_knobs=_normalized_runtime_knobs,
        apply_managed_accelerator_environment=(
            _apply_managed_accelerator_environment
        ),
        invoking_uid=current_uid,
        process_start_boundary=_process_start_boundary,
        current_process_identity=_current_process_identity,
        proc_identity=_proc_identity,
        wait_managed_ready=_wait_managed_ready,
        track_managed_child=_track_managed_child,
        terminate_verified_group=_terminate_verified_group,
        terminate_direct_child=_terminate_direct_child,
        forget_managed_child=_forget_managed_child,
    )


@_exclusive_lifecycle(require_process_control=True)
def stop(args=None):
    return _lifecycle_stop(
        args,
        load_manifest=_load_manifest,
        discover_managed_launches=_managed_launch_processes,
        process_matches=_process_matches,
        process_group_members=_process_group_members,
        group_alive=_group_alive,
        managed_child_liveness=_managed_child_liveness,
        save_manifest=_save_manifest,
        terminate_verified_group=_terminate_verified_group,
        merge_usage=_merge_usage,
        bind_usage_transaction=_bind_usage_transaction,
        recover_benchmark_workspace=(
            _benchmark_workspace_manager().recover
        ),
    )


def _busy_mount_references(path, *, ops=None, hardware=None):
    return _mounts_busy_mount_references(
        path,
        ops=ops,
        hardware=hardware,
    )


def _ensure_busy_mount_scan_available(path, hardware=None):
    return _linux_ensure_busy_mount_scan_available(
        path,
        hardware,
        trusted_system_binary=_trusted_system_binary,
    )


def _managed_path(path, mount_root):
    return _lifecycle_managed_path(path, mount_root)


@_exclusive_lifecycle(require_process_control=True)
def _destroy_locked(args, expected_manifest_token=None):
    return _lifecycle_destroy(
        args,
        expected_manifest_token=expected_manifest_token,
        load_manifest=_load_manifest,
        save_manifest=_save_manifest,
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
        recover_benchmark_workspace=(
            _benchmark_workspace_manager().recover
        ),
    )


def destroy(args, expected_manifest_token=None):
    """Destroy a reviewed deployment, with a pre-lock stale-token check."""
    if expected_manifest_token is not None:
        token = _validate_token(expected_manifest_token, "deployment token")
        manifest = _load_manifest(required=True)
        if _manifest_confirmation_token(manifest) != token:
            raise RamdiskError(
                "RAM workspace changed since review; inspect the active "
                "deployment and confirm Destroy again"
            )
    else:
        # Retained for Python compatibility; every public CLI path supplies a
        # validated deployment token before reaching this compatibility seam.
        token = None
    return _destroy_locked(args, expected_manifest_token=token)


def status(deep=True):
    """Return lifecycle status, optionally skipping deep revalidation."""
    return _lifecycle_status(
        deep=deep,
        load_manifest=_load_manifest,
        manifest_path=_manifest_path,
        source_still_matches=_source_still_matches,
        mount_at=_mount_at,
        validate_mount=_validate_mount,
        validate_namespace=_validate_namespace,
        process_matches=_process_matches,
        managed_child_liveness=_managed_child_liveness,
        deployment_token=_manifest_confirmation_token,
    )


def verify():
    """Return the versioned deep-verification projection for one snapshot."""
    report = status(deep=True)
    mounts = report.get("mounts") or []
    processes = report.get("processes") or []
    state = report.get("state")
    mounts_verified = bool(mounts) and all(
        mount.get("verified") is True
        and mount.get("namespace_verified") is True
        for mount in mounts
    )
    if state == "running":
        processes_verified = bool(processes) and all(
            process.get("running") is True
            and process.get("verified") is True
            for process in processes
        )
    elif state == "stopped":
        processes_verified = all(
            not process.get("running") and process.get("reason") == "stopped"
            for process in processes
        )
    else:
        processes_verified = state == "ready" and not processes
    verified = bool(
        report.get("present")
        and report.get("deep_validation") is True
        and report.get("source_fingerprint_verified") is True
        and state in ("ready", "running", "stopped")
        and mounts_verified
        and processes_verified
        and not report.get("recovery")
    )
    return {
        "schema": "colibri.ramdisk.verify.v1",
        "version": 1,
        "verified": verified,
        "deployment_token": report.get("deployment_token"),
        "report": report,
    }


def _node_core_count(plan, node=None):
    return _planning_node_core_count(
        plan,
        node=node,
    )


def _engine_cpu_list(plan, node=None):
    return _planning_engine_cpu_list(
        plan,
        node=node,
    )


def _memory_node_list(plan, node=None):
    return _planning_memory_node_list(
        plan,
        node=node,
    )


def _managed_numa_enabled(plan, node=None):
    return _planning_managed_numa_enabled(
        plan,
        node=node,
    )


def _normalized_runtime_knobs(plan, knobs, node=None):
    return _common_normalized_runtime_knobs(
        plan,
        knobs,
        node=node,
        node_core_count=_node_core_count,
    )


def _human_status(report):
    return _presentation_module()._human_status(report)


def _managed_process_metrics(record):
    return _processes_module()._managed_process_metrics(
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
    return _presentation_module()._placement_summary(
        plan,
        base_port=base_port,
    )


def _plan_confirmation_token(plan):
    return _plan_token(plan)


def _manifest_confirmation_token(manifest):
    return _deployment_token(
        manifest,
        persisted_base_port=_persisted_base_port,
    )


@_exclusive_lifecycle(require_process_control=True)
def benchmark(args, cli_path=None, engine_path=None, cancel_event=None):
    """Run the append-only causal benchmark without tuning manifest state."""
    module = _benchmark_module()
    return module.run_benchmark(
        args,
        cli_path=cli_path or os.path.join(os.path.dirname(__file__), "coli"),
        engine_path=engine_path,
        cancel_event=cancel_event,
        load_manifest=_load_manifest,
        assert_effective_masks_unchanged=_assert_effective_masks_unchanged,
        assert_ready_mounts=_assert_ready_mounts,
        resolve_engine_path=_resolve_engine_path,
        source_build_identity=lambda: module._source_build_identity(
            __file__,
            environ=os.environ,
            which=shutil.which,
            run=_run,
        ),
        fingerprint_file=lambda path: "sha256:" + _sha256_file(path),
        state_root=_state_root,
        ensure_private_dir=_ensure_private_dir,
        assert_durable_state_dir=_assert_durable_state_dir,
        admit_runtime=_admit_runtime,
        fresh_user_binary=_fresh_user_binary,
        workspace_manager=_benchmark_workspace_manager(),
    )


def dispatch(args, cli_path=None, engine_path=None, system=None):
    return _cli_dispatch(
        args,
        cli_path=cli_path,
        engine_path=engine_path,
        system=system,
        build_plan=build_plan,
        stage=stage,
        status=status,
        verify=verify,
        destroy=destroy,
        benchmark=benchmark,
        plan_token=_plan_confirmation_token,
        deployment_token=_manifest_confirmation_token,
        validate_token=_validate_token,
        human_plan=_human_plan,
        human_status=_human_status,
        human_benchmark=_presentation_module()._human_benchmark_summary,
        json_print=_json_print,
        termination_guard=_cli_termination_guard,
    )


__all__ = sorted(
    set(
        name
        for name in globals()
        if not name.startswith("_") and name not in {"start", "stop"}
    )
    | set(
        name
        for name in _LAZY_ATTRIBUTES
        if not name.startswith("_")
    )
)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--copy-worker":
        try:
            sys.exit(_copy_worker_main(sys.argv[2], sys.argv[3], sys.argv[4]))
        except Exception as error:
            print(error, file=sys.stderr)
            sys.exit(1)
    print("ramdisk.py is a support module; run `coli ramdisk`", file=sys.stderr)
    sys.exit(2)
