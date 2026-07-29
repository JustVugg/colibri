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
from ramdisk_support.curses_ui import (
    _TUI_SCREENS,
    _TuiTerminationSignal,
    _curses_termination_guard,
    _join_tui_worker,
    _run_tui_frontend as _curses_run_tui_frontend,
    _tui as _curses_tui,
    _tui_review_scroll,
    _tui_wrap_rows,
)
from ramdisk_support.cli import (
    _add_lifecycle_options as _cli_add_lifecycle_options,
    _cli_exit_after_signal,
    _cli_termination_guard,
    _confirm as _cli_confirm,
    _json_print as _cli_json_print,
    _load_textual_frontend as _cli_load_textual_frontend,
    _textual_dependency_missing as _cli_textual_dependency_missing,
    configure_parser as _cli_configure_parser,
    dispatch as _cli_dispatch,
    launch_tui as _cli_launch_tui,
)
from ramdisk_support.benchmark import (
    BENCHMARK_PROMPT,
    _aggregate_score as _benchmark_aggregate_score,
    _benchmark_environment as _benchmark_make_environment,
    _benchmark_generate as _benchmark_generate_turn,
    _cancellable_engine_type as _benchmark_cancellable_engine_type,
    _normalized_runtime_knobs as _benchmark_normalized_runtime_knobs,
    _parse_profiler as _benchmark_parse_profiler,
    _score_variant as _benchmark_score_variant,
    _source_build_identity as _benchmark_source_build_identity,
    _system_score as _benchmark_system_score,
    run_benchmark as _run_benchmark,
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
from ramdisk_support.presentation import (
    _human_benchmark as _presentation_human_benchmark,
    _human_plan as _presentation_human_plan,
    _human_status as _presentation_human_status,
    _manifest_confirmation_token as _presentation_manifest_confirmation_token,
    _placement_summary as _presentation_placement_summary,
    _plan_confirmation_token as _presentation_plan_confirmation_token,
    _prepare_confirmation as _presentation_prepare_confirmation,
    _prepare_confirmation_rows as _presentation_prepare_confirmation_rows,
    _tui_activity_rows as _presentation_tui_activity_rows,
    _tui_benchmark_rows as _presentation_tui_benchmark_rows,
    _tui_hardware_rows as _presentation_tui_hardware_rows,
    _tui_help_rows as _presentation_tui_help_rows,
    _tui_idle_action_hint as _presentation_tui_idle_action_hint,
    _tui_plan_rows as _presentation_tui_plan_rows,
    _tui_preset_rows as _presentation_tui_preset_rows,
    _tui_settings_rows as _presentation_tui_settings_rows,
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
    return _presentation_human_plan(plan)


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
        apply_managed_accelerator_environment=(
            _apply_managed_accelerator_environment
        ),
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
    return _benchmark_normalized_runtime_knobs(
        plan,
        knobs,
        node=node,
        node_core_count=_node_core_count,
    )


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
        apply_managed_accelerator_environment=(
            _apply_managed_accelerator_environment
        ),
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
    return _presentation_human_status(report)


def _human_benchmark(result):
    return _presentation_human_benchmark(result)


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
    return _presentation_placement_summary(
        plan,
        base_port=base_port,
    )


def _plan_confirmation_token(plan):
    return _presentation_plan_confirmation_token(plan)


def _manifest_confirmation_token(manifest):
    return _presentation_manifest_confirmation_token(
        manifest,
        persisted_base_port=_persisted_base_port,
    )


def _prepare_confirmation(plan, base_port=8000):
    return _presentation_prepare_confirmation(
        plan,
        base_port=base_port,
    )


def _prepare_confirmation_rows(plan, base_port=8000):
    return _presentation_prepare_confirmation_rows(
        plan,
        base_port=base_port,
    )


def dispatch(args, cli_path=None, engine_path=None, system=None):
    return _cli_dispatch(
        args,
        cli_path=cli_path,
        engine_path=engine_path,
        system=system,
        build_plan=build_plan,
        prepare=prepare,
        status=status,
        benchmark=benchmark,
        start=start,
        stop=stop,
        destroy=destroy,
        human_plan=_human_plan,
        human_status=_human_status,
        human_benchmark=_human_benchmark,
        json_print=_json_print,
        termination_guard=_cli_termination_guard,
    )


_tui_worker_guard = threading.Lock()
_tui_worker = None


def _tui_plan_rows(plan, report, active=False, base_port=8000, confirmation=None):
    return _presentation_tui_plan_rows(
        plan,
        report,
        active=active,
        base_port=base_port,
        confirmation=confirmation,
    )


def _tui_preset_rows():
    return _presentation_tui_preset_rows()


def _tui_hardware_rows(hardware):
    return _presentation_tui_hardware_rows(hardware)


def _tui_activity_rows(report, hardware, process_metrics=None):
    return _presentation_tui_activity_rows(
        report,
        hardware,
        process_metrics=process_metrics,
        meminfo=_meminfo,
    )


def _tui_benchmark_rows(history):
    return _presentation_tui_benchmark_rows(history)


def _tui_settings_rows(args, plan, report, base_port=8000):
    return _presentation_tui_settings_rows(
        args,
        plan,
        report,
        base_port=base_port,
    )


def _tui_help_rows():
    return _presentation_tui_help_rows()


def _tui_idle_action_hint(screen, plan, report):
    return _presentation_tui_idle_action_hint(
        screen,
        plan,
        report,
    )


def _tui(stdscr, initial, cli_path, engine_path):
    return _curses_tui(
        stdscr,
        initial,
        cli_path,
        engine_path,
        bindings=sys.modules[__name__],
    )


def _load_textual_frontend():
    return _cli_load_textual_frontend()


def _textual_dependency_missing(error):
    return _cli_textual_dependency_missing(error)


def _run_tui_frontend(callback):
    return _curses_run_tui_frontend(
        callback,
        bindings=sys.modules[__name__],
    )


def launch_tui(args, cli_path=None, engine_path=None, system=None):
    global _tui_worker

    def finish_frontend():
        global _tui_worker
        with _tui_worker_guard:
            if (
                _tui_worker is not None
                and not _tui_worker["thread"].is_alive()
            ):
                _tui_worker = None

    return _cli_launch_tui(
        args,
        cli_path=cli_path,
        engine_path=engine_path,
        system=system,
        lifecycle=sys.modules[__name__],
        run_tui_frontend=_run_tui_frontend,
        legacy_tui=_tui,
        curses_termination_guard=_curses_termination_guard,
        finish_frontend=finish_frontend,
        load_textual_frontend=_load_textual_frontend,
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
