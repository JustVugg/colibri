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
    build_plan as _planning_build_plan,
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
    """Import the optional frontend without making scriptable commands depend on it."""
    import ramdisk_textual

    return ramdisk_textual


def _textual_dependency_missing(error):
    missing = getattr(error, "name", "") or ""
    return missing == "textual" or missing.startswith("textual.")


def _run_tui_frontend(callback):
    return _curses_run_tui_frontend(
        callback,
        bindings=sys.modules[__name__],
    )


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
