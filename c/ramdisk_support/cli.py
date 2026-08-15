"""Scriptable orchestration for the dependency-free RAM-workspace CLI."""

from __future__ import print_function

import argparse
import contextlib
import json
import signal
import subprocess
import sys
import threading

from .accelerator import GPU_LAYOUT_CHOICES, GPU_LAYOUT_EXPERTS_ONLY
from .common import DEFAULT_MOUNT_ROOT, RamdiskError


def _add_lifecycle_options(parser, suppress=False):
    default = argparse.SUPPRESS if suppress else None
    # ``coli`` already supplies --model/--ctx on the outer ramdisk parser via
    # its common parent. Add them only when this module is used standalone;
    # the action-local parser always receives suppressing copies so the same
    # options can also appear after ``plan``/``prepare`` without overwriting a
    # value parsed before the action.
    if "--model" not in parser._option_string_actions:
        parser.add_argument(
            "--model",
            default=default,
            help="canonical model directory on durable storage",
        )
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
            "interleaved = one shared model copy and one engine; "
            "per-node = one complete copy and independent engine per "
            "NUMA node (replication, not sharding)"
        ),
    )
    parser.add_argument(
        "--memory-nodes",
        default=default,
        metavar="NODELIST",
        help=(
            "effective NUMA memory nodes (for example 0-3,8); "
            "defaults to allowed CPU-bearing nodes"
        ),
    )
    parser.add_argument(
        "--cpu-list",
        default=default,
        metavar="CPULIST",
        help=(
            "whole-core managed-engine CPUs (for example 0-15,32-47); "
            "defaults to allowed CPUs on the selected memory nodes"
        ),
    )
    parser.add_argument(
        "--capacity-gb",
        type=float,
        default=default,
        help="per-copy staging budget; required for partial mode",
    )
    parser.add_argument(
        "--profile",
        default=default,
        help="compatible .coli_usage text or JSON profile",
    )
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
    if "--gpu" not in parser._option_string_actions:
        parser.add_argument(
            "--gpu",
            default=default,
            help="auto, none, or an exact device list such as 0,1",
        )
    if "--gpu-layout" not in parser._option_string_actions:
        parser.add_argument(
            "--gpu-layout",
            choices=GPU_LAYOUT_CHOICES,
            default=(
                argparse.SUPPRESS
                if suppress
                else GPU_LAYOUT_EXPERTS_ONLY
            ),
            help=(
                "experts-only (stable), dense-attention, or "
                "dense-attention-sharded (experimental)"
            ),
        )


def configure_parser(parser, common_parent=None):
    """Attach scriptable subcommands; options work before or after the action."""
    _add_lifecycle_options(parser, suppress=False)
    after = argparse.ArgumentParser(
        add_help=False,
        argument_default=argparse.SUPPRESS,
    )
    _add_lifecycle_options(after, suppress=True)
    actions = parser.add_subparsers(
        dest="ramdisk_action",
        metavar="ACTION",
    )
    plan = actions.add_parser(
        "plan",
        parents=[after],
        help="show an exact staging and reserve plan",
    )
    plan.add_argument("--json", action="store_true")
    stage_parsers = (
        actions.add_parser(
            "stage",
            parents=[after],
            help="mount, stage, and validate the token-bound plan",
        ),
        actions.add_parser(
            "prepare",
            parents=[after],
            help="exact alias of stage",
        ),
    )
    for stage_parser in stage_parsers:
        stage_parser.add_argument(
            "--plan-token",
            default=None,
            help="lowercase 64-hex identity emitted by plan --json",
        )
        stage_parser.add_argument(
            "--yes",
            action="store_true",
            help="accept the reviewed plan non-interactively",
        )
        stage_parser.add_argument("--json", action="store_true")
    verify_parser = actions.add_parser(
        "verify",
        parents=[after],
        help="deeply validate sources, mounts, namespaces, and processes",
    )
    verify_parser.add_argument("--json", action="store_true")
    status_parser = actions.add_parser(
        "status",
        parents=[after],
        help="show mounts and managed processes",
    )
    status_parser.add_argument("--json", action="store_true")
    status_parser.add_argument(
        "--runtime",
        action="store_true",
        help="include advisory live serving/GPU telemetry (requires --json)",
    )
    for action, help_text in (
        ("start", "start token-bound managed engines"),
        ("stop", "stop token-bound managed engines and merge usage"),
    ):
        managed = actions.add_parser(action, parents=[after], help=help_text)
        managed.add_argument(
            "--deployment-token",
            default=None,
            help="lowercase 64-hex identity emitted by status --json",
        )
        managed.add_argument("--yes", action="store_true")
        managed.add_argument("--json", action="store_true")
        if action == "start":
            managed.add_argument(
                "--base-port",
                type=int,
                default=None,
                help=(
                    "first numeric-loopback managed endpoint port; omitted "
                    "reuses the prepared deployment's persisted port"
                ),
            )
    destroy_parser = actions.add_parser(
        "destroy",
        parents=[after],
        help="unmount volatile weights safely",
    )
    destroy_parser.add_argument(
        "--deployment-token",
        default=None,
        help="lowercase 64-hex identity emitted by status --json",
    )
    destroy_parser.add_argument("--yes", action="store_true")
    destroy_parser.add_argument("--json", action="store_true")
    benchmark_parser = actions.add_parser(
        "benchmark",
        parents=[after],
        help="run the fixed causal RAMMAP evidence protocol",
    )
    benchmark_parser.add_argument(
        "--evidence-profile",
        default=None,
        help="frozen usage profile shared by every treatment",
    )
    benchmark_parser.add_argument(
        "--residency-gb",
        type=float,
        default=None,
        help="numeric CPU residency budget; auto/all are forbidden",
    )
    benchmark_parser.add_argument("--cuda-host-gb", type=float, default=None)
    benchmark_parser.add_argument("--cuda-expert-gb", type=float, default=None)
    benchmark_parser.add_argument(
        "--replicates",
        type=int,
        default=7,
        help="fresh processes per treatment (minimum 7)",
    )
    benchmark_parser.add_argument("--seed", type=int, default=377)
    benchmark_parser.add_argument(
        "--practical-threshold",
        type=float,
        default=0.05,
        help="predeclared minimum relative throughput improvement",
    )
    benchmark_parser.add_argument("--confidence", type=float, default=0.95)
    benchmark_parser.add_argument(
        "--raw-evidence",
        default=None,
        help="append-only JSONL destination outside volatile storage",
    )
    benchmark_parser.add_argument("--json", action="store_true")


def _json_print(value):
    print(json.dumps(value, indent=2, sort_keys=True))


def _stage_projection(manifest, reviewed_plan_token, deployment_token):
    return {
        "schema": "colibri.ramdisk.stage.v1",
        "version": 1,
        "state": manifest.get("state"),
        "deployment_id": manifest.get("deployment_id"),
        "plan_token": reviewed_plan_token,
        "deployment_token": deployment_token(manifest),
        "mounts": [
            {"path": record.get("path"), "node": record.get("node")}
            for record in manifest.get("mounts", [])
        ],
    }


def _destroy_projection(result):
    projection = {
        "schema": "colibri.ramdisk.destroy.v1",
        "version": 1,
    }
    for name in (
        "destroyed",
        "durable_state_preserved",
        "benchmark_history_preserved",
        "empty_mountpoints_preserved",
    ):
        if name in result:
            projection[name] = result[name]
    return projection


def _start_projection(manifest, deployment_token):
    ports = sorted(
        record.get("port")
        for record in manifest.get("processes", [])
        if not record.get("stopped_at") and isinstance(record.get("port"), int)
    )
    return {
        "schema": "colibri.ramdisk.start.v1",
        "version": 1,
        "state": manifest.get("state"),
        "deployment_id": manifest.get("deployment_id"),
        "deployment_token": deployment_token(manifest),
        "ports": ports,
        "endpoints": sorted(
            (
                {
                    "port": record.get("port"),
                    "node": record.get("node"),
                    "url": "http://127.0.0.1:%d" % record["port"],
                }
                for record in manifest.get("processes", [])
                if not record.get("stopped_at")
                and isinstance(record.get("port"), int)
            ),
            key=lambda item: (item["port"] is None, item["port"] or 0),
        ),
        "containment_mode": _containment_mode(manifest),
        "usage_merge_summary": _usage_merge_summary(manifest),
        "recovery_attention": _recovery_attention(manifest),
    }


def _stop_projection(manifest, deployment_token):
    return {
        "schema": "colibri.ramdisk.stop.v1",
        "version": 1,
        "state": manifest.get("state"),
        "deployment_id": manifest.get("deployment_id"),
        "deployment_token": deployment_token(manifest),
        "stopped_count": int(
            (manifest.get("_operation_summary") or {}).get("stopped_count", 0)
        ),
        "containment_mode": _containment_mode(manifest),
        "usage_merge_summary": _usage_merge_summary(manifest),
        "recovery_attention": _recovery_attention(manifest),
    }


def _containment_mode(manifest):
    return (
        "cgroup-v2"
        if manifest.get("process_supervision_version") == 1
        else "legacy-process-group"
    )


def _usage_merge_summary(manifest):
    summary = (manifest.get("_operation_summary") or {}).get("usage_merge") or {}
    return {
        "merged_count": int(summary.get("merged_count", 0)),
        "pending_count": int(summary.get("pending_count", 0)),
        "error_count": int(summary.get("error_count", 0)),
    }


def _recovery_attention(manifest):
    recovery = manifest.get("recovery")
    return bool(
        manifest.get("state") == "error"
        or manifest.get("pending_launches")
        or manifest.get("benchmark_workspace")
        or (
            isinstance(recovery, dict)
            and (
                recovery.get("retained_processes")
                or recovery.get("retained_mounts")
                or recovery.get("state") == "attention-required"
            )
        )
        or any(
            record.get("stop_error") or record.get("usage_merge_error")
            for record in manifest.get("processes", [])
            if isinstance(record, dict)
        )
    )


@contextlib.contextmanager
def _interruptible_confirmation():
    """Let terminal Ctrl-C interrupt a blocking confirmation read."""
    sigint = getattr(signal, "SIGINT", None)
    previous = None
    installed = False
    if (
        sigint is not None
        and threading.current_thread() is threading.main_thread()
    ):
        try:
            previous = signal.getsignal(sigint)
            signal.signal(sigint, signal.default_int_handler)
            installed = True
        except (OSError, ValueError):
            pass
    try:
        yield
    finally:
        if installed:
            try:
                signal.signal(sigint, previous)
            except (OSError, ValueError):
                pass


def _confirm(message, accepted=False):
    if accepted:
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise RamdiskError(
            message
            + "; rerun with --yes after reviewing `ramdisk plan`"
        )
    with _interruptible_confirmation():
        answer = input(message + " [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        raise RamdiskError("cancelled")


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
        for name in ("SIGINT", "SIGHUP", "SIGTERM"):
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


def dispatch(
    args,
    cli_path=None,
    engine_path=None,
    system=None,
    *,
    build_plan,
    stage,
    status,
    verify,
    destroy,
    benchmark,
    plan_token,
    deployment_token,
    validate_token,
    human_plan,
    human_status,
    human_benchmark,
    json_print=None,
    termination_guard=None,
    start=None,
    stop=None,
):
    """Route one parsed action through explicitly supplied application seams."""
    emit_json = _json_print if json_print is None else json_print
    guard = (
        _cli_termination_guard
        if termination_guard is None
        else termination_guard
    )
    action = getattr(args, "ramdisk_action", None)
    termination = None
    try:
        if action == "plan":
            value = build_plan(args)
            if getattr(args, "json", False):
                projection = dict(value)
                projection["plan_token"] = plan_token(value)
                emit_json(projection)
            else:
                human_plan(value)
            return 2 if value["blockers"] else 0
        if action in ("stage", "prepare"):
            json_mode = bool(getattr(args, "json", False))
            reviewed_token = validate_token(
                getattr(args, "plan_token", None),
                "plan token",
            )
            if not getattr(args, "yes", False):
                raise RamdiskError("stage requires --yes")
            progress = (lambda _name, _size, _elapsed: None) if json_mode else None
            with guard(True) as termination:
                value = stage(
                    args,
                    progress=progress,
                    display_plan=not json_mode,
                    expected_plan_token=reviewed_token,
                    cancel_event=termination["cancel_event"],
                )
            if json_mode:
                emit_json(
                    _stage_projection(
                        value,
                        reviewed_token,
                        deployment_token,
                    )
                )
            else:
                print(
                    "RAM-disk ready: %s"
                    % ", ".join(record["path"] for record in value["mounts"])
                )
            return _cli_exit_after_signal(termination, 0)
        if action == "verify":
            value = verify()
            if getattr(args, "json", False):
                emit_json(value)
            else:
                human_status(value["report"])
                print(
                    "RAM-disk deep verification: %s"
                    % ("verified" if value["verified"] else "not verified")
                )
            return 0 if value["verified"] else 2
        if action == "status":
            include_runtime = bool(getattr(args, "runtime", False))
            if include_runtime and not getattr(args, "json", False):
                raise RamdiskError("status --runtime requires --json")
            value = status(runtime=True) if include_runtime else status()
            if getattr(args, "json", False):
                emit_json(value)
            else:
                human_status(value)
            return 0
        if action in ("start", "stop"):
            if not getattr(args, "json", False):
                raise RamdiskError("%s requires --json" % action)
            reviewed_token = validate_token(
                getattr(args, "deployment_token", None),
                "deployment token",
            )
            if not getattr(args, "yes", False):
                raise RamdiskError("%s requires --yes" % action)
            operation = start if action == "start" else stop
            if operation is None:
                raise RamdiskError("%s is unavailable" % action)
            with guard(action == "start") as termination:
                if action == "start":
                    value = operation(
                        args,
                        cli_path=cli_path,
                        engine_path=engine_path,
                        cancel_event=termination["cancel_event"],
                        expected_manifest_token=reviewed_token,
                    )
                    emit_json(_start_projection(value, deployment_token))
                else:
                    value = operation(
                        args,
                        expected_manifest_token=reviewed_token,
                    )
                    emit_json(_stop_projection(value, deployment_token))
            return _cli_exit_after_signal(termination, 0)
        if action == "destroy":
            reviewed_token = validate_token(
                getattr(args, "deployment_token", None),
                "deployment token",
            )
            if not getattr(args, "yes", False):
                raise RamdiskError("destroy requires --yes")
            with guard(False) as termination:
                value = destroy(
                    args,
                    expected_manifest_token=reviewed_token,
                )
            if getattr(args, "json", False):
                emit_json(_destroy_projection(value))
            else:
                print("RAM-disk destroyed; durable state preserved")
            return _cli_exit_after_signal(termination, 0)
        if action == "benchmark":
            with guard(True) as termination:
                value = benchmark(
                    args,
                    cli_path=cli_path,
                    engine_path=engine_path,
                    cancel_event=termination["cancel_event"],
                )
            if getattr(args, "json", False):
                emit_json(value)
            else:
                human_benchmark(value)
            return _cli_exit_after_signal(termination, 0)
        raise RamdiskError("choose a ramdisk action; run with --help")
    except (RamdiskError, OSError, subprocess.SubprocessError) as exc:
        if getattr(args, "json", False):
            public_error = (
                exc.public_message
                if isinstance(exc, RamdiskError)
                else (
                    "RAM-disk operation failed; rerun without --json for "
                    "protected diagnostics"
                )
            )
            emit_json(
                {
                    "schema": "colibri.ramdisk.error.v1",
                    "version": 1,
                    "error": public_error,
                }
            )
        else:
            print("coli ramdisk: %s" % exc, file=sys.stderr)
        return _cli_exit_after_signal(termination, 2)
