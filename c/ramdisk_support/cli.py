"""Scriptable CLI orchestration and lazy terminal-frontend selection."""

from __future__ import print_function

import argparse
import contextlib
import json
import os
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
    prepare_parser = actions.add_parser(
        "prepare",
        parents=[after],
        help="mount, stage, and validate weights",
    )
    prepare_parser.add_argument(
        "--yes",
        action="store_true",
        help="accept the reviewed plan non-interactively",
    )
    status_parser = actions.add_parser(
        "status",
        parents=[after],
        help="show mounts and managed processes",
    )
    status_parser.add_argument("--json", action="store_true")
    benchmark_parser = actions.add_parser(
        "benchmark",
        parents=[after],
        help="run equal RAM/SSD scorecards",
    )
    benchmark_parser.add_argument("--json", action="store_true")
    start_parser = actions.add_parser(
        "start",
        parents=[after],
        help="start managed engine process(es)",
    )
    start_parser.add_argument(
        "--base-port",
        type=int,
        default=None,
        help=(
            "managed base port "
            "(defaults to the prepared deployment's last value)"
        ),
    )
    actions.add_parser(
        "stop",
        parents=[after],
        help="stop only verified managed processes",
    )
    destroy_parser = actions.add_parser(
        "destroy",
        parents=[after],
        help="unmount volatile weights safely",
    )
    destroy_parser.add_argument("--yes", action="store_true")


def _json_print(value):
    print(json.dumps(value, indent=2, sort_keys=True))


def _confirm(message, accepted=False):
    if accepted:
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise RamdiskError(
            message
            + "; rerun with --yes after reviewing `ramdisk plan`"
        )
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


def dispatch(
    args,
    cli_path=None,
    engine_path=None,
    system=None,
    *,
    build_plan,
    prepare,
    status,
    benchmark,
    start,
    stop,
    destroy,
    human_plan,
    human_status,
    human_benchmark,
    json_print=None,
    termination_guard=None,
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
                emit_json(value)
            else:
                human_plan(value)
            return 2 if value["blockers"] else 0
        if action == "prepare":
            with guard(True) as termination:
                value = prepare(
                    args,
                    cancel_event=termination["cancel_event"],
                )
            print(
                "RAM-disk ready: %s"
                % ", ".join(record["path"] for record in value["mounts"])
            )
            return _cli_exit_after_signal(termination, 0)
        if action == "status":
            value = status()
            if getattr(args, "json", False):
                emit_json(value)
            else:
                human_status(value)
            return 0
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
        if action == "start":
            with guard(True) as termination:
                value = start(
                    args,
                    cli_path=cli_path,
                    engine_path=engine_path,
                    cancel_event=termination["cancel_event"],
                )
            print(
                "managed engine ports: %s"
                % ", ".join(str(port) for port in value["ports"])
            )
            return _cli_exit_after_signal(termination, 0)
        if action == "stop":
            with guard(False) as termination:
                value = stop(args)
            if value.get("state") == "error":
                print(
                    "managed engine cleanup completed, but the RAM "
                    "workspace is incomplete; review `coli ramdisk status`, "
                    "then run destroy",
                    file=sys.stderr,
                )
                return _cli_exit_after_signal(termination, 2)
            print("managed engines stopped; usage deltas merged")
            return _cli_exit_after_signal(termination, 0)
        if action == "destroy":
            with guard(False) as termination:
                value = destroy(args)
            print(
                "RAM-disk destroyed; durable state and benchmark history "
                "preserved"
            )
            return _cli_exit_after_signal(termination, 0)
        raise RamdiskError(
            "choose a ramdisk action or run the interactive TUI"
        )
    except (RamdiskError, OSError, subprocess.SubprocessError) as exc:
        if getattr(args, "json", False):
            emit_json(
                {
                    "schema": "colibri.ramdisk.error.v1",
                    "version": 1,
                    "error": str(exc),
                }
            )
        else:
            print("coli ramdisk: %s" % exc, file=sys.stderr)
        return _cli_exit_after_signal(termination, 2)


def _load_textual_frontend():
    """Import the optional frontend only when terminal routing selects it."""
    import ramdisk_textual

    return ramdisk_textual


def _textual_dependency_missing(error):
    missing = getattr(error, "name", "") or ""
    return missing == "textual" or missing.startswith("textual.")


def launch_tui(
    args,
    cli_path=None,
    engine_path=None,
    system=None,
    *,
    lifecycle,
    run_tui_frontend,
    legacy_tui=None,
    curses_termination_guard=None,
    finish_frontend=None,
    load_textual_frontend=None,
    curses_wrapper=None,
    target_platform=None,
    environment=None,
):
    """Select Textual or curses without importing either frontend eagerly."""
    platform_name = (
        sys.platform if target_platform is None else target_platform
    )
    if not platform_name.startswith("linux"):
        print(
            "coli ramdisk: the TUI is supported only on Linux",
            file=sys.stderr,
        )
        return 2

    environment = os.environ if environment is None else environment
    requested_ui = environment.get(
        "COLI_RAMDISK_UI",
        "auto",
    ).strip().lower()
    if requested_ui not in ("auto", "textual", "curses"):
        print(
            "coli ramdisk: COLI_RAMDISK_UI must be auto, textual, or curses",
            file=sys.stderr,
        )
        return 2

    loader = (
        _load_textual_frontend
        if load_textual_frontend is None
        else load_textual_frontend
    )
    textual_frontend = None
    if requested_ui in ("auto", "textual"):
        try:
            textual_frontend = loader()
        except ModuleNotFoundError as exc:
            if not _textual_dependency_missing(exc):
                raise
            if requested_ui == "textual":
                print(
                    "coli ramdisk: Textual UI requested but Textual is not "
                    "installed; install the TUI dependency or set "
                    "COLI_RAMDISK_UI=curses",
                    file=sys.stderr,
                )
                return 2

    try:
        if textual_frontend is not None:
            return run_tui_frontend(
                lambda: textual_frontend.launch_tui(
                    args,
                    cli_path=cli_path,
                    engine_path=engine_path,
                    lifecycle=lifecycle,
                )
            )

        if curses_wrapper is None:
            import curses

            curses_wrapper = curses.wrapper
        if legacy_tui is None or curses_termination_guard is None:
            raise TypeError(
                "curses routing requires legacy_tui and "
                "curses_termination_guard callbacks"
            )
        with curses_termination_guard():
            return run_tui_frontend(
                lambda: curses_wrapper(
                    legacy_tui,
                    args,
                    cli_path,
                    engine_path,
                )
            )
    finally:
        if finish_frontend is not None:
            finish_frontend()
