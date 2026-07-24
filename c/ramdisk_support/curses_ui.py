"""Legacy curses frontend with facade-owned runtime state bindings."""

from __future__ import print_function

import argparse
import contextlib
import math
import signal
import subprocess
import sys
import textwrap
import threading
import time

from .common import (
    DEFAULT_MOUNT_ROOT,
    GIB,
    MIB,
    RamdiskError,
    _OperationCancelled,
)
from ramdisk_ui import ActionPolicy, ReviewIdentity


_TUI_SCREENS = (
    "Plan",
    "Hardware",
    "Activity",
    "Benchmarks",
    "Settings",
)


def _tui_review_scroll(pending_action, requested_scroll):
    """Keep prepare-review facts visible until review completes or cancels."""
    if pending_action == "prepare":
        return 0
    return max(0, requested_scroll)


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
        wrapped.extend(
            (style, part)
            for part in (parts or [leading])
        )
    return wrapped


def _tui(
    stdscr,
    initial,
    cli_path,
    engine_path,
    *,
    bindings,
):
    """Run the curses dashboard against live facade/runtime bindings."""
    import curses

    _benchmarks_path = bindings._benchmarks_path
    _load_manifest = bindings._load_manifest
    _managed_ports_for_plan = bindings._managed_ports_for_plan
    _managed_process_metrics = bindings._managed_process_metrics
    _manifest_confirmation_token = bindings._manifest_confirmation_token
    _noninteractive_privilege = bindings._noninteractive_privilege
    _persisted_base_port = bindings._persisted_base_port
    _placement_summary = bindings._placement_summary
    _plan_confirmation_token = bindings._plan_confirmation_token
    _prepare_confirmation = bindings._prepare_confirmation
    _read_json = bindings._read_json
    _trusted_system_binary = bindings._trusted_system_binary
    _tui_activity_rows = bindings._tui_activity_rows
    _tui_benchmark_rows = bindings._tui_benchmark_rows
    _tui_hardware_rows = bindings._tui_hardware_rows
    _tui_help_rows = bindings._tui_help_rows
    _tui_idle_action_hint = bindings._tui_idle_action_hint
    _tui_plan_rows = bindings._tui_plan_rows
    _tui_settings_rows = bindings._tui_settings_rows
    _validate_noninteractive_sudo = bindings._validate_noninteractive_sudo
    benchmark = bindings.benchmark
    build_plan = bindings.build_plan
    current_euid = bindings.current_euid
    destroy = bindings.destroy
    discover_hardware = bindings.discover_hardware
    prepare = bindings.prepare
    scan_model = bindings.scan_model
    start = bindings.start
    status = bindings.status
    stop = bindings.stop
    worker_guard = bindings._tui_worker_guard

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
            (
                curses.COLOR_CYAN,
                curses.COLOR_GREEN,
                curses.COLOR_YELLOW,
                curses.COLOR_RED,
            ),
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
    message = (
        "Shared placement is selected: one model copy, one engine. "
        "Press ? for help."
    )
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
        if (
            row < 0
            or row >= height
            or column < 0
            or column >= width
            or limit <= 0
        ):
            return
        try:
            stdscr.addnstr(
                row,
                column,
                str(value),
                min(limit, width - column),
                attribute,
            )
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

    def begin_operation(
        action,
        label,
        target,
        cancelable=False,
    ):
        nonlocal operation
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
                    cancel_event=(
                        op["cancel_event"]
                        if cancelable
                        else None
                    ),
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
        thread = threading.Thread(
            target=runner,
            name="coli-ramdisk-%s" % action,
        )
        op["thread"] = thread
        with worker_guard:
            bindings._tui_worker = op
        try:
            thread.start()
        except BaseException:
            with worker_guard:
                bindings._tui_worker = None
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
        current_text = (
            current
            if current not in (None, "")
            else "<auto>"
        )
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
                "%s · current %s · Enter keeps it"
                % (label, current_text),
                width - 1,
                attrs["dim"],
            )
            stdscr.move(height - 2, 0)
            stdscr.clrtoeol()
            safe_add(
                height - 2,
                0,
                prompt,
                width - 1,
                attrs["accent"],
            )
            stdscr.refresh()
            entry_column = len(prompt)
            value = stdscr.getstr(
                height - 2,
                entry_column,
                max(1, width - entry_column - 1),
            )
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
        """Obtain sudo credentials before workers need the controlling TTY."""
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
                result = subprocess.run(
                    [sudo, "-v"],
                    check=False,
                )
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
            message = (
                "Sudo authorization was cancelled; "
                "no mount operation started."
            )
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
                cancelled_cleanly = isinstance(
                    operation_error,
                    _OperationCancelled,
                )
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
                        message = (
                            "%s cleanup failed after cancellation: %s · "
                            "review Activity before quitting"
                            % (
                                operation_action.capitalize(),
                                operation_error,
                            )
                        )
                        quit_when_idle = False
                    else:
                        message = "%s failed: %s" % (
                            operation_action.capitalize(),
                            operation_error,
                        )
                elif operation_action == "prepare":
                    message = (
                        "RAM workspace is ready. Open Activity or press s "
                        "to start."
                    )
                elif operation_action == "start":
                    message = "Engine ready on %s." % _placement_summary(
                        operation_result["plan"],
                        args.base_port,
                    )["endpoints"]
                elif operation_action == "stop":
                    if operation_result.get("state") == "error":
                        message = (
                            "Engine cleanup finished, but the workspace is "
                            "incomplete. Review Activity, then Destroy."
                        )
                    else:
                        message = (
                            "Managed engines stopped; RAM weights remain "
                            "prepared."
                        )
                elif operation_action == "destroy":
                    message = (
                        "RAM workspace removed; durable KV and benchmark "
                        "state preserved."
                    )
                elif operation_action == "benchmark":
                    message = (
                        "Benchmark complete; best path: %s."
                        % operation_result.get("best_variant")
                    )
                operation = None
                with worker_guard:
                    bindings._tui_worker = None
                cancel_confirmation()
                invalidate(deep=False, model=False)
                if quit_after_cleanup:
                    return quit_exit_code

        plan = report = hardware = None
        rows = []
        try:
            if (
                hardware_cache is None
                or now - hardware_checked >= 30.0
            ):
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
                    args.base_port = _persisted_base_port(
                        active_manifest_cache
                    )
                    active_deployment_identity = deployment_identity
                if (
                    report.get("state") in ("running", "starting")
                    and report.get("ports")
                    and processes
                ):
                    first = processes[0]
                    args.base_port = (
                        int(first["port"])
                        - int(first.get("node") or 0)
                    )
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
                if (
                    plan_cache is None
                    or plan_key != plan_key_cache
                ):
                    plan_cache = build_plan(
                        args,
                        hardware=hardware,
                        model=model_cache,
                    )
                    plan_key_cache = plan_key
                plan = plan_cache

            if pending_action == "prepare":
                current_token = _plan_confirmation_token(plan)
                current_review = ReviewIdentity.for_prepare(
                    current_token,
                    plan,
                    args.base_port,
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
                rows = _tui_plan_rows(
                    plan,
                    report,
                    active,
                    args.base_port,
                    confirmation,
                )
            elif screen == 1:
                rows = _tui_hardware_rows(hardware)
            elif screen == 2:
                if now - metrics_checked >= 2.0:
                    metrics_cache = {
                        process.get("pid"): _managed_process_metrics(
                            process
                        )
                        for process in report.get("processes", [])
                        if process.get("pid") is not None
                    }
                    metrics_checked = now
                rows = _tui_activity_rows(
                    report,
                    hardware,
                    metrics_cache,
                )
            elif screen == 3:
                if history_cache is None:
                    history_cache = (
                        _read_json(_benchmarks_path())
                        or {"results": []}
                    )
                rows = _tui_benchmark_rows(history_cache)
            else:
                rows = _tui_settings_rows(
                    args,
                    plan,
                    report,
                    args.base_port,
                )
        except Exception as exc:
            rows = [
                ("bad", "THIS PAGE COULD NOT BE RENDERED"),
                ("bad", str(exc)),
                (
                    "dim",
                    "Press R to retry a deep refresh. "
                    "No lifecycle action was taken.",
                ),
            ]

        if operation is not None:
            with operation_lock:
                op_label = operation["label"]
                op_detail = operation["detail"]
                op_started = operation["started"]
            spinner = "|/-\\"[
                int((now - op_started) * 5) % 4
            ]
            rows = [
                (
                    "warn",
                    "%s %s · %.1fs"
                    % (spinner, op_label, now - op_started),
                ),
                ("normal", op_detail),
                ("normal", ""),
            ] + rows

        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < 8 or width < 38:
            safe_add(
                0,
                0,
                "COLIBRÍ · RAM WORKSPACE",
                max(1, width - 1),
                attrs["accent"],
            )
            safe_add(
                2,
                0,
                "Resize to at least 38 x 8.",
                max(1, width - 1),
                attrs["warn"],
            )
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

        state_text = (
            report.get("state", "unknown")
            if report
            else "loading"
        )
        safe_add(
            0,
            0,
            "COLIBRÍ · RAM WORKSPACE",
            width - 1,
            attrs["accent"],
        )
        right = "state · %s" % state_text
        safe_add(
            0,
            max(0, width - len(right) - 1),
            right,
            len(right),
            attrs["dim"],
        )
        tab_text = "  ".join(
            (
                "[%s]" % name.upper()
                if index == screen and not help_open
                else name
            )
            for index, name in enumerate(_TUI_SCREENS)
        )
        if help_open:
            tab_text = "[HELP]  " + tab_text
        safe_add(
            1,
            0,
            tab_text,
            width - 1,
            attrs["heading"],
        )
        safe_add(
            2,
            0,
            "─" * max(1, width - 1),
            width - 1,
            attrs["dim"],
        )

        rendered = _tui_wrap_rows(rows, width - 3)
        content_height = max(1, height - 5)
        max_scroll = max(0, len(rendered) - content_height)
        scroll = min(
            _tui_review_scroll(pending_action, scroll),
            max_scroll,
        )
        for offset, (style, line) in enumerate(
            rendered[scroll : scroll + content_height]
        ):
            safe_add(
                3 + offset,
                1,
                line,
                width - 3,
                attrs.get(style, attrs["normal"]),
            )
        if max_scroll:
            indicator = "%d–%d / %d" % (
                scroll + 1,
                min(
                    len(rendered),
                    scroll + content_height,
                ),
                len(rendered),
            )
            safe_add(
                2,
                max(0, width - len(indicator) - 1),
                indicator,
                len(indicator),
                attrs["dim"],
            )

        if operation is not None:
            action_hint = (
                "Operation in progress · [c] cancel · "
                "navigation remains available"
                if operation["cancelable"]
                else "Operation in progress · navigation remains available"
            )
        elif help_open:
            action_hint = "[?] close help"
        else:
            action_hint = _tui_idle_action_hint(
                screen,
                plan,
                report,
            )
        footer = (
            "%s · ←/→ pages · ↑/↓ scroll · [?] help · [q] quit"
            % action_hint
        )
        safe_add(
            height - 2,
            0,
            str(message).ljust(width - 1),
            width - 1,
            attrs["dim"],
        )
        safe_add(
            height - 1,
            0,
            footer.ljust(width - 1),
            width - 1,
            curses.A_REVERSE,
        )
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
                    message = (
                        "Cancelling safely; Colibri will quit after "
                        "rollback/cleanup finishes."
                    )
                else:
                    message = (
                        "This cleanup step cannot be interrupted safely; "
                        "Colibri will quit when it finishes."
                    )
            else:
                return 130 if key == 3 else 0
            continue
        if (
            key == ord("c")
            and operation is not None
            and operation["cancelable"]
        ):
            operation["cancel_event"].set()
            message = (
                "Cancellation requested; waiting for "
                "rollback/cleanup checkpoints."
            )
            continue
        if key == ord("?"):
            if pending_action == "prepare":
                cancel_confirmation()
                message = (
                    "Prepare review cancelled when help was opened."
                )
            help_open = not help_open
            scroll = 0
            continue
        if key in (ord("l"), curses.KEY_RIGHT):
            if pending_action == "prepare":
                cancel_confirmation()
                message = (
                    "Prepare review cancelled when leaving the Plan page."
                )
            screen = (screen + 1) % len(_TUI_SCREENS)
            help_open = False
            scroll = 0
            continue
        if key in (ord("h"), curses.KEY_LEFT):
            if pending_action == "prepare":
                cancel_confirmation()
                message = (
                    "Prepare review cancelled when leaving the Plan page."
                )
            screen = (screen - 1) % len(_TUI_SCREENS)
            help_open = False
            scroll = 0
            continue
        if key in (ord("j"), curses.KEY_DOWN):
            scroll = _tui_review_scroll(
                pending_action,
                scroll + 1,
            )
            continue
        if key in (ord("k"), curses.KEY_UP):
            scroll = _tui_review_scroll(
                pending_action,
                scroll - 1,
            )
            continue
        if key == curses.KEY_NPAGE:
            scroll = _tui_review_scroll(
                pending_action,
                scroll + content_height,
            )
            continue
        if key == curses.KEY_PPAGE:
            scroll = _tui_review_scroll(
                pending_action,
                scroll - content_height,
            )
            continue
        if key == ord("R"):
            invalidate(deep=True, model=True)
            cancel_confirmation()
            message = (
                "Refreshing hardware, model plan, and lifecycle "
                "validation…"
            )
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
                review = ReviewIdentity.for_prepare(
                    token,
                    plan,
                    args.base_port,
                )
                if (
                    pending_action != "prepare"
                    or pending_review != review
                    or now > pending_deadline
                ):
                    pending_action = "prepare"
                    pending_review = review
                    pending_deadline = now + 10.0
                    message = (
                        "TOTAL RAM %.2f GiB · press p again only if "
                        "the three facts above are correct."
                        % (
                            plan["staging"]["total_staged_bytes"]
                            / float(GIB)
                        )
                    )
                    scroll = 0
                else:
                    if not authorize_privileged_mounts():
                        cancel_confirmation()
                        continue
                    reviewed_token = pending_review.token
                    prepared_args = argparse.Namespace(
                        **dict(vars(args), yes=True)
                    )
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
                                        (
                                            size / elapsed / MIB
                                            if elapsed > 0
                                            else 0.0
                                        ),
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
                    message = (
                        "Preparation started; progress stays visible and "
                        "navigation remains available."
                    )
            continue

        if key == ord("s"):
            action_policy = ActionPolicy.from_state(plan, report)
            if not action_policy.start.enabled:
                message = action_policy.start.reason
            else:
                start_args = argparse.Namespace(
                    base_port=args.base_port
                )
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
                message = (
                    "Engine startup runs in the background; this can "
                    "take a while for a large model."
                )
            continue

        if key == ord("x"):
            action_policy = ActionPolicy.from_state(plan, report)
            if not action_policy.stop.enabled:
                message = action_policy.stop.reason
            else:
                begin_operation(
                    "stop",
                    "Stopping managed engines",
                    lambda op: stop(argparse.Namespace()),
                )
                message = (
                    "Stopping only verified managed process groups."
                )
            continue

        if key == ord("d"):
            action_policy = ActionPolicy.from_state(plan, report)
            if not action_policy.destroy.enabled:
                message = action_policy.destroy.reason
            else:
                try:
                    current_manifest = _load_manifest(required=True)
                    current_token = _manifest_confirmation_token(
                        current_manifest
                    )
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
                if (
                    pending_action != "destroy"
                    or now > pending_deadline
                ):
                    pending_action = "destroy"
                    pending_review = current_review
                    pending_deadline = now + 10.0
                    message = (
                        "CONFIRM DESTROY: engines are stopped; unmount "
                        "volatile weights now by pressing d again within 10s."
                    )
                elif pending_review != current_review:
                    cancel_confirmation()
                    invalidate(deep=True)
                    message = (
                        "The active deployment changed; "
                        "review Destroy again."
                    )
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
                    message = (
                        "Destroy started; durable state will be preserved."
                    )
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
                message = (
                    "Benchmark is running; results will be saved "
                    "to the scorecard."
                )
            continue

        if screen != 4:
            continue
        action_policy = ActionPolicy.from_state(plan, report)
        if not action_policy.edit_weights.enabled and (
            key != ord("P")
            or not action_policy.edit_base_port.enabled
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
                message = (
                    "Placement changed to one shared model copy "
                    "and one engine."
                )
            elif key == ord("m"):
                args.mode = (
                    "partial"
                    if args.mode == "full"
                    else "full"
                )
                args.capacity_gb = (
                    16.0
                    if args.mode == "partial"
                    else None
                )
                changed = True
                message = "Staging mode changed to %s." % args.mode
            elif key == ord("H"):
                choices = ("auto", "within_size", "advise")
                args.thp = choices[
                    (choices.index(args.thp) + 1) % len(choices)
                ]
                changed = True
                message = (
                    "Huge-page policy changed to %s." % args.thp
                )
            elif key == ord("f"):
                effective = (
                    args.prefault
                    if args.prefault is not None
                    else args.mode == "full"
                )
                args.prefault = 0 if effective else 1
                changed = True
                message = "Prefault %s." % (
                    "enabled"
                    if args.prefault
                    else "disabled"
                )
            elif key == ord("y"):
                args.allow_swappable = not args.allow_swappable
                changed = True
                message = "Swappable tmpfs %s." % (
                    "allowed"
                    if args.allow_swappable
                    else "refused"
                )
            elif key in (
                ord("c"),
                ord("r"),
                ord("o"),
                ord("P"),
                ord("w"),
            ):
                label, current = {
                    ord("c"): (
                        "Per-copy budget in GiB",
                        args.capacity_gb or 16.0,
                    ),
                    ord("r"): (
                        "Usage profile path ('-' = model default)",
                        args.profile or "<model default>",
                    ),
                    ord("o"): (
                        "Mount root",
                        args.mount_root,
                    ),
                    ord("P"): (
                        "Start base port",
                        args.base_port,
                    ),
                    ord("w"): (
                        "Concurrent copy workers",
                        args.parallel,
                    ),
                }[key]
                value = prompt_value(label, current)
                if value is not None:
                    if key == ord("c"):
                        value = float(value)
                        if (
                            not math.isfinite(value)
                            or value <= 0
                        ):
                            raise ValueError(
                                "budget must be positive"
                            )
                        args.capacity_gb = value
                    elif key == ord("r"):
                        args.profile = (
                            None
                            if value in ("-", "<model default>")
                            else value
                        )
                    elif key == ord("o"):
                        args.mount_root = value
                    elif key == ord("P"):
                        value = int(value)
                        ports = _managed_ports_for_plan(
                            plan,
                            value,
                        )
                        if (
                            not 1 <= value <= 65535
                            or len(set(ports)) != len(ports)
                            or any(
                                port < 1 or port > 65535
                                for port in ports
                            )
                        ):
                            raise ValueError(
                                "base port produces invalid or "
                                "duplicate replica ports"
                            )
                        args.base_port = value
                    elif key == ord("w"):
                        value = int(value)
                        if not 1 <= value <= 64:
                            raise ValueError(
                                "workers must be between 1 and 64"
                            )
                        args.parallel = value
                    changed = True
                    message = "%s updated." % label
        except (TypeError, ValueError) as exc:
            message = "Invalid setting: %s" % exc
        if changed:
            cancel_confirmation()
            plan_cache = plan_key_cache = None
            scroll = 0


class _TuiTerminationSignal(BaseException):
    def __init__(self, signum):
        super().__init__("terminal signal %s" % signum)
        self.signum = int(signum)


@contextlib.contextmanager
def _curses_termination_guard():
    """Make curses SIGHUP/SIGTERM follow its existing cleanup path."""
    previous = {}
    first_signal = {"signum": None}

    def terminate(received, _frame):
        if first_signal["signum"] is not None:
            # Cleanup is already running under this guard. Repeated service
            # manager/SSH signals must not restore the default disposition
            # and kill rollback halfway through.
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
    """Wait through terminal interrupts until cleanup reaches a checkpoint."""
    while True:
        try:
            active["thread"].join()
            return
        except (KeyboardInterrupt, _TuiTerminationSignal):
            if active.get("cancelable"):
                active["cancel_event"].set()


def _run_tui_frontend(callback, *, bindings):
    """Run a frontend while preserving active cleanup on every exit."""
    try:
        return callback()
    except (
        KeyboardInterrupt,
        _TuiTerminationSignal,
    ) as interruption:
        signum = getattr(interruption, "signum", None)
        exit_code = (
            128 + signum
            if signum is not None
            else 130
        )
        with bindings._tui_worker_guard:
            active = bindings._tui_worker
        if active is not None:
            if active.get("cancelable"):
                active["cancel_event"].set()
                print(
                    "coli ramdisk: termination received; "
                    "waiting for safe rollback/cleanup",
                    file=sys.stderr,
                )
            else:
                print(
                    "coli ramdisk: termination received during "
                    "non-interruptible cleanup; waiting",
                    file=sys.stderr,
                )
            _join_tui_worker(active)
            error = active.get("error")
            if (
                error is not None
                and not isinstance(error, _OperationCancelled)
            ):
                print(
                    "coli ramdisk: cleanup failed after interrupt: %s"
                    % error,
                    file=sys.stderr,
                )
                return 2
        return exit_code
    except BaseException as interface_error:
        with bindings._tui_worker_guard:
            active = bindings._tui_worker
        if active is not None:
            if active.get("cancelable"):
                active["cancel_event"].set()
            print(
                "coli ramdisk: interface exited; "
                "waiting for active cleanup",
                file=sys.stderr,
            )
            _join_tui_worker(active)
            operation_error = active.get("error")
            if (
                operation_error is not None
                and not isinstance(
                    operation_error,
                    _OperationCancelled,
                )
            ):
                print(
                    "coli ramdisk: active operation/cleanup also failed: %s"
                    % operation_error,
                    file=sys.stderr,
                )
                raise RamdiskError(
                    "interface failed while active operation cleanup "
                    "also failed: %s"
                    % operation_error
                ) from interface_error
        raise
