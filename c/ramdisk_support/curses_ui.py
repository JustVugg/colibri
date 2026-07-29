"""Legacy curses frontend with facade-owned runtime state bindings."""

from __future__ import print_function

import argparse
import copy
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
from .runtime_monitor import RuntimeMonitor
from ramdisk_ui import ActionPolicy, ReviewIdentity


_TUI_SCREENS = (
    "Plan",
    "Hardware",
    "GPUs",
    "Activity",
    "Benchmarks",
    "Settings",
)

_GPU_LAYOUTS = (
    ("experts-only", "Hot experts only"),
    ("dense-attention", "Dense + GPU attention · experimental"),
    (
        "dense-attention-sharded",
        "Dense + sharded attention · experimental",
    ),
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


def _metric_gib(value):
    if value is None:
        return "n/a"
    try:
        return "%.1f GiB" % (float(value) / GIB)
    except (TypeError, ValueError, OverflowError):
        return "n/a"


def _runtime_failure_snapshot(previous, error, observed_at=None):
    """Retain the last sample without leaving it looking live after a crash."""
    snapshot = copy.deepcopy(previous) if isinstance(previous, dict) else {}
    message = "runtime sampler failed: %s" % (
        str(error).strip() or type(error).__name__
    )
    service = snapshot.setdefault("service", {})
    service.update(
        {
            "state": "degraded",
            "label": "DEGRADED",
            "active": service.get("active"),
            "queued": service.get("queued"),
            "endpoints": list(service.get("endpoints") or ()),
            "error": message,
            "stale": True,
            "observed_at": service.get("observed_at"),
        }
    )
    freshness = snapshot.setdefault("freshness", {})
    for channel in (
        "service",
        "cards",
        "model",
        "tiers",
        "profile",
        "process",
    ):
        prior = freshness.get(channel)
        prior = prior if isinstance(prior, dict) else {}
        freshness[channel] = {
            "stale": True,
            "observed_at": prior.get("observed_at"),
            "error": message,
        }
    for card in snapshot.get("gpus") or ():
        if isinstance(card, dict):
            card["card_stale"] = True
            card["model_stale"] = True
            card["process_stale"] = True
    snapshot["tiers_stale"] = True
    snapshot["profile_stale"] = True
    snapshot["process_stale"] = True
    snapshot["sampler_error_at"] = (
        float(observed_at) if observed_at is not None else time.time()
    )
    return snapshot


def _metric_percent(value):
    if value is None:
        return "n/a"
    try:
        return "%.0f%%" % float(value)
    except (TypeError, ValueError, OverflowError):
        return "n/a"


def _metric_observed_at(value):
    try:
        return time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(float(value)),
        )
    except (OSError, OverflowError, TypeError, ValueError):
        return "unknown time"


def _gpu_eligibility(device, hardware):
    """Dependency-free fallback for builds without the shared selector."""
    if not isinstance(device, dict):
        return False, "malformed discovery record"
    index = device.get("index")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
    ):
        return False, "invalid device index"
    discovery = hardware.get("gpu_discovery") or {}
    if discovery.get("cuda_visible_devices_present"):
        return False, (
            discovery.get("selection_error")
            or "ambient CUDA_VISIBLE_DEVICES prevents safe GPU selection"
        )
    node = device.get("numa_node")
    effective = set(hardware.get("effective_nodes") or ())
    if device.get("locality") not in ("resolved", "single-node"):
        return False, "NUMA locality is unavailable"
    if (
        isinstance(node, bool)
        or not isinstance(node, int)
        or node not in effective
    ):
        return False, "NUMA node is outside the effective host mask"
    return True, None


def _same_gpu_identity(left, right):
    left_uuid = str(left.get("uuid") or "").lower()
    right_uuid = str(right.get("uuid") or "").lower()
    if left_uuid and right_uuid:
        return left_uuid == right_uuid
    left_pci = str(
        left.get("pci_bus_id")
        or left.get("pci")
        or left.get("bus_id")
        or ""
    ).lower()
    right_pci = str(
        right.get("pci_bus_id")
        or right.get("pci")
        or right.get("bus_id")
        or ""
    ).lower()
    if left_pci and right_pci:
        return left_pci == right_pci
    if (left_uuid or left_pci) and (right_uuid or right_pci):
        return False
    return left.get("index") == right.get("index")


def _selected_gpu_indices(args, plan, active=False):
    planned = (plan or {}).get("managed_accelerator") or {}
    draft = getattr(args, "managed_accelerator", None) or {}
    contract = planned if active else (draft or planned)
    return {
        int(device["index"])
        for device in contract.get("devices") or ()
        if isinstance(device, dict)
        and isinstance(device.get("index"), int)
        and not isinstance(device.get("index"), bool)
    }


def _gpu_layout(args, plan, active=False):
    planned = (plan or {}).get("managed_accelerator") or {}
    draft = getattr(args, "managed_accelerator", None) or {}
    contract = planned if active else (draft or planned)
    return str(
        contract.get("layout")
        or getattr(args, "gpu_layout", None)
        or "experts-only"
    )


def _apply_tui_gpu_selection(
    bindings,
    args,
    hardware,
    selected,
    layout,
    *,
    reset_placement,
):
    """Apply a curses draft edit through the shared planning contract."""
    apply_selection = getattr(bindings, "apply_gpu_selection", None)
    if apply_selection is None:
        raise RamdiskError(
            "this Colibri build does not expose managed GPU selection"
        )
    selector = (
        "none"
        if not selected
        else ",".join(str(index) for index in sorted(selected))
    )
    apply_selection(
        args,
        hardware,
        selector=selector,
        layout=layout,
        cuda_capable=(
            True
            if (
                getattr(args, "managed_accelerator", None) or {}
            ).get("capability") == "available"
            else None
        ),
        reset_placement=reset_placement,
    )
    args.gpu_placement = (
        "auto"
        if reset_placement
        else getattr(args, "gpu_placement", "custom")
    )
    marker = getattr(bindings, "mark_preset_custom", None)
    if marker is not None:
        marker(args)
    return selector


def _tui_gpu_rows(
    hardware,
    args,
    plan,
    report,
    runtime,
    cursor=0,
):
    """Render compact selection and live per-card metrics."""
    hardware = hardware or {}
    active = bool((report or {}).get("present"))
    devices = sorted(
        [
            device
            for device in (hardware or {}).get("gpus") or ()
            if isinstance(device, dict)
        ],
        key=lambda device: int(device.get("index", 1 << 30)),
    )
    planned = (plan or {}).get("managed_accelerator") or {}
    draft = getattr(args, "managed_accelerator", None) or {}
    selected_devices = list(
        (planned if active else (draft or planned)).get("devices") or ()
    )
    for selected_device in selected_devices:
        if not any(
            _same_gpu_identity(selected_device, device)
            for device in devices
        ):
            devices.append(dict(selected_device))
    devices.sort(key=lambda device: int(device.get("index", 1 << 30)))
    layout = _gpu_layout(args, plan, active=active)
    layout_label = dict(_GPU_LAYOUTS).get(layout, layout)
    rows = [
        (
            "heading",
            "ACCELERATORS · %s"
            % ("LOCKED BY ACTIVE DEPLOYMENT" if active else "SELECT CARDS"),
        ),
        (
            "accent" if selected_devices else "dim",
            "Layout · %s" % layout_label,
        ),
    ]
    if active:
        rows.append(
            (
                "dim",
                "Destroy the RAM workspace before changing cards or layout.",
            )
        )
    else:
        rows.extend(
            [
                (
                    "dim",
                    "↑/↓ card · Space toggle · [g] layout · "
                    "[a] all + GPU-local · [u] reset locality · [c] CPU",
                ),
                (
                    "dim",
                    "At least one card is required for GPU serving. "
                    "Dense layouts are experimental.",
                ),
            ]
        )
    rows.append(("normal", ""))

    runtime_rows = [
        item
        for item in (runtime or {}).get("gpus") or ()
        if isinstance(item, dict)
    ]
    if not devices:
        rows.append(
            (
                "warn",
                (hardware.get("gpu_discovery") or {}).get("error")
                or "No NVIDIA GPUs were discovered.",
            )
        )
        return rows
    cursor = max(0, min(int(cursor), len(devices) - 1))
    for position, device in enumerate(devices):
        index = int(device["index"])
        eligible, reason = _gpu_eligibility(device, hardware)
        is_selected = any(
            _same_gpu_identity(device, selected_device)
            for selected_device in selected_devices
        )
        marker = "▶" if position == cursor and not active else " "
        checkbox = (
            "[×]"
            if is_selected
            else "[ ]"
            if eligible
            else "[-]"
        )
        free_bytes = device.get("free_bytes")
        total_bytes = device.get("total_bytes")
        card = next(
            (
                row
                for row in runtime_rows
                if _same_gpu_identity(device, row)
            ),
            {},
        )
        if card.get("memory_free_bytes") is not None:
            free_bytes = card["memory_free_bytes"]
        if card.get("memory_total_bytes") is not None:
            total_bytes = card["memory_total_bytes"]
        style = (
            "warn"
            if not eligible
            else "accent"
            if is_selected
            else "normal"
        )
        rows.append(
            (
                style,
                "%s %s GPU %d · %s · %s · free %s / %s · NUMA %s"
                % (
                    marker,
                    checkbox,
                    index,
                    device.get("name") or "NVIDIA GPU",
                    device.get("uuid")
                    or device.get("pci_bus_id")
                    or "identity unavailable",
                    _metric_gib(free_bytes),
                    _metric_gib(total_bytes),
                    device.get("numa_node", "n/a"),
                ),
            )
        )
        if not eligible:
            rows.append(("warn", "    unavailable · %s" % reason))
        if card:
            stale_parts = []
            if card.get("card_stale"):
                stale_parts.append(
                    "card stale (last %s)"
                    % _metric_observed_at(card.get("observed_at"))
                )
            if card.get("model_stale"):
                stale_parts.append("model stale")
            if card.get("process_stale"):
                stale_parts.append("process stale")
            rows.append(
                (
                    "warn" if stale_parts else "dim",
                    "    card %s · used %s · Colibri %s · model %s"
                    "%s"
                    % (
                        _metric_percent(
                            card.get("utilization_percent")
                        ),
                        _metric_gib(card.get("memory_used_bytes")),
                        _metric_gib(card.get("process_vram_bytes")),
                        _metric_gib(
                            card.get("model_resident_bytes")
                        ),
                        (
                            " · " + " · ".join(stale_parts)
                            if stale_parts
                            else ""
                        ),
                    ),
                )
            )
            rows.append(
                (
                    "dim",
                    "    experts %s / %s · non-expert %s"
                    % (
                        card.get("expert_count")
                        if card.get("expert_count") is not None
                        else "n/a",
                        _metric_gib(card.get("expert_bytes")),
                        _metric_gib(card.get("non_expert_bytes")),
                    ),
                )
            )
    return rows


def _tui_runtime_rows(runtime):
    """Render serving proof and aggregate request/resource telemetry."""
    if not runtime:
        return [
            ("heading", "COLIBRI SERVING · SAMPLING"),
            ("dim", "Waiting for the first runtime telemetry sample."),
            ("normal", ""),
        ]
    service = runtime.get("service") or {}
    state = service.get("state", "degraded")
    style = {
        "serving": "good",
        "starting": "warn",
        "stopped": "dim",
        "degraded": "bad",
    }.get(state, "warn")
    rows = [
        (
            "heading",
            "COLIBRI SERVING · %s" % service.get("label", state.upper()),
        ),
        (
            style,
            "API %s · active %s · queued %s · process RSS %s%s"
            % (
                "healthy" if state == "serving" else state,
                service.get("active")
                if service.get("active") is not None
                else "n/a",
                service.get("queued")
                if service.get("queued") is not None
                else "n/a",
                _metric_gib(runtime.get("process_rss_bytes")),
                " · stale"
                if service.get("stale") or runtime.get("process_stale")
                else "",
            ),
        ),
    ]
    for endpoint in service.get("endpoints") or ():
        rows.append(
            (
                "good"
                if endpoint.get("process_verified")
                and endpoint.get("health_ok")
                else "bad",
                "  %s · process %s · health %s"
                % (
                    endpoint.get("url"),
                    "verified"
                    if endpoint.get("process_verified")
                    else "unverified",
                    "ok" if endpoint.get("health_ok") else "unavailable",
                ),
            )
        )
    if service.get("error"):
        rows.append(("warn", "  %s" % service["error"]))
    tiers = runtime.get("tiers") or {}
    rows.extend(
        [
            ("normal", ""),
            ("heading", "MODEL PLACEMENT"),
            (
                "warn" if runtime.get("tiers_stale") else "normal",
                "Experts · VRAM %s (%s) · RAM %s (%s) · disk %s%s"
                % (
                    tiers.get("vram", "n/a"),
                    (
                        "%.1f GB" % float(tiers["vram_gb"])
                        if tiers.get("vram_gb") is not None
                        else "n/a"
                    ),
                    tiers.get("ram", "n/a"),
                    (
                        "%.1f GB" % float(tiers["ram_gb"])
                        if tiers.get("ram_gb") is not None
                        else "n/a"
                    ),
                    tiers.get("disk", "n/a"),
                    " · stale" if runtime.get("tiers_stale") else "",
                ),
            ),
        ]
    )
    profile = runtime.get("latest_profile")
    if profile:
        rows.extend(
            [
                ("normal", ""),
                ("heading", "LATEST COMPLETED REQUEST"),
                (
                    "warn" if runtime.get("profile_stale") else "normal",
                    "%s tok/s · TTFT %s ms · wall %s s%s"
                    % (
                        (
                            "%.2f"
                            % float(profile["tokens_per_second"])
                            if profile.get("tokens_per_second") is not None
                            else "n/a"
                        ),
                        (
                            "%.1f" % float(profile["ttft_ms"])
                            if profile.get("ttft_ms") is not None
                            else "n/a"
                        ),
                        (
                            "%.2f" % float(profile["wall_s"])
                            if profile.get("wall_s") is not None
                            else "n/a"
                        ),
                        " · stale"
                        if runtime.get("profile_stale")
                        else "",
                    ),
                ),
                (
                    "dim",
                    "Expert disk/wait/matmul %s/%s/%s s · "
                    "attention/head %s/%s s"
                    % tuple(
                        (
                            "%.2f" % float(profile[key])
                            if profile.get(key) is not None
                            else "n/a"
                        )
                        for key in (
                            "expert_disk_s",
                            "expert_wait_s",
                            "expert_matmul_s",
                            "attention_s",
                            "lm_head_s",
                        )
                    ),
                ),
            ]
        )
    else:
        rows.append(("dim", "No completed profiled request yet."))
    freshness = runtime.get("freshness") or {}
    for name, channel in freshness.items():
        if not isinstance(channel, dict) or not channel.get("stale"):
            continue
        rows.append(
            (
                "warn",
                "%s telemetry stale · last successful sample %s"
                % (
                    name,
                    _metric_observed_at(channel.get("observed_at")),
                ),
            )
        )
    rows.append(("normal", ""))
    return rows


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
    _tui_preset_rows = bindings._tui_preset_rows
    _tui_settings_rows = bindings._tui_settings_rows
    _validate_noninteractive_sudo = bindings._validate_noninteractive_sudo
    benchmark = bindings.benchmark
    build_plan = bindings.build_plan
    current_euid = bindings.current_euid
    destroy = bindings.destroy
    discover_hardware = bindings.discover_hardware
    mark_preset_custom = bindings.mark_preset_custom
    prepare = bindings.prepare
    resolve_preset = bindings.resolve_preset
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
        ("memory_nodes", None),
        ("cpu_list", None),
        ("gpu", None),
        ("gpu_layout", "experts-only"),
    ):
        if not hasattr(args, name):
            setattr(args, name, value)
    if not hasattr(args, "gpu_placement"):
        args.gpu_placement = (
            "custom"
            if args.memory_nodes is not None or args.cpu_list is not None
            else "auto"
        )

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
    runtime_monitor = RuntimeMonitor()
    runtime_snapshot = None
    runtime_checked = 0.0
    runtime_job = None
    runtime_generation = 0
    runtime_lock = threading.Lock()
    hardware_job = None
    hardware_generation = 0
    hardware_lock = threading.Lock()
    gpu_cursor = 0
    deep_status_refresh = True
    operation = None
    quit_when_idle = False
    quit_exit_code = 0
    operation_lock = threading.Lock()
    preset_prompt_pending = not bool(
        getattr(args, "ramdisk_preset", None)
    )
    preset_prompt_suppressed = False

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
        nonlocal deep_status_refresh, hardware_checked
        nonlocal hardware_generation
        plan_cache = plan_key_cache = None
        report_cache = active_manifest_cache = history_cache = None
        if model:
            model_cache = None
        if deep:
            # Keep the last snapshot visible while a replacement is discovered
            # off the input thread.
            hardware_checked = 0.0
            hardware_generation += 1
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

    def harvest_runtime_poll():
        nonlocal runtime_job, runtime_snapshot, runtime_checked
        if runtime_job is None:
            return
        with runtime_lock:
            if not runtime_job["done"]:
                return
            result = runtime_job["result"]
            generation = runtime_job["generation"]
            runtime_job = None
        runtime_checked = time.monotonic()
        if result is not None and generation == runtime_generation:
            runtime_snapshot = result

    def harvest_hardware_poll():
        nonlocal hardware_job, hardware_cache, hardware_checked
        nonlocal plan_cache, plan_key_cache, message
        if hardware_job is None:
            return
        with hardware_lock:
            if not hardware_job["done"]:
                return
            result = hardware_job["result"]
            error = hardware_job["error"]
            generation = hardware_job["generation"]
            hardware_job = None
        hardware_checked = time.monotonic()
        if generation != hardware_generation:
            return
        if result is not None:
            hardware_cache = result
            plan_cache = plan_key_cache = None
        elif error is not None:
            message = (
                "Hardware refresh failed; retaining the last snapshot: %s"
                % error
            )

    def begin_hardware_poll():
        nonlocal hardware_job
        if hardware_job is not None:
            return
        job = {
            "done": False,
            "result": None,
            "error": None,
            "generation": hardware_generation,
        }

        def runner():
            try:
                result = discover_hardware()
                error = None
            except Exception as exc:
                result = None
                error = exc
            with hardware_lock:
                job["result"] = result
                job["error"] = error
                job["done"] = True

        hardware_job = job
        thread = threading.Thread(
            target=runner,
            name="coli-ramdisk-hardware-discovery",
        )
        thread.daemon = True
        job["thread"] = thread
        thread.start()

    def begin_runtime_poll(manifest, report, plan, hardware, checked):
        nonlocal runtime_job, runtime_checked
        if runtime_job is not None:
            return
        job = {
            "done": False,
            "result": None,
            "generation": runtime_generation,
            "previous": copy.deepcopy(runtime_snapshot),
        }

        def runner():
            try:
                result = runtime_monitor.sample(
                    manifest,
                    report,
                    plan,
                    hardware,
                )
            except Exception as exc:
                result = _runtime_failure_snapshot(
                    job["previous"],
                    exc,
                )
            with runtime_lock:
                job["result"] = result
                job["done"] = True

        runtime_job = job
        runtime_checked = checked
        thread = threading.Thread(
            target=runner,
            name="coli-ramdisk-runtime-monitor",
        )
        thread.daemon = True
        job["thread"] = thread
        thread.start()

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
        harvest_hardware_poll()
        harvest_runtime_poll()
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
        preset_selecting = False
        rows = []
        try:
            if hardware_cache is None:
                hardware_cache = discover_hardware()
                hardware_checked = now
                plan_key_cache = None
            elif (
                hardware_job is None
                and now - hardware_checked >= 30.0
            ):
                begin_hardware_poll()
            hardware = hardware_cache
            if report_cache is None or now - report_checked >= 2.0:
                report_cache = status(deep=deep_status_refresh)
                deep_status_refresh = False
                report_checked = now
                active_manifest_cache = None
            report = report_cache
            active = bool(report.get("present"))
            if active:
                preset_prompt_suppressed = True
                if active_manifest_cache is None:
                    active_manifest_cache = _load_manifest(required=True)
                plan = active_manifest_cache["plan"]
                processes = report.get("processes", [])
                deployment_identity = (
                    active_manifest_cache.get("deployment_id"),
                    active_manifest_cache.get("created_at"),
                )
                if deployment_identity != active_deployment_identity:
                    runtime_generation += 1
                    runtime_snapshot = None
                    runtime_checked = 0.0
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
                if active_deployment_identity is not None:
                    runtime_generation += 1
                    runtime_snapshot = None
                    runtime_checked = 0.0
                active_deployment_identity = None
                preset_selecting = bool(
                    preset_prompt_pending
                    and not preset_prompt_suppressed
                )
                if not preset_selecting:
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
                        args.memory_nodes,
                        args.cpu_list,
                        repr(getattr(args, "managed_accelerator", None)),
                        getattr(args, "ramdisk_preset", None),
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

            if pending_action == "prepare" and plan is not None:
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
            if (
                not preset_selecting
                and plan is not None
                and runtime_job is None
                and now - runtime_checked >= 2.0
            ):
                begin_runtime_poll(
                    copy.deepcopy(active_manifest_cache)
                    if active
                    else None,
                    copy.deepcopy(report),
                    copy.deepcopy(plan),
                    copy.deepcopy(hardware),
                    now,
                )
            if preset_selecting:
                rows = _tui_preset_rows()
            elif help_open:
                rows = _tui_help_rows() + [
                    ("normal", ""),
                    ("heading", "GPUs PAGE"),
                    (
                        "normal",
                        "↑/↓ · choose card    Space · toggle selected card",
                    ),
                    (
                        "normal",
                        "g · cycle placement layout    "
                        "a/u · select all or reset GPU-local placement",
                    ),
                    (
                        "normal",
                        "c · switch draft to CPU-only serving",
                    ),
                ]
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
                rows = _tui_gpu_rows(
                    hardware,
                    args,
                    plan,
                    report,
                    runtime_snapshot,
                    cursor=gpu_cursor,
                )
            elif screen == 3:
                if now - metrics_checked >= 2.0:
                    metrics_cache = {
                        process.get("pid"): _managed_process_metrics(
                            process
                        )
                        for process in report.get("processes", [])
                        if process.get("pid") is not None
                    }
                    metrics_checked = now
                rows = _tui_runtime_rows(
                    runtime_snapshot
                ) + _tui_activity_rows(
                    report,
                    hardware,
                    metrics_cache,
                )
            elif screen == 4:
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
        elif preset_selecting:
            action_hint = "[Enter] default · [1-4] choose"
        elif screen == 2 and not (report or {}).get("present"):
            action_hint = (
                "[Space] select  [g] layout  [a] all  [u] GPU-local"
            )
        else:
            policy_screen = screen - 1 if screen >= 3 else screen
            action_hint = _tui_idle_action_hint(
                policy_screen,
                plan,
                report,
            )
        navigation = (
            "←/→ pages · ↑/↓ cards · PgUp/PgDn scroll"
            if screen == 2 and not help_open
            else "←/→ pages · ↑/↓ scroll"
        )
        footer = "%s · %s · [?] help · [q] quit" % (
            action_hint,
            navigation,
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
        if preset_selecting:
            choices = tuple(
                preset_id
                for preset_id, _label, _description
                in bindings.PRESET_CHOICES
            )
            if key in (10, 13, curses.KEY_ENTER):
                preset_id = choices[0]
            elif ord("1") <= key <= ord("4"):
                preset_id = choices[key - ord("1")]
            else:
                message = (
                    "Choose 1-4, or press Enter for Fastest GPU staging."
                )
                continue
            try:
                if model_cache is None:
                    model_cache = scan_model(args.model)
                result = resolve_preset(
                    preset_id,
                    args,
                    hardware=hardware,
                    model=model_cache,
                    cli_path=cli_path,
                    engine_path=engine_path,
                )
                vars(args).clear()
                vars(args).update(vars(result["args"]))
                args.gpu_placement = (
                    "auto"
                    if getattr(args, "managed_accelerator", None)
                    else getattr(args, "gpu_placement", "auto")
                )
                plan_cache = result["plan"]
                plan_key_cache = None
                preset_prompt_pending = False
                scroll = 0
                message = (
                    "%s populated the draft; review the exact plan."
                    % result["plan"].get("preset", {}).get(
                        "label",
                        "Preset",
                    )
                )
            except (OSError, ValueError, RamdiskError) as exc:
                message = "Preset could not be resolved: %s" % exc
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
        if (
            screen == 2
            and not help_open
            and key in (ord("j"), curses.KEY_DOWN)
        ):
            gpu_cursor = min(
                max(0, len((hardware or {}).get("gpus") or ()) - 1),
                gpu_cursor + 1,
            )
            continue
        if (
            screen == 2
            and not help_open
            and key in (ord("k"), curses.KEY_UP)
        ):
            gpu_cursor = max(0, gpu_cursor - 1)
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

        if screen == 2 and key in (
            ord(" "),
            ord("g"),
            ord("a"),
            ord("u"),
            ord("c"),
        ):
            if report.get("present"):
                message = (
                    "GPU selection is locked by the active deployment; "
                    "Destroy it before editing."
                )
                continue
            devices = sorted(
                [
                    device
                    for device in (hardware or {}).get("gpus") or ()
                    if isinstance(device, dict)
                ],
                key=lambda device: int(device.get("index", 1 << 30)),
            )
            gpu_cursor = max(
                0,
                min(gpu_cursor, max(0, len(devices) - 1)),
            )
            selected = _selected_gpu_indices(args, plan)
            layout = _gpu_layout(args, plan)
            reset_placement = (
                getattr(args, "gpu_placement", "auto") != "custom"
            )
            try:
                if key == ord(" "):
                    if not devices:
                        raise RamdiskError("no NVIDIA GPU was discovered")
                    device = devices[gpu_cursor]
                    eligible, reason = _gpu_eligibility(device, hardware)
                    if not eligible:
                        raise RamdiskError(
                            "GPU %s is unavailable: %s"
                            % (device.get("index"), reason)
                        )
                    index = int(device["index"])
                    if index in selected:
                        selected.remove(index)
                    else:
                        selected.add(index)
                    if not selected:
                        raise RamdiskError(
                            "a GPU layout requires at least one card; "
                            "press c for CPU-only serving"
                        )
                elif key == ord("a"):
                    selected = {
                        int(device["index"])
                        for device in devices
                        if _gpu_eligibility(device, hardware)[0]
                    }
                    if not selected:
                        raise RamdiskError(
                            "no usable NVIDIA GPU was discovered"
                        )
                    reset_placement = True
                elif key == ord("u"):
                    if not selected:
                        raise RamdiskError(
                            "select at least one GPU before resetting locality"
                        )
                    reset_placement = True
                elif key == ord("c"):
                    selected = set()
                    layout = "experts-only"
                    reset_placement = True
                else:
                    if not selected:
                        raise RamdiskError(
                            "select at least one GPU before changing layout"
                        )
                    choices = [
                        value for value, _label in _GPU_LAYOUTS
                    ]
                    if len(selected) < 2:
                        choices.remove("dense-attention-sharded")
                    layout = choices[
                        (choices.index(layout) + 1) % len(choices)
                    ]
                selector = _apply_tui_gpu_selection(
                    bindings,
                    args,
                    hardware,
                    selected,
                    layout,
                    reset_placement=reset_placement,
                )
                cancel_confirmation()
                plan_cache = plan_key_cache = None
                scroll = 0
                if selector == "none":
                    message = "Managed serving changed to CPU only."
                else:
                    message = (
                        "Managed GPUs %s · %s%s"
                        % (
                            selector,
                            dict(_GPU_LAYOUTS).get(layout, layout),
                            " · placement reset to GPU-local"
                            if reset_placement
                            else " · custom placement preserved",
                        )
                    )
            except (OSError, TypeError, ValueError, RamdiskError) as exc:
                message = "GPU selection was not changed: %s" % exc
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

        if key == ord("b") and screen == 4:
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

        if screen != 5:
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
            if key != ord("P"):
                mark_preset_custom(args)
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
