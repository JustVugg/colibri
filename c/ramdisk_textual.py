"""Guided Textual frontend for the Colibri RAM-workspace lifecycle.

The frontend deliberately owns no placement or lifecycle rules.  It presents
the immutable projections from :mod:`ramdisk_ui`, mutates only the shared
argument namespace, and sends every plan rebuild and lifecycle action back to
``ramdisk.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import math
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Static

from ramdisk_support.presets import (
    PRESET_CHOICES,
    PRESET_GPU_FASTEST,
)
from ramdisk_ui import (
    ActionPermission,
    ActionPolicy,
    DeploymentHealth,
    HealthLevel,
    PlacementContract,
    ReviewAction,
    ReviewIdentity,
)


GIB = 1 << 30
MIB = 1 << 20
STEPS = (
    ("inspect", "Inspect"),
    ("placement", "Placement"),
    ("capacity", "Capacity"),
    ("runtime", "Runtime"),
    ("review", "Review"),
    ("operate", "Operate"),
)


@dataclass(frozen=True)
class ConsoleSnapshot:
    """One coherent read of the draft or persisted deployment."""

    plan: Optional[Mapping[str, Any]]
    report: Mapping[str, Any]
    hardware: Mapping[str, Any]
    manifest: Optional[Mapping[str, Any]] = None
    history: Optional[Mapping[str, Any]] = None
    base_port: int = 8000
    error: str = ""

    @classmethod
    def loading(cls, base_port: int = 8000) -> "ConsoleSnapshot":
        return cls(
            plan=None,
            report={"present": False, "state": "loading"},
            hardware={},
            base_port=base_port,
        )


def _normalize_args(initial: argparse.Namespace) -> argparse.Namespace:
    args = argparse.Namespace(**vars(initial))
    defaults = (
        ("mode", "full"),
        ("topology", "interleaved"),
        ("mount_root", "/mnt/colibri-ram"),
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
    )
    for name, value in defaults:
        if not hasattr(args, name):
            setattr(args, name, value)
    if args.base_port is None:
        args.base_port = 8000
    return args


class LifecycleSession:
    """Cached, serialized reads around the lifecycle module."""

    def __init__(self, lifecycle: Any, args: argparse.Namespace) -> None:
        self.lifecycle = lifecycle
        self.args = args
        self._lock = threading.RLock()
        self._hardware: Optional[Mapping[str, Any]] = None
        self._model: Optional[Mapping[str, Any]] = None
        self._plan: Optional[Mapping[str, Any]] = None
        self._plan_key: Optional[tuple[Any, ...]] = None
        self._hardware_checked = 0.0
        self._first_status = True

    def invalidate(self, *, deep: bool = False, model: bool = False) -> None:
        with self._lock:
            self._plan = None
            self._plan_key = None
            if deep:
                self._hardware = None
                self._first_status = True
            if model:
                self._model = None

    def resolve_preset(
        self,
        preset_id: str,
        *,
        cli_path: Optional[str] = None,
        engine_path: Optional[str] = None,
    ) -> Mapping[str, Any]:
        with self._lock:
            if self._hardware is None:
                self._hardware = self.lifecycle.discover_hardware()
                self._hardware_checked = time.monotonic()
            if self._model is None:
                self._model = self.lifecycle.scan_model(self.args.model)
            result = self.lifecycle.resolve_preset(
                preset_id,
                self.args,
                hardware=self._hardware,
                model=self._model,
                cli_path=cli_path,
                engine_path=engine_path,
            )
            resolved_args = result["args"]
            vars(self.args).clear()
            vars(self.args).update(vars(resolved_args))
            self._plan = result["plan"]
            self._plan_key = None
            return result

    def refresh(self, *, deep: bool = False, model: bool = False) -> ConsoleSnapshot:
        with self._lock:
            if model:
                self._model = None
            now = time.monotonic()
            if deep or self._hardware is None or now - self._hardware_checked >= 30.0:
                self._hardware = self.lifecycle.discover_hardware()
                self._hardware_checked = now
                self._plan = None
                self._plan_key = None

            report = self.lifecycle.status(deep=bool(deep or self._first_status))
            self._first_status = False
            manifest = None
            if report.get("present"):
                manifest = self.lifecycle._load_manifest(required=True)
                plan = manifest["plan"]
                self.args.base_port = self.lifecycle._persisted_base_port(manifest)
                processes = tuple(report.get("processes", ()))
                if (
                    report.get("state") in ("running", "starting")
                    and processes
                    and processes[0].get("port") is not None
                ):
                    first = processes[0]
                    self.args.base_port = int(first["port"]) - int(
                        first.get("node") or 0
                    )
            else:
                if self._model is None:
                    self._model = self.lifecycle.scan_model(self.args.model)
                plan_key = (
                    self.args.mode,
                    self.args.topology,
                    self.args.mount_root,
                    self.args.capacity_gb,
                    self.args.profile,
                    self.args.allow_swappable,
                    self.args.thp,
                    self.args.prefault,
                    self.args.parallel,
                    self.args.ctx,
                    self.args.memory_nodes,
                    self.args.cpu_list,
                    self._hardware_checked,
                )
                if self._plan is None or self._plan_key != plan_key:
                    self._plan = self.lifecycle.build_plan(
                        self.args,
                        hardware=self._hardware,
                        model=self._model,
                    )
                    self._plan_key = plan_key
                plan = self._plan

            history = None
            read_json = getattr(self.lifecycle, "_read_json", None)
            benchmark_path = getattr(self.lifecycle, "_benchmarks_path", None)
            if read_json is not None and benchmark_path is not None:
                history = read_json(benchmark_path()) or {"results": []}
            return ConsoleSnapshot(
                plan=plan,
                report=report,
                hardware=self._hardware or {},
                manifest=manifest,
                history=history,
                base_port=int(self.args.base_port),
            )


def _format_gib(value: Any) -> str:
    try:
        return f"{int(value) / float(GIB):,.2f} GiB"
    except (TypeError, ValueError):
        return "unknown"


def _format_list(values: Any, *, empty: str = "none") -> str:
    if values is None:
        return empty
    if isinstance(values, str):
        return values or empty
    try:
        rendered = ", ".join(str(value) for value in values)
    except TypeError:
        rendered = str(values)
    return rendered or empty


def _line(
    content: Text,
    label: str,
    value: Any = "",
    *,
    style: str = "",
    label_style: str = "bold #91a7a6",
) -> None:
    content.append(label, style=label_style)
    if value != "":
        content.append("  ")
        content.append(str(value), style=style)
    content.append("\n")


def _section(content: Text, title: str) -> None:
    if content:
        content.append("\n")
    content.append(title.upper(), style="bold #63bec2")
    content.append("\n")


def _selected_memory_nodes(plan: Mapping[str, Any]) -> Sequence[Any]:
    placement = plan.get("placement", {})
    selected = placement.get("memory_nodes")
    if selected is None:
        selected = plan.get("hardware", {}).get("online_nodes", ())
    return selected


def _selected_cpu_list(plan: Mapping[str, Any]) -> str:
    placement = plan.get("placement", {})
    value = placement.get("cpu_list")
    if value:
        return str(value)
    cpus = placement.get("cpus")
    if cpus:
        return _format_list(cpus)
    return "all effective whole cores on selected NUMA nodes"


def _engine_cpu_masks(plan: Mapping[str, Any]) -> str:
    placement = plan.get("placement", {})
    masks = placement.get("engine_cpu_masks")
    if masks:
        return _format_list(masks)
    engine_sets = placement.get("engine_cpu_sets") or ()
    rendered = []
    for target in engine_sets:
        node = target.get("node")
        label = "shared" if node is None else f"N{node}"
        rendered.append(f"{label}: {target.get('cpu_list') or 'none'}")
    return "; ".join(rendered)


def placement_backplane(snapshot: ConsoleSnapshot) -> Text:
    """Build the always-visible placement contract."""

    text = Text()
    plan = snapshot.plan
    if plan is None:
        text.append("PLACEMENT BACKPLANE", style="bold #63bec2")
        text.append("\nInspecting model, memory nodes, and runtime endpoints…")
        return text

    contract = PlacementContract.from_plan(plan, snapshot.base_port)
    nodes = tuple(_selected_memory_nodes(plan))
    node_cells = " │ ".join(f"N{node}" for node in nodes) or "host"
    ports = ", ".join(str(port) for port in contract.ports)
    total = _format_gib(contract.total_staged_bytes)
    if contract.is_replication:
        text.append(
            "DANGER · PER-NODE REPLICATION · NOT MODEL SHARDING",
            style="bold white on #8f2730",
        )
        text.append("\n")
        text.append(
            f"MODEL ×{contract.copy_count}  ──  RAM [{node_cells}]  ──  "
            f"ENGINES ×{contract.engine_count}  ──  PORTS {ports}",
            style="bold #e2675a",
        )
        text.append("\n")
        text.append(
            f"{contract.copy_count} complete staged copies consume {total} total RAM.",
            style="#e98d85",
        )
    else:
        text.append("PLACEMENT BACKPLANE · SHARED", style="bold #63bec2")
        text.append("\n")
        text.append(
            f"MODEL ×1  ──  RAM [{node_cells}]  ──  ENGINE ×1  ──  PORT {ports}",
            style="bold #d7e4e3",
        )
        text.append("\n")
        text.append(
            f"One staged copy ({total}). One engine. CPU count multiplies neither.",
            style="#91a7a6",
        )
    return text


def render_step(snapshot: ConsoleSnapshot, step: str) -> Text:
    plan = snapshot.plan
    report = snapshot.report
    hardware = snapshot.hardware
    text = Text()
    if snapshot.error:
        text.append("REFRESH NEEDS ATTENTION\n", style="bold #e2675a")
        text.append(snapshot.error, style="#e98d85")
        text.append("\nNo lifecycle action was taken. Press R to retry.")
        return text
    if plan is None:
        text.append("Inspecting this server…", style="bold #d7a84e")
        return text

    contract = PlacementContract.from_plan(plan, snapshot.base_port)
    model = plan.get("model", {})
    staging = plan.get("staging", {})
    reserve = plan.get("reserve", {})

    if step == "inspect":
        text.append("First, verify the machine and canonical model.\n", style="#d7e4e3")
        _section(text, "Model source")
        _line(text, "Path", model.get("path", "unknown"))
        _line(text, "Model", model.get("name", os.path.basename(str(model.get("path", ""))) or "unknown"))
        _line(text, "Shards", model.get("shard_count", len(model.get("shards", ()))))
        _line(text, "Selected for RAM", len(staging.get("selected_shards", ())))
        _section(text, "Server")
        memory = hardware.get("memory", {})
        _line(text, "Available memory", _format_gib(memory.get("available_bytes")))
        _line(text, "Host physical cores", hardware.get("physical_cores", "unknown"))
        _line(
            text,
            "Effective whole cores",
            hardware.get("effective_physical_cores", "unknown"),
        )
        _line(text, "Online NUMA nodes", _format_list(hardware.get("online_nodes", ())))
        _line(
            text,
            "Effective NUMA nodes",
            _format_list(
                hardware.get(
                    "effective_nodes", hardware.get("online_nodes", ())
                )
            ),
        )
        _line(text, "Kernel", hardware.get("kernel_release", "unknown"))
        node_rows = tuple(hardware.get("nodes", ()))
        if node_rows:
            _section(text, "NUMA inventory")
            online_nodes = tuple(hardware.get("online_nodes", ()))
            for node in node_rows:
                node_id = node.get("id", "?")
                effective_cpu_list = node.get("effective_cpu_list") or "none allowed"
                total = _format_gib(node.get("memory_total_bytes"))
                available = _format_gib(node.get("memory_available_bytes"))
                cores = node.get("physical_cores", "unknown")
                distances = tuple(node.get("distance", ()))
                distance_text = " ".join(
                    f"N{other}:{distance}"
                    for other, distance in zip(online_nodes, distances)
                )
                _line(
                    text,
                    f"N{node_id}",
                    f"{available} free / {total} · {cores} physical cores",
                    style="bold #d7e4e3",
                )
                _line(text, "  Effective CPUs", effective_cpu_list)
                if distance_text:
                    _line(text, "  NUMA distance", distance_text)
        text.append(
            "\nNUMA nodes control RAM placement. CPU cores do not create model copies.",
            style="bold #d7a84e",
        )
    elif step == "placement":
        text.append("Choose where memory pages and whole CPU cores may run.\n")
        _section(text, "Placement contract")
        _line(
            text,
            "Topology",
            "shared / interleaved" if contract.is_shared else "per-node replicas",
            style="bold #63bec2" if contract.is_shared else "bold #e2675a",
        )
        _line(text, "Memory NUMA nodes", _format_list(_selected_memory_nodes(plan)))
        _line(text, "Whole-core CPU list", _selected_cpu_list(plan))
        _line(text, "Copies", contract.copy_count)
        _line(text, "Engines", contract.engine_count)
        _line(text, "Mounts", _format_list(contract.mount_paths))
        engine_masks = _engine_cpu_masks(plan)
        if engine_masks:
            _line(text, "Per-engine CPU masks", engine_masks)
        remote_cpu_list = plan.get("placement", {}).get("remote_cpu_list")
        if remote_cpu_list:
            _line(
                text,
                "Remote-memory CPUs",
                remote_cpu_list,
                style="bold #d7a84e",
            )
        if contract.is_replication:
            text.append(
                "\nDANGER: every selected NUMA node receives the complete staged set "
                "and an independent engine. This is replication, not sharding.",
                style="bold #e2675a",
            )
        else:
            text.append(
                "\nRecommended: one model copy is spread across selected memory nodes "
                "and served by one engine.",
                style="#8bd3d6",
            )
            if len(_selected_memory_nodes(plan)) > 1:
                text.append(
                    "\nLinux interleave is balanced and sampled after staging, but may "
                    "fall back outside the mask under severe memory pressure.",
                    style="#d7a84e",
                )
        text.append(
            "\nDIMM and memory-channel details are informational; Linux exposes "
            "NUMA-node placement as the selectable control.",
            style="#91a7a6",
        )
    elif step == "capacity":
        text.append("Confirm that the staged set and operating reserve fit.\n")
        _section(text, "RAM cost")
        _line(text, "Staging mode", contract.mode)
        _line(text, "Per copy", _format_gib(contract.staged_bytes_per_copy))
        _line(text, "Copy count", contract.copy_count)
        _line(text, "Total staged RAM", _format_gib(contract.total_staged_bytes), style="bold")
        _line(text, "Host memory required", _format_gib(reserve.get("total_required_bytes")))
        _line(text, "Currently available", _format_gib(reserve.get("available_bytes")))
        _line(
            text,
            "OS reserve",
            _format_gib(
                reserve.get("total_os_margin_bytes", reserve.get("os_margin_bytes"))
            ),
        )
        blockers = tuple(plan.get("blockers", ()))
        warnings = tuple(plan.get("warnings", ()))
        _section(text, "Readiness")
        if blockers:
            text.append("NOT READY\n", style="bold #e2675a")
            for blocker in blockers:
                text.append(f"• {blocker}\n", style="#e98d85")
        else:
            text.append("READY · required capacity is available.\n", style="bold #73be8b")
        for warning in warnings:
            text.append(f"• {warning}\n", style="#d7a84e")
    elif step == "runtime":
        text.append("Review the process and endpoint consequences before staging.\n")
        _section(text, "Managed runtime")
        _line(text, "Engines after Start", contract.engine_count)
        _line(text, "Ports", _format_list(contract.ports))
        _line(text, "Whole-core CPU list", _selected_cpu_list(plan))
        engine_masks = _engine_cpu_masks(plan)
        if engine_masks:
            _line(text, "Per-engine CPU masks", engine_masks)
        _line(text, "Context length", plan.get("managed_runtime", {}).get("ctx", getattr(plan, "ctx", "default")))
        _line(text, "Copy workers", plan.get("parallel", "unknown"))
        _line(text, "Prefault", "on" if plan.get("prefault") else "off")
        _line(text, "Huge pages", plan.get("mount_options", {}).get("thp", "auto"))
        text.append(
            "\nCopy workers affect staging throughput only; they do not create model replicas.",
            style="bold #d7a84e",
        )
    elif step == "review":
        text.append("Nothing changes until Prepare is confirmed.\n")
        _section(text, "Exact preparation")
        copy_name = "complete model" if contract.mode == "full" else "selected shard set"
        _line(text, "Staged copies", f"{contract.copy_count} {copy_name} " + ("copy" if contract.copy_count == 1 else "copies"))
        _line(text, "Total staged RAM", _format_gib(contract.total_staged_bytes), style="bold")
        _line(text, "Engines after Start", contract.engine_count)
        _line(text, "Ports", _format_list(contract.ports))
        _line(text, "Memory NUMA nodes", _format_list(_selected_memory_nodes(plan)))
        _line(text, "Whole-core CPU list", _selected_cpu_list(plan))
        _line(text, "Mount paths", _format_list(contract.mount_paths))
        if contract.is_replication:
            text.append(
                "\nREPLICA REVIEW: the full staged set is copied once per node. "
                "Confirm only if independent full-model services are intended.",
                style="bold white on #8f2730",
            )
        elif plan.get("blockers"):
            text.append("\nPreparation is blocked. Return to Capacity.", style="bold #e2675a")
        else:
            text.append(
                "\nShared contract confirmed: one copy and one engine.",
                style="bold #73be8b",
            )
    else:
        state = report.get("state", "unknown")
        text.append("Operate only the deployment Colibri can verify.\n")
        _section(text, "Lifecycle")
        _line(text, "State", state, style="bold")
        _line(text, "Mount records", len(report.get("mounts", ())))
        _line(text, "Managed processes", len(report.get("processes", ())))
        _line(text, "Ports", _format_list(report.get("ports", contract.ports)))
        if report.get("present"):
            try:
                health = DeploymentHealth.from_report(plan, report)
                health_labels = {
                    HealthLevel.VERIFIED: ("verified", "bold #73be8b"),
                    HealthLevel.FAST_CHECK: ("fast check passed", "bold #d7a84e"),
                    HealthLevel.NEEDS_ATTENTION: ("needs attention", "bold #e2675a"),
                }
                label, style = health_labels[health.level]
                _line(text, "Deployment health", label, style=style)
            except (KeyError, TypeError, ValueError):
                _line(text, "Deployment health", "refresh required", style="#d7a84e")
        else:
            _line(text, "Deployment health", "not prepared")
        if report.get("error"):
            text.append(f"\n{report['error']}", style="#e2675a")
        history = snapshot.history or {}
        results = tuple(history.get("results", ()))
        if results:
            _section(text, "Last benchmark")
            latest = results[-1]
            _line(text, "Best path", latest.get("best_variant", latest.get("variant", "recorded")))
    return text


def _permission_line(name: str, permission: ActionPermission) -> tuple[str, str]:
    if permission.enabled:
        return (f"✓ {name}: available", "#73be8b")
    return (f"— {name}: {permission.reason}", "#91a7a6")


def review_facts(
    identity: ReviewIdentity,
    plan: Optional[Mapping[str, Any]] = None,
) -> Text:
    contract = identity.placement
    text = Text()
    action = identity.action
    if action is ReviewAction.PREPARE:
        text.append("PREPARE EXACTLY THIS CONTRACT\n", style="bold #d7a84e")
    else:
        text.append("DESTROY EXACTLY THIS WORKSPACE\n", style="bold #e2675a")
    _line(text, "Topology", contract.topology)
    _line(
        text,
        "Complete staged copies"
        if contract.mode == "full"
        else "Selected-set copies",
        contract.copy_count,
        style="bold",
    )
    _line(text, "Total staged RAM", _format_gib(contract.total_staged_bytes), style="bold")
    _line(text, "Managed engines", contract.engine_count, style="bold")
    _line(text, "Ports", _format_list(contract.ports))
    selected_nodes = (
        _selected_memory_nodes(plan)
        if plan is not None
        else contract.numa_nodes
    )
    _line(text, "Memory NUMA nodes", _format_list(selected_nodes))
    if plan is not None:
        _line(text, "Whole-core CPU list", _selected_cpu_list(plan))
        engine_masks = _engine_cpu_masks(plan)
        if engine_masks:
            _line(text, "Per-engine CPU masks", engine_masks)
        accelerator = plan.get("managed_accelerator") or {}
        if accelerator.get("mode") == "cuda":
            _line(
                text,
                "Managed GPUs",
                _format_list(
                    [
                        "%s (%s)"
                        % (
                            device.get("index"),
                            device.get("name") or "unnamed",
                        )
                        for device in accelerator.get("devices", [])
                    ]
                ),
                style="bold",
            )
            _line(
                text,
                "RAM → VRAM source",
                "mmap · async · automatic VRAM tier",
                style="bold",
            )
    _line(text, "Mount paths", _format_list(identity.mount_paths))
    if contract.is_replication:
        text.append(
            "\nDANGER: this is per-node replication, not model sharding.",
            style="bold white on #8f2730",
        )
    elif action is ReviewAction.PREPARE:
        text.append("\nShared means one copy and one engine.", style="bold #73be8b")
    else:
        text.append(
            "\nNo managed engine may be running. If process state changes, "
            "this confirmation becomes invalid.",
            style="bold #d7a84e",
        )
    text.append("\n\nThis review expires in 10 seconds.", style="#91a7a6")
    return text


class PresetScreen(ModalScreen[Optional[str]]):
    """One first-run intent question that only populates a draft."""

    BINDINGS = [
        Binding("escape", "cancel", "Advanced setup"),
        Binding("q", "cancel", "Advanced setup"),
        Binding("enter", "select_default", "Select default"),
        Binding("1", "select_index(0)", "Fastest GPU staging", show=False),
        Binding("2", "select_index(1)", "Single RAM copy", show=False),
        Binding("3", "select_index(2)", "Minimal RAM", show=False),
        Binding("4", "select_index(3)", "Multiple replicas", show=False),
    ]

    CSS = """
    PresetScreen {
        align: center middle;
        background: rgba(5, 9, 13, 0.78);
    }
    #preset-card {
        width: 76;
        max-width: 94%;
        height: auto;
        max-height: 94%;
        padding: 1 2;
        background: #18222d;
        border: tall #63bec2;
    }
    #preset-title {
        height: 2;
        color: #63bec2;
        text-style: bold;
    }
    #preset-help {
        height: auto;
        color: #91a7a6;
        margin-bottom: 1;
    }
    .preset-choice {
        width: 100%;
        height: 3;
        margin-bottom: 1;
    }
    #preset-cancel {
        width: 100%;
        height: 3;
    }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="preset-card"):
            yield Static(
                "WHAT SHOULD COLIBRI OPTIMIZE?",
                id="preset-title",
            )
            yield Static(
                "This only prepopulates a draft. You will review the exact "
                "plan before anything is mounted or copied.",
                id="preset-help",
            )
            for index, (preset_id, label, description) in enumerate(
                PRESET_CHOICES,
                1,
            ):
                yield Button(
                    "%d  %s%s\n%s"
                    % (
                        index,
                        label,
                        " · default" if index == 1 else "",
                        description,
                    ),
                    id="preset-%s" % preset_id,
                    classes="preset-choice",
                    variant="primary" if index == 1 else "default",
                )
            yield Button(
                "Continue with advanced setup",
                id="preset-cancel",
            )

    def on_mount(self) -> None:
        self.query_one(
            "#preset-%s" % PRESET_GPU_FASTEST,
            Button,
        ).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_select_default(self) -> None:
        self.dismiss(PRESET_GPU_FASTEST)

    def action_select_index(self, index: int) -> None:
        try:
            self.dismiss(PRESET_CHOICES[int(index)][0])
        except (IndexError, TypeError, ValueError):
            return

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "preset-cancel":
            self.dismiss(None)
        elif button_id.startswith("preset-"):
            self.dismiss(button_id.removeprefix("preset-"))


class ContractReviewScreen(ModalScreen[bool]):
    """Token-bound preparation or destruction review."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Cancel"),
    ]

    CSS = """
    ContractReviewScreen {
        align: center middle;
        background: rgba(5, 9, 13, 0.78);
    }
    #review-card {
        width: 76;
        max-width: 94%;
        height: 90%;
        padding: 1 2;
        background: #18222d;
        border: tall #d7a84e;
    }
    #review-scroll {
        height: 1fr;
        scrollbar-color: #405564;
        scrollbar-color-hover: #63bec2;
    }
    #review-facts {
        height: auto;
        min-height: 0;
    }
    #review-buttons {
        height: 3;
        align-horizontal: right;
    }
    #review-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        identity: ReviewIdentity,
        plan: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.identity = identity
        self.plan = plan
        self._expired = False

    def compose(self) -> ComposeResult:
        confirm_label = (
            "Prepare RAM workspace"
            if self.identity.action is ReviewAction.PREPARE
            else "Destroy volatile workspace"
        )
        confirm_variant = (
            "warning"
            if self.identity.action is ReviewAction.PREPARE
            else "error"
        )
        with Vertical(id="review-card"):
            with VerticalScroll(id="review-scroll"):
                yield Static(
                    review_facts(self.identity, self.plan),
                    id="review-facts",
                )
            with Horizontal(id="review-buttons"):
                yield Button("Cancel", id="cancel-review")
                yield Button(
                    confirm_label,
                    id="confirm-review",
                    variant=confirm_variant,
                )

    def on_mount(self) -> None:
        self.set_timer(10.0, self._expire)

    def _expire(self) -> None:
        self._expired = True
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-review":
            self.dismiss(True)
        elif event.button.id == "cancel-review":
            self.dismiss(False)


class RamdiskTextualApp(App[int]):
    """A placement-first, guided server setup and operations console."""

    TITLE = "Colibri RAM workspace"
    SUB_TITLE = "one placement contract, one lifecycle authority"

    CSS = """
    $surface: #0d1519;
    $panel: #162329;
    $panel-soft: #1b2c32;
    $ink: #d7e4e3;
    $muted: #91a7a6;
    $cyan: #63bec2;
    $amber: #d7a84e;
    $danger: #e2675a;
    $good: #73be8b;

    Screen {
        background: $surface;
        color: $ink;
        layers: base overlay;
    }
    #too-small {
        display: none;
        layer: overlay;
        width: 100%;
        height: 100%;
        padding: 1;
        background: $surface;
        color: $amber;
        text-style: bold;
    }
    #masthead {
        height: 3;
        padding: 0 1;
        background: #0c1218;
        border-bottom: solid #2c3a47;
    }
    #brand {
        height: 1;
        color: $cyan;
        text-style: bold;
    }
    #subtitle {
        height: 1;
        color: $muted;
    }
    #placement-rail {
        height: 5;
        padding: 1 2;
        background: $panel;
        border-bottom: heavy #304554;
    }
    #step-nav {
        height: 3;
        padding: 0 1;
        background: #111b24;
    }
    .step-tab {
        width: 1fr;
        min-width: 5;
        height: 3;
        padding: 0;
        border: none;
        background: #111b24;
        color: $muted;
    }
    .step-tab:hover {
        background: $panel-soft;
        color: $ink;
    }
    .step-tab.step-active {
        background: $panel-soft;
        color: $cyan;
        text-style: bold;
        border-bottom: heavy $cyan;
    }
    #step-scroll {
        height: 1fr;
        padding: 1 2;
        background: $surface;
        scrollbar-color: #405564;
        scrollbar-color-hover: $cyan;
    }
    #step-body {
        height: auto;
        min-height: 8;
    }
    .control-panel {
        display: none;
        height: auto;
        margin-top: 1;
        padding: 1;
        background: $panel;
        border-left: thick #304554;
    }
    .control-title {
        height: 1;
        color: $cyan;
        text-style: bold;
        margin-bottom: 1;
    }
    .field-row {
        height: 3;
    }
    .field-label {
        width: 24;
        height: 3;
        padding: 1 1 0 0;
        color: $muted;
    }
    Input {
        height: 3;
        background: #0c1218;
        border: tall #304554;
    }
    Input:focus {
        border: tall $cyan;
    }
    .choice-row {
        height: 3;
    }
    .choice-row Button {
        margin-right: 1;
    }
    #placement-note {
        height: auto;
        color: $muted;
        margin-top: 1;
    }
    #review-panel Button {
        width: 32;
    }
    #prepare-reason {
        height: auto;
        color: $muted;
        margin-top: 1;
    }
    #operation-grid {
        height: auto;
        grid-size: 2;
        grid-gutter: 1;
    }
    #operation-grid Button {
        width: 1fr;
    }
    #action-reasons {
        height: auto;
        margin-top: 1;
    }
    #operation-status {
        display: none;
        height: 2;
        padding: 0 1;
        background: #302718;
        color: $amber;
        text-style: bold;
    }
    #cancel-operation {
        display: none;
        width: 28;
        height: 3;
        margin: 0 1;
    }
    #message {
        height: 2;
        padding: 0 1;
        background: #0c1218;
        color: $muted;
    }
    #wizard-controls {
        height: 3;
        padding: 0 1;
        background: #111b24;
    }
    #wizard-controls Button {
        width: 14;
        height: 3;
    }
    #step-counter {
        width: 1fr;
        height: 3;
        content-align: center middle;
        color: $muted;
    }
    Footer {
        height: 1;
        background: #253440;
        color: $ink;
    }
    Screen.compact #masthead {
        height: 1;
    }
    Screen.compact #subtitle {
        display: none;
    }
    Screen.compact #placement-rail {
        height: 3;
        padding: 0 1;
    }
    Screen.compact #step-nav {
        height: 2;
        padding: 0;
    }
    Screen.compact .step-tab {
        height: 2;
    }
    Screen.compact #step-scroll {
        padding: 0 1;
    }
    Screen.compact #wizard-controls {
        height: 2;
        padding: 0;
    }
    Screen.compact #wizard-controls Button {
        height: 2;
    }
    Screen.too-small #too-small {
        display: block;
    }
    Screen.too-small #masthead,
    Screen.too-small #placement-rail,
    Screen.too-small #step-nav,
    Screen.too-small #step-scroll,
    Screen.too-small #operation-status,
    Screen.too-small #cancel-operation,
    Screen.too-small #message,
    Screen.too-small #wizard-controls,
    Screen.too-small Footer {
        display: none;
    }
    """

    BINDINGS = [
        Binding("q", "request_quit", "Quit"),
        Binding("escape", "request_quit", "Quit", show=False),
        Binding("ctrl+c", "interrupt", "Interrupt", show=False, priority=True),
        Binding("c", "cancel_operation", "Cancel operation"),
        Binding("r", "deep_refresh", "Deep refresh"),
        Binding("left", "previous_step", "Previous", show=False),
        Binding("right", "next_step", "Next", show=False),
        Binding("1", "show_step(0)", "Inspect", show=False),
        Binding("2", "show_step(1)", "Placement", show=False),
        Binding("3", "show_step(2)", "Capacity", show=False),
        Binding("4", "show_step(3)", "Runtime", show=False),
        Binding("5", "show_step(4)", "Review", show=False),
        Binding("6", "show_step(5)", "Operate", show=False),
    ]

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        cli_path: Optional[str] = None,
        engine_path: Optional[str] = None,
        lifecycle: Any = None,
        initial_snapshot: Optional[ConsoleSnapshot] = None,
        auto_refresh: bool = True,
        show_preset_prompt: Optional[bool] = None,
        privilege_authorizer: Optional[
            Callable[["RamdiskTextualApp"], Any]
        ] = None,
    ) -> None:
        super().__init__()
        self.lifecycle = lifecycle or importlib.import_module("ramdisk")
        self.args = _normalize_args(args)
        self.cli_path = cli_path
        self.engine_path = engine_path
        self.session = LifecycleSession(self.lifecycle, self.args)
        self.snapshot = initial_snapshot or ConsoleSnapshot.loading(self.args.base_port)
        self.refresh_enabled = bool(auto_refresh)
        self._preset_prompt_enabled = (
            initial_snapshot is None
            if show_preset_prompt is None
            else bool(show_preset_prompt)
        )
        self._preset_prompted = bool(
            getattr(self.args, "ramdisk_preset", None)
            or self.snapshot.report.get("present")
        )
        self._preset_screen_open = False
        self.privilege_authorizer = privilege_authorizer
        self.current_step = 0
        self.policy = ActionPolicy.from_state(
            self.snapshot.plan, self.snapshot.report
        )
        self.operation: Optional[dict[str, Any]] = None
        self._operation_lock = threading.Lock()
        self._refresh_inflight = False
        self._refresh_pending = False
        self._refresh_pending_deep = False
        self._refresh_pending_model = False
        self._dirty_inputs: set[str] = set()
        self._syncing_inputs = False
        self._base_port_override: Optional[int] = None
        self._plan_stale = False
        self._quit_when_idle = False
        self._quit_code = 0
        self._pending_review: Optional[ReviewIdentity] = None
        self._previous_signal_handlers: dict[int, Any] = {}
        self._message = (
            "Shared placement is the default: one model copy and one engine."
        )

    def compose(self) -> ComposeResult:
        yield Static(
            "Resize to at least 72 × 24. No lifecycle action was taken.",
            id="too-small",
        )
        with Vertical(id="masthead"):
            yield Static("COLIBRÍ  /  RAM WORKSPACE", id="brand")
            yield Static(
                "guided server staging · settings → exact contract → verified operation",
                id="subtitle",
            )
        yield Static(placement_backplane(self.snapshot), id="placement-rail")
        with Horizontal(id="step-nav"):
            for index, (step_id, label) in enumerate(STEPS, 1):
                yield Button(
                    f"{index} {label}",
                    id=f"step-{step_id}",
                    classes="step-tab",
                )
        with VerticalScroll(id="step-scroll"):
            yield Static(render_step(self.snapshot, STEPS[0][0]), id="step-body")
            with Vertical(id="placement-controls", classes="control-panel"):
                yield Static("PLACEMENT INPUTS", classes="control-title")
                with Horizontal(classes="field-row"):
                    yield Static("Memory NUMA nodes", classes="field-label")
                    yield Input(
                        value=str(self.args.memory_nodes or ""),
                        placeholder="effective allowed nodes",
                        id="memory-nodes",
                    )
                with Horizontal(classes="field-row"):
                    yield Static("Whole-core CPU list", classes="field-label")
                    yield Input(
                        value=str(self.args.cpu_list or ""),
                        placeholder="effective CPUs on selected nodes",
                        id="cpu-list",
                    )
                with Horizontal(classes="field-row"):
                    yield Static("Managed mount root", classes="field-label")
                    yield Input(value=str(self.args.mount_root), id="mount-root")
                with Horizontal(classes="choice-row"):
                    yield Button(
                        "Use shared placement",
                        id="use-shared",
                        variant="primary",
                    )
                yield Static(
                    "Submit raw Linux range lists; the planner validates effective "
                    "NUMA nodes and whole cores. DIMM/channel data is informational.",
                    id="placement-note",
                )
            with Vertical(id="capacity-controls", classes="control-panel"):
                yield Static("STAGING INPUTS", classes="control-title")
                with Horizontal(classes="choice-row"):
                    yield Button("Stage full model", id="mode-full")
                    yield Button("Use profile-guided set", id="mode-partial")
                with Horizontal(classes="field-row"):
                    yield Static("Per-copy budget GiB", classes="field-label")
                    yield Input(
                        value="" if self.args.capacity_gb is None else str(self.args.capacity_gb),
                        placeholder="required for profile-guided mode",
                        id="capacity-gb",
                    )
                with Horizontal(classes="field-row"):
                    yield Static("Usage profile path", classes="field-label")
                    yield Input(
                        value=str(self.args.profile or ""),
                        placeholder="model default",
                        id="profile-path",
                    )
            with Vertical(id="runtime-controls", classes="control-panel"):
                yield Static("RUNTIME INPUTS", classes="control-title")
                with Horizontal(classes="field-row"):
                    yield Static("Managed base port", classes="field-label")
                    yield Input(value=str(self.args.base_port), id="base-port")
                with Horizontal(classes="field-row"):
                    yield Static("Concurrent copy workers", classes="field-label")
                    yield Input(value=str(self.args.parallel), id="copy-workers")
                with Horizontal(classes="field-row"):
                    yield Static("Context length (0 = 4096)", classes="field-label")
                    yield Input(value=str(self.args.ctx), id="context-length")
                with Horizontal(classes="choice-row"):
                    yield Button("Toggle prefault", id="toggle-prefault")
                    yield Button("Cycle huge pages", id="cycle-thp")
                    yield Button("Toggle swappable tmpfs", id="toggle-swappable")
            with Vertical(id="review-panel", classes="control-panel"):
                yield Static("EXACT ACTION REVIEW", classes="control-title")
                yield Button(
                    "Prepare RAM workspace",
                    id="action-prepare",
                    variant="warning",
                )
                yield Static("", id="prepare-reason")
            with Vertical(id="operate-panel", classes="control-panel"):
                yield Static("VERIFIED LIFECYCLE ACTIONS", classes="control-title")
                with Grid(id="operation-grid"):
                    yield Button("Start managed engine", id="action-start", variant="success")
                    yield Button("Stop managed engine", id="action-stop", variant="warning")
                    yield Button("Run path benchmark", id="action-benchmark")
                    yield Button("Destroy volatile workspace", id="action-destroy", variant="error")
                yield Static("", id="action-reasons")
        yield Static("", id="operation-status")
        yield Button(
            "Cancel active operation",
            id="cancel-operation",
            variant="warning",
        )
        yield Static(self._message, id="message")
        with Horizontal(id="wizard-controls"):
            yield Button("← Back", id="previous-step")
            yield Static("STEP 1 OF 6", id="step-counter")
            yield Button("Next →", id="next-step", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        if threading.current_thread() is threading.main_thread():
            for name in ("SIGHUP", "SIGTERM"):
                signum = getattr(signal, name, None)
                if signum is None:
                    continue
                try:
                    self._previous_signal_handlers[signum] = signal.getsignal(
                        signum
                    )
                    signal.signal(signum, self._termination_signal)
                except (OSError, ValueError):
                    self._previous_signal_handlers.pop(signum, None)
        self._apply_snapshot(self.snapshot)
        self._show_step(0)
        self._maybe_prompt_preset()
        if self.refresh_enabled:
            self.set_interval(2.0, self._automatic_refresh)
            self._request_refresh(deep=True, model=True)

    def on_unmount(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum, previous in self._previous_signal_handlers.items():
            try:
                signal.signal(signum, previous)
            except (OSError, ValueError):
                pass
        self._previous_signal_handlers.clear()

    def _termination_signal(self, signum: int, _frame: Any) -> None:
        """Turn service-manager/SSH termination into cooperative rollback."""
        try:
            self.call_later(self._request_quit, 128 + int(signum))
        except RuntimeError:
            operation = self.operation
            if operation is not None and operation["cancelable"]:
                operation["cancel_event"].set()
            self._quit_when_idle = operation is not None
            self._quit_code = 128 + int(signum)

    def on_resize(self, event: events.Resize) -> None:
        too_small = event.size.width < 72 or event.size.height < 24
        if too_small and isinstance(self.screen, ContractReviewScreen):
            self.screen.dismiss(False)
        root_screen = self.screen_stack[0]
        compact = event.size.width < 90 or event.size.height < 30
        root_screen.set_class(compact, "compact")
        root_screen.set_class(too_small, "too-small")
        for index, (step_id, label) in enumerate(STEPS, 1):
            button = self.query_one(f"#step-{step_id}", Button)
            button.label = str(index) if event.size.width < 64 else f"{index} {label}"

    def _automatic_refresh(self) -> None:
        if self.operation is None and not self._refresh_inflight:
            self._request_refresh()

    def _request_refresh(self, *, deep: bool = False, model: bool = False) -> None:
        if not self.is_running:
            return
        if self._refresh_inflight:
            self._refresh_pending = True
            self._refresh_pending_deep = self._refresh_pending_deep or deep
            self._refresh_pending_model = self._refresh_pending_model or model
            return
        self._refresh_inflight = True

        def refresh_thread() -> None:
            try:
                snapshot = self.session.refresh(deep=deep, model=model)
            except BaseException as exc:
                try:
                    self.call_from_thread(self._finish_refresh, None, exc)
                except RuntimeError:
                    pass
            else:
                try:
                    self.call_from_thread(self._finish_refresh, snapshot, None)
                except RuntimeError:
                    pass

        # Refresh is read-only and may be abandoned when the interface closes.
        # Lifecycle mutations use the non-daemon, cleanup-aware path below.
        threading.Thread(
            target=refresh_thread,
            name="coli-ramdisk-refresh",
            daemon=True,
        ).start()

    def _finish_refresh(
        self,
        snapshot: Optional[ConsoleSnapshot],
        error: Optional[BaseException],
    ) -> None:
        self._refresh_inflight = False
        if self._refresh_pending:
            deep = self._refresh_pending_deep
            model = self._refresh_pending_model
            self._refresh_pending = False
            self._refresh_pending_deep = False
            self._refresh_pending_model = False
            self._request_refresh(deep=deep, model=model)
            return
        if error is not None:
            self._apply_snapshot(replace(self.snapshot, error=str(error)))
            self._set_message(
                f"Refresh failed: {error}. Press R to retry; no lifecycle action was taken."
            )
        elif snapshot is not None:
            self._plan_stale = False
            self._apply_snapshot(snapshot)

    def _apply_snapshot(self, snapshot: ConsoleSnapshot) -> None:
        state = snapshot.report.get("state")
        if self._base_port_override is not None:
            persisted_port = None
            if snapshot.manifest is not None:
                try:
                    persisted_port = self.lifecycle._persisted_base_port(
                        snapshot.manifest
                    )
                except (AttributeError, KeyError, TypeError, ValueError):
                    persisted_port = snapshot.manifest.get("base_port")
            if state in ("running", "starting"):
                self._base_port_override = None
            elif persisted_port == self._base_port_override:
                self._base_port_override = None
            else:
                snapshot = replace(
                    snapshot,
                    base_port=self._base_port_override,
                )
        self.snapshot = snapshot
        self.args.base_port = snapshot.base_port
        self.query_one("#placement-rail", Static).update(
            placement_backplane(snapshot)
        )
        self.query_one("#step-body", Static).update(
            render_step(snapshot, STEPS[self.current_step][0])
        )
        self._sync_controls()
        self._update_action_policy()
        self._maybe_prompt_preset()

    def _maybe_prompt_preset(self) -> None:
        if (
            not self._preset_prompt_enabled
            or self._preset_prompted
            or self._preset_screen_open
            or not self.is_mounted
            or self.snapshot.plan is None
            or self.snapshot.report.get("state") == "loading"
        ):
            return
        if self.snapshot.report.get("present"):
            self._preset_prompted = True
            return
        self._preset_prompted = True
        self._preset_screen_open = True
        self.push_screen(PresetScreen(), self._preset_selected)

    def _preset_selected(self, preset_id: Optional[str]) -> None:
        self._preset_screen_open = False
        if preset_id is None:
            self._set_message(
                "Advanced setup retained; no preset values were applied."
            )
            return
        try:
            result = self.session.resolve_preset(
                preset_id,
                cli_path=self.cli_path,
                engine_path=self.engine_path,
            )
        except (OSError, ValueError, self.lifecycle.RamdiskError) as exc:
            self._set_message(
                "Preset could not be resolved: %s. No lifecycle action was taken."
                % exc
            )
            return
        self._plan_stale = False
        self._apply_snapshot(
            replace(
                self.snapshot,
                plan=result["plan"],
                error="",
            )
        )
        self._show_step(4)
        preset = result["plan"].get("preset") or {}
        self._set_message(
            "%s populated the draft. Review the exact plan; advanced "
            "settings remain editable."
            % preset.get("label", "Preset")
        )

    def _sync_input(self, selector: str, value: Any) -> None:
        widget = self.query_one(selector, Input)
        if widget.id in self._dirty_inputs or widget.has_focus:
            return
        self._syncing_inputs = True
        try:
            widget.value = "" if value is None else str(value)
        finally:
            self._syncing_inputs = False

    def _sync_controls(self) -> None:
        self._sync_input("#memory-nodes", self.args.memory_nodes)
        self._sync_input("#cpu-list", self.args.cpu_list)
        self._sync_input("#mount-root", self.args.mount_root)
        self._sync_input("#capacity-gb", self.args.capacity_gb)
        self._sync_input("#profile-path", self.args.profile)
        self._sync_input("#base-port", self.args.base_port)
        self._sync_input("#copy-workers", self.args.parallel)
        self._sync_input("#context-length", self.args.ctx)

        policy = ActionPolicy.from_state(self.snapshot.plan, self.snapshot.report)
        operation_locked = self.operation is not None
        weight_locked = operation_locked or not policy.edit_weights.enabled
        for selector in (
            "#memory-nodes",
            "#cpu-list",
            "#mount-root",
            "#capacity-gb",
            "#profile-path",
            "#copy-workers",
            "#context-length",
        ):
            self.query_one(selector).disabled = weight_locked
        for selector in (
            "#mode-full",
            "#mode-partial",
            "#toggle-prefault",
            "#cycle-thp",
            "#toggle-swappable",
        ):
            self.query_one(selector, Button).disabled = weight_locked
        self.query_one("#base-port", Input).disabled = (
            operation_locked
            or self.snapshot.plan is None
            or not policy.edit_base_port.enabled
        )
        use_shared = self.query_one("#use-shared", Button)
        use_shared.display = self.args.topology == "per-node"
        use_shared.disabled = weight_locked

    def _update_action_policy(self) -> None:
        self.policy = ActionPolicy.from_state(
            self.snapshot.plan, self.snapshot.report
        )
        mapping = {
            "prepare": ("#action-prepare", self.policy.prepare),
            "start": ("#action-start", self.policy.start),
            "stop": ("#action-stop", self.policy.stop),
            "benchmark": ("#action-benchmark", self.policy.benchmark),
            "destroy": ("#action-destroy", self.policy.destroy),
        }
        for _, (selector, permission) in mapping.items():
            self.query_one(selector, Button).disabled = (
                self.operation is not None
                or bool(self.snapshot.error)
                or self._plan_stale
                or not permission.enabled
            )

        prepare_reason = (
            "Ready. Confirmation will repeat every copy, engine, port and mount."
            if self.policy.prepare.enabled
            else self.policy.prepare.reason
        )
        if self.snapshot.error:
            prepare_reason = "Resolve the refresh error before taking an action."
        elif self._plan_stale:
            prepare_reason = (
                "Rebuilding the authoritative plan after your settings change."
            )
        self.query_one("#prepare-reason", Static).update(prepare_reason)

        reasons = Text()
        for name, permission in (
            ("Prepare", self.policy.prepare),
            ("Start", self.policy.start),
            ("Stop", self.policy.stop),
            ("Benchmark", self.policy.benchmark),
            ("Destroy", self.policy.destroy),
        ):
            line, style = _permission_line(name, permission)
            reasons.append(line, style=style)
            reasons.append("\n")
        if self.snapshot.error:
            reasons.append(
                "Actions are disabled until refresh succeeds.", style="bold #e2675a"
            )
        self.query_one("#action-reasons", Static).update(reasons)

    def _show_step(self, index: int) -> None:
        self.current_step = max(0, min(len(STEPS) - 1, int(index)))
        active_id = STEPS[self.current_step][0]
        for _, (step_id, _) in enumerate(STEPS):
            self.query_one(f"#step-{step_id}", Button).set_class(
                step_id == active_id, "step-active"
            )
        panel_for_step = {
            "placement": "#placement-controls",
            "capacity": "#capacity-controls",
            "runtime": "#runtime-controls",
            "review": "#review-panel",
            "operate": "#operate-panel",
        }
        for step_id, selector in panel_for_step.items():
            self.query_one(selector).display = step_id == active_id
        self.query_one("#step-body", Static).update(
            render_step(self.snapshot, active_id)
        )
        self.query_one("#step-counter", Static).update(
            f"STEP {self.current_step + 1} OF {len(STEPS)} · {STEPS[self.current_step][1].upper()}"
        )
        self.query_one("#previous-step", Button).disabled = self.current_step == 0
        self.query_one("#next-step", Button).disabled = (
            self.current_step == len(STEPS) - 1
        )
        self.query_one("#step-scroll", VerticalScroll).scroll_home(animate=False)

    def _set_message(self, message: str) -> None:
        self._message = str(message)
        if self.is_mounted:
            self.query_one("#message", Static).update(self._message)

    def action_show_step(self, index: int) -> None:
        if self._commit_dirty_inputs():
            self._show_step(index)

    def action_previous_step(self) -> None:
        if self._commit_dirty_inputs():
            self._show_step(self.current_step - 1)

    def action_next_step(self) -> None:
        if self._commit_dirty_inputs():
            self._show_step(self.current_step + 1)

    def action_deep_refresh(self) -> None:
        if not self._commit_dirty_inputs():
            return
        self.session.invalidate(deep=True, model=True)
        self._set_message("Refreshing hardware, model plan and lifecycle validation…")
        self._request_refresh(deep=True, model=True)

    def action_request_quit(self) -> None:
        self._request_quit(0)

    def action_interrupt(self) -> None:
        self._request_quit(130)

    def _request_quit(self, code: int) -> None:
        if self.operation is None:
            self.exit(code)
            return
        self._quit_when_idle = True
        self._quit_code = int(code)
        if self.operation["cancelable"]:
            self.operation["cancel_event"].set()
            self._set_message(
                "Cancelling safely; Colibri will quit after rollback and cleanup finish."
            )
        else:
            self._set_message(
                "Cleanup cannot be interrupted safely; Colibri will quit when it finishes."
            )

    def action_cancel_operation(self) -> None:
        if self.operation is None:
            self._set_message("No cancellable lifecycle operation is running.")
        elif not self.operation["cancelable"]:
            self._set_message("This cleanup step cannot be interrupted safely.")
        else:
            self.operation["cancel_event"].set()
            cancel_button = self.query_one("#cancel-operation", Button)
            cancel_button.disabled = True
            self._set_message(
                "Cancellation requested; waiting for rollback and cleanup checkpoints."
            )

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "cancel-operation":
            self.action_cancel_operation()
            return
        if not self._commit_dirty_inputs():
            return
        if button_id.startswith("step-"):
            step_id = button_id.removeprefix("step-")
            self._show_step(next(index for index, item in enumerate(STEPS) if item[0] == step_id))
        elif button_id == "previous-step":
            self.action_previous_step()
        elif button_id == "next-step":
            self.action_next_step()
        elif button_id == "use-shared":
            if self._can_edit_weights():
                self.args.topology = "interleaved"
                self._settings_changed(
                    "Placement changed to one shared model copy and one engine."
                )
        elif button_id == "mode-full":
            if self._can_edit_weights():
                self.args.mode = "full"
                self.args.capacity_gb = None
                self._settings_changed("Staging changed to the full model.")
        elif button_id == "mode-partial":
            if self._can_edit_weights():
                self.args.mode = "partial"
                if self.args.capacity_gb is None:
                    self.args.capacity_gb = 16.0
                self._settings_changed(
                    "Staging changed to a profile-guided set; verify its capacity."
                )
        elif button_id == "toggle-prefault":
            if self._can_edit_weights():
                effective = (
                    self.args.prefault
                    if self.args.prefault is not None
                    else self.args.mode == "full"
                )
                self.args.prefault = 0 if effective else 1
                self._settings_changed(
                    f"Prefault {'enabled' if self.args.prefault else 'disabled'}."
                )
        elif button_id == "cycle-thp":
            if self._can_edit_weights():
                choices = ("auto", "within_size", "advise")
                self.args.thp = choices[
                    (choices.index(self.args.thp) + 1) % len(choices)
                ]
                self._settings_changed(
                    f"Huge-page policy changed to {self.args.thp}."
                )
        elif button_id == "toggle-swappable":
            if self._can_edit_weights():
                self.args.allow_swappable = not self.args.allow_swappable
                self._settings_changed(
                    "Swappable tmpfs "
                    + ("allowed." if self.args.allow_swappable else "refused.")
                )
        elif button_id == "action-prepare":
            self._open_prepare_review()
        elif button_id == "action-start":
            self._start_engine()
        elif button_id == "action-stop":
            self._stop_engine()
        elif button_id == "action-benchmark":
            self._benchmark()
        elif button_id == "action-destroy":
            self._open_destroy_review()

    @on(Input.Changed)
    def _input_changed(self, event: Input.Changed) -> None:
        field = event.input.id
        if not field or self._syncing_inputs:
            return
        attributes = {
            "memory-nodes": "memory_nodes",
            "cpu-list": "cpu_list",
            "mount-root": "mount_root",
            "capacity-gb": "capacity_gb",
            "profile-path": "profile",
            "base-port": "base_port",
            "copy-workers": "parallel",
            "context-length": "ctx",
        }
        attribute = attributes.get(field)
        if attribute is None:
            return
        current = getattr(self.args, attribute)
        expected = "" if current is None else str(current)
        if event.value != expected:
            self._dirty_inputs.add(field)
        else:
            self._dirty_inputs.discard(field)

    def _parse_input_value(
        self,
        field: str,
        raw_value: str,
    ) -> tuple[str, Any, str, bool]:
        value = raw_value.strip()
        if field == "memory-nodes":
            return (
                "memory_nodes",
                value or None,
                "memory NUMA node selection",
                False,
            )
        if field == "cpu-list":
            return ("cpu_list", value or None, "whole-core CPU selection", False)
        if field == "mount-root":
            if not value:
                raise ValueError("mount root cannot be empty")
            return ("mount_root", value, "mount root", False)
        if field == "capacity-gb":
            if not value and self.args.mode == "full":
                return ("capacity_gb", None, "per-copy capacity", False)
            capacity = float(value)
            if not math.isfinite(capacity) or capacity <= 0:
                raise ValueError("capacity must be a finite positive number")
            return ("capacity_gb", capacity, "per-copy capacity", False)
        if field == "profile-path":
            return ("profile", value or None, "usage profile", False)
        if field == "base-port":
            port = int(value)
            if self.snapshot.plan is None:
                raise ValueError(
                    "wait for the authoritative plan before selecting engine ports"
                )
            ports = self.lifecycle._managed_ports_for_plan(
                self.snapshot.plan, port
            )
            if (
                not 1 <= port <= 65535
                or len(set(ports)) != len(ports)
                or any(item < 1 or item > 65535 for item in ports)
            ):
                raise ValueError(
                    "base port produces invalid or duplicate engine ports"
                )
            return ("base_port", port, "managed base port", True)
        if field == "copy-workers":
            workers = int(value)
            if not 1 <= workers <= 64:
                raise ValueError("copy workers must be between 1 and 64")
            return ("parallel", workers, "concurrent copy workers", False)
        if field == "context-length":
            ctx = int(value)
            if ctx < 0:
                raise ValueError("context length cannot be negative")
            return ("ctx", ctx, "context length", False)
        raise ValueError(f"unknown setting {field!r}")

    def _commit_dirty_inputs(
        self,
        fields: Optional[set[str]] = None,
    ) -> bool:
        order = (
            "memory-nodes",
            "cpu-list",
            "mount-root",
            "capacity-gb",
            "profile-path",
            "base-port",
            "copy-workers",
            "context-length",
        )
        wanted = [
            field
            for field in order
            if field in self._dirty_inputs
            and (fields is None or field in fields)
        ]
        if not wanted:
            return True
        if self.operation is not None:
            self._set_message(
                "Settings stay locked until the active lifecycle operation finishes."
            )
            return False
        includes_base_port = "base-port" in wanted
        includes_weights = any(field != "base-port" for field in wanted)
        if includes_base_port and not self.policy.edit_base_port.enabled:
            self._set_message(self.policy.edit_base_port.reason)
            return False
        if includes_weights and not self.policy.edit_weights.enabled:
            self._set_message(self.policy.edit_weights.reason)
            return False

        parsed: list[tuple[str, Any, str, bool]] = []
        try:
            for field in wanted:
                widget = self.query_one(f"#{field}", Input)
                parsed.append(self._parse_input_value(field, widget.value))
        except (TypeError, ValueError) as exc:
            self._set_message(
                f"Invalid setting: {exc}. Fix the highlighted step before continuing."
            )
            return False

        for attribute, value, _, is_base_port in parsed:
            setattr(self.args, attribute, value)
            if is_base_port:
                self._base_port_override = int(value)
                self.snapshot = replace(self.snapshot, base_port=int(value))
        for field in wanted:
            self._dirty_inputs.discard(field)

        labels = ", ".join(item[2] for item in parsed)
        if includes_weights:
            message = f"Updated {labels}; rebuilding the authoritative plan."
            self._settings_changed(message)
        else:
            self._settings_changed(
                f"Updated {labels}; the exact endpoint contract now reflects it.",
                rebuild=False,
            )
        return True

    @on(Input.Submitted)
    def _input_submitted(self, event: Input.Submitted) -> None:
        field = event.input.id
        if field is None:
            return
        self._dirty_inputs.add(field)
        self._commit_dirty_inputs({field})

    def _can_edit_weights(self) -> bool:
        if self.operation is not None:
            self._set_message(
                "Settings stay locked until the active lifecycle operation finishes."
            )
            return False
        if self.policy.edit_weights.enabled:
            return True
        self._set_message(self.policy.edit_weights.reason)
        return False

    def _settings_changed(self, message: str, *, rebuild: bool = True) -> None:
        if rebuild:
            marker = getattr(self.lifecycle, "mark_preset_custom", None)
            if marker is not None:
                marker(self.args)
            self.session.invalidate()
            self._plan_stale = True
        self._pending_review = None
        self._apply_snapshot(self.snapshot)
        self._sync_controls()
        self._set_message(message)
        if rebuild:
            self._request_refresh()

    def _authorize_privileged_mounts(self) -> bool:
        if self.privilege_authorizer is not None:
            result = self.privilege_authorizer(self)
            if isinstance(result, tuple):
                allowed, message = result
                if message:
                    self._set_message(str(message))
                return bool(allowed)
            return bool(result)
        if os.geteuid() == 0:
            return True
        try:
            sudo = self.lifecycle._trusted_system_binary("sudo")
        except (OSError, self.lifecycle.RamdiskError) as exc:
            self._set_message(f"Cannot authorize mount operations: {exc}")
            return False
        try:
            with self.suspend():
                result = subprocess.run([sudo, "-v"], check=False)
                reusable = (
                    self.lifecycle._validate_noninteractive_sudo(sudo)
                    if result.returncode == 0
                    else None
                )
        except OSError as exc:
            self._set_message(f"Sudo authorization failed: {exc}")
            return False
        if result.returncode:
            self._set_message(
                "Sudo authorization was cancelled; no mount operation started."
            )
            return False
        if reusable is None or reusable.returncode:
            self._set_message(
                "Sudo policy cannot reuse authorization without prompting; "
                "no mount operation started."
            )
            return False
        return True

    def _open_prepare_review(self) -> None:
        if not self._commit_dirty_inputs():
            return
        permission = self.policy.prepare
        if not permission.enabled:
            self._set_message(permission.reason)
            return
        plan = self.snapshot.plan
        if plan is None:
            self._set_message("The authoritative plan is not available yet.")
            return
        token = self.lifecycle._plan_confirmation_token(plan)
        review = ReviewIdentity.for_prepare(token, plan, self.snapshot.base_port)
        self._pending_review = review
        self.push_screen(
            ContractReviewScreen(review, plan),
            callback=self._prepare_review_result,
        )

    def _prepare_review_result(self, confirmed: bool) -> None:
        review = self._pending_review
        self._pending_review = None
        if not confirmed or review is None:
            self._set_message("Preparation review cancelled or expired; nothing changed.")
            return
        plan = self.snapshot.plan
        if plan is None:
            self._set_message("The plan disappeared; inspect and review again.")
            return
        current = ReviewIdentity.for_prepare(
            self.lifecycle._plan_confirmation_token(plan),
            plan,
            self.snapshot.base_port,
        )
        if current != review:
            self._set_message("The plan changed after review; inspect and confirm it again.")
            self._request_refresh(deep=True)
            return
        if not self._authorize_privileged_mounts():
            return
        prepared_args = argparse.Namespace(**dict(vars(self.args), yes=True))
        expected_copies = (
            len(plan.get("staging", {}).get("selected_shards", ()))
            * review.placement.copy_count
        )
        copied = [0]
        copied_lock = threading.Lock()

        def run_prepare(operation: dict[str, Any]) -> Any:
            def progress(name: str, size: int, elapsed: float) -> None:
                with copied_lock:
                    copied[0] += 1
                    rate = size / elapsed / MIB if elapsed > 0 else 0.0
                    self._update_operation_from_thread(
                        operation,
                        f"Copy {copied[0]}/{expected_copies} · {name} · {rate:.1f} MiB/s",
                    )

            return self.lifecycle.prepare(
                prepared_args,
                progress=progress,
                display_plan=False,
                expected_plan_token=review.token,
                cancel_event=operation["cancel_event"],
            )

        self._begin_operation(
            "prepare",
            "Preparing RAM workspace",
            run_prepare,
            cancelable=True,
        )

    def _start_engine(self) -> None:
        if not self._commit_dirty_inputs():
            return
        permission = self.policy.start
        if not permission.enabled:
            self._set_message(permission.reason)
            return
        start_args = argparse.Namespace(base_port=self.snapshot.base_port)
        self._begin_operation(
            "start",
            "Loading managed engine",
            lambda operation: self.lifecycle.start(
                start_args,
                cli_path=self.cli_path,
                engine_path=self.engine_path,
                cancel_event=operation["cancel_event"],
            ),
            cancelable=True,
        )

    def _stop_engine(self) -> None:
        if not self._commit_dirty_inputs():
            return
        permission = self.policy.stop
        if not permission.enabled:
            self._set_message(permission.reason)
            return
        self._begin_operation(
            "stop",
            "Stopping verified managed engines",
            lambda operation: self.lifecycle.stop(argparse.Namespace()),
            cancelable=False,
        )

    def _benchmark(self) -> None:
        if not self._commit_dirty_inputs():
            return
        permission = self.policy.benchmark
        if not permission.enabled:
            self._set_message(permission.reason)
            return
        self._begin_operation(
            "benchmark",
            "Running deterministic path benchmark",
            lambda operation: self.lifecycle.benchmark(
                argparse.Namespace(),
                cli_path=self.cli_path,
                engine_path=self.engine_path,
                cancel_event=operation["cancel_event"],
            ),
            cancelable=True,
        )

    def _open_destroy_review(self) -> None:
        if not self._commit_dirty_inputs():
            return
        permission = self.policy.destroy
        if not permission.enabled:
            self._set_message(permission.reason)
            return
        manifest = self.snapshot.manifest
        if manifest is None:
            try:
                manifest = self.lifecycle._load_manifest(required=True)
            except (OSError, self.lifecycle.RamdiskError) as exc:
                self._set_message(f"Destroy review failed: {exc}")
                self._request_refresh(deep=True)
                return
        token = self.lifecycle._manifest_confirmation_token(manifest)
        review = ReviewIdentity.for_destroy(
            token,
            manifest["plan"],
            manifest.get("mounts", ()),
            self.lifecycle._persisted_base_port(manifest),
        )
        self._pending_review = review
        self.push_screen(
            ContractReviewScreen(review, manifest["plan"]),
            callback=self._destroy_review_result,
        )

    def _destroy_review_result(self, confirmed: bool) -> None:
        review = self._pending_review
        self._pending_review = None
        if not confirmed or review is None:
            self._set_message("Destroy review cancelled or expired; nothing changed.")
            return
        try:
            manifest = self.lifecycle._load_manifest(required=True)
        except (OSError, self.lifecycle.RamdiskError) as exc:
            self._set_message(f"Destroy review is stale: {exc}")
            self._request_refresh(deep=True)
            return
        current = ReviewIdentity.for_destroy(
            self.lifecycle._manifest_confirmation_token(manifest),
            manifest["plan"],
            manifest.get("mounts", ()),
            self.lifecycle._persisted_base_port(manifest),
        )
        if current != review:
            self._set_message(
                "The active deployment changed after review; inspect and confirm again."
            )
            self._request_refresh(deep=True)
            return
        if not self._authorize_privileged_mounts():
            return
        self._begin_operation(
            "destroy",
            "Destroying volatile RAM workspace",
            lambda operation: self.lifecycle.destroy(
                argparse.Namespace(yes=True),
                expected_manifest_token=review.token,
            ),
            cancelable=False,
        )

    def _set_shared_worker(self, operation: Optional[dict[str, Any]]) -> None:
        guard = getattr(self.lifecycle, "_tui_worker_guard", None)
        if guard is None:
            setattr(self.lifecycle, "_tui_worker", operation)
        else:
            with guard:
                setattr(self.lifecycle, "_tui_worker", operation)

    def _begin_operation(
        self,
        action: str,
        label: str,
        target: Callable[[dict[str, Any]], Any],
        *,
        cancelable: bool,
    ) -> None:
        if self.operation is not None:
            self._set_message("A lifecycle operation is already running.")
            return
        operation: dict[str, Any] = {
            "action": action,
            "label": label,
            "detail": "Starting…",
            "started": time.monotonic(),
            "cancelable": bool(cancelable),
            "cancel_event": threading.Event(),
            "done": False,
            "result": None,
            "error": None,
        }

        def runner() -> None:
            try:
                privilege = getattr(
                    self.lifecycle,
                    "_noninteractive_privilege",
                    None,
                )
                privilege_context = (
                    privilege(
                        keepalive=action in ("prepare", "destroy"),
                        cancel_event=(
                            operation["cancel_event"] if cancelable else None
                        ),
                    )
                    if privilege is not None
                    else contextlib.nullcontext()
                )
                with privilege_context:
                    result = target(operation)
                with self._operation_lock:
                    operation["result"] = result
            except BaseException as exc:
                with self._operation_lock:
                    operation["error"] = exc
            finally:
                with self._operation_lock:
                    operation["done"] = True
                try:
                    self.call_from_thread(self._finish_operation, operation)
                except RuntimeError:
                    pass

        thread = threading.Thread(
            target=runner,
            name=f"coli-ramdisk-{action}",
        )
        operation["thread"] = thread
        self.operation = operation
        self._set_shared_worker(operation)
        self._sync_controls()
        self._update_action_policy()
        status = self.query_one("#operation-status", Static)
        status.display = True
        status.update(f"{label} · Starting…")
        cancel_button = self.query_one("#cancel-operation", Button)
        cancel_button.display = bool(cancelable)
        cancel_button.disabled = False
        self._set_message(
            f"{label} started; navigation remains available"
            + (
                " and the visible Cancel control requests safe rollback."
                if cancelable
                else "."
            )
        )
        try:
            thread.start()
        except BaseException:
            self.operation = None
            self._set_shared_worker(None)
            status.display = False
            cancel_button.display = False
            self._sync_controls()
            self._update_action_policy()
            raise

    def _update_operation_from_thread(
        self, operation: dict[str, Any], detail: str
    ) -> None:
        with self._operation_lock:
            operation["detail"] = detail
        try:
            self.call_from_thread(self._show_operation_progress, operation, detail)
        except RuntimeError:
            pass

    def _show_operation_progress(
        self, operation: dict[str, Any], detail: str
    ) -> None:
        if self.operation is operation:
            elapsed = time.monotonic() - operation["started"]
            self.query_one("#operation-status", Static).update(
                f"{operation['label']} · {detail} · {elapsed:.1f}s"
            )

    def _finish_operation(self, operation: dict[str, Any]) -> None:
        if self.operation is not operation:
            return
        operation["thread"].join(timeout=0)
        with self._operation_lock:
            error = operation["error"]
            result = operation["result"]
        action = operation["action"]
        cancel_requested = operation["cancel_event"].is_set()
        cancelled_type = getattr(self.lifecycle, "_OperationCancelled", ())
        cancelled_cleanly = bool(cancelled_type) and isinstance(error, cancelled_type)
        if error is not None:
            if cancelled_cleanly:
                message = f"{action.capitalize()} cancelled safely: {error}"
            elif cancel_requested:
                message = (
                    f"{action.capitalize()} cleanup failed after cancellation: {error}. "
                    "Review Operate before quitting."
                )
                self._quit_when_idle = False
            else:
                message = f"{action.capitalize()} failed: {error}"
        elif action == "prepare":
            message = "RAM workspace is ready. Continue to Operate to start the engine."
        elif action == "start":
            ports = result.get("ports", ()) if isinstance(result, Mapping) else ()
            message = f"Managed engine ready on {_format_list(ports, empty='its prepared port')}."
        elif action == "stop":
            if isinstance(result, Mapping) and result.get("state") == "error":
                message = (
                    "Engine cleanup finished, but the workspace is incomplete. "
                    "Refresh Operate, then Destroy."
                )
            else:
                message = "Managed engines stopped; RAM weights remain prepared."
        elif action == "destroy":
            message = "RAM workspace removed; durable KV and benchmark state were preserved."
        else:
            best = result.get("best_variant") if isinstance(result, Mapping) else None
            message = (
                f"Benchmark complete; best path: {best}."
                if best
                else "Benchmark complete; scorecard saved."
            )
        self.operation = None
        self._set_shared_worker(None)
        self.query_one("#operation-status", Static).display = False
        self.query_one("#cancel-operation", Button).display = False
        self.session.invalidate()
        self._set_message(message)
        self._sync_controls()
        self._update_action_policy()
        if self._quit_when_idle and (error is None or cancelled_cleanly):
            self.exit(self._quit_code)
        else:
            self._request_refresh(deep=action in ("prepare", "destroy"))


def launch_tui(
    args: argparse.Namespace,
    *,
    cli_path: Optional[str] = None,
    engine_path: Optional[str] = None,
    lifecycle: Any = None,
) -> int:
    """Run the Textual frontend and return a CLI-compatible status code."""

    app = RamdiskTextualApp(
        args,
        cli_path=cli_path,
        engine_path=engine_path,
        lifecycle=lifecycle,
    )
    result = app.run()
    return 0 if result is None else int(result)
