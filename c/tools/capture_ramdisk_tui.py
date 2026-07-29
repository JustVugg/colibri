#!/usr/bin/env python3
"""Generate deterministic SVG screenshots for the RAM-workspace TUI how-to.

The capture uses the production Textual application with illustrative data.
It deliberately has no access to the real lifecycle implementation: automatic
refresh is disabled, privileged authorization always fails, and every
lifecycle read or mutation outside the pure confirmation-token helpers raises.

Run from the repository root:

    python3 c/tools/capture_ramdisk_tui.py
"""

from __future__ import annotations

import argparse
import asyncio
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


C_DIR = Path(__file__).resolve().parents[1]
ROOT = C_DIR.parent
ASSET_DIR = ROOT / "docs" / "media" / "ramdisk-tui"
VIEWPORT = (110, 34)
GIB = 1 << 30
MODEL_PATH = "/srv/colibri/models/GLM-5.2-colibri-int4"
MOUNT_ROOT = "/mnt/colibri-ram"
SHARED_MOUNT = f"{MOUNT_ROOT}/shared"
EXPECTED_ASSETS = (
    "01-inspect.svg",
    "02-review.svg",
    "03-prepare-confirmation.svg",
    "04-ready.svg",
    "05-running.svg",
    "06-stopped.svg",
    "07-absent.svg",
)


# Import the installed/source-tree frontend only after computing paths. The
# script resides below c/, so direct execution already puts c/tools on
# sys.path, not its parent.
import sys

sys.path.insert(0, str(C_DIR))

import ramdisk_textual  # noqa: E402
from textual.screen import Screen  # noqa: E402
from textual.widgets import Button, Static  # noqa: E402


class CaptureSafetyError(RuntimeError):
    """The documentation renderer attempted an unsafe or incoherent action."""


class RefusingLifecycle:
    """Lifecycle double that permits pure identity tokens and nothing else."""

    class RamdiskError(RuntimeError):
        """Compatibility exception used by the production frontend."""

    MUTATIONS = ("prepare", "start", "stop", "benchmark", "destroy")
    HOST_READS = (
        "discover_hardware",
        "status",
        "scan_model",
        "build_plan",
        "_load_manifest",
    )

    def __init__(self) -> None:
        self.attempts: list[str] = []
        self.plan_token_reads = 0

    def _refuse(self, operation: str) -> Any:
        self.attempts.append(operation)
        raise CaptureSafetyError(
            f"documentation capture refused lifecycle call: {operation}"
        )

    def _plan_confirmation_token(self, _plan: Mapping[str, Any]) -> str:
        self.plan_token_reads += 1
        return "docs-plan-token-shared-full-v1"

    def _manifest_confirmation_token(
        self, _manifest: Mapping[str, Any]
    ) -> str:
        return "docs-manifest-token-shared-full-v1"

    def _persisted_base_port(self, manifest: Mapping[str, Any]) -> int:
        return int(manifest.get("base_port", 8000))

    def discover_hardware(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._refuse("discover_hardware")

    def status(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._refuse("status")

    def scan_model(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._refuse("scan_model")

    def build_plan(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._refuse("build_plan")

    def _load_manifest(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._refuse("_load_manifest")

    def prepare(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._refuse("prepare")

    def start(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._refuse("start")

    def stop(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._refuse("stop")

    def benchmark(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._refuse("benchmark")

    def destroy(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._refuse("destroy")


class NonAuthorizingPrivilege:
    """Privilege callback that records and refuses every authorization."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _app: ramdisk_textual.RamdiskTextualApp) -> bool:
        self.calls += 1
        return False


@dataclass(frozen=True)
class CaptureBundle:
    """Rendered assets plus the safety audit for the capture session."""

    screenshots: Mapping[str, str]
    lifecycle_attempts: tuple[str, ...]
    plan_token_reads: int
    privilege_calls: int


def documentation_plan() -> dict[str, Any]:
    """Return one blocker-free illustrative shared/full placement plan."""

    staged_bytes = int(357.4 * GIB)
    runtime_reserve = 24 * GIB
    page_table_reserve = 2 * GIB
    os_reserve = 64 * GIB
    return {
        "schema": "colibri.ramdisk.plan.v1",
        "version": 1,
        "topology": "interleaved",
        "mode": "full",
        "mount_root": MOUNT_ROOT,
        "capacity_bytes": 360 * GIB,
        "prefault": True,
        "parallel": 4,
        "hardware": {"online_nodes": [0, 1]},
        "placement": {
            "memory_nodes": [0, 1],
            "memory_node_list": "0-1",
            "cpus": list(range(128)),
            "cpu_list": "0-127",
            "engine_cpu_sets": [
                {
                    "node": None,
                    "cpus": list(range(128)),
                    "cpu_list": "0-127",
                    "physical_cores": 64,
                }
            ],
            "dimm_control": "informational-only",
        },
        "model": {
            "path": MODEL_PATH,
            "name": "GLM-5.2 Colibri INT4",
            "fingerprint": "sha256:docs-example-7f3a9d21",
            "shard_count": 144,
            "dense_tensor_bytes": 18 * GIB,
        },
        "staging": {
            "selected_shards": [
                f"model-{index:05d}-of-00144.safetensors"
                for index in range(1, 145)
            ],
            "linked_shards": [],
            "direct_mapped_expert_count": 256,
            "replica_count": 1,
            "staged_bytes": staged_bytes,
            "total_staged_bytes": staged_bytes,
        },
        "reserve": {
            "runtime_reserve_bytes": runtime_reserve,
            "page_table_bytes": page_table_reserve,
            "os_margin_bytes": os_reserve,
            "total_os_margin_bytes": os_reserve,
            "total_required_bytes": (
                staged_bytes
                + runtime_reserve
                + page_table_reserve
                + os_reserve
            ),
            "available_bytes": 744 * GIB,
        },
        "managed_runtime": {"ctx": 8192},
        "managed_accelerator": {
            "mode": "cuda",
            "layout": "experts-only",
            "devices": [
                {
                    "index": 0,
                    "uuid": "GPU-docs-0000",
                    "name": "NVIDIA RTX 5090",
                    "pci_bus_id": "0000:41:00.0",
                    "numa_node": 0,
                },
                {
                    "index": 1,
                    "uuid": "GPU-docs-1111",
                    "name": "NVIDIA RTX 5090",
                    "pci_bus_id": "0000:61:00.0",
                    "numa_node": 1,
                },
            ],
            "mmap": True,
            "rammap": False,
            "async_copy": True,
            "vram_budget": "auto",
            "capability": "available",
        },
        "accelerator_projection": {
            "dense_tensor_bytes": 18 * GIB,
            "dense_gpu_bytes": 0,
            "vram_reserve_per_device_bytes": 2 * GIB,
            "selected_free_bytes": 60 * GIB,
            "selected_total_bytes": 64 * GIB,
            "expert_headroom_bytes": 56 * GIB,
            "exact_per_device_at_runtime": True,
        },
        "mount_options": {"thp": "within_size"},
        "mounts": [{"path": SHARED_MOUNT, "node": None}],
        "blockers": [],
        "warnings": [],
    }


def documentation_hardware() -> dict[str, Any]:
    """Return deterministic NUMA and memory facts for the illustrative host."""

    return {
        "online_nodes": [0, 1],
        "effective_nodes": [0, 1],
        "physical_cores": 64,
        "effective_physical_cores": 64,
        "kernel_release": "6.12.24-docs",
        "nodes": [
            {
                "id": 0,
                "physical_cores": 32,
                "cpu_list": "0-31,64-95",
                "effective_cpu_list": "0-31,64-95",
                "memory_total_bytes": 384 * GIB,
                "memory_available_bytes": 372 * GIB,
                "distance": [10, 20],
            },
            {
                "id": 1,
                "physical_cores": 32,
                "cpu_list": "32-63,96-127",
                "effective_cpu_list": "32-63,96-127",
                "memory_total_bytes": 384 * GIB,
                "memory_available_bytes": 372 * GIB,
                "distance": [20, 10],
            },
        ],
        "memory": {
            "available_bytes": 744 * GIB,
            "total_bytes": 768 * GIB,
        },
        "gpu_discovery": {"status": "available", "error": None},
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-docs-0000",
                "name": "NVIDIA RTX 5090",
                "pci_bus_id": "0000:41:00.0",
                "numa_node": 0,
                "locality": "resolved",
                "total_bytes": 32 * GIB,
                "free_bytes": 30 * GIB,
            },
            {
                "index": 1,
                "uuid": "GPU-docs-1111",
                "name": "NVIDIA RTX 5090",
                "pci_bus_id": "0000:61:00.0",
                "numa_node": 1,
                "locality": "resolved",
                "total_bytes": 32 * GIB,
                "free_bytes": 30 * GIB,
            },
        ],
    }


def documentation_manifest() -> dict[str, Any]:
    """Return a stable manifest for prepared lifecycle screenshots."""

    plan = documentation_plan()
    return {
        "version": 1,
        "deployment_id": "7f3a9d214cc9486aae4275b040000001",
        "created_at": "2026-07-24T12:00:00+00:00",
        "model_fingerprint": plan["model"]["fingerprint"],
        "plan": plan,
        "mounts": [
            {
                "path": SHARED_MOUNT,
                "node": None,
                "identity": {"mount_id": 4012, "device": "0:86"},
            }
        ],
        "processes": [],
        "base_port": 8000,
    }


def documentation_snapshot(state: str) -> ramdisk_textual.ConsoleSnapshot:
    """Build a coherent absent, ready, running, or stopped UI snapshot."""

    if state not in {"absent", "ready", "running", "stopped"}:
        raise ValueError(f"unsupported documentation state: {state}")
    plan = documentation_plan()
    if state == "absent":
        return ramdisk_textual.ConsoleSnapshot(
            plan=plan,
            report={
                "present": False,
                "state": "absent",
                "mounts": [],
                "processes": [],
                "ports": [8000],
            },
            hardware=documentation_hardware(),
            base_port=8000,
        )

    process_rows: list[dict[str, Any]] = []
    if state == "running":
        process_rows.append(
            {
                "pid": 42420,
                "pgid": 42420,
                "port": 8000,
                "node": None,
                "cpu_list": "0-127",
                "running": True,
                "verified": True,
                "reason": "running",
                "identity": {
                    "starttime": 987654321,
                    "deployment_id": "7f3a9d214cc9486aae4275b040000001",
                },
            }
        )
    running = state == "running"
    runtime = {
        "service": {
            "state": "serving" if running else "stopped",
            "label": "SERVING" if running else "STOPPED",
            "endpoints": (
                [
                    {
                        "port": 8000,
                        "node": None,
                        "pid": 42420,
                        "url": "http://127.0.0.1:8000",
                        "process_verified": True,
                        "health_ok": True,
                        "profile_ok": True,
                        "error": None,
                    }
                ]
                if running
                else []
            ),
            "active": 1 if running else 0,
            "queued": 0,
            "error": None,
            "stale": False,
            "observed_at": 1784916000.0,
        },
        "gpus": [
            {
                "index": index,
                "uuid": f"GPU-docs-{index * 1111:04d}",
                "pci_bus_id": (
                    "0000:41:00.0" if index == 0 else "0000:61:00.0"
                ),
                "name": "NVIDIA RTX 5090",
                "selected": True,
                "numa_node": index,
                "utilization_percent": (82.0 - index * 7) if running else 0.0,
                "memory_used_bytes": (29 - index) * GIB if running else 2 * GIB,
                "memory_free_bytes": (3 + index) * GIB if running else 30 * GIB,
                "memory_total_bytes": 32 * GIB,
                "process_vram_bytes": (28 - index) * GIB if running else 0,
                "model_resident_bytes": (26 - index) * GIB if running else 0,
                "expert_bytes": (22 - index) * GIB if running else 0,
                "expert_count": (64 - index * 4) if running else 0,
                "non_expert_bytes": 4 * GIB if running else 0,
                "card_stale": False,
                "model_stale": False,
                "process_stale": False,
                "observed_at": 1784916000.0,
            }
            for index in (0, 1)
        ],
        "tiers": (
            {
                "vram": 124,
                "ram": 96,
                "disk": 36,
                "vram_gb": 43.0,
                "ram_gb": 76.0,
            }
            if running
            else None
        ),
        "tiers_stale": False,
        "latest_profile": (
            {
                "tokens_per_second": 17.8,
                "ttft_ms": 84.0,
                "expert_disk_s": 0.18,
                "expert_wait_s": 0.04,
                "expert_matmul_s": 0.72,
                "attention_s": 0.31,
                "lm_head_s": 0.05,
            }
            if running
            else None
        ),
        "profile_stale": False,
        "process_rss_bytes": 31 * GIB if running else 0,
        "process_stale": False,
        "freshness": {},
    }
    return ramdisk_textual.ConsoleSnapshot(
        plan=plan,
        report={
            "present": True,
            "state": state,
            "mounts": [
                {
                    "path": SHARED_MOUNT,
                    "verified": True,
                    "namespace_verified": True,
                    "identity": {"mount_id": 4012, "device": "0:86"},
                    "memory_nodes": [0, 1],
                }
            ],
            "processes": process_rows,
            "deep_validation": True,
            "source_fingerprint_verified": True,
            "ports": [8000],
        },
        hardware=documentation_hardware(),
        manifest=documentation_manifest(),
        runtime=runtime,
        base_port=8000,
    )


def documentation_args() -> argparse.Namespace:
    """Return CLI arguments matching the illustrative plan."""

    return argparse.Namespace(
        model=MODEL_PATH,
        mode="full",
        topology="interleaved",
        mount_root=MOUNT_ROOT,
        capacity_gb=None,
        profile=None,
        allow_swappable=False,
        thp="within_size",
        prefault=1,
        parallel=4,
        ctx=8192,
        base_port=8000,
        memory_nodes="0-1",
        cpu_list="0-127",
        gpu="0,1",
        gpu_layout="experts-only",
        gpu_placement="auto",
    )


def _make_app(
    snapshot: ramdisk_textual.ConsoleSnapshot,
    lifecycle: RefusingLifecycle,
    privilege: NonAuthorizingPrivilege,
) -> ramdisk_textual.RamdiskTextualApp:
    return ramdisk_textual.RamdiskTextualApp(
        documentation_args(),
        cli_path="/usr/bin/coli",
        engine_path="/usr/libexec/colibri/colibri",
        lifecycle=lifecycle,
        initial_snapshot=snapshot,
        auto_refresh=False,
        privilege_authorizer=privilege,
    )


def _validate_svg(name: str, svg: str) -> None:
    if "Resize to at least 72" in svg:
        raise CaptureSafetyError(f"{name} rendered the minimum-size guard")
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise CaptureSafetyError(f"{name} is not valid SVG: {exc}") from exc
    if not root.tag.endswith("svg"):
        raise CaptureSafetyError(f"{name} has unexpected root element {root.tag}")
    try:
        view_box = tuple(float(value) for value in root.attrib["viewBox"].split())
        width, height = view_box[2], view_box[3]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise CaptureSafetyError(f"{name} has no usable viewBox") from exc
    if width <= 0 or height <= 0:
        raise CaptureSafetyError(f"{name} has empty dimensions")


def _normalize_svg(svg: str) -> str:
    """Remove template-only trailing whitespace while preserving SVG content."""

    return "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"


async def _capture_step(
    *,
    name: str,
    state: str,
    step_key: str,
    expected_text: tuple[str, ...],
    title: str,
    lifecycle: RefusingLifecycle,
    privilege: NonAuthorizingPrivilege,
    message: str = "",
) -> str:
    app = _make_app(documentation_snapshot(state), lifecycle, privilege)
    async with app.run_test(size=VIEWPORT) as pilot:
        await pilot.pause()
        if app.screen.has_class("too-small"):
            raise CaptureSafetyError(f"{name} used an unsafe viewport")
        await pilot.press(step_key)
        await pilot.pause()
        if message:
            app._set_message(message)
            await pilot.pause()
        body = str(app.query_one("#step-body", Static).content)
        for expected in expected_text:
            if expected not in body:
                raise CaptureSafetyError(
                    f"{name} did not render expected text: {expected}"
                )
        if app.operation is not None:
            raise CaptureSafetyError(f"{name} unexpectedly started an operation")
        svg = _normalize_svg(
            app.export_screenshot(title=title, simplify=True)
        )
    _validate_svg(name, svg)
    return svg


async def _capture_prepare_confirmation(
    lifecycle: RefusingLifecycle,
    privilege: NonAuthorizingPrivilege,
) -> str:
    name = "03-prepare-confirmation.svg"
    app = _make_app(documentation_snapshot("absent"), lifecycle, privilege)
    async with app.run_test(size=VIEWPORT) as pilot:
        await pilot.pause()
        if app.screen.has_class("too-small"):
            raise CaptureSafetyError(f"{name} used an unsafe viewport")
        await pilot.press("6")
        await pilot.pause()
        prepare = app.query_one("#action-prepare", Button)
        prepare.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#action-prepare")
        await pilot.pause()
        if not isinstance(app.screen, ramdisk_textual.ContractReviewScreen):
            raise CaptureSafetyError(f"{name} did not open the contract review")
        facts = str(app.screen.query_one("#review-facts", Static).content)
        for expected in (
            "PREPARE EXACTLY THIS CONTRACT",
            "Complete staged copies",
            "Shared means one copy and one engine.",
        ):
            if expected not in facts:
                raise CaptureSafetyError(
                    f"{name} did not render expected text: {expected}"
                )
        if app.operation is not None:
            raise CaptureSafetyError(f"{name} unexpectedly started an operation")
        svg = _normalize_svg(
            app.export_screenshot(
                title="Colibri RAM workspace — Prepare confirmation",
                simplify=True,
            )
        )
    _validate_svg(name, svg)
    return svg


async def render_screenshots() -> CaptureBundle:
    """Render all seven screenshots without writing or touching host state."""

    lifecycle = RefusingLifecycle()
    privilege = NonAuthorizingPrivilege()
    screenshots: dict[str, str] = {}
    screenshots["01-inspect.svg"] = await _capture_step(
        name="01-inspect.svg",
        state="absent",
        step_key="1",
        expected_text=("First, verify the machine", "NUMA INVENTORY"),
        title="Colibri RAM workspace — Inspect",
        lifecycle=lifecycle,
        privilege=privilege,
    )
    screenshots["02-review.svg"] = await _capture_step(
        name="02-review.svg",
        state="absent",
        step_key="6",
        expected_text=(
            "Nothing changes until Prepare is confirmed.",
            "Shared contract confirmed",
        ),
        title="Colibri RAM workspace — Review",
        lifecycle=lifecycle,
        privilege=privilege,
    )
    screenshots["03-prepare-confirmation.svg"] = (
        await _capture_prepare_confirmation(lifecycle, privilege)
    )
    screenshots["04-ready.svg"] = await _capture_step(
        name="04-ready.svg",
        state="ready",
        step_key="7",
        expected_text=(
            "STOPPED",
            "State",
            "ready",
            "Deployment health",
            "verified",
        ),
        title="Colibri RAM workspace — Ready",
        lifecycle=lifecycle,
        privilege=privilege,
        message="Preparation complete; staged weights and placement are verified.",
    )
    screenshots["05-running.svg"] = await _capture_step(
        name="05-running.svg",
        state="running",
        step_key="7",
        expected_text=(
            "SERVING",
            "State",
            "running",
            "Managed processes",
            "verified",
        ),
        title="Colibri RAM workspace — Running",
        lifecycle=lifecycle,
        privilege=privilege,
        message="Managed engine is running at http://127.0.0.1:8000.",
    )
    screenshots["06-stopped.svg"] = await _capture_step(
        name="06-stopped.svg",
        state="stopped",
        step_key="7",
        expected_text=(
            "STOPPED",
            "State",
            "stopped",
            "Mount records",
            "verified",
        ),
        title="Colibri RAM workspace — Stopped",
        lifecycle=lifecycle,
        privilege=privilege,
        message="Managed engine stopped; staged weights remain ready to restart.",
    )
    screenshots["07-absent.svg"] = await _capture_step(
        name="07-absent.svg",
        state="absent",
        step_key="7",
        expected_text=(
            "STOPPED",
            "State",
            "absent",
            "Deployment health",
            "not prepared",
        ),
        title="Colibri RAM workspace — Destroyed",
        lifecycle=lifecycle,
        privilege=privilege,
        message=(
            "RAM workspace removed; durable KV and benchmark state were preserved."
        ),
    )

    if tuple(screenshots) != EXPECTED_ASSETS:
        raise CaptureSafetyError("capture asset set or order changed unexpectedly")
    if lifecycle.attempts:
        raise CaptureSafetyError(
            "lifecycle calls crossed the capture boundary: "
            + ", ".join(lifecycle.attempts)
        )
    if privilege.calls:
        raise CaptureSafetyError(
            "documentation capture unexpectedly requested privilege"
        )
    if lifecycle.plan_token_reads != 1:
        raise CaptureSafetyError(
            "Prepare review did not read exactly one illustrative plan token"
        )
    return CaptureBundle(
        screenshots=screenshots,
        lifecycle_attempts=tuple(lifecycle.attempts),
        plan_token_reads=lifecycle.plan_token_reads,
        privilege_calls=privilege.calls,
    )


def write_screenshots(bundle: CaptureBundle) -> tuple[Path, ...]:
    """Atomically write the exact documentation asset set."""

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    unexpected = {
        path.name for path in ASSET_DIR.glob("*.svg")
    } - set(EXPECTED_ASSETS)
    if unexpected:
        raise CaptureSafetyError(
            "refusing to mix unexpected SVG assets into capture directory: "
            + ", ".join(sorted(unexpected))
        )
    written = []
    for name in EXPECTED_ASSETS:
        svg = bundle.screenshots[name]
        destination = ASSET_DIR / name
        temporary = ASSET_DIR / f".{name}.tmp"
        temporary.write_text(_normalize_svg(svg), encoding="utf-8")
        temporary.replace(destination)
        written.append(destination)
    actual = {path.name for path in ASSET_DIR.glob("*.svg")}
    if actual != set(EXPECTED_ASSETS):
        raise CaptureSafetyError("written screenshot set is incomplete")
    return tuple(written)


def main() -> int:
    bundle = asyncio.run(render_screenshots())
    for path in write_screenshots(bundle):
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
