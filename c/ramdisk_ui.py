"""Dependency-free UI projections for the RAM-disk operator console.

This module converts lifecycle dictionaries into small immutable values that
terminal frontends can render without reimplementing placement, health, or
action rules.  It deliberately performs no I/O and owns no lifecycle state;
``ramdisk.py`` remains the authority for planning, confirmation tokens, and
all mutations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple


class HealthLevel(str, Enum):
    """How strongly an active deployment has been verified."""

    VERIFIED = "verified"
    FAST_CHECK = "fast-check"
    NEEDS_ATTENTION = "needs-attention"


class ReviewAction(str, Enum):
    """Lifecycle actions whose reviewed identity must remain exact."""

    PREPARE = "prepare"
    DESTROY = "destroy"


@dataclass(frozen=True)
class PlacementContract:
    """Exact operator-visible consequences of a staging plan."""

    topology: str
    mode: str
    copy_count: int
    engine_count: int
    staged_bytes_per_copy: int
    total_staged_bytes: int
    numa_nodes: Tuple[int, ...]
    ports: Tuple[int, ...]
    mount_paths: Tuple[str, ...]

    @classmethod
    def from_plan(
        cls, plan: Mapping[str, Any], base_port: int = 8000
    ) -> "PlacementContract":
        mounts = tuple(plan["mounts"])
        topology = str(plan["topology"])
        copy_count = len(mounts) if topology == "per-node" else 1
        staged_bytes_per_copy = int(plan["staging"]["staged_bytes"])
        nodes = tuple(
            int(node)
            for node in plan.get("placement", {}).get(
                "memory_nodes",
                plan.get("hardware", {}).get("online_nodes", ()),
            )
        )
        ports = tuple(
            int(base_port)
            + (0 if mount.get("node") is None else int(mount["node"]))
            for mount in mounts
        )
        return cls(
            topology=topology,
            mode=str(plan["mode"]),
            copy_count=copy_count,
            engine_count=len(mounts),
            staged_bytes_per_copy=staged_bytes_per_copy,
            total_staged_bytes=staged_bytes_per_copy * copy_count,
            numa_nodes=nodes,
            ports=ports,
            mount_paths=tuple(str(mount.get("path", "")) for mount in mounts),
        )

    @property
    def is_replication(self) -> bool:
        return self.topology == "per-node"

    @property
    def is_shared(self) -> bool:
        return self.topology == "interleaved"


@dataclass(frozen=True)
class DeploymentHealth:
    """Immutable health projection for one persisted deployment."""

    level: HealthLevel
    mounts_healthy: bool
    processes_healthy: bool
    lifecycle_state_healthy: bool
    source_fingerprint_verified: Optional[bool]
    deep_validation: bool
    namespaces_verified: bool

    @classmethod
    def from_report(
        cls, plan: Mapping[str, Any], report: Mapping[str, Any]
    ) -> "DeploymentHealth":
        mounts = tuple(report.get("mounts", ()))
        processes = tuple(report.get("processes", ()))
        state = report.get("state")
        expected_mounts = len(plan.get("mounts", ()))
        expected_processes = len(plan.get("mounts", ()))
        mounts_healthy = (
            expected_mounts > 0
            and len(mounts) == expected_mounts
            and all(bool(mount.get("verified")) for mount in mounts)
        )
        if state == "running":
            processes_healthy = len(processes) == expected_processes and all(
                bool(process.get("running")) and bool(process.get("verified"))
                for process in processes
            )
        else:
            processes_healthy = all(
                not bool(process.get("running"))
                and process.get("reason") == "stopped"
                for process in processes
            )
        lifecycle_state_healthy = state in ("ready", "running", "stopped")
        source_fingerprint_verified = report.get("source_fingerprint_verified")
        source_check_not_failed = source_fingerprint_verified is not False
        deep_validation = report.get("deep_validation") is True
        namespaces_verified = (
            expected_mounts > 0
            and len(mounts) == expected_mounts
            and all(mount.get("namespace_verified") is True for mount in mounts)
        )
        fast_check_ok = (
            mounts_healthy
            and processes_healthy
            and lifecycle_state_healthy
            and source_check_not_failed
        )
        deployment_verified = (
            fast_check_ok
            and deep_validation
            and source_fingerprint_verified is True
            and namespaces_verified
        )
        if deployment_verified:
            level = HealthLevel.VERIFIED
        elif fast_check_ok:
            level = HealthLevel.FAST_CHECK
        else:
            level = HealthLevel.NEEDS_ATTENTION
        return cls(
            level=level,
            mounts_healthy=mounts_healthy,
            processes_healthy=processes_healthy,
            lifecycle_state_healthy=lifecycle_state_healthy,
            source_fingerprint_verified=source_fingerprint_verified,
            deep_validation=deep_validation,
            namespaces_verified=namespaces_verified,
        )

    @property
    def fast_check_ok(self) -> bool:
        return self.level in (HealthLevel.VERIFIED, HealthLevel.FAST_CHECK)

    @property
    def verified(self) -> bool:
        return self.level is HealthLevel.VERIFIED


@dataclass(frozen=True)
class ActionPermission:
    """Whether an operator action is currently safe and, if not, why."""

    enabled: bool
    reason: str = ""


@dataclass(frozen=True)
class ActionPolicy:
    """Lifecycle and settings guards derived from a plan and status report."""

    prepare: ActionPermission
    start: ActionPermission
    stop: ActionPermission
    destroy: ActionPermission
    benchmark: ActionPermission
    edit_weights: ActionPermission
    edit_base_port: ActionPermission

    @classmethod
    def from_state(
        cls,
        plan: Optional[Mapping[str, Any]],
        report: Optional[Mapping[str, Any]],
    ) -> "ActionPolicy":
        report = report or {}
        present = bool(report.get("present"))
        state = report.get("state")

        if present:
            prepare = ActionPermission(
                False,
                "A persisted workspace already exists; Destroy it before preparing another.",
            )
        elif plan is None or plan.get("blockers"):
            prepare = ActionPermission(
                False,
                "Preparation is blocked; review every NOT READY item above.",
            )
        else:
            prepare = ActionPermission(True)

        recovery = report.get("recovery")
        recovery = recovery if isinstance(recovery, Mapping) else {}
        retained_processes = recovery.get("retained_processes") or ()
        pending_launches = recovery.get("pending_launches") or ()
        retained_recovery = bool(retained_processes)
        outcome_unknown = bool(pending_launches)
        recovery_blocks_start = retained_recovery or outcome_unknown
        start_enabled = (
            state in ("ready", "stopped") and not recovery_blocks_start
        )
        start = ActionPermission(
            start_enabled,
            ""
            if start_enabled
            else "Resolve managed-process recovery before starting."
            if recovery_blocks_start
            else "Start requires a prepared or stopped workspace.",
        )
        process_records = tuple(report.get("processes", ()))
        process_cleanup_pending = any(
            bool(process.get("running"))
            or process.get("reason") != "stopped"
            for process in process_records
        )
        retained_cleanup_pending = present and retained_recovery
        cleanup_pending = (
            process_cleanup_pending
            or retained_cleanup_pending
            or (present and outcome_unknown)
        )
        stop_enabled = (present and outcome_unknown) or (
            state in ("running", "starting")
            or (present and cleanup_pending)
        )
        if stop_enabled:
            stop_reason = ""
        elif state == "error":
            stop_reason = (
                "No managed engine cleanup is pending; use Destroy to recover "
                "the incomplete workspace."
            )
        else:
            stop_reason = "No managed engine is running."
        stop = ActionPermission(
            stop_enabled,
            stop_reason,
        )
        destroy_enabled = present and not stop_enabled and not outcome_unknown
        if destroy_enabled:
            destroy_reason = ""
        elif outcome_unknown:
            destroy_reason = (
                "Destroy is unavailable while a managed launch outcome is unknown; "
                "use Stop to discover, terminate, and reconcile it first."
            )
        elif retained_cleanup_pending:
            destroy_reason = (
                "Use Stop to reconcile the retained managed process before "
                "destroying its RAM workspace."
            )
        elif stop_enabled:
            destroy_reason = (
                "Stop every managed engine before destroying its RAM workspace."
            )
        else:
            destroy_reason = "There is no RAM workspace to destroy."
        destroy = ActionPermission(destroy_enabled, destroy_reason)
        benchmark_enabled = (
            state in ("ready", "stopped") and not recovery_blocks_start
        )
        benchmark = ActionPermission(
            benchmark_enabled,
            ""
            if benchmark_enabled
            else "Benchmark requires prepared weights and stopped managed engines.",
        )
        if stop_enabled:
            locked_reason = (
                "The persisted weights plan is locked; Stop and Destroy before changing it."
            )
        else:
            locked_reason = (
                "The persisted weights plan is locked; Destroy it before changing it."
            )
        edit_weights = ActionPermission(not present, "" if not present else locked_reason)
        base_port_enabled = (
            (not present or state in ("ready", "stopped"))
            and not recovery_blocks_start
        )
        if base_port_enabled:
            base_port_reason = ""
        elif outcome_unknown:
            base_port_reason = (
                "Use Stop to reconcile the outcome-unknown managed launch "
                "before changing the base port."
            )
        elif state == "error" and not cleanup_pending:
            base_port_reason = (
                "Destroy the incomplete workspace before changing the base port."
            )
        else:
            base_port_reason = "Stop managed engines before changing the base port."
        edit_base_port = ActionPermission(
            base_port_enabled,
            base_port_reason,
        )
        return cls(
            prepare=prepare,
            start=start,
            stop=stop,
            destroy=destroy,
            benchmark=benchmark,
            edit_weights=edit_weights,
            edit_base_port=edit_base_port,
        )


@dataclass(frozen=True)
class ReviewIdentity:
    """The immutable facts displayed when an exact lifecycle token was reviewed."""

    action: ReviewAction
    token: str
    placement: PlacementContract
    mount_paths: Tuple[str, ...]

    @classmethod
    def for_prepare(
        cls, token: str, plan: Mapping[str, Any], base_port: int = 8000
    ) -> "ReviewIdentity":
        placement = PlacementContract.from_plan(plan, base_port)
        return cls(
            action=ReviewAction.PREPARE,
            token=str(token),
            placement=placement,
            mount_paths=placement.mount_paths,
        )

    @classmethod
    def for_destroy(
        cls,
        token: str,
        plan: Mapping[str, Any],
        mounts: Sequence[Mapping[str, Any]],
        base_port: int = 8000,
    ) -> "ReviewIdentity":
        placement = PlacementContract.from_plan(plan, base_port)
        return cls(
            action=ReviewAction.DESTROY,
            token=str(token),
            placement=placement,
            mount_paths=tuple(str(mount.get("path", "")) for mount in mounts),
        )
