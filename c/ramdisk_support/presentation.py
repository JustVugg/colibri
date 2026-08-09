"""Human-readable RAM-disk reports, review tokens, and view projections."""

from __future__ import print_function

import hashlib
import json

from .common import GIB, _format_range_list
from .contracts import (
    ActionPolicy,
    DeploymentHealth,
    HealthLevel,
    PlacementContract,
)
from .presets import PRESET_CHOICES


def _placement_summary(plan, base_port=8000):
    """Describe placement in user terms instead of implementation terms."""
    contract = PlacementContract.from_plan(plan, base_port)
    copies = contract.copy_count
    engines = contract.engine_count
    nodes = list(contract.numa_nodes)
    ports = list(contract.ports)
    each_gib = contract.staged_bytes_per_copy / float(GIB)
    total_gib = contract.total_staged_bytes / float(GIB)
    full = contract.mode == "full"
    copy_name = "complete model" if full else "selected shard set"
    copy_word = "copy" if copies == 1 else "copies"
    engine_word = "engine" if engines == 1 else "independent engines"
    port_word = "port" if len(ports) == 1 else "ports"
    endpoints = "%s %s" % (
        port_word,
        ", ".join(str(port) for port in ports),
    )
    node_labels = ["N%s" % node for node in nodes]
    selected_cpus = plan.get("placement", {}).get("cpu_list")
    cpu_clause = (
        " Selected engine CPUs: %s." % selected_cpus
        if selected_cpus
        else ""
    )

    if contract.is_shared:
        title = "Single shared model (recommended)"
        cost = "%d %s %s (%.2f GiB) · %d %s" % (
            copies,
            copy_name,
            copy_word,
            total_gib,
            engines,
            engine_word,
        )
        explanation = (
            "Stored once; RAM pages are spread across %d NUMA %s selected "
            "for this plan and one engine serves one endpoint.%s"
            % (
                len(nodes),
                "node" if len(nodes) == 1 else "nodes",
                cpu_clause,
            )
        )
        rail = "MODEL x1  ->  RAM [%s]  ->  ENGINE x1" % (
            " | ".join(node_labels) if node_labels else "host"
        )
    else:
        title = (
            "Independent full-model replicas (advanced)"
            if full
            else "Independent staged-set replicas (advanced)"
        )
        cost = "%d %s %s (%d x %.2f GiB = %.2f GiB) · %d %s" % (
            copies,
            copy_name,
            copy_word,
            copies,
            each_gib,
            total_gib,
            engines,
            engine_word,
        )
        explanation = (
            "This is replication, not model sharding: every NUMA node "
            "stores the entire staged set and serves a separate endpoint.%s"
            % cpu_clause
        )
        rail = "MODEL x%d  ->  %s  ->  ENGINES x%d" % (
            copies,
            "  ".join("[%s]" % label for label in node_labels)
            or "[host]",
            engines,
        )
    return {
        "title": title,
        "cost": cost,
        "explanation": explanation,
        "rail": rail,
        "endpoints": endpoints,
        "copy_count": copies,
        "engine_count": engines,
        "ports": ports,
    }


def _accelerator_review(plan):
    accelerator = plan.get("managed_accelerator") or {}
    if accelerator.get("mode") != "cuda":
        return None
    devices = accelerator.get("devices") or []
    projection = plan.get("accelerator_projection") or {}
    dense_gpu_bytes = projection.get("dense_gpu_bytes")
    expert_headroom_bytes = projection.get("expert_headroom_bytes")
    reserve_per_device = projection.get(
        "vram_reserve_per_device_bytes"
    )
    has_projection = all(
        isinstance(value, int)
        for value in (
            dense_gpu_bytes,
            expert_headroom_bytes,
            reserve_per_device,
        )
    )
    return {
        "devices": devices,
        "indices": ",".join(str(device["index"]) for device in devices),
        "layout": str(accelerator.get("layout") or "experts-only"),
        "dense_gpu_gib": (
            float(dense_gpu_bytes) / GIB
            if has_projection
            else None
        ),
        "expert_headroom_gib": (
            float(expert_headroom_bytes) / GIB
            if has_projection
            else None
        ),
        "reserve_per_device_gib": (
            float(reserve_per_device) / GIB
            if has_projection
            else None
        ),
    }


def _human_plan(plan):
    placement = _placement_summary(plan)
    print("RAM-disk plan: %s" % placement["title"])
    preset = plan.get("preset") or {}
    if preset:
        print(
            "  preset: %s%s"
            % (
                preset.get("label", preset.get("id", "Custom")),
                " (Custom)" if preset.get("state") == "custom" else "",
            )
        )
        if preset.get("reason"):
            print("  preset decision: %s" % preset["reason"])
    print("  model: %s" % plan["model"]["path"])
    print("  placement: %s" % placement["cost"])
    print("  endpoints after start: %s" % placement["endpoints"])
    print(
        "  NUMA memory nodes: %s; managed engine CPUs: %s"
        % (
            plan.get("placement", {}).get(
                "memory_node_list",
                "all",
            ),
            plan.get("placement", {}).get("cpu_list", "all"),
        )
    )
    print(
        "  DIMM/channel placement: informational only; "
        "Linux allocates by NUMA node"
    )
    print("  %s" % placement["explanation"])
    print(
        "  staged set: %d shard(s); %d direct expert(s)"
        % (
            len(plan["staging"]["selected_shards"]),
            plan["staging"]["direct_mapped_expert_count"],
        )
    )
    accelerator = _accelerator_review(plan)
    if accelerator is not None:
        print(
            "  accelerator: CUDA selected GPU indices %s; devices %s; "
            "layout %s; GPU-local NUMA %s; mmap upload; VRAM budget auto"
            % (
                accelerator["indices"],
                ", ".join(
                    "%s (%s)"
                    % (
                        device["index"],
                        device.get("name") or "unnamed",
                    )
                    for device in accelerator["devices"]
                ),
                accelerator["layout"],
                _format_range_list(
                    sorted(
                        {
                            int(device["numa_node"])
                            for device in accelerator["devices"]
                        }
                    )
                ),
            )
        )
        if accelerator["dense_gpu_gib"] is not None:
            print(
                "  GPU projection: dense %.2f GiB; expert headroom %.2f GiB; "
                "reserve %.2f GiB/card"
                % (
                    accelerator["dense_gpu_gib"],
                    accelerator["expert_headroom_gib"],
                    accelerator["reserve_per_device_gib"],
                )
            )
    print(
        "  total staged + OS/runtime projection: %.2f GiB; "
        "available: %.2f GiB"
        % (
            plan["reserve"]["total_required_bytes"] / float(GIB),
            plan["reserve"]["available_bytes"] / float(GIB),
        )
    )
    if plan["mode"] == "partial":
        print(
            "  profile coverage: %.1f%%; staging efficiency: %.1f%%"
            % (
                plan["profile"]["coverage"] * 100,
                plan["profile"]["staging_efficiency"] * 100,
            )
        )
        pin = plan["profile"]["pin_comparison"]
        print(
            "  same-budget hot PIN comparison: %.1f%% profile coverage "
            "with %d expert(s)"
            % (
                pin["coverage"] * 100,
                len(pin["selected_experts"]),
            )
        )
    for warning in plan["warnings"]:
        print("  warning: %s" % warning)
    for blocker in plan["blockers"]:
        print("  BLOCKED: %s" % blocker)


def _human_status(report):
    print("RAM-disk state: %s" % report["state"])
    if not report["present"]:
        return
    for mount in report["mounts"]:
        print(
            "  %s: %s"
            % (
                mount["path"],
                "verified tmpfs"
                if mount["verified"]
                else "missing/unverified",
            )
        )
    for process in report["processes"]:
        print(
            "  port %s PID %s: %s"
            % (
                process["port"],
                process["pid"],
                process["reason"],
            )
        )
    recovery = report.get("recovery")
    if not isinstance(recovery, dict):
        return
    print(
        "  recovery: %s / %s"
        % (recovery.get("operation"), recovery.get("state"))
    )
    for path in recovery.get("retained_mounts", []):
        print("    retained mount: %s" % path)
    for path in recovery.get("released_mounts", []):
        print("    released mount: %s" % path)
    for process in recovery.get("retained_processes", []):
        print(
            "    retained PID %s: %s (%s)"
            % (
                process.get("pid"),
                process.get("state_dir"),
                process.get("error") or "absence unproven",
            )
        )
    for pending in recovery.get("pending_launches", []):
        print(
            "    outcome-unknown launch node %s port %s: %s"
            % (
                pending.get("node"),
                pending.get("port"),
                pending.get("state_dir"),
            )
        )
    errors = recovery.get("errors", {})
    if isinstance(errors, dict):
        for name, value in errors.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        print(
                            "    %s PID %s: %s"
                            % (name, item.get("pid"), item.get("error"))
                        )
                    else:
                        print("    %s: %s" % (name, item))
            else:
                print("    %s: %s" % (name, value))
    if recovery.get("action"):
        print("    action: %s" % recovery["action"])


def _human_benchmark_summary(result):
    """Render only the operator-facing causal result and evidence location."""
    print(
        "Causal benchmark: %s / %s"
        % (result.get("status"), result.get("claim"))
    )
    protocol_id = result.get("protocol_id")
    if protocol_id:
        print("  protocol: %s" % protocol_id)
    print(
        "  %d replicate attempt(s); %d successful"
        % (
            int(result.get("attempted_replicates", 0)),
            int(result.get("successful_replicates", 0)),
        )
    )
    if result.get("raw_evidence_path"):
        print("  append-only evidence: %s" % result["raw_evidence_path"])
    for reason in result.get("reasons") or []:
        print("  neutral: %s" % reason)
