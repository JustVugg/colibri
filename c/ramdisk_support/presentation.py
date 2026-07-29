"""Human-readable RAM-disk reports, review tokens, and view projections."""

from __future__ import print_function

import hashlib
import json

from .common import GIB, _format_range_list
from .presets import PRESET_CHOICES
from ramdisk_ui import (
    ActionPolicy,
    DeploymentHealth,
    HealthLevel,
    PlacementContract,
)


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


def _human_benchmark(result):
    print(
        "RAM-disk benchmark (%s / %s)"
        % (result["mode"], result["topology"])
    )
    for variant in result["variants"]:
        if variant.get("status") != "ok":
            print(
                "  %-30s %s"
                % (variant["name"], variant.get("status"))
            )
            continue
        interactive = variant["interactive"]
        print(
            "  %-30s TTFT %s ms  tok/s p50 %s p95 %s  RAM %.1f%%"
            % (
                variant["name"],
                "%.1f" % interactive["ttft_ms"]
                if interactive["ttft_ms"] is not None
                else "n/a",
                "%.3f" % interactive["p50_tokens_per_second"]
                if interactive["p50_tokens_per_second"] is not None
                else "n/a",
                "%.3f" % interactive["p95_tokens_per_second"]
                if interactive["p95_tokens_per_second"] is not None
                else "n/a",
                interactive["ram_map_coverage"] * 100,
            )
        )
        print(
            "    forward p50/p99 %s/%s ms  SSD %s bytes/token"
            % (
                "%.1f" % interactive["forward_p50_ms"]
                if interactive["forward_p50_ms"] is not None
                else "n/a",
                "%.1f" % interactive["forward_p99_ms"]
                if interactive["forward_p99_ms"] is not None
                else "n/a",
                "%.0f" % interactive["ssd_bytes_per_token"]
                if interactive["ssd_bytes_per_token"] is not None
                else "n/a",
            )
        )
    aggregate = result["aggregate"]
    print(
        "  aggregate: %s  slowest %s tok/s  total %s tok/s"
        % (
            aggregate.get("status"),
            "%.3f" % aggregate["slowest_node_tokens_per_second"]
            if aggregate.get("slowest_node_tokens_per_second")
            is not None
            else "n/a",
            "%.3f" % aggregate["total_tokens_per_second"]
            if aggregate.get("total_tokens_per_second") is not None
            else "n/a",
        )
    )
    system = result["system"]
    print(
        "  system: stage %s s  prefault %s s  RSS %s GiB  "
        "mount shmem %.2f GiB"
        % (
            "%.1f" % system["stage_seconds"]
            if system.get("stage_seconds") is not None
            else "n/a",
            "%.2f" % system["prefault_seconds"]
            if system.get("prefault_seconds") is not None
            else "n/a",
            "%.2f" % (system["rss_bytes"] / GIB)
            if system.get("rss_bytes") is not None
            else "n/a",
            system["shmem_bytes"] / float(GIB),
        )
    )
    print(
        "    swap +%.3f GiB  host huge-page coverage %.1f%%  NUMA %s"
        % (
            system["swap_delta_bytes"] / float(GIB),
            system["huge_page_coverage"] * 100,
            system["numa_page_placement"],
        )
    )
    print(
        "  acceptance: paths=%s outputs=%s no-swap-growth=%s "
        "within-budget=%s"
        % (
            result["acceptance"].get(
                "all_required_paths_succeeded"
            ),
            result["acceptance"]["greedy_outputs_identical"],
            result["acceptance"]["no_swap_growth"],
            result["acceptance"]["staging_within_budget"],
        )
    )
    if (
        result["acceptance"].get(
            "full_zero_physical_ssd_reads_verified"
        )
        is not None
    ):
        print(
            "    full direct physical SSD reads measured zero: %s"
            % result["acceptance"][
                "full_zero_physical_ssd_reads_verified"
            ]
        )
    print(
        "  best knobs for this topology: %s"
        % result["best_runtime_knobs"]
    )


def _plan_confirmation_token(plan):
    """Stable identity for exactly the plan a user reviewed in the TUI."""
    reviewed = {
        "schema": plan.get("schema"),
        "version": plan.get("version"),
        "model_fingerprint": plan.get("model", {}).get("fingerprint"),
        "mode": plan.get("mode"),
        "topology": plan.get("topology"),
        "placement": plan.get("placement"),
        "mount_root": plan.get("mount_root"),
        "capacity_bytes": plan.get("capacity_bytes"),
        "selected_shards": plan.get("staging", {}).get(
            "selected_shards"
        ),
        "linked_shards": plan.get("staging", {}).get(
            "linked_shards"
        ),
        "total_staged_bytes": plan.get("staging", {}).get(
            "total_staged_bytes"
        ),
        "total_required_bytes": plan.get("reserve", {}).get(
            "total_required_bytes"
        ),
        "mounts": plan.get("mounts"),
        "mount_options": plan.get("mount_options"),
        "prefault": plan.get("prefault"),
        "parallel": plan.get("parallel"),
        "managed_runtime": plan.get("managed_runtime"),
        "managed_accelerator": plan.get("managed_accelerator"),
        "preset": plan.get("preset"),
    }
    payload = json.dumps(
        reviewed,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_confirmation_token(
    manifest,
    *,
    persisted_base_port,
):
    """Bind a destructive confirmation to one prepared deployment."""
    mounts = []
    for record in manifest.get("mounts", []):
        identity = record.get("identity", {})
        mounts.append(
            {
                "path": record.get("path"),
                "node": record.get("node"),
                "mount_id": identity.get("mount_id"),
                "device": identity.get("device"),
            }
        )
    processes = []
    for record in manifest.get("processes", []):
        processes.append(
            {
                "pid": record.get("pid"),
                "pgid": record.get("pgid"),
                "uid": record.get("uid"),
                "starttime": record.get("starttime"),
                "nonce": record.get("nonce"),
                "port": record.get("port"),
                "node": record.get("node"),
            }
        )
    reviewed = {
        "version": manifest.get("version"),
        "deployment_id": manifest.get("deployment_id"),
        "created_at": manifest.get("created_at"),
        "state": manifest.get("state"),
        "base_port": persisted_base_port(manifest),
        "model_fingerprint": manifest.get("model_fingerprint"),
        "plan_token": _plan_confirmation_token(
            manifest.get("plan", {})
        ),
        "mounts": mounts,
        "processes": processes,
    }
    payload = json.dumps(
        reviewed,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prepare_confirmation(plan, base_port=8000):
    contract = PlacementContract.from_plan(plan, base_port)
    placement = _placement_summary(plan, base_port)
    copies = contract.copy_count
    each_gib = contract.staged_bytes_per_copy / float(GIB)
    total_gib = contract.total_staged_bytes / float(GIB)
    copy_name = (
        "complete model"
        if contract.mode == "full"
        else "selected shard set"
    )
    if contract.is_shared:
        nodes = len(contract.numa_nodes)
        accelerator = _accelerator_review(plan)
        accelerator_text = ""
        if accelerator is not None:
            accelerator_text = (
                "GPU(s) %s use layout %s. "
                % (
                    accelerator["indices"],
                    accelerator["layout"],
                )
            )
            if accelerator["dense_gpu_gib"] is not None:
                accelerator_text += (
                    "Projected dense VRAM %.2f GiB and expert headroom "
                    "%.2f GiB after %.2f GiB/card reserve. "
                    % (
                        accelerator["dense_gpu_gib"],
                        accelerator["expert_headroom_gib"],
                        accelerator["reserve_per_device_gib"],
                    )
                )
        return (
            "CONFIRM SHARED PLAN: stage %d %s copy (%.2f GiB) at %s, "
            "spread across %d NUMA %s. Memory nodes %s; engine CPUs %s. "
            "%sStart will launch 1 engine on %s. tmpfs size is a cap, THP "
            "is requested rather than guaranteed, and copy workers do not "
            "create replicas. Press p again within 10s."
            % (
                copies,
                copy_name,
                total_gib,
                plan["mount_root"],
                nodes,
                "node" if nodes == 1 else "nodes",
                plan.get("placement", {}).get(
                    "memory_node_list",
                    "all",
                ),
                plan.get("placement", {}).get("cpu_list", "all"),
                accelerator_text,
                placement["endpoints"],
            )
        )
    return (
        "CONFIRM REPLICA PLAN: stage %d %s copies "
        "(%d x %.2f GiB = %.2f GiB) at %s. Memory nodes %s; "
        "selected CPUs %s. Start will launch %d independent engines on %s. "
        "This is replication, not sharding, and does not accelerate one "
        "request. Press p again within 10s."
        % (
            copies,
            copy_name,
            copies,
            each_gib,
            total_gib,
            plan["mount_root"],
            plan.get("placement", {}).get(
                "memory_node_list",
                "all",
            ),
            plan.get("placement", {}).get("cpu_list", "all"),
            placement["engine_count"],
            placement["endpoints"],
        )
    )


def _prepare_confirmation_rows(plan, base_port=8000):
    """Put the irreversible topology facts in the first three TUI rows."""
    contract = PlacementContract.from_plan(plan, base_port)
    placement = _placement_summary(plan, base_port)
    copies = contract.copy_count
    engines = contract.engine_count
    if contract.mode == "full":
        copies_text = "%d complete model %s" % (
            copies,
            "copy" if copies == 1 else "copies",
        )
    else:
        copies_text = "%d selected shard-set %s" % (
            copies,
            "copy" if copies == 1 else "copies",
        )
    accelerator = None
    if contract.is_replication:
        placement_text = "DANGER · replication, not sharding"
        engines_text = "%d independent engines" % engines
    else:
        nodes = len(contract.numa_nodes)
        placement_text = "SHARED · pages span %d NUMA %s" % (
            nodes,
            "node" if nodes == 1 else "nodes",
        )
        engines_text = "%d engine on %s" % (
            engines,
            placement["endpoints"],
        )
        accelerator = _accelerator_review(plan)
        if accelerator is not None:
            engines_text += " · GPU(s) %s" % accelerator["indices"]
            placement_text += " · layout %s" % accelerator["layout"]
    rows = [
        ("warn", "REVIEW · %s" % copies_text),
        ("warn", "START · %s" % engines_text),
        (
            "bad" if contract.is_replication else "accent",
            placement_text,
        ),
    ]
    if (
        not contract.is_replication
        and accelerator is not None
        and accelerator["dense_gpu_gib"] is not None
    ):
        rows.append(
            (
                "normal",
                "VRAM · dense %.2f GiB · expert headroom %.2f GiB · "
                "reserve %.2f GiB/card"
                % (
                    accelerator["dense_gpu_gib"],
                    accelerator["expert_headroom_gib"],
                    accelerator["reserve_per_device_gib"],
                ),
            )
        )
    return rows


def _tui_plan_rows(
    plan,
    report,
    active=False,
    base_port=8000,
    confirmation=None,
):
    placement = _placement_summary(plan, base_port)
    rows = []
    if confirmation:
        rows.extend(_prepare_confirmation_rows(plan, base_port))
        rows.append(("normal", ""))
    rows.extend(
        [
            (
                "dim",
                "ACTIVE DEPLOYMENT"
                if active
                else "DRAFT PLAN · nothing has been changed yet",
            ),
            (
                "warn"
                if plan["topology"] == "per-node"
                else "accent",
                placement["title"],
            ),
            ("accent", placement["rail"]),
            ("normal", placement["cost"]),
            (
                "normal",
                "After Start: %s" % placement["endpoints"],
            ),
            (
                "warn" if plan["topology"] == "per-node" else "dim",
                placement["explanation"],
            ),
            ("normal", ""),
            *(
                [
                    (
                        "accent",
                        "PRESET · %s%s"
                        % (
                            plan["preset"].get("label", "Custom"),
                            (
                                " · CUSTOM"
                                if plan["preset"].get("state") == "custom"
                                else ""
                            ),
                        ),
                    ),
                    (
                        "dim",
                        plan["preset"].get("reason") or "Reviewed draft values.",
                    ),
                    ("normal", ""),
                ]
                if plan.get("preset")
                else []
            ),
            (
                "heading",
                "STAGING · %s"
                % (
                    "full model"
                    if plan["mode"] == "full"
                    else "profile-selected shard set"
                ),
            ),
            (
                "normal",
                "%d of %d shards in RAM · %d direct-mapped experts · "
                "prefault %s"
                % (
                    len(plan["staging"]["selected_shards"]),
                    plan["model"]["shard_count"],
                    plan["staging"]["direct_mapped_expert_count"],
                    "on" if plan["prefault"] else "off",
                ),
            ),
            (
                "normal",
                "%s host memory %.2f GiB · %s available %.2f GiB"
                % (
                    "Planned" if active else "Projected",
                    plan["reserve"]["total_required_bytes"]
                    / float(GIB),
                    "at preparation" if active else "currently",
                    plan["reserve"]["available_bytes"] / float(GIB),
                ),
            ),
        ]
    )
    accelerator = _accelerator_review(plan)
    if accelerator is not None:
        devices = accelerator["devices"]
        rows.extend(
            [
                (
                    "heading",
                    "GPU STAGING · one shared source, one multi-GPU engine",
                ),
                (
                    "accent",
                    "GPU(s) %s · PCI/NUMA %s"
                    % (
                        " · ".join(
                            "%s %s"
                            % (
                                device["index"],
                                device.get("name") or "unnamed",
                            )
                            for device in devices
                        ),
                        " · ".join(
                            "%s → N%s"
                            % (
                                device.get("pci_bus_id", "?"),
                                device.get("numa_node", "?"),
                            )
                            for device in devices
                        ),
                    ),
                ),
                (
                    "normal",
                    "Selected GPU indices %s · layout %s · "
                    "COLI_MMAP upload · async CUDA copies"
                    % (
                        accelerator["indices"],
                        accelerator["layout"],
                    ),
                ),
            ]
        )
        if accelerator["dense_gpu_gib"] is not None:
            rows.append(
                (
                    "normal",
                    "Projected dense VRAM %.2f GiB · expert headroom "
                    "%.2f GiB · reserve %.2f GiB/card"
                    % (
                        accelerator["dense_gpu_gib"],
                        accelerator["expert_headroom_gib"],
                        accelerator["reserve_per_device_gib"],
                    ),
                )
            )
    if plan["mode"] == "partial":
        rows.append(
            (
                "normal",
                "Profile coverage %.1f%% · staging efficiency %.1f%%"
                % (
                    plan["profile"]["coverage"] * 100,
                    plan["profile"]["staging_efficiency"] * 100,
                ),
            )
        )
    if confirmation:
        rows.extend(
            [
                ("normal", ""),
                ("warn", "FULL PREPARATION DETAIL"),
                ("warn", confirmation),
            ]
        )
    if active:
        health = DeploymentHealth.from_report(plan, report)
        if health.level is HealthLevel.VERIFIED:
            health_style = "good"
            health_title = "DEPLOYMENT VERIFIED"
            health_detail = (
                "Persisted settings are locked. Activity shows current "
                "mount and engine health."
            )
        elif health.level is HealthLevel.FAST_CHECK:
            health_style = "warn"
            health_title = (
                "FAST CHECK PASSED · DEEP VERIFICATION PENDING"
            )
            health_detail = (
                "Press R for source and NUMA verification; Start also "
                "revalidates before launch."
            )
        else:
            health_style = "bad"
            health_title = "DEPLOYMENT NEEDS ATTENTION"
            health_detail = (
                "Open Activity and press R before Start; Destroy "
                "revalidates every exact mount."
            )
        rows.extend(
            [
                ("normal", ""),
                (health_style, health_title),
                (
                    "dim" if health.fast_check_ok else "bad",
                    health_detail,
                ),
            ]
        )
    elif plan["blockers"]:
        rows.extend([("normal", ""), ("bad", "NOT READY")])
        rows.extend(
            ("bad", blocker)
            for blocker in plan["blockers"]
        )
    else:
        rows.extend([("normal", ""), ("good", "READY")])
        if confirmation:
            rows.append(
                (
                    "warn",
                    "Press p again before the confirmation expires, "
                    "or any change cancels it.",
                )
            )
        else:
            rows.append(
                (
                    "dim",
                    "Review the copy count and memory total above, "
                    "then press p to prepare.",
                )
            )
    rows.extend(
        ("warn", warning)
        for warning in plan["warnings"]
    )
    if report.get("present"):
        rows.append(
            (
                "dim",
                "Lifecycle state: %s"
                % report.get("state", "unknown"),
            )
        )
    return rows


def _tui_hardware_rows(hardware):
    nodes = hardware.get("nodes", [])
    rows = [
        ("heading", "HOST MEMORY TOPOLOGY"),
        (
            "normal",
            "%.1f GiB available / %.1f GiB total · %d physical cores · "
            "%d NUMA %s"
            % (
                hardware["memory"]["available_bytes"] / float(GIB),
                hardware["memory"]["total_bytes"] / float(GIB),
                hardware["physical_cores"],
                len(nodes),
                "node" if len(nodes) == 1 else "nodes",
            ),
        ),
        (
            "dim",
            "NUMA nodes determine RAM placement. CPU cores do not "
            "create model copies.",
        ),
        (
            "normal",
            "Kernel %s · tmpfs %s · noswap %s · THP %s"
            % (
                hardware["kernel_release"],
                "available"
                if hardware["tmpfs"]["supported"]
                else "missing",
                "available"
                if hardware["tmpfs"]["noswap_supported"]
                else "missing",
                hardware["thp"]["shmem_enabled"] or "unknown",
            ),
        ),
        (
            "warn" if hardware["swap"]["used_bytes"] else "dim",
            "Swap in use %.2f GiB"
            % (
                hardware["swap"]["used_bytes"]
                / float(GIB)
            ),
        ),
        ("normal", ""),
    ]
    for node in nodes:
        rows.extend(
            [
                (
                    "accent",
                    "NUMA %s · CPUs %s · %d physical cores"
                    % (
                        node["id"],
                        node["cpu_list"],
                        node["physical_cores"],
                    ),
                ),
                (
                    "normal",
                    "  %.1f GiB available / %.1f GiB total · "
                    "distance %s"
                    % (
                        node["memory_available_bytes"]
                        / float(GIB),
                        node["memory_total_bytes"] / float(GIB),
                        node["distance"],
                    ),
                ),
            ]
        )
    return rows


def _tui_activity_rows(
    report,
    hardware,
    process_metrics=None,
    *,
    meminfo,
):
    rows = [
        (
            "heading",
            "LIFECYCLE · %s"
            % report.get("state", "unknown").upper(),
        )
    ]
    if not report.get("present"):
        rows.extend(
            [
                ("dim", "No RAM workspace exists yet."),
                (
                    "normal",
                    "Review the Plan page, then prepare it with p.",
                ),
            ]
        )
        return rows
    rows.append(
        (
            "dim",
            "%s validation · manifest %s"
            % (
                "deep"
                if report.get("deep_validation")
                else "fast",
                report.get("manifest_path"),
            ),
        )
    )
    rows.extend(
        [("normal", ""), ("heading", "RAM MOUNTS")]
    )
    for mount in report.get("mounts", []):
        rows.append(
            (
                "good" if mount.get("verified") else "bad",
                "%s · %s · NUMA pages %s"
                % (
                    mount["path"],
                    "verified"
                    if mount.get("verified")
                    else "missing or unverified",
                    mount.get("numa_allocation")
                    or "not sampled",
                ),
            )
        )
    rows.extend(
        [("normal", ""), ("heading", "MANAGED ENGINES")]
    )
    processes = report.get("processes", [])
    if not processes:
        rows.append(
            (
                "dim",
                "No engine is running. Prepared weights stay "
                "resident until Destroy.",
            )
        )
    for process in processes:
        rows.append(
            (
                "good" if process.get("running") else "dim",
                "port %s · PID %s · node %s · %s"
                % (
                    process.get("port"),
                    process.get("pid"),
                    process.get("node"),
                    process.get("reason"),
                ),
            )
        )
        metrics = (process_metrics or {}).get(
            process.get("pid"),
            {},
        )
        if metrics.get("rss_bytes") is not None:
            rows.append(
                (
                    "dim",
                    "  RSS %.2f GiB across %d processes · "
                    "RAM map %s experts / %s GiB"
                    % (
                        metrics["rss_bytes"] / float(GIB),
                        metrics["rss_processes"],
                        metrics["rammap_experts"]
                        if metrics["rammap_experts"] is not None
                        else "n/a",
                        "%.2f"
                        % (metrics["rammap_bytes"] / GIB)
                        if metrics["rammap_bytes"] is not None
                        else "n/a",
                    ),
                )
            )
    mem = meminfo()
    rows.extend(
        [
            ("normal", ""),
            (
                "dim",
                "Host shared memory %.2f GiB · swap %.3f GiB"
                % (
                    mem.get("Shmem", 0) / float(GIB),
                    hardware["swap"]["used_bytes"] / float(GIB),
                ),
            ),
        ]
    )
    return rows


def _tui_benchmark_rows(history):
    rows = [("heading", "PERSISTENT PATH SCORECARD")]
    results = (history or {}).get("results", [])
    if not results:
        rows.extend(
            [
                ("dim", "No benchmark history yet."),
                (
                    "normal",
                    "Prepare the workspace, stop managed engines, "
                    "then press b here.",
                ),
            ]
        )
        return rows
    latest = results[-1]
    rows.append(
        (
            "accent",
            "Latest %s · best %s"
            % (
                latest.get("created_at"),
                latest.get("best_variant"),
            ),
        )
    )
    for variant in latest.get("variants", []):
        if variant.get("status") != "ok":
            rows.append(
                (
                    "warn",
                    "%s · %s"
                    % (
                        variant.get("name"),
                        variant.get("status"),
                    ),
                )
            )
            continue
        score = variant.get("interactive", {})
        rows.append(
            (
                "normal",
                "%s · TTFT %s ms · %.2f tok/s p50 · RAM %.1f%% · "
                "SSD %s B/token"
                % (
                    variant.get("name"),
                    "%.1f" % score["ttft_ms"]
                    if score.get("ttft_ms") is not None
                    else "n/a",
                    score.get("p50_tokens_per_second") or 0.0,
                    (
                        score.get("ram_map_coverage")
                        or 0.0
                    )
                    * 100,
                    "%.0f" % score["ssd_bytes_per_token"]
                    if score.get("ssd_bytes_per_token")
                    is not None
                    else "n/a",
                ),
            )
        )
    aggregate = latest.get("aggregate", {})
    rows.append(
        (
            "dim",
            "Aggregate %s · slowest %s tok/s · total %s tok/s"
            % (
                aggregate.get("status", "n/a"),
                aggregate.get(
                    "slowest_node_tokens_per_second",
                    "n/a",
                ),
                aggregate.get(
                    "total_tokens_per_second",
                    "n/a",
                ),
            ),
        )
    )
    return rows


def _tui_settings_rows(
    args,
    plan,
    report,
    base_port=8000,
):
    rows = [("heading", "WORKSPACE SETTINGS")]
    preset = plan.get("preset") or {}
    if preset:
        rows.extend(
            [
                (
                    "accent" if preset.get("state") != "custom" else "warn",
                    "Preset · %s%s"
                    % (
                        preset.get("label", preset.get("id", "Custom")),
                        " · Custom" if preset.get("state") == "custom" else "",
                    ),
                ),
                ("dim", preset.get("reason") or "Advanced draft values."),
            ]
        )
    if report.get("present"):
        placement = _placement_summary(plan, base_port)
        can_change_port = report.get("state") in (
            "ready",
            "stopped",
        )
        rows.extend(
            [
                ("warn", "LOCKED BY ACTIVE DEPLOYMENT"),
                ("normal", placement["title"]),
                ("normal", placement["cost"]),
                (
                    "normal" if can_change_port else "dim",
                    (
                        "[P] Next Start base port %s"
                        if can_change_port
                        else "Current base port        %s"
                    )
                    % base_port,
                ),
                (
                    "dim",
                    "Start uses the persisted weights plan shown here. "
                    "Stop before changing its next endpoint; Destroy "
                    "before changing placement or staging.",
                ),
            ]
        )
        return rows
    placement = _placement_summary(plan, base_port)
    rows.extend(
        [
            (
                "warn"
                if plan["topology"] == "per-node"
                else "accent",
                "Placement · %s" % placement["title"],
            ),
            ("normal", placement["cost"]),
            ("dim", placement["explanation"]),
        ]
    )
    if plan["topology"] == "per-node":
        rows.append(
            ("good", "[i] Return to one shared copy")
        )
    else:
        rows.append(
            (
                "dim",
                "Replica mode is explicit-only: choose Multiple NUMA "
                "replicas at startup or pass --topology per-node.",
            )
        )
    rows.extend(
        [
            ("normal", ""),
            ("normal", "[m] Staging mode        %s" % args.mode),
            (
                "normal",
                "[c] Per-copy budget    %s"
                % (
                    "%.1f GiB" % args.capacity_gb
                    if args.capacity_gb
                    else "full model size"
                ),
            ),
            (
                "normal",
                "[r] Usage profile      %s"
                % (args.profile or "<model>/.coli_usage"),
            ),
            (
                "normal",
                "[o] Mount root         %s" % args.mount_root,
            ),
            (
                "normal",
                "[P] Base port          %s" % args.base_port,
            ),
            (
                "normal",
                "[w] Copy workers       %s "
                "(copy concurrency only)" % args.parallel,
            ),
            (
                "normal",
                "[H] Huge pages         %s" % args.thp,
            ),
            (
                "normal",
                "[f] Prefault           %s"
                % ("on" if plan["prefault"] else "off"),
            ),
            (
                "normal",
                "[y] Swappable tmpfs    %s"
                % (
                    "allowed"
                    if args.allow_swappable
                    else "refused"
                ),
            ),
            ("normal", ""),
            (
                "dim",
                "Full mode always stages the full model; capacity "
                "changes only apply to partial mode.",
            ),
        ]
    )
    return rows


def _tui_preset_rows():
    rows = [
        ("heading", "WHAT SHOULD COLIBRI OPTIMIZE?"),
        (
            "dim",
            "Choose once to prepopulate the draft. Nothing is mounted or copied.",
        ),
        ("normal", ""),
    ]
    for index, (_preset_id, label, description) in enumerate(
        PRESET_CHOICES,
        1,
    ):
        rows.append(
            (
                "accent" if index == 1 else "normal",
                "[%d] %s%s"
                % (
                    index,
                    label,
                    " · default" if index == 1 else "",
                ),
            )
        )
        rows.append(("dim", "    %s" % description))
    rows.extend(
        [
            ("normal", ""),
            (
                "dim",
                "Enter selects Fastest GPU staging. Advanced settings remain editable.",
            ),
        ]
    )
    return rows


def _tui_help_rows():
    return [
        ("heading", "HOW THIS WORKS"),
        (
            "normal",
            "1. Plan shows exactly how many model copies, engines, "
            "ports, and GiB will be created.",
        ),
        (
            "normal",
            "2. Prepare mounts tmpfs and copies weights. It does not "
            "start an engine.",
        ),
        (
            "normal",
            "3. Start launches the persisted deployment; Stop keeps "
            "RAM weights; Destroy unmounts them.",
        ),
        ("normal", ""),
        (
            "accent",
            "Shared placement is the normal path: one model copy and "
            "one engine across all NUMA nodes.",
        ),
        (
            "warn",
            "Per-node means independent full replicas, not a model "
            "split. It is never enabled by a TUI toggle.",
        ),
        ("normal", ""),
        ("heading", "KEYS"),
        ("normal", "Left/Right or h/l · change page"),
        ("normal", "Up/Down or j/k · scroll"),
        (
            "normal",
            "p · review/prepare     s · start     "
            "x · stop     d · destroy",
        ),
        (
            "normal",
            "b · benchmark          R · deep refresh          "
            "? · close help",
        ),
        (
            "normal",
            "c · cancel prepare/start/benchmark at a safe "
            "cleanup checkpoint",
        ),
        (
            "normal",
            "Settings page · edit draft settings before preparation",
        ),
        (
            "normal",
            "q or Esc · quit; long operations cancel safely first, "
            "cleanup finishes before exit",
        ),
    ]


def _tui_idle_action_hint(screen, plan, report):
    """Return only actions the shared lifecycle policy permits."""
    policy = ActionPolicy.from_state(plan, report)
    if (
        screen == 0
        and report
        and not report.get("present")
        and policy.prepare.enabled
    ):
        return "[p] review / prepare"
    if screen == 3 and policy.benchmark.enabled:
        return "[b] benchmark"
    if policy.start.enabled:
        return "[s] start  [d] destroy"
    if policy.stop.enabled:
        return "[x] stop  [d] destroy"
    if policy.destroy.enabled:
        return "[d] destroy"
    return "[R] refresh"
