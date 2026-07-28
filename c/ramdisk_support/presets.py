"""Pure first-run RAM-workspace preset resolution."""

from __future__ import print_function

import argparse
import copy
import os

from .common import GIB, MIB, RamdiskError, _format_range_list


PRESET_GPU_FASTEST = "gpu-fastest"
PRESET_SINGLE = "single"
PRESET_MINIMAL = "minimal"
PRESET_REPLICAS = "replicas"

PRESET_CHOICES = (
    (
        PRESET_GPU_FASTEST,
        "Fastest GPU staging",
        "One shared copy on GPU-local NUMA nodes; one multi-GPU engine.",
    ),
    (
        PRESET_SINGLE,
        "Single RAM copy",
        "One full shared copy using the normal effective NUMA placement.",
    ),
    (
        PRESET_MINIMAL,
        "Minimal RAM",
        "Largest safe profile-guided partial staging set.",
    ),
    (
        PRESET_REPLICAS,
        "Multiple NUMA replicas",
        "Advanced: one complete copy and independent engine per NUMA node.",
    ),
)

_PRESET_LABELS = {
    preset_id: label for preset_id, label, _description in PRESET_CHOICES
}


def _engine_cuda_capable(engine_path):
    """Return whether a local engine contains the CUDA backend marker."""
    if not engine_path:
        return None
    try:
        with open(os.path.realpath(engine_path), "rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    return False
                if b"[CUDA] mode: routed experts" in block:
                    return True
    except OSError:
        return False


def _namespace(args):
    return argparse.Namespace(**copy.deepcopy(vars(args)))


def _set_common(draft, preset_id):
    draft.ramdisk_preset = preset_id
    draft.ramdisk_preset_label = _PRESET_LABELS[preset_id]
    draft.ramdisk_preset_reason = ""
    draft.ramdisk_preset_fallback = None
    draft.managed_accelerator = None
    draft.prefault = None
    return draft


def _single_draft(args, preset_id=PRESET_SINGLE):
    draft = _set_common(_namespace(args), preset_id)
    draft.mode = "full"
    draft.topology = "interleaved"
    draft.capacity_gb = None
    draft.memory_nodes = None
    draft.cpu_list = None
    return draft


def _replica_draft(args):
    draft = _set_common(_namespace(args), PRESET_REPLICAS)
    draft.mode = "full"
    draft.topology = "per-node"
    draft.capacity_gb = None
    draft.memory_nodes = None
    draft.cpu_list = None
    draft.ramdisk_preset_reason = (
        "Explicit replication: every selected NUMA node receives a complete "
        "copy and an independent managed engine."
    )
    return draft


def _gpu_local_draft(args, hardware, devices, cuda_capable):
    draft = _set_common(_namespace(args), PRESET_GPU_FASTEST)
    nodes = []
    for device in devices:
        node = int(device["numa_node"])
        if node not in nodes:
            nodes.append(node)
    node_rows = {
        int(row["id"]): row
        for row in hardware.get("nodes", [])
        if isinstance(row, dict) and isinstance(row.get("id"), int)
    }
    cpus = set()
    for node in nodes:
        row = node_rows.get(node, {})
        node_cpus = (
            row.get("effective_cpus", [])
            if "effective_cpus" in row
            else row.get("cpus", [])
        )
        cpus.update(int(cpu) for cpu in node_cpus)
    cpus = sorted(cpus)
    if not cpus:
        raise RamdiskError(
            "GPU-local NUMA nodes expose no complete effective CPU cores"
        )
    draft.mode = "full"
    draft.topology = "interleaved"
    draft.capacity_gb = None
    draft.memory_nodes = _format_range_list(nodes)
    draft.cpu_list = _format_range_list(cpus)
    draft.managed_accelerator = {
        "mode": "cuda",
        "devices": [
            {
                "index": int(device["index"]),
                "name": str(device["name"]),
                "pci_bus_id": str(device["pci_bus_id"]),
                "numa_node": int(device["numa_node"]),
            }
            for device in devices
        ],
        "mmap": True,
        "rammap": False,
        "async_copy": True,
        "vram_budget": "auto",
        "capability": (
            "available"
            if cuda_capable is True
            else "unverified"
        ),
    }
    draft.ramdisk_preset_reason = (
        "One shared model copy across GPU-local NUMA node(s) %s; "
        "one managed engine uses GPU(s) %s."
        % (
            _format_range_list(nodes),
            ",".join(str(device["index"]) for device in devices),
        )
    )
    return draft


def _memory_admitted(plan):
    reserve = plan.get("reserve") or {}
    available = reserve.get("available_bytes")
    required = reserve.get("total_required_bytes")
    if not isinstance(available, int) or not isinstance(required, int):
        return False
    if available < required:
        return False
    memory_blockers = (
        "memory hard-limit",
        "memory headroom",
        "cannot hold",
        "breach the runtime/OS reserve",
        "replicas and reserves",
    )
    return not any(
        any(marker in blocker for marker in memory_blockers)
        for blocker in plan.get("blockers", [])
    )


def _partial_upper_bytes(full_plan, model):
    reserve = full_plan.get("reserve") or {}
    available = int(reserve.get("available_bytes") or 0)
    runtime = int(reserve.get("runtime_bytes") or 0)
    margin = int(reserve.get("os_margin_bytes") or 0)
    upper = max(0, available - runtime - margin)
    return min(int(model["total_shard_bytes"]), upper)


def _minimum_profile_closure(model, counts):
    shard_sizes = {
        item["name"]: int(item["size_bytes"])
        for item in model["shards"]
    }
    costs = []
    for key in counts:
        expert = model["experts"].get(key)
        if expert is None:
            continue
        costs.append(
            sum(shard_sizes[name] for name in set(expert["shards"]))
        )
    return min(costs) if costs else 0


def _blocked_from(plan, message, draft):
    blocked = copy.deepcopy(plan)
    blocked["blockers"] = sorted(
        set(list(blocked.get("blockers", [])) + [message])
    )
    blocked["preset"] = {
        "id": draft.ramdisk_preset,
        "label": draft.ramdisk_preset_label,
        "state": "selected",
        "reason": draft.ramdisk_preset_reason,
        "fallback": draft.ramdisk_preset_fallback,
    }
    return blocked


def _partial_plan(
    draft,
    full_plan,
    hardware,
    model,
    *,
    build_plan,
    load_profile,
):
    upper = _partial_upper_bytes(full_plan, model)
    draft.mode = "partial"
    draft.capacity_gb = max(float(MIB) / GIB, float(upper) / GIB)
    try:
        _profile_path, counts = load_profile(
            getattr(draft, "profile", None),
            model,
        )
    except RamdiskError as exc:
        return draft, _blocked_from(
            full_plan,
            "profile-guided staging is unavailable: %s" % exc,
            draft,
        )
    minimum = _minimum_profile_closure(model, counts)
    if upper < minimum or minimum <= 0:
        return draft, _blocked_from(
            full_plan,
            "no complete profile-guided shard closure fits the safe RAM budget",
            draft,
        )

    # The planner owns reserve arithmetic. Start from its projected staging
    # ceiling, then remove exactly the reported deficit until admitted.
    budget = upper
    last_plan = None
    for _attempt in range(16):
        draft.capacity_gb = max(
            float(MIB) / GIB,
            float(budget + 1023) / GIB,
        )
        try:
            candidate = build_plan(
                draft,
                hardware=hardware,
                model=model,
            )
        except RamdiskError:
            candidate = None
        if candidate is not None:
            last_plan = candidate
            if _memory_admitted(candidate):
                return draft, candidate
            reserve = candidate.get("reserve") or {}
            deficit = max(
                MIB,
                int(reserve.get("total_required_bytes") or 0)
                - int(reserve.get("available_bytes") or 0),
            )
        else:
            deficit = MIB
        budget -= deficit
        if budget < minimum:
            break
    return draft, _blocked_from(
        last_plan or full_plan,
        "no complete profile-guided shard closure fits the safe RAM budget",
        draft,
    )


def _annotate(plan, draft):
    plan = copy.deepcopy(plan)
    plan["preset"] = {
        "id": draft.ramdisk_preset,
        "label": draft.ramdisk_preset_label,
        "state": (
            "custom"
            if draft.ramdisk_preset == "custom"
            else "selected"
        ),
        "reason": draft.ramdisk_preset_reason,
        "fallback": draft.ramdisk_preset_fallback,
    }
    return plan


def mark_preset_custom(args):
    """Mark a selected draft custom without discarding accelerator settings."""
    selected = getattr(args, "ramdisk_preset", None)
    if not selected or selected == "custom":
        return args
    previous = getattr(args, "ramdisk_preset_label", None) or str(selected)
    args.ramdisk_preset = "custom"
    args.ramdisk_preset_label = "Custom"
    args.ramdisk_preset_reason = "Advanced values edited from %s." % previous
    args.ramdisk_preset_fallback = None
    return args


def resolve_preset(
    preset_id,
    args,
    *,
    hardware,
    model,
    build_plan,
    load_profile,
    cuda_capable=None,
):
    """Return populated draft arguments and the authoritative reviewed plan."""
    if preset_id not in _PRESET_LABELS:
        raise RamdiskError("unknown RAM-workspace preset: %s" % preset_id)

    if preset_id == PRESET_REPLICAS:
        draft = _replica_draft(args)
        plan = build_plan(draft, hardware=hardware, model=model)
        return {"args": draft, "plan": _annotate(plan, draft)}

    if preset_id == PRESET_SINGLE:
        draft = _single_draft(args)
        plan = build_plan(draft, hardware=hardware, model=model)
        return {"args": draft, "plan": _annotate(plan, draft)}

    if preset_id == PRESET_GPU_FASTEST:
        devices = list(hardware.get("gpus") or [])
        effective_nodes = set(hardware.get("effective_nodes") or [])
        usable = [
            device
            for device in devices
            if device.get("locality") in ("resolved", "single-node")
            and device.get("numa_node") in effective_nodes
        ]
        fallback_reason = None
        if cuda_capable is False:
            fallback_reason = (
                "the selected engine does not contain the CUDA backend"
            )
        elif cuda_capable is not True:
            fallback_reason = (
                "CUDA engine capability could not be established"
            )
        elif not devices:
            fallback_reason = (
                hardware.get("gpu_discovery", {}).get("error")
                or "no NVIDIA GPU was detected"
            )
        elif len(usable) != len(devices):
            fallback_reason = (
                "one or more GPU PCI devices have no usable NUMA-local node"
            )
        if fallback_reason:
            draft = _single_draft(args, PRESET_GPU_FASTEST)
            draft.ramdisk_preset_fallback = PRESET_SINGLE
            draft.ramdisk_preset_reason = (
                "GPU-aware staging fell back to one ordinary shared copy: %s."
                % fallback_reason
            )
            plan = build_plan(draft, hardware=hardware, model=model)
            plan = _annotate(plan, draft)
            plan["warnings"] = list(plan.get("warnings", [])) + [
                draft.ramdisk_preset_reason
            ]
            return {"args": draft, "plan": plan}
        draft = _gpu_local_draft(
            args,
            hardware,
            usable,
            cuda_capable,
        )
    else:
        draft = _set_common(_namespace(args), PRESET_MINIMAL)
        draft.topology = "interleaved"
        draft.memory_nodes = None
        draft.cpu_list = None
        draft.ramdisk_preset_reason = (
            "Profile-guided staging is sized to the largest safely admitted "
            "shard closure."
        )

    full_draft = _namespace(draft)
    full_draft.mode = "full"
    full_draft.capacity_gb = None
    full_plan = build_plan(
        full_draft,
        hardware=hardware,
        model=model,
    )
    if preset_id == PRESET_GPU_FASTEST and _memory_admitted(full_plan):
        return {"args": draft, "plan": _annotate(full_plan, draft)}

    draft, plan = _partial_plan(
        draft,
        full_plan,
        hardware,
        model,
        build_plan=build_plan,
        load_profile=load_profile,
    )
    return {"args": draft, "plan": _annotate(plan, draft)}
