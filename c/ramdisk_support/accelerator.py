"""Managed accelerator environment contracts for RAM-workspace processes."""

from __future__ import print_function

import re

from .common import GIB, RamdiskError, _format_range_list


GPU_LAYOUT_EXPERTS_ONLY = "experts-only"
GPU_LAYOUT_DENSE_ATTENTION = "dense-attention"
GPU_LAYOUT_DENSE_ATTENTION_SHARDED = "dense-attention-sharded"
GPU_LAYOUT_CHOICES = (
    GPU_LAYOUT_EXPERTS_ONLY,
    GPU_LAYOUT_DENSE_ATTENTION,
    GPU_LAYOUT_DENSE_ATTENTION_SHARDED,
)
GPU_VRAM_RESERVE_BYTES = 2 * GIB

_GPU_LAYOUT_ENVIRONMENT = {
    GPU_LAYOUT_EXPERTS_ONLY: {
        "CUDA_DENSE": "0",
        "COLI_CUDA_ATTN": "0",
        "COLI_CUDA_ATTN_SHARD": "0",
    },
    GPU_LAYOUT_DENSE_ATTENTION: {
        "CUDA_DENSE": "1",
        "COLI_CUDA_ATTN": "1",
        "COLI_CUDA_ATTN_SHARD": "0",
    },
    GPU_LAYOUT_DENSE_ATTENTION_SHARDED: {
        "CUDA_DENSE": "1",
        "COLI_CUDA_ATTN": "1",
        "COLI_CUDA_ATTN_SHARD": "1",
    },
}


ACCELERATOR_ENVIRONMENT_KEYS = (
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "COLI_CUDA",
    "COLI_METAL",
    "COLI_VULKAN",
    "COLI_CUDA_DUAL_PROJ",
    "COLI_CUDA_MTP",
    "COLI_CUDA_PIPE",
    "COLI_CUDA_PIPE_SHARD",
    "COLI_CUDA_PIPE_S_MIN",
    "COLI_CUDA_PROFILE",
    "COLI_CUDA_RESID",
    "COLI_CUDA_ROUTER",
    "COLI_CUDA_SHARED_W4A16",
    "COLI_CUDA_SHARED_W4A16_MIN_ROWS",
    "COLI_CUDA_TC_INT4",
    "COLI_CUDA_TC_MIN_ROWS",
    "COLI_CUDA_TC_W4A16",
    "COLI_CUDA_TC_W4A16_MIN",
    "COLI_CUDA_W4_PACKED",
    "COLI_GPU",
    "COLI_GPUS",
    "COLI_GPU_FAIL_AFTER",
    "CUDA_EXPERT_GB",
    "CUDA_DENSE",
    "CUDA_RAW_EXPERTS",
    "CUDA_RESERVE_GB",
    "COLI_CUDA_ATTN",
    "COLI_CUDA_ATTN_SHARD",
    "CUDA_RELEASE_HOST",
    "COLI_CUDA_ASYNC",
    "COLI_GROUP_ASYNC",
    "COLI_DSA_GATHER",
    "DRAFT",
    "COLI_MMAP",
    "COLI_RAMMAP",
    "PIN",
    "PIN_GB",
    "PIN_FILL",
    "REPIN",
    "REPIN_VERBOSE",
    "SPEC_PIN",
)

ACCELERATOR_ENVIRONMENT_PREFIXES = (
    "COLI_ANS_",
    "COLI_CUDA_",
    "COLI_METAL_",
    "COLI_VK_",
)


def _normalize_gpu_layout(value):
    layout = str(value or GPU_LAYOUT_EXPERTS_ONLY)
    if layout not in GPU_LAYOUT_CHOICES:
        raise RamdiskError(
            "GPU layout must be one of: %s"
            % ", ".join(GPU_LAYOUT_CHOICES)
        )
    return layout


def gpu_device_eligibility(device, hardware):
    """Return ``(eligible, reason)`` for one discovered NVIDIA GPU."""
    if not isinstance(device, dict):
        return False, "GPU discovery returned a malformed device record"
    index = device.get("index")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
    ):
        return False, "GPU discovery returned an invalid device index"
    discovery = hardware.get("gpu_discovery") or {}
    if discovery.get("cuda_visible_devices_present"):
        return False, (
            discovery.get("selection_error")
            or "ambient CUDA_VISIBLE_DEVICES prevents safe GPU selection"
        )
    node = device.get("numa_node")
    if device.get("locality") not in ("resolved", "single-node"):
        if device.get("locality") == "outside-effective-mask":
            return False, "its NUMA node is outside the effective host mask"
        return False, "its NUMA-local node could not be resolved"
    if (
        isinstance(node, bool)
        or not isinstance(node, int)
        or node < 0
    ):
        return False, "its NUMA-local node is invalid"
    effective_nodes = set(hardware.get("effective_nodes") or [])
    if node not in effective_nodes:
        return False, "its NUMA node is outside the effective host mask"
    return True, None


def eligible_gpu_devices(hardware):
    """Return eligible physical NVIDIA devices in stable index order."""
    devices = []
    for device in hardware.get("gpus") or []:
        eligible, _reason = gpu_device_eligibility(device, hardware)
        if eligible:
            devices.append(device)
    return sorted(devices, key=lambda item: int(item["index"]))


def _parse_gpu_selector(selector):
    if isinstance(selector, str):
        value = selector.strip().lower()
        if value in ("auto", "none"):
            return value
        if not value or len(value) > 4096:
            raise RamdiskError(
                "--gpu must be auto, none, or a device list such as 0,1"
            )
        selected = []
        for token in value.split(","):
            token = token.strip()
            match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
            if not match:
                raise RamdiskError(
                    "--gpu must be auto, none, or a device list such as 0,1"
                )
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if end < start:
                raise RamdiskError("--gpu contains a descending range")
            if end > 65535:
                raise RamdiskError("--gpu contains an unreasonable device index")
            selected.extend(range(start, end + 1))
    elif isinstance(selector, (list, tuple, set)):
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in selector
        ):
            raise RamdiskError(
                "--gpu device lists must contain non-negative integers"
            )
        selected = list(selector)
    else:
        raise RamdiskError(
            "--gpu must be auto, none, or a device list such as 0,1"
        )
    selected = sorted(set(selected))
    if not selected:
        raise RamdiskError(
            "--gpu device list is empty; use --gpu none for CPU mode"
        )
    return selected


def _selected_gpu_devices(selector, hardware):
    parsed = _parse_gpu_selector(selector)
    if parsed == "none":
        return []
    eligible = eligible_gpu_devices(hardware)
    if parsed == "auto":
        if not eligible:
            discovery = hardware.get("gpu_discovery") or {}
            detail = (
                discovery.get("selection_error")
                or discovery.get("error")
            )
            raise RamdiskError(
                detail or "no usable NVIDIA GPU was detected"
            )
        return eligible
    discovered = {
        int(device["index"]): device
        for device in hardware.get("gpus") or []
        if isinstance(device, dict)
        and isinstance(device.get("index"), int)
        and not isinstance(device.get("index"), bool)
    }
    devices = []
    for index in parsed:
        device = discovered.get(index)
        if device is None:
            raise RamdiskError(
                "--gpu selects NVIDIA device %d, which was not discovered"
                % index
            )
        eligible_device, reason = gpu_device_eligibility(device, hardware)
        if not eligible_device:
            raise RamdiskError(
                "--gpu selects unusable NVIDIA device %d: %s"
                % (index, reason)
            )
        devices.append(device)
    return devices


def _gpu_local_placement(hardware, devices):
    nodes = sorted(set(int(device["numa_node"]) for device in devices))
    rows = {
        int(row["id"]): row
        for row in hardware.get("nodes") or []
        if isinstance(row, dict)
        and isinstance(row.get("id"), int)
        and not isinstance(row.get("id"), bool)
    }
    effective_cpus = set(hardware.get("effective_cpus") or [])
    cpus = set()
    for node in nodes:
        row = rows.get(node, {})
        node_cpus = (
            row.get("effective_cpus", [])
            if "effective_cpus" in row
            else row.get("cpus", [])
        )
        cpus.update(
            int(cpu)
            for cpu in node_cpus
            if isinstance(cpu, int)
            and not isinstance(cpu, bool)
            and (not effective_cpus or cpu in effective_cpus)
        )
    if not cpus:
        raise RamdiskError(
            "GPU-local NUMA nodes expose no complete effective CPU cores"
        )
    return _format_range_list(nodes), _format_range_list(sorted(cpus))


def apply_gpu_selection(
    args,
    hardware,
    selector=None,
    layout=None,
    *,
    cuda_capable=None,
    reset_placement=True,
):
    """Apply one exact GPU selection and high-level layout to draft arguments."""
    if selector is None:
        selector = getattr(args, "gpu", None)
    if selector is None:
        selector = "auto"
    previous_accelerator = getattr(args, "managed_accelerator", None) or {}
    parsed = _parse_gpu_selector(selector)
    layout = _normalize_gpu_layout(
        layout
        if layout is not None
        else getattr(args, "gpu_layout", None)
    )
    if parsed == "none":
        if layout != GPU_LAYOUT_EXPERTS_ONLY:
            raise RamdiskError(
                "%s requires one or more selected GPUs" % layout
            )
        # Commit only after every validation succeeds.  Frontends keep one
        # mutable draft Namespace, so a rejected edit must leave it untouched.
        args.gpu_layout = layout
        args.gpu = "none"
        args.managed_accelerator = None
        if reset_placement:
            args.memory_nodes = None
            args.cpu_list = None
        return args

    devices = _selected_gpu_devices(parsed, hardware)
    if (
        layout == GPU_LAYOUT_DENSE_ATTENTION_SHARDED
        and len(devices) < 2
    ):
        raise RamdiskError(
            "dense-attention-sharded requires at least two selected GPUs"
        )
    memory_nodes = getattr(args, "memory_nodes", None)
    cpu_list = getattr(args, "cpu_list", None)
    if reset_placement:
        memory_nodes, cpu_list = _gpu_local_placement(
            hardware,
            devices,
        )
    gpu = (
        "auto"
        if parsed == "auto"
        else ",".join(str(device["index"]) for device in devices)
    )
    managed_accelerator = {
        "mode": "cuda",
        "layout": layout,
        "devices": [
            {
                "index": int(device["index"]),
                "cuda_ordinal": ordinal,
                "name": str(device.get("name") or ""),
                "uuid": str(device.get("uuid") or ""),
                "pci_bus_id": str(device.get("pci_bus_id") or ""),
                "numa_node": int(device["numa_node"]),
            }
            for ordinal, device in enumerate(devices)
        ],
        "mmap": True,
        "rammap": False,
        "async_copy": True,
        "vram_budget": "auto",
        "capability": (
            "available"
            if (
                cuda_capable is True
                or (
                    cuda_capable is None
                    and previous_accelerator.get("capability") == "available"
                )
            )
            else "unverified"
        ),
    }
    args.gpu_layout = layout
    if reset_placement:
        args.memory_nodes = memory_nodes
        args.cpu_list = cpu_list
    args.topology = "interleaved"
    args.gpu = gpu
    args.managed_accelerator = managed_accelerator
    return args


def _same_gpu_identity(expected, observed):
    """Compare UUIDs when both are known, otherwise compare PCI identities."""
    expected_uuid = str(expected.get("uuid") or "")
    observed_uuid = str(observed.get("uuid") or "")
    if expected_uuid and observed_uuid:
        return expected_uuid == observed_uuid
    return (
        bool(expected.get("pci_bus_id"))
        and expected.get("pci_bus_id") == observed.get("pci_bus_id")
    )


def _managed_accelerator_contract(plan):
    contract = plan.get("managed_accelerator")
    if contract is None:
        return {
            "mode": "cpu",
            "layout": GPU_LAYOUT_EXPERTS_ONLY,
            "devices": [],
            "mmap": False,
            "rammap": True,
            "async_copy": False,
            "vram_budget": None,
            "capability": "legacy",
        }
    if not isinstance(contract, dict):
        raise RamdiskError("managed accelerator plan is malformed")
    mode = contract.get("mode")
    if mode == "cpu":
        if contract.get("devices") not in (None, []):
            raise RamdiskError("CPU accelerator plan cannot select GPUs")
        if _normalize_gpu_layout(contract.get("layout")) != (
            GPU_LAYOUT_EXPERTS_ONLY
        ):
            raise RamdiskError("CPU accelerator plan cannot use a GPU layout")
        return {
            "mode": "cpu",
            "layout": GPU_LAYOUT_EXPERTS_ONLY,
            "devices": [],
            "mmap": False,
            "rammap": True,
            "async_copy": False,
            "vram_budget": None,
            "capability": str(
                contract.get("capability") or "not-requested"
            ),
        }
    if mode != "cuda":
        raise RamdiskError("managed accelerator mode is invalid")
    layout = _normalize_gpu_layout(contract.get("layout"))
    devices = contract.get("devices")
    if not isinstance(devices, list) or not devices:
        raise RamdiskError("managed CUDA plan has no devices")
    normalized = []
    seen = set()
    ordinals = []
    for device in devices:
        if not isinstance(device, dict):
            raise RamdiskError("managed CUDA device record is malformed")
        index = device.get("index")
        node = device.get("numa_node")
        uuid = device.get("uuid")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index in seen
            or isinstance(node, bool)
            or not isinstance(node, int)
            or node < 0
            or not isinstance(device.get("pci_bus_id"), str)
            or not device["pci_bus_id"]
            or (uuid is not None and not isinstance(uuid, str))
        ):
            raise RamdiskError("managed CUDA device identity is invalid")
        seen.add(index)
        cuda_ordinal = device.get("cuda_ordinal")
        if (
            cuda_ordinal is not None
            and (
                isinstance(cuda_ordinal, bool)
                or not isinstance(cuda_ordinal, int)
                or cuda_ordinal < 0
            )
        ):
            raise RamdiskError("managed CUDA ordinal mapping is invalid")
        ordinals.append(cuda_ordinal)
        normalized.append(
            {
                "index": index,
                **(
                    {"cuda_ordinal": cuda_ordinal}
                    if cuda_ordinal is not None
                    else {}
                ),
                "name": str(device.get("name") or ""),
                "uuid": str(uuid or ""),
                "pci_bus_id": device["pci_bus_id"],
                "numa_node": node,
            }
        )
    if any(ordinal is not None for ordinal in ordinals):
        if (
            any(ordinal is None for ordinal in ordinals)
            or ordinals != list(range(len(normalized)))
            or any(not device["uuid"] for device in normalized)
        ):
            raise RamdiskError("managed CUDA ordinal mapping is invalid")
    if (
        contract.get("mmap") is not True
        or contract.get("rammap") is not False
        or contract.get("vram_budget") != "auto"
    ):
        raise RamdiskError("managed CUDA memory contract is invalid")
    if (
        layout == GPU_LAYOUT_DENSE_ATTENTION_SHARDED
        and len(normalized) < 2
    ):
        raise RamdiskError(
            "dense-attention-sharded requires at least two selected GPUs"
        )
    return {
        "mode": "cuda",
        "layout": layout,
        "devices": normalized,
        "mmap": True,
        "rammap": False,
        "async_copy": bool(contract.get("async_copy", True)),
        "vram_budget": "auto",
        "capability": str(contract.get("capability") or "unverified"),
    }


def _managed_accelerator_environment(plan):
    """Return the exact sanitized accelerator variables for one plan."""
    contract = _managed_accelerator_contract(plan)
    if contract["mode"] == "cpu":
        return {
            "COLI_CUDA": "0",
            "CUDA_DENSE": "0",
            "COLI_CUDA_ATTN": "0",
            "COLI_CUDA_ATTN_SHARD": "0",
            "DRAFT": "0",
            "COLI_MMAP": "0",
            "COLI_RAMMAP": "1",
        }
    # Pin the process to reviewed physical identities. CUDA renumbers this
    # visible list, and the persisted cuda_ordinal field gives telemetry a
    # lossless way to map logical ordinals back to physical cards. A legacy
    # manifest cannot safely assume NVML physical indices are CUDA ordinals.
    logical_mapping = all(
        "cuda_ordinal" in device for device in contract["devices"]
    )
    if not logical_mapping:
        raise RamdiskError(
            "legacy managed CUDA plan has no safe ordinal mapping; "
            "stop, destroy, and prepare the workspace again"
        )
    indices = [
        str(device["cuda_ordinal"])
        for device in contract["devices"]
    ]
    environment = {
        "COLI_CUDA": "1",
        "CUDA_EXPERT_GB": "auto",
        "CUDA_RESERVE_GB": "%.9f" % (
            GPU_VRAM_RESERVE_BYTES / 1e9
        ),
        "COLI_CUDA_ASYNC": "1" if contract["async_copy"] else "0",
        "DRAFT": "0",
        "COLI_MMAP": "1",
        "COLI_RAMMAP": "0",
        # The VRAM tier is populated by pin_load. PIN_FILL lets a compatible
        # history fill otherwise-unused VRAM after its measured hot prefix.
        "PIN": "auto",
        "PIN_GB": "all",
        "PIN_FILL": "1",
        "REPIN": "16",
    }
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        device["uuid"] for device in contract["devices"]
    )
    environment.update(_GPU_LAYOUT_ENVIRONMENT[contract["layout"]])
    if len(indices) == 1:
        environment["COLI_GPU"] = indices[0]
    else:
        environment["COLI_GPUS"] = ",".join(indices)
    return environment


def _apply_managed_accelerator_environment(environment, plan):
    contract = _managed_accelerator_contract(plan)
    if (
        contract["mode"] == "cuda"
        and "CUDA_VISIBLE_DEVICES" in environment
    ):
        raise RamdiskError(
            "ambient CUDA_VISIBLE_DEVICES prevents safe physical GPU "
            "selection; relaunch with it unset"
        )
    for key in tuple(environment):
        if (
            key in ACCELERATOR_ENVIRONMENT_KEYS
            or key.startswith(ACCELERATOR_ENVIRONMENT_PREFIXES)
        ):
            environment.pop(key, None)
    applied = _managed_accelerator_environment(
        {"managed_accelerator": contract}
    )
    environment.update(applied)
    return applied
