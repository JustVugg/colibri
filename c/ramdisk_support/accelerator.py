"""Managed accelerator environment contracts for RAM-workspace processes."""

from __future__ import print_function

from .common import RamdiskError


ACCELERATOR_ENVIRONMENT_KEYS = (
    "COLI_CUDA",
    "COLI_GPU",
    "COLI_GPUS",
    "CUDA_EXPERT_GB",
    "CUDA_DENSE",
    "CUDA_RELEASE_HOST",
    "COLI_CUDA_ASYNC",
    "COLI_MMAP",
    "COLI_RAMMAP",
    "PIN",
    "PIN_GB",
    "PIN_FILL",
)


def _managed_accelerator_contract(plan):
    contract = plan.get("managed_accelerator")
    if contract is None:
        return {
            "mode": "cpu",
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
        return {
            "mode": "cpu",
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
    devices = contract.get("devices")
    if not isinstance(devices, list) or not devices:
        raise RamdiskError("managed CUDA plan has no devices")
    normalized = []
    seen = set()
    for device in devices:
        if not isinstance(device, dict):
            raise RamdiskError("managed CUDA device record is malformed")
        index = device.get("index")
        node = device.get("numa_node")
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
        ):
            raise RamdiskError("managed CUDA device identity is invalid")
        seen.add(index)
        normalized.append(
            {
                "index": index,
                "name": str(device.get("name") or ""),
                "pci_bus_id": device["pci_bus_id"],
                "numa_node": node,
            }
        )
    if (
        contract.get("mmap") is not True
        or contract.get("rammap") is not False
        or contract.get("vram_budget") != "auto"
    ):
        raise RamdiskError("managed CUDA memory contract is invalid")
    return {
        "mode": "cuda",
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
            "COLI_MMAP": "0",
            "COLI_RAMMAP": "1",
        }
    indices = [str(device["index"]) for device in contract["devices"]]
    environment = {
        "COLI_CUDA": "1",
        "CUDA_EXPERT_GB": "auto",
        "COLI_CUDA_ASYNC": "1" if contract["async_copy"] else "0",
        "COLI_MMAP": "1",
        "COLI_RAMMAP": "0",
        # The VRAM tier is populated by pin_load. PIN_FILL lets a compatible
        # history fill otherwise-unused VRAM after its measured hot prefix.
        "PIN": "auto",
        "PIN_GB": "all",
        "PIN_FILL": "1",
    }
    if len(indices) == 1:
        environment["COLI_GPU"] = indices[0]
    else:
        environment["COLI_GPUS"] = ",".join(indices)
    return environment


def _apply_managed_accelerator_environment(environment, plan):
    for key in ACCELERATOR_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    applied = _managed_accelerator_environment(plan)
    environment.update(applied)
    return applied
