#!/usr/bin/env python3
"""Hardware and model placement planning for colibri's disk/RAM/VRAM tiers."""

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


GB = 1_000_000_000
EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")


def _tensor_sizes(path):
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"short safetensors header: {path}")
        length = int.from_bytes(raw, "little")
        if length < 2 or length > file_size - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(stream.read(length))
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        if not 0 <= start <= end <= file_size - 8 - length:
            raise ValueError(f"invalid tensor offsets for {name}: {path}")
        yield name, end - start


def analyze_model(model):
    model = Path(model).resolve()
    config_path = model / "config.json"
    if not config_path.is_file():
        raise ValueError(f"missing config.json: {model}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no safetensors shards: {model}")

    dense_bytes = 0
    expert_groups = {}
    for shard in shards:
        for name, size in _tensor_sizes(shard):
            match = EXPERT_RE.search(name)
            if match:
                key = tuple(map(int, match.groups()))
                expert_groups[key] = expert_groups.get(key, 0) + size
            else:
                dense_bytes += size

    layer_sizes = {}
    for (layer, _), size in expert_groups.items():
        layer_sizes.setdefault(layer, []).append(size)
    per_layer = {layer: int(statistics.median(sizes)) for layer, sizes in layer_sizes.items()}
    per_cap_bytes = sum(per_layer.values())
    typical_expert_bytes = int(statistics.median(per_layer.values())) if per_layer else 0
    model_bytes = sum(shard.stat().st_size for shard in shards)
    return {
        "path": str(model),
        "shards": len(shards),
        "model_bytes": model_bytes,
        "dense_bytes": dense_bytes,
        "expert_bytes": sum(expert_groups.values()),
        "expert_count": len(expert_groups),
        "expert_layers": len(per_layer),
        "typical_expert_bytes": typical_expert_bytes,
        "per_cap_bytes": per_cap_bytes,
        "config": config,
    }


def memory_available():
    # Linux (and MSYS2/Git-Bash CPython where /proc exists): MemAvailable.
    try:
        text = Path("/proc/meminfo").read_text()
        return int(re.search(r"MemAvailable:\s+(\d+)", text).group(1)) * 1024
    except (OSError, AttributeError):
        pass
    # Windows native CPython: GlobalMemoryStatusEx -> ullAvailPhys.
    # Same definition the C engine uses (compat_meminfo in compat.h):
    # standby/free/zero pages, i.e. reclaimable without swapping.
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
            kernel32 = ctypes.windll.kernel32
            kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]
            kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)) and stat.ullAvailPhys:
                return stat.ullAvailPhys
            # Fallback (e.g. sandboxed callers where GlobalMemoryStatusEx reports
            # nothing): total installed RAM in KB. Less precise than ullAvailPhys
            # — it ignores standby/reclaimable pages — but never returns 0 on a
            # real machine, which keeps the expert cache from being mis-sized.
            total_kb = ctypes.c_ulonglong(0)
            kernel32.GetPhysicallyInstalledSystemMemory.argtypes = [ctypes.c_void_p]
            kernel32.GetPhysicallyInstalledSystemMemory.restype = ctypes.c_int
            if kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(total_kb)):
                return total_kb.value * 1024
        except OSError:
            pass
    # macOS: no /proc and not win32. Sum the reclaimable pages reported by vm_stat
    # (free + inactive + speculative + purgeable) — the same "reclaimable without swapping"
    # definition the C engine's compat_meminfo uses. Fall back to total RAM (never 0 on a Mac).
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["vm_stat"], text=True, capture_output=True, timeout=5).stdout
            page_match = re.search(r"page size of (\d+) bytes", out)
            page = int(page_match.group(1)) if page_match else os.sysconf("SC_PAGE_SIZE")
            pages = 0
            for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                match = re.search(rf"{key}:\s+(\d+)\.", out)
                if match:
                    pages += int(match.group(1))
            if pages:
                return pages * page
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        try:
            total = subprocess.run(["sysctl", "-n", "hw.memsize"], text=True,
                                   capture_output=True, timeout=5).stdout.strip()
            if total:
                return int(total)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return 0


def discover_gpus():
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    import csv
    for fields in csv.reader(result.stdout.splitlines()):
        fields = [f.strip() for f in fields]
        if len(fields) != 4:
            continue
        try:
            index, total, free = int(fields[0]), int(fields[2]), int(fields[3])
        except ValueError:
            continue
        devices.append({"index": index, "name": fields[1],
                        "total_bytes": total * 1024 * 1024,
                        "free_bytes": free * 1024 * 1024})
    return devices


def physical_cpu_count():
    if sys.platform == "win32":
        # os.cpu_count() conta i processori logici (SMT): 2 thread/core saturano
        # le unita' AVX-512 e peggiorano il matmul. Contiamo i core fisici veri
        # con GetLogicalProcessorInformationEx(RelationProcessorCore).
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            need = ctypes.c_ulong(0)
            k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(need))
            buf = (ctypes.c_char * need.value)()
            if k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(need)):
                raw, cores, off = bytes(buf), 0, 0
                while off + 8 <= need.value:
                    relationship = int.from_bytes(raw[off:off + 4], "little")
                    size = int.from_bytes(raw[off + 4:off + 8], "little")
                    if size <= 0:
                        break
                    if relationship == 0:  # RelationProcessorCore
                        cores += 1
                    off += size
                if cores:
                    return cores
        except (OSError, ValueError, AttributeError):
            pass
    try:
        result = subprocess.run(["lscpu", "-p=core,socket"], text=True,
                                capture_output=True, check=True, timeout=5)
        cores = {tuple(map(int, line.split(","))) for line in result.stdout.splitlines()
                 if line and not line.startswith("#")}
        if cores:
            return len(cores)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return os.cpu_count() or 1


POLICIES = {
    "quality": {"preserve_quantization": True, "preserve_router": True},
    "balanced": {"preserve_quantization": True, "preserve_router": True},
    "experimental-fast": {"preserve_quantization": False, "preserve_router": False},
}


def build_plan(model, ram_gb=0, context=4096, gpu_indices=None, vram_gb=0,
               available_memory=None, available_disk=None, gpus=None,
               policy="quality", physical_cpus=None):
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    info = analyze_model(model)
    physical_cpus = physical_cpu_count() if physical_cpus is None else physical_cpus
    cfg = info["config"]
    available_memory = memory_available() if available_memory is None else available_memory
    if available_disk is None:
        try:
            usage = shutil.disk_usage(info["path"])
            available_disk = usage.free
        except OSError:
            available_disk = 500 * GB
    gpus = discover_gpus() if gpus is None else gpus
    if gpu_indices is not None:
        wanted = set(gpu_indices)
        gpus = [gpu for gpu in gpus if gpu["index"] in wanted]

    ram_budget = int(ram_gb * GB) if ram_gb > 0 else int(available_memory * 0.88)
    if ram_budget < 4 * GB:
        ram_budget = 8 * GB
    typical = info["typical_expert_bytes"]
    layers = int(cfg.get("num_hidden_layers") or 0) + 1
    kv_bytes = layers * context * (int(cfg.get("kv_lora_rank") or 0) +
                                   int(cfg.get("qk_rope_head_dim") or 0)) * 4
    kv_buffer = context * int(cfg.get("num_attention_heads") or 0) * (
        int(cfg.get("qk_nope_head_dim") or 0) + int(cfg.get("v_head_dim") or 0)) * 4
    runtime_bytes = int(1.2 * GB + 2.5 * GB + 64 * typical + kv_bytes + kv_buffer)
    cache_bytes = max(0, ram_budget - info["dense_bytes"] - runtime_bytes)
    per_cap = info["per_cap_bytes"]
    configured_experts = int(cfg.get("n_routed_experts") or 0)
    cap = int(cache_bytes // per_cap) if per_cap else 0
    if configured_experts:
        cap = min(cap, configured_experts)

    reserve = 2 * GB
    gpu_plan = []
    safe_vram = 0
    for gpu in gpus:
        usable = max(0, gpu["free_bytes"] - reserve)
        safe_vram += usable
        gpu_plan.append(dict(gpu, reserve_bytes=reserve, usable_bytes=usable))
    requested_vram = int(vram_gb * GB) if vram_gb > 0 else safe_vram
    # VRAM-resident experts do not need duplicate RAM backing: the checkpoint is
    # their recovery source. RAM is therefore an independent warm compute tier.
    vram_budget = min(requested_vram, safe_vram, info["expert_bytes"])
    vram_experts = int(vram_budget // typical) if typical else 0
    hot_bytes = min(info["expert_bytes"], vram_experts * typical)
    warm_bytes = min(max(0, info["expert_bytes"] - hot_bytes), cache_bytes)
    cold_bytes = max(0, info["expert_bytes"] - hot_bytes - warm_bytes)

    warnings = []
    if cap < 1:
        warnings.append("RAM budget cannot hold one expert slot per sparse layer")
    if gpu_indices is not None and len(gpus) != len(set(gpu_indices)):
        warnings.append("one or more requested GPUs were not detected")
    if gpus and vram_budget < requested_vram:
        warnings.append("VRAM tier was clamped by free VRAM or model expert size")
    if cold_bytes:
        warnings.append("cold expert misses may reach disk; normal decode speed depends on hit rate")

    if cold_bytes:
        bottleneck = "disk expert misses"
    elif warm_bytes:
        bottleneck = "CPU expert compute and RAM bandwidth"
    else:
        bottleneck = "GPU compute and interconnect"

    return {
        "version": 2,
        "policy": {"name": policy, **POLICIES[policy],
                   "quality_preserving": policy != "experimental-fast"},
        "model": {key: value for key, value in info.items() if key != "config"},
        "cpu": {"physical_cores": max(1, int(physical_cpus)),
                "thread_policy": "physical-cores"},
        "tiers": {
            "disk": {"role": "cold-backing", "model_bytes": info["model_bytes"],
                     "available_bytes": available_disk, "cold_expert_bytes": cold_bytes},
            "ram": {"role": "resident+warm-experts", "available_bytes": available_memory,
                    "budget_bytes": ram_budget, "dense_bytes": info["dense_bytes"],
                    "runtime_bytes": runtime_bytes, "expert_cache_bytes": cache_bytes,
                    "warm_expert_bytes": warm_bytes, "cache_slots_per_layer": cap},
            "vram": {"role": "hot-experts", "devices": gpu_plan,
                     "budget_bytes": vram_budget, "hot_expert_bytes": hot_bytes,
                     "expert_capacity": vram_experts, "requires_host_backing": False},
        },
        "expected_bottleneck": bottleneck,
        "decisions": [
            {"target": "VRAM", "reason": "profile-ranked hot experts"},
            {"target": "RAM", "reason": "warm experts execute on CPU without quality loss"},
            {"target": "Disk", "reason": "immutable recovery source for cold experts"},
        ],
        "warnings": warnings,
    }


def environment_for_plan(plan, env=None, cuda_enabled=True):
    """Apply a plan without overriding explicit user environment settings."""
    result = dict(env or {})
    result.setdefault("COLI_POLICY", plan["policy"]["name"])
    result.setdefault("OMP_NUM_THREADS", str(plan["cpu"]["physical_cores"]))
    if sys.platform != "win32":
        # la libgomp di MinGW non supporta l'affinity su Windows
        # ("Affinity not supported on this configuration"): non impostarle li'.
        result.setdefault("OMP_PROC_BIND", "spread")
        result.setdefault("OMP_PLACES", "cores")
    if plan["policy"]["name"] == "balanced":
        result.setdefault("REPIN", "64")
    ram = plan["tiers"]["ram"]
    result.setdefault("RAM_GB", f"{ram['budget_bytes'] / GB:.3f}")

    vram = plan["tiers"]["vram"]
    devices = [device["index"] for device in vram["devices"]]
    if not cuda_enabled or not devices or vram["budget_bytes"] <= 0:
        return result
    if result.get("COLI_CUDA", "1") == "0":
        return result

    result.setdefault("COLI_CUDA", "1")
    if "COLI_GPU" not in result and "COLI_GPUS" not in result:
        key = "COLI_GPU" if len(devices) == 1 else "COLI_GPUS"
        result[key] = ",".join(map(str, devices))
    result.setdefault("CUDA_EXPERT_GB", f"{vram['budget_bytes'] / GB:.3f}")
    if result.get("PIN"):
        result.setdefault("PIN_GB", f"{vram['budget_bytes'] / GB:.3f}")
    return result


def format_bytes(value):
    return f"{value / GB:.1f} GB"


def format_plan(plan):
    model, tiers = plan["model"], plan["tiers"]
    policy=plan["policy"]
    lines = [f"policy {policy['name']} · quality-preserving {'yes' if policy['quality_preserving'] else 'no'}",
             f"model  {model['shards']} shards · {format_bytes(model['model_bytes'])}",
             f"disk   {format_bytes(tiers['disk']['cold_expert_bytes'])} cold experts · "
             f"{format_bytes(tiers['disk']['available_bytes'])} free",
             f"RAM    {format_bytes(tiers['ram']['budget_bytes'])} budget · "
             f"{format_bytes(tiers['ram']['dense_bytes'])} dense · "
             f"{format_bytes(tiers['ram']['runtime_bytes'])} runtime · "
             f"{format_bytes(tiers['ram']['warm_expert_bytes'])} warm experts · "
             f"cap {tiers['ram']['cache_slots_per_layer']}/layer"]
    vram = tiers["vram"]
    if vram["devices"]:
        names = ", ".join(f"{gpu['index']}:{gpu['name']}" for gpu in vram["devices"])
        lines.append(f"VRAM   {format_bytes(vram['budget_bytes'])} hot tier · "
                     f"~{vram['expert_capacity']} experts · {names}")
    else:
        lines.append("VRAM   no NVIDIA device detected · CPU path")
    lines.append(f"limit  {plan['expected_bottleneck']}")
    lines.extend(f"warn   {warning}" for warning in plan["warnings"])
    return "\n".join(lines)
