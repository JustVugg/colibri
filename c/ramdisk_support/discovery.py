"""Normalized Linux hardware and cgroup discovery."""

from __future__ import print_function

import csv
import io
import os
import posixpath
import re
import subprocess

from .common import RamdiskError, _format_range_list, _parse_range_list
from .platform_ops import get_platform_ops


def _normalize_pci_bus_id(value):
    """Return Linux's canonical PCI domain:bus:device.function spelling."""
    match = re.fullmatch(
        r"\s*([0-9A-Fa-f]{1,8}):([0-9A-Fa-f]{1,2}):"
        r"([0-9A-Fa-f]{1,2})\.([0-7])\s*",
        str(value),
    )
    if not match:
        return None
    domain, bus, device, function = (
        int(item, 16) for item in match.groups()
    )
    if domain > 0xFFFF or bus > 0xFF or device > 0x1F:
        return None
    return "%04x:%02x:%02x.%x" % (
        domain,
        bus,
        device,
        function,
    )


def _discover_gpus(
    effective_nodes,
    *,
    ops=None,
    run=None,
    environ=None,
):
    """Discover NVIDIA devices and resolve their PCI-local Linux NUMA nodes."""
    ops = get_platform_ops() if ops is None else ops
    environ = os.environ if environ is None else environ
    visibility_present = "CUDA_VISIBLE_DEVICES" in environ
    visibility_value = (
        str(environ.get("CUDA_VISIBLE_DEVICES") or "")
        if visibility_present
        else None
    )
    visibility = {
        "cuda_visible_devices_present": visibility_present,
        "cuda_visible_devices": visibility_value,
        "selection_error": (
            "ambient CUDA_VISIBLE_DEVICES prevents safe physical GPU "
            "selection; relaunch the RAM TUI with it unset"
            if visibility_present
            else None
        ),
    }
    if not ops.is_linux:
        return {
            "status": "unsupported",
            "error": "GPU NUMA discovery is supported only on Linux",
            "devices": [],
            **visibility,
        }
    run = subprocess.run if run is None else run
    executable = ops.executable_path("nvidia-smi")
    if not executable:
        return {
            "status": "unavailable",
            "error": "nvidia-smi is not available",
            "devices": [],
            **visibility,
        }
    command = [
        executable,
        "--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "unavailable",
            "error": "cannot query NVIDIA GPUs: %s" % exc,
            "devices": [],
            **visibility,
        }
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        return {
            "status": "unavailable",
            "error": (
                "nvidia-smi GPU query failed"
                + (": %s" % detail if detail else "")
            ),
            "devices": [],
            **visibility,
        }

    allowed = sorted(set(int(node) for node in effective_nodes))
    devices = []
    malformed = 0
    seen_indices = set()
    seen_uuids = set()
    for fields in csv.reader(io.StringIO(result.stdout or "")):
        fields = [field.strip() for field in fields]
        if len(fields) != 6:
            malformed += 1
            continue
        try:
            index = int(fields[0])
            total_mib = int(fields[4])
            free_mib = int(fields[5])
        except ValueError:
            malformed += 1
            continue
        uuid = fields[2]
        pci_bus_id = _normalize_pci_bus_id(fields[3])
        if (
            index < 0
            or index in seen_indices
            or total_mib < 0
            or free_mib < 0
            or free_mib > total_mib
            or not uuid
            or uuid.lower() == "n/a"
            or uuid in seen_uuids
            or pci_bus_id is None
        ):
            malformed += 1
            continue
        seen_indices.add(index)
        seen_uuids.add(uuid)
        raw_node = ops.read_text(
            "/sys/bus/pci/devices/%s/numa_node" % pci_bus_id,
            "",
        ).strip()
        try:
            numa_node = int(raw_node)
        except ValueError:
            numa_node = None
        if numa_node == -1 and len(allowed) == 1:
            numa_node = allowed[0]
            locality = "single-node"
        elif numa_node is None or numa_node < 0:
            numa_node = None
            locality = "unknown"
        elif numa_node not in allowed:
            locality = "outside-effective-mask"
        else:
            locality = "resolved"
        devices.append(
            {
                "index": index,
                "name": fields[1],
                "uuid": uuid,
                "pci_bus_id": pci_bus_id,
                "numa_node": numa_node,
                "locality": locality,
                "total_bytes": total_mib * 1024 * 1024,
                "free_bytes": free_mib * 1024 * 1024,
            }
        )
    devices.sort(key=lambda item: item["index"])
    if devices:
        status = "available" if not malformed else "partial"
        error = (
            None
            if not malformed
            else "%d malformed nvidia-smi row(s) were ignored" % malformed
        )
    elif malformed:
        status = "unavailable"
        error = "nvidia-smi returned no valid GPU rows"
    else:
        status = "none"
        error = None
    return {
        "status": status,
        "error": error,
        "devices": devices,
        **visibility,
    }


def _mountinfo_unescape(value):
    """Decode the octal escapes used for paths in ``/proc/*/mountinfo``."""
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _cgroup_mounts(mountinfo):
    """Return normalized cgroup mount records from one mountinfo snapshot."""
    records = []
    for line in mountinfo.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
            root = _mountinfo_unescape(fields[3])
            mountpoint = _mountinfo_unescape(fields[4])
            mount_options = fields[5].split(",")
            filesystem = fields[separator + 1]
            source = fields[separator + 2]
            super_options = fields[separator + 3].split(",")
        except (IndexError, ValueError):
            continue
        if filesystem not in ("cgroup", "cgroup2"):
            continue
        records.append(
            {
                "filesystem": filesystem,
                "root": posixpath.normpath(root),
                "mountpoint": posixpath.normpath(mountpoint),
                "source": source,
                "mount_options": mount_options,
                "optional_fields": fields[6:separator],
                "super_options": super_options,
            }
        )
    return records


def _cgroup_memberships(cgroup_text):
    """Parse v1 controller paths and the v2 unified path."""
    memberships = {"v1": {}, "v2": None}
    for line in cgroup_text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, path = fields
        if not path.startswith("/"):
            continue
        normalized = posixpath.normpath(path)
        if not controllers:
            memberships["v2"] = normalized
        else:
            for controller in controllers.split(","):
                if controller:
                    memberships["v1"][controller] = normalized
    return memberships


def _resolve_cgroup_directory(membership, mounts, filesystem, controller=None):
    """Map a membership path to the most-specific visible cgroup mount."""
    candidates = []
    for mount in mounts:
        if mount["filesystem"] != filesystem:
            continue
        controller_options = set(mount["super_options"]) | set(
            mount["mount_options"]
        ) | set(mount["source"].split(",")) | set(mount["optional_fields"])
        if controller is not None and controller not in controller_options:
            continue
        root = mount["root"]
        explicit_root = membership == root or membership.startswith(
            root.rstrip("/") + "/"
        )
        relative = (
            posixpath.relpath(membership, root)
            if explicit_root
            else membership.lstrip("/") or "."
        )
        resolved = posixpath.normpath(
            posixpath.join(mount["mountpoint"], relative)
        )
        try:
            contained = (
                posixpath.commonpath([resolved, mount["mountpoint"]])
                == mount["mountpoint"]
            )
        except ValueError:
            contained = False
        if contained:
            candidates.append((int(explicit_root), len(root), mount, resolved))
    if not candidates:
        return None, None
    _, _, mount, resolved = max(candidates, key=lambda item: item[:2])
    return mount, resolved


def _cgroup_ancestors(path, mountpoint):
    """Yield a cgroup and every visible ancestor through its mount root."""
    current = posixpath.normpath(path)
    root = posixpath.normpath(mountpoint)
    while True:
        try:
            if posixpath.commonpath([current, root]) != root:
                raise ValueError("cgroup path escaped its mount")
        except ValueError:
            raise RamdiskError("resolved cgroup path is outside its controller mount")
        yield current
        if current == root:
            break
        parent = posixpath.dirname(current)
        if parent == current:
            raise RamdiskError("cgroup ancestry did not reach its controller mount")
        current = parent


def _parse_cgroup_bytes(value, path, unlimited_word=False, v1_unlimited=False):
    if value is None:
        return None
    if unlimited_word and value == "max":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise RamdiskError("invalid cgroup memory value in %s" % path)
    if parsed < 0:
        raise RamdiskError("negative cgroup memory value in %s" % path)
    if v1_unlimited and parsed >= (1 << 60):
        return None
    return parsed


def _empty_cgroup_memory():
    return {
        "version": None,
        "status": "none",
        "path": None,
        "mountpoint": None,
        "limit_bytes": None,
        "current_bytes": None,
        "available_bytes": None,
        "limiting_path": None,
        "high_bytes": None,
        "high_available_bytes": None,
        "high_limiting_path": None,
        "error": None,
    }


def _discover_cgroup_memory_with_ops(
    cgroup_text=None,
    mountinfo_text=None,
    ops=None,
):
    """Return hard/high headroom across every limiting cgroup ancestor."""
    ops = get_platform_ops() if ops is None else ops
    result = _empty_cgroup_memory()
    if not ops.is_linux:
        if cgroup_text is None or mountinfo_text is None:
            return result
        # Synthetic contracts are portable parser inputs.  Use only the
        # import-safe file helpers needed to inspect their fixture hierarchy.
        from .linux_ops import LinuxPlatformOps

        ops = LinuxPlatformOps(ops.platform_name)
    try:
        if cgroup_text is None:
            cgroup_text = ops.read_cgroup_contract("/proc/self/cgroup")
        if mountinfo_text is None:
            mountinfo_text = ops.read_cgroup_contract("/proc/self/mountinfo")
    except RamdiskError as exc:
        result.update({"status": "unavailable", "error": str(exc)})
        return result
    memberships = _cgroup_memberships(cgroup_text)
    mounts = _cgroup_mounts(mountinfo_text)
    version = None
    membership = None
    mount = resolved = None
    if memberships["v2"] is not None:
        version = 2
        membership = memberships["v2"]
        mount, resolved = _resolve_cgroup_directory(
            membership, mounts, "cgroup2"
        )
        v2_memory_visible = (
            mount is not None
            and resolved is not None
            and any(
                ops.path_exists(posixpath.join(ancestor, leaf))
                for ancestor in _cgroup_ancestors(resolved, mount["mountpoint"])
                for leaf in ("memory.current", "memory.max", "memory.high")
            )
        )
        if "memory" in memberships["v1"] and not v2_memory_visible:
            v1_membership = memberships["v1"]["memory"]
            v1_mount, v1_resolved = _resolve_cgroup_directory(
                v1_membership, mounts, "cgroup", controller="memory"
            )
            if v1_mount is not None and v1_resolved is not None:
                version = 1
                membership = v1_membership
                mount, resolved = v1_mount, v1_resolved
    elif "memory" in memberships["v1"]:
        version = 1
        membership = memberships["v1"]["memory"]
        mount, resolved = _resolve_cgroup_directory(
            membership, mounts, "cgroup", controller="memory"
        )
    if version is None:
        return result
    result.update({"version": version, "path": membership})
    if mount is None or resolved is None:
        result.update(
            {
                "status": "unavailable",
                "error": "memory cgroup membership has no visible controller mount",
            }
        )
        return result
    result["mountpoint"] = mount["mountpoint"]
    try:
        for ancestor in _cgroup_ancestors(resolved, mount["mountpoint"]):
            if version == 2:
                limit_path = posixpath.join(ancestor, "memory.max")
                current_path = posixpath.join(ancestor, "memory.current")
                high_path = posixpath.join(ancestor, "memory.high")
                limit = _parse_cgroup_bytes(
                    ops.read_cgroup_value(limit_path),
                    limit_path,
                    unlimited_word=True,
                )
                high = _parse_cgroup_bytes(
                    ops.read_cgroup_value(high_path),
                    high_path,
                    unlimited_word=True,
                )
            else:
                limit_path = posixpath.join(
                    ancestor,
                    "memory.limit_in_bytes",
                )
                current_path = posixpath.join(
                    ancestor,
                    "memory.usage_in_bytes",
                )
                limit = _parse_cgroup_bytes(
                    ops.read_cgroup_value(limit_path),
                    limit_path,
                    v1_unlimited=True,
                )
                high = None
            if limit is None and high is None:
                continue
            current = _parse_cgroup_bytes(
                ops.read_cgroup_value(current_path),
                current_path,
            )
            if current is None:
                raise RamdiskError(
                    "cgroup memory limit is visible but usage is unavailable at %s"
                    % ancestor
                )
            if limit is not None:
                available = max(0, limit - current)
                if (
                    result["available_bytes"] is None
                    or available < result["available_bytes"]
                ):
                    result.update(
                        {
                            "limit_bytes": limit,
                            "current_bytes": current,
                            "available_bytes": available,
                            "limiting_path": ancestor,
                        }
                    )
            if high is not None:
                high_available = max(0, high - current)
                if (
                    result["high_available_bytes"] is None
                    or high_available < result["high_available_bytes"]
                ):
                    result.update(
                        {
                            "high_bytes": high,
                            "high_available_bytes": high_available,
                            "high_limiting_path": ancestor,
                        }
                    )
        result["status"] = (
            "limited"
            if result["available_bytes"] is not None
            or result["high_available_bytes"] is not None
            else "unlimited"
        )
    except RamdiskError as exc:
        result.update({"status": "unavailable", "error": str(exc)})
    return result


def _discover_cgroup_memory(cgroup_text=None, mountinfo_text=None):
    return _discover_cgroup_memory_with_ops(
        cgroup_text=cgroup_text,
        mountinfo_text=mountinfo_text,
    )


def _unsupported_hardware(ops):
    cpus = list(range(ops.cpu_count()))
    cpu_list = _format_range_list(cpus)
    return {
        "linux": False,
        "capabilities": ops.capabilities(),
        "kernel_release": ops.kernel_release(),
        "online_nodes": [0],
        "effective_nodes": [0],
        "effective_cpus": cpus,
        "effective_cpu_list": cpu_list,
        "effective_mask_source": "portable-fallback",
        "core_groups": [[cpu] for cpu in cpus],
        "nodes": [
            {
                "id": 0,
                "cpus": cpus,
                "cpu_list": cpu_list,
                "physical_cores": len(cpus),
                "memory_total_bytes": 0,
                "memory_available_bytes": 0,
                "distance": [],
                "effective_cpus": cpus,
                "effective_cpu_list": cpu_list,
            }
        ],
        "physical_cores": len(cpus),
        "effective_physical_cores": len(cpus),
        "memory": {"total_bytes": 0, "available_bytes": 0},
        "cgroup_memory": _empty_cgroup_memory(),
        "swap": {"configured": [], "used_bytes": 0},
        "tmpfs": {"supported": False, "noswap_supported": False},
        "thp": {
            "shmem_enabled": "",
            "modes": [],
            "within_size_supported": False,
            "advise_supported": False,
        },
        "numactl": None,
        "gpus": [],
        "gpu_discovery": {
            "status": "unsupported",
            "error": "GPU NUMA discovery is supported only on Linux",
        },
        "mount": None,
        "umount": None,
        "sudo": None,
        "hugetlb": {
            "total_pages": 0,
            "free_pages": 0,
            "page_size_bytes": 0,
        },
    }


def discover_hardware(ops=None, gpu_discovery=None):
    """Return normalized Linux discovery or explicit unsupported capabilities."""
    ops = get_platform_ops() if ops is None else ops
    if not ops.is_linux:
        return _unsupported_hardware(ops)
    online_text = ops.read_text("/sys/devices/system/node/online", "0")
    try:
        online = _parse_range_list(online_text)
    except ValueError:
        online = [0]
    if not online:
        online = [0]
    nodes = []
    all_cpus = []
    for node in online:
        cpus_text = ops.read_text(
            "/sys/devices/system/node/node%d/cpulist" % node,
            ops.read_text("/sys/devices/system/cpu/online", "0"),
        )
        try:
            cpus = _parse_range_list(cpus_text)
        except ValueError:
            cpus = []
        all_cpus.extend(cpus)
        memory = ops.node_meminfo(node)
        distance = []
        for word in ops.read_text(
            "/sys/devices/system/node/node%d/distance" % node
        ).split():
            try:
                distance.append(int(word))
            except ValueError:
                pass
        nodes.append(
            {
                "id": node,
                "cpus": cpus,
                "cpu_list": cpus_text.strip(),
                "physical_cores": ops.physical_cores(cpus),
                "memory_total_bytes": memory.get("MemTotal", 0),
                "memory_available_bytes": memory.get(
                    "MemFree", memory.get("MemAvailable", 0)
                ),
                "distance": distance,
            }
        )
    all_cpus = sorted(set(all_cpus))
    affinity = ops.cpu_affinity()
    if affinity is None:
        affinity = ops.status_allowed_list("Cpus_allowed_list", all_cpus)
    effective_cpus = sorted(set(affinity) & set(all_cpus))
    effective_nodes = sorted(
        set(ops.status_allowed_list("Mems_allowed_list", online)) & set(online)
    )
    core_groups = ops.thread_sibling_groups(effective_cpus)
    for node in nodes:
        node["effective_cpus"] = sorted(
            set(node["cpus"]) & set(effective_cpus)
        )
        node["effective_cpu_list"] = _format_range_list(node["effective_cpus"])
    memory = ops.meminfo()
    swaps = []
    swap_text = ops.read_text("/proc/swaps")
    for line in swap_text.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 5:
            swaps.append(
                {
                    "path": fields[0],
                    "type": fields[1],
                    "size_bytes": int(fields[2]) * 1024,
                    "used_bytes": int(fields[3]) * 1024,
                }
            )
    shmem_enabled = ops.read_text(
        "/sys/kernel/mm/transparent_hugepage/shmem_enabled"
    ).strip()
    thp_modes = re.findall(r"\[?([A-Za-z_]+)\]?", shmem_enabled)
    filesystems = ops.read_text("/proc/filesystems")
    cgroup_memory = _discover_cgroup_memory_with_ops(ops=ops)
    gpu_report = (
        _discover_gpus(effective_nodes, ops=ops)
        if gpu_discovery is None
        else gpu_discovery(effective_nodes, ops=ops)
    )
    return {
        "linux": True,
        "capabilities": ops.capabilities(),
        "kernel_release": ops.kernel_release(),
        "online_nodes": online,
        "effective_nodes": effective_nodes,
        "effective_cpus": effective_cpus,
        "effective_cpu_list": _format_range_list(effective_cpus),
        "effective_mask_source": "kernel-task-status",
        "core_groups": core_groups,
        "nodes": nodes,
        "physical_cores": ops.physical_cores(all_cpus),
        "effective_physical_cores": len(core_groups),
        "memory": {
            "total_bytes": memory.get("MemTotal", 0),
            "available_bytes": memory.get(
                "MemAvailable",
                memory.get("MemFree", 0),
            ),
        },
        "cgroup_memory": cgroup_memory,
        "swap": {
            "configured": swaps,
            "used_bytes": sum(item["used_bytes"] for item in swaps),
        },
        "tmpfs": {
            "supported": any(
                line.strip().endswith("tmpfs")
                for line in filesystems.splitlines()
            ),
            "noswap_supported": ops.kernel_at_least(6, 4),
        },
        "thp": {
            "shmem_enabled": shmem_enabled,
            "modes": sorted(set(thp_modes)),
            "within_size_supported": "within_size" in thp_modes,
            "advise_supported": "advise" in thp_modes or bool(shmem_enabled),
        },
        "numactl": ops.executable_path("numactl"),
        "gpus": list(gpu_report.get("devices") or []),
        "gpu_discovery": {
            "status": gpu_report.get("status", "unavailable"),
            "error": gpu_report.get("error"),
            "cuda_visible_devices_present": bool(
                gpu_report.get("cuda_visible_devices_present")
            ),
            "cuda_visible_devices": gpu_report.get(
                "cuda_visible_devices"
            ),
            "selection_error": gpu_report.get("selection_error"),
        },
        "mount": ops.executable_path("mount"),
        "umount": ops.executable_path("umount"),
        "sudo": ops.executable_path("sudo"),
        "hugetlb": {
            "total_pages": memory.get("HugePages_Total", 0) // 1024,
            "free_pages": memory.get("HugePages_Free", 0) // 1024,
            "page_size_bytes": memory.get("Hugepagesize", 0),
        },
    }
