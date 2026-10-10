#!/usr/bin/env python3
"""Hardware detection for the one-step setup (`coli setup`, the MCP server).

What it answers, and nothing more: how much RAM there is, how much disk is free
where the model would go, which CPU features the engine can use, and which GPUs
are there (Vulkan devices with their type and memory, NVIDIA cards through
nvidia-smi with their compute capability). It never changes anything on the
machine.

Every probe is split in two: a function that runs a command or a system call,
and a pure parser that turns its output into data. The parsers are what the
tests exercise with mocked hardware (tests/test_setup_hw.py); the probes are
thin and fail closed (a probe that cannot run reports nothing, never a guess).

Vulkan is probed through the loader itself with ctypes, in a child process: the
same vkEnumeratePhysicalDevices the engine calls, the same device ranking
(discrete > integrated > virtual > other > CPU), and the device-local heap
budget from VK_EXT_memory_budget when the driver has it. A child process,
because a broken driver can crash the process that loads it. When the loader
cannot be loaded, `vulkaninfo --summary` is parsed instead (names and types,
no memory figures).
"""
import ctypes
import glob
import json
import os
import platform
import posixpath
import re
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
GB = 1_000_000_000


def host_os():
    """The OS whose conventions decide which probes apply (WSL driver search,
    no Vulkan on macOS). Patchable, so tests model any OS on any host."""
    return sys.platform

# ---------------------------------------------------------------- memory


def memory_from_meminfo(text):
    """{"total": bytes, "available": bytes} from /proc/meminfo text.

    Missing lines come back as None rather than 0: zero would read as a
    measurement ("no RAM free") when it is only an absence."""
    out = {"total": None, "available": None}
    for key, field in (("MemTotal", "total"), ("MemAvailable", "available")):
        match = re.search(rf"^{key}:\s+(\d+)\s+kB\s*$", text or "", re.M)
        if match:
            out[field] = int(match.group(1)) * 1024
    return out


def memory_from_windows_status(total_phys, avail_phys, avail_pagefile=None):
    """GlobalMemoryStatusEx figures. Available is the smaller of free physical
    memory and the commit Windows can still grant (resource_plan's rule, #1375)."""
    available = avail_phys
    if avail_pagefile and 0 < avail_pagefile < avail_phys:
        available = avail_pagefile
    return {"total": int(total_phys) if total_phys else None,
            "available": int(available) if available else None}


def memory_from_darwin(memsize_text, vm_stat_text=""):
    """hw.memsize plus the reclaimable pages of vm_stat (free, inactive,
    speculative, purgeable), the engine's own definition of available."""
    try:
        total = int((memsize_text or "").strip())
    except ValueError:
        total = None
    page_match = re.search(r"page size of (\d+) bytes", vm_stat_text or "")
    page = int(page_match.group(1)) if page_match else 16384
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        match = re.search(rf"{key}:\s+(\d+)\.", vm_stat_text or "")
        if match:
            pages += int(match.group(1))
    return {"total": total, "available": pages * page if pages else None}


def _windows_memory():
    try:
        from resource_plan import WINDOWS_MEMORYSTATUSEX_FIELDS

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [(name, getattr(ctypes, kind))
                        for name, kind in WINDOWS_MEMORYSTATUSEX_FIELDS]

        stat = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
        kernel32 = ctypes.windll.kernel32
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]
        kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return memory_from_windows_status(stat.ullTotalPhys, stat.ullAvailPhys,
                                              stat.ullAvailPageFile)
    except (OSError, AttributeError, ImportError, ValueError):
        pass
    return {"total": None, "available": None}


def detect_memory():
    if sys.platform == "win32":
        return _windows_memory()
    if sys.platform == "darwin":
        return memory_from_darwin(_run(["sysctl", "-n", "hw.memsize"]), _run(["vm_stat"]))
    try:
        with open("/proc/meminfo", encoding="ascii", errors="replace") as handle:
            mem = memory_from_meminfo(handle.read(256 * 1024))
    except OSError:
        mem = {"total": None, "available": None}
    # A container limit is what kills the process, whatever the host has free:
    # resource_plan already knows how to read it.
    try:
        from resource_plan import memory_available
        budget = memory_available()
        if budget is not None:
            mem["available"] = budget if mem["available"] is None else min(mem["available"], budget)
    except Exception:  # a malformed cgroup is reported by `coli plan`, not here
        pass
    return mem


# ---------------------------------------------------------------- CPU

#: The flags that change which kernels the engine compiles in (ARCH=native).
CPU_FEATURES_OF_INTEREST = (
    "avx2", "fma", "f16c", "avx_vnni", "avx512f", "avx512bw", "avx512_vnni",
    "avx512_bf16", "amx_tile", "asimd", "asimddp", "i8mm", "sve",
)


def cpu_from_cpuinfo(text):
    """Model name and feature flags from /proc/cpuinfo (x86 `flags`, Arm `Features`)."""
    name = None
    flags = set()
    for line in (text or "").splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip().lower()
        if key in ("model name", "hardware", "cpu model") and not name:
            name = value.strip() or None
        elif key in ("flags", "features") and not flags:
            flags = set(value.split())
    # Linux spells the VNNI flags avx512_vnni / avx_vnni; some kernels avx512vnni.
    if "avx512vnni" in flags:
        flags.add("avx512_vnni")
    features = [f for f in CPU_FEATURES_OF_INTEREST if f in flags]
    return {"name": name, "features": features}


#: IsProcessorFeaturePresent codes (winnt.h).
WINDOWS_PF = {"avx2": 40, "avx512f": 41, "asimd": 43}


def cpu_features_from_windows(is_present):
    """`is_present(code) -> bool` is kernel32.IsProcessorFeaturePresent."""
    return [name for name, code in WINDOWS_PF.items() if is_present(code)]


def _windows_cpu():
    name, features = None, []
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (OSError, ImportError):
        pass
    try:
        k32 = ctypes.windll.kernel32
        k32.IsProcessorFeaturePresent.argtypes = [ctypes.c_uint]
        k32.IsProcessorFeaturePresent.restype = ctypes.c_int
        features = cpu_features_from_windows(lambda code: bool(k32.IsProcessorFeaturePresent(code)))
    except (OSError, AttributeError):
        pass
    return {"name": name, "features": features}


def detect_cpu():
    if sys.platform == "win32":
        cpu = _windows_cpu()
    elif sys.platform == "darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or None
        feats = _run(["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"]).lower().split()
        flags = set(feats)
        if platform.machine().lower() in ("arm64", "aarch64"):
            flags.update({"asimd", "asimddp"})
        cpu = {"name": name, "features": [f for f in CPU_FEATURES_OF_INTEREST if f in flags]}
    else:
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                cpu = cpu_from_cpuinfo(handle.read(1 << 20))
        except OSError:
            cpu = {"name": None, "features": []}
    cpu["arch"] = platform.machine().lower() or None
    cpu["logical_cores"] = os.cpu_count()
    try:
        from resource_plan import physical_cpu_count
        cpu["physical_cores"] = physical_cpu_count()
    except Exception:
        cpu["physical_cores"] = None
    return cpu


# ---------------------------------------------------------------- OS / WSL


def is_wsl(osrelease_text=None, environ=None):
    environ = os.environ if environ is None else environ
    if environ.get("WSL_DISTRO_NAME") or environ.get("WSL_INTEROP"):
        return True
    if osrelease_text is None:
        try:
            with open("/proc/sys/kernel/osrelease", encoding="ascii", errors="replace") as handle:
                osrelease_text = handle.read(4096)
        except OSError:
            return False
    lowered = osrelease_text.lower()
    return "microsoft" in lowered or "wsl" in lowered


def os_release(text=None):
    """ID and ID_LIKE from /etc/os-release, for the package hints."""
    if text is None:
        try:
            with open("/etc/os-release", encoding="utf-8", errors="replace") as handle:
                text = handle.read(65536)
        except OSError:
            return {}
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key in ("ID", "ID_LIKE", "PRETTY_NAME", "VERSION_ID"):
            out[key] = value.strip().strip('"')
    return out


def detect_os():
    info = {"platform": sys.platform, "machine": platform.machine().lower(),
            "wsl": sys.platform.startswith("linux") and is_wsl()}
    if sys.platform.startswith("linux"):
        info.update({k.lower(): v for k, v in os_release().items()})
    elif sys.platform == "win32":
        info["pretty_name"] = f"Windows {platform.release()}"
    elif sys.platform == "darwin":
        info["pretty_name"] = f"macOS {platform.mac_ver()[0]}"
    return info


# ---------------------------------------------------------------- disk


def existing_ancestor(path):
    path = os.path.abspath(os.path.expanduser(path))
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def disk_free(path):
    try:
        return shutil.disk_usage(existing_ancestor(path)).free
    except OSError:
        return None


def path_warnings(path, wsl):
    """Advice about where a model lives; the engine streams it, so it matters.

    The WSL check is about a Linux path, so it is normalized with POSIX rules
    whatever Python runs it: os.path on Windows would turn /mnt/c/x into
    C:\\mnt\\c\\x and the check would silently pass everything."""
    warnings = []
    if not wsl:
        return warnings
    expanded = os.path.expanduser(path)
    if not expanded.startswith("/"):
        expanded = posixpath.join(os.getcwd(), expanded)
    absolute = posixpath.normpath(expanded)
    if re.match(r"^/mnt/[a-zA-Z](/|$)", absolute):
        warnings.append(
            f"{absolute} is a Windows drive seen from WSL: reading it is many times "
            "slower than the Linux disk, and the engine streams the model from it. "
            "Keep the model in the Linux home (for example ~/colibri-models).")
    return warnings


def fixed_drives_from_mask(mask, drive_type):
    """Drive roots from GetLogicalDrives' bitmask, kept when GetDriveTypeW says
    DRIVE_FIXED (3): no network shares, no removable sticks."""
    roots = []
    for index in range(26):
        if mask & (1 << index):
            root = f"{chr(ord('A') + index)}:\\"
            if drive_type(root) == 3:
                roots.append(root)
    return roots


def windows_fixed_drives():
    if sys.platform != "win32":
        return []
    try:
        k32 = ctypes.windll.kernel32
        k32.GetDriveTypeW.argtypes = [ctypes.c_wchar_p]
        k32.GetDriveTypeW.restype = ctypes.c_uint
        return fixed_drives_from_mask(k32.GetLogicalDrives(), k32.GetDriveTypeW)
    except (OSError, AttributeError):
        return []


# ---------------------------------------------------------------- NVIDIA

#: What the setup asks nvidia-smi, in this order. compute_cap is the card's CUDA
#: compute capability ("7.0" for a V100): which nvcc can build for the card
#: depends on it (setup_flow.choose_backend). pci.bus_id ties a row to the CUDA
#: driver's own device when compute_cap has to come from there.
NVIDIA_FIELDS = ("index", "name", "memory.total", "memory.free", "driver_version",
                 "compute_cap", "pci.bus_id")
#: The same without compute_cap: a driver that predates the field refuses the
#: whole query and prints no rows, so the setup asks again without it.
NVIDIA_FIELDS_NO_CC = tuple(f for f in NVIDIA_FIELDS if f != "compute_cap")


def nvidia_smi_query(fields=NVIDIA_FIELDS):
    return [f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]


NVIDIA_SMI_QUERY = nvidia_smi_query()


def parse_nvidia_smi(text, fields=NVIDIA_FIELDS):
    """Rows of `nvidia-smi --query-gpu=<fields> --format=csv,noheader,nounits`
    (memory in MiB). A unified-memory part prints [N/A] for memory: that is
    None, not zero; so is a compute capability the driver does not report.
    Rows with fewer columns than `fields` (an older query) leave the missing
    ones None."""
    import csv
    gpus = []
    for row in csv.reader((text or "").splitlines()):
        row = [f.strip() for f in row]
        if len(row) < 4:
            continue
        values = dict(zip(fields, row))
        try:
            index = int(values["index"])
        except (KeyError, ValueError):
            continue

        def mib(value):
            try:
                return int(float(value)) * 1024 * 1024
            except (TypeError, ValueError):
                return None

        def known(value):
            return value if value and not value.startswith("[") else None

        gpus.append({"index": index, "name": values.get("name"),
                     "total_bytes": mib(values.get("memory.total")),
                     "free_bytes": mib(values.get("memory.free")),
                     "driver": known(values.get("driver_version")),
                     "compute_cap": compute_cap_text(values.get("compute_cap")),
                     "pci_bus_id": known(values.get("pci.bus_id"))})
    return gpus


def compute_cap_text(value):
    """A compute capability as "7.0", from nvidia-smi or the CUDA driver; None
    for [N/A], an empty field or anything that is not major.minor."""
    match = re.match(r"^\s*(\d+)\.(\d+)\s*$", value or "")
    return f"{int(match.group(1))}.{int(match.group(2))}" if match else None


def compute_cap_sm(value):
    """The number nvcc names an architecture by: "7.0" -> 70 (compute_70,
    sm_70), "12.0" -> 120, "10.3" -> 103. None when the capability is unknown."""
    text = compute_cap_text(value)
    if text is None:
        return None
    major, minor = text.split(".")
    return int(major) * 10 + int(minor)


def pci_bus_key(value):
    """nvidia-smi writes 00000000:01:00.0, the CUDA driver 0000:01:00.0: the
    same slot. (domain, bus, device, function) as integers, or None."""
    match = re.match(r"^\s*([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-7])\s*$", value or "")
    return tuple(int(part, 16) for part in match.groups()) if match else None


def fill_compute_caps(gpus, cuda_devices):
    """Give each nvidia-smi row without a compute capability the one the CUDA
    driver reports for the same PCI slot. Rows that cannot be matched stay None:
    an unknown capability is not a guess."""
    by_slot = {pci_bus_key(d.get("pci_bus_id")): compute_cap_text(d.get("compute_cap"))
               for d in cuda_devices or []}
    by_slot.pop(None, None)
    for gpu in gpus:
        if gpu.get("compute_cap") is None:
            gpu["compute_cap"] = by_slot.get(pci_bus_key(gpu.get("pci_bus_id")))
    return gpus


def nvidia_smi_path():
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for candidate in ("/usr/lib/wsl/lib/nvidia-smi",
                      r"C:\Windows\System32\nvidia-smi.exe"):
        if os.path.isfile(candidate):
            return candidate
    return None


def detect_nvidia():
    """The NVIDIA cards, each with its compute capability when anything can tell.

    First nvidia-smi with compute_cap. A driver that does not know that field
    refuses the whole query, so no rows means asking again without it; then
    (and for any card printed as [N/A]) the CUDA driver itself is asked, in a
    child process like the Vulkan probe, and its answer matched by PCI slot."""
    exe = nvidia_smi_path()
    if not exe:
        return []
    gpus = parse_nvidia_smi(_run([exe] + nvidia_smi_query(NVIDIA_FIELDS), timeout=10), NVIDIA_FIELDS)
    if not gpus:
        gpus = parse_nvidia_smi(_run([exe] + nvidia_smi_query(NVIDIA_FIELDS_NO_CC), timeout=10),
                                NVIDIA_FIELDS_NO_CC)
    if any(gpu.get("compute_cap") is None for gpu in gpus):
        probe = _probe_cuda_child()
        if probe:
            fill_compute_caps(gpus, probe.get("devices"))
    return gpus


def _cuda_driver_library():
    names = {"win32": ["nvcuda.dll"]}.get(sys.platform, ["libcuda.so.1", "libcuda.so"])
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _cuda_probe_inprocess():
    """Compute capability and PCI slot of every device, from the CUDA driver
    API (the same driver the engine's CUDA runtime talks to)."""
    lib = _cuda_driver_library()
    if lib is None:
        return {"driver": False, "devices": []}
    c_int = ctypes.c_int
    if lib.cuInit(0) != 0:
        return {"driver": True, "devices": []}
    count = c_int(0)
    if lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
        return {"driver": True, "devices": []}
    devices = []
    for ordinal in range(count.value):
        device = c_int(0)
        if lib.cuDeviceGet(ctypes.byref(device), ordinal) != 0:
            continue
        major, minor = c_int(0), c_int(0)
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75, _MINOR = 76 (cuda.h)
        lib.cuDeviceGetAttribute(ctypes.byref(major), 75, device)
        lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, device)
        bus = ctypes.create_string_buffer(64)
        name = ctypes.create_string_buffer(256)
        lib.cuDeviceGetPCIBusId(bus, 64, device)
        lib.cuDeviceGetName(name, 256, device)
        devices.append({"name": name.value.decode("utf-8", "replace"),
                        "pci_bus_id": bus.value.decode("ascii", "replace"),
                        "compute_cap": f"{major.value}.{minor.value}" if major.value else None})
    return {"driver": True, "devices": devices}


def _probe_cuda_child(timeout=25):
    try:
        result = subprocess.run([sys.executable, os.path.join(HERE, "setup_hw.py"), "--cuda-probe"],
                                capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------- Vulkan

VK_DEVICE_TYPES = {0: "other", 1: "integrated", 2: "discrete", 3: "virtual", 4: "cpu"}
#: The engine's own ranking (backend_vulkan.c, coli_vk_init).
VK_TYPE_RANK = {"discrete": 4, "integrated": 3, "virtual": 2, "other": 1, "cpu": 0}
VK_VENDORS = {0x10DE: "NVIDIA", 0x1002: "AMD", 0x1022: "AMD", 0x8086: "Intel",
              0x13B5: "Arm", 0x5143: "Qualcomm", 0x106B: "Apple", 0x1414: "Microsoft",
              0x10005: "Mesa"}


def vk_version(value):
    return f"{value >> 22 & 0x7F}.{value >> 12 & 0x3FF}.{value & 0xFFF}"


def _vulkan_library():
    names = {"win32": ["vulkan-1.dll"],
             "darwin": ["libvulkan.1.dylib", "libMoltenVK.dylib"]}.get(
                 sys.platform, ["libvulkan.so.1", "libvulkan.so"])
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _vk_probe_inprocess():
    """Enumerate Vulkan physical devices with the loader, as the engine does.

    Raw buffers and explicit offsets rather than full ctypes mirrors of the
    Vulkan structs: only the leading fields are read, and a buffer larger than
    the struct cannot be overrun by a newer driver. Offsets are the C layout on
    every 64-bit ABI colibri builds for."""
    lib = _vulkan_library()
    if lib is None:
        return {"loader": False, "devices": []}
    u32, vp = ctypes.c_uint32, ctypes.c_void_p

    lib.vkCreateInstance.argtypes = [vp, vp, ctypes.POINTER(vp)]
    lib.vkCreateInstance.restype = ctypes.c_int32
    lib.vkEnumeratePhysicalDevices.argtypes = [vp, ctypes.POINTER(u32), vp]
    lib.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
    lib.vkGetPhysicalDeviceProperties.argtypes = [vp, vp]
    lib.vkGetPhysicalDeviceProperties.restype = None
    lib.vkGetPhysicalDeviceMemoryProperties.argtypes = [vp, vp]
    lib.vkGetPhysicalDeviceMemoryProperties.restype = None
    lib.vkEnumerateDeviceExtensionProperties.argtypes = [vp, ctypes.c_char_p, ctypes.POINTER(u32), vp]
    lib.vkEnumerateDeviceExtensionProperties.restype = ctypes.c_int32
    lib.vkDestroyInstance.argtypes = [vp, vp]
    lib.vkDestroyInstance.restype = None

    instance_version = 1 << 22  # 1.0
    if hasattr(lib, "vkEnumerateInstanceVersion"):
        lib.vkEnumerateInstanceVersion.argtypes = [ctypes.POINTER(u32)]
        lib.vkEnumerateInstanceVersion.restype = ctypes.c_int32
        value = u32(0)
        if lib.vkEnumerateInstanceVersion(ctypes.byref(value)) == 0:
            instance_version = value.value
    api = min(instance_version, (1 << 22) | (3 << 12))

    class AppInfo(ctypes.Structure):
        _fields_ = [("sType", u32), ("pNext", vp), ("pApplicationName", ctypes.c_char_p),
                    ("applicationVersion", u32), ("pEngineName", ctypes.c_char_p),
                    ("engineVersion", u32), ("apiVersion", u32)]

    class CreateInfo(ctypes.Structure):
        _fields_ = [("sType", u32), ("pNext", vp), ("flags", u32), ("pApplicationInfo", vp),
                    ("enabledLayerCount", u32), ("ppEnabledLayerNames", vp),
                    ("enabledExtensionCount", u32), ("ppEnabledExtensionNames", vp)]

    app = AppInfo(0, None, b"colibri-setup", 1, b"colibri", 1, api)
    info = CreateInfo(1, None, 0, ctypes.cast(ctypes.byref(app), vp), 0, None, 0, None)
    instance = vp()
    result = lib.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(instance))
    if result != 0 or not instance:
        return {"loader": True, "instance_error": result, "devices": []}
    devices = []
    try:
        count = u32(0)
        lib.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)
        handles = (vp * max(count.value, 1))()
        lib.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), handles)
        props2 = None
        if api >= (1 << 22) | (1 << 12):
            props2 = getattr(lib, "vkGetPhysicalDeviceMemoryProperties2", None)
            if props2 is not None:
                props2.argtypes = [vp, vp]
                props2.restype = None
        for handle in handles[:count.value]:
            raw = (ctypes.c_ubyte * 2048)()
            lib.vkGetPhysicalDeviceProperties(handle, raw)
            props = bytes(raw)
            api_v, driver_v, vendor, device_id, dtype = struct.unpack_from("<5I", props, 0)
            name = props[20:276].split(b"\0", 1)[0].decode("utf-8", "replace")

            ext_count = u32(0)
            lib.vkEnumerateDeviceExtensionProperties(handle, None, ctypes.byref(ext_count), None)
            ext_raw = (ctypes.c_ubyte * (260 * max(ext_count.value, 1)))()
            lib.vkEnumerateDeviceExtensionProperties(handle, None, ctypes.byref(ext_count), ext_raw)
            ext_bytes = bytes(ext_raw)
            extensions = {ext_bytes[i * 260:i * 260 + 256].split(b"\0", 1)[0].decode("ascii", "replace")
                          for i in range(ext_count.value)}
            has_budget = "VK_EXT_memory_budget" in extensions and props2 is not None

            budget_raw = (ctypes.c_ubyte * 512)()
            if has_budget:
                mem2 = (ctypes.c_ubyte * 1024)()
                struct.pack_into("<I", budget_raw, 0, 1000237000)
                struct.pack_into("<I", mem2, 0, 1000059006)
                struct.pack_into("<Q", mem2, 8, ctypes.addressof(budget_raw))
                props2(handle, mem2)
                memory = bytes(mem2)[16:16 + 520]
            else:
                mem = (ctypes.c_ubyte * 1024)()
                lib.vkGetPhysicalDeviceMemoryProperties(handle, mem)
                memory = bytes(mem)[:520]
            devices.append(_vk_device_record(name, dtype, vendor, device_id, api_v, driver_v,
                                             memory, bytes(budget_raw) if has_budget else None,
                                             "VK_KHR_cooperative_matrix" in extensions))
    finally:
        lib.vkDestroyInstance(instance, None)
    return {"loader": True, "instance_version": vk_version(instance_version), "devices": devices}


def _vk_device_record(name, dtype, vendor, device_id, api_v, driver_v, memory, budget, coop):
    """Turn VkPhysicalDeviceMemoryProperties (and the budget struct) into figures."""
    heap_count = struct.unpack_from("<I", memory, 260)[0]
    heaps = []
    for index in range(min(heap_count, 16)):
        size, flags = struct.unpack_from("<QI", memory, 264 + index * 16)
        heap = {"size": size, "device_local": bool(flags & 1)}
        if budget is not None:
            heap["budget"] = struct.unpack_from("<Q", budget, 16 + index * 8)[0]
            heap["usage"] = struct.unpack_from("<Q", budget, 144 + index * 8)[0]
        heaps.append(heap)
    local = [h for h in heaps if h["device_local"]]
    return {
        "name": name,
        "type": VK_DEVICE_TYPES.get(dtype, "other"),
        "vendor": VK_VENDORS.get(vendor, f"0x{vendor:04x}"),
        "vendor_id": vendor,
        "device_id": device_id,
        "api_version": vk_version(api_v),
        "api_version_raw": api_v,
        "device_local_bytes": sum(h["size"] for h in local) or None,
        "budget_bytes": (sum(h.get("budget", 0) for h in local) or None) if budget is not None else None,
        "memory_budget_ext": budget is not None,
        "cooperative_matrix": coop,
        "heaps": heaps,
    }


def parse_vulkaninfo_summary(text):
    """Devices from `vulkaninfo --summary`: names, types and API versions. The
    summary carries no memory figures, so those stay None."""
    devices = []
    current = None
    for line in (text or "").splitlines():
        stripped = line.strip()
        if re.match(r"^GPU\d+:$", stripped):
            current = {"name": None, "type": "other", "vendor": None, "vendor_id": None,
                       "device_id": None, "api_version": None, "api_version_raw": None,
                       "device_local_bytes": None, "budget_bytes": None,
                       "memory_budget_ext": False, "cooperative_matrix": False, "heaps": []}
            devices.append(current)
            continue
        if current is None:
            continue
        key, sep, value = stripped.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if key == "deviceName":
            current["name"] = value
        elif key == "deviceType":
            kind = value.replace("PHYSICAL_DEVICE_TYPE_", "").lower()
            current["type"] = {"integrated_gpu": "integrated", "discrete_gpu": "discrete",
                               "virtual_gpu": "virtual"}.get(kind, kind if kind in VK_TYPE_RANK else "other")
        elif key == "apiVersion":
            version = value.split()[0]
            current["api_version"] = version
            parts = [int(p) for p in re.findall(r"\d+", version)[:3]] + [0, 0, 0]
            current["api_version_raw"] = (parts[0] << 22) | (parts[1] << 12) | parts[2]
        elif key == "vendorID":
            try:
                vendor = int(value.split()[0], 0)
                current["vendor_id"] = vendor
                current["vendor"] = VK_VENDORS.get(vendor, f"0x{vendor:04x}")
            except ValueError:
                pass
        elif key == "deviceID":
            try:
                current["device_id"] = int(value.split()[0], 0)
            except ValueError:
                pass
    return [d for d in devices if d["name"]]


def vulkan_usable(device):
    """A device the GPU path can use: a real GPU (not a CPU rasterizer such as
    llvmpipe/Lavapipe) with Vulkan 1.2, the engine's floor."""
    if device.get("type") not in ("discrete", "integrated", "virtual"):
        return False
    raw = device.get("api_version_raw")
    return raw is None or raw >= (1 << 22) | (2 << 12)


def best_vulkan_device(devices):
    """The device the engine will pick: the highest-ranked type, first wins."""
    best, rank = None, -1
    for device in devices:
        value = VK_TYPE_RANK.get(device.get("type"), 0)
        if value > rank:
            best, rank = device, value
    return best


def find_user_icds(home=None):
    """Vulkan driver manifests installed under the home directory.

    A driver built by hand (Mesa's Dozen for WSL2 is the usual case) installs
    its manifest under a prefix in $HOME, which the loader does not search. Only
    manifests are returned; whether one gives a usable device is the probe's
    question."""
    home = home or os.path.expanduser("~")
    patterns = (".local/share/vulkan/icd.d/*.json",
                "*/share/vulkan/icd.d/*.json",
                "*/install/share/vulkan/icd.d/*.json",
                "*/*/share/vulkan/icd.d/*.json")
    found = []
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(home, pattern))):
            if path not in found and os.path.isfile(path):
                found.append(os.path.normpath(path))
    return found


def _probe_vulkan_child(extra_env=None, timeout=25):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    try:
        result = subprocess.run([sys.executable, os.path.join(HERE, "setup_hw.py"), "--vulkan-probe"],
                                capture_output=True, text=True, timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                return None
    return None


def detect_vulkan(wsl=None, home=None):
    """{"devices": [...], "icd": manifest-or-None, "source": "loader"|"vulkaninfo"|None}.

    On WSL the system drivers usually see no GPU (only the CPU rasterizer); a
    Dozen driver installed under $HOME does. When the default search finds no
    usable GPU there, each user manifest is tried in turn and the first one that
    yields a usable device is reported with its path, so the run configuration
    can point the engine at the same driver."""
    if host_os() == "darwin":
        return {"devices": [], "icd": None, "source": None}
    wsl = is_wsl() if wsl is None else wsl
    probe = _probe_vulkan_child()
    if probe is not None and probe.get("loader"):
        result = {"devices": probe.get("devices", []), "icd": None, "source": "loader"}
    else:
        summary = _run(["vulkaninfo", "--summary"], timeout=20) if shutil.which("vulkaninfo") else ""
        result = {"devices": parse_vulkaninfo_summary(summary), "icd": None,
                  "source": "vulkaninfo" if summary else None}
    if not any(vulkan_usable(d) for d in result["devices"]) and host_os().startswith("linux") \
            and not os.environ.get("VK_ICD_FILENAMES") and not os.environ.get("VK_DRIVER_FILES"):
        for manifest in find_user_icds(home):
            extra = _probe_vulkan_child({"VK_ICD_FILENAMES": manifest, "VK_DRIVER_FILES": manifest})
            if extra and any(vulkan_usable(d) for d in extra.get("devices", [])):
                result = {"devices": extra["devices"], "icd": manifest, "source": "loader"}
                break
    if os.environ.get("VK_ICD_FILENAMES") or os.environ.get("VK_DRIVER_FILES"):
        result["icd"] = os.environ.get("VK_ICD_FILENAMES") or os.environ.get("VK_DRIVER_FILES")
    return result


# ---------------------------------------------------------------- Windows video (informational)


def parse_win_video_controllers(text):
    """`Name|AdapterRAM` lines from Win32_VideoController. AdapterRAM is a uint32
    and saturates at 4 GiB, so it is shown, never planned with."""
    out = []
    for line in (text or "").splitlines():
        name, sep, ram = line.strip().partition("|")
        if not name:
            continue
        try:
            ram_bytes = int(ram) if sep and ram.strip() else None
        except ValueError:
            ram_bytes = None
        out.append({"name": name.strip(), "adapter_ram_bytes": ram_bytes})
    return out


def detect_windows_video():
    if sys.platform != "win32":
        return []
    text = _run(["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.AdapterRAM }"],
                timeout=20)
    return parse_win_video_controllers(text)


# ---------------------------------------------------------------- the report


def _run(cmd, timeout=10):
    try:
        # errors="replace": a GPU name in the console code page must not turn
        # into a UnicodeDecodeError that throws the whole probe away.
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                              timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


def gpu_summary(vulkan, nvidia):
    """What the setup acts on: the Vulkan device the engine would pick (when it
    is a usable GPU) and the NVIDIA cards."""
    devices = vulkan.get("devices", []) if vulkan else []
    best = best_vulkan_device(devices)
    usable = best if best is not None and vulkan_usable(best) else None
    return {"vulkan": usable, "vulkan_icd": vulkan.get("icd") if vulkan else None,
            "nvidia": nvidia, "has_gpu": bool(usable or nvidia)}


def detect(target_dir=None, probe_gpu=True):
    """The whole report, as plain data (JSON-serialisable)."""
    os_info = detect_os()
    target = os.path.abspath(os.path.expanduser(target_dir)) if target_dir else None
    report = {
        "os": os_info,
        "cpu": detect_cpu(),
        "memory": detect_memory(),
        "disk": None,
        "vulkan": {"devices": [], "icd": None, "source": None},
        "nvidia": [],
        "windows_video": [],
    }
    if target:
        report["disk"] = {"path": target, "free_bytes": disk_free(target),
                          "warnings": path_warnings(target, os_info.get("wsl"))}
    if probe_gpu:
        report["vulkan"] = detect_vulkan(wsl=os_info.get("wsl"))
        report["nvidia"] = detect_nvidia()
        if not report["vulkan"]["devices"]:
            report["windows_video"] = detect_windows_video()
    report["gpu"] = gpu_summary(report["vulkan"], report["nvidia"])
    return report


def fmt_gb(value):
    if value is None:
        return "?"
    return f"{value / GB:.1f} GB" if value < 100 * GB else f"{value / GB:.0f} GB"


def format_report(report):
    cpu, mem, osi = report["cpu"], report["memory"], report["os"]
    lines = []
    cores = cpu.get("physical_cores")
    threads = cpu.get("logical_cores")
    core_text = f"{cores} cores ({threads} threads)" if cores and threads and cores != threads \
        else f"{threads or cores or '?'} cores"
    features = ", ".join(f.upper() for f in cpu.get("features", []) if f in
                         ("avx2", "avx_vnni", "avx512f", "avx512_vnni", "avx512_bf16", "asimddp", "sve"))
    lines.append(f"  CPU     {cpu.get('name') or cpu.get('arch') or '?'}, {core_text}"
                 + (f", {features}" if features else ""))
    lines.append(f"  RAM     {fmt_gb(mem.get('total'))}"
                 + (f" ({fmt_gb(mem.get('available'))} free now)" if mem.get("available") else ""))
    disk = report.get("disk")
    if disk:
        lines.append(f"  Disk    {fmt_gb(disk.get('free_bytes'))} free in {disk['path']}")
        for warning in disk.get("warnings", []):
            lines.append(f"          warning: {warning}")
    gpu = report.get("gpu", {})
    shown = False
    if gpu.get("vulkan"):
        device = gpu["vulkan"]
        memory = device.get("budget_bytes") or device.get("device_local_bytes")
        share = ", shares RAM" if device["type"] == "integrated" else ""
        lines.append(f"  GPU     {device['name']} via Vulkan {device.get('api_version') or ''}"
                     f" ({device['type']}{share}"
                     + (f", {fmt_gb(memory)} {'budget' if device.get('budget_bytes') else 'device memory'}" if memory else "")
                     + ")")
        if gpu.get("vulkan_icd"):
            lines.append(f"          driver manifest: {gpu['vulkan_icd']}")
        shown = True
    for card in gpu.get("nvidia", []):
        lines.append(f"  GPU     {card['name']} (NVIDIA, {fmt_gb(card.get('total_bytes'))} VRAM"
                     + (f", compute {card['compute_cap']}" if card.get("compute_cap") else "")
                     + (f", driver {card['driver']}" if card.get("driver") else "") + ")")
        shown = True
    if not shown:
        cpu_only = [d for d in report.get("vulkan", {}).get("devices", []) if d.get("type") == "cpu"]
        others = report.get("windows_video") or []
        if others:
            names = ", ".join(v["name"] for v in others)
            lines.append(f"  GPU     {names} (no Vulkan driver answered; the engine will use the CPU)")
        elif cpu_only:
            lines.append(f"  GPU     none usable (Vulkan sees only {cpu_only[0]['name']}, a CPU rasterizer)")
        else:
            lines.append("  GPU     none found (the engine runs on the CPU)")
    system = osi.get("pretty_name") or osi.get("platform")
    if osi.get("wsl"):
        system = f"{system} under WSL2"
    lines.append(f"  System  {system}")
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--vulkan-probe"]:
        try:
            print(json.dumps(_vk_probe_inprocess()))
        except Exception as error:  # report, never crash the parent's parse
            print(json.dumps({"loader": False, "error": str(error), "devices": []}))
        return 0
    if argv[:1] == ["--cuda-probe"]:
        try:
            print(json.dumps(_cuda_probe_inprocess()))
        except Exception as error:  # same rule as the Vulkan probe
            print(json.dumps({"driver": False, "error": str(error), "devices": []}))
        return 0
    sys.path.insert(0, HERE)
    targets = [arg for arg in argv if not arg.startswith("--")]
    report = detect(targets[0] if targets else os.path.expanduser("~"))
    if "--json" in argv:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
