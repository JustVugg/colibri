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


# Strict .coli_ssd grammar -- the byte-for-byte mirror of colibri.c's
# coli_ssd_cache_parse() (keep the two in lockstep; test_resource_plan.py and
# test_ssd_probe.c chew the same vector file, tests/fixtures/ssd_cache_vectors.txt):
#   v2:     b"v2 <gbs> <st_dev>"   single spaces, at most one trailing \n
#   legacy: b"<gbs>"               the pre-fix format, never trusted
# where <gbs> = digits["."digits] with 0 < gbs < 1000 and <st_dev> = 1..20
# digits fitting unsigned 64-bit; total length <= 64 bytes, no NULs, nothing
# else. float() permissiveness ("inf", "nan", "1e99", whitespace, signs) is
# deliberately out: it let corrupt caches surface as measurements, and "inf"
# reached doctor's JSON as the invalid literal Infinity.
_SSD_CACHE_V2 = re.compile(rb"\Av2 (\d+(?:\.\d+)?) (\d{1,20})\n?\Z")
_SSD_CACHE_LEGACY = re.compile(rb"\A(\d+(?:\.\d+)?)\n?\Z")


def parse_ssd_cache(data):
    """Classify raw .coli_ssd bytes under the strict grammar above. Returns
    ("v2", gbs, st_dev), ("legacy", gbs, None), or (None, None, None) for
    garbage. Classification only -- trust (the st_dev match) is the caller's."""
    if not data or len(data) > 64 or b"\x00" in data:
        return (None, None, None)
    match = _SSD_CACHE_V2.match(data)
    if match:
        gbs, dev = float(match.group(1)), int(match.group(2))
        if 0 < gbs < 1000 and dev <= 0xFFFFFFFFFFFFFFFF:
            return ("v2", gbs, dev)
        return (None, None, None)
    match = _SSD_CACHE_LEGACY.match(data)
    if match:
        gbs = float(match.group(1))
        if 0 < gbs < 1000:
            return ("legacy", gbs, None)
    return (None, None, None)


def ssd_probe_state(model_dir):
    """Classify the cached F_NOCACHE storage probe the C engine writes to
    <model>/.coli_ssd on its first Metal+darwin startup (colibri.c
    coli_ssd_probe_cached, issue #379). Read-only: never re-measures, never
    guesses -- mirrors S4's "read-and-display only" contract for `coli
    doctor`/`coli plan`. Returns (state, gbs):
      ("ok", gbs)        v2 cache recorded on THIS volume -- the one case the
                         engine itself would trust
      ("legacy", None)   pre-v2 bare number; the engine re-probes + upgrades
      ("foreign", None)  v2 from another volume (st_dev mismatch); re-probed
      ("garbage", None)  a file exists but fails the strict grammar
      ("absent", None)   no cache file at all
    The distinctions matter for wording (#386 r2, F10): "no cached probe yet"
    is a lie when a file exists. The read is bounded to 65 bytes (F13): the
    strict grammar caps a well-formed cache at 64, so byte 65 alone already
    convicts -- no reason to slurp an arbitrarily large impostor file."""
    try:
        with open(Path(model_dir) / ".coli_ssd", "rb") as fh:
            data = fh.read(65)
    except OSError:
        return ("absent", None)
    kind, gbs, dev = parse_ssd_cache(data)
    if kind == "v2":
        try:
            if dev == os.stat(model_dir).st_dev:
                return ("ok", gbs)
        except OSError:
            pass
        return ("foreign", None)
    if kind == "legacy":
        return ("legacy", None)
    return ("garbage", None)


def read_ssd_probe(model_dir):
    """The measured GB/s as a float when the engine itself would trust the
    cache (ssd_probe_state "ok"), else None."""
    return ssd_probe_state(model_dir)[1]


# What doctor/plan say for a cache that exists but is not trusted (#386 r2,
# F10): each state names what will actually happen, never "no cached probe
# yet" while a file sits right there.
SSD_PROBE_PENDING = {
    "legacy": "legacy cache pending engine upgrade; re-measured on the next Metal+darwin start",
    "foreign": "cache from another volume; the engine will re-probe here",
    "garbage": "unreadable cache; the engine will re-probe",
}


def discover_gpus():
    # NVIDIA first; if there are none (or no nvidia-smi), fall back to ROCm/HIP so
    # a working AMD engine isn't planned CPU-only and --gpu N stops failing (#662).
    devices = _discover_nvidia_gpus()
    if devices:
        return devices
    return _discover_amd_gpus()


def _discover_nvidia_gpus():
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None:
        devices = []
        import csv
        for fields in csv.reader(result.stdout.splitlines()):
            fields = [f.strip() for f in fields]
            if len(fields) != 4:
                continue
            try:
                index = int(fields[0])
            except ValueError:
                continue
            # Unified-memory chips (e.g. NVIDIA GB10 Grace Blackwell) have no
            # discrete VRAM pool, so nvidia-smi reports memory.total/memory.free
            # as "[N/A]" rather than a number. Fall back to system RAM figures
            # in that case instead of silently dropping the GPU from discovery.
            try:
                total, free = int(fields[2]), int(fields[3])
            except ValueError:
                try:
                    meminfo = Path("/proc/meminfo").read_text()
                    total = int(re.search(r"MemTotal:\s+(\d+)", meminfo).group(1)) // 1024
                    free = memory_available() // (1024 * 1024)
                except (OSError, AttributeError):
                    total = free = 0
            devices.append({"index": index, "name": fields[1],
                            "total_bytes": total * 1024 * 1024,
                            "free_bytes": free * 1024 * 1024})
        if devices:
            return devices

    # ROCm fallback. rocm-smi's text format is stable across the supported
    # releases, but its optional product-name fields are not; keep the name
    # generic and only trust explicitly reported VRAM totals/usage.
    try:
        result = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                                text=True, capture_output=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is None:
        # HIP SDK for Windows ships hipInfo instead of rocm-smi. It reports
        # totalGlobalMem but not a reliable process-wide free value, so use the
        # total as an upper bound; the engine still clamps uploads at runtime.
        try:
            result = subprocess.run(["hipInfo"], text=True, capture_output=True,
                                    check=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return []
        devices = []
        blocks = re.split(r"(?=^\s*Name:\s*)", result.stdout, flags=re.MULTILINE)
        for block in blocks:
            name = re.search(r"^\s*Name:\s*(.+)$", block, flags=re.MULTILINE)
            memory = re.search(r"totalGlobalMem:\s*([0-9.]+)\s*GB", block)
            if not memory:
                continue
            total = int(float(memory.group(1)) * GB)
            devices.append({"index": len(devices),
                            "name": name.group(1).strip() if name else "AMD GPU",
                            "total_bytes": total, "free_bytes": total})
        return devices
    devices = {}
    current = None
    for line in result.stdout.splitlines():
        match = re.search(r"GPU\[(\d+)\]", line)
        if match:
            current = int(match.group(1))
            devices.setdefault(current, {})
        if current is None:
            continue
        number = re.search(r"(\d+)\s*$", line)
        if not number:
            continue
        value = int(number.group(1))
        if "VRAM Total Memory" in line:
            devices[current]["total_bytes"] = value
        elif "VRAM Total Used Memory" in line:
            devices[current]["used_bytes"] = value
    return [{"index": index, "name": f"AMD GPU {index}",
             "total_bytes": item["total_bytes"],
             "free_bytes": max(0, item["total_bytes"] - item.get("used_bytes", 0))}
            for index, item in sorted(devices.items()) if "total_bytes" in item]


def _discover_amd_gpus():
    """ROCm/HIP discovery via rocm-smi (#662). Absent on non-AMD hosts, so this
    returns [] there. rocm-smi --showmeminfo vram reports BYTES (unlike nvidia-smi's
    MiB), so no unit scaling. Column names drift across ROCm versions, so match them
    by substring rather than position. VERIFY on AMD hardware (labelled
    hardware-owner-needed) -- authored without a ROCm host to test against."""
    command = ["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--csv"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    import csv
    rows = list(csv.DictReader(result.stdout.splitlines()))
    if not rows:
        return []

    def find_col(row, *needles):
        for key in row:
            low = (key or "").lower()
            if all(n in low for n in needles):
                return key
        return None

    devices = []
    for i, row in enumerate(rows):
        dev = (row.get("device") or "").strip()
        m = re.search(r"(\d+)", dev)
        index = int(m.group(1)) if m else i
        total_col = find_col(row, "vram", "total", "memory")
        used_col = find_col(row, "vram", "used")
        name_col = (find_col(row, "card", "series") or find_col(row, "card", "model")
                    or find_col(row, "product"))
        try:
            total = int((row.get(total_col) or "0").strip())
        except (ValueError, TypeError):
            total = 0
        try:
            used = int((row.get(used_col) or "0").strip())
        except (ValueError, TypeError):
            used = 0
        free = max(total - used, 0)
        name = (row.get(name_col) or "").strip() if name_col else ""
        devices.append({"index": index, "name": name or f"AMD GPU {index}",
                        "total_bytes": total, "free_bytes": free})
    return devices


def _physical_cores_warn(message):
    """Visibility for a mis-detected core count: a silent "1" here becomes
    OMP_NUM_THREADS=1 and pins the whole run to a single core (#325). Emit on
    stderr so it surfaces in the [PLAN]/[OMP] stream without being swallowed."""
    print(f"[plan] warning: {message}", file=sys.stderr)


def physical_cpu_count():
    """Number of physical CPU cores (not SMT siblings).

    Per-expert matmul regions are tiny and back-to-back; two SMT siblings share
    one AVX-512 unit and contend, so logical (SMT) counts over-subscribe and
    hurt throughput. We want true physical cores. A silent 1 here propagates to
    OMP_NUM_THREADS=1 and pins the run to one core (#325), so every fallback
    must be visible, never just ``or 1``.
    """
    if sys.platform == "win32":
        # Contiamo i core fisici veri con GetLogicalProcessorInformationEx
        # (RelationProcessorCore). Le firme vanno dichiarate: su Python a 64 bit
        # una WinAPI non dichiarata ritorna c_int (32 bit) e riceve i puntatori
        # come c_int di default, quindi il probe puo' fallire silenziosamente.
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetLogicalProcessorInformationEx.argtypes = [
                ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            k32.GetLogicalProcessorInformationEx.restype = ctypes.c_int
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
            _physical_cores_warn("GetLogicalProcessorInformationEx returned no cores")
        except (OSError, ValueError, AttributeError) as error:
            _physical_cores_warn(f"Windows core probe failed: {error}")
    try:
        # Ask lscpu for exactly core,socket and dedupe on (core, socket).
        # Counting un-deduplicated rows would return logical threads (SMT),
        # which was the original over-subscription bug. Empty fields ("-")
        # mark an offline core/socket and fail int() -> skipped.
        #
        # Column layout robustness: `lscpu -p=<list>` emits *exactly* the
        # requested columns (no CPU prefix), while bare `lscpu -p` prepends
        # CPU. We requested two columns, but take the LAST TWO fields so the
        # parser stays correct whether or not a CPU column is present
        # (JustVugg review: the previous fields[1]/fields[2] indexing assumed
        #  a 3-column layout and regressed 2-column output to the logical
        # count -- the opposite of the fix).
        result = subprocess.run(["lscpu", "-p=core,socket"], text=True,
                                capture_output=True, check=True, timeout=5)
        cores = set()
        for line in result.stdout.splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split(",")
            if len(fields) < 2:
                continue
            try:
                core, socket = int(fields[-2]), int(fields[-1])
            except ValueError:
                continue  # "-" for an offline core/socket
            cores.add((core, socket))
        if cores:
            return len(cores)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        _physical_cores_warn(f"lscpu core probe failed: {error}")
    logical = os.cpu_count()
    if not logical:
        _physical_cores_warn(
            "could not detect any CPU cores; falling back to 1. "
            "Set OMP_NUM_THREADS manually to fix single-core decode (#325).")
        return 1
    _physical_cores_warn(
        f"physical-core probes unavailable; using {logical} logical CPUs "
        f"(SMT may over-subscribe). Set OMP_NUM_THREADS to physical cores if slow.")
    return logical


def _resolve_physical_cores(physical_cpus):
    """Coerce the build_plan() physical-core argument to a sane positive int.

    A None/0/None-ish value reaching here means physical_cpu_count() already
    warned; clamp to 1 (so the engine always gets a positive team size) but keep
    that clamp visible rather than silently masking it as the old ``max(1, int())``
    did (#325)."""
    try:
        count = int(physical_cpus or 0)
    except (TypeError, ValueError):
        count = 0
    if count < 1:
        _physical_cores_warn(
            "physical core count resolved to 0; defaulting to 1. "
            "Set OMP_NUM_THREADS to fix single-core decode (#325).")
        return 1
    return count


def cpu_socket_count():
    """Return the number of physical CPU sockets visible to this process."""
    if not sys.platform.startswith("linux"):
        return 1
    try:
        result = subprocess.run(["lscpu", "-p=socket"], text=True,
                                capture_output=True, check=True, timeout=5)
        sockets = {int(line) for line in result.stdout.splitlines()
                   if line and not line.startswith("#")}
        if sockets:
            return len(sockets)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 1


def _auto_tune(bottleneck_class, projected_hit, gpus, cpu_sockets, plan_has_metal):
    """Derive tuning knobs from the bottleneck classification."""
    tune = {}
    has_gpu = bool(gpus)
    n_gpu = len(gpus)

    # MTP: costs more than it saves when compute-bound (#389 measured 42% loss)
    # or streaming-bound (#467 measured 32% loss under CUDA at 85% hit).
    # EXCEPTION: an explicit COLI_CUDA_MTP=1 in the environment is a documented
    # opt-in to test speculation under CUDA (glm.c resolves DRAFT=-1 -> 3 only
    # when it sees the var). Exporting DRAFT=0 here preempted that auto path,
    # so the opt-in was silently inert on the Windows bare-run/auto-tier flows
    # (#467): respect it and let the engine's auto path take over. Unset still
    # gets DRAFT=0 -> MTP off, which is the measured-correct default.
    if os.environ.get("COLI_CUDA_MTP") == "1":
        pass  # explicit opt-in: leave DRAFT to the engine's auto resolution
    elif bottleneck_class == "compute":
        tune["DRAFT"] = {"value": "0",
                         "reason": "compute-bound: MTP batch overhead exceeds yield"}
    elif bottleneck_class == "disk" and projected_hit < 0.90:
        tune["DRAFT"] = {"value": "0",
                         "reason": "low hit rate: MTP widens expert union, adds disk reads"}
    # otherwise leave DRAFT unset (engine default: auto)

    # PIPE: resident pipeline mode depends on GPU count
    if has_gpu and n_gpu == 1:
        tune["COLI_CUDA_PIPE"] = {"value": "1",
                                  "reason": "single GPU: S=1 pipeline gate"}
    elif has_gpu and n_gpu > 1:
        tune["COLI_CUDA_PIPE"] = {"value": "2",
                                  "reason": "multi-GPU: residual stays on-device across layers"}
    elif not has_gpu and bottleneck_class == "disk":
        tune["PIPE"] = {"value": "1",
                        "reason": "overlap disk reads with resident expert compute"}

    # NUMA: selective interleave for GPU hosts, blanket hint for CPU-only
    if cpu_sockets > 1 and has_gpu:
        tune["COLI_NUMA"] = {"value": "1",
                             "reason": "multi-socket + GPU: interleave expert slabs, protect DMA buffers"}
    elif cpu_sockets > 1 and not has_gpu:
        tune["COLI_NUMA"] = {"value": "1",
                             "reason": "multi-socket CPU-only: interleave expert slabs across nodes"}
        tune["_numa_hint"] = "numactl --interleave=all may perform better on CPU-only hosts"

    # OMP: kill hot-thread spin when GPU/Metal owns the power budget
    if plan_has_metal:
        tune["COLI_NO_OMP_TUNE"] = {"value": "1",
                                    "reason": "Metal: OMP spin-wait steals GPU power budget"}

    # PIN: fully resident if RAM allows and no GPU tier competes
    if projected_hit >= 0.99 and not has_gpu:
        tune["PIN_GB"] = {"value": "all",
                          "reason": "enough RAM for full expert residency"}

    return tune


POLICIES = {
    "quality": {"preserve_quantization": True, "preserve_router": True},
    "balanced": {"preserve_quantization": True, "preserve_router": True},
    "experimental-fast": {"preserve_quantization": False, "preserve_router": False},
}


def build_plan(model, ram_gb=0, context=4096, gpu_indices=None, vram_gb=0,
               available_memory=None, available_disk=None, gpus=None,
               policy="quality", physical_cpus=None, cpu_sockets=None):
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    info = analyze_model(model)
    physical_cpus = physical_cpu_count() if physical_cpus is None else physical_cpus
    cpu_sockets = cpu_socket_count() if cpu_sockets is None else cpu_sockets
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
    layers = int(cfg.get("num_hidden_layers", 0)) + 1
    if cfg.get("model_type") == "hy_v3" or cfg.get("kv_lora_rank") is None:
        hd = int(cfg.get("head_dim") or cfg.get("hidden_size", 0) // max(1, int(cfg.get("num_attention_heads", 1))))
        n_kv = int(cfg.get("num_key_value_heads") or cfg.get("num_attention_heads", 0))
        n_heads = int(cfg.get("num_attention_heads", 0))
        kv_bytes = layers * context * n_kv * hd * 2 * 4
        kv_buffer = context * n_heads * hd * 4
    else:
        kv_bytes = layers * context * (int(cfg.get("kv_lora_rank", 0)) +
                                       int(cfg.get("qk_rope_head_dim", 0))) * 4
        kv_buffer = context * int(cfg.get("num_attention_heads", 0)) * (
            int(cfg.get("qk_nope_head_dim", 0)) + int(cfg.get("v_head_dim", 0))) * 4
    runtime_bytes = int(1.2 * GB + 2.5 * GB + 64 * typical + kv_bytes + kv_buffer)
    cache_bytes = max(0, ram_budget - info["dense_bytes"] - runtime_bytes)
    per_cap = info["per_cap_bytes"]
    configured_experts = int(cfg.get("n_routed_experts") or cfg.get("num_experts") or 0)
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

    total_expert = info["expert_bytes"]
    resident_expert = hot_bytes + warm_bytes
    projected_hit = resident_expert / total_expert if total_expert else 1.0

    if cold_bytes:
        bottleneck = "disk expert misses"
        bottleneck_class = "disk"
    elif warm_bytes and gpus:
        bottleneck = "CPU expert tail and GPU compute"
        bottleneck_class = "mixed"
    elif projected_hit >= 0.99:
        if gpus:
            bottleneck = "GPU compute and interconnect"
        else:
            bottleneck = "CPU expert compute (fully resident)"
        bottleneck_class = "compute"
    else:
        bottleneck = "CPU expert compute and RAM bandwidth"
        bottleneck_class = "memory"

    tune = _auto_tune(bottleneck_class, projected_hit, gpus, cpu_sockets,
                      plan_has_metal=False)
    probe_state, probe_gbs = ssd_probe_state(info["path"])

    return {
        "version": 2,
        "policy": {"name": policy, **POLICIES[policy],
                   "quality_preserving": policy != "experimental-fast"},
        "model": {key: value for key, value in info.items() if key != "config"},
        "cpu": {"physical_cores": _resolve_physical_cores(physical_cpus),
                "sockets": max(1, int(cpu_sockets)),
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
        "bottleneck_class": bottleneck_class,
        "projected_hit_rate": round(projected_hit, 4),
        "tune": tune,
        "decisions": [
            {"target": "VRAM", "reason": "profile-ranked hot experts"},
            {"target": "RAM", "reason": "warm experts execute on CPU without quality loss"},
            {"target": "Disk", "reason": "immutable recovery source for cold experts"},
        ],
        "warnings": warnings,
        # #379: read-only surfacing of the cached Metal-cache storage probe, if
        # the engine has already measured this model dir. gbs is None unless
        # the engine itself would trust the cache; the state says WHY (#386 r2,
        # F10) -- never re-measured or guessed here.
        "ssd_probe_gbs": probe_gbs,
        "ssd_probe_state": probe_state,
    }


def environment_for_plan(plan, env=None, cuda_enabled=True):
    """Apply a plan without overriding explicit user environment settings."""
    result = dict(env or {})
    result.setdefault("COLI_POLICY", plan["policy"]["name"])
    result.setdefault("OMP_NUM_THREADS", str(plan["cpu"]["physical_cores"]))
    # NOTE: we intentionally do NOT set OMP_PROC_BIND / OMP_PLACES here.
    # The engine's own hot-thread tuning (glm.c main(), the COLI_OMP_TUNED
    # self-exec) sets OMP_PROC_BIND=close with overwrite=0 -- it prefers
    # packing the team onto adjacent cores for the tiny back-to-back per-expert
    # matmuls. Pre-setting OMP_PROC_BIND=spread here ran first and won (the
    # engine's overwrite=0 setenv could not override an already-set var), and
    # spread + OMP_PLACES=cores collapsed the team to one CPU on some libgomp /
    # multi-socket topologies (#325: --auto-tier pinned decode to 1 core on a
    # 64-core box even with OMP_NUM_THREADS=64). Leaving affinity to the engine
    # makes --auto-tier match the plain (working) path. A user who wants a
    # specific policy can still set OMP_PROC_BIND/OMP_PLACES in the environment
    # themselves -- setdefault above only covers OMP_NUM_THREADS.
    tune = plan.get("tune", {})
    for key, entry in tune.items():
        if key.startswith("_"):
            continue
        result.setdefault(key, entry["value"])
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
    if plan.get("ssd_probe_gbs") is not None:
        lines.append(f"ssd    {plan['ssd_probe_gbs']:.1f} GB/s F_NOCACHE (cached probe, #379)")
    elif plan.get("ssd_probe_state") in SSD_PROBE_PENDING:
        lines.append(f"ssd    {SSD_PROBE_PENDING[plan['ssd_probe_state']]}")
    lines.append(f"limit  {plan['expected_bottleneck']}")
    hit = plan.get("projected_hit_rate", 0)
    lines.append(f"hit    {hit:.0%} projected expert residency")
    tune = plan.get("tune", {})
    if tune:
        lines.append("")
        lines.append("auto-tune:")
        for key, entry in tune.items():
            if key.startswith("_"):
                continue
            lines.append(f"  {key}={entry['value']:12s} {entry['reason']}")
        hint = tune.get("_numa_hint")
        if hint:
            lines.append(f"  hint: {hint}")
    lines.extend(f"warn   {warning}" for warning in plan["warnings"])
    return "\n".join(lines)
