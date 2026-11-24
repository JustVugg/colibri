#!/usr/bin/env python3
"""Which model runs best on this machine, and how fast it decodes there: the choice `coli setup`
(START-HERE) offers first, and the speeds its menu shows.

A model's decode speed is the bytes a token reads over the bandwidth that serves them, plus a
fixed cost a layer (the kernels and the small operations between the matrices), over the tokens
a forward gives (an MTP head drafts). Where the bytes come from is where the model fits:

  gpu     every weight on the card: its memory's bandwidth
  hybrid  the dense part and the experts the card holds on it, the other experts in RAM, which
          the CPU runs at the RAM's bandwidth (the expert tiers balance the two)
  ram     every weight in RAM, the CPU: the RAM's bandwidth
  ssd     the dense part and an expert cache in RAM (and on the card), the misses from the disk

The constants (GPU_EFF, CPU_EFF, FAMILY) are fitted to decode speeds measured on two machines
(CALIBRATION, tests/test_model_fit.py holds the estimates to them). The machine: setup_hw's
report (RAM, cards, CPU, disk), the RAM's read bandwidth measured here (a process a core, each
scanning its own buffer with the C library's memchr: within 3% of numpy's sums on three
machines), a card's bandwidth from the CUDA driver (memory clock x bus width) or from its name,
and the disk's kind (NVMe, SSD, HDD). Standard library only: the setup runs before anything
else is installed.
"""
import json
import os
import re
import subprocess
import sys
import time

GB = 1_000_000_000

#: A forward's share of a card's peak bandwidth, by backend (CALIBRATION: Inkling-2-Mini's two
#: quantizations on CUDA part at 310 GB/s of the RTX PRO 4000 SFF's 432; Vulkan's chain reads
#: Qwen3.6 at ~0.6 of it)
GPU_EFF = {"cuda": 0.72, "vulkan": 0.62}
#: The CPU's share of the RAM's measured bandwidth in a decode (Qwen3.6 int4 on ds: 19.8 tok/s
#: from 2.5 GB a token at 61.8 GB/s)
CPU_EFF = 0.8
#: An integrated GPU reads the same RAM: what it changes is the compute. Measured on a Radeon 780M
#: against its Ryzen 7 PRO 8700GE (docs/vulkan.md): decode on the GPU over decode on the CPU, for
#: that GPU only (another integrated one is estimated as its CPU).
IGPU_NAME = "780M"
IGPU_RATIO = {"qwen36": 9.94 / 6.01, "qwen38": 3.81 / 3.49}
#: A family's fixed cost a layer in ms (on the card by backend, on the CPU), and the tokens a
#: forward gives with the head its downloads carry. Layers are the ones a token runs (a loop's
#: passes counted). Fitted to CALIBRATION; a family without measurements takes DEFAULT_FAMILY.
FAMILY = {
    "qwen36": {"gpu_ms": {"cuda": 0.035, "vulkan": 0.02}, "cpu_ms": 0.0},
    "qwen38": {"gpu_ms": {"cuda": 0.05, "vulkan": 0.15}, "cpu_ms": 0.1, "tpf": 1.45},
    "inkling2": {"gpu_ms": {"cuda": 0.35, "vulkan": 0.33}, "cpu_ms": 1.0},   # Inkling-2's loop and mixes
}
DEFAULT_FAMILY = {"gpu_ms": {"cuda": 0.08, "vulkan": 0.12}, "cpu_ms": 0.3}
#: The families whose engines quantize a checkpoint's 16-bit dense matrices to int8 as they load
#: them: they hold and read half those bytes (ui/models.py's DENSE_I8_AT_LOAD).
DENSE_I8_AT_LOAD = {"qwen36"}

#: Random 1 MiB reads, 16 in flight, by the disk's kind (GB/s): how fast the expert misses come
DISK_GBS = {"nvme": 2.0, "ssd": 0.45, "hdd": 0.12, None: 0.8}
#: Memory a run keeps for itself past the weights: the system and the program in RAM, the
#: driver's context, the KV cache and scratch on a card
RAM_KEEP, RAM_SHARE = 2 * GB, 0.9
VRAM_KEEP = 2 * GB

#: Decode speeds a pick holds to: smooth chat (a thinking model's reasoning too: a thousand tokens
#: in under a minute), and the least worth offering first
SMOOTH, USABLE = 20.0, 4.0
#: The share of its experts a token reads, for an entry without its own figures (the catalog's
#: MoE models read 2-4%: 8 of 256, 10 of 512, 8 of 128)
ROUGH_ACTIVE = 0.04
#: The RAM's bandwidth assumed where nothing measured it (GB/s): a dual-channel desktop's
DEFAULT_RAM_GBS = 30.0

#: A card's peak memory bandwidth (GB/s) by its name, for a card the CUDA driver does not
#: describe (AMD, Intel, an NVIDIA card without the driver's answer). The first match wins:
#: the longer names first. Laptop parts are slower than the desktop cards of the same number.
CARD_GBS = (
    (r"rtx pro 6000", 1792), (r"rtx pro 5000", 1344), (r"rtx pro 4500", 896),
    (r"rtx pro 4000 .*sff|rtx pro 4000 sff", 432), (r"rtx pro 4000", 672),
    (r"rtx 6000 ada", 960), (r"rtx 5000 ada", 576), (r"rtx 4500 ada", 432),
    (r"rtx 4000 sff ada|rtx 4000 ada sff", 280), (r"rtx 4000 ada", 360),
    (r"rtx a6000|a6000", 768), (r"rtx a5000|a5000", 768), (r"rtx a4000|a4000", 448),
    (r"rtx a2000|a2000", 288),
    (r"5090 laptop", 896), (r"5080 laptop", 896), (r"5070 ti laptop", 672), (r"5070 laptop", 384),
    (r"5060 laptop", 384), (r"4090 laptop", 576), (r"4080 laptop", 432), (r"4070 laptop", 256),
    (r"4060 laptop", 256), (r"4050 laptop", 192), (r"3080 ti laptop", 512), (r"3080 laptop", 448),
    (r"3070 ti laptop", 448), (r"3070 laptop", 448), (r"3060 laptop", 336), (r"3050 laptop", 192),
    (r"rtx 5090", 1792), (r"rtx 5080", 960), (r"rtx 5070 ti", 896), (r"rtx 5070", 672),
    (r"rtx 5060 ti", 448), (r"rtx 5060", 448),
    (r"rtx 4090", 1008), (r"rtx 4080 super", 736), (r"rtx 4080", 717), (r"rtx 4070 ti super", 672),
    (r"rtx 4070 ti", 504), (r"rtx 4070 super", 504), (r"rtx 4070", 504), (r"rtx 4060 ti", 288),
    (r"rtx 4060", 272),
    (r"rtx 3090 ti", 1008), (r"rtx 3090", 936), (r"rtx 3080 ti", 912), (r"rtx 3080", 760),
    (r"rtx 3070 ti", 608), (r"rtx 3070", 448), (r"rtx 3060 ti", 448), (r"rtx 3060", 360),
    (r"rtx 3050", 224), (r"rtx 2080 ti", 616), (r"rtx 2080", 448), (r"rtx 2070", 448),
    (r"rtx 2060", 336), (r"gtx 1080 ti", 484), (r"gtx 1080", 320), (r"gtx 1070", 256),
    (r"gtx 16[56]0", 192),
    (r"h200", 4800), (r"h100.*(sxm|hbm3)", 3350), (r"h100", 2000), (r"a100.*80", 2039), (r"a100", 1555),
    (r"l40s|l40", 864), (r"\bl4\b", 300), (r"\ba10\b", 600), (r"\ba40\b", 696), (r"v100", 900),
    (r"\bt4\b", 320),
    (r"9070 xt|9070", 640), (r"9060 xt", 320), (r"7900 xtx", 960), (r"7900 xt", 800),
    (r"7900 gre", 576), (r"7800 xt", 624), (r"7700 xt", 432), (r"7600 xt|7600", 288),
    (r"6950 xt", 576), (r"6900 xt|6800 xt|6800", 512), (r"6750 xt", 432), (r"6700 xt", 384),
    (r"6650 xt", 280), (r"6600 xt", 256), (r"6600", 224), (r"w7900", 864), (r"mi300", 5300),
    (r"mi250", 3200),
    (r"arc.*a770", 560), (r"arc.*a750|arc.*a580", 512), (r"arc.*a380", 186), (r"arc.*b580", 456),
    (r"arc.*b570", 380),
)


def card_gbs(name, vram_bytes=None):
    """A card's peak bandwidth from its name, else from its memory's size (a card of that size's
    usual bus); None for no name and no size."""
    text = (name or "").lower()
    for pattern, gbs in CARD_GBS:
        if re.search(pattern, text):
            return float(gbs)
    if not vram_bytes:
        return None
    gb = vram_bytes / GB
    for below, gbs in ((6, 192), (10, 300), (14, 400), (20, 500), (30, 800)):
        if gb < below:
            return float(gbs)
    return 1200.0


def driver_gbs(device):
    """Peak bandwidth from the CUDA driver's memory clock (kHz) and bus width (bits), the two
    attributes setup_hw's probe reads: two transfers a clock (GDDR and HBM alike)."""
    clock, bits = (device or {}).get("memory_clock_khz"), (device or {}).get("memory_bus_bits")
    if not clock or not bits:
        return None
    return 2.0 * clock * 1e3 * bits / 8 / GB


# ---------------------------------------------------------------- the RAM's bandwidth

_CHILD = r'''
import sys, time
n, secs = int(sys.argv[1]), float(sys.argv[2])
buf = b"\x02" * n          # written: a page the system has not handed out reads as a cached zero
buf.find(b"\x01")
print("ready", flush=True)
sys.stdin.readline()
done, t0 = 0, time.perf_counter()
while time.perf_counter() - t0 < secs:
    buf.find(b"\x01")      # memchr over the whole buffer: no byte matches
    done += n
print(done / (time.perf_counter() - t0), flush=True)
'''


def measure_ram_bandwidth(cpus=None, avail_bytes=None, mb=128, seconds=0.6, runs=2):
    """The RAM's read bandwidth in GB/s, the best of `runs`: a process a logical CPU, each
    scanning its own `mb` MiB with memchr (past any cache), started together. Fewer processes
    where the free RAM is short (a quarter of it at most). None if it cannot run."""
    cpus = cpus or os.cpu_count() or 2
    procs = cpus
    if avail_bytes:
        procs = max(1, min(cpus, int(avail_bytes * 0.25 // (mb << 20))))
    best = None
    for _ in range(runs):
        children = []
        try:
            for _ in range(procs):
                children.append(subprocess.Popen([sys.executable, "-c", _CHILD, str(mb << 20), str(seconds)],
                                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True))
            for c in children:
                if (c.stdout.readline() or "").strip() != "ready":
                    raise OSError("a probe did not start")
            for c in children:
                c.stdin.write("go\n")
                c.stdin.flush()
            total = sum(float(c.communicate(timeout=30)[0]) for c in children)
        except (OSError, ValueError, subprocess.SubprocessError):
            for c in children:
                try:
                    c.kill()
                except OSError:
                    pass
            continue
        best = max(best or 0, total / GB)
    return round(best, 1) if best else None


# ---------------------------------------------------------------- the disk

def _block_kind(d):
    """One block device's kind, from its sysfs folder."""
    if os.path.basename(d).startswith("nvme"):
        return "nvme"
    try:
        with open(os.path.join(d, "queue", "rotational")) as f:
            rot = f.read().strip() == "1"
    except OSError:
        return None
    if not rot:
        return "ssd"
    try:
        with open(os.path.join(d, "device", "model")) as f:
            model = f.read().strip()
    except OSError:
        model = ""
    # Hyper-V's virtual disk reads rotational whatever holds it: WSL2's own disk on an NVMe drive
    return None if model == "Virtual Disk" else "hdd"


def _linux_disk_kind(path):
    try:
        st = os.stat(path)
        dev = "/sys/dev/block/%d:%d" % (os.major(st.st_dev), os.minor(st.st_dev))
        real = os.path.realpath(dev)
    except (OSError, ValueError):
        return None
    if not os.path.isdir(real):
        return None
    disk = real if os.path.exists(os.path.join(real, "queue")) else os.path.dirname(real)
    members = []
    slaves = os.path.join(disk, "slaves")
    if os.path.isdir(slaves):      # md RAID, LVM, dm-crypt: the disks under it
        for s in os.listdir(slaves):
            m = os.path.realpath(os.path.join(slaves, s))
            members.append(m if os.path.exists(os.path.join(m, "queue")) else os.path.dirname(m))
    kinds = [_block_kind(d) for d in members or [disk]]
    for k in ("hdd", "ssd", None, "nvme"):   # the slowest member sets the pace
        if k in kinds:
            return k
    return None


def _windows_disk_kind(path):
    drive = os.path.splitdrive(os.path.abspath(path))[0].rstrip(":")
    if not re.fullmatch(r"[A-Za-z]", drive or ""):
        return None
    script = ("$n=(Get-Partition -DriveLetter %s -ErrorAction Stop).DiskNumber;"
              "$d=Get-PhysicalDisk | Where-Object DeviceId -eq $n | Select-Object -First 1;"
              "Write-Output \"$($d.MediaType)|$($d.BusType)\"" % drive)
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                             capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    media, _, bus = out.partition("|")
    if bus.strip().lower() == "nvme":
        return "nvme"
    return {"ssd": "ssd", "hdd": "hdd"}.get(media.strip().lower())


def disk_kind(path):
    """nvme, ssd, hdd or None (unknown: a network share, WSL's view of a Windows drive or its
    own virtual disk)."""
    path = path or "."
    while path and not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    if sys.platform == "win32":
        return _windows_disk_kind(path)
    if sys.platform == "darwin":
        return "nvme"           # every Apple Silicon Mac's internal drive
    return _linux_disk_kind(path)


# ---------------------------------------------------------------- the machine

def machine_from_hw(hw, ram_gbs=None, disk=None, cuda_devices=None):
    """setup_hw's report as what the estimates read: RAM (bytes), its bandwidth (GB/s), the cards
    (discrete ones with their memory and bandwidth, an integrated one), the disk's kind."""
    mem = hw.get("memory") or {}
    cards = []
    by_slot = {}
    for d in cuda_devices or []:
        by_slot[(d.get("pci_bus_id") or "").lower().lstrip("0")] = d
        by_slot[d.get("name")] = d
    for g in (hw.get("gpu") or {}).get("nvidia") or hw.get("nvidia") or []:
        dev = by_slot.get((g.get("pci_bus_id") or "").lower().lstrip("0")) or by_slot.get(g.get("name"))
        gbs = driver_gbs(dev) or card_gbs(g.get("name"), g.get("total_bytes"))
        cards.append({"name": g.get("name"), "vendor": "NVIDIA", "kind": "discrete",
                      "vram": g.get("total_bytes") or 0, "gbs": gbs, "source": "driver" if driver_gbs(dev) else "name"})
    vk = (hw.get("gpu") or {}).get("vulkan")
    integrated = None
    if vk and vk.get("type") == "discrete" and (vk.get("vendor") or "") != "NVIDIA":
        name = re.sub(r"^Microsoft Direct3D12 \((.+)\)$", r"\1", vk.get("name") or "")
        cards.append({"name": name, "vendor": vk.get("vendor"), "kind": "discrete",
                      "vram": vk.get("device_local_bytes") or 0,
                      "gbs": card_gbs(name, vk.get("device_local_bytes")), "source": "name"})
    elif vk and vk.get("type") == "integrated":
        integrated = re.sub(r"^Microsoft Direct3D12 \((.+)\)$", r"\1", vk.get("name") or "")
    return {"ram": mem.get("total") or 0, "ram_avail": mem.get("available") or 0,
            "ram_gbs": ram_gbs, "cards": cards, "integrated": integrated,
            "disk": disk, "disk_gbs": DISK_GBS.get(disk, DISK_GBS[None]),
            "cpu": (hw.get("cpu") or {}).get("name"),
            "cores": (hw.get("cpu") or {}).get("physical_cores") or (hw.get("cpu") or {}).get("logical_cores")}


# ---------------------------------------------------------------- a model's bytes

def geometry(entry):
    """What a token costs a catalog entry, as the engine holds it (GB): resident (dense part,
    embeddings, heads: always in memory), experts (all of them), the dense bytes and the expert
    bytes a token reads, the layers it runs, the tokens a forward gives. An entry without these
    figures (one from COLI_SETUP_CATALOG) gets rough ones from its sizes: its dense part resident
    and read whole, the rest of the download experts of which a token reads ROUGH_ACTIVE, 48
    layers. None for an image model."""
    fam = FAMILY.get(getattr(entry, "cost", "") or entry.family, DEFAULT_FAMILY)
    if not getattr(entry, "active_gb", 0):
        if getattr(entry, "modality", "text") != "text" or not getattr(entry, "disk_gb", 0):
            return None
        dense = min(entry.dense_gb, entry.disk_gb)
        experts = max(0.0, entry.disk_gb - dense)
        return {"resident": dense * GB, "experts": experts * GB, "dense": dense * GB,
                "expert_read": experts * ROUGH_ACTIVE * GB, "layers": 48, "tpf": fam.get("tpf", 1.0), "rough": True}
    return {"resident": entry.resident_gb * GB, "experts": entry.experts_gb * GB,
            "dense": entry.dense_read_gb * GB, "expert_read": (entry.active_gb - entry.dense_read_gb) * GB,
            "layers": entry.layers, "tpf": entry.tpf or fam.get("tpf", 1.0)}


def _hit(fraction):
    """A cache holding `fraction` of the experts serves more of the reads than that: the routers
    pick some experts far more often (a square root fits GLM-5.3-Flash's and DeepSeek V4 Flash's
    streamed runs, CALIBRATION)."""
    return 1.0 if fraction >= 1 else max(0.0, fraction) ** 0.5


def estimate(entry, machine, backend):
    """How `entry` runs on `machine` with `backend` (cuda, vulkan or cpu): {"tier", "tok_s",
    "where"} (gpu, hybrid, ram or ssd; decode tokens a second), or {"tier": None, "why"} when it
    does not fit anywhere here."""
    g = geometry(entry)
    if g is None:
        return {"tier": None, "why": "no figures for this model"}
    fam = FAMILY.get(getattr(entry, "cost", "") or entry.family, DEFAULT_FAMILY)
    ram = max(0.0, machine["ram"] * RAM_SHARE - RAM_KEEP)
    ram_gbs = (machine.get("ram_gbs") or DEFAULT_RAM_GBS) * GB
    disk_gbs = (machine.get("disk_gbs") or DISK_GBS[None]) * GB
    cpu_bw = ram_gbs * CPU_EFF
    cpu_fixed = g["layers"] * fam["cpu_ms"] / 1000
    total = g["resident"] + g["experts"]
    cards = [c for c in machine.get("cards") or [] if c.get("vram")]
    if backend in ("cuda", "vulkan") and cards:
        vram = sum(c["vram"] for c in cards) - VRAM_KEEP * len(cards)
        gbs = min(c.get("gbs") or card_gbs(c.get("name"), c["vram"]) or 300.0 for c in cards) * GB
        bw = gbs * GPU_EFF[backend]
        fixed = g["layers"] * fam["gpu_ms"].get(backend, DEFAULT_FAMILY["gpu_ms"][backend]) / 1000
        if total <= vram:
            t = (g["dense"] + g["expert_read"]) / bw + fixed
            return {"tier": "gpu", "tok_s": round(g["tpf"] / t, 1),
                    "where": "every weight on the card (%.0f of %.0f GB)" % (total / GB, vram / GB)}
        if g["resident"] <= vram:
            on_card = vram - g["resident"]
            rest = g["experts"] - on_card
            h_card = _hit(on_card / g["experts"])
            in_ram = min(rest, max(0.0, ram - g["resident"]))      # the other experts, as RAM holds them
            h_ram = _hit((on_card + in_ram) / g["experts"]) - h_card
            miss = max(0.0, 1.0 - h_card - h_ram)
            t = (g["dense"] + g["expert_read"] * h_card) / bw + g["expert_read"] * h_ram / cpu_bw \
                + g["expert_read"] * miss / disk_gbs + fixed
            tier = "hybrid" if miss < 0.01 else "ssd"
            where = ("the dense part and %.0f GB of experts on the card, %.0f GB in RAM"
                     % (on_card / GB, in_ram / GB)) + ("" if tier == "hybrid" else ", the rest from the disk")
            return {"tier": tier, "tok_s": round(g["tpf"] / t, 2 if tier == "ssd" else 1), "where": where}
    if total <= ram:
        t = (g["dense"] + g["expert_read"]) / cpu_bw + cpu_fixed
        tok = g["tpf"] / t
        where = "every weight in RAM (%.0f of %.0f GB), the CPU" % (total / GB, ram / GB)
        if backend == "vulkan" and machine.get("integrated"):
            if IGPU_NAME in machine["integrated"]:
                tok *= IGPU_RATIO.get(entry.family, 1.0)
            where = "every weight in RAM, the integrated GPU (%s)" % machine["integrated"]
        return {"tier": "ram", "tok_s": round(tok, 1), "where": where}
    if g["resident"] < ram:
        cache = ram - g["resident"]
        h = _hit(cache / g["experts"])
        t = (g["dense"] + g["expert_read"] * h) / cpu_bw + g["expert_read"] * (1 - h) / disk_gbs + cpu_fixed
        return {"tier": "ssd", "tok_s": round(g["tpf"] / t, 2),
                "where": "the dense part and %.0f GB of experts in RAM, the rest from the disk" % (cache / GB)}
    return {"tier": None, "why": "its dense part (%.0f GB) does not fit in RAM" % (g["resident"] / GB)}


def sure_speed(est):
    """The speed a pick counts on: an estimate that streams from the disk at half, since how often
    a token misses the cache depends on the prompts (the streamed anchors part by up to 66%,
    CALIBRATION), while the others came within a fifth of the measurements."""
    return est["tok_s"] / (2.0 if est["tier"] == "ssd" else 1.0)


def worth(row):
    """What a row is worth as the first offer: its capability (the catalog's rank) times the square
    of how far below smooth chat it decodes (1 from SMOOTH up, sure_speed). A model just under
    SMOOTH loses little to one just over it; one at a third of it keeps a ninth of its rank."""
    return row["entry"].rank * min(1.0, sure_speed(row["est"]) / SMOOTH) ** 2


def pick(rows):
    """The row `coli setup` offers first among those that fit, each with its estimate in
    row["est"]: the one worth the most (worth), a tie to the faster, then the smaller download.
    None when nothing fits."""
    text = [r for r in rows if r["fits"] and r["entry"].modality == "text" and (r.get("est") or {}).get("tier")]
    if not text:
        return None
    return max(text, key=lambda r: (worth(r), r["est"]["tok_s"], -r["entry"].disk_gb))


def describe(machine):
    """The machine as the menu's header says it."""
    parts = ["RAM %.0f GB%s" % (machine["ram"] / GB, " at %.0f GB/s" % machine["ram_gbs"] if machine.get("ram_gbs") else "")]
    for c in machine.get("cards") or []:
        parts.append("%s %.0f GB%s" % (c["name"], c["vram"] / GB, " at %.0f GB/s" % c["gbs"] if c.get("gbs") else ""))
    if machine.get("integrated"):
        parts.append("integrated GPU %s" % machine["integrated"])
    if machine.get("disk"):
        parts.append("disk %s" % {"nvme": "NVMe", "ssd": "SSD", "hdd": "HDD"}[machine["disk"]])
    return ", ".join(parts)


if __name__ == "__main__":   # `python3 model_fit.py`: this machine as the setup sees it
    import setup_hw
    hw = setup_hw.detect(os.path.expanduser("~"))
    probe = setup_hw._probe_cuda_child() if hw.get("nvidia") else None
    m = machine_from_hw(hw, measure_ram_bandwidth(avail_bytes=(hw.get("memory") or {}).get("available")),
                        disk_kind(os.path.expanduser("~")), (probe or {}).get("devices"))
    print(json.dumps(m, indent=1, default=str))
