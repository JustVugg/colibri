#!/usr/bin/env python3
"""The Models page's service: what fits the machine it runs on (all in RAM, with the GPU,
the experts streamed from the SSD), the speed to expect, the models on this disk, our
catalog and the GGUFs on Hugging Face that colibri reads, downloads with progress, and
switching the model the gateway serves.

It runs inside `coli ui` when the gateway is on this machine (the default), or on its own
on the engine's machine, for `coli ui --models-backend`:

    python3 ui/models.py [--backend http://127.0.0.1:8000] [--models-dir DIR] [--port 8791]

Everything is read on the machine it runs on: the CPU, the RAM, the GPUs and the free disk
from colibri's own probe (setup_hw), the RAM bandwidth and the SSD's random reads measured
here. Speeds are labelled: "measured" is the median of the gateway's own finished turns with
that model, on the CPU or with the GPU; "estimate" is the bytes a token reads over the
bandwidth measured here, never shown as measured.

Switching restarts the running `coli serve` with its own command, folder and environment,
only --model changed (on Linux, where /proc names the process that holds the port); when the
new model does not start, the previous one is started again. Standard library only (numpy,
when it is there, measures the RAM bandwidth)."""
import argparse
import json
import os
import random
import re
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
_UP = os.path.dirname(HERE)
# the release archive keeps coli next to ui/, a source checkout keeps it in c/
COLI_DIR = _UP if os.path.isfile(os.path.join(_UP, "coli")) else os.path.join(_UP, "c")
sys.path.insert(0, COLI_DIR)
import setup_catalog  # noqa: E402
import setup_download  # noqa: E402
import setup_flow  # noqa: E402
import setup_hw  # noqa: E402

GB = 1_000_000_000
LINUX = sys.platform.startswith("linux")
# The share of the measured RAM bandwidth a decode forward reaches, until this machine's own
# turns on the CPU calibrate it (calibrate below). The one measured: Qwen3.6 int4-g64 on an
# i5-13500 writes 13.8 tok/s reading 2.5 GB a token, 0.9 of the 41 GB/s measured there; a little
# under it for the engines not measured. An estimate, not a fact.
ETA = 0.8
# The engines that quantize a checkpoint's 16-bit (and f32) dense matrices to int8 as they load
# them (qwen36.c, COLI_DENSE_I8, on by default): they hold and read half the bytes on the disk.
DENSE_I8_AT_LOAD = {"qwen36"}
# Decode on a Radeon 780M against the CPU of the same Ryzen 7 PRO 8700GE (docs/vulkan.md, "The
# chain on a Radeon 780M"). An integrated GPU reads the same RAM, so what changes is the compute;
# the ratio is applied only to that GPU. Which engines use an integrated GPU at all is
# setup_flow.VULKAN_IGPU_MEASURED, the rule `coli setup` follows.
IGPU_NAME = "780M"
IGPU_RATIO = {"qwen36": {"model": "Qwen3.6", "gpu": 9.94, "cpu": 6.01},
              "qwen38": {"model": "Qwen3.8", "gpu": 3.81, "cpu": 3.49}}
# What the MTP head gives an engine on the CPU, measured: Qwen3.6 13.7-13.9 -> 14.7-14.9 tok/s
# with one draft (#1940 on an i5-13500). A family measured here uses its own tokens per forward.
MTP_GAIN = {"qwen36": 14.8 / 13.8}
# How capable a family is, for the best model each way it can run (higher first).
FAMILY_RANK = {"olmoe": 1, "qwen36": 3, "glm53": 4, "qwen38": 5, "deepseek_v4": 5, "mimo": 6,
               "deepseek_v41": 6, "glm": 8, "inkling": 9, "kimi": 10}
# A GGUF's general.architecture -> the engine family that reads it (tools/gguf_import.py ARCHES)
GGUF_FAMILY = {"qwen35moe": "qwen36", "qwen35": "qwen36", "qwen3moe": "qwen36", "qwen3": "qwen36",
               "flint": "qwen36", "muse-spark": "qwen36", "lfm3": "qwen36", "qwen4exp": "qwen38",
               "deepseek4": "deepseek_v4", "deepseek41": "deepseek_v41", "inkling": "inkling",
               "glm5-next": "glm53", "glm-dsa": "glm", "mistral-large4": "glm", "mimo2": "mimo",
               "kimi-k3": "kimi", "olmoe": "olmoe"}


def gguf_arches():
    """the architectures colibri imports: the importer's own table"""
    try:
        sys.path.insert(0, os.path.join(COLI_DIR, "tools"))
        import gguf_import
        return set(gguf_import.ARCHES)
    except Exception:   # noqa: BLE001 - a tree without tools/: the table above
        return set(GGUF_FAMILY)


# ------------------------------------------------------------------ configuration
class Cfg:
    gateway = "http://127.0.0.1:8000"
    host, port = "127.0.0.1", 8000
    dirs = []            # --models-dir; empty: colibri's own models folder
    state = None


CFG = Cfg()


def configure(backend, models_dirs=(), state=None):
    u = urllib.parse.urlparse(backend or CFG.gateway)
    CFG.gateway = "%s://%s" % (u.scheme or "http", u.netloc)
    CFG.host, CFG.port = u.hostname or "127.0.0.1", u.port or (443 if u.scheme == "https" else 80)
    CFG.dirs = [os.path.realpath(os.path.expanduser(d)) for d in models_dirs if d]
    CFG.state = state or os.path.join(setup_flow.setup_home(), "ui")
    os.makedirs(CFG.state, exist_ok=True)
    for name, into in (("machine.json", MACHINE), ("geometry-v2.json", GEO), ("measured.json", MEASURED)):
        into.clear()
        into.update(jload(spath(name), {}))


def spath(name):
    return os.path.join(CFG.state, name)


def roots():
    """Where models are: --models-dir, else the folder `coli setup` used (~/colibri-models by
    default), and the folder of the model the gateway serves when it is elsewhere. Downloads
    go to the first."""
    if CFG.dirs:
        out = list(CFG.dirs)
    else:
        cfg = setup_flow.load_config() or {}
        md = cfg.get("model_dir")
        out = [os.path.realpath(os.path.dirname(md) if md else setup_flow.default_models_root())]
    cur = gateway_model_dir()
    if cur and not any(cur.startswith(r + os.sep) for r in out):
        out.append(os.path.dirname(cur))
    return out


def under_roots(path):
    return any(path.startswith(r + os.sep) for r in roots())


# ------------------------------------------------------------------ small helpers
def jload(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def jsave(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def http_json(url, timeout=30, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "colibri-models/1", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_range(url, start, end, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "colibri-models/1", "Range": "bytes=%d-%d" % (start, end)})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def gw_json(path, timeout=5):
    key = (gateway_proc() or {}).get("key") or os.environ.get("COLI_API_KEY")
    return http_json(CFG.gateway + path, timeout=timeout, headers={"Authorization": "Bearer " + key} if key else None)


def gw_idle():
    """no turn in flight or queued: nothing running, or its scheduler empty"""
    try:
        s = gw_json("/health").get("scheduler") or {}
        return (s.get("active", 0) + s.get("queued", 0)) == 0
    except urllib.error.HTTPError as e:   # 503: the engine is down; 401: busy or not, unknown
        return e.code == 503
    except (OSError, ValueError):         # nothing listening
        return True


# ------------------------------------------------------------------ the gateway's process
def _argval(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return None


def _without(argv, flags):
    """argv without these options and their values"""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in flags:
            skip = True
            continue
        if any(a.startswith(f + "=") for f in flags):
            continue
        out.append(a)
    return out


def _listening_inodes(port):
    inodes = set()
    for fn in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(fn) as f:
                rows = f.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            p = row.split()
            if len(p) > 9 and p[3] == "0A" and int(p[1].rsplit(":", 1)[1], 16) == port:
                inodes.add("socket:[%s]" % p[9])
    return inodes


def _holds(pid, inodes):
    try:
        for fd in os.listdir("/proc/%s/fd" % pid):
            try:
                if os.readlink("/proc/%s/fd/%s" % (pid, fd)) in inodes:
                    return True
            except OSError:
                pass
    except OSError:
        pass
    return False


_gw = {"pid": None, "at": 0, "info": None}


def gateway_proc():
    """{pid, argv, cwd, env, key} of the process listening on the gateway's port, when it is
    on this machine and /proc shows it (Linux); None otherwise. Found, it is kept a few seconds."""
    if not LINUX or CFG.host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        return None
    now = time.time()
    if _gw["info"] and now - _gw["at"] < 3:
        return _gw["info"]
    inodes = _listening_inodes(CFG.port)     # nobody listening: known at once, nothing cached
    pid = None
    if inodes:
        if _gw["pid"] and _holds(_gw["pid"], inodes):
            pid = _gw["pid"]
        else:
            pid = next((int(p) for p in os.listdir("/proc") if p.isdigit() and _holds(p, inodes)), None)
    info = None
    if pid:
        try:
            with open("/proc/%d/cmdline" % pid, "rb") as f:
                argv = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
            with open("/proc/%d/environ" % pid, "rb") as f:
                env = dict(x.split("=", 1) for x in f.read().decode(errors="replace").split("\0") if "=" in x)
            info = {"pid": pid, "argv": argv, "cwd": os.readlink("/proc/%d/cwd" % pid), "env": env,
                    "coli": any(os.path.basename(a) == "coli" for a in argv[:3]) and "serve" in argv}
            info["key"] = _argval(argv, "--api-key") or env.get("COLI_API_KEY")
        except OSError:
            info = None
    _gw.update(pid=pid, at=now, info=info)
    return info


def gateway_model_dir():
    p = gateway_proc()
    m = _argval(p["argv"], "--model") if p and p["coli"] else None
    return os.path.realpath(os.path.join(p["cwd"], os.path.expanduser(m))) if m else None


def tier_of(argv, env):
    """ram when the server keeps everything on the CPU, gpu otherwise on a machine with one"""
    if not MACHINE.get("gpus"):
        return "ram"
    gpu = _argval(argv, "--gpu")
    if gpu == "none":
        return "ram"
    if gpu or _argval(argv, "--vram") or env.get("COLI_CUDA") == "1":
        return "gpu"
    return "ram" if env.get("COLI_VULKAN") == "0" else "gpu"


def gateway_tier():
    p = gateway_proc()
    return tier_of(p["argv"], p["env"]) if p and p["coli"] else None


# ------------------------------------------------------------------ the machine
MACHINE = {}
_mlock = threading.Lock()


def detect_machine():
    root = roots()[0]
    hw = setup_hw.detect(setup_hw.existing_ancestor(root), probe_gpu=True)
    cpu, mem, osi = hw.get("cpu") or {}, hw.get("memory") or {}, hw.get("os") or {}
    gpus = [{"name": g.get("name"), "kind": "discrete", "vendor": "NVIDIA", "vram_gb": round((g.get("total_bytes") or 0) / GB, 1)}
            for g in hw.get("nvidia") or []]
    vk = (hw.get("gpu") or {}).get("vulkan")
    if vk and not gpus and vk.get("type") in ("integrated", "discrete"):
        # WSL's Vulkan goes through Direct3D 12: "Microsoft Direct3D12 (the card)"
        d3d = re.fullmatch(r"Microsoft Direct3D12 \((.+)\)", vk.get("name") or "")
        gpus.append({"name": d3d.group(1) + ", via Direct3D 12" if d3d else vk.get("name"), "kind": vk.get("type"),
                     "vendor": vk.get("vendor"),
                     "vram_gb": round((vk.get("device_local_bytes") or 0) / GB, 1)})
    if not gpus and sys.platform == "darwin" and osi.get("machine") in ("arm64", "aarch64"):
        gpus.append({"name": (cpu.get("name") or "Apple") + " GPU", "kind": "integrated", "vendor": "Apple", "vram_gb": 0})
    return {"cpu": cpu.get("name") or cpu.get("arch"), "cores": cpu.get("logical_cores"),
            "physical_cores": cpu.get("physical_cores"), "features": cpu.get("features"),
            "ram_total_gb": round((mem.get("total") or 0) / GB, 1), "ram_avail_gb": round((mem.get("available") or 0) / GB, 1),
            "gpus": gpus, "disk_free_gb": round(shutil.disk_usage(setup_hw.existing_ancestor(root)).free / GB, 1),
            "models_dir": root, "os": osi.get("pretty_name"), "wsl": bool(osi.get("wsl"))}


def measure_ram_bw(seconds=1.5):
    """One thread per core summing its own 64 MiB array: the DRAM read bandwidth a decode sees.
    numpy's sum runs outside the GIL; in plain Python the threads would take turns."""
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError("serve numpy per misurare la banda della RAM") from None
    n = os.cpu_count() or 4
    per = 64 << 20
    arrs = [np.ones(per // 8, dtype=np.float64) for _ in range(n)]
    for a in arrs:
        a.sum()
    counts = [0] * n
    stop = time.time() + seconds

    def work(i):
        a, c = arrs[i], 0
        while time.time() < stop:
            a.sum()
            c += 1
        counts[i] = c
    th = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    return sum(counts) * per / (time.time() - t0) / GB


def biggest_file(dirs, depth=3):
    best = (0, None)
    for root in dirs:
        for dirpath, sub, files in os.walk(root):
            if dirpath[len(root):].count(os.sep) >= depth:
                sub[:] = []
            for f in files:
                p = os.path.join(dirpath, f)
                try:
                    s = os.path.getsize(p)
                except OSError:
                    continue
                if s > best[0] and not os.path.islink(p):
                    best = (s, p)
    return best[1]


def measure_disk(seconds=2.0, block=1 << 20, threads=16):
    """Random 1 MiB direct reads (past the page cache) from the biggest model file, 16 in
    flight: how fast experts come from the SSD. Linux only (O_DIRECT)."""
    import mmap
    if not LINUX:
        raise RuntimeError("le letture dirette dal disco si misurano solo su Linux")
    path = biggest_file([r for r in roots() if os.path.isdir(r)])
    if not path:
        raise RuntimeError("nessun modello sul disco da leggere: rimisura dopo il primo download")
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    tot = [0] * threads
    stop = time.time() + seconds

    def work(i):
        buf, r, n = mmap.mmap(-1, block), random.Random(i), 0
        while time.time() < stop:
            n += os.preadv(fd, [buf], r.randrange(0, max(1, (size - block) // 4096)) * 4096)
        tot[i] = n
    th = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    t0 = time.time()
    try:
        for t in th:
            t.start()
        for t in th:
            t.join()
    finally:
        os.close(fd)
    return sum(tot) / (time.time() - t0) / GB


def refresh_machine(measure=False):
    """the machine again; the two benchmarks too when asked, or when they never ran (and
    the engine is idle: they would slow it down and be slowed down by it)"""
    with _mlock:
        m = detect_machine()
        bench = dict(MACHINE.get("bench") or {})
        if (measure or not bench.get("ram_bw_gbs")) and gw_idle():
            bench = {"at": time.strftime("%Y-%m-%d %H:%M")}
            for key, fn, nd in (("ram_bw_gbs", measure_ram_bw, 1), ("disk_gbs", measure_disk, 2)):
                try:
                    v = fn()
                    if v:
                        bench[key] = round(v, nd)
                except Exception as e:   # noqa: BLE001 - numpy missing, O_DIRECT refused: said, not fatal
                    bench.setdefault("errors", {})[key] = str(e)[:200]
        m["bench"] = bench
        MACHINE.clear()
        MACHINE.update(m)
        jsave(spath("machine.json"), MACHINE)
        return MACHINE


# ------------------------------------------------------------------ geometry: bytes per token
GEO = {}   # geometry-v2.json: v2 keeps the dense part's 16-bit and f32 bytes apart
_glock = threading.Lock()
K_KEYS = ("num_experts_per_tok", "moe_k", "num_experts_per_token", "experts_per_token", "moe_topk", "top_k")
E_KEYS = ("n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts", "n_experts", "num_routed_experts")


def _first(d, keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, int) and v > 0:
            return v
    return None


def geo_from_config(cfg):
    t = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    return {"k": _first(t, K_KEYS) or _first(cfg, K_KEYS), "E": _first(t, E_KEYS) or _first(cfg, E_KEYS),
            "hidden": t.get("hidden_size") or cfg.get("hidden_size"), "vocab": t.get("vocab_size") or cfg.get("vocab_size"),
            "layers": t.get("num_hidden_layers") or cfg.get("num_hidden_layers")}


def classify(name, gguf=False):
    """What a tensor costs a token: routed experts (k of E read), dense (read whole), the
    token embedding (resident, one row read), other lookup tables such as Qwen3.8's n-gram
    embeddings (neither resident nor read whole), the MTP head, a vision tower."""
    n = name.lower()
    if n.startswith("mtp.") or ".mtp." in n or "nextn" in n:
        return "mtp"
    if ("_exps" in n) if gguf else (".experts." in n and "shared" not in n):
        return "experts"
    if "visual" in n or "vision" in n or "mm_projector" in n:
        return "vision"
    if n.startswith("token_embd") or "embed_tokens" in n:
        return "embed"
    if "embd" in n or "embed" in n or "ngram" in n:
        return "lookup"
    return "dense"


def geo_from_sizes(sz):
    return {"experts": sz.get("experts", 0), "dense_active": sz.get("dense", 0), "lookup": sz.get("lookup", 0),
            "resident": sz.get("dense", 0) + sz.get("embed", 0) + sz.get("mtp", 0) + sz.get("vision", 0), "sizes": sz}


def st_header_sizes(raw_header, sz):
    """bytes by kind; the dense part's 16-bit and f32 bytes apart too (an engine may requantize them)"""
    for k, v in json.loads(raw_header).items():
        if k == "__metadata__" or not isinstance(v, dict):
            continue
        a, b = v.get("data_offsets") or (0, 0)
        c = classify(k)
        sz[c] = sz.get(c, 0) + (b - a)
        wide = {"BF16": "dense16", "F16": "dense16", "F32": "dense32"}.get(v.get("dtype")) if c == "dense" else None
        if wide:
            sz[wide] = sz.get(wide, 0) + (b - a)


def st_local(path, sz):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        st_header_sizes(f.read(n), sz)


def st_remote(repo, revision, path, sz):
    url = setup_download.file_url(repo, revision, path)
    head = http_range(url, 0, (256 << 10) - 1)    # most headers fit; a longer one is read to its end
    n = struct.unpack("<Q", head[:8])[0]
    raw = head[8:8 + n] if len(head) >= 8 + n else head[8:] + http_range(url, len(head), 8 + n - 1)
    st_header_sizes(raw, sz)


def _geo_put(key, g):
    with _glock:
        GEO[key] = g
        jsave(spath("geometry-v2.json"), GEO)
    return g


def catalog_geometry(entry, fetch=True):
    """Bytes per token for a catalog model: the tensor sizes from every shard's header (HTTP
    Range reads, nothing downloaded) and k of E from its config.json. With fetch=False (a page
    waiting) only what is cached, or the catalog's rough sizes."""
    key = "%s@%s" % (entry.repo, entry.revision)
    with _glock:
        if key in GEO and (GEO[key].get("source") == "headers" or not fetch):
            return GEO[key]
    resident = entry.dense_gb * GB
    rough = {"resident": resident, "experts": max(0.0, entry.disk_gb * GB - resident), "dense_active": resident * 0.75}
    if not fetch:
        return dict(rough, source="pending")
    g = {"source": "catalog"}
    try:
        g.update(geo_from_config(http_json(setup_download.file_url(entry.repo, entry.revision, "config.json"), timeout=30)))
    except Exception as e:   # noqa: BLE001
        g["error"] = "config.json: %s" % e
    try:
        files = setup_download.list_repo_files(entry.repo, entry.revision, include=entry.include, exclude=entry.exclude)
        shards = [f["path"] for f in files if f["path"].endswith(".safetensors")]
        if not shards:
            raise ValueError("no safetensors")
        sz = {}
        for p in shards:
            st_remote(entry.repo, entry.revision, p, sz)
        g.update(geo_from_sizes(sz), source="headers")
    except Exception as e:   # noqa: BLE001 - the catalog's own sizes, rougher
        g["error"] = (g.get("error", "") + " headers: %s" % e).strip()
        g.update(rough)
    return _geo_put(key, g)


def gguf_parts(first):
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", first)
    return ["%s-%05d-of-%s.gguf" % (m.group(1), i, m.group(3)) for i in range(1, int(m.group(3)) + 1)] if m else [first]


def local_geometry(path):
    """The same for a model folder on this disk: its shards, a linked GGUF, or the expert
    container next to the shards (Qwen3.8's experts-<format>, the one the gateway uses when it
    serves this model, else the smallest)."""
    ptr = jload(os.path.join(path, "gguf.json"), None)
    parts = None
    if ptr and ptr.get("gguf"):       # a linked GGUF can be gone while its folder stays
        parts = gguf_parts(ptr["gguf"] if os.path.isabs(ptr["gguf"]) else os.path.join(path, ptr["gguf"]))
        gone = [p for p in parts if not os.path.isfile(p)]
        if gone:
            raise ValueError("manca il GGUF: %s" % gone[0])
    key = "local:" + path
    with _glock:
        if key in GEO:
            return GEO[key]
    g = {"source": "local"}
    g.update(geo_from_config(jload(os.path.join(path, "config.json"), {})))
    if parts:
        gg = gguf_geometry([(p, os.path.getsize(p), None) for p in parts])
        g.update({k: gg[k] for k in ("experts", "dense_active", "resident", "k", "E", "sizes")})
    else:
        sz = {}
        for f in sorted(os.listdir(path)):
            if f.endswith(".safetensors"):
                st_local(os.path.join(path, f), sz)
        packs = sorted(d for d in os.listdir(path) if d.startswith("experts-") and os.path.isdir(os.path.join(path, d)))
        if packs and not sz.get("experts"):
            p = gateway_proc()
            want = p["env"].get("Q38_EXPERT_LOWBIT") if p and gateway_model_dir() == path else None
            pick = "experts-" + want if want and "experts-" + want in packs else min(packs, key=lambda d: dir_bytes(os.path.join(path, d)))
            sz["experts"] = dir_bytes(os.path.join(path, pick))
        g.update(geo_from_sizes(sz))
    return _geo_put(key, g)


def overhead_s(g):
    """Fixed work a forward does whatever the bytes: sampling over the vocabulary and the
    threads meeting once a block. Small models are bound by it, not by the RAM."""
    return 0.002 + 0.0001 * (g.get("layers") or 40)


def held(g, family):
    """(resident, dense read a token) as the engine holds them, not as the files store them"""
    sz = g.get("sizes") or {}
    cut = sz.get("dense16", 0) / 2 + sz.get("dense32", 0) * 3 / 4 if family in DENSE_I8_AT_LOAD else 0
    return g["resident"] - cut, g["dense_active"] - cut


def per_token(g, family=""):
    if not g.get("k") or not g.get("E"):
        return None
    return held(g, family)[1] + g["experts"] * g["k"] / g["E"]


# Calibration from this machine's own turns on the CPU: the share of the RAM bandwidth a
# forward really reaches here, and the tokens a forward gives with the MTP head.
CAL = {"eta": ETA, "tpf": None, "from": None, "family": None}


def calibrate(rows):
    bw = ((MACHINE.get("bench") or {}).get("ram_bw_gbs") or 0) * GB
    etas, tpfs, names, fams = [], [], [], []   # names: the turns it was calibrated on, for the page
    for r in rows:
        m, g = (r.get("measured") or {}).get("ram"), r.get("_geo")
        if not (m and g and bw and m.get("tok_s")):
            continue
        tpf = m.get("tpf") or 1.0
        b = per_token(g, r.get("family"))
        busy = tpf / m["tok_s"] - overhead_s(g) if b else 0      # one forward, less its fixed part
        if busy <= 0:
            continue
        etas.append(b / busy / bw)
        if m.get("tpf"):
            tpfs.append(m["tpf"])
        names.append({"name": r["name"], "tok_s": m["tok_s"], "tpf": tpf})
        fams.append(r.get("family"))
    CAL.update(eta=min(0.9, statistics.median(etas)) if etas else ETA, tpf=statistics.median(tpfs) if tpfs else None,
               family=fams[0] if fams else None, **{"from": names or None})


def usable_ram():
    return max(0.0, (MACHINE.get("ram_total_gb") or 0) * GB * 0.9 - 2 * GB)


def tiers_for(g, disk_bytes, ram_min_gb=0, have=False, family="", mtp=None, accel=True):
    """The three ways a model can run here, each with fits / why / speed estimate. `have`:
    already on this disk, so the download size does not matter. `accel`: its engine has a GPU
    path at all. A model with an MTP head gets the tokens per forward measured here."""
    bench = MACHINE.get("bench") or {}
    bw = (bench.get("ram_bw_gbs") or 0) * GB * CAL["eta"]
    if mtp is None:
        mtp = (g.get("sizes") or {}).get("mtp", 0) > 0
    gain = 1.0 if not mtp else CAL["tpf"] if family and family == CAL["family"] and CAL["tpf"] else MTP_GAIN.get(family, 1.0)
    disk = (bench.get("disk_gbs") or 0) * GB
    resident, dense = held(g, family)
    total = resident + g["experts"]
    ram = usable_ram()
    b = per_token(g, family)
    oh = overhead_s(g)
    full = round(gain / (b / bw + oh), 1) if b and bw else None
    out = {}
    fits_ram = total <= ram
    out["ram"] = {"fits": fits_ram, "need_gb": round(total / GB, 1),
                  "why": ("tutto in RAM: %.0f GB su %.0f utili" % (total / GB, ram / GB) if fits_ram
                          else "servono %.0f GB, la RAM utile è %.0f GB" % (total / GB, ram / GB)),
                  "est": full if fits_ram else None}
    gpus = MACHINE.get("gpus") or []
    if gpus:
        disc = [x for x in gpus if x["kind"] == "discrete"]
        name = gpus[0]["name"] + (" ×%d" % len(disc) if len(disc) > 1 else "")
        vram = sum(x.get("vram_gb") or 0 for x in disc) * GB
        cap = ram + vram
        fits = total <= cap
        t = {"fits": fits, "need_gb": round(total / GB, 1), "gpu": name, "est": None,
             "est_note": "su GPU dedicata la velocità si misura al primo uso" if disc else None,
             "why": (("%s: GPU integrata, usa la stessa RAM (%.0f GB utili)" % (name, ram / GB) if not disc
                      else "%.0f GB di VRAM + %.0f GB di RAM" % (vram / GB, ram / GB)) if fits
                     else "servono %.0f GB, VRAM + RAM fanno %.0f GB" % (total / GB, cap / GB))}
        if not accel:
            t.update(fits=False, est_note=None, why="il motore %s gira solo sulla CPU" % (family or "di questo modello"))
        elif not disc and family in setup_flow.VULKAN_IGPU_MEASURED:
            ratio = IGPU_RATIO.get(family) if IGPU_NAME in (name or "") else None
            if ratio and fits and full:
                r = ratio["gpu"] / ratio["cpu"]
                t.update(est=round(full * r, 1), est_note=None, basis=dict(ratio, ratio=round(r, 3)))
            else:
                t["est_note"] = "su questa GPU integrata la velocità si misura al primo uso"
        elif not disc and fits:
            t.update(fits=False, why="su questa GPU integrata il motore %s resta sulla CPU: nessuna misura dice che "
                                     "vada più veloce" % (family or "di questo modello"))
        out["gpu"] = t
    cache = ram - resident
    fits_ssd = cache > 0 and not fits_ram and ram_min_gb * GB <= (MACHINE.get("ram_total_gb") or 0) * GB
    lo = None
    if fits_ssd and b and bw and disk:
        hit = min(1.0, cache / g["experts"]) if g["experts"] else 1.0
        exp_act = g["experts"] * g["k"] / g["E"]
        # the MTP head speeds up what is read from RAM; the misses still come from the disk
        lo = round(1 / (((dense + exp_act * hit) / bw + oh) / gain + exp_act * (1 - hit) / disk), 2)
    out["ssd"] = {"fits": fits_ssd, "need_gb": round(resident / GB, 1),
                  "why": ("parte densa %.0f GB in RAM, cache esperti %.0f GB, il resto dall'SSD" % (resident / GB, cache / GB)
                          if fits_ssd else "entra già tutto in RAM" if fits_ram
                          else "la parte densa (%.0f GB) non entra nella RAM utile" % (resident / GB)),
                  "est_lo": lo, "est_hi": full if fits_ssd else None}
    out["mtp"] = gain if gain > 1 else None
    out["disk_ok"] = have or (MACHINE.get("disk_free_gb") or 0) * GB >= disk_bytes * 1.02 + 2 * GB
    if not out["disk_ok"]:
        for k in ("ram", "gpu", "ssd"):
            if k in out and out[k]["fits"]:
                out[k]["why"] += " · ma il download (%.0f GB) non entra nel disco" % (disk_bytes / GB)
    return out


# ------------------------------------------------------------------ measured speeds
MEASURED = {}   # model dir -> {turns: [...], last_seq, pid}


def sample_measured():
    """the gateway's finished turns since the last look, each with the way it ran (ram, gpu)"""
    d, tier, p = gateway_model_dir(), gateway_tier(), gateway_proc()
    if not d:
        return
    try:
        prof = gw_json("/profile")
    except (OSError, ValueError):
        return
    rec = MEASURED.setdefault(d, {"turns": [], "last_seq": 0, "pid": None})
    if rec.get("pid") != p["pid"]:          # a new server: its /profile restarts from 0
        rec["pid"], rec["last_seq"] = p["pid"], 0
    seq = prof.get("seq", 0)
    new = seq - rec["last_seq"]
    if new <= 0:
        return
    for t in (prof.get("turns") or [])[-new:]:
        if t.get("completion_tokens", 0) >= 64 and t.get("wall_s", 0) > 0:
            rec["turns"].append({"tok_s": round(t["completion_tokens"] / t["wall_s"], 2), "tokens": t["completion_tokens"],
                                 "prompt": t.get("prompt_tokens"), "tier": tier,
                                 "tpf": round(t["completion_tokens"] / t["forwards"], 2) if t.get("forwards") else None,
                                 "at": int(time.time())})
    rec["turns"] = rec["turns"][-60:]
    rec["last_seq"] = seq
    jsave(spath("measured.json"), MEASURED)


def measured_for(path):
    """{ram: {...}, gpu: {...}}: the median speed of the turns that ran each way"""
    rec = MEASURED.get(os.path.realpath(path)) if path else None
    out = {}
    for tier in ("ram", "gpu"):
        turns = [t for t in (rec or {}).get("turns", []) if (t.get("tier") or "ram") == tier]
        if turns:
            tpfs = [t["tpf"] for t in turns if t.get("tpf")]
            out[tier] = {"tok_s": round(statistics.median(t["tok_s"] for t in turns), 1), "turns": len(turns),
                         "tpf": round(statistics.median(tpfs), 2) if tpfs else None}
    return out or None


# ------------------------------------------------------------------ models on this disk
def dir_bytes(path):
    tot = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            p = os.path.join(dirpath, f)
            if not os.path.islink(p):
                try:
                    tot += os.path.getsize(p)
                except OSError:
                    pass
    return tot


_inst = {"at": 0, "rows": []}


def installed():
    if time.time() - _inst["at"] < 30:
        return _inst["rows"]
    rows = []
    for root in roots():
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if name.startswith(".") or not os.path.isdir(d):
                continue
            has = {f: os.path.exists(os.path.join(d, f)) for f in ("config.json", "gguf.json", "model_index.json")}
            if not any(has.values()):
                continue
            man = setup_download.read_manifest(d) or {}
            cfg = jload(os.path.join(d, "config.json"), {})
            gg = cfg.get("colibri_gguf") if isinstance(cfg.get("colibri_gguf"), dict) else {}
            rows.append({"path": os.path.realpath(d), "name": gg.get("name") or name,
                         "arch": (cfg.get("architectures") or [cfg.get("model_type") or ("image" if has["model_index.json"] else "")])[0],
                         "gguf": has["gguf.json"], "bytes": None if has["gguf.json"] else dir_bytes(d),
                         "repo": man.get("repo"), "complete": man.get("complete", True)})
    _inst.update(at=time.time(), rows=rows)
    return rows


def family_of(path):
    """(family id, has a GPU path, kind: text, decision or image) from colibri's own registry;
    ValueError for a folder it cannot serve"""
    import family_registry
    d = family_registry.resolve_model(path).descriptor
    caps = d.capabilities
    kind = "decision" if caps.decision else "image" if d.engine_artifact == "qwenimage" else "text"
    return d.id, d.supports_accelerator, kind


def installed_rows():
    """Models on this disk as candidates: their own geometry, measured speeds."""
    cur = gateway_model_dir()
    rows = []
    for r in installed():
        r = dict(r, id="local:" + os.path.basename(r["path"]), kind="text", installed=r["path"], in_use=cur == r["path"])
        try:
            fam, accel, kind = family_of(r["path"])
        except Exception as e:   # noqa: BLE001 - coli serve would refuse it too
            r.update(tiers=None, error="colibri non sa servire questa cartella: %s" % str(e)[:200], status="incomplete", rank=0)
            rows.append(r)
            continue
        if kind != "text":       # served, but nothing to estimate: a decision model, a picture model
            r.update(kind=kind, family=fam, tiers=None, status="good", rank=0)
            rows.append(r)
            continue
        try:
            g = local_geometry(r["path"])
        except Exception as e:   # noqa: BLE001
            r.update(tiers=None, error="non leggibile: %s" % str(e)[:200], status="incomplete", rank=0)
            rows.append(r)
            continue
        r.update(rank=FAMILY_RANK.get(fam, 2), family=fam, _accel=accel, k=g.get("k"), E=g.get("E"), _geo=g,
                 measured=measured_for(r["path"]), status="good")
        if (g.get("E") or 0) > 1 and not g.get("experts"):
            r.update(tiers=None, status="incomplete", summary="incompleto: mancano gli esperti")
        rows.append(r)
    calibrate(rows)
    for r in rows:
        if r.get("_geo") and r.get("status") == "good":
            r["tiers"] = tiers_for(r["_geo"], 0, 0, True, r["family"], accel=r["_accel"])
        r.pop("_geo", None)
        r.pop("_accel", None)
    return rows


# ------------------------------------------------------------------ catalog
def catalog_rows():
    inst = {r.get("repo"): r for r in installed() if r.get("repo")}
    m, rows = MACHINE, []
    try:
        import family_registry
    except ImportError:
        family_registry = None
    for e in setup_catalog.catalog():
        status, reason = setup_catalog.assess(e, (m.get("ram_total_gb") or 0) * GB, (m.get("disk_free_gb") or 0) * GB)
        text = e.modality == "text"
        g = catalog_geometry(e, fetch=False) if text else None
        here = inst.get(e.repo)
        try:
            accel = family_registry.family_by_id(e.family).supports_accelerator
        except Exception:   # noqa: BLE001 - a family the registry does not know
            accel = True
        rows.append({"id": e.id, "name": e.name, "repo": e.repo, "family": e.family, "kind": e.modality,
                     "download_gb": e.disk_gb, "ram_min_gb": e.ram_min_gb, "ram_good_gb": e.ram_good_gb,
                     "dense_gb": e.dense_gb, "rank": e.rank, "summary": e.summary, "license_note": e.license_note,
                     "status": status, "reason": reason, "ours": "colibri" in e.repo.lower(),
                     "installed": here["path"] if here else None,
                     "tiers": tiers_for(g, e.disk_gb * GB, e.ram_min_gb, bool(here), e.family, accel=accel) if text else None,
                     "geometry": g.get("source") if g else None, "k": g.get("k") if g else None, "E": g.get("E") if g else None,
                     "measured": measured_for(here["path"]) if here else None})
    return rows


def best_per_tier(rows):
    """The most capable text model each way it can run here and can be had: on this disk
    already, or a download that fits it. Ties go to the faster estimate, then the smaller."""
    out = {}
    for tier in ("ram", "gpu", "ssd"):
        ok = [r for r in rows if r["kind"] == "text" and r.get("tiers") and (r["tiers"].get(tier) or {}).get("fits")
              and r["tiers"].get("disk_ok") and r.get("status") not in ("unsupported", "needs-ram")]
        if ok:
            def key(r):
                t = r["tiers"][tier]
                return (r["rank"], bool(r.get("installed")), t.get("est") or t.get("est_lo") or 0, -(r.get("download_gb") or 0))
            out[tier] = max(ok, key=key)["id"]
    return out


# ------------------------------------------------------------------ GGUFs on Hugging Face
QUANT_RE = re.compile(r"(UD-)?(IQ\d_(?:XXS|XS|S|M|NL)|Q\d_K(?:_(?:XL|L|M|S))?|Q\d_\d|TQ\d_\d|MXFP4(?:_MOE)?|NVFP4|BF16|F16|F32)", re.I)
_search_cache = {}


def hf_search(q, limit=40):
    key = (q or "").strip().lower()
    hit = _search_cache.get(key)
    if hit and hit[1] > time.time():
        return hit[0]
    params = [("filter", "gguf"), ("sort", "downloads"), ("direction", "-1"), ("limit", str(limit)),
              ("expand[]", "gguf"), ("expand[]", "downloads"), ("expand[]", "likes"), ("expand[]", "lastModified")]
    if key:
        params.insert(0, ("search", q.strip()))
    supported = gguf_arches()
    out = []
    for r in http_json(setup_download.endpoint() + "/api/models?" + urllib.parse.urlencode(params), timeout=30):
        gg = r.get("gguf") or {}
        arch = gg.get("architecture")
        out.append({"repo": r.get("id"), "arch": arch, "engine": GGUF_FAMILY.get(arch, arch), "supported": arch in supported,
                    "params_b": round((gg.get("total") or 0) / 1e9, 1), "context": gg.get("context_length"),
                    "downloads": r.get("downloads"), "likes": r.get("likes"), "updated": (r.get("lastModified") or "")[:10]})
    _search_cache[key] = (out, time.time() + 600)
    return out


def _gguf_parse(buf):
    """Metadata and tensor table of a GGUF header held in buf. IndexError when buf is too
    short (the caller reads more)."""
    o = [0]

    def rd(fmt):
        s = struct.calcsize(fmt)
        if o[0] + s > len(buf):
            raise IndexError
        v = struct.unpack_from(fmt, buf, o[0])
        o[0] += s
        return v[0]

    def rstr():
        n = rd("<Q")
        if o[0] + n > len(buf):
            raise IndexError
        s = bytes(buf[o[0]:o[0] + n]).decode("utf-8", "replace")
        o[0] += n
        return s
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmts = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            at, n = rd("<I"), rd("<Q")
            if at in sizes:
                o[0] += sizes[at] * n
                if o[0] > len(buf):
                    raise IndexError
                return None
            for _ in range(n):
                rval(at)
            return None
        return rd(fmts[t])
    if bytes(buf[:4]) != b"GGUF":
        raise ValueError("not a GGUF")
    o[0] = 4
    rd("<I")
    nt, nkv = rd("<Q"), rd("<Q")
    meta = {}
    for _ in range(nkv):
        k = rstr()
        v = rval(rd("<I"))
        if v is not None and not k.startswith("tokenizer."):
            meta[k] = v
    tensors = []
    for _ in range(nt):
        name = rstr()
        for _ in range(rd("<I")):
            rd("<Q")
        rd("<I")
        tensors.append((name, rd("<Q")))
    align = meta.get("general.alignment", 32)
    return meta, tensors, (o[0] + align - 1) // align * align


def gguf_geometry(parts):
    """Exact bytes by kind from the headers of every part of one GGUF (HTTP Range reads of a
    few MB a part, or the local file): the tensor table gives each tensor's offset."""
    sz, meta0 = {}, None
    for path, size, url in parts:
        n = 4 << 20
        while True:
            if url:
                buf = http_range(url, 0, min(size, n) - 1)
            else:
                with open(path, "rb") as f:
                    buf = f.read(min(size, n))
            try:
                meta, tensors, start = _gguf_parse(buf)
                break
            except IndexError:
                if n >= size or n >= (256 << 20):
                    raise ValueError("%s: header larger than %d MB" % (path, n >> 20)) from None
                n *= 4
        meta0 = meta0 or meta
        offs = sorted(t[1] for t in tensors)
        ends = {o: (offs[i + 1] if i + 1 < len(offs) else size - start) for i, o in enumerate(offs)}
        for name, off in tensors:
            c = classify(name, gguf=True)
            sz[c] = sz.get(c, 0) + ends[off] - off
    meta0 = meta0 or {}
    arch = meta0.get("general.architecture", "")
    g = {"source": "gguf", "arch": arch, "layers": meta0.get(arch + ".block_count"),
         "k": meta0.get(arch + ".expert_used_count"), "E": meta0.get(arch + ".expert_count"), **geo_from_sizes(sz)}
    if not g["E"]:                       # a dense model: everything is read every token
        g.update(k=1, E=1)
    return g


def gguf_variants(repo):
    groups = {}
    for f in setup_download.list_repo_files(repo, "main"):
        p = f["path"]
        base = os.path.basename(p).lower()
        if not p.endswith(".gguf") or base.startswith("mmproj") or "imatrix" in base:
            continue
        m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", p)
        groups.setdefault(m.group(1) if m else p[:-5], []).append(f)
    out = []
    for stem, parts in groups.items():
        parts.sort(key=lambda f: f["path"])
        q = QUANT_RE.search(os.path.basename(stem))
        out.append({"stem": stem, "quant": q.group(0).upper() if q else "?", "parts": len(parts), "files": parts,
                    "bytes": sum(f["size"] for f in parts), "mtp": "mtp" in os.path.basename(stem).lower()})
    out.sort(key=lambda v: v["bytes"])
    return [v for v in out if not v["mtp"]], [v for v in out if v["mtp"]]


def gguf_files(repo):
    from concurrent.futures import ThreadPoolExecutor
    variants, mtp = gguf_variants(repo)
    stems = [os.path.basename(v["stem"]) for v in variants]
    pre = os.path.commonprefix(stems) if len(stems) > 1 else ""
    pre = pre[:pre.rfind("-") + 1] if "-" in pre else ""

    def one(v):
        row = dict(v, label=os.path.basename(v["stem"])[len(pre):] or v["quant"])
        try:
            key = "gguf:%s:%s" % (repo, ",".join(f["path"] for f in v["files"]))
            with _glock:
                g = GEO.get(key)
            if g is None:
                g = _geo_put(key, gguf_geometry([(f["path"], f["size"], setup_download.file_url(repo, "main", f["path"]))
                                                  for f in v["files"]]))
            row["tiers"] = tiers_for(g, v["bytes"], 0, False, GGUF_FAMILY.get(g.get("arch"), ""), bool(mtp) or None)
            row["k"], row["E"] = g.get("k"), g.get("E")
        except Exception as e:   # noqa: BLE001
            row["tiers"], row["error"] = None, "header: %s" % e
        return row
    with ThreadPoolExecutor(8) as pool:      # a few MB of header each, read in parallel
        rows = list(pool.map(one, variants))
    return {"repo": repo, "variants": rows, "mtp": mtp}


# ------------------------------------------------------------------ jobs: download, import, switch
JOBS = {}
_jlock = threading.Lock()


class Cancelled(BaseException):
    pass


def new_job(kind, title, **kw):
    jid = "%x" % int(time.time() * 1000)
    job = {"id": jid, "kind": kind, "title": title, "state": "queued", "done": 0, "total": 0, "rate": 0, "eta": None,
           "msg": "", "started": time.time(), "cancel": False, **kw}
    with _jlock:
        JOBS[jid] = job
    return job


def job_view(j):
    return {k: v for k, v in j.items() if k != "cancel"}


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")[:80]


def run_download(job, repo, revision, files, dest, after=None):
    try:
        os.makedirs(dest, exist_ok=True)
        total = sum(int(f["size"]) for f in files)
        need = total * 1.02 + 2 * GB - setup_download.bytes_present(dest, files)
        free = shutil.disk_usage(dest).free
        if free < need:
            raise RuntimeError("spazio insufficiente: servono %.0f GB, liberi %.0f GB" % (need / GB, free / GB))
        job.update(state="downloading", total=total)
        mark = [time.time(), 0]

        def progress(ev):
            if job["cancel"]:
                raise Cancelled()
            now = time.time()
            job["done"], job["file"] = ev["done"], ev["file"]
            if now - mark[0] >= 1.0:
                rate = (ev["done"] - mark[1]) / (now - mark[0])
                job["rate"] = rate if not job["rate"] else 0.7 * job["rate"] + 0.3 * rate
                job["eta"] = (total - ev["done"]) / job["rate"] if job["rate"] > 0 else None
                mark[0], mark[1] = now, ev["done"]
        setup_download.download_repo(repo, revision, dest, files, token=setup_download.hf_token(), progress=progress)
        job["done"] = total
        if after:
            job["state"] = "installing"
            after(job)
        job.update(state="done", msg=job.get("msg") or "pronto")
        _inst["at"] = 0
    except Cancelled:
        job.update(state="cancelled", msg="annullato: i file parziali restano e il download riprende da lì")
    except Exception as e:   # noqa: BLE001 - said on the page
        job.update(state="error", msg=str(e)[:400])


def start_catalog_download(model_id):
    e = setup_catalog.by_id(model_id)
    dest = os.path.join(roots()[0], e.id)
    job = new_job("download", e.name, repo=e.repo, dest=dest)

    def go():
        try:
            files = setup_download.list_repo_files(e.repo, e.revision, include=e.include, exclude=e.exclude,
                                                   token=setup_download.hf_token())
        except Exception as err:   # noqa: BLE001
            job.update(state="error", msg=str(err)[:400])
            return

        def after(j):
            for tool, creates in e.post_install:
                if os.path.exists(os.path.join(dest, creates)):
                    continue
                r = subprocess.run([sys.executable, os.path.join(COLI_DIR, "tools", tool + ".py"), dest],
                                   capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError("tools/%s.py: %s" % (tool, (r.stderr or r.stdout)[-300:]))
        run_download(job, e.repo, e.revision, files, dest, after)
    threading.Thread(target=go, daemon=True).start()
    return job


def start_gguf_download(repo, stem):
    variants, mtp = gguf_variants(repo)
    v = next((x for x in variants if x["stem"] == stem), None)
    if not v:
        raise ValueError("%s: no variant %s" % (repo, stem))
    files = list(v["files"])
    # Qwen3.8: the MTP head ships as its own GGUF; the importer finds it next to the model
    extra = [m for m in mtp if v["quant"] in m["stem"].upper()] or mtp[:1]
    for m in extra[:1]:
        files += m["files"]
    name = slug(re.sub(r"-gguf$", "", repo.split("/")[-1], flags=re.I) + "-" + v["quant"])
    root = roots()[0]
    dest, out = os.path.join(root, "gguf", name), os.path.join(root, name)
    job = new_job("download", "%s %s" % (repo.split("/")[-1], v["quant"]), repo=repo, dest=out)

    def after(j):
        r = subprocess.run([sys.executable, os.path.join(COLI_DIR, "coli"), "import", os.path.join(dest, v["files"][0]["path"]),
                            "--out", out, "--mode", "link"], capture_output=True, text=True, cwd=COLI_DIR)
        if r.returncode:
            raise RuntimeError("coli import: %s" % (r.stderr or r.stdout)[-400:])
        j["msg"] = "importato in %s" % out
    threading.Thread(target=run_download, args=(job, repo, "main", files, dest, after), daemon=True).start()
    return job


def proc_children(pid):
    out = []
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                with open("/proc/%s/stat" % p) as f:
                    if int(f.read().rsplit(")", 1)[1].split()[1]) == pid:
                        out.append(int(p))
            except (OSError, ValueError, IndexError):
                pass
    return out


def stop_gateway(p, job):
    kids = proc_children(p["pid"])
    job.update(state="stopping", msg="fermo il server attuale")
    os.kill(p["pid"], signal.SIGTERM)
    for _ in range(60):
        if not os.path.exists("/proc/%d" % p["pid"]):
            break
        time.sleep(1)
    else:
        os.kill(p["pid"], signal.SIGKILL)
    for k in kids:                     # the engine, if the gateway left it behind
        if os.path.exists("/proc/%d" % k):
            os.kill(k, signal.SIGTERM)
    _gw["at"] = 0


def launch(argv, cwd, env, log):
    if os.path.exists(log):
        os.replace(log, log + ".prev")
    out = open(log, "wb")
    kw = {"start_new_session": True} if os.name == "posix" else \
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
    # its own session: it outlives this service, as a `coli serve` started by hand would
    return subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, **kw)


def wait_up(proc, log, job, limit_s=1800):
    for _ in range(limit_s // 2):
        time.sleep(2)
        if proc.poll() is not None:
            try:      # its last words, a traceback's frames left out
                with open(log, errors="replace") as f:
                    said = [ln.strip() for ln in f.read()[-20000:].splitlines()
                            if ln.strip() and not ln.startswith((" ", "\t", "Traceback"))]
            except OSError:
                said = []
            raise RuntimeError("il server si è chiuso (codice %s): %s" % (proc.returncode, " · ".join(said[-2:])[-400:]))
        try:
            if gw_json("/health", timeout=3).get("status") == "ok":
                return
        except (OSError, ValueError):
            pass
    raise RuntimeError("il server non risponde dopo %d minuti" % (limit_s // 60))


_switching = threading.Lock()


def switch_model(path, force=False, tier=None):
    """Restart the gateway on another model of this disk, the way it was started (its command,
    folder and environment) with --model changed; `tier` ram or gpu also turns the GPU off or
    on. A model that does not start brings the previous one back."""
    path = os.path.realpath(path)
    if not os.path.isdir(path) or not under_roots(path):
        raise ValueError("cartella del modello non valida")
    old = gateway_proc()
    answering = False
    try:
        gw_json("/health", timeout=3)
        answering = True
    except urllib.error.HTTPError:
        answering = True
    except (OSError, ValueError):
        pass
    if answering and old and not old["coli"]:
        raise RuntimeError("la porta %d è di un altro processo (%d): non lo tocco" % (CFG.port, old["pid"]))
    if answering and not old:
        raise RuntimeError("non trovo il processo di coli serve sulla porta %d su questa macchina: fermalo e "
                           "riavvialo con coli serve --model %s" % (CFG.port, path))
    if answering and not force and not gw_idle():
        raise RuntimeError("il motore sta rispondendo a una richiesta: riprova quando ha finito")
    if not _switching.acquire(blocking=False):
        raise RuntimeError("sto già cambiando modello: aspetta che finisca")
    job = new_job("switch", os.path.basename(path), dest=path)

    def go():
        try:
            import family_registry
            # what can fail before the old server stops: the user keeps a working model
            d = family_registry.resolve_model(path).descriptor
            binary = os.path.join(COLI_DIR, d.engine_artifact)
            if not os.path.exists(binary) and not os.path.exists(binary + ".exe"):
                if not (os.path.exists(os.path.join(COLI_DIR, "Makefile")) and shutil.which("make")):
                    raise RuntimeError("manca il motore %s in %s" % (d.engine_artifact, COLI_DIR))
                job.update(state="building", msg="compilo il motore %s (una volta sola)" % d.build_target)
                r = subprocess.run(["make", "-j%d" % max(1, (os.cpu_count() or 2) // 2), d.build_target],
                                   cwd=COLI_DIR, capture_output=True, text=True)
                if r.returncode or not os.path.exists(binary):
                    raise RuntimeError("make %s non riuscito: %s" % (d.build_target, (r.stderr or r.stdout)[-300:]))
            if old and old["coli"]:
                argv, cwd, env = list(old["argv"]), old["cwd"], dict(old["env"])
                at = argv.index("serve") + 1
                argv = argv[:at] + ["--model", path] + _without(argv[at:], ("--model",))
            else:
                argv = [sys.executable, os.path.join(COLI_DIR, "coli"), "serve", "--model", path,
                        "--host", CFG.host, "--port", str(CFG.port)]
                cwd, env = COLI_DIR, dict(os.environ)
            if tier == "ram":
                argv = _without(argv, ("--gpu", "--vram")) + ["--gpu", "none"]
                env["COLI_VULKAN"] = "0"
            elif tier == "gpu":
                argv = _without(argv, ("--gpu",)) if _argval(argv, "--gpu") == "none" else argv
                for k in ("COLI_CUDA", "COLI_VULKAN"):
                    if env.get(k) == "0":
                        env.pop(k)
                nvidia = any(x.get("vendor") == "NVIDIA" for x in MACHINE.get("gpus") or [])
                if nvidia and env.get("COLI_CUDA") != "1" and not _argval(argv, "--gpu"):
                    argv += ["--gpu", "auto"]
            if old and old["coli"]:
                stop_gateway(old, job)
            log = spath("serve.log")
            job.update(state="starting", msg="avvio il nuovo modello")
            proc = launch(argv, cwd, env, log)
            try:
                wait_up(proc, log, job)
            except RuntimeError as e:
                if proc.poll() is None:
                    proc.terminate()
                if not (old and old["coli"]):
                    raise
                job.update(state="starting", msg=str(e)[:400], note="non parte: riavvio il modello di prima")
                back = launch(old["argv"], old["cwd"], old["env"], spath("serve-previous.log"))
                try:
                    wait_up(back, spath("serve-previous.log"), job)
                    job["note"] = "riavviato il modello di prima"
                except RuntimeError as e2:
                    job["note"] = "e il modello di prima non riparte: %s" % str(e2)[:300]
                raise
            job.update(state="done", msg="in uso")
            _inst["at"] = 0
            _gw["at"] = 0
        except Exception as e:   # noqa: BLE001 - said on the page
            job.update(state="error", msg=str(e)[:400])
        finally:
            _switching.release()
    threading.Thread(target=go, daemon=True).start()
    return job


# ------------------------------------------------------------------ the API
def get(path, q):
    """(status, body) for GET /models/..."""
    if path == "/models/machine":
        return 200, MACHINE or refresh_machine()
    if path == "/models/catalog":
        inst = installed_rows()           # first: it calibrates the estimates
        rows = catalog_rows()
        return 200, {"machine": MACHINE, "eta": CAL["eta"], "cal": CAL, "catalog": rows, "installed": inst,
                     "best": best_per_tier(rows + [r for r in inst if r.get("tiers")]),
                     "in_use": gateway_model_dir(), "in_use_tier": gateway_tier(), "roots": roots(),
                     "can_switch": LINUX, "gguf_arches": sorted(gguf_arches())}
    if path == "/models/search":
        return 200, {"q": q.get("q", ""), "results": hf_search(q.get("q", ""))}
    if path == "/models/files":
        if not CAL["from"]:
            installed_rows()
        return 200, gguf_files(q["repo"])
    if path == "/models/jobs":
        with _jlock:
            return 200, {"jobs": [job_view(j) for j in sorted(JOBS.values(), key=lambda j: -j["started"])]}
    return 404, {"error": "not found"}


def post(path, b):
    """(status, body) for POST /models/..."""
    if path == "/models/machine/measure":
        if not gw_idle():
            return 409, {"error": "il motore sta lavorando: la misura verrebbe falsata"}
        return 200, refresh_machine(measure=True)
    if path == "/models/download":
        return 200, job_view(start_catalog_download(b["id"]) if b.get("id") else start_gguf_download(b["repo"], b["stem"]))
    if path == "/models/use":
        return 200, job_view(switch_model(b["path"], bool(b.get("force")), b.get("tier")))
    m = re.fullmatch(r"/models/jobs/([0-9a-f]+)/cancel", path)
    if m and m.group(1) in JOBS:
        JOBS[m.group(1)]["cancel"] = True
        return 200, {"ok": True}
    return 404, {"error": "not found"}


def handle(method, path, query, body):
    """one request, its errors as a status the page shows"""
    try:
        if method == "GET":
            return get(path, dict(urllib.parse.parse_qsl(query)))
        b = json.loads(body or b"{}")
        if not isinstance(b, dict):
            raise ValueError("a JSON object")
        return post(path, b)
    except (KeyError, ValueError) as e:
        return 400, {"error": str(e)[:400]}
    except RuntimeError as e:
        return 409, {"error": str(e)[:400]}
    except Exception as e:   # noqa: BLE001 - the network's, the disk's: said to the page
        return 502, {"error": "%s: %s" % (type(e).__name__, str(e)[:400])}


# ------------------------------------------------------------------ in the background
def warm_geometry():
    for r in installed():
        try:
            local_geometry(r["path"])
        except Exception:   # noqa: BLE001
            pass
    for e in setup_catalog.catalog():          # bytes per token for every catalog model, once
        if e.modality == "text":
            try:
                catalog_geometry(e)
            except Exception:   # noqa: BLE001 - offline: the catalog's own sizes
                pass


def background():
    try:
        refresh_machine()
    except Exception:   # noqa: BLE001
        pass
    threading.Thread(target=warm_geometry, daemon=True).start()
    while True:
        time.sleep(20)
        try:
            sample_measured()
            if not (MACHINE.get("bench") or {}).get("ram_bw_gbs"):
                refresh_machine()
            else:                       # the cheap part: free RAM and disk
                mem = setup_hw.detect_memory()
                with _mlock:
                    MACHINE["ram_avail_gb"] = round((mem.get("available") or 0) / GB, 1)
                    MACHINE["disk_free_gb"] = round(shutil.disk_usage(setup_hw.existing_ancestor(roots()[0])).free / GB, 1)
        except Exception:   # noqa: BLE001 - the next round tries again
            pass


_started = []


def start(backend, models_dirs=(), state=None):
    if not _started:
        configure(backend, models_dirs, state)
        threading.Thread(target=background, daemon=True).start()
        _started.append(True)


class Handler(BaseHTTPRequestHandler):
    """alone on the engine's machine, for `coli ui --models-backend`: 127.0.0.1 only, a loopback
    Host, and on a POST the page's own header (no other site, no DNS rebinding)"""
    server_version = "colibri-models/1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _go(self):
        u = urllib.parse.urlsplit(self.path)
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]   # any port: an ssh tunnel moves it
        post = self.command == "POST"
        if host not in ("127.0.0.1", "localhost") or (post and self.headers.get("X-Colibri-UI") != "1"):
            return self._send(403, {"error": "solo dalla pagina di colibri"})
        n = int(self.headers.get("Content-Length") or 0)
        if n > 65536:
            return self._send(413, {"error": "too large"})
        self._send(*handle(self.command, u.path, u.query, self.rfile.read(n) if n else b""))

    do_GET = do_POST = _go


def main():
    ap = argparse.ArgumentParser(description="the Models page's service, alone on the engine's machine")
    ap.add_argument("--backend", default=os.environ.get("COLIBRI_BACKEND", "http://127.0.0.1:8000"),
                    help="the colibri gateway (coli serve) on this machine")
    ap.add_argument("--models-dir", action="append", default=[], help="where models are (default: colibri's own folder)")
    ap.add_argument("--port", type=int, default=8791)
    a = ap.parse_args()
    start(a.backend, a.models_dir)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print("colibri models service on 127.0.0.1:%d (the gateway: %s)" % (a.port, CFG.gateway), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
