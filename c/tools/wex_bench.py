#!/usr/bin/env python3
"""wex_bench.py -- the WEX measurements of #2860 in one command, with a report to paste there.

    python3 c/tools/wex_bench.py STAGE [STAGE ...] [options]

Stages (each runs alone; results accumulate in OUT/wex-bench-results.json):
    env      OS, CPU model and flags, cores, RAM, GPUs, compiler, the container's drive type
    verify   tools/wex4_verify on --container: time, GB/s, the root
    decode   tools/wex_decode_bench decode: ms per record on N seeded WNR3 records for the
             reference decoder, the fast decoder per stream count (--lanes), the CUDA helper
             when COLI_WEX_CUDA_LIB (or --helper) names it; every arm checked byte for byte
    io       tools/wex_decode_bench io: read GB/s of N seeded records, uncached and warm
    engine   one engine, a fixed prompt, greedy, N tokens: the shards against the container,
             fast decoder on/off, helper on/off; prefill, decode, tok/s, token ids, decode counts
    report   OUT/wex-bench-report.md and .json from the results so far
    all      env verify decode io (engine too with --snap), then report

Standard library only; it builds the C tools with `make -C c` when they are missing.
Nothing it prints or writes holds the host name, the user name or a path: paths are replaced by
placeholders, and the container is named by its file name. See docs/wex-bench.md.
"""
import argparse
import ctypes
import datetime
import getpass
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

C_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(C_DIR)
IS_WIN = sys.platform == "win32"
EXE = ".exe" if IS_WIN else ""
RESULTS = "wex-bench-results.json"

# BOS + "The three primary colors of light are" for the DeepSeek V4 / V4.1 tokenizer: the prompt of #2879 and #2889
V41_PROMPT_IDS = [0, 671, 2038, 6074, 11273, 294, 2900, 477]


# ---- privacy: no host name, user name or path leaves this script ----------------------------
class Scrubber:
    def __init__(self):
        self.pairs = []
        self.words = []
        generic = {"root", "user", "admin", "administrator", "runner", "ubuntu", "guest", "localhost", "debian", "fedora"}
        for word in {self._safe(getpass.getuser), self._safe(socket.gethostname), self._safe(platform.node)}:
            if word and len(word) >= 3 and word.lower() not in generic:   # "root" names no one, and is a word of the report
                self.words.append(word)
        self.add(os.path.expanduser("~"), "<home>")
        self.add(REPO, "<repo>")
        self.add(os.getcwd(), "<cwd>")

    @staticmethod
    def _safe(fn):
        try:
            return fn()
        except Exception:
            return ""

    def add(self, path, label):
        if not path:
            return
        path = os.path.abspath(path)
        forms = {path, path.replace("\\", "/")}
        if IS_WIN and len(path) > 2 and path[1] == ":":   # the MSYS2 and WSL spellings of the same path
            rest = path[2:].replace("\\", "/")
            forms |= {"/" + path[0].lower() + rest, "/mnt/" + path[0].lower() + rest}
        for f in forms:
            if len(f) > 3:
                self.pairs.append((f, label))
        self.pairs.sort(key=lambda p: -len(p[0]))

    def __call__(self, text):
        if text is None:
            return text
        if isinstance(text, (list, tuple)):
            return [self(t) for t in text]
        if isinstance(text, dict):
            return {k: self(v) for k, v in text.items()}
        if not isinstance(text, str):
            return text
        for raw, label in self.pairs:
            text = re.sub(re.escape(raw), label, text, flags=re.I if IS_WIN else 0)
        text = re.sub(r"\\\\[\w.$-]+\\[^\s\"'|,;)]*", "<path>", text)                 # UNC
        text = re.sub(r"\b[A-Za-z]:[\\/][^\s\"'|,;)]*", "<path>", text)               # C:\... or C:/...
        text = re.sub(r"(?<![\w.<>])/(?:home|Users|mnt|root|media|Volumes|tmp|var|opt|data|srv|run|[a-z])/[^\s\"'|,;)]*",
                      "<path>", text)
        for word in self.words:
            text = re.sub(r"(?i)(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", "<name>", text)
        return text


SCRUB = Scrubber()


def say(*parts):
    print(SCRUB(" ".join(str(p) for p in parts)), flush=True)


def run(cmd, env=None, cwd=None, timeout=None):
    """(exit code, stdout, stderr, seconds); a missing program is (127, '', reason, 0)."""
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, errors="replace", timeout=timeout)
    except FileNotFoundError as e:
        return 127, "", str(e), 0.0
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return 124, out, err + "\n(timed out)", time.perf_counter() - t0
    return p.returncode, p.stdout, p.stderr, time.perf_counter() - t0


def first_line(cmd):
    code, out, err, _ = run(cmd, timeout=30)
    text = (out or err).strip().splitlines()
    return text[0].strip() if code == 0 and text else ""


# ---- results file ------------------------------------------------------------------------
def load_results(out):
    path = os.path.join(out, RESULTS)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


PATH_VARS = ("COLI_WEX_PATH", "SNAP", "COLI_WEX_CUDA_LIB", "COLI_MODEL_MIRROR", "CC")


def env_vars(extra=None):
    """The variables that change what the tools and engines do, as set now; values that are paths: <set>."""
    env = dict(os.environ)
    env.update(extra or {})
    picked = {k: v for k, v in sorted(env.items())
              if k.startswith(("COLI_", "WEX_", "OMP_", "GOMP_", "KMP_", "CUDA_VISIBLE", "V41_", "Q38_", "DSV4_")) or k in ("SNAP", "CC")}
    return {k: ("<set>" if k in PATH_VARS and v not in ("", "0") else v) for k, v in picked.items()}


def save_stage(out, stage, data, extra_env=None):
    results = load_results(out)
    data["when"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    data["env_vars"] = env_vars(extra_env)
    results[stage] = SCRUB(data)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, RESULTS), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)


def build(target, force=False):
    """Build c/<target> with make when it is missing (or --build); returns its path or None."""
    path = os.path.join(C_DIR, target + EXE)
    if os.path.exists(path) and not force:
        return path
    make = os.environ.get("MAKE", "make")
    say("building", target, "with", make, "...")
    code, out, err, _ = run([make, "-C", C_DIR, target + EXE])
    if code != 0 or not os.path.exists(path):
        say("could not build", target + ":", (err or out).strip()[-800:])
        return None
    return path


# ---- env ---------------------------------------------------------------------------------
def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _sysctl(name):
    return first_line(["sysctl", "-n", name])


def _powershell(script):
    exe = shutil.which("powershell") or shutil.which("powershell.exe") or shutil.which("pwsh")
    if not exe:
        return ""
    code, out, _, _ = run([exe, "-NoProfile", "-NonInteractive", "-Command", script], timeout=60)
    return out.strip() if code == 0 else ""


def cpu_info():
    info = {"arch": platform.machine(), "logical_cores": os.cpu_count()}
    flags = set()
    if sys.platform.startswith("linux"):
        text = _read("/proc/cpuinfo")
        m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
        info["model"] = m.group(1).strip() if m else first_line(["sh", "-c", "lscpu | sed -n 's/^Model name: *//p'"])
        fl = re.search(r"^(?:flags|Features)\s*:\s*(.+)$", text, re.M)
        have = set(fl.group(1).split()) if fl else set()
        for name, keys in (("AVX2", ["avx2"]), ("AVX-512F", ["avx512f"]), ("BMI2", ["bmi2"]), ("SHA-NI", ["sha_ni"]),
                           ("NEON", ["asimd", "neon"]), ("SVE", ["sve"]), ("SVE2", ["sve2"])):
            if any(k in have for k in keys):
                flags.add(name)
        pairs = set(re.findall(r"^physical id\s*:\s*(\d+)[\s\S]*?^core id\s*:\s*(\d+)", text, re.M))
        info["physical_cores"] = len(pairs) or None
    elif sys.platform == "darwin":
        info["model"] = _sysctl("machdep.cpu.brand_string")
        info["physical_cores"] = int(_sysctl("hw.physicalcpu") or 0) or None
        p0, p1 = _sysctl("hw.perflevel0.physicalcpu"), _sysctl("hw.perflevel1.physicalcpu")
        if p0:
            info["core_kinds"] = "%s performance + %s efficiency" % (p0, p1 or "0")
        for name, key in (("AVX2", "hw.optional.avx2_0"), ("NEON", "hw.optional.AdvSIMD"), ("NEON", "hw.optional.neon"),
                          ("SVE", "hw.optional.arm.FEAT_SVE"), ("SME", "hw.optional.arm.FEAT_SME")):
            if _sysctl(key) == "1":
                flags.add(name)
    elif IS_WIN:
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            info["model"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            info["model"] = platform.processor()
        try:
            present = ctypes.windll.kernel32.IsProcessorFeaturePresent
            for name, pf in (("AVX2", 40), ("AVX-512F", 41), ("NEON", 19), ("SVE", 46)):
                if present(pf):
                    flags.add(name)
        except Exception:
            pass
        cores = _powershell("(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum")
        info["physical_cores"] = int(cores) if cores.isdigit() else None
    info["flags"] = sorted(flags)
    return info


def ram_gb():
    if sys.platform.startswith("linux"):
        m = re.search(r"MemTotal:\s*(\d+)", _read("/proc/meminfo"))
        return round(int(m.group(1)) / 1048576, 1) if m else None
    if sys.platform == "darwin":
        v = _sysctl("hw.memsize")
        return round(int(v) / 2 ** 30, 1) if v.isdigit() else None
    if IS_WIN:
        class MS(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
        s = MS()
        s.len = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return round(s.total / 2 ** 30, 1)
    return None


def gpus():
    found = []
    smi = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi") else None)
    if smi:
        code, out, _, _ = run([smi, "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"], timeout=60)
        if code != 0:
            code, out, _, _ = run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], timeout=60)
        for line in out.strip().splitlines() if code == 0 else []:
            parts = [p.strip() for p in line.split(",")]
            found.append({"via": "nvidia-smi", "name": parts[0], "driver": parts[1] if len(parts) > 1 else "",
                          "memory": parts[2] if len(parts) > 2 else "", "compute_cap": parts[3] if len(parts) > 3 else ""})
        code, out, _, _ = run([smi], timeout=60)
        m = re.search(r"CUDA Version:\s*([\d.]+)", out or "")
        if m and found:
            for g in found:
                g["cuda_driver_api"] = m.group(1)
    vk = shutil.which("vulkaninfo")
    if vk:
        code, out, _, _ = run([vk, "--summary"], timeout=60)
        names = re.findall(r"deviceName\s*=\s*(.+)", out or "")
        drivers = re.findall(r"driverName\s*=\s*(.+)", out or "")
        infos = re.findall(r"driverInfo\s*=\s*(.+)", out or "")
        apis = re.findall(r"apiVersion\s*=\s*(.+)", out or "")
        for i, n in enumerate(names):
            found.append({"via": "vulkaninfo", "name": n.strip(), "driver": (drivers[i].strip() if i < len(drivers) else "") +
                          (" " + infos[i].strip() if i < len(infos) else ""), "api": apis[i].strip() if i < len(apis) else ""})
    if sys.platform == "darwin" and not found:
        code, out, _, _ = run(["system_profiler", "SPDisplaysDataType"], timeout=60)
        for m in re.finditer(r"Chipset Model:\s*(.+)", out or ""):
            found.append({"via": "system_profiler", "name": m.group(1).strip()})
    return found


def compilers():
    seen = {}
    for name in [os.environ.get("CC"), "cc", "gcc", "clang"]:
        if name and name not in seen and shutil.which(name):
            seen[name] = first_line([name, "--version"])
    nvcc = shutil.which("nvcc")
    if nvcc:
        code, out, _, _ = run([nvcc, "--version"], timeout=30)
        m = re.search(r"release ([\d.]+)", out or "")
        seen["nvcc"] = "CUDA " + m.group(1) if m else "present"
    if IS_WIN and shutil.which("cl"):
        code, out, err, _ = run(["cl"], timeout=30)
        seen["cl"] = (err or out).strip().splitlines()[0] if (err or out).strip() else "present"
    return seen


def _mount_point(path):
    path = os.path.abspath(path)
    while not os.path.ismount(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def drive_of(path):
    """What the OS says about the drive that holds `path`: bus, media, model; never the path."""
    if not path or not os.path.exists(path):
        return {"known": False, "why": "no container given"}
    full = os.path.abspath(path)
    if IS_WIN:
        if full.startswith("\\\\"):
            return {"known": False, "why": "a network share: the OS does not tell the drive"}
        letter = full[0].upper()
        out = _powershell(
            "$p = Get-Partition -DriveLetter %s -ErrorAction Stop; $d = Get-Disk -Number $p.DiskNumber; "
            "$m = (Get-PhysicalDisk | Where-Object DeviceId -eq ([string]$p.DiskNumber)).MediaType; "
            "'{0}|{1}|{2}' -f $d.BusType, $m, $d.FriendlyName" % letter)
        if "|" in out:
            bus, media, model = (out.splitlines()[-1].split("|") + ["", "", ""])[:3]
            return {"known": True, "bus": bus, "media": media, "model": model}
        return {"known": False, "why": "a mapped or virtual drive: the OS does not tell the drive"}
    if sys.platform == "darwin":
        code, out, _, _ = run(["diskutil", "info", _mount_point(full)], timeout=60)
        def field(name):
            m = re.search(r"^\s*%s:\s*(.+)$" % re.escape(name), out or "", re.M)
            return m.group(1).strip() if m else ""
        return {"known": code == 0, "bus": field("Protocol"), "media": "SSD" if field("Solid State") == "Yes" else field("Solid State"),
                "location": field("Device Location"), "model": field("Device / Media Name")}
    # Linux
    mp = _mount_point(full)
    fstype = ""
    for line in _read("/proc/mounts").splitlines():
        parts = line.split()
        if len(parts) > 2 and parts[1] == mp:
            fstype = parts[2]
    if fstype in ("9p", "drvfs", "virtiofs", "fuse.sshfs", "nfs", "nfs4", "cifs", "smb3"):
        return {"known": False, "fs": fstype, "why": "%s: a host or network volume, the drive is not visible from here" % fstype}
    try:
        st = os.stat(full)
        dev = os.path.realpath("/sys/dev/block/%d:%d" % (os.major(st.st_dev), os.minor(st.st_dev)))
        disk = dev if os.path.exists(os.path.join(dev, "queue")) else os.path.dirname(dev)
        rot = _read(os.path.join(disk, "queue", "rotational")).strip()
        model = _read(os.path.join(disk, "device", "model")).strip()
        bus = "NVMe" if "nvme" in disk else "USB" if "/usb" in disk else "SATA/SAS" if "/ata" in disk else "virtual" if "virtual" in disk else ""
        return {"known": True, "fs": fstype, "bus": bus, "media": {"0": "SSD", "1": "HDD"}.get(rot, ""), "model": model}
    except Exception as e:
        return {"known": False, "fs": fstype, "why": "not found (%s)" % type(e).__name__}


def os_info():
    info = {"system": platform.system(), "release": platform.release(), "python": platform.python_version()}
    if sys.platform.startswith("linux"):
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)', _read("/etc/os-release"), re.M)
        info["distro"] = m.group(1) if m else ""
        info["wsl"] = "microsoft" in platform.release().lower()
    elif sys.platform == "darwin":
        info["distro"] = "macOS " + platform.mac_ver()[0]
    elif IS_WIN:
        info["distro"] = "Windows " + platform.release() + " " + platform.version()
    return info


def stage_env(args):
    data = {"os": os_info(), "cpu": cpu_info(), "ram_gb": ram_gb(), "gpus": gpus(), "compilers": compilers(),
            "container_drive": drive_of(args.container), "checkpoint_drive": drive_of(args.snap) if args.snap else None}
    say("env:", json.dumps(SCRUB(data), indent=1))
    save_stage(args.out, "env", data)
    return 0


# ---- verify ------------------------------------------------------------------------------
def container_desc(path):
    return {"file": os.path.basename(path), "bytes": os.path.getsize(path)} if path and os.path.exists(path) else None


def stage_verify(args):
    if not args.container:
        say("verify: --container FILE is required")
        return 2
    tool = build("tools/wex4_verify", args.build)
    if not tool:
        return 2
    code, out, err, secs = run([tool, args.container, str(args.verify_threads)])
    root = re.search(r"COLI_WEX_ROOT=([0-9a-f]{64})", out)
    head = re.search(r"WEX4 (\d+)\.(\d+), (\d+) blobs, (\d+) records, (\d+) tensors, grid (\d+) x (\d+)", out)
    size = os.path.getsize(args.container)
    data = {"container": container_desc(args.container), "exit": code, "seconds": round(secs, 1),
            "gb_per_s": round(size / secs / 1e9, 3) if secs > 0 else None, "threads": args.verify_threads,
            "root": root.group(1) if root else None, "records": int(head.group(4)) if head else None,
            "grid": "%s x %s" % (head.group(6), head.group(7)) if head else None,
            "output": SCRUB((out + err).strip().splitlines()[-3:])}
    say("verify: exit %d, %.1f s, %.3f GB/s, root %s" % (code, secs, data["gb_per_s"] or 0, data["root"]))
    save_stage(args.out, "verify", data)
    return code


# ---- decode and io (tools/wex_decode_bench) ----------------------------------------------
def bench_tool(args, mode, extra):
    if not args.container:
        say(mode + ": --container FILE is required")
        return 2, None, ""
    tool = build("tools/wex_decode_bench", args.build)
    if not tool:
        return 2, None, ""
    env = dict(os.environ)
    if args.helper:
        env["COLI_WEX_CUDA_LIB"] = os.path.abspath(args.helper)
    cmd = [tool, mode, args.container, "--records", str(args.records), "--seed", str(args.seed)] + extra
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    code, out, err, secs = run(cmd, env=env)
    table = [l for l in out.splitlines() if not l.startswith("JSON ")]
    for l in table:
        say(l)
    if err.strip():
        say(err.strip()[-1500:])
    js = [l[5:] for l in out.splitlines() if l.startswith("JSON ")]
    data = json.loads(js[-1]) if js else None
    return code, data, "\n".join(table)


def stage_decode(args):
    extra = ["--lanes", args.lanes]
    if args.passes:
        extra += ["--passes", str(args.passes)]
    code, data, table = bench_tool(args, "decode", extra)
    if data is None:
        return code or 2
    root = re.search(r"root ([0-9a-f]{64})", table)
    data.update({"exit": code, "container": container_desc(args.container), "root": root.group(1) if root else None,
                 "helper": "set" if (args.helper or os.environ.get("COLI_WEX_CUDA_LIB", "0") not in ("", "0")) else "unset"})
    save_stage(args.out, "decode", data, {"COLI_WEX_CUDA_LIB": args.helper} if args.helper else None)
    return code


def stage_io(args):
    extra = ["--passes", str(args.passes)] if args.passes else []
    code, data, table = bench_tool(args, "io", extra)
    if data is None:
        return code or 2
    data.update({"exit": code, "container": container_desc(args.container)})
    save_stage(args.out, "io", data)
    return code


# ---- engine ------------------------------------------------------------------------------
ENGINE_DEFAULT_CAP = {"deepseek_v41": 8, "qwen38": 64, "custom": 0}


def parse_engine(engine, out, err):
    r = {}
    text = out + "\n" + err
    if engine == "deepseek_v41":
        ids = [l for l in out.splitlines() if re.fullmatch(r"\s*\d+(\s+\d+)*\s*", l)]
        r["tokens"] = [int(t) for t in ids[0].split()] if ids else []
        m = re.search(r"\[v41\] prefill (\d+) tokens in ([\d.]+)s \| decode (\d+) in ([\d.]+)s \(([\d.]+) tok/s\)", err)
        if m:
            r.update(prefill_s=float(m.group(2)), decode_s=float(m.group(4)), tok_s=float(m.group(5)))
        m = re.search(r"loaded in ([\d.]+)s", err)
        r["load_s"] = float(m.group(1)) if m else None
        m = re.search(r"\[v41\] experts: (\d+) hits, (\d+) misses, ([\d.]+) MB read", err)
        if m:
            r.update(expert_hits=int(m.group(1)), expert_misses=int(m.group(2)), mb_read=float(m.group(3)))
        m = re.search(r"\[v41_wex\] container: (\d+) records read \((\d+) WNR3 decoded\), (\d+) refused", err)
        if m:
            r.update(container_reads=int(m.group(1)), wnr3_decoded=int(m.group(2)), refused=int(m.group(3)))
        r["container_used"] = bool(m) and "is not used" not in err
    elif engine == "qwen38":
        m = re.search(r"C engine :\s*([\d ]+)", out)
        r["tokens"] = [int(t) for t in m.group(1).split()] if m else []
        m = re.search(r"TTFT: ([\d.]+) s", err)
        r["prefill_s"] = float(m.group(1)) if m else None
        m = re.search(r"Speed: ([\d.]+) tok/s \(([\d.]+)s for (\d+) tokens\)", err)
        if m:
            r.update(tok_s=float(m.group(1)), decode_s=float(m.group(2)))
        r["container_used"] = "COLI_WEX_PATH" in err or "[q38" in err
    else:
        import hashlib
        r["tokens"] = []
        r["stdout_sha256"] = hashlib.sha256(out.encode()).hexdigest()[:16]
        m = re.search(r"([\d.]+) tok/s", text)
        r["tok_s"] = float(m.group(1)) if m else None
    m = re.search(r"\[wex\] stats: helper=(\d+) device=(\d+) fast=(\d+) reference=(\d+) failed=(\d+) cpu=(\S+)", err)
    if m:
        r["decodes"] = {"helper": int(m.group(1)), "device": int(m.group(2)), "fast": int(m.group(3)),
                        "reference": int(m.group(4)), "failed": int(m.group(5)), "cpu_decoder": m.group(6)}
    m = re.search(r"\[wex\] WNR3 records decode on the CPU with the (.+)", err)
    r["cpu_decoder_line"] = m.group(1).strip() if m else None
    m = re.search(r"\[wex\] CUDA helper [^:]*: (.+)", err)
    r["helper_line"] = m.group(1).strip() if m else None
    return r


def stage_engine(args):
    if not args.snap:
        say("engine: --snap CHECKPOINT_DIR is required")
        return 2
    engine = args.engine
    SCRUB.add(args.snap, "<checkpoint>")
    if engine == "custom" and not args.cmd:
        say("engine custom: --cmd 'COMMAND ...' is required")
        return 2
    binary = None
    if engine != "custom":
        binary = os.path.abspath(args.engine_bin) if args.engine_bin else build(engine, args.build_engine)
        if not binary or not os.path.exists(binary):
            say("engine: no", engine, "binary (make -C c %s, or --engine-bin)" % engine)
            return 2
    ids = [int(x) for x in args.prompt_ids.split(",")] if args.prompt_ids else (V41_PROMPT_IDS if engine == "deepseek_v41" else None)
    if engine == "qwen38" and not ids:
        say("engine qwen38: --prompt-ids is required (the token ids of the prompt for the Qwen3.8 tokenizer)")
        return 2
    root = args.root or (load_results(args.out).get("verify") or {}).get("root")
    helper = os.path.abspath(args.helper) if args.helper else (os.environ.get("COLI_WEX_CUDA_LIB") or "")
    if helper == "0":
        helper = ""
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else \
        ["shards", "container-fast", "container-reference"] + (["container-helper"] if helper else [])
    if not args.container and any(a.startswith("container") for a in arms):
        say("engine: --container FILE is required for the container arms")
        return 2
    os.makedirs(args.out, exist_ok=True)
    ref = os.path.join(args.out, "wex-bench-prompt.json")
    if ids is not None:
        with open(ref, "w", encoding="utf-8") as f:
            if engine == "qwen38":
                json.dump({"prompt_ids": ids, "full_ids": ids + [0] * args.tokens}, f)
            else:
                json.dump({"prompt_ids": ids, "output_ids": [0] * args.tokens}, f)
    cap = args.cap or ENGINE_DEFAULT_CAP.get(engine, 8)
    results = []
    for arm in arms:
        env = {k: v for k, v in os.environ.items() if not k.startswith("COLI_WEX_")}
        env["SNAP"] = args.snap
        env["COLI_WEX_STATS"] = "1"
        if arm.startswith("container"):
            env["COLI_WEX_PATH"] = args.container
            if root:
                env["COLI_WEX_ROOT"] = root
            env["COLI_WEX_FAST_DECODE"] = "0" if arm == "container-reference" else "1"
            env["COLI_WEX_CUDA_LIB"] = helper if arm == "container-helper" else "0"
            if arm == "container-helper" and not helper:
                say("engine: arm container-helper needs --helper or COLI_WEX_CUDA_LIB")
                continue
        elif arm != "shards":
            say("engine: unknown arm", arm, "(shards, container-fast, container-reference, container-helper)")
            continue
        if engine == "deepseek_v41":
            cmd = [binary, str(cap), ref]
        elif engine == "qwen38":
            cmd = [binary, str(cap), str(args.q38_bits), ref]
        else:
            cmd = shlex.split(args.cmd, posix=not IS_WIN)
        say("engine %s, arm %s ..." % (engine, arm))
        code, out, err, secs = run(cmd, env=env, cwd=C_DIR, timeout=args.timeout)
        with open(os.path.join(args.out, "wex-bench-engine-%s.log" % arm), "w", encoding="utf-8") as f:
            f.write(SCRUB(out + "\n----- stderr -----\n" + err))
        r = parse_engine(engine, out, err)
        r.update(arm=arm, exit=code, wall_s=round(secs, 1),
                 env={k: ("<container>" if k == "COLI_WEX_PATH" else "<helper>" if k == "COLI_WEX_CUDA_LIB" and env[k] != "0"
                          else "<checkpoint>" if k == "SNAP" else env[k]) for k in sorted(env)
                      if k.startswith("COLI_WEX_") or k == "SNAP"})
        results.append(r)
        say("  exit %d, prefill %s s, decode %s s, %s tok/s, tokens %s, decodes %s" % (
            code, r.get("prefill_s"), r.get("decode_s"), r.get("tok_s"), " ".join(map(str, r.get("tokens", []))), r.get("decodes")))
    base = results[0].get("tokens") if results else None
    same = bool(results) and all(r.get("tokens") == base for r in results) and bool(base)
    if engine == "custom":
        same = bool(results) and len({r.get("stdout_sha256") for r in results}) == 1
    data = {"engine": engine, "cap": cap, "tokens_asked": args.tokens, "prompt_ids": ids, "root_pinned": bool(root),
            "container": container_desc(args.container), "arms": results, "same_tokens": same}
    say("engine: the same tokens in every arm" if same else "engine: TOKENS DIFFER between arms (or none were read)")
    save_stage(args.out, "engine", data)
    return 0 if same else 1


# ---- report ------------------------------------------------------------------------------
def git_info():
    def g(*a):
        return first_line(["git", "-C", REPO] + list(a))
    commit = g("rev-parse", "--short=12", "HEAD")
    if not commit:
        return {"commit": "unknown (no git)"}
    code, out, _, _ = run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], timeout=30)
    return {"commit": commit, "subject": g("log", "-1", "--format=%s"), "dirty": bool(out.strip())}


def fmt(v, nd=2):
    return "-" if v is None else ("%.*f" % (nd, v) if isinstance(v, float) else str(v))


def stage_report(args):
    res = load_results(args.out)
    if not res:
        say("report: no results in", args.out, "(run some stages first)")
        return 2
    conditions = SCRUB({"git": git_info(), "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")})
    L = ["## WEX bench report", ""]
    gi = conditions["git"]
    L.append("Commit `%s`%s (%s), %s. Generated by `c/tools/wex_bench.py`." % (
        gi.get("commit"), " with local changes" if gi.get("dirty") else "", gi.get("subject", ""), conditions["date"]))
    L.append("")
    env = res.get("env")
    if env:
        cpu, o = env.get("cpu", {}), env.get("os", {})
        L += ["### Machine", "", "| | |", "|---|---|",
              "| OS | %s%s |" % (o.get("distro") or o.get("system"), " (WSL2)" if o.get("wsl") else ""),
              "| CPU | %s, %s, %s physical / %s logical cores%s |" % (
                  cpu.get("model"), cpu.get("arch"), fmt(cpu.get("physical_cores")), fmt(cpu.get("logical_cores")),
                  ", " + cpu["core_kinds"] if cpu.get("core_kinds") else ""),
              "| CPU flags | %s |" % (", ".join(cpu.get("flags") or []) or "-"),
              "| RAM | %s GB |" % fmt(env.get("ram_gb"), 1),
              "| GPU | %s |" % ("; ".join("%s: %s (%s%s%s)" % (g.get("via"), g.get("name"), g.get("driver", ""),
                                                             ", " + g["memory"] if g.get("memory") else "",
                                                             ", sm " + g["compute_cap"] if g.get("compute_cap") else
                                                             ", Vulkan " + g["api"] if g.get("api") else "")
                                          for g in env.get("gpus") or []) or "none found"),
              "| Compiler | %s |" % ("; ".join("%s" % v for v in (env.get("compilers") or {}).values() if v) or "-")]
        d = env.get("container_drive") or {}
        L.append("| Container's drive | %s |" % (" ".join(x for x in (d.get("bus"), d.get("media"), d.get("model"), d.get("location")) if x)
                                               if d.get("known") else d.get("why", "-")))
        if env.get("checkpoint_drive"):
            d = env["checkpoint_drive"]
            L.append("| Checkpoint's drive | %s |" % (" ".join(x for x in (d.get("bus"), d.get("media"), d.get("model")) if x)
                                                    if d.get("known") else d.get("why", "-")))
        L.append("")
    v = res.get("verify")
    if v:
        c = v.get("container") or {}
        L += ["### verify (`tools/wex4_verify`)", "",
              "`%s`, %.2f GB, %s records (grid %s): exit %s, %s s on %s threads, **%s GB/s**; root `%s`." % (
                  c.get("file"), (c.get("bytes") or 0) / 1e9, fmt(v.get("records")), v.get("grid"), v.get("exit"), fmt(v.get("seconds"), 1),
                  v.get("threads"), fmt(v.get("gb_per_s"), 3), v.get("root")), ""]
    d = res.get("decode")
    if d and d.get("skipped"):
        L += ["### decode", "", "`%s`: %s." % ((d.get("container") or {}).get("file"), d["skipped"]), ""]
    elif d:
        c = d.get("container") or {}
        L += ["### decode (`tools/wex_decode_bench decode`)", "",
              "`%s` (root `%s`), %d WNR3 records picked by seed %s (%s), %d decoded bytes each, stored/decoded %.3f. "
              "Fast decoder kernel: %s; build default `WEX_FAST_LANES` %s. CUDA helper: %s." % (
                  c.get("file"), (d.get("root") or "")[:16] + "...", d.get("records", 0), d.get("seed"), ", ".join(d.get("records_picked") or []),
                  d.get("decoded_bytes", 0), (d.get("stored_bytes", 0) / max(1, d.get("decoded_bytes", 1) * d.get("records", 1))),
                  d.get("kernel"), d.get("build_lanes"), "features %s" % d["helper_features"] if d.get("helper_features") else d.get("helper")), "",
              "| arm | threads | ms/record, best | median | passes | GB/s decoded | bytes |", "|---|---:|---:|---:|---:|---:|---|"]
        ref = next((a for a in d.get("arms", []) if a["name"].startswith("reference") and a["threads"] == 1), None)
        for a in d.get("arms", []):
            L.append("| %s | %d | %.2f | %.2f | %d | %.3f | %s |" % (
                a["name"] + (" (%.1fx)" % (ref["best_ms"] / a["best_ms"]) if ref and a is not ref and a["best_ms"] > 0 else ""),
                a["threads"], a["best_ms"], a["median_ms"], a["passes"], a["gbs"], "MISMATCH" if a["mismatches"] else "same"))
        L += ["", "Every arm checked against the reference's bytes: **%s**." % ("identical" if d.get("identical") else "NOT identical"), ""]
    io = res.get("io")
    if io:
        c = io.get("container") or {}
        L += ["### io (`tools/wex_decode_bench io`)", "",
              "`%s`, %d records picked by the seed (%d RAW), %.1f MB, one read a record." % (
                  c.get("file"), io.get("records", 0), io.get("raw_records", 0), io.get("bytes", 0) / 1e6), "",
              "| read | threads | GB/s, best | median | ms/record | how |", "|---|---:|---:|---:|---:|---|"]
        for r in io.get("rows", []):
            L.append("| %s | %d | %.3f | %.3f | %.2f | %s |" % (r["read"], r["threads"], r["best_gbs"], r["median_gbs"], r["ms_per_record"], r["how"]))
        L.append("")
    e = res.get("engine")
    if e:
        L += ["### engine (`%s`)" % e.get("engine"), "",
              "Prompt ids `%s`, %s greedy tokens, %s expert slots a layer, root %s, container `%s`." % (
                  ",".join(map(str, e.get("prompt_ids") or [])), e.get("tokens_asked"), e.get("cap"),
                  "pinned (first-read SHA-256 on)" if e.get("root_pinned") else "not pinned",
                  (e.get("container") or {}).get("file")), "",
              "| arm | prefill s | decode s | tok/s | container reads (WNR3) | decodes: helper / fast / reference / failed | tokens |",
              "|---|---:|---:|---:|---|---|---|"]
        for a in e.get("arms", []):
            dc = a.get("decodes")
            L.append("| %s | %s | %s | %s | %s | %s | `%s` |" % (
                a["arm"], fmt(a.get("prefill_s")), fmt(a.get("decode_s")), fmt(a.get("tok_s")),
                "%s (%s)" % (a["container_reads"], a["wnr3_decoded"]) if "container_reads" in a else "-",
                "%d / %d / %d / %d" % (dc["helper"] + dc["device"], dc["fast"], dc["reference"], dc["failed"]) if dc else "-",
                " ".join(map(str, a.get("tokens") or [])) or a.get("stdout_sha256", "")))
        L += ["", "Tokens in every arm: **%s**." % ("the same" if e.get("same_tokens") else "DIFFERENT"), ""]
    L += ["### Conditions", "", "| stage | run at | variables set (the engine arms set their own `COLI_WEX_*`, `SNAP`, `COLI_WEX_STATS=1`) |",
          "|---|---|---|"]
    for k in ("env", "verify", "decode", "io", "engine"):
        if k in res:
            L.append("| %s | %s | %s |" % (k, res[k].get("when"), ", ".join("`%s=%s`" % kv for kv in (res[k].get("env_vars") or {}).items()) or "none"))
    L.append("")
    md = SCRUB("\n".join(L))
    with open(os.path.join(args.out, "wex-bench-report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(args.out, "wex-bench-report.json"), "w", encoding="utf-8") as f:
        json.dump(SCRUB({"conditions": conditions, "results": res}), f, indent=1)
    say(md)
    say("wrote wex-bench-report.md and wex-bench-report.json in the --out directory")
    return 0


STAGES = {"env": stage_env, "verify": stage_verify, "decode": stage_decode, "io": stage_io, "engine": stage_engine, "report": stage_report}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=list(STAGES) + ["all"])
    ap.add_argument("--container", help="a WEX4 container (WNR3 for decode; any for verify and io)")
    ap.add_argument("--out", default=".", help="where the results and the report go (default: here)")
    ap.add_argument("--records", type=int, default=16, help="records for decode and io (default 16)")
    ap.add_argument("--seed", type=int, default=1, help="which records (default 1)")
    ap.add_argument("--passes", type=int, default=0, help="passes per arm (default: decode 3, the reference 1; io 1)")
    ap.add_argument("--lanes", default="2,4,6,8", help="fast decoder stream counts (any of 1,2,3,4,6,8)")
    ap.add_argument("--threads", type=int, default=0, help="records at once / read threads (default: the OpenMP team, at most 8)")
    ap.add_argument("--verify-threads", type=int, default=8)
    ap.add_argument("--helper", help="the CUDA helper (kaira_wex_cuda.dll / .so); default: COLI_WEX_CUDA_LIB")
    ap.add_argument("--build", action="store_true", help="rebuild the C tools first")
    ap.add_argument("--engine", default="deepseek_v41", choices=["deepseek_v41", "qwen38", "custom"])
    ap.add_argument("--engine-bin", help="the engine binary (default: c/<engine>, built if missing)")
    ap.add_argument("--build-engine", action="store_true", help="rebuild the engine first")
    ap.add_argument("--snap", help="the checkpoint directory (the trunk; the shards for the shards arm)")
    ap.add_argument("--root", help="COLI_WEX_ROOT for the container arms (default: the verify stage's)")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--prompt-ids", help="comma-separated prompt token ids (deepseek_v41 default: BOS + 'The three primary colors of light are')")
    ap.add_argument("--cap", type=int, default=0, help="expert slots a layer (deepseek_v41 default 8, qwen38 64)")
    ap.add_argument("--q38-bits", type=int, default=8)
    ap.add_argument("--arms", help="comma-separated: shards, container-fast, container-reference, container-helper")
    ap.add_argument("--cmd", help="engine custom: the command to run in c/ for each arm (the bench sets SNAP and COLI_WEX_*)")
    ap.add_argument("--timeout", type=int, default=7200, help="seconds per engine run")
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    SCRUB.add(args.out, "<out>")
    if args.container:
        args.container = os.path.abspath(args.container)
        SCRUB.add(os.path.dirname(args.container), "<container dir>")
    if args.snap:
        args.snap = os.path.abspath(args.snap)
        SCRUB.add(args.snap, "<checkpoint>")
    stages = args.stages
    if "all" in stages:
        stages = ["env", "verify", "decode", "io"] + (["engine"] if args.snap else []) + ["report"]
    worst = 0
    for s in stages:
        code = STAGES[s](args)
        if code:
            say("stage %s: exit %d" % (s, code))
        worst = max(worst, code)
    return worst


if __name__ == "__main__":
    sys.exit(main())
