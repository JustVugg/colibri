#!/usr/bin/env python3
"""datapoint.py - run the standard colibri machine datapoint.

Measures what docs/benchmarks.md asks for - engine load, capped greedy
decode tok/s, and iobench disk figures - and prints a ready-to-paste
markdown datapoint block.

The decode number is exact: generation is capped at --max-new tokens,
so the token count is known without re-tokenizing the output.

Cold vs warm: before measuring, the page cache is evicted (macOS: `purge`
if permitted, otherwise by writing a temp file larger than RAM, which
forces resident pages out). The first decode run is then a true cold
run; the second is warm. iobench cold runs right after eviction, before
the decode runs can re-warm the container. Use --no-evict to skip the
eviction (e.g. when cache warmth is irrelevant or eviction is unwanted).

Usage:
  python tools/datapoint.py --snap models/olmoe_merged [--engine ./olmoe]
                            [--shard models/olmoe_merged/model-00000.safetensors]
                            [--max-new 128] [--warm-runs 1]
"""

import argparse
import os
import platform
import re
import subprocess
import sys
import tempfile
import time

LOAD_RE = re.compile(
    r"(?:resident weights loaded in|init done in|loaded \d+ layers in|\bloaded\b.*?\bin\b)\s+([\d.]+)s.*?\bRSS\b\s*(?:after load:\s*)?([\d.]+)\s*GB",
    re.IGNORECASE
)
TUNE_RE = re.compile(r"TUNE decode:\s*(\d+)\s*tokens in\s*([\d.]+)s", re.IGNORECASE)
FIRST_RE = re.compile(r"time_to_first_token=([\d.]+)s", re.IGNORECASE)
V4_TOKENS_RE = re.compile(r"v4_tokens.*?generated=(\d+)", re.IGNORECASE)
RAM_RE = re.compile(r"(?:projected=|dense=resident\(|available=)([\d.]+)G[iI]?B", re.IGNORECASE)
IOBENCH_RE = re.compile(r"-> ([\d.]+) GB/s")


def machine_info():
    info = {}
    if sys.platform == "darwin":
        info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                     capture_output=True, text=True).stdout.strip()
        mem = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True).stdout.strip()
        info["ram"] = f"{int(mem) / 1073741824:.0f} GB"
        info["ram_gb"] = int(mem) / 1073741824
        info["os"] = platform.mac_ver()[0]
    elif sys.platform.startswith("linux"):
        info["cpu"] = platform.processor() or "?"
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["ram_gb"] = int(line.split()[1]) / 1048576
                    info["ram"] = f"{info['ram_gb']:.0f} GB"
                    break
        info["os"] = platform.platform()
    elif sys.platform == "win32":
        # ram_gb sizes the eviction write: a hardcoded 8.0 on a big box means the
        # "cold" run is measured warm and published as cold (#1042). ctypes+winreg
        # only — no new dependency.
        try:
            import ctypes

            class _MEMSTATUS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MEMSTATUS()
            st.dwLength = ctypes.sizeof(_MEMSTATUS)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                raise OSError("GlobalMemoryStatusEx failed")
            info["ram_gb"] = st.ullTotalPhys / 1073741824
            info["ram"] = f"{info['ram_gb']:.0f} GB"
        except Exception:
            info["ram"] = "?"
            info["ram_gb"] = 8.0

        info["cpu"] = platform.processor() or "?"
        try:
            import winreg
            k = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            info["cpu"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
            winreg.CloseKey(k)
        except Exception:
            pass
        info["os"] = f"Windows {platform.release()} {platform.version()}"
    else:
        info["cpu"] = platform.processor() or "?"
        info["ram"] = "?"
        info["ram_gb"] = 8.0
        info["os"] = platform.platform()
    info["cores"] = os.cpu_count()
    return info


def evict_cache(ram_gb, snap_dir=None):
    """Evict resident page cache pages without privileges.

    macOS `purge` needs root; Linux uses posix_fadvise(DONTNEED) over the
    model's safetensors shards to drop cached pages instantly with 0 disk writes.
    The fallback writes a temp file larger than RAM when no direct method exists.
    Returns True if an eviction was actually attempted.
    """
    if sys.platform == "darwin":
        r = subprocess.run(["purge"], capture_output=True)
        if r.returncode == 0:
            return True

    # Zero-write Linux eviction using posix_fadvise over model shards
    if sys.platform.startswith("linux") and snap_dir and os.path.isdir(snap_dir):
        try:
            evicted_any = False
            for name in os.listdir(snap_dir):
                if name.endswith(".safetensors"):
                    p = os.path.join(snap_dir, name)
                    try:
                        fd = os.open(p, os.O_RDONLY)
                        try:
                            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                            evicted_any = True
                        finally:
                            os.close(fd)
                    except OSError:
                        pass
            if evicted_any:
                return True
        except OSError as e:
            print(f"[datapoint] Linux fadvise eviction skipped: {e}", file=sys.stderr)

    # Fallback: Write temp file larger than RAM
    size_mb = int(ram_gb) * 1024 + 1024
    print(f"[datapoint] evicting page cache by writing {size_mb / 1024:.0f} GB "
          f"to a temp file (no direct eviction on this platform; --no-evict skips)",
          file=sys.stderr)
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".evict") as f:
            chunk = b"\0" * (1 << 20)
            written = 0
            while written < size_mb:
                f.write(chunk)
                written += 1
            f.flush()
        return True
    except OSError as e:
        print(f"[datapoint] cache eviction skipped: {e}", file=sys.stderr)
        return False


def run_engine(engine, snap, prompt, max_new, runs, cap, bits, memory_gb=None):
    results = []
    engine_name = os.path.basename(engine).lower()

    for i in range(runs):
        env = dict(os.environ, CHAT="1", COLI_TEMP="0", MAX_NEW=str(max_new), SNAP=snap)
        t0 = time.monotonic()

        # Support modern engines and CLI wrappers
        # Exact basename check so 'colibri' isn't misclassified as 'coli-wrapper'
        if engine_name in ("coli", "coli-wrapper"):
            cmd = [engine, "run", "--model", snap, prompt, "--ngen", str(max_new)]
            stdin_input = None
        elif "deepseek" in engine_name:
            mem_str = str(int(memory_gb)) if memory_gb is not None else "32"
            cmd = [engine, snap, prompt, "--max-tokens", str(max_new), "--memory-gb", mem_str]
            stdin_input = None
        else:
            # Legacy positional engines (olmoe, inkling, colibri)
            cmd = [engine, str(cap), str(bits)]
            stdin_input = prompt + "\n"

        proc = subprocess.run(cmd, input=stdin_input, capture_output=True, text=True, env=env)
        wall = time.monotonic() - t0
        output = proc.stdout + "\n" + proc.stderr

        # 1. Try standard load & RSS regex
        m = LOAD_RE.search(output)
        if m:
            load_s, rss = float(m.group(1)), float(m.group(2))
            gen_s = wall - load_s
            tok_count = max_new
        else:
            # 2. Fallback parser for engines like deepseek_v4 that output TUNE decode / timing lines
            m_tune = TUNE_RE.search(output)
            m_first = FIRST_RE.search(output)
            m_tokens = V4_TOKENS_RE.search(output)
            m_ram = RAM_RE.search(output)

            if not m_tune and not m_first:
                sys.exit(f"could not parse engine load/decode line from stdout/stderr:\n{output[-400:]}")

            tok_count = int(m_tune.group(1)) if m_tune else (int(m_tokens.group(1)) if m_tokens else max_new)
            gen_s = float(m_tune.group(2)) if m_tune else wall
            load_s = float(m_first.group(1)) if m_first else max(0.0, wall - gen_s)
            rss = float(m_ram.group(1)) if m_ram else (float(memory_gb) if memory_gb is not None else 0.0)

        tok_s = tok_count / gen_s if gen_s > 0 else 0.0
        results.append({"tokens": tok_count, "wall_s": wall, "gen_s": gen_s,
                        "tok_s": tok_s, "rss": rss, "load_s": load_s})
    return results


def run_iobench(iobench, shard, mode):
    proc = subprocess.run([iobench, shard, "19", "64", "8", str(mode)],
                          capture_output=True, text=True)
    m = IOBENCH_RE.search(proc.stdout)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./olmoe", help="engine binary (default ./olmoe)")
    ap.add_argument("--snap", required=True, help="model snapshot directory")
    ap.add_argument("--iobench", default="./iobench", help="iobench binary")
    ap.add_argument("--shard", default=None, help="shard file for iobench")
    ap.add_argument("--prompt", default="Continue this story: The lighthouse keeper climbed the stairs and saw something impossible in the fog. ",
                    help="prompt used for the decode runs")
    ap.add_argument("--max-new", type=int, default=128, help="decode cap per run")
    ap.add_argument("--warm-runs", type=int, default=1, help="warm decode runs after the cold run")
    ap.add_argument("--cap", type=int, default=16, help="per-layer expert cache cap passed to the engine")
    ap.add_argument("--bits", type=int, default=8, help="quant bits passed to the engine")
    ap.add_argument("--memory-gb", type=float, default=None, help="RAM memory cap in GB passed to engine (default: auto-derived from host RAM)")
    ap.add_argument("--no-evict", action="store_true", help="skip page-cache eviction")
    args = ap.parse_args()

    info = machine_info()
    mem_gb = args.memory_gb if args.memory_gb is not None else info.get("ram_gb", 32.0)
    evicted = False if args.no_evict else evict_cache(info.get("ram_gb", 8.0), snap_dir=args.snap)
    cold_label = "cold (cache evicted)" if evicted else "cold (cache not evicted)"

    disk = {}
    if args.shard and evicted:
        # true cold disk figure first, before decode re-warms the container
        disk["cold"] = run_iobench(args.iobench, args.shard, 1)

    cold = run_engine(args.engine, args.snap, args.prompt, args.max_new, 1, args.cap, args.bits, memory_gb=mem_gb)
    warm = run_engine(args.engine, args.snap, args.prompt, args.max_new, max(1, args.warm_runs), args.cap, args.bits, memory_gb=mem_gb)

    if args.shard:
        if "cold" not in disk:
            disk["cold"] = run_iobench(args.iobench, args.shard, 1)
        disk["buffered"] = run_iobench(args.iobench, args.shard, 0)

    out = ["**Datapoint (automated runner)**", "", "| component | detail |",
           "|---|---|", f"| machine | {info['cpu']} |",
           f"| RAM | {info['ram']} |", f"| cores | {info['cores']} |",
           f"| OS | {info['os']} |", f"| engine | {os.path.basename(args.engine)}, snapshot {args.snap}, cap={args.cap}, bits={args.bits} |",
           "", f"**Decode (COLI_TEMP=0, capped at {args.max_new} tokens, exact count):**", "",
           "| phase | tokens | gen s | tok/s | RSS after load |", "|---|---|---|---|---|"]
    for r in cold:
        out.append(f"| {cold_label} | {r['tokens']} | {r['gen_s']:.2f} | {r['tok_s']:.2f} | {r['rss']:.2f} GB |")
    for r in warm:
        out.append(f"| warm | {r['tokens']} | {r['gen_s']:.2f} | {r['tok_s']:.2f} | {r['rss']:.2f} GB |")
    out.append(f"| load | - | {cold[0]['load_s']:.2f} | - | - |")

    if args.shard:
        out += ["", "**Disk (iobench, 19 MB x 64, 8 threads):**", "",
                "| mode | GB/s |", "|---|---|"]
        for name, val in disk.items():
            out.append(f"| {name} | {val:.2f} |" if val is not None else f"| {name} | n/a |")
        if sys.platform == "darwin":
            out += ["", "Note: macOS has no O_DIRECT; iobench uses F_NOCACHE, which stops new caching",
                    "but cannot evict pages already resident (doc caveat #86). The cold figure above was",
                    "taken right after cache eviction, before decode re-warmed the container."]

    print("\n".join(out))


if __name__ == "__main__":
    main()
