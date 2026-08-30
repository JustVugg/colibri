#!/usr/bin/env python3
"""datapoint.py - run the standard cross-engine colibri datapoint.

The default ``persistent`` mode starts one engine through the shared mux
``SERVE=1`` protocol and keeps it alive for the complete campaign: one cold
request, a repeated-prompt warm upper bound, then a deterministic rotation of
different prompts as the primary serving workload.  Expert, KV, GPU-residency
and allocator state therefore survive exactly as they do in ``coli serve``,
without letting one repeated prompt define the headline result.  The engine's
DONE frame supplies the real completion count, decode tok/s, expert hit rate
and RSS; no output re-tokenization is needed.

``--mode fresh-process`` preserves the historical runner for startup/cold-cache
work.  In that mode every row is a new process, so a row labelled warm means
only that the operating-system page cache may be warm -- the in-process expert
cache is empty again.  Keeping the two modes explicit prevents those different
states from being compared as if they were the same benchmark.

Before either campaign, the page cache is evicted when possible (macOS:
``purge``; Linux: ``posix_fadvise(DONTNEED)`` over the model shards).  iobench
runs immediately after eviction, before model loading can re-warm the files.

Usage:
  python tools/datapoint.py --snap models/olmoe_merged
  python tools/datapoint.py --snap models/v4 --engine ./deepseek_v4 \
      --rotating-runs 8
  python tools/datapoint.py --mode fresh-process --snap models/olmoe_merged \
      --engine ./olmoe
"""

import argparse
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LOAD_RE = re.compile(
    r"(?:resident weights loaded in|init done in|loaded \d+ layers in|\bloaded\b.*?\bin\b)\s+([\d.]+)s.*?\bRSS\b\s*(?:after load:\s*)?([\d.]+)\s*GB",
    re.IGNORECASE
)
TUNE_RE = re.compile(r"TUNE decode:\s*(\d+)\s*tokens in\s*([\d.]+)s", re.IGNORECASE)
FIRST_RE = re.compile(r"time_to_first_token=([\d.]+)s", re.IGNORECASE)
V4_TOKENS_RE = re.compile(r"v4_tokens.*?generated=(\d+)", re.IGNORECASE)
QWEN38_SPEED_RE = re.compile(
    r"Speed:\s*[\d.]+\s*tok/s\s*\(([\d.]+)s\s+for\s+(\d+)\s+tokens\)",
    re.IGNORECASE,
)
RAM_RE = re.compile(r"(?:projected=|dense=resident\(|available=)([\d.]+)G[iI]?B", re.IGNORECASE)
IOBENCH_RE = re.compile(r"-> ([\d.]+) GB/s")

HERE = Path(__file__).resolve().parent
C_ROOT = HERE.parent

# Fixed order, no network corpus and no RNG: two machines run the same request
# mix.  The prompts deliberately span reasoning, code, systems and multilingual
# prose so a MoE cache must serve changing routes instead of memorising one
# prompt.  --rotate-prompt replaces this suite for model-specific campaigns.
DEFAULT_ROTATING_PROMPTS = (
    "Explain how a database transaction can deadlock, give a concrete example, "
    "and compare two practical prevention strategies in detail.",
    "Write a Python function that merges overlapping intervals. Explain its "
    "correctness, time complexity, edge cases, and include two examples.",
    "A service has high median throughput but poor tail latency under load. "
    "Develop a step-by-step investigation plan covering CPU, memory, and storage.",
    "Rispondi in italiano: confronta memoria virtuale e memoria fisica, spiegando "
    "page fault, cache del filesystem e un esempio pratico.",
)


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
        # platform.release() says "10" on Windows 11: Microsoft never moved the
        # internal version off 10.0, so the BUILD number is the only tell —
        # 22000 and up is Windows 11 (#1042 follow-up: every Win11 datapoint in
        # the tracker was being filed as Windows 10).
        release = platform.release()
        try:
            build = int(platform.version().split(".")[2])
            if release == "10" and build >= 22000:
                release = "11"
        except (IndexError, ValueError):
            pass
        info["os"] = f"Windows {release} {platform.version()}"
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
    # Confirm the temp volume can hold it before writing a byte. This is the only
    # eviction route on Windows (macOS has purge, Linux has fadvise), and since
    # #1042 it writes the machine's real RAM rather than a hardcoded 8 GB, so the
    # write is now as large as the box. tempfile.gettempdir() is usually the
    # system drive while the model lives elsewhere, so on a 128 GB machine with
    # 124 GB free on C: this fills the system volume to zero and only then raises.
    tmp_dir = tempfile.gettempdir()
    need = size_mb * (1 << 20)
    try:
        free = shutil.disk_usage(tmp_dir).free
    except OSError:
        free = None
    if free is not None and free < need:
        print(f"[datapoint] cache eviction skipped: need {need / (1 << 30):.0f} GB "
              f"free in {tmp_dir} but only {free / (1 << 30):.0f} GB available. "
              f"Point TMPDIR (TEMP on Windows) at a larger volume, or pass "
              f"--no-evict and label the run warm.", file=sys.stderr)
        return False
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


def _load_server_runtime():
    """Import the shared gateway from either a source tree or an installation."""
    root = str(C_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import openai_server
    return openai_server


def physical_core_count():
    """Use the launcher's cross-platform topology probe, without making it required."""
    try:
        root = str(C_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from resource_plan import physical_cpu_count
        return physical_cpu_count()
    except (ImportError, OSError, ValueError):
        return None


def _persistent_executable(runtime, family, engine):
    if engine:
        name = os.path.basename(engine).lower()
        if name in ("coli", "coli.exe", "coli-wrapper", "coli-wrapper.exe"):
            raise ValueError(
                "persistent mode needs an engine binary, not the coli wrapper; "
                "omit --engine to select the model family's engine automatically"
            )
        path = Path(engine).expanduser()
    else:
        path = Path(runtime.default_engine(family))
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"{family.display_name} engine is not built: {path} "
            f"(run: make -C c {family.build_target})"
        )
    return str(path)


def _persistent_environment(family, max_new, memory_gb, physical_cores):
    env = dict(os.environ, COLI_TEMP="0", MAX_NEW=str(max_new), NGEN=str(max_new))
    if memory_gb is not None:
        env["RAM_GB"] = f"{memory_gb:g}"
    # Match the public coli launcher: sister engines use physical cores; V4's
    # runtime deliberately owns its compute/loader split (#897/#958).
    if (family.id != "deepseek_v4" and not env.get("COLI_NO_OMP_TUNE") and
            physical_cores):
        env.setdefault("OMP_NUM_THREADS", str(physical_cores))
    return env


def _measure_persistent_request(engine, prompt, max_new):
    first_text_at = [None]
    start = time.monotonic()
    profile_seq = getattr(engine, "profile_seq", 0)

    def on_text(_text):
        if first_text_at[0] is None:
            first_text_at[0] = time.monotonic()

    stats = engine.generate(prompt, max_new, 0.0, 1.0, on_text, cache_slot=0)
    wall = time.monotonic() - start
    tok_count = int(stats["completion_tokens"])
    tok_s = float(stats["tokens_per_second"])
    # The engine's own decode rate is authoritative.  This reconstructed
    # duration is useful for a compact table and avoids conflating prefill with
    # decode when the request wall includes both.
    family_id = getattr(getattr(engine, "family", None), "id", None)
    rate_tokens = max(tok_count - 1, 0) if family_id == "qwen38" else tok_count
    gen_s = rate_tokens / tok_s if rate_tokens > 0 and tok_s > 0 else 0.0
    ttft_s = ((first_text_at[0] - start) if first_text_at[0] is not None else wall)

    # Some engines emit PROF immediately before DONE, while DeepSeek V4 emits
    # it immediately after.  generate() returns on DONE, so give the dispatcher
    # a small bounded window to consume trailing telemetry.  This wait is not
    # included in request wall/TTFT and does not affect the authoritative tok/s.
    profile_deadline = time.monotonic() + 0.1
    while (getattr(engine, "profile_seq", 0) <= profile_seq and
           time.monotonic() < profile_deadline):
        time.sleep(0.001)
    profile = None
    if (getattr(engine, "profile_seq", 0) > profile_seq and
            getattr(engine, "profile", None)):
        profile = dict(engine.profile[-1])
    return {
        "tokens": tok_count,
        "prompt_tokens": int(stats.get("prompt_tokens", 0)),
        "wall_s": wall,
        "request_s": wall,
        "ttft_s": ttft_s,
        "gen_s": gen_s,
        "tok_s": tok_s,
        "hit": float(stats.get("cache_hit_percent", 0.0)),
        "rss": float(stats.get("rss_gb", 0.0)),
        "length_limited": bool(stats.get("length_limited", False)),
        "profile": profile,
    }


def _rotating_prompt_suite(prompts=None):
    selected = tuple(DEFAULT_ROTATING_PROMPTS if prompts is None else prompts)
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in selected):
        raise ValueError("rotating prompts must be non-empty strings")
    if len(set(selected)) < 2:
        raise ValueError("rotating workload needs at least two distinct prompts")
    return selected


def _render_prompt(runtime, prompt):
    return runtime.render_chat_for_arch(
        [{"role": "user", "content": prompt}], enable_thinking=False)


def run_persistent_engine(engine, snap, prompt, max_new, warmup_runs, warm_runs,
                          cap, memory_gb=None, runtime=None, physical_cores=None,
                          rotating_prompts=None, rotating_runs=4):
    """Measure repeated and rotating requests in one SERVE=1 process."""
    if rotating_runs < 2:
        raise ValueError("rotating_runs must be at least 2")
    rotating_prompt_source = ("custom" if rotating_prompts is not None
                              else "built-in")
    rotating_prompts = _rotating_prompt_suite(rotating_prompts)
    runtime = runtime or _load_server_runtime()
    resolved = runtime.resolve_model(snap)
    family = resolved.descriptor
    if not family.has_gateway_adapter:
        raise ValueError(f"{family.display_name} has no persistent gateway adapter")
    executable = _persistent_executable(runtime, family, engine)
    if physical_cores is None:
        physical_cores = physical_core_count()
    env = _persistent_environment(family, max_new, memory_gb, physical_cores)
    runtime.ARCH = family.id
    rendered = _render_prompt(runtime, prompt)
    rendered_rotating = [_render_prompt(runtime, item)
                         for item in rotating_prompts]

    started = time.monotonic()
    persistent = runtime.Engine(executable, snap, cap=cap, max_tokens=max_new,
                                env=env, kv_slots=1, family=family)
    load_s = time.monotonic() - started
    cold = []
    warmup = []
    warm = []
    rotating = []
    try:
        print("[datapoint] persistent cold request 1/1", file=sys.stderr)
        cold.append(_measure_persistent_request(persistent, rendered, max_new))
        for index in range(warmup_runs):
            print(f"[datapoint] discarded warm-up {index + 1}/{warmup_runs}",
                  file=sys.stderr)
            warmup.append(_measure_persistent_request(persistent, rendered, max_new))
        for index in range(warm_runs):
            print(f"[datapoint] warm-identical upper bound {index + 1}/{warm_runs}",
                  file=sys.stderr)
            warm.append(_measure_persistent_request(persistent, rendered, max_new))
        for index in range(rotating_runs):
            prompt_index = index % len(rendered_rotating)
            print(f"[datapoint] rotating prompt {index + 1}/{rotating_runs} "
                  f"(suite {prompt_index + 1}/{len(rendered_rotating)})",
                  file=sys.stderr)
            rotating.append(_measure_persistent_request(
                persistent, rendered_rotating[prompt_index], max_new))
        tiers = dict(persistent.tiers) if getattr(persistent, "tiers", None) else None
        hwinfo = dict(persistent.hwinfo) if getattr(persistent, "hwinfo", None) else None
    finally:
        # All measurements above are already collected; a teardown that fails
        # must not discard them. At --memory-gb 126 the engine can outlast the
        # close() grace period while it frees a 111 GB resident cache, and a
        # raised exception here would swallow the return and lose a complete run
        # (observed on the EPYC 7282 datapoint, #1154).
        try:
            persistent.close()
        except Exception as exc:  # noqa: BLE001 - teardown is best-effort
            print(f"[datapoint] warning: engine teardown did not complete "
                  f"cleanly, results are still valid: {exc}", file=sys.stderr)

    return {
        "mode": "persistent",
        "family": family.id,
        "family_name": family.display_name,
        "engine": executable,
        "load_s": load_s,
        "cold": cold,
        "warmup": warmup,
        "warm": warm,
        "rotating": rotating,
        "rotating_prompt_count": len(rotating_prompts),
        "rotating_prompt_source": rotating_prompt_source,
        "tiers": tiers,
        "hwinfo": hwinfo,
        "omp_threads": env.get("OMP_NUM_THREADS", "engine runtime auto"),
        "loader_lanes": (env.get("V4_LOADER_LANES", "engine default")
                         if family.id == "deepseek_v4" else "n/a"),
    }


def run_fresh_engine(engine, snap, prompt, max_new, runs, cap, bits, memory_gb=None):
    results = []
    engine_name = os.path.basename(engine).lower()

    for i in range(runs):
        env = dict(os.environ, CHAT="1", COLI_TEMP="0", MAX_NEW=str(max_new),
                   N_NEW=str(max_new), SNAP=snap)
        t0 = time.monotonic()
        prompt_path = None

        # Support modern engines and CLI wrappers
        # Exact basename check so 'colibri' isn't misclassified as 'coli-wrapper'
        if engine_name in ("coli", "coli-wrapper"):
            cmd = [engine, "run", "--model", snap, prompt, "--ngen", str(max_new)]
            stdin_input = None
        elif "deepseek" in engine_name:
            mem_str = str(int(memory_gb)) if memory_gb is not None else "32"
            cmd = [engine, snap, prompt, "--max-tokens", str(max_new), "--memory-gb", mem_str]
            stdin_input = None
        elif engine_name in ("qwen38", "qwen38.exe"):
            # Qwen3.8's direct CLI takes a prompt filename at argv[3]; stdin is
            # reserved for its persistent SERVE protocol.  Close the temporary
            # file before launch so this path also works on Windows.
            handle = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".txt", delete=False)
            try:
                handle.write(prompt)
                prompt_path = handle.name
            finally:
                handle.close()
            cmd = [engine, str(cap), str(bits), prompt_path]
            stdin_input = None
        else:
            # Legacy positional engines (olmoe, inkling, colibri)
            cmd = [engine, str(cap), str(bits)]
            stdin_input = prompt + "\n"

        try:
            proc = subprocess.run(cmd, input=stdin_input, capture_output=True,
                                  text=True, env=env)
        finally:
            if prompt_path:
                try:
                    os.unlink(prompt_path)
                except FileNotFoundError:
                    pass
        wall = time.monotonic() - t0
        output = proc.stdout + "\n" + proc.stderr
        if proc.returncode:
            sys.exit(
                f"engine exited with status {proc.returncode}:\n{output[-400:]}"
            )

        # 1. Try standard load & RSS regex
        m = LOAD_RE.search(output)
        if m:
            load_s, rss = float(m.group(1)), float(m.group(2))
            speed = QWEN38_SPEED_RE.search(output) if engine_name in ("qwen38", "qwen38.exe") else None
            if engine_name in ("qwen38", "qwen38.exe") and not speed:
                sys.exit(f"could not parse Qwen3.8 generation count/speed:\n{output[-400:]}")
            gen_s = float(speed.group(1)) if speed else wall - load_s
            tok_count = int(speed.group(2)) if speed else max_new
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
        results.append({"tokens": tok_count, "prompt_tokens": 0,
                        "wall_s": wall, "request_s": max(0.0, wall - load_s),
                        "ttft_s": None, "gen_s": gen_s, "tok_s": tok_s,
                        "hit": None, "rss": rss, "load_s": load_s,
                        "length_limited": tok_count >= max_new, "profile": None})
    return results


def run_iobench(iobench, shard, mode):
    proc = subprocess.run([iobench, shard, "19", "64", "8", str(mode)],
                          capture_output=True, text=True)
    m = IOBENCH_RE.search(proc.stdout)
    return float(m.group(1)) if m else None


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("persistent", "fresh-process"),
                    default="persistent",
                    help="one long-lived serve engine (default), or one process per row")
    ap.add_argument("--engine", default=None,
                    help="engine binary (default: select from model config.json)")
    ap.add_argument("--snap", required=True, help="model snapshot directory")
    ap.add_argument("--iobench", default="./iobench", help="iobench binary")
    ap.add_argument("--shard", default=None, help="shard file for iobench")
    ap.add_argument("--prompt", default="Continue this story: The lighthouse keeper climbed the stairs and saw something impossible in the fog. ",
                    help="prompt used for cold and warm-identical requests")
    ap.add_argument("--rotate-prompt", action="append", dest="rotating_prompts",
                    help="custom rotating prompt; repeat at least twice to replace the built-in suite")
    ap.add_argument("--max-new", type=int, default=128, help="decode cap per run")
    ap.add_argument("--warmup-runs", type=int, default=0,
                    help="extra discarded identical-prompt warm-ups (default: 0; cold already warms it)")
    ap.add_argument("--warm-runs", type=int, default=1,
                    help="measured identical-prompt upper-bound requests (default: 1)")
    ap.add_argument("--rotating-runs", type=int, default=4,
                    help="measured changing-prompt requests (primary result; default: 4)")
    ap.add_argument("--cap", type=int, default=16, help="per-layer expert cache cap passed to the engine")
    ap.add_argument("--bits", type=int, default=8,
                    help="quant bits passed by fresh-process legacy engines")
    ap.add_argument("--memory-gb", type=float, default=None,
                    help="RAM cap in GB (default: auto-derived from host RAM)")
    ap.add_argument("--no-evict", action="store_true", help="skip page-cache eviction")
    return ap


def _median(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def _percentile(rows, key, quantile):
    values = sorted(row[key] for row in rows if row.get(key) is not None)
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _cell(value, suffix="", digits=2):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def _append_result_row(out, label, result):
    out.append(
        f"| {label} | {result['prompt_tokens']} | {result['tokens']} | "
        f"{_cell(result['request_s'])} | {_cell(result['ttft_s'])} | "
        f"{_cell(result['tok_s'])} | {_cell(result['hit'], '%', 1)} | "
        f"{_cell(result['rss'], ' GB')} |"
    )


def _append_profile_summary(out, label, rows):
    profiles = [row["profile"] for row in rows if row.get("profile")]
    if not profiles:
        return
    fields = (
        ("expert disk", "expert_disk_s"),
        ("expert wait", "expert_wait_s"),
        ("expert matmul", "expert_matmul_s"),
        ("attention", "attention_s"),
        ("lm head", "lm_head_s"),
    )
    out += ["", f"**{label} profile medians (engine telemetry):**", "",
            "| phase | seconds/request |", "|---|---|"]
    for phase, key in fields:
        values = [profile[key] for profile in profiles if key in profile]
        if values:
            out.append(f"| {phase} | {statistics.median(values):.3f} |")


def _append_workload_summary(out, campaign):
    persistent = campaign["mode"] == "persistent"
    workloads = []
    if persistent and campaign.get("rotating"):
        workloads.append(("**rotating prompts (primary)**", campaign["rotating"]))
    if campaign.get("warm"):
        label = ("warm-identical (upper bound)" if persistent else
                 "fresh-process repeats")
        workloads.append((label, campaign["warm"]))
    if not workloads:
        return
    out += ["", "**Workload summary:**", "",
            "| workload | n | median tok/s | sigma tok/s | p95 request s | median hit |",
            "|---|---:|---:|---:|---:|---:|"]
    for label, rows in workloads:
        tok_values = [row["tok_s"] for row in rows]
        sigma = statistics.pstdev(tok_values) if len(tok_values) > 1 else 0.0
        out.append(
            f"| {label} | {len(rows)} | {_cell(_median(rows, 'tok_s'))} | "
            f"{sigma:.3f} | {_cell(_percentile(rows, 'request_s', 0.95))} | "
            f"{_cell(_median(rows, 'hit'), '%', 1)} |"
        )


def format_report(info, args, campaign, cold_label, disk):
    logical = info.get("cores") or "?"
    physical = info.get("physical_cores") or "?"
    engine_hw = campaign.get("hwinfo") or {}
    cpu = engine_hw.get("cpu") or info["cpu"]
    mode_note = (
        "one SERVE=1 process; expert/KV/GPU state retained across requests"
        if campaign["mode"] == "persistent" else
        "one process per row; warm rows retain OS page cache only"
    )
    family = campaign.get("family_name") or campaign.get("family") or "legacy"
    out = ["**Datapoint (automated runner)**", "", "| component | detail |",
           "|---|---|", f"| machine | {cpu} |",
           f"| RAM | {info['ram']} |",
           f"| CPU topology | {physical} physical / {logical} logical |",
           f"| OS | {info['os']} |",
           f"| mode | {campaign['mode']}: {mode_note} |",
           f"| engine | {os.path.basename(campaign['engine'])} ({family}) |",
           f"| snapshot | {args.snap} |",
           f"| cache settings | cap={args.cap}, RAM={args.memory_gb or info.get('ram_gb', '?')} GB |",
           f"| OpenMP team | {campaign.get('omp_threads', 'process default')} |",
           f"| V4 loader lanes | {campaign.get('loader_lanes', 'n/a')} |",
           f"| engine load | {campaign['load_s']:.2f} s |"]
    if engine_hw.get("gpu"):
        out.append(f"| GPU | {engine_hw['gpu']} ({engine_hw.get('gpus', '?')} device(s), "
                   f"{engine_hw.get('vram_total_gb', 0):.2f} GB VRAM) |")
    tiers = campaign.get("tiers")
    if tiers:
        out.append("| expert tiers | "
                   f"VRAM {tiers['vram']} / RAM {tiers['ram']} / disk {tiers['disk']} "
                   f"({tiers['vram_gb']:.2f} + {tiers['ram_gb']:.2f} GB resident) |")
    if campaign["mode"] == "persistent":
        out.append("| prompt workload | "
                   f"{campaign.get('rotating_prompt_count', 0)} "
                   f"{campaign.get('rotating_prompt_source', 'unknown')} prompts, "
                   "fixed rotation |")

    count_note = ("actual counts from engine" if campaign["mode"] == "persistent"
                  else "legacy count inference")
    out += ["", f"**Decode (COLI_TEMP=0, capped at {args.max_new} tokens; "
            f"{count_note}):**", "",
            "| phase | prompt tok | completion tok | request s | TTFT s | decode tok/s | hit | RSS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for result in campaign["cold"]:
        _append_result_row(out, cold_label, result)
    for index, result in enumerate(campaign.get("warmup", ())):
        _append_result_row(out, f"identical warm-up {index + 1} (discarded)", result)
    for index, result in enumerate(campaign["warm"]):
        label = (f"warm-identical {index + 1} (upper bound)"
                 if campaign["mode"] == "persistent" else
                 f"fresh-process repeat {index + 1}")
        _append_result_row(out, label, result)
    for index, result in enumerate(campaign.get("rotating", ())):
        _append_result_row(out, f"rotating {index + 1} (primary)", result)

    _append_workload_summary(out, campaign)
    if campaign.get("rotating"):
        _append_profile_summary(out, "Rotating workload", campaign["rotating"])
    if campaign.get("warm"):
        profile_label = ("Warm-identical upper bound"
                         if campaign["mode"] == "persistent" else
                         "Fresh-process repeats")
        _append_profile_summary(out, profile_label, campaign["warm"])

    if campaign["mode"] == "persistent":
        out += ["", "Rotating prompts are the primary serving result: each request changes prompt",
                "while the engine and expert cache remain alive. Warm-identical intentionally reuses",
                "the same prompt and cache slot and is reported only as a cache/KV upper bound."]
    else:
        out += ["", "Fresh-process warm rows do not retain the engine's expert, KV, allocator, or GPU state."]

    if args.shard:
        out += ["", "**Disk (iobench, 19 MB x 64, 8 threads):**", "",
                "| mode | GB/s |", "|---|---|"]
        for name, val in disk.items():
            out.append(f"| {name} | {val:.2f} |" if val is not None else f"| {name} | n/a |")
        if sys.platform == "darwin":
            out += ["", "Note: macOS has no O_DIRECT; iobench uses F_NOCACHE, which stops new caching",
                    "but cannot evict pages already resident. The cold figure was taken immediately",
                    "after the attempted cache eviction, before the engine loaded the container."]
    return "\n".join(out)


def _auto_engine_for_fresh(snap):
    runtime = _load_server_runtime()
    family = runtime.resolve_model(snap).descriptor
    return _persistent_executable(runtime, family, None), family


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.max_new < 1:
        ap.error("--max-new must be positive")
    if args.warmup_runs < 0:
        ap.error("--warmup-runs cannot be negative")
    if args.warm_runs < 1:
        ap.error("--warm-runs must be positive")
    if args.memory_gb is not None and args.memory_gb <= 0:
        ap.error("--memory-gb must be positive")
    if args.mode == "persistent":
        if args.rotating_runs < 2:
            ap.error("--rotating-runs must be at least 2")
        try:
            _rotating_prompt_suite(args.rotating_prompts)
        except ValueError as error:
            ap.error(str(error))

    info = machine_info()
    info["physical_cores"] = physical_core_count()
    mem_gb = args.memory_gb if args.memory_gb is not None else info.get("ram_gb", 32.0)
    evicted = False if args.no_evict else evict_cache(info.get("ram_gb", 8.0), snap_dir=args.snap)
    cold_label = ("cold request (cache evicted before engine load)" if evicted else
                  "cold request (page cache retained)")

    disk = {}
    if args.shard and evicted:
        # true cold disk figure first, before decode re-warms the container
        disk["cold"] = run_iobench(args.iobench, args.shard, 1)

    if args.mode == "persistent":
        campaign = run_persistent_engine(
            args.engine, args.snap, args.prompt, args.max_new,
            args.warmup_runs, args.warm_runs, args.cap,
            memory_gb=mem_gb, physical_cores=info["physical_cores"],
            rotating_prompts=args.rotating_prompts,
            rotating_runs=args.rotating_runs)
    else:
        family = None
        engine = args.engine
        if engine is None:
            engine, family = _auto_engine_for_fresh(args.snap)
        cold = run_fresh_engine(engine, args.snap, args.prompt, args.max_new, 1,
                                args.cap, args.bits, memory_gb=mem_gb)
        warm = run_fresh_engine(engine, args.snap, args.prompt, args.max_new,
                                args.warm_runs, args.cap, args.bits, memory_gb=mem_gb)
        campaign = {
            "mode": "fresh-process",
            "family": family.id if family else None,
            "family_name": family.display_name if family else None,
            "engine": str(Path(engine).expanduser()),
            "load_s": cold[0]["load_s"],
            "cold": cold,
            "warmup": [],
            "warm": warm,
            "rotating": [],
            "tiers": None,
            "hwinfo": None,
            "omp_threads": os.environ.get("OMP_NUM_THREADS", "process default"),
            "loader_lanes": os.environ.get("V4_LOADER_LANES", "engine default"),
        }

    if args.shard:
        if "cold" not in disk:
            disk["cold"] = run_iobench(args.iobench, args.shard, 1)
        disk["buffered"] = run_iobench(args.iobench, args.shard, 0)

    print(format_report(info, args, campaign, cold_label, disk))


if __name__ == "__main__":
    main()
