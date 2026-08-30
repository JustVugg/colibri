"""Measured, quality-preserving execution tuning for colibri.

The tuner changes execution scheduling only.  It never sweeps quantization,
expert selection, sampling, or model weights.  GLM uses its fixed-token replay
protocol.  The sibling engines keep one serve process alive per candidate and
rotate deterministic prompts, so their expert caches behave like a real chat
instead of being reset between every sample.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCHEMA_VERSION = 1
PROMPT_RE = re.compile(r"\[PROMPT_TOKENS\]\s+\d+:\s*([0-9 ]+)")
TOKENS_RE = re.compile(r"\[TOKENS\]\s+\d+\s+generated:\s*([0-9 ]+)")
# Two shapes, one meaning. colibri emits the richer REPLAY line from its fixed
# oracle replay; the sibling engines emit `TUNE decode: N tokens in T.TTTs` at
# the end of an ordinary greedy run, which is deterministic for a fixed prompt
# and NGEN and therefore comparable across candidates. Accepting both is what
# lets `coli tune` work on more than GLM (#898).
TIMING_RE = re.compile(r"(?:REPLAY|TUNE) decode:\s+(\d+)\s+tokens in\s+([0-9.]+)s")
SPEED_RE = re.compile(r"REPLAY decode:\s+\d+\s+tokens.*?\|\s*([0-9.]+)\s+tok/s")
HIT_RE = re.compile(r"expert hit\s+([0-9.]+)%")
LATENCY_RE = re.compile(r"latency p50\s+([0-9.]+)\s*ms.*?p99\s+([0-9.]+)\s*ms")

# Only scheduling / placement knobs belong here.  Quality-affecting knobs such
# as TOPK, TOPP, DRAFT, quantization and CACHE_ROUTE are intentionally excluded.
TUNABLE_KEYS = frozenset({
    "OMP_NUM_THREADS", "COLI_NUMA", "PIPE", "DIRECT",
    "COLI_CUDA_PIPE", "COLI_CUDA_ASYNC", "V4_LOADER_LANES",
    "RAM_GB", "K3_EXPERT_GB",
})

GIB = 1024 ** 3
CAP_ARCHES = frozenset({"glm", "inkling", "olmoe", "qwen36", "qwen38"})


def _config_path(profile_dir: str | None, fingerprint: str) -> Path:
    if profile_dir:
        root = Path(profile_dir).expanduser()
    elif os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", "~/AppData/Local")).expanduser() / "colibri" / "tuning"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser() / "colibri" / "tuning"
    return root / f"{fingerprint}.json"


def machine_fingerprint(plan: dict, model: str, engine: str) -> str:
    """Stable identity for execution-relevant hardware, model, and engine."""
    model_path = Path(model)
    files = []
    paths = [model_path / name for name in ("config.json", "model.safetensors.index.json")]
    paths.extend(sorted(model_path.glob("*.safetensors")))
    for path in paths:
        name = path.name
        try:
            stat = path.stat()
            files.append((name, stat.st_size, stat.st_mtime_ns))
        except OSError:
            files.append((name, None, None))
    try:
        engine_stat = Path(engine).stat()
        engine_id = (engine_stat.st_size, engine_stat.st_mtime_ns)
    except OSError:
        engine_id = (None, None)
    cpu_model = platform.processor()
    if sys_platform := getattr(platform, "system", lambda: "")():
        cpu_model = f"{sys_platform}:{platform.machine()}:{cpu_model}"
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model += ":" + line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    payload = {
        "schema": SCHEMA_VERSION,
        "cpu_model": cpu_model,
        "cpu": plan.get("cpu"),
        "gpus": [
            {"name": gpu.get("name"), "total_bytes": gpu.get("total_bytes")}
            for gpu in plan.get("tiers", {}).get("vram", {}).get("devices", [])
        ],
        "model": str(model_path.resolve()),
        "model_files": files,
        "engine": engine_id,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def load_profile(plan: dict, model: str, engine: str, profile_dir: str | None = None) -> dict | None:
    fingerprint = machine_fingerprint(plan, model, engine)
    path = _config_path(profile_dir, fingerprint)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (profile.get("schema_version") != SCHEMA_VERSION
            or profile.get("fingerprint") != fingerprint
            or not profile.get("accepted")):
        return None
    winner = profile.get("winner", {})
    env = winner.get("env", {}) if isinstance(winner, dict) else None
    if not isinstance(env, dict) or any(key not in TUNABLE_KEYS for key in env):
        return None
    # Available RAM is deliberately absent from the machine fingerprint: it is
    # volatile, so including it would create a new profile whenever another
    # application opened.  Resource winners therefore need a second admission
    # gate at load time.  A profile measured yesterday may be reused today only
    # when its cap/budget still fits the freshly-built safe plan.
    ram = plan.get("tiers", {}).get("ram", {})
    cap = winner.get("cap")
    if cap is not None:
        safe_cap = ram.get("cache_slots_per_layer")
        if (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1
                or not isinstance(safe_cap, int) or cap > safe_cap):
            return None
    for key, divisor, plan_key in (
            ("RAM_GB", GIB, "budget_bytes"),
            ("K3_EXPERT_GB", 1_000_000_000, "expert_cache_bytes")):
        if key not in env:
            continue
        try:
            requested = float(env[key])
            safe = float(ram[plan_key]) / divisor
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        # Values are persisted to three decimals, hence the one-MiB-ish
        # rounding allowance.  NaN/inf and zero are never useful budgets.
        if (not math.isfinite(requested) or requested <= 0
                or requested > safe + 0.001):
            return None
    return profile


def apply_profile(env: dict, profile: dict, explicit_keys=()) -> dict:
    """Apply a validated profile without overriding the caller's environment."""
    result = dict(env)
    explicit = set(explicit_keys)
    for key, value in profile["winner"]["env"].items():
        if key not in explicit:
            result[key] = str(value)
    return result


def save_profile(profile: dict, profile_dir: str | None = None) -> Path:
    path = _config_path(profile_dir, profile["fingerprint"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(profile, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return path


def candidate_steps(plan: dict, base_env: dict, arch: str = "glm") -> list[tuple[str, dict]]:
    """Return a bounded coordinate-descent sweep for this topology and engine.

    `arch` gates the knobs that are not universal. OMP_NUM_THREADS and COLI_NUMA
    are read by every engine (they are OpenMP and NUMA, not engine features), so
    they are always eligible. PIPE and DIRECT are read by colibri alone --
    `grep -c 'getenv("PIPE")'` is 3 in colibri.c and 0 in the other four -- so
    offering them elsewhere would sweep candidates that cannot differ, spend the
    replay budget proving it, and report a 0% gain as if it were a measurement
    (#898)."""
    steps = []
    cores = max(1, int(plan.get("cpu", {}).get("physical_cores", os.cpu_count() or 1)))
    current_threads = int(base_env.get("OMP_NUM_THREADS", cores))
    if not base_env.get("COLI_NO_OMP_TUNE"):
        for threads in dict.fromkeys((cores, max(1, cores // 2))):
            if threads != current_threads:
                steps.append((f"omp-{threads}", {"OMP_NUM_THREADS": str(threads)}))
    sockets = int(plan.get("cpu", {}).get("sockets", 1))
    if sockets > 1 and base_env.get("COLI_NUMA") != "1":
        steps.append(("numa-on", {"COLI_NUMA": "1"}))
    has_gpu = bool(plan.get("tiers", {}).get("vram", {}).get("devices"))
    if has_gpu:
        pipe = int(base_env.get("COLI_CUDA_PIPE", "0"))
        for value in (1, 2):
            if value != pipe:
                steps.append((f"cuda-pipe-{value}", {"COLI_CUDA_PIPE": str(value)}))
        steps.append(("cuda-sync", {"COLI_CUDA_ASYNC": "0"}))
    if arch == "glm" and plan.get("tiers", {}).get("disk", {}).get("cold_expert_bytes", 0) > 0:
        direct = int(base_env.get("DIRECT", "0"))
        if direct != 1:
            steps.append(("direct-on", {"DIRECT": "1"}))
        pipe = int(base_env.get("PIPE", "0"))
        if pipe != 1:
            steps.append(("io-pipe-on", {"PIPE": "1"}))
    if (arch == "deepseek_v4"
            and plan.get("tiers", {}).get("disk", {}).get("cold_expert_bytes", 0) > 0):
        # The pool default (9) was measured on one CPU/NVMe pair.  GPU-tier
        # systems often prefer 3 because a deeper reader pool contends with
        # bank refill, while other SSDs keep scaling past 9.  Sweep a bounded
        # set around both measured optima; lanes block in pread and therefore
        # do not need one CPU apiece.
        current_lanes = int(base_env.get("V4_LOADER_LANES", "9"))
        lane_values = (3, 6, 9, 12)
        for lanes in lane_values:
            if lanes != current_lanes:
                steps.append((f"v4-loader-{lanes}",
                              {"V4_LOADER_LANES": str(lanes)}))
    return steps


def _scaled_ram_budget(plan: dict, fraction: float) -> float | None:
    """Scale only the expert-cache portion, never dense/runtime reservations."""
    ram = plan.get("tiers", {}).get("ram", {})
    try:
        budget = int(ram["budget_bytes"])
        cache = max(0, int(ram["expert_cache_bytes"]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    fixed = max(0, budget - cache)
    return (fixed + int(cache * fraction)) / GIB


def resource_candidate_steps(plan: dict, base_env: dict, arch: str, cap: int,
                             explicit_resources=()) -> list[tuple[str, dict, int]]:
    """Return bounded, never-larger RAM/cache candidates for disk-backed MoE.

    The planner's current result is the upper bound.  We only remove 25% or
    50% of the expert-cache portion, leaving dense weights, KV, workspaces and
    OS reserve untouched.  This can recover page-cache/I/O throughput on a
    memory-pressured host, but the ordinary hit/TTFT/tail gates decide whether
    that trade is actually worthwhile on this model and workload.
    """
    if plan.get("tiers", {}).get("disk", {}).get("cold_expert_bytes", 0) <= 0:
        return []
    explicit = set(explicit_resources)
    fractions = (0.75, 0.50)
    steps = []

    if arch in CAP_ARCHES:
        if ("cap" in explicit or cap <= 1
                or (arch == "glm" and "RAM_GB" in explicit)):
            return []
        seen = set()
        for fraction in fractions:
            candidate_cap = max(1, int(cap * fraction))
            if candidate_cap == cap or candidate_cap in seen:
                continue
            seen.add(candidate_cap)
            overlay = {}
            if arch == "glm" and "RAM_GB" not in explicit:
                budget = _scaled_ram_budget(plan, fraction)
                if budget is not None:
                    overlay["RAM_GB"] = f"{budget:.3f}"
            steps.append((f"cache-{int(fraction * 100)}", overlay, candidate_cap))
        return steps

    if arch == "deepseek_v4":
        if "RAM_GB" in explicit:
            return []
        seen = set()
        for fraction in fractions:
            budget = _scaled_ram_budget(plan, fraction)
            if budget is None:
                continue
            value = f"{budget:.3f}"
            if value == base_env.get("RAM_GB") or value in seen:
                continue
            seen.add(value)
            steps.append((f"ram-cache-{int(fraction * 100)}",
                          {"RAM_GB": value}, cap))
        return steps

    if arch == "kimi":
        if "K3_EXPERT_GB" in explicit:
            return []
        try:
            requested = float(base_env.get("K3_EXPERT_GB", "8"))
            planned = (float(plan["tiers"]["ram"]["expert_cache_bytes"])
                       / 1_000_000_000)
        except (KeyError, TypeError, ValueError, OverflowError):
            return []
        baseline = min(requested, planned)
        if not math.isfinite(baseline) or baseline <= 0:
            return []
        seen = set()
        for fraction in fractions:
            value = f"{baseline * fraction:.3f}"
            if value in seen:
                continue
            seen.add(value)
            steps.append((f"k3-cache-{int(fraction * 100)}",
                          {"K3_EXPERT_GB": value}, cap))
        return steps

    return []


def parse_replay(output: str) -> dict:
    timing = TIMING_RE.search(output)
    speed = SPEED_RE.search(output)
    if not timing and not speed:
        raise ValueError("engine emitted neither a REPLAY nor a TUNE decode line")
    # Prefer tokens/elapsed: the printed tok/s carries two decimals, which is one
    # significant digit at 0.04 tok/s and decides the 3% gate on rounding (#852).
    if timing and float(timing.group(2)) > 0:
        tok_s = int(timing.group(1)) / float(timing.group(2))
    elif speed:
        tok_s = float(speed.group(1))
    else:
        raise ValueError("TUNE decode line reported zero elapsed time")
    hit = HIT_RE.search(output)
    latency = LATENCY_RE.search(output)
    return {
        "tok_s": tok_s,
        "hit_pct": float(hit.group(1)) if hit else None,
        "p50_ms": float(latency.group(1)) if latency else None,
        "p99_ms": float(latency.group(2)) if latency else None,
    }


def _summarize_measurement(name: str, overlay: dict, samples: list[dict],
                           cap: int | None = None) -> dict:
    """Reduce one candidate's fixed request budget to comparable medians."""
    summary = {
        "name": name,
        "env": dict(overlay),
        "tok_s": statistics.median(sample["tok_s"] for sample in samples),
        "hit_pct": statistics.median(
            sample["hit_pct"] for sample in samples
            if sample["hit_pct"] is not None
        ) if any(sample["hit_pct"] is not None for sample in samples) else None,
        "p99_ms": statistics.median(
            sample["p99_ms"] for sample in samples
            if sample["p99_ms"] is not None
        ) if any(sample["p99_ms"] is not None for sample in samples) else None,
        "ttft_s": statistics.median(
            sample["ttft_s"] for sample in samples
            if sample.get("ttft_s") is not None
        ) if any(sample.get("ttft_s") is not None for sample in samples) else None,
        "samples": samples,
    }
    if cap is not None:
        summary["cap"] = cap
    return summary


def _safety_gates(baseline: dict, trial: dict) -> dict:
    return {
        "hit_gate": (baseline["hit_pct"] is None or trial["hit_pct"] is None
                     or trial["hit_pct"] >= baseline["hit_pct"] - 0.5),
        "tail_gate": (baseline["p99_ms"] is None or trial["p99_ms"] is None
                      or trial["p99_ms"] <= baseline["p99_ms"] * 1.20),
        "ttft_gate": (baseline.get("ttft_s") is None or trial.get("ttft_s") is None
                      or trial["ttft_s"] <= baseline["ttft_s"] * 1.20),
    }


class OutputDrift(RuntimeError):
    """A run produced different tokens than the session's first run.

    The sweep only offers quality-preserving scheduling knobs, so any output
    change disqualifies that candidate outright (or, on the baseline itself,
    proves the engine is not deterministic and cannot be tuned this way)."""


def _run(command: list[str], env: dict, timeout: int) -> subprocess.CompletedProcess:
    if Path(command[0]).suffix.lower() == ".py":
        command = [sys.executable, *command]
    return subprocess.run(command, env=env, text=True, capture_output=True, timeout=timeout)


def create_replay(engine: str, cap: int, env: dict, prompt: str, tokens: int,
                  timeout: int) -> tuple[dict, str]:
    calibration = dict(env, PROMPT=prompt, NGEN=str(tokens), TOKENS="1", PROF="1")
    proc = _run([engine, str(cap)], calibration, timeout)
    output = proc.stdout + "\n" + proc.stderr
    if proc.returncode:
        raise RuntimeError(f"calibration failed ({proc.returncode})\n{output[-2000:]}")
    prompt_match, token_match = PROMPT_RE.search(output), TOKENS_RE.search(output)
    if not prompt_match or not token_match:
        raise RuntimeError("engine did not emit calibration token trace; rebuild the engine")
    prompt_ids = [int(value) for value in prompt_match.group(1).split()]
    continuation = [int(value) for value in token_match.group(1).split()]
    if len(prompt_ids) < 2 or not continuation:
        raise RuntimeError("calibration produced an empty token trace")
    return {"prompt_ids": prompt_ids, "full_ids": prompt_ids + continuation}, output


def run_tune(engine: str, cap: int, base_env: dict, plan: dict, model: str,
             prompt: str, arch: str = "glm",
             tokens: int = 16, repeats: int = 2, timeout: int = 900,
             min_gain: float = 0.03, profile_dir: str | None = None,
             progress=None, family=None, engine_cls=None,
             prompts=None, explicit_resources=()) -> tuple[dict, Path]:
    """Coordinate-descent tuning with a reverse-order confirmation gate."""
    progress = progress or (lambda _message: None)
    if cap is None:
        # --cap has default=None ("auto"). str(None) used to reach the engine
        # as argv[1]="None", which atoi() turns into 0: qwen36 refuses it and
        # tuning dies with "calibration failed (1)", while GLM silently swept
        # its platform default instead of the cap the plan just resolved
        # (#1190). The plan is authoritative here, exactly like `coli chat`;
        # a plan without a resolved cap is a loud error, never "None".
        cap = plan.get("tiers", {}).get("ram", {}).get("cache_slots_per_layer")
        if cap is None:
            raise ValueError(
                "--cap not given and the resource plan carries no resolved "
                "cache_slots_per_layer")
    import contextlib
    # GLM measures through its native replay protocol (REF/REF_FORCE/REPLAY,
    # the strongest fixed-workload guarantee). The sibling engines implement
    # none of it (#1191) but ALL of them speak the serve protocol -- it is how
    # the gateway and tools/datapoint.py drive every engine in production --
    # so their measurement runs go through openai_server.Engine instead: one
    # greedy request per run (COLI_TEMP=0, a deterministic fixed workload) with
    # the DONE frame supplying decode tok/s and cache hit%. Quality is not
    # assumed but enforced: the first run's bytes are the contract, and any
    # run that emits different bytes raises OutputDrift -- a drifting
    # candidate is disqualified, a drifting baseline aborts the tune.
    if arch == "glm":
        replay, _ = create_replay(engine, cap, base_env, prompt, tokens, timeout)
        context = tempfile.TemporaryDirectory(prefix="coli-tune-")
    else:
        replay = None
        context = contextlib.nullcontext()
    serving_prompts = tuple(prompts or (prompt,))
    if not serving_prompts or any(not isinstance(item, str) or not item
                                  for item in serving_prompts):
        raise ValueError("tuning prompts must be non-empty strings")
    with context as directory:
        if arch == "glm":
            ref = Path(directory) / "replay.json"
            ref.write_text(json.dumps(replay), encoding="utf-8")

            def run_once(name, overlay, launch_cap):
                env = dict(base_env)
                env.update(overlay)
                env.update({"REF": str(ref), "REF_FORCE": "1", "REPLAY": "1", "PROF": "1"})
                env.pop("PROMPT", None); env.pop("TOKENS", None)
                proc = _run([engine, str(launch_cap)], env, timeout)
                output = proc.stdout + "\n" + proc.stderr
                if proc.returncode:
                    raise RuntimeError(f"{name} failed ({proc.returncode})\n{output[-2000:]}")
                return parse_replay(output)
        else:
            if engine_cls is None:
                from openai_server import Engine as engine_cls
            # One expected byte stream per prompt. Different prompts naturally
            # produce different text; the same prompt changing across
            # candidates is the quality violation.
            expected = {}

            def measure(name, overlay, launch_cap):
                env = dict(base_env)
                env.update(overlay)
                env["COLI_TEMP"] = "0"
                served = engine_cls(engine, model, launch_cap, tokens, env, 1, family)
                samples = []
                try:
                    # Keep this engine alive for the whole candidate. `repeats`
                    # is still the request budget, but requests now rotate
                    # instead of repeatedly teaching the cache one prompt.
                    for repeat in range(repeats):
                        active_prompt = serving_prompts[repeat % len(serving_prompts)]
                        progress(f"{name} ({repeat + 1}/{repeats}, prompt "
                                 f"{repeat % len(serving_prompts) + 1}/"
                                 f"{len(serving_prompts)})")
                        pieces = []
                        started = time.perf_counter()
                        first_text_at = [None]

                        def collect(piece):
                            if piece and first_text_at[0] is None:
                                first_text_at[0] = time.perf_counter()
                            pieces.append(piece)

                        result = served.generate(active_prompt, tokens, 0.0, 1.0,
                                                 collect)
                        finished = time.perf_counter()
                        text = "".join(pieces)
                        contract = expected.get(active_prompt)
                        if contract is None:
                            expected[active_prompt] = text
                        elif text != contract:
                            raise OutputDrift(
                                f"{name} changed the generated tokens for prompt "
                                f"{repeat % len(serving_prompts) + 1}; a scheduling "
                                f"knob must never do that")
                        tok_s = float(result.get("tokens_per_second") or 0.0)
                        if tok_s <= 0:
                            raise RuntimeError(
                                f"{name}: engine reported no decode speed")
                        reported_ttft = result.get("time_to_first_token")
                        if reported_ttft is None:
                            ttft = ((first_text_at[0] or finished) - started)
                        else:
                            ttft = float(reported_ttft)
                        if not math.isfinite(ttft) or ttft < 0:
                            raise RuntimeError(f"{name}: engine reported invalid TTFT")
                        samples.append({
                            "tok_s": tok_s,
                            "hit_pct": result.get("cache_hit_percent"),
                            "p50_ms": None,
                            "p99_ms": None,
                            "ttft_s": ttft,
                        })
                finally:
                    served.close()
                recorded_cap = launch_cap if arch in CAP_ARCHES else None
                return _summarize_measurement(name, overlay, samples, recorded_cap)

        if arch == "glm":
            def measure(name, overlay, launch_cap):
                samples = []
                for repeat in range(repeats):
                    progress(f"{name} ({repeat + 1}/{repeats})")
                    sample = run_once(name, overlay, launch_cap)
                    sample["ttft_s"] = None
                    samples.append(sample)
                recorded_cap = launch_cap if arch in CAP_ARCHES else None
                return _summarize_measurement(name, overlay, samples, recorded_cap)

        baseline = measure("baseline", {}, cap)
        winner = baseline
        accumulated = {}
        winner_cap = cap
        candidates = [baseline]

        def consider(name, change, trial_cap):
            nonlocal winner, accumulated, winner_cap
            trial_env = dict(accumulated)
            trial_env.update(change)
            try:
                trial = measure(name, trial_env, trial_cap)
            except OutputDrift as drift:
                progress(f"{name}: {drift} -- disqualified")
                return
            candidates.append(trial)
            gates = _safety_gates(baseline, trial)
            if (all(gates.values())
                    and trial["tok_s"] > winner["tok_s"] * (1.0 + min_gain)):
                winner = trial
                accumulated = trial_env
                winner_cap = trial_cap

        for name, change in candidate_steps(plan, base_env, arch):
            consider(name, change, winner_cap)
        for name, change, trial_cap in resource_candidate_steps(
                plan, base_env, arch, cap, explicit_resources):
            consider(name, change, trial_cap)

        validation = None
        accepted = False
        gain = 0.0
        if winner is not baseline:
            # Candidate trials necessarily run after the first baseline and can
            # benefit from thermal/page-cache drift.  Confirm in the opposite
            # order: winner first, baseline last.  This is deliberately
            # conservative — a later baseline gets any remaining warm-cache
            # advantage.  A profile is accepted only if it still wins.
            try:
                confirmed_winner = measure("confirm-winner", winner["env"], winner_cap)
            except OutputDrift as drift:
                progress(f"confirm-winner: {drift} -- profile rejected")
                confirmed_winner = None
            if confirmed_winner is None:
                winner = baseline
                validation = {"winner": None, "baseline": None,
                              "hit_gate": False, "tail_gate": False,
                              "ttft_gate": False}
                accepted = False
                gain = 0.0
            else:
                confirmed_winner["name"] = winner["name"]
                confirmed_baseline = measure("confirm-baseline", {}, cap)
                gates = _safety_gates(confirmed_baseline, confirmed_winner)
                gain = confirmed_winner["tok_s"] / confirmed_baseline["tok_s"] - 1.0
                accepted = all(gates.values()) and gain >= min_gain
                validation = {
                    "winner": confirmed_winner,
                    "baseline": confirmed_baseline,
                    **gates,
                }
                if accepted:
                    winner = confirmed_winner
    fingerprint = machine_fingerprint(plan, model, engine)
    profile = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(Path(model).resolve()),
        "accepted": accepted,
        "minimum_gain": min_gain,
        "gain": gain if accepted else 0.0,
        "baseline": baseline,
        "winner": winner if accepted else baseline,
        "candidates": candidates,
        "validation": validation,
        "plan": plan,
    }
    return profile, save_profile(profile, profile_dir)
