"""Measured, quality-preserving execution tuning for colibri.

The tuner changes execution scheduling only.  It never sweeps quantization,
expert selection, sampling, or model weights.  A calibration continuation is
generated once and then teacher-forced for every candidate so every run sees
the same token and routing inputs.
"""
from __future__ import annotations

import hashlib
import json
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
    "COLI_CUDA_PIPE", "COLI_CUDA_ASYNC",
})


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
    env = profile.get("winner", {}).get("env", {})
    if not isinstance(env, dict) or any(key not in TUNABLE_KEYS for key in env):
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
    return steps


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
             progress=None, family=None, engine_cls=None) -> tuple[dict, Path]:
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
    with context as directory:
        if arch == "glm":
            ref = Path(directory) / "replay.json"
            ref.write_text(json.dumps(replay), encoding="utf-8")

            def run_once(name, overlay):
                env = dict(base_env)
                env.update(overlay)
                env.update({"REF": str(ref), "REF_FORCE": "1", "REPLAY": "1", "PROF": "1"})
                env.pop("PROMPT", None); env.pop("TOKENS", None)
                proc = _run([engine, str(cap)], env, timeout)
                output = proc.stdout + "\n" + proc.stderr
                if proc.returncode:
                    raise RuntimeError(f"{name} failed ({proc.returncode})\n{output[-2000:]}")
                return parse_replay(output)
        else:
            if engine_cls is None:
                from openai_server import Engine as engine_cls
            expected = []      # bytes of the session's first run: the contract

            def run_once(name, overlay):
                env = dict(base_env)
                env.update(overlay)
                env["COLI_TEMP"] = "0"
                served = engine_cls(engine, model, cap, tokens, env, 1, family)
                pieces = []
                try:
                    result = served.generate(prompt, tokens, 0.0, 1.0,
                                             pieces.append)
                finally:
                    served.close()
                text = "".join(pieces)
                if not expected:
                    expected.append(text)
                elif text != expected[0]:
                    raise OutputDrift(
                        f"{name} changed the generated tokens; a scheduling "
                        f"knob must never do that")
                tok_s = float(result.get("tokens_per_second") or 0.0)
                if tok_s <= 0:
                    raise RuntimeError(
                        f"{name}: engine reported no decode speed")
                return {"tok_s": tok_s,
                        "hit_pct": result.get("cache_hit_percent"),
                        "p50_ms": None, "p99_ms": None}

        def measure(name, overlay):
            samples = []
            for repeat in range(repeats):
                progress(f"{name} ({repeat + 1}/{repeats})")
                samples.append(run_once(name, overlay))
            return {
                "name": name,
                "env": dict(overlay),
                "tok_s": statistics.median(sample["tok_s"] for sample in samples),
                "hit_pct": statistics.median(
                    sample["hit_pct"] for sample in samples if sample["hit_pct"] is not None
                ) if any(sample["hit_pct"] is not None for sample in samples) else None,
                "p99_ms": statistics.median(
                    sample["p99_ms"] for sample in samples if sample["p99_ms"] is not None
                ) if any(sample["p99_ms"] is not None for sample in samples) else None,
                "samples": samples,
            }

        baseline = measure("baseline", {})
        winner = baseline
        accumulated = {}
        candidates = [baseline]
        for name, change in candidate_steps(plan, base_env, arch):
            trial_env = dict(accumulated)
            trial_env.update(change)
            try:
                trial = measure(name, trial_env)
            except OutputDrift as drift:
                progress(f"{name}: {drift} -- disqualified")
                continue
            candidates.append(trial)
            hit_ok = (baseline["hit_pct"] is None or trial["hit_pct"] is None
                      or trial["hit_pct"] >= baseline["hit_pct"] - 0.5)
            tail_ok = (baseline["p99_ms"] is None or trial["p99_ms"] is None
                       or trial["p99_ms"] <= baseline["p99_ms"] * 1.20)
            if hit_ok and tail_ok and trial["tok_s"] > winner["tok_s"] * (1.0 + min_gain):
                winner = trial
                accumulated = trial_env

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
                confirmed_winner = measure("confirm-winner", winner["env"])
            except OutputDrift as drift:
                progress(f"confirm-winner: {drift} -- profile rejected")
                confirmed_winner = None
            if confirmed_winner is None:
                winner = baseline
                validation = {"winner": None, "baseline": None,
                              "hit_gate": False, "tail_gate": False}
                accepted = False
                gain = 0.0
            else:
                confirmed_winner["name"] = winner["name"]
                confirmed_baseline = measure("confirm-baseline", {})
                hit_ok = (confirmed_baseline["hit_pct"] is None
                          or confirmed_winner["hit_pct"] is None
                          or confirmed_winner["hit_pct"] >= confirmed_baseline["hit_pct"] - 0.5)
                tail_ok = (confirmed_baseline["p99_ms"] is None
                           or confirmed_winner["p99_ms"] is None
                           or confirmed_winner["p99_ms"] <= confirmed_baseline["p99_ms"] * 1.20)
                gain = confirmed_winner["tok_s"] / confirmed_baseline["tok_s"] - 1.0
                accepted = hit_ok and tail_ok and gain >= min_gain
                validation = {
                    "winner": confirmed_winner,
                    "baseline": confirmed_baseline,
                    "hit_gate": hit_ok,
                    "tail_gate": tail_ok,
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
