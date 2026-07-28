"""Deterministic RAM-disk benchmark execution and score construction.

The module owns benchmark computation but not lifecycle locking or durable
state policy.  Those operations are supplied by the facade at call time so
the benchmark layer remains import-safe and independently testable.
"""

from __future__ import print_function

import concurrent.futures
import datetime
import hashlib
import os
import queue
import re
import subprocess
import threading
import time

from .common import (
    BENCHMARK_SCHEMA,
    MANIFEST_VERSION,
    RamdiskError,
    _EngineCleanupError,
    _OperationCancelled,
    _percentile,
    _raise_if_cancelled,
    _utc_now,
)


BENCHMARK_PROMPT = (
    "Explain in two sentences why deterministic validation matters."
)


def _source_build_identity(
    source_file,
    *,
    environ,
    which,
    run,
):
    """Return best-effort revision metadata for reproducible reports."""
    explicit = environ.get("COLI_BUILD_COMMIT")
    if explicit:
        return {
            "revision": explicit,
            "working_tree_modified": None,
        }
    git = which("git")
    if not git:
        return {
            "revision": None,
            "working_tree_modified": None,
        }
    source_dir = os.path.dirname(os.path.abspath(source_file))
    revision = run(
        [git, "-C", source_dir, "rev-parse", "HEAD"]
    )
    if revision.returncode:
        return {
            "revision": None,
            "working_tree_modified": None,
        }
    status = run(
        [git, "-C", source_dir, "status", "--porcelain"]
    )
    return {
        "revision": revision.stdout.strip() or None,
        "working_tree_modified": (
            None
            if status.returncode
            else bool(status.stdout.strip())
        ),
    }


def _parse_profiler(text, elapsed):
    rates = [
        float(value)
        for value in re.findall(
            r"([0-9]+(?:\.[0-9]+)?)\s*tok(?:en)?s?/s",
            text,
            re.I,
        )
    ]
    forward_p50 = None
    forward_p99 = None
    match = re.search(
        r"forward[^\n]*p50[=: ]+([0-9.]+)\s*ms"
        r"[^\n]*p99[=: ]+([0-9.]+)\s*ms",
        text,
        re.I,
    )
    if match:
        forward_p50 = float(match.group(1))
        forward_p99 = float(match.group(2))
    ram_experts = None
    ram_bytes = None
    match = re.search(
        r"RAM map:\s*(\d+) experts / ([0-9.]+) GB",
        text,
    )
    if match:
        ram_experts = int(match.group(1))
        ram_bytes = float(match.group(2)) * 1e9
    io_bytes = None
    match = re.search(
        r"(?:physical SSD|disk I/O|expert I/O)"
        r"[^\n]*?([0-9.]+)\s*(GB|MB|bytes)",
        text,
        re.I,
    )
    if match:
        scale = {
            "gb": 1e9,
            "mb": 1e6,
            "bytes": 1,
        }[match.group(2).lower()]
        io_bytes = float(match.group(1)) * scale
    prefault = None
    match = re.search(
        r"prefaulted in ([0-9.]+)s",
        text,
    )
    if match:
        prefault = float(match.group(1))
    ttft_ms = None
    match = re.search(
        r"TTFT\s+([0-9.]+)s",
        text,
        re.I,
    )
    if match:
        ttft_ms = float(match.group(1)) * 1000.0
    rss_bytes = None
    match = re.search(
        r"\bRSS\s+([0-9.]+)\s+GB",
        text,
        re.I,
    )
    if match:
        rss_bytes = float(match.group(1)) * 1e9
    return {
        "elapsed_seconds": elapsed,
        "tokens_per_second": (
            rates[-1]
            if rates
            else (32.0 / elapsed if elapsed else None)
        ),
        "forward_p50_ms": forward_p50,
        "forward_p99_ms": forward_p99,
        "rammap_experts": ram_experts,
        "rammap_bytes": ram_bytes,
        "physical_ssd_bytes": io_bytes,
        "prefault_seconds": prefault,
        "ttft_ms": ttft_ms,
        "rss_bytes": rss_bytes,
    }


def _normalized_runtime_knobs(
    plan,
    knobs,
    node=None,
    *,
    node_core_count,
):
    """Validate the managed benchmark-knob vocabulary for one target."""
    result = {}
    thread_limit = node_core_count(plan, node)
    for key, value in (knobs or {}).items():
        if key in ("PIPE", "DIRECT", "URING"):
            parsed = int(value)
            if parsed not in (0, 1):
                raise RamdiskError(
                    "%s benchmark knob must be 0 or 1" % key
                )
            result[key] = parsed
        elif key == "PIPE_WORKERS":
            parsed = int(value)
            if not 1 <= parsed <= max(64, thread_limit):
                raise RamdiskError(
                    "PIPE_WORKERS benchmark knob is outside its safe range"
                )
            result[key] = parsed
        elif key == "OMP_NUM_THREADS":
            parsed = int(value)
            if not 1 <= parsed <= thread_limit:
                raise RamdiskError(
                    "OMP_NUM_THREADS=%s exceeds the %s-core benchmark target"
                    % (parsed, thread_limit)
                )
            result[key] = parsed
        elif key == "OMP_PROC_BIND":
            if value not in ("close", "spread"):
                raise RamdiskError(
                    "OMP_PROC_BIND benchmark knob must be close or spread"
                )
            result[key] = value
        else:
            raise RamdiskError(
                "unsupported managed benchmark knob: %s" % key
            )
    return result


def _benchmark_environment(
    manifest,
    weights_dir,
    state_dir,
    rammap,
    node=None,
    knobs=None,
    *,
    environ,
    node_core_count,
    engine_cpu_list,
    memory_node_list,
    managed_numa_enabled,
    normalized_runtime_knobs,
    apply_managed_accelerator_environment=None,
):
    if apply_managed_accelerator_environment is None:
        from .accelerator import _apply_managed_accelerator_environment

        apply_managed_accelerator_environment = (
            _apply_managed_accelerator_environment
        )
    plan = manifest["plan"]
    runtime = plan.get("managed_runtime", {})
    environment = environ.copy()
    for inherited in (
        "COLI_MMAP",
        "PIN",
        "PIN_GB",
        "PIN_FILL",
        "RAM_GB",
        "COLI_RAM_OVERCOMMIT",
        "CUDA_EXPERT_GB",
        "CUDA_DENSE",
        "COLI_GPUS",
        "COLI_GPU",
        "COLI_CUDA",
        "COLI_METAL",
        "COLI_NO_OMP_TUNE",
        "COLI_OMP_TUNED",
        "DIRECT",
        "PIPE",
        "PIPE_WORKERS",
        "URING",
        "DSA_TOPK",
        "GRAMMAR",
    ):
        environment.pop(inherited, None)
    environment.update(
        {
            "COLI_WEIGHTS_DIR": weights_dir,
            "COLI_STATE_DIR": state_dir,
            "COLI_NUMA": (
                "1"
                if managed_numa_enabled(plan, node)
                else "0"
            ),
            "COLI_NUMA_NODES": memory_node_list(
                plan,
                node=node,
            ),
            "COLI_CPU_AFFINITY": engine_cpu_list(
                plan,
                node=node,
            ),
            "OMP_NUM_THREADS": str(
                node_core_count(plan, node)
            ),
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "TEMP": "0",
            "DRAFT": "0",
            "KVSAVE": "0",
            "AUTOPIN": "0",
            "REPIN": "0",
            "CACHE_ROUTE": "0",
            "TOPK": "0",
            "TOPP": "0",
            "EXPERT_BUDGET": "0",
            "PREFETCH": "0",
            "PILOT": "0",
            "PILOT_REAL": "0",
            "CAP_RAISE": "0",
            "COLI_POLICY": "quality",
            "CTX": str(int(runtime.get("ctx", 4096))),
            "KV_SLOTS": "1",
            "COLI_KV_SLOTS": "1",
            "PROF": "1",
        }
    )
    applied_accelerator = apply_managed_accelerator_environment(
        environment,
        plan,
    )
    if applied_accelerator.get("COLI_CUDA") == "0":
        environment["COLI_MMAP"] = "0"
        environment["COLI_RAMMAP"] = "1" if rammap else "0"
    environment["COLI_RAM_PREFAULT"] = str(
        plan["prefault"]
        if environment.get("COLI_RAMMAP") == "1"
        else 0
    )
    for key, value in normalized_runtime_knobs(
        plan,
        knobs,
        node=node,
    ).items():
        environment[key] = str(value)
    return environment


def _cancellable_engine_type(
    engine_type,
    read_engine_turn,
    ready_marker,
    cancel_event,
):
    """Adapt benchmark engine startup without changing the shared API."""
    if cancel_event is None:
        return engine_type

    class CancellableEngine(engine_type):
        @classmethod
        def _wait_until_ready(cls, process, timeout):
            outcome = queue.Queue(maxsize=1)

            def read_ready():
                try:
                    read_engine_turn(
                        process.stdout,
                        ready_marker,
                        lambda _: None,
                    )
                except BaseException as error:
                    outcome.put(error)
                else:
                    outcome.put(None)

            reader = threading.Thread(
                target=read_ready,
                name="colibri-benchmark-ready",
                daemon=True,
            )
            reader.start()
            deadline = time.monotonic() + timeout
            try:
                while True:
                    _raise_if_cancelled(cancel_event)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(
                            "colibri engine did not become ready "
                            "within %.3g seconds" % timeout
                        )
                    try:
                        error = outcome.get(
                            timeout=min(0.2, remaining)
                        )
                    except queue.Empty:
                        continue
                    if error is not None:
                        raise error
                    break
            except BaseException:
                cls._terminate_process(process)
                reader.join(timeout=5)
                raise
            reader.join()

    return CancellableEngine


def _benchmark_generate(
    engine,
    prompt,
    on_text,
    cancel_event,
    client_cancelled_type,
):
    """Run one benchmark turn and interrupt before the first token."""
    if cancel_event is None:
        return engine.generate(
            prompt,
            32,
            0.0,
            1.0,
            on_text,
            cache_slot=0,
        )

    done = threading.Event()
    close_errors = []

    def cancel_watch():
        while not done.wait(0.1):
            if not cancel_event.is_set():
                continue
            try:
                # generate() waits on its response queue before the first
                # token, so its callback alone cannot cancel a long TTFT.
                # Closing the benchmark-only engine wakes that queue and
                # terminates the child.
                engine.close()
            except BaseException as exc:
                close_errors.append(exc)
            return

    watcher = threading.Thread(
        target=cancel_watch,
        name="colibri-benchmark-cancel",
        daemon=True,
    )
    watcher.start()
    try:
        try:
            result = engine.generate(
                prompt,
                32,
                0.0,
                1.0,
                on_text,
                cache_slot=0,
                cancelled=cancel_event.is_set,
            )
        except client_cancelled_type:
            watcher.join()
            if close_errors:
                raise _EngineCleanupError(
                    "benchmark cancellation could not close its "
                    "engine: %s" % close_errors[0]
                )
            _raise_if_cancelled(cancel_event)
            raise
        except BaseException as exc:
            if cancel_event.is_set():
                watcher.join()
                if close_errors:
                    raise _EngineCleanupError(
                        "benchmark cancellation could not close its "
                        "engine: %s" % close_errors[0]
                    ) from exc
                raise _OperationCancelled(
                    "benchmark cancelled by user at a safe checkpoint"
                ) from exc
            raise
        if cancel_event.is_set():
            watcher.join()
            if close_errors:
                raise _EngineCleanupError(
                    "benchmark cancellation could not close its "
                    "engine: %s" % close_errors[0]
                )
            _raise_if_cancelled(cancel_event)
        return result
    finally:
        done.set()
        watcher.join(timeout=1)


def _score_variant(
    engine_path,
    manifest,
    name,
    weights_dir,
    rammap,
    knobs,
    cancel_event=None,
    *,
    state_root,
    ensure_private_dir,
    assert_durable_state_dir,
    admit_runtime,
    fresh_user_binary,
    engine_cpu_list,
    benchmark_environment,
    cancellable_engine_type,
    benchmark_generate,
):
    # Import the stdlib-only engine protocol client lazily. This keeps
    # plan/status usable in minimal packaging probes while ensuring one
    # persistent process receives the warm-up and all measured turns.
    from openai_server import (
        READY,
        ClientCancelled,
        Engine as BaseEngine,
        read_engine_turn,
        render_chat,
    )

    Engine = cancellable_engine_type(
        BaseEngine,
        read_engine_turn,
        READY,
        cancel_event,
    )

    plan = manifest["plan"]
    runtime = plan.get("managed_runtime", {})
    fingerprint_dir = manifest["model_fingerprint"].split(
        ":",
        1,
    )[-1]
    safe_name = re.sub(
        r"[^a-z0-9_.-]",
        "-",
        name.lower(),
    )
    state_dir = os.path.join(
        state_root(),
        "benchmark-state",
        fingerprint_dir,
        safe_name,
    )
    target_mount = manifest["mounts"][0]
    command_prefix = []
    node = None
    if plan["topology"] == "per-node":
        node = int(target_mount["node"])
        command_prefix = [
            fresh_user_binary("numactl"),
            "--physcpubind=%s"
            % engine_cpu_list(plan, node=node),
            "--membind=%d" % node,
        ]
    environment = benchmark_environment(
        manifest,
        weights_dir,
        state_dir,
        rammap,
        node=node,
        knobs=knobs,
    )
    ensure_private_dir(state_dir)
    assert_durable_state_dir(
        state_dir,
        plan=plan,
    )
    admit_runtime(
        plan,
        target_mount,
        benchmark=not rammap,
    )
    prompt = render_chat(
        [
            {
                "role": "user",
                "content": BENCHMARK_PROMPT,
            }
        ],
        False,
        None,
        None,
        None,
    )
    log_path = os.path.join(
        state_dir,
        "benchmark.log",
    )
    log = open(
        log_path,
        "ab",
        buffering=0,
    )
    engine = None
    try:
        _raise_if_cancelled(cancel_event)
        engine = Engine(
            engine_path,
            plan["model"]["path"],
            cap=int(runtime.get("cache_cap", 8)),
            max_tokens=32,
            env=environment,
            kv_slots=1,
            command_prefix=command_prefix,
            stderr=log,
        )

        def run_once():
            _raise_if_cancelled(cancel_event)
            parts = []
            profile_seq = engine.profile_seq
            started = time.monotonic()
            stats = benchmark_generate(
                engine,
                prompt,
                parts.append,
                cancel_event,
                ClientCancelled,
            )
            _raise_if_cancelled(cancel_event)
            elapsed = time.monotonic() - started
            if stats.get("completion_tokens") != 32:
                raise RamdiskError(
                    "benchmark produced %s tokens instead of the "
                    "required 32"
                    % stats.get("completion_tokens")
                )
            if (
                engine.profile_seq <= profile_seq
                or not engine.profile
            ):
                raise RamdiskError(
                    "engine did not emit the required benchmark "
                    "telemetry"
                )
            profile = dict(engine.profile[-1])
            output = "".join(parts)
            return {
                "elapsed_seconds": elapsed,
                "tokens_per_second": stats.get(
                    "tokens_per_second"
                ),
                "forward_p50_ms": profile.get(
                    "forward_p50_ms"
                ),
                "forward_p99_ms": profile.get(
                    "forward_p99_ms"
                ),
                "rammap_experts": profile.get(
                    "rammap_experts"
                ),
                "rammap_bytes": profile.get("rammap_bytes"),
                "physical_ssd_bytes": profile.get(
                    "physical_ssd_bytes"
                ),
                "physical_ssd_valid": profile.get(
                    "physical_ssd_valid"
                ),
                "prefault_seconds": profile.get(
                    "prefault_seconds"
                ),
                "ttft_ms": profile.get("ttft_ms"),
                "rss_bytes": (
                    float(stats.get("rss_gb", 0.0)) * 1e9
                ),
                "output_sha256": hashlib.sha256(
                    output.encode("utf-8")
                ).hexdigest(),
            }

        run_once()  # warm-up on this exact process/LRU
        runs = [
            run_once()
            for _ in range(3)
        ]
    finally:
        cleanup_failures = []
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                cleanup_failures.append(
                    "engine: %s" % exc
                )
        try:
            log.close()
        except Exception as exc:
            cleanup_failures.append(
                "log: %s" % exc
            )
        if cleanup_failures:
            raise _EngineCleanupError(
                "benchmark variant cleanup failed: %s"
                % "; ".join(cleanup_failures)
            )
    outputs = {
        run["output_sha256"]
        for run in runs
    }
    if len(outputs) != 1:
        raise RamdiskError(
            "greedy benchmark output changed across deterministic "
            "runs"
        )
    observed_experts = {
        run.get("rammap_experts")
        for run in runs
    }
    observed_bytes = {
        run.get("rammap_bytes")
        for run in runs
    }
    expected_experts = (
        plan["staging"]["direct_mapped_expert_count"]
        if rammap
        else 0
    )
    expected_bytes = (
        plan["staging"]["direct_mapped_bytes"]
        if rammap
        else 0
    )
    if (
        observed_experts != {expected_experts}
        or observed_bytes != {expected_bytes}
    ):
        raise RamdiskError(
            "RAM-map telemetry mismatch: expected %d experts/%d "
            "bytes, observed %s/%s"
            % (
                expected_experts,
                expected_bytes,
                sorted(observed_experts, key=str),
                sorted(observed_bytes, key=str),
            )
        )
    if rammap and plan["mode"] == "full":
        if any(
            run.get("physical_ssd_valid") is not True
            for run in runs
        ):
            raise RamdiskError(
                "full direct RAM-map benchmark could not verify "
                "physical SSD reads"
            )
        if any(
            run.get("physical_ssd_bytes") != 0
            for run in runs
        ):
            raise RamdiskError(
                "full direct RAM-map benchmark performed physical "
                "SSD expert reads"
            )
    rates = [
        run["tokens_per_second"]
        for run in runs
        if run["tokens_per_second"] is not None
    ]
    forwards50 = [
        run["forward_p50_ms"]
        for run in runs
        if run["forward_p50_ms"] is not None
    ]
    forwards99 = [
        run["forward_p99_ms"]
        for run in runs
        if run["forward_p99_ms"] is not None
    ]
    ssd = [
        run["physical_ssd_bytes"]
        for run in runs
        if run["physical_ssd_bytes"] is not None
    ]
    ttfts = [
        run["ttft_ms"]
        for run in runs
        if run["ttft_ms"] is not None
    ]
    direct_total = plan["model"]["complete_experts"]
    direct_count = next(iter(observed_experts))
    return {
        "name": name,
        "status": "ok",
        "knobs": knobs,
        "runs": runs,
        "output_sha256": next(iter(outputs)),
        "persistent_engine": True,
        "log": log_path,
        "interactive": {
            "ttft_ms": _percentile(ttfts, 0.50),
            "p50_tokens_per_second": _percentile(
                rates,
                0.50,
            ),
            "p95_tokens_per_second": _percentile(
                rates,
                0.95,
            ),
            "forward_p50_ms": _percentile(
                forwards50,
                0.50,
            ),
            "forward_p99_ms": _percentile(
                forwards99,
                0.99,
            ),
            "ram_map_coverage": (
                float(direct_count) / direct_total
                if direct_total
                else 0.0
            ),
            "ssd_bytes_per_token": (
                sum(ssd) / len(ssd) / 32.0
                if ssd
                else None
            ),
        },
    }


def _aggregate_score(
    manifest,
    engine_path=None,
    knobs=None,
    cancel_event=None,
    *,
    state_root,
    ensure_private_dir,
    assert_durable_state_dir,
    admit_concurrent_runtimes,
    fresh_user_binary,
    engine_cpu_list,
    normalized_runtime_knobs,
    benchmark_environment,
    cancellable_engine_type,
    benchmark_generate,
):
    """Benchmark every node-local replica in one fixed environment."""
    if manifest["plan"]["topology"] != "per-node":
        return {
            "status": "not-run",
            "reason": (
                "aggregate score applies only to per-node topology"
            ),
            "per_node_tokens_per_second": [],
            "slowest_node_tokens_per_second": None,
            "total_tokens_per_second": None,
        }
    if manifest.get("state") == "running":
        return {
            "status": "not-run",
            "reason": (
                "stop managed engines before the fixed-environment "
                "aggregate benchmark"
            ),
            "per_node_tokens_per_second": [],
            "slowest_node_tokens_per_second": None,
            "total_tokens_per_second": None,
        }
    if not engine_path:
        return {
            "status": "error",
            "reason": (
                "aggregate benchmark engine path was not resolved"
            ),
            "per_node_tokens_per_second": [],
            "slowest_node_tokens_per_second": None,
            "total_tokens_per_second": None,
        }

    from openai_server import (
        READY,
        ClientCancelled,
        Engine as BaseEngine,
        read_engine_turn,
        render_chat,
    )

    Engine = cancellable_engine_type(
        BaseEngine,
        read_engine_turn,
        READY,
        cancel_event,
    )

    plan = manifest["plan"]
    runtime = plan.get("managed_runtime", {})
    fingerprint_dir = manifest["model_fingerprint"].split(
        ":",
        1,
    )[-1]
    prompt = render_chat(
        [
            {
                "role": "user",
                "content": BENCHMARK_PROMPT,
            }
        ],
        False,
        None,
        None,
        None,
    )
    launched = []
    launched_lock = threading.Lock()
    normalized_knobs = None
    try:
        _raise_if_cancelled(cancel_event)
        numactl = fresh_user_binary("numactl")
        mounts = list(manifest.get("mounts", []))
        if not mounts:
            raise RamdiskError(
                "per-node aggregate benchmark has no planned "
                "replicas"
            )
        for mount in mounts:
            node_knobs = normalized_runtime_knobs(
                plan,
                knobs or {"PIPE": 0},
                mount["node"],
            )
            if normalized_knobs is None:
                normalized_knobs = node_knobs
            elif node_knobs != normalized_knobs:
                raise RamdiskError(
                    "aggregate runtime knobs are not valid uniformly "
                    "across nodes"
                )
        admit_concurrent_runtimes(
            plan,
            mounts,
            benchmark=False,
        )

        def launch(mount):
            _raise_if_cancelled(cancel_event)
            node = int(mount["node"])
            state_dir = os.path.join(
                state_root(),
                "benchmark-state",
                fingerprint_dir,
                "aggregate-node-%d" % node,
            )
            ensure_private_dir(state_dir)
            assert_durable_state_dir(
                state_dir,
                plan=plan,
            )
            environment = benchmark_environment(
                manifest,
                mount["path"],
                state_dir,
                True,
                node=node,
                knobs=normalized_knobs,
            )
            log_path = os.path.join(
                state_dir,
                "benchmark.log",
            )
            log = open(
                log_path,
                "ab",
                buffering=0,
            )
            try:
                engine = Engine(
                    engine_path,
                    plan["model"]["path"],
                    cap=int(runtime.get("cache_cap", 8)),
                    max_tokens=32,
                    env=environment,
                    kv_slots=1,
                    command_prefix=[
                        numactl,
                        "--physcpubind=%s"
                        % engine_cpu_list(
                            plan,
                            node=node,
                        ),
                        "--membind=%d" % node,
                    ],
                    stderr=log,
                )
            except BaseException:
                log.close()
                raise
            entry = {
                "node": node,
                "engine": engine,
                "log_stream": log,
                "log": log_path,
            }
            with launched_lock:
                launched.append(entry)
            return entry

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(mounts)
        ) as executor:
            entries = list(
                executor.map(launch, mounts)
            )
        entries.sort(
            key=lambda entry: entry["node"]
        )

        def request(entry):
            _raise_if_cancelled(cancel_event)
            parts = []
            engine = entry["engine"]
            profile_seq = engine.profile_seq
            started = time.monotonic()
            stats = benchmark_generate(
                engine,
                prompt,
                parts.append,
                cancel_event,
                ClientCancelled,
            )
            _raise_if_cancelled(cancel_event)
            elapsed = time.monotonic() - started
            tokens = int(
                stats.get("completion_tokens", 0) or 0
            )
            if tokens != 32:
                raise RamdiskError(
                    "node %s produced %d tokens instead of 32"
                    % (entry["node"], tokens)
                )
            if (
                engine.profile_seq <= profile_seq
                or not engine.profile
            ):
                raise RamdiskError(
                    "node %s did not emit required aggregate "
                    "telemetry" % entry["node"]
                )
            profile = dict(engine.profile[-1])
            return {
                "node": entry["node"],
                "tokens_per_second": (
                    tokens / elapsed
                    if elapsed
                    else None
                ),
                "elapsed_seconds": elapsed,
                "rammap_experts": profile.get(
                    "rammap_experts"
                ),
                "rammap_bytes": profile.get("rammap_bytes"),
                "physical_ssd_bytes": profile.get(
                    "physical_ssd_bytes"
                ),
                "physical_ssd_valid": profile.get(
                    "physical_ssd_valid"
                ),
                "prefault_seconds": profile.get(
                    "prefault_seconds"
                ),
                "rss_bytes": (
                    float(stats.get("rss_gb", 0.0)) * 1e9
                ),
                "output_sha256": hashlib.sha256(
                    "".join(parts).encode("utf-8")
                ).hexdigest(),
            }

        def concurrent_round():
            _raise_if_cancelled(cancel_event)
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(entries)
            ) as executor:
                rows = list(
                    executor.map(request, entries)
                )
            wall = time.monotonic() - started
            return {
                "rows": rows,
                "wall_seconds": wall,
                "total_tokens_per_second": (
                    32.0 * len(entries) / wall
                    if wall
                    else None
                ),
            }

        concurrent_round()
        rounds = [
            concurrent_round()
            for _ in range(3)
        ]
        hashes = {
            row["output_sha256"]
            for round_result in rounds
            for row in round_result["rows"]
        }
        if len(hashes) != 1:
            raise RamdiskError(
                "deterministic aggregate outputs differed across "
                "replicas or runs"
            )
        expected_experts = plan["staging"][
            "direct_mapped_expert_count"
        ]
        expected_bytes = plan["staging"][
            "direct_mapped_bytes"
        ]
        for round_result in rounds:
            for row in round_result["rows"]:
                if (
                    row["rammap_experts"]
                    != expected_experts
                    or row["rammap_bytes"]
                    != expected_bytes
                ):
                    raise RamdiskError(
                        "node %s aggregate RAM-map telemetry does "
                        "not match the staging plan"
                        % row["node"]
                    )
                if plan["mode"] == "full":
                    if (
                        row.get("physical_ssd_valid")
                        is not True
                    ):
                        raise RamdiskError(
                            "node %s could not verify physical SSD "
                            "reads" % row["node"]
                        )
                    if row["physical_ssd_bytes"] != 0:
                        raise RamdiskError(
                            "node %s full aggregate run performed "
                            "physical SSD expert reads"
                            % row["node"]
                        )
        by_node = {}
        for round_result in rounds:
            for row in round_result["rows"]:
                by_node.setdefault(
                    row["node"],
                    [],
                ).append(
                    row["tokens_per_second"]
                )
        summary = [
            {
                "node": node,
                "p50_tokens_per_second": _percentile(
                    values,
                    0.50,
                ),
                "p95_tokens_per_second": _percentile(
                    values,
                    0.95,
                ),
            }
            for node, values in sorted(by_node.items())
        ]
        p50_values = [
            row["p50_tokens_per_second"]
            for row in summary
        ]
        total_values = [
            item["total_tokens_per_second"]
            for item in rounds
        ]
        return {
            "status": "ok",
            "warmups": 1,
            "measured_rounds": 3,
            "persistent_engines": True,
            "fixed_environment": {
                "TEMP": 0,
                "DRAFT": 0,
                "KVSAVE": 0,
                "AUTOPIN": 0,
                "PROF": 1,
            },
            "runtime_knobs": normalized_knobs,
            "per_node_tokens_per_second": summary,
            "slowest_node_tokens_per_second": min(
                p50_values
            ),
            "total_tokens_per_second": _percentile(
                total_values,
                0.50,
            ),
            "output_sha256": next(iter(hashes)),
            "rounds": rounds,
            "logs": [
                entry["log"]
                for entry in entries
            ],
        }
    except _OperationCancelled:
        raise
    except (
        RamdiskError,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        if (
            cancel_event is not None
            and cancel_event.is_set()
        ):
            raise
        return {
            "status": "error",
            "error": str(exc),
            "per_node_tokens_per_second": [],
            "slowest_node_tokens_per_second": None,
            "total_tokens_per_second": None,
        }
    finally:
        cleanup_failures = []
        for entry in launched:
            try:
                entry["engine"].close()
            except Exception as exc:
                cleanup_failures.append(
                    "node %s engine: %s"
                    % (entry.get("node"), exc)
                )
            try:
                entry["log_stream"].close()
            except Exception as exc:
                cleanup_failures.append(
                    "node %s log: %s"
                    % (entry.get("node"), exc)
                )
        if cleanup_failures:
            raise _EngineCleanupError(
                "aggregate benchmark cleanup failed: %s"
                % "; ".join(cleanup_failures)
            )


def _system_score(
    manifest,
    variants,
    swap_before,
    swap_after,
    aggregate=None,
    *,
    meminfo,
    statvfs,
):
    """Build host and mount metrics without assuming ``os.statvfs``."""
    memory = meminfo()
    prefaults = [
        run["prefault_seconds"]
        for variant in variants
        if variant.get("status") == "ok"
        for run in variant.get("runs", [])
        if run.get("prefault_seconds") is not None
    ]
    rss_values = [
        run["rss_bytes"]
        for variant in variants
        if variant.get("status") == "ok"
        for run in variant.get("runs", [])
        if run.get("rss_bytes") is not None
    ]
    aggregate_rows = []
    aggregate_rss_totals = []
    aggregate_prefaults = []
    if aggregate and aggregate.get("status") == "ok":
        aggregate_rows = [
            row
            for round_result in aggregate.get("rounds", [])
            for row in round_result.get("rows", [])
        ]
        aggregate_prefaults = [
            row["prefault_seconds"]
            for row in aggregate_rows
            if row.get("prefault_seconds") is not None
        ]
        rss_values.extend(
            row["rss_bytes"]
            for row in aggregate_rows
            if row.get("rss_bytes") is not None
        )
        aggregate_rss_totals = [
            sum(
                row["rss_bytes"]
                for row in round_result.get("rows", [])
                if row.get("rss_bytes") is not None
            )
            for round_result in aggregate.get("rounds", [])
        ]
    created = manifest.get("created_at")
    ready = manifest.get("ready_at")
    stage_seconds = None
    try:
        if created and ready:
            stage_seconds = (
                datetime.datetime.fromisoformat(ready)
                - datetime.datetime.fromisoformat(created)
            ).total_seconds()
    except ValueError:
        pass
    shmem = memory.get("Shmem", 0)
    huge = memory.get("ShmemPmdMapped", 0)
    placement = []
    mount_shmem = 0
    for record in manifest.get("mounts", []):
        counts = record.get("numa_allocation", {})
        total = sum(
            int(value)
            for value in counts.values()
        )
        node = record.get("node")
        local = (
            int(counts.get(str(node), 0))
            if node is not None
            else None
        )
        allocated_bytes = None
        if statvfs is not None:
            try:
                filesystem = statvfs(record["path"])
                allocated_bytes = (
                    filesystem.f_blocks
                    - filesystem.f_bfree
                ) * filesystem.f_frsize
                mount_shmem += allocated_bytes
            except OSError:
                pass
        placement.append(
            {
                "path": record["path"],
                "target_node": node,
                "sampled_pages": total,
                "local_pages": local,
                "remote_pages": (
                    total - local
                    if local is not None
                    else None
                ),
                "by_node": counts,
                "allocated_bytes": allocated_bytes,
            }
        )
    return {
        "stage_seconds": stage_seconds,
        "prefault_seconds": (
            max(aggregate_prefaults)
            if aggregate_prefaults
            else _percentile(prefaults, 0.50)
        ),
        "rss_bytes": (
            max(aggregate_rss_totals)
            if aggregate_rss_totals
            else max(rss_values)
            if rss_values
            else None
        ),
        "per_process_peak_rss_bytes": (
            max(rss_values)
            if rss_values
            else None
        ),
        "aggregate_rss_bytes": (
            max(aggregate_rss_totals)
            if aggregate_rss_totals
            else None
        ),
        "shmem_bytes": mount_shmem,
        "host_shmem_bytes": shmem,
        "swap_before_bytes": swap_before,
        "swap_after_bytes": swap_after,
        "swap_delta_bytes": max(
            0,
            swap_after - swap_before,
        ),
        "huge_page_coverage": (
            float(huge) / shmem
            if shmem
            else 0.0
        ),
        "huge_page_coverage_scope": (
            "host-global ShmemPmdMapped/Shmem; per-mount THP "
            "accounting is not exposed by tmpfs"
        ),
        "numa_allocation": [
            record.get("numa_allocation", {})
            for record in manifest.get("mounts", [])
        ],
        "numa_page_placement": placement,
        "numa_traffic_note": (
            "dependency-free v1 reports sampled page placement; "
            "PMU traffic counters require external perf privileges"
        ),
    }


def run_benchmark(
    args,
    cli_path,
    engine_path=None,
    cancel_event=None,
    *,
    load_manifest,
    assert_effective_masks_unchanged,
    assert_ready_mounts,
    resolve_engine_path,
    node_core_count,
    score_variant,
    discover_hardware,
    aggregate_score,
    system_score,
    filesystem_for_path,
    source_build_identity,
    read_json,
    benchmarks_path,
    atomic_json,
    save_manifest,
    argv,
):
    """Run a benchmark while the facade owns locking and dependencies."""
    del args
    manifest = load_manifest(required=True)
    _raise_if_cancelled(cancel_event)
    if manifest.get("state") not in (
        "ready",
        "running",
        "stopped",
    ):
        raise RamdiskError(
            "benchmark requires a ready RAM-disk manifest"
        )
    assert_effective_masks_unchanged(manifest["plan"])
    assert_ready_mounts(manifest)
    engine_path = resolve_engine_path(
        cli_path,
        engine_path,
    )
    model = manifest["plan"]["model"]["path"]
    mount = manifest["mounts"][0]["path"]
    running = manifest.get("state") == "running"
    if running:
        raise RamdiskError(
            "stop managed engines before benchmarking so every "
            "score uses the fixed environment"
        )
    specs = [
        (
            "ssd_baseline",
            model,
            False,
            {
                "PIPE": 0,
                "DIRECT": 0,
                "URING": 0,
            },
        ),
        (
            "tmpfs_pread_slabs",
            mount,
            False,
            {
                "PIPE": 0,
                "DIRECT": 0,
                "URING": 0,
            },
        ),
    ]
    if manifest["plan"]["mode"] == "full":
        benchmark_node = (
            manifest["mounts"][0].get("node")
            if manifest["plan"]["topology"] == "per-node"
            else None
        )
        cores = node_core_count(
            manifest["plan"],
            benchmark_node,
        )
        specs.extend(
            [
                (
                    "full_direct_half_threads",
                    mount,
                    True,
                    {
                        "PIPE": 0,
                        "OMP_NUM_THREADS": max(
                            1,
                            cores // 2,
                        ),
                        "OMP_PROC_BIND": "close",
                    },
                ),
                (
                    "full_direct_pipe0",
                    mount,
                    True,
                    {
                        "PIPE": 0,
                        "OMP_NUM_THREADS": cores,
                        "OMP_PROC_BIND": "close",
                    },
                ),
                (
                    "full_direct_pipe1",
                    mount,
                    True,
                    {
                        "PIPE": 1,
                        "OMP_NUM_THREADS": cores,
                        "OMP_PROC_BIND": "spread",
                    },
                ),
            ]
        )
        skipped = {
            "name": "partial_direct_ssd_fallback",
            "status": "not-applicable",
            "reason": "manifest is full mode",
        }
    else:
        specs.extend(
            [
                (
                    "partial_direct_buffered",
                    mount,
                    True,
                    {
                        "PIPE": 1,
                        "DIRECT": 0,
                        "PIPE_WORKERS": 4,
                        "URING": 0,
                    },
                ),
                (
                    "partial_direct_ssd",
                    mount,
                    True,
                    {
                        "PIPE": 1,
                        "DIRECT": 1,
                        "PIPE_WORKERS": 8,
                        "URING": 0,
                    },
                ),
                (
                    "partial_direct_uring",
                    mount,
                    True,
                    {
                        "PIPE": 1,
                        "DIRECT": 1,
                        "PIPE_WORKERS": 8,
                        "URING": 1,
                    },
                ),
            ]
        )
        skipped = {
            "name": "full_direct",
            "status": "not-applicable",
            "reason": "manifest is partial mode",
        }
    variants = []
    swap_before = discover_hardware()["swap"]["used_bytes"]
    for name, weights, rammap, knobs in specs:
        _raise_if_cancelled(cancel_event)
        try:
            if cancel_event is None:
                score = score_variant(
                    engine_path,
                    manifest,
                    name,
                    weights,
                    rammap,
                    knobs,
                )
            else:
                score = score_variant(
                    engine_path,
                    manifest,
                    name,
                    weights,
                    rammap,
                    knobs,
                    cancel_event=cancel_event,
                )
            variants.append(score)
        except (
            _OperationCancelled,
            _EngineCleanupError,
        ):
            raise
        except (
            RamdiskError,
            RuntimeError,
            OSError,
            subprocess.SubprocessError,
        ) as exc:
            if (
                cancel_event is not None
                and cancel_event.is_set()
            ):
                raise
            variants.append(
                {
                    "name": name,
                    "status": "error",
                    "error": str(exc),
                    "knobs": knobs,
                }
            )
    variants.append(skipped)
    baseline = next(
        (
            variant
            for variant in variants
            if (
                variant.get("name") == "ssd_baseline"
                and variant.get("status") == "ok"
            )
        ),
        None,
    )
    token_mismatches = []
    if baseline:
        for variant in variants:
            if variant.get("status") != "ok":
                continue
            equivalent = (
                variant.get("output_sha256")
                == baseline.get("output_sha256")
            )
            variant["greedy_output_matches_ssd"] = equivalent
            if not equivalent:
                token_mismatches.append(variant["name"])
                variant["status"] = "error"
                variant["error"] = (
                    "greedy output differs from the SSD baseline"
                )
    successful = [
        variant
        for variant in variants
        if (
            variant.get("status") == "ok"
            and (
                variant.get("greedy_output_matches_ssd")
                is True
            )
            and variant["name"].startswith(
                "%s_direct" % manifest["plan"]["mode"]
            )
        )
    ]
    best = (
        max(
            successful,
            key=lambda variant: (
                variant["interactive"][
                    "p50_tokens_per_second"
                ]
                or 0
            ),
        )
        if successful
        else None
    )
    previous_best = manifest.get(
        "best_runtime",
        {},
    ).get(
        manifest["plan"]["topology"],
        {},
    )
    aggregate_knobs = dict(
        best.get("knobs")
        if best
        else previous_best.get(
            "knobs",
            {"PIPE": 0},
        )
    )
    # The interactive sweep targets the first replica. Thread counts are
    # node-relative, so aggregate engines derive their local count rather
    # than reusing node 0's absolute value on asymmetric machines.
    aggregate_knobs.pop(
        "OMP_NUM_THREADS",
        None,
    )
    if cancel_event is None:
        aggregate = aggregate_score(
            manifest,
            engine_path=engine_path,
            knobs=aggregate_knobs,
        )
    else:
        aggregate = aggregate_score(
            manifest,
            engine_path=engine_path,
            knobs=aggregate_knobs,
            cancel_event=cancel_event,
        )
    if baseline and aggregate.get("status") == "ok":
        aggregate_matches = (
            aggregate.get("output_sha256")
            == baseline.get("output_sha256")
        )
        aggregate[
            "greedy_output_matches_ssd"
        ] = aggregate_matches
        if not aggregate_matches:
            token_mismatches.append(
                "aggregate_per_node"
            )
            aggregate["status"] = "error"
            aggregate["error"] = (
                "aggregate greedy output differs from the SSD "
                "baseline"
            )
    tmpfs_ok = any(
        (
            variant.get("name") == "tmpfs_pread_slabs"
            and variant.get("status") == "ok"
            and (
                variant.get("greedy_output_matches_ssd")
                is True
            )
        )
        for variant in variants
    )
    aggregate_ok = (
        manifest["plan"]["topology"] != "per-node"
        or (
            aggregate.get("status") == "ok"
            and (
                aggregate.get("greedy_output_matches_ssd")
                is True
            )
        )
    )
    required_paths_ok = (
        bool(baseline)
        and tmpfs_ok
        and bool(successful)
        and aggregate_ok
    )
    full_zero_verified = None
    if manifest["plan"]["mode"] == "full":
        full_zero_verified = bool(successful) and all(
            (
                len(variant.get("runs", [])) == 3
                and all(
                    (
                        run.get("physical_ssd_valid")
                        is True
                        and run.get("physical_ssd_bytes") == 0
                    )
                    for run in variant["runs"]
                )
            )
            for variant in successful
        )
    swap_after = discover_hardware()["swap"]["used_bytes"]
    result = {
        "schema": BENCHMARK_SCHEMA,
        "version": MANIFEST_VERSION,
        "created_at": _utc_now(),
        "model_fingerprint": manifest["model_fingerprint"],
        "topology": manifest["plan"]["topology"],
        "mode": manifest["plan"]["mode"],
        "prompt": BENCHMARK_PROMPT,
        "source": source_build_identity(),
        "command": list(argv),
        "hardware": {
            "kernel_release": manifest["plan"][
                "hardware"
            ].get("kernel_release"),
            "physical_cores": manifest["plan"][
                "hardware"
            ].get("physical_cores"),
            "nodes": manifest["plan"]["hardware"].get(
                "nodes"
            ),
            "memory": manifest["plan"]["hardware"].get(
                "memory"
            ),
        },
        "storage": {
            "canonical_model_filesystem": (
                filesystem_for_path(model)
            ),
            "mounts": [
                {
                    "path": record["path"],
                    "node": record.get("node"),
                    "filesystem": record.get(
                        "identity",
                        {},
                    ).get("filesystem"),
                    "options": record.get(
                        "identity",
                        {},
                    ).get(
                        "all_options",
                        [],
                    ),
                }
                for record in manifest.get("mounts", [])
            ],
        },
        "warmups": 1,
        "measured_runs": 3,
        "tokens_per_run": 32,
        "variants": variants,
        "aggregate": aggregate,
        "system": system_score(
            manifest,
            variants,
            swap_before,
            swap_after,
            aggregate=aggregate,
        ),
        "acceptance": {
            "all_required_paths_succeeded": required_paths_ok,
            "greedy_outputs_identical": (
                required_paths_ok
                and not token_mismatches
            ),
            "output_mismatches": token_mismatches,
            "full_zero_physical_ssd_reads_verified": (
                full_zero_verified
            ),
            "no_swap_growth": swap_after <= swap_before,
            "staging_within_budget": (
                manifest["plan"]["staging"]["staged_bytes"]
                <= manifest["plan"]["capacity_bytes"]
            ),
        },
        "best_runtime_knobs": (
            best.get("knobs")
            if best
            else previous_best.get("knobs")
        ),
        "best_variant": (
            best.get("name")
            if best
            else previous_best.get("variant")
        ),
    }
    _raise_if_cancelled(cancel_event)
    history = read_json(
        benchmarks_path()
    ) or {
        "version": 1,
        "results": [],
    }
    if (
        not isinstance(history, dict)
        or history.get("version") != 1
        or not isinstance(history.get("results"), list)
    ):
        raise RamdiskError(
            "benchmark history is malformed or unsupported"
        )
    history.setdefault(
        "results",
        [],
    ).append(result)
    atomic_json(
        benchmarks_path(),
        history,
    )
    manifest.setdefault(
        "benchmark_results",
        [],
    ).append(result)
    if best:
        manifest.setdefault(
            "best_runtime",
            {},
        )[manifest["plan"]["topology"]] = {
            "variant": result["best_variant"],
            "knobs": result["best_runtime_knobs"],
        }
    save_manifest(manifest)
    return result
