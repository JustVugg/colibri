"""RAM-disk preparation, engine, teardown, and status orchestration."""

from __future__ import print_function

import copy
import errno
import math
import os
import secrets
import socket
import subprocess
import sys
import threading
import time

from .common import (
    MANIFEST_VERSION,
    MIB,
    STATUS_SCHEMA,
    RamdiskError,
    _OperationCancelled,
    _raise_if_cancelled,
    _utc_now,
)


def _managed_ports_for_plan(plan, base_port=8000):
    return [
        int(base_port)
        + (0 if mount.get("node") is None else int(mount["node"]))
        for mount in plan["mounts"]
    ]


def _persisted_base_port(manifest):
    """Recover the last base port, including manifests predating that field."""
    explicit = manifest.get("base_port")
    if (
        isinstance(explicit, int)
        and not isinstance(explicit, bool)
        and 1 <= explicit <= 65535
    ):
        return explicit

    candidates = []
    for process in manifest.get("processes", []):
        port = process.get("port")
        node = process.get("node")
        if isinstance(port, int) and not isinstance(port, bool):
            candidates.append(
                port - (0 if node is None else int(node))
            )
    if not candidates:
        for mount, port in zip(
            manifest.get("mounts", []),
            manifest.get("ports", []),
        ):
            if isinstance(port, int) and not isinstance(port, bool):
                node = mount.get("node")
                candidates.append(
                    port - (0 if node is None else int(node))
                )
    if (
        candidates
        and len(set(candidates)) == 1
        and 1 <= candidates[0] <= 65535
    ):
        return candidates[0]
    return 8000


def _managed_path(path, mount_root):
    normalized = os.path.normpath(os.path.abspath(path))
    root = os.path.normpath(os.path.abspath(mount_root))
    if root in ("/", "", os.path.expanduser("~")):
        return False
    return (
        normalized == root
        or os.path.commonpath([normalized, root]) == root
    )


def _assert_ready_mounts(
    manifest,
    *,
    source_still_matches,
    validate_mount,
    validate_namespace,
):
    plan = manifest["plan"]
    source_still_matches(plan)
    for record in manifest.get("mounts", []):
        actual = validate_mount(record, plan)
        expected = record.get("identity", {})
        if expected and (
            actual["mount_id"] != expected.get("mount_id")
            or actual["device"] != expected.get("device")
        ):
            raise RamdiskError(
                "mount identity changed at %s" % record["path"]
            )
        validate_namespace(plan, record)


def prepare(
    args,
    progress=None,
    display_plan=True,
    expected_plan_token=None,
    cancel_event=None,
    *,
    load_manifest,
    build_plan,
    managed_ports_for_plan,
    plan_confirmation_token,
    render_plan,
    confirm,
    save_manifest,
    mount_at,
    mount_tmpfs,
    umount_path,
    validate_mount,
    populate_mount,
    validate_namespace,
    source_still_matches,
    durable_unlink,
    manifest_path,
):
    if load_manifest(required=False) is not None:
        raise RamdiskError(
            "a RAM-disk manifest already exists; stop/destroy it before "
            "preparing another"
        )
    plan = build_plan(args)
    try:
        base_port = int(getattr(args, "base_port", 8000))
    except (TypeError, ValueError):
        raise RamdiskError("managed base port must be an integer")
    planned_ports = managed_ports_for_plan(plan, base_port)
    if (
        not 1 <= base_port <= 65535
        or len(set(planned_ports)) != len(planned_ports)
        or any(
            port < 1 or port > 65535
            for port in planned_ports
        )
    ):
        raise RamdiskError(
            "managed base port produces invalid or duplicate replica ports"
        )
    _raise_if_cancelled(cancel_event)
    if (
        expected_plan_token is not None
        and plan_confirmation_token(plan) != expected_plan_token
    ):
        raise RamdiskError(
            "RAM-disk plan changed since review; inspect the updated plan "
            "and confirm again"
        )
    if plan["blockers"]:
        raise RamdiskError(
            "preparation blocked: " + "; ".join(plan["blockers"])
        )
    if display_plan:
        render_plan(plan)
    confirm(
        "Mount tmpfs and stage the reviewed bytes?",
        bool(getattr(args, "yes", False)),
    )
    if progress is None:
        progress_lock = threading.Lock()

        def progress(name, size, elapsed):
            with progress_lock:
                rate = size / elapsed / MIB if elapsed > 0 else 0.0
                print(
                    "  staged %-36s %8.1f MiB/s" % (name, rate),
                    flush=True,
                )

    manifest = {
        "version": MANIFEST_VERSION,
        "deployment_id": secrets.token_hex(16),
        "base_port": base_port,
        "state": "preparing",
        "created_at": _utc_now(),
        "plan": plan,
        "model_fingerprint": plan["model"]["fingerprint"],
        "mounts": [],
        "processes": [],
        "ports": [],
        "benchmark_results": [],
        "initial_swap_used_bytes": plan["hardware"]["swap"]["used_bytes"],
    }
    save_manifest(manifest)
    mounted = []
    try:
        for mount in plan["mounts"]:
            _raise_if_cancelled(cancel_event)
            if mount_at(mount["path"]):
                raise RamdiskError(
                    "refusing already-mounted path: %s"
                    % mount["path"]
                )
            mount_tmpfs(plan, mount)
            mounted_actual = mount_at(mount["path"])
            if not mounted_actual:
                # A successful helper without an observable identity must be
                # rolled back immediately, before general cleanup could mistake
                # a later foreign mount for the one this operation created.
                try:
                    umount_path(
                        mount["path"],
                        plan["hardware"],
                    )
                except Exception as cleanup_exc:
                    raise RamdiskError(
                        "mounted %s but could not identify or roll it back: %s"
                        % (mount["path"], cleanup_exc)
                    )
                raise RamdiskError(
                    "mounted %s but could not read its mount identity; "
                    "rolled it back" % mount["path"]
                )
            mounted.append((mount, mounted_actual))
            record = dict(mount)
            record["identity"] = mounted_actual
            record["validated"] = False
            manifest["mounts"].append(record)
            save_manifest(manifest)
            identity = validate_mount(mount, plan)
            record["identity"] = identity
            record["validated"] = True
            save_manifest(manifest)

        seed = None
        for index, mount in enumerate(plan["mounts"]):
            _raise_if_cancelled(cancel_event)
            source = (
                seed
                if index and plan["topology"] == "per-node"
                else None
            )
            populate_mount(
                plan,
                mount,
                source_root=source,
                progress=progress,
                cancel_event=cancel_event,
            )
            if seed is None:
                seed = mount["path"]
            manifest["mounts"][index]["numa_allocation"] = (
                validate_namespace(plan, mount)
            )
            save_manifest(manifest)
        _raise_if_cancelled(cancel_event)
        source_still_matches(plan)
        manifest["state"] = "ready"
        manifest["ready_at"] = _utc_now()
        save_manifest(manifest)
        return manifest
    except BaseException as exc:
        manifest["state"] = "error"
        manifest["error"] = str(exc)
        cleanup_errors = []
        try:
            save_manifest(manifest)
        except Exception as save_exc:
            cleanup_errors.append(
                "could not persist preparation error: %s" % save_exc
            )
        for mount, expected in reversed(mounted):
            try:
                actual = mount_at(mount["path"])
                if actual:
                    if (
                        expected
                        and actual["filesystem"] == "tmpfs"
                        and actual["source"] == "tmpfs"
                        and actual["mount_id"]
                        == expected.get("mount_id")
                        and actual["device"]
                        == expected.get("device")
                    ):
                        umount_path(
                            mount["path"],
                            plan["hardware"],
                        )
                    else:
                        cleanup_errors.append(
                            "refusing changed mount during preparation "
                            "rollback: %s" % mount["path"]
                        )
            except Exception as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
        if isinstance(exc, _OperationCancelled) and not cleanup_errors:
            try:
                durable_unlink(manifest_path())
            except Exception as cleanup_exc:
                cleanup_errors.append(
                    "could not remove cancelled preparation manifest: %s"
                    % cleanup_exc
                )
            if not cleanup_errors:
                raise
        if cleanup_errors:
            manifest["cleanup_errors"] = cleanup_errors
            try:
                save_manifest(manifest)
            except Exception as save_exc:
                cleanup_errors.append(
                    "could not persist cleanup result: %s" % save_exc
                )
            raise RamdiskError(
                "%s; preparation rollback/reporting errors: %s"
                % (exc, "; ".join(cleanup_errors))
            ) from exc
        raise


def start(
    args,
    cli_path=None,
    engine_path=None,
    cancel_event=None,
    *,
    default_cli_path,
    load_manifest,
    assert_effective_masks_unchanged,
    assert_ready_mounts,
    process_matches,
    save_manifest,
    merge_usage,
    persisted_base_port,
    fresh_user_binary,
    admit_concurrent_runtimes,
    state_root,
    ensure_private_dir,
    assert_durable_state_dir,
    recover_delta,
    usage_read,
    usage_write,
    managed_numa_enabled,
    memory_node_list,
    engine_cpu_list,
    node_core_count,
    normalized_runtime_knobs,
    apply_managed_accelerator_environment=None,
    proc_identity,
    wait_managed_ready,
    track_managed_child,
    terminate_verified_group,
    terminate_direct_child,
    forget_managed_child,
):
    if apply_managed_accelerator_environment is None:
        from .accelerator import _apply_managed_accelerator_environment

        apply_managed_accelerator_environment = (
            _apply_managed_accelerator_environment
        )
    manifest = load_manifest(required=True)
    _raise_if_cancelled(cancel_event)
    if manifest.get("state") not in ("ready", "stopped"):
        raise RamdiskError(
            "manifest state is %s, not ready"
            % manifest.get("state")
        )
    assert_effective_masks_unchanged(manifest["plan"])
    assert_ready_mounts(manifest)
    plan = manifest["plan"]
    cli_path = cli_path or default_cli_path
    model = plan["model"]["path"]
    canonical_usage = os.path.join(model, ".coli_usage")
    foreign = []
    recovered = False
    for record in manifest.get("processes", []):
        _raise_if_cancelled(cancel_event)
        if record.get("stopped_at"):
            if not record.get("usage_merged_at"):
                record.setdefault(
                    "usage_merge_id",
                    secrets.token_hex(16),
                )
                save_manifest(manifest)
                merge_usage(
                    record,
                    canonical_usage,
                    plan=plan,
                )
                record["usage_merged_at"] = _utc_now()
                record.pop("usage_merge_error", None)
                recovered = True
                save_manifest(manifest)
            continue
        matches, reason, _ = process_matches(record)
        if matches:
            raise RamdiskError(
                "managed engine is already running on port %s"
                % record.get("port")
            )
        if reason == "not-running":
            # Crash recovery must merge post-baseline counts before the stable
            # state directory is seeded for a replacement process.
            record.setdefault(
                "usage_merge_id",
                secrets.token_hex(16),
            )
            save_manifest(manifest)
            merge_usage(
                record,
                canonical_usage,
                plan=plan,
            )
            record["usage_merged_at"] = _utc_now()
            record["stopped_at"] = _utc_now()
            record["crash_recovered_at"] = _utc_now()
            record.pop("usage_merge_error", None)
            recovered = True
            save_manifest(manifest)
        else:
            foreign.append(
                "PID %s (%s)"
                % (record.get("pid"), reason)
            )
    if foreign:
        raise RamdiskError(
            "refusing stale foreign process records: "
            + ", ".join(foreign)
        )
    if recovered:
        save_manifest(manifest)

    requested_base_port = getattr(args, "base_port", None)
    if requested_base_port is None:
        base_port = persisted_base_port(manifest)
    else:
        if isinstance(requested_base_port, bool):
            raise RamdiskError(
                "managed base port must be an integer"
            )
        try:
            base_port = int(requested_base_port)
        except (TypeError, ValueError):
            raise RamdiskError(
                "managed base port must be an integer"
            )
    ports = [
        base_port
        + (
            0
            if record.get("node") is None
            else int(record["node"])
        )
        for record in manifest["mounts"]
    ]
    if (
        len(set(ports)) != len(ports)
        or any(port < 1 or port > 65535 for port in ports)
    ):
        raise RamdiskError(
            "managed ports are invalid or duplicated"
        )
    for port in ports:
        probe = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RamdiskError(
                "port %d is unavailable: %s" % (port, exc)
            )
        finally:
            probe.close()

    previous_state = manifest["state"]
    previous_processes = copy.deepcopy(
        manifest.get("processes", [])
    )
    previous_ports = list(manifest.get("ports", []))
    previous_base_port = persisted_base_port(manifest)
    manifest["base_port"] = base_port
    nonce = secrets.token_hex(24)
    managed_numactl = (
        fresh_user_binary("numactl")
        if plan["topology"] == "per-node"
        else None
    )
    records = []
    runtime = plan.get("managed_runtime", {})
    saved_runtime_knobs = dict(
        manifest.get("best_runtime", {})
        .get(plan["topology"], {})
        .get("knobs")
        or {}
    )
    # Thread counts are node-relative and the managed topology contract always
    # uses every physical core. Retain other measured knobs for this topology.
    saved_runtime_knobs.pop("OMP_NUM_THREADS", None)
    managed_ctx = int(runtime.get("ctx", 4096))
    managed_slots = int(runtime.get("kv_slots", 1))
    managed_cap = int(runtime.get("cache_cap", 8))
    fingerprint_dir = manifest["model_fingerprint"].split(
        ":",
        1,
    )[-1]
    try:
        startup_timeout = float(
            os.environ.get(
                "COLI_RAMDISK_START_TIMEOUT",
                "7200",
            )
        )
    except ValueError:
        raise RamdiskError(
            "COLI_RAMDISK_START_TIMEOUT must be numeric"
        )
    if (
        not math.isfinite(startup_timeout)
        or not 1 <= startup_timeout <= 86400
    ):
        raise RamdiskError(
            "COLI_RAMDISK_START_TIMEOUT must be between 1 and 86400 "
            "seconds"
        )
    # Admit the complete replica set from one shared cgroup snapshot before
    # spawning the first child.
    admit_concurrent_runtimes(
        plan,
        manifest["mounts"],
        benchmark=False,
    )
    spawned = []
    launch_contexts = []
    manifest["processes"] = []
    manifest["ports"] = []
    manifest["state"] = "starting"
    save_manifest(manifest)

    try:
        for index, mount in enumerate(manifest["mounts"]):
            _raise_if_cancelled(cancel_event)
            node = mount.get("node")
            label = (
                "interleaved"
                if node is None
                else "node-%d" % node
            )
            state_dir = os.path.join(
                state_root(),
                "engines",
                fingerprint_dir,
                label,
            )
            ensure_private_dir(state_dir)
            assert_durable_state_dir(state_dir, plan=plan)
            recover_delta(
                state_dir,
                canonical_usage,
                plan=plan,
            )
            baseline = usage_read(canonical_usage)
            usage_write(
                os.path.join(state_dir, ".coli_usage"),
                baseline,
            )
            context = {
                "state_dir": state_dir,
                "usage_baseline": baseline,
                "record": None,
            }
            launch_contexts.append(context)
            port = base_port + (
                0 if node is None else int(node)
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "COLI_WEIGHTS_DIR": mount["path"],
                    "COLI_STATE_DIR": state_dir,
                    "COLI_MANAGED_NONCE": nonce,
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
                    "CTX": str(managed_ctx),
                    "KV_SLOTS": str(managed_slots),
                    "COLI_KV_SLOTS": str(managed_slots),
                    # Managed engines deliberately keep durable node-specific
                    # KV state; shell overrides may not disable that promise.
                    "KVSAVE": "1",
                    "CAP_RAISE": "0",
                    "AUTOPIN": "0",
                    "PROF": "1",
                }
            )
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
            ):
                environment.pop(inherited, None)
            applied_runtime_knobs = normalized_runtime_knobs(
                plan,
                saved_runtime_knobs,
                node=node,
            )
            for key, value in applied_runtime_knobs.items():
                environment[key] = str(value)
            applied_accelerator = apply_managed_accelerator_environment(
                environment,
                plan,
            )
            environment["COLI_RAM_PREFAULT"] = str(
                plan["prefault"]
                if applied_accelerator.get("COLI_RAMMAP") == "1"
                else 0
            )
            command = [
                cli_path,
                "serve",
                "--model",
                model,
                "--port",
                str(port),
                "--cap",
                str(managed_cap),
                "--ctx",
                str(managed_ctx),
                "--kv-slots",
                str(managed_slots),
            ]
            if not os.access(cli_path, os.X_OK):
                command.insert(0, sys.executable)
            if node is not None:
                command = [
                    managed_numactl,
                    "--physcpubind=%s"
                    % engine_cpu_list(plan, node=node),
                    "--membind=%d" % node,
                ] + command
            log_path = os.path.join(
                state_dir,
                "engine.log",
            )
            log = open(log_path, "ab", buffering=0)
            try:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            finally:
                log.close()
            spawned.append(process)
            identity = None
            identity_deadline = time.monotonic() + 1.0
            while (
                time.monotonic() < identity_deadline
                and process.poll() is None
            ):
                _raise_if_cancelled(cancel_event)
                identity = proc_identity(process.pid)
                if (
                    identity
                    and identity.get("pgid") == process.pid
                    and identity.get("nonce") == nonce
                ):
                    break
                time.sleep(0.01)
            if (
                process.poll() is not None
                or not identity
                or identity.get("pgid") != process.pid
                or identity.get("nonce") != nonce
            ):
                raise RamdiskError(
                    "managed engine exited during launch; see %s"
                    % log_path
                )
            record = {
                "pid": process.pid,
                "pgid": identity["pgid"],
                "uid": identity["uid"],
                "starttime": identity["starttime"],
                "nonce": nonce,
                "port": port,
                "node": node,
                "command": command,
                "state_dir": state_dir,
                "weights_dir": mount["path"],
                "usage_baseline": baseline,
                "started_at": _utc_now(),
                "log": log_path,
                "runtime_knobs": applied_runtime_knobs,
                "accelerator_environment": applied_accelerator,
            }
            context["record"] = record
            records.append(record)
            manifest["processes"] = records
            manifest["ports"] = [
                item["port"]
                for item in records
            ]
            manifest["state"] = "starting"
            save_manifest(manifest)

        # Launch all nodes before waiting so replicated model loading proceeds
        # concurrently. Publish `running` only after every health check passes.
        for record in records:
            wait_managed_ready(
                record,
                startup_timeout,
                api_key=os.environ.get("COLI_API_KEY"),
                cancel_event=cancel_event,
            )
            save_manifest(manifest)
        _raise_if_cancelled(cancel_event)
        manifest["state"] = "running"
        save_manifest(manifest)
        for process in spawned:
            track_managed_child(process)
        return manifest
    except BaseException as launch_error:
        cleanup_failures = []
        surviving_groups = set()

        def rollback_save(label):
            try:
                save_manifest(manifest)
                return True
            except Exception as save_exc:
                cleanup_failures.append(
                    "%s: %s" % (label, save_exc)
                )
                return False

        records_by_pid = {
            int(record["pid"]): record
            for record in records
        }
        # Published children get immediate identity+nonce revalidation.
        # An unpublished child is signaled only through its exact Popen PID.
        for process in reversed(spawned):
            record = records_by_pid.get(int(process.pid))
            if record is not None:
                track_managed_child(process)
                failure = terminate_verified_group(record)
            else:
                failure = terminate_direct_child(process)
            try:
                process.wait(timeout=1)
            except (
                subprocess.TimeoutExpired,
                ChildProcessError,
            ):
                pass
            forget_managed_child(process.pid)
            if record is not None:
                still_alive = process_matches(record)[0]
                surviving_pgid = int(record["pgid"])
            else:
                still_alive = process.poll() is None
                surviving_pgid = int(process.pid)
            if failure and still_alive:
                cleanup_failures.append(failure)
                surviving_groups.add(surviving_pgid)

        for context in launch_contexts:
            record = context.get("record") or {
                "state_dir": context["state_dir"],
                "usage_baseline": context["usage_baseline"],
            }
            if (
                context.get("record")
                and record["pgid"] in surviving_groups
            ):
                record["stop_error"] = (
                    "launch rollback could not terminate process group"
                )
                continue
            record.setdefault(
                "usage_merge_id",
                secrets.token_hex(16),
            )
            transaction_persisted = bool(
                context.get("record")
            ) and rollback_save(
                "could not persist usage transaction for %s"
                % context["state_dir"]
            )
            try:
                # If the transaction id could not be serialized, retain the
                # journal after applying it so a future start can recognize it.
                merge_usage(
                    record,
                    canonical_usage,
                    plan=plan,
                    keep_journal=not transaction_persisted,
                )
                if context.get("record"):
                    record["usage_merged_at"] = _utc_now()
                    record["stopped_at"] = _utc_now()
                    rollback_save(
                        "could not persist usage recovery for %s"
                        % context["state_dir"]
                    )
            except Exception as exc:
                if context.get("record"):
                    record["usage_merge_error"] = str(exc)
                cleanup_failures.append(
                    "usage recovery for %s: %s"
                    % (context["state_dir"], exc)
                )
        if (
            isinstance(launch_error, _OperationCancelled)
            and not cleanup_failures
        ):
            manifest["state"] = previous_state
            manifest["processes"] = previous_processes
            manifest["ports"] = previous_ports
            manifest["base_port"] = previous_base_port
            manifest.pop("launch_error", None)
            manifest.pop("cleanup_errors", None)
            try:
                save_manifest(manifest)
            except Exception as save_exc:
                cleanup_failures.append(
                    "could not persist clean launch cancellation: %s"
                    % save_exc
                )
            if not cleanup_failures:
                raise

        manifest["state"] = "error"
        manifest["launch_error"] = str(launch_error)
        manifest["cleanup_errors"] = cleanup_failures
        rollback_save("could not persist launch rollback")
        if cleanup_failures:
            raise RamdiskError(
                "%s; launch rollback/reporting errors: %s"
                % (
                    launch_error,
                    "; ".join(cleanup_failures),
                )
            ) from launch_error
        raise


def stop(
    args=None,
    *,
    load_manifest,
    process_matches,
    save_manifest,
    terminate_verified_group,
    merge_usage,
):
    manifest = load_manifest(required=True)
    plan = manifest["plan"]
    canonical_usage = os.path.join(
        manifest["plan"]["model"]["path"],
        ".coli_usage",
    )
    refusals = []
    identities = []
    for record in manifest.get("processes", []):
        if record.get("stopped_at"):
            identities.append(
                (record, False, "already-stopped", None)
            )
            continue
        matches, reason, actual = process_matches(record)
        if not matches and reason != "not-running":
            refusals.append(
                "PID %s is %s"
                % (record.get("pid"), reason)
            )
        identities.append(
            (record, matches, reason, actual)
        )
    # Validate every identity before signaling any process.
    if refusals:
        raise RamdiskError(
            "refusing to signal unverified processes: "
            + "; ".join(refusals)
        )
    for record, _, _, _ in identities:
        if not record.get("usage_merged_at"):
            record.setdefault(
                "usage_merge_id",
                secrets.token_hex(16),
            )
    # Persist transaction ids before signaling. Recovery can then recognize a
    # marker if the manager dies between usage replacement and manifest save.
    save_manifest(manifest)
    failures = []
    for record, matches, reason, actual in identities:
        if matches:
            pgid = int(record.get("pgid", record["pid"]))
            failure = terminate_verified_group(record)
            if failure:
                record["stop_error"] = failure
                failures.append(
                    "PID/PGID %s survived SIGKILL" % pgid
                )
                continue
        if not record.get("usage_merged_at"):
            try:
                merge_usage(
                    record,
                    canonical_usage,
                    plan=plan,
                )
                record["usage_merged_at"] = _utc_now()
                record.pop("usage_merge_error", None)
                # Every node is its own committed transaction.
                save_manifest(manifest)
            except Exception as exc:
                record["usage_merge_error"] = str(exc)
                failures.append(
                    "PID %s usage delta was not merged: %s"
                    % (record.get("pid"), exc)
                )
        record.setdefault("stopped_at", _utc_now())
        record.pop("stop_error", None)
    planned_paths = {
        record["path"]
        for record in plan["mounts"]
    }
    recorded_paths = {
        record["path"]
        for record in manifest.get("mounts", [])
    }
    incomplete_mount_layout = recorded_paths != planned_paths
    manifest["state"] = (
        "error"
        if (
            failures
            or incomplete_mount_layout
            or any(
                record.get("stop_error")
                or record.get("usage_merge_error")
                for record in manifest.get("processes", [])
            )
        )
        else "stopped"
    )
    save_manifest(manifest)
    if failures:
        raise RamdiskError(
            "engines were signaled but cleanup is incomplete: "
            + "; ".join(failures)
        )
    return manifest


def destroy(
    args,
    expected_manifest_token=None,
    *,
    load_manifest,
    manifest_confirmation_token,
    confirm,
    stop_action,
    mount_table,
    path_is_below,
    managed_path,
    mount_at,
    validate_mount,
    validate_namespace,
    busy_mount_references,
    umount_path,
    durable_unlink,
    manifest_path,
):
    manifest = load_manifest(required=True)
    if (
        expected_manifest_token is not None
        and manifest_confirmation_token(manifest)
        != expected_manifest_token
    ):
        raise RamdiskError(
            "RAM workspace changed since review; inspect the active "
            "deployment and confirm Destroy again"
        )
    confirm(
        "Stop engines and unmount all volatile RAM-disk weights?",
        bool(getattr(args, "yes", False)),
    )
    if manifest.get("processes"):
        manifest = stop_action(args)
    root = manifest["plan"]["mount_root"]
    preserved_mountpoints = []
    all_mounts_verified_here = True
    verified_mounts = []
    # Preflight every replica before changing any mount. A foreign or busy
    # final node must not leave earlier nodes already unmounted.
    planned_mounts = manifest["plan"]["mounts"]
    managed_paths = [
        record["path"]
        for record in planned_mounts
    ]
    recorded_by_path = {
        record["path"]: record
        for record in manifest.get("mounts", [])
    }
    nested_mounts = sorted(
        mount["path"]
        for mount in mount_table()
        if any(
            path_is_below(mount["path"], path)
            for path in managed_paths
        )
    )
    if nested_mounts:
        raise RamdiskError(
            "refusing managed mount(s) with nested child mounts: %s"
            % ", ".join(nested_mounts)
        )
    for planned in planned_mounts:
        path = planned["path"]
        if not managed_path(path, root):
            raise RamdiskError(
                "refusing unsafe managed path: %s" % path
            )
        actual = mount_at(path)
        record = recorded_by_path.get(path)
        if actual and record is None:
            # Without a recorded mount-id/device pair, a surviving mount could
            # now be foreign. Retain the recovery manifest for an operator.
            raise RamdiskError(
                "refusing unverified surviving mount at planned path: %s"
                % path
            )
        if record is None:
            all_mounts_verified_here = False
            preserved_mountpoints.append(path)
            continue
        expected = record.get("identity", {})
        if actual:
            if (
                actual.get("filesystem") != "tmpfs"
                or actual.get("source") != "tmpfs"
                or actual.get("mount_id")
                != expected.get("mount_id")
                or actual.get("device")
                != expected.get("device")
            ):
                raise RamdiskError(
                    "refusing foreign or replaced mount: %s"
                    % path
                )
            try:
                validated = validate_mount(
                    record,
                    manifest["plan"],
                )
            except RamdiskError as exc:
                raise RamdiskError(
                    "refusing foreign or altered mount at %s: %s"
                    % (path, exc)
                )
            if (
                validated["mount_id"]
                != expected.get("mount_id")
                or validated["device"]
                != expected.get("device")
            ):
                raise RamdiskError(
                    "refusing foreign or replaced mount: %s"
                    % path
                )
            if manifest.get("state") in ("ready", "stopped"):
                validate_namespace(
                    manifest["plan"],
                    record,
                    sample_numa=False,
                )
            busy = busy_mount_references(path)
            if busy:
                raise RamdiskError(
                    "mount %s is busy in PID(s): %s"
                    % (
                        path,
                        ",".join(str(pid) for pid in busy),
                    )
                )
            verified_mounts.append(record)
        else:
            # An externally unmounted path no longer has an identity we can
            # prove. Never remove it based only on serialized metadata.
            all_mounts_verified_here = False
            preserved_mountpoints.append(path)
            continue

    for record in reversed(verified_mounts):
        path = record["path"]
        umount_path(
            path,
            manifest["plan"]["hardware"],
        )
        if record.get("path_preexisting"):
            preserved_mountpoints.append(path)
            continue
        try:
            os.rmdir(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EPERM):
                # X-mount.mkdir creates /mnt targets as root. Sudo stays
                # deliberately limited to umount.
                preserved_mountpoints.append(path)
                continue
            if exc.errno not in (
                errno.ENOTEMPTY,
                errno.EBUSY,
            ):
                raise
            raise RamdiskError(
                "refusing to remove non-empty mount path: %s"
                % path
            )
    if (
        manifest["plan"]["topology"] == "per-node"
        and all_mounts_verified_here
        and not manifest["plan"].get(
            "mount_root_preexisting",
            True,
        )
    ):
        try:
            os.rmdir(root)
        except FileNotFoundError:
            pass
        except OSError:
            pass
    durable_unlink(manifest_path())
    return {
        "destroyed": True,
        "durable_state_preserved": True,
        "benchmark_history_preserved": True,
        "empty_mountpoints_preserved": sorted(
            set(preserved_mountpoints)
        ),
    }


def status(
    deep=True,
    *,
    load_manifest,
    manifest_path,
    source_still_matches,
    mount_at,
    validate_mount,
    validate_namespace,
    process_matches,
):
    """Return lifecycle status, optionally skipping shard/header revalidation.

    Scriptable ``status`` always uses the deep default.  The curses dashboard
    polls the cheap form and exposes an explicit refresh for a new deep model
    scan, avoiding repeated reads of every safetensors header on large models.
    """
    manifest = load_manifest(required=False)
    result = {
        "schema": STATUS_SCHEMA,
        "version": MANIFEST_VERSION,
        "manifest_path": manifest_path(),
        "present": bool(manifest),
        "state": (
            "absent"
            if not manifest
            else manifest.get("state", "unknown")
        ),
        "deep_validation": bool(deep),
        "mounts": [],
        "processes": [],
    }
    if not manifest:
        return result
    source_verified = None if not deep else True
    source_error = None
    if deep:
        try:
            source_still_matches(manifest["plan"])
        except RamdiskError as exc:
            source_verified = False
            source_error = str(exc)
    for record in manifest.get("mounts", []):
        actual = mount_at(record["path"])
        expected = record.get("identity", {})
        identity_verified = bool(
            actual
            and actual["filesystem"] == "tmpfs"
            and actual["source"] == "tmpfs"
            and actual["mount_id"]
            == expected.get("mount_id")
            and actual["device"]
            == expected.get("device")
        )
        options_verified = False
        namespace_verified = None if not deep else False
        option_error = namespace_error = None
        if identity_verified:
            try:
                validate_mount(record, manifest["plan"])
                options_verified = True
            except RamdiskError as exc:
                option_error = str(exc)
            if deep and options_verified and source_verified:
                try:
                    validate_namespace(
                        manifest["plan"],
                        record,
                        sample_numa=False,
                    )
                    namespace_verified = True
                except (OSError, RamdiskError) as exc:
                    namespace_error = str(exc)
        result["mounts"].append(
            {
                "path": record["path"],
                "node": record.get("node"),
                "mounted": bool(actual),
                "verified": (
                    identity_verified
                    and options_verified
                    and (
                        namespace_verified
                        if deep
                        else True
                    )
                ),
                "identity_verified": identity_verified,
                "options_verified": options_verified,
                "namespace_verified": namespace_verified,
                "option_error": option_error,
                "namespace_error": namespace_error,
                "filesystem": (
                    actual.get("filesystem")
                    if actual
                    else None
                ),
                "numa_allocation": record.get(
                    "numa_allocation",
                    {},
                ),
            }
        )
    for record in manifest.get("processes", []):
        if record.get("stopped_at"):
            matches, reason = False, "stopped"
        else:
            matches, reason, _ = process_matches(record)
        result["processes"].append(
            {
                "pid": record.get("pid"),
                "port": record.get("port"),
                "node": record.get("node"),
                "running": matches,
                "verified": matches,
                "reason": reason,
                "state_dir": record.get("state_dir"),
                "log": record.get("log"),
            }
        )
    result["model_fingerprint"] = manifest.get(
        "model_fingerprint"
    )
    result["mode"] = manifest["plan"].get("mode")
    result["topology"] = manifest["plan"].get("topology")
    result["ports"] = manifest.get("ports", [])
    result["source_fingerprint_verified"] = source_verified
    result["source_fingerprint_error"] = source_error
    return result
