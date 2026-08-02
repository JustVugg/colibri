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
    _MountHelperCompletedError,
    _OperationCancelled,
    _raise_if_cancelled,
    _utc_now,
)


_POPEN_BASE_TYPE = subprocess.Popen


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


def _retained_process_recovery(manifest):
    recovery = manifest.get("recovery")
    if not isinstance(recovery, dict):
        return []
    retained = recovery.get("retained_processes")
    return retained if isinstance(retained, list) else []


def _pending_launch_recovery(manifest):
    pending = manifest.get("pending_launches")
    return pending if isinstance(pending, list) else []


def _unresolved_process_recovery_error(action):
    return RamdiskError(
        "refusing %s while unpublished managed-child absence is unproven; "
        "run `coli ramdisk stop` to reconcile it after the process group "
        "exits, then inspect recovery.retained_processes in status"
        % action
    )


def _pending_launch_recovery_error(action):
    return RamdiskError(
        "refusing %s while a pre-spawn managed launch has an unknown "
        "outcome; inspect pending_launches in status and recover the "
        "deployment explicitly" % action
    )


def _reconcile_unpublished_processes(
    manifest,
    *,
    group_alive,
    merge_usage,
    save_manifest,
):
    """Merge an unpublished child only after its direct-created PGID is absent."""
    retained = list(_retained_process_recovery(manifest))
    if not retained:
        return manifest
    plan = manifest["plan"]
    canonical_usage = os.path.join(
        plan["model"]["path"],
        ".coli_usage",
    )
    failures = []
    remaining = []
    released = []
    for entry in retained:
        pid = entry.get("pgid", entry.get("pid"))
        try:
            alive = group_alive(int(pid))
        except BaseException as exc:
            alive = None
            failure = (
                "PID/PGID %s absence check failed during unpublished "
                "recovery: %s" % (pid, exc)
            )
        else:
            failure = None
        if alive is not False:
            failure = failure or (
                "PID/PGID %s unpublished managed process group is still "
                "live or its absence is unproven" % pid
            )
        baseline = entry.get("usage_baseline")
        merge_id = entry.get("usage_merge_id")
        if not isinstance(baseline, dict) or not (
            isinstance(merge_id, str)
            and len(merge_id) == 32
            and all(character in "0123456789abcdef" for character in merge_id)
        ):
            failure = failure or (
                "PID/PGID %s unpublished recovery is missing exact durable "
                "usage accounting metadata" % pid
            )
        if failure:
            retained_entry = dict(entry)
            retained_entry["error"] = failure
            remaining.append(retained_entry)
            failures.append(failure)
            continue
        try:
            merge_usage(
                entry,
                canonical_usage,
                plan=plan,
            )
            entry["usage_merged_at"] = _utc_now()
            save_manifest(manifest)
        except BaseException as exc:
            failure = (
                "PID/PGID %s unpublished usage delta was not merged: %s"
                % (pid, exc)
            )
            retained_entry = dict(entry)
            retained_entry["error"] = failure
            remaining.append(retained_entry)
            failures.append(failure)
            continue
        released.append(
            {
                "pid": entry.get("pid"),
                "pgid": pid,
                "state_dir": entry.get("state_dir"),
                "usage_merged_at": entry.get("usage_merged_at"),
            }
        )

    recovery = manifest.setdefault("recovery", {})
    recovery["retained_processes"] = remaining
    recovery["released_processes"] = released
    recovery["state"] = (
        "attention-required" if remaining else "reconciled"
    )
    manifest["state"] = "error"
    if failures:
        manifest.setdefault("cleanup_errors", []).extend(failures)
    save_manifest(manifest)
    if failures:
        raise RamdiskError(
            "unpublished managed-child recovery is incomplete: "
            + "; ".join(failures)
        )
    return manifest


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
        if record.get("ownership", "managed") != "managed":
            raise RamdiskError(
                "mount ownership is still pending at %s" % record["path"]
            )
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


def _same_managed_mount(actual, expected):
    """Return whether an observed mount is the exact recorded tmpfs."""
    return bool(
        actual
        and isinstance(expected, dict)
        and actual.get("filesystem") == "tmpfs"
        and actual.get("source") == "tmpfs"
        and actual.get("mount_id") == expected.get("mount_id")
        and actual.get("device") == expected.get("device")
    )


def _rollback_preparation_mounts(
    manifest,
    *,
    mount_at,
    mount_table,
    path_is_below,
    busy_mount_references,
    umount_path,
    validate_mount,
):
    """Roll back only mounts whose exact persisted ownership is still valid."""
    plan = manifest["plan"]
    cleanup_errors = []
    retained = set()
    released = set()
    candidates = []
    records = list(manifest.get("mounts", []))
    if not records:
        return cleanup_errors, retained, released

    def retain(record, message):
        path = record["path"]
        cleanup_errors.append(message)
        retained.add(path)
        record["cleanup"] = {
            "state": "retained",
            "error": message,
        }

    def nested_paths(table, path):
        return sorted(
            item["path"]
            for item in table
            if isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and path_is_below(item["path"], path)
        )

    def inspect_candidate(record, table, phase):
        path = record["path"]
        try:
            actual = mount_at(path)
        except BaseException as exc:
            return None, (
                "could not inspect mount %s at %s: %s"
                % (path, phase, exc)
            )
        if actual is None:
            return "absent", None
        if not _same_managed_mount(actual, record.get("identity")):
            return None, (
                "refusing changed mount during preparation rollback at %s: %s"
                % (phase, path)
            )
        try:
            verified = validate_mount(record, plan)
        except BaseException as exc:
            return None, (
                "could not validate mount during rollback at %s (%s): %s"
                % (path, phase, exc)
            )
        if not _same_managed_mount(verified, record.get("identity")):
            return None, (
                "refusing changed mount during preparation rollback at %s: %s"
                % (phase, path)
            )
        nested = nested_paths(table, path)
        if nested:
            return None, (
                "refusing preparation rollback for %s with nested mount(s) "
                "at %s: %s" % (path, phase, ", ".join(nested))
            )
        try:
            busy = busy_mount_references(
                path,
                hardware=plan["hardware"],
            )
        except BaseException as exc:
            return None, (
                "could not inspect busy references during rollback at %s "
                "(%s): %s" % (path, phase, exc)
            )
        if busy:
            return None, (
                "refusing preparation rollback for busy mount %s at %s; "
                "referenced by PID(s): %s"
                % (path, phase, ",".join(str(pid) for pid in busy))
            )
        if phase == "immediately before unmount":
            try:
                final_actual = mount_at(path)
                final_verified = validate_mount(record, plan)
            except BaseException as exc:
                return None, (
                    "could not revalidate exact mount identity after busy "
                    "scan at %s: %s" % (path, exc)
                )
            if (
                not _same_managed_mount(
                    final_actual,
                    record.get("identity"),
                )
                or not _same_managed_mount(
                    final_verified,
                    record.get("identity"),
                )
            ):
                return None, (
                    "refusing changed mount after busy scan immediately "
                    "before unmount: %s" % path
                )
        return "verified", None

    try:
        table = mount_table()
    except BaseException as exc:
        message = "could not inspect nested mounts during rollback: %s" % exc
        for record in records:
            retain(record, message)
        return cleanup_errors, retained, released

    for record in reversed(records):
        path = record["path"]
        ownership = record.get("ownership", "managed")
        if ownership == "pending":
            retain(record, (
                "refusing pathname-only rollback for pending mount: %s"
                % path
            ))
            continue
        result, error = inspect_candidate(record, table, "initial preflight")
        if result == "absent":
            released.add(path)
            record["cleanup"] = {"state": "absent"}
            continue
        if error:
            retain(record, error)
            continue
        candidates.append(record)

    if cleanup_errors:
        message = (
            "rollback withheld because another managed mount could not be "
            "verified safely"
        )
        for record in candidates:
            path = record["path"]
            retained.add(path)
            record["cleanup"] = {
                "state": "retained",
                "error": message,
            }
        return cleanup_errors, retained, released

    # Repeat the all-mount safety preflight at the latest shared boundary. This
    # preserves the no-known-partial-cleanup contract when a mount changes
    # after the initial scan but before the first unmount.
    try:
        latest_table = mount_table()
    except BaseException as exc:
        message = "could not refresh nested mounts before rollback: %s" % exc
        for record in candidates:
            retain(record, message)
        return cleanup_errors, retained, released

    latest_candidates = []
    latest_errors = False
    for record in candidates:
        path = record["path"]
        result, error = inspect_candidate(
            record,
            latest_table,
            "latest all-mount preflight",
        )
        if result == "absent":
            released.add(path)
            record["cleanup"] = {"state": "absent"}
        elif error:
            retain(record, error)
            latest_errors = True
        else:
            latest_candidates.append(record)
    if latest_errors:
        message = (
            "rollback withheld because another managed mount changed after "
            "initial preflight"
        )
        for record in latest_candidates:
            retain(record, message)
        return cleanup_errors, retained, released

    for index, record in enumerate(latest_candidates):
        path = record["path"]
        try:
            immediate_table = mount_table()
            result, error = inspect_candidate(
                record,
                immediate_table,
                "immediately before unmount",
            )
        except BaseException as exc:
            result, error = None, (
                "could not perform final rollback preflight at %s: %s"
                % (path, exc)
            )
        if result == "absent":
            released.add(path)
            record["cleanup"] = {"state": "absent"}
            continue
        if error:
            retain(record, error)
            message = (
                "rollback withheld after a late mount change at %s" % path
            )
            for pending in latest_candidates[index + 1:]:
                retain(pending, message)
            break
        try:
            umount_path(path, plan["hardware"])
        except BaseException as exc:
            message = "could not unmount %s during preparation rollback: %s" % (
                path,
                exc,
            )
            retain(record, message)
            for pending in latest_candidates[index + 1:]:
                retain(
                    pending,
                    "rollback withheld after unmount failure at %s" % path,
                )
            break
        try:
            after = mount_at(path)
        except BaseException as exc:
            retain(
                record,
                "could not verify mount absence after rollback at %s: %s"
                % (path, exc),
            )
            for pending in latest_candidates[index + 1:]:
                retain(
                    pending,
                    "rollback withheld after post-unmount verification "
                    "failure at %s" % path,
                )
            break
        if after is not None:
            retain(
                record,
                "mount remains or was replaced after rollback helper at %s"
                % path,
            )
            for pending in latest_candidates[index + 1:]:
                retain(
                    pending,
                    "rollback withheld after mount remained at %s" % path,
                )
            break
        released.add(path)
        record["cleanup"] = {"state": "unmounted"}

    return cleanup_errors, retained, released


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
    ensure_busy_mount_scan_available,
    durable_unlink,
    manifest_path,
    mount_table,
    path_is_below,
    busy_mount_references,
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
    ensure_busy_mount_scan_available(
        plan["mount_root"],
        hardware=plan["hardware"],
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
    durable_manifest = None

    def persist_manifest():
        nonlocal durable_manifest
        save_manifest(manifest)
        durable_manifest = copy.deepcopy(manifest)

    persist_manifest()
    try:
        for mount_index, mount in enumerate(plan["mounts"]):
            _raise_if_cancelled(cancel_event)
            if mount_at(mount["path"]):
                raise RamdiskError(
                    "refusing already-mounted path: %s"
                    % mount["path"]
                )
            record = dict(mount)
            record["ownership"] = "pending"
            record["operation_id"] = "%s:mount:%d" % (
                manifest["deployment_id"],
                mount_index,
            )
            record["requested"] = {
                "filesystem": "tmpfs",
                "source": "tmpfs",
                "size_bytes": mount.get("size_bytes"),
                "policy": mount.get("policy"),
                "thp": plan.get("mount_options", {}).get("thp"),
                "noswap": plan.get("mount_options", {}).get("noswap"),
                "safety_options": [
                    "noatime",
                    "nodev",
                    "nosuid",
                    "noexec",
                    "mode=0700",
                ],
            }
            manifest["mounts"].append(record)
            # This atomic write is the recovery boundary: no helper is invoked
            # until the intended pathname exists durably as unowned/pending.
            persist_manifest()
            try:
                mount_tmpfs(plan, mount)
            except _MountHelperCompletedError:
                # Only this typed failure proves the helper process completed.
                # A generic runner exception can happen while a privileged
                # helper is still in flight and must leave ownership pending.
                # Even after completed failure, observation proves absence but
                # cannot prove that an observed mount belongs to this attempt.
                try:
                    failed_actual = mount_at(mount["path"])
                except Exception:
                    pass
                else:
                    if failed_actual is None:
                        manifest["mounts"] = [
                            candidate
                            for candidate in manifest["mounts"]
                            if candidate.get("operation_id")
                            != record["operation_id"]
                        ]
                        persist_manifest()
                raise
            mounted_actual = mount_at(mount["path"])
            if not mounted_actual:
                raise RamdiskError(
                    "mounted %s but could not read its mount identity; "
                    "retained its pending recovery record"
                    % mount["path"]
                )
            for key in ("effective_thp", "effective_noswap"):
                if key in mount:
                    record[key] = mount[key]
            record["identity"] = mounted_actual
            record["ownership"] = "identified"
            record["validated"] = False
            persist_manifest()
            identity = validate_mount(mount, plan)
            if not _same_managed_mount(identity, mounted_actual):
                raise RamdiskError(
                    "mount identity changed while validating %s"
                    % mount["path"]
                )
            record["identity"] = identity
            record["ownership"] = "managed"
            record["validated"] = True
            persist_manifest()

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
            persist_manifest()
        _raise_if_cancelled(cancel_event)
        source_still_matches(plan)
        manifest["state"] = "ready"
        manifest["ready_at"] = _utc_now()
        persist_manifest()
        return manifest
    except BaseException as exc:
        # Rollback authority comes only from a manifest write that returned
        # successfully. An in-memory identity promotion whose write failed
        # must never authorize a pathname unmount during error handling.
        recovery_manifest = copy.deepcopy(durable_manifest)
        recovery_manifest["state"] = "error"
        recovery_manifest["error"] = str(exc)
        cleanup_errors = []
        try:
            save_manifest(recovery_manifest)
        except BaseException as save_exc:
            cleanup_errors.append(
                "could not persist preparation error: %s" % save_exc
            )
        (
            mount_cleanup_errors,
            retained_mounts,
            released_mounts,
        ) = _rollback_preparation_mounts(
            recovery_manifest,
            mount_at=mount_at,
            mount_table=mount_table,
            path_is_below=path_is_below,
            busy_mount_references=busy_mount_references,
            umount_path=umount_path,
            validate_mount=validate_mount,
        )
        cleanup_errors.extend(mount_cleanup_errors)
        recovery_manifest["recovery"] = {
            "operation": "prepare",
            "state": (
                "attention-required"
                if retained_mounts
                else "clean"
            ),
            "retained_mounts": sorted(retained_mounts),
            "released_mounts": sorted(released_mounts),
        }
        if cleanup_errors:
            recovery_manifest["cleanup_errors"] = cleanup_errors
        else:
            recovery_manifest.pop("cleanup_errors", None)
        try:
            save_manifest(recovery_manifest)
        except BaseException as save_exc:
            cleanup_errors.append(
                "could not persist cleanup result: %s" % save_exc
            )
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
            recovery_manifest["cleanup_errors"] = cleanup_errors
            raise RamdiskError(
                "%s; preparation rollback/reporting errors: %s"
                % (exc, "; ".join(cleanup_errors))
            ) from exc
        raise


def _rollback_launched_children(
    spawned,
    records,
    *,
    process_matches,
    group_alive,
    track_managed_child,
    terminate_verified_group,
    terminate_direct_child,
    forget_managed_child,
    launch_contexts=None,
):
    """Terminate launch children without mistaking uncertainty for absence."""
    cleanup_failures = []
    surviving_groups = set()
    records_by_pid = {
        int(record["pid"]): record
        for record in records
    }
    contexts_by_pid = {
        int(process.pid): context
        for process, context in zip(
            spawned,
            launch_contexts or (),
        )
    }

    # Published children get immediate identity+nonce revalidation. An
    # unpublished child is signaled only through its retained Popen handle.
    for process in reversed(spawned):
        pid = int(process.pid)
        record = records_by_pid.get(pid)
        context = contexts_by_pid.get(pid)
        termination_failure = None
        try:
            if record is not None:
                track_managed_child(process)
                termination_failure = terminate_verified_group(record)
                # Identity drift can make group signaling unsafe while the
                # retained Popen still proves this exact direct child is ours.
                # Terminating that child by handle is safe and may also let a
                # later group scan prove that no forked engine survived.
                try:
                    direct_child_needs_fallback = process.poll() is None
                except BaseException:
                    direct_child_needs_fallback = True
                if direct_child_needs_fallback:
                    try:
                        direct_failure = terminate_direct_child(process)
                    except BaseException as exc:
                        direct_failure = (
                            "direct-child termination attempt failed: %s"
                            % exc
                        )
                    if direct_failure:
                        termination_failure = "; ".join(
                            item
                            for item in (
                                str(termination_failure or ""),
                                str(direct_failure),
                            )
                            if item
                        )
            else:
                termination_failure = terminate_direct_child(process)
        except BaseException as exc:
            termination_failure = "termination attempt failed: %s" % exc

        wait_failure = None
        try:
            process.wait(timeout=1)
        except (
            subprocess.TimeoutExpired,
            ChildProcessError,
        ):
            pass
        except BaseException as exc:
            wait_failure = "could not wait for direct child: %s" % exc

        poll_failure = None
        try:
            direct_child_alive = process.poll() is None
        except BaseException as exc:
            direct_child_alive = True
            poll_failure = "could not poll direct child: %s" % exc

        identity_matches = False
        identity_reason = None
        identity_failure = None
        direct_created_group_alive = None
        group_identity_failure = None
        if record is not None:
            try:
                (
                    identity_matches,
                    identity_reason,
                    _,
                ) = process_matches(record)
            except BaseException as exc:
                identity_reason = "identity-check-failed"
                identity_failure = (
                    "could not establish managed process identity: %s" % exc
                )
        else:
            # Popen.poll() establishes only whether the direct wrapper exited.
            # start_new_session=True made its PID the directly-created PGID,
            # and a forked engine can remain there after the wrapper dies.
            try:
                direct_created_group_alive = group_alive(pid)
            except BaseException as exc:
                group_identity_failure = (
                    "could not establish direct-created process group %s "
                    "absence: %s" % (pid, exc)
                )

        absence_proven = (
            not direct_child_alive
            and (
                (
                    record is None
                    and direct_created_group_alive is False
                )
                or (
                    not identity_matches
                    and identity_reason == "not-running"
                )
            )
        )
        if absence_proven:
            try:
                forget_managed_child(pid)
            except BaseException as exc:
                cleanup_failures.append(
                    "could not forget reaped direct child %s: %s"
                    % (pid, exc)
                )
            continue

        details = []
        if termination_failure:
            details.append(str(termination_failure))
        if wait_failure:
            details.append(wait_failure)
        if poll_failure:
            details.append(poll_failure)
        if identity_failure:
            details.append(identity_failure)
        if group_identity_failure:
            details.append(group_identity_failure)
        if direct_child_alive:
            details.append("direct child is still alive")
        elif identity_matches:
            details.append("persisted process identity is still running")
        elif record is None and direct_created_group_alive:
            details.append(
                "direct-created process group %s is still alive" % pid
            )
        elif record is None:
            details.append(
                "direct-created process group %s absence remains unproven"
                % pid
            )
        else:
            details.append(
                "managed process absence remains unproven (%s)"
                % (identity_reason or "unknown")
            )
        failure = "; ".join(details)
        cleanup_failures.append(failure)
        surviving_pgid = (
            int(record["pgid"])
            if record is not None
            else pid
        )
        surviving_groups.add(surviving_pgid)
        if record is not None:
            record["stop_error"] = failure
        if context is not None:
            context["rollback_process_alive"] = True
            context["rollback_pid"] = pid
            context["rollback_error"] = failure

    return cleanup_failures, surviving_groups


def _construct_retained_popen(popen_factory, *args, **kwargs):
    """Construct Popen while retaining a real partially initialized attempt.

    Normal callable test doubles remain opaque: if they raise, no object exists
    whose child-creation fields can be inspected, so their outcome is unknown.
    """
    if not (
        isinstance(popen_factory, type)
        and issubclass(popen_factory, _POPEN_BASE_TYPE)
    ):
        try:
            return popen_factory(*args, **kwargs), None, False
        except BaseException as exc:
            return None, exc, False

    try:
        attempt = popen_factory.__new__(popen_factory)
    except BaseException as exc:
        return None, exc, False

    try:
        popen_factory.__init__(attempt, *args, **kwargs)
    except BaseException as exc:
        return attempt, exc, True
    return attempt, None, True


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
    group_alive,
    managed_child_liveness,
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
    validate_usage_for_plan,
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
    if _pending_launch_recovery(manifest):
        raise _pending_launch_recovery_error("start")
    if _retained_process_recovery(manifest):
        raise _unresolved_process_recovery_error("start")
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
    resolved_engine_path = None
    if engine_path is not None:
        resolved_engine_path = os.path.realpath(
            os.path.abspath(str(engine_path))
        )
        if (
            not os.path.isfile(resolved_engine_path)
            or not os.access(resolved_engine_path, os.X_OK)
        ):
            raise RamdiskError(
                "reviewed engine is not an executable file: %s"
                % resolved_engine_path
            )
    model = plan["model"]["path"]
    canonical_usage = os.path.join(model, ".coli_usage")
    foreign = []
    recovered = False
    for record in manifest.get("processes", []):
        _raise_if_cancelled(cancel_event)
        try:
            child_alive = managed_child_liveness(record["pid"])
            child_liveness_failure = None
        except BaseException as child_exc:
            child_alive = True
            child_liveness_failure = (
                "retained-child-liveness-check-failed: %s" % child_exc
            )
        try:
            matches, reason, _ = process_matches(record)
        except BaseException as identity_exc:
            matches, reason = (
                False,
                "identity-check-failed: %s" % identity_exc,
            )
        if child_liveness_failure:
            matches = False
            reason = child_liveness_failure
        if record.get("stopped_at"):
            if matches or reason != "not-running" or child_alive is True:
                foreign.append(
                    "PID %s (%s)"
                    % (
                        record.get("pid"),
                        (
                            "stopped-record-process-group-live"
                            if matches
                            else "retained-managed-child-live"
                            if reason == "not-running"
                            and child_alive is True
                            else reason
                        ),
                    )
                )
                continue
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
        if matches:
            raise RamdiskError(
                "managed engine is already running on port %s"
                % record.get("port")
            )
        if reason == "not-running" and child_alive is not True:
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
                % (
                    record.get("pid"),
                    (
                        "retained-managed-child-live"
                        if reason == "not-running"
                        and child_alive is True
                        else reason
                    ),
                )
            )
    if foreign:
        refusal = RamdiskError(
            "refusing stale foreign process records: "
            + ", ".join(foreign)
        )
        manifest["state"] = "error"
        manifest["launch_error"] = str(refusal)
        manifest.setdefault("cleanup_errors", []).append(str(refusal))
        try:
            save_manifest(manifest)
        except Exception as save_exc:
            raise RamdiskError(
                "%s; could not persist managed-child recovery: %s"
                % (refusal, save_exc)
            ) from refusal
        raise refusal
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
    # Recover every stable replica state first, then validate one canonical
    # seed before creating any per-engine copy or spawning the first child.
    # A later replica's journal therefore cannot introduce an incompatible
    # identified header after an earlier replica has already launched.
    seed_state_dirs = []
    for mount in manifest["mounts"]:
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
        seed_state_dirs.append(state_dir)
    canonical_baseline = usage_read(canonical_usage)
    validate_usage_for_plan(
        canonical_baseline,
        plan,
        source="canonical usage history %s" % canonical_usage,
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
    manifest["pending_launches"] = []
    manifest["state"] = "starting"
    save_manifest(manifest)

    try:
        for index, mount in enumerate(manifest["mounts"]):
            _raise_if_cancelled(cancel_event)
            node = mount.get("node")
            state_dir = seed_state_dirs[index]
            baseline = dict(canonical_baseline)
            usage_write(
                os.path.join(state_dir, ".coli_usage"),
                baseline,
            )
            port = base_port + (
                0 if node is None else int(node)
            )
            launch_nonce = secrets.token_hex(24)
            usage_merge_id = secrets.token_hex(16)
            pending_entry = {
                "operation_id": "start:%s" % usage_merge_id,
                "nonce": launch_nonce,
                "port": port,
                "node": node,
                "state_dir": state_dir,
                "usage_baseline": baseline,
                "usage_merge_id": usage_merge_id,
            }
            context = {
                "node": node,
                "state_dir": state_dir,
                "usage_baseline": baseline,
                "usage_merge_id": usage_merge_id,
                "pending_entry": pending_entry,
                "spawn_outcome": "not-attempted",
                "record": None,
            }
            launch_contexts.append(context)
            manifest["pending_launches"].append(pending_entry)
            # A hard manager crash after this durable write but before exact
            # process publication leaves an outcome-unknown launch that every
            # mutating action must retain and surface, never silently ignore.
            save_manifest(manifest)
            environment = os.environ.copy()
            environment.update(
                {
                    "COLI_WEIGHTS_DIR": mount["path"],
                    "COLI_STATE_DIR": state_dir,
                    "COLI_MANAGED_NONCE": launch_nonce,
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
                "COLI_ENGINE",
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
                "COLI_NO_OMP_TUNE",
                "COLI_OMP_TUNED",
                "DIRECT",
                "PIPE",
                "PIPE_WORKERS",
                "URING",
            ):
                environment.pop(inherited, None)
            if resolved_engine_path is not None:
                environment["COLI_ENGINE"] = resolved_engine_path
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
            _raise_if_cancelled(cancel_event)
            log = open(log_path, "ab", buffering=0)
            child_stdin = None
            try:
                child_stdin = open(os.devnull, "rb")
                # Retain outcome-unknown until a successful handle or a real
                # partially initialized Popen attempt proves what happened.
                context["spawn_outcome"] = "outcome-unknown"
                (
                    process,
                    construction_error,
                    attempt_inspected,
                ) = _construct_retained_popen(
                    subprocess.Popen,
                    command,
                    env=environment,
                    stdin=child_stdin,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
                if construction_error is not None:
                    child_created = bool(
                        getattr(process, "_child_created", False)
                    ) if process is not None else False
                    child_pid = (
                        getattr(process, "pid", None)
                        if process is not None
                        else None
                    )
                    exact_child_handle = (
                        isinstance(child_pid, int)
                        and not isinstance(child_pid, bool)
                        and child_pid > 0
                        and attempt_inspected
                    )
                    if exact_child_handle:
                        # Popen writes pid immediately before _child_created;
                        # normalize an interruption in that narrow window so
                        # poll/wait/destruction retain the exact child handle.
                        process._child_created = True
                        spawned.append(process)
                        context["spawn_outcome"] = "created"
                    elif (
                        isinstance(construction_error, Exception)
                        and attempt_inspected
                        and not child_created
                        and child_pid is None
                    ):
                        # A normal exception plus the inspected attempt state
                        # proves construction never published a child PID.
                        context["spawn_outcome"] = "proven-absent"
                    raise construction_error
                if process is None:
                    raise RamdiskError(
                        "Popen returned no process handle"
                    )
                spawned.append(process)
                context["spawn_outcome"] = "created"
            finally:
                try:
                    if child_stdin is not None:
                        child_stdin.close()
                finally:
                    log.close()
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
                    and identity.get("nonce") == launch_nonce
                ):
                    break
                time.sleep(0.01)
            if (
                process.poll() is not None
                or not identity
                or identity.get("pgid") != process.pid
                or identity.get("nonce") != launch_nonce
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
                "nonce": launch_nonce,
                "port": port,
                "node": node,
                "command": command,
                "state_dir": state_dir,
                "weights_dir": mount["path"],
                "usage_baseline": baseline,
                "usage_merge_id": usage_merge_id,
                "started_at": _utc_now(),
                "log": log_path,
                "runtime_knobs": applied_runtime_knobs,
                "accelerator_environment": applied_accelerator,
            }
            candidate_records = records + [record]
            manifest["processes"] = candidate_records
            manifest["ports"] = [
                item["port"]
                for item in candidate_records
            ]
            manifest["pending_launches"] = [
                pending
                for pending in manifest["pending_launches"]
                if pending.get("operation_id")
                != pending_entry["operation_id"]
            ]
            manifest["state"] = "starting"
            try:
                save_manifest(manifest)
            except BaseException:
                manifest["processes"] = list(records)
                manifest["ports"] = [
                    item["port"]
                    for item in records
                ]
                manifest["pending_launches"].append(pending_entry)
                raise
            records.append(record)
            context["record"] = record

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

        # Close the two line-level publication windows before rollback. A
        # handle present in ``spawned`` proves Popen returned even if control
        # flow landed before ``spawn_outcome`` was updated. Likewise, an exact
        # record still present in the candidate manifest is the published
        # authority even if interruption landed before the local records list
        # and context were updated. Normalizing both views prevents one
        # state_dir from being serialized as pending+retained or
        # published+retained.
        for context in launch_contexts[:len(spawned)]:
            context["spawn_outcome"] = "created"
        published_by_launch = {
            (
                record.get("usage_merge_id"),
                record.get("state_dir"),
            ): record
            for record in manifest.get("processes", [])
            if isinstance(record, dict)
        }
        for context in launch_contexts:
            published = published_by_launch.get(
                (
                    context.get("usage_merge_id"),
                    context.get("state_dir"),
                )
            )
            if published is not None:
                context["record"] = published
        rollback_records = list(manifest.get("processes", []))

        def rollback_save(label):
            try:
                save_manifest(manifest)
                return True
            except Exception as save_exc:
                cleanup_failures.append(
                    "%s: %s" % (label, save_exc)
                )
                return False

        (
            child_cleanup_failures,
            surviving_groups,
        ) = _rollback_launched_children(
            spawned,
            rollback_records,
            process_matches=process_matches,
            group_alive=group_alive,
            track_managed_child=track_managed_child,
            terminate_verified_group=terminate_verified_group,
            terminate_direct_child=terminate_direct_child,
            forget_managed_child=forget_managed_child,
            launch_contexts=launch_contexts,
        )
        cleanup_failures.extend(child_cleanup_failures)
        outcome_unknown_contexts = [
            context
            for context in launch_contexts
            if context.get("spawn_outcome") == "outcome-unknown"
        ]
        for context in outcome_unknown_contexts:
            cleanup_failures.append(
                "Popen was interrupted after launch began for %s; process "
                "creation outcome is unknown and pending recovery was retained"
                % context["state_dir"]
            )
        retained_processes = [
            {
                "pid": context.get("rollback_pid"),
                "pgid": context.get("rollback_pid"),
                "node": context.get("node"),
                "state_dir": context["state_dir"],
                "usage_baseline": context["usage_baseline"],
                "usage_merge_id": context["usage_merge_id"],
                "error": context.get("rollback_error"),
            }
            for context in launch_contexts
            if (
                context.get("rollback_process_alive")
                and context.get("record") is None
            )
        ]
        # Cooperative rollback has now classified every pending launch as
        # never-created, proven absent, or retained with an exact direct PGID.
        # Publish that transition before any usage transaction can be saved.
        manifest["pending_launches"] = [
            context["pending_entry"]
            for context in outcome_unknown_contexts
        ]
        if retained_processes:
            manifest["recovery"] = {
                "operation": "start",
                "state": "attention-required",
                "retained_processes": retained_processes,
            }

        for context in launch_contexts:
            record = context.get("record") or {
                "state_dir": context["state_dir"],
                "usage_baseline": context["usage_baseline"],
                "usage_merge_id": context["usage_merge_id"],
            }
            if context.get("rollback_process_alive"):
                # The child may still be writing this state directory. Never
                # merge or publish its usage while direct-child absence is
                # unproven, even when identity publication never completed.
                continue
            if context.get("spawn_outcome") == "outcome-unknown":
                # An unknown child may still be writing. Its baseline and
                # transaction id stay private and durable in pending_launches.
                continue
            if (
                context.get("record")
                and record["pgid"] in surviving_groups
            ):
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
    group_alive,
    managed_child_liveness,
    save_manifest,
    terminate_verified_group,
    merge_usage,
):
    manifest = load_manifest(required=True)
    if _pending_launch_recovery(manifest):
        refusal = _pending_launch_recovery_error("stop")
        manifest["state"] = "error"
        manifest.setdefault("cleanup_errors", []).append(str(refusal))
        try:
            save_manifest(manifest)
        except BaseException as save_exc:
            raise RamdiskError(
                "%s; could not persist pending-launch recovery: %s"
                % (refusal, save_exc)
            ) from refusal
        raise refusal
    if _retained_process_recovery(manifest):
        manifest = _reconcile_unpublished_processes(
            manifest,
            group_alive=group_alive,
            merge_usage=merge_usage,
            save_manifest=save_manifest,
        )
    non_managed_mounts = [
        record
        for record in manifest.get("mounts", [])
        if record.get("ownership", "managed") != "managed"
    ]
    if non_managed_mounts:
        refusal = RamdiskError(
            "refusing stop while non-managed mount ownership requires "
            "recovery: "
            + ", ".join(
                "%s (%s)"
                % (record.get("path"), record.get("ownership"))
                for record in non_managed_mounts
            )
        )
        manifest["state"] = "error"
        manifest.setdefault("cleanup_errors", []).append(str(refusal))
        try:
            save_manifest(manifest)
        except BaseException as save_exc:
            raise RamdiskError(
                "%s; could not persist mount recovery refusal: %s"
                % (refusal, save_exc)
            ) from refusal
        raise refusal
    plan = manifest["plan"]
    canonical_usage = os.path.join(
        manifest["plan"]["model"]["path"],
        ".coli_usage",
    )
    refusals = []
    identities = []
    for record in manifest.get("processes", []):
        try:
            child_alive = managed_child_liveness(record["pid"])
            child_liveness_failure = None
        except BaseException as child_exc:
            child_alive = True
            child_liveness_failure = (
                "retained-child-liveness-check-failed: %s" % child_exc
            )
        try:
            matches, reason, actual = process_matches(record)
        except BaseException as identity_exc:
            matches, reason, actual = (
                False,
                "identity-check-failed: %s" % identity_exc,
                None,
            )
        if child_liveness_failure:
            matches = False
            reason = child_liveness_failure
        if record.get("stopped_at"):
            if matches or reason != "not-running" or child_alive is True:
                record["stop_error"] = (
                    "stopped-record-process-group-live"
                    if matches
                    else child_liveness_failure
                    or "retained-managed-child-live"
                    if reason == "not-running"
                    and child_alive is True
                    else reason
                )
                refusals.append(
                    "PID %s is %s"
                    % (record.get("pid"), record["stop_error"])
                )
            identities.append(
                (record, False, "already-stopped", None)
            )
            continue
        if not matches and reason == "not-running" and child_alive is True:
            reason = "retained-managed-child-live"
        if not matches and reason != "not-running":
            record["stop_error"] = reason
            refusals.append(
                "PID %s is %s"
                % (record.get("pid"), reason)
            )
        identities.append(
            (record, matches, reason, actual)
        )
    # Validate every identity before signaling any process.
    if refusals:
        refusal = RamdiskError(
            "refusing to signal unverified processes: "
            + "; ".join(refusals)
        )
        manifest["state"] = "error"
        manifest.setdefault("cleanup_errors", []).append(str(refusal))
        try:
            save_manifest(manifest)
        except Exception as save_exc:
            raise RamdiskError(
                "%s; could not persist stop recovery: %s"
                % (refusal, save_exc)
            ) from refusal
        raise refusal
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
            try:
                failure = terminate_verified_group(record)
            except BaseException as termination_exc:
                failure = (
                    "PID/PGID %s termination revalidation failed: %s"
                    % (pgid, termination_exc)
                )
            try:
                post_matches, post_reason, _ = process_matches(record)
            except BaseException as identity_exc:
                post_matches, post_reason = (
                    False,
                    "identity-check-failed: %s" % identity_exc,
                )
            try:
                child_alive = managed_child_liveness(record["pid"])
                child_liveness_failure = None
            except BaseException as child_exc:
                child_alive = True
                child_liveness_failure = (
                    "retained-child liveness check failed: %s" % child_exc
                )
            absence_proven = (
                not post_matches
                and post_reason == "not-running"
                and child_alive is not True
            )
            if not absence_proven:
                retained_failure = "; ".join(
                    item
                    for item in (
                        str(failure or ""),
                        "PID/PGID %s absence is unproven after termination "
                        "(%s)" % (pgid, post_reason),
                        str(child_liveness_failure or ""),
                    )
                    if item
                )
                record["stop_error"] = retained_failure
                failures.append(retained_failure)
                continue
            record.pop("stop_error", None)
        if record.get("usage_merged_at"):
            # The applied transaction marker is authoritative. Older/error
            # snapshots could retain a save failure in usage_merge_error after
            # the exact merge had already completed; do not let that stale
            # diagnostic make every later stop permanently fail.
            record.pop("usage_merge_error", None)
        else:
            try:
                merge_usage(
                    record,
                    canonical_usage,
                    plan=plan,
                )
                record["usage_merged_at"] = _utc_now()
                record.pop("usage_merge_error", None)
            except Exception as exc:
                record["usage_merge_error"] = str(exc)
                failures.append(
                    "PID %s usage delta was not merged: %s"
                    % (record.get("pid"), exc)
                )
            else:
                try:
                    # Every node is its own committed transaction. If this
                    # intermediate write fails, the final recovery write (or a
                    # fresh retry using the same stable transaction id) can
                    # publish the already-applied marker without reapplying it.
                    save_manifest(manifest)
                except Exception as exc:
                    failures.append(
                        "PID %s usage merge completed but its manifest marker "
                        "could not be persisted: %s"
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
    save_manifest,
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
    if (
        manifest.get("processes")
        or _retained_process_recovery(manifest)
        or _pending_launch_recovery(manifest)
    ):
        manifest = stop_action(args)
    retained_processes = _retained_process_recovery(manifest)
    if retained_processes:
        raise RamdiskError(
            "refusing destroy while unpublished managed-child absence is "
            "unproven; inspect recovery.retained_processes"
        )
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
    released_mounts = set()

    def persist_destroy_failure(error):
        message = str(error)
        retained_mounts = sorted(
            set(managed_paths) - released_mounts
        )
        manifest["state"] = "error"
        manifest["destroy_error"] = message
        manifest["recovery"] = {
            "operation": "destroy",
            "state": "attention-required",
            "retained_mounts": retained_mounts,
            "released_mounts": sorted(released_mounts),
        }
        retained_set = set(retained_mounts)
        for record in manifest.get("mounts", []):
            if record.get("path") in retained_set:
                record["cleanup"] = {
                    "state": "retained",
                    "error": message,
                }
        try:
            save_manifest(manifest)
        except Exception as save_exc:
            raise RamdiskError(
                "%s; could not persist destroy recovery: %s"
                % (message, save_exc)
            ) from error
        if isinstance(error, RamdiskError):
            raise error
        raise RamdiskError(message) from error

    recorded_by_path = {
        record["path"]: record
        for record in manifest.get("mounts", [])
    }
    pending_mounts = sorted(
        record["path"]
        for record in manifest.get("mounts", [])
        if record.get("ownership") == "pending"
    )
    if pending_mounts:
        # A hard manager crash can leave a privileged mount helper in flight.
        # Path absence is only a momentary observation and cannot prove that
        # helper will not publish an untracked tmpfs after this process exits.
        persist_destroy_failure(RamdiskError(
            "refusing destroy while managed mount helper outcome is unknown "
            "for pending path(s): %s" % ", ".join(pending_mounts)
        ))
    try:
        observed_mounts = mount_table()
    except Exception as exc:
        persist_destroy_failure(exc)
    nested_mounts = sorted(
        mount["path"]
        for mount in observed_mounts
        if any(
            path_is_below(mount["path"], path)
            for path in managed_paths
        )
    )
    if nested_mounts:
        persist_destroy_failure(RamdiskError(
            "refusing managed mount(s) with nested child mounts: %s"
            % ", ".join(nested_mounts)
        ))
    for planned in planned_mounts:
        path = planned["path"]
        if not managed_path(path, root):
            persist_destroy_failure(RamdiskError(
                "refusing unsafe managed path: %s" % path
            ))
        try:
            actual = mount_at(path)
        except Exception as exc:
            persist_destroy_failure(exc)
        record = recorded_by_path.get(path)
        if actual and (
            record is None
            or record.get("ownership") == "pending"
        ):
            # Without a recorded mount-id/device pair, a surviving mount could
            # now be foreign. Retain the recovery manifest for an operator.
            persist_destroy_failure(RamdiskError(
                "refusing unverified surviving mount at planned path: %s"
                % path
            ))
        if record is None:
            all_mounts_verified_here = False
            preserved_mountpoints.append(path)
            released_mounts.add(path)
            continue
        expected = record.get("identity") or {}
        if actual:
            if not _same_managed_mount(actual, expected):
                persist_destroy_failure(RamdiskError(
                    "refusing foreign or replaced mount: %s"
                    % path
                ))
            try:
                validated = validate_mount(
                    record,
                    manifest["plan"],
                )
            except RamdiskError as exc:
                persist_destroy_failure(RamdiskError(
                    "refusing foreign or altered mount at %s: %s"
                    % (path, exc)
                ))
            except Exception as exc:
                persist_destroy_failure(exc)
            if not _same_managed_mount(validated, expected):
                persist_destroy_failure(RamdiskError(
                    "refusing foreign or replaced mount: %s"
                    % path
                ))
            if manifest.get("state") in ("ready", "stopped"):
                try:
                    validate_namespace(
                        manifest["plan"],
                        record,
                        sample_numa=False,
                    )
                except Exception as exc:
                    persist_destroy_failure(exc)
            try:
                busy = busy_mount_references(
                    path,
                    hardware=manifest["plan"]["hardware"],
                )
            except Exception as exc:
                persist_destroy_failure(exc)
            if busy:
                persist_destroy_failure(RamdiskError(
                    "mount %s is busy in PID(s): %s"
                    % (
                        path,
                        ",".join(str(pid) for pid in busy),
                    )
                ))
            verified_mounts.append(record)
        else:
            # An externally unmounted path no longer has an identity we can
            # prove. Never remove it based only on serialized metadata.
            all_mounts_verified_here = False
            preserved_mountpoints.append(path)
            released_mounts.add(path)
            continue

    for record in reversed(verified_mounts):
        path = record["path"]
        expected = record.get("identity") or {}
        # The initial sweep prevents known partial teardown. Re-read every
        # safety predicate again at the latest possible boundary so a mount
        # replacement, nested mount, or new busy reference cannot ride a stale
        # preflight into a pathname-based unmount.
        try:
            latest = mount_at(path)
        except BaseException as exc:
            persist_destroy_failure(exc)
        if not _same_managed_mount(latest, expected):
            persist_destroy_failure(RamdiskError(
                "refusing foreign or replaced mount immediately before "
                "unmount: %s" % path
            ))
        try:
            latest_validated = validate_mount(record, manifest["plan"])
        except BaseException as exc:
            persist_destroy_failure(exc)
        if not _same_managed_mount(latest_validated, expected):
            persist_destroy_failure(RamdiskError(
                "refusing foreign or replaced mount immediately before "
                "unmount: %s" % path
            ))
        try:
            latest_table = mount_table()
        except BaseException as exc:
            persist_destroy_failure(exc)
        latest_nested = sorted(
            item["path"]
            for item in latest_table
            if isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and path_is_below(item["path"], path)
        )
        if latest_nested:
            persist_destroy_failure(RamdiskError(
                "refusing managed mount with nested child mounts "
                "immediately before unmount at %s: %s"
                % (path, ", ".join(latest_nested))
            ))
        try:
            latest_busy = busy_mount_references(
                path,
                hardware=manifest["plan"]["hardware"],
            )
        except BaseException as exc:
            persist_destroy_failure(exc)
        if latest_busy:
            persist_destroy_failure(RamdiskError(
                "mount %s became busy before unmount in PID(s): %s"
                % (
                    path,
                    ",".join(str(pid) for pid in latest_busy),
                )
            ))
        try:
            post_busy_identity = mount_at(path)
            post_busy_validated = validate_mount(
                record,
                manifest["plan"],
            )
        except BaseException as exc:
            persist_destroy_failure(exc)
        if (
            not _same_managed_mount(post_busy_identity, expected)
            or not _same_managed_mount(post_busy_validated, expected)
        ):
            persist_destroy_failure(RamdiskError(
                "refusing foreign or replaced mount after busy scan "
                "immediately before unmount: %s" % path
            ))
        try:
            umount_path(
                path,
                manifest["plan"]["hardware"],
            )
        except BaseException as exc:
            persist_destroy_failure(exc)
        try:
            after_unmount = mount_at(path)
        except BaseException as exc:
            persist_destroy_failure(exc)
        if after_unmount is not None:
            persist_destroy_failure(RamdiskError(
                "mount remains or was replaced after unmount helper at %s"
                % path
            ))
        released_mounts.add(path)
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
                persist_destroy_failure(exc)
            persist_destroy_failure(RamdiskError(
                "refusing to remove non-empty mount path: %s"
                % path
            ))
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
    try:
        durable_unlink(manifest_path())
    except Exception as exc:
        persist_destroy_failure(exc)
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
    managed_child_liveness,
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
        "recovery": None,
    }
    if not manifest:
        return result
    recovery = manifest.get("recovery")
    retained_processes = _retained_process_recovery(manifest)
    pending_launches = _pending_launch_recovery(manifest)
    planned_mount_paths = {
        record.get("path")
        for record in manifest.get("plan", {}).get("mounts", [])
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    serialized_retained_mounts = (
        recovery.get("retained_mounts", [])
        if isinstance(recovery, dict)
        and isinstance(recovery.get("retained_mounts", []), list)
        else []
    )
    serialized_released_mounts = (
        recovery.get("released_mounts", [])
        if isinstance(recovery, dict)
        and isinstance(recovery.get("released_mounts", []), list)
        else []
    )
    ownership_recovery_mounts = [
        record.get("path")
        for record in manifest.get("mounts", [])
        if (
            isinstance(record, dict)
            and record.get("ownership", "managed") != "managed"
            and isinstance(record.get("path"), str)
        )
    ]
    retained_mounts = sorted(
        set(
            path
            for path in serialized_retained_mounts
            if isinstance(path, str) and path in planned_mount_paths
        )
        | set(ownership_recovery_mounts)
    )
    released_mounts = sorted(
        path
        for path in serialized_released_mounts
        if isinstance(path, str) and path in planned_mount_paths
    )
    cleanup_errors = manifest.get("cleanup_errors", [])
    if not isinstance(cleanup_errors, list):
        cleanup_errors = [str(cleanup_errors)]
    stop_errors = [
        {
            "pid": record.get("pid"),
            "error": str(
                record.get("stop_error")
                or record.get("usage_merge_error")
            ),
        }
        for record in manifest.get("processes", [])
        if record.get("stop_error") or record.get("usage_merge_error")
    ]
    error_summary = {
        key: str(manifest[key])
        for key in ("error", "launch_error", "destroy_error")
        if manifest.get(key)
    }
    if cleanup_errors:
        error_summary["cleanup_errors"] = [
            str(item) for item in cleanup_errors
        ]
    if stop_errors:
        error_summary["stop_errors"] = stop_errors
    if (
        isinstance(recovery, dict)
        or pending_launches
        or retained_mounts
        or error_summary
    ):
        result["recovery"] = {
            "operation": (
                recovery.get("operation")
                if isinstance(recovery, dict)
                else "prepare"
                if ownership_recovery_mounts
                else "start"
                if pending_launches
                else None
            ),
            "state": (
                "attention-required"
                if (
                    pending_launches
                    or retained_processes
                    or ownership_recovery_mounts
                )
                else recovery.get("state")
                if isinstance(recovery, dict)
                else "attention-required"
            ),
            "retained_mounts": retained_mounts,
            "released_mounts": released_mounts,
            "retained_processes": [
                {
                    "pid": entry.get("pid"),
                    "pgid": entry.get("pgid", entry.get("pid")),
                    "state_dir": entry.get("state_dir"),
                    "error": entry.get("error"),
                }
                for entry in retained_processes
                if isinstance(entry, dict)
            ],
            "pending_launches": [
                {
                    "port": entry.get("port"),
                    "node": entry.get("node"),
                    "state_dir": entry.get("state_dir"),
                    "state": "outcome-unknown",
                }
                for entry in pending_launches
                if isinstance(entry, dict)
            ],
            "errors": error_summary,
            "action": (
                "Do not start or destroy this deployment; inspect the "
                "outcome-unknown pending launch and reconcile it explicitly."
                if pending_launches
                else "Run `coli ramdisk stop` after the retained process "
                "group exits to reconcile its exact usage transaction."
                if retained_processes
                else "Inspect pending ownership, the retained mount identity, "
                "nested mounts, and busy references; explicitly reconcile "
                "any uncertainty, then retry `coli ramdisk destroy` only "
                "after confirming it is safe."
                if retained_mounts
                else None
            ),
        }
    source_verified = None if not deep else True
    source_error = None
    if deep:
        try:
            source_still_matches(manifest["plan"])
        except Exception as exc:
            source_verified = False
            source_error = str(exc)
    for record in manifest.get("mounts", []):
        try:
            actual = mount_at(record["path"])
            mount_read_error = None
        except Exception as mount_exc:
            actual = None
            mount_read_error = str(mount_exc)
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
        option_error = mount_read_error
        namespace_error = None
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
                "ownership": record.get("ownership", "managed"),
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
        try:
            matches, reason, _ = process_matches(record)
        except Exception as identity_exc:
            matches, reason = (
                False,
                "identity-check-failed: %s" % identity_exc,
            )
        try:
            child_alive = managed_child_liveness(record.get("pid"))
        except Exception as child_exc:
            child_alive = None
            if reason == "not-running":
                reason = "retained-child-liveness-check-failed: %s" % child_exc
        attention_required = False
        if record.get("stopped_at"):
            if matches:
                reason = "stopped-record-process-group-live"
                attention_required = True
            elif child_alive is True:
                reason = "stopped-record-retained-child-live"
                attention_required = True
            elif reason != "not-running":
                attention_required = True
            else:
                reason = "stopped"
        elif not matches and child_alive is True:
            reason = "retained-managed-child-live"
            attention_required = True
        elif not matches and reason != "not-running":
            attention_required = True
        result["processes"].append(
            {
                "pid": record.get("pid"),
                "port": record.get("port"),
                "node": record.get("node"),
                "running": bool(matches or child_alive is True),
                "verified": matches,
                "reason": reason,
                "attention_required": attention_required,
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
