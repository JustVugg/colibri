"""Durable RAM-disk state, manifests, locking, and usage recovery."""

from __future__ import print_function

import contextlib
import datetime
import hashlib
import json
import os
import posixpath
import re
import secrets
import stat
import threading

from .common import (
    DEFAULT_MOUNT_ROOT,
    MANIFEST_VERSION,
    MIB,
    PROFILE_LINE_RE,
    USAGE_MERGE_RE,
    RamdiskError,
    _path_is_below,
    _path_without_symlinks,
    _positive_int,
    _validated_usage_header,
    _utc_now,
    _usage_engine_id,
    _usage_engine_name,
)
from .platform_ops import current_uid, get_platform_ops
from .accelerator import _managed_accelerator_contract


CONTAINMENT_VERSION = 1


def validate_containment(value):
    """Load runner containment validation only for versioned manifests."""
    from .supervision import validate_containment as validate

    return validate(value)


try:
    import fcntl
except ImportError:
    fcntl = None


_lifecycle_local = threading.local()
_fallback_usage_lock = threading.RLock()
_NATIVE_DIRFD_PRIMITIVES = (
    os.name == "posix"
    and os.open in getattr(os, "supports_dir_fd", set())
    and os.stat in getattr(os, "supports_dir_fd", set())
    and os.unlink in getattr(os, "supports_dir_fd", set())
    and os.rename in getattr(os, "supports_dir_fd", set())
)


def _supports_native_dirfd():
    """Return whether the current runtime can bind every required file step."""
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    return (
        _NATIVE_DIRFD_PRIMITIVES
        and all(
            isinstance(getattr(os, name, None), int)
            and getattr(os, name) != 0
            for name in required_flags
        )
    )


def _valid_utc_timestamp(value):
    if not isinstance(value, str) or not value:
        return False
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return (
        parsed.tzinfo is not None
        and parsed.utcoffset() == datetime.timedelta(0)
    )


def _validate_supervised_containment(record, *, pending):
    """Validate one versioned cgroup identity and its durable transitions."""
    try:
        validate_containment(record.get("containment"))
    except RamdiskError as exc:
        raise RamdiskError(
            "RAM-disk manifest has invalid managed containment: %s" % exc
        ) from exc
    authorized_at = record.get("containment_removal_authorized_at")
    removed_at = record.get("containment_removed_at")
    if (
        authorized_at is not None
        and not _valid_utc_timestamp(authorized_at)
    ) or (
        removed_at is not None
        and not _valid_utc_timestamp(removed_at)
    ) or (removed_at is not None and authorized_at is None):
        raise RamdiskError(
            "RAM-disk manifest has invalid containment removal state"
        )
    if not pending:
        return
    phase = record.get("containment_phase")
    pid = record.get("pid")
    phases = (
        "cgroup-created",
        "gate-spawned",
        "attached-verified",
        "gate-released",
    )
    if phase not in phases:
        raise RamdiskError(
            "RAM-disk manifest has invalid managed containment phase"
        )
    if (
        phase == "cgroup-created"
        and pid is not None
    ) or (
        phase != "cgroup-created"
        and not _positive_int(pid)
    ):
        raise RamdiskError(
            "RAM-disk manifest has invalid managed gate identity"
        )


def _benchmark_workspace_source_fingerprint(plan):
    selected = set((plan.get("staging") or {}).get("selected_shards") or [])
    shards = []
    for item in plan.get("source_shards") or []:
        if isinstance(item, dict) and item.get("name") in selected:
            shards.append(
                {
                    "name": item.get("name"),
                    "size_bytes": item.get("size_bytes"),
                    "header_sha256": item.get("header_sha256"),
                }
            )
    projection = {
        "model_fingerprint": (plan.get("model") or {}).get("fingerprint"),
        "selected_shards": sorted(selected),
        "selected_source_identities": sorted(
            shards,
            key=lambda item: item["name"],
        ),
        "linked_shards": sorted(
            (plan.get("staging") or {}).get("linked_shards") or []
        ),
    }
    return hashlib.sha256(
        json.dumps(
            projection,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _validate_benchmark_pending_process(record, *, workspace, manifest, state_root):
    required = {
        "version", "operation_id", "workspace_operation_id", "protocol_id",
        "treatment_id", "block_index", "sequence", "launch_id", "uid",
        "state_dir", "weights_dir", "expected_command", "environment_sha256",
        "containment", "containment_phase", "pid", "starttime", "created_at",
    }
    optional = {
        "containment_removal_authorized_at", "containment_removed_at",
        "recovery_error",
    }
    phases = (
        "create-intent", "cgroup-created", "gate-spawned",
        "attached-verified", "gate-released",
    )
    treatments = {
        "anon-pin-interleaved", "anon-pin-local",
        "tmpfs-rammap-interleaved", "tmpfs-rammap-local",
        "ssd-slab-control", "tmpfs-slab-control",
        "cuda-fixed-budget-validation",
    }
    if (
        not isinstance(record, dict)
        or not required.issubset(record)
        or not set(record).issubset(required | optional)
        or record.get("version") != 1
        or isinstance(record.get("version"), bool)
        or not isinstance(record.get("operation_id"), str)
        or re.fullmatch(r"replicate:[0-9a-f]{32}", record["operation_id"])
        is None
        or record.get("workspace_operation_id") != workspace.get("operation_id")
        or record.get("protocol_id") != workspace.get("protocol_id")
        or not isinstance(record.get("protocol_id"), str)
        or re.fullmatch(r"[0-9a-f]{64}", record["protocol_id"]) is None
        or record.get("treatment_id") not in treatments
        or isinstance(record.get("block_index"), bool)
        or not isinstance(record.get("block_index"), int)
        or record["block_index"] < 0
        or isinstance(record.get("sequence"), bool)
        or not isinstance(record.get("sequence"), int)
        or record["sequence"] < 0
        or record["sequence"] // len(treatments) != record["block_index"]
        or not isinstance(record.get("launch_id"), str)
        or re.fullmatch(r"[0-9a-f]{32}", record["launch_id"]) is None
        or record["operation_id"] != "replicate:" + record["launch_id"]
        or isinstance(record.get("uid"), bool)
        or not isinstance(record.get("uid"), int)
        or record["uid"] != current_uid()
        or not isinstance(record.get("environment_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", record["environment_sha256"])
        is None
        or record.get("containment_phase") not in phases
        or not _valid_utc_timestamp(record.get("created_at"))
        or not isinstance(record.get("expected_command"), list)
        or not record["expected_command"]
        or any(
            not isinstance(item, str) or not item
            for item in record["expected_command"]
        )
    ):
        raise RamdiskError("RAM-disk manifest has invalid benchmark process")
    phase = record["containment_phase"]
    pid = record.get("pid")
    starttime = record.get("starttime")
    containment = record.get("containment")
    if phase == "create-intent":
        if containment is not None or pid is not None or starttime is not None:
            raise RamdiskError("RAM-disk manifest has invalid benchmark create intent")
    else:
        validated = validate_containment(containment)
        expected_relative = "colibri/d%s/o%s" % (
            hashlib.sha256(manifest["deployment_id"].encode("utf-8")).hexdigest()[:24],
            hashlib.sha256(record["operation_id"].encode("utf-8")).hexdigest()[:24],
        )
        if validated["relative_path"] != expected_relative:
            raise RamdiskError("RAM-disk benchmark containment belongs to another operation")
        if phase == "cgroup-created":
            if pid is not None or starttime is not None:
                raise RamdiskError("RAM-disk manifest has invalid benchmark cgroup phase")
        elif not _positive_int(pid) or not _positive_int(starttime):
            raise RamdiskError("RAM-disk manifest has invalid benchmark gate identity")
    state_dir = record.get("state_dir")
    prefix = os.path.join(
        state_root(), "causal-benchmark-state", record["protocol_id"]
    )
    expected_name = "%04d-%03d-%s-%s" % (
        record["block_index"], record["sequence"], record["treatment_id"],
        record["launch_id"],
    )
    if (
        not isinstance(state_dir, str)
        or not os.path.isabs(state_dir)
        or os.path.dirname(state_dir) != prefix
        or os.path.basename(state_dir) != expected_name
        or not _path_without_symlinks(state_dir)
    ):
        raise RamdiskError("RAM-disk manifest has invalid benchmark process state path")
    workspace_names = {
        "tmpfs-rammap-interleaved": "interleaved",
        "tmpfs-slab-control": "interleaved",
        "tmpfs-rammap-local": "local",
    }
    if record["treatment_id"] in workspace_names:
        expected_weights = next(
            root["path"] for root in workspace["roots"]
            if root["name"] == workspace_names[record["treatment_id"]]
        )
    else:
        expected_weights = manifest["plan"]["model"]["path"]
    if record.get("weights_dir") != expected_weights:
        raise RamdiskError("RAM-disk manifest has invalid benchmark process weights")
    for name in (
        "containment_removal_authorized_at", "containment_removed_at"
    ):
        if record.get(name) is not None and not _valid_utc_timestamp(record[name]):
            raise RamdiskError("RAM-disk manifest has invalid benchmark recovery time")
    if record.get("containment_removed_at") and not record.get(
        "containment_removal_authorized_at"
    ):
        raise RamdiskError("RAM-disk manifest has invalid benchmark removal authority")
    if phase == "create-intent" and (
        record.get("containment_removal_authorized_at") is not None
        or record.get("containment_removed_at") is not None
    ):
        raise RamdiskError(
            "RAM-disk benchmark create intent has removal authority"
        )
    ordered_times = [record["created_at"]] + [
        record[name]
        for name in (
            "containment_removal_authorized_at",
            "containment_removed_at",
        )
        if record.get(name) is not None
    ]
    parsed_times = [
        datetime.datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
        for value in ordered_times
    ]
    if parsed_times != sorted(parsed_times):
        raise RamdiskError("RAM-disk manifest has invalid benchmark recovery order")
    if record.get("recovery_error") is not None and (
        not isinstance(record["recovery_error"], str)
        or not record["recovery_error"]
    ):
        raise RamdiskError("RAM-disk manifest has invalid benchmark recovery error")


def _validate_benchmark_workspace_v2(workspace, *, manifest, state_root):
    """Validate borrowed-deployment plus one-scratch recovery authority."""
    allowed_workspace = {
        "version",
        "operation_id",
        "protocol_id",
        "phase",
        "operation_path",
        "source_fingerprint",
        "size_bytes",
        "created_at",
        "roots",
    }
    allowed_with_process = allowed_workspace | {"pending_process"}
    deployment_id = manifest.get("deployment_id")
    plan = manifest.get("plan") or {}
    staged_bytes = (plan.get("staging") or {}).get("staged_bytes")
    expected_size = (
        max(staged_bytes + max(64 * MIB, staged_bytes // 100), 64 * MIB)
        if _positive_int(staged_bytes)
        else None
    )
    operation_id = workspace.get("operation_id")
    protocol_id = workspace.get("protocol_id")
    if (
        set(workspace) not in (allowed_workspace, allowed_with_process)
        or workspace.get("version") != 2
        or workspace.get("phase")
        not in ("pending", "mounted", "staged", "cleanup")
        or not isinstance(deployment_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", deployment_id) is None
        or not isinstance(operation_id, str)
        or re.fullmatch(r"benchmark:[0-9a-f]{32}", operation_id) is None
        or not isinstance(protocol_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", protocol_id) is None
        or not isinstance(workspace.get("source_fingerprint"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", workspace["source_fingerprint"]
        ) is None
        or workspace.get("source_fingerprint")
        != _benchmark_workspace_source_fingerprint(plan)
        or workspace.get("size_bytes") != expected_size
        or not _valid_utc_timestamp(workspace.get("created_at"))
    ):
        raise RamdiskError("RAM-disk manifest has invalid benchmark workspace")
    operation_path = os.path.join(
        state_root(),
        "benchmark-workspaces",
        deployment_id,
        operation_id.split(":", 1)[1],
    )
    if (
        workspace.get("operation_path") != operation_path
        or not os.path.isabs(operation_path)
        or os.path.normpath(operation_path) != operation_path
        or not _path_without_symlinks(operation_path)
    ):
        raise RamdiskError("RAM-disk manifest has invalid benchmark workspace path")
    roots_list = workspace.get("roots")
    if not isinstance(roots_list, list) or len(roots_list) != 2:
        raise RamdiskError("RAM-disk manifest has invalid benchmark workspace roots")
    roots = {
        root.get("name"): root
        for root in roots_list
        if isinstance(root, dict)
    }
    if set(roots) != {"interleaved", "local"}:
        raise RamdiskError("RAM-disk manifest has invalid benchmark workspace roots")
    deployment_name = (
        "interleaved" if plan.get("topology") == "interleaved" else "local"
        if plan.get("topology") == "per-node" else None
    )
    if deployment_name is None:
        raise RamdiskError("RAM-disk manifest has invalid benchmark topology")
    expected_specs = {
        "interleaved": ("interleave", [0, 1], None, "interleave=static:0-1"),
        "local": ("local", [0], 0, "bind=static:0"),
    }
    identity_keys = {
        "mount_id",
        "parent_id",
        "device",
        "root",
        "path",
        "options",
        "optional",
        "filesystem",
        "source",
        "super_options",
    }
    identities = []
    for name in ("interleaved", "local"):
        root = roots[name]
        role = "deployment" if name == deployment_name else "scratch"
        mode, nodes, node, policy = expected_specs[name]
        common_keys = {
            "name",
            "role",
            "operation_id",
            "path",
            "path_preexisting",
            "mode",
            "nodes",
            "node",
            "policy",
            "size_bytes",
            "source_fingerprint",
            "requested",
            "ownership",
            "stage_phase",
            "effective_thp",
            "effective_noswap",
            "identity",
            "numa_allocation",
            "staged_at",
        }
        scratch_only = {
            "helper_started_at",
            "helper_completed_at",
            "cleanup_authorized_at",
            "unmounted_at",
            "removed_at",
        }
        if (
            not set(root).issubset(common_keys | (scratch_only if role == "scratch" else set()))
            or root.get("name") != name
            or root.get("role") != role
            or root.get("mode") != mode
            or not isinstance(root.get("nodes"), list)
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in root["nodes"]
            )
            or root["nodes"] != nodes
            or (node is None and root.get("node") is not None)
            or (
                node is not None
                and (
                    isinstance(root.get("node"), bool)
                    or not isinstance(root.get("node"), int)
                    or root["node"] != node
                )
            )
            or root.get("policy") != policy
            or root.get("size_bytes") != expected_size
            or root.get("source_fingerprint") != workspace["source_fingerprint"]
        ):
            raise RamdiskError("RAM-disk manifest has invalid benchmark workspace root")
        requested = root.get("requested")
        mount_options = plan.get("mount_options")
        if (
            not isinstance(mount_options, dict)
            or not isinstance(mount_options.get("thp"), str)
            or mount_options["thp"] not in ("always", "advise", "never")
            or not isinstance(mount_options.get("noswap"), bool)
        ):
            raise RamdiskError("RAM-disk benchmark mount options are malformed")
        base_requested = {
            "filesystem": "tmpfs",
            "source": "tmpfs",
            "size_bytes": expected_size,
            "thp": mount_options["thp"],
            "noswap": mount_options["noswap"],
            "safety_options": [
                "noatime", "nodev", "nosuid", "noexec", "mode=0700"
            ],
            "policy": policy,
        }
        expected_requested = dict(base_requested)
        for key in ("effective_thp", "effective_noswap"):
            if key in root:
                expected_requested[key] = root[key]
        if (
            "effective_thp" in root
            and (
                not isinstance(root["effective_thp"], str)
                or root["effective_thp"]
                not in ("always", "advise", "never")
            )
        ) or (
            "effective_noswap" in root
            and not isinstance(root["effective_noswap"], bool)
        ):
            raise RamdiskError(
                "RAM-disk benchmark effective mount policy is malformed"
            )
        if requested != expected_requested:
            raise RamdiskError("RAM-disk manifest has invalid benchmark mount policy")
        identity = root.get("identity")
        exact_identity = (
            isinstance(identity, dict)
            and set(identity) in (identity_keys, identity_keys | {"all_options"})
            and _positive_int(identity.get("mount_id"))
            and _positive_int(identity.get("parent_id"))
            and isinstance(identity.get("device"), str)
            and re.fullmatch(r"[0-9]+:[0-9]+", identity["device"]) is not None
            and identity.get("root") == "/"
            and identity.get("path") == root.get("path")
            and identity.get("filesystem") == "tmpfs"
            and identity.get("source") == "tmpfs"
            and all(
                isinstance(identity.get(key), list)
                for key in ("options", "optional", "super_options")
            )
            and (
                "all_options" not in identity
                or isinstance(identity.get("all_options"), list)
            )
        )
        if role == "deployment":
            mount = next(
                (
                    item for item in manifest.get("mounts", [])
                    if isinstance(item, dict) and item.get("path") == root.get("path")
                ),
                None,
            )
            planned = next(
                (
                    item for item in plan.get("mounts", [])
                    if isinstance(item, dict) and item.get("path") == root.get("path")
                ),
                None,
            )
            if (
                mount is None
                or planned is None
                or mount.get("ownership", "managed") != "managed"
                or mount.get("identity") != identity
                or root.get("operation_id") != mount.get("operation_id")
                or not isinstance(root.get("operation_id"), str)
                or re.fullmatch(
                    re.escape(deployment_id) + r":mount:[0-9]+",
                    root["operation_id"],
                ) is None
                or root.get("path_preexisting") is not True
                or root.get("ownership") != "managed"
                or root.get("stage_phase") != "staged"
                or not exact_identity
                or not isinstance(root.get("numa_allocation"), dict)
                or not _valid_utc_timestamp(root.get("staged_at"))
                or (node is None and planned.get("node") is not None)
                or (
                    node is not None
                    and (
                        isinstance(planned.get("node"), bool)
                        or not isinstance(planned.get("node"), int)
                        or planned["node"] != node
                    )
                )
                or planned.get("policy") != policy
                or planned.get("size_bytes") != expected_size
            ):
                raise RamdiskError(
                    "RAM-disk manifest has invalid borrowed deployment root"
                )
        else:
            path = os.path.join(operation_path, name)
            ownership = root.get("ownership")
            stage_phase = root.get("stage_phase")
            if (
                root.get("operation_id") != operation_id
                or root.get("path") != path
                or not _path_without_symlinks(path)
                or root.get("path_preexisting") is not False
                or ownership not in ("pending", "identified", "managed")
                or stage_phase not in ("not-started", "pending", "staged")
                or (ownership == "pending" and identity is not None)
                or (ownership != "pending" and not exact_identity)
                or (stage_phase in ("pending", "staged") and ownership != "managed")
                or (
                    stage_phase == "staged"
                    and (
                        not isinstance(root.get("numa_allocation"), dict)
                        or not _valid_utc_timestamp(root.get("staged_at"))
                    )
                )
                or (
                    stage_phase != "staged"
                    and (
                        root.get("numa_allocation") is not None
                        or root.get("staged_at") is not None
                    )
                )
            ):
                raise RamdiskError("RAM-disk manifest has invalid scratch transition")
            if ownership != "pending" and (
                not _valid_utc_timestamp(root.get("helper_started_at"))
                or not _valid_utc_timestamp(root.get("helper_completed_at"))
                or not isinstance(root.get("effective_thp"), str)
                or not isinstance(root.get("effective_noswap"), bool)
            ):
                raise RamdiskError("RAM-disk manifest has invalid scratch helper state")
            if root.get("helper_completed_at") and not root.get("helper_started_at"):
                raise RamdiskError("RAM-disk manifest has invalid scratch helper state")
            for timestamp in (
                "helper_started_at", "helper_completed_at", "staged_at",
                "cleanup_authorized_at", "unmounted_at", "removed_at",
            ):
                if root.get(timestamp) is not None and not _valid_utc_timestamp(
                    root[timestamp]
                ):
                    raise RamdiskError("RAM-disk manifest has invalid scratch timestamp")
            if (
                root.get("unmounted_at") and not root.get("cleanup_authorized_at")
            ) or (
                root.get("removed_at") and not root.get("unmounted_at")
            ) or (
                any(root.get(key) for key in (
                    "cleanup_authorized_at", "unmounted_at", "removed_at"
                )) and workspace["phase"] != "cleanup"
            ):
                raise RamdiskError("RAM-disk manifest has invalid scratch cleanup state")
            ordered = [workspace["created_at"]] + [
                root[key]
                for key in (
                    "helper_started_at", "helper_completed_at", "staged_at",
                    "cleanup_authorized_at", "unmounted_at", "removed_at",
                )
                if root.get(key) is not None
            ]
            parsed = [
                datetime.datetime.fromisoformat(
                    value[:-1] + "+00:00" if value.endswith("Z") else value
                )
                for value in ordered
            ]
            if parsed != sorted(parsed):
                raise RamdiskError("RAM-disk manifest has invalid scratch transition order")
        if exact_identity:
            identities.append(identity)
    if len(identities) == 2 and (
        identities[0]["mount_id"] == identities[1]["mount_id"]
        or identities[0]["device"] == identities[1]["device"]
    ):
        raise RamdiskError("RAM-disk benchmark roots are not physically distinct")
    scratch = roots["local" if deployment_name == "interleaved" else "interleaved"]
    if workspace["phase"] == "pending" and scratch.get("stage_phase") != "not-started":
        raise RamdiskError("RAM-disk manifest has invalid benchmark pending phase")
    if workspace["phase"] in ("mounted", "staged") and scratch.get("ownership") != "managed":
        raise RamdiskError("RAM-disk manifest has invalid benchmark mount phase")
    if workspace["phase"] == "staged" and scratch.get("stage_phase") != "staged":
        raise RamdiskError("RAM-disk manifest has invalid benchmark staging phase")
    if workspace.get("pending_process") is not None:
        if workspace["phase"] != "staged":
            raise RamdiskError("RAM-disk benchmark process requires staged workspace")
        if manifest.get("processes") or manifest.get("pending_launches"):
            raise RamdiskError(
                "RAM-disk benchmark process conflicts with managed engines"
            )
        _validate_benchmark_pending_process(
            workspace["pending_process"],
            workspace=workspace,
            manifest=manifest,
            state_root=state_root,
        )
    return None


def _validate_benchmark_workspace_v1_unshipped(
    workspace, *, manifest, state_root
):
    """Historical validator for the unshipped two-scratch draft."""
    if not isinstance(workspace, dict):
        raise RamdiskError("RAM-disk manifest has invalid benchmark workspace")
    allowed_workspace = {
        "version",
        "operation_id",
        "protocol_id",
        "phase",
        "operation_path",
        "source_fingerprint",
        "size_bytes",
        "created_at",
        "roots",
    }
    deployment_id = manifest.get("deployment_id")
    if not isinstance(deployment_id, str) or re.fullmatch(
        r"[0-9a-f]{32}", deployment_id
    ) is None:
        raise RamdiskError(
            "RAM-disk benchmark workspace requires a deployment identity"
        )
    plan = manifest.get("plan") or {}
    staged_bytes = (plan.get("staging") or {}).get("staged_bytes")
    expected_size = (
        max(staged_bytes + max(64 * MIB, staged_bytes // 100), 64 * MIB)
        if _positive_int(staged_bytes)
        else None
    )
    if (
        set(workspace) != allowed_workspace
        or not isinstance(workspace.get("version"), int)
        or isinstance(workspace.get("version"), bool)
        or workspace.get("version") != 1
        or workspace.get("phase")
        not in ("pending", "mounted", "staged", "cleanup")
        or not re.fullmatch(
            r"benchmark:[0-9a-f]{32}",
            str(workspace.get("operation_id", "")),
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(workspace.get("protocol_id", "")),
        )
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(workspace.get("source_fingerprint", "")),
        )
        or workspace.get("size_bytes") != expected_size
        or not _valid_utc_timestamp(workspace.get("created_at"))
        or workspace.get("source_fingerprint")
        != _benchmark_workspace_source_fingerprint(plan)
    ):
        raise RamdiskError("RAM-disk manifest has invalid benchmark workspace")
    operation_path = workspace.get("operation_path")
    operation_suffix = workspace["operation_id"].split(":", 1)[1]
    expected_operation_path = os.path.join(
        state_root(),
        "benchmark-workspaces",
        deployment_id,
        operation_suffix,
    )
    if (
        not isinstance(operation_path, str)
        or not os.path.isabs(operation_path)
        or os.path.normpath(operation_path) != operation_path
        or operation_path != expected_operation_path
        or not _path_without_symlinks(operation_path)
    ):
        raise RamdiskError(
            "RAM-disk manifest has invalid benchmark workspace path"
        )
    roots = workspace.get("roots")
    if not isinstance(roots, list) or len(roots) != 2:
        raise RamdiskError(
            "RAM-disk manifest has invalid benchmark workspace roots"
        )
    expected = {
        "interleaved": ("interleave", [0, 1], "interleave=static:0-1"),
        "local": ("local", [0], "bind=static:0"),
    }
    seen_paths = set()
    for root in roots:
        if not isinstance(root, dict) or root.get("name") not in expected:
            raise RamdiskError(
                "RAM-disk manifest has invalid benchmark workspace root"
            )
        mode, nodes, policy = expected.pop(root["name"])
        path = root.get("path")
        requested = root.get("requested")
        requested_base_keys = {
            "filesystem",
            "source",
            "size_bytes",
            "thp",
            "noswap",
            "safety_options",
            "policy",
        }
        allowed_root = {
            "name",
            "path",
            "path_preexisting",
            "mode",
            "nodes",
            "node",
            "policy",
            "size_bytes",
            "source_fingerprint",
            "requested",
            "ownership",
            "stage_phase",
            "helper_started_at",
            "helper_completed_at",
            "effective_thp",
            "effective_noswap",
            "identity",
            "numa_allocation",
            "staged_at",
            "cleanup_authorized_at",
            "unmounted_at",
            "removed_at",
        }
        if (
            not set(root).issubset(allowed_root)
            or not isinstance(path, str)
            or not os.path.isabs(path)
            or os.path.normpath(path) != path
            or path != os.path.join(operation_path, root["name"])
            or not _path_without_symlinks(path)
            or path in seen_paths
            or root.get("mode") != mode
            or root.get("nodes") != nodes
            or root.get("policy") != policy
            or root.get("size_bytes") != workspace["size_bytes"]
            or root.get("source_fingerprint")
            != workspace["source_fingerprint"]
            or root.get("ownership")
            not in ("pending", "identified", "managed")
            or root.get("stage_phase")
            not in ("not-started", "pending", "staged")
            or not isinstance(requested, dict)
            or set(requested)
            not in (
                requested_base_keys,
                requested_base_keys | {"effective_thp", "effective_noswap"},
            )
            or requested.get("filesystem") != "tmpfs"
            or requested.get("source") != "tmpfs"
            or requested.get("size_bytes") != workspace["size_bytes"]
            or requested.get("policy") != policy
            or not isinstance(requested.get("thp"), str)
            or not isinstance(requested.get("noswap"), bool)
            or requested.get("safety_options")
            != ["noatime", "nodev", "nosuid", "noexec", "mode=0700"]
            or root.get("path_preexisting") is not False
            or root.get("node") != (None if root["name"] == "interleaved" else 0)
        ):
            raise RamdiskError(
                "RAM-disk manifest has invalid benchmark workspace root"
            )
        seen_paths.add(path)
        identity = root.get("identity")
        identity_base_keys = {
            "mount_id",
            "parent_id",
            "device",
            "root",
            "path",
            "options",
            "optional",
            "filesystem",
            "source",
            "super_options",
        }
        exact_identity = (
            isinstance(identity, dict)
            and set(identity)
            in (identity_base_keys, identity_base_keys | {"all_options"})
            and _positive_int(identity.get("mount_id"))
            and _positive_int(identity.get("parent_id"))
            and isinstance(identity.get("device"), str)
            and re.fullmatch(r"[0-9]+:[0-9]+", identity["device"]) is not None
            and identity.get("root") == "/"
            and identity.get("filesystem") == "tmpfs"
            and identity.get("source") == "tmpfs"
            and identity.get("path") == path
            and all(
                isinstance(identity.get(name), list)
                for name in ("options", "optional", "super_options")
            )
            and (
                "all_options" not in identity
                or isinstance(identity["all_options"], list)
            )
        )
        if (
            root["ownership"] == "pending" and identity is not None
        ) or (
            root["ownership"] in ("identified", "managed")
            and not exact_identity
        ):
            raise RamdiskError(
                "RAM-disk manifest has invalid benchmark mount identity"
            )
        if root["stage_phase"] == "staged" and (
            root["ownership"] != "managed"
            or not _valid_utc_timestamp(root.get("staged_at"))
            or not isinstance(root.get("numa_allocation"), dict)
        ):
            raise RamdiskError(
                "RAM-disk manifest has invalid benchmark staging state"
            )
        if root["stage_phase"] in ("pending", "staged") and (
            root["ownership"] != "managed"
        ):
            raise RamdiskError(
                "RAM-disk manifest has invalid benchmark staging state"
            )
        if root["stage_phase"] != "staged" and (
            root.get("staged_at") is not None
            or root.get("numa_allocation") is not None
        ):
            raise RamdiskError(
                "RAM-disk manifest has invalid benchmark staging state"
            )
        if root["ownership"] in ("identified", "managed") and (
            not root.get("helper_started_at")
            or not root.get("helper_completed_at")
            or not isinstance(root.get("effective_thp"), str)
            or not isinstance(root.get("effective_noswap"), bool)
        ):
            raise RamdiskError(
                "RAM-disk manifest has invalid benchmark helper state"
            )
        if root["ownership"] in ("identified", "managed") and (
            requested.get("effective_thp") != root.get("effective_thp")
            or requested.get("effective_noswap")
            is not root.get("effective_noswap")
        ):
            raise RamdiskError(
                "RAM-disk manifest has invalid benchmark effective policy"
            )
        for timestamp in (
            "helper_started_at",
            "helper_completed_at",
            "cleanup_authorized_at",
            "unmounted_at",
            "removed_at",
        ):
            if root.get(timestamp) is not None and not _valid_utc_timestamp(
                root[timestamp]
            ):
                raise RamdiskError(
                    "RAM-disk manifest has invalid benchmark recovery time"
                )
        if root.get("helper_completed_at") and not root.get("helper_started_at"):
            raise RamdiskError(
                "RAM-disk manifest has invalid benchmark helper state"
            )
        if root.get("unmounted_at") and not root.get("cleanup_authorized_at"):
            raise RamdiskError(
                "RAM-disk manifest has invalid benchmark cleanup state"
            )
        if any(
            root.get(name)
            for name in (
                "cleanup_authorized_at",
                "unmounted_at",
                "removed_at",
            )
        ) and workspace["phase"] != "cleanup":
            raise RamdiskError(
                "RAM-disk manifest has invalid benchmark cleanup state"
            )
        if root.get("removed_at") and not root.get("unmounted_at"):
            raise RamdiskError(
                "RAM-disk manifest has invalid benchmark cleanup state"
            )
        ordered_times = [workspace["created_at"]] + [
            root[name]
            for name in (
                "helper_started_at",
                "helper_completed_at",
                "staged_at",
                "cleanup_authorized_at",
                "unmounted_at",
                "removed_at",
            )
            if root.get(name) is not None
        ]
        parsed_times = [
            datetime.datetime.fromisoformat(
                value[:-1] + "+00:00" if value.endswith("Z") else value
            )
            for value in ordered_times
        ]
        if parsed_times != sorted(parsed_times):
            raise RamdiskError(
                "RAM-disk manifest has invalid benchmark transition order"
            )
    if expected:
        raise RamdiskError(
            "RAM-disk manifest has incomplete benchmark workspace roots"
        )
    if workspace["phase"] in ("mounted", "staged") and any(
        root.get("ownership") != "managed" for root in roots
    ):
        raise RamdiskError(
            "RAM-disk manifest has invalid benchmark mount phase"
        )
    if workspace["phase"] == "pending" and any(
        root.get("stage_phase") != "not-started" for root in roots
    ):
        raise RamdiskError(
            "RAM-disk manifest has invalid benchmark pending phase"
        )
    if workspace["phase"] == "staged" and any(
        root.get("stage_phase") != "staged" for root in roots
    ):
        raise RamdiskError(
            "RAM-disk manifest has invalid benchmark staging phase"
        )


def _validate_benchmark_workspace(workspace, *, manifest, state_root):
    """Validate the shipped borrowed-root workspace recovery authority."""
    if not isinstance(workspace, dict):
        raise RamdiskError("RAM-disk manifest has invalid benchmark workspace")
    if workspace.get("version") != 2:
        raise RamdiskError("RAM-disk manifest has unsupported benchmark workspace")
    return _validate_benchmark_workspace_v2(
        workspace,
        manifest=manifest,
        state_root=state_root,
    )


def _valid_usage_snapshot(value):
    if not isinstance(value, dict):
        return False
    for key, count in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(count, int)
            or isinstance(count, bool)
        ):
            return False
        if re.fullmatch(r"-[12]:[1-9]\d*", key):
            if count <= 0:
                return False
        elif re.fullmatch(r"\d+:\d+", key):
            if count < 0:
                return False
        else:
            return False
    return True


def _state_root():
    base = os.environ.get("XDG_STATE_HOME")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".local", "state")
    base = os.path.expanduser(base)
    if not os.path.isabs(base):
        raise RamdiskError("XDG_STATE_HOME must be an absolute durable path")
    return os.path.normpath(os.path.join(base, "colibri", "ramdisk"))


def _manifest_path():
    override = os.environ.get("COLI_RAMDISK_MANIFEST")
    if override:
        override = os.path.expanduser(override)
        if not os.path.isabs(override):
            raise RamdiskError("COLI_RAMDISK_MANIFEST must be an absolute durable path")
        return os.path.normpath(override)
    return os.path.join(_state_root(), "manifest.json")


def _benchmarks_path():
    return os.path.join(_state_root(), "benchmarks.json")


def _ensure_private_dir(path):
    path = os.path.normpath(path)
    if not _path_without_symlinks(path):
        raise RamdiskError("private state path contains a symlink: %s" % path)
    os.makedirs(path, mode=0o700, exist_ok=True)
    if not _path_without_symlinks(path):
        raise RamdiskError("private state path changed through a symlink: %s" % path)
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RamdiskError("private state path is not a real directory: %s" % path)
    os.chmod(path, 0o700)


def _assert_durable_state_dir(
    path,
    plan=None,
    *,
    filesystem_for_path=None,
):
    """Revalidate a derived engine/benchmark state directory before use."""
    path = os.path.normpath(path)
    if not _path_without_symlinks(path):
        raise RamdiskError("managed state path contains a symlink: %s" % path)
    if filesystem_for_path is None:
        raise RamdiskError("managed state filesystem validation is unavailable")
    if filesystem_for_path(path) in ("tmpfs", "ramfs"):
        raise RamdiskError("managed state path is on a volatile filesystem: %s" % path)
    if plan is not None:
        for mount in plan.get("mounts", []):
            weight_path = mount.get("path")
            if isinstance(weight_path, str) and _path_is_below(
                os.path.realpath(path),
                os.path.realpath(weight_path),
                allow_equal=True,
            ):
                raise RamdiskError(
                    "managed state path overlaps volatile weights: %s" % path
                )
    return path


def _ensure_atomic_parent(path):
    """Create a missing atomic-write parent without mutating an existing one."""
    if os.path.lexists(path):
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RamdiskError(
                "atomic-state parent is not a real directory: %s" % path
            )
        return
    try:
        os.makedirs(path, mode=0o700)
    except FileExistsError:
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RamdiskError(
                "atomic-state parent is not a real directory: %s" % path
            )
        return
    os.chmod(path, 0o700)


@contextlib.contextmanager
def _lifecycle_lock():
    """Serialize all manifest-changing operations for this invoking user."""
    depth = getattr(_lifecycle_local, "depth", 0)
    if depth:
        _lifecycle_local.depth = depth + 1
        try:
            yield
        finally:
            _lifecycle_local.depth -= 1
        return
    if fcntl is None or not get_platform_ops().is_linux:
        raise RamdiskError("RAM-disk lifecycle locking is supported only on Linux")
    root = _state_root()
    _ensure_private_dir(root)
    lock_path = os.path.join(root, "lifecycle.lock")
    with open(lock_path, "a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RamdiskError("another `coli ramdisk` lifecycle operation is active")
        _lifecycle_local.depth = 1
        try:
            yield
        finally:
            _lifecycle_local.depth = 0
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _close_preserving_primary(close):
    try:
        yield
    except BaseException:
        try:
            close()
        except BaseException:
            pass
        raise
    else:
        close()


@contextlib.contextmanager
def _fdopen_preserving_primary(descriptor, *args, **kwargs):
    try:
        stream = os.fdopen(descriptor, *args, **kwargs)
    except BaseException:
        try:
            os.close(descriptor)
        except BaseException:
            pass
        raise
    with _close_preserving_primary(stream.close):
        yield stream


def _stat_identity(info):
    return (info.st_dev, info.st_ino)


def _real_directory_info(path, source):
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise RamdiskError("%s directory is unavailable: %s" % (source, exc)) from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RamdiskError("%s parent is not a real directory: %s" % (source, path))
    return info


def _revalidate_bound_parent(bound):
    final = _real_directory_info(bound["parent"], bound["source"])
    if _stat_identity(final) != bound["identity"]:
        raise RamdiskError(
            "%s parent identity changed during access" % bound["source"]
        )
    validator = bound.get("validator")
    if validator is not None:
        validator()
        final = _real_directory_info(bound["parent"], bound["source"])
        if _stat_identity(final) != bound["identity"]:
            raise RamdiskError(
                "%s parent identity changed during validation" % bound["source"]
            )


@contextlib.contextmanager
def _bound_parent_descriptor(
    parent,
    *,
    source,
    validator=None,
    expected_identity=None,
    require_native=False,
):
    parent = os.path.normpath(parent)
    before = _real_directory_info(parent, source)
    before_identity = _stat_identity(before)
    if expected_identity is not None and before_identity != expected_identity:
        raise RamdiskError("%s parent identity changed before open" % source)
    if validator is not None:
        validator()
    if not _supports_native_dirfd():
        if require_native:
            raise RamdiskError(
                "%s requires descriptor-relative filesystem operations" % source
            )
        bound = {
            "descriptor": None,
            "identity": before_identity,
            "parent": parent,
            "native": False,
            "source": source,
            "validator": validator,
        }
        try:
            yield bound
        except BaseException:
            try:
                _revalidate_bound_parent(bound)
            except BaseException:
                pass
            raise
        else:
            _revalidate_bound_parent(bound)
        return

    flags = (
        os.O_RDONLY
        | int(getattr(os, "O_DIRECTORY"))
        | int(getattr(os, "O_NOFOLLOW"))
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(parent, flags)
    except OSError as exc:
        raise RamdiskError("cannot open %s parent safely: %s" % (source, exc)) from exc
    with _close_preserving_primary(lambda: os.close(descriptor)):
        opened = os.fstat(descriptor)
        after_open = _real_directory_info(parent, source)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _stat_identity(opened) != before_identity
            or _stat_identity(after_open) != before_identity
        ):
            raise RamdiskError("%s parent identity changed during verified open" % source)
        if validator is not None:
            validator()
        after_validation = _real_directory_info(parent, source)
        if _stat_identity(after_validation) != before_identity:
            raise RamdiskError("%s parent identity changed during validation" % source)
        bound = {
            "descriptor": descriptor,
            "identity": before_identity,
            "parent": parent,
            "native": True,
            "source": source,
            "validator": validator,
        }
        try:
            yield bound
        except BaseException:
            try:
                _revalidate_bound_parent(bound)
            except BaseException:
                pass
            raise
        else:
            _revalidate_bound_parent(bound)


def _target_info(bound, name):
    try:
        if bound["native"]:
            return os.stat(
                name,
                dir_fd=bound["descriptor"],
                follow_symlinks=False,
            )
        return os.lstat(os.path.join(bound["parent"], name))
    except FileNotFoundError:
        return None


def _require_regular_target(info, path, source, *, allow_missing):
    if info is None:
        if allow_missing:
            return
        raise RamdiskError(
            "%s must be an existing regular non-symlink file: %s"
            % (source, path)
        )
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RamdiskError(
            "%s must be a regular non-symlink file: %s" % (source, path)
        )


def _read_regular_text_from_bound_impl(
    bound,
    name,
    *,
    source,
    allow_missing,
    consumer=None,
):
    path = os.path.join(bound["parent"], name)
    before = _target_info(bound, name)
    _require_regular_target(before, path, source, allow_missing=allow_missing)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | int(getattr(os, "O_NONBLOCK", 0) or 0)
    )
    if isinstance(getattr(os, "O_NOFOLLOW", None), int):
        flags |= int(getattr(os, "O_NOFOLLOW"))
    try:
        if bound["native"]:
            descriptor = os.open(
                name,
                flags,
                dir_fd=bound["descriptor"],
            )
        else:
            descriptor = os.open(path, flags)
    except FileNotFoundError:
        if allow_missing and before is None:
            return {
                "text": "",
                "exists": False,
                "parent_identity": bound["identity"],
                "target_identity": None,
            }
        raise RamdiskError("%s changed before open: %s" % (source, path))
    except OSError:
        # The public reader wrapper normalizes ordinary I/O failures while
        # preserving RamdiskError and BaseException control flow.
        raise

    with _close_preserving_primary(lambda: os.close(descriptor)):
        opened = os.fstat(descriptor)
        _require_regular_target(opened, path, source, allow_missing=False)
        opened_identity = _stat_identity(opened)
        after_open = _target_info(bound, name)
        if (
            before is None
            or after_open is None
            or _stat_identity(before) != opened_identity
            or _stat_identity(after_open) != opened_identity
        ):
            raise RamdiskError("%s changed during verified open: %s" % (source, path))
        with _fdopen_preserving_primary(
            descriptor,
            "r",
            encoding="utf-8",
            errors="strict",
            closefd=False,
        ) as stream:
            try:
                text = stream.read()
                consumed = consumer(text) if consumer is not None else None
            except RamdiskError:
                raise
            except (OSError, UnicodeError) as exc:
                raise RamdiskError(
                    "cannot read %s: %s" % (source, exc)
                ) from exc
        final = _target_info(bound, name)
        if final is None or _stat_identity(final) != opened_identity:
            raise RamdiskError("%s changed during read: %s" % (source, path))
        snapshot = {
            "text": text,
            "exists": True,
            "parent_identity": bound["identity"],
            "target_identity": opened_identity,
        }
        if consumer is not None:
            snapshot["value"] = consumed
        return snapshot


def _read_regular_text_from_bound(
    bound,
    name,
    *,
    source,
    allow_missing,
    consumer=None,
):
    try:
        return _read_regular_text_from_bound_impl(
            bound,
            name,
            source=source,
            allow_missing=allow_missing,
            consumer=consumer,
        )
    except RamdiskError:
        raise
    except (OSError, UnicodeError) as exc:
        raise RamdiskError("cannot read %s: %s" % (source, exc)) from exc


def _read_bound_regular_text(
    path,
    *,
    source,
    allow_missing,
    validator=None,
    expected_parent_identity=None,
    require_native=False,
    consumer=None,
):
    parent = os.path.dirname(path)
    name = os.path.basename(path)
    if not name or os.path.join(parent, name) != path:
        raise RamdiskError("%s path is not normalized: %s" % (source, path))
    with _bound_parent_descriptor(
        parent,
        source=source,
        validator=validator,
        expected_identity=expected_parent_identity,
        require_native=require_native,
    ) as bound:
        return _read_regular_text_from_bound(
            bound,
            name,
            source=source,
            allow_missing=allow_missing,
            consumer=consumer,
        )


def _open_bound_temporary(bound, prefix, mode):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if bound["native"]:
        flags |= int(getattr(os, "O_NOFOLLOW"))
    for _ in range(128):
        name = prefix + secrets.token_hex(16)
        try:
            if bound["native"]:
                descriptor = os.open(
                    name,
                    flags,
                    mode,
                    dir_fd=bound["descriptor"],
                )
            else:
                descriptor = os.open(
                    os.path.join(bound["parent"], name),
                    flags,
                    mode,
                )
        except FileExistsError:
            continue
        return descriptor, name
    raise RamdiskError("could not allocate an atomic temporary file")


def _fsync_bound_directory(descriptor):
    os.fsync(descriptor)


def _atomic_replace_stream_from_bound(
    bound,
    name,
    *,
    source,
    prefix,
    mode,
    writer,
    expected_snapshot=None,
):
    path = os.path.join(bound["parent"], name)
    if (
        not name
        or os.path.basename(name) != name
        or os.path.normpath(path) != path
    ):
        raise RamdiskError("%s path is not normalized: %s" % (source, path))
    if isinstance(expected_snapshot, dict) and (
        expected_snapshot.get("parent_identity") != bound["identity"]
    ):
        raise RamdiskError("%s parent identity changed before replacement" % source)

    _revalidate_bound_parent(bound)
    initial = _target_info(bound, name)
    _require_regular_target(initial, path, source, allow_missing=True)
    initial_identity = _stat_identity(initial) if initial is not None else None
    if isinstance(expected_snapshot, dict) and (
        initial_identity != expected_snapshot.get("target_identity")
    ):
        raise RamdiskError("%s changed before atomic replacement" % source)

    descriptor, tmp_name = _open_bound_temporary(bound, prefix, mode)
    tmp_path = os.path.join(bound["parent"], tmp_name)
    try:
        opened_temp = os.fstat(descriptor)
        _require_regular_target(
            opened_temp,
            tmp_path,
            "atomic temporary file",
            allow_missing=False,
        )
        temp_identity = _stat_identity(opened_temp)
        visible_temp = _target_info(bound, tmp_name)
        if (
            visible_temp is None
            or _stat_identity(visible_temp) != temp_identity
        ):
            raise RamdiskError("atomic temporary file escaped its bound parent")
    except BaseException:
        try:
            os.close(descriptor)
        except BaseException:
            pass
        try:
            if bound["native"]:
                os.unlink(tmp_name, dir_fd=bound["descriptor"])
            else:
                os.unlink(tmp_path)
        except BaseException:
            pass
        raise

    replaced = False
    try:
        with _fdopen_preserving_primary(
            descriptor,
            "w",
            encoding="utf-8",
        ) as stream:
            writer(stream)
            stream.flush()
            if hasattr(os, "fchmod"):
                os.fchmod(stream.fileno(), mode)
            else:
                os.chmod(tmp_path, mode)
            os.fsync(stream.fileno())

        _revalidate_bound_parent(bound)
        visible_temp = _target_info(bound, tmp_name)
        if (
            visible_temp is None
            or _stat_identity(visible_temp) != temp_identity
        ):
            raise RamdiskError("atomic temporary file changed before replacement")
        current = _target_info(bound, name)
        _require_regular_target(current, path, source, allow_missing=True)
        current_identity = _stat_identity(current) if current is not None else None
        if current_identity != initial_identity:
            raise RamdiskError("%s changed before atomic replacement" % source)

        if bound["native"]:
            os.replace(
                tmp_name,
                name,
                src_dir_fd=bound["descriptor"],
                dst_dir_fd=bound["descriptor"],
            )
        else:
            os.replace(tmp_path, path)
        replaced = True

        committed = _target_info(bound, name)
        _require_regular_target(committed, path, source, allow_missing=False)
        if _stat_identity(committed) != temp_identity:
            raise RamdiskError("%s changed during atomic replacement" % source)
        _revalidate_bound_parent(bound)
        if bound["native"]:
            _fsync_bound_directory(bound["descriptor"])
        else:
            _fsync_directory(bound["parent"])
        return {
            "text": None,
            "exists": True,
            "parent_identity": bound["identity"],
            "target_identity": temp_identity,
        }
    except BaseException:
        if not replaced:
            try:
                if bound["native"]:
                    os.unlink(tmp_name, dir_fd=bound["descriptor"])
                else:
                    os.unlink(tmp_path)
            except BaseException:
                pass
        raise


def _atomic_replace_stream(
    path,
    *,
    prefix,
    mode,
    writer,
    expected_snapshot=None,
    validator=None,
    require_native=False,
    source=None,
):
    parent = os.path.dirname(path)
    name = os.path.basename(path)
    source = source or "atomic target %s" % path
    expected_parent = (
        expected_snapshot.get("parent_identity")
        if isinstance(expected_snapshot, dict)
        else None
    )
    with _bound_parent_descriptor(
        parent,
        source=source,
        validator=validator,
        expected_identity=expected_parent,
        require_native=require_native,
    ) as bound:
        return _atomic_replace_stream_from_bound(
            bound,
            name,
            source=source,
            prefix=prefix,
            mode=mode,
            writer=writer,
            expected_snapshot=expected_snapshot,
        )


def _atomic_json_from_bound(
    bound,
    name,
    value,
    *,
    mode=0o600,
    source="atomic JSON target",
    expected_snapshot=None,
):
    def write(stream):
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")

    return _atomic_replace_stream_from_bound(
        bound,
        name,
        source=source,
        prefix=".tmp-",
        mode=mode,
        writer=write,
        expected_snapshot=expected_snapshot,
    )


def _atomic_json(
    path,
    value,
    mode=0o600,
    *,
    expected_snapshot=None,
    validator=None,
    require_native=False,
):
    parent = os.path.dirname(path)
    _ensure_atomic_parent(parent)

    def write(stream):
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")

    _atomic_replace_stream(
        path,
        prefix=".tmp-",
        mode=mode,
        writer=write,
        expected_snapshot=expected_snapshot,
        validator=validator,
        require_native=require_native,
    )


def _read_json(path, required=False):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError:
        if required:
            raise RamdiskError("RAM-disk manifest not found; run `coli ramdisk prepare`")
        return None
    except (OSError, ValueError) as exc:
        raise RamdiskError("cannot read %s: %s" % (path, exc))


def _manifest_mount_layout(plan):
    """Validate and return the exact v1 mount layout encoded by a plan."""
    topology = plan.get("topology")
    root = plan.get("mount_root")
    planned = plan.get("mounts")
    model = plan.get("model", {}).get("path")
    if topology not in ("interleaved", "per-node") or not isinstance(root, str):
        raise RamdiskError("RAM-disk manifest has an invalid topology")
    root = posixpath.normpath(root)
    if (
        not posixpath.isabs(root)
        or posixpath.commonpath([root, "/mnt"]) != "/mnt"
        or root == "/mnt"
        or not _path_without_symlinks(root)
        or root in ("/mnt", DEFAULT_MOUNT_ROOT + "/..")
    ):
        raise RamdiskError("RAM-disk manifest has an unsafe mount root")
    try:
        normalized_model = str(model).replace("\\", "/")
        if re.match(r"^[A-Za-z]:/", normalized_model):
            common = None
        else:
            normalized_model = posixpath.normpath(normalized_model)
            common = posixpath.commonpath([root, normalized_model])
        if common in (root, normalized_model):
            raise RamdiskError(
                "RAM-disk manifest mount root overlaps its canonical model"
            )
    except (TypeError, ValueError):
        raise RamdiskError(
            "RAM-disk manifest has incompatible model and mount paths"
        )
    if not isinstance(planned, list) or not planned:
        raise RamdiskError("RAM-disk manifest has no planned mounts")
    planned_nodes = plan.get("placement", {}).get(
        "memory_nodes",
        plan.get("hardware", {}).get("online_nodes"),
    )
    if topology == "interleaved":
        expected = [(None, root)]
    else:
        if (
            not isinstance(planned_nodes, list)
            or not planned_nodes
            or any(
                not isinstance(node, int)
                or isinstance(node, bool)
                or node < 0
                for node in planned_nodes
            )
            or len(set(planned_nodes)) != len(planned_nodes)
        ):
            raise RamdiskError("RAM-disk manifest has an invalid NUMA node set")
        expected = [
            (node, posixpath.join(root, "node%d" % node))
            for node in planned_nodes
        ]
    observed = []
    for record in planned:
        if not isinstance(record, dict):
            raise RamdiskError(
                "RAM-disk manifest has invalid planned mount paths"
            )
        path = record.get("path")
        node = record.get("node")
        if (
            not isinstance(path, str)
            or not posixpath.isabs(path)
            or posixpath.normpath(path) != path
            or not _path_without_symlinks(path)
        ):
            raise RamdiskError(
                "RAM-disk manifest has invalid planned mount paths"
            )
        observed.append((node, path))
    if observed != expected:
        raise RamdiskError(
            "RAM-disk manifest mounts do not match its topology"
        )
    return root, expected


def _load_manifest(
    required=False,
    *,
    filesystem_for_path=None,
    read_json=None,
    manifest_path=None,
    state_root=None,
    benchmarks_path=None,
    assert_durable_state_dir=None,
    uid_provider=None,
):
    """Read and minimally validate lifecycle state before it can drive actions."""
    read_json = _read_json if read_json is None else read_json
    manifest_path = _manifest_path if manifest_path is None else manifest_path
    state_root = _state_root if state_root is None else state_root
    benchmarks_path = _benchmarks_path if benchmarks_path is None else benchmarks_path
    uid_provider = current_uid if uid_provider is None else uid_provider
    if assert_durable_state_dir is None:
        def assert_durable_state_dir(path, plan=None):
            return _assert_durable_state_dir(
                path,
                plan=plan,
                filesystem_for_path=filesystem_for_path,
            )

    manifest = read_json(manifest_path(), required=required)
    if manifest is None:
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != MANIFEST_VERSION
    ):
        raise RamdiskError("unsupported or malformed RAM-disk manifest")
    base_port = manifest.get("base_port")
    if base_port is not None and (
        isinstance(base_port, bool)
        or not isinstance(base_port, int)
        or not 1 <= base_port <= 65535
    ):
        raise RamdiskError("RAM-disk manifest has an invalid managed base port")
    deployment_id = manifest.get("deployment_id")
    if deployment_id is not None and (
        not isinstance(deployment_id, str)
        or not re.fullmatch(r"[0-9a-f]{32}", deployment_id)
    ):
        raise RamdiskError("RAM-disk manifest has an invalid deployment identity")
    process_supervision_version = manifest.get("process_supervision_version")
    if (
        process_supervision_version is not None
        and (
            not isinstance(process_supervision_version, int)
            or isinstance(process_supervision_version, bool)
            or process_supervision_version != CONTAINMENT_VERSION
        )
    ):
        raise RamdiskError(
            "RAM-disk manifest has an unsupported process supervision version"
        )
    if process_supervision_version is not None and deployment_id is None:
        raise RamdiskError(
            "RAM-disk manifest supervision is missing deployment identity"
        )
    plan = manifest.get("plan")
    mounts = manifest.get("mounts")
    processes = manifest.get("processes", [])
    if (
        not isinstance(plan, dict)
        or not isinstance(mounts, list)
        or not isinstance(processes, list)
    ):
        raise RamdiskError("RAM-disk manifest is missing lifecycle records")
    if (
        not isinstance(plan.get("model"), dict)
        or not isinstance(plan.get("mounts"), list)
    ):
        raise RamdiskError("RAM-disk manifest has an invalid plan")
    fingerprint = manifest.get("model_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint)
        or plan["model"].get("fingerprint") != fingerprint
        or not isinstance(plan["model"].get("path"), str)
        or not os.path.isabs(plan["model"]["path"])
        or not plan["mounts"]
    ):
        raise RamdiskError("RAM-disk manifest has an invalid model identity")
    if manifest.get("benchmark_workspace") is not None:
        _validate_benchmark_workspace(
            manifest["benchmark_workspace"],
            manifest=manifest,
            state_root=state_root,
        )
    _managed_accelerator_contract(plan)
    mount_root, _ = _manifest_mount_layout(plan)
    planned_paths = {record["path"] for record in plan["mounts"]}
    if len(planned_paths) != len(plan["mounts"]):
        raise RamdiskError("RAM-disk manifest has invalid planned mount paths")
    mount_by_path = {record["path"]: record for record in plan["mounts"]}
    mount_ownership = {}
    for record in mounts:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not os.path.isabs(record["path"])
        ):
            raise RamdiskError(
                "RAM-disk manifest contains an unsafe mount record"
            )
        if record["path"] not in planned_paths:
            raise RamdiskError(
                "RAM-disk manifest mount does not belong to its plan"
            )
        if record.get("node") != mount_by_path[record["path"]].get("node"):
            raise RamdiskError(
                "RAM-disk manifest mount has the wrong NUMA node"
            )
        identity = record.get("identity")
        identity_is_exact = (
            isinstance(identity, dict)
            and _positive_int(identity.get("mount_id"))
            and isinstance(identity.get("device"), str)
            and bool(identity["device"])
        )
        ownership = record.get("ownership")
        if ownership is None and identity_is_exact:
            # Manifests written before ownership transitions were explicit
            # contain only exact identities and are managed by definition.
            ownership = "managed"
        if ownership not in ("pending", "identified", "managed"):
            raise RamdiskError(
                "RAM-disk manifest mount has an invalid ownership state"
            )
        if ownership == "pending" and identity is not None:
            raise RamdiskError(
                "RAM-disk pending mount unexpectedly has an identity"
            )
        if ownership != "pending" and not identity_is_exact:
            raise RamdiskError(
                "RAM-disk manifest mount is missing its identity"
            )
        if record["path"] in mount_ownership:
            raise RamdiskError(
                "RAM-disk manifest contains duplicate mount records"
            )
        mount_ownership[record["path"]] = ownership
    state = manifest.get("state")
    if state not in (
        "preparing",
        "ready",
        "starting",
        "running",
        "stopped",
        "error",
    ):
        raise RamdiskError("RAM-disk manifest has an invalid lifecycle state")
    recovery = manifest.get("recovery")
    if recovery is not None and not isinstance(recovery, dict):
        raise RamdiskError("RAM-disk manifest has invalid recovery metadata")
    if isinstance(recovery, dict) and (
        recovery.get("operation")
        not in (None, "prepare", "start", "stop", "destroy")
        or recovery.get("state")
        not in (None, "attention-required", "clean", "reconciled")
    ):
        raise RamdiskError("RAM-disk manifest has invalid recovery metadata")
    retained_processes = (
        recovery.get("retained_processes", [])
        if isinstance(recovery, dict)
        else []
    )
    if not isinstance(retained_processes, list):
        raise RamdiskError(
            "RAM-disk manifest has invalid retained process recovery"
        )
    if retained_processes and state != "error":
        raise RamdiskError(
            "RAM-disk retained process recovery requires the error state"
        )
    if process_supervision_version is not None and retained_processes:
        raise RamdiskError(
            "versioned cgroup supervision cannot contain legacy retained "
            "process recovery"
        )
    pending_launches = manifest.get("pending_launches", [])
    if not isinstance(pending_launches, list):
        raise RamdiskError("RAM-disk manifest has invalid pending launches")
    if pending_launches and state not in ("starting", "error"):
        raise RamdiskError(
            "RAM-disk pending launches require starting or error state"
        )
    invoking_uid = uid_provider()
    pending_operation_ids = set()
    pending_nonces = set()
    usage_merge_ids = set()
    for pending in pending_launches:
        operation_id = (
            pending.get("operation_id") if isinstance(pending, dict) else None
        )
        nonce = pending.get("nonce") if isinstance(pending, dict) else None
        uid = pending.get("uid") if isinstance(pending, dict) else None
        port = pending.get("port") if isinstance(pending, dict) else None
        node = pending.get("node") if isinstance(pending, dict) else None
        state_dir = (
            pending.get("state_dir") if isinstance(pending, dict) else None
        )
        weights_dir = (
            pending.get("weights_dir") if isinstance(pending, dict) else None
        )
        launch_not_before = (
            pending.get("launch_not_before")
            if isinstance(pending, dict)
            else None
        )
        launcher_cmdline = (
            pending.get("launcher_cmdline")
            if isinstance(pending, dict)
            else None
        )
        launcher_pid = (
            pending.get("launcher_pid")
            if isinstance(pending, dict)
            else None
        )
        launcher_starttime = (
            pending.get("launcher_starttime")
            if isinstance(pending, dict)
            else None
        )
        expected_command = (
            pending.get("expected_command")
            if isinstance(pending, dict)
            else None
        )
        baseline = (
            pending.get("usage_baseline")
            if isinstance(pending, dict)
            else None
        )
        merge_id = (
            pending.get("usage_merge_id")
            if isinstance(pending, dict)
            else None
        )
        observed_group = (
            pending.get("observed_group")
            if isinstance(pending, dict)
            else None
        )
        usage_merged_at = (
            pending.get("usage_merged_at")
            if isinstance(pending, dict)
            else None
        )
        recovery_error = (
            pending.get("recovery_error")
            if isinstance(pending, dict)
            else None
        )
        if process_supervision_version is not None:
            _validate_supervised_containment(pending, pending=True)
        elif isinstance(pending, dict) and any(
            key in pending
            for key in (
                "containment",
                "containment_phase",
                "containment_removal_authorized_at",
                "containment_removed_at",
            )
        ):
            raise RamdiskError(
                "RAM-disk manifest has unversioned managed containment"
            )
        observed_pgid = (
            observed_group.get("pgid")
            if isinstance(observed_group, dict)
            else None
        )
        observed_uid = (
            observed_group.get("uid")
            if isinstance(observed_group, dict)
            else None
        )
        leader_starttime = (
            observed_group.get("leader_starttime")
            if isinstance(observed_group, dict)
            else None
        )
        planned_mount = next(
            (
                record
                for record in plan["mounts"]
                if record.get("node") == node
            ),
            None,
        )
        if (
            not isinstance(operation_id, str)
            or not re.fullmatch(r"start:[0-9a-f]{32}", operation_id)
            or not isinstance(nonce, str)
            or not re.fullmatch(r"[0-9a-f]{48}", nonce)
            or uid != invoking_uid
            or not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65535
            or node not in {record.get("node") for record in plan["mounts"]}
            or not isinstance(state_dir, str)
            or not os.path.isabs(state_dir)
            or planned_mount is None
            or weights_dir != planned_mount.get("path")
            or not isinstance(launch_not_before, int)
            or isinstance(launch_not_before, bool)
            or launch_not_before < 0
            or not _positive_int(launcher_pid)
            or not _positive_int(launcher_starttime)
            or not isinstance(launcher_cmdline, list)
            or not launcher_cmdline
            or any(
                not isinstance(item, str) or not item
                for item in launcher_cmdline
            )
            or not isinstance(expected_command, list)
            or not expected_command
            or any(
                not isinstance(item, str) or not item
                for item in expected_command
            )
            or not _valid_usage_snapshot(baseline)
            or not isinstance(merge_id, str)
            or not re.fullmatch(r"[0-9a-f]{32}", merge_id)
            or (
                usage_merged_at is not None
                and not _valid_utc_timestamp(usage_merged_at)
            )
            or (
                recovery_error is not None
                and (
                    not isinstance(recovery_error, str)
                    or not recovery_error
                )
            )
            or (
                observed_group is not None
                and (
                    not isinstance(observed_group, dict)
                    or not _positive_int(observed_pgid)
                    or observed_uid != uid
                    or (
                        leader_starttime is not None
                        and not _positive_int(leader_starttime)
                    )
                )
            )
        ):
            raise RamdiskError(
                "RAM-disk manifest has unsafe pending launch recovery"
            )
        if operation_id != "start:" + merge_id:
            raise RamdiskError(
                "RAM-disk manifest has unsafe pending launch recovery"
            )
        if operation_id in pending_operation_ids or nonce in pending_nonces:
            raise RamdiskError(
                "RAM-disk manifest has duplicate pending launch recovery"
            )
        if merge_id in usage_merge_ids:
            raise RamdiskError(
                "RAM-disk manifest has duplicate usage transaction authority"
            )
        pending_operation_ids.add(operation_id)
        pending_nonces.add(nonce)
        usage_merge_ids.add(merge_id)
    for retained in retained_processes:
        pid = retained.get("pid") if isinstance(retained, dict) else None
        pgid = retained.get("pgid") if isinstance(retained, dict) else None
        node = retained.get("node") if isinstance(retained, dict) else None
        state_dir = (
            retained.get("state_dir") if isinstance(retained, dict) else None
        )
        baseline = (
            retained.get("usage_baseline")
            if isinstance(retained, dict)
            else None
        )
        merge_id = (
            retained.get("usage_merge_id")
            if isinstance(retained, dict)
            else None
        )
        error = retained.get("error") if isinstance(retained, dict) else None
        if (
            not _positive_int(pid)
            or pgid != pid
            or node not in {record.get("node") for record in plan["mounts"]}
            or not isinstance(state_dir, str)
            or not os.path.isabs(state_dir)
            or not _valid_usage_snapshot(baseline)
            or not isinstance(merge_id, str)
            or not re.fullmatch(r"[0-9a-f]{32}", merge_id)
            or (
                error is not None
                and (not isinstance(error, str) or not error)
            )
        ):
            raise RamdiskError(
                "RAM-disk manifest has unsafe retained process recovery"
            )
        if merge_id in usage_merge_ids:
            raise RamdiskError(
                "RAM-disk manifest has duplicate usage transaction authority"
            )
        usage_merge_ids.add(merge_id)
    recorded_paths = {record["path"] for record in mounts}
    non_managed = sorted(
        path
        for path, ownership in mount_ownership.items()
        if ownership != "managed"
    )
    if state not in ("preparing", "error") and non_managed:
        raise RamdiskError(
            "RAM-disk manifest contains pending mount ownership outside "
            "preparation/recovery: %s" % ", ".join(non_managed)
        )
    if (
        state in ("ready", "starting", "running", "stopped")
        and recorded_paths != planned_paths
    ):
        raise RamdiskError(
            "ready RAM-disk manifest does not contain every planned mount"
        )
    if state == "running" and len(processes) != len(mounts):
        raise RamdiskError(
            "running RAM-disk manifest has an incomplete process set"
        )
    durable = plan.get("durable_state")
    expected_durable = {
        "root": state_root(),
        "manifest": manifest_path(),
        "benchmarks": benchmarks_path(),
    }
    if durable != expected_durable or any(
        not _path_without_symlinks(path)
        for path in expected_durable.values()
    ):
        raise RamdiskError(
            "RAM-disk manifest has an invalid durable-state identity"
        )
    if filesystem_for_path is None:
        raise RamdiskError("durable-state filesystem validation is unavailable")
    if any(
        filesystem_for_path(path) in ("tmpfs", "ramfs")
        for path in expected_durable.values()
    ):
        raise RamdiskError(
            "RAM-disk durable state is on a volatile filesystem"
        )
    if any(
        _path_is_below(path, mount_root, allow_equal=True)
        for path in expected_durable.values()
    ):
        raise RamdiskError(
            "RAM-disk durable state overlaps the volatile mount"
        )
    source_shards = plan.get("source_shards")
    if not isinstance(source_shards, list) or not source_shards:
        raise RamdiskError(
            "RAM-disk manifest is missing canonical shard identities"
        )
    fingerprint_dir = fingerprint.split(":", 1)[1]
    expected_state_root = os.path.join(
        state_root(),
        "engines",
        fingerprint_dir,
    )
    recovery_state_dirs = set()
    for recovery_record in pending_launches + retained_processes:
        node = recovery_record.get("node")
        label = "interleaved" if node is None else "node-%d" % node
        expected_recovery_state_dir = os.path.join(
            expected_state_root,
            label,
        )
        state_dir = recovery_record["state_dir"]
        if (
            state_dir != expected_recovery_state_dir
            or state_dir in recovery_state_dirs
        ):
            raise RamdiskError(
                "RAM-disk recovery has an unsafe or duplicate state directory"
            )
        assert_durable_state_dir(state_dir, plan=plan)
        recovery_state_dirs.add(state_dir)
    process_keys = {
        "pid": set(),
        "pgid": set(),
        "port": set(),
        "node": set(),
        "state_dir": set(recovery_state_dirs),
        "weights_dir": set(),
    }
    for record in processes:
        if not isinstance(record, dict):
            raise RamdiskError(
                "RAM-disk manifest contains an invalid process record"
            )
        pid, pgid = record.get("pid"), record.get("pgid")
        uid, starttime = record.get("uid"), record.get("starttime")
        nonce, port = record.get("nonce"), record.get("port")
        node, weights_dir = record.get("node"), record.get("weights_dir")
        state_dir, command = record.get("state_dir"), record.get("command")
        usage_baseline = record.get("usage_baseline")
        usage_merge_id = record.get("usage_merge_id")
        usage_merged_at = record.get("usage_merged_at")
        if process_supervision_version is not None:
            _validate_supervised_containment(record, pending=False)
        elif any(
            key in record
            for key in (
                "containment",
                "containment_removal_authorized_at",
                "containment_removed_at",
            )
        ):
            raise RamdiskError(
                "RAM-disk manifest has unversioned managed containment"
            )
        mount = next(
            (
                item
                for item in plan["mounts"]
                if item.get("node") == node
            ),
            None,
        )
        label = (
            "interleaved"
            if node is None
            else "node-%d" % node
            if isinstance(node, int)
            else ""
        )
        expected_state_dir = os.path.join(expected_state_root, label)
        valid_command = (
            isinstance(command, list)
            and command
            and all(isinstance(item, str) and item for item in command)
        )
        try:
            model_at = command.index("--model") if valid_command else -1
            port_at = command.index("--port") if valid_command else -1
            serves = "serve" in command
            command_matches = (
                serves
                and command[model_at + 1] == plan["model"]["path"]
                and int(command[port_at + 1]) == port
            )
        except (ValueError, IndexError, TypeError):
            command_matches = False
        if (
            not _positive_int(pid)
            or not _positive_int(pgid)
            or pid != pgid
            or uid != invoking_uid
            or not _positive_int(starttime)
            or not isinstance(nonce, str)
            or not re.fullmatch(r"[0-9a-f]{48}", nonce)
            or not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65535
            or mount is None
            or weights_dir != mount["path"]
            or state_dir != expected_state_dir
            or not command_matches
            or not _valid_usage_snapshot(usage_baseline)
            or (
                usage_merge_id is not None
                and (
                    not isinstance(usage_merge_id, str)
                    or not re.fullmatch(r"[0-9a-f]{32}", usage_merge_id)
                )
            )
            or (
                usage_merged_at is not None
                and (
                    not _valid_utc_timestamp(usage_merged_at)
                    or usage_merge_id is None
                )
            )
        ):
            raise RamdiskError(
                "RAM-disk manifest contains an unsafe managed process record"
            )
        if usage_merge_id is not None:
            if usage_merge_id in usage_merge_ids:
                raise RamdiskError(
                    "RAM-disk manifest has duplicate usage transaction authority"
                )
            usage_merge_ids.add(usage_merge_id)
        assert_durable_state_dir(state_dir, plan=plan)
        for key, value in (
            ("pid", pid),
            ("pgid", pgid),
            ("port", port),
            ("node", node),
            ("state_dir", state_dir),
            ("weights_dir", weights_dir),
        ):
            if value in process_keys[key]:
                raise RamdiskError(
                    "RAM-disk manifest contains duplicate managed process records"
                )
            process_keys[key].add(value)
    return manifest


def _save_manifest(
    manifest,
    *,
    atomic_json=None,
    manifest_path=None,
):
    atomic_json = _atomic_json if atomic_json is None else atomic_json
    manifest_path = _manifest_path if manifest_path is None else manifest_path
    manifest["updated_at"] = _utc_now()
    atomic_json(manifest_path(), manifest)


def _read_optional_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as stream:
            return stream.read()
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeError) as exc:
        raise RamdiskError("cannot read usage state %s: %s" % (path, exc))


def _usage_header(counts, source="usage history"):
    if not isinstance(counts, dict):
        raise RamdiskError("%s counts must contain an object" % source)
    records = []
    for key, value in counts.items():
        if not isinstance(key, str):
            continue
        match = re.fullmatch(r"(-?\d+):(\d+)", key)
        if not match:
            continue
        layer, second = (int(item) for item in match.groups())
        if layer in (-1, -2):
            try:
                records.append((layer, second, int(value)))
            except (TypeError, ValueError):
                raise RamdiskError("%s has a malformed usage header" % source)
    return _validated_usage_header(records, source=source)


def _validate_usage_for_plan(counts, plan, source="usage history"):
    """Validate identified usage metadata against the managed model."""
    header = _usage_header(counts, source=source)
    if header is None:
        return None
    try:
        model_path = plan["model"]["path"]
        with open(
            os.path.join(model_path, "config.json"),
            "r",
            encoding="utf-8",
        ) as stream:
            config = json.load(stream)
        if not isinstance(config, dict):
            raise ValueError("config root is not an object")
        dimensions = (
            int(config["num_hidden_layers"]),
            int(config["n_routed_experts"]),
        )
        engine_id = _usage_engine_id(
            _usage_engine_name(config.get("model_type"))
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise RamdiskError(
            "cannot validate %s against the managed model config: %s"
            % (source, exc)
        )
    return _validated_usage_header(
        [
            (-1, header["n_layers"], header["n_experts"]),
            (
                -2,
                header["format_version"],
                header["engine_id"],
            ),
        ],
        source=source,
        expected_dimensions=dimensions,
        expected_engine_id=engine_id,
    )


def _usage_header_counts(header):
    if not header:
        return {}
    return {
        "-1:%d" % header["n_layers"]: header["n_experts"],
        "-2:%d" % header["format_version"]: header["engine_id"],
    }


def _usage_data_counts(counts):
    result = {}
    for key, value in counts.items():
        if not isinstance(key, str):
            continue
        match = re.fullmatch(r"(\d+):(\d+)", key)
        if match:
            result[key] = int(value)
    return result


def _compatible_usage_header(*histories):
    reference = None
    reference_source = None
    for source, counts in histories:
        header = _usage_header(counts, source=source)
        if header is None:
            continue
        if reference is None:
            reference = header
            reference_source = source
            continue
        if (
            header["n_layers"],
            header["n_experts"],
        ) != (
            reference["n_layers"],
            reference["n_experts"],
        ):
            raise RamdiskError(
                "%s history dimensions do not match %s"
                % (source, reference_source)
            )
        if header["format_version"] != reference["format_version"]:
            raise RamdiskError(
                "%s usage format version does not match %s"
                % (source, reference_source)
            )
        if header["engine_id"] != reference["engine_id"]:
            raise RamdiskError(
                "%s engine identity does not match %s"
                % (source, reference_source)
            )
    return reference


def _usage_parse(text, source):
    counts = {}
    header_records = []
    for line in text.splitlines():
        match = PROFILE_LINE_RE.match(line)
        if not match:
            if re.match(r"^\s*-(?:1|2)(?:\s|$)", line):
                raise RamdiskError("%s has a malformed usage header" % source)
            continue
        layer, expert, count = (int(value) for value in match.groups())
        if layer in (-1, -2):
            header_records.append((layer, expert, count))
        elif layer >= 0:
            counts["%d:%d" % (layer, expert)] = count
    header = _validated_usage_header(header_records, source=source)
    counts.update(_usage_header_counts(header))
    return counts


def _usage_merge_ids_parse(text):
    result = set()
    for line in text.splitlines():
        match = USAGE_MERGE_RE.match(line.strip())
        if match:
            result.add(match.group(1))
    return result


def _usage_snapshot_from_bound(
    bound,
    name,
    path,
    *,
    source,
    allow_missing,
):
    def consume(text):
        return {
            "counts": _usage_parse(text, path),
            "merge_ids": _usage_merge_ids_parse(text),
        }

    snapshot = _read_regular_text_from_bound(
        bound,
        name,
        source=source,
        allow_missing=allow_missing,
        consumer=consume,
    )
    if snapshot["exists"]:
        snapshot.update(snapshot.pop("value"))
    else:
        snapshot["counts"] = {}
        snapshot["merge_ids"] = set()
    return snapshot


def _usage_snapshot(
    path,
    *,
    source=None,
    allow_missing=True,
    validator=None,
    expected_parent_identity=None,
    require_native=False,
):
    source = source or "usage state %s" % path
    def consume(text):
        return {
            "counts": _usage_parse(text, path),
            "merge_ids": _usage_merge_ids_parse(text),
        }

    snapshot = _read_bound_regular_text(
        path,
        source=source,
        allow_missing=allow_missing,
        validator=validator,
        expected_parent_identity=expected_parent_identity,
        require_native=require_native,
        consumer=consume,
    )
    if snapshot["exists"]:
        snapshot.update(snapshot.pop("value"))
    else:
        snapshot["counts"] = {}
        snapshot["merge_ids"] = set()
    return snapshot


def _usage_read(path):
    return _usage_snapshot(path)["counts"]


def _managed_usage_read(
    path,
    plan=None,
    *,
    filesystem_for_path=None,
):
    if os.path.basename(path) != ".coli_usage" or os.path.normpath(path) != path:
        raise RamdiskError("managed usage history path is not canonical: %s" % path)
    state_dir = os.path.dirname(path)
    validator = None
    if filesystem_for_path is not None:
        validator = lambda: _assert_durable_state_dir(
            state_dir,
            plan=plan,
            filesystem_for_path=filesystem_for_path,
        )
    return _usage_snapshot(
        path,
        source="managed usage history",
        allow_missing=False,
        validator=validator,
        require_native=True,
    )["counts"]


def _usage_merge_ids(path):
    return _usage_snapshot(path)["merge_ids"]


def _fsync_directory(path):
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    except BaseException:
        try:
            os.close(descriptor)
        except BaseException:
            # The fsync failure is the authoritative durability error.
            pass
        raise
    else:
        # When fsync succeeded, a close failure remains observable.
        os.close(descriptor)


def _durable_unlink(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    _fsync_directory(os.path.dirname(path))


def _durable_unlink_from_bound(
    bound,
    name,
    *,
    source,
    expected_snapshot=None,
):
    path = os.path.join(bound["parent"], name)
    _revalidate_bound_parent(bound)
    current = _target_info(bound, name)
    current_identity = _stat_identity(current) if current is not None else None
    if isinstance(expected_snapshot, dict) and (
        expected_snapshot.get("parent_identity") != bound["identity"]
        or expected_snapshot.get("target_identity") != current_identity
    ):
        raise RamdiskError("%s changed before durable unlink" % source)
    if current is not None:
        _require_regular_target(
            current,
            path,
            source,
            allow_missing=False,
        )
        if bound["native"]:
            os.unlink(name, dir_fd=bound["descriptor"])
        else:
            os.unlink(path)
    if bound["native"]:
        _fsync_bound_directory(bound["descriptor"])
    else:
        _fsync_directory(bound["parent"])
    _revalidate_bound_parent(bound)


def _usage_stream_writer(path, counts, merge_id=None, merge_ids=None):
    markers = set(merge_ids or ())
    if merge_id:
        markers.add(merge_id)
    header = _usage_header(counts, source=path)
    data_counts = _usage_data_counts(counts)

    def write(stream):
        # route_trace.h deliberately keeps an all-zero history zero-byte,
        # so emit identifying records only when there are data records.
        if header and data_counts:
            stream.write(
                "-1 %d %d\n"
                % (header["n_layers"], header["n_experts"])
            )
            stream.write(
                "-2 %d %d\n"
                % (header["format_version"], header["engine_id"])
            )
        for key in sorted(
            data_counts,
            key=lambda item: tuple(
                int(value)
                for value in item.split(":")
            ),
        ):
            layer, expert = key.split(":")
            stream.write(
                "%s %s %d\n" % (layer, expert, data_counts[key])
            )
        for marker in sorted(markers):
            stream.write("# coli-ramdisk-merge %s\n" % marker)

    return write


def _usage_write_from_bound(
    bound,
    name,
    path,
    counts,
    merge_id=None,
    merge_ids=None,
    *,
    expected_snapshot=None,
    source="usage state",
):
    return _atomic_replace_stream_from_bound(
        bound,
        name,
        source=source,
        prefix=".usage-",
        mode=0o600,
        writer=_usage_stream_writer(path, counts, merge_id, merge_ids),
        expected_snapshot=expected_snapshot,
    )


def _usage_write(
    path,
    counts,
    merge_id=None,
    merge_ids=None,
    *,
    expected_snapshot=None,
    validator=None,
    require_native=False,
):
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        raise RamdiskError(
            "usage-state parent directory is absent: %s" % parent
        )

    _atomic_replace_stream(
        path,
        prefix=".usage-",
        mode=0o600,
        writer=_usage_stream_writer(path, counts, merge_id, merge_ids),
        expected_snapshot=expected_snapshot,
        validator=validator,
        require_native=require_native,
    )


def _assert_canonical_usage_target(
    canonical_path,
    plan=None,
    *,
    source_still_matches=None,
):
    normalized = os.path.normpath(canonical_path)
    if normalized != canonical_path or os.path.basename(canonical_path) != ".coli_usage":
        raise RamdiskError(
            "canonical usage target is not an exact normalized .coli_usage path"
        )
    parent = os.path.dirname(normalized)
    if not os.path.isdir(parent):
        raise RamdiskError(
            "canonical model directory is absent; usage delta remains journaled"
        )
    if plan is not None:
        expected = os.path.normpath(
            os.path.join(plan["model"]["path"], ".coli_usage")
        )
        if normalized != expected:
            raise RamdiskError(
                "canonical usage target is not the managed model history"
            )
        if os.path.realpath(parent) != os.path.realpath(plan["model"]["path"]):
            raise RamdiskError(
                "canonical usage target no longer matches the managed model"
            )
        if source_still_matches is None:
            raise RamdiskError(
                "canonical source identity validation is unavailable"
            )
        source_still_matches(plan)


@contextlib.contextmanager
def _canonical_usage_descriptor(
    canonical_path,
    plan=None,
    *,
    source_still_matches=None,
):
    validator = lambda: _assert_canonical_usage_target(
        canonical_path,
        plan=plan,
        source_still_matches=source_still_matches,
    )
    with _bound_parent_descriptor(
        os.path.dirname(canonical_path),
        source="canonical usage target",
        validator=validator,
        require_native=True,
    ) as bound:
        yield bound


def _canonical_usage_snapshot(
    canonical_path,
    plan=None,
    *,
    source_still_matches=None,
):
    with _canonical_usage_descriptor(
        canonical_path,
        plan=plan,
        source_still_matches=source_still_matches,
    ) as bound:
        return _usage_snapshot_from_bound(
            bound,
            ".coli_usage",
            canonical_path,
            source="canonical usage target",
            allow_missing=True,
        )


def _canonical_usage_read(
    canonical_path,
    plan=None,
    *,
    source_still_matches=None,
):
    return _canonical_usage_snapshot(
        canonical_path,
        plan=plan,
        source_still_matches=source_still_matches,
    )["counts"]


def _managed_state_validator(
    state_dir,
    plan,
    filesystem_for_path,
):
    return lambda: _assert_durable_state_dir(
        state_dir,
        plan=plan,
        filesystem_for_path=filesystem_for_path,
    )


@contextlib.contextmanager
def _managed_state_descriptor(
    state_dir,
    plan=None,
    *,
    filesystem_for_path=None,
):
    validator = _managed_state_validator(
        state_dir,
        plan,
        filesystem_for_path,
    )
    with _bound_parent_descriptor(
        state_dir,
        source="managed state",
        validator=validator,
        require_native=True,
    ) as bound:
        yield bound


def _managed_usage_write(
    path,
    counts,
    plan=None,
    *,
    filesystem_for_path=None,
):
    if os.path.basename(path) != ".coli_usage" or os.path.normpath(path) != path:
        raise RamdiskError("managed usage history path is not canonical: %s" % path)
    state_dir = os.path.dirname(path)
    with _managed_state_descriptor(
        state_dir,
        plan=plan,
        filesystem_for_path=filesystem_for_path,
    ) as bound:
        snapshot = _usage_snapshot_from_bound(
            bound,
            ".coli_usage",
            path,
            source="managed usage history",
            allow_missing=True,
        )
        return _usage_write_from_bound(
            bound,
            ".coli_usage",
            path,
            counts,
            expected_snapshot=snapshot,
            source="managed usage history",
        )


def _journal_snapshot_from_bound(bound, state_dir):
    path = os.path.join(state_dir, ".coli_usage.delta.json")

    def consume(text):
        try:
            payload = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise RamdiskError("cannot read %s: %s" % (path, exc)) from exc
        if not isinstance(payload, dict):
            raise RamdiskError("usage delta journal must contain a JSON object")
        return payload

    return _read_regular_text_from_bound(
        bound,
        ".coli_usage.delta.json",
        source="usage delta journal",
        allow_missing=True,
        consumer=consume,
    )


def _managed_state_snapshots(
    state_dir,
    plan=None,
    *,
    filesystem_for_path=None,
    include_usage=False,
    include_journal=False,
):
    with _managed_state_descriptor(
        state_dir,
        plan=plan,
        filesystem_for_path=filesystem_for_path,
    ) as bound:
        result = {}
        if include_usage:
            usage_path = os.path.join(state_dir, ".coli_usage")
            result["usage"] = _usage_snapshot_from_bound(
                bound,
                ".coli_usage",
                usage_path,
                source="managed usage history",
                allow_missing=False,
            )
        if include_journal:
            result["journal"] = _journal_snapshot_from_bound(bound, state_dir)
        return result


def _journal_payload(snapshot, path):
    if not snapshot["exists"]:
        return None
    payload = snapshot.get("value")
    if not isinstance(payload, dict):
        raise RamdiskError("usage delta journal must contain a JSON object")
    return payload


def _read_usage_journal(
    state_dir,
    plan=None,
    *,
    filesystem_for_path=None,
):
    snapshots = _managed_state_snapshots(
        state_dir,
        plan=plan,
        filesystem_for_path=filesystem_for_path,
        include_journal=True,
    )
    path = os.path.join(state_dir, ".coli_usage.delta.json")
    return _journal_payload(snapshots["journal"], path)


def _journal_merge_id(payload):
    if not isinstance(payload, dict):
        raise RamdiskError("usage delta journal must contain a JSON object")
    merge_id = payload.get("id")
    if not merge_id:
        raise RamdiskError(
            "usage delta journal is missing its transaction id"
        )
    if (
        not isinstance(merge_id, str)
        or not re.fullmatch(r"[0-9a-f]{32}", merge_id)
    ):
        raise RamdiskError(
            "usage delta journal has an invalid transaction id"
        )
    return merge_id


def _bind_usage_transaction(
    record,
    plan=None,
    *,
    filesystem_for_path=None,
    reserved_ids=None,
):
    reserved_ids = set(reserved_ids or ())
    persisted = record.get("usage_merge_id")
    if persisted is not None and (
        not isinstance(persisted, str)
        or not re.fullmatch(r"[0-9a-f]{32}", persisted)
    ):
        raise RamdiskError(
            "managed usage recovery has an invalid transaction id"
        )
    payload = _read_usage_journal(
        record["state_dir"],
        plan=plan,
        filesystem_for_path=filesystem_for_path,
    )
    journal_id = _journal_merge_id(payload) if payload is not None else None
    if persisted is not None and journal_id is not None and persisted != journal_id:
        raise RamdiskError(
            "usage delta journal transaction does not match managed record"
        )
    authoritative = persisted or journal_id
    if authoritative is not None and authoritative in reserved_ids:
        raise RamdiskError(
            "duplicate usage transaction authority: %s" % authoritative
        )
    if authoritative is None:
        for _ in range(128):
            candidate = secrets.token_hex(16)
            if candidate not in reserved_ids:
                authoritative = candidate
                break
        else:
            raise RamdiskError("could not allocate a unique usage transaction id")
    merge_id = authoritative
    operation_id = record.get("operation_id")
    if operation_id is not None and operation_id != "start:" + merge_id:
        raise RamdiskError(
            "managed usage transaction does not match its operation authority"
        )
    record["usage_merge_id"] = merge_id
    return merge_id


@contextlib.contextmanager
def _usage_lock(lock):
    if fcntl is None:
        with _fallback_usage_lock:
            yield
        return
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _validated_usage_delta(payload, expected_merge_id=None):
    merge_id = _journal_merge_id(payload)
    delta = payload.get("delta", {})
    if not isinstance(delta, dict):
        raise RamdiskError("usage delta journal has invalid counts")
    if any(
        not isinstance(key, str)
        or re.fullmatch(r"\d+:\d+", key) is None
        or not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for key, value in delta.items()
    ):
        raise RamdiskError("usage delta journal has invalid counts")
    headers = payload.get("headers", {})
    if not isinstance(headers, dict):
        raise RamdiskError("usage delta journal has invalid headers")
    _compatible_usage_header(
        ("usage delta journal", headers),
    )
    if expected_merge_id is not None:
        if (
            not isinstance(expected_merge_id, str)
            or not re.fullmatch(r"[0-9a-f]{32}", expected_merge_id)
        ):
            raise RamdiskError(
                "managed usage recovery has an invalid expected transaction id"
            )
        if merge_id != expected_merge_id:
            raise RamdiskError(
                "usage delta journal transaction does not match managed record"
            )
    return merge_id, delta, headers


def _usage_journal_transaction_id(
    state_dir,
    plan=None,
    *,
    filesystem_for_path=None,
):
    """Return one validated live journal ID without applying its delta."""
    payload = _read_usage_journal(
        state_dir,
        plan=plan,
        filesystem_for_path=filesystem_for_path,
    )
    if payload is None:
        return None
    merge_id, _, _ = _validated_usage_delta(payload)
    return merge_id


def _apply_usage_delta(
    canonical_path,
    merge_id,
    delta,
    headers,
    plan=None,
    *,
    source_still_matches=None,
):
    lock_path = os.path.join(_state_root(), "usage.lock")
    _ensure_private_dir(os.path.dirname(lock_path))
    with open(lock_path, "a+", encoding="utf-8") as lock:
        with _usage_lock(lock):
            with _canonical_usage_descriptor(
                canonical_path,
                plan=plan,
                source_still_matches=source_still_matches,
            ) as canonical_bound:
                canonical_snapshot = _usage_snapshot_from_bound(
                    canonical_bound,
                    ".coli_usage",
                    canonical_path,
                    source="canonical usage target",
                    allow_missing=True,
                )
                applied = set(canonical_snapshot["merge_ids"])
                if merge_id in applied:
                    # A prior replace may have published this marker while its
                    # parent-directory fsync reported an uncertain outcome.
                    # Re-prove the canonical namespace before the caller is
                    # allowed to delete the last durable delta journal.
                    _revalidate_bound_parent(canonical_bound)
                    _fsync_bound_directory(canonical_bound["descriptor"])
                    _revalidate_bound_parent(canonical_bound)
                    return
                canonical = dict(canonical_snapshot["counts"])
                merged_header = _compatible_usage_header(
                    ("usage delta journal", headers),
                    ("canonical usage history", canonical),
                )
                for key, value in delta.items():
                    canonical[key] = canonical.get(key, 0) + value
                canonical.update(_usage_header_counts(merged_header))
                applied.add(merge_id)
                _usage_write_from_bound(
                    canonical_bound,
                    ".coli_usage",
                    canonical_path,
                    canonical,
                    merge_ids=applied,
                    expected_snapshot=canonical_snapshot,
                    source="canonical usage target",
                )


def _recover_delta_from_bound(
    state_bound,
    journal_snapshot,
    state_dir,
    canonical_path,
    plan=None,
    expected_merge_id=None,
    *,
    source_still_matches=None,
    keep_journal=False,
):
    delta_path = os.path.join(state_dir, ".coli_usage.delta.json")
    payload = _journal_payload(journal_snapshot, delta_path)
    if payload is None:
        if expected_merge_id is not None:
            if (
                not isinstance(expected_merge_id, str)
                or re.fullmatch(r"[0-9a-f]{32}", expected_merge_id) is None
            ):
                raise RamdiskError(
                    "managed usage recovery has an invalid expected transaction id"
                )
            raise RamdiskError(
                "expected usage delta journal is absent: %s" % delta_path
            )
        _durable_unlink_from_bound(
            state_bound,
            ".coli_usage.delta.json",
            source="usage delta journal",
            expected_snapshot=journal_snapshot,
        )
        return None
    merge_id, delta, headers = _validated_usage_delta(
        payload,
        expected_merge_id=expected_merge_id,
    )
    _revalidate_bound_parent(state_bound)
    _apply_usage_delta(
        canonical_path,
        merge_id,
        delta,
        headers,
        plan=plan,
        source_still_matches=source_still_matches,
    )
    _revalidate_bound_parent(state_bound)
    if not keep_journal:
        _durable_unlink_from_bound(
            state_bound,
            ".coli_usage.delta.json",
            source="usage delta journal",
            expected_snapshot=journal_snapshot,
        )
    return merge_id


def _recover_delta(
    state_dir,
    canonical_path,
    plan=None,
    expected_merge_id=None,
    *,
    filesystem_for_path=None,
    source_still_matches=None,
):
    with _managed_state_descriptor(
        state_dir,
        plan=plan,
        filesystem_for_path=filesystem_for_path,
    ) as state_bound:
        journal_snapshot = _journal_snapshot_from_bound(state_bound, state_dir)
        return _recover_delta_from_bound(
            state_bound,
            journal_snapshot,
            state_dir,
            canonical_path,
            plan=plan,
            expected_merge_id=expected_merge_id,
            source_still_matches=source_still_matches,
        )


def _merge_usage(
    record,
    canonical_path,
    plan=None,
    keep_journal=False,
    *,
    filesystem_for_path=None,
    source_still_matches=None,
):
    baseline = record.get("usage_baseline")
    if not _valid_usage_snapshot(baseline):
        raise RamdiskError(
            "managed usage recovery is missing a valid exact baseline"
        )
    persisted_merge_id = record.get("usage_merge_id")
    if persisted_merge_id is not None and (
        not isinstance(persisted_merge_id, str)
        or not re.fullmatch(r"[0-9a-f]{32}", persisted_merge_id)
    ):
        raise RamdiskError(
            "managed usage recovery has an invalid transaction id"
        )
    state_dir = record["state_dir"]
    state_usage = os.path.join(state_dir, ".coli_usage")
    delta_path = os.path.join(state_dir, ".coli_usage.delta.json")
    with _managed_state_descriptor(
        state_dir,
        plan=plan,
        filesystem_for_path=filesystem_for_path,
    ) as state_bound:
        usage_snapshot = _usage_snapshot_from_bound(
            state_bound,
            ".coli_usage",
            state_usage,
            source="managed usage history",
            allow_missing=False,
        )
        journal_snapshot = _journal_snapshot_from_bound(state_bound, state_dir)
        current = usage_snapshot["counts"]
        source_header = _compatible_usage_header(
            ("managed usage history", current),
            ("usage baseline", baseline),
        )
        current_counts = _usage_data_counts(current)
        baseline_counts = _usage_data_counts(baseline)
        for key, baseline_count in baseline_counts.items():
            if baseline_count <= 0:
                continue
            if key not in current_counts:
                raise RamdiskError(
                    "managed usage history is missing positive baseline counter %s"
                    % key
                )
            if current_counts[key] < baseline_count:
                raise RamdiskError(
                    "managed usage history counter %s regressed below baseline"
                    % key
                )
        delta = {
            key: value - baseline_counts.get(key, 0)
            for key, value in current_counts.items()
            if value > baseline_counts.get(key, 0)
        }

        payload = _journal_payload(journal_snapshot, delta_path)
        if payload is not None:
            journal_merge_id = _journal_merge_id(payload)
            if (
                persisted_merge_id is not None
                and persisted_merge_id != journal_merge_id
            ):
                raise RamdiskError(
                    "usage delta journal transaction does not match managed record"
                )
            record["usage_merge_id"] = journal_merge_id
            _recover_delta_from_bound(
                state_bound,
                journal_snapshot,
                state_dir,
                canonical_path,
                plan=plan,
                expected_merge_id=journal_merge_id,
                source_still_matches=source_still_matches,
                keep_journal=keep_journal,
            )
            return

        # Prove the absent journal in the same durable directory authority that
        # will create and later unlink this transaction's journal.
        _durable_unlink_from_bound(
            state_bound,
            ".coli_usage.delta.json",
            source="usage delta journal",
            expected_snapshot=journal_snapshot,
        )
        merge_id = persisted_merge_id or secrets.token_hex(16)
        record["usage_merge_id"] = merge_id

        lock_path = os.path.join(_state_root(), "usage.lock")
        _ensure_private_dir(os.path.dirname(lock_path))
        with open(lock_path, "a+", encoding="utf-8") as lock:
            with _usage_lock(lock):
                with _canonical_usage_descriptor(
                    canonical_path,
                    plan=plan,
                    source_still_matches=source_still_matches,
                ) as canonical_bound:
                    canonical_snapshot = _usage_snapshot_from_bound(
                        canonical_bound,
                        ".coli_usage",
                        canonical_path,
                        source="canonical usage target",
                        allow_missing=True,
                    )
                    if merge_id in canonical_snapshot["merge_ids"]:
                        return
                    if not delta:
                        canonical = dict(canonical_snapshot["counts"])
                        canonical_header = _compatible_usage_header(
                            ("managed usage history", current),
                            ("usage baseline", baseline),
                            ("canonical usage history", canonical),
                        )
                        if (
                            source_header is not None
                            and _usage_header(canonical) is None
                        ):
                            canonical.update(
                                _usage_header_counts(canonical_header)
                            )
                            _usage_write_from_bound(
                                canonical_bound,
                                ".coli_usage",
                                canonical_path,
                                canonical,
                                merge_ids=canonical_snapshot["merge_ids"],
                                expected_snapshot=canonical_snapshot,
                                source="canonical usage target",
                            )
                        return

        journal_payload = {
            "version": 1,
            "id": merge_id,
            "delta": delta,
            "headers": _usage_header_counts(source_header),
            "created_at": _utc_now(),
        }
        _revalidate_bound_parent(state_bound)
        journal_snapshot = _atomic_json_from_bound(
            state_bound,
            ".coli_usage.delta.json",
            journal_payload,
            source="usage delta journal",
            expected_snapshot=journal_snapshot,
        )
        journal_snapshot["value"] = journal_payload
        _recover_delta_from_bound(
            state_bound,
            journal_snapshot,
            state_dir,
            canonical_path,
            plan=plan,
            expected_merge_id=merge_id,
            source_still_matches=source_still_matches,
            keep_journal=keep_journal,
        )
