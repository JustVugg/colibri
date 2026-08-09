"""Deterministic causal RAMMAP experiments with append-only raw evidence.

This module deliberately does not contain the donor's tuning sweep.  Every
measured row represents one fresh engine process in a predeclared randomized
block.  Raw evidence is immutable JSONL and claims remain neutral unless all
correctness, measurement, repetition, and paired-interval gates pass.

DRAM traffic is supplied by an optional counter helper named by
``COLI_DRAM_COLLECTOR``.  The helper protocol is intentionally small: the
executable must accept ``--preflight --json`` and
``--snapshot --pid PID --json`` and emit cumulative byte counters.  When no
verified helper is available, experiments still preserve raw rows but the
claim is explicitly incomplete and neutral.
"""

from __future__ import print_function

import copy
import contextlib
import errno
import hashlib
import json
import math
import os
import queue
import random
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import threading
import time

from .common import (
    GIB,
    MANIFEST_VERSION,
    MIB,
    RamdiskError,
    _EngineCleanupError,
    _MountHelperCompletedError,
    _OperationCancelled,
    _format_range_list,
    _parse_range_list,
    _path_without_symlinks,
    _raise_if_cancelled,
    _utc_now,
)


BENCHMARK_SCHEMA = "colibri.ramdisk.causal-benchmark.v1"
PROTOCOL_SCHEMA = "colibri.ramdisk.causal-protocol.v1"
RAW_EVIDENCE_SCHEMA = "colibri.ramdisk.causal-raw.v1"
BENCHMARK_PROMPT = (
    "Explain in two sentences why deterministic validation matters."
)
MIN_REPETITIONS = 7
CI_METHOD = "paired-bootstrap-v1"
BOOTSTRAP_RESAMPLES = 10000

TREATMENT_IDS = (
    "anon-pin-interleaved",
    "anon-pin-local",
    "tmpfs-rammap-interleaved",
    "tmpfs-rammap-local",
    "ssd-slab-control",
    "tmpfs-slab-control",
    "cuda-fixed-budget-validation",
)


class WorkspaceCleanupError(RamdiskError):
    """A durable benchmark workspace could not be cleaned up safely."""


class WorkspaceVerificationError(RamdiskError):
    """A prepared workspace changed after its durable review."""


class UnavailableWorkspaceManager:
    """Fail-closed placeholder until lifecycle wires durable scratch staging."""

    def open(self, manifest, protocol, cancel_event):
        del manifest, protocol, cancel_event
        raise RamdiskError("durable causal benchmark workspace is unavailable")

    def verify(self, descriptor):
        del descriptor
        return False

_MEASURED_ENVIRONMENT_KEYS = (
    "AUTOPIN",
    "CAP_RAISE",
    "CACHE_ROUTE",
    "COLI_CPU_AFFINITY",
    "COLI_CUDA",
    "COLI_GPU",
    "COLI_GPUS",
    "COLI_KV_SLOTS",
    "COLI_MMAP",
    "COLI_NUMA",
    "COLI_NUMA_NODES",
    "COLI_POLICY",
    "COLI_RAMMAP",
    "COLI_RAM_PREFAULT",
    "COLI_STATE_DIR",
    "COLI_WEIGHTS_DIR",
    "CUDA_DENSE",
    "CUDA_EXPERT_GB",
    "CUDA_VISIBLE_DEVICES",
    "DRAFT",
    "CTX",
    "EXPERT_BUDGET",
    "KVSAVE",
    "KV_SLOTS",
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_PLACES",
    "OMP_PROC_BIND",
    "PIN",
    "PIN_FILL",
    "PIN_GB",
    "PILOT",
    "PILOT_REAL",
    "PREFETCH",
    "PROF",
    "REPIN",
    "TEMP",
    "TOPK",
    "TOPP",
)

_CLEARED_ENVIRONMENT_KEYS = set(_MEASURED_ENVIRONMENT_KEYS) | {
    "COLI_CUDA_ATTN",
    "COLI_CUDA_ATTN_SHARD",
    "COLI_CUDA_MTP",
    "COLI_NO_OMP_TUNE",
    "COLI_OMP_TUNED",
    "COLI_RAM_OVERCOMMIT",
    "CUDA_RELEASE_HOST",
    "DSA_TOPK",
    "DIRECT",
    "GRAMMAR",
    "PIPE",
    "PIPE_WORKERS",
    "RAM_GB",
    "URING",
}

_SAFE_INHERITED_ENVIRONMENT_KEYS = {
    "HOME",
    "LANG",
    "LD_LIBRARY_PATH",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TMPDIR",
    "TZ",
}


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _canonical_sha256(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _same_mount_identity(actual, expected):
    return bool(
        isinstance(actual, dict)
        and isinstance(expected, dict)
        and actual.get("filesystem") == "tmpfs"
        and actual.get("source") == "tmpfs"
        and actual.get("mount_id") == expected.get("mount_id")
        and actual.get("device") == expected.get("device")
    )


def _workspace_source_fingerprint(plan):
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
    return _canonical_sha256(
        {
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
    )


def _workspace_requirements(plan):
    """Freeze logical policy/capacity without physical attempt identities."""
    size_bytes = DurableWorkspaceManager._workspace_size(plan)
    source_fingerprint = _workspace_source_fingerprint(plan)
    topology = plan.get("topology")
    if topology not in ("interleaved", "per-node"):
        raise RamdiskError("causal benchmark workspace topology is invalid")
    deployment_name = "interleaved" if topology == "interleaved" else "local"
    roots = {}
    for name in ("interleaved", "local"):
        mode, nodes, node, policy = DurableWorkspaceManager._root_spec(name)
        roots[name] = {
            "role": "deployment" if name == deployment_name else "scratch",
            "mode": mode,
            "nodes": nodes,
            "node": node,
            "policy": policy,
            "size_bytes": size_bytes,
            "source_fingerprint": source_fingerprint,
        }
    return {
        "version": 1,
        "topology": topology,
        "size_bytes": size_bytes,
        "source_fingerprint": source_fingerprint,
        "roots": roots,
    }


class DurableWorkspaceManager:
    """Journal one opposite-policy scratch beside a prepared deployment root."""

    VERSION = 2
    PHASES = ("pending", "mounted", "staged", "cleanup")

    def __init__(
        self,
        *,
        load_manifest,
        save_manifest,
        state_root,
        ensure_private_dir,
        assert_durable_state_dir,
        mount_at,
        mount_table,
        path_is_below,
        busy_mount_references,
        mount_tmpfs,
        validate_mount,
        populate_mount,
        validate_namespace,
        source_still_matches,
        umount_path,
        available_for_mount=None,
    ):
        self._load_manifest = load_manifest
        self._save_manifest = save_manifest
        self._state_root = state_root
        self._ensure_private_dir = ensure_private_dir
        self._assert_durable_state_dir = assert_durable_state_dir
        self._mount_at = mount_at
        self._mount_table = mount_table
        self._path_is_below = path_is_below
        self._busy_mount_references = busy_mount_references
        self._mount_tmpfs = mount_tmpfs
        self._validate_mount = validate_mount
        self._populate_mount = populate_mount
        self._validate_namespace = validate_namespace
        self._source_still_matches = source_still_matches
        self._umount_path = umount_path
        self._available_for_mount = available_for_mount

    @staticmethod
    def _validate_nodes(plan):
        placement = (plan.get("placement") or {}).get("memory_nodes") or []
        online = (plan.get("hardware") or {}).get("online_nodes") or []
        rows = (plan.get("hardware") or {}).get("nodes") or []
        try:
            placement = {int(node) for node in placement}
            online = {int(node) for node in online}
            described = {
                int(row["id"])
                for row in rows
                if isinstance(row, dict) and "id" in row
            }
        except (TypeError, ValueError, KeyError) as error:
            raise RamdiskError(
                "causal workspace has malformed persisted NUMA hardware"
            ) from error
        required = {0, 1}
        if not required.issubset(placement & online & described):
            raise RamdiskError(
                "causal workspace requires persisted NUMA nodes 0 and 1"
            )

    @staticmethod
    def _workspace_size(plan):
        staged = (plan.get("staging") or {}).get("staged_bytes")
        if (
            not isinstance(staged, int)
            or isinstance(staged, bool)
            or staged <= 0
        ):
            raise RamdiskError(
                "causal workspace has no exact staged-byte capacity"
            )
        return max(staged + max(64 * MIB, staged // 100), 64 * MIB)

    @staticmethod
    def _root_plan(plan, root):
        result = copy.deepcopy(plan)
        result.setdefault("placement", {})["memory_nodes"] = list(root["nodes"])
        return result

    @staticmethod
    def _descriptor(root):
        keys = (
            "name",
            "role",
            "operation_id",
            "path",
            "mode",
            "nodes",
            "size_bytes",
            "policy",
            "source_fingerprint",
            "requested",
            "effective_thp",
            "effective_noswap",
            "identity",
        )
        result = {
            key: copy.deepcopy(root.get(key))
            for key in keys
        }
        result["verified"] = True
        return result

    @staticmethod
    def _root_spec(name):
        if name == "interleaved":
            return ("interleave", [0, 1], None, "interleave=static:0-1")
        if name == "local":
            return ("local", [0], 0, "bind=static:0")
        raise RamdiskError("causal benchmark workspace root name is invalid")

    @classmethod
    def _requested(cls, plan, *, name, size_bytes):
        _mode, _nodes, _node, policy = cls._root_spec(name)
        return {
            "filesystem": "tmpfs",
            "source": "tmpfs",
            "size_bytes": size_bytes,
            "thp": plan["mount_options"]["thp"],
            "noswap": bool(plan["mount_options"]["noswap"]),
            "safety_options": [
                "noatime",
                "nodev",
                "nosuid",
                "noexec",
                "mode=0700",
            ],
            "policy": policy,
        }

    def _prepared_root(self, manifest, *, source_fingerprint, size_bytes):
        """Return the exact prepared root that supplies one treatment policy."""
        plan = manifest["plan"]
        topology = plan.get("topology")
        if topology == "interleaved":
            name = "interleaved"
            candidates = [
                record for record in manifest.get("mounts", [])
                if isinstance(record, dict) and record.get("node") is None
            ]
        elif topology == "per-node":
            name = "local"
            candidates = [
                record for record in manifest.get("mounts", [])
                if isinstance(record, dict) and record.get("node") == 0
            ]
        else:
            raise RamdiskError("causal workspace has an unsupported topology")
        if len(candidates) != 1:
            raise RamdiskError(
                "causal workspace requires one exact prepared %s root" % name
            )
        record = candidates[0]
        planned = next(
            (
                item for item in plan.get("mounts", [])
                if isinstance(item, dict) and item.get("path") == record.get("path")
            ),
            None,
        )
        mode, nodes, node, policy = self._root_spec(name)
        identity = record.get("identity")
        if (
            planned is None
            or record.get("ownership", "managed") != "managed"
            or not isinstance(identity, dict)
            or isinstance(identity.get("mount_id"), bool)
            or not isinstance(identity.get("mount_id"), int)
            or identity.get("mount_id") <= 0
            or not isinstance(identity.get("device"), str)
            or not identity["device"]
            or planned.get("node") != node
            or planned.get("policy") != policy
            or planned.get("size_bytes") != size_bytes
            or not isinstance(record.get("operation_id"), str)
            or re.fullmatch(r"[0-9a-f]{32}:mount:[0-9]+", record["operation_id"])
            is None
        ):
            raise RamdiskError(
                "causal workspace prepared root does not match its reviewed plan"
            )
        requested = self._requested(plan, name=name, size_bytes=size_bytes)
        persisted_requested = record.get("requested")
        if persisted_requested is not None:
            for key, value in requested.items():
                if persisted_requested.get(key) != value:
                    raise RamdiskError(
                        "causal workspace prepared root changed reviewed options"
                    )
        root = {
            "name": name,
            "role": "deployment",
            "operation_id": record.get("operation_id"),
            "path": record["path"],
            "path_preexisting": True,
            "mode": mode,
            "nodes": nodes,
            "node": node,
            "policy": policy,
            "size_bytes": size_bytes,
            "source_fingerprint": source_fingerprint,
            "requested": requested,
            "ownership": "managed",
            "stage_phase": "staged",
            "identity": copy.deepcopy(identity),
            "numa_allocation": copy.deepcopy(record.get("numa_allocation") or {}),
            "staged_at": _utc_now(),
        }
        for key in ("effective_thp", "effective_noswap"):
            if key in record:
                root[key] = record[key]
                root["requested"][key] = record[key]
        self._source_still_matches(plan)
        self._verify_root(manifest, root, namespace=True)
        return root

    def _preflight_capacity(self, plan, scratch):
        if not callable(self._available_for_mount):
            raise RamdiskError(
                "causal workspace live capacity preflight is unavailable"
            )
        available = self._available_for_mount(scratch, plan=plan)
        reserve = plan.get("reserve") or {}
        protected = sum(
            int(reserve.get(name) or 0)
            for name in (
                "benchmark_runtime_bytes",
                "page_table_bytes",
                "os_margin_bytes",
            )
        )
        required = int((plan.get("staging") or {}).get("staged_bytes") or 0) + protected
        if (
            isinstance(available, bool)
            or not isinstance(available, int)
            or available < required
        ):
            raise RamdiskError(
                "causal workspace requires %d additional bytes but only %s are available"
                % (required, available)
            )

    @staticmethod
    def _root_by_descriptor(workspace, descriptor):
        if not isinstance(descriptor, dict):
            return None
        return next(
            (
                root
                for root in workspace.get("roots", [])
                if root.get("name") == descriptor.get("name")
                and root.get("path") == descriptor.get("path")
            ),
            None,
        )

    def _validate_workspace_record(self, manifest):
        workspace = manifest.get("benchmark_workspace")
        if not isinstance(workspace, dict):
            raise WorkspaceCleanupError(
                "durable benchmark workspace record is malformed"
            )
        if (
            workspace.get("version") != self.VERSION
            or workspace.get("phase") not in self.PHASES
            or not isinstance(workspace.get("roots"), list)
            or not workspace["roots"]
        ):
            raise WorkspaceCleanupError(
                "durable benchmark workspace record is malformed"
            )
        operation_id = workspace.get("operation_id")
        deployment_id = manifest.get("deployment_id")
        if (
            not isinstance(operation_id, str)
            or re.fullmatch(r"benchmark:[0-9a-f]{32}", operation_id) is None
            or not isinstance(deployment_id, str)
        ):
            raise WorkspaceCleanupError(
                "durable benchmark workspace operation is malformed"
            )
        expected_path = os.path.join(
            self._state_root(),
            "benchmark-workspaces",
            deployment_id,
            operation_id.split(":", 1)[1],
        )
        if (
            workspace.get("operation_path") != expected_path
            or not _path_without_symlinks(expected_path)
            or workspace.get("size_bytes") != self._workspace_size(manifest["plan"])
            or workspace.get("source_fingerprint")
            != _workspace_source_fingerprint(manifest["plan"])
        ):
            raise WorkspaceCleanupError(
                "durable benchmark workspace authority changed"
            )
        roots = {
            root.get("name"): root
            for root in workspace["roots"]
            if isinstance(root, dict)
        }
        if set(roots) != {"interleaved", "local"}:
            raise WorkspaceCleanupError(
                "durable benchmark workspace roots are malformed"
            )
        roles = {root.get("role") for root in roots.values()}
        if roles != {"deployment", "scratch"}:
            raise WorkspaceCleanupError(
                "durable benchmark workspace roles are malformed"
            )
        for name in ("interleaved", "local"):
            root = roots[name]
            mode, nodes, node, policy = self._root_spec(name)
            expected_requested = self._requested(
                manifest["plan"], name=name, size_bytes=workspace["size_bytes"]
            )
            for key in ("effective_thp", "effective_noswap"):
                if key in root:
                    expected_requested[key] = root[key]
            expected_root_path = (
                os.path.join(expected_path, name)
                if root.get("role") == "scratch"
                else root.get("path")
            )
            if (
                root.get("path") != expected_root_path
                or not _path_without_symlinks(root["path"])
                or root.get("mode") != mode
                or root.get("nodes") != nodes
                or root.get("node") != node
                or root.get("policy") != policy
                or root.get("size_bytes") != workspace["size_bytes"]
                or root.get("source_fingerprint")
                != workspace["source_fingerprint"]
                or root.get("requested") != expected_requested
            ):
                raise WorkspaceCleanupError(
                    "durable benchmark workspace root authority changed"
                )
        deployment = next(
            root for root in roots.values() if root["role"] == "deployment"
        )
        expected_deployment_name = (
            "interleaved"
            if manifest["plan"].get("topology") == "interleaved"
            else "local"
        )
        mount = next(
            (
                record for record in manifest.get("mounts", [])
                if isinstance(record, dict) and record.get("path") == deployment["path"]
            ),
            None,
        )
        if (
            deployment["name"] != expected_deployment_name
            or mount is None
            or mount.get("ownership", "managed") != "managed"
            or not _same_mount_identity(
                mount.get("identity"), deployment.get("identity")
            )
            or deployment.get("operation_id") != mount.get("operation_id")
            or deployment.get("path_preexisting") is not True
        ):
            raise WorkspaceCleanupError(
                "durable benchmark borrowed deployment authority changed"
            )
        scratch = next(root for root in roots.values() if root["role"] == "scratch")
        if (
            scratch["name"] == deployment["name"]
            or scratch.get("path_preexisting") is not False
            or scratch.get("operation_id") != workspace["operation_id"]
        ):
            raise WorkspaceCleanupError(
                "durable benchmark scratch authority changed"
            )
        return workspace

    def _verify_root(self, manifest, root, descriptor=None, *, namespace=False):
        plan = manifest["plan"]
        if root.get("source_fingerprint") != _workspace_source_fingerprint(plan):
            raise WorkspaceVerificationError(
                "causal benchmark source fingerprint changed"
            )
        if descriptor is not None:
            expected = self._descriptor(root)
            for key, value in expected.items():
                if descriptor.get(key) != value:
                    raise WorkspaceVerificationError(
                        "causal benchmark workspace descriptor changed"
                    )
        actual = self._mount_at(root["path"])
        if not _same_mount_identity(actual, root.get("identity")):
            raise WorkspaceVerificationError(
                "causal benchmark workspace identity changed"
            )
        validated = self._validate_mount(root, self._root_plan(plan, root))
        if not _same_mount_identity(validated, root.get("identity")):
            raise WorkspaceVerificationError(
                "causal benchmark workspace policy identity changed"
            )
        if namespace:
            self._validate_namespace(
                self._root_plan(plan, root),
                root,
                sample_numa=False,
            )
        return True

    def verify(self, descriptor):
        """Rebind a public descriptor to the durable record and live mount."""
        try:
            manifest = self._load_manifest(required=True)
            workspace = self._validate_workspace_record(manifest)
            if workspace.get("phase") != "staged":
                return False
            root = self._root_by_descriptor(workspace, descriptor)
            if root is None or root.get("stage_phase") != "staged":
                return False
            self._source_still_matches(manifest["plan"])
            return self._verify_root(
                manifest,
                root,
                descriptor,
                namespace=True,
            )
        except (RamdiskError, OSError, ValueError, TypeError):
            return False

    def bind_protocol(self, protocol):
        """Prove the physical attempt belongs to the stable logical protocol."""
        protocol_id = protocol.get("protocol_id") if isinstance(protocol, dict) else None
        if not isinstance(protocol_id, str) or re.fullmatch(
            r"[0-9a-f]{64}", protocol_id
        ) is None:
            raise RamdiskError("causal benchmark protocol identity is invalid")
        manifest = self._load_manifest(required=True)
        workspace = self._validate_workspace_record(manifest)
        if workspace.get("phase") != "staged":
            raise RamdiskError(
                "causal benchmark workspace is not staged for protocol binding"
            )
        if workspace.get("protocol_id") != protocol_id:
            raise RamdiskError(
                "causal benchmark workspace belongs to a different protocol"
            )
        return protocol_id

    def _preflight_cleanup_root(self, manifest, workspace, root, table):
        path = root.get("path")
        actual = self._mount_at(path)
        started = root.get("helper_started_at")
        completed = root.get("helper_completed_at")
        ownership = root.get("ownership")
        authorized = root.get("cleanup_authorized_at")
        if ownership == "pending":
            if actual is not None:
                raise WorkspaceCleanupError(
                    "refusing unverified pending benchmark mount at %s" % path
                )
            if started and not completed:
                raise WorkspaceCleanupError(
                    "benchmark mount helper outcome is unknown at %s" % path
                )
            return
        if actual is None:
            if not authorized:
                raise WorkspaceCleanupError(
                    "benchmark workspace disappeared before durable cleanup "
                    "authority at %s" % path
                )
            return
        self._verify_root(manifest, root)
        nested = sorted(
            item.get("path")
            for item in table
            if isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and self._path_is_below(item["path"], path)
        )
        if nested:
            raise WorkspaceCleanupError(
                "benchmark workspace has nested mounts at %s: %s"
                % (path, ", ".join(nested))
            )
        busy = self._busy_mount_references(
            path,
            hardware=manifest["plan"]["hardware"],
        )
        if busy:
            raise WorkspaceCleanupError(
                "benchmark workspace is busy at %s in PID(s): %s"
                % (path, ",".join(str(pid) for pid in busy))
            )

    def _remove_mountpoint(self, path):
        try:
            os.rmdir(path)
        except FileNotFoundError:
            return
        except OSError as error:
            if error.errno in (errno.ENOTEMPTY, errno.EBUSY):
                raise WorkspaceCleanupError(
                    "benchmark workspace mountpoint is not empty: %s" % path
                ) from error
            raise WorkspaceCleanupError(
                "cannot remove benchmark workspace mountpoint %s: %s"
                % (path, error)
            ) from error

    def recover(self, manifest=None):
        """Complete or refuse one durable workspace cleanup state machine."""
        manifest = (
            self._load_manifest(required=True)
            if manifest is None
            else manifest
        )
        if manifest.get("benchmark_workspace") is None:
            return True
        workspace = self._validate_workspace_record(manifest)
        scratch_roots = [
            root for root in workspace["roots"] if root.get("role") == "scratch"
        ]
        if len(scratch_roots) != 1:
            raise WorkspaceCleanupError(
                "durable benchmark scratch recovery authority is malformed"
            )
        if workspace.get("phase") != "cleanup":
            workspace["phase"] = "cleanup"
            self._save_manifest(manifest)
        try:
            table = self._mount_table()
            for root in scratch_roots:
                self._preflight_cleanup_root(manifest, workspace, root, table)
            now = _utc_now()
            for root in scratch_roots:
                root.setdefault("cleanup_authorized_at", now)
            # The borrowed deployment root is deliberately absent from this
            # teardown authority.  Only the one journaled scratch may unmount.
            self._save_manifest(manifest)

            for root in reversed(scratch_roots):
                path = root["path"]
                actual = self._mount_at(path)
                if actual is not None:
                    latest_table = self._mount_table()
                    self._preflight_cleanup_root(
                        manifest,
                        workspace,
                        root,
                        latest_table,
                    )
                    self._umount_path(path, manifest["plan"]["hardware"])
                    if self._mount_at(path) is not None:
                        raise WorkspaceCleanupError(
                            "benchmark workspace remains mounted at %s" % path
                        )
                root.setdefault("unmounted_at", _utc_now())
                self._save_manifest(manifest)
                self._remove_mountpoint(path)
                root.setdefault("removed_at", _utc_now())
                self._save_manifest(manifest)

            operation_path = workspace.get("operation_path")
            if isinstance(operation_path, str):
                self._remove_mountpoint(operation_path)
            manifest.pop("benchmark_workspace", None)
            self._save_manifest(manifest)
            return True
        except WorkspaceCleanupError:
            raise
        except BaseException as error:
            raise WorkspaceCleanupError(str(error)) from error

    def _prepare(self, manifest, protocol, cancel_event):
        if manifest.get("benchmark_workspace") is not None:
            self.recover(manifest)
        plan = manifest["plan"]
        self._validate_nodes(plan)
        size_bytes = self._workspace_size(plan)
        source_fingerprint = _workspace_source_fingerprint(plan)
        deployment = self._prepared_root(
            manifest,
            source_fingerprint=source_fingerprint,
            size_bytes=size_bytes,
        )
        scratch_name = (
            "local" if deployment["name"] == "interleaved" else "interleaved"
        )
        operation_id = "benchmark:%s" % secrets.token_hex(16)
        operation_path = os.path.join(
            self._state_root(),
            "benchmark-workspaces",
            manifest["deployment_id"],
            operation_id.split(":", 1)[1],
        )
        mode, nodes, node, policy = self._root_spec(scratch_name)
        scratch = {
            "name": scratch_name,
            "role": "scratch",
            "operation_id": operation_id,
            "path": os.path.join(operation_path, scratch_name),
            "path_preexisting": False,
            "mode": mode,
            "nodes": nodes,
            "node": node,
            "policy": policy,
            "size_bytes": size_bytes,
            "source_fingerprint": source_fingerprint,
            "requested": self._requested(
                plan, name=scratch_name, size_bytes=size_bytes
            ),
            "ownership": "pending",
            "stage_phase": "not-started",
        }
        # This is a live, current-capacity admission.  It happens before any
        # mount/staging journal exists and counts only the one additional copy.
        self._preflight_capacity(plan, scratch)
        self._ensure_private_dir(operation_path)
        self._assert_durable_state_dir(operation_path, plan=plan)
        self._ensure_private_dir(scratch["path"])
        self._assert_durable_state_dir(scratch["path"], plan=plan)
        roots = [deployment, scratch]
        workspace = {
            "version": self.VERSION,
            "operation_id": operation_id,
            "protocol_id": protocol.get("protocol_id"),
            "phase": "pending",
            "operation_path": operation_path,
            "source_fingerprint": source_fingerprint,
            "size_bytes": size_bytes,
            "created_at": _utc_now(),
            "roots": roots,
        }
        manifest["benchmark_workspace"] = workspace
        self._save_manifest(manifest)

        _raise_if_cancelled(cancel_event)
        if self._mount_at(scratch["path"]) is not None:
            raise RamdiskError(
                "refusing already-mounted benchmark workspace: %s"
                % scratch["path"]
            )
        scratch["helper_started_at"] = _utc_now()
        self._save_manifest(manifest)
        try:
            self._mount_tmpfs(self._root_plan(plan, scratch), scratch)
        except _MountHelperCompletedError:
            scratch["helper_completed_at"] = _utc_now()
            self._save_manifest(manifest)
            raise
        scratch["helper_completed_at"] = _utc_now()
        for key in ("effective_thp", "effective_noswap"):
            if key in scratch:
                scratch["requested"][key] = scratch[key]
        self._save_manifest(manifest)
        actual = self._mount_at(scratch["path"])
        if actual is None:
            raise RamdiskError(
                "benchmark mount completed without a readable identity"
            )
        scratch["identity"] = copy.deepcopy(actual)
        scratch["ownership"] = "identified"
        self._save_manifest(manifest)
        validated = self._validate_mount(scratch, self._root_plan(plan, scratch))
        if not _same_mount_identity(validated, actual):
            raise RamdiskError(
                "benchmark workspace changed while validating %s"
                % scratch["path"]
            )
        scratch["identity"] = copy.deepcopy(validated)
        scratch["ownership"] = "managed"
        self._save_manifest(manifest)

        first = deployment["identity"]
        second = scratch["identity"]
        if (
            first.get("mount_id") == second.get("mount_id")
            or first.get("device") == second.get("device")
        ):
            raise RamdiskError(
                "causal benchmark roots require distinct devices and mount identities"
            )

        workspace["phase"] = "mounted"
        self._save_manifest(manifest)
        _raise_if_cancelled(cancel_event)
        scratch["stage_phase"] = "pending"
        self._save_manifest(manifest)
        self._populate_mount(
            self._root_plan(plan, scratch),
            scratch,
            source_root=deployment["path"],
            progress=None,
            cancel_event=cancel_event,
        )
        scratch["numa_allocation"] = self._validate_namespace(
            self._root_plan(plan, scratch),
            scratch,
        )
        self._source_still_matches(plan)
        if scratch["source_fingerprint"] != _workspace_source_fingerprint(plan):
            raise RamdiskError("benchmark source changed while staging")
        scratch["stage_phase"] = "staged"
        scratch["staged_at"] = _utc_now()
        self._save_manifest(manifest)
        workspace["phase"] = "staged"
        self._save_manifest(manifest)
        return {
            root["name"]: self._descriptor(root)
            for root in roots
        }

    @contextlib.contextmanager
    def open(self, manifest, protocol, cancel_event):
        try:
            roots = self._prepare(manifest, protocol, cancel_event)
            yield roots
        except BaseException as primary:
            try:
                self.recover(manifest)
            except BaseException as cleanup:
                raise WorkspaceCleanupError(
                    "%s; benchmark workspace cleanup failed: %s"
                    % (primary, cleanup)
                ) from primary
            raise
        else:
            self.recover(manifest)


def _causal_protocol_identity(protocol):
    projection = dict(protocol)
    projection.pop("created_at", None)
    projection.pop("protocol_id", None)
    return _canonical_sha256(projection)


def _finite_number(value, label, *, positive=False, minimum=None, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RamdiskError("%s must be a finite numeric value" % label)
    parsed = float(value)
    if positive and parsed <= 0:
        raise RamdiskError("%s must be positive" % label)
    if minimum is not None and parsed < minimum:
        raise RamdiskError("%s must be at least %s" % (label, minimum))
    if maximum is not None and parsed > maximum:
        raise RamdiskError("%s must be at most %s" % (label, maximum))
    return parsed


def _numeric_gib(value, label):
    return int(_finite_number(value, label, positive=True) * GIB)


def _format_gib(byte_count):
    return "%.12g" % (float(byte_count) / GIB)


def seeded_block_order(seed, treatment_ids, repetitions):
    """Return the persisted deterministic treatment order for every block."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RamdiskError("benchmark seed must be a non-negative integer")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < MIN_REPETITIONS
    ):
        raise RamdiskError(
            "causal benchmark requires at least %d fresh processes per treatment"
            % MIN_REPETITIONS
        )
    if repetitions > 1000:
        raise RamdiskError("causal benchmark repetitions exceed the safety limit")
    identifiers = list(treatment_ids)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise RamdiskError("causal benchmark treatment identities must be unique")
    rng = random.Random(seed)
    result = []
    for block_index in range(repetitions):
        order = list(identifiers)
        rng.shuffle(order)
        result.append({"block_index": block_index, "order": order})
    return result


def _numa_policy(mode, nodes, cpu_list, numactl):
    nodes = [int(node) for node in nodes]
    node_list = ",".join(str(node) for node in nodes)
    if mode == "interleave":
        prefix = [numactl, "--interleave=%s" % node_list]
    elif mode == "local":
        prefix = [
            numactl,
            "--membind=%d" % nodes[0],
            "--cpunodebind=%d" % nodes[0],
        ]
    else:
        raise RamdiskError("unsupported causal NUMA policy: %s" % mode)
    cpus = _parse_range_list(cpu_list or "")
    if not cpus:
        raise RamdiskError("causal NUMA policy requires a non-empty CPU set")
    normalized_cpu_list = _format_range_list(cpus)
    prefix.append("--physcpubind=%s" % normalized_cpu_list)
    return {
        "mode": mode,
        "nodes": nodes,
        "command_prefix": prefix,
        "cpu_list": normalized_cpu_list,
        "thread_count": len(cpus),
        "min_expected_fraction": 0.95,
        "max_imbalance": 0.25,
    }


def _base_treatment_environment(runtime):
    ctx = int(runtime.get("ctx", 4096))
    if ctx <= 0:
        raise RamdiskError("causal benchmark context must be positive")
    return {
        "AUTOPIN": "0",
        "CAP_RAISE": "0",
        "CACHE_ROUTE": "0",
        "COLI_KV_SLOTS": "1",
        "COLI_POLICY": "quality",
        "CTX": str(ctx),
        "DRAFT": "0",
        "EXPERT_BUDGET": "0",
        "KVSAVE": "0",
        "KV_SLOTS": "1",
        "OMP_PLACES": "cores",
        "OMP_PROC_BIND": "close",
        "OMP_DYNAMIC": "FALSE",
        "PIN_FILL": "0",
        "PILOT": "0",
        "PILOT_REAL": "0",
        "PREFETCH": "0",
        "PROF": "1",
        "REPIN": "0",
        "TEMP": "0",
        "TOPK": "0",
        "TOPP": "0",
    }


def build_causal_protocol(
    manifest,
    *,
    engine_path,
    profile_path,
    residency_gib,
    cuda_host_gib,
    cuda_expert_gib,
    repetitions=MIN_REPETITIONS,
    seed=377,
    practical_threshold=0.05,
    confidence=0.95,
    ci_method=CI_METHOD,
    fingerprint_file=None,
    source_identity=None,
    numactl="numactl",
    created_at=None,
    inherited_environment=None,
    dram_collector=None,
    engine_environment_defaults=None,
    model_content_fingerprints=None,
):
    """Freeze the full experiment before the first engine is launched."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("plan"), dict):
        raise RamdiskError("causal benchmark requires a RAM-disk manifest")
    plan = manifest["plan"]
    if manifest.get("state") not in ("ready", "stopped"):
        raise RamdiskError("causal benchmark requires a ready or stopped deployment")
    if not isinstance(profile_path, str) or not profile_path:
        raise RamdiskError("causal benchmark requires one frozen evidence profile")
    planned_profile = (plan.get("profile") or {}).get("path")
    if (
        planned_profile is not None
        and (
            not isinstance(planned_profile, str)
            or not planned_profile
            or os.path.realpath(profile_path) != os.path.realpath(planned_profile)
        )
    ):
        raise RamdiskError(
            "causal benchmark frozen evidence profile must match the plan"
        )
    if not isinstance(engine_path, str) or not engine_path:
        raise RamdiskError("causal benchmark requires an engine binary")
    if not isinstance(manifest.get("deployment_id"), str) or not manifest["deployment_id"]:
        raise RamdiskError("causal benchmark deployment identity is missing")
    if ci_method != CI_METHOD:
        raise RamdiskError("unsupported paired interval method: %s" % ci_method)

    residency_bytes = _numeric_gib(residency_gib, "residency budget")
    cuda_host_bytes = _numeric_gib(cuda_host_gib, "CUDA host budget")
    cuda_expert_bytes = _numeric_gib(cuda_expert_gib, "CUDA expert budget")
    threshold = _finite_number(
        practical_threshold,
        "practical threshold",
        minimum=0.0,
        maximum=10.0,
    )
    confidence = _finite_number(
        confidence,
        "confidence level",
        minimum=0.5,
        maximum=0.9999,
    )
    if confidence == 0.5:
        raise RamdiskError("confidence level must be greater than 0.5")

    expert_set = list(plan.get("staging", {}).get("direct_mapped_experts") or [])
    direct_bytes = int(plan.get("staging", {}).get("direct_mapped_bytes") or 0)
    if not expert_set or direct_bytes <= 0:
        raise RamdiskError("causal benchmark requires a fixed direct-mapped expert set")
    if residency_bytes < direct_bytes:
        raise RamdiskError(
            "residency budget cannot hold the fixed direct-mapped expert set"
        )

    placement = plan.get("placement") or {}
    selected_nodes = [int(node) for node in placement.get("memory_nodes") or []]
    if not {0, 1}.issubset(selected_nodes):
        raise RamdiskError(
            "causal benchmark requires reviewed NUMA nodes 0 and 1"
        )
    cpu_list = placement.get("cpu_list")
    selected_cpus = placement.get("cpus")
    if not isinstance(selected_cpus, list):
        selected_cpus = _parse_range_list(cpu_list or "")
    node_zero = next(
        (
            row
            for row in (plan.get("hardware") or {}).get("nodes", [])
            if isinstance(row, dict) and row.get("id") == 0
        ),
        None,
    )
    local_cpus = sorted(
        set(int(cpu) for cpu in selected_cpus)
        & set(int(cpu) for cpu in (node_zero or {}).get("cpus", []))
    )
    if not local_cpus:
        raise RamdiskError(
            "causal benchmark requires reviewed CPUs local to NUMA node 0"
        )
    interleaved = _numa_policy("interleave", [0, 1], cpu_list, numactl)
    local = _numa_policy(
        "local",
        [0],
        _format_range_list(local_cpus),
        numactl,
    )
    canonical_model = plan.get("model", {}).get("path")
    mounts = manifest.get("mounts") or []
    staged_model = mounts[0].get("path") if mounts else None
    if not canonical_model or not staged_model:
        raise RamdiskError("causal benchmark requires canonical and staged model paths")

    file_fingerprint = fingerprint_file or _sha256_path
    binary_fingerprint = file_fingerprint(engine_path)
    profile_fingerprint = file_fingerprint(profile_path)
    if not binary_fingerprint or not profile_fingerprint:
        raise RamdiskError("causal benchmark fingerprints cannot be empty")
    model_fingerprint = manifest.get("model_fingerprint") or plan.get("model", {}).get(
        "fingerprint"
    )
    if not model_fingerprint:
        raise RamdiskError("causal benchmark model fingerprint is missing")

    runtime = plan.get("managed_runtime") or {}
    cache_cap = int(runtime.get("cache_cap", 8))
    if cache_cap <= 0:
        raise RamdiskError("causal benchmark cache cap must be positive")
    base = {
        name: str(value)
        for name, value in (inherited_environment or {}).items()
        if name in _SAFE_INHERITED_ENVIRONMENT_KEYS and value is not None
    }
    base.update(_base_treatment_environment(runtime))
    defaults = {
        "SERVE": "1",
        "SERVE_BATCH": "1",
        "NGEN": "32",
        "KV_SLOTS": "1",
    }
    defaults.update(
        {
            str(name): str(value)
            for name, value in (engine_environment_defaults or {}).items()
        }
    )
    base.update(defaults)

    def treatment(identifier, block, causal, storage, weights, policy, updates):
        environment = dict(base)
        environment.update(updates)
        environment["COLI_CPU_AFFINITY"] = policy["cpu_list"]
        environment["OMP_NUM_THREADS"] = str(policy["thread_count"])
        return {
            "id": identifier,
            "block": block,
            "causal": bool(causal),
            "storage": storage,
            "weights_path": weights,
            "numa_policy": policy,
            "environment": environment,
            "requires_zero_physical_ssd_reads": (
                storage == "tmpfs-rammap" and plan.get("mode") == "full"
            ),
        }

    pin = {
        "COLI_CUDA": "0",
        "COLI_MMAP": "0",
        "COLI_RAMMAP": "0",
        "COLI_RAM_PREFAULT": "0",
        "PIN": profile_path,
        "PIN_GB": _format_gib(residency_bytes),
    }
    rammap = {
        "COLI_CUDA": "0",
        "COLI_MMAP": "0",
        "COLI_RAMMAP": "1",
        "COLI_RAM_PREFAULT": "1",
        "PIN": "off",
    }
    slab = {
        "COLI_CUDA": "0",
        "COLI_MMAP": "0",
        "COLI_RAMMAP": "0",
        "COLI_RAM_PREFAULT": "0",
        "PIN": "off",
    }
    accelerator = plan.get("managed_accelerator") or {}
    devices = accelerator.get("devices") or []
    device_indices = [str(int(device["index"])) for device in devices if "index" in device]
    if accelerator.get("mode") != "cuda" or not device_indices:
        raise RamdiskError(
            "causal benchmark requires at least one reviewed CUDA device"
        )
    cuda = {
        "COLI_CUDA": "1",
        "COLI_MMAP": "1",
        "COLI_RAMMAP": "0",
        "COLI_RAM_PREFAULT": "0",
        "CUDA_DENSE": (
            "0" if accelerator.get("layout", "experts-only") == "experts-only" else "1"
        ),
        "CUDA_EXPERT_GB": _format_gib(cuda_expert_bytes),
        "PIN": profile_path,
        "PIN_GB": _format_gib(cuda_host_bytes),
    }
    if device_indices:
        cuda["COLI_GPUS"] = ",".join(device_indices)
        cuda["CUDA_VISIBLE_DEVICES"] = ",".join(device_indices)

    treatments = [
        treatment(TREATMENT_IDS[0], "cpu-causal", True, "anonymous-pin", canonical_model, interleaved, pin),
        treatment(TREATMENT_IDS[1], "cpu-causal", True, "anonymous-pin", canonical_model, local, pin),
        treatment(TREATMENT_IDS[2], "cpu-causal", True, "tmpfs-rammap", staged_model, interleaved, rammap),
        treatment(TREATMENT_IDS[3], "cpu-causal", True, "tmpfs-rammap", staged_model, local, rammap),
        treatment(TREATMENT_IDS[4], "media-control", False, "ssd-slab", canonical_model, interleaved, slab),
        treatment(TREATMENT_IDS[5], "media-control", False, "tmpfs-slab", staged_model, interleaved, slab),
        treatment(TREATMENT_IDS[6], "cuda-validation", False, "cuda-mmap", canonical_model, local, cuda),
    ]
    workspace_assignments = {
        "tmpfs-rammap-interleaved": "interleaved",
        "tmpfs-rammap-local": "local",
        "tmpfs-slab-control": "interleaved",
    }
    for current in treatments:
        current["workspace"] = workspace_assignments.get(current["id"])
        if current["workspace"] is not None:
            current["weights_path"] = "workspace://%s" % current["workspace"]
    order = seeded_block_order(seed, [item["id"] for item in treatments], repetitions)
    frozen = {
        "schema": PROTOCOL_SCHEMA,
        "version": MANIFEST_VERSION,
        "created_at": created_at or _utc_now(),
        "deployment_id": manifest.get("deployment_id"),
        "model_mode": plan.get("mode"),
        "engine_path": engine_path,
        "model_path": canonical_model,
        "profile_path": profile_path,
        "expert_set": expert_set,
        "expert_set_sha256": _canonical_sha256(expert_set),
        "direct_mapped_bytes": direct_bytes,
        "residency_budget_bytes": residency_bytes,
        "cache_cap": cache_cap,
        "cuda_validation": {
            "host_budget_bytes": cuda_host_bytes,
            "gpu_budget_bytes": cuda_expert_bytes,
            "devices": device_indices,
        },
        "repetitions": repetitions,
        "seed": seed,
        "predeclared": {
            "direction": "higher-throughput",
            "practical_threshold": threshold,
            "confidence": confidence,
            "ci_method": ci_method,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "fingerprints": {
            "binary": binary_fingerprint,
            "model": model_fingerprint,
            "model_shards": dict(model_content_fingerprints or {}),
            "profile": profile_fingerprint,
            "expert_set": _canonical_sha256(expert_set),
        },
        "source": source_identity,
        "workspace_requirements": _workspace_requirements(plan),
        "dram_collector": copy.deepcopy(
            dram_collector
            or {
                "available": False,
                "collector": None,
                "collector_identity": None,
                "error": "DRAM collector unavailable",
            }
        ),
        "prompt": BENCHMARK_PROMPT,
        "tokens_per_replicate": 32,
        "treatments": treatments,
        "randomized_blocks": order,
    }
    frozen["protocol_id"] = _causal_protocol_identity(frozen)
    return frozen


def _validate_workspace_roots(workspace_manager, roots, protocol=None):
    if not isinstance(roots, dict):
        raise RamdiskError("causal benchmark workspace roots are malformed")
    expected = {
        "interleaved": ("interleave", [0, 1]),
        "local": ("local", [0]),
    }
    validated = {}
    for name, (mode, nodes) in expected.items():
        descriptor = roots.get(name)
        if not isinstance(descriptor, dict):
            raise RamdiskError("causal benchmark %s workspace is missing" % name)
        descriptor = dict(descriptor)
        path = descriptor.get("path")
        if (
            not isinstance(path, str)
            or not os.path.isabs(path)
            or not _path_without_symlinks(path)
            or not os.path.isdir(path)
        ):
            raise RamdiskError(
                "causal benchmark %s workspace path is invalid" % name
            )
        if (
            descriptor.get("mode") != mode
            or descriptor.get("nodes") != nodes
            or descriptor.get("verified") is not True
            or workspace_manager.verify(descriptor) is not True
        ):
            raise RamdiskError(
                "causal benchmark %s workspace policy is unverified" % name
            )
        required_identity = {
            "name",
            "role",
            "operation_id",
            "identity",
            "size_bytes",
            "policy",
            "requested",
            "source_fingerprint",
        }
        if not required_identity.issubset(descriptor):
            raise RamdiskError(
                "causal benchmark %s workspace identity is incomplete" % name
            )
        validated[name] = copy.deepcopy(descriptor)
        validated[name]["path"] = os.path.abspath(path)
        validated[name]["mode"] = mode
        validated[name]["nodes"] = list(nodes)
        validated[name]["verified"] = True
        requirements = (protocol or {}).get("workspace_requirements") or {}
        expected_root = (requirements.get("roots") or {}).get(name)
        if expected_root is not None:
            for key in (
                "role",
                "mode",
                "nodes",
                "node",
                "policy",
                "size_bytes",
                "source_fingerprint",
            ):
                if validated[name].get(key) != expected_root.get(key):
                    raise RamdiskError(
                        "causal benchmark %s workspace differs from the logical protocol"
                        % name
                    )
    if os.path.realpath(validated["interleaved"]["path"]) == os.path.realpath(
        validated["local"]["path"]
    ):
        raise RamdiskError(
            "causal benchmark interleaved and local workspaces must be distinct"
        )
    first_identity = validated["interleaved"].get("identity") or {}
    second_identity = validated["local"].get("identity") or {}
    if (
        first_identity.get("mount_id") == second_identity.get("mount_id")
        or first_identity.get("device") == second_identity.get("device")
    ):
        raise RamdiskError(
            "causal benchmark roots require distinct devices and mount identities"
        )
    return validated


def _bind_workspace_roots(protocol, roots):
    """Create an ephemeral execution view without changing logical identity."""
    frozen = copy.deepcopy(protocol)
    attempt_ids = {
        root.get("operation_id")
        for root in roots.values()
        if root.get("role") == "scratch"
    }
    if len(attempt_ids) != 1:
        raise RamdiskError("causal benchmark workspace attempt identity is invalid")
    attempt_roots = {}
    for name, root in roots.items():
        identity = root.get("identity") or {}
        attempt_roots[name] = {
            "role": root.get("role"),
            "path": root.get("path"),
            "mode": root.get("mode"),
            "nodes": copy.deepcopy(root.get("nodes")),
            "node": root.get("node"),
            "policy": root.get("policy"),
            "size_bytes": root.get("size_bytes"),
            "source_fingerprint": root.get("source_fingerprint"),
            "identity": {
                "mount_id": identity.get("mount_id"),
                "device": identity.get("device"),
            },
        }
    attempt = {
        "operation_id": next(iter(attempt_ids)),
        "roots": attempt_roots,
    }
    for treatment in frozen["treatments"]:
        workspace = treatment.get("workspace")
        if workspace is not None:
            treatment["weights_path"] = roots[workspace]["path"]
            treatment["_workspace_attempt"] = copy.deepcopy(attempt)
    return frozen


def _safe_parent(path):
    parent = os.path.dirname(os.path.abspath(os.fspath(path)))
    os.makedirs(parent, mode=0o700, exist_ok=True)
    if not _path_without_symlinks(parent):
        raise RamdiskError("causal evidence parent contains a symlink")
    mode = os.lstat(parent).st_mode
    if not stat.S_ISDIR(mode) or mode & 0o077:
        raise RamdiskError("causal evidence parent must be a private directory")
    return parent


def _paths_alias(first, second):
    first = os.path.abspath(os.fspath(first))
    second = os.path.abspath(os.fspath(second))
    if os.path.realpath(first) == os.path.realpath(second):
        return True
    if os.path.lexists(first) and os.path.lexists(second):
        try:
            return os.path.samefile(first, second)
        except OSError:
            return True
    return False


def _validate_evidence_paths(raw_path, protocol_path, *, reserved_paths=()):
    raw_path = os.path.abspath(os.fspath(raw_path))
    protocol_path = os.path.abspath(os.fspath(protocol_path))
    for path, label in ((raw_path, "raw evidence"), (protocol_path, "protocol")):
        if not _path_without_symlinks(path):
            raise RamdiskError("%s path contains a symlink" % label)
    if _paths_alias(raw_path, protocol_path):
        raise RamdiskError("raw evidence and protocol paths must be distinct")
    for reserved in reserved_paths:
        if not reserved:
            continue
        for path, label in ((raw_path, "raw evidence"), (protocol_path, "protocol")):
            if _paths_alias(path, reserved):
                raise RamdiskError(
                    "%s path collides with durable deployment authority" % label
                )
    return raw_path, protocol_path


def _require_evidence_root(raw_path, protocol_path, evidence_root):
    evidence_root = os.path.abspath(os.fspath(evidence_root))
    for path, label in ((raw_path, "raw evidence"), (protocol_path, "protocol")):
        try:
            inside = os.path.commonpath((evidence_root, path)) == evidence_root
        except ValueError:
            inside = False
        if not inside or os.path.normpath(path) == evidence_root:
            raise RamdiskError(
                "%s path must be below the durable causal-evidence directory"
                % label
            )


def _fsync_parent(path):
    parent = os.path.dirname(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_private_regular(path, label):
    if not _path_without_symlinks(path):
        raise RamdiskError("%s path contains a symlink" % label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RamdiskError("%s is unreadable: %s" % (label, exc))
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode):
            raise RamdiskError("%s path is not a regular file" % label)
        if current.st_mode & 0o077:
            raise RamdiskError("%s path must be private" % label)
        return os.fdopen(descriptor, "r", encoding="utf-8")
    except Exception:
        os.close(descriptor)
        raise


def persist_causal_protocol(path, protocol):
    """Create one immutable protocol file, or resume the exact protocol ID."""
    path = os.path.abspath(os.fspath(path))
    _safe_parent(path)
    payload = _canonical_json(protocol) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        try:
            with _open_private_regular(path, "persisted causal protocol") as stream:
                current = json.load(stream)
        except RamdiskError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise RamdiskError("persisted causal protocol is unreadable: %s" % exc)
        expected_id = protocol.get("protocol_id")
        if (
            current.get("protocol_id") != expected_id
            or _causal_protocol_identity(current) != expected_id
            or _causal_protocol_identity(protocol) != expected_id
        ):
            raise RamdiskError("evidence path already contains a different protocol")
        return current
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise RamdiskError("causal protocol write was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_parent(path)
    return protocol


def _require_mapping(row, name):
    value = row.get(name)
    if not isinstance(value, dict):
        raise RamdiskError("raw evidence %s must be an object" % name)
    return value


def _is_hex_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _numeric_node_map(value, label):
    if not isinstance(value, dict):
        raise RamdiskError("raw evidence %s must be an object" % label)
    result = {}
    for node, count in value.items():
        if (
            not isinstance(node, str)
            or re.fullmatch(r"(?:0|[1-9][0-9]*)", node) is None
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise RamdiskError(
                "raw evidence %s must contain non-negative numeric node counts"
                % label
            )
        result[node] = count
    return result


def _validate_workspace_attempt(value):
    if not isinstance(value, dict) or set(value) != {"operation_id", "roots"}:
        raise RamdiskError("raw evidence workspace attempt is malformed")
    if re.fullmatch(
        r"benchmark:[0-9a-f]{32}", str(value.get("operation_id", ""))
    ) is None:
        raise RamdiskError("raw evidence workspace attempt identity is malformed")
    roots = value.get("roots")
    if not isinstance(roots, dict) or set(roots) != {"interleaved", "local"}:
        raise RamdiskError("raw evidence workspace roots are malformed")
    expected_keys = {
        "role",
        "path",
        "mode",
        "nodes",
        "node",
        "policy",
        "size_bytes",
        "source_fingerprint",
        "identity",
    }
    for name, expected_mode, expected_nodes, expected_node in (
        ("interleaved", "interleave", [0, 1], None),
        ("local", "local", [0], 0),
    ):
        root = roots[name]
        identity = root.get("identity") if isinstance(root, dict) else None
        if (
            not isinstance(root, dict)
            or set(root) != expected_keys
            or root.get("role") not in ("deployment", "scratch")
            or not isinstance(root.get("path"), str)
            or not os.path.isabs(root["path"])
            or not _path_without_symlinks(root["path"])
            or root.get("mode") != expected_mode
            or root.get("nodes") != expected_nodes
            or root.get("node") != expected_node
            or not isinstance(root.get("policy"), str)
            or isinstance(root.get("size_bytes"), bool)
            or not isinstance(root.get("size_bytes"), int)
            or root["size_bytes"] <= 0
            or not _is_hex_digest(root.get("source_fingerprint"))
            or not isinstance(identity, dict)
            or set(identity) != {"mount_id", "device"}
            or isinstance(identity.get("mount_id"), bool)
            or not isinstance(identity.get("mount_id"), int)
            or identity["mount_id"] <= 0
            or not isinstance(identity.get("device"), str)
            or re.fullmatch(r"[0-9]+:[0-9]+", identity["device"]) is None
        ):
            raise RamdiskError("raw evidence workspace root is malformed")
    if {root["role"] for root in roots.values()} != {"deployment", "scratch"}:
        raise RamdiskError("raw evidence workspace roles are malformed")
    if os.path.realpath(roots["interleaved"]["path"]) == os.path.realpath(
        roots["local"]["path"]
    ):
        raise RamdiskError("raw evidence workspace paths alias")
    identities = [roots[name]["identity"] for name in ("interleaved", "local")]
    if (
        identities[0]["mount_id"] == identities[1]["mount_id"]
        or identities[0]["device"] == identities[1]["device"]
    ):
        raise RamdiskError("raw evidence workspace roots are not physically distinct")
    return value


def validate_raw_evidence_row(row):
    """Validate structure without discarding failed/incomplete measurements."""
    if not isinstance(row, dict):
        raise RamdiskError("raw evidence row must be an object")
    if row.get("schema") != RAW_EVIDENCE_SCHEMA or row.get("version") != 1:
        raise RamdiskError("raw evidence schema is malformed or unsupported")
    if not _is_hex_digest(row.get("protocol_id")):
        raise RamdiskError("raw evidence protocol_id must be lowercase 64-hex")
    record_type = row.get("record_type", "replicate")
    if record_type == "workspace-cleanup":
        if row.get("status") != "error":
            raise RamdiskError("workspace cleanup evidence must record an error")
        if not isinstance(row.get("error"), str) or not row["error"]:
            raise RamdiskError("workspace cleanup evidence requires an error")
        sequence = row.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise RamdiskError("workspace cleanup evidence sequence is invalid")
        return row
    if record_type != "replicate":
        raise RamdiskError("raw evidence record_type is unsupported")
    if row.get("workspace_attempt") is not None:
        _validate_workspace_attempt(row["workspace_attempt"])
    if not isinstance(row.get("treatment_id"), str) or not row["treatment_id"]:
        raise RamdiskError("raw evidence treatment_id is missing")
    for name in ("block_index", "sequence"):
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RamdiskError("raw evidence %s must be non-negative" % name)
    if row.get("status") not in ("ok", "error"):
        raise RamdiskError("raw evidence status must be ok or error")
    for name in (
        "process",
        "applied_environment",
        "numa_policy",
        "fingerprints",
        "profiler",
        "swap",
        "rss",
        "numa_placement",
        "dram_traffic",
        "performance",
        "correctness",
    ):
        _require_mapping(row, name)
    if row["status"] == "ok":
        process = row["process"]
        if not isinstance(process.get("launch_id"), str) or not process["launch_id"]:
            raise RamdiskError("raw evidence process launch_id is missing")
        if not isinstance(process.get("pid"), int) or process["pid"] <= 0:
            raise RamdiskError("raw evidence process PID is invalid")
        if (
            isinstance(process.get("starttime"), bool)
            or not isinstance(process.get("starttime"), int)
            or process["starttime"] <= 0
        ):
            raise RamdiskError("raw evidence process starttime is invalid")
        if not _is_hex_digest(row.get("output_sha256")):
            raise RamdiskError("raw evidence output_sha256 must be lowercase 64-hex")
        throughput = row["performance"].get("tokens_per_second")
        _finite_number(throughput, "raw evidence throughput", positive=True)
        for name in ("elapsed_seconds", "ttft_ms", "forward_p50_ms", "forward_p99_ms"):
            value = row["performance"].get(name)
            if value is not None:
                _finite_number(
                    value,
                    "raw evidence performance %s" % name,
                    positive=name == "elapsed_seconds",
                    minimum=None if name == "elapsed_seconds" else 0.0,
                )
        for name in ("anonymous_bytes", "file_bytes", "shmem_bytes"):
            value = row["rss"].get(name)
            if value is not None:
                _finite_number(
                    value,
                    "raw evidence RSS %s" % name,
                    minimum=0.0,
                )
        for name in ("before_bytes", "after_bytes", "process_bytes"):
            value = row["swap"].get(name)
            if value is not None:
                _finite_number(
                    value,
                    "raw evidence swap %s" % name,
                    minimum=0.0,
                )
        if row["swap"].get("delta_bytes") is not None:
            _finite_number(
                row["swap"]["delta_bytes"],
                "raw evidence swap delta_bytes",
            )
        before = row["swap"].get("before_bytes")
        after = row["swap"].get("after_bytes")
        delta = row["swap"].get("delta_bytes")
        if before is None or after is None or delta is None:
            raise RamdiskError("successful raw evidence requires complete swap counters")
        if float(delta) != float(after) - float(before):
            raise RamdiskError("raw evidence swap delta is inconsistent")

        placement = row["numa_placement"]
        by_node = _numeric_node_map(placement.get("by_node"), "NUMA by_node")
        bytes_by_node = _numeric_node_map(
            placement.get("bytes_by_node"), "NUMA bytes_by_node"
        )
        page_size = placement.get("page_size")
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or page_size <= 0
            or set(bytes_by_node) != set(by_node)
            or any(
                bytes_by_node[node] != by_node[node] * page_size
                for node in by_node
            )
        ):
            raise RamdiskError("raw evidence NUMA byte counts are inconsistent")
        verified = _placement_verified(by_node, row["numa_policy"])
        if placement.get("verified") is not verified:
            raise RamdiskError("raw evidence NUMA verification is inconsistent")
    elif not isinstance(row.get("error"), str) or not row["error"]:
        raise RamdiskError("failed raw evidence row requires an error")
    dram = row["dram_traffic"]
    if not isinstance(dram.get("available"), bool):
        raise RamdiskError("raw evidence dram_traffic.available must be boolean")
    if dram["available"]:
        for name in ("read_bytes", "write_bytes", "total_bytes"):
            _finite_number(dram.get(name), "raw evidence DRAM %s" % name, minimum=0.0)
        if float(dram["total_bytes"]) != (
            float(dram["read_bytes"]) + float(dram["write_bytes"])
        ):
            raise RamdiskError("raw evidence DRAM total is inconsistent")
        if not isinstance(dram.get("collector"), str) or not dram["collector"]:
            raise RamdiskError("available raw DRAM evidence requires a collector")
        if not _is_hex_digest(dram.get("collector_identity")):
            raise RamdiskError(
                "available raw DRAM evidence requires a frozen collector identity"
            )
        if dram.get("error") is not None:
            raise RamdiskError("available raw DRAM evidence cannot contain an error")
    elif (
        any(dram.get(name) is not None for name in (
            "read_bytes", "write_bytes", "total_bytes", "collector",
            "collector_identity",
        ))
        or not isinstance(dram.get("error"), str)
        or not dram["error"]
    ):
        raise RamdiskError("unavailable raw DRAM evidence has inconsistent fields")
    for name in ("zero_swap_growth", "physical_reads_ok", "placement_ok"):
        if not isinstance(row["correctness"].get(name), bool):
            raise RamdiskError("raw evidence correctness.%s must be boolean" % name)
    if row["status"] == "ok":
        expected_zero_swap = row["swap"]["after_bytes"] <= row["swap"]["before_bytes"]
        expected_placement = row["numa_placement"]["verified"] is True
        if row["correctness"]["zero_swap_growth"] is not expected_zero_swap:
            raise RamdiskError("raw evidence swap correctness is inconsistent")
        if row["correctness"]["placement_ok"] is not expected_placement:
            raise RamdiskError("raw evidence placement correctness is inconsistent")
    return row


def append_raw_evidence(path, row):
    """Append exactly one validated JSON row without rewriting history."""
    validate_raw_evidence_row(row)
    path = os.path.abspath(os.fspath(path))
    _safe_parent(path)
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    created = not os.path.lexists(path)
    descriptor = os.open(path, flags, 0o600)
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode):
            raise RamdiskError("raw evidence path is not a regular file")
        if current.st_mode & 0o077:
            raise RamdiskError("raw evidence path must be private")
        payload = _canonical_json(row) + b"\n"
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise RamdiskError("raw evidence append was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if created:
        _fsync_parent(path)
    return row


def read_raw_evidence(path):
    path = os.path.abspath(os.fspath(path))
    if not os.path.lexists(path):
        return []
    rows = []
    try:
        with _open_private_regular(path, "raw evidence") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.endswith("\n"):
                    raise RamdiskError(
                        "raw evidence ends with a partial row at line %d" % line_number
                    )
                if not line.strip():
                    raise RamdiskError("raw evidence has an empty row at line %d" % line_number)
                row = json.loads(line)
                validate_raw_evidence_row(row)
                rows.append(row)
    except RamdiskError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise RamdiskError("raw evidence is unreadable: %s" % exc)
    return rows


class DurableEvidenceStore:
    """Bind append-only evidence to retained parent and file descriptors."""

    def __init__(
        self,
        raw_path,
        protocol_path,
        *,
        assert_durable=None,
    ):
        raw_path, protocol_path = _validate_evidence_paths(
            raw_path, protocol_path
        )
        raw_parent = _safe_parent(raw_path)
        protocol_parent = _safe_parent(protocol_path)
        if os.path.realpath(raw_parent) != os.path.realpath(protocol_parent):
            raise RamdiskError(
                "raw evidence and protocol must share one bound durable parent"
            )
        required = ("O_DIRECTORY", "O_NOFOLLOW")
        if any(
            not isinstance(getattr(os, name, None), int)
            or getattr(os, name) == 0
            for name in required
        ):
            raise RamdiskError(
                "durable causal evidence requires no-follow directory opens"
            )
        self.raw_path = raw_path
        self.protocol_path = protocol_path
        self.parent = raw_parent
        self.raw_name = os.path.basename(raw_path)
        self.protocol_name = os.path.basename(protocol_path)
        self._assert_durable = assert_durable
        self._parent_fd = os.open(
            self.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        self._parent_identity = self._identity(os.fstat(self._parent_fd))
        self._raw_fd = None
        self._raw_identity = None
        self._protocol_fd = None
        self._protocol_identity = None
        self._protocol_payload = None
        self._revalidate_parent()

    @staticmethod
    def _identity(info):
        return (int(info.st_dev), int(info.st_ino))

    @staticmethod
    def _read_all(descriptor):
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _validate_private_file(info, label):
        if not stat.S_ISREG(info.st_mode):
            raise RamdiskError("%s path is not a regular file" % label)
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise RamdiskError("%s path must be private" % label)
        if int(info.st_nlink) != 1:
            raise RamdiskError("%s path must not have hard links" % label)
        getuid = getattr(os, "getuid", None)
        if callable(getuid) and int(info.st_uid) != int(getuid()):
            raise RamdiskError("%s path has the wrong owner" % label)

    def _revalidate_parent(self):
        if not _path_without_symlinks(self.parent):
            raise RamdiskError("causal evidence parent contains a symlink")
        path_info = os.stat(self.parent, follow_symlinks=False)
        open_info = os.fstat(self._parent_fd)
        if (
            not stat.S_ISDIR(path_info.st_mode)
            or self._identity(path_info) != self._parent_identity
            or self._identity(open_info) != self._parent_identity
        ):
            raise RamdiskError("causal evidence parent identity changed")
        if callable(self._assert_durable):
            self._assert_durable(self.parent)

    def _path_info(self, name, label):
        try:
            info = os.stat(
                name,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RamdiskError("%s path identity is unavailable: %s" % (label, exc))
        self._validate_private_file(info, label)
        return info

    def _revalidate_file(self, descriptor, identity, name, label):
        open_info = os.fstat(descriptor)
        path_info = self._path_info(name, label)
        if (
            self._identity(open_info) != identity
            or self._identity(path_info) != identity
        ):
            raise RamdiskError("%s path identity changed" % label)

    def _revalidate(self):
        self._revalidate_parent()
        if self._protocol_fd is not None:
            self._revalidate_file(
                self._protocol_fd,
                self._protocol_identity,
                self.protocol_name,
                "persisted causal protocol",
            )
            if self._read_all(self._protocol_fd) != self._protocol_payload:
                raise RamdiskError("persisted causal protocol content changed")
        if self._raw_fd is not None:
            self._revalidate_file(
                self._raw_fd,
                self._raw_identity,
                self.raw_name,
                "raw evidence",
            )

    def _create_protocol(self, payload):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(
            self.protocol_name,
            flags,
            0o600,
            dir_fd=self._parent_fd,
        )
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise RamdiskError("causal protocol write was incomplete")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(self._parent_fd)

    def bind_protocol(self, protocol):
        """Create or bind the immutable protocol and the append-only raw file."""
        self._revalidate_parent()
        requested_payload = _canonical_json(protocol) + b"\n"
        try:
            self._create_protocol(requested_payload)
        except FileExistsError:
            pass
        self._protocol_fd = os.open(
            self.protocol_name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=self._parent_fd,
        )
        protocol_info = os.fstat(self._protocol_fd)
        self._validate_private_file(protocol_info, "persisted causal protocol")
        self._protocol_identity = self._identity(protocol_info)
        payload = self._read_all(self._protocol_fd)
        try:
            current = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, TypeError) as exc:
            raise RamdiskError(
                "persisted causal protocol is unreadable: %s" % exc
            ) from exc
        expected_id = protocol.get("protocol_id")
        if (
            not payload.endswith(b"\n")
            or current.get("protocol_id") != expected_id
            or _causal_protocol_identity(current) != expected_id
            or _causal_protocol_identity(protocol) != expected_id
        ):
            raise RamdiskError("evidence path already contains a different protocol")
        self._protocol_payload = payload

        created = False
        raw_flags = os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW
        try:
            self._raw_fd = os.open(
                self.raw_name,
                raw_flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=self._parent_fd,
            )
            created = True
        except FileExistsError:
            self._raw_fd = os.open(
                self.raw_name,
                raw_flags,
                dir_fd=self._parent_fd,
            )
        raw_info = os.fstat(self._raw_fd)
        self._validate_private_file(raw_info, "raw evidence")
        self._raw_identity = self._identity(raw_info)
        if created:
            os.fsync(self._raw_fd)
            os.fsync(self._parent_fd)
        self._revalidate()
        return current

    @staticmethod
    def _parse_rows(payload):
        if payload and not payload.endswith(b"\n"):
            raise RamdiskError("raw evidence ends with a partial row")
        rows = []
        for line_number, line in enumerate(payload.splitlines(), 1):
            if not line.strip():
                raise RamdiskError(
                    "raw evidence has an empty row at line %d" % line_number
                )
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError, TypeError) as exc:
                raise RamdiskError(
                    "raw evidence is unreadable at line %d: %s"
                    % (line_number, exc)
                ) from exc
            validate_raw_evidence_row(row)
            rows.append(row)
        return rows

    def read_rows(self):
        self._revalidate()
        rows = self._parse_rows(self._read_all(self._raw_fd))
        self._revalidate()
        return rows

    def append(self, row):
        validate_raw_evidence_row(row)
        self._revalidate()
        payload = _canonical_json(row) + b"\n"
        written = os.write(self._raw_fd, payload)
        if written != len(payload):
            raise RamdiskError("raw evidence append was incomplete")
        os.fsync(self._raw_fd)
        self._revalidate()
        return row

    def close(self):
        for name in ("_raw_fd", "_protocol_fd", "_parent_fd"):
            descriptor = getattr(self, name, None)
            if descriptor is not None:
                try:
                    os.close(descriptor)
                finally:
                    setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(self, _kind, _value, _traceback):
        self.close()


def failed_raw_evidence_row(
    protocol,
    treatment,
    *,
    block_index,
    sequence,
    error,
    started_at=None,
    finished_at=None,
):
    error_text = str(error).strip() or "replicate failed without an error message"
    row = {
        "schema": RAW_EVIDENCE_SCHEMA,
        "version": 1,
        "record_type": "replicate",
        "protocol_id": protocol["protocol_id"],
        "treatment_id": treatment["id"],
        "block_index": block_index,
        "sequence": sequence,
        "started_at": started_at or _utc_now(),
        "finished_at": finished_at or _utc_now(),
        "status": "error",
        "process": {"launch_id": None, "pid": None, "starttime": None},
        "applied_environment": dict(treatment.get("environment") or {}),
        "numa_policy": dict(treatment.get("numa_policy") or {}),
        "fingerprints": dict(protocol.get("fingerprints") or {}),
        "output_sha256": None,
        "profiler": {
            "rammap_experts": None,
            "rammap_bytes": None,
            "physical_ssd_bytes": None,
            "physical_ssd_valid": None,
        },
        "swap": {
            "before_bytes": None,
            "after_bytes": None,
            "delta_bytes": None,
            "process_bytes": None,
        },
        "rss": {
            "anonymous_bytes": None,
            "file_bytes": None,
            "shmem_bytes": None,
        },
        "numa_placement": {
            "by_node": {},
            "bytes_by_node": {},
            "page_size": None,
            "verified": False,
        },
        "dram_traffic": {
            "available": False,
            "read_bytes": None,
            "write_bytes": None,
            "total_bytes": None,
            "collector": None,
            "collector_identity": None,
            "error": "replicate failed before complete DRAM evidence",
        },
        "performance": {
            "tokens_per_second": None,
            "elapsed_seconds": None,
            "ttft_ms": None,
            "forward_p50_ms": None,
            "forward_p99_ms": None,
        },
        "correctness": {
            "zero_swap_growth": False,
            "physical_reads_ok": False,
            "placement_ok": False,
        },
        "error": error_text,
    }
    if treatment.get("_workspace_attempt") is not None:
        row["workspace_attempt"] = copy.deepcopy(
            treatment["_workspace_attempt"]
        )
    return row


def workspace_cleanup_evidence_row(protocol, error):
    error_text = str(error).strip() or "workspace cleanup failed"
    return {
        "schema": RAW_EVIDENCE_SCHEMA,
        "version": 1,
        "record_type": "workspace-cleanup",
        "protocol_id": protocol["protocol_id"],
        "sequence": int(protocol["repetitions"]) * len(protocol["treatments"]),
        "started_at": _utc_now(),
        "finished_at": _utc_now(),
        "status": "error",
        "error": error_text,
    }


def _sha256_path(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise RamdiskError("cannot fingerprint %s: %s" % (path, exc))
    return "sha256:" + digest.hexdigest()


def _model_content_fingerprints(plan, fingerprint_file):
    """Hash every source shard payload, not only safetensors metadata."""
    shards = plan.get("source_shards") or []
    if not isinstance(shards, list) or not shards:
        raise RamdiskError("causal benchmark model shard inventory is missing")
    result = {}
    for record in shards:
        if not isinstance(record, dict):
            raise RamdiskError("causal benchmark model shard inventory is malformed")
        name = record.get("name")
        path = record.get("path")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(path, str)
            or not os.path.isabs(path)
        ):
            raise RamdiskError("causal benchmark model shard identity is malformed")
        digest = fingerprint_file(path)
        if not isinstance(digest, str) or not digest:
            raise RamdiskError("causal benchmark model shard digest is unavailable")
        result[name] = digest
    if len(result) != len(shards):
        raise RamdiskError("causal benchmark model shard names are duplicated")
    return result


def _assert_frozen_artifacts(protocol, fingerprint_file, treatment=None):
    for name, path in (
        ("binary", protocol.get("engine_path")),
        ("profile", protocol.get("profile_path")),
    ):
        expected = (protocol.get("fingerprints") or {}).get(name)
        if not path or not expected or fingerprint_file(path) != expected:
            raise RamdiskError(
                "causal benchmark %s changed after protocol freeze" % name
            )
    shard_fingerprints = (protocol.get("fingerprints") or {}).get(
        "model_shards"
    ) or {}
    if shard_fingerprints:
        roots = [protocol.get("model_path")]
        if isinstance(treatment, dict):
            roots.append(treatment.get("weights_path"))
        for root in {path for path in roots if isinstance(path, str) and path}:
            for shard, expected in sorted(shard_fingerprints.items()):
                path = os.path.join(root, shard)
                if fingerprint_file(path) != expected:
                    raise RamdiskError(
                        "causal benchmark model shard changed after protocol "
                        "freeze: %s" % path
                    )
    collector = protocol.get("dram_collector") or {}
    argv = collector.get("argv")
    expected = collector.get("executable_fingerprint")
    if collector.get("available") is True:
        if (
            not isinstance(argv, list)
            or not argv
            or not isinstance(argv[0], str)
            or not expected
            or fingerprint_file(argv[0]) != expected
        ):
            raise RamdiskError(
                "causal benchmark DRAM collector changed after protocol freeze"
            )


def _source_build_identity(source_file, *, environ, which, run):
    """Return best-effort revision metadata for reproducible reports."""
    explicit = environ.get("COLI_BUILD_COMMIT")
    if explicit:
        return {"revision": explicit, "working_tree_modified": None}
    git = which("git")
    if not git:
        return {"revision": None, "working_tree_modified": None}
    source_dir = os.path.dirname(os.path.abspath(source_file))
    revision = run([git, "-C", source_dir, "rev-parse", "HEAD"])
    if revision.returncode:
        return {"revision": None, "working_tree_modified": None}
    status_result = run([git, "-C", source_dir, "status", "--porcelain"])
    return {
        "revision": revision.stdout.strip() or None,
        "working_tree_modified": (
            None if status_result.returncode else bool(status_result.stdout.strip())
        ),
    }


def _parse_profiler(text, elapsed):
    """Parse legacy human profiler output used by fixture engines."""
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
    match = re.search(r"RAM map:\s*(\d+) experts / ([0-9.]+) GB", text)
    if match:
        ram_experts = int(match.group(1))
        ram_bytes = float(match.group(2)) * 1e9
    io_bytes = None
    match = re.search(
        r"(?:physical SSD|disk I/O|expert I/O)[^\n]*?"
        r"([0-9.]+)\s*(GB|MB|bytes)",
        text,
        re.I,
    )
    if match:
        scale = {"gb": 1e9, "mb": 1e6, "bytes": 1}[match.group(2).lower()]
        io_bytes = float(match.group(1)) * scale
    prefault = None
    match = re.search(r"prefaulted in ([0-9.]+)s", text)
    if match:
        prefault = float(match.group(1))
    ttft_ms = None
    match = re.search(r"TTFT\s+([0-9.]+)s", text, re.I)
    if match:
        ttft_ms = float(match.group(1)) * 1000.0
    rss_bytes = None
    match = re.search(r"\bRSS\s+([0-9.]+)\s+GB", text, re.I)
    if match:
        rss_bytes = float(match.group(1)) * 1e9
    return {
        "elapsed_seconds": elapsed,
        "tokens_per_second": (
            rates[-1] if rates else (32.0 / elapsed if elapsed else None)
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


class _UnavailableDramCollector:
    available = False

    def __init__(self, reason):
        self.reason = str(reason)
        self.description = {
            "available": False,
            "collector": None,
            "collector_identity": None,
            "error": self.reason,
        }

    def start(self, pid, treatment):
        del pid, treatment
        return None

    def finish(self, handle):
        del handle
        return {
            "available": False,
            "read_bytes": None,
            "write_bytes": None,
            "total_bytes": None,
            "collector": None,
            "collector_identity": None,
            "error": self.reason,
        }


def _unavailable_dram(reason):
    return _UnavailableDramCollector(reason).finish(None)


def _normalize_dram_measurement(value):
    """Downgrade runtime counter loss to explicit incomplete evidence."""
    if not isinstance(value, dict):
        return _unavailable_dram("DRAM collector returned a non-object snapshot")
    if value.get("available") is not True:
        return _unavailable_dram(
            value.get("error") or "DRAM collector snapshot is unavailable"
        )
    try:
        read_bytes = int(value["read_bytes"])
        write_bytes = int(value["write_bytes"])
        total_bytes = int(value["total_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        return _unavailable_dram(
            "DRAM collector returned malformed runtime counters: %s" % error
        )
    if (
        read_bytes < 0
        or write_bytes < 0
        or total_bytes < 0
        or total_bytes != read_bytes + write_bytes
    ):
        return _unavailable_dram(
            "DRAM collector returned inconsistent runtime counters"
        )
    result = dict(value)
    result.update(
        available=True,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        total_bytes=total_bytes,
    )
    return result


class _CommandDramCollector:
    available = True

    def __init__(self, command, run, metadata, executable_fingerprint):
        self.command = list(command)
        self.run = run
        self.metadata = dict(metadata)
        self.identity = _canonical_sha256(
            {
                "argv": self.command,
                "executable_fingerprint": executable_fingerprint,
                "metadata": self.metadata,
                "unit": "bytes",
            }
        )
        self.description = {
            "available": True,
            "collector": " ".join(self.command),
            "collector_identity": self.identity,
            "argv": list(self.command),
            "executable_fingerprint": executable_fingerprint,
            "metadata": self.metadata,
            "unit": "bytes",
        }

    def _snapshot(self, pid):
        result = self.run(
            self.command + ["--snapshot", "--pid", str(int(pid)), "--json"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise RamdiskError(
                "DRAM collector snapshot failed: %s"
                % ((result.stderr or result.stdout).strip() or result.returncode)
            )
        try:
            value = json.loads(result.stdout)
            read_bytes = int(value["read_bytes"])
            write_bytes = int(value["write_bytes"])
        except (ValueError, TypeError, KeyError) as exc:
            raise RamdiskError("DRAM collector returned malformed counters: %s" % exc)
        if read_bytes < 0 or write_bytes < 0:
            raise RamdiskError("DRAM collector counters must be non-negative")
        return {"pid": int(pid), "read_bytes": read_bytes, "write_bytes": write_bytes}

    def start(self, pid, treatment):
        del treatment
        return self._snapshot(pid)

    def finish(self, handle):
        current = self._snapshot(handle["pid"])
        read_bytes = current["read_bytes"] - handle["read_bytes"]
        write_bytes = current["write_bytes"] - handle["write_bytes"]
        if read_bytes < 0 or write_bytes < 0:
            return _UnavailableDramCollector(
                "DRAM collector counters reset during the replicate"
            ).finish(None)
        return {
            "available": True,
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "total_bytes": read_bytes + write_bytes,
            "collector": " ".join(self.command),
            "collector_identity": self.identity,
            "error": None,
        }


def preflight_dram_collector(
    *,
    environ=None,
    which=shutil.which,
    run=subprocess.run,
    fingerprint_file=_sha256_path,
):
    """Verify an external byte-counter helper before any process launch."""
    environment = os.environ if environ is None else environ
    configured = environment.get("COLI_DRAM_COLLECTOR")
    if not configured:
        return _UnavailableDramCollector("DRAM collector unavailable")
    try:
        command = shlex.split(configured)
    except ValueError as exc:
        return _UnavailableDramCollector("DRAM collector command is malformed: %s" % exc)
    if not command:
        return _UnavailableDramCollector("DRAM collector unavailable")
    if not os.path.isabs(command[0]):
        resolved = which(command[0])
        if not resolved:
            return _UnavailableDramCollector("DRAM collector executable unavailable")
        command[0] = resolved
    try:
        executable_fingerprint = fingerprint_file(command[0])
    except (OSError, RamdiskError, ValueError, TypeError) as exc:
        return _UnavailableDramCollector(
            "DRAM collector executable cannot be fingerprinted: %s" % exc
        )
    if not isinstance(executable_fingerprint, str) or not executable_fingerprint:
        return _UnavailableDramCollector(
            "DRAM collector executable fingerprint is unavailable"
        )
    try:
        result = run(
            command + ["--preflight", "--json"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            return _UnavailableDramCollector(
                "DRAM collector preflight failed: %s"
                % ((result.stderr or result.stdout).strip() or result.returncode)
            )
    except (OSError, ValueError, TypeError) as exc:
        return _UnavailableDramCollector("DRAM collector preflight failed: %s" % exc)
    try:
        metadata = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        return _UnavailableDramCollector(
            "DRAM collector preflight failed: %s" % exc
        )
    if not isinstance(metadata, dict):
        return _UnavailableDramCollector(
            "DRAM collector preflight JSON must be an object"
        )
    if metadata.get("available") is not True or metadata.get("unit") != "bytes":
        return _UnavailableDramCollector(
            "DRAM collector preflight did not verify byte counters"
        )
    return _CommandDramCollector(
        command,
        run,
        metadata,
        executable_fingerprint,
    )


def _parse_process_status(text):
    values = {}
    names = {
        "RssAnon": "anonymous_bytes",
        "RssFile": "file_bytes",
        "RssShmem": "shmem_bytes",
        "VmSwap": "process_swap_bytes",
    }
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z]+):\s*(\d+)\s+kB\s*$", line)
        if match and match.group(1) in names:
            values[names[match.group(1)]] = int(match.group(2)) * 1024
    return {name: values.get(name) for name in names.values()}


def _parse_numa_maps(text, *, page_size):
    by_node = {}
    for line in text.splitlines():
        for node, pages in re.findall(r"\bN(\d+)=(\d+)\b", line):
            by_node[node] = by_node.get(node, 0) + int(pages)
    return {
        "by_node": by_node,
        "bytes_by_node": {
            node: pages * int(page_size) for node, pages in by_node.items()
        },
        "page_size": int(page_size),
    }


def _read_process_status(pid):
    with open("/proc/%d/status" % int(pid), "r", encoding="utf-8") as stream:
        return _parse_process_status(stream.read())


def _placement_verified(by_node, policy):
    expected = [str(int(node)) for node in policy.get("nodes") or []]
    counts = {str(node): int(value) for node, value in by_node.items()}
    if not expected or any(counts.get(node, 0) <= 0 for node in expected):
        return False
    expected_total = sum(counts.get(node, 0) for node in expected)
    total = sum(counts.values())
    minimum_fraction = float(policy.get("min_expected_fraction", 1.0))
    if total <= 0 or float(expected_total) / total < minimum_fraction:
        return False
    if policy.get("mode") == "local":
        return len(expected) == 1
    if policy.get("mode") == "interleave":
        average = float(expected_total) / len(expected)
        maximum_imbalance = float(policy.get("max_imbalance", 0.25))
        return all(
            abs(counts[node] - average) / average <= maximum_imbalance
            for node in expected
        )
    return False


def _sample_process_numa(pid, policy):
    with open("/proc/%d/numa_maps" % int(pid), "r", encoding="utf-8") as stream:
        result = _parse_numa_maps(
            stream.read(),
            page_size=os.sysconf("SC_PAGE_SIZE"),
        )
    result["verified"] = _placement_verified(result["by_node"], policy)
    return result


def _process_starttime(pid):
    with open("/proc/%d/stat" % int(pid), "r", encoding="utf-8") as stream:
        value = stream.read().strip()
    right = value.rfind(")")
    fields = value[right + 2 :].split()
    if right < 0 or len(fields) <= 19:
        raise RamdiskError("engine process identity is malformed")
    return int(fields[19])


def _swap_used_bytes():
    values = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as stream:
        for line in stream:
            match = re.match(r"^(SwapTotal|SwapFree):\s*(\d+)\s+kB", line)
            if match:
                values[match.group(1)] = int(match.group(2)) * 1024
    if set(values) != {"SwapTotal", "SwapFree"}:
        raise RamdiskError("cannot measure host swap usage")
    return max(0, values["SwapTotal"] - values["SwapFree"])


def _environment_for_treatment(treatment, *, environ, state_dir=None):
    # Ambient state is frozen into the protocol once.  Replicates never inherit
    # a value which was not reviewed and recorded by the protocol.
    del environ
    environment = {
        name: str(value) for name, value in treatment["environment"].items()
    }
    environment["COLI_WEIGHTS_DIR"] = treatment["weights_path"]
    environment["SNAP"] = treatment["weights_path"]
    if state_dir is not None:
        environment["COLI_STATE_DIR"] = state_dir
    policy = treatment["numa_policy"]
    environment["COLI_NUMA"] = "1"
    environment["COLI_NUMA_NODES"] = ",".join(str(node) for node in policy["nodes"])
    return environment


def _benchmark_environment(
    manifest,
    weights_dir,
    state_dir,
    rammap,
    node=None,
    knobs=None,
    *,
    environ=None,
):
    """Compatibility projection for callers that review benchmark placement.

    The causal production runner builds frozen treatment environments instead;
    this helper deliberately refuses the donor tuning-knob surface.
    """
    if knobs:
        raise RamdiskError(
            "causal benchmark environments do not accept tuning knobs"
        )
    plan = manifest.get("plan") if isinstance(manifest, dict) else None
    if not isinstance(plan, dict):
        raise RamdiskError("benchmark environment requires a RAM-disk plan")
    placement = plan.get("placement") or {}
    selected_nodes = (
        [int(node)]
        if node is not None
        else [int(item) for item in placement.get("memory_nodes") or []]
    )
    if not selected_nodes:
        raise RamdiskError("benchmark environment has no reviewed NUMA nodes")
    if node is None:
        cpu_list = placement.get("cpu_list")
    else:
        cpu_list = next(
            (
                item.get("cpu_list")
                for item in placement.get("engine_cpu_sets") or []
                if item.get("node") == int(node)
            ),
            None,
        )
    if not cpu_list:
        cpu_list = placement.get("cpu_list")
    policy = _numa_policy(
        "local" if len(selected_nodes) == 1 else "interleave",
        selected_nodes,
        cpu_list,
        "numactl",
    )
    environment = _base_treatment_environment(plan.get("managed_runtime") or {})
    environment.update(
        {
            "COLI_CPU_AFFINITY": policy["cpu_list"],
            "COLI_CUDA": "0",
            "COLI_MMAP": "0",
            "COLI_RAMMAP": "1" if rammap else "0",
            "COLI_RAM_PREFAULT": (
                str(int(plan.get("prefault", 1))) if rammap else "0"
            ),
            "OMP_NUM_THREADS": str(policy["thread_count"]),
            "PIN": "off",
        }
    )
    treatment = {
        "environment": environment,
        "weights_path": weights_dir,
        "numa_policy": policy,
    }
    return _environment_for_treatment(
        treatment,
        environ=os.environ if environ is None else environ,
        state_dir=state_dir,
    )


def _environment_projection(environment):
    return {name: str(environment[name]) for name in sorted(environment)}


def _cancellable_engine_type(engine_type, read_engine_turn, ready_marker, cancel_event):
    """Adapt benchmark startup so cancellation interrupts the READY wait."""
    if cancel_event is None:
        return engine_type

    class CancellableEngine(engine_type):
        @classmethod
        def _wait_until_ready(cls, process, timeout):
            outcome = queue.Queue(maxsize=1)

            def read_ready():
                try:
                    read_engine_turn(process.stdout, ready_marker, lambda _: None)
                except BaseException as error:
                    outcome.put(error)
                else:
                    outcome.put(None)

            reader = threading.Thread(
                target=read_ready,
                name="colibri-causal-benchmark-ready",
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
                            "colibri engine did not become ready within %.3g seconds"
                            % timeout
                        )
                    try:
                        error = outcome.get(timeout=min(0.2, remaining))
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


def _benchmark_generate(engine, prompt, on_text, cancel_event, client_cancelled_type):
    """Run one measured turn and wake it safely on cancellation."""
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
                engine.close()
            except BaseException as exc:
                close_errors.append(exc)
            return

    watcher = threading.Thread(
        target=cancel_watch,
        name="colibri-causal-benchmark-cancel",
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
                    "benchmark cancellation could not close its engine: %s"
                    % close_errors[0]
                )
            _raise_if_cancelled(cancel_event)
            raise
        except BaseException as exc:
            if cancel_event.is_set():
                watcher.join()
                if close_errors:
                    raise _EngineCleanupError(
                        "benchmark cancellation could not close its engine: %s"
                        % close_errors[0]
                    ) from exc
                raise _OperationCancelled(
                    "benchmark cancelled by user at a safe checkpoint"
                ) from exc
            raise
        if cancel_event.is_set():
            watcher.join()
            if close_errors:
                raise _EngineCleanupError(
                    "benchmark cancellation could not close its engine: %s"
                    % close_errors[0]
                )
            _raise_if_cancelled(cancel_event)
        return result
    finally:
        done.set()
        watcher.join(timeout=1)


def _default_engine_dependencies(cancel_event):
    from openai_server import (
        READY,
        ClientCancelled,
        Engine as BaseEngine,
        model_arch,
        read_engine_turn,
        render_chat,
        tune_child_env,
    )

    cancellable_type = _cancellable_engine_type(
        BaseEngine,
        read_engine_turn,
        READY,
        cancel_event,
    )

    class RecordedEnvironmentEngine(cancellable_type):
        def __init__(
            self,
            executable,
            model,
            cap=None,
            max_tokens=1024,
            env=None,
            kv_slots=1,
            command_prefix=None,
            stderr=None,
        ):
            child_environment = dict(
                env or os.environ,
                SNAP=str(model),
                SERVE="1",
                SERVE_BATCH="1",
                NGEN=str(max_tokens),
                KV_SLOTS=str(kv_slots),
            )
            tune_child_env(child_environment, model_arch(model))
            self.benchmark_child_environment = dict(child_environment)
            super().__init__(
                executable,
                model,
                cap=cap,
                max_tokens=max_tokens,
                env=child_environment,
                kv_slots=kv_slots,
                command_prefix=command_prefix,
                stderr=stderr,
            )

    def render(prompt):
        return render_chat(
            [{"role": "user", "content": prompt}],
            False,
            None,
            None,
            None,
        )

    return RecordedEnvironmentEngine, render, ClientCancelled


def _production_engine_environment_defaults(model_path, max_tokens=32, kv_slots=1):
    """Freeze the exact defaults which ``openai_server.Engine`` adds."""
    from openai_server import model_arch, tune_child_env

    defaults = {
        "SERVE": "1",
        "SERVE_BATCH": "1",
        "NGEN": str(int(max_tokens)),
        "KV_SLOTS": str(int(kv_slots)),
    }
    tune_child_env(defaults, model_arch(model_path))
    return {name: str(value) for name, value in defaults.items()}


def run_causal_replicate(
    protocol,
    treatment,
    block_index,
    sequence,
    *,
    engine_path=None,
    engine_factory=None,
    render_prompt=None,
    client_cancelled_type=None,
    process_starttime=_process_starttime,
    swap_used_bytes=_swap_used_bytes,
    rss_sampler=_read_process_status,
    numa_sampler=_sample_process_numa,
    dram_collector=None,
    benchmark_generate=_benchmark_generate,
    monotonic=time.monotonic,
    utc_now=_utc_now,
    launch_id_factory=lambda: secrets.token_hex(16),
    environ=None,
    state_dir=None,
    cancel_event=None,
    stderr=None,
    fingerprint_file=None,
):
    """Launch one fresh engine and return one complete raw evidence row."""
    _raise_if_cancelled(cancel_event)
    if treatment.get("id") not in {
        item.get("id") for item in protocol.get("treatments", [])
    }:
        raise RamdiskError("replicate treatment does not belong to the protocol")
    if engine_factory is None or render_prompt is None or client_cancelled_type is None:
        default_engine, default_render, default_cancelled = _default_engine_dependencies(
            cancel_event
        )
        if engine_factory is None:
            engine_factory = default_engine
        if render_prompt is None:
            render_prompt = default_render
        if client_cancelled_type is None:
            client_cancelled_type = default_cancelled
    if engine_path is None:
        engine_path = protocol["engine_path"]
    environment = _environment_for_treatment(
        treatment,
        environ=os.environ if environ is None else environ,
        state_dir=state_dir,
    )
    prompt = render_prompt(protocol.get("prompt") or BENCHMARK_PROMPT)
    collector = dram_collector or _UnavailableDramCollector(
        "DRAM collector unavailable"
    )
    engine = None
    dram_handle = None
    dram_start_error = None
    cleanup_errors = []
    try:
        if fingerprint_file is not None:
            # This is the last userspace boundary before the engine factory
            # reaches Popen.  Rehash payloads here to close the prestage gap.
            _assert_frozen_artifacts(
                protocol,
                fingerprint_file,
                treatment=treatment,
            )
        engine = engine_factory(
            engine_path,
            treatment["weights_path"],
            cap=int(protocol.get("cache_cap", 8)),
            max_tokens=int(protocol.get("tokens_per_replicate", 32)),
            env=environment,
            kv_slots=1,
            command_prefix=list(treatment["numa_policy"]["command_prefix"]),
            stderr=stderr if stderr is not None else subprocess.DEVNULL,
        )
        reported_environment = getattr(
            engine,
            "benchmark_child_environment",
            None,
        )
        if reported_environment is not None:
            if not isinstance(reported_environment, dict):
                raise RamdiskError("engine reported a malformed child environment")
            reported_environment = {
                str(name): str(value)
                for name, value in reported_environment.items()
            }
            if reported_environment != environment:
                raise RamdiskError(
                    "actual engine child environment differs from the frozen protocol"
                )
            environment = reported_environment
        pid = int(engine.process.pid)
        starttime = process_starttime(pid)
        launch_id = launch_id_factory()
        if not isinstance(launch_id, str) or not launch_id:
            raise RamdiskError("replicate launch identity is invalid")
        _raise_if_cancelled(cancel_event)
        swap_before = int(swap_used_bytes())
        try:
            dram_handle = collector.start(pid, treatment)
        except Exception as error:
            dram_start_error = "DRAM collector start failed: %s" % error
        started_at = utc_now()
        started = monotonic()
        parts = []
        profile_seq = int(getattr(engine, "profile_seq", 0))
        stats = benchmark_generate(
            engine,
            prompt,
            parts.append,
            cancel_event,
            client_cancelled_type,
        )
        elapsed = monotonic() - started
        finished_at = utc_now()
        if dram_start_error is not None:
            dram = _unavailable_dram(dram_start_error)
        else:
            try:
                dram = _normalize_dram_measurement(
                    collector.finish(dram_handle)
                )
            except Exception as error:
                dram = _unavailable_dram(
                    "DRAM collector finish failed: %s" % error
                )
            finally:
                dram_handle = None
        _raise_if_cancelled(cancel_event)
        tokens = int(stats.get("completion_tokens", 0) or 0)
        expected_tokens = int(protocol.get("tokens_per_replicate", 32))
        if tokens != expected_tokens:
            raise RamdiskError(
                "causal replicate produced %d tokens instead of %d"
                % (tokens, expected_tokens)
            )
        profiles = getattr(engine, "profile", None)
        if int(getattr(engine, "profile_seq", 0)) <= profile_seq or not profiles:
            raise RamdiskError("engine did not emit required causal PROF telemetry")
        profile = dict(profiles[-1])
        swap_after = int(swap_used_bytes())
        rss = dict(rss_sampler(pid))
        placement = dict(numa_sampler(pid, treatment["numa_policy"]))
        physical_required = bool(treatment.get("requires_zero_physical_ssd_reads"))
        physical_ok = (
            not physical_required
            or (
                profile.get("physical_ssd_valid") is True
                and profile.get("physical_ssd_bytes") == 0
            )
        )
        output = "".join(parts)
        rate = stats.get("tokens_per_second")
        if rate is None and elapsed > 0:
            rate = float(tokens) / elapsed
        row = {
            "schema": RAW_EVIDENCE_SCHEMA,
            "version": 1,
            "record_type": "replicate",
            "protocol_id": protocol["protocol_id"],
            "treatment_id": treatment["id"],
            "block_index": int(block_index),
            "sequence": int(sequence),
            "started_at": started_at,
            "finished_at": finished_at,
            "status": "ok",
            "process": {
                "launch_id": launch_id,
                "pid": pid,
                "starttime": starttime,
            },
            "applied_environment": _environment_projection(environment),
            "numa_policy": dict(treatment["numa_policy"]),
            "fingerprints": dict(protocol["fingerprints"]),
            "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "profiler": {
                "rammap_experts": profile.get("rammap_experts"),
                "rammap_bytes": profile.get("rammap_bytes"),
                "physical_ssd_bytes": profile.get("physical_ssd_bytes"),
                "physical_ssd_valid": profile.get("physical_ssd_valid"),
            },
            "swap": {
                "before_bytes": swap_before,
                "after_bytes": swap_after,
                "delta_bytes": swap_after - swap_before,
                "process_bytes": rss.pop("process_swap_bytes", None),
            },
            "rss": {
                "anonymous_bytes": rss.get("anonymous_bytes"),
                "file_bytes": rss.get("file_bytes"),
                "shmem_bytes": rss.get("shmem_bytes"),
            },
            "numa_placement": placement,
            "dram_traffic": dram,
            "performance": {
                "tokens_per_second": rate,
                "elapsed_seconds": elapsed,
                "ttft_ms": profile.get("ttft_ms"),
                "forward_p50_ms": profile.get("forward_p50_ms"),
                "forward_p99_ms": profile.get("forward_p99_ms"),
            },
            "correctness": {
                "zero_swap_growth": swap_after <= swap_before,
                "physical_reads_ok": physical_ok,
                "placement_ok": placement.get("verified") is True,
            },
            "error": None,
        }
        if treatment.get("_workspace_attempt") is not None:
            row["workspace_attempt"] = copy.deepcopy(
                treatment["_workspace_attempt"]
            )
        return validate_raw_evidence_row(row)
    finally:
        if dram_handle is not None:
            try:
                collector.finish(dram_handle)
            except Exception:
                pass
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                cleanup_errors.append("engine: %s" % exc)
        if cleanup_errors:
            raise _EngineCleanupError(
                "causal benchmark cleanup failed: %s" % "; ".join(cleanup_errors)
            )


def _percentile_linear(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise RamdiskError("paired interval has no bootstrap samples")
    position = max(0.0, min(1.0, float(probability))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_interval(
    baseline,
    treatment,
    *,
    confidence=0.95,
    seed=377,
    resamples=BOOTSTRAP_RESAMPLES,
):
    """Return a deterministic paired bootstrap interval for relative change."""
    if len(baseline) != len(treatment) or len(baseline) < 2:
        raise RamdiskError("paired interval requires equal paired samples")
    confidence = _finite_number(
        confidence,
        "confidence level",
        minimum=0.5,
        maximum=0.9999,
    )
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1000:
        raise RamdiskError("paired bootstrap requires at least 1000 resamples")
    differences = []
    for before, after in zip(baseline, treatment):
        before = _finite_number(before, "paired baseline", positive=True)
        after = _finite_number(after, "paired treatment", positive=True)
        differences.append((after - before) / before)
    rng = random.Random(seed)
    count = len(differences)
    bootstrapped = []
    for _ in range(resamples):
        bootstrapped.append(
            sum(differences[rng.randrange(count)] for _ in range(count)) / count
        )
    alpha = 1.0 - confidence
    return {
        "method": CI_METHOD,
        "confidence": confidence,
        "pairs": count,
        "point_estimate": sum(differences) / count,
        "lower_bound": _percentile_linear(bootstrapped, alpha / 2.0),
        "upper_bound": _percentile_linear(bootstrapped, 1.0 - alpha / 2.0),
    }


def _workspace_attempt_error(protocol, treatment, row):
    workspace_name = treatment.get("workspace")
    attempt = row.get("workspace_attempt")
    if workspace_name is None:
        return (
            "non-workspace treatment recorded a workspace attempt"
            if attempt is not None
            else None
        )
    if row.get("status") != "ok" and attempt is None:
        return None
    if attempt is None:
        return "workspace treatment omitted its physical attempt binding"
    try:
        _validate_workspace_attempt(attempt)
    except RamdiskError as error:
        return str(error)
    requirements = protocol.get("workspace_requirements") or {}
    expected_roots = requirements.get("roots") or {}
    roots = attempt["roots"]
    for name in ("interleaved", "local"):
        expected = expected_roots.get(name)
        if not isinstance(expected, dict):
            return "logical workspace requirements are incomplete"
        for key in (
            "role",
            "mode",
            "nodes",
            "node",
            "policy",
            "size_bytes",
            "source_fingerprint",
        ):
            if roots[name].get(key) != expected.get(key):
                return "physical workspace differs from frozen logical requirements"
    applied_snap = (row.get("applied_environment") or {}).get("SNAP")
    if applied_snap != roots[workspace_name]["path"]:
        return "physical workspace path differs from the launched child environment"
    return None


def evaluate_causal_claim(protocol, rows):
    """Evaluate correctness/completeness before considering performance."""
    if (
        not isinstance(protocol, dict)
        or not _is_hex_digest(protocol.get("protocol_id"))
        or _causal_protocol_identity(protocol) != protocol["protocol_id"]
    ):
        raise RamdiskError("causal protocol identity is invalid")
    treatment_by_id = {
        treatment["id"]: treatment for treatment in protocol["treatments"]
    }
    expected_schedule = {}
    for block in protocol["randomized_blocks"]:
        block_index = int(block["block_index"])
        expected_schedule[block_index] = {
            treatment_id: block_index * len(treatment_by_id) + position
            for position, treatment_id in enumerate(block["order"])
        }
    reasons_invalid = []
    reasons_incomplete = []
    by_treatment = {identifier: {} for identifier in treatment_by_id}
    seen = set()
    launch_ids = set()
    process_identities = set()
    hashes = set()
    expected_append_order = [
        (int(block["block_index"]), treatment_id, sequence)
        for block in protocol["randomized_blocks"]
        for sequence, treatment_id in enumerate(
            block["order"],
            start=int(block["block_index"]) * len(treatment_by_id),
        )
    ]
    actual_append_order = [
        (row.get("block_index"), row.get("treatment_id"), row.get("sequence"))
        for row in rows
        if row.get("record_type", "replicate") == "replicate"
    ]
    if actual_append_order != expected_append_order[: len(actual_append_order)]:
        reasons_invalid.append(
            "raw evidence append order is not a prefix of the frozen schedule"
        )
    for row in rows:
        validate_raw_evidence_row(row)
        if row.get("record_type") == "workspace-cleanup":
            if row["protocol_id"] != protocol["protocol_id"]:
                reasons_invalid.append(
                    "workspace cleanup row belongs to a different protocol"
                )
            else:
                reasons_incomplete.append(
                    "workspace cleanup failed: %s" % row["error"]
                )
            continue
        if row["protocol_id"] != protocol["protocol_id"]:
            reasons_invalid.append("raw row belongs to a different protocol")
            continue
        treatment_id = row["treatment_id"]
        if treatment_id not in treatment_by_id:
            reasons_invalid.append("unknown treatment %s" % treatment_id)
            continue
        block_index = row["block_index"]
        if block_index not in expected_schedule:
            reasons_invalid.append(
                "unexpected block %d for treatment %s"
                % (block_index, treatment_id)
            )
            continue
        if expected_schedule[block_index].get(treatment_id) != row["sequence"]:
            reasons_invalid.append(
                "sequence does not match the frozen randomized schedule for %s"
                % treatment_id
            )
        key = (row["block_index"], treatment_id)
        if key in seen:
            reasons_invalid.append(
                "duplicate raw row for block %d treatment %s" % key
            )
            continue
        seen.add(key)
        by_treatment[treatment_id][row["block_index"]] = row
        if row["status"] != "ok":
            reasons_invalid.append(
                "treatment %s block %d failed: %s"
                % (treatment_id, row["block_index"], row.get("error"))
            )
            continue
        launch_id = row["process"]["launch_id"]
        if launch_id in launch_ids:
            reasons_invalid.append(
                "causal replicates did not use a globally fresh process"
            )
        launch_ids.add(launch_id)
        process_identity = (
            row["process"]["pid"],
            row["process"]["starttime"],
        )
        if process_identity in process_identities:
            reasons_invalid.append(
                "process identity was reused instead of starting a fresh process"
            )
        process_identities.add(process_identity)
        hashes.add(row["output_sha256"])
        treatment = treatment_by_id[treatment_id]
        workspace_error = _workspace_attempt_error(protocol, treatment, row)
        if workspace_error is not None:
            reasons_invalid.append(
                "%s for %s" % (workspace_error, treatment_id)
            )
        expected_environment = _environment_projection(
            _environment_for_treatment(treatment, environ={})
        )
        applied_environment = dict(row["applied_environment"])
        state_dir = applied_environment.pop("COLI_STATE_DIR", None)
        if state_dir is not None and (
            not isinstance(state_dir, str) or not os.path.isabs(state_dir)
        ):
            reasons_invalid.append(
                "applied environment has an invalid fresh state directory"
            )
        if treatment.get("workspace") is not None:
            expected_environment.pop("SNAP", None)
            applied_environment.pop("SNAP", None)
        if applied_environment != expected_environment:
            reasons_invalid.append(
                "applied environment differs from the frozen treatment for %s"
                % treatment_id
            )
        if row["numa_policy"] != treatment["numa_policy"]:
            reasons_invalid.append(
                "NUMA policy differs from the frozen treatment for %s"
                % treatment_id
            )
        if row["fingerprints"] != protocol["fingerprints"]:
            reasons_invalid.append(
                "fingerprint differs from the frozen protocol for %s"
                % treatment_id
            )
        if row["swap"].get("delta_bytes") is None:
            reasons_incomplete.append("swap measurement is missing")
        elif row["swap"]["delta_bytes"] > 0:
            reasons_invalid.append("swap grew during %s" % treatment_id)
        process_swap = row["swap"].get("process_bytes")
        if process_swap is None:
            reasons_incomplete.append(
                "process swap measurement is missing for %s" % treatment_id
            )
        elif process_swap > 0:
            reasons_invalid.append(
                "process swap was nonzero during %s" % treatment_id
            )
        if row["numa_placement"].get("verified") is not True:
            reasons_invalid.append("NUMA placement failed for %s" % treatment_id)
        profiler = row["profiler"]
        if treatment.get("storage") == "tmpfs-rammap":
            if not (
                profiler.get("rammap_experts") == len(protocol["expert_set"])
                and profiler.get("rammap_bytes")
                == protocol["direct_mapped_bytes"]
            ):
                reasons_invalid.append(
                    "RAMMAP telemetry does not match the frozen expert set for %s"
                    % treatment_id
                )
        elif profiler.get("rammap_experts") not in (0, None):
            reasons_invalid.append(
                "non-RAMMAP treatment emitted RAMMAP telemetry for %s"
                % treatment_id
            )
        if treatment.get("requires_zero_physical_ssd_reads") and not (
            profiler.get("physical_ssd_valid") is True
            and profiler.get("physical_ssd_bytes") == 0
        ):
            reasons_invalid.append(
                "full RAMMAP physical reads were not zero for %s" % treatment_id
            )
        expected_physical_ok = (
            not treatment.get("requires_zero_physical_ssd_reads")
            or (
                profiler.get("physical_ssd_valid") is True
                and profiler.get("physical_ssd_bytes") == 0
            )
        )
        if row["correctness"].get("physical_reads_ok") is not expected_physical_ok:
            reasons_invalid.append(
                "physical-read correctness projection is inconsistent for %s"
                % treatment_id
            )
        dram = row["dram_traffic"]
        if dram.get("available") is not True:
            reasons_incomplete.append("DRAM counters unavailable for %s" % treatment_id)
        else:
            expected_collector = protocol.get("dram_collector") or {}
            if (
                expected_collector.get("available") is not True
                or dram.get("collector_identity")
                != expected_collector.get("collector_identity")
                or dram.get("collector") != expected_collector.get("collector")
            ):
                reasons_invalid.append(
                    "DRAM collector differs from the frozen protocol for %s"
                    % treatment_id
                )
        for name in ("anonymous_bytes", "file_bytes", "shmem_bytes"):
            if row["rss"].get(name) is None:
                reasons_incomplete.append("RSS %s is missing for %s" % (name, treatment_id))
        performance = row["performance"]
        for name in (
            "tokens_per_second",
            "elapsed_seconds",
            "ttft_ms",
            "forward_p50_ms",
            "forward_p99_ms",
        ):
            if performance.get(name) is None:
                reasons_incomplete.append(
                    "performance %s is missing for %s" % (name, treatment_id)
                )
    if len(hashes) > 1:
        reasons_invalid.append("deterministic output hashes differ across treatments")

    repetitions = int(protocol["repetitions"])
    for treatment_id, blocks in by_treatment.items():
        expected = set(range(repetitions))
        missing = sorted(expected - set(blocks))
        if missing:
            reasons_incomplete.append(
                "treatment %s is missing blocks %s" % (treatment_id, missing)
            )
        launch_ids = {
            row["process"].get("launch_id")
            for row in blocks.values()
            if row["status"] == "ok"
        }
        launch_ids.discard(None)
        if len(launch_ids) != len(
            [row for row in blocks.values() if row["status"] == "ok"]
        ):
            reasons_invalid.append(
                "treatment %s did not use a fresh process per replicate"
                % treatment_id
            )

    comparisons = {}
    if not reasons_invalid and not reasons_incomplete:
        for topology, baseline_id, treatment_id in (
            (
                "interleaved",
                "anon-pin-interleaved",
                "tmpfs-rammap-interleaved",
            ),
            ("local", "anon-pin-local", "tmpfs-rammap-local"),
        ):
            baseline = [
                by_treatment[baseline_id][block]["performance"]["tokens_per_second"]
                for block in range(repetitions)
            ]
            treatment = [
                by_treatment[treatment_id][block]["performance"]["tokens_per_second"]
                for block in range(repetitions)
            ]
            comparisons[topology] = paired_interval(
                baseline,
                treatment,
                confidence=protocol["predeclared"]["confidence"],
                seed=protocol["seed"] + (0 if topology == "interleaved" else 1),
                resamples=protocol["predeclared"]["bootstrap_resamples"],
            )

    if reasons_invalid:
        status = "invalid"
        claim = "neutral"
        reasons = sorted(set(reasons_invalid + reasons_incomplete))
    elif reasons_incomplete:
        status = "incomplete"
        claim = "neutral"
        reasons = sorted(set(reasons_incomplete))
    else:
        status = "complete"
        threshold = protocol["predeclared"]["practical_threshold"]
        claim = (
            "improvement"
            if comparisons
            and all(
                interval["lower_bound"] > threshold
                for interval in comparisons.values()
            )
            else "neutral"
        )
        reasons = [] if claim == "improvement" else [
            "paired lower bound did not clear the practical threshold"
        ]
    return {
        "schema": BENCHMARK_SCHEMA,
        "version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": status,
        "claim": claim,
        "direction": protocol["predeclared"]["direction"],
        "practical_threshold": protocol["predeclared"]["practical_threshold"],
        "comparisons": comparisons,
        "reasons": reasons,
        "replicates_by_treatment": {
            treatment_id: len(blocks)
            for treatment_id, blocks in by_treatment.items()
        },
    }


def _causal_result_from_rows(
    protocol,
    rows,
    *,
    raw_evidence_path,
    protocol_path,
    dram_collector,
):
    result = evaluate_causal_claim(protocol, rows)
    replicate_rows = [
        row
        for row in rows
        if row.get("record_type", "replicate") == "replicate"
    ]
    result.update(
        {
            "attempted_replicates": len(replicate_rows),
            "successful_replicates": len(
                [row for row in replicate_rows if row.get("status") == "ok"]
            ),
            "raw_evidence_path": os.path.abspath(raw_evidence_path),
            "protocol_path": os.path.abspath(protocol_path),
            "dram_collector": dict(
                getattr(dram_collector, "description", {}) or {}
            ),
        }
    )
    return result


def _execute_causal_schedule(
    protocol,
    *,
    raw_evidence_path,
    protocol_path,
    prestage,
    replicate_runner,
    dram_collector,
    load_rows,
    append,
    cancel_event=None,
    final_reload=True,
):
    rows = list(load_rows())
    for row in rows:
        validate_raw_evidence_row(row)
        if row["protocol_id"] != protocol["protocol_id"]:
            raise RamdiskError("raw evidence path contains a different protocol")
    expected_prefix = [
        (int(block["block_index"]), treatment_id, sequence)
        for block in protocol["randomized_blocks"]
        for sequence, treatment_id in enumerate(
            block["order"],
            start=int(block["block_index"]) * len(protocol["treatments"]),
        )
    ]
    replicate_rows = [
        row
        for row in rows
        if row.get("record_type", "replicate") == "replicate"
    ]
    actual_prefix = [
        (row["block_index"], row["treatment_id"], row["sequence"])
        for row in replicate_rows
    ]
    if actual_prefix != expected_prefix[: len(actual_prefix)]:
        raise RamdiskError(
            "raw evidence replicates are not a prefix of the frozen schedule"
        )
    completed = {
        (row["block_index"], row["treatment_id"])
        for row in rows
        if row.get("record_type", "replicate") == "replicate"
    }
    treatment_by_id = {
        treatment["id"]: treatment for treatment in protocol["treatments"]
    }
    for block in protocol["randomized_blocks"]:
        block_index = int(block["block_index"])
        for position, treatment_id in enumerate(block["order"]):
            key = (block_index, treatment_id)
            if key in completed:
                continue
            _raise_if_cancelled(cancel_event)
            treatment = treatment_by_id[treatment_id]
            execution_treatment = treatment
            sequence = block_index * len(treatment_by_id) + position
            started_at = _utc_now()
            try:
                prepared = prestage(protocol, treatment)
                if prepared is not None:
                    if (
                        not isinstance(prepared, dict)
                        or prepared.get("id") != treatment["id"]
                    ):
                        raise RamdiskError(
                            "causal prestage returned an invalid treatment binding"
                        )
                    execution_treatment = prepared
                row = replicate_runner(
                    protocol,
                    execution_treatment,
                    block_index,
                    sequence,
                    dram_collector=dram_collector,
                    cancel_event=cancel_event,
                )
            except (
                _EngineCleanupError,
                _OperationCancelled,
                WorkspaceVerificationError,
            ) as exc:
                row = failed_raw_evidence_row(
                    protocol,
                    execution_treatment,
                    block_index=block_index,
                    sequence=sequence,
                    error=str(exc),
                    started_at=started_at,
                )
                validate_raw_evidence_row(row)
                append(row)
                rows.append(row)
                raise
            except Exception as exc:
                row = failed_raw_evidence_row(
                    protocol,
                    execution_treatment,
                    block_index=block_index,
                    sequence=sequence,
                    error=str(exc),
                    started_at=started_at,
                )
            validate_raw_evidence_row(row)
            append(row)
            rows.append(row)
            completed.add(key)
    # Claims are based only on the exact durable bytes which survived every
    # append and cleanup boundary, never on the optimistic in-memory list.
    if final_reload:
        rows = list(load_rows())
    return _causal_result_from_rows(
        protocol,
        rows,
        raw_evidence_path=raw_evidence_path,
        protocol_path=protocol_path,
        dram_collector=dram_collector,
    )


def _existing_complete_causal_result(
    protocol,
    *,
    raw_evidence_path,
    protocol_path,
    dram_collector,
    assert_durable,
    recover=None,
):
    """Return descriptor-bound complete evidence without allocating scratch."""
    if not (
        os.path.lexists(raw_evidence_path)
        and os.path.lexists(protocol_path)
    ):
        return None
    with DurableEvidenceStore(
        raw_evidence_path,
        protocol_path,
        assert_durable=assert_durable,
    ) as evidence:
        bound_protocol = evidence.bind_protocol(protocol)
        rows = evidence.read_rows()
        result = _causal_result_from_rows(
            bound_protocol,
            rows,
            raw_evidence_path=raw_evidence_path,
            protocol_path=protocol_path,
            dram_collector=dram_collector,
        )
        if result["status"] != "complete":
            return None
        if callable(recover):
            recover()
            recovered_rows = evidence.read_rows()
            if recovered_rows != rows:
                raise RamdiskError(
                    "completed causal evidence changed during workspace recovery"
                )
    return result


def run_causal_benchmark(
    protocol,
    *,
    raw_evidence_path,
    prestage,
    replicate_runner,
    dram_collector,
    protocol_path=None,
    persist_protocol=persist_causal_protocol,
    existing_rows=read_raw_evidence,
    append_row=append_raw_evidence,
    cancel_event=None,
    evidence_store_factory=None,
    assert_durable=None,
):
    """Execute/resume against one descriptor-bound durable evidence store."""
    raw_evidence_path = os.fspath(raw_evidence_path)
    protocol_path = (
        os.fspath(protocol_path)
        if protocol_path is not None
        else raw_evidence_path + ".protocol.json"
    )
    raw_evidence_path, protocol_path = _validate_evidence_paths(
        raw_evidence_path,
        protocol_path,
    )
    use_bound_store = (
        evidence_store_factory is not None
        or (
            persist_protocol is persist_causal_protocol
            and existing_rows is read_raw_evidence
            and append_row is append_raw_evidence
        )
    )
    if use_bound_store:
        factory = evidence_store_factory or DurableEvidenceStore
        with factory(
            raw_evidence_path,
            protocol_path,
            assert_durable=assert_durable,
        ) as store:
            protocol = store.bind_protocol(protocol)
            return _execute_causal_schedule(
                protocol,
                raw_evidence_path=raw_evidence_path,
                protocol_path=protocol_path,
                prestage=prestage,
                replicate_runner=replicate_runner,
                dram_collector=dram_collector,
                load_rows=store.read_rows,
                append=store.append,
                cancel_event=cancel_event,
                final_reload=True,
            )

    # Dependency-injected unit seams retain their historical callback shape.
    protocol = persist_protocol(protocol_path, protocol)
    return _execute_causal_schedule(
        protocol,
        raw_evidence_path=raw_evidence_path,
        protocol_path=protocol_path,
        prestage=prestage,
        replicate_runner=replicate_runner,
        dram_collector=dram_collector,
        load_rows=lambda: existing_rows(raw_evidence_path),
        append=lambda row: append_row(raw_evidence_path, row),
        cancel_event=cancel_event,
        final_reload=False,
    )


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
    source_build_identity,
    fingerprint_file,
    state_root,
    ensure_private_dir,
    assert_durable_state_dir,
    admit_runtime,
    fresh_user_binary,
    workspace_manager,
    environ=None,
    dram_preflight=preflight_dram_collector,
    causal_runner=run_causal_benchmark,
    replicate_runner=run_causal_replicate,
    freeze_engine_environment=_production_engine_environment_defaults,
    fingerprint_model_content=_model_content_fingerprints,
):
    """Build and execute a causal protocol from the reviewed manifest."""
    manifest = load_manifest(required=True)
    _raise_if_cancelled(cancel_event)
    if manifest.get("state") not in ("ready", "stopped"):
        raise RamdiskError(
            "stop managed engines before the causal benchmark"
        )
    assert_effective_masks_unchanged(manifest["plan"])
    assert_ready_mounts(manifest)
    resolved_engine = resolve_engine_path(cli_path, engine_path)
    profile_path = getattr(args, "evidence_profile", None) or (
        manifest["plan"].get("profile") or {}
    ).get("path")
    raw_path = getattr(args, "raw_evidence", None) or os.path.join(
        state_root(),
        "causal-evidence",
        "raw.v1.jsonl",
    )
    environment = os.environ if environ is None else environ
    plan = manifest["plan"]
    protocol_path = os.fspath(raw_path) + ".protocol.json"
    manifest_path = environment.get("COLI_RAMDISK_MANIFEST") or os.path.join(
        state_root(), "manifest.json"
    )
    reserved_paths = [
        manifest_path,
        os.path.join(state_root(), "benchmarks.json"),
        os.path.join(state_root(), "lifecycle.lock"),
        os.path.join(state_root(), "usage.lock"),
        resolved_engine,
        profile_path,
        (plan.get("model") or {}).get("path"),
    ]
    reserved_paths.extend(
        item.get("path")
        for item in plan.get("source_shards") or []
        if isinstance(item, dict)
    )
    raw_path, protocol_path = _validate_evidence_paths(
        raw_path,
        protocol_path,
        reserved_paths=reserved_paths,
    )
    _require_evidence_root(
        raw_path,
        protocol_path,
        os.path.join(state_root(), "causal-evidence"),
    )
    evidence_parent = _safe_parent(raw_path)
    assert_durable_state_dir(evidence_parent, plan=plan)
    numactl = fresh_user_binary("numactl")
    collector = dram_preflight(environ=environment)
    engine_environment_defaults = freeze_engine_environment(
        plan["model"]["path"],
        max_tokens=32,
        kv_slots=1,
    )
    model_content_fingerprints = fingerprint_model_content(
        plan,
        fingerprint_file,
    )
    draft_protocol = build_causal_protocol(
        manifest,
        engine_path=resolved_engine,
        profile_path=profile_path,
        residency_gib=getattr(args, "residency_gb", None),
        cuda_host_gib=getattr(args, "cuda_host_gb", None),
        cuda_expert_gib=getattr(args, "cuda_expert_gb", None),
        repetitions=getattr(args, "replicates", MIN_REPETITIONS),
        seed=getattr(args, "seed", 377),
        practical_threshold=getattr(args, "practical_threshold", 0.05),
        confidence=getattr(args, "confidence", 0.95),
        ci_method=CI_METHOD,
        fingerprint_file=fingerprint_file,
        source_identity=source_build_identity(),
        numactl=numactl,
        inherited_environment=environment,
        dram_collector=dict(getattr(collector, "description", {}) or {}),
        engine_environment_defaults=engine_environment_defaults,
        model_content_fingerprints=model_content_fingerprints,
    )
    if causal_runner is run_causal_benchmark:
        durable_check = lambda path: assert_durable_state_dir(
            path,
            plan=plan,
        )
        completed = _existing_complete_causal_result(
            draft_protocol,
            raw_evidence_path=raw_path,
            protocol_path=protocol_path,
            dram_collector=collector,
            assert_durable=durable_check,
            recover=(
                (lambda: workspace_manager.recover(manifest))
                if callable(getattr(workspace_manager, "recover", None))
                else None
            ),
        )
        if completed is not None:
            return completed
    mount_by_path = {
        record.get("path"): record for record in manifest.get("mounts", [])
    }
    protocol = None
    workspace_by_path = {}

    try:
        with workspace_manager.open(
            manifest,
            draft_protocol,
            cancel_event,
        ) as yielded_roots:
            roots = _validate_workspace_roots(
                workspace_manager, yielded_roots, draft_protocol
            )
            execution_protocol = _bind_workspace_roots(draft_protocol, roots)
            execution_by_id = {
                item["id"]: item for item in execution_protocol["treatments"]
            }
            protocol = draft_protocol
            binder = getattr(workspace_manager, "bind_protocol", None)
            if not callable(binder):
                raise RamdiskError(
                    "durable benchmark workspace cannot bind the final protocol"
                )
            binder(protocol)
            workspace_by_path = {
                descriptor["path"]: descriptor
                for descriptor in roots.values()
            }

            def prestage(current_protocol, treatment):
                assert_ready_mounts(manifest)
                execution_treatment = execution_by_id[treatment["id"]]
                _assert_frozen_artifacts(
                    current_protocol,
                    fingerprint_file,
                    treatment=execution_treatment,
                )
                weights = execution_treatment["weights_path"]
                workspace = workspace_by_path.get(weights)
                if workspace is not None and workspace_manager.verify(workspace) is not True:
                    raise WorkspaceVerificationError(
                        "causal benchmark workspace changed before %s"
                        % treatment["id"]
                    )
                if not os.path.isdir(weights):
                    raise RamdiskError(
                        "prestage path is missing for %s: %s"
                        % (treatment["id"], weights)
                    )
                for shard in plan.get("staging", {}).get("selected_shards", []):
                    candidate = os.path.join(weights, shard)
                    if not os.path.exists(candidate):
                        raise RamdiskError(
                            "prestage shard is missing for %s: %s"
                            % (treatment["id"], candidate)
                        )
                mount = mount_by_path.get(weights)
                if mount is None and workspace is not None:
                    mount = {
                        "path": weights,
                        "node": 0 if workspace["mode"] == "local" else None,
                    }
                if mount is None:
                    mount = (manifest.get("mounts") or [{}])[0]
                admit_runtime(
                    plan,
                    mount,
                    benchmark=treatment["storage"] != "tmpfs-rammap",
                )
                return execution_treatment

            def run_one(
                current_protocol,
                treatment,
                block_index,
                sequence,
                **kwargs
            ):
                state_dir = os.path.join(
                    state_root(),
                    "causal-benchmark-state",
                    current_protocol["protocol_id"],
                    "%04d-%03d-%s-%s"
                    % (
                        block_index,
                        sequence,
                        treatment["id"],
                        secrets.token_hex(8),
                    ),
                )
                ensure_private_dir(state_dir)
                assert_durable_state_dir(state_dir, plan=plan)
                return replicate_runner(
                    current_protocol,
                    treatment,
                    block_index,
                    sequence,
                    engine_path=resolved_engine,
                    state_dir=state_dir,
                    environ=environment,
                    fingerprint_file=fingerprint_file,
                    **kwargs
                )

            return causal_runner(
                protocol,
                raw_evidence_path=raw_path,
                protocol_path=protocol_path,
                prestage=prestage,
                replicate_runner=run_one,
                dram_collector=collector,
                cancel_event=cancel_event,
                assert_durable=lambda path: assert_durable_state_dir(
                    path,
                    plan=plan,
                ),
            )
    except WorkspaceCleanupError as exc:
        if protocol is None:
            raise
        event = workspace_cleanup_evidence_row(protocol, exc)
        with DurableEvidenceStore(
            raw_path,
            protocol_path,
            assert_durable=lambda path: assert_durable_state_dir(
                path,
                plan=plan,
            ),
        ) as evidence:
            protocol = evidence.bind_protocol(protocol)
            evidence.append(event)
            rows = evidence.read_rows()
        result = evaluate_causal_claim(protocol, rows)
        replicate_rows = [
            row
            for row in rows
            if row.get("record_type", "replicate") == "replicate"
        ]
        result.update(
            {
                "attempted_replicates": len(replicate_rows),
                "successful_replicates": len(
                    [row for row in replicate_rows if row.get("status") == "ok"]
                ),
                "raw_evidence_path": os.path.abspath(raw_path),
                "protocol_path": os.path.abspath(raw_path + ".protocol.json"),
                "dram_collector": dict(
                    getattr(collector, "description", {}) or {}
                ),
                "workspace_cleanup": {
                    "status": "error",
                    "error": str(exc),
                },
            }
        )
        return result
