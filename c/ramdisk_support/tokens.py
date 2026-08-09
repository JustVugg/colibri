"""Canonical, dependency-free authorization tokens for RAM-disk mutations."""

from __future__ import print_function

import hashlib
import json
import re

from .common import RamdiskError


TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_plan_projection(plan):
    """Bind stable mutation authority while excluding volatile observations."""
    reserve = plan.get("reserve") or {}
    hardware = plan.get("hardware") or {}
    stable_reserve = {
        name: value
        for name, value in reserve.items()
        if name
        not in {
            "available_bytes",
            "host_available_bytes",
            "cgroup_available_bytes",
            "cgroup_high_available_bytes",
        }
    }
    return {
        "schema": plan.get("schema"),
        "version": plan.get("version"),
        "model": plan.get("model"),
        "source_shards": plan.get("source_shards"),
        "mode": plan.get("mode"),
        "topology": plan.get("topology"),
        "hardware": {
            "effective_mask_source": hardware.get("effective_mask_source"),
        },
        "placement": plan.get("placement"),
        "mount_root": plan.get("mount_root"),
        "mount_root_preexisting": plan.get("mount_root_preexisting"),
        "capacity_bytes": plan.get("capacity_bytes"),
        "staging": plan.get("staging"),
        "reserve": stable_reserve,
        "mounts": plan.get("mounts"),
        "mount_options": plan.get("mount_options"),
        "prefault": plan.get("prefault"),
        "parallel": plan.get("parallel"),
        "managed_runtime": plan.get("managed_runtime"),
        "managed_accelerator": plan.get("managed_accelerator"),
        "preset": plan.get("preset"),
        "durable_state": plan.get("durable_state"),
    }


def _hash_projection(projection):
    payload = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def plan_token(plan):
    """Return the lowercase SHA-256 identity of one reviewed plan."""
    return _hash_projection(canonical_plan_projection(plan))


def canonical_deployment_projection(manifest, *, persisted_base_port):
    """Bind every persisted field that can authorize a live mutation.

    Process commands, nonces, paths, and containment identities remain private
    because only their canonical hash leaves this module.  Binding the full
    validated authority records prevents a stale token from approving a
    different signal, cgroup-removal, mount, usage, or recovery transaction.
    """
    return {
        "version": manifest.get("version"),
        "deployment_id": manifest.get("deployment_id"),
        "created_at": manifest.get("created_at"),
        "state": manifest.get("state"),
        "base_port": persisted_base_port(manifest),
        "model_fingerprint": manifest.get("model_fingerprint"),
        "plan_token": plan_token(manifest.get("plan", {})),
        "process_supervision_version": manifest.get(
            "process_supervision_version"
        ),
        "mounts": manifest.get("mounts", []),
        "processes": manifest.get("processes", []),
        "pending_launches": manifest.get("pending_launches", []),
        "recovery": manifest.get("recovery"),
        "benchmark_workspace": manifest.get("benchmark_workspace"),
        "best_runtime": manifest.get("best_runtime"),
    }


def deployment_token(manifest, *, persisted_base_port):
    """Return the lowercase SHA-256 identity of one deployment snapshot."""
    return _hash_projection(
        canonical_deployment_projection(
            manifest,
            persisted_base_port=persisted_base_port,
        )
    )


def validate_token(value, label="token"):
    """Return a token or reject any non-canonical spelling."""
    if not isinstance(value, str) or TOKEN_RE.fullmatch(value) is None:
        raise RamdiskError(
            "%s must be exactly 64 lowercase hexadecimal characters" % label
        )
    return value


# Preserve the donor's private helper spellings for Python callers while the
# public CLI and JSON contracts consistently use plan/deployment terminology.
_plan_confirmation_token = plan_token
_manifest_confirmation_token = deployment_token
