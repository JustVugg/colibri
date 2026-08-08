"""Canonical, dependency-free authorization tokens for RAM-disk mutations."""

from __future__ import print_function

import hashlib
import json
import re

from .common import RamdiskError


TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_plan_projection(plan):
    """Return exactly the reviewed fields bound by a plan token."""
    hardware = plan.get("hardware") or {}
    return {
        "schema": plan.get("schema"),
        "version": plan.get("version"),
        "model_fingerprint": plan.get("model", {}).get("fingerprint"),
        "mode": plan.get("mode"),
        "topology": plan.get("topology"),
        "hardware": {
            "effective_mask_source": hardware.get("effective_mask_source"),
        },
        "placement": plan.get("placement"),
        "mount_root": plan.get("mount_root"),
        "capacity_bytes": plan.get("capacity_bytes"),
        "selected_shards": plan.get("staging", {}).get(
            "selected_shards"
        ),
        "linked_shards": plan.get("staging", {}).get("linked_shards"),
        "total_staged_bytes": plan.get("staging", {}).get(
            "total_staged_bytes"
        ),
        "total_required_bytes": plan.get("reserve", {}).get(
            "total_required_bytes"
        ),
        "mounts": plan.get("mounts"),
        "mount_options": plan.get("mount_options"),
        "prefault": plan.get("prefault"),
        "parallel": plan.get("parallel"),
        "managed_runtime": plan.get("managed_runtime"),
        "managed_accelerator": plan.get("managed_accelerator"),
        "preset": plan.get("preset"),
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
    """Return exactly the deployment fields bound by destructive review."""
    mounts = []
    for record in manifest.get("mounts", []):
        identity = record.get("identity", {})
        mounts.append(
            {
                "path": record.get("path"),
                "node": record.get("node"),
                "mount_id": identity.get("mount_id"),
                "device": identity.get("device"),
            }
        )
    processes = []
    for record in manifest.get("processes", []):
        processes.append(
            {
                "pid": record.get("pid"),
                "pgid": record.get("pgid"),
                "uid": record.get("uid"),
                "starttime": record.get("starttime"),
                "nonce": record.get("nonce"),
                "port": record.get("port"),
                "node": record.get("node"),
            }
        )
    return {
        "version": manifest.get("version"),
        "deployment_id": manifest.get("deployment_id"),
        "created_at": manifest.get("created_at"),
        "state": manifest.get("state"),
        "base_port": persisted_base_port(manifest),
        "model_fingerprint": manifest.get("model_fingerprint"),
        "plan_token": plan_token(manifest.get("plan", {})),
        "mounts": mounts,
        "processes": processes,
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
