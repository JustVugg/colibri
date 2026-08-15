"""Read-only serving and accelerator telemetry for headless status clients.

``RuntimeMonitor.sample`` returns a dependency-free plain dictionary:

.. code-block:: text

    {
      "service": {
        "state": "stopped|starting|serving|degraded",
        "label": str,
        "endpoints": [dict],
        "active": int | None,
        "queued": int | None,
        "error": str | None,
        "stale": bool,
        "observed_at": float | None,
      },
      "gpus": [{
        "index": int,
        "uuid": str | None,
        "pci_bus_id": str,
        "name": str,
        "selected": bool,
        "numa_node": int | None,
        "utilization_percent": float | None,
        "memory_used_bytes": int | None,
        "memory_free_bytes": int | None,
        "memory_total_bytes": int | None,
        "process_vram_bytes": int | None,
        "model_resident_bytes": int | None,
        "expert_bytes": int | None,
        "expert_count": int | None,
        "non_expert_bytes": int | None,
        "card_stale": bool,
        "model_stale": bool,
        "process_stale": bool,
        "observed_at": float | None,
      }],
      "tiers": dict | None,
      "tiers_stale": bool,
      "latest_profile": dict | None,
      "profile_stale": bool,
      "process_rss_bytes": int | None,
      "process_stale": bool,
      "freshness": {
        "service|cards|model|tiers|profile|process": {
          "stale": bool, "observed_at": float | None, "error": str | None
        }
      },
    }

All HTTP URLs are synthesized from validated manifest ports and the numeric
loopback address. Serialized host names or URLs are never used. Endpoint and
compute-process attribution happen only after every live readable member of
the managed process group has passed the persisted UID, nonce, and path
checks. Proven inert members must still retain the exact managed UID, process
group, and session, and are excluded from runtime sampling.
"""

from __future__ import print_function

import csv
import copy
import importlib
import io
import json
import math
import os
import subprocess
import threading
import time

MIB = 1024 * 1024
_MAX_RESPONSE_BYTES = 2 * MIB


_LOOPBACK_OPENER = None
_LOOPBACK_OPENER_LOCK = threading.Lock()


def _processes_module():
    return importlib.import_module("ramdisk_support.processes")


def _process_matches(*args, **kwargs):
    return _processes_module()._process_matches(*args, **kwargs)


def _process_group_members(*args, **kwargs):
    return _processes_module()._process_group_members(*args, **kwargs)


def _managed_process_metrics(*args, **kwargs):
    return _processes_module()._managed_process_metrics(*args, **kwargs)


def _build_loopback_opener():
    """Build an HTTP-only client with neither proxies nor redirects."""
    request_module = importlib.import_module("urllib.request")

    class _NoRedirect(request_module.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            del req, fp, code, msg, headers, newurl
            return None

    opener = request_module.OpenerDirector()
    for handler in (
        request_module.UnknownHandler(),
        request_module.HTTPHandler(),
        request_module.HTTPDefaultErrorHandler(),
        _NoRedirect(),
        request_module.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener


def _loopback_opener():
    global _LOOPBACK_OPENER
    if _LOOPBACK_OPENER is None:
        with _LOOPBACK_OPENER_LOCK:
            if _LOOPBACK_OPENER is None:
                _LOOPBACK_OPENER = _build_loopback_opener()
    return _LOOPBACK_OPENER


def _loopback_urlopen(request, timeout):
    parse_module = importlib.import_module("urllib.parse")
    url = getattr(request, "full_url", request)
    parsed = parse_module.urlsplit(str(url))
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("invalid loopback telemetry port") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or port is None
        or not 1 <= port <= 65535
        or parsed.fragment
    ):
        raise ValueError(
            "telemetry requests require an HTTP numeric-loopback URL with a port"
        )
    return _loopback_opener().open(request, timeout=timeout)


def _http_request(url, headers, method):
    request_module = importlib.import_module("urllib.request")
    return request_module.Request(url, headers=headers, method=method)


def _safe_int(value):
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _text(value):
    if value is None:
        return ""
    return str(value).strip()


def _optional_text(value):
    value = _text(value)
    if not value or value.lower() in ("n/a", "na", "none", "[not supported]"):
        return None
    return value


def _mib_bytes(value):
    amount = _safe_float(value)
    if amount is None or amount < 0:
        return None
    return int(amount * MIB)


def _freshness(stale=True, observed_at=None, error=None):
    return {
        "stale": bool(stale),
        "observed_at": observed_at,
        "error": error,
    }


def _error_text(error):
    text = str(error).strip()
    return text or type(error).__name__


def _gpu_key(record):
    uuid = _optional_text(record.get("uuid"))
    if uuid:
        return ("uuid", uuid.lower())
    pci = _optional_text(
        record.get("pci_bus_id")
        or record.get("pci")
        or record.get("bus_id")
    )
    if pci:
        return ("pci", pci.lower())
    index = _safe_int(record.get("index"))
    if index is not None:
        return ("index", index)
    return None


def _gpu_aliases(record):
    aliases = []
    uuid = _optional_text(record.get("uuid"))
    if uuid:
        aliases.append(("uuid", uuid.lower()))
    pci = _optional_text(
        record.get("pci_bus_id")
        or record.get("pci")
        or record.get("bus_id")
    )
    if pci:
        aliases.append(("pci", pci.lower()))
    index = _safe_int(record.get("index"))
    if index is not None:
        aliases.append(("index", index))
    return tuple(aliases)


def _same_gpu_record(left, right):
    left_uuid = _optional_text(left.get("uuid"))
    right_uuid = _optional_text(right.get("uuid"))
    if left_uuid and right_uuid:
        return left_uuid.lower() == right_uuid.lower()
    left_pci = _optional_text(
        left.get("pci_bus_id")
        or left.get("pci")
        or left.get("bus_id")
    )
    right_pci = _optional_text(
        right.get("pci_bus_id")
        or right.get("pci")
        or right.get("bus_id")
    )
    if left_pci and right_pci:
        return left_pci.lower() == right_pci.lower()
    if (left_uuid or left_pci) and (right_uuid or right_pci):
        return False
    left_index = _safe_int(left.get("index"))
    right_index = _safe_int(right.get("index"))
    return (
        left_index is not None
        and right_index is not None
        and left_index == right_index
    )


def _blank_gpu(record):
    return {
        "index": _safe_int(record.get("index")),
        "uuid": _optional_text(record.get("uuid")),
        "pci_bus_id": _text(
            record.get("pci_bus_id")
            or record.get("pci")
            or record.get("bus_id")
        ),
        "name": _text(record.get("name")),
        "selected": bool(record.get("selected")),
        "numa_node": _safe_int(record.get("numa_node")),
        "utilization_percent": None,
        "memory_used_bytes": None,
        "memory_free_bytes": None,
        "memory_total_bytes": None,
        "process_vram_bytes": None,
        "model_resident_bytes": None,
        "expert_bytes": None,
        "expert_count": None,
        "non_expert_bytes": None,
        "card_stale": True,
        "model_stale": True,
        "process_stale": True,
        "observed_at": None,
    }


def _planned_gpus(plan, hardware):
    accelerator = (plan or {}).get("managed_accelerator") or {}
    selected = list(accelerator.get("devices") or ())
    rows = []
    for source in (
        list((hardware or {}).get("gpus") or ()),
        selected,
    ):
        for item in source:
            if not isinstance(item, dict):
                continue
            key = _gpu_key(item)
            if key is None:
                continue
            row = next(
                (
                    candidate
                    for candidate in rows
                    if _same_gpu_record(candidate, item)
                ),
                None,
            )
            if row is None:
                row = _blank_gpu(item)
                rows.append(row)
            for field, aliases in (
                ("uuid", ("uuid",)),
                ("pci_bus_id", ("pci_bus_id", "pci", "bus_id")),
                ("name", ("name",)),
                ("numa_node", ("numa_node",)),
            ):
                for alias in aliases:
                    value = item.get(alias)
                    if value not in (None, ""):
                        row[field] = (
                            _safe_int(value)
                            if field == "numa_node"
                            else _text(value)
                        )
                        break
            index = _safe_int(item.get("index"))
            if index is not None:
                row["index"] = index
            row["selected"] = any(
                isinstance(candidate, dict)
                and _same_gpu_record(row, candidate)
                for candidate in selected
            )
    return rows


def _merge_gpu_identity(target, source):
    for field in ("index", "uuid", "pci_bus_id", "name", "numa_node"):
        value = source.get(field)
        if value not in (None, ""):
            target[field] = value


def _normalize_health_gpu_rows(payload):
    raw = payload.get("gpus")
    if raw is None:
        raw = payload.get("gpu_details")
    if isinstance(raw, dict):
        raw = raw.get("devices") or raw.get("gpus")
    if not isinstance(raw, list):
        return []
    normalized = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        identity = _optional_text(item.get("identity"))
        identity_uuid = (
            identity if identity and identity.upper().startswith("GPU-") else None
        )
        identity_pci = (
            identity
            if identity
            and not identity.upper().startswith("GPU-")
            and ":" in identity
            else None
        )
        model_bytes = _safe_int(
            item.get(
                "model_resident_bytes",
                item.get(
                    "resident_model_bytes",
                    item.get("model_bytes"),
                ),
            )
        )
        expert_bytes = _safe_int(
            item.get(
                "expert_bytes",
                item.get("expert_resident_bytes"),
            )
        )
        expert_count = _safe_int(item.get("expert_count"))
        non_expert_bytes = _safe_int(
            item.get(
                "non_expert_bytes",
                item.get("nonexpert_bytes"),
            )
        )
        exact = (
            all(
                value is not None and value >= 0
                for value in (
                    model_bytes,
                    expert_bytes,
                    expert_count,
                    non_expert_bytes,
                )
            )
            and expert_bytes + non_expert_bytes == model_bytes
        )
        row = {
            "index": _safe_int(
                item.get(
                    "index",
                    item.get("device_index", item.get("device")),
                )
            ),
            "uuid": _optional_text(item.get("uuid")) or identity_uuid,
            "pci_bus_id": _text(
                item.get("pci_bus_id")
                or item.get("pci")
                or item.get("bus_id")
                or identity_pci
            ),
            "name": _text(item.get("name")),
            "numa_node": _safe_int(item.get("numa_node")),
            "model_resident_bytes": model_bytes,
            "expert_bytes": expert_bytes,
            "expert_count": expert_count,
            "non_expert_bytes": non_expert_bytes,
            "_exact": exact,
        }
        if _gpu_key(row) is not None:
            normalized.append(row)
    return normalized


def _bind_model_rows_to_plan(plan, rows):
    """Resolve logical CUDA ordinals through a reviewed launch mapping."""
    devices = list(
        (
            ((plan or {}).get("managed_accelerator") or {}).get("devices")
            or ()
        )
    )
    if not devices:
        return rows
    if not _plan_has_safe_gpu_mapping(plan):
        return []
    by_ordinal = {
        _safe_int(device.get("cuda_ordinal")): device
        for device in devices
    }
    bound = []
    for source in rows:
        row = dict(source)
        device = by_ordinal.get(_safe_int(row.get("index")))
        if device is None:
            continue
        has_identity = bool(
            _optional_text(row.get("uuid"))
            or _optional_text(row.get("pci_bus_id"))
        )
        if has_identity and not _same_gpu_record(device, row):
            # A logical ordinal claiming another physical identity is not
            # safe to attribute to the reviewed deployment.
            continue
        row["index"] = _safe_int(device.get("index"))
        row["uuid"] = (
            _optional_text(row.get("uuid"))
            or _optional_text(device.get("uuid"))
        )
        row["pci_bus_id"] = (
            _text(row.get("pci_bus_id"))
            or _text(
                device.get("pci_bus_id")
                or device.get("pci")
                or device.get("bus_id")
            )
        )
        row["name"] = _text(row.get("name")) or _text(
            device.get("name")
        )
        row["numa_node"] = (
            _safe_int(row.get("numa_node"))
            if _safe_int(row.get("numa_node")) is not None
            else _safe_int(device.get("numa_node"))
        )
        bound.append(row)
    return bound


def _valid_nonnegative_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _valid_nonnegative_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def _valid_scheduler(row):
    return (
        isinstance(row, dict)
        and _valid_nonnegative_int(row.get("active"))
        and _valid_nonnegative_int(row.get("queued"))
    )


def _valid_tiers(row):
    return (
        isinstance(row, dict)
        and all(
            _valid_nonnegative_int(row.get(key))
            for key in ("vram", "ram", "disk")
        )
        and all(
            _valid_nonnegative_number(row.get(key))
            for key in ("vram_gb", "ram_gb")
        )
    )


def _sum_tiers(rows):
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        return None
    result = {}
    for key in set().union(*(row.keys() for row in rows)):
        values = [row.get(key) for row in rows]
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
            if value is not None
        ):
            present = [value for value in values if value is not None]
            if present:
                result[key] = sum(present)
        else:
            for value in values:
                if value is not None:
                    result[key] = value
                    break
    return result or None


def _model_rows_complete(plan, rows):
    selected = list(
        (
            ((plan or {}).get("managed_accelerator") or {}).get("devices")
            or ()
        )
    )
    if not selected:
        return True
    if not _plan_has_safe_gpu_mapping(plan) or len(rows) != len(selected):
        return False
    return all(
        sum(_same_gpu_record(device, row) for row in rows) == 1
        for device in selected
    )


def _plan_has_safe_gpu_mapping(plan):
    devices = list(
        (
            ((plan or {}).get("managed_accelerator") or {}).get("devices")
            or ()
        )
    )
    if not devices:
        return True
    if any(not isinstance(device, dict) for device in devices):
        return False
    ordinals = [_safe_int(device.get("cuda_ordinal")) for device in devices]
    uuids = [_optional_text(device.get("uuid")) for device in devices]
    return (
        ordinals == list(range(len(devices)))
        and all(uuid is not None for uuid in uuids)
        and len({uuid.lower() for uuid in uuids}) == len(uuids)
    )


_CONTAINMENT_IDENTITY_FIELDS = (
    "version",
    "mode",
    "relative_path",
    "device",
    "inode",
)


class _ManagedIdentityChanged(ValueError):
    """The authority for an in-flight endpoint response is no longer live."""


def _managed_response_identity(manifest, record):
    """Return the immutable authority to which one response is bound.

    Endpoint telemetry is meaningful only for the exact persisted deployment,
    process, and kernel-containment record that authorized the request.  Keep
    raw persisted values here: removing a field must differ from retaining its
    effective fallback, and every cgroup identity component is independently
    security-sensitive.
    """
    containment = record.get("containment")
    containment_identity = (
        tuple(containment.get(key) for key in _CONTAINMENT_IDENTITY_FIELDS)
        if isinstance(containment, dict)
        else None
    )
    return (
        manifest.get("deployment_id"),
        manifest.get("created_at"),
        manifest.get("state"),
        record.get("pid"),
        record.get("pgid"),
        record.get("uid"),
        record.get("starttime"),
        record.get("nonce"),
        record.get("port"),
        record.get("node"),
        record.get("state_dir"),
        record.get("weights_dir"),
        record.get("stopped_at"),
        containment_identity,
    )


class RuntimeMonitor:
    """Stateful, read-only sampler for explicit headless status requests."""

    def __init__(
        self,
        *,
        urlopen=None,
        subprocess_run=None,
        monotonic=None,
        wall_time=None,
        process_matches=None,
        process_group_members=None,
        process_metrics=None,
        load_manifest=None,
        verify_record=None,
        api_key=None,
        timeout=0.75,
    ):
        self._urlopen = urlopen or _loopback_urlopen
        self._subprocess_run = subprocess_run or subprocess.run
        self._monotonic = monotonic or time.monotonic
        self._wall_time = wall_time or time.time
        self._process_matches = process_matches or _process_matches
        self._process_group_members = (
            process_group_members or _process_group_members
        )
        self._process_metrics = process_metrics or _managed_process_metrics
        self._load_manifest = load_manifest
        self._verify_record = verify_record
        self._api_key = api_key
        self._timeout = max(0.05, min(float(timeout), 5.0))
        self._last = {
            "service": None,
            "cards": None,
            "model": None,
            "tiers": None,
            "profile": None,
            "process": None,
        }
        self._deployment_identity = None
        self._profile_versions = {}
        self._profile_order = {}
        self._profile_counter = 0
        self._profile_ambiguous = False
        self._gpu_versions = {}
        self._gpu_observed = {}
        self._gpu_dirty_baseline = None

    def _request_json(self, url):
        headers = {"Accept": "application/json"}
        api_key = (
            self._api_key
            if self._api_key is not None
            else os.environ.get("COLI_API_KEY")
        )
        if api_key:
            headers["Authorization"] = "Bearer " + str(api_key)
        request = _http_request(url, headers=headers, method="GET")
        with self._urlopen(request, timeout=self._timeout) as response:
            status = getattr(response, "status", None)
            if status is None and hasattr(response, "getcode"):
                status = response.getcode()
            if status is not None and int(status) != 200:
                raise ValueError("HTTP %s" % status)
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("response exceeds telemetry size limit")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("telemetry response is not an object")
        return payload

    def _current_manifest(self, fallback):
        if self._load_manifest is None:
            return fallback
        manifest = self._load_manifest(required=True)
        if not isinstance(manifest, dict):
            raise ValueError("persisted deployment manifest is unavailable")
        return manifest

    def _revalidate_response_identity(self, identity, fallback):
        """Reopen and prove the exact persisted/live response authority."""
        manifest = self._current_manifest(fallback)
        candidates = [
            record
            for record in list(manifest.get("processes") or ())
            if isinstance(record, dict)
            and _managed_response_identity(manifest, record) == identity
        ]
        if len(candidates) != 1:
            raise ValueError(
                "managed deployment or process identity changed while sampling"
            )
        record = candidates[0]
        active_manifest = dict(manifest)
        active_manifest["processes"] = [record]
        verified, rejected = self._verify_records(active_manifest)
        if len(verified) != 1:
            pid = _safe_int(record.get("pid"))
            raise ValueError(
                rejected.get(pid)
                or "managed process identity changed while sampling"
            )

        containment = record.get("containment")
        if containment is None:
            # Legacy records predate cgroup-v2 authority and retain the donor's
            # exact process-group validation. New managed records always carry
            # containment and therefore require the injected kernel verifier.
            return
        if self._verify_record is None:
            raise ValueError(
                "managed containment identity cannot be revalidated"
            )
        result = self._verify_record(record)
        if isinstance(result, tuple):
            matches = result[0] is True if result else False
            reason = result[1] if len(result) > 1 else None
        else:
            matches = result is True
            reason = None
        if not matches:
            raise ValueError(
                str(
                    reason
                    or "managed containment identity changed while sampling"
                )
            )

    def _request_bound_json(self, url, manifest, record):
        """Fetch JSON only across matching pre/post identity proofs."""
        if (
            self._load_manifest is None
            and self._verify_record is None
            and record.get("containment") is None
        ):
            # Preserve the donor API for legacy callers. Managed PR3 callers
            # inject both persisted-manifest and cgroup verification seams.
            return self._request_json(url)
        identity = _managed_response_identity(manifest, record)
        try:
            self._revalidate_response_identity(identity, manifest)
        except Exception as error:
            raise _ManagedIdentityChanged(_error_text(error)) from error
        payload = self._request_json(url)
        try:
            self._revalidate_response_identity(identity, manifest)
        except Exception as error:
            raise _ManagedIdentityChanged(_error_text(error)) from error
        return payload

    def _response_state_snapshot(self):
        return {
            "last": copy.deepcopy(self._last),
            "deployment_identity": copy.deepcopy(
                self._deployment_identity
            ),
            "profile_versions": copy.deepcopy(self._profile_versions),
            "profile_order": copy.deepcopy(self._profile_order),
            "profile_counter": self._profile_counter,
            "profile_ambiguous": self._profile_ambiguous,
            "gpu_versions": copy.deepcopy(self._gpu_versions),
            "gpu_observed": copy.deepcopy(self._gpu_observed),
            "gpu_dirty_baseline": copy.deepcopy(
                self._gpu_dirty_baseline
            ),
        }

    def _restore_response_state(self, snapshot, preserve_profile=False):
        profile = None
        if preserve_profile:
            profile = {
                "last": copy.deepcopy(self._last["profile"]),
                "versions": copy.deepcopy(self._profile_versions),
                "order": copy.deepcopy(self._profile_order),
                "counter": self._profile_counter,
                "ambiguous": self._profile_ambiguous,
            }
        self._last = copy.deepcopy(snapshot["last"])
        self._deployment_identity = copy.deepcopy(
            snapshot["deployment_identity"]
        )
        self._profile_versions = copy.deepcopy(snapshot["profile_versions"])
        self._profile_order = copy.deepcopy(snapshot["profile_order"])
        self._profile_counter = snapshot["profile_counter"]
        self._profile_ambiguous = snapshot["profile_ambiguous"]
        self._gpu_versions = copy.deepcopy(snapshot["gpu_versions"])
        self._gpu_observed = copy.deepcopy(snapshot["gpu_observed"])
        self._gpu_dirty_baseline = copy.deepcopy(
            snapshot["gpu_dirty_baseline"]
        )
        if profile is not None:
            self._last["profile"] = profile["last"]
            self._profile_versions = profile["versions"]
            self._profile_order = profile["order"]
            self._profile_counter = profile["counter"]
            self._profile_ambiguous = profile["ambiguous"]

    def _verify_records(self, manifest):
        verified = []
        rejected = {}
        for record in list((manifest or {}).get("processes") or ()):
            if not isinstance(record, dict):
                continue
            pid = _safe_int(record.get("pid"))
            pgid = _safe_int(record.get("pgid", pid))
            if pid is None or pgid is None:
                rejected[pid] = "invalid managed process identity"
                continue
            try:
                matches, reason, _actual = self._process_matches(record)
            except Exception as error:
                rejected[pid] = _error_text(error)
                continue
            if not matches:
                rejected[pid] = str(reason or "process identity mismatch")
                continue
            try:
                members, unreadable = self._process_group_members(pgid)
            except Exception as error:
                rejected[pid] = _error_text(error)
                continue
            if unreadable:
                rejected[pid] = "managed process group has unreadable members"
                continue
            expected_uid = record.get("uid")
            expected_nonce = record.get("nonce")
            if expected_uid is None or not expected_nonce:
                rejected[pid] = "managed process identity is incomplete"
                continue
            trusted_count = 0
            sampled_pids = set()
            for member in members:
                if (
                    not isinstance(member, dict)
                    or _safe_int(member.get("pid")) is None
                    or _safe_int(member.get("pgid")) != pgid
                    or _safe_int(member.get("sid")) != pgid
                    or member.get("uid") != expected_uid
                ):
                    continue
                if member.get("inert") is True:
                    trusted_count += 1
                    continue
                if member.get("nonce") != expected_nonce or any(
                    key in record and member.get(key) != record.get(key)
                    for key in ("state_dir", "weights_dir")
                ):
                    continue
                trusted_count += 1
                sampled_pids.add(int(member["pid"]))
            if (
                not members
                or trusted_count != len(members)
                or not sampled_pids
            ):
                rejected[pid] = "managed process group identity mismatch"
                continue
            verified.append(
                {
                    "record": record,
                    "pids": sampled_pids,
                }
            )
        return verified, rejected

    def _endpoint(self, record):
        port = _safe_int(record.get("port"))
        if port is None or not 1 <= port <= 65535:
            return None
        return {
            "port": port,
            "node": _safe_int(record.get("node")),
            "pid": _safe_int(record.get("pid")),
            "url": "http://127.0.0.1:%d" % port,
        }

    def _sample_endpoints(self, manifest, verified, rejected, now):
        by_pid = {
            _safe_int(item["record"].get("pid")): item
            for item in verified
        }
        endpoints = []
        health_rows = []
        health_snapshots = []
        tiers_rows = []
        profile_rows = []
        scheduler_rows = []
        errors = []
        identity_stale = {"health": False, "profile": False}
        expected = list((manifest or {}).get("processes") or ())
        for record in expected:
            if not isinstance(record, dict):
                continue
            endpoint = self._endpoint(record)
            if endpoint is None:
                errors.append("managed endpoint has an invalid port")
                continue
            pid = endpoint["pid"]
            endpoint.update(
                {
                    "process_verified": pid in by_pid,
                    "health_ok": False,
                    "profile_ok": False,
                    "error": None,
                }
            )
            if pid not in by_pid:
                endpoint["error"] = rejected.get(
                    pid,
                    "managed process identity is unverified",
                )
                errors.append(
                    "port %d: %s"
                    % (endpoint["port"], endpoint["error"])
                )
                endpoints.append(endpoint)
                continue
            try:
                health = self._request_bound_json(
                    endpoint["url"] + "/health",
                    manifest,
                    record,
                )
                if health.get("status") != "ok":
                    detail = _text(
                        health.get("error")
                        or health.get("detail")
                        or health.get("reason")
                    )
                    raise ValueError(
                        "health status is not ok"
                        + (": " + detail if detail else "")
                    )
                endpoint["health_ok"] = True
                normalized_gpus = _normalize_health_gpu_rows(health)
                scheduler_row = health.get("scheduler")
                tiers_row = health.get("tiers")
                health_rows.extend(normalized_gpus)
                health_snapshots.append(
                    {
                        "port": endpoint["port"],
                        "gpus": normalized_gpus,
                        "gpus_present": (
                            isinstance(health.get("gpus"), list)
                            or isinstance(health.get("gpu_details"), list)
                            or isinstance(health.get("gpus"), dict)
                            or isinstance(health.get("gpu_details"), dict)
                        ),
                        "gpus_seq": _safe_int(
                            health.get("gpus_seq")
                        ),
                        "tiers_valid": _valid_tiers(tiers_row),
                        "scheduler_valid": _valid_scheduler(
                            scheduler_row
                        ),
                    }
                )
                if _valid_tiers(tiers_row):
                    tiers_rows.append(dict(tiers_row))
                if _valid_scheduler(scheduler_row):
                    scheduler_rows.append(dict(scheduler_row))
            except _ManagedIdentityChanged as error:
                identity_stale["health"] = True
                endpoint["error"] = "health: " + _error_text(error)
                errors.append(
                    "port %d: %s"
                    % (endpoint["port"], endpoint["error"])
                )
            except Exception as error:
                endpoint["error"] = "health: " + _error_text(error)
                errors.append(
                    "port %d: %s"
                    % (endpoint["port"], endpoint["error"])
                )
            try:
                profile = self._request_bound_json(
                    endpoint["url"] + "/profile",
                    manifest,
                    record,
                )
                turns = profile.get("turns")
                if not isinstance(turns, list):
                    raise ValueError("profile turns are unavailable")
                endpoint["profile_ok"] = True
                if turns and isinstance(turns[-1], dict):
                    latest = dict(turns[-1])
                    latest["endpoint_port"] = endpoint["port"]
                    latest["profile_seq"] = _safe_int(profile.get("seq"))
                    if latest.get("tokens_per_second") is None:
                        tokens = _safe_float(
                            latest.get("completion_tokens")
                        )
                        wall = _safe_float(latest.get("wall_s"))
                        if tokens is not None and wall and wall > 0:
                            latest["tokens_per_second"] = tokens / wall
                    profile_rows.append(latest)
            except _ManagedIdentityChanged as error:
                identity_stale["profile"] = True
                detail = "profile: " + _error_text(error)
                endpoint["error"] = (
                    endpoint["error"] + "; " + detail
                    if endpoint["error"]
                    else detail
                )
                errors.append("port %d: %s" % (endpoint["port"], detail))
            except Exception as error:
                detail = "profile: " + _error_text(error)
                endpoint["error"] = (
                    endpoint["error"] + "; " + detail
                    if endpoint["error"]
                    else detail
                )
                errors.append("port %d: %s" % (endpoint["port"], detail))
            endpoints.append(endpoint)
        all_health = (
            bool(expected)
            and len(endpoints) == len(expected)
            and all(
            item["process_verified"] and item["health_ok"]
            for item in endpoints
            )
        )
        all_profiles = (
            bool(expected)
            and len(endpoints) == len(expected)
            and all(
            item["process_verified"] and item["profile_ok"]
            for item in endpoints
            )
        )
        scheduler = None
        scheduler_complete = (
            all_health and len(scheduler_rows) == len(expected)
        )
        tiers_complete = (
            all_health and len(tiers_rows) == len(expected)
        )
        if scheduler_complete:
            scheduler = {
                "active": sum(
                    _safe_int(row.get("active")) or 0
                    for row in scheduler_rows
                ),
                "queued": sum(
                    _safe_int(row.get("queued")) or 0
                    for row in scheduler_rows
                ),
            }
        latest_profile = None
        if profile_rows:
            latest_profile = max(
                profile_rows,
                key=lambda row: (
                    _safe_int(row.get("profile_seq")) or -1,
                    -(_safe_int(row.get("endpoint_port")) or 0),
                ),
            )
        return {
            "endpoints": endpoints,
            "all_health": all_health,
            "all_profiles": all_profiles,
            "scheduler": scheduler,
            "scheduler_complete": scheduler_complete,
            "health_gpus": health_rows,
            "health_snapshots": health_snapshots,
            "tiers": _sum_tiers(tiers_rows) if tiers_complete else None,
            "tiers_complete": tiers_complete,
            "latest_profile": latest_profile,
            "profile_rows": profile_rows,
            "errors": errors,
            "identity_stale": identity_stale,
            "observed_at": now,
        }

    def _select_latest_profile(self, rows):
        """Order endpoint-local profile sequences by observed changes."""
        rows = [dict(row) for row in rows if isinstance(row, dict)]
        initial = []
        changed = []
        for row in rows:
            port = _safe_int(row.get("endpoint_port"))
            sequence = _safe_int(row.get("profile_seq"))
            if port is None or sequence is None:
                continue
            previous = self._profile_versions.get(port)
            if previous is None:
                initial.append(row)
            elif sequence != previous:
                changed.append(row)
            self._profile_versions[port] = sequence
        self._profile_ambiguous = len(initial) + len(changed) > 1
        # On the first observation only, a larger endpoint-local sequence is
        # the best evidence available. Thereafter, any sequence change seen in
        # this poll is newer than every unchanged endpoint, regardless of its
        # incomparable local counter.
        initial.sort(
            key=lambda row: (
                _safe_int(row.get("profile_seq")) or -1,
                -(_safe_int(row.get("endpoint_port")) or 0),
            )
        )
        changed.sort(
            key=lambda row: (
                _safe_int(row.get("endpoint_port")) or 0,
                _safe_int(row.get("profile_seq")) or -1,
            )
        )
        for row in initial + changed:
            port = _safe_int(row.get("endpoint_port"))
            self._profile_counter += 1
            self._profile_order[port] = self._profile_counter
        candidates = [
            row
            for row in rows
            if _safe_int(row.get("endpoint_port"))
            in self._profile_order
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda row: self._profile_order[
                _safe_int(row.get("endpoint_port"))
            ],
        )

    def _run_nvidia_smi(self, query):
        result = self._subprocess_run(
            [
                "nvidia-smi",
                "--query-%s=%s" % (query[0], query[1]),
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=self._timeout,
        )
        if result.returncode:
            raise OSError(
                _text(result.stderr)
                or "nvidia-smi exited %d" % result.returncode
            )
        return list(csv.reader(io.StringIO(result.stdout)))

    def _sample_cards(self, now):
        rows = self._run_nvidia_smi(
            (
                "gpu",
                "index,uuid,pci.bus_id,name,"
                "utilization.gpu,memory.used,memory.free,memory.total",
            )
        )
        cards = []
        malformed = 0
        for fields in rows:
            if len(fields) != 8:
                malformed += 1
                continue
            index = _safe_int(fields[0])
            if index is None:
                malformed += 1
                continue
            cards.append(
                {
                    "index": index,
                    "uuid": _optional_text(fields[1]),
                    "pci_bus_id": _text(fields[2]),
                    "name": _text(fields[3]),
                    "numa_node": None,
                    "utilization_percent": _safe_float(fields[4]),
                    "memory_used_bytes": _mib_bytes(fields[5]),
                    "memory_free_bytes": _mib_bytes(fields[6]),
                    "memory_total_bytes": _mib_bytes(fields[7]),
                    "observed_at": now,
                }
            )
        if not cards and rows:
            raise ValueError("nvidia-smi returned no valid GPU rows")
        return cards, (
            "%d malformed nvidia-smi GPU row(s) were ignored" % malformed
            if malformed
            else None
        )

    def _sample_compute_apps(self, verified_pids):
        if not verified_pids:
            return {}
        rows = self._run_nvidia_smi(
            (
                "compute-apps",
                "pid,gpu_uuid,used_gpu_memory",
            )
        )
        totals = {}
        malformed = False
        for fields in rows:
            if len(fields) != 3:
                message = ",".join(fields).strip().lower()
                if "no running processes" not in message:
                    malformed = True
                continue
            pid = _safe_int(fields[0])
            uuid = _optional_text(fields[1])
            used = _mib_bytes(fields[2])
            if pid not in verified_pids:
                continue
            if uuid is None or used is None:
                raise ValueError(
                    "nvidia-smi returned incomplete managed process metrics"
                )
            key = ("uuid", uuid.lower())
            totals[key] = totals.get(key, 0) + used
        if malformed:
            raise ValueError(
                "nvidia-smi returned malformed compute process rows"
            )
        return totals

    def _sample_process_rss(self, verified):
        total = 0
        count = 0
        for item in verified:
            metrics = self._process_metrics(item["record"])
            rss = _safe_int((metrics or {}).get("rss_bytes"))
            if rss is None:
                raise ValueError("managed process RSS is unavailable")
            total += rss
            count += 1
        return total if count else 0

    def _merge_gpus(
        self,
        plan,
        hardware,
        cards,
        model_rows,
        process_vram,
        freshness,
        stopped,
    ):
        base = _planned_gpus(plan, hardware)
        sources = [base]
        if cards is not None:
            sources.append([_blank_gpu(row) for row in cards])
        if model_rows is not None:
            sources.append([_blank_gpu(row) for row in model_rows])
        if self._last["cards"] is not None:
            sources.append(
                [_blank_gpu(row) for row in self._last["cards"]["value"]]
            )
        rows = []

        def target_for(source):
            for target in rows:
                if _same_gpu_record(target, source):
                    return target
            return None

        for source in sources:
            for item in source:
                if not _gpu_aliases(item):
                    continue
                row = target_for(item)
                if row is None:
                    row = _blank_gpu(item)
                    rows.append(row)
                selected = row["selected"]
                _merge_gpu_identity(row, item)
                row["selected"] = selected or bool(item.get("selected"))

        if cards is not None:
            current_targets = set()
            for item in cards:
                target = target_for(item)
                if target is None:
                    continue
                current_targets.add(id(target))
                _merge_gpu_identity(target, item)
                for field in (
                    "utilization_percent",
                    "memory_used_bytes",
                    "memory_free_bytes",
                    "memory_total_bytes",
                ):
                    target[field] = item.get(field)
                target["card_stale"] = False
                target["observed_at"] = item.get("observed_at")
            if self._last["cards"] is not None:
                for item in self._last["cards"]["value"]:
                    target = target_for(item)
                    if target is None or id(target) in current_targets:
                        continue
                    for field in (
                        "utilization_percent",
                        "memory_used_bytes",
                        "memory_free_bytes",
                        "memory_total_bytes",
                    ):
                        target[field] = item.get(field)
                    target["card_stale"] = True
                    target["observed_at"] = item.get(
                        "observed_at",
                        self._last["cards"]["observed_at"],
                    )
        elif self._last["cards"] is not None:
            for item in self._last["cards"]["value"]:
                target = target_for(item)
                if target is None:
                    continue
                for field in (
                    "utilization_percent",
                    "memory_used_bytes",
                    "memory_free_bytes",
                    "memory_total_bytes",
                ):
                    target[field] = item.get(field)
                target["card_stale"] = True
                target["observed_at"] = self._last["cards"]["observed_at"]

        if stopped:
            for target in rows:
                target["model_resident_bytes"] = 0
                target["expert_bytes"] = 0
                target["expert_count"] = 0
                target["non_expert_bytes"] = 0
                target["model_stale"] = False
        elif model_rows is not None:
            accumulated = {}
            for item in model_rows:
                target = target_for(item)
                if target is None:
                    continue
                key = id(target)
                aggregate = accumulated.setdefault(
                    key,
                    {
                        "model_resident_bytes": 0,
                        "expert_bytes": 0,
                        "expert_count": 0,
                        "non_expert_bytes": 0,
                    },
                )
                for field in aggregate:
                    value = _safe_int(item.get(field))
                    if value is not None:
                        aggregate[field] += value
                _merge_gpu_identity(target, item)
            for key, values in accumulated.items():
                target = next(row for row in rows if id(row) == key)
                target.update(values)
                target["model_stale"] = False
        elif self._last["model"] is not None:
            for item in self._last["model"]["value"]:
                target = target_for(item)
                if target is None:
                    continue
                for field in (
                    "model_resident_bytes",
                    "expert_bytes",
                    "expert_count",
                    "non_expert_bytes",
                ):
                    target[field] = item.get(field)
                target["model_stale"] = True

        if stopped:
            for target in rows:
                target["process_vram_bytes"] = 0
                target["process_stale"] = False
        elif process_vram is not None:
            for target in rows:
                target["process_vram_bytes"] = next(
                    (
                        process_vram[alias]
                        for alias in _gpu_aliases(target)
                        if alias in process_vram
                    ),
                    0,
                )
                target["process_stale"] = False
        elif self._last["process"] is not None:
            cached = self._last["process"].get("gpu_vram") or {}
            for target in rows:
                target["process_vram_bytes"] = next(
                    (
                        cached[alias]
                        for alias in _gpu_aliases(target)
                        if alias in cached
                    ),
                    None,
                )
                target["process_stale"] = True

        for row in rows:
            if row["card_stale"] and row["observed_at"] is None:
                row["observed_at"] = freshness["cards"]["observed_at"]
        return sorted(
            rows,
            key=lambda row: (
                row.get("index") is None,
                row.get("index") if row.get("index") is not None else 0,
            ),
        )

    def sample(self, manifest=None, report=None, plan=None, hardware=None):
        """Return one safe runtime snapshot without mutating lifecycle state."""
        response_state = self._response_state_snapshot()
        self._monotonic()
        now = float(self._wall_time())
        manifest = manifest if isinstance(manifest, dict) else {}
        report = report if isinstance(report, dict) else {}
        plan = (
            plan
            if isinstance(plan, dict)
            else manifest.get("plan")
            if isinstance(manifest.get("plan"), dict)
            else {}
        )
        hardware = hardware if isinstance(hardware, dict) else {}
        deployment_identity = (
            manifest.get("deployment_id"),
            manifest.get("created_at"),
        ) if manifest else None
        if deployment_identity != self._deployment_identity:
            self._last = {
                key: None for key in self._last
            }
            self._deployment_identity = deployment_identity
            self._profile_versions = {}
            self._profile_order = {}
            self._profile_counter = 0
            self._profile_ambiguous = False
            self._gpu_versions = {}
            self._gpu_observed = {}
            self._gpu_dirty_baseline = None
        lifecycle_state = str(
            report.get("state") or manifest.get("state") or "absent"
        ).lower()
        expected = [
            record
            for record in list(manifest.get("processes") or ())
            if isinstance(record, dict) and not record.get("stopped_at")
        ]
        stopped = lifecycle_state in (
            "absent",
            "ready",
            "stopped",
            "error",
        ) and not expected
        if stopped:
            # A stop/start cycle can reuse one manifest identity. Live serving,
            # model, tier, request, and process samples from the prior engine
            # must not be resurrected while its replacement is starting.
            for channel in (
                "service",
                "model",
                "tiers",
                "profile",
                "process",
            ):
                self._last[channel] = None
            self._profile_versions = {}
            self._profile_order = {}
            self._profile_counter = 0
            self._profile_ambiguous = False
            self._gpu_versions = {}
            self._gpu_observed = {}
            self._gpu_dirty_baseline = None

        active_manifest = dict(manifest)
        active_manifest["processes"] = expected
        verified, rejected = self._verify_records(active_manifest)
        endpoints = self._sample_endpoints(
            active_manifest,
            verified,
            rejected,
            now,
        )

        freshness = {
            "service": _freshness(),
            "cards": _freshness(),
            "model": _freshness(),
            "tiers": _freshness(),
            "profile": _freshness(),
            "process": _freshness(),
        }

        if stopped:
            service_state = "stopped"
            service_stale = False
            service_error = None
            scheduler = {"active": 0, "queued": 0}
            service_observed = now
        elif endpoints["all_health"]:
            service_state = "serving"
            service_stale = False
            service_error = None
            scheduler = endpoints["scheduler"] or {
                "active": None,
                "queued": None,
            }
            service_observed = now
            self._last["service"] = {
                "value": dict(scheduler),
                "observed_at": now,
            }
        else:
            service_state = (
                "starting"
                if lifecycle_state in ("preparing", "starting")
                else "degraded"
            )
            service_stale = True
            service_error = "; ".join(endpoints["errors"]) or (
                "managed service health is unavailable"
            )
            cached = self._last["service"]
            scheduler = (
                dict(cached["value"])
                if cached is not None
                else endpoints["scheduler"]
                or {"active": None, "queued": None}
            )
            service_observed = (
                cached["observed_at"] if cached is not None else None
            )
        freshness["service"] = _freshness(
            service_stale,
            service_observed,
            service_error,
        )

        selected_gpu = bool(
            ((plan or {}).get("managed_accelerator") or {}).get("devices")
        )
        safe_gpu_mapping = _plan_has_safe_gpu_mapping(plan)
        known_gpus = bool((hardware or {}).get("gpus"))
        cards = None
        cards_error = None
        try:
            if not selected_gpu and not known_gpus:
                cards = []
                partial_cards_error = None
            else:
                cards, partial_cards_error = self._sample_cards(now)
            cards_error = partial_cards_error
            freshness["cards"] = _freshness(False, now, None)
            if partial_cards_error:
                freshness["cards"] = _freshness(
                    True,
                    now,
                    partial_cards_error,
                )
        except Exception as error:
            cards_error = _error_text(error)
            cached = self._last["cards"]
            freshness["cards"] = _freshness(
                True,
                cached["observed_at"] if cached is not None else None,
                cards_error,
            )

        model_rows = None
        exact_health_gpus = []
        model_complete = (
            endpoints["all_health"]
            and len(endpoints["health_snapshots"]) == len(expected)
        )
        model_observed = None
        current_gpu_versions = {}
        for endpoint_snapshot in endpoints["health_snapshots"]:
            rows = _bind_model_rows_to_plan(
                plan,
                [
                    {
                        key: value
                        for key, value in row.items()
                        if key != "_exact"
                    }
                    for row in endpoint_snapshot["gpus"]
                    if row.get("_exact")
                ],
            )
            if selected_gpu and (
                not endpoint_snapshot["gpus_present"]
                or endpoint_snapshot["gpus_seq"] is None
                or not safe_gpu_mapping
                or not _model_rows_complete(plan, rows)
            ):
                model_complete = False
            if endpoint_snapshot["gpus_seq"] is not None:
                current_gpu_versions[
                    endpoint_snapshot["port"]
                ] = endpoint_snapshot["gpus_seq"]
            exact_health_gpus.extend(rows)
        if selected_gpu and model_complete:
            observed = []
            for endpoint_snapshot in endpoints["health_snapshots"]:
                port = endpoint_snapshot["port"]
                sequence = endpoint_snapshot["gpus_seq"]
                if self._gpu_versions.get(port) != sequence:
                    self._gpu_versions[port] = sequence
                    self._gpu_observed[port] = now
                observed.append(self._gpu_observed.get(port, now))
            model_observed = min(observed) if observed else now
        scheduler_quiescent = (
            endpoints["scheduler_complete"]
            and _safe_int((endpoints["scheduler"] or {}).get("active")) == 0
        )
        if selected_gpu and not scheduler_quiescent:
            if self._gpu_dirty_baseline is None:
                self._gpu_dirty_baseline = {
                    item["port"]: current_gpu_versions.get(item["port"])
                    for item in endpoints["endpoints"]
                    if _safe_int(item.get("port")) is not None
                }
        elif (
            selected_gpu
            and self._gpu_dirty_baseline is not None
            and model_complete
            and set(current_gpu_versions)
            == set(self._gpu_dirty_baseline)
            and all(
                current_gpu_versions[port]
                != self._gpu_dirty_baseline[port]
                for port in current_gpu_versions
            )
        ):
            self._gpu_dirty_baseline = None
        placement_snapshot_pending = (
            selected_gpu and self._gpu_dirty_baseline is not None
        )
        if stopped:
            freshness["model"] = _freshness(False, now, None)
        elif not selected_gpu:
            model_rows = []
            self._last["model"] = {
                "value": model_rows,
                "observed_at": now,
            }
            freshness["model"] = _freshness(False, now, None)
        elif model_complete:
            self._last["model"] = {
                "value": exact_health_gpus,
                "observed_at": model_observed,
            }
            if scheduler_quiescent and not placement_snapshot_pending:
                model_rows = exact_health_gpus
                freshness["model"] = _freshness(
                    False,
                    model_observed,
                    None,
                )
            else:
                freshness["model"] = _freshness(
                    True,
                    model_observed,
                    (
                        "model placement may be changing during an active "
                        "request"
                        if (
                            endpoints["scheduler_complete"]
                            and not scheduler_quiescent
                        )
                        else (
                            "awaiting a post-request GPU placement snapshot"
                            if scheduler_quiescent
                            else "scheduler telemetry is incomplete"
                        )
                    ),
                )
        else:
            cached = self._last["model"]
            model_error = (
                (
                    "managed CUDA plan has no safe ordinal mapping; "
                    "stop, destroy, and prepare the workspace again"
                )
                if selected_gpu and not safe_gpu_mapping
                else "model GPU telemetry is incomplete or unavailable"
                if endpoints["all_health"] and selected_gpu
                else service_error
            )
            freshness["model"] = _freshness(
                True,
                cached["observed_at"] if cached is not None else None,
                model_error,
            )

        tiers = None
        tiers_stale = not stopped
        if stopped:
            freshness["tiers"] = _freshness(False, now, None)
        elif endpoints["tiers_complete"] and endpoints["tiers"] is not None:
            tiers = endpoints["tiers"]
            cached = self._last["tiers"]
            tiers_observed = (
                cached["observed_at"]
                if cached is not None and cached["value"] == tiers
                else now
            )
            self._last["tiers"] = {
                "value": tiers,
                "observed_at": tiers_observed,
            }
            tiers_stale = (
                not scheduler_quiescent or placement_snapshot_pending
            )
            freshness["tiers"] = _freshness(
                tiers_stale,
                tiers_observed,
                (
                    None
                    if not tiers_stale
                    else (
                        "expert tiers may be changing during an active request"
                        if (
                            endpoints["scheduler_complete"]
                            and not scheduler_quiescent
                        )
                        else (
                            "awaiting a post-request GPU placement snapshot"
                            if scheduler_quiescent
                            else "scheduler telemetry is incomplete"
                        )
                    )
                ),
            )
        elif self._last["tiers"] is not None:
            tiers = dict(self._last["tiers"]["value"])
            freshness["tiers"] = _freshness(
                True,
                self._last["tiers"]["observed_at"],
                "expert tier telemetry is incomplete or unavailable",
            )
        else:
            freshness["tiers"] = _freshness(
                True,
                None,
                "expert tier telemetry is incomplete or unavailable",
            )

        latest_profile = None
        profile_stale = not stopped
        if stopped:
            freshness["profile"] = _freshness(False, now, None)
        elif endpoints["all_profiles"]:
            latest_profile = self._select_latest_profile(
                endpoints["profile_rows"]
            )
            profile_stale = self._profile_ambiguous
            profile_observed = now
            if latest_profile is not None:
                cached = self._last["profile"]
                if (
                    cached is not None
                    and _safe_int(
                        cached["value"].get("endpoint_port")
                    )
                    == _safe_int(latest_profile.get("endpoint_port"))
                    and _safe_int(
                        cached["value"].get("profile_seq")
                    )
                    == _safe_int(latest_profile.get("profile_seq"))
                ):
                    profile_observed = cached["observed_at"]
                self._last["profile"] = {
                    "value": latest_profile,
                    "observed_at": profile_observed,
                }
            freshness["profile"] = _freshness(
                profile_stale,
                profile_observed,
                (
                    "multiple endpoints completed requests in one poll; "
                    "latest completion is ambiguous"
                    if profile_stale
                    else None
                ),
            )
        else:
            cached = self._last["profile"]
            if cached is not None:
                latest_profile = dict(cached["value"])
            freshness["profile"] = _freshness(
                True,
                cached["observed_at"] if cached is not None else None,
                "; ".join(endpoints["errors"]) or "profile is unavailable",
            )

        verified_pids = set()
        for item in verified:
            verified_pids.update(item["pids"])
        process_vram = None
        process_rss = None
        process_error = None
        if stopped:
            process_vram = {}
            process_rss = 0
            freshness["process"] = _freshness(False, now, None)
        elif len(verified) == len(expected) and verified:
            errors = []
            try:
                process_rss = self._sample_process_rss(verified)
            except Exception as error:
                errors.append(_error_text(error))
            if selected_gpu:
                try:
                    process_vram = self._sample_compute_apps(verified_pids)
                except Exception as error:
                    errors.append(_error_text(error))
            else:
                process_vram = {}
            if not errors and process_rss is not None:
                self._last["process"] = {
                    "rss_bytes": process_rss,
                    "gpu_vram": process_vram or {},
                    "observed_at": now,
                }
                freshness["process"] = _freshness(False, now, None)
            else:
                process_error = "; ".join(errors) or (
                    "managed process metrics are unavailable"
                )
        else:
            process_error = "managed process identity is unverified"
        if process_error is not None:
            cached = self._last["process"]
            if cached is not None:
                process_rss = cached["rss_bytes"]
                process_vram = None
            freshness["process"] = _freshness(
                True,
                cached["observed_at"] if cached is not None else None,
                process_error,
            )

        gpus = self._merge_gpus(
            plan,
            hardware,
            cards,
            model_rows,
            process_vram,
            freshness,
            stopped,
        )
        if model_rows is not None:
            self._last["model"]["value"] = [
                dict(row) for row in gpus if row.get("selected")
            ]
        if cards is not None:
            self._last["cards"] = {
                "value": [dict(row) for row in gpus],
                "observed_at": now,
            }

        labels = {
            "stopped": "STOPPED",
            "starting": "STARTING",
            "serving": "SERVING",
            "degraded": "DEGRADED",
        }
        snapshot = {
            "service": {
                "state": service_state,
                "label": labels[service_state],
                "endpoints": endpoints["endpoints"],
                "active": scheduler.get("active"),
                "queued": scheduler.get("queued"),
                "error": service_error,
                "stale": service_stale,
                "observed_at": service_observed,
            },
            "gpus": gpus,
            "tiers": tiers,
            "tiers_stale": tiers_stale,
            "latest_profile": latest_profile,
            "profile_stale": profile_stale,
            "process_rss_bytes": process_rss,
            "process_stale": freshness["process"]["stale"],
            "freshness": freshness,
        }
        if endpoints["identity_stale"]["health"]:
            self._restore_response_state(
                response_state,
                preserve_profile=(
                    endpoints["all_profiles"]
                    and not endpoints["identity_stale"]["profile"]
                ),
            )
        elif endpoints["identity_stale"]["profile"]:
            self._last["profile"] = copy.deepcopy(
                response_state["last"]["profile"]
            )
            self._profile_versions = copy.deepcopy(
                response_state["profile_versions"]
            )
            self._profile_order = copy.deepcopy(
                response_state["profile_order"]
            )
            self._profile_counter = response_state["profile_counter"]
            self._profile_ambiguous = response_state["profile_ambiguous"]
        return snapshot
