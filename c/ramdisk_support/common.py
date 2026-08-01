"""Dependency-free constants, errors, and pure helpers for RAM-disk control."""

from __future__ import print_function

import datetime
import math
import os
import re


MANIFEST_VERSION = 1

PLAN_SCHEMA = "colibri.ramdisk.plan.v1"

STATUS_SCHEMA = "colibri.ramdisk.status.v1"

BENCHMARK_SCHEMA = "colibri.ramdisk.benchmark.v1"

DEFAULT_MOUNT_ROOT = "/mnt/colibri-ram"

GIB = 1 << 30

MIB = 1 << 20

TMPFS_MAGIC = 0x01021994

PROFILE_LINE_RE = re.compile(r"^\s*(-?\d+)\s+(\d+)\s+(\d+)\s*$")

USAGE_FORMAT_VERSION = 1

USAGE_MERGE_RE = re.compile(r"^# coli-ramdisk-merge ([0-9a-f]{32})$")

class RamdiskError(RuntimeError):
    """An expected, user-actionable lifecycle failure."""

class _OperationCancelled(RamdiskError):
    """A cooperative cancellation that reached a clean lifecycle checkpoint."""

class _EngineCleanupError(RamdiskError):
    """A benchmark engine may still be live, so no later variant may launch."""


def _usage_engine_id(name):
    """Return route_trace.h's stable 32-bit FNV-1a engine identity."""
    value = 2166136261
    for byte in str(name).encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def _usage_engine_name(model_type):
    """Mirror the launcher's model-type to route_trace engine mapping."""
    normalized = str(model_type or "").lower()
    if "inkling" in normalized:
        return "inkling"
    if "kimi" in normalized:
        return "kimi_k3"
    return "glm_moe_dsa"


def _validated_usage_header(
    records,
    source="usage history",
    expected_dimensions=None,
    expected_engine_id=None,
):
    """Validate the two route_trace.h identified-history records."""
    dimensions = []
    formats = []
    for layer, second, third in records:
        if layer == -1:
            dimensions.append((int(second), int(third)))
        elif layer == -2:
            formats.append((int(second), int(third)))
    if not dimensions and not formats:
        return None
    if len(dimensions) != 1 or len(formats) != 1:
        raise RamdiskError(
            "%s must contain both usage headers exactly once" % source
        )
    n_layers, n_experts = dimensions[0]
    version, engine_id = formats[0]
    if n_layers < 0 or n_experts < 1:
        raise RamdiskError("%s has invalid history dimensions" % source)
    if version > USAGE_FORMAT_VERSION:
        raise RamdiskError(
            "%s uses unsupported usage format version %d" % (source, version)
        )
    if not 0 <= engine_id <= 0xFFFFFFFF:
        raise RamdiskError("%s has an invalid engine identity" % source)
    if expected_dimensions is not None and (
        n_layers,
        n_experts,
    ) != tuple(expected_dimensions):
        raise RamdiskError(
            "%s dimensions %d x %d do not match the selected model's %d x %d"
            % (
                source,
                n_layers,
                n_experts,
                expected_dimensions[0],
                expected_dimensions[1],
            )
        )
    if expected_engine_id is not None and engine_id != expected_engine_id:
        raise RamdiskError(
            "%s engine identity does not match the selected model" % source
        )
    return {
        "n_layers": n_layers,
        "n_experts": n_experts,
        "format_version": version,
        "engine_id": engine_id,
    }


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def _path_without_symlinks(path):
    """True when no existing component redirects the reviewed absolute path."""
    return os.path.isabs(path) and os.path.realpath(path) == os.path.normpath(path)

def _path_is_below(path, parent, allow_equal=False):
    try:
        normalized = os.path.normpath(os.path.abspath(path))
        root = os.path.normpath(os.path.abspath(parent))
        return os.path.commonpath([normalized, root]) == root and (allow_equal or normalized != root)
    except ValueError:
        return False

def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0

def _parse_range_list(value):
    result = []
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise ValueError("descending range")
            result.extend(range(start, end + 1))
        else:
            result.append(int(item))
    return sorted(set(result))

def _format_range_list(values):
    values = sorted(set(int(value) for value in values))
    groups = []
    for value in values:
        if not groups or value != groups[-1][1] + 1:
            groups.append([value, value])
        else:
            groups[-1][1] = value
    return ",".join(
        str(start) if start == end else "%d-%d" % (start, end)
        for start, end in groups
    )

def _raise_if_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise _OperationCancelled("operation cancelled by user at a safe checkpoint")

def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(percentile * len(ordered))) - 1))
    return ordered[index]
