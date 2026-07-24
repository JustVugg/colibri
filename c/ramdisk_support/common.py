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

PROFILE_LINE_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$")

USAGE_MERGE_RE = re.compile(r"^# coli-ramdisk-merge ([0-9a-f]{32})$")

class RamdiskError(RuntimeError):
    """An expected, user-actionable lifecycle failure."""

class _OperationCancelled(RamdiskError):
    """A cooperative cancellation that reached a clean lifecycle checkpoint."""

class _EngineCleanupError(RamdiskError):
    """A benchmark engine may still be live, so no later variant may launch."""

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
