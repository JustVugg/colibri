"""Linux filesystem and kernel operations used by RAM-disk discovery."""

from __future__ import print_function

import os
import platform
import re
import shutil

from .common import RamdiskError, _parse_range_list


def _read_text(path, default=""):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            return stream.read()
    except OSError:
        return default


def _status_allowed_list(field, fallback):
    """Read the kernel's effective task mask from ``/proc/self/status``."""
    status = _read_text("/proc/self/status")
    match = re.search(r"^%s:\s*(.*?)\s*$" % re.escape(field), status, re.MULTILINE)
    if match:
        try:
            return _parse_range_list(match.group(1))
        except (TypeError, ValueError):
            pass
    return sorted(set(int(value) for value in fallback))


def _thread_sibling_groups(cpus):
    """Return physical-core sibling groups clipped to the supplied CPU mask."""
    remaining = set(int(cpu) for cpu in cpus)
    groups = []
    while remaining:
        cpu = min(remaining)
        siblings_text = _read_text(
            "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % cpu,
            str(cpu),
        )
        try:
            siblings = set(_parse_range_list(siblings_text)) & set(cpus)
        except ValueError:
            siblings = {cpu}
        if not siblings:
            siblings = {cpu}
        groups.append(sorted(siblings))
        remaining.difference_update(siblings)
    return groups


def _meminfo(path="/proc/meminfo"):
    values = {}
    for line in _read_text(path).splitlines():
        match = re.match(r"^([^:]+):\s*(\d+)(?:\s+kB)?", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    return values


def _read_cgroup_value(path):
    """Read one controller file, distinguishing absence from access failure."""
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as stream:
            return stream.read().strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RamdiskError("cannot read cgroup controller file %s: %s" % (path, exc))


def _read_cgroup_contract(path):
    """Read a procfs cgroup contract without treating denial as absence."""
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as stream:
            return stream.read()
    except (OSError, UnicodeError) as exc:
        raise RamdiskError("cannot read cgroup contract %s: %s" % (path, exc))


def _node_meminfo(node):
    values = {}
    path = "/sys/devices/system/node/node%d/meminfo" % node
    for line in _read_text(path).splitlines():
        match = re.search(r"Node\s+\d+\s+([^:]+):\s*(\d+)\s+kB", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    return values


def _physical_cores(cpus):
    cores = set()
    for cpu in cpus:
        base = "/sys/devices/system/cpu/cpu%d/topology" % cpu
        package = _read_text(os.path.join(base, "physical_package_id"), "0").strip()
        core = _read_text(os.path.join(base, "core_id"), str(cpu)).strip()
        cores.add((package, core))
    return max(1, len(cores))


def _kernel_at_least(major, minor):
    match = re.match(r"^(\d+)\.(\d+)", platform.release())
    return bool(match and (int(match.group(1)), int(match.group(2))) >= (major, minor))


class LinuxPlatformOps:
    """Narrow Linux discovery operations with no import-time probes."""

    is_linux = True

    def __init__(self, platform_name="linux"):
        self.platform_name = platform_name

    def capabilities(self):
        return {
            "platform": self.platform_name,
            "hardware_discovery": True,
            "cgroup_memory": True,
            "numa": True,
            "ramdisk_lifecycle": True,
            "reason": None,
        }

    read_text = staticmethod(_read_text)
    status_allowed_list = staticmethod(_status_allowed_list)
    thread_sibling_groups = staticmethod(_thread_sibling_groups)
    meminfo = staticmethod(_meminfo)
    read_cgroup_value = staticmethod(_read_cgroup_value)
    read_cgroup_contract = staticmethod(_read_cgroup_contract)
    node_meminfo = staticmethod(_node_meminfo)
    physical_cores = staticmethod(_physical_cores)

    @staticmethod
    def path_exists(path):
        return os.path.exists(path)

    @staticmethod
    def cpu_affinity():
        get_affinity = getattr(os, "sched_getaffinity", None)
        if get_affinity is None:
            return None
        try:
            return sorted(int(cpu) for cpu in get_affinity(0))
        except OSError:
            return None

    @staticmethod
    def kernel_release():
        return platform.release()

    @staticmethod
    def kernel_at_least(major, minor):
        return _kernel_at_least(major, minor)

    @staticmethod
    def executable_path(name):
        return shutil.which(name)
