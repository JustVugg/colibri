"""Platform selection and explicit RAM-disk capability reports."""

from __future__ import print_function

import os
import platform
import sys

from .common import RamdiskError


UNSUPPORTED_PLATFORM_REASON = "coli ramdisk is supported only on Linux"


def current_uid():
    """Return the invoking UID without assuming a POSIX ``os`` module."""
    getuid = getattr(os, "getuid", None)
    return int(getuid()) if getuid is not None else 1000


def current_euid():
    """Return the effective UID, falling back to the portable invoking UID."""
    geteuid = getattr(os, "geteuid", None)
    return int(geteuid()) if geteuid is not None else current_uid()


def _capabilities(platform_name, supported, reason=None):
    return {
        "platform": platform_name,
        "hardware_discovery": bool(supported),
        "cgroup_memory": bool(supported),
        "numa": bool(supported),
        "ramdisk_lifecycle": bool(supported),
        "reason": reason,
    }


def _unsupported_process_operation(*args, **kwargs):
    del args, kwargs
    raise RamdiskError(UNSUPPORTED_PLATFORM_REASON)


class UnsupportedPlatformOps:
    """Portable facts for a host without a RAM-disk lifecycle backend."""

    is_linux = False
    process_control_supported = False
    process_control_reason = UNSUPPORTED_PLATFORM_REASON

    def __init__(self, platform_name):
        self.platform_name = platform_name

    def capabilities(self):
        return _capabilities(
            self.platform_name,
            supported=False,
            reason=UNSUPPORTED_PLATFORM_REASON,
        )

    def cpu_count(self):
        return max(1, int(os.cpu_count() or 1))

    def kernel_release(self):
        return platform.release()

    process_start_boundary = staticmethod(_unsupported_process_operation)
    process_identity = staticmethod(_unsupported_process_operation)
    managed_launch_processes = staticmethod(_unsupported_process_operation)
    process_group_member_pids = staticmethod(_unsupported_process_operation)
    process_group_alive = staticmethod(_unsupported_process_operation)
    signal_verified_process_group = staticmethod(
        _unsupported_process_operation
    )
    process_status = staticmethod(_unsupported_process_operation)
    busy_mount_references = staticmethod(_unsupported_process_operation)


def get_platform_ops(platform_name=None):
    """Select an operations backend without probing host facilities."""
    selected = sys.platform if platform_name is None else str(platform_name)
    if selected.startswith("linux"):
        from .linux_ops import LinuxPlatformOps

        return LinuxPlatformOps(selected)
    return UnsupportedPlatformOps(selected)
