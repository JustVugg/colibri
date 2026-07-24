"""Compatibility exports for RAM-disk test fixtures.

The test cases were split into responsibility-focused modules.  Keep the
shared fixtures importable here for integration tests and external test
invocations that historically imported ``test_ramdisk``.
"""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403
