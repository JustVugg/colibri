"""Opt-in real-tmpfs lifecycle smoke test.

Run in a privileged or isolated user/mount namespace:

    COLI_RAMDISK_INTEGRATION=1 python3 -m unittest -v tests.test_ramdisk_integration

The ordinary dependency-free suite skips this test because mounting is a
machine-level capability, not a unit-test prerequisite.
"""

import argparse
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

try:
    from . import test_ramdisk
except ImportError:  # unittest discovery imports tests as top-level modules
    import test_ramdisk

import ramdisk


@unittest.skipUnless(
    os.environ.get("COLI_RAMDISK_INTEGRATION") == "1",
    "set COLI_RAMDISK_INTEGRATION=1 inside a private mount namespace",
)
class RealTmpfsLifecycleTest(unittest.TestCase):
    def test_prepare_status_destroy_on_real_tmpfs(self):
        with test_ramdisk.ModelFixture() as fixture, tempfile.TemporaryDirectory(
            prefix="colibri-ramdisk-state-", dir="/var/tmp"
        ) as durable:
            mount_root = "/mnt/colibri-ramdisk-it-%d" % os.getpid()
            args = test_ramdisk.plan_args(
                fixture.root,
                mount_root=mount_root,
                allow_swappable=True,
                thp="auto",
                yes=True,
            )
            before_swap = ramdisk.discover_hardware()["swap"]["used_bytes"]
            with mock.patch.dict(
                os.environ,
                {
                    "XDG_STATE_HOME": durable,
                    "COLI_RAMDISK_MANIFEST": os.path.join(
                        durable, "colibri", "ramdisk", "manifest.json"
                    ),
                },
            ):
                plan = ramdisk.build_plan(args)
                # This test validates the prepare/status/destroy lifecycle, not the
                # production 16 GiB OS reserve. Fail on every non-memory prerequisite,
                # then bypass only that reserve in the reviewed plan returned to
                # prepare; production planning and admission remain untouched.
                blockers = plan["blockers"]
                memory_blockers = [
                    b for b in blockers if "memory" in b.lower() or "reserve" in b.lower()
                ]
                self.assertEqual(
                    [b for b in blockers if b not in memory_blockers],
                    [],
                    "non-memory plan blockers: %s" % blockers,
                )
                plan["blockers"] = []
                plan["reserve"]["os_margin_bytes"] = 0
                plan["reserve"]["total_os_margin_bytes"] = 0
                plan["reserve"]["required_global_bytes"] = (
                    plan["staging"]["staged_bytes"]
                    + plan["reserve"]["runtime_bytes"]
                    + plan["reserve"]["page_table_bytes"]
                )
                plan["reserve"]["total_required_bytes"] = plan["reserve"][
                    "required_global_bytes"
                ]
                with mock.patch.object(ramdisk, "build_plan", return_value=plan):
                    prepared = ramdisk.prepare(args, display_plan=False)
                self.assertEqual(prepared["state"], "ready")
                report = ramdisk.status()
                self.assertTrue(report["source_fingerprint_verified"])
                self.assertTrue(all(item["verified"] for item in report["mounts"]))
                self.assertEqual(
                    {item["filesystem"] for item in report["mounts"]}, {"tmpfs"}
                )
                durable_kv = Path(ramdisk._state_root()) / "engines" / "preserved" / ".coli_kv"
                durable_kv.parent.mkdir(parents=True, mode=0o700)
                durable_kv.write_bytes(b"durable-test-state")
                destroyed = ramdisk.destroy(argparse.Namespace(yes=True))
                self.assertTrue(destroyed["destroyed"])
                self.assertFalse(os.path.ismount(mount_root))
                self.assertEqual(ramdisk.status()["state"], "absent")
                self.assertEqual(durable_kv.read_bytes(), b"durable-test-state")
            after_swap = ramdisk.discover_hardware()["swap"]["used_bytes"]
            self.assertLessEqual(after_swap, before_swap)


if __name__ == "__main__":
    unittest.main()
