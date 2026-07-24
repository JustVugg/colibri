"""Direct contracts for the dependency-injected presentation module."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403

from ramdisk_support import presentation


class PresentationModuleTest(unittest.TestCase):
    def test_shared_and_replica_summaries_preserve_operator_consequences(self):
        with ModelFixture() as fixture:
            shared = ramdisk.build_plan(
                plan_args(fixture.root),
                hardware=hardware_fixture(nodes=2),
            )
            replica = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"),
                hardware=hardware_fixture(nodes=2),
            )

        shared_summary = presentation._placement_summary(shared, base_port=9100)
        replica_summary = presentation._placement_summary(replica, base_port=9100)

        self.assertEqual(shared_summary["copy_count"], 1)
        self.assertEqual(shared_summary["ports"], [9100])
        self.assertEqual(replica_summary["copy_count"], 2)
        self.assertEqual(replica_summary["ports"], [9100, 9101])
        self.assertIn("replication, not model sharding", replica_summary["explanation"])

    def test_manifest_token_gets_persisted_port_through_its_callback(self):
        manifest = {
            "version": 1,
            "deployment_id": "a" * 32,
            "created_at": "2026-07-23T12:00:00+00:00",
            "state": "ready",
            "mounts": [],
            "processes": [],
        }
        seen = []

        def persisted_base_port(value):
            seen.append(value)
            return 9100

        first = presentation._manifest_confirmation_token(
            manifest,
            persisted_base_port=persisted_base_port,
        )
        second = presentation._manifest_confirmation_token(
            manifest,
            persisted_base_port=lambda _manifest: 9200,
        )

        self.assertEqual(seen, [manifest])
        self.assertNotEqual(first, second)

    def test_activity_rows_only_read_meminfo_for_a_present_workspace(self):
        hardware = hardware_fixture()

        absent = presentation._tui_activity_rows(
            {"present": False, "state": "absent"},
            hardware,
            meminfo=lambda: self.fail("absent workspace must not probe meminfo"),
        )
        active = presentation._tui_activity_rows(
            {
                "present": True,
                "state": "ready",
                "deep_validation": False,
                "manifest_path": "/tmp/manifest.json",
                "mounts": [],
                "processes": [],
            },
            hardware,
            meminfo=lambda: {"Shmem": 4 * ramdisk.GIB},
        )

        self.assertIn(("dim", "No RAM workspace exists yet."), absent)
        self.assertEqual(
            active[-1],
            ("dim", "Host shared memory 4.00 GiB · swap 0.000 GiB"),
        )


if __name__ == "__main__":
    unittest.main()
