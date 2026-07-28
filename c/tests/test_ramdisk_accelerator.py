"""Managed accelerator plan and environment tests."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403

import copy

from ramdisk_support import accelerator, presets, processes


def _gpu_plan(root):
    hardware = hardware_fixture(nodes=4)
    hardware["gpus"] = [
        {
            "index": 0,
            "name": "GPU 0",
            "pci_bus_id": "0000:41:00.0",
            "numa_node": 1,
            "locality": "resolved",
            "total_bytes": 32 * ramdisk.GIB,
            "free_bytes": 28 * ramdisk.GIB,
        },
        {
            "index": 1,
            "name": "GPU 1",
            "pci_bus_id": "0000:c1:00.0",
            "numa_node": 3,
            "locality": "resolved",
            "total_bytes": 32 * ramdisk.GIB,
            "free_bytes": 28 * ramdisk.GIB,
        },
    ]
    hardware["gpu_discovery"] = {
        "status": "available",
        "error": None,
    }
    model = ramdisk.scan_model(str(root))
    return presets.resolve_preset(
        presets.PRESET_GPU_FASTEST,
        plan_args(root),
        hardware=hardware,
        model=model,
        build_plan=ramdisk.build_plan,
        load_profile=ramdisk._load_profile,
        cuda_capable=True,
    )["plan"]


class ManagedAcceleratorTest(unittest.TestCase):
    def test_cpu_contract_sanitizes_hostile_ambient_gpu_values(self):
        environment = {
            "COLI_CUDA": "1",
            "COLI_GPUS": "9",
            "CUDA_EXPERT_GB": "999",
            "COLI_MMAP": "1",
            "PIN": "/tmp/hostile",
            "KEEP": "yes",
        }

        applied = accelerator._apply_managed_accelerator_environment(
            environment,
            {"managed_accelerator": {"mode": "cpu", "devices": []}},
        )

        self.assertEqual(
            applied,
            {
                "COLI_CUDA": "0",
                "COLI_MMAP": "0",
                "COLI_RAMMAP": "1",
            },
        )
        self.assertEqual(environment["KEEP"], "yes")
        self.assertNotIn("COLI_GPUS", environment)
        self.assertNotIn("CUDA_EXPERT_GB", environment)
        self.assertNotIn("PIN", environment)

    def test_gpu_contract_applies_only_reviewed_devices_and_mmap_upload(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        environment = {
            "COLI_GPU": "7",
            "COLI_GPUS": "7,8",
            "CUDA_EXPERT_GB": "4",
            "COLI_RAMMAP": "1",
        }

        applied = accelerator._apply_managed_accelerator_environment(
            environment,
            plan,
        )

        self.assertEqual(applied["COLI_CUDA"], "1")
        self.assertEqual(applied["COLI_GPUS"], "0,1")
        self.assertNotIn("COLI_GPU", applied)
        self.assertEqual(applied["CUDA_EXPERT_GB"], "auto")
        self.assertEqual(applied["COLI_CUDA_ASYNC"], "1")
        self.assertEqual(applied["COLI_MMAP"], "1")
        self.assertEqual(applied["COLI_RAMMAP"], "0")
        self.assertEqual(applied["PIN"], "auto")
        self.assertEqual(applied["PIN_GB"], "all")

    def test_gpu_identity_drift_fails_closed(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        plan["hardware"]["effective_mask_source"] = "test-fixture"
        current = copy.deepcopy(plan["hardware"])
        current["gpus"][1]["pci_bus_id"] = "0000:d1:00.0"

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "GPU/NUMA identity changed",
        ):
            processes._assert_effective_masks_unchanged(
                plan,
                discover_hardware=lambda: current,
            )

    def test_legacy_plan_retains_cpu_direct_rammap_contract(self):
        contract = accelerator._managed_accelerator_contract({})
        environment = accelerator._managed_accelerator_environment({})

        self.assertEqual(contract["mode"], "cpu")
        self.assertEqual(environment["COLI_RAMMAP"], "1")
        self.assertEqual(environment["COLI_MMAP"], "0")
        self.assertEqual(environment["COLI_CUDA"], "0")


if __name__ == "__main__":
    unittest.main()
