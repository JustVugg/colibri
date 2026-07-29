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
            "uuid": "GPU-test-0",
            "pci_bus_id": "0000:41:00.0",
            "numa_node": 1,
            "locality": "resolved",
            "total_bytes": 32 * ramdisk.GIB,
            "free_bytes": 28 * ramdisk.GIB,
        },
        {
            "index": 1,
            "name": "GPU 1",
            "uuid": "GPU-test-1",
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
            "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
            "CUDA_VISIBLE_DEVICES": "7",
            "COLI_CUDA": "1",
            "COLI_CUDA_MTP": "1",
            "COLI_GPUS": "9",
            "CUDA_EXPERT_GB": "999",
            "CUDA_RESERVE_GB": "0",
            "COLI_MMAP": "1",
            "PIN": "/tmp/hostile",
            "REPIN": "0",
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
                "CUDA_DENSE": "0",
                "COLI_CUDA_ATTN": "0",
                "COLI_CUDA_ATTN_SHARD": "0",
                "DRAFT": "0",
                "COLI_MMAP": "0",
                "COLI_RAMMAP": "1",
            },
        )
        self.assertEqual(environment["KEEP"], "yes")
        self.assertNotIn("COLI_GPUS", environment)
        self.assertNotIn("CUDA_DEVICE_ORDER", environment)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
        self.assertNotIn("COLI_CUDA_MTP", environment)
        self.assertNotIn("CUDA_EXPERT_GB", environment)
        self.assertNotIn("CUDA_RESERVE_GB", environment)
        self.assertNotIn("PIN", environment)
        self.assertNotIn("REPIN", environment)

    def test_gpu_contract_applies_only_reviewed_devices_and_mmap_upload(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        environment = {
            "COLI_GPU": "7",
            "COLI_GPUS": "7,8",
            "COLI_CUDA_PIPE": "1",
            "CUDA_EXPERT_GB": "4",
            "CUDA_RESERVE_GB": "0",
            "REPIN": "0",
            "COLI_RAMMAP": "1",
        }

        applied = accelerator._apply_managed_accelerator_environment(
            environment,
            plan,
        )

        self.assertEqual(applied["COLI_CUDA"], "1")
        self.assertEqual(applied["COLI_GPUS"], "0,1")
        self.assertEqual(
            applied["CUDA_VISIBLE_DEVICES"],
            "GPU-test-0,GPU-test-1",
        )
        self.assertNotIn("COLI_GPU", applied)
        self.assertNotIn("COLI_CUDA_PIPE", applied)
        self.assertEqual(applied["CUDA_EXPERT_GB"], "auto")
        self.assertEqual(applied["CUDA_RESERVE_GB"], "2.147483648")
        self.assertEqual(applied["COLI_CUDA_ASYNC"], "1")
        self.assertEqual(applied["COLI_MMAP"], "1")
        self.assertEqual(applied["COLI_RAMMAP"], "0")
        self.assertEqual(applied["PIN"], "auto")
        self.assertEqual(applied["PIN_GB"], "all")
        self.assertEqual(applied["REPIN"], "16")
        self.assertEqual(applied["CUDA_DENSE"], "0")
        self.assertEqual(applied["COLI_CUDA_ATTN"], "0")
        self.assertEqual(applied["COLI_CUDA_ATTN_SHARD"], "0")

    def test_gpu_contract_fails_closed_on_ambient_visibility_mask(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        environment = {
            "CUDA_VISIBLE_DEVICES": "GPU-allocation-boundary",
            "KEEP": "yes",
        }

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "CUDA_VISIBLE_DEVICES",
        ):
            accelerator._apply_managed_accelerator_environment(
                environment,
                plan,
            )

        self.assertEqual(
            environment,
            {
                "CUDA_VISIBLE_DEVICES": "GPU-allocation-boundary",
                "KEEP": "yes",
            },
        )

    def test_nonzero_physical_devices_launch_as_reviewed_logical_ordinals(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        devices = plan["managed_accelerator"]["devices"]
        devices[0].update(index=2, uuid="GPU-physical-2", cuda_ordinal=0)
        devices[1].update(index=5, uuid="GPU-physical-5", cuda_ordinal=1)

        applied = accelerator._managed_accelerator_environment(plan)

        self.assertEqual(
            applied["CUDA_VISIBLE_DEVICES"],
            "GPU-physical-2,GPU-physical-5",
        )
        self.assertEqual(applied["COLI_GPUS"], "0,1")

    def test_legacy_gpu_plan_without_ordinal_mapping_fails_closed(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        devices = plan["managed_accelerator"]["devices"]
        devices[0]["index"] = 2
        devices[1]["index"] = 5
        for device in devices:
            device.pop("cuda_ordinal")

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "no safe ordinal mapping",
        ):
            accelerator._managed_accelerator_environment(plan)

    def test_logical_ordinal_records_must_stay_in_launch_order(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        plan["managed_accelerator"]["devices"].reverse()

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "ordinal mapping",
        ):
            accelerator._managed_accelerator_environment(plan)

    def test_logical_ordinal_mapping_requires_stable_uuid(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        plan["managed_accelerator"]["devices"][0]["uuid"] = ""

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "ordinal mapping",
        ):
            accelerator._managed_accelerator_environment(plan)

    def test_dense_layout_applies_exact_sanitized_environment(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        plan["managed_accelerator"]["layout"] = (
            accelerator.GPU_LAYOUT_DENSE_ATTENTION_SHARDED
        )
        environment = {
            "CUDA_DENSE": "0",
            "COLI_CUDA_ATTN": "0",
            "COLI_CUDA_ATTN_SHARD": "0",
            "COLI_GROUP_ASYNC": "hostile",
            "COLI_DSA_GATHER": "hostile",
            "DRAFT": "1",
        }

        applied = accelerator._apply_managed_accelerator_environment(
            environment,
            plan,
        )

        self.assertEqual(applied["CUDA_DENSE"], "1")
        self.assertEqual(applied["COLI_CUDA_ATTN"], "1")
        self.assertEqual(applied["COLI_CUDA_ATTN_SHARD"], "1")
        self.assertEqual(environment, applied)
        self.assertNotIn("COLI_GROUP_ASYNC", environment)
        self.assertNotIn("COLI_DSA_GATHER", environment)
        self.assertEqual(environment["DRAFT"], "0")

    def test_gpu_identity_drift_fails_closed(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        plan["hardware"]["effective_mask_source"] = "test-fixture"
        current = copy.deepcopy(plan["hardware"])
        current["gpus"][1]["uuid"] = "GPU-replacement"

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "GPU/NUMA identity changed",
        ):
            processes._assert_effective_masks_unchanged(
                plan,
                discover_hardware=lambda: current,
            )

    def test_legacy_gpu_identity_falls_back_to_pci(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        plan["hardware"]["effective_mask_source"] = "test-fixture"
        current = copy.deepcopy(plan["hardware"])
        plan["managed_accelerator"]["devices"][1].pop("uuid")
        current["gpus"][1].pop("uuid")
        current["gpus"][1]["pci_bus_id"] = "0000:d1:00.0"

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "GPU/NUMA identity changed",
        ):
            processes._assert_effective_masks_unchanged(
                plan,
                discover_hardware=lambda: current,
            )

    def test_uuid_is_primary_when_a_gpu_pci_address_changes(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        plan["hardware"]["effective_mask_source"] = "test-fixture"
        current = copy.deepcopy(plan["hardware"])
        current["gpus"][1]["pci_bus_id"] = "0000:d1:00.0"

        processes._assert_effective_masks_unchanged(
            plan,
            discover_hardware=lambda: current,
        )

    def test_legacy_plan_retains_cpu_direct_rammap_contract(self):
        contract = accelerator._managed_accelerator_contract({})
        environment = accelerator._managed_accelerator_environment({})

        self.assertEqual(contract["mode"], "cpu")
        self.assertEqual(contract["layout"], "experts-only")
        self.assertEqual(environment["COLI_RAMMAP"], "1")
        self.assertEqual(environment["COLI_MMAP"], "0")
        self.assertEqual(environment["COLI_CUDA"], "0")

    def test_legacy_gpu_plan_defaults_to_experts_only_layout(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        plan["managed_accelerator"].pop("layout")

        contract = accelerator._managed_accelerator_contract(plan)
        environment = accelerator._managed_accelerator_environment(plan)

        self.assertEqual(contract["layout"], "experts-only")
        self.assertEqual(environment["CUDA_DENSE"], "0")
        self.assertEqual(environment["COLI_CUDA_ATTN"], "0")
        self.assertEqual(environment["COLI_CUDA_ATTN_SHARD"], "0")

    def test_confirmation_binds_gpu_identity_and_layout_not_free_vram(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        token = ramdisk._plan_confirmation_token(plan)
        changed_projection = copy.deepcopy(plan)
        changed_projection["accelerator_projection"][
            "selected_free_bytes"
        ] -= ramdisk.GIB
        changed_layout = copy.deepcopy(plan)
        changed_layout["managed_accelerator"]["layout"] = (
            accelerator.GPU_LAYOUT_DENSE_ATTENTION
        )
        changed_identity = copy.deepcopy(plan)
        changed_identity["managed_accelerator"]["devices"][0]["uuid"] = (
            "GPU-replacement"
        )

        self.assertEqual(
            ramdisk._plan_confirmation_token(changed_projection),
            token,
        )
        self.assertNotEqual(
            ramdisk._plan_confirmation_token(changed_layout),
            token,
        )
        self.assertNotEqual(
            ramdisk._plan_confirmation_token(changed_identity),
            token,
        )

    def test_tui_selection_change_preserves_proven_cuda_capability(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        args = argparse.Namespace(
            gpu="auto",
            gpu_layout="experts-only",
            managed_accelerator=copy.deepcopy(
                plan["managed_accelerator"]
            ),
            memory_nodes="1,3",
            cpu_list="2-3,6-7",
            topology="interleaved",
        )

        accelerator.apply_gpu_selection(
            args,
            plan["hardware"],
            selector="0",
            reset_placement=True,
        )

        self.assertEqual(
            args.managed_accelerator["capability"],
            "available",
        )

    def test_rejected_selection_does_not_partially_mutate_tui_draft(self):
        with ModelFixture() as fixture:
            plan = _gpu_plan(fixture.root)
        args = argparse.Namespace(
            gpu="auto",
            gpu_layout="experts-only",
            managed_accelerator=copy.deepcopy(
                plan["managed_accelerator"]
            ),
            memory_nodes="1,3",
            cpu_list="2-3,6-7",
            topology="interleaved",
        )
        before = copy.deepcopy(vars(args))

        with self.assertRaisesRegex(
            ramdisk.RamdiskError,
            "requires at least two selected GPUs",
        ):
            accelerator.apply_gpu_selection(
                args,
                plan["hardware"],
                selector="0",
                layout="dense-attention-sharded",
                reset_placement=True,
            )

        self.assertEqual(vars(args), before)


if __name__ == "__main__":
    unittest.main()
