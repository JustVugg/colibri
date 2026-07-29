"""GPU discovery and first-run RAM-workspace preset contracts."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403

from ramdisk_support import presets


def gpu_record(index, node, bus):
    return {
        "index": index,
        "name": "GPU %d" % index,
        "uuid": "GPU-test-%d" % index,
        "pci_bus_id": bus,
        "numa_node": node,
        "locality": "resolved",
        "total_bytes": 32 * ramdisk.GIB,
        "free_bytes": 28 * ramdisk.GIB,
    }


def gpu_hardware(available=128 * ramdisk.GIB):
    hardware = hardware_fixture(available=available, nodes=4)
    hardware["gpus"] = [
        gpu_record(0, 1, "0000:41:00.0"),
        gpu_record(1, 3, "0000:c1:00.0"),
    ]
    hardware["gpu_discovery"] = {
        "status": "available",
        "error": None,
    }
    return hardware


class _GpuOps:
    is_linux = True

    def __init__(self, nodes):
        self.nodes = nodes

    @staticmethod
    def executable_path(name):
        return "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None

    def read_text(self, path, default=None):
        return self.nodes.get(path, default)


class GpuDiscoveryTest(unittest.TestCase):
    def test_discovers_and_normalizes_gpu_pci_numa_locality(self):
        output = (
            "0, RTX 5090, GPU-aaaa, 00000000:41:00.0, 32768, 30000\n"
            "1, RTX 5090, GPU-bbbb, 00000000:C1:00.0, 32768, 29000\n"
        )
        ops = _GpuOps(
            {
                "/sys/bus/pci/devices/0000:41:00.0/numa_node": "1\n",
                "/sys/bus/pci/devices/0000:c1:00.0/numa_node": "3\n",
            }
        )
        run = mock.Mock(
            return_value=argparse.Namespace(
                returncode=0,
                stdout=output,
                stderr="",
            )
        )

        report = ramdisk._discover_gpus(
            [0, 1, 2, 3],
            ops=ops,
            run=run,
        )

        self.assertEqual(report["status"], "available")
        self.assertIn("uuid", run.call_args.args[0][1])
        self.assertEqual(
            [
                (
                    gpu["index"],
                    gpu["uuid"],
                    gpu["pci_bus_id"],
                    gpu["numa_node"],
                )
                for gpu in report["devices"]
            ],
            [
                (0, "GPU-aaaa", "0000:41:00.0", 1),
                (1, "GPU-bbbb", "0000:c1:00.0", 3),
            ],
        )
        self.assertTrue(
            all(gpu["locality"] == "resolved" for gpu in report["devices"])
        )

    def test_maps_minus_one_only_on_a_single_effective_node(self):
        ops = _GpuOps(
            {"/sys/bus/pci/devices/0000:01:00.0/numa_node": "-1\n"}
        )
        run = mock.Mock(
            return_value=argparse.Namespace(
                returncode=0,
                stdout=(
                    "0, GPU, GPU-aaaa, 0000:01:00.0, 1024, 512\n"
                ),
                stderr="",
            )
        )

        single = ramdisk._discover_gpus([7], ops=ops, run=run)
        multiple = ramdisk._discover_gpus([0, 7], ops=ops, run=run)

        self.assertEqual(single["devices"][0]["numa_node"], 7)
        self.assertEqual(single["devices"][0]["locality"], "single-node")
        self.assertIsNone(multiple["devices"][0]["numa_node"])
        self.assertEqual(multiple["devices"][0]["locality"], "unknown")

    def test_query_failure_is_explicit_and_nonfatal(self):
        ops = _GpuOps({})
        run = mock.Mock(
            return_value=argparse.Namespace(
                returncode=1,
                stdout="",
                stderr="Failed to initialize NVML",
            )
        )

        report = ramdisk._discover_gpus([0], ops=ops, run=run)

        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["devices"], [])
        self.assertIn("Failed to initialize NVML", report["error"])

    def test_records_ambient_cuda_visibility_as_selection_boundary(self):
        ops = _GpuOps(
            {"/sys/bus/pci/devices/0000:01:00.0/numa_node": "0\n"}
        )
        run = mock.Mock(
            return_value=argparse.Namespace(
                returncode=0,
                stdout=(
                    "0, GPU, GPU-aaaa, 0000:01:00.0, 1024, 512\n"
                ),
                stderr="",
            )
        )

        report = ramdisk._discover_gpus(
            [0],
            ops=ops,
            run=run,
            environ={"CUDA_VISIBLE_DEVICES": "0"},
        )

        self.assertTrue(report["cuda_visible_devices_present"])
        self.assertEqual(report["cuda_visible_devices"], "0")
        self.assertIn("relaunch", report["selection_error"])


class PresetResolutionTest(unittest.TestCase):
    def resolve(
        self,
        preset,
        args,
        hardware,
        model,
        cuda_capable=True,
    ):
        return presets.resolve_preset(
            preset,
            args,
            hardware=hardware,
            model=model,
            build_plan=ramdisk.build_plan,
            load_profile=ramdisk._load_profile,
            cuda_capable=cuda_capable,
        )

    def test_gpu_fastest_selects_only_gpu_local_nodes_for_one_engine(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            result = self.resolve(
                presets.PRESET_GPU_FASTEST,
                plan_args(fixture.root),
                gpu_hardware(),
                model,
            )

        args = result["args"]
        plan = result["plan"]
        self.assertEqual(args.memory_nodes, "1,3")
        self.assertEqual(args.cpu_list, "2-3,6-7")
        self.assertEqual(plan["topology"], "interleaved")
        self.assertEqual(plan["staging"]["replica_count"], 1)
        self.assertEqual(len(plan["placement"]["engine_cpu_sets"]), 1)
        self.assertEqual(
            [gpu["index"] for gpu in plan["managed_accelerator"]["devices"]],
            [0, 1],
        )
        self.assertTrue(plan["managed_accelerator"]["mmap"])
        self.assertFalse(plan["managed_accelerator"]["rammap"])
        self.assertEqual(
            [gpu["uuid"] for gpu in plan["managed_accelerator"]["devices"]],
            ["GPU-test-0", "GPU-test-1"],
        )
        self.assertEqual(
            plan["managed_accelerator"]["layout"],
            "experts-only",
        )
        self.assertIn("dense_tensor_bytes", plan["model"])

    def test_gpu_fastest_uses_all_eligible_gpus_and_ignores_unusable_rows(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            hardware = gpu_hardware()
            hardware["gpus"][0]["numa_node"] = None
            hardware["gpus"][0]["locality"] = "unknown"
            result = self.resolve(
                presets.PRESET_GPU_FASTEST,
                plan_args(fixture.root),
                hardware,
                model,
            )

        self.assertIsNone(result["plan"]["preset"]["fallback"])
        self.assertEqual(result["args"].memory_nodes, "3")
        self.assertEqual(
            [
                gpu["index"]
                for gpu in result["plan"]["managed_accelerator"]["devices"]
            ],
            [1],
        )

    def test_gpu_fastest_honors_exact_subset_and_dense_layout(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            result = self.resolve(
                presets.PRESET_GPU_FASTEST,
                plan_args(
                    fixture.root,
                    gpu="1",
                    gpu_layout="dense-attention",
                ),
                gpu_hardware(),
                model,
            )

        plan = result["plan"]
        self.assertEqual(result["args"].memory_nodes, "3")
        self.assertEqual(
            [gpu["index"] for gpu in plan["managed_accelerator"]["devices"]],
            [1],
        )
        self.assertEqual(
            plan["managed_accelerator"]["layout"],
            "dense-attention",
        )
        self.assertEqual(
            plan["accelerator_projection"]["dense_gpu_bytes"],
            plan["model"]["dense_tensor_bytes"],
        )

    def test_sharded_dense_layout_requires_multiple_selected_gpus(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "requires at least two",
            ):
                self.resolve(
                    presets.PRESET_GPU_FASTEST,
                    plan_args(
                        fixture.root,
                        gpu="0",
                        gpu_layout="dense-attention-sharded",
                    ),
                    gpu_hardware(),
                    model,
                )

    def test_prepopulated_cuda_contract_cannot_bypass_topology_guard(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            hardware = gpu_hardware()
            device = hardware["gpus"][0]
            args = plan_args(
                fixture.root,
                topology="per-node",
                managed_accelerator={
                    "mode": "cuda",
                    "layout": "experts-only",
                    "devices": [dict(device)],
                    "mmap": True,
                    "rammap": False,
                    "async_copy": True,
                    "vram_budget": "auto",
                    "capability": "available",
                },
            )

            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "requires interleaved topology",
            ):
                ramdisk.build_plan(
                    args,
                    hardware=hardware,
                    model=model,
                )

    def test_explicit_unusable_gpu_is_rejected_instead_of_widened(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            hardware = gpu_hardware()
            hardware["gpus"][0]["numa_node"] = None
            hardware["gpus"][0]["locality"] = "unknown"
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "selects unusable NVIDIA device 0",
            ):
                self.resolve(
                    presets.PRESET_GPU_FASTEST,
                    plan_args(fixture.root, gpu="0"),
                    hardware,
                    model,
                )

    def test_scriptable_plan_resolves_gpu_selector_without_a_preset(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            plan = ramdisk.build_plan(
                plan_args(
                    fixture.root,
                    gpu="0",
                    gpu_layout="dense-attention",
                ),
                hardware=gpu_hardware(),
                model=model,
            )

        self.assertEqual(
            [gpu["index"] for gpu in plan["managed_accelerator"]["devices"]],
            [0],
        )
        self.assertEqual(plan["placement"]["memory_node_list"], "1")
        self.assertEqual(
            plan["managed_accelerator"]["layout"],
            "dense-attention",
        )

    def test_custom_numa_masks_block_an_incompatible_gpu_change(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "outside the reviewed NUMA placement",
            ):
                ramdisk.build_plan(
                    plan_args(
                        fixture.root,
                        gpu="0",
                        memory_nodes="3",
                        cpu_list="6-7",
                    ),
                    hardware=gpu_hardware(),
                    model=model,
                )

    def test_dense_layout_blocks_obvious_gpu_capacity_failure(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            hardware = gpu_hardware()
            hardware["gpus"][0]["free_bytes"] = ramdisk.GIB
            plan = ramdisk.build_plan(
                plan_args(
                    fixture.root,
                    gpu="0",
                    gpu_layout="dense-attention",
                ),
                hardware=hardware,
                model=model,
            )

        self.assertIn(
            "selected GPU free VRAM cannot hold the projected dense tensors "
            "and per-device reserve",
            plan["blockers"],
        )
        self.assertEqual(
            plan["accelerator_projection"]["expert_headroom_bytes"],
            0,
        )

    def test_dense_layout_checks_each_cards_balanced_projection(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            model["dense_tensor_bytes"] = 20 * ramdisk.GIB
            hardware = gpu_hardware()
            hardware["gpus"][0]["free_bytes"] = 3 * ramdisk.GIB
            hardware["gpus"][1]["free_bytes"] = 30 * ramdisk.GIB
            plan = ramdisk.build_plan(
                plan_args(
                    fixture.root,
                    gpu="0,1",
                    gpu_layout="dense-attention",
                ),
                hardware=hardware,
                model=model,
            )

        projection = plan["accelerator_projection"]
        self.assertGreater(
            projection["selected_free_bytes"],
            projection["dense_gpu_bytes"]
            + 2 * projection["vram_reserve_per_device_bytes"],
        )
        self.assertFalse(projection["per_device"][0]["admission_ok"])
        self.assertTrue(projection["per_device"][1]["admission_ok"])
        self.assertIn(
            "selected GPU free VRAM cannot hold the projected dense tensors "
            "and per-device reserve",
            plan["blockers"],
        )
        self.assertIn("balanced estimate", " ".join(plan["warnings"]))

    def test_gpu_fastest_falls_back_to_single_when_locality_is_unavailable(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            hardware = hardware_fixture(nodes=4)
            hardware["gpus"] = []
            hardware["gpu_discovery"] = {
                "status": "unavailable",
                "error": "NVML mismatch",
            }
            result = self.resolve(
                presets.PRESET_GPU_FASTEST,
                plan_args(fixture.root),
                hardware,
                model,
            )

        self.assertEqual(result["args"].topology, "interleaved")
        self.assertIsNone(result["args"].memory_nodes)
        self.assertEqual(result["plan"]["preset"]["fallback"], "single")
        self.assertIn("NVML mismatch", result["plan"]["preset"]["reason"])
        self.assertEqual(result["plan"]["managed_accelerator"]["mode"], "cpu")

    def test_gpu_fastest_falls_back_when_cuda_capability_is_unproven(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            result = self.resolve(
                presets.PRESET_GPU_FASTEST,
                plan_args(fixture.root),
                gpu_hardware(),
                model,
                cuda_capable=None,
            )

        self.assertEqual(result["plan"]["preset"]["fallback"], "single")
        self.assertIn(
            "CUDA engine capability could not be established",
            result["plan"]["preset"]["reason"],
        )
        self.assertEqual(result["plan"]["managed_accelerator"]["mode"], "cpu")

    def test_gpu_fastest_falls_back_to_partial_without_ever_replicating(self):
        with ModelFixture() as fixture:
            (fixture.root / ".coli_usage").write_text(
                "0 1 100\n",
                encoding="utf-8",
            )
            model = ramdisk.scan_model(str(fixture.root))
            for shard in model["shards"]:
                shard["size_bytes"] = 10 * ramdisk.GIB
            model["total_shard_bytes"] = 20 * ramdisk.GIB
            result = self.resolve(
                presets.PRESET_GPU_FASTEST,
                plan_args(fixture.root),
                gpu_hardware(available=64 * ramdisk.GIB),
                model,
            )

        self.assertEqual(result["args"].mode, "partial")
        self.assertEqual(result["plan"]["mode"], "partial")
        self.assertEqual(result["plan"]["topology"], "interleaved")
        self.assertEqual(result["plan"]["staging"]["replica_count"], 1)
        self.assertFalse(
            any("replica" in blocker for blocker in result["plan"]["blockers"])
        )

    def test_minimal_missing_profile_is_an_actionable_blocker(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            result = self.resolve(
                presets.PRESET_MINIMAL,
                plan_args(fixture.root),
                hardware_fixture(),
                model,
            )

        self.assertEqual(result["args"].mode, "partial")
        self.assertTrue(
            any(
                "profile-guided staging is unavailable" in blocker
                for blocker in result["plan"]["blockers"]
            )
        )

    def test_replicas_is_explicit_and_multiplies_engines(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            result = self.resolve(
                presets.PRESET_REPLICAS,
                plan_args(fixture.root),
                hardware_fixture(nodes=4),
                model,
            )

        self.assertEqual(result["plan"]["topology"], "per-node")
        self.assertEqual(result["plan"]["staging"]["replica_count"], 4)
        self.assertEqual(
            len(result["plan"]["placement"]["engine_cpu_sets"]),
            4,
        )

    def test_advanced_edit_marks_custom_without_dropping_accelerator(self):
        args = argparse.Namespace(
            ramdisk_preset="gpu-fastest",
            ramdisk_preset_label="Fastest GPU staging",
            ramdisk_preset_reason="GPU-local nodes.",
            ramdisk_preset_fallback=None,
            managed_accelerator={"mode": "cuda", "devices": [{"index": 0}]},
        )
        accelerator = args.managed_accelerator

        presets.mark_preset_custom(args)

        self.assertEqual(args.ramdisk_preset, "custom")
        self.assertEqual(args.ramdisk_preset_label, "Custom")
        self.assertIn("Fastest GPU staging", args.ramdisk_preset_reason)
        self.assertIs(args.managed_accelerator, accelerator)


if __name__ == "__main__":
    unittest.main()
