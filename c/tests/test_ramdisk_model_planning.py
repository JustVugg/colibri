"""RAM-disk model scanning, hardware discovery, and placement planning tests."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403


class ScanAndPlanTest(unittest.TestCase):
    def test_scan_indexes_complete_six_tensor_experts_and_sorted_shards(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
        self.assertEqual(model["shard_names"], sorted(model["shard_names"]))
        self.assertEqual(set(model["experts"]), {"0:0", "0:1"})
        self.assertEqual(len(model["experts"]["0:0"]["tensors"]), 6)
        self.assertEqual(model["experts"]["0:0"]["shards"], model["shard_names"])
        self.assertTrue(model["experts"]["0:1"]["direct_map_eligible"])

    def test_fingerprint_changes_when_source_identity_changes(self):
        with ModelFixture() as fixture:
            before = ramdisk.scan_model(str(fixture.root))["fingerprint"]
            shard = fixture.root / "model-00002-of-00002.safetensors"
            stat = shard.stat()
            os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            after = ramdisk.scan_model(str(fixture.root))["fingerprint"]
        self.assertNotEqual(before, after)

    def test_fingerprint_includes_configuration_and_tokenizer_content(self):
        with ModelFixture() as fixture:
            before = ramdisk.scan_model(str(fixture.root))["fingerprint"]
            tokenizer = fixture.root / "tokenizer.json"
            tokenizer.write_text('{"version":"changed"}', encoding="utf-8")
            after_tokenizer = ramdisk.scan_model(str(fixture.root))["fingerprint"]
            config = json.loads((fixture.root / "config.json").read_text(encoding="utf-8"))
            config["vocab_size"] += 1
            (fixture.root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            after_config = ramdisk.scan_model(str(fixture.root))["fingerprint"]
        self.assertNotEqual(before, after_tokenizer)
        self.assertNotEqual(after_tokenizer, after_config)

    def test_model_and_profile_json_require_object_roots(self):
        with ModelFixture() as fixture:
            config_path = fixture.root / "config.json"
            config_path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ramdisk.RamdiskError, "JSON object"):
                ramdisk.scan_model(str(fixture.root))

        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            profile = fixture.root / "profile.json"
            profile.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ramdisk.RamdiskError, "contain an object"):
                ramdisk._load_profile(str(profile), model)

    def test_partial_selection_is_profile_driven_deterministic_and_budgeted(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            shard2 = next(item for item in model["shards"] if item["name"].startswith("model-00002"))
            selected, experts = ramdisk._select_partial(
                model, {"0:0": 1000, "0:1": 100}, shard2["size_bytes"]
            )
        self.assertEqual(selected, [shard2["name"]])
        self.assertEqual(experts, ["0:1"])
        self.assertLessEqual(sum(item["size_bytes"] for item in model["shards"] if item["name"] in selected), shard2["size_bytes"])

    def test_partial_selection_can_stage_ineligible_experts_via_tmpfs_slabs(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            model["experts"]["0:1"]["direct_map_eligible"] = False
            shard2 = next(
                item for item in model["shards"] if item["name"].startswith("model-00002")
            )
            selected, direct_experts = ramdisk._select_partial(
                model, {"0:1": 100}, shard2["size_bytes"]
            )
        self.assertEqual(selected, [shard2["name"]])
        self.assertEqual(direct_experts, [])

    def test_partial_plan_compares_shard_closures_with_same_budget_pinning(self):
        with ModelFixture() as fixture:
            profile = fixture.root / ".coli_usage"
            profile.write_text("0 0 1000\n0 1 10\n", encoding="utf-8")
            plan = ramdisk.build_plan(
                plan_args(fixture.root, mode="partial", capacity_gb=1),
                hardware=hardware_fixture(),
            )
        comparison = plan["profile"]["pin_comparison"]
        self.assertEqual(comparison["budget_bytes"], ramdisk.GIB)
        self.assertGreaterEqual(comparison["coverage"], plan["profile"]["coverage"])
        self.assertGreater(
            plan["profile"]["predicted_expert_bytes_avoided_per_staged_byte"], 0
        )

    def test_partial_mode_requires_a_profile_and_positive_budget(self):
        with ModelFixture() as fixture:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "positive --capacity"):
                ramdisk.build_plan(
                    plan_args(fixture.root, mode="partial"),
                    hardware=hardware_fixture(),
                )
            with self.assertRaisesRegex(ramdisk.RamdiskError, "requires .coli_usage"):
                ramdisk.build_plan(
                    plan_args(fixture.root, mode="partial", capacity_gb=1),
                    hardware=hardware_fixture(),
                )

    def test_profile_fingerprint_mismatch_is_rejected(self):
        with ModelFixture() as fixture:
            profile = fixture.root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "model_fingerprint": "sha256:not-this-model",
                        "counts": [{"layer": 0, "expert": 1, "count": 9}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ramdisk.RamdiskError, "fingerprint"):
                ramdisk.build_plan(
                    plan_args(fixture.root, mode="partial", capacity_gb=1, profile=str(profile)),
                    hardware=hardware_fixture(),
                )

    def test_capacity_refusal_preserves_os_and_runtime_reserve(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(available=1 * ramdisk.GIB)
            )
        self.assertTrue(any("reserve" in blocker for blocker in plan["blockers"]))
        self.assertGreaterEqual(plan["reserve"]["os_margin_bytes"], 16 * ramdisk.GIB)

    def test_cgroup_v2_uses_tightest_limiting_ancestor_headroom(self):
        with canonical_temporary_directory() as temporary:
            mountpoint = Path(temporary) / "cgroup"
            parent = mountpoint / "service.slice"
            leaf = parent / "colibri.scope"
            leaf.mkdir(parents=True)
            for directory, maximum, current, high in (
                (mountpoint, "max", "0", "max"),
                (parent, "1000", "450", "700"),
                (leaf, "900", "100", "650"),
            ):
                (directory / "memory.max").write_text(maximum, encoding="utf-8")
                (directory / "memory.current").write_text(current, encoding="utf-8")
                (directory / "memory.high").write_text(high, encoding="utf-8")
            info = ramdisk._discover_cgroup_memory(
                cgroup_text="0::/service.slice/colibri.scope\n",
                mountinfo_text=(
                    "36 25 0:32 /host.slice/container.scope %s "
                    "rw,nosuid,nodev,noexec - cgroup2 cgroup rw\n"
                    % mountpoint
                ),
            )

        self.assertEqual(info["version"], 2)
        self.assertEqual(info["status"], "limited")
        self.assertEqual(info["available_bytes"], 550)
        self.assertEqual(info["limit_bytes"], 1000)
        self.assertEqual(info["current_bytes"], 450)
        self.assertEqual(info["high_available_bytes"], 250)
        self.assertTrue(info["limiting_path"].endswith("service.slice"))

    def test_cgroup_discovery_fails_closed_when_proc_contract_is_unreadable(self):
        for cgroup_text, mountinfo_text, expected_path in (
            (None, "", "/proc/self/cgroup"),
            ("0::/\n", None, "/proc/self/mountinfo"),
        ):
            with self.subTest(path=expected_path), mock.patch(
                "builtins.open", side_effect=PermissionError("denied")
            ):
                info = ramdisk._discover_cgroup_memory(
                    cgroup_text=cgroup_text,
                    mountinfo_text=mountinfo_text,
                )

            self.assertEqual(info["status"], "unavailable")
            self.assertIn(expected_path, info["error"])

    def test_hybrid_cgroup_falls_back_to_visible_v1_memory_mount(self):
        with canonical_temporary_directory() as temporary:
            mountpoint = Path(temporary) / "memory"
            leaf = mountpoint / "legacy"
            leaf.mkdir(parents=True)
            (mountpoint / "memory.limit_in_bytes").write_text(
                "9223372036854771712", encoding="utf-8"
            )
            (mountpoint / "memory.usage_in_bytes").write_text(
                "0", encoding="utf-8"
            )
            (leaf / "memory.limit_in_bytes").write_text(
                "4096", encoding="utf-8"
            )
            (leaf / "memory.usage_in_bytes").write_text(
                "1024", encoding="utf-8"
            )
            info = ramdisk._discover_cgroup_memory(
                cgroup_text="0::/unified\n7:memory:/legacy\n",
                mountinfo_text=(
                    "42 25 0:38 / %s rw,nosuid,nodev,noexec "
                    "- cgroup cgroup rw,memory\n" % mountpoint
                ),
            )

        self.assertEqual(info["version"], 1)
        self.assertEqual(info["status"], "limited")
        self.assertEqual(info["available_bytes"], 3072)

    def test_cgroup_v1_memory_limit_has_compatible_headroom(self):
        with canonical_temporary_directory() as temporary:
            mountpoint = Path(temporary) / "memory"
            leaf = mountpoint / "colibri"
            leaf.mkdir(parents=True)
            (mountpoint / "memory.limit_in_bytes").write_text(
                "9223372036854771712", encoding="utf-8"
            )
            (mountpoint / "memory.usage_in_bytes").write_text("0", encoding="utf-8")
            (leaf / "memory.limit_in_bytes").write_text("2048", encoding="utf-8")
            (leaf / "memory.usage_in_bytes").write_text("512", encoding="utf-8")
            info = ramdisk._discover_cgroup_memory(
                cgroup_text="5:cpu:/other\n7:memory:/colibri\n",
                mountinfo_text=(
                    "42 25 0:38 / %s rw,nosuid,nodev,noexec - cgroup cgroup rw,memory\n"
                    % mountpoint
                ),
            )

        self.assertEqual(info["version"], 1)
        self.assertEqual(info["available_bytes"], 1536)
        self.assertEqual(info["limit_bytes"], 2048)
        self.assertIsNone(info["high_available_bytes"])

    def test_plan_caps_capacity_at_cgroup_hard_limit_and_warns_on_high(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture()
            hardware["cgroup_memory"] = {
                "version": 2,
                "status": "limited",
                "available_bytes": ramdisk.GIB,
                "high_available_bytes": ramdisk.GIB // 2,
                "error": None,
            }
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware)

        self.assertEqual(plan["reserve"]["available_bytes"], ramdisk.GIB)
        self.assertEqual(plan["reserve"]["host_available_bytes"], 128 * ramdisk.GIB)
        self.assertTrue(
            any("cgroup memory hard-limit headroom" in item for item in plan["blockers"])
        )
        self.assertTrue(any("memory.high" in item for item in plan["warnings"]))

    def test_plan_blocks_when_cgroup_memory_contract_cannot_be_read(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture()
            hardware["cgroup_memory"] = {
                "version": 2,
                "status": "unavailable",
                "available_bytes": None,
                "high_available_bytes": None,
                "error": "permission denied",
            }
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware)

        self.assertTrue(
            any(
                "cannot validate cgroup memory headroom" in item
                for item in plan["blockers"]
            )
        )

    def test_runtime_availability_is_capped_by_current_cgroup_headroom(self):
        cgroup = {"available_bytes": 400, "error": None}
        with mock.patch.object(
            ramdisk, "_meminfo", return_value={"MemAvailable": 1000}
        ), mock.patch.object(
            ramdisk, "_discover_cgroup_memory", return_value=cgroup
        ):
            self.assertEqual(ramdisk._available_memory(), 400)
        with mock.patch.object(
            ramdisk, "_node_meminfo", return_value={"MemFree": 900}
        ), mock.patch.object(
            ramdisk, "_discover_cgroup_memory", return_value=cgroup
        ):
            self.assertEqual(
                ramdisk._available_for_mount({"node": 0}, plan={}), 400
            )

    def test_per_node_replication_refuses_any_under_capacity_node(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture(available=18 * ramdisk.GIB, nodes=2)
            plan = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"), hardware=hardware
            )
        self.assertTrue(any("NUMA node" in blocker for blocker in plan["blockers"]))

    def test_missing_noswap_blocks_unless_explicitly_accepted(self):
        with ModelFixture() as fixture:
            blocked = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(noswap=False)
            )
            accepted = ramdisk.build_plan(
                plan_args(fixture.root, allow_swappable=True), hardware=hardware_fixture(noswap=False)
            )
        self.assertTrue(any("noswap" in blocker for blocker in blocked["blockers"]))
        self.assertFalse(any("noswap" in blocker for blocker in accepted["blockers"]))

    def test_protected_or_model_overlapping_mount_roots_are_blocked(self):
        with ModelFixture() as fixture:
            broad = ramdisk.build_plan(
                plan_args(fixture.root, mount_root="/"), hardware=hardware_fixture()
            )
            overlap = ramdisk.build_plan(
                plan_args(fixture.root, mount_root=str(fixture.root / "ram")), hardware=hardware_fixture()
            )
        self.assertTrue(any("protected broad" in blocker for blocker in broad["blockers"]))
        self.assertTrue(any("canonical model" in blocker for blocker in overlap["blockers"]))

    def test_invalid_numeric_planning_inputs_are_actionable(self):
        with ModelFixture() as fixture:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "finite positive"):
                ramdisk.build_plan(
                    plan_args(fixture.root, capacity_gb=float("nan")),
                    hardware=hardware_fixture(),
                )
            with self.assertRaisesRegex(ramdisk.RamdiskError, "--ctx"):
                ramdisk.build_plan(
                    plan_args(fixture.root, ctx=-1), hardware=hardware_fixture()
                )
            with self.assertRaisesRegex(ramdisk.RamdiskError, "--parallel"):
                ramdisk.build_plan(
                    plan_args(fixture.root, parallel=0), hardware=hardware_fixture()
                )

    def test_per_node_plan_reports_exact_replica_totals(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"),
                hardware=hardware_fixture(nodes=2),
            )
        self.assertEqual(plan["staging"]["replica_count"], 2)
        self.assertEqual(
            plan["staging"]["total_staged_bytes"],
            plan["staging"]["staged_bytes"] * 2,
        )
        self.assertEqual(
            plan["reserve"]["total_runtime_bytes"],
            plan["reserve"]["runtime_bytes"] * 2,
        )

    def test_sparse_effective_masks_remain_exact_in_shared_plan(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture(nodes=3)
            for row, node_id, cpus in zip(
                hardware["nodes"],
                (0, 2, 8),
                ([0, 1], [4, 5], [16, 17]),
            ):
                row.update(
                    {
                        "id": node_id,
                        "cpus": list(cpus),
                        "cpu_list": ramdisk._format_range_list(cpus),
                    }
                )
            hardware.update(
                {
                    "online_nodes": [0, 2, 8],
                    "effective_nodes": [0, 8],
                    "effective_cpus": [0, 1, 16, 17],
                    "core_groups": [[0], [1], [16], [17]],
                }
            )
            plan = ramdisk.build_plan(
                plan_args(
                    fixture.root,
                    memory_nodes="0,8",
                    cpu_list="0-1,16-17",
                ),
                hardware=hardware,
            )

        self.assertEqual(plan["placement"]["memory_nodes"], [0, 8])
        self.assertEqual(plan["placement"]["cpu_list"], "0-1,16-17")
        self.assertEqual(
            plan["mounts"][0]["policy"], r"interleave=static:0\,8"
        )
        self.assertEqual(plan["staging"]["replica_count"], 1)
        self.assertTrue(
            any("may fall back" in warning for warning in plan["warnings"])
        )

    def test_selected_replica_nodes_create_only_selected_full_copies(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(
                    fixture.root,
                    topology="per-node",
                    memory_nodes="0,2",
                    cpu_list="0-1,4-5",
                ),
                hardware=hardware_fixture(nodes=4),
            )

        self.assertEqual([mount["node"] for mount in plan["mounts"]], [0, 2])
        self.assertEqual(
            [mount["policy"] for mount in plan["mounts"]],
            ["bind=static:0", "bind=static:2"],
        )
        self.assertEqual(plan["staging"]["replica_count"], 2)
        self.assertEqual(
            [entry["cpu_list"] for entry in plan["placement"]["engine_cpu_sets"]],
            ["0-1", "4-5"],
        )

    def test_remote_cpu_selection_is_explicit_in_shared_mode(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(
                    fixture.root,
                    memory_nodes="0",
                    cpu_list="2-3",
                ),
                hardware=hardware_fixture(nodes=2),
        )

        self.assertEqual(plan["placement"]["remote_cpu_list"], "2-3")
        self.assertEqual(plan["placement"]["memory_policy"], "strict-bind")
        self.assertEqual(plan["mounts"][0]["policy"], "bind=static:0")
        environment = ramdisk._benchmark_environment(
            {"plan": plan},
            plan["mounts"][0]["path"],
            "/durable/state",
            True,
        )
        self.assertEqual(environment["COLI_NUMA"], "1")
        self.assertEqual(environment["COLI_NUMA_NODES"], "0")
        self.assertTrue(
            any("remote NUMA access" in warning for warning in plan["warnings"])
        )

    def test_replica_cpu_selection_cannot_name_unselected_nodes(self):
        with ModelFixture() as fixture:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError, "outside the selected replica nodes"
            ):
                ramdisk.build_plan(
                    plan_args(
                        fixture.root,
                        topology="per-node",
                        memory_nodes="0",
                        cpu_list="2-3",
                    ),
                    hardware=hardware_fixture(nodes=2),
                )

    def test_effective_cpuset_is_a_hard_placement_boundary(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture(nodes=2)
            hardware["effective_nodes"] = [0]
            hardware["effective_cpus"] = [0, 1]
            hardware["core_groups"] = [[0], [1]]
            with self.assertRaisesRegex(ramdisk.RamdiskError, "effective host mask"):
                ramdisk.build_plan(
                    plan_args(fixture.root, memory_nodes="1"),
                    hardware=hardware,
                )
            with self.assertRaisesRegex(ramdisk.RamdiskError, "effective host mask"):
                ramdisk.build_plan(
                    plan_args(fixture.root, cpu_list="2-3"),
                    hardware=hardware,
                )

    def test_cpu_selection_must_keep_effective_sibling_groups_whole(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture(nodes=1)
            hardware["core_groups"] = [[0, 1]]
            with self.assertRaisesRegex(ramdisk.RamdiskError, "whole effective physical cores"):
                ramdisk.build_plan(
                    plan_args(fixture.root, cpu_list="0"),
                    hardware=hardware,
                )

    def test_engine_threads_are_counted_only_inside_selected_cpu_mask(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture(nodes=1)
            hardware["nodes"][0]["physical_cores"] = 64
            hardware["physical_cores"] = 64
            hardware["effective_physical_cores"] = 64
            hardware["effective_cpus"] = [0]
            hardware["core_groups"] = [[0]]
            plan = ramdisk.build_plan(
                plan_args(fixture.root, cpu_list="0"),
                hardware=hardware,
            )

        self.assertEqual(
            plan["placement"]["engine_cpu_sets"][0]["physical_cores"], 1
        )
        self.assertEqual(ramdisk._node_core_count(plan), 1)

    def test_effective_mask_drift_changes_review_identity(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture(nodes=2)
            broad = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware)
            hardware["effective_nodes"] = [0]
            hardware["effective_cpus"] = [0, 1]
            hardware["core_groups"] = [[0], [1]]
            constrained = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware
            )

        self.assertNotEqual(
            ramdisk._plan_confirmation_token(broad),
            ramdisk._plan_confirmation_token(constrained),
        )

    def test_managed_launch_revalidation_rejects_cpuset_drift(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture(nodes=2)
            hardware["effective_mask_source"] = "kernel-task-status"
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware)
        current = hardware_fixture(nodes=2)
        current["effective_nodes"] = [0]
        current["effective_cpus"] = [0, 1]
        with mock.patch.object(ramdisk, "discover_hardware", return_value=current):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "changed since preparation"):
                ramdisk._assert_effective_masks_unchanged(plan)

    def test_numa_sampling_stride_visits_every_round_robin_residue(self):
        for nodes in (2, 4):
            indices = ramdisk._sample_page_indices(4096, 256, nodes)
            self.assertEqual({index % nodes for index in indices}, set(range(nodes)))

