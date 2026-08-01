"""RAM-disk model scanning, hardware discovery, and placement planning tests."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403


def quantized_expert_tensors(fmt, hidden=384, intermediate=256):
    tensors = []
    for projection, rows, columns in (
        ("gate_proj", intermediate, hidden),
        ("up_proj", intermediate, hidden),
        ("down_proj", hidden, intermediate),
    ):
        name = "model.layers.0.mlp.experts.0.%s.weight" % projection
        if fmt == 1:
            weight_bytes, scale_bytes = rows * columns, rows * 4
        elif fmt == 5:
            groups = (columns + 63) // 64
            weight_bytes, scale_bytes = rows * groups * 24, rows * groups * 4
        elif fmt == 6:
            weight_bytes = rows * ((columns + 255) // 256) * 98
            scale_bytes = 4
        elif fmt == 8:
            weight_bytes = rows * columns
            scale_bytes = ((rows + 127) // 128) * ((columns + 127) // 128) * 4
        else:
            raise AssertionError("unsupported test format")
        tensors.append((name, "U8", weight_bytes, [weight_bytes]))
        tensors.append((name + ".qs", "F32", scale_bytes, [scale_bytes // 4]))
    return tensors


class ScanAndPlanTest(unittest.TestCase):
    GLM_USAGE_HEADER = "-1 1 2\n-2 1 3815245270\n"

    def test_scan_indexes_complete_six_tensor_experts_and_sorted_shards(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
        self.assertEqual(model["shard_names"], sorted(model["shard_names"]))
        self.assertEqual(set(model["experts"]), {"0:0", "0:1"})
        self.assertEqual(len(model["experts"]["0:0"]["tensors"]), 6)
        self.assertEqual(model["experts"]["0:0"]["shards"], model["shard_names"])
        self.assertTrue(model["experts"]["0:1"]["direct_map_eligible"])

    def test_scan_accepts_all_direct_engine_expert_formats(self):
        for fmt in (5, 6, 8):
            with self.subTest(fmt=fmt), ModelFixture() as fixture:
                for shard in fixture.root.glob("*.safetensors"):
                    shard.unlink()
                tensors = quantized_expert_tensors(fmt)
                write_safetensors(fixture.root / "model.safetensors", tensors)
                config_path = fixture.root / "config.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                config.update(hidden_size=384, moe_intermediate_size=256)
                config_path.write_text(json.dumps(config), encoding="utf-8")
                model = ramdisk.scan_model(str(fixture.root))

                expert = model["experts"]["0:0"]
                self.assertTrue(expert["direct_map_eligible"])
                self.assertEqual(expert["tensor_bytes"], sum(item[2] for item in tensors))

    def test_scan_rejects_ambiguous_or_corrupt_direct_format_geometry(self):
        fixtures = {
            "unstamped E8/FP8 collision": (6, 98, 64, None),
            "wrong FP8 scale count": (8, 384, 256, 4),
        }
        for label, (fmt, hidden, intermediate, extra_scale_bytes) in fixtures.items():
            with self.subTest(case=label), ModelFixture() as fixture:
                for shard in fixture.root.glob("*.safetensors"):
                    shard.unlink()
                tensors = quantized_expert_tensors(fmt, hidden, intermediate)
                if extra_scale_bytes:
                    name, dtype, size, shape = tensors[1]
                    tensors[1] = (name, dtype, size + extra_scale_bytes,
                                  [shape[0] + extra_scale_bytes // 4])
                write_safetensors(fixture.root / "model.safetensors", tensors)
                config_path = fixture.root / "config.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                config.update(hidden_size=hidden, moe_intermediate_size=intermediate)
                config_path.write_text(json.dumps(config), encoding="utf-8")
                model = ramdisk.scan_model(str(fixture.root))

                self.assertFalse(model["experts"]["0:0"]["direct_map_eligible"])

    def test_scan_preserves_unstamped_int8_fp8_collision_inversion(self):
        with ModelFixture() as fixture:
            for shard in fixture.root.glob("*.safetensors"):
                shard.unlink()
            # gate/up [2,256] have two per-row scales and two FP8 blocks. The
            # engine's unstamped policy deliberately selects incumbent int8.
            tensors = quantized_expert_tensors(1, hidden=256, intermediate=2)
            write_safetensors(fixture.root / "model.safetensors", tensors)
            config_path = fixture.root / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config.update(hidden_size=256, moe_intermediate_size=2)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            model = ramdisk.scan_model(str(fixture.root))

        self.assertTrue(model["experts"]["0:0"]["direct_map_eligible"])

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

    def test_headered_profile_validates_model_dimensions_and_engine(self):
        with ModelFixture() as fixture:
            model = ramdisk.scan_model(str(fixture.root))
            profile = fixture.root / ".coli_usage"
            profile.write_text(
                self.GLM_USAGE_HEADER + "0 1 10\n",
                encoding="utf-8",
            )
            _, counts = ramdisk._load_profile(str(profile), model)
            self.assertEqual(counts, {"0:1": 10})

            profile.write_text(
                "-1 9 2\n-2 1 3815245270\n0 1 10\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ramdisk.RamdiskError, "dimensions"):
                ramdisk._load_profile(str(profile), model)

            profile.write_text(
                "-1 1 2\n-2 1 1\n0 1 10\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ramdisk.RamdiskError, "engine"):
                ramdisk._load_profile(str(profile), model)

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

    def test_numa_sampling_avoids_page_order_resonance(self):
        total_pages = 1 << 20
        sample_pages = 256
        for nodes in (2, 4):
            indices = ramdisk._sample_page_indices(
                total_pages,
                sample_pages,
                nodes,
            )
            self.assertEqual(
                indices,
                ramdisk._sample_page_indices(
                    total_pages,
                    sample_pages,
                    nodes,
                ),
            )
            self.assertEqual(len(indices), sample_pages)
            self.assertEqual(len(set(indices)), sample_pages)
            self.assertTrue(
                all(0 <= index < total_pages for index in indices)
            )
            for order in range(10):
                allocation_units = (
                    total_pages + (1 << order) - 1
                ) // (1 << order)
                if allocation_units < 7 * nodes:
                    continue
                counts = [
                    sum(
                        1
                        for index in indices
                        if (index >> order) % nodes == node
                    )
                    for node in range(nodes)
                ]
                ideal = float(sample_pages) / nodes
                deviation = max(
                    abs(count - ideal) / ideal
                    for count in counts
                )
                self.assertLessEqual(
                    deviation,
                    0.15,
                    "order %d across %d nodes: %r"
                    % (order, nodes, counts),
                )
        self.assertEqual(
            ramdisk._sample_page_indices(8, 20, 4),
            list(range(8)),
        )
