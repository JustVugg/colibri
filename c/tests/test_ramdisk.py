import argparse
import io
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))
import ramdisk  # noqa: E402


def write_safetensors(path, tensors):
    """Write a tiny valid safetensors file without third-party packages."""
    offset = 0
    header = {}
    payload = bytearray()
    for tensor in tensors:
        name, dtype, size = tensor[:3]
        shape = tensor[3] if len(tensor) > 3 else [size if dtype in ("U8", "I8") else size // 4]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        payload.extend(bytes(((offset + index) % 251 for index in range(size))))
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    while (8 + len(raw)) % 4:
        raw += b" "
    with open(path, "wb") as stream:
        stream.write(len(raw).to_bytes(8, "little"))
        stream.write(raw)
        stream.write(payload)


def expert_tensors(layer, expert, projections=("gate_proj", "up_proj", "down_proj")):
    rows = []
    for projection in projections:
        name = "model.layers.%d.mlp.experts.%d.%s.weight" % (layer, expert, projection)
        rows.append((name, "U8", 16, [4, 4]))
        rows.append((name + ".qs", "F32", 16, [4]))
    return rows


def hardware_fixture(available=128 * ramdisk.GIB, nodes=1, noswap=True, numactl="/usr/bin/numactl"):
    per_node = available // nodes
    node_rows = []
    for node in range(nodes):
        node_rows.append(
            {
                "id": node,
                "cpus": [node * 2, node * 2 + 1],
                "cpu_list": "%d-%d" % (node * 2, node * 2 + 1),
                "physical_cores": 2,
                "memory_total_bytes": per_node * 2,
                "memory_available_bytes": per_node,
                "distance": [10 if other == node else 20 for other in range(nodes)],
            }
        )
    effective_cpus = [
        cpu for node in node_rows for cpu in node["cpus"]
    ]
    return {
        "linux": True,
        "kernel_release": "6.8.0-test",
        "online_nodes": list(range(nodes)),
        "effective_nodes": list(range(nodes)),
        "effective_cpus": effective_cpus,
        "effective_cpu_list": ramdisk._format_range_list(effective_cpus),
        "core_groups": [[cpu] for cpu in effective_cpus],
        "nodes": node_rows,
        "physical_cores": nodes * 2,
        "effective_physical_cores": nodes * 2,
        "memory": {"total_bytes": available * 2, "available_bytes": available},
        "swap": {"configured": [], "used_bytes": 0},
        "tmpfs": {"supported": True, "noswap_supported": noswap},
        "thp": {
            "shmem_enabled": "always within_size [advise] never",
            "modes": ["always", "within_size", "advise", "never"],
            "within_size_supported": True,
            "advise_supported": True,
        },
        "numactl": numactl,
        "mount": "/bin/mount",
        "umount": "/bin/umount",
        "sudo": "/usr/bin/sudo",
        "hugetlb": {"total_pages": 0, "free_pages": 0, "page_size_bytes": 0},
    }


def set_asymmetric_node_cores(hardware, counts=(3, 5)):
    """Give fixture nodes realistic, disjoint single-thread core masks."""
    cpu = 0
    groups = []
    for node, count in zip(hardware["nodes"], counts):
        cpus = list(range(cpu, cpu + count))
        node["cpus"] = cpus
        node["cpu_list"] = ramdisk._format_range_list(cpus)
        node["physical_cores"] = count
        groups.extend([[item] for item in cpus])
        cpu += count
    effective_cpus = [item for group in groups for item in group]
    hardware["effective_cpus"] = effective_cpus
    hardware["effective_cpu_list"] = ramdisk._format_range_list(effective_cpus)
    hardware["core_groups"] = groups
    hardware["physical_cores"] = sum(counts)
    hardware["effective_physical_cores"] = sum(counts)


class ModelFixture:
    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # Expert 0 spans both shards. Expert 1 is a complete one-shard closure.
        write_safetensors(
            self.root / "model-00001-of-00002.safetensors",
            expert_tensors(0, 0, ("gate_proj",)) + [("model.embed_tokens.weight", "U8", 64)],
        )
        write_safetensors(
            self.root / "model-00002-of-00002.safetensors",
            expert_tensors(0, 0, ("up_proj", "down_proj")) + expert_tensors(0, 1),
        )
        (self.root / "config.json").write_text(
            json.dumps(
                {
                    "hidden_size": 4,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "n_routed_experts": 2,
                    "num_experts_per_tok": 1,
                    "moe_intermediate_size": 4,
                    "intermediate_size": 4,
                    "kv_lora_rank": 4,
                    "qk_nope_head_dim": 4,
                    "qk_rope_head_dim": 4,
                    "v_head_dim": 4,
                    "n_shared_experts": 1,
                    "vocab_size": 32,
                    "index_head_dim": 0,
                }
            ),
            encoding="utf-8",
        )
        (self.root / "tokenizer.json").write_text("{}", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self.temp.cleanup()


def plan_args(model, **overrides):
    values = {
        "model": str(model),
        "mode": "full",
        "topology": "interleaved",
        "capacity_gb": None,
        "profile": None,
        "mount_root": "/mnt/colibri-ram",
        "allow_swappable": False,
        "prefault": None,
        "parallel": 2,
        "ctx": 4096,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


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
        with tempfile.TemporaryDirectory() as temporary:
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
        with tempfile.TemporaryDirectory() as temporary:
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
        with tempfile.TemporaryDirectory() as temporary:
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


class TuiPlacementContractTest(unittest.TestCase):
    def test_error_footer_advertises_the_available_recovery_actions(self):
        plan = {
            "mounts": [{"path": "/mnt/colibri-test", "node": None}],
            "blockers": [],
        }

        destroy_only = ramdisk._tui_idle_action_hint(
            0, plan, {"present": True, "state": "error", "processes": []}
        )
        stop_then_destroy = ramdisk._tui_idle_action_hint(
            0,
            plan,
            {"present": True, "state": "error", "processes": [{"pid": 123}]},
        )

        self.assertEqual(destroy_only, "[d] destroy")
        self.assertEqual(stop_then_destroy, "[x] stop  [d] destroy")

    def test_four_node_default_is_one_shared_model_and_one_engine(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=4)
            )

        placement = ramdisk._placement_summary(plan, base_port=8000)
        confirmation = ramdisk._prepare_confirmation(plan, base_port=8000)

        self.assertEqual(plan["topology"], "interleaved")
        self.assertEqual(plan["staging"]["replica_count"], 1)
        self.assertEqual(placement["copy_count"], 1)
        self.assertEqual(placement["engine_count"], 1)
        self.assertIn("Single shared model", placement["title"])
        self.assertIn("1 complete model copy", placement["cost"])
        self.assertIn("1 engine", placement["cost"])
        self.assertIn("spread across 4 NUMA nodes", placement["explanation"])
        self.assertIn("1 complete model copy", confirmation)
        self.assertIn("1 engine on port 8000", confirmation)

    def test_four_node_replica_mode_names_every_full_copy_and_endpoint(self):
        with ModelFixture() as fixture:
            shared = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=4)
            )
            replicated = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"),
                hardware=hardware_fixture(nodes=4),
            )

        placement = ramdisk._placement_summary(replicated, base_port=8000)
        confirmation = ramdisk._prepare_confirmation(replicated, base_port=8000)

        self.assertEqual(placement["copy_count"], 4)
        self.assertEqual(placement["engine_count"], 4)
        self.assertIn("Independent full-model replicas", placement["title"])
        self.assertIn("4 complete model copies", placement["cost"])
        self.assertIn("4 independent engines", placement["cost"])
        self.assertIn("ports 8000, 8001, 8002, 8003", placement["endpoints"])
        self.assertIn("replication, not model sharding", placement["explanation"])
        self.assertIn("4 complete model copies", confirmation)
        self.assertIn("4 independent engines", confirmation)
        self.assertIn("not sharding", confirmation)
        self.assertNotEqual(
            ramdisk._plan_confirmation_token(shared),
            ramdisk._plan_confirmation_token(replicated),
        )

    def test_prepare_rejects_a_plan_changed_after_tui_confirmation(self):
        with ModelFixture() as fixture:
            shared = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
            replicated = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"),
                hardware=hardware_fixture(nodes=2),
            )
            reviewed = ramdisk._plan_confirmation_token(shared)
            with mock.patch.object(ramdisk, "_load_manifest", return_value=None), mock.patch.object(
                ramdisk, "build_plan", return_value=replicated
            ), mock.patch.object(ramdisk, "_mount_tmpfs") as mount:
                with self.assertRaisesRegex(ramdisk.RamdiskError, "changed since review"):
                    ramdisk.prepare.__wrapped__(
                        plan_args(fixture.root, yes=True),
                        display_plan=False,
                        expected_plan_token=reviewed,
                    )
        mount.assert_not_called()

    def test_destroy_rejects_a_replacement_after_tui_confirmation(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
        mount = dict(plan["mounts"][0])
        mount["identity"] = {"mount_id": 41, "device": "0:41"}
        reviewed_manifest = {
            "version": ramdisk.MANIFEST_VERSION,
            "deployment_id": "a" * 32,
            "created_at": "2026-07-22T10:00:00+00:00",
            "model_fingerprint": plan["model"]["fingerprint"],
            "plan": plan,
            "mounts": [mount],
            "processes": [],
        }
        replacement_manifest = dict(reviewed_manifest)
        replacement_mount = dict(mount)
        replacement_mount["identity"] = {"mount_id": 42, "device": "0:42"}
        replacement_manifest.update(
            {
                "deployment_id": "b" * 32,
                "created_at": "2026-07-22T10:00:01+00:00",
                "mounts": [replacement_mount],
            }
        )
        reviewed_token = ramdisk._manifest_confirmation_token(reviewed_manifest)

        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=replacement_manifest
        ), mock.patch.object(ramdisk, "_confirm") as confirm, mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "changed since review"):
                ramdisk.destroy.__wrapped__(
                    argparse.Namespace(yes=True),
                    expected_manifest_token=reviewed_token,
                )

        confirm.assert_not_called()
        unmount.assert_not_called()

    def test_destroy_confirmation_expires_when_process_state_changes(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
        manifest = {
            "version": ramdisk.MANIFEST_VERSION,
            "deployment_id": "a" * 32,
            "created_at": "2026-07-22T10:00:00+00:00",
            "state": "ready",
            "model_fingerprint": plan["model"]["fingerprint"],
            "plan": plan,
            "mounts": [],
            "processes": [],
        }
        reviewed_token = ramdisk._manifest_confirmation_token(manifest)
        started = dict(manifest)
        started["state"] = "starting"
        started["processes"] = [
            {
                "pid": 1234,
                "pgid": 1234,
                "uid": 1000,
                "starttime": 5678,
                "nonce": "managed",
                "port": 8000,
                "node": None,
            }
        ]

        self.assertNotEqual(
            reviewed_token,
            ramdisk._manifest_confirmation_token(started),
        )

    def test_destroy_confirmation_expires_when_endpoint_changes(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
        manifest = {
            "version": ramdisk.MANIFEST_VERSION,
            "deployment_id": "a" * 32,
            "created_at": "2026-07-22T10:00:00+00:00",
            "state": "ready",
            "base_port": 8000,
            "model_fingerprint": plan["model"]["fingerprint"],
            "plan": plan,
            "mounts": [],
            "processes": [],
        }
        reviewed_token = ramdisk._manifest_confirmation_token(manifest)
        changed_endpoint = dict(manifest, base_port=9000)

        self.assertNotEqual(
            reviewed_token,
            ramdisk._manifest_confirmation_token(changed_endpoint),
        )

    def test_minimum_viewport_pins_replica_warning_before_confirmation(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"),
                hardware=hardware_fixture(nodes=4),
            )
        report = {"present": False, "state": "absent"}
        rows = ramdisk._tui_plan_rows(
            plan,
            report,
            active=False,
            base_port=8000,
            confirmation=ramdisk._prepare_confirmation(plan, 8000),
        )
        minimum_content = "\n".join(
            line for _, line in ramdisk._tui_wrap_rows(rows, 35)[:3]
        )

        self.assertIn("4 complete model copies", minimum_content)
        self.assertIn("4 independent engines", minimum_content)
        self.assertIn("replication, not sharding", minimum_content)

    def test_prepare_confirmation_cannot_coexist_with_scrolled_content(self):
        for requested_scroll in (1, 3, 100):
            with self.subTest(requested_scroll=requested_scroll):
                self.assertEqual(
                    ramdisk._tui_review_scroll("prepare", requested_scroll), 0
                )

        self.assertEqual(ramdisk._tui_review_scroll(None, 3), 3)
        self.assertEqual(ramdisk._tui_review_scroll("destroy", 3), 3)

    def test_running_plan_does_not_verify_a_dead_managed_process(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
        report = {
            "present": True,
            "state": "running",
            "deep_validation": False,
            "source_fingerprint_verified": None,
            "mounts": [
                {
                    "verified": True,
                    "namespace_verified": None,
                }
            ],
            "processes": [
                {
                    "running": False,
                    "verified": False,
                }
            ],
        }
        rendered = "\n".join(
            text for _, text in ramdisk._tui_plan_rows(plan, report, active=True)
        )

        self.assertIn("DEPLOYMENT NEEDS ATTENTION", rendered)
        self.assertNotIn("DEPLOYMENT VERIFIED", rendered)

    def test_legacy_stopped_manifest_recovers_its_previous_base_port(self):
        manifest = {
            "processes": [
                {
                    "port": 9100,
                    "node": None,
                    "stopped_at": "2026-07-22T10:00:00+00:00",
                }
            ],
            "ports": [9100],
            "mounts": [{"node": None}],
        }
        self.assertEqual(ramdisk._persisted_base_port(manifest), 9100)

    def test_manifest_rejects_boolean_base_port(self):
        with mock.patch.object(
            ramdisk,
            "_read_json",
            return_value={"version": ramdisk.MANIFEST_VERSION, "base_port": True},
        ):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "base port"):
                ramdisk._load_manifest(required=True)

    def test_cancelled_prepare_stops_before_mounting(self):
        cancel = threading.Event()
        cancel.set()
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
            with mock.patch.object(
                ramdisk, "_load_manifest", return_value=None
            ), mock.patch.object(ramdisk, "build_plan", return_value=plan), mock.patch.object(
                ramdisk, "_mount_tmpfs"
            ) as mount:
                with self.assertRaisesRegex(
                    ramdisk._OperationCancelled, "cancelled by user"
                ):
                    ramdisk.prepare.__wrapped__(
                        plan_args(fixture.root, yes=True),
                        display_plan=False,
                        cancel_event=cancel,
                    )
        mount.assert_not_called()

    def test_cancelled_prepare_does_not_hide_rollback_failure(self):
        cancel = threading.Event()
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
        actual = {
            "mount_id": 91,
            "device": "0:91",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }

        def cancel_copy(*args, **kwargs):
            cancel.set()
            ramdisk._raise_if_cancelled(cancel)

        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=None
        ), mock.patch.object(ramdisk, "build_plan", return_value=plan), mock.patch.object(
            ramdisk, "_save_manifest"
        ), mock.patch.object(
            ramdisk, "_mount_at", side_effect=[None, actual, actual]
        ), mock.patch.object(ramdisk, "_mount_tmpfs"), mock.patch.object(
            ramdisk, "_validate_mount", return_value=actual
        ), mock.patch.object(ramdisk, "_populate_mount", side_effect=cancel_copy), mock.patch.object(
            ramdisk, "_umount_path", side_effect=OSError("sudo ticket expired")
        ):
            with self.assertRaises(ramdisk.RamdiskError) as raised:
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True),
                    display_plan=False,
                    cancel_event=cancel,
                )

        self.assertNotIsInstance(raised.exception, ramdisk._OperationCancelled)
        self.assertIn("rollback/reporting errors", str(raised.exception))
        self.assertIn("sudo ticket expired", str(raised.exception))

    def test_clean_prepare_cancellation_removes_recovery_manifest(self):
        cancel = threading.Event()
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture(nodes=2)
            )
        actual = {
            "mount_id": 92,
            "device": "0:92",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }

        def cancel_copy(*args, **kwargs):
            cancel.set()
            ramdisk._raise_if_cancelled(cancel)

        with mock.patch.object(
            ramdisk, "_load_manifest", return_value=None
        ), mock.patch.object(ramdisk, "build_plan", return_value=plan), mock.patch.object(
            ramdisk, "_save_manifest"
        ), mock.patch.object(
            ramdisk, "_mount_at", side_effect=[None, actual, actual]
        ), mock.patch.object(ramdisk, "_mount_tmpfs"), mock.patch.object(
            ramdisk, "_validate_mount", return_value=actual
        ), mock.patch.object(ramdisk, "_populate_mount", side_effect=cancel_copy), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount, mock.patch.object(ramdisk, "_durable_unlink") as unlink:
            with self.assertRaises(ramdisk._OperationCancelled):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True),
                    display_plan=False,
                    cancel_event=cancel,
                )

        unmount.assert_called_once_with(plan["mounts"][0]["path"], plan["hardware"])
        unlink.assert_called_once_with(ramdisk._manifest_path())

    def test_tui_workers_use_noninteractive_sudo_after_foreground_validation(self):
        command = ["/usr/bin/mount", "-t", "tmpfs"]
        with mock.patch.object(ramdisk.os, "geteuid", return_value=1000), mock.patch.object(
            ramdisk, "_trusted_system_binary", return_value="/usr/bin/sudo"
        ):
            foreground = ramdisk._privileged(command, {})
            with ramdisk._noninteractive_privilege():
                background = ramdisk._privileged(command, {})

        self.assertEqual(foreground[:2], ["/usr/bin/sudo", "--"])
        self.assertEqual(background[:3], ["/usr/bin/sudo", "-n", "--"])

    def test_sudo_keepalive_never_prompts(self):
        stop = mock.Mock()
        stop.is_set.return_value = False
        stop.wait.return_value = True
        completed = subprocess.CompletedProcess(
            ["/usr/bin/sudo", "-n", "-v"],
            0,
        )
        with mock.patch.object(
            ramdisk.subprocess,
            "run",
            return_value=completed,
        ) as run:
            ramdisk._sudo_ticket_keepalive(
                stop,
                "/usr/bin/sudo",
                interval=0.01,
            )

        run.assert_called_once_with(
            ["/usr/bin/sudo", "-n", "-v"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5.0,
        )
        stop.wait.assert_called_once_with(0.01)

    def test_failed_sudo_keepalive_requests_cancellable_rollback(self):
        stop = mock.Mock()
        stop.is_set.return_value = False
        failure = threading.Event()
        cancel = threading.Event()
        completed = subprocess.CompletedProcess(
            ["/usr/bin/sudo", "-n", "-v"],
            1,
        )
        with mock.patch.object(
            ramdisk.subprocess,
            "run",
            return_value=completed,
        ):
            ramdisk._sudo_ticket_keepalive(
                stop,
                "/usr/bin/sudo",
                interval=0.01,
                failure_event=failure,
                cancel_event=cancel,
            )

        self.assertTrue(failure.is_set())
        self.assertTrue(cancel.is_set())
        stop.wait.assert_not_called()

    def test_sudo_authorization_is_checked_for_background_reuse(self):
        completed = subprocess.CompletedProcess(
            ["/usr/bin/sudo", "-n", "-v"],
            0,
        )
        with mock.patch.object(
            ramdisk.subprocess,
            "run",
            return_value=completed,
        ) as run:
            result = ramdisk._validate_noninteractive_sudo("/usr/bin/sudo")

        self.assertEqual(result.returncode, 0)
        run.assert_called_once_with(
            ["/usr/bin/sudo", "-n", "-v"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def test_interface_failure_reports_concurrent_cleanup_failure(self):
        interface_error = RuntimeError("display failed")
        cleanup_error = ramdisk.RamdiskError("rollback failed")
        worker_thread = mock.Mock()
        worker_thread.is_alive.return_value = False
        cancel_event = mock.Mock()
        ramdisk._tui_worker = {
            "cancelable": True,
            "cancel_event": cancel_event,
            "thread": worker_thread,
            "error": cleanup_error,
        }
        self.addCleanup(setattr, ramdisk, "_tui_worker", None)

        with mock.patch.dict(
            os.environ, {"COLI_RAMDISK_UI": "curses"}
        ), mock.patch("curses.wrapper", side_effect=interface_error):
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "interface failed while active operation cleanup also failed",
            ) as caught:
                ramdisk.launch_tui(argparse.Namespace())

        self.assertIs(caught.exception.__cause__, interface_error)
        cancel_event.set.assert_called_once_with()
        worker_thread.join.assert_called_once_with()
        self.assertIsNone(ramdisk._tui_worker)

    def test_minimum_width_settings_input_uses_a_full_entry_row(self):
        import curses

        class FakeScreen:
            def __init__(self):
                self.keys = iter(
                    [curses.KEY_RIGHT] * (len(ramdisk._TUI_SCREENS) - 1)
                    + [ord("o"), ord("q")]
                )
                self.getstr_call = None

            def timeout(self, milliseconds):
                pass

            def getmaxyx(self):
                return (8, 38)

            def erase(self):
                pass

            def addnstr(self, row, column, value, limit, attribute=0):
                pass

            def refresh(self):
                pass

            def getch(self):
                return next(self.keys)

            def move(self, row, column):
                pass

            def clrtoeol(self):
                pass

            def getstr(self, row, column, limit):
                self.getstr_call = (row, column, limit)
                return b""

        with ModelFixture() as fixture:
            hardware = hardware_fixture(nodes=2)
            model = ramdisk.scan_model(str(fixture.root))
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware, model=model
            )
            report = {
                "present": False,
                "state": "absent",
                "mounts": [],
                "processes": [],
            }
            screen = FakeScreen()
            with mock.patch.object(
                ramdisk, "discover_hardware", return_value=hardware
            ), mock.patch.object(ramdisk, "scan_model", return_value=model), mock.patch.object(
                ramdisk, "build_plan", return_value=plan
            ), mock.patch.object(ramdisk, "status", return_value=report), mock.patch.object(
                curses, "curs_set"
            ), mock.patch.object(curses, "echo"), mock.patch.object(curses, "noecho"):
                result = ramdisk._tui(
                    screen, plan_args(fixture.root), "/fake/coli", "/fake/engine"
                )

        self.assertEqual(result, 0)
        self.assertEqual(screen.getstr_call[1], 2)
        self.assertGreaterEqual(screen.getstr_call[2], 34)

    def test_tui_has_no_single_key_path_into_replica_mode(self):
        class FakeScreen:
            def __init__(self):
                self.keys = iter((ord("t"), ord("q")))
                self.output = []

            def timeout(self, milliseconds):
                self.timeout_ms = milliseconds

            def getmaxyx(self):
                return (24, 100)

            def erase(self):
                pass

            def addnstr(self, row, column, value, limit, attribute=0):
                self.output.append(str(value)[:limit])

            def refresh(self):
                pass

            def getch(self):
                return next(self.keys)

        with ModelFixture() as fixture:
            hardware = hardware_fixture(nodes=4)
            model = ramdisk.scan_model(str(fixture.root))
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware, model=model
            )
            report = {
                "present": False,
                "state": "absent",
                "mounts": [],
                "processes": [],
                "deep_validation": True,
            }
            observed = []

            def build(args, **kwargs):
                observed.append(args.topology)
                return plan

            screen = FakeScreen()
            with mock.patch.object(ramdisk, "discover_hardware", return_value=hardware), mock.patch.object(
                ramdisk, "scan_model", return_value=model
            ), mock.patch.object(ramdisk, "build_plan", side_effect=build), mock.patch.object(
                ramdisk, "status", return_value=report
            ):
                result = ramdisk._tui(
                    screen, plan_args(fixture.root), "/fake/coli", "/fake/engine"
                )

        self.assertEqual(result, 0)
        self.assertEqual(observed, ["interleaved"])
        rendered = "\n".join(screen.output)
        self.assertIn("Single shared model", rendered)
        self.assertIn("1 complete model copy", rendered)


class MountAndCopyTest(unittest.TestCase):
    def test_mountinfo_preserves_noncontiguous_mpol_nodemask(self):
        line = (
            "36 25 0:32 / /mnt/colibri-ram rw,noatime - tmpfs tmpfs "
            "rw,noswap,nodev,nosuid,noexec,mode=700,huge=within_size,"
            "mpol=interleave:0-1,3\n"
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            parsed = ramdisk._mount_table(stream.name)
        self.assertEqual(parsed[0]["super_options"][-1], "rw")
        self.assertIn("mpol=interleave:0-1,3", parsed[0]["super_options"])

    def test_mount_falls_back_from_within_size_to_advise_only_on_option_error(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture())
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if len(calls) == 1:
                return subprocess.CompletedProcess(command, 32, "", "mount: invalid argument")
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(
            ramdisk, "_trusted_system_binary", return_value="/bin/mount"
        ), mock.patch.object(ramdisk, "_run", side_effect=run), mock.patch.object(
            ramdisk, "_privileged", side_effect=lambda command, hardware: command
        ):
            ramdisk._mount_tmpfs(plan, plan["mounts"][0])
        self.assertIn("huge=within_size", calls[0][4])
        self.assertIn("huge=advise", calls[1][4])
        self.assertTrue(plan["mounts"][0]["effective_noswap"])

    def test_swappable_fallback_preserves_supported_within_size_thp(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root, allow_swappable=True),
                hardware=hardware_fixture(),
            )
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if len(calls) <= 2:
                return subprocess.CompletedProcess(
                    command, 32, "", "mount: invalid argument"
                )
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(
            ramdisk, "_trusted_system_binary", return_value="/bin/mount"
        ), mock.patch.object(ramdisk, "_run", side_effect=run), mock.patch.object(
            ramdisk, "_privileged", side_effect=lambda command, hardware: command
        ):
            ramdisk._mount_tmpfs(plan, plan["mounts"][0])
        self.assertEqual(len(calls), 3)
        self.assertIn("huge=within_size", calls[2][4])
        self.assertNotIn("noswap", calls[2][4].split(","))
        self.assertFalse(plan["mounts"][0]["effective_noswap"])

    def test_mount_keeps_private_tmpfs_over_reusable_underlying_directory(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture())
        options = ramdisk._mount_option_list(plan, plan["mounts"][0])
        self.assertIn("mode=0700", options)
        self.assertIn("X-mount.mkdir=0755", options)

    def test_non_option_mount_error_does_not_retry_weaker_options(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture())
        result = subprocess.CompletedProcess([], 1, "", "permission denied")
        with mock.patch.object(
            ramdisk, "_trusted_system_binary", return_value="/bin/mount"
        ), mock.patch.object(
            ramdisk, "_run", return_value=result
        ) as run, mock.patch.object(
            ramdisk, "_privileged", side_effect=lambda command, hardware: command
        ):
            with self.assertRaises(ramdisk.RamdiskError):
                ramdisk._mount_tmpfs(plan, plan["mounts"][0])
        self.assertEqual(run.call_count, 1)

    def test_interrupted_successful_mount_is_identified_and_rolled_back_locally(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root),
                hardware=hardware_fixture(),
            )
        mount = plan["mounts"][0]
        actual = {
            "mount_id": 44,
            "device": "0:44",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        interrupted = ramdisk._TuiTerminationSignal(signal.SIGTERM)
        with mock.patch.object(
            ramdisk,
            "_trusted_system_binary",
            return_value="/bin/mount",
        ), mock.patch.object(
            ramdisk,
            "_privileged",
            side_effect=lambda command, hardware: command,
        ), mock.patch.object(
            ramdisk,
            "_run",
            side_effect=interrupted,
        ), mock.patch.object(
            ramdisk,
            "_mount_at",
            return_value=actual,
        ), mock.patch.object(
            ramdisk,
            "_validate_mount",
            return_value=actual,
        ) as validate, mock.patch.object(
            ramdisk,
            "_umount_path",
        ) as unmount:
            with self.assertRaises(ramdisk._TuiTerminationSignal):
                ramdisk._mount_tmpfs(plan, mount)

        attempted = validate.call_args.args[0]
        self.assertEqual(attempted["effective_thp"], "within_size")
        self.assertTrue(attempted["effective_noswap"])
        unmount.assert_called_once_with(mount["path"], plan["hardware"])

    def test_prepare_immediately_rolls_back_identityless_successful_mount(self):
        with ModelFixture() as fixture, mock.patch.object(
            ramdisk, "_filesystem_for_path", return_value="ext4"
        ):
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
        with mock.patch.object(ramdisk, "_load_manifest", return_value=None), mock.patch.object(
            ramdisk, "build_plan", return_value=plan
        ), mock.patch.object(ramdisk, "_save_manifest"), mock.patch.object(
            ramdisk, "_mount_at", side_effect=[None, None]
        ), mock.patch.object(ramdisk, "_mount_tmpfs"), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "rolled it back"):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True), display_plan=False
                )
        unmount.assert_called_once_with(plan["mounts"][0]["path"], plan["hardware"])

    def test_prepare_cleanup_runs_even_when_error_manifest_cannot_be_saved(self):
        with ModelFixture() as fixture, mock.patch.object(
            ramdisk, "_filesystem_for_path", return_value="ext4"
        ):
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
        actual = {
            "mount_id": 9,
            "device": "0:42",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        with mock.patch.object(ramdisk, "_load_manifest", return_value=None), mock.patch.object(
            ramdisk, "build_plan", return_value=plan
        ), mock.patch.object(
            ramdisk, "_save_manifest", side_effect=[None, OSError("state full"), OSError("state full"), OSError("state full")]
        ), mock.patch.object(
            ramdisk, "_mount_at", side_effect=[None, actual, actual]
        ), mock.patch.object(ramdisk, "_mount_tmpfs"), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "rollback/reporting errors"):
                ramdisk.prepare.__wrapped__(
                    plan_args(fixture.root, yes=True), display_plan=False
                )
        unmount.assert_called_once_with(plan["mounts"][0]["path"], plan["hardware"])

    def test_copy_uses_atomic_publish_validates_header_and_removes_source_cache(self):
        with ModelFixture() as fixture, tempfile.TemporaryDirectory() as destination:
            source = fixture.root / "model-00001-of-00002.safetensors"
            target = Path(destination) / source.name
            ramdisk._copy_one(
                str(source), str(target), source.stat().st_size, 0, available=lambda: ramdisk.GIB
            )
            self.assertEqual(target.stat().st_size, source.stat().st_size)
            self.assertEqual(target.stat().st_mode & 0o222, 0)
            self.assertFalse(any(".coli-copy-" in item.name for item in Path(destination).iterdir()))

    def test_mount_validation_rejects_foreign_filesystem(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture())
        mount = plan["mounts"][0]
        foreign = {
            "mount_id": 3,
            "device": "8:1",
            "filesystem": "ext4",
            "source": "/dev/sda1",
            "options": ["rw", "noatime", "nodev", "nosuid", "noexec"],
            "super_options": [],
        }
        with mock.patch.object(ramdisk, "_mount_at", return_value=foreign):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "foreign"):
                ramdisk._validate_mount(mount, plan)

    def test_mount_validation_requires_managed_thp_numa_and_safety_options(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture(nodes=2))
        mount = plan["mounts"][0]
        actual = {
            "mount_id": 9,
            "device": "0:42",
            "filesystem": "tmpfs",
            "source": "tmpfs",
            "options": ["rw", "noatime", "nodev", "nosuid", "noexec"],
            "super_options": [
                "mode=700",
                "noswap",
                "huge=within_size",
                "mpol=interleave=static:0-1",
            ],
        }
        with mock.patch.object(ramdisk, "_mount_at", return_value=actual):
            self.assertEqual(ramdisk._validate_mount(mount, plan)["mount_id"], 9)
        actual["super_options"].remove("mpol=interleave=static:0-1")
        with mock.patch.object(ramdisk, "_mount_at", return_value=actual):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "NUMA policy"):
                ramdisk._validate_mount(mount, plan)

    def test_single_selected_node_rejects_sampled_pages_on_another_host_node(self):
        with ModelFixture() as fixture, tempfile.TemporaryDirectory() as destination:
            plan = ramdisk.build_plan(
                plan_args(fixture.root, memory_nodes="0"),
                hardware=hardware_fixture(nodes=2),
            )
            for source in fixture.root.glob("*.safetensors"):
                target = Path(destination) / source.name
                shutil.copy2(source, target)
                target.chmod(0o400)
            mount = {"path": destination, "node": None}
            with mock.patch.object(
                ramdisk,
                "_sample_numa_allocation",
                return_value={"1": 128},
            ):
                with self.assertRaisesRegex(
                    ramdisk.RamdiskError, "escaped the reviewed NUMA"
                ):
                    ramdisk._validate_namespace(plan, mount)

    def test_mount_lookup_rejects_stacked_exact_paths(self):
        mounts = [
            {"mount_id": 4, "path": "/mnt/colibri-test"},
            {"mount_id": 9, "path": "/mnt/colibri-test"},
        ]
        with mock.patch.object(ramdisk, "_mount_table", return_value=mounts):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "stacked mounts"):
                ramdisk._mount_at("/mnt/colibri-test")


class StateAndSafetyTest(unittest.TestCase):
    FINGERPRINT = "sha256:" + ("a" * 64)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(
            os.environ,
            {
                "XDG_STATE_HOME": os.path.join(self.temp.name, "state"),
                "COLI_RAMDISK_MANIFEST": os.path.join(self.temp.name, "manifest.json"),
            },
        )
        self.env.start()
        self.filesystem = mock.patch.object(ramdisk, "_filesystem_for_path", return_value="ext4")
        self.filesystem.start()

    def tearDown(self):
        self.filesystem.stop()
        self.env.stop()
        self.temp.cleanup()

    def manifest(self, state="ready", mount_paths=None, processes=None):
        """Return a schema-valid lifecycle record for focused safety tests."""
        mount_paths = mount_paths or ["/mnt/colibri-test"]
        processes = processes or []
        topology = "per-node" if len(mount_paths) > 1 else "interleaved"
        mount_root = mount_paths[0] if topology == "interleaved" else os.path.dirname(mount_paths[0])
        nodes = list(range(len(mount_paths))) if topology == "per-node" else [0]
        planned = [
            {"path": path, "node": nodes[index] if topology == "per-node" else None}
            for index, path in enumerate(mount_paths)
        ]
        mounted = [
            {
                "path": path,
                "node": planned[index]["node"],
                "identity": {
                    "mount_id": index + 4,
                    "device": "0:%d" % (index + 9),
                },
            }
            for index, path in enumerate(mount_paths)
        ]
        fingerprint_dir = self.FINGERPRINT.split(":", 1)[1]
        complete_processes = []
        for index, partial in enumerate(processes):
            record = dict(partial)
            node = planned[index]["node"]
            port = 8000 + index
            label = "interleaved" if node is None else "node-%d" % node
            record.update(
                {
                    "pgid": record["pid"],
                    "uid": os.getuid(),
                    "starttime": 100 + record["pid"],
                    "nonce": "%048x" % (index + 1),
                    "port": port,
                    "node": node,
                    "weights_dir": planned[index]["path"],
                    "state_dir": os.path.join(
                        ramdisk._state_root(), "engines", fingerprint_dir, label
                    ),
                    "command": [
                        str(C_DIR / "coli"),
                        "serve",
                        "--model",
                        os.path.join(self.temp.name, "model"),
                        "--port",
                        str(port),
                    ],
                }
            )
            complete_processes.append(record)
        return {
            "version": ramdisk.MANIFEST_VERSION,
            "state": state,
            "model_fingerprint": self.FINGERPRINT,
            "plan": {
                "topology": topology,
                "mount_root": mount_root,
                "mounts": planned,
                "hardware": hardware_fixture(nodes=len(mount_paths) if topology == "per-node" else 1),
                "model": {
                    "path": os.path.join(self.temp.name, "model"),
                    "fingerprint": self.FINGERPRINT,
                },
                "durable_state": {
                    "root": ramdisk._state_root(),
                    "manifest": ramdisk._manifest_path(),
                    "benchmarks": ramdisk._benchmarks_path(),
                },
                "source_shards": [{"name": "model.safetensors"}],
            },
            "mounts": mounted,
            "processes": complete_processes,
        }

    def test_usage_delta_merge_and_crash_recovery_are_idempotent(self):
        model_dir = os.path.join(self.temp.name, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        state_dir = os.path.join(self.temp.name, "node-state")
        os.makedirs(model_dir, mode=0o750)
        os.makedirs(state_dir, mode=0o700)
        ramdisk._usage_write(canonical, {"0:1": 10, "0:2": 3})
        ramdisk._usage_write(os.path.join(state_dir, ".coli_usage"), {"0:1": 14, "0:2": 3, "0:3": 2})
        record = {"state_dir": state_dir, "usage_baseline": {"0:1": 10, "0:2": 3}}
        ramdisk._merge_usage(record, canonical)
        self.assertEqual(ramdisk._usage_read(canonical), {"0:1": 14, "0:2": 3, "0:3": 2})
        ramdisk._merge_usage(record, canonical)
        self.assertEqual(ramdisk._usage_read(canonical), {"0:1": 14, "0:2": 3, "0:3": 2})

        merge_id = "a" * 32
        delta_path = os.path.join(state_dir, ".coli_usage.delta.json")
        ramdisk._usage_write(canonical, {"0:1": 15}, merge_id=merge_id)
        ramdisk._atomic_json(delta_path, {"version": 1, "id": merge_id, "delta": {"0:1": 1}})
        ramdisk._recover_delta(state_dir, canonical)
        self.assertEqual(ramdisk._usage_read(canonical)["0:1"], 15)
        self.assertFalse(os.path.exists(delta_path))
        self.assertEqual(os.stat(model_dir).st_mode & 0o777, 0o750)

    def test_atomic_json_never_chmods_an_existing_override_parent(self):
        parent = os.path.join(self.temp.name, "shared-parent")
        os.mkdir(parent, 0o755)
        before = os.stat(parent).st_mode & 0o777
        ramdisk._atomic_json(os.path.join(parent, "manifest.json"), {"ok": True})
        self.assertEqual(os.stat(parent).st_mode & 0o777, before)

    def test_private_state_directory_rejects_existing_symlink_without_chmod(self):
        target = os.path.join(self.temp.name, "redirect-target")
        link = os.path.join(self.temp.name, "redirect-link")
        os.mkdir(target, 0o755)
        os.symlink(target, link)
        before = os.stat(target).st_mode & 0o777
        with self.assertRaisesRegex(ramdisk.RamdiskError, "contains a symlink"):
            ramdisk._ensure_private_dir(link)
        self.assertEqual(os.stat(target).st_mode & 0o777, before)

    def test_derived_state_directory_must_remain_on_durable_filesystem(self):
        state_dir = os.path.join(self.temp.name, "engine-state")
        os.mkdir(state_dir)
        with mock.patch.object(ramdisk, "_filesystem_for_path", return_value="tmpfs"):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "volatile filesystem"):
                ramdisk._assert_durable_state_dir(state_dir, plan=self.manifest()["plan"])

    def test_two_node_usage_markers_survive_crash_between_manifest_saves(self):
        model_dir = os.path.join(self.temp.name, "model")
        canonical = os.path.join(model_dir, ".coli_usage")
        os.makedirs(model_dir)
        ramdisk._usage_write(canonical, {"0:1": 10})
        records = []
        for index, value in enumerate((12, 13), 1):
            state_dir = os.path.join(self.temp.name, "node-%d" % index)
            os.makedirs(state_dir)
            ramdisk._usage_write(os.path.join(state_dir, ".coli_usage"), {"0:1": value})
            record = {
                "state_dir": state_dir,
                "usage_baseline": {"0:1": 10},
                "usage_merge_id": ("%x" % index) * 32,
            }
            ramdisk._merge_usage(record, canonical)
            records.append(record)
        self.assertEqual(ramdisk._usage_read(canonical), {"0:1": 15})
        self.assertEqual(ramdisk._usage_merge_ids(canonical), {"1" * 32, "2" * 32})

        # Simulate both records still looking uncommitted after a manager crash.
        for index, record in enumerate(records, 1):
            ramdisk._atomic_json(
                os.path.join(record["state_dir"], ".coli_usage.delta.json"),
                {"version": 1, "id": record["usage_merge_id"], "delta": {"0:1": index + 1}},
            )
            ramdisk._recover_delta(record["state_dir"], canonical)
        self.assertEqual(ramdisk._usage_read(canonical), {"0:1": 15})

    def test_process_identity_rejects_uid_starttime_and_nonce_mismatch(self):
        record = {"pid": 44, "uid": 1000, "starttime": 99, "nonce": "expected"}
        with mock.patch.object(
            ramdisk,
            "_proc_identity",
            return_value={"pid": 44, "uid": 1000, "starttime": 99, "nonce": "other", "pgid": 44},
        ):
            matches, reason, _ = ramdisk._process_matches(record)
        self.assertFalse(matches)
        self.assertEqual(reason, "foreign-nonce")

    def test_manifest_rejects_missing_nonce_before_process_signaling(self):
        manifest = self.manifest(
            state="running", processes=[{"pid": 12345}]
        )
        manifest["processes"][0].pop("nonce")
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        with self.assertRaisesRegex(ramdisk.RamdiskError, "unsafe managed process"):
            ramdisk._load_manifest(required=True)

    def test_manifest_rejects_mount_layout_outside_v1_root(self):
        manifest = self.manifest()
        manifest["plan"]["mount_root"] = os.path.join(self.temp.name, "mount")
        manifest["plan"]["mounts"][0]["path"] = manifest["plan"]["mount_root"]
        manifest["mounts"][0]["path"] = manifest["plan"]["mount_root"]
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        with self.assertRaisesRegex(ramdisk.RamdiskError, "unsafe mount root"):
            ramdisk._load_manifest(required=True)

    def test_stop_validates_every_pid_before_signaling_any(self):
        manifest = self.manifest(
            state="running",
            mount_paths=["/mnt/colibri-test/node0", "/mnt/colibri-test/node1"],
            processes=[{"pid": 1}, {"pid": 2}],
        )
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        def match(record):
            if record["pid"] == 1:
                return True, "running", {"pgid": 1}
            return False, "foreign-uid", {"pgid": 2}

        with mock.patch.object(ramdisk, "_process_matches", side_effect=match), mock.patch.object(
            os, "killpg"
        ) as kill:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "unverified"):
                ramdisk.stop()
        kill.assert_not_called()

    def test_stop_revalidates_identity_before_escalating_to_sigkill(self):
        record = {
            "pid": 12345,
            "pgid": 12345,
            "uid": os.getuid(),
            "starttime": 91,
            "nonce": "a" * 48,
        }
        running = (
            True,
            "running",
            {"pid": 12345, "pgid": 12345},
        )
        reused = (
            False,
            "foreign-starttime",
            {"pid": 12345, "pgid": 12345},
        )
        with mock.patch.object(
            ramdisk,
            "_process_matches",
            side_effect=(running, reused),
        ), mock.patch.object(os, "killpg") as kill:
            failure = ramdisk._terminate_verified_group(
                record,
                term_seconds=0,
                kill_seconds=0,
            )

        kill.assert_called_once_with(12345, signal.SIGTERM)
        self.assertIn("identity changed before SIGKILL", failure)

    def test_verified_stop_reaps_a_locally_owned_zombie_before_escalation(self):
        record = {
            "pid": 12346,
            "pgid": 12346,
            "uid": os.getuid(),
            "starttime": 92,
            "nonce": "b" * 48,
        }
        process = mock.Mock(pid=12346)
        process.poll.side_effect = (None, 0)
        running = (
            True,
            "running",
            {"pid": 12346, "pgid": 12346},
        )
        stopped = (False, "not-running", None)
        ramdisk._track_managed_child(process)
        self.addCleanup(ramdisk._forget_managed_child, process.pid)
        with mock.patch.object(
            ramdisk,
            "_process_matches",
            side_effect=(running, stopped),
        ), mock.patch.object(os, "killpg") as kill:
            failure = ramdisk._terminate_verified_group(
                record,
                term_seconds=0,
                kill_seconds=0,
            )

        self.assertIsNone(failure)
        kill.assert_called_once_with(12346, signal.SIGTERM)
        self.assertNotIn(12346, ramdisk._managed_children)

    def test_stop_persists_error_when_usage_merge_fails(self):
        manifest = self.manifest(state="running", processes=[{"pid": 12345}])
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        with mock.patch.object(
            ramdisk, "_process_matches", return_value=(False, "not-running", None)
        ), mock.patch.object(
            ramdisk, "_merge_usage", side_effect=ramdisk.RamdiskError("disk unavailable")
        ):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "cleanup is incomplete"):
                ramdisk.stop()
        persisted = ramdisk._read_json(ramdisk._manifest_path())
        self.assertEqual(persisted["state"], "error")
        self.assertIn("disk unavailable", persisted["processes"][0]["usage_merge_error"])

    def test_stop_preserves_recoverable_error_for_incomplete_mount_layout(self):
        manifest = self.manifest(
            state="error",
            mount_paths=["/mnt/colibri-test/node0", "/mnt/colibri-test/node1"],
        )
        manifest["mounts"].pop()
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        stopped = ramdisk.stop()

        self.assertEqual(stopped["state"], "error")
        self.assertEqual(ramdisk._load_manifest(required=True)["state"], "error")

    def test_managed_readiness_requires_verified_health_response(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"status":"ok"}'

        record = {"pid": 123, "port": 8123, "log": "/tmp/engine.log"}
        with mock.patch.object(
            ramdisk, "_process_matches", return_value=(True, "running", {})
        ), mock.patch.object(ramdisk.urllib.request, "urlopen", return_value=Response()):
            ramdisk._wait_managed_ready(record, timeout=1, api_key="secret")
        self.assertIn("ready_at", record)

    def test_destroy_refuses_replaced_mount_identity(self):
        mount_path = "/mnt/colibri-test"
        manifest = self.manifest(mount_paths=[mount_path])
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        replacement = {
            "mount_id": 5,
            "device": "0:10",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        args = argparse.Namespace(yes=True)
        with mock.patch.object(ramdisk, "_mount_at", return_value=replacement), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "foreign or replaced"):
                ramdisk.destroy(args)
        unmount.assert_not_called()

    def test_destroy_retains_manifest_for_unrecorded_surviving_mount(self):
        mount_path = "/mnt/colibri-test"
        manifest = self.manifest(state="error", mount_paths=[mount_path])
        manifest["mounts"] = []
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        surviving = {
            "mount_id": 17,
            "device": "0:77",
            "filesystem": "tmpfs",
            "source": "tmpfs",
        }
        with mock.patch.object(ramdisk, "_mount_table", return_value=[]), mock.patch.object(
            ramdisk, "_mount_at", return_value=surviving
        ), mock.patch.object(ramdisk, "_umount_path") as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "unverified surviving mount"):
                ramdisk.destroy(argparse.Namespace(yes=True))
        unmount.assert_not_called()
        self.assertTrue(os.path.exists(ramdisk._manifest_path()))

    def test_destroy_preflights_every_busy_mount_before_unmounting(self):
        paths = ["/mnt/colibri-test/node0", "/mnt/colibri-test/node1"]
        manifest = self.manifest(mount_paths=paths)
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)

        def mounted(path):
            record = next(item for item in manifest["mounts"] if item["path"] == path)
            return dict(record["identity"], filesystem="tmpfs", source="tmpfs")

        with mock.patch.object(ramdisk, "_mount_at", side_effect=mounted), mock.patch.object(
            ramdisk, "_validate_mount", side_effect=lambda record, plan: mounted(record["path"])
        ), mock.patch.object(ramdisk, "_validate_namespace"), mock.patch.object(
            ramdisk, "_busy_mount_references", side_effect=[[], [999]]
        ), mock.patch.object(ramdisk, "_umount_path") as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "busy"):
                ramdisk.destroy(argparse.Namespace(yes=True))
        unmount.assert_not_called()

    def test_destroy_rejects_nested_child_mounts_before_any_unmount(self):
        paths = ["/mnt/colibri-test/node0", "/mnt/colibri-test/node1"]
        manifest = self.manifest(mount_paths=paths)
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        child = {
            "mount_id": 99,
            "path": paths[1] + "/foreign-child",
            "filesystem": "ext4",
            "source": "/dev/loop0",
        }
        with mock.patch.object(ramdisk, "_mount_table", return_value=[child]), mock.patch.object(
            ramdisk, "_umount_path"
        ) as unmount:
            with self.assertRaisesRegex(ramdisk.RamdiskError, "nested child mounts"):
                ramdisk.destroy(argparse.Namespace(yes=True))
        unmount.assert_not_called()

    def test_busy_mount_scan_includes_the_manager_process(self):
        held = os.path.join(self.temp.name, "held-mount")
        child = os.path.join(held, "inside")
        os.makedirs(child)
        previous = os.getcwd()
        try:
            os.chdir(child)
            self.assertIn(os.getpid(), ramdisk._busy_mount_references(held))
        finally:
            os.chdir(previous)

    def test_dashboard_rss_sums_verified_wrapper_and_engine_group(self):
        record = {
            "pid": 101,
            "pgid": 101,
            "uid": os.getuid(),
            "nonce": "a" * 48,
        }
        members = [
            {"pid": 101, "uid": os.getuid(), "nonce": "a" * 48},
            {"pid": 102, "uid": os.getuid(), "nonce": "a" * 48},
        ]

        def proc_text(path, default=""):
            if path == "/proc/101/status":
                return "VmRSS:\t100 kB\n"
            if path == "/proc/102/status":
                return "VmRSS:\t900 kB\n"
            return default

        with mock.patch.object(
            ramdisk, "_process_matches", return_value=(True, "running", {})
        ), mock.patch.object(
            ramdisk, "_process_group_members", return_value=(members, [])
        ), mock.patch.object(ramdisk, "_read_text", side_effect=proc_text):
            metrics = ramdisk._managed_process_metrics(record)
        self.assertEqual(metrics["rss_bytes"], 1000 * 1024)
        self.assertEqual(metrics["rss_processes"], 2)

    def test_status_absent_is_versioned(self):
        report = ramdisk.status()
        self.assertEqual(report["schema"], ramdisk.STATUS_SCHEMA)
        self.assertEqual(report["state"], "absent")

    def test_manifest_rejects_volatile_durable_state(self):
        manifest = self.manifest()
        ramdisk._atomic_json(ramdisk._manifest_path(), manifest)
        with mock.patch.object(ramdisk, "_filesystem_for_path", return_value="tmpfs"):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "volatile"):
                ramdisk._load_manifest(required=True)


class ManagedLaunchTest(unittest.TestCase):
    def test_per_node_launch_forces_durable_kv_and_node_local_core_counts(self):
        captures = []
        nonce = "a" * 48

        class FakeSocket:
            def bind(self, address):
                pass

            def close(self):
                pass

        class FakeProcess:
            next_pid = 4100

            def __init__(self):
                type(self).next_pid += 1
                self.pid = type(self).next_pid

            def poll(self):
                return None

        def popen(command, **kwargs):
            process = FakeProcess()
            captures.append(
                {
                    "command": list(command),
                    "environment": dict(kwargs["env"]),
                    "pid": process.pid,
                }
            )
            return process

        def identity(pid):
            return {
                "pid": pid,
                "pgid": pid,
                "uid": os.getuid(),
                "starttime": 1000 + pid,
                "nonce": nonce,
            }

        with ModelFixture() as fixture, tempfile.TemporaryDirectory() as state:
            hardware = hardware_fixture(nodes=2)
            set_asymmetric_node_cores(hardware)
            plan = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"), hardware=hardware
            )
            manifest = {
                "state": "ready",
                "base_port": 8100,
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(item) for item in plan["mounts"]],
                "processes": [],
                "best_runtime": {
                    "per-node": {
                        "variant": "partial_direct",
                        "knobs": {
                            "PIPE": 1,
                            "OMP_NUM_THREADS": 3,
                            "OMP_PROC_BIND": "spread",
                        },
                    }
                },
            }
            with mock.patch.dict(
                os.environ,
                {
                    "XDG_STATE_HOME": state,
                    "KVSAVE": "0",
                    "COLI_NO_OMP_TUNE": "1",
                    "COLI_OMP_TUNED": "1",
                },
            ), mock.patch.object(
                ramdisk, "_filesystem_for_path", return_value="ext4"
            ), mock.patch.object(ramdisk, "_load_manifest", return_value=manifest), mock.patch.object(
                ramdisk, "_assert_ready_mounts"
            ), mock.patch.object(ramdisk, "_save_manifest"), mock.patch.object(
                ramdisk, "_admit_concurrent_runtimes"
            ) as admit, mock.patch.object(ramdisk, "_recover_delta"), mock.patch.object(
                ramdisk, "_usage_read", return_value={}
            ), mock.patch.object(ramdisk, "_usage_write"), mock.patch.object(
                ramdisk, "_fresh_user_binary", return_value="/usr/bin/numactl"
            ), mock.patch.object(
                ramdisk.socket, "socket", side_effect=lambda *args, **kwargs: FakeSocket()
            ), mock.patch.object(ramdisk.subprocess, "Popen", side_effect=popen), mock.patch.object(
                ramdisk, "_proc_identity", side_effect=identity
            ), mock.patch.object(ramdisk, "_wait_managed_ready"), mock.patch.object(
                ramdisk.secrets, "token_hex", return_value=nonce
            ):
                result = ramdisk.start.__wrapped__(
                    argparse.Namespace(base_port=None), cli_path=sys.executable
                )
        for launch in captures:
            self.addCleanup(ramdisk._forget_managed_child, launch["pid"])

        self.assertEqual(result["state"], "running")
        admit.assert_called_once_with(plan, manifest["mounts"], benchmark=False)
        self.assertEqual(len(captures), 2)
        for index, (expected_cores, expected_cpus) in enumerate(
            ((3, "0-2"), (5, "3-7"))
        ):
            launch = captures[index]
            environment = launch["environment"]
            self.assertEqual(environment["KVSAVE"], "1")
            self.assertEqual(environment["PROF"], "1")
            self.assertEqual(environment["PIPE"], "1")
            self.assertEqual(environment["OMP_NUM_THREADS"], str(expected_cores))
            self.assertEqual(environment["OMP_PROC_BIND"], "spread")
            self.assertEqual(environment["COLI_NUMA"], "0")
            self.assertNotIn("COLI_NO_OMP_TUNE", environment)
            self.assertNotIn("COLI_OMP_TUNED", environment)
            self.assertEqual(environment["COLI_NUMA_NODES"], str(index))
            self.assertEqual(environment["COLI_CPU_AFFINITY"], expected_cpus)
            self.assertEqual(
                launch["command"][:3],
                [
                    "/usr/bin/numactl",
                    "--physcpubind=%s" % expected_cpus,
                    "--membind=%d" % index,
                ],
            )
            self.assertTrue(environment["COLI_STATE_DIR"].endswith("node-%d" % index))
        self.assertEqual([record["port"] for record in result["processes"]], [8100, 8101])
        self.assertEqual(result["base_port"], 8100)

    def test_clean_start_cancellation_restores_retryable_manifest(self):
        cancel = threading.Event()
        nonce = "c" * 48

        class FakeSocket:
            def bind(self, address):
                pass

            def close(self):
                pass

        class FakeProcess:
            pid = 6100

            def poll(self):
                return None

            def wait(self, timeout=None):
                return 0

        def cancel_ready(*args, **kwargs):
            cancel.set()
            ramdisk._raise_if_cancelled(cancel)

        identity = {
            "pid": 6100,
            "pgid": 6100,
            "uid": os.getuid(),
            "starttime": 16100,
            "nonce": nonce,
        }
        with ModelFixture() as fixture, tempfile.TemporaryDirectory() as state:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
            manifest = {
                "state": "ready",
                "base_port": 9000,
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(plan["mounts"][0])],
                "processes": [],
                "ports": [],
            }
            with mock.patch.dict(
                os.environ, {"XDG_STATE_HOME": state}
            ), mock.patch.object(
                ramdisk, "_filesystem_for_path", return_value="ext4"
            ), mock.patch.object(
                ramdisk, "_load_manifest", return_value=manifest
            ), mock.patch.object(ramdisk, "_assert_ready_mounts"), mock.patch.object(
                ramdisk, "_save_manifest"
            ), mock.patch.object(ramdisk, "_admit_concurrent_runtimes"), mock.patch.object(
                ramdisk, "_recover_delta"
            ), mock.patch.object(ramdisk, "_usage_read", return_value={}), mock.patch.object(
                ramdisk, "_usage_write"
            ), mock.patch.object(
                ramdisk.socket, "socket", side_effect=lambda *args, **kwargs: FakeSocket()
            ), mock.patch.object(
                ramdisk.subprocess, "Popen", return_value=FakeProcess()
            ), mock.patch.object(
                ramdisk, "_proc_identity", return_value=identity
            ), mock.patch.object(
                ramdisk, "_wait_managed_ready", side_effect=cancel_ready
            ), mock.patch.object(
                ramdisk, "_terminate_verified_group", return_value=None
            ), mock.patch.object(
                ramdisk, "_group_alive", return_value=False
            ), mock.patch.object(ramdisk, "_merge_usage"), mock.patch.object(
                ramdisk.secrets, "token_hex", return_value=nonce
            ):
                with self.assertRaises(ramdisk._OperationCancelled):
                    ramdisk.start.__wrapped__(
                        argparse.Namespace(base_port=None),
                        cli_path=sys.executable,
                        cancel_event=cancel,
                    )

        self.assertEqual(manifest["state"], "ready")
        self.assertEqual(manifest["base_port"], 9000)
        self.assertEqual(manifest["processes"], [])
        self.assertEqual(manifest["ports"], [])
        self.assertNotIn("launch_error", manifest)

    def test_launch_rollback_merges_every_context_when_manifest_saves_fail(self):
        nonce = "b" * 48

        class FakeSocket:
            def bind(self, address):
                pass

            def close(self):
                pass

        class FakeProcess:
            pid = 5100

            def poll(self):
                return None

            def wait(self, timeout=None):
                return 0

        with ModelFixture() as fixture, tempfile.TemporaryDirectory() as state:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture())
            manifest = {
                "state": "ready",
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(plan["mounts"][0])],
                "processes": [],
            }

            def save(current):
                if current.get("state") == "error" or any(
                    record.get("usage_merge_id") for record in current.get("processes", [])
                ):
                    raise OSError("state filesystem full")

            identity = {
                "pid": 5100,
                "pgid": 5100,
                "uid": os.getuid(),
                "starttime": 15100,
                "nonce": nonce,
            }
            with mock.patch.dict(os.environ, {"XDG_STATE_HOME": state}), mock.patch.object(
                ramdisk, "_filesystem_for_path", return_value="ext4"
            ), mock.patch.object(
                ramdisk, "_load_manifest", return_value=manifest
            ), mock.patch.object(ramdisk, "_assert_ready_mounts"), mock.patch.object(
                ramdisk, "_save_manifest", side_effect=save
            ), mock.patch.object(ramdisk, "_admit_concurrent_runtimes"), mock.patch.object(
                ramdisk, "_recover_delta"
            ), mock.patch.object(ramdisk, "_usage_read", return_value={}), mock.patch.object(
                ramdisk, "_usage_write"
            ), mock.patch.object(
                ramdisk.socket, "socket", side_effect=lambda *args, **kwargs: FakeSocket()
            ), mock.patch.object(
                ramdisk.subprocess, "Popen", return_value=FakeProcess()
            ), mock.patch.object(
                ramdisk, "_proc_identity", return_value=identity
            ), mock.patch.object(
                ramdisk, "_wait_managed_ready", side_effect=ramdisk.RamdiskError("not ready")
            ), mock.patch.object(
                ramdisk, "_terminate_verified_group", return_value=None
            ), mock.patch.object(ramdisk, "_group_alive", return_value=False), mock.patch.object(
                ramdisk, "_merge_usage"
            ) as merge, mock.patch.object(ramdisk.secrets, "token_hex", return_value=nonce):
                with self.assertRaisesRegex(ramdisk.RamdiskError, "rollback/reporting errors"):
                    ramdisk.start.__wrapped__(
                        argparse.Namespace(base_port=8200), cli_path=sys.executable
                    )

        merge.assert_called_once()
        self.assertTrue(merge.call_args.kwargs["keep_journal"])


class BenchmarkTest(unittest.TestCase):
    def test_cancellable_engine_startup_terminates_before_ready_timeout(self):
        cancel = threading.Event()
        reader_started = threading.Event()
        release_reader = threading.Event()

        class FakeBaseEngine:
            @staticmethod
            def _terminate_process(process):
                process.terminated = True
                release_reader.set()

        class FakeProcess:
            stdout = object()
            terminated = False

        def read_engine_turn(stream, marker, callback):
            reader_started.set()
            release_reader.wait(timeout=2)

        engine_type = ramdisk._cancellable_engine_type(
            FakeBaseEngine,
            read_engine_turn,
            b"READY",
            cancel,
        )

        def request_cancel():
            reader_started.wait(timeout=2)
            cancel.set()

        canceller = threading.Thread(target=request_cancel)
        canceller.start()
        process = FakeProcess()
        with self.assertRaises(ramdisk._OperationCancelled):
            engine_type._wait_until_ready(process, timeout=7200)
        canceller.join(timeout=2)

        self.assertTrue(process.terminated)

    def test_benchmark_cancel_wakes_generation_before_first_token(self):
        cancel = threading.Event()
        entered_generate = threading.Event()
        release_generate = threading.Event()

        class FakeCancelled(Exception):
            pass

        class FakeEngine:
            closed = False

            def generate(self, *args, **kwargs):
                entered_generate.set()
                release_generate.wait(timeout=2)
                raise RuntimeError("engine is shutting down")

            def close(self):
                self.closed = True
                release_generate.set()

        def request_cancel():
            entered_generate.wait(timeout=2)
            cancel.set()

        engine = FakeEngine()
        canceller = threading.Thread(target=request_cancel)
        canceller.start()
        with self.assertRaises(ramdisk._OperationCancelled):
            ramdisk._benchmark_generate(
                engine,
                "prompt",
                lambda text: None,
                cancel,
                FakeCancelled,
            )
        canceller.join(timeout=2)

        self.assertTrue(engine.closed)

    def test_benchmark_cancel_does_not_hide_engine_close_failure(self):
        cancel = threading.Event()
        entered_generate = threading.Event()

        class FakeCancelled(Exception):
            pass

        class FakeEngine:
            def generate(self, *args, **kwargs):
                entered_generate.set()
                cancel.wait(timeout=2)
                raise FakeCancelled()

            def close(self):
                raise OSError("engine survived termination")

        def request_cancel():
            entered_generate.wait(timeout=2)
            cancel.set()

        canceller = threading.Thread(target=request_cancel)
        canceller.start()
        with self.assertRaises(ramdisk.RamdiskError) as raised:
            ramdisk._benchmark_generate(
                FakeEngine(),
                "prompt",
                lambda text: None,
                cancel,
                FakeCancelled,
            )
        canceller.join(timeout=2)

        self.assertNotIsInstance(raised.exception, ramdisk._OperationCancelled)
        self.assertIn("survived termination", str(raised.exception))

    def test_engine_resolution_prefers_current_binary_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = root / "coli"
            cli.write_text("", encoding="utf-8")
            suffix = ".exe" if os.name == "nt" else ""
            current = root / ("colibri" + suffix)
            legacy = root / ("glm" + suffix)
            for engine in (current, legacy):
                engine.write_text("", encoding="utf-8")
                engine.chmod(0o755)

            resolved = ramdisk._resolve_engine_path(str(cli))

        self.assertEqual(resolved, str(current.resolve()))

    def test_system_score_uses_concurrent_aggregate_rss_and_mount_shmem(self):
        filesystem = mock.Mock(f_blocks=10, f_bfree=6, f_frsize=4096)
        manifest = {
            "mounts": [
                {"path": "/mnt/colibri-test/node0", "node": 0, "numa_allocation": {"0": 4}},
                {"path": "/mnt/colibri-test/node1", "node": 1, "numa_allocation": {"1": 4}},
            ]
        }
        variants = [
            {
                "status": "ok",
                "runs": [{"rss_bytes": 2, "prefault_seconds": 0.5}],
            }
        ]
        aggregate = {
            "status": "ok",
            "rounds": [
                {
                    "rows": [
                        {"rss_bytes": 3, "prefault_seconds": 1.0},
                        {"rss_bytes": 5, "prefault_seconds": 2.0},
                    ]
                }
            ],
        }
        with mock.patch.object(
            ramdisk, "_meminfo", return_value={"Shmem": 100, "ShmemPmdMapped": 25}
        ), mock.patch.object(ramdisk.os, "statvfs", return_value=filesystem):
            result = ramdisk._system_score(
                manifest, variants, 10, 10, aggregate=aggregate
            )
        self.assertEqual(result["rss_bytes"], 8)
        self.assertEqual(result["aggregate_rss_bytes"], 8)
        self.assertEqual(result["per_process_peak_rss_bytes"], 5)
        self.assertEqual(result["prefault_seconds"], 2.0)
        self.assertEqual(result["shmem_bytes"], 2 * 4 * 4096)
        self.assertEqual(result["huge_page_coverage_scope"].split()[0], "host-global")

    def test_variant_uses_one_persistent_engine_and_measured_rammap_coverage(self):
        import openai_server

        with ModelFixture() as fixture, tempfile.TemporaryDirectory() as state:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
            manifest = {
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(plan["mounts"][0])],
            }
            expected_experts = plan["staging"]["direct_mapped_expert_count"]
            expected_bytes = plan["staging"]["direct_mapped_bytes"]

            class FakeEngine:
                instances = 0
                environments = []

                def __init__(self, *args, **kwargs):
                    type(self).instances += 1
                    type(self).environments.append(dict(kwargs["env"]))
                    self.profile_seq = 0
                    self.profile = []

                def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0):
                    on_text("identical output")
                    self.profile_seq += 1
                    self.profile.append(
                        {
                            "forward_p50_ms": 10.0,
                            "forward_p99_ms": 20.0,
                            "physical_ssd_bytes": 0,
                            "physical_ssd_valid": True,
                            "rammap_experts": expected_experts,
                            "rammap_bytes": expected_bytes,
                            "ttft_ms": 30.0,
                            "prefault_seconds": 0.5,
                        }
                    )
                    return {"completion_tokens": 32, "tokens_per_second": 4.0, "rss_gb": 2.0}

                def close(self):
                    pass

            with mock.patch.dict(
                os.environ,
                {
                    "XDG_STATE_HOME": state,
                    "TEMP": "1",
                    "DRAFT": "1",
                    "KVSAVE": "1",
                    "AUTOPIN": "1",
                    "PROF": "0",
                    "COLI_NO_OMP_TUNE": "1",
                    "COLI_OMP_TUNED": "1",
                },
            ), mock.patch.object(
                openai_server, "Engine", FakeEngine
            ), mock.patch.object(
                ramdisk, "_filesystem_for_path", return_value="ext4"
            ), mock.patch.object(ramdisk, "_admit_runtime"):
                result = ramdisk._score_variant(
                    "/fake/glm", manifest, "full_direct", plan["mounts"][0]["path"], True, {"PIPE": 0}
                )
        self.assertEqual(FakeEngine.instances, 1)
        self.assertTrue(result["persistent_engine"])
        self.assertEqual(result["interactive"]["ram_map_coverage"], 1.0)
        self.assertEqual(len(result["runs"]), 3)
        benchmark_environment = FakeEngine.environments[0]
        self.assertEqual(
            {key: benchmark_environment[key] for key in ("TEMP", "DRAFT", "KVSAVE", "AUTOPIN", "PROF")},
            {"TEMP": "0", "DRAFT": "0", "KVSAVE": "0", "AUTOPIN": "0", "PROF": "1"},
        )
        self.assertNotIn("COLI_NO_OMP_TUNE", benchmark_environment)
        self.assertNotIn("COLI_OMP_TUNED", benchmark_environment)

    def test_aggregate_launches_fixed_node_local_engines_and_runs_concurrently(self):
        import openai_server

        barrier = threading.Barrier(2)

        class FakeEngine:
            instances = []

            def __init__(self, *args, **kwargs):
                self.environment = dict(kwargs["env"])
                self.command_prefix = list(kwargs["command_prefix"])
                self.node = int(self.command_prefix[2].split("=", 1)[1])
                self.maximum = kwargs["max_tokens"]
                self.kv_slots = kwargs["kv_slots"]
                self.profile_seq = 0
                self.profile = []
                self.calls = 0
                self.closed = False
                type(self).instances.append(self)

            def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0):
                barrier.wait(timeout=3)
                self.calls += 1
                on_text("identical output")
                self.profile_seq += 1
                self.profile.append(
                    {
                        "physical_ssd_bytes": 0,
                        "physical_ssd_valid": True,
                        "rammap_experts": expected_experts,
                        "rammap_bytes": expected_bytes,
                    }
                )
                return {"completion_tokens": 32}

            def close(self):
                self.closed = True

        with ModelFixture() as fixture, tempfile.TemporaryDirectory() as state:
            hardware = hardware_fixture(nodes=2)
            set_asymmetric_node_cores(hardware)
            plan = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"), hardware=hardware
            )
            manifest = {
                "state": "ready",
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(item) for item in plan["mounts"]],
                "processes": [],
            }
            expected_experts = plan["staging"]["direct_mapped_expert_count"]
            expected_bytes = plan["staging"]["direct_mapped_bytes"]
            with mock.patch.dict(
                os.environ,
                {
                    "XDG_STATE_HOME": state,
                    "TEMP": "1",
                    "DRAFT": "1",
                    "KVSAVE": "1",
                    "AUTOPIN": "1",
                    "PROF": "0",
                    "COLI_NO_OMP_TUNE": "1",
                    "COLI_OMP_TUNED": "1",
                },
            ), mock.patch.object(openai_server, "Engine", FakeEngine), mock.patch.object(
                ramdisk, "_filesystem_for_path", return_value="ext4"
            ), mock.patch.object(
                ramdisk, "_fresh_user_binary", return_value="/usr/bin/numactl"
            ), mock.patch.object(
                ramdisk, "_admit_concurrent_runtimes"
            ) as admit:
                result = ramdisk._aggregate_score(
                    manifest, engine_path="/fake/glm", knobs={"PIPE": 0}
                )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["warmups"], 1)
        self.assertEqual(result["measured_rounds"], 3)
        self.assertTrue(result["persistent_engines"])
        self.assertEqual(result["runtime_knobs"], {"PIPE": 0})
        self.assertEqual(
            result["fixed_environment"],
            {"TEMP": 0, "DRAFT": 0, "KVSAVE": 0, "AUTOPIN": 0, "PROF": 1},
        )
        admit.assert_called_once_with(plan, manifest["mounts"], benchmark=False)
        self.assertEqual(len(FakeEngine.instances), 2)
        for engine, (expected_cores, expected_cpus) in zip(
            sorted(FakeEngine.instances, key=lambda item: item.node),
            ((3, "0-2"), (5, "3-7")),
        ):
            self.assertEqual(engine.calls, 4)
            self.assertTrue(engine.closed)
            self.assertEqual(engine.maximum, 32)
            self.assertEqual(engine.kv_slots, 1)
            self.assertEqual(engine.environment["OMP_NUM_THREADS"], str(expected_cores))
            self.assertEqual(engine.environment["COLI_NUMA"], "0")
            self.assertNotIn("COLI_NO_OMP_TUNE", engine.environment)
            self.assertNotIn("COLI_OMP_TUNED", engine.environment)
            self.assertEqual(engine.environment["COLI_NUMA_NODES"], str(engine.node))
            self.assertEqual(engine.environment["COLI_CPU_AFFINITY"], expected_cpus)
            self.assertEqual(
                {key: engine.environment[key] for key in ("TEMP", "DRAFT", "KVSAVE", "AUTOPIN", "PROF")},
                {"TEMP": "0", "DRAFT": "0", "KVSAVE": "0", "AUTOPIN": "0", "PROF": "1"},
            )
            self.assertEqual(
                engine.command_prefix,
                [
                    "/usr/bin/numactl",
                    "--physcpubind=%s" % expected_cpus,
                    "--membind=%d" % engine.node,
                ],
            )

    def test_concurrent_runtime_admission_reserves_shared_cgroup_headroom(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"),
                hardware=hardware_fixture(nodes=2),
            )
        runtime = int(plan["reserve"]["managed_runtime_bytes"])
        page_tables = int(plan["reserve"]["page_table_bytes"])
        per_node_required = [
            runtime + page_tables + int(node["reserve_bytes"])
            for node in plan["hardware"]["nodes"]
        ]
        shared_headroom = max(per_node_required)

        with mock.patch.object(
            ramdisk,
            "_node_meminfo",
            return_value={"MemFree": sum(per_node_required) * 2},
        ), mock.patch.object(
            ramdisk,
            "_cgroup_available_memory",
            return_value=shared_headroom,
        ) as cgroup_available:
            with self.assertRaisesRegex(
                ramdisk.RamdiskError,
                "cgroup.*aggregate runtime/OS floor",
            ):
                ramdisk._admit_concurrent_runtimes(
                    plan,
                    plan["mounts"],
                    benchmark=False,
                )

        cgroup_available.assert_called_once_with()

    def test_full_per_node_benchmark_sizes_thread_sweep_to_target_node(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture(nodes=2)
            set_asymmetric_node_cores(hardware)
            plan = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"), hardware=hardware
            )
            manifest = {
                "state": "ready",
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(item) for item in plan["mounts"]],
                "processes": [],
            }
            seen_knobs = {}

            def score(engine_path, current_manifest, name, weights, rammap, knobs):
                seen_knobs[name] = dict(knobs)
                return {
                    "name": name,
                    "status": "ok",
                    "knobs": dict(knobs),
                    "runs": [],
                    "output_sha256": "same-output",
                    "interactive": {"p50_tokens_per_second": 1.0},
                }

            with mock.patch.object(ramdisk, "_load_manifest", return_value=manifest), mock.patch.object(
                ramdisk, "_assert_ready_mounts"
            ), mock.patch.object(ramdisk, "_resolve_engine_path", return_value="/fake/glm"), mock.patch.object(
                ramdisk, "_score_variant", side_effect=score
            ), mock.patch.object(
                ramdisk, "discover_hardware", return_value={"swap": {"used_bytes": 0}}
            ), mock.patch.object(
                ramdisk, "_aggregate_score", return_value={"status": "not-run"}
            ) as aggregate, mock.patch.object(ramdisk, "_system_score", return_value={}), mock.patch.object(
                ramdisk, "_read_json", return_value={"version": 1, "results": []}
            ), mock.patch.object(ramdisk, "_atomic_json"), mock.patch.object(ramdisk, "_save_manifest"):
                ramdisk.benchmark.__wrapped__(argparse.Namespace(), cli_path="/fake/coli")

        self.assertEqual(seen_knobs["full_direct_half_threads"]["OMP_NUM_THREADS"], 1)
        self.assertEqual(seen_knobs["full_direct_pipe0"]["OMP_NUM_THREADS"], 3)
        self.assertEqual(seen_knobs["full_direct_pipe1"]["OMP_NUM_THREADS"], 3)
        self.assertNotIn("OMP_NUM_THREADS", aggregate.call_args.kwargs["knobs"])

    def test_best_runtime_knobs_are_saved_only_for_current_topology(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture())
            other_topology = {"variant": "node-local-best", "knobs": {"PIPE": 0}}
            manifest = {
                "state": "ready",
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(plan["mounts"][0])],
                "processes": [],
                "best_runtime": {"per-node": dict(other_topology)},
            }
            rates = {
                "ssd_baseline": 1.0,
                "tmpfs_pread_slabs": 2.0,
                "full_direct_half_threads": 5.0,
                "full_direct_pipe0": 7.0,
                "full_direct_pipe1": 9.0,
            }

            def score(engine_path, current_manifest, name, weights, rammap, knobs):
                return {
                    "name": name,
                    "status": "ok",
                    "knobs": dict(knobs),
                    "runs": [],
                    "output_sha256": "same-output",
                    "interactive": {"p50_tokens_per_second": rates[name]},
                }

            with mock.patch.object(ramdisk, "_load_manifest", return_value=manifest), mock.patch.object(
                ramdisk, "_assert_ready_mounts"
            ), mock.patch.object(ramdisk, "_resolve_engine_path", return_value="/fake/glm"), mock.patch.object(
                ramdisk, "_score_variant", side_effect=score
            ), mock.patch.object(
                ramdisk, "discover_hardware", return_value={"swap": {"used_bytes": 0}}
            ), mock.patch.object(
                ramdisk, "_aggregate_score", return_value={"status": "not-run"}
            ), mock.patch.object(ramdisk, "_system_score", return_value={}), mock.patch.object(
                ramdisk, "_read_json", return_value={"version": 1, "results": []}
            ), mock.patch.object(ramdisk, "_atomic_json"), mock.patch.object(ramdisk, "_save_manifest"):
                result = ramdisk.benchmark.__wrapped__(argparse.Namespace(), cli_path="/fake/coli")

        self.assertEqual(result["best_variant"], "full_direct_pipe1")
        self.assertEqual(result["best_runtime_knobs"], manifest["best_runtime"]["interleaved"]["knobs"])
        self.assertEqual(manifest["best_runtime"]["per-node"], other_topology)

    def test_acceptance_is_false_when_applicable_paths_fail(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture())
            manifest = {
                "state": "ready",
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(plan["mounts"][0])],
                "processes": [],
            }

            def score(engine_path, current_manifest, name, weights, rammap, knobs):
                if name != "ssd_baseline":
                    raise ramdisk.RamdiskError("synthetic path failure")
                return {
                    "name": name,
                    "status": "ok",
                    "knobs": dict(knobs),
                    "runs": [],
                    "output_sha256": "baseline-only",
                    "interactive": {"p50_tokens_per_second": 1.0},
                }

            with mock.patch.object(
                ramdisk, "_load_manifest", return_value=manifest
            ), mock.patch.object(ramdisk, "_assert_ready_mounts"), mock.patch.object(
                ramdisk, "_resolve_engine_path", return_value="/fake/glm"
            ), mock.patch.object(
                ramdisk, "_score_variant", side_effect=score
            ), mock.patch.object(
                ramdisk, "discover_hardware", return_value={"swap": {"used_bytes": 0}}
            ), mock.patch.object(
                ramdisk, "_aggregate_score", return_value={"status": "not-run"}
            ), mock.patch.object(
                ramdisk, "_system_score", return_value={}
            ), mock.patch.object(
                ramdisk, "_read_json", return_value={"version": 1, "results": []}
            ), mock.patch.object(ramdisk, "_atomic_json"), mock.patch.object(
                ramdisk, "_save_manifest"
            ):
                result = ramdisk.benchmark.__wrapped__(
                    argparse.Namespace(), cli_path="/fake/coli"
                )

        self.assertFalse(result["acceptance"]["all_required_paths_succeeded"])
        self.assertFalse(result["acceptance"]["greedy_outputs_identical"])

    def test_variant_cleanup_failure_aborts_before_another_engine_launch(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(
                plan_args(fixture.root), hardware=hardware_fixture()
            )
            manifest = {
                "state": "ready",
                "model_fingerprint": plan["model"]["fingerprint"],
                "plan": plan,
                "mounts": [dict(plan["mounts"][0])],
                "processes": [],
            }
            with mock.patch.object(
                ramdisk, "_load_manifest", return_value=manifest
            ), mock.patch.object(ramdisk, "_assert_ready_mounts"), mock.patch.object(
                ramdisk, "_resolve_engine_path", return_value="/fake/glm"
            ), mock.patch.object(
                ramdisk,
                "_score_variant",
                side_effect=ramdisk._EngineCleanupError("engine survived"),
            ) as score, mock.patch.object(
                ramdisk, "discover_hardware", return_value={"swap": {"used_bytes": 0}}
            ), mock.patch.object(ramdisk, "_aggregate_score") as aggregate:
                with self.assertRaisesRegex(
                    ramdisk._EngineCleanupError, "engine survived"
                ):
                    ramdisk.benchmark.__wrapped__(
                        argparse.Namespace(), cli_path="/fake/coli"
                    )

        score.assert_called_once()
        aggregate.assert_not_called()


class CliJsonSmokeTest(unittest.TestCase):
    @unittest.skipUnless(hasattr(signal, "SIGTERM"), "SIGTERM is unavailable")
    def test_curses_sigterm_uses_cleanup_exception_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(ramdisk._TuiTerminationSignal) as raised:
            with ramdisk._curses_termination_guard():
                handler = signal.getsignal(signal.SIGTERM)
                self.assertTrue(callable(handler))
                handler(signal.SIGTERM, None)

        self.assertEqual(raised.exception.signum, int(signal.SIGTERM))
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    @unittest.skipUnless(hasattr(signal, "SIGTERM"), "SIGTERM is unavailable")
    def test_curses_repeated_sigterm_is_deferred_until_cleanup_guard_exits(self):
        previous = signal.getsignal(signal.SIGTERM)
        with ramdisk._curses_termination_guard():
            handler = signal.getsignal(signal.SIGTERM)
            with self.assertRaises(ramdisk._TuiTerminationSignal):
                handler(signal.SIGTERM, None)
            handler(signal.SIGTERM, None)

        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    @unittest.skipUnless(hasattr(signal, "SIGTERM"), "SIGTERM is unavailable")
    def test_cli_sigterm_requests_cooperative_prepare_rollback(self):
        args = argparse.Namespace(ramdisk_action="prepare", json=False)
        previous = signal.getsignal(signal.SIGTERM)

        def interrupted_prepare(_args, cancel_event=None):
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            handler(signal.SIGTERM, None)
            self.assertTrue(cancel_event.is_set())
            raise ramdisk._OperationCancelled("termination requested")

        with mock.patch.object(
            ramdisk,
            "prepare",
            side_effect=interrupted_prepare,
        ), mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(
                ramdisk.dispatch(args),
                128 + int(signal.SIGTERM),
            )

        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    @unittest.skipUnless(hasattr(signal, "SIGTERM"), "SIGTERM is unavailable")
    def test_cli_sigterm_defers_until_stop_transaction_finishes(self):
        args = argparse.Namespace(ramdisk_action="stop", json=False)
        completed = []

        def interrupted_stop(_args):
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            handler(signal.SIGTERM, None)
            completed.append(True)
            return {"state": "stopped"}

        with mock.patch.object(ramdisk, "stop", side_effect=interrupted_stop):
            self.assertEqual(
                ramdisk.dispatch(args),
                128 + int(signal.SIGTERM),
            )

        self.assertEqual(completed, [True])

    def test_stop_dispatch_surfaces_an_incomplete_recovery_workspace(self):
        args = argparse.Namespace(ramdisk_action="stop", json=False)
        with mock.patch.object(
            ramdisk, "stop", return_value={"state": "error"}
        ), mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(ramdisk.dispatch(args), 2)

        self.assertIn("workspace is incomplete", stderr.getvalue())
        self.assertIn("ramdisk status", stderr.getvalue())

    def test_benchmark_dispatch_preserves_versioned_json_schema(self):
        payload = {
            "schema": ramdisk.BENCHMARK_SCHEMA,
            "version": ramdisk.MANIFEST_VERSION,
            "variants": [],
        }
        args = argparse.Namespace(ramdisk_action="benchmark", json=True)
        with mock.patch.object(ramdisk, "benchmark", return_value=payload), mock.patch.object(
            ramdisk, "_json_print"
        ) as emit:
            self.assertEqual(ramdisk.dispatch(args), 0)
        emit.assert_called_once_with(payload)

    def test_plan_json_is_parseable_even_when_host_has_blockers(self):
        with ModelFixture() as fixture:
            result = subprocess.run(
                [sys.executable, str(C_DIR / "coli"), "ramdisk", "plan", "--model", str(fixture.root), "--json"],
                cwd=C_DIR,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["schema"], ramdisk.PLAN_SCHEMA)
        self.assertIn(result.returncode, (0, 2))
        self.assertEqual(result.stderr, "")

    def test_invalid_plan_and_absent_status_keep_json_contract(self):
        with ModelFixture() as fixture, tempfile.TemporaryDirectory() as state:
            environment = dict(
                os.environ,
                XDG_STATE_HOME=state,
                COLI_RAMDISK_MANIFEST=os.path.join(state, "manifest.json"),
            )
            invalid = subprocess.run(
                [
                    sys.executable,
                    str(C_DIR / "coli"),
                    "ramdisk",
                    "plan",
                    "--model",
                    str(fixture.root),
                    "--capacity-gb",
                    "nan",
                    "--json",
                ],
                cwd=C_DIR,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
            status = subprocess.run(
                [sys.executable, str(C_DIR / "coli"), "ramdisk", "status", "--json"],
                cwd=C_DIR,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )
        self.assertEqual(invalid.returncode, 2)
        self.assertEqual(json.loads(invalid.stdout)["schema"], "colibri.ramdisk.error.v1")
        self.assertEqual(invalid.stderr, "")
        self.assertEqual(status.returncode, 0)
        self.assertEqual(json.loads(status.stdout)["schema"], ramdisk.STATUS_SCHEMA)


if __name__ == "__main__":
    unittest.main()
