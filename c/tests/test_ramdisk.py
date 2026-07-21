import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))
import ramdisk


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
    return {
        "linux": True,
        "kernel_release": "6.8.0-test",
        "online_nodes": list(range(nodes)),
        "nodes": node_rows,
        "physical_cores": nodes * 2,
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

    def test_numa_sampling_stride_visits_every_round_robin_residue(self):
        for nodes in (2, 4):
            indices = ramdisk._sample_page_indices(4096, 256, nodes)
            self.assertEqual({index % nodes for index in indices}, set(range(nodes)))


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

        with mock.patch.object(ramdisk, "_run", side_effect=run), mock.patch.object(
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

        with mock.patch.object(ramdisk, "_run", side_effect=run), mock.patch.object(
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
        with mock.patch.object(ramdisk, "_run", return_value=result) as run, mock.patch.object(
            ramdisk, "_privileged", side_effect=lambda command, hardware: command
        ):
            with self.assertRaises(ramdisk.RamdiskError):
                ramdisk._mount_tmpfs(plan, plan["mounts"][0])
        self.assertEqual(run.call_count, 1)

    def test_prepare_immediately_rolls_back_identityless_successful_mount(self):
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture())
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
        with ModelFixture() as fixture:
            plan = ramdisk.build_plan(plan_args(fixture.root), hardware=hardware_fixture())
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
            "super_options": ["mode=700", "noswap", "huge=within_size", "mpol=interleave:0-1"],
        }
        with mock.patch.object(ramdisk, "_mount_at", return_value=actual):
            self.assertEqual(ramdisk._validate_mount(mount, plan)["mount_id"], 9)
        actual["super_options"].remove("mpol=interleave:0-1")
        with mock.patch.object(ramdisk, "_mount_at", return_value=actual):
            with self.assertRaisesRegex(ramdisk.RamdiskError, "NUMA policy"):
                ramdisk._validate_mount(mount, plan)

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
            hardware["nodes"][0]["physical_cores"] = 3
            hardware["nodes"][1]["physical_cores"] = 5
            hardware["physical_cores"] = 8
            plan = ramdisk.build_plan(
                plan_args(fixture.root, topology="per-node"), hardware=hardware
            )
            manifest = {
                "state": "ready",
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
                os.environ, {"XDG_STATE_HOME": state, "KVSAVE": "0"}
            ), mock.patch.object(
                ramdisk, "_filesystem_for_path", return_value="ext4"
            ), mock.patch.object(ramdisk, "_load_manifest", return_value=manifest), mock.patch.object(
                ramdisk, "_assert_ready_mounts"
            ), mock.patch.object(ramdisk, "_save_manifest"), mock.patch.object(
                ramdisk, "_admit_runtime"
            ), mock.patch.object(ramdisk, "_recover_delta"), mock.patch.object(
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
                    argparse.Namespace(base_port=8100), cli_path=sys.executable
                )

        self.assertEqual(result["state"], "running")
        self.assertEqual(len(captures), 2)
        for index, expected_cores in enumerate((3, 5)):
            launch = captures[index]
            environment = launch["environment"]
            self.assertEqual(environment["KVSAVE"], "1")
            self.assertEqual(environment["PROF"], "1")
            self.assertEqual(environment["PIPE"], "1")
            self.assertEqual(environment["OMP_NUM_THREADS"], str(expected_cores))
            self.assertEqual(environment["OMP_PROC_BIND"], "spread")
            self.assertEqual(environment["COLI_NUMA"], "0")
            self.assertEqual(
                launch["command"][:3],
                ["/usr/bin/numactl", "--cpunodebind=%d" % index, "--membind=%d" % index],
            )
            self.assertTrue(environment["COLI_STATE_DIR"].endswith("node-%d" % index))
        self.assertEqual([record["port"] for record in result["processes"]], [8100, 8101])

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
            ), mock.patch.object(ramdisk, "_admit_runtime"), mock.patch.object(
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
                ramdisk, "_terminate_group", return_value=None
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

    def test_aggregate_launches_fixed_node_local_engines_and_runs_concurrently(self):
        import openai_server

        barrier = threading.Barrier(2)

        class FakeEngine:
            instances = []

            def __init__(self, *args, **kwargs):
                self.environment = dict(kwargs["env"])
                self.command_prefix = list(kwargs["command_prefix"])
                self.node = int(self.command_prefix[1].split("=", 1)[1])
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
            hardware["nodes"][0]["physical_cores"] = 3
            hardware["nodes"][1]["physical_cores"] = 5
            hardware["physical_cores"] = 8
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
                },
            ), mock.patch.object(openai_server, "Engine", FakeEngine), mock.patch.object(
                ramdisk, "_filesystem_for_path", return_value="ext4"
            ), mock.patch.object(
                ramdisk, "_fresh_user_binary", return_value="/usr/bin/numactl"
            ), mock.patch.object(ramdisk, "_admit_runtime") as admit:
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
        self.assertEqual(admit.call_count, 2)
        self.assertEqual(len(FakeEngine.instances), 2)
        for engine, expected_cores in zip(
            sorted(FakeEngine.instances, key=lambda item: item.node), (3, 5)
        ):
            self.assertEqual(engine.calls, 4)
            self.assertTrue(engine.closed)
            self.assertEqual(engine.maximum, 32)
            self.assertEqual(engine.kv_slots, 1)
            self.assertEqual(engine.environment["OMP_NUM_THREADS"], str(expected_cores))
            self.assertEqual(engine.environment["COLI_NUMA"], "0")
            self.assertEqual(
                {key: engine.environment[key] for key in ("TEMP", "DRAFT", "KVSAVE", "AUTOPIN", "PROF")},
                {"TEMP": "0", "DRAFT": "0", "KVSAVE": "0", "AUTOPIN": "0", "PROF": "1"},
            )
            self.assertEqual(
                engine.command_prefix,
                [
                    "/usr/bin/numactl",
                    "--cpunodebind=%d" % engine.node,
                    "--membind=%d" % engine.node,
                ],
            )

    def test_full_per_node_benchmark_sizes_thread_sweep_to_target_node(self):
        with ModelFixture() as fixture:
            hardware = hardware_fixture(nodes=2)
            hardware["nodes"][0]["physical_cores"] = 3
            hardware["nodes"][1]["physical_cores"] = 5
            hardware["physical_cores"] = 8
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


class CliJsonSmokeTest(unittest.TestCase):
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
