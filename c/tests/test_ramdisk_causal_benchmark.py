"""Causal RAMMAP experiment protocol, evidence, and claim contracts."""

import argparse
import contextlib
import copy
import io
import json
import math
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

from ramdisk_support import benchmark  # noqa: E402


GIB = 1 << 30
TREATMENT_IDS = (
    "anon-pin-interleaved",
    "anon-pin-local",
    "tmpfs-rammap-interleaved",
    "tmpfs-rammap-local",
    "ssd-slab-control",
    "tmpfs-slab-control",
    "cuda-fixed-budget-validation",
)
FIXTURE_COLLECTOR_ID = "b" * 64


def sample_manifest(profile_path="/profiles/frozen.coli_usage"):
    return {
        "state": "ready",
        "deployment_id": "deployment-test",
        "model_fingerprint": "sha256:model-fingerprint",
        "plan": {
            "mode": "full",
            "topology": "interleaved",
            "hardware": {
                "nodes": [
                    {"id": 0, "cpus": [0, 1, 2, 3], "cpu_list": "0-3"},
                    {"id": 1, "cpus": [4, 5, 6, 7], "cpu_list": "4-7"},
                ],
            },
            "model": {
                "path": "/models/canonical",
                "fingerprint": "sha256:model-fingerprint",
            },
            "profile": {"path": profile_path},
            "placement": {
                "memory_nodes": [0, 1],
                "memory_node_list": "0-1",
                "cpu_list": "0-7",
            },
            "staging": {
                "direct_mapped_experts": ["0:0", "0:1"],
                "direct_mapped_expert_count": 2,
                "direct_mapped_bytes": 2 * GIB,
                "selected_shards": ["model-00001.safetensors"],
                "staged_bytes": 2 * GIB,
            },
            "mount_options": {"thp": "advise", "noswap": True},
            "mounts": [
                {
                    "path": "/mnt/colibri-ram",
                    "node": None,
                    "policy": "interleave=static:0-1",
                    "size_bytes": 2 * GIB + max(64 * (1 << 20), (2 * GIB) // 100),
                }
            ],
            "managed_runtime": {"cache_cap": 8, "ctx": 4096},
            "managed_accelerator": {
                "mode": "cuda",
                "layout": "experts-only",
                "devices": [{"index": 0, "numa_node": 0}],
            },
        },
        "mounts": [{"path": "/mnt/colibri-ram", "node": None}],
        "processes": [],
    }


def build_protocol(**overrides):
    values = {
        "manifest": sample_manifest(),
        "engine_path": "/opt/colibri/bin/colibri",
        "profile_path": "/profiles/frozen.coli_usage",
        "residency_gib": 4.0,
        "cuda_host_gib": 3.0,
        "cuda_expert_gib": 8.0,
        "repetitions": 7,
        "seed": 377,
        "practical_threshold": 0.05,
        "confidence": 0.95,
        "fingerprint_file": lambda path: "sha256:" + Path(path).name,
        "created_at": "2026-08-08T00:00:00+00:00",
        "dram_collector": {
            "available": True,
            "collector": "fixture",
            "collector_identity": FIXTURE_COLLECTOR_ID,
            "argv": ["/opt/colibri/bin/dram-collector"],
            "executable_fingerprint": "sha256:dram-collector",
            "metadata": {"available": True, "unit": "bytes"},
            "unit": "bytes",
        },
    }
    values.update(overrides)
    return benchmark.build_causal_protocol(**values)


def raw_row(protocol, treatment_id, block_index, throughput=120.0, sequence=None):
    treatment = next(
        item for item in protocol["treatments"]
        if item["id"] == treatment_id
    )
    treatment_index = next(
        index
        for index, item in enumerate(protocol["treatments"])
        if item["id"] == treatment_id
    )
    if sequence is None:
        position = protocol["randomized_blocks"][block_index]["order"].index(
            treatment_id
        )
        sequence = block_index * len(protocol["treatments"]) + position
    rammap = treatment["storage"] == "tmpfs-rammap"
    local = treatment["numa_policy"]["mode"] == "local"
    placement_pages = {"0": 200} if local else {"0": 100, "1": 100}
    applied_environment = benchmark._environment_projection(
        benchmark._environment_for_treatment(
            treatment,
            environ={},
        )
    )
    row = {
        "schema": benchmark.RAW_EVIDENCE_SCHEMA,
        "version": 1,
        "protocol_id": protocol["protocol_id"],
        "treatment_id": treatment_id,
        "block_index": block_index,
        "sequence": sequence,
        "started_at": "2026-08-08T00:00:00+00:00",
        "finished_at": "2026-08-08T00:00:01+00:00",
        "status": "ok",
        "process": {
            "launch_id": "%032x" % (
                1 + block_index * len(protocol["treatments"]) + treatment_index
            ),
            "pid": 1000 + block_index * len(protocol["treatments"]) + treatment_index,
            "starttime": 9000 + block_index * len(protocol["treatments"]) + treatment_index,
            "supervision_mode": "cgroup-v2",
            "containment_identity_sha256": "c" * 64,
        },
        "applied_environment": applied_environment,
        "numa_policy": copy.deepcopy(treatment["numa_policy"]),
        "fingerprints": copy.deepcopy(protocol["fingerprints"]),
        "output_sha256": "a" * 64,
        "profiler": {
            "rammap_experts": 2 if rammap else 0,
            "rammap_bytes": 2 * GIB if rammap else 0,
            "physical_ssd_bytes": 0,
            "physical_ssd_valid": True,
        },
        "swap": {
            "before_bytes": 0,
            "after_bytes": 0,
            "delta_bytes": 0,
            "process_bytes": 0,
        },
        "rss": {
            "anonymous_bytes": 3 * GIB,
            "file_bytes": GIB,
            "shmem_bytes": 2 * GIB,
        },
        "numa_placement": {
            "by_node": placement_pages,
            "bytes_by_node": {
                node: pages * 4096 for node, pages in placement_pages.items()
            },
            "page_size": 4096,
            "verified": True,
        },
        "dram_traffic": {
            "available": True,
            "read_bytes": 10 * GIB,
            "write_bytes": GIB,
            "total_bytes": 11 * GIB,
            "collector": "fixture",
            "collector_identity": FIXTURE_COLLECTOR_ID,
            "error": None,
        },
        "performance": {
            "tokens_per_second": float(throughput),
            "elapsed_seconds": 1.0,
            "ttft_ms": 10.0,
            "forward_p50_ms": 20.0,
            "forward_p99_ms": 30.0,
        },
        "correctness": {
            "zero_swap_growth": True,
            "physical_reads_ok": True,
            "placement_ok": True,
        },
        "error": None,
    }
    workspace_name = treatment.get("workspace")
    if workspace_name is not None:
        requirements = protocol["workspace_requirements"]["roots"]
        roots = {}
        for index, name in enumerate(("interleaved", "local"), 1):
            expected = requirements[name]
            roots[name] = dict(
                copy.deepcopy(expected),
                path="/mnt/causal-attempt/%s" % name,
                identity={"mount_id": 100 + index, "device": "%d:1" % index},
            )
        row["workspace_attempt"] = {
            "operation_id": "benchmark:" + "c" * 32,
            "roots": roots,
        }
        row["applied_environment"]["SNAP"] = roots[workspace_name]["path"]
    return row


def complete_rows(protocol, pin=100.0, rammap=120.0):
    rates = {
        "anon-pin-interleaved": pin,
        "anon-pin-local": pin,
        "tmpfs-rammap-interleaved": rammap,
        "tmpfs-rammap-local": rammap,
        "ssd-slab-control": pin,
        "tmpfs-slab-control": pin,
        "cuda-fixed-budget-validation": pin,
    }
    return [
        raw_row(protocol, treatment_id, block["block_index"], rates[treatment_id])
        for block in protocol["randomized_blocks"]
        for treatment_id in block["order"]
    ]


class CausalProtocolTest(unittest.TestCase):
    def test_fixed_treatments_controls_and_numeric_budgets(self):
        protocol = build_protocol()

        self.assertEqual(
            tuple(item["id"] for item in protocol["treatments"]),
            TREATMENT_IDS,
        )
        self.assertEqual(
            [item["id"] for item in protocol["treatments"] if item["causal"]],
            list(TREATMENT_IDS[:4]),
        )
        self.assertEqual(protocol["residency_budget_bytes"], 4 * GIB)
        self.assertEqual(protocol["cuda_validation"]["host_budget_bytes"], 3 * GIB)
        self.assertEqual(protocol["cuda_validation"]["gpu_budget_bytes"], 8 * GIB)
        self.assertEqual(protocol["expert_set"], ["0:0", "0:1"])
        self.assertEqual(protocol["predeclared"]["direction"], "higher-throughput")
        self.assertEqual(protocol["predeclared"]["ci_method"], "paired-bootstrap-v1")
        self.assertRegex(protocol["protocol_id"], r"^[0-9a-f]{64}$")
        treatments = {item["id"]: item for item in protocol["treatments"]}
        self.assertIn(
            "--physcpubind=0-7",
            treatments["tmpfs-rammap-interleaved"]["numa_policy"]["command_prefix"],
        )
        self.assertIn(
            "--physcpubind=0-3",
            treatments["tmpfs-rammap-local"]["numa_policy"]["command_prefix"],
        )
        local_environment = benchmark._environment_for_treatment(
            treatments["tmpfs-rammap-local"],
            environ={
                "RAM_GB": "999",
                "PILOT": "1",
                "CTX": "999",
                "COLI_CPU_AFFINITY": "99",
            },
        )
        self.assertNotIn("RAM_GB", local_environment)
        self.assertEqual(local_environment["PILOT"], "0")
        self.assertEqual(local_environment["CTX"], "4096")
        self.assertEqual(local_environment["COLI_CPU_AFFINITY"], "0-3")
        self.assertEqual(local_environment["OMP_NUM_THREADS"], "4")
        self.assertEqual(protocol["cache_cap"], 8)

    def test_child_environment_is_frozen_allowlisted_and_fully_recorded(self):
        protocol = build_protocol(
            inherited_environment={
                "HOME": "/durable/home",
                "PATH": "/reviewed/bin",
                "LANG": "C.UTF-8",
                "CUDA_VISIBLE_DEVICES": "ambient-device",
                "OMP_DYNAMIC": "TRUE",
                "LD_PRELOAD": "/tmp/inject.so",
                "SECRET_TOKEN": "must-not-leak",
            }
        )
        treatment = protocol["treatments"][0]
        child = benchmark._environment_for_treatment(
            treatment,
            environ={
                "HOME": "/changed",
                "PATH": "/changed/bin",
                "LD_PRELOAD": "/tmp/changed.so",
            },
        )
        recorded = benchmark._environment_projection(child)

        self.assertEqual(child["HOME"], "/durable/home")
        self.assertEqual(child["PATH"], "/reviewed/bin")
        self.assertEqual(child["OMP_DYNAMIC"], "FALSE")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", child)
        self.assertNotIn("LD_PRELOAD", child)
        self.assertNotIn("SECRET_TOKEN", child)
        self.assertEqual(set(recorded), set(child))
        self.assertEqual(recorded, child)

    def test_auto_nonfinite_and_undersized_budgets_are_rejected(self):
        invalid = (
            ({"residency_gib": "all"}, "residency budget"),
            ({"residency_gib": float("nan")}, "residency budget"),
            ({"residency_gib": 1.0}, "direct-mapped expert set"),
            ({"cuda_host_gib": "auto"}, "CUDA host budget"),
            ({"cuda_expert_gib": math.inf}, "CUDA expert budget"),
        )
        for overrides, message in invalid:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                benchmark.RamdiskError,
                message,
            ):
                build_protocol(**overrides)

        with self.assertRaisesRegex(
            benchmark.RamdiskError,
            "frozen evidence profile",
        ):
            build_protocol(profile_path="/profiles/different.coli_usage")

        no_cuda = sample_manifest()
        no_cuda["plan"]["managed_accelerator"]["devices"] = []
        with self.assertRaisesRegex(benchmark.RamdiskError, "CUDA device"):
            build_protocol(manifest=no_cuda)

    def test_seeded_order_is_deterministic_complete_and_minimum_seven(self):
        first = benchmark.seeded_block_order(377, TREATMENT_IDS, 7)
        second = benchmark.seeded_block_order(377, TREATMENT_IDS, 7)
        different = benchmark.seeded_block_order(378, TREATMENT_IDS, 7)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(len(first), 7)
        for index, block in enumerate(first):
            self.assertEqual(block["block_index"], index)
            self.assertEqual(set(block["order"]), set(TREATMENT_IDS))
        with self.assertRaisesRegex(benchmark.RamdiskError, "at least 7"):
            benchmark.seeded_block_order(377, TREATMENT_IDS, 6)

    def test_frozen_binary_and_profile_are_revalidated(self):
        protocol = build_protocol()
        expected = {
            protocol["engine_path"]: protocol["fingerprints"]["binary"],
            protocol["profile_path"]: protocol["fingerprints"]["profile"],
            protocol["dram_collector"]["argv"][0]: (
                protocol["dram_collector"]["executable_fingerprint"]
            ),
        }
        benchmark._assert_frozen_artifacts(protocol, expected.__getitem__)
        expected[protocol["engine_path"]] = "sha256:replacement"
        with self.assertRaisesRegex(benchmark.RamdiskError, "binary changed"):
            benchmark._assert_frozen_artifacts(protocol, expected.__getitem__)


class RawEvidenceTest(unittest.TestCase):
    def test_protocol_is_immutable_and_rows_are_append_only(self):
        protocol = build_protocol()
        row0 = raw_row(protocol, TREATMENT_IDS[0], 0)
        row1 = raw_row(protocol, TREATMENT_IDS[1], 0)
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.v1.jsonl"
            protocol_path = Path(directory) / "protocol.v1.json"

            persisted = benchmark.persist_causal_protocol(protocol_path, protocol)
            benchmark.append_raw_evidence(raw_path, row0)
            prefix = raw_path.read_bytes()
            benchmark.append_raw_evidence(raw_path, row1)

            self.assertEqual(persisted["protocol_id"], protocol["protocol_id"])
            self.assertTrue(raw_path.read_bytes().startswith(prefix))
            self.assertEqual(
                [item["treatment_id"] for item in benchmark.read_raw_evidence(raw_path)],
                list(TREATMENT_IDS[:2]),
            )
            changed = copy.deepcopy(protocol)
            changed["protocol_id"] = "f" * 64
            with self.assertRaisesRegex(benchmark.RamdiskError, "different protocol"):
                benchmark.persist_causal_protocol(protocol_path, changed)

            tampered = copy.deepcopy(protocol)
            tampered["residency_budget_bytes"] += GIB
            protocol_path.write_bytes(
                benchmark._canonical_json(tampered) + b"\n"
            )
            with self.assertRaisesRegex(benchmark.RamdiskError, "different protocol"):
                benchmark.persist_causal_protocol(protocol_path, protocol)

    def test_raw_schema_rejects_missing_fields_but_preserves_failed_attempts(self):
        protocol = build_protocol()
        row = raw_row(protocol, TREATMENT_IDS[0], 0)
        self.assertIs(benchmark.validate_raw_evidence_row(row), row)

        missing = copy.deepcopy(row)
        missing.pop("dram_traffic")
        with self.assertRaisesRegex(benchmark.RamdiskError, "dram_traffic"):
            benchmark.validate_raw_evidence_row(missing)

        missing_identity = copy.deepcopy(row)
        missing_identity["process"]["starttime"] = None
        with self.assertRaisesRegex(benchmark.RamdiskError, "starttime"):
            benchmark.validate_raw_evidence_row(missing_identity)

        failed = benchmark.failed_raw_evidence_row(
            protocol,
            protocol["treatments"][0],
            block_index=0,
            sequence=0,
            error="engine did not start",
        )
        self.assertEqual(failed["status"], "error")
        self.assertIn("engine did not start", failed["error"])
        benchmark.validate_raw_evidence_row(failed)

    def test_raw_schema_rejects_boolean_numeric_authority(self):
        protocol = build_protocol()
        row = raw_row(protocol, "tmpfs-rammap-local", 0)

        boolean_pid = copy.deepcopy(row)
        boolean_pid["process"]["pid"] = True
        with self.assertRaisesRegex(benchmark.RamdiskError, "PID"):
            benchmark.validate_raw_evidence_row(boolean_pid)

        boolean_prof = copy.deepcopy(row)
        boolean_prof["profiler"]["physical_ssd_bytes"] = False
        with self.assertRaisesRegex(benchmark.RamdiskError, "physical SSD"):
            benchmark.validate_raw_evidence_row(boolean_prof)

        unavailable_prof = copy.deepcopy(row)
        unavailable_prof["profiler"].update(
            physical_ssd_bytes=None,
            physical_ssd_valid=None,
        )
        benchmark.validate_raw_evidence_row(unavailable_prof)

        boolean_nodes = copy.deepcopy(row)
        boolean_nodes["workspace_attempt"]["roots"]["local"].update(
            nodes=[False],
            node=False,
        )
        with self.assertRaisesRegex(benchmark.RamdiskError, "workspace root"):
            benchmark.validate_raw_evidence_row(boolean_nodes)

    def test_protocol_rejects_string_or_numeric_mount_policy_flags(self):
        for name, value in (("noswap", "false"), ("thp", 1)):
            manifest = sample_manifest()
            manifest["plan"]["mount_options"][name] = value
            with self.subTest(name=name), self.assertRaisesRegex(
                benchmark.RamdiskError,
                "mount options",
            ):
                build_protocol(manifest=manifest)

        manifest = sample_manifest()
        manifest["plan"]["placement"]["memory_nodes"] = [False, True]
        with self.assertRaisesRegex(
            benchmark.RamdiskError,
            "NUMA nodes",
        ):
            build_protocol(manifest=manifest)

    def test_protocol_and_raw_reads_reject_symlink_substitution(self):
        protocol = build_protocol()
        row = raw_row(protocol, TREATMENT_IDS[0], 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_protocol = root / "real-protocol.json"
            protocol_link = root / "protocol.json"
            real_protocol.write_bytes(
                benchmark._canonical_json(protocol) + b"\n"
            )
            protocol_link.symlink_to(real_protocol)
            with self.assertRaisesRegex(benchmark.RamdiskError, "symlink"):
                benchmark.persist_causal_protocol(protocol_link, protocol)

            real_raw = root / "real-raw.jsonl"
            raw_link = root / "raw.jsonl"
            real_raw.write_bytes(benchmark._canonical_json(row) + b"\n")
            raw_link.symlink_to(real_raw)
            with self.assertRaisesRegex(benchmark.RamdiskError, "symlink"):
                benchmark.read_raw_evidence(raw_link)

    def test_evidence_paths_reject_reserved_hardlinks_and_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reserved = root / "benchmarks.json"
            reserved.write_text("{}\n", encoding="utf-8")
            alias = root / "raw.jsonl"
            alias.hardlink_to(reserved)
            with self.assertRaisesRegex(
                benchmark.RamdiskError,
                "durable deployment authority",
            ):
                benchmark._validate_evidence_paths(
                    alias,
                    root / "protocol.json",
                    reserved_paths=[reserved],
                )

            with self.assertRaisesRegex(benchmark.RamdiskError, "distinct"):
                benchmark._validate_evidence_paths(
                    root / "same.json",
                    root / "." / "same.json",
                )

            evidence_root = root / "state" / "causal-evidence"
            with self.assertRaisesRegex(benchmark.RamdiskError, "below"):
                benchmark._require_evidence_root(
                    evidence_root,
                    evidence_root / "protocol.json",
                    evidence_root,
                )
            with self.assertRaisesRegex(benchmark.RamdiskError, "below"):
                benchmark._require_evidence_root(
                    root / "model" / "raw.jsonl",
                    evidence_root / "protocol.json",
                    evidence_root,
                )

    def test_bound_store_rejects_raw_protocol_and_parent_replacement(self):
        protocol = build_protocol()
        for replacement in ("raw", "protocol", "parent"):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory) / "evidence"
                parent.mkdir(mode=0o700)
                raw_path = parent / "raw.jsonl"
                protocol_path = parent / "protocol.json"
                calls = []

                def prestage(_protocol, _treatment):
                    calls.append(True)
                    if len(calls) != 2:
                        return
                    if replacement == "raw":
                        raw_path.unlink()
                        raw_path.write_text("", encoding="utf-8")
                        raw_path.chmod(0o600)
                    elif replacement == "protocol":
                        protocol_path.unlink()
                    else:
                        old = parent.with_name("evidence-old")
                        parent.rename(old)
                        parent.mkdir(mode=0o700)

                with self.assertRaisesRegex(
                    benchmark.RamdiskError,
                    "identity|unavailable|protocol",
                ):
                    benchmark.run_causal_benchmark(
                        protocol,
                        raw_evidence_path=raw_path,
                        protocol_path=protocol_path,
                        prestage=prestage,
                        replicate_runner=lambda current, treatment, block, sequence, **_kw: raw_row(
                            current,
                            treatment["id"],
                            block,
                            sequence=sequence,
                        ),
                        dram_collector=mock.Mock(description=protocol["dram_collector"]),
                        assert_durable=lambda _path: None,
                    )

    def test_bound_store_rejects_private_hardlinked_evidence(self):
        protocol = build_protocol()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            raw_path = parent / "raw.jsonl"
            raw_path.write_text("", encoding="utf-8")
            raw_path.chmod(0o600)
            (parent / "alias.jsonl").hardlink_to(raw_path)

            with self.assertRaisesRegex(benchmark.RamdiskError, "hard links"):
                benchmark.run_causal_benchmark(
                    protocol,
                    raw_evidence_path=raw_path,
                    prestage=lambda *_args: None,
                    replicate_runner=lambda *_args, **_kwargs: None,
                    dram_collector=mock.Mock(description=protocol["dram_collector"]),
                    assert_durable=lambda _path: None,
                )


class CausalExecutionTest(unittest.TestCase):
    def test_full_shard_payload_is_rehashed_at_spawn_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            shard = model / "model-00001.safetensors"
            shard.write_bytes(b"header-payload-a")
            engine = root / "colibri"
            engine.write_bytes(b"engine")
            profile = root / "profile.coli_usage"
            profile.write_bytes(b"profile")
            manifest = sample_manifest(str(profile))
            manifest["plan"]["model"]["path"] = str(model)
            manifest["mounts"] = [{"path": str(model), "node": None}]
            manifest["plan"]["source_shards"] = [
                {"name": shard.name, "path": str(shard)}
            ]
            content = benchmark._model_content_fingerprints(
                manifest["plan"], benchmark._sha256_path
            )
            protocol = build_protocol(
                manifest=manifest,
                engine_path=str(engine),
                profile_path=str(profile),
                fingerprint_file=benchmark._sha256_path,
                model_content_fingerprints=content,
            )
            original_stat = shard.stat()
            shard.write_bytes(b"header-payload-b")
            os.utime(
                shard,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            launched = []

            with self.assertRaisesRegex(
                benchmark.RamdiskError, "model shard changed"
            ):
                benchmark.run_causal_replicate(
                    protocol,
                    protocol["treatments"][0],
                    0,
                    0,
                    engine_factory=lambda *_args, **_kwargs: launched.append(True),
                    render_prompt=lambda value: value,
                    client_cancelled_type=RuntimeError,
                    fingerprint_file=benchmark._sha256_path,
                )
            self.assertEqual(launched, [])

    def test_recorded_environment_is_the_exact_engine_popen_environment(self):
        import openai_server

        class Process:
            def __init__(self):
                self.pid = 4321
                self.returncode = None
                self.stdin = io.BytesIO()
                self.stdout = io.BytesIO(
                    openai_server.READY + b"STAT 0 0 0 0\n"
                )

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def wait(self, timeout=None):
                del timeout
                return self.returncode

            def kill(self):
                self.returncode = -9

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(
                json.dumps({"model_type": "deepseek_v4"}),
                encoding="utf-8",
            )
            defaults = benchmark._production_engine_environment_defaults(
                str(model)
            )
            supplied = {
                **defaults,
                "SNAP": str(model),
                "COLI_WEIGHTS_DIR": str(model),
                "SERVE": "1",
                "SERVE_BATCH": "1",
                "NGEN": "32",
                "KV_SLOTS": "1",
                "COLI_KV_SLOTS": "1",
                "PATH": "/reviewed/bin",
            }
            captured = {}
            retired = []
            real_popen = subprocess.Popen

            def popen(command, **kwargs):
                if "env" not in kwargs:
                    return real_popen(command, **kwargs)
                captured.update(command=command, environment=dict(kwargs["env"]))
                return Process()

            def retire(process):
                retired.append(process.pid)
                process.terminate()

            engine_type, _render, _cancelled = benchmark._default_engine_dependencies(
                None,
                terminate_process=retire,
            )
            with mock.patch.object(openai_server.subprocess, "Popen", side_effect=popen):
                engine = engine_type(
                    "/engine",
                    str(model),
                    max_tokens=32,
                    kv_slots=1,
                    env=supplied,
                )
                engine.close()

            self.assertEqual(
                captured["environment"],
                engine.benchmark_child_environment,
            )
            self.assertEqual(set(captured["environment"]), set(supplied))
            self.assertEqual(retired, [4321])
            for name in (
                "SNAP", "SERVE", "SERVE_BATCH", "NGEN", "KV_SLOTS",
                "GOMP_SPINCOUNT", "OMP_WAIT_POLICY", "V4_MTP_CONF",
            ):
                self.assertIn(name, captured["environment"])
    def test_dram_metadata_and_runtime_counter_loss_are_unavailable(self):
        result = mock.Mock(returncode=0, stdout="[]", stderr="")
        collector = benchmark.preflight_dram_collector(
            environ={"COLI_DRAM_COLLECTOR": "/collector"},
            run=lambda *_args, **_kwargs: result,
            fingerprint_file=lambda _path: "sha256:" + "a" * 64,
        )
        self.assertFalse(collector.available)
        self.assertIn("object", collector.reason)

        for snapshot in ([], {}, {"available": True, "read_bytes": 1}):
            with self.subTest(snapshot=snapshot):
                normalized = benchmark._normalize_dram_measurement(snapshot)
                self.assertFalse(normalized["available"])
                self.assertIsNone(normalized["total_bytes"])

    def test_replicate_uses_one_fresh_engine_and_records_every_measurement(self):
        protocol = build_protocol()
        treatment = protocol["treatments"][2]
        events = []

        class FakeProcess:
            pid = 4321

        class FakeEngine:
            process = FakeProcess()
            profile_seq = 0
            profile = []
            closed = False

            def __init__(self, *args, **kwargs):
                events.append(("engine", args, kwargs))

            def generate(self, prompt, maximum, temperature, top_p, on_text, **kwargs):
                del prompt, maximum, temperature, top_p, kwargs
                events.append(("generate",))
                on_text("deterministic output")
                self.profile_seq += 1
                self.profile.append(
                    {
                        "rammap_experts": 2,
                        "rammap_bytes": 2 * GIB,
                        "physical_ssd_bytes": 0,
                        "physical_ssd_valid": True,
                        "ttft_ms": 9.0,
                        "forward_p50_ms": 12.0,
                        "forward_p99_ms": 18.0,
                    }
                )
                return {"completion_tokens": 32, "tokens_per_second": 16.0}

            def close(self):
                self.closed = True
                events.append(("close",))

        class FakeDram:
            available = True

            def start(self, pid, current_treatment):
                events.append(("dram-start", pid, current_treatment["id"]))
                return "handle"

            def finish(self, handle):
                events.append(("dram-finish", handle))
                return {
                    "available": True,
                    "read_bytes": 100,
                    "write_bytes": 20,
                    "total_bytes": 120,
                    "collector": "fixture",
                    "collector_identity": FIXTURE_COLLECTOR_ID,
                    "error": None,
                }

        swap = iter((1000, 1000))
        clocks = iter((10.0, 12.0))
        timestamps = iter(("start", "finish"))
        engine = object.__new__(FakeEngine)
        engine.process = FakeProcess()
        engine.profile_seq = 0
        engine.profile = []
        engine.closed = False
        row = benchmark.run_causal_replicate(
            protocol,
            treatment,
            block_index=0,
            sequence=0,
            engine_path="/fake/colibri",
            engine_factory=lambda *args, **kwargs: (
                FakeEngine.__init__(engine, *args, **kwargs) or engine
            ),
            render_prompt=lambda _prompt: "rendered",
            process_starttime=lambda _pid: 7654,
            swap_used_bytes=lambda: next(swap),
            rss_sampler=lambda _pid: {
                "anonymous_bytes": 10,
                "file_bytes": 20,
                "shmem_bytes": 30,
                "process_swap_bytes": 0,
            },
            numa_sampler=lambda _pid, _policy: {
                "by_node": {"0": 5, "1": 5},
                "bytes_by_node": {"0": 20, "1": 20},
                "page_size": 4,
                "verified": True,
            },
            dram_collector=FakeDram(),
            monotonic=lambda: next(clocks),
            utc_now=lambda: next(timestamps),
            launch_id_factory=lambda: "launch-1",
            environ={},
        )

        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["process"], {"launch_id": "launch-1", "pid": 4321, "starttime": 7654})
        self.assertEqual(row["rss"]["anonymous_bytes"], 10)
        self.assertEqual(row["dram_traffic"]["total_bytes"], 120)
        self.assertEqual(row["performance"]["elapsed_seconds"], 2.0)
        self.assertEqual(row["performance"]["tokens_per_second"], 16.0)
        self.assertRegex(row["output_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(row["correctness"]["physical_reads_ok"])
        self.assertTrue(engine.closed)
        self.assertEqual(
            [event[0] for event in events],
            ["engine", "dram-start", "generate", "dram-finish", "close"],
        )

    def test_prestage_precedes_every_fresh_process_and_every_attempt_appends(self):
        protocol = build_protocol()
        events = []
        appended = []

        def prestage(_protocol, treatment):
            events.append(("prestage", treatment["id"]))

        def run_replicate(_protocol, treatment, block_index, sequence, **_kwargs):
            events.append(("launch", treatment["id"]))
            return raw_row(
                protocol,
                treatment["id"],
                block_index,
                throughput=100 + sequence,
            )

        result = benchmark.run_causal_benchmark(
            protocol,
            raw_evidence_path="/not-written/raw.v1.jsonl",
            prestage=prestage,
            replicate_runner=run_replicate,
            dram_collector=mock.sentinel.collector,
            persist_protocol=lambda _path, value: value,
            existing_rows=lambda _path: [],
            append_row=lambda _path, row: appended.append(row),
        )

        expected = len(TREATMENT_IDS) * 7
        self.assertEqual(len(appended), expected)
        self.assertEqual(result["attempted_replicates"], expected)
        self.assertEqual(
            len({row["process"]["launch_id"] for row in appended}),
            expected,
        )
        for index in range(0, len(events), 2):
            self.assertEqual(events[index][0], "prestage")
            self.assertEqual(events[index + 1][0], "launch")
            self.assertEqual(events[index][1], events[index + 1][1])

    def test_existing_block_rows_resume_without_duplicate_launches(self):
        protocol = build_protocol()
        first = protocol["randomized_blocks"][0]["order"][0]
        existing = [raw_row(protocol, first, 0)]
        launches = []
        appended = []

        benchmark.run_causal_benchmark(
            protocol,
            raw_evidence_path="/not-written/raw.v1.jsonl",
            prestage=lambda *_args: None,
            replicate_runner=lambda p, t, block_index, sequence, **kwargs: (
                launches.append((block_index, t["id"]))
                or raw_row(p, t["id"], block_index)
            ),
            dram_collector=mock.sentinel.collector,
            persist_protocol=lambda _path, value: value,
            existing_rows=lambda _path: list(existing),
            append_row=lambda _path, row: appended.append(row),
        )

        self.assertNotIn((0, first), launches)
        self.assertEqual(len(appended), len(TREATMENT_IDS) * 7 - 1)

    def test_resume_rejects_non_prefix_rows_before_launch(self):
        protocol = build_protocol()
        second = protocol["randomized_blocks"][0]["order"][1]
        existing = [raw_row(protocol, second, 0)]
        launches = []

        with self.assertRaisesRegex(benchmark.RamdiskError, "prefix"):
            benchmark.run_causal_benchmark(
                protocol,
                raw_evidence_path="/not-written/raw.v1.jsonl",
                prestage=lambda *_args: None,
                replicate_runner=lambda *_args, **_kwargs: launches.append(True),
                dram_collector=mock.sentinel.collector,
                persist_protocol=lambda _path, value: value,
                existing_rows=lambda _path: existing,
                append_row=lambda *_args: None,
            )
        self.assertEqual(launches, [])

    def test_unexpected_replicate_failure_still_appends_one_failed_row(self):
        protocol = build_protocol()
        appended = []
        attempts = []

        def run_replicate(current, treatment, block_index, sequence, **_kwargs):
            attempts.append((block_index, treatment["id"]))
            if len(attempts) == 1:
                raise TypeError("unexpected collector adapter failure")
            return raw_row(
                current,
                treatment["id"],
                block_index,
                sequence=sequence,
            )

        result = benchmark.run_causal_benchmark(
            protocol,
            raw_evidence_path="/not-written/raw.v1.jsonl",
            prestage=lambda *_args: None,
            replicate_runner=run_replicate,
            dram_collector=mock.sentinel.collector,
            persist_protocol=lambda _path, value: value,
            existing_rows=lambda _path: [],
            append_row=lambda _path, row: appended.append(row),
        )

        self.assertEqual(len(appended), len(TREATMENT_IDS) * 7)
        self.assertEqual(appended[0]["status"], "error")
        self.assertIn("collector adapter", appended[0]["error"])
        self.assertEqual(result["attempted_replicates"], len(appended))

    def test_engine_cleanup_failure_is_not_persisted_and_aborts_later_launches(self):
        protocol = build_protocol()
        appended = []
        launches = []

        def fail_cleanup(*_args, **_kwargs):
            launches.append("launch")
            raise benchmark._EngineCleanupError("fresh engine survived cleanup")

        with self.assertRaisesRegex(
            benchmark._EngineCleanupError,
            "survived cleanup",
        ):
            benchmark.run_causal_benchmark(
                protocol,
                raw_evidence_path="/not-written/raw.v1.jsonl",
                prestage=lambda *_args: None,
                replicate_runner=fail_cleanup,
                dram_collector=mock.sentinel.collector,
                persist_protocol=lambda _path, value: value,
                existing_rows=lambda _path: [],
                append_row=lambda _path, row: appended.append(row),
            )

        self.assertEqual(launches, ["launch"])
        self.assertEqual(appended, [])

    def test_pending_process_recovery_failure_is_not_persisted_or_skipped(self):
        protocol = build_protocol()
        appended = []
        launches = []

        def fail_recovery(*_args, **_kwargs):
            launches.append("recovery")
            raise benchmark.WorkspaceCleanupError(
                "pending benchmark process is inconclusive"
            )

        with self.assertRaisesRegex(
            benchmark.WorkspaceCleanupError,
            "process is inconclusive",
        ):
            benchmark.run_causal_benchmark(
                protocol,
                raw_evidence_path="/not-written/raw.v1.jsonl",
                prestage=lambda *_args: None,
                replicate_runner=fail_recovery,
                dram_collector=mock.sentinel.collector,
                persist_protocol=lambda _path, value: value,
                existing_rows=lambda _path: [],
                append_row=lambda _path, row: appended.append(row),
            )

        self.assertEqual(launches, ["recovery"])
        self.assertEqual(appended, [])

    def test_workspace_reverification_failure_aborts_later_launches(self):
        protocol = build_protocol()
        appended = []
        launches = []

        with self.assertRaisesRegex(
            benchmark.WorkspaceVerificationError,
            "workspace identity changed",
        ):
            benchmark.run_causal_benchmark(
                protocol,
                raw_evidence_path="/not-written/raw.v1.jsonl",
                prestage=lambda *_args: (_ for _ in ()).throw(
                    benchmark.WorkspaceVerificationError(
                        "workspace identity changed"
                    )
                ),
                replicate_runner=lambda *_args, **_kwargs: launches.append("launch"),
                dram_collector=mock.sentinel.collector,
                persist_protocol=lambda _path, value: value,
                existing_rows=lambda _path: [],
                append_row=lambda _path, row: appended.append(row),
            )

        self.assertEqual(launches, [])
        self.assertEqual(len(appended), 1)
        self.assertEqual(appended[0]["status"], "error")

    def test_high_level_runner_prestages_and_uses_unique_retry_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical"
            staged = root / "staged"
            local_staged = root / "local-staged"
            state = root / "state"
            canonical.mkdir()
            staged.mkdir()
            local_staged.mkdir()
            shard = "model-00001.safetensors"
            (canonical / shard).write_bytes(b"canonical")
            (staged / shard).write_bytes(b"staged")
            (local_staged / shard).write_bytes(b"local-staged")
            profile = root / "frozen.coli_usage"
            profile.write_bytes(b"profile")
            engine = root / "colibri"
            engine.write_bytes(b"engine")
            manifest = sample_manifest(str(profile))
            manifest["plan"]["model"]["path"] = str(canonical)
            manifest["mounts"] = [{"path": str(staged), "node": None}]
            args = argparse.Namespace(
                evidence_profile=str(profile),
                residency_gb=4.0,
                cuda_host_gb=3.0,
                cuda_expert_gb=8.0,
                replicates=7,
                seed=377,
                practical_threshold=0.05,
                confidence=0.95,
                raw_evidence=str(state / "causal-evidence" / "raw.v1.jsonl"),
            )
            created_state = []
            replicate_calls = []
            admissions = []
            verified_roots = []

            class WorkspaceManager:
                @contextlib.contextmanager
                def open(self, current_manifest, protocol, cancel_event):
                    self.opened = (current_manifest, protocol, cancel_event)
                    required = protocol["workspace_requirements"]["roots"]
                    yield {
                        "interleaved": dict(required["interleaved"], **{
                            "name": "interleaved",
                            "operation_id": "deployment:mount:0",
                            "path": str(staged),
                            "requested": {"source": "tmpfs"},
                            "identity": {"mount_id": 10, "device": "1:10"},
                            "verified": True,
                        }),
                        "local": dict(required["local"], **{
                            "name": "local",
                            "operation_id": "benchmark:" + "a" * 32,
                            "path": str(local_staged),
                            "requested": {"source": "tmpfs"},
                            "identity": {"mount_id": 11, "device": "1:11"},
                            "verified": True,
                        }),
                    }

                def verify(self, descriptor):
                    verified_roots.append(descriptor["path"])
                    return True

                def bind_protocol(self, protocol):
                    self.bound_protocol_id = protocol["protocol_id"]

            workspace_manager = WorkspaceManager()

            def ensure_private(path):
                Path(path).mkdir(parents=True, exist_ok=False)
                created_state.append(path)

            def replicate(*call_args, **call_kwargs):
                replicate_calls.append((call_args, call_kwargs))
                return mock.sentinel.row

            def causal_runner(protocol, **kwargs):
                by_id = {item["id"]: item for item in protocol["treatments"]}
                self.assertEqual(
                    by_id["tmpfs-rammap-interleaved"]["weights_path"],
                    "workspace://interleaved",
                )
                self.assertEqual(
                    by_id["tmpfs-rammap-local"]["weights_path"],
                    "workspace://local",
                )
                kwargs["prestage"](protocol, by_id["anon-pin-interleaved"])
                bound_interleaved = kwargs["prestage"](
                    protocol, by_id["tmpfs-rammap-interleaved"]
                )
                bound_local = kwargs["prestage"](
                    protocol, by_id["tmpfs-rammap-local"]
                )
                self.assertEqual(bound_interleaved["weights_path"], str(staged))
                self.assertEqual(bound_local["weights_path"], str(local_staged))
                for _ in range(2):
                    kwargs["replicate_runner"](
                        protocol,
                        by_id["anon-pin-interleaved"],
                        0,
                        0,
                        dram_collector=kwargs["dram_collector"],
                        cancel_event=None,
                    )
                return {"protocol_id": protocol["protocol_id"]}

            result = benchmark.run_benchmark(
                args,
                cli_path="/opt/colibri/coli",
                load_manifest=lambda required: manifest,
                assert_effective_masks_unchanged=lambda _plan: None,
                assert_ready_mounts=lambda _manifest: None,
                resolve_engine_path=lambda _cli, _engine: str(engine),
                source_build_identity=lambda: {"revision": "95128b5"},
                fingerprint_file=lambda path: "sha256:" + Path(path).name,
                state_root=lambda: str(state),
                ensure_private_dir=ensure_private,
                assert_durable_state_dir=lambda _path, plan: None,
                admit_runtime=lambda _plan, _mount, benchmark: admissions.append(benchmark),
                fresh_user_binary=lambda name: "/usr/bin/" + name,
                environ={},
                dram_preflight=lambda environ: mock.sentinel.collector,
                causal_runner=causal_runner,
                replicate_runner=replicate,
                workspace_manager=workspace_manager,
                freeze_engine_environment=lambda *_args, **_kwargs: {},
                fingerprint_model_content=lambda *_args, **_kwargs: {},
            )

        self.assertRegex(result["protocol_id"], r"^[0-9a-f]{64}$")
        self.assertEqual(admissions, [True, False, False])
        self.assertEqual(
            verified_roots,
            [str(staged), str(local_staged), str(staged), str(local_staged)],
        )
        self.assertEqual(len(replicate_calls), 2)
        state_dirs = [call[1]["state_dir"] for call in replicate_calls]
        self.assertEqual(len(set(state_dirs)), 2)
        self.assertEqual(state_dirs, created_state)
        self.assertTrue(all(Path(path).is_absolute() for path in state_dirs))

    def test_high_level_two_invocation_resume_uses_stable_logical_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical"
            prepared = root / "prepared"
            state = root / "state"
            for path in (canonical, prepared):
                path.mkdir()
            shard = "model-00001.safetensors"
            (canonical / shard).write_bytes(b"canonical")
            (prepared / shard).write_bytes(b"prepared")
            profile = root / "frozen.coli_usage"
            profile.write_bytes(b"profile")
            engine = root / "colibri"
            engine.write_bytes(b"engine")
            manifest = sample_manifest(str(profile))
            manifest["plan"]["model"]["path"] = str(canonical)
            manifest["mounts"] = [{"path": str(prepared), "node": None}]
            raw_path = state / "causal-evidence" / "raw.v1.jsonl"
            args = argparse.Namespace(
                evidence_profile=str(profile),
                residency_gb=4.0,
                cuda_host_gb=3.0,
                cuda_expert_gb=8.0,
                replicates=7,
                seed=377,
                practical_threshold=0.05,
                confidence=0.95,
                raw_evidence=str(raw_path),
            )
            bound_ids = []

            class Collector:
                description = {
                    "available": True,
                    "collector": "fixture",
                    "collector_identity": FIXTURE_COLLECTOR_ID,
                    "argv": ["/opt/colibri/bin/dram-collector"],
                    "executable_fingerprint": "sha256:dram-collector",
                    "metadata": {"available": True, "unit": "bytes"},
                    "unit": "bytes",
                }

            class WorkspaceManager:
                attempt = 0
                fail_unused_cleanup = False
                replace_evidence_during_recovery = False

                @contextlib.contextmanager
                def open(self, _manifest, protocol, _cancel_event):
                    self.attempt += 1
                    scratch = root / ("scratch-%d" % self.attempt)
                    scratch.mkdir()
                    (scratch / shard).write_bytes(b"scratch")
                    required = protocol["workspace_requirements"]["roots"]
                    yield {
                        "interleaved": dict(required["interleaved"], **{
                            "name": "interleaved",
                            "operation_id": "deployment:mount:0",
                            "path": str(prepared),
                            "requested": {"source": "tmpfs"},
                            "identity": {"mount_id": 10, "device": "1:10"},
                            "verified": True,
                        }),
                        "local": dict(required["local"], **{
                            "name": "local",
                            "operation_id": "benchmark:%032x" % self.attempt,
                            "path": str(scratch),
                            "requested": {"source": "tmpfs"},
                            "identity": {
                                "mount_id": 20 + self.attempt,
                                "device": "%d:20" % (2 + self.attempt),
                            },
                            "verified": True,
                        }),
                    }
                    if self.fail_unused_cleanup:
                        raise benchmark.WorkspaceCleanupError(
                            "unused scratch cleanup failed"
                        )

                @staticmethod
                def verify(_descriptor):
                    return True

                @staticmethod
                def bind_protocol(protocol):
                    bound_ids.append(protocol["protocol_id"])
                    return protocol["protocol_id"]

                def recover(self, _manifest):
                    if not self.replace_evidence_during_recovery:
                        return
                    replacement = raw_path.with_name("replacement.jsonl")
                    replacement.write_bytes(raw_path.read_bytes())
                    replacement.chmod(0o600)
                    os.replace(replacement, raw_path)

            manager = WorkspaceManager()
            first_cancel = threading.Event()
            launched = [0]

            def replicate(protocol, treatment, block_index, sequence, **_kwargs):
                row = raw_row(
                    protocol,
                    treatment["id"],
                    block_index,
                    throughput=(
                        120.0
                        if treatment["id"].startswith("tmpfs-rammap-")
                        else 100.0
                    ),
                    sequence=sequence,
                )
                if treatment.get("_workspace_attempt") is not None:
                    row["workspace_attempt"] = copy.deepcopy(
                        treatment["_workspace_attempt"]
                    )
                    row["applied_environment"]["SNAP"] = treatment["weights_path"]
                launched[0] += 1
                if launched[0] == 1:
                    first_cancel.set()
                return row

            common = dict(
                load_manifest=lambda required: manifest,
                assert_effective_masks_unchanged=lambda _plan: None,
                assert_ready_mounts=lambda _manifest: None,
                resolve_engine_path=lambda _cli, _engine: str(engine),
                source_build_identity=lambda: {"revision": "95128b5"},
                fingerprint_file=lambda path: "sha256:" + Path(path).name,
                state_root=lambda: str(state),
                ensure_private_dir=lambda path: Path(path).mkdir(parents=True),
                assert_durable_state_dir=lambda _path, plan: None,
                admit_runtime=lambda *_args, **_kwargs: None,
                fresh_user_binary=lambda name: "/usr/bin/" + name,
                environ={},
                dram_preflight=lambda environ: Collector(),
                replicate_runner=replicate,
                workspace_manager=manager,
                freeze_engine_environment=lambda *_args, **_kwargs: {},
                fingerprint_model_content=lambda *_args, **_kwargs: {},
            )
            with self.assertRaises(benchmark._OperationCancelled):
                benchmark.run_benchmark(
                    args,
                    cli_path="/opt/colibri/coli",
                    cancel_event=first_cancel,
                    **common
                )
            launched[0] = 1
            result = benchmark.run_benchmark(
                args,
                cli_path="/opt/colibri/coli",
                cancel_event=threading.Event(),
                **common
            )
            manager.fail_unused_cleanup = True
            resumed_result = benchmark.run_benchmark(
                args,
                cli_path="/opt/colibri/coli",
                cancel_event=threading.Event(),
                **common
            )
            manager.replace_evidence_during_recovery = True
            with self.assertRaisesRegex(
                benchmark.RamdiskError,
                "raw evidence path identity changed",
            ):
                benchmark.run_benchmark(
                    args,
                    cli_path="/opt/colibri/coli",
                    cancel_event=threading.Event(),
                    **common
                )
            rows = benchmark.read_raw_evidence(raw_path)

        self.assertEqual(len(set(bound_ids)), 1)
        self.assertEqual(len(rows), 49)
        self.assertEqual((result["status"], result["claim"]), ("complete", "improvement"))
        self.assertEqual(
            (resumed_result["status"], resumed_result["claim"]),
            ("complete", "improvement"),
        )
        self.assertEqual(manager.attempt, 2)
        workspace_attempts = {
            row["workspace_attempt"]["operation_id"]
            for row in rows
            if row.get("workspace_attempt") is not None
        }
        self.assertEqual(len(workspace_attempts), 2)

    def test_workspace_cleanup_failure_is_persisted_as_neutral_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical"
            interleaved = root / "interleaved"
            local = root / "local"
            state = root / "state"
            for path in (canonical, interleaved, local):
                path.mkdir()
            shard = "model-00001.safetensors"
            for path in (canonical, interleaved, local):
                (path / shard).write_bytes(path.name.encode("ascii"))
            profile = root / "frozen.coli_usage"
            profile.write_bytes(b"profile")
            engine = root / "colibri"
            engine.write_bytes(b"engine")
            raw_path = state / "causal-evidence" / "raw.v1.jsonl"
            manifest = sample_manifest(str(profile))
            manifest["plan"]["model"]["path"] = str(canonical)
            manifest["mounts"] = [{"path": str(interleaved), "node": None}]
            args = argparse.Namespace(
                evidence_profile=str(profile),
                residency_gb=4.0,
                cuda_host_gb=3.0,
                cuda_expert_gb=8.0,
                replicates=7,
                seed=377,
                practical_threshold=0.05,
                confidence=0.95,
                raw_evidence=str(raw_path),
            )

            class FailingCleanupManager:
                @contextlib.contextmanager
                def open(self, _manifest, protocol, _cancel_event):
                    required = protocol["workspace_requirements"]["roots"]
                    yield {
                        "interleaved": dict(required["interleaved"], **{
                            "name": "interleaved",
                            "operation_id": "deployment:mount:0",
                            "path": str(interleaved),
                            "requested": {"source": "tmpfs"},
                            "identity": {"mount_id": 10, "device": "1:10"},
                            "verified": True,
                        }),
                        "local": dict(required["local"], **{
                            "name": "local",
                            "operation_id": "benchmark:" + "a" * 32,
                            "path": str(local),
                            "requested": {"source": "tmpfs"},
                            "identity": {"mount_id": 11, "device": "1:11"},
                            "verified": True,
                        }),
                    }
                    raise benchmark.WorkspaceCleanupError("scratch unmount failed")

                @staticmethod
                def verify(_descriptor):
                    return True

                @staticmethod
                def bind_protocol(protocol):
                    return protocol["protocol_id"]

            result = benchmark.run_benchmark(
                args,
                cli_path="/opt/colibri/coli",
                load_manifest=lambda required: manifest,
                assert_effective_masks_unchanged=lambda _plan: None,
                assert_ready_mounts=lambda _manifest: None,
                resolve_engine_path=lambda _cli, _engine: str(engine),
                source_build_identity=lambda: {"revision": "95128b5"},
                fingerprint_file=lambda path: "sha256:" + Path(path).name,
                state_root=lambda: str(state),
                ensure_private_dir=lambda path: Path(path).mkdir(parents=True),
                assert_durable_state_dir=lambda _path, plan: None,
                admit_runtime=lambda *_args, **_kwargs: None,
                fresh_user_binary=lambda name: "/usr/bin/" + name,
                environ={},
                dram_preflight=lambda environ: mock.sentinel.collector,
                causal_runner=lambda protocol, **_kwargs: {
                    "protocol_id": protocol["protocol_id"],
                },
                workspace_manager=FailingCleanupManager(),
                freeze_engine_environment=lambda *_args, **_kwargs: {},
                fingerprint_model_content=lambda *_args, **_kwargs: {},
            )
            persisted = benchmark.read_raw_evidence(raw_path)

            manifest["benchmark_workspace"] = {
                "pending_process": {"recovery_error": "inconclusive"}
            }
            with self.assertRaisesRegex(
                benchmark.WorkspaceCleanupError,
                "process recovery authority remains durable",
            ):
                benchmark.run_benchmark(
                    args,
                    cli_path="/opt/colibri/coli",
                    load_manifest=lambda required: manifest,
                    assert_effective_masks_unchanged=lambda _plan: None,
                    assert_ready_mounts=lambda _manifest: None,
                    resolve_engine_path=lambda _cli, _engine: str(engine),
                    source_build_identity=lambda: {"revision": "95128b5"},
                    fingerprint_file=lambda path: "sha256:" + Path(path).name,
                    state_root=lambda: str(state),
                    ensure_private_dir=lambda path: Path(path).mkdir(
                        parents=True,
                        exist_ok=True,
                    ),
                    assert_durable_state_dir=lambda _path, plan: None,
                    admit_runtime=lambda *_args, **_kwargs: None,
                    fresh_user_binary=lambda name: "/usr/bin/" + name,
                    environ={},
                    dram_preflight=lambda environ: mock.sentinel.collector,
                    causal_runner=lambda protocol, **_kwargs: {
                        "protocol_id": protocol["protocol_id"],
                    },
                    workspace_manager=FailingCleanupManager(),
                    freeze_engine_environment=lambda *_args, **_kwargs: {},
                    fingerprint_model_content=lambda *_args, **_kwargs: {},
                )
            self.assertEqual(len(benchmark.read_raw_evidence(raw_path)), 1)

        self.assertEqual((result["status"], result["claim"]), ("incomplete", "neutral"))
        self.assertIn("scratch unmount failed", " ".join(result["reasons"]))
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0]["record_type"], "workspace-cleanup")


class CausalClaimTest(unittest.TestCase):
    def test_protocol_identity_is_revalidated_before_a_claim(self):
        protocol = build_protocol(practical_threshold=0.50)
        rows = complete_rows(protocol, pin=100.0, rammap=120.0)
        original_id = protocol["protocol_id"]
        protocol["predeclared"]["practical_threshold"] = 0.05

        self.assertNotEqual(
            benchmark._causal_protocol_identity(protocol),
            original_id,
        )
        with self.assertRaisesRegex(benchmark.RamdiskError, "protocol identity"):
            benchmark.evaluate_causal_claim(protocol, rows)

    def test_persisted_derived_measurements_cannot_override_raw_counters(self):
        protocol = build_protocol()
        mutations = (
            lambda row: row["dram_traffic"].update(total_bytes=999),
            lambda row: row["swap"].update(after_bytes=10**12, delta_bytes=0),
            lambda row: row["numa_placement"].update(
                by_node={}, bytes_by_node="not-a-map", verified=True
            ),
            lambda row: row["correctness"].update(zero_swap_growth=False),
            lambda row: row["correctness"].update(placement_ok=False),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                row = complete_rows(protocol)[0]
                mutate(row)
                with self.assertRaises(benchmark.RamdiskError):
                    benchmark.evaluate_causal_claim(protocol, [row])

    def test_mixed_dram_collector_identity_is_invalid(self):
        protocol = build_protocol()
        rows = complete_rows(protocol)
        rows[0]["dram_traffic"]["collector_identity"] = "c" * 64

        result = benchmark.evaluate_causal_claim(protocol, rows)

        self.assertEqual((result["status"], result["claim"]), ("invalid", "neutral"))
        self.assertIn("DRAM collector", " ".join(result["reasons"]))

    def test_paired_interval_is_deterministic_and_claim_requires_lower_bounds(self):
        protocol = build_protocol(practical_threshold=0.10)
        interval = benchmark.paired_interval(
            [100.0] * 7,
            [125.0] * 7,
            confidence=0.95,
            seed=377,
        )
        self.assertEqual(interval, benchmark.paired_interval(
            [100.0] * 7,
            [125.0] * 7,
            confidence=0.95,
            seed=377,
        ))
        self.assertAlmostEqual(interval["point_estimate"], 0.25)
        self.assertAlmostEqual(interval["lower_bound"], 0.25)
        self.assertAlmostEqual(interval["upper_bound"], 0.25)

        positive = benchmark.evaluate_causal_claim(
            protocol,
            complete_rows(protocol, pin=100.0, rammap=125.0),
        )
        neutral = benchmark.evaluate_causal_claim(
            protocol,
            complete_rows(protocol, pin=100.0, rammap=105.0),
        )
        self.assertEqual((positive["status"], positive["claim"]), ("complete", "improvement"))
        self.assertEqual((neutral["status"], neutral["claim"]), ("complete", "neutral"))

    def test_each_correctness_gate_invalidates_evidence(self):
        protocol = build_protocol()
        mutations = (
            ("output", lambda row: row.__setitem__("output_sha256", "b" * 64)),
            (
                "swap",
                lambda row: (
                    row["swap"].update(after_bytes=1, delta_bytes=1),
                    row["correctness"].update(zero_swap_growth=False),
                ),
            ),
            ("process swap", lambda row: row["swap"].update(process_bytes=4096)),
            ("physical", lambda row: row["profiler"].update(physical_ssd_bytes=4096)),
            (
                "placement",
                lambda row: (
                    row["numa_placement"].update(
                        by_node={"0": 200},
                        bytes_by_node={"0": 819200},
                        verified=False,
                    ),
                    row["correctness"].update(placement_ok=False),
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                rows = complete_rows(protocol)
                target = next(
                    row for row in rows
                    if row["treatment_id"] == "tmpfs-rammap-interleaved"
                )
                mutate(target)
                result = benchmark.evaluate_causal_claim(protocol, rows)
                self.assertEqual(result["status"], "invalid")
                self.assertEqual(result["claim"], "neutral")
                self.assertTrue(result["reasons"])

    def test_missing_dram_counters_make_claim_incomplete_and_neutral(self):
        protocol = build_protocol()
        rows = complete_rows(protocol)
        rows[0]["dram_traffic"] = {
            "available": False,
            "read_bytes": None,
            "write_bytes": None,
            "total_bytes": None,
            "collector": None,
            "collector_identity": None,
            "error": "collector unavailable",
        }

        result = benchmark.evaluate_causal_claim(protocol, rows)

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["claim"], "neutral")
        self.assertIn("DRAM", " ".join(result["reasons"]))

    def test_missing_required_latency_makes_claim_incomplete_and_neutral(self):
        protocol = build_protocol()
        rows = complete_rows(protocol)
        rows[0]["performance"]["forward_p99_ms"] = None

        result = benchmark.evaluate_causal_claim(protocol, rows)

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["claim"], "neutral")
        self.assertIn("forward_p99_ms", " ".join(result["reasons"]))

    def test_launch_identity_must_be_unique_across_all_treatments(self):
        protocol = build_protocol()
        rows = complete_rows(protocol)
        rows[1]["process"]["launch_id"] = rows[0]["process"]["launch_id"]

        result = benchmark.evaluate_causal_claim(protocol, rows)

        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["claim"], "neutral")
        self.assertIn("fresh process", " ".join(result["reasons"]))

    def test_reordered_complete_evidence_cannot_produce_a_claim(self):
        protocol = build_protocol()
        rows = complete_rows(protocol)
        rows[0], rows[1] = rows[1], rows[0]

        result = benchmark.evaluate_causal_claim(protocol, rows)

        self.assertEqual((result["status"], result["claim"]), ("invalid", "neutral"))
        self.assertIn("append order", " ".join(result["reasons"]))

    def test_frozen_schedule_environment_fingerprints_and_rammap_are_claim_gates(self):
        protocol = build_protocol()

        def change_numa_policy(rows):
            row = rows[0]
            row["numa_policy"].update(nodes=[7])
            row["numa_placement"].update(
                by_node={"7": 200},
                bytes_by_node={"7": 819200},
                verified=True,
            )

        mutations = (
            (
                "sequence",
                lambda rows: rows[0].__setitem__("sequence", rows[0]["sequence"] + 1),
            ),
            (
                "environment",
                lambda rows: rows[0]["applied_environment"].update(PIN_GB="999"),
            ),
            (
                "NUMA policy",
                change_numa_policy,
            ),
            (
                "fingerprint",
                lambda rows: rows[0]["fingerprints"].update(binary="sha256:wrong"),
            ),
            (
                "RAMMAP telemetry",
                lambda rows: next(
                    row
                    for row in rows
                    if row["treatment_id"] == "tmpfs-rammap-interleaved"
                )["profiler"].update(rammap_experts=0),
            ),
            (
                "process identity",
                lambda rows: rows[1]["process"].update(
                    pid=rows[0]["process"]["pid"],
                    starttime=rows[0]["process"]["starttime"],
                ),
            ),
        )
        for expected, mutate in mutations:
            with self.subTest(expected=expected):
                rows = complete_rows(protocol)
                mutate(rows)
                result = benchmark.evaluate_causal_claim(protocol, rows)
                self.assertEqual(result["status"], "invalid")
                self.assertEqual(result["claim"], "neutral")
                self.assertIn(expected, " ".join(result["reasons"]))

    def test_unexpected_block_cannot_be_ignored_by_the_paired_interval(self):
        protocol = build_protocol()
        rows = complete_rows(protocol)
        extra = raw_row(protocol, TREATMENT_IDS[0], 0)
        extra["block_index"] = protocol["repetitions"]
        extra["sequence"] = protocol["repetitions"] * len(protocol["treatments"])
        extra["process"]["launch_id"] = "f" * 32
        extra["process"]["pid"] = 50000
        extra["process"]["starttime"] = 60000
        rows.append(extra)

        result = benchmark.evaluate_causal_claim(protocol, rows)

        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["claim"], "neutral")
        self.assertIn("unexpected block", " ".join(result["reasons"]))


if __name__ == "__main__":
    unittest.main()
