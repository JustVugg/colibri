"""RAM-disk benchmark lifecycle and scoring tests."""

if __package__:
    from .ramdisk_test_support import *  # noqa: F401,F403
else:
    from ramdisk_test_support import *  # noqa: F401,F403


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
        with canonical_temporary_directory() as directory:
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
        ), mock.patch.object(
            ramdisk.os, "statvfs", return_value=filesystem, create=True
        ):
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

        with ModelFixture() as fixture, canonical_temporary_directory() as state:
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

        with ModelFixture() as fixture, canonical_temporary_directory() as state:
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

