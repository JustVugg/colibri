import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from resource_plan import (
    GB,
    analyze_model,
    build_plan,
    cpu_socket_count,
    environment_for_plan,
    format_plan,
    memory_available,
    parse_ssd_cache,
    physical_cpu_count,
    read_ssd_probe,
    ssd_probe_state,
)


def write_shard(path, tensors):
    offset = 0
    header = {}
    payload = b""
    for name, size in tensors:
        header[name] = {"dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]}
        payload += b"\0" * size
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class ResourcePlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name)
        (self.model / "config.json").write_text(json.dumps({
            "num_hidden_layers": 2,
            "n_routed_experts": 2,
            "kv_lora_rank": 4,
            "qk_rope_head_dim": 2,
            "qk_nope_head_dim": 3,
            "v_head_dim": 5,
            "num_attention_heads": 2,
        }))
        write_shard(self.model / "model.safetensors", [
            ("model.embed_tokens.weight", 100),
            ("model.layers.0.self_attn.q_a_proj.weight", 200),
            ("model.layers.1.mlp.experts.0.gate_proj.weight", 30),
            ("model.layers.1.mlp.experts.0.up_proj.weight", 30),
            ("model.layers.1.mlp.experts.1.gate_proj.weight", 30),
            ("model.layers.1.mlp.experts.1.up_proj.weight", 30),
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def test_analyzes_dense_and_expert_storage(self):
        info = analyze_model(self.model)
        self.assertEqual(info["dense_bytes"], 300)
        self.assertEqual(info["expert_bytes"], 120)
        self.assertEqual(info["expert_count"], 2)
        self.assertEqual(info["per_cap_bytes"], 60)

    def test_memory_available_is_positive(self):
        # Regression: on native Windows CPython, /proc/meminfo does not exist,
        # so the Linux-only path returned 0 and the expert cache was sized to
        # 0 slots/layer. The value must be a sane positive number of bytes.
        self.assertGreater(memory_available(), 0)

    def test_cpu_socket_count_is_positive(self):
        self.assertGreaterEqual(cpu_socket_count(), 1)

    # #379: read_ssd_probe reads the cached F_NOCACHE measurement colibri.c writes to
    # <model>/.coli_ssd on its first Metal+darwin startup. This is the read-only
    # Python-side half of the probe contract (S4) -- never re-measures, never
    # guesses, so every case here is pure file parsing. Trust mirrors the
    # engine: only a v2 cache recorded on THIS volume (matching st_dev) counts.
    def _write_v2_cache(self, gbs="14.322", dev=None):
        if dev is None:
            dev = os.stat(self.model).st_dev
        # byte-exact: text mode would CRLF-translate on Windows and the strict
        # reader would (correctly) reject the fixture as garbage
        (self.model / ".coli_ssd").write_bytes(f"v2 {gbs} {dev}\n".encode("ascii"))

    def test_ssd_probe_missing_file_returns_none(self):
        self.assertIsNone(read_ssd_probe(self.model))

    def test_ssd_probe_reads_cached_v2_value(self):
        self._write_v2_cache("14.322")
        self.assertEqual(read_ssd_probe(self.model), 14.322)

    def test_ssd_probe_foreign_volume_v2_returns_none(self):
        # A model dir rsync'd to another drive carries the OLD volume's
        # measurement; the engine re-probes it, so doctor/plan must not show it.
        self._write_v2_cache("14.322", dev=os.stat(self.model).st_dev + 1)
        self.assertIsNone(read_ssd_probe(self.model))

    def test_ssd_probe_legacy_bare_number_returns_none(self):
        # Pre-v2 caches were written before cold-range steering existed, so the
        # value may be page-cache contamination; the engine re-probes and
        # upgrades, and until then there is no number worth surfacing.
        (self.model / ".coli_ssd").write_bytes(b"14.322\n")
        self.assertIsNone(read_ssd_probe(self.model))

    def test_ssd_probe_unparsable_file_returns_none(self):
        (self.model / ".coli_ssd").write_bytes(b"not-a-number\n")
        self.assertIsNone(read_ssd_probe(self.model))

    def test_ssd_probe_empty_file_returns_none(self):
        (self.model / ".coli_ssd").write_bytes(b"")
        self.assertIsNone(read_ssd_probe(self.model))

    def test_ssd_probe_grammar_matches_c_reader_vectors(self):
        # THE parity pin (#386 fix round): parse_ssd_cache() must accept exactly
        # the grammar colibri.c's coli_ssd_cache_parse() accepts. Both suites
        # consume the same vector file; edit it and both sides re-judge.
        vectors = Path(__file__).parent / "fixtures" / "ssd_cache_vectors.txt"
        unescape = {"n": b"\n", "r": b"\r", "t": b"\t", "0": b"\x00",
                    "s": b" ", "\\": b"\\"}
        checked = 0
        for raw in vectors.read_text(encoding="utf-8").splitlines():
            if not raw or raw.startswith("#"):
                continue
            fields = raw.split("\t")
            payload = b""
            escaped = fields[-1] if len(fields) > 1 else ""
            i = 0
            while i < len(escaped):
                if escaped[i] == "\\" and i + 1 < len(escaped):
                    self.assertIn(escaped[i + 1], unescape, f"bad escape in: {raw!r}")
                    payload += unescape[escaped[i + 1]]
                    i += 2
                else:
                    payload += escaped[i].encode("utf-8")
                    i += 1
            kind, gbs, dev = parse_ssd_cache(payload)
            if fields[0] == "garbage":
                self.assertEqual((kind, gbs, dev), (None, None, None),
                                 f"garbage accepted: {payload!r} -> {(kind, gbs, dev)}")
            elif fields[0] == "legacy":
                self.assertEqual((kind, gbs, dev), ("legacy", float(fields[1]), None),
                                 f"legacy misread: {payload!r}")
            elif fields[0] == "v2":
                self.assertEqual((kind, gbs, dev), ("v2", float(fields[1]), int(fields[2])),
                                 f"v2 misread: {payload!r}")
            else:
                self.fail(f"unknown vector kind: {fields[0]}")
            checked += 1
        self.assertGreaterEqual(checked, 40, "vector file suspiciously short")

    def test_ssd_probe_state_classifies_every_case(self):
        # #386 r2, F10: doctor/plan wording keys off these states -- a cache
        # that exists but is not trusted must never read "no cached probe yet".
        self.assertEqual(ssd_probe_state(self.model), ("absent", None))
        self._write_v2_cache("14.322")
        self.assertEqual(ssd_probe_state(self.model), ("ok", 14.322))
        self._write_v2_cache("14.322", dev=os.stat(self.model).st_dev + 1)
        self.assertEqual(ssd_probe_state(self.model), ("foreign", None))
        (self.model / ".coli_ssd").write_bytes(b"14.322\n")
        self.assertEqual(ssd_probe_state(self.model), ("legacy", None))
        (self.model / ".coli_ssd").write_bytes(b"inf\n")
        self.assertEqual(ssd_probe_state(self.model), ("garbage", None))

    def test_ssd_probe_surfaces_in_plan_and_format(self):
        self._write_v2_cache("14.3")
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1)
        self.assertEqual(plan["ssd_probe_gbs"], 14.3)
        self.assertEqual(plan["ssd_probe_state"], "ok")
        self.assertIn("14.3 GB/s", format_plan(plan))

    def test_ssd_probe_pending_states_surface_in_format(self):
        (self.model / ".coli_ssd").write_bytes(b"14.3\n")   # legacy
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1)
        self.assertIsNone(plan["ssd_probe_gbs"])
        self.assertEqual(plan["ssd_probe_state"], "legacy")
        self.assertIn("legacy cache pending engine upgrade", format_plan(plan))

    def test_ssd_probe_absent_from_plan_and_format_when_not_cached(self):
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1)
        self.assertIsNone(plan["ssd_probe_gbs"])
        self.assertNotIn("F_NOCACHE", format_plan(plan))

    def test_builds_bounded_three_tier_plan(self):
        gpus = [{"index": 0, "name": "test-gpu", "total_bytes": 12 * GB,
                 "free_bytes": 10 * GB}]
        plan = build_plan(self.model, ram_gb=16, context=32, vram_gb=20,
                          available_memory=32 * GB, available_disk=100 * GB, gpus=gpus,
                          physical_cpus=24, cpu_sockets=2)
        self.assertEqual(plan["version"], 2)
        self.assertEqual(plan["policy"]["name"], "quality")
        self.assertEqual(plan["cpu"]["physical_cores"], 24)
        self.assertEqual(plan["cpu"]["sockets"], 2)
        self.assertTrue(plan["policy"]["preserve_quantization"])
        self.assertFalse(plan["tiers"]["vram"]["requires_host_backing"])
        self.assertEqual(plan["tiers"]["ram"]["budget_bytes"], 16 * GB)
        self.assertLessEqual(plan["tiers"]["vram"]["budget_bytes"], 8 * GB)
        self.assertIn("clamped", plan["warnings"][0])
        self.assertIn("0:test-gpu", format_plan(plan))

    def test_unified_memory_uses_one_shared_pool(self):
        gpus = [{"index": 0, "name": "NVIDIA GB10", "total_bytes": 130 * GB,
                 "free_bytes": 128 * GB, "unified_memory": True}]
        plan = build_plan(self.model, ram_gb=100, vram_gb=65,
                          available_memory=121 * GB, available_disk=1,
                          gpus=gpus, physical_cpus=8, cpu_sockets=1)
        ram = plan["tiers"]["ram"]["budget_bytes"]
        vram = plan["tiers"]["vram"]["budget_bytes"]
        self.assertTrue(plan["memory"]["unified"])
        self.assertLessEqual(ram + vram + plan["model"]["dense_bytes"], 121 * GB)
        self.assertTrue(any("share one physical memory" in warning
                            for warning in plan["warnings"]))

    def test_nvidia_unified_device_is_marked_from_name(self):
        output = "0, NVIDIA GB10, 130000, 120000\n"
        with mock.patch("resource_plan.subprocess.run",
                        return_value=subprocess.CompletedProcess(
                            args=[], returncode=0, stdout=output, stderr="")):
            from resource_plan import _discover_nvidia_gpus
            devices = _discover_nvidia_gpus()
        self.assertTrue(devices[0]["unified_memory"])

    def test_auto_tier_thread_count_uses_physical_cores(self):
        # End-to-end for #325: build_plan + environment_for_plan must export the
        # physical (not logical SMT) core count as OMP_NUM_THREADS. The original
        # suite passed physical_cpus=24 explicitly, so it never exercised the
        # real physical_cpu_count() probe whose single-core failure pinned decode.
        def lscpu(stdout):
            return subprocess.CompletedProcess(args=[], returncode=0,
                                               stdout=stdout, stderr="")
        # 1 socket, 12 cores, 2 SMT siblings -> 24 threads, 12 physical cores.

        # The parser must return 12 physical cores under BOTH lscpu layouts:
        #  - 2-col: `lscpu -p=core,socket` emits exactly [core,socket] (this is
        #           what the probe actually requests; the previous fields[1]/[2]
        #           indexing skipped every line here and fell through to the
        #           logical count -> the regression JustVugg caught).
        #  - 3-col: bare `lscpu -p` prepends a CPU column -> [cpu,core,socket].
        # Taking the last two fields is correct in both cases.
        layouts = {
            "2-col (-p=core,socket)": (
                "# core,socket\n" +
                "\n".join(f"{core},0" for core in range(12) for _ in range(2))),
            "3-col (bare -p, CPU prefix)": (
                "# CPU,Core,Socket\n" +
                "\n".join(f"{cpu},{core},0" for core in range(12) for cpu in range(2))),
        }
        for label, blob in layouts.items():
            with mock.patch("resource_plan.subprocess.run",
                            return_value=lscpu(blob)), \
                 mock.patch.object(sys, "platform", "linux"):
                plan = build_plan(self.model, available_memory=16 * GB,
                                  available_disk=1, gpus=[])
                env = environment_for_plan(plan)
            self.assertEqual(plan["cpu"]["physical_cores"], 12, label)
            self.assertEqual(env["OMP_NUM_THREADS"], "12", label)

    def test_plan_does_not_set_omp_affinity_vars(self):
        # The real #325 regression: --auto-tier set OMP_PROC_BIND=spread +
        # OMP_PLACES=cores, which ran before the engine's overwrite=0 setenv and
        # so won, collapsing the OpenMP team to one CPU on the reporter's 64-core
        # Linux box even though OMP_NUM_THREADS was correct. The plan must leave
        # affinity to the engine's own hot-thread tuning (which prefers 'close').
        plan = build_plan(self.model, available_memory=16 * GB,
                          available_disk=1, gpus=[], physical_cpus=64)
        env = environment_for_plan(plan)
        self.assertEqual(env["OMP_NUM_THREADS"], "64")
        self.assertNotIn("OMP_PROC_BIND", env)
        self.assertNotIn("OMP_PLACES", env)

    def test_plan_conserves_budget_and_experts_above_256gb(self):
        # Regression for #325's reporter: a 512 GB machine loading the whole
        # model into RAM. Verify the budget math stays exact at large RAM sizes
        # (no integer truncation, no over-allocation, no experts lost between
        # tiers). Checked at 256/512/800 GB to bracket the reporter's box.
        for ram_gb in (256, 512, 800):
            plan = build_plan(self.model, ram_gb=ram_gb, available_disk=1,
                              gpus=[], physical_cpus=64)
            ram = plan["tiers"]["ram"]
            # RAM budget never over-allocated: dense + runtime + cache <= budget.
            allocated = (ram["dense_bytes"] + ram["runtime_bytes"]
                         + ram["expert_cache_bytes"])
            self.assertLessEqual(allocated, ram["budget_bytes"],
                                 f"over-allocated RAM at {ram_gb} GB")
            # Every expert byte is accounted for exactly once across the tiers.
            tiers = plan["tiers"]
            tiered = (tiers["vram"]["hot_expert_bytes"]
                      + ram["warm_expert_bytes"]
                      + tiers["disk"]["cold_expert_bytes"])
            self.assertEqual(tiered, plan["model"]["expert_bytes"],
                             f"expert bytes lost/duplicated at {ram_gb} GB")
            # A positive RAM budget yields a non-negative cache and a sensible cap.
            self.assertGreaterEqual(ram["expert_cache_bytes"], 0)
            self.assertGreaterEqual(ram["cache_slots_per_layer"], 0)

    def test_filters_requested_devices(self):
        gpus = [{"index": 0, "name": "a", "total_bytes": 8 * GB, "free_bytes": 8 * GB}]
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1,
                          gpus=gpus, gpu_indices=[1])
        self.assertEqual(plan["tiers"]["vram"]["devices"], [])
        self.assertIn("not detected", plan["warnings"][0])

    def test_cli_emits_versioned_json(self):
        cli = Path(__file__).parents[1] / "coli"
        run = subprocess.run([
            sys.executable, str(cli), "plan", "--model", str(self.model),
            "--gpu", "none", "--json",
        ], text=True, capture_output=True, check=True)
        plan = json.loads(run.stdout)
        self.assertEqual(plan["version"], 2)
        self.assertEqual(plan["model"]["expert_count"], 2)

    def test_applies_plan_without_overriding_explicit_settings(self):
        gpus = [
            {"index": 0, "name": "a", "total_bytes": 12 * GB, "free_bytes": 10 * GB},
            {"index": 1, "name": "b", "total_bytes": 12 * GB, "free_bytes": 10 * GB},
        ]
        plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                          available_disk=1, gpus=gpus, cpu_sockets=2)
        env = environment_for_plan(plan, {"RAM_GB": "12", "PIN": "stats.txt",
                                               "COLI_GPUS": "1"})
        self.assertEqual(env["RAM_GB"], "12")
        self.assertEqual(env["COLI_CUDA"], "1")
        self.assertEqual(env["COLI_GPUS"], "1")
        self.assertEqual(env["OMP_NUM_THREADS"], str(plan["cpu"]["physical_cores"]))
        # The plan must NOT set OMP_PROC_BIND / OMP_PLACES on any platform:
        # the engine's own hot-thread tuning owns affinity (it prefers
        # OMP_PROC_BIND=close for the back-to-back per-expert matmuls). Setting
        # spread + cores here ran before the engine's overwrite=0 setenv and so
        # won, collapsing the team to one CPU on some libgomp topologies (#325).
        self.assertNotIn("OMP_PROC_BIND", env)
        self.assertNotIn("OMP_PLACES", env)
        self.assertEqual(env["PIN_GB"], env["CUDA_EXPERT_GB"])

        explicit_threads = environment_for_plan(plan, {"OMP_NUM_THREADS": "7",
                                                        "OMP_PROC_BIND": "close"})
        self.assertEqual(explicit_threads["OMP_NUM_THREADS"], "7")
        self.assertEqual(explicit_threads["OMP_PROC_BIND"], "close")

        if sys.platform.startswith("linux"):
            self.assertEqual(env["COLI_NUMA"], "1")
            explicit_numa = environment_for_plan(plan, {"COLI_NUMA": "0"})
            self.assertEqual(explicit_numa["COLI_NUMA"], "0")

    def test_single_socket_plan_does_not_enable_numa(self):
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1,
                          gpus=[], physical_cpus=8, cpu_sockets=1)
        self.assertNotIn("COLI_NUMA", environment_for_plan(plan))

    def test_auto_tune_mtp_off_when_compute_bound(self):
        # Tiny model with 64 GB RAM and no GPU: all experts fit in RAM with no
        # warm tier, so the plan classifies as compute-bound.
        plan = build_plan(self.model, ram_gb=64, available_memory=64 * GB,
                          available_disk=100 * GB, gpus=[], physical_cpus=24,
                          cpu_sockets=2)
        # With such a small model fully in RAM and no GPU, bottleneck is compute
        self.assertEqual(plan["bottleneck_class"], "compute")
        self.assertIn("DRAFT", plan["tune"])
        self.assertEqual(plan["tune"]["DRAFT"]["value"], "0")
        env = environment_for_plan(plan)
        self.assertEqual(env["DRAFT"], "0")
        explicit = environment_for_plan(plan, {"DRAFT": "3"})
        self.assertEqual(explicit["DRAFT"], "3")

    def test_auto_tune_mtp_off_when_disk_low_hit(self):
        # Use a model large enough that 8 GB RAM can't hold all experts.
        big = tempfile.TemporaryDirectory()
        bigmodel = Path(big.name)
        (bigmodel / "config.json").write_text(json.dumps({
            "num_hidden_layers": 2, "n_routed_experts": 4,
            "kv_lora_rank": 4, "qk_rope_head_dim": 2,
            "qk_nope_head_dim": 3, "v_head_dim": 5, "num_attention_heads": 2,
        }))
        expert_size = 3 * GB  # each expert 3 GB → 12 GB total, won't fit in 8 GB budget
        write_shard(bigmodel / "out-00000.safetensors", [
            ("model.embed_tokens.weight", 100),
            ("model.layers.0.self_attn.q_a_proj.weight", 200),
        ])
        for i in range(4):
            write_shard(bigmodel / f"out-{i+1:05d}.safetensors", [
                (f"model.layers.1.mlp.experts.{i}.gate_proj.weight", expert_size),
            ])
        plan = build_plan(bigmodel, ram_gb=0, available_memory=4 * GB,
                          available_disk=100 * GB, gpus=[], physical_cpus=8,
                          cpu_sockets=1)
        big.cleanup()
        self.assertEqual(plan["bottleneck_class"], "disk")
        self.assertLess(plan["projected_hit_rate"], 0.90)
        self.assertEqual(plan["tune"]["DRAFT"]["value"], "0")

    def test_auto_tune_pipe_multi_gpu(self):
        gpus = [
            {"index": 0, "name": "a", "total_bytes": 32 * GB, "free_bytes": 30 * GB},
            {"index": 1, "name": "b", "total_bytes": 32 * GB, "free_bytes": 30 * GB},
        ]
        plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                          available_disk=1, gpus=gpus, cpu_sockets=2)
        self.assertEqual(plan["tune"]["COLI_CUDA_PIPE"]["value"], "2")
        env = environment_for_plan(plan)
        self.assertEqual(env["COLI_CUDA_PIPE"], "2")

    def test_auto_tune_pipe_single_gpu(self):
        gpus = [{"index": 0, "name": "a", "total_bytes": 12 * GB, "free_bytes": 10 * GB}]
        plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                          available_disk=1, gpus=gpus, cpu_sockets=1)
        self.assertEqual(plan["tune"]["COLI_CUDA_PIPE"]["value"], "1")

    def test_auto_tune_numa_hint_for_cpu_only(self):
        plan = build_plan(self.model, ram_gb=64, available_memory=64 * GB,
                          available_disk=1, gpus=[], physical_cpus=64, cpu_sockets=2)
        self.assertIn("_numa_hint", plan["tune"])
        self.assertIn("numactl", plan["tune"]["_numa_hint"])
        self.assertIn("auto-tune", format_plan(plan))

    def test_format_plan_shows_tune_and_hit_rate(self):
        plan = build_plan(self.model, ram_gb=64, available_memory=64 * GB,
                          available_disk=100 * GB, gpus=[], physical_cpus=24,
                          cpu_sockets=1)
        text = format_plan(plan)
        self.assertIn("hit", text)
        self.assertIn("auto-tune", text)
        self.assertIn("DRAFT", text)

    def test_cpu_binary_does_not_apply_gpu_tier(self):
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1,
                          gpus=[{"index": 0, "name": "a", "total_bytes": 8 * GB,
                                 "free_bytes": 8 * GB}])
        env = environment_for_plan(plan, cuda_enabled=False)
        self.assertIn("RAM_GB", env)
        self.assertNotIn("COLI_CUDA", env)
        disabled = environment_for_plan(plan, {"COLI_CUDA": "0"}, cuda_enabled=True)
        self.assertNotIn("COLI_GPU", disabled)
        self.assertNotIn("CUDA_EXPERT_GB", disabled)

    def test_rejects_unknown_policy_and_marks_experimental_policy(self):
        with self.assertRaisesRegex(ValueError, "unknown policy"):
            build_plan(self.model, available_memory=16 * GB, available_disk=1,
                       gpus=[], policy="fast-ish")
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1,
                          gpus=[], policy="experimental-fast")
        self.assertFalse(plan["policy"]["quality_preserving"])
        self.assertFalse(plan["policy"]["preserve_router"])

    def test_balanced_policy_enables_lossless_live_repin(self):
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1,
                          gpus=[], policy="balanced")
        env = environment_for_plan(plan)
        self.assertEqual(env["COLI_POLICY"], "balanced")
        self.assertEqual(env["REPIN"], "64")
        explicit = environment_for_plan(plan, {"REPIN": "0"})
        self.assertEqual(explicit["REPIN"], "0")

    def test_plan_explains_hot_warm_and_cold_placement(self):
        plan = build_plan(self.model, ram_gb=4, vram_gb=0,
                          available_memory=4 * GB, available_disk=1, gpus=[])
        self.assertEqual([item["target"] for item in plan["decisions"]],
                         ["VRAM", "RAM", "Disk"])
        self.assertIn("quality-preserving yes", format_plan(plan))
        self.assertIn("expected_bottleneck", plan)


class PhysicalCpuCountTest(unittest.TestCase):
    """Regression for #325: --auto-tier pinned decode to one core because
    physical_cpu_count() silently returned 1.

    Two root causes this locks down:
      1. lscpu -p prepends a CPU column, so `-p=core,socket` emits
         CPU,Core,Socket; counting rows counted logical SMT siblings.
      2. any probe failure fell through to ``os.cpu_count() or 1`` and the
         ``or 1`` could pin a constrained/cgroup'd box to a single core.
    """

    def _lscpu(self, stdout):
        return subprocess.CompletedProcess(args=[], returncode=0,
                                           stdout=stdout, stderr="")

    def _lscpu_topology(self, sockets, cores_per_socket, threads_per_core):
        # Real lscpu shape: socket-local core IDs repeat across sockets; the
        # CPU column (always prepended) is a unique logical-CPU index.
        rows, cpu = [], 0
        for sock in range(sockets):
            for core in range(cores_per_socket):
                for _ in range(threads_per_core):
                    rows.append(f"{cpu},{core},{sock}")
                    cpu += 1
        return "# CPU,Core,Socket\n" + "\n".join(rows)

    def test_counts_physical_cores_not_smt_threads(self):
        blob = self._lscpu_topology(sockets=2, cores_per_socket=16, threads_per_core=2)
        with mock.patch("resource_plan.subprocess.run", return_value=self._lscpu(blob)), \
             mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(physical_cpu_count(), 32)

    def test_single_socket_no_smt(self):
        blob = self._lscpu_topology(sockets=1, cores_per_socket=8, threads_per_core=1)
        with mock.patch("resource_plan.subprocess.run", return_value=self._lscpu(blob)), \
             mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(physical_cpu_count(), 8)

    def test_skips_offline_core_socket_fields(self):
        # VMs / large NUMA boxes emit "-" for offline core or socket IDs; that
        # used to raise ValueError, discard the whole parse, and fall through
        # to the single-core fallback.
        blob = "# CPU,Core,Socket\n0,0,0\n1,-,0\n2,1,0\n3,1,0\n"
        with mock.patch("resource_plan.subprocess.run", return_value=self._lscpu(blob)), \
             mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(physical_cpu_count(), 2)

    def test_lscpu_missing_falls_back_to_logical_not_silent_one(self):
        # The bug: lscpu absent -> os.cpu_count() or 1. On a constrained box
        # os.cpu_count() can be 1. We still must never silently pick 1 without
        # a warning, and when logical cores exist they must be used.
        import os
        with mock.patch("resource_plan.subprocess.run", side_effect=FileNotFoundError), \
             mock.patch.object(sys, "platform", "linux"), \
             mock.patch("resource_plan.os.cpu_count", return_value=16), \
             mock.patch("sys.stderr"):
            self.assertEqual(physical_cpu_count(), 16)

    def test_zero_logical_cores_warns_and_returns_one(self):
        # The genuine degenerate case: no probe works and os.cpu_count() is
        # None/1. Must return 1 (engine needs a positive team size) but warn.
        with mock.patch("resource_plan.subprocess.run", side_effect=FileNotFoundError), \
             mock.patch.object(sys, "platform", "linux"), \
             mock.patch("resource_plan.os.cpu_count", return_value=None), \
             mock.patch("sys.stderr"):
            self.assertEqual(physical_cpu_count(), 1)

    def test_apple_silicon_prefers_performance_cores(self):
        def sysctl(command, **kwargs):
            value = "8\n" if command[-1] == "hw.perflevel0.logicalcpu" else "10\n"
            return subprocess.CompletedProcess(args=command, returncode=0,
                                               stdout=value, stderr="")
        with mock.patch("resource_plan.subprocess.run", side_effect=sysctl), \
             mock.patch.object(sys, "platform", "darwin"):
            self.assertEqual(physical_cpu_count(), 8)


if __name__ == "__main__":
    unittest.main()
