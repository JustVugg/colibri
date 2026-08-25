"""machine_info() must report the machine's real RAM on win32 (#1042).

ram_gb sizes the eviction write in evict_cache(): a hardcoded 8.0 on a
128 GB box writes 9 GB, evicts nothing, and the run labelled "cold" is
measured warm and published as cold. These tests run on any host by
mocking sys.platform and the two win32 probes.
"""
import ctypes
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from family_registry import FAMILIES
from tools import datapoint

GB = 1073741824


class MachineInfoWin32Test(unittest.TestCase):
    def _win32_info(self, memstatus_ok, total_phys=128 * GB):
        def fake_memstatus(argp):
            if not memstatus_ok:
                return 0
            argp._obj.ullTotalPhys = total_phys
            return 1

        windll = mock.MagicMock()
        windll.kernel32.GlobalMemoryStatusEx.side_effect = fake_memstatus
        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch.object(ctypes, "windll", windll, create=True):
            return datapoint.machine_info()

    def test_win32_reports_real_ram(self):
        info = self._win32_info(memstatus_ok=True)
        self.assertAlmostEqual(info["ram_gb"], 128.0)
        self.assertEqual(info["ram"], "128 GB")
        self.assertTrue(info["os"].startswith("Windows"))

    def test_win32_build_number_disambiguates_windows_11(self):
        # platform.release() says "10" on Windows 11; build >= 22000 is the tell
        import platform as plat
        from unittest import mock as m2
        with m2.patch.object(plat, "release", return_value="10"), \
             m2.patch.object(plat, "version", return_value="10.0.26200"):
            info = self._win32_info(memstatus_ok=True)
        self.assertTrue(info["os"].startswith("Windows 11"), info["os"])
        with m2.patch.object(plat, "release", return_value="10"), \
             m2.patch.object(plat, "version", return_value="10.0.19045"):
            info = self._win32_info(memstatus_ok=True)
        self.assertTrue(info["os"].startswith("Windows 10"), info["os"])

    def test_win32_probe_failure_falls_back(self):
        """A failed probe keeps the old conservative default rather than 0.0 —
        an eviction write sized from 0 GB would silently evict nothing."""
        info = self._win32_info(memstatus_ok=False)
        self.assertEqual(info["ram_gb"], 8.0)
        self.assertEqual(info["ram"], "?")

    def test_unknown_platform_keeps_fallback(self):
        with mock.patch.object(sys, "platform", "freebsd14"):
            info = datapoint.machine_info()
        self.assertEqual(info["ram_gb"], 8.0)
        self.assertEqual(info["ram"], "?")


class PersistentDatapointTest(unittest.TestCase):
    def setUp(self):
        self.family = SimpleNamespace(
            id="deepseek_v4",
            display_name="DeepSeek V4",
            has_gateway_adapter=True,
            build_target="deepseek_v4",
        )
        self.instances = []

    def _runtime(self, engine_type=None):
        instances = self.instances

        class FakeEngine:
            def __init__(self, executable, model, cap, max_tokens, env, kv_slots,
                         family):
                self.executable = executable
                self.model = model
                self.cap = cap
                self.max_tokens = max_tokens
                self.env = env
                self.kv_slots = kv_slots
                self.family = family
                self.calls = []
                self.closed = False
                self.tiers = {"vram": 2, "ram": 3, "disk": 4,
                              "vram_gb": 1.5, "ram_gb": 2.5}
                self.hwinfo = {"cores": 16}
                self.profile = []
                self.profile_seq = 0
                instances.append(self)

            def generate(self, prompt, max_tokens, temperature, top_p, on_text,
                         cache_slot=0):
                self.calls.append((prompt, max_tokens, temperature, top_p, cache_slot))
                on_text("token")
                self.profile.append({"expert_disk_s": 0.1,
                                     "expert_wait_s": 0.2,
                                     "expert_matmul_s": 0.3,
                                     "attention_s": 0.4,
                                     "lm_head_s": 0.5})
                self.profile_seq += 1
                return {"completion_tokens": max_tokens,
                        "tokens_per_second": float(len(self.calls)),
                        "cache_hit_percent": float(len(self.calls) * 10),
                        "rss_gb": 12.5,
                        "prompt_tokens": 17,
                        "length_limited": True}

            def close(self):
                self.closed = True

        return SimpleNamespace(
            ARCH=None,
            Engine=engine_type or FakeEngine,
            resolve_model=lambda _snap: SimpleNamespace(descriptor=self.family),
            default_engine=lambda _family: "/unused/default-engine",
            render_chat_for_arch=lambda messages, enable_thinking: (
                f"rendered:{messages[0]['content']}:{enable_thinking}"),
        )

    def test_default_campaign_uses_rotating_prompts_as_primary_workload(self):
        runtime = self._runtime()
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "deepseek_v4"
            executable.touch()
            with mock.patch.dict(os.environ, {}, clear=True):
                campaign = datapoint.run_persistent_engine(
                    str(executable), "/model", "hello", 8, 0, 1, 16,
                    memory_gb=64, runtime=runtime, physical_cores=8)

        self.assertEqual(len(self.instances), 1)
        engine = self.instances[0]
        self.assertEqual(len(engine.calls), 6)  # cold + identical + 4 rotating
        self.assertTrue(all(call[0] == "rendered:hello:False"
                            for call in engine.calls[:2]))
        self.assertEqual(
            [call[0] for call in engine.calls[2:]],
            [f"rendered:{prompt}:False"
             for prompt in datapoint.DEFAULT_ROTATING_PROMPTS],
        )
        self.assertTrue(all(call[4] == 0 for call in engine.calls))
        self.assertTrue(engine.closed)
        self.assertEqual(engine.env["RAM_GB"], "64")
        self.assertEqual(engine.env["COLI_TEMP"], "0")
        self.assertNotIn("OMP_NUM_THREADS", engine.env)  # V4 owns its thread split
        self.assertEqual(runtime.ARCH, "deepseek_v4")
        self.assertEqual(len(campaign["cold"]), 1)
        self.assertEqual(len(campaign["warmup"]), 0)
        self.assertEqual(len(campaign["warm"]), 1)
        self.assertEqual(len(campaign["rotating"]), 4)
        self.assertEqual([row["tok_s"] for row in campaign["warm"]], [2.0])
        self.assertEqual([row["tok_s"] for row in campaign["rotating"]],
                         [3.0, 4.0, 5.0, 6.0])
        self.assertEqual(campaign["warm"][0]["tokens"], 8)
        self.assertEqual(campaign["warm"][0]["prompt_tokens"], 17)
        self.assertEqual(campaign["rotating_prompt_count"], 4)
        self.assertEqual(campaign["rotating_prompt_source"], "built-in")

    def test_teardown_failure_does_not_discard_collected_results(self):
        # At --memory-gb 126 the engine can outlast the close() grace period
        # while it frees a large resident cache; the raised TimeoutExpired used
        # to swallow the return and lose a complete run (#1154). A teardown that
        # fails must still yield the measurements already gathered.
        base = self._runtime()

        class RaisingCloseEngine(base.Engine):
            def close(self):
                self.closed = True
                raise subprocess.TimeoutExpired(cmd="engine", timeout=5)

        runtime = self._runtime(engine_type=RaisingCloseEngine)
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "deepseek_v4"
            executable.touch()
            with mock.patch.dict(os.environ, {}, clear=True):
                campaign = datapoint.run_persistent_engine(
                    str(executable), "/model", "hello", 8, 0, 1, 16,
                    memory_gb=126, runtime=runtime, physical_cores=8)

        self.assertEqual(len(self.instances), 1)
        self.assertTrue(self.instances[0].closed)
        self.assertEqual(len(campaign["cold"]), 1)
        self.assertEqual(len(campaign["warm"]), 1)
        self.assertEqual([row["tok_s"] for row in campaign["rotating"]],
                         [3.0, 4.0, 5.0, 6.0])

    def test_engine_is_closed_when_a_request_fails(self):
        instances = self.instances

        class FailingEngine:
            def __init__(self, *_args, **_kwargs):
                self.profile_seq = 0
                self.profile = []
                self.closed = False
                instances.append(self)

            def generate(self, *_args, **_kwargs):
                raise RuntimeError("generation failed")

            def close(self):
                self.closed = True

        runtime = self._runtime(FailingEngine)
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "deepseek_v4"
            executable.touch()
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                datapoint.run_persistent_engine(
                    str(executable), "/model", "hello", 8, 1, 3, 16,
                    runtime=runtime, physical_cores=8)
        self.assertTrue(self.instances[0].closed)

    def test_trailing_profile_after_done_is_attached_to_the_request(self):
        class DelayedProfileEngine:
            profile_seq = 0
            profile = []

            def generate(self, _prompt, max_tokens, _temperature, _top_p,
                         on_text, cache_slot=0):
                self.asserted_slot = cache_slot
                on_text("token")

                def emit_profile():
                    self.profile = [{"expert_disk_s": 1.25}]
                    self.profile_seq += 1

                self.timer = threading.Timer(0.01, emit_profile)
                self.timer.start()
                return {"completion_tokens": max_tokens,
                        "tokens_per_second": 2.0,
                        "cache_hit_percent": 50.0,
                        "rss_gb": 4.0,
                        "prompt_tokens": 3,
                        "length_limited": True}

        engine = DelayedProfileEngine()
        result = datapoint._measure_persistent_request(engine, "prompt", 8)
        engine.timer.join()
        self.assertEqual(engine.asserted_slot, 0)
        self.assertEqual(result["profile"]["expert_disk_s"], 1.25)

    def test_sister_engines_use_physical_cores_without_overriding_user_value(self):
        family = SimpleNamespace(id="qwen36")
        with mock.patch.dict(os.environ, {}, clear=True):
            env = datapoint._persistent_environment(family, 128, 32, 16)
            self.assertEqual(env["OMP_NUM_THREADS"], "16")
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "6"}, clear=True):
            env = datapoint._persistent_environment(family, 128, 32, 16)
            self.assertEqual(env["OMP_NUM_THREADS"], "6")

    def test_persistent_mode_rejects_the_cli_wrapper(self):
        with self.assertRaisesRegex(ValueError, "engine binary"):
            datapoint._persistent_executable(None, self.family, "./coli")

    def test_parser_defaults_to_persistent_campaign(self):
        args = datapoint.build_parser().parse_args(["--snap", "/model"])
        self.assertEqual(args.mode, "persistent")
        self.assertEqual(args.warmup_runs, 0)
        self.assertEqual(args.warm_runs, 1)
        self.assertEqual(args.rotating_runs, 4)
        self.assertIsNone(args.rotating_prompts)
        self.assertIsNone(args.engine)

    def test_custom_rotating_suite_requires_two_distinct_nonempty_prompts(self):
        self.assertEqual(datapoint._rotating_prompt_suite(["one", "two"]),
                         ("one", "two"))
        with self.assertRaisesRegex(ValueError, "two distinct"):
            datapoint._rotating_prompt_suite(["same", "same"])
        with self.assertRaisesRegex(ValueError, "two distinct"):
            datapoint._rotating_prompt_suite([])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            datapoint._rotating_prompt_suite(["one", " "])

    def test_report_makes_rotating_primary_and_identical_an_upper_bound(self):
        args = SimpleNamespace(snap="/model", cap=16, memory_gb=64,
                               max_new=8, shard=None)
        row = {"prompt_tokens": 4, "tokens": 8, "request_s": 2.0,
               "ttft_s": 0.5, "tok_s": 4.0, "hit": 60.0,
               "rss": 12.0, "profile": {"expert_disk_s": 0.25}}
        campaign = {
            "mode": "persistent", "family_name": "DeepSeek V4",
            "engine": "/tmp/deepseek_v4", "load_s": 1.0,
            "cold": [row], "warmup": [], "warm": [dict(row, tok_s=9.0)],
            "rotating": [dict(row, tok_s=value, request_s=10.0 - value)
                         for value in (3.0, 4.0, 5.0, 6.0)],
            "rotating_prompt_count": 4, "rotating_prompt_source": "built-in",
            "tiers": None, "hwinfo": None, "omp_threads": "8",
            "loader_lanes": "engine default",
        }
        info = {"cpu": "test CPU", "ram": "64 GB", "ram_gb": 64,
                "cores": 16, "physical_cores": 8, "os": "test OS"}
        report = datapoint.format_report(
            info, args, campaign, "cold request", {})
        rotating = report.index("| **rotating prompts (primary)** |")
        identical = report.index("| warm-identical (upper bound) |")
        self.assertLess(rotating, identical)
        self.assertIn("| **rotating prompts (primary)** | 4 | 4.50 |", report)
        self.assertIn("p95 request s", report)
        self.assertIn("4 built-in prompts, fixed rotation", report)

    def test_every_registered_family_has_the_shared_persistent_adapter(self):
        expected = {"glm", "inkling", "kimi", "olmoe", "qwen36", "deepseek_v4"}
        self.assertTrue(expected.issubset({family.id for family in FAMILIES}))
        self.assertTrue(all(family.has_gateway_adapter for family in FAMILIES))


class EvictCacheSpaceTest(unittest.TestCase):
    """The portable fallback must not fill the temp volume.

    It is the only eviction route on Windows, and since #1042 it sizes the write
    from the machine's real RAM rather than a hardcoded 8 GB. A 128 GB box whose
    temp dir is on a drive with 124 GB free would write until that volume hit
    zero and only then raise OSError.
    """

    def _run(self, ram_gb, free_gb, tmp="/tmp"):
        usage = SimpleNamespace(free=int(free_gb * GB))
        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch.object(datapoint.tempfile, "gettempdir", return_value=tmp), \
             mock.patch.object(datapoint.shutil, "disk_usage", return_value=usage), \
             mock.patch.object(datapoint.tempfile, "NamedTemporaryFile") as ntf:
            return datapoint.evict_cache(ram_gb), ntf

    def test_refuses_when_the_temp_volume_is_too_small(self):
        result, ntf = self._run(ram_gb=128, free_gb=124)
        self.assertFalse(result)
        ntf.assert_not_called()          # nothing written, not even partially

    def test_writes_when_there_is_room(self):
        result, ntf = self._run(ram_gb=8, free_gb=500)
        self.assertTrue(ntf.called)
        self.assertTrue(result)

    def test_unreadable_temp_volume_does_not_block_the_write(self):
        """A failed disk_usage must not become a refusal. The pre-existing
        OSError handler around the write is still the backstop."""
        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch.object(datapoint.shutil, "disk_usage",
                               side_effect=OSError("no stat")), \
             mock.patch.object(datapoint.tempfile, "NamedTemporaryFile") as ntf:
            self.assertTrue(datapoint.evict_cache(1))
            self.assertTrue(ntf.called)


if __name__ == "__main__":
    unittest.main()
