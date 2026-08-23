import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotune import (
    apply_profile,
    candidate_steps,
    load_profile,
    machine_fingerprint,
    parse_replay,
    run_tune,
    save_profile,
)


def plan(cores=8, sockets=1, gpu=True, cold=0):
    devices = [{"name": "Test GPU", "total_bytes": 24_000_000_000}] if gpu else []
    return {
        "version": 2,
        "cpu": {"physical_cores": cores, "sockets": sockets},
        "tiers": {
            "disk": {"cold_expert_bytes": cold},
            "vram": {"devices": devices},
        },
    }


class AutotuneUnitTest(unittest.TestCase):
    def test_candidates_are_bounded_and_quality_preserving(self):
        steps = candidate_steps(plan(sockets=2, cold=10), {
            "OMP_NUM_THREADS": "8", "COLI_CUDA_PIPE": "0",
            "DIRECT": "0", "PIPE": "0",
        })
        names = [name for name, _ in steps]
        self.assertEqual(names, [
            "omp-4", "numa-on", "cuda-pipe-1", "cuda-pipe-2",
            "cuda-sync", "direct-on", "io-pipe-on",
        ])
        keys = {key for _, values in steps for key in values}
        self.assertNotIn("TOPK", keys)
        self.assertNotIn("DRAFT", keys)

    def test_parse_replay(self):
        result = parse_replay(
            "REPLAY decode: 16 tokens in 1.000s | 16.00 tok/s | expert hit 92.5%\n"
            "[PROF] decode forwards: 16 | latency p50 50.0 ms | p90 70.0 ms | "
            "p99 80.0 ms | max 90.0 ms"
        )
        self.assertEqual(result["tok_s"], 16.0)
        self.assertEqual(result["hit_pct"], 92.5)
        self.assertEqual(result["p99_ms"], 80.0)

    def test_parse_replay_uses_elapsed_time_below_point_one_tok_s(self):
        result = parse_replay(
            "REPLAY decode: 16 tokens in 355.556s | 0.04 tok/s | expert hit 34.4%\n"
            "[PROF] decode forwards: 16 | latency p50 4000.0 ms | p90 5000.0 ms | "
            "p99 60000.0 ms | max 70000.0 ms"
        )
        self.assertAlmostEqual(result["tok_s"], 16 / 355.556)
        self.assertNotEqual(result["tok_s"], 0.04)

    def test_parse_replay_accepts_legacy_speed_without_elapsed_time(self):
        result = parse_replay(
            "REPLAY decode: 16 tokens | 0.04 tok/s | expert hit 34.4%\n"
            "[PROF] decode forwards: 16 | latency p50 4000.0 ms | p90 5000.0 ms | "
            "p99 60000.0 ms | max 70000.0 ms"
        )
        self.assertEqual(result["tok_s"], 0.04)

    def test_parse_replay_falls_back_when_elapsed_time_rounds_to_zero(self):
        result = parse_replay(
            "REPLAY decode: 1 tokens in 0.000s | 1234.56 tok/s"
        )
        self.assertEqual(result["tok_s"], 1234.56)

    def test_accepts_the_common_TUNE_line_from_the_sibling_engines(self):
        """#898: only colibri emitted a parseable throughput line, so the tuner
        was GLM-only. The four sibling engines now print `TUNE decode: N tokens
        in T.TTTs` at the end of an ordinary greedy run."""
        result = parse_replay("TUNE decode: 16 tokens in 8.000s\n")
        self.assertAlmostEqual(result["tok_s"], 2.0)

    def test_refuses_output_with_neither_line(self):
        """An engine that printed nothing parseable used to raise 'did not emit
        REPLAY throughput'. It must still refuse rather than score it as zero."""
        with self.assertRaises(ValueError):
            parse_replay("no decode line here\n")

    def test_glm_only_knobs_are_not_swept_on_other_engines(self):
        """PIPE and DIRECT are read by colibri alone -- 3 and 1 occurrences in
        colibri.c, 0 in each of the other four. Sweeping them elsewhere spends
        the replay budget proving two identical runs are identical, then reports
        the 0% difference as a measurement."""
        plan = {"cpu": {"physical_cores": 8, "sockets": 1},
                "tiers": {"disk": {"cold_expert_bytes": 1 << 30}, "vram": {"devices": []}}}
        glm = dict(candidate_steps(plan, {}, "glm"))
        self.assertIn("direct-on", glm)
        self.assertIn("io-pipe-on", glm)
        for arch in ("deepseek_v4", "kimi", "inkling", "olmoe"):
            with self.subTest(arch=arch):
                steps = dict(candidate_steps(plan, {}, arch))
                self.assertNotIn("direct-on", steps)
                self.assertNotIn("io-pipe-on", steps)
                # OMP and NUMA are engine-agnostic and must survive, or the
                # sweep has nothing left to measure on those engines.
                self.assertTrue(any(name.startswith("omp-") for name in steps), steps)

    def test_profile_round_trip_and_explicit_environment_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory) / "engine"
            engine.write_bytes(b"engine")
            p = plan()
            fingerprint = machine_fingerprint(p, directory, str(engine))
            profile = {
                "schema_version": 1,
                "fingerprint": fingerprint,
                "accepted": True,
                "gain": 0.10,
                "winner": {"env": {"COLI_CUDA_PIPE": "2"}},
            }
            save_profile(profile, directory)
            loaded = load_profile(p, directory, str(engine), directory)
            self.assertEqual(loaded["fingerprint"], fingerprint)
            applied = apply_profile(
                {"COLI_CUDA_PIPE": "0"}, loaded, {"COLI_CUDA_PIPE"}
            )
            self.assertEqual(applied["COLI_CUDA_PIPE"], "0")
            applied = apply_profile({}, loaded)
            self.assertEqual(applied["COLI_CUDA_PIPE"], "2")

    def test_profile_rejects_unknown_knobs(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory) / "engine"
            engine.write_bytes(b"engine")
            p = plan()
            fingerprint = machine_fingerprint(p, directory, str(engine))
            save_profile({
                "schema_version": 1, "fingerprint": fingerprint, "accepted": True,
                "winner": {"env": {"TOPK": "1"}},
            }, directory)
            self.assertIsNone(load_profile(p, directory, str(engine), directory))


class FakeServeEngine:
    """Serve-protocol stand-in for the sibling engines: deterministic greedy
    text, speed keyed off the universal knobs the sweep offers."""

    launches = []
    drift_on = None            # knob name that changes the OUTPUT (illegal)
    drift_baseline = False     # second run of ANY env differs (nondeterminism)
    calls = 0

    def __init__(self, executable, model, cap, max_tokens, env, kv_slots,
                 family):
        FakeServeEngine.launches.append(
            {"cap": cap, "env": dict(env), "family": family})
        self.env = env

    def generate(self, prompt, max_tokens, temperature, top_p, on_text,
                 cache_slot=0):
        FakeServeEngine.calls += 1
        text = "hello world"
        if FakeServeEngine.drift_on and \
                self.env.get(FakeServeEngine.drift_on) is not None:
            text = "HELLO DRIFT"
        if FakeServeEngine.drift_baseline and FakeServeEngine.calls > 1:
            text = f"drift {FakeServeEngine.calls}"
        on_text(text)
        threads = self.env.get("OMP_NUM_THREADS")
        speed = {None: 10.0, "8": 13.0, "4": 11.0}.get(threads, 10.0)
        return {"completion_tokens": max_tokens, "tokens_per_second": speed,
                "cache_hit_percent": 95.0, "prompt_tokens": 3}

    def close(self):
        pass

    @classmethod
    def reset(cls, drift_on=None, drift_baseline=False):
        cls.launches, cls.calls = [], 0
        cls.drift_on, cls.drift_baseline = drift_on, drift_baseline


class ServeTuneTest(unittest.TestCase):
    """The sibling engines have no replay protocol (#1191): their measurement
    runs go through the serve protocol every engine already speaks."""

    def run_serve_tune(self, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            p = plan(cores=8, gpu=False)
            p["tiers"]["ram"] = {"cache_slots_per_layer": 16}
            arguments = dict(
                engine="/unused-binary", cap=None, base_env={}, plan=p,
                model=directory, prompt="prompt", arch="deepseek_v4",
                tokens=4, repeats=2, timeout=5, min_gain=0.03,
                profile_dir=directory, family="fake-family",
                engine_cls=FakeServeEngine)
            arguments.update(overrides)
            return run_tune(**arguments)

    def test_sibling_arch_tunes_through_the_serve_protocol(self):
        FakeServeEngine.reset()
        profile, _ = self.run_serve_tune()
        self.assertTrue(profile["accepted"])
        # cores=8 with OMP unset means only omp-4 is offered (8 is already
        # the implicit default), so the sweep's one candidate must win
        self.assertEqual(profile["winner"]["env"], {"OMP_NUM_THREADS": "4"})
        self.assertAlmostEqual(profile["gain"], 0.10)
        # the resolved plan cap and the family reach every engine launch
        for launch in FakeServeEngine.launches:
            self.assertEqual(launch["cap"], 16)
            self.assertEqual(launch["family"], "fake-family")
            self.assertEqual(launch["env"].get("COLI_TEMP"), "0")

    def test_candidate_that_changes_output_is_disqualified(self):
        FakeServeEngine.reset(drift_on="OMP_NUM_THREADS")
        profile, _ = self.run_serve_tune()
        # every OMP candidate drifts, so nothing may beat the baseline
        self.assertFalse(profile["accepted"])
        self.assertEqual(profile["winner"]["env"], {})
        names = [candidate["name"] for candidate in profile["candidates"]]
        self.assertNotIn("omp-4", names)

    def test_nondeterministic_baseline_aborts_the_tune(self):
        FakeServeEngine.reset(drift_baseline=True)
        with self.assertRaises(RuntimeError):
            self.run_serve_tune()


class AutotuneIntegrationTest(unittest.TestCase):
    def test_unset_cap_resolves_from_the_plan_not_str_none(self):
        # --cap has default=None; str(None) used to reach every engine as
        # argv[1]="None" -> atoi -> 0: qwen36 aborted, GLM silently swept the
        # platform default instead of the plan's cap (#1190).
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory) / "cap-recorder.py"
            caps = Path(directory) / "caps"
            engine.write_text(
                "#!/usr/bin/env python3\n"
                "import os,sys\n"
                f"open({str(caps)!r},'a').write(sys.argv[1]+'\\n')\n"
                "if os.environ.get('PROMPT'):\n"
                " print('[PROMPT_TOKENS] 3: 10 11 12')\n"
                " print('[TOKENS] 4 generated: 20 21 22 23')\n"
                "else:\n"
                " print('REPLAY decode: 4 tokens | 10.00 tok/s | expert hit 95.0%')\n"
                " print('[PROF] decode forwards: 4 | latency p50 80.0 ms "
                "| p90 90.0 ms | p99 100.0 ms | max 100.0 ms')\n",
                encoding="utf-8",
            )
            engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
            p = plan(cores=1)
            p["tiers"]["ram"] = {"cache_slots_per_layer": 84}
            run_tune(
                str(engine), None, dict(os.environ), p, directory, "prompt",
                tokens=4, repeats=1, timeout=5, min_gain=0.03,
                profile_dir=directory,
            )
            seen = set(caps.read_text().split())
            self.assertEqual(seen, {"84"},
                             f"engine saw caps {seen}, expected the plan's 84")

    def test_unset_cap_with_capless_plan_refuses_loudly(self):
        with self.assertRaises(ValueError):
            run_tune("/nonexistent-engine", None, dict(os.environ),
                     plan(cores=1), "/tmp", "prompt", tokens=4, repeats=1,
                     timeout=5, min_gain=0.03)

    def test_fixed_replay_selects_and_persists_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory) / "fake-engine.py"
            engine.write_text(
                "#!/usr/bin/env python3\n"
                "import os\n"
                "if os.environ.get('PROMPT'):\n"
                " print('calibration')\n"
                " print('[PROMPT_TOKENS] 3: 10 11 12')\n"
                " print('[TOKENS] 4 generated: 20 21 22 23')\n"
                "else:\n"
                " pipe=os.environ.get('COLI_CUDA_PIPE','0')\n"
                " speed={'0':10.0,'1':12.0,'2':11.0}[pipe]\n"
                " if os.environ.get('COLI_CUDA_ASYNC') == '0': speed=9.0\n"
                " print(f'REPLAY decode: 4 tokens | {speed:.2f} tok/s | expert hit 95.0%')\n"
                " print('[PROF] decode forwards: 4 | latency p50 80.0 ms | p90 90.0 ms | p99 100.0 ms | max 100.0 ms')\n",
                encoding="utf-8",
            )
            engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
            p = plan(cores=1)
            profile, path = run_tune(
                str(engine), 8, dict(os.environ, COLI_CUDA_PIPE="0"),
                p, directory, "prompt", tokens=4, repeats=1, timeout=5,
                min_gain=0.03, profile_dir=directory,
            )
            self.assertTrue(profile["accepted"])
            self.assertEqual(profile["winner"]["env"], {"COLI_CUDA_PIPE": "1"})
            self.assertAlmostEqual(profile["gain"], 0.20)
            self.assertEqual(json.loads(path.read_text())["winner"]["tok_s"], 12.0)

    def test_reverse_confirmation_rejects_warmup_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory) / "drifting-engine.py"
            counter = Path(directory) / "counter"
            engine.write_text(
                "#!/usr/bin/env python3\n"
                "import os\n"
                f"counter={str(counter)!r}\n"
                "if os.environ.get('PROMPT'):\n"
                " print('[PROMPT_TOKENS] 3: 10 11 12')\n"
                " print('[TOKENS] 4 generated: 20 21 22 23')\n"
                "else:\n"
                " try: n=int(open(counter).read())\n"
                " except FileNotFoundError: n=0\n"
                " open(counter,'w').write(str(n+1))\n"
                " speed=10.0+n\n"
                " print(f'REPLAY decode: 4 tokens | {speed:.2f} tok/s | expert hit 95.0%')\n"
                " print('[PROF] decode forwards: 4 | latency p50 80.0 ms | p90 90.0 ms | p99 100.0 ms | max 100.0 ms')\n",
                encoding="utf-8",
            )
            engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
            profile, _ = run_tune(
                str(engine), 8, dict(os.environ, COLI_CUDA_PIPE="0"),
                plan(cores=1), directory, "prompt", tokens=4, repeats=1,
                timeout=5, min_gain=0.03, profile_dir=directory,
            )
            self.assertFalse(profile["accepted"])
            self.assertIsNotNone(profile["validation"])
            self.assertLess(
                profile["validation"]["winner"]["tok_s"],
                profile["validation"]["baseline"]["tok_s"],
            )


if __name__ == "__main__":
    unittest.main()
