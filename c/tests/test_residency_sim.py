import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
import residency_sim as sim


class TraceParserTest(unittest.TestCase):
    def test_batch_rows_keep_union_order_and_selection_multiplicity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.txt"
            path.write_text(
                "0 0 3 7:0.6000 2:0.4000\n"
                "0 1 3 2:0.7000 9:0.3000\n"
                "1 0 4 1:1.0000\n",
                encoding="utf-8",
            )
            events = sim.parse_trace(path)
        self.assertEqual(events, [
            sim.AccessEvent(3, (7, 2, 9), (7, 2, 2, 9), 0, str(path)),
            sim.AccessEvent(4, (1,), (1,), 1, str(path)),
        ])

    def test_multiple_calls_to_same_layer_remain_separate_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.txt"
            path.write_text("0 0 3 1:1.0000\n1 0 3 2:1.0000\n", encoding="utf-8")
            events = sim.parse_trace(path)
        self.assertEqual([event.experts for event in events], [(1,), (2,)])

    def test_noncontiguous_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.txt"
            path.write_text(
                "0 0 3 1:1.0000\n1 0 4 2:1.0000\n0 1 3 3:1.0000\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "noncontiguous"):
                sim.parse_trace(path)

    def test_kimi_style_constant_call_ids_are_rejected(self):
        events = [
            sim.AccessEvent(0, (1,), call=0),
            sim.AccessEvent(1, (2,), call=0),
            sim.AccessEvent(2, (3,), call=0),
        ]
        with self.assertRaisesRegex(ValueError, "lacks advancing GLM call ids"):
            sim.validate_glm_trace(events, Path("kimi.trace"))

    def test_glm_call_ids_advance_across_layer_wraps(self):
        events = [
            sim.AccessEvent(0, (1,), call=0),
            sim.AccessEvent(1, (2,), call=1),
            sim.AccessEvent(2, (3,), call=2),
            sim.AccessEvent(0, (4,), call=3),
            sim.AccessEvent(1, (5,), call=4),
        ]
        sim.validate_glm_trace(events, Path("glm.trace"))

    def test_glm_call_ids_must_start_at_zero_and_advance_at_wrap(self):
        invalid = (
            [sim.AccessEvent(0, (1,), call=1)],
            [sim.AccessEvent(0, (1,), call=0), sim.AccessEvent(1, (2,), call=1),
             sim.AccessEvent(0, (3,), call=1)],
        )
        for events in invalid:
            with self.subTest(events=events), self.assertRaisesRegex(
                    ValueError, "lacks advancing GLM call ids"):
                sim.validate_glm_trace(events, Path("glm.trace"))

    def test_rows_must_start_at_zero_and_be_contiguous(self):
        invalid = (
            "0 1 3 7:1.0000\n",
            "0 0 3 7:1.0000\n0 2 3 8:1.0000\n",
        )
        for text in invalid:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "trace.txt"
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "expected row"):
                    sim.parse_trace(path)

    def test_malformed_or_nonfinite_gate_is_rejected_with_location(self):
        invalid = ("0 0 3 bad\n", "0 0 3 7:nan\n", "0 0 3 7:1:2\n")
        for text in invalid:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "trace.txt"
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, r"trace\.txt:1"):
                    sim.parse_trace(path)


class SpecificationTest(unittest.TestCase):
    def test_manifest_separates_resident_and_read_bytes(self):
        traces = [[sim.AccessEvent(3, (0, 1))]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"layers": {"3": {
                "resident_bytes": 123,
                "read_bytes": 456,
                "felt_miss_us": 45.5,
                "max_experts": 9,
            }}}), encoding="utf-8")
            specs = sim.load_specs(traces, path, 1.0, 1.0)
        self.assertEqual(specs[3], sim.LayerSpec(123, 456, 45.5, 9))

    def test_manifest_bounds_are_required_for_observed_layers_and_experts(self):
        traces = [[sim.AccessEvent(3, (9,))]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"layers": {"3": {
                "resident_bytes": 123,
                "max_experts": 9,
            }}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exceeds manifest bound"):
                sim.load_specs(traces, path, 1.0, 1.0)

    def test_default_felt_cost_uses_read_bytes(self):
        traces = [[sim.AccessEvent(0, (0,))]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"layers": {"0": {
                "resident_bytes": 10,
                "read_bytes": 1_000_000_000,
                "max_experts": 1,
            }}}), encoding="utf-8")
            specs = sim.load_specs(traces, path, 2.0, 0.5)
        self.assertEqual(specs[0].felt_miss_us, 250_000)

    def test_manifest_layers_absent_from_traces_do_not_consume_budget(self):
        traces = [[sim.AccessEvent(0, (0,))]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"layers": {
                "0": {"resident_bytes": 10, "max_experts": 1},
                "1": {"resident_bytes": 10, "max_experts": 1},
            }}), encoding="utf-8")
            specs = sim.load_specs(traces, path, 1.0, 1.0)
        self.assertEqual(set(specs), {0})

    def test_profile_costs_extracts_aggregate_felt_wait(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.out"
            path.write_text(
                "[PROF] expert I/O: 38.178 GB fetched | "
                "hit 42.9% (100 pin + 125 lru / 300 load) | 525.0 loads/token | "
                "7.2s read service / 1.4s felt wait\n",
                encoding="utf-8",
            )
            costs = sim.profile_costs([path])
        self.assertEqual(costs["physical_misses"], 300)
        self.assertEqual(costs["requests_per_token"], [525.0])
        self.assertTrue(costs["complete"])
        self.assertEqual(costs["records"], 1)
        self.assertEqual(costs["felt_wait_seconds"], 1.4)
        self.assertAlmostEqual(costs["felt_us_per_physical_miss"], 4666.6666667)

    def test_profile_costs_without_physical_misses_is_not_calibrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.out"
            path.write_text(
                "[PROF] legacy | 525.0 loads/token | "
                "7.2s read service / 1.4s felt wait\n",
                encoding="utf-8",
            )
            costs = sim.profile_costs([path])
        self.assertIsNone(costs["physical_misses"])
        self.assertEqual(costs["requests_per_token"], [])
        self.assertFalse(costs["complete"])
        self.assertIsNone(costs["felt_us_per_physical_miss"])

    def test_profile_costs_require_a_complete_record_from_every_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            complete = Path(tmp) / "complete.out"
            empty = Path(tmp) / "empty.out"
            complete.write_text(
                "[PROF] expert I/O: 1 GB fetched | "
                "hit 50.0% (1 pin + 1 lru / 2 load) | 4.0 loads/token | "
                "0.2s read service / 0.1s felt wait\n",
                encoding="utf-8",
            )
            empty.write_text("engine stopped before PROF output\n", encoding="utf-8")
            costs = sim.profile_costs([complete, empty])
        self.assertFalse(costs["complete"])
        self.assertIsNone(costs["physical_misses"])
        self.assertIsNone(costs["felt_us_per_physical_miss"])

    def test_profile_costs_distinguish_zero_misses_from_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.out"
            path.write_text(
                "[PROF] expert I/O: 0 GB fetched | "
                "hit 100.0% (1 pin + 1 lru / 0 load) | 2.0 loads/token | "
                "0.0s read service / 0.0s felt wait\n",
                encoding="utf-8",
            )
            costs = sim.profile_costs([path])
        self.assertTrue(costs["complete"])
        self.assertEqual(costs["physical_misses"], 0)
        self.assertIsNone(costs["felt_us_per_physical_miss"])


class PolicyBehaviorTest(unittest.TestCase):
    def specs(self):
        return {0: sim.LayerSpec(1, 4, 10, 100)}

    def events(self, sequence):
        return [sim.AccessEvent(0, (expert,), (expert,)) for expert in sequence]

    def test_lru_known_sequence_and_read_bytes(self):
        stats = sim.run_policy("lru", {0: 2}, self.specs(),
                               self.events([1, 2, 1, 3, 1, 2]))
        self.assertEqual((stats.hits, stats.misses), (2, 4))
        self.assertEqual(stats.bytes_read, 16)
        self.assertEqual(stats.felt_wait_us, 40)

    def test_frequency_admission_rejects_scan_pollution(self):
        warm = [0, 1] * 20
        sequence = warm[:]
        for cold in range(2, 22):
            sequence.extend([cold, 0, 1])
        lru = sim.run_policy("lru", {0: 2}, self.specs(), self.events(sequence))
        frequency = sim.run_policy(
            "frequency", {0: 2}, self.specs(), self.events(sequence))
        self.assertGreater(frequency.hits, lru.hits)
        self.assertLess(frequency.misses, lru.misses)
        self.assertGreater(frequency.rejected_admissions, 0)

    def test_training_uses_row_selection_multiplicity(self):
        event = sim.AccessEvent(0, (7, 8), (7, 7, 7, 8))
        counts = sim.training_counts([[event]])
        self.assertEqual(counts[0], {7: 3, 8: 1})

    def test_training_counts_seed_frequency_and_half_pinned_policies(self):
        learned = {0: {7: 100, 8: 50, 9: 1}}
        first = self.events([7, 8])
        frequency = sim.run_policy(
            "frequency", {0: 2}, self.specs(), first, learned)
        pinned = sim.run_policy(
            "half-pinned", {0: 2}, self.specs(), first, learned)
        self.assertEqual(frequency.hits, 2)
        self.assertEqual(frequency.seeded_objects, 2)
        self.assertEqual(pinned.hits, 1)
        self.assertEqual(pinned.seeded_objects, 1)

    def test_slru_protects_reused_objects_from_one_hit_scan(self):
        warm = [0, 1] * 20
        sequence = warm + sum(([cold, 0, 1] for cold in range(2, 22)), [])
        lru = sim.run_policy("lru", {0: 3}, self.specs(), self.events(sequence))
        slru = sim.run_policy("slru", {0: 3}, self.specs(), self.events(sequence))
        self.assertGreaterEqual(slru.hits, lru.hits)

    def test_lru_promotes_between_64_expert_blocks(self):
        event = sim.AccessEvent(0, tuple(range(65)) + (0,))
        stats = sim.run_policy("lru", {0: 64}, self.specs(), [event])
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.misses, 65)


class AllocationTest(unittest.TestCase):
    def test_uniform_allocation_respects_resident_bytes(self):
        specs = {
            0: sim.LayerSpec(10, 100, 1, 8),
            1: sim.LayerSpec(20, 100, 1, 8),
        }
        caps = sim.uniform_capacities(specs, 95)
        self.assertEqual(caps, {0: 3, 1: 3})
        self.assertEqual(sim.capacity_bytes(caps, specs), 90)

    def test_dynamic_budget_uses_exact_policy_replay(self):
        specs = {
            0: sim.LayerSpec(1, 1, 1, 8),
            1: sim.LayerSpec(1, 1, 10, 8),
        }
        trace = [[
            *[sim.AccessEvent(0, (expert,)) for expert in [0, 1, 2, 3] * 20],
            *[sim.AccessEvent(1, (expert,)) for expert in [0, 1] * 40],
        ]]
        counts = sim.training_counts(trace)
        caps = sim.dynamic_capacities(trace, specs, 4, "lru", counts)
        self.assertGreater(caps[1], caps[0])
        self.assertLessEqual(sim.capacity_bytes(caps, specs), 4)

    def test_dynamic_budget_evaluates_intermediate_capacity(self):
        specs = {0: sim.LayerSpec(1, 1, 1, 4)}
        sequence = [0, 1, 2, 0, 1, 2, 3] * 20
        traces = [[sim.AccessEvent(0, (expert,)) for expert in sequence]]
        counts = sim.training_counts(traces)
        self.assertEqual(
            sim.run_policy("lru", {0: 0}, specs, traces[0], counts).misses, 140)
        self.assertEqual(
            sim.run_policy("lru", {0: 2}, specs, traces[0], counts).misses, 140)
        self.assertEqual(
            sim.run_policy("lru", {0: 3}, specs, traces[0], counts).misses, 80)
        caps = sim.dynamic_capacities(traces, specs, 3, "lru", counts)
        self.assertEqual(caps, {0: 3})
        result = sim.analyze(traces, traces, specs, 3, ["lru"], counts)
        self.assertEqual(result["allocations"]["dynamic-lru"]["unused_bytes"], 0)

    def test_frontier_matches_exhaustive_greedy_on_small_specs(self):
        specs = {
            0: sim.LayerSpec(2, 1, 1, 5),
            1: sim.LayerSpec(3, 1, 4, 5),
        }
        traces = [[
            *[sim.AccessEvent(0, (expert,)) for expert in [0, 1, 2, 0, 1, 3] * 4],
            *[sim.AccessEvent(1, (expert,)) for expert in [0, 1, 0, 2, 0, 1] * 4],
        ]]
        counts = sim.training_counts(traces)

        def exhaustive(policy, scenario_specs, budget):
            capacities = {layer: 0 for layer in scenario_specs}
            cache = sim.LayerReplayCache(traces, scenario_specs, counts)
            waits = {
                layer: cache.layer_wait(policy, layer, 0, scenario_specs)
                for layer in scenario_specs
            }
            used = 0
            while True:
                best = None
                for layer, spec in scenario_specs.items():
                    current = capacities[layer]
                    for target in range(current + 1, spec.max_experts + 1):
                        added = (target - current) * spec.resident_bytes
                        if used + added > budget:
                            break
                        wait = cache.layer_wait(policy, layer, target, scenario_specs)
                        benefit = waits[layer] - wait
                        candidate = (benefit / added, benefit, -added, -layer,
                                     layer, target, wait, added)
                        if best is None or candidate > best:
                            best = candidate
                if best is None or best[1] <= 0:
                    break
                _, _, _, _, layer, target, wait, added = best
                capacities[layer] = target
                waits[layer] = wait
                used += added
            return capacities

        for policy in ("lru", "half-pinned", "frequency"):
            for felt_scale in (0.1, 0.2, 1.0, 3.5):
                scaled = sim.scale_specs(specs, felt_scale, target_layer=0)
                for budget in range(1, 26):
                    with self.subTest(
                            policy=policy, felt_scale=felt_scale, budget=budget):
                        self.assertEqual(
                            sim.dynamic_capacities(
                                traces, scaled, budget, policy, counts),
                            exhaustive(policy, scaled, budget),
                        )

    def test_allocation_reports_unusable_budget_remainder(self):
        specs = {0: sim.LayerSpec(2, 1, 1, 1)}
        traces = [[sim.AccessEvent(0, (0,))]]
        result = sim.analyze(traces, traces, specs, 3, ["lru"])
        self.assertEqual(result["allocations"]["uniform"]["unused_bytes"], 1)
        # A one-access training trace has no reuse benefit, so the dynamic
        # allocator intentionally buys no residency and exposes all 3 bytes as
        # unused rather than hiding the remainder.
        self.assertEqual(result["allocations"]["dynamic-lru"]["unused_bytes"], 3)

    def test_training_traces_are_independent_cold_runs(self):
        specs = {0: sim.LayerSpec(1, 1, 1, 4)}
        traces = [
            [sim.AccessEvent(0, (1,))],
            [sim.AccessEvent(0, (1,))],
        ]
        wait = sim._training_wait(traces, {0: 0}, specs, "lru", {})
        self.assertEqual(wait, 2)

    def test_shifted_heldout_dynamic_lru_regression_is_visible(self):
        train, evaluation, specs, budget = sim.synthetic_trace("shifted")
        result = sim.analyze(train, evaluation, specs, budget, ["lru"])
        uniform = result["results"]["uniform"]["lru"]["aggregate"]
        dynamic = result["results"]["dynamic-lru"]["lru"]["aggregate"]
        self.assertGreater(dynamic["felt_wait_us"], uniform["felt_wait_us"])


class DecisionGateTest(unittest.TestCase):
    def result(self, category_waits):
        def payload(values):
            total = sum(values)
            return {
                "aggregate": {"felt_wait_us": total},
                "traces": [{"felt_wait_us": value} for value in values],
            }

        return {"results": {
            "uniform": {
                "lru": payload([100, 100, 100]),
                "frequency": payload(category_waits),
            },
            "dynamic-frequency": {
                "frequency": payload([70, 70, 70]),
            },
        }}

    def test_gate_equal_weights_categories_and_rejects_worst_regression(self):
        categories = ["chat", "chat", "code"]
        gate = sim.decision_gate(self.result([80, 80, 104]), categories)
        frequency = next(
            row for row in gate["candidates"]
            if row["allocation"] == "uniform" and row["policy"] == "frequency")
        self.assertAlmostEqual(frequency["category_mean_gain"], 0.08)
        self.assertAlmostEqual(frequency["worst_category_gain"], -0.04)
        self.assertFalse(frequency["pass"])
        dynamic = next(
            row for row in gate["candidates"]
            if row["allocation"] == "dynamic-frequency")
        self.assertTrue(dynamic["pass"])

    def test_trace_categories_strip_numeric_replicates_or_use_explicit_names(self):
        traces = [
            [sim.AccessEvent(0, (0,), source="/tmp/chat_1.trace")],
            [sim.AccessEvent(0, (0,), source="/tmp/chat_2.trace")],
            [sim.AccessEvent(0, (0,), source="/tmp/code-1.trace")],
        ]
        self.assertEqual(sim.trace_categories(traces), ["chat", "chat", "code"])
        self.assertEqual(
            sim.trace_categories(traces, ["a", "b", "c"]), ["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "one value per"):
            sim.trace_categories(traces, ["a"])

    def test_layer_specific_sensitivity_can_invalidate_candidate(self):
        train, evaluation, specs, budget = sim.synthetic_trace("stationary")
        sensitivity = sim.sensitivity_analysis(
            train, evaluation, specs, budget, ["lru", "frequency"],
            ["stationary"], [0.5, 1.0, 1.5])
        frequency = next(
            row for row in sensitivity["candidates"]
            if row["allocation"] == "uniform" and row["policy"] == "frequency")
        self.assertEqual(len(frequency["outcomes"]), 1 + 2 * len(specs))
        self.assertTrue(any(
            outcome["scenario"] == "layer-1-x0.5"
            for outcome in frequency["outcomes"]))

    def test_replay_cache_preserves_full_replay_metrics(self):
        train, evaluation, specs, budget = sim.synthetic_trace("stationary")
        counts = sim.training_counts(train)
        capacities = sim.uniform_capacities(specs, budget)
        cached = sim.evaluate(
            evaluation, capacities, specs, ["lru", "frequency"], counts)
        for policy in ("lru", "frequency"):
            direct = sim.PolicyStats()
            for trace in evaluation:
                direct.add(sim.run_policy(policy, capacities, specs, trace, counts))
            self.assertEqual(cached[policy]["aggregate"], direct.report())

    def test_combined_category_suite_applies_worst_category_guard(self):
        train, evaluation, specs, budget, categories = sim.synthetic_category_suite()
        result = sim.analyze(train, evaluation, specs, budget, ["lru", "frequency"])
        gate = sim.decision_gate(result, categories)
        frequency = next(
            candidate for candidate in gate["candidates"]
            if candidate["allocation"] == "uniform"
            and candidate["policy"] == "frequency")
        self.assertGreater(frequency["category_mean_gain"], 0.10)
        self.assertLess(frequency["worst_category_gain"], -0.03)

    def test_synthetic_budget_sweep_exposes_robust_and_fragile_regions(self):
        rows = sim.synthetic_budget_sweep(["lru", "frequency"], [2, 8, 12])
        frequency = {
            row["budget_mb"]: row
            for row in rows
            if row["allocation"] == "dynamic-frequency"
        }
        self.assertTrue(frequency[2]["pass_sensitivity"])
        self.assertFalse(frequency[8]["pass_sensitivity"])
        self.assertTrue(frequency[12]["pass_sensitivity"])


if __name__ == "__main__":
    unittest.main()
