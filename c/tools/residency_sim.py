#!/usr/bin/env python3
"""Offline GLM expert-residency policy simulator.

The simulator consumes Colibri ROUTE_TRACE output without loading a model. It
replays GLM's routed-expert unions through bounded per-layer caches and compares
placement policies under the same expert-residency byte budget. It deliberately
models demand residency only: no speculative reads, routing changes, or token
semantics are involved.

ROUTE_TRACE lines have this form (one row per token position and MoE call):

    <call> <row> <layer> <expert>:<gate> ...

Rows sharing one contiguous (call, layer) group are a runtime batch. The parser
retains both the first-seen expert union for demand replay and per-row selection
counts for training startup residents.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


EXPERT_FIELD = re.compile(r"^(\d+):([^:]+)$")


@dataclass(frozen=True)
class AccessEvent:
    layer: int
    experts: tuple[int, ...]
    selections: tuple[int, ...] = ()
    call: int | None = None
    source: str = ""

    def selection_counts(self) -> Counter[int]:
        return Counter(self.selections or self.experts)


@dataclass(frozen=True)
class LayerSpec:
    resident_bytes: int
    read_bytes: int
    felt_miss_us: float
    max_experts: int


@dataclass
class PolicyStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    bytes_read: int = 0
    felt_wait_us: float = 0.0
    admissions: int = 0
    rejected_admissions: int = 0
    evictions: int = 0
    seeded_objects: int = 0

    def add(self, other: "PolicyStats") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def report(self) -> dict:
        accesses = self.hits + self.misses
        return {
            **asdict(self),
            "accesses": accesses,
            "hit_rate": self.hits / accesses if accesses else 1.0,
        }


def _build_event(key, union, selections, source):
    if key is None or not union:
        return None
    return AccessEvent(key[1], tuple(union), tuple(selections), key[0], source)


def parse_trace(path: Path) -> list[AccessEvent]:
    """Parse contiguous GLM ROUTE_TRACE call groups into demand-union events."""
    events = []
    key = None
    union: list[int] = []
    known: set[int] = set()
    selections: list[int] = []
    seen_keys: set[tuple[int, int]] = set()
    next_row = 0
    last_call = -1
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) < 4:
            raise ValueError(f"{path}:{lineno}: expected call row layer expert:gate ...")
        try:
            call, row, layer = map(int, fields[:3])
        except ValueError as error:
            raise ValueError(f"{path}:{lineno}: invalid call/row/layer") from error
        if call < 0 or row < 0 or layer < 0:
            raise ValueError(f"{path}:{lineno}: negative call/row/layer")
        next_key = (call, layer)
        if next_key != key:
            event = _build_event(key, union, selections, str(path))
            if event:
                events.append(event)
            if next_key in seen_keys:
                raise ValueError(f"{path}:{lineno}: noncontiguous call/layer group {next_key}")
            if call < last_call:
                raise ValueError(f"{path}:{lineno}: call id moved backwards")
            seen_keys.add(next_key)
            key = next_key
            union = []
            known = set()
            selections = []
            next_row = 0
            last_call = call
        if row != next_row:
            raise ValueError(
                f"{path}:{lineno}: expected row {next_row} in call/layer group, got {row}")
        next_row += 1
        for value in fields[3:]:
            match = EXPERT_FIELD.fullmatch(value)
            if not match:
                raise ValueError(f"{path}:{lineno}: invalid expert field {value!r}")
            expert = int(match.group(1))
            try:
                gate = float(match.group(2))
            except ValueError as error:
                raise ValueError(f"{path}:{lineno}: invalid gate in {value!r}") from error
            if not math.isfinite(gate):
                raise ValueError(f"{path}:{lineno}: non-finite gate in {value!r}")
            selections.append(expert)
            if expert not in known:
                known.add(expert)
                union.append(expert)
    event = _build_event(key, union, selections, str(path))
    if event:
        events.append(event)
    return events


def validate_glm_trace(events: Sequence[AccessEvent], path: Path) -> None:
    if not events or any(event.call is None for event in events):
        return
    calls = [event.call for event in events]
    if calls[0] != 0 or any(
            call != previous + 1 for previous, call in zip(calls, calls[1:])):
        raise ValueError(
            f"{path}: trace lacks advancing GLM call ids; unsupported engine or corrupt trace")


def read_traces(paths: Sequence[Path]) -> list[list[AccessEvent]]:
    traces = [parse_trace(path) for path in paths]
    if not traces or any(not trace for trace in traces):
        raise ValueError("every trace must contain at least one routing event")
    for path, trace in zip(paths, traces):
        validate_glm_trace(trace, path)
    return traces


def load_specs(
    traces: Sequence[Sequence[AccessEvent]],
    manifest: Path,
    read_gbps: float,
    felt_fraction: float,
) -> dict[int, LayerSpec]:
    if not math.isfinite(read_gbps) or read_gbps <= 0:
        raise ValueError("read GB/s must be positive and finite")
    if not math.isfinite(felt_fraction) or not 0 <= felt_fraction <= 1:
        raise ValueError("felt fraction must be finite and in [0,1]")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("layers"), dict):
        raise ValueError("manifest.layers must be an object")
    specs = {}
    for raw_layer, raw_spec in document["layers"].items():
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid manifest layer {raw_layer!r}") from error
        if layer < 0 or not isinstance(raw_spec, dict):
            raise ValueError(f"invalid layer {raw_layer!r} specification")
        try:
            resident_bytes = int(raw_spec["resident_bytes"])
            read_bytes = int(raw_spec.get("read_bytes", resident_bytes))
            max_experts = int(raw_spec["max_experts"])
            default_felt = (read_bytes / (read_gbps * 1e9) * 1e6 * felt_fraction)
            felt = float(raw_spec.get("felt_miss_us", default_felt))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid layer {layer} specification") from error
        if (resident_bytes <= 0 or read_bytes <= 0 or max_experts <= 0 or
                felt < 0 or not math.isfinite(felt)):
            raise ValueError(f"invalid layer {layer} specification")
        specs[layer] = LayerSpec(resident_bytes, read_bytes, felt, max_experts)
    observed = defaultdict(set)
    for trace in traces:
        for event in trace:
            observed[event.layer].update(event.experts)
    for layer, experts in observed.items():
        if layer not in specs:
            raise ValueError(f"trace layer {layer} is absent from manifest")
        invalid = [expert for expert in experts if expert >= specs[layer].max_experts]
        if invalid:
            raise ValueError(f"trace layer {layer} expert {min(invalid)} exceeds manifest bound")
    return {layer: spec for layer, spec in specs.items() if layer in observed}


def uniform_capacities(specs: dict[int, LayerSpec], budget_bytes: int) -> dict[int, int]:
    """Allocate equal slot counts per layer, bounded by each layer's expert count."""
    if budget_bytes < 0:
        raise ValueError("budget must be non-negative")
    capacities = {layer: 0 for layer in specs}
    used = 0
    while True:
        cost = sum(spec.resident_bytes for layer, spec in specs.items()
                   if capacities[layer] < spec.max_experts)
        if cost <= 0 or used + cost > budget_bytes:
            break
        for layer, spec in specs.items():
            if capacities[layer] < spec.max_experts:
                capacities[layer] += 1
                used += spec.resident_bytes
    return capacities


class BasePolicy:
    def __init__(self, capacities, specs, learned_counts=None):
        self.capacities = capacities
        self.specs = specs
        self.learned_counts = learned_counts or {}
        self.stats = PolicyStats()

    def lookup(self, layer: int, expert: int) -> bool:
        raise NotImplementedError

    def observe(self, layer: int, expert: int, hit: bool) -> None:
        pass

    def admit_batch(self, layer: int, misses: Sequence[int]) -> None:
        raise NotImplementedError

    def access(self, event: AccessEvent) -> None:
        # GLM resolves/promotes at most 64 experts at a time.
        for start in range(0, len(event.experts), 64):
            misses = []
            spec = self.specs[event.layer]
            for expert in event.experts[start:start + 64]:
                self.stats.requests += 1
                hit = self.lookup(event.layer, expert)
                self.observe(event.layer, expert, hit)
                if hit:
                    self.stats.hits += 1
                else:
                    self.stats.misses += 1
                    self.stats.bytes_read += spec.read_bytes
                    self.stats.felt_wait_us += spec.felt_miss_us
                    misses.append(expert)
            self.admit_batch(event.layer, misses)


class LRUPolicy(BasePolicy):
    def __init__(self, capacities, specs, learned_counts=None):
        super().__init__(capacities, specs, learned_counts)
        self.cache: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)

    def lookup(self, layer, expert):
        cache = self.cache[layer]
        if expert not in cache:
            return False
        cache.move_to_end(expert)
        return True

    def admit_batch(self, layer, misses):
        capacity = self.capacities.get(layer, 0)
        if capacity <= 0:
            self.stats.rejected_admissions += len(misses)
            return
        selected = list(misses[-capacity:])
        self.stats.rejected_admissions += len(misses) - len(selected)
        cache = self.cache[layer]
        for expert in reversed(selected):
            if expert in cache:
                cache.move_to_end(expert)
                continue
            while len(cache) >= capacity:
                cache.popitem(last=False)
                self.stats.evictions += 1
            cache[expert] = None
            self.stats.admissions += 1


class LFUPolicy(BasePolicy):
    """Always-admit LFU with per-layer periodic decay and recency ties."""
    def __init__(self, capacities, specs, learned_counts=None,
                 decay_interval=4096, decay=0.5):
        super().__init__(capacities, specs, learned_counts)
        self.resident: dict[int, set[int]] = defaultdict(set)
        self.frequency: dict[int, Counter[int]] = defaultdict(Counter)
        self.last: dict[int, dict[int, int]] = defaultdict(dict)
        self.clock: Counter[int] = Counter()
        self.decay_interval = decay_interval
        self.decay = decay
        for layer, capacity in capacities.items():
            counts = Counter(self.learned_counts.get(layer, {}))
            self.frequency[layer].update(counts)
            ranked = sorted(counts, key=lambda expert: (-counts[expert], expert))[:capacity]
            self.resident[layer].update(ranked)
            self.stats.seeded_objects += len(ranked)

    def lookup(self, layer, expert):
        return expert in self.resident[layer]

    def observe(self, layer, expert, hit):
        self.clock[layer] += 1
        if self.decay_interval and self.clock[layer] % self.decay_interval == 0:
            for candidate in list(self.frequency[layer]):
                self.frequency[layer][candidate] *= self.decay
                if self.frequency[layer][candidate] < 0.5:
                    del self.frequency[layer][candidate]
        self.frequency[layer][expert] += 1
        self.last[layer][expert] = self.clock[layer]

    def _victim(self, layer):
        return min(self.resident[layer],
                   key=lambda expert: (self.frequency[layer][expert],
                                       self.last[layer].get(expert, 0), expert))

    def admit_batch(self, layer, misses):
        capacity = self.capacities.get(layer, 0)
        if capacity <= 0:
            self.stats.rejected_admissions += len(misses)
            return
        resident = self.resident[layer]
        for expert in misses:
            if expert in resident:
                continue
            if len(resident) >= capacity:
                resident.remove(self._victim(layer))
                self.stats.evictions += 1
            resident.add(expert)
            self.stats.admissions += 1


class SLRUPolicy(BasePolicy):
    """Segmented LRU: one-hit objects enter probation; reused objects are protected."""
    def __init__(self, capacities, specs, learned_counts=None, protected_fraction=0.5):
        super().__init__(capacities, specs, learned_counts)
        self.protected_fraction = protected_fraction
        self.probation: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
        self.protected: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)

    def _limits(self, layer):
        capacity = self.capacities.get(layer, 0)
        if capacity <= 1:
            return capacity, 0
        protected = min(capacity - 1, max(1, int(capacity * self.protected_fraction)))
        return capacity - protected, protected

    def lookup(self, layer, expert):
        protected = self.protected[layer]
        probation = self.probation[layer]
        if expert in protected:
            protected.move_to_end(expert)
            return True
        if expert not in probation:
            return False
        del probation[expert]
        probation_limit, protected_limit = self._limits(layer)
        if protected_limit:
            while len(protected) >= protected_limit:
                demoted, _ = protected.popitem(last=False)
                probation[demoted] = None
                while len(probation) > probation_limit:
                    probation.popitem(last=False)
                    self.stats.evictions += 1
            protected[expert] = None
        else:
            probation[expert] = None
        return True

    def admit_batch(self, layer, misses):
        capacity = self.capacities.get(layer, 0)
        probation_limit, _protected_limit = self._limits(layer)
        if capacity <= 0:
            self.stats.rejected_admissions += len(misses)
            return
        probation = self.probation[layer]
        protected = self.protected[layer]
        for expert in misses:
            if expert in probation or expert in protected:
                continue
            probation[expert] = None
            self.stats.admissions += 1
            while len(probation) > probation_limit:
                probation.popitem(last=False)
                self.stats.evictions += 1


class FrequencyAdmissionPolicy(BasePolicy):
    """Decayed-frequency admission: demand may execute without polluting cache."""
    def __init__(self, capacities, specs, learned_counts=None,
                 decay_interval=4096, decay=0.5, hysteresis=0.10):
        super().__init__(capacities, specs, learned_counts)
        self.resident: dict[int, set[int]] = defaultdict(set)
        self.frequency: dict[int, Counter[int]] = defaultdict(Counter)
        self.last: dict[int, dict[int, int]] = defaultdict(dict)
        self.clock: Counter[int] = Counter()
        self.decay_interval = decay_interval
        self.decay = decay
        self.hysteresis = hysteresis
        for layer, capacity in capacities.items():
            counts = Counter(self.learned_counts.get(layer, {}))
            self.frequency[layer].update(counts)
            ranked = sorted(counts, key=lambda expert: (-counts[expert], expert))[:capacity]
            self.resident[layer].update(ranked)
            self.stats.seeded_objects += len(ranked)

    def lookup(self, layer, expert):
        return expert in self.resident[layer]

    def observe(self, layer, expert, hit):
        self.clock[layer] += 1
        if self.decay_interval and self.clock[layer] % self.decay_interval == 0:
            for candidate in list(self.frequency[layer]):
                self.frequency[layer][candidate] *= self.decay
                if self.frequency[layer][candidate] < 0.5:
                    del self.frequency[layer][candidate]
        self.frequency[layer][expert] += 1
        self.last[layer][expert] = self.clock[layer]

    def _victim(self, layer):
        return min(self.resident[layer],
                   key=lambda expert: (self.frequency[layer][expert],
                                       self.last[layer].get(expert, 0), expert))

    def admit_batch(self, layer, misses):
        capacity = self.capacities.get(layer, 0)
        if capacity <= 0:
            self.stats.rejected_admissions += len(misses)
            return
        resident = self.resident[layer]
        for expert in misses:
            if expert in resident:
                continue
            if len(resident) < capacity:
                resident.add(expert)
                self.stats.admissions += 1
                continue
            victim = self._victim(layer)
            if (self.frequency[layer][expert] <=
                    self.frequency[layer][victim] * (1 + self.hysteresis)):
                self.stats.rejected_admissions += 1
                continue
            resident.remove(victim)
            resident.add(expert)
            self.stats.evictions += 1
            self.stats.admissions += 1


class HalfPinnedLRUPolicy(BasePolicy):
    """Synthetic half-capacity frequency pins plus adaptive LRU."""
    def __init__(self, capacities, specs, learned_counts=None, pin_fraction=0.5):
        super().__init__(capacities, specs, learned_counts)
        self.pins: dict[int, set[int]] = defaultdict(set)
        self.cache: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
        self.adaptive_capacity = {}
        for layer, capacity in capacities.items():
            pin_count = min(capacity, int(capacity * pin_fraction))
            counts = self.learned_counts.get(layer, {})
            ranked = sorted(counts, key=lambda expert: (-counts[expert], expert))[:pin_count]
            self.pins[layer].update(ranked)
            self.adaptive_capacity[layer] = capacity - len(ranked)
            self.stats.seeded_objects += len(ranked)

    def lookup(self, layer, expert):
        if expert in self.pins[layer]:
            return True
        cache = self.cache[layer]
        if expert not in cache:
            return False
        cache.move_to_end(expert)
        return True

    def admit_batch(self, layer, misses):
        capacity = self.adaptive_capacity.get(layer, 0)
        if capacity <= 0:
            self.stats.rejected_admissions += len(misses)
            return
        selected = list(misses[-capacity:])
        self.stats.rejected_admissions += len(misses) - len(selected)
        cache = self.cache[layer]
        for expert in reversed(selected):
            if expert in self.pins[layer] or expert in cache:
                continue
            while len(cache) >= capacity:
                cache.popitem(last=False)
                self.stats.evictions += 1
            cache[expert] = None
            self.stats.admissions += 1


POLICIES = {
    "lru": LRUPolicy,
    "half-pinned": HalfPinnedLRUPolicy,
    "lfu": LFUPolicy,
    "slru": SLRUPolicy,
    "frequency": FrequencyAdmissionPolicy,
}


def training_counts(traces):
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for trace in traces:
        for event in trace:
            counts[event.layer].update(event.selection_counts())
    return dict(counts)


def run_policy(policy_name, capacities, specs, trace, learned_counts=None):
    policy = POLICIES[policy_name](capacities, specs, learned_counts)
    for event in trace:
        policy.access(event)
    return policy.stats


class LayerReplayCache:
    """Memoize exact per-layer replays; all implemented policies are layer-local."""
    def __init__(self, traces, specs, learned_counts=None, base=None):
        self.traces = traces
        self.specs = specs
        self.learned_counts = learned_counts or {}
        self.layer_traces = base.layer_traces if base is not None else {
            layer: [[event for event in trace if event.layer == layer] for trace in traces]
            for layer in specs
        }
        self.cache = base.cache if base is not None else {}
        self.miss_cache = base.miss_cache if base is not None else {}
        self.frontier_cache = base.frontier_cache if base is not None else {}

    def layer_stats(self, policy_name, layer, capacity, trace_index):
        key = (policy_name, layer, capacity, trace_index)
        if key not in self.cache:
            self.cache[key] = run_policy(
                policy_name,
                {layer: capacity},
                {layer: self.specs[layer]},
                self.layer_traces[layer][trace_index],
                {layer: self.learned_counts.get(layer, {})},
            )
        return self.cache[key]

    def trace_stats(self, policy_name, capacities, specs, trace_index):
        combined = PolicyStats()
        felt_wait_us = 0.0
        for layer, spec in specs.items():
            stats = self.layer_stats(
                policy_name, layer, capacities.get(layer, 0), trace_index)
            combined.add(stats)
            felt_wait_us += stats.misses * spec.felt_miss_us
        combined.felt_wait_us = felt_wait_us
        return combined

    def layer_wait(self, policy_name, layer, capacity, specs):
        return self.layer_misses(policy_name, layer, capacity) * specs[layer].felt_miss_us

    def layer_misses(self, policy_name, layer, capacity):
        key = (policy_name, layer, capacity)
        if key not in self.miss_cache:
            self.miss_cache[key] = sum(
                self.layer_stats(policy_name, layer, capacity, trace_index).misses
                for trace_index in range(len(self.traces))
            )
        return self.miss_cache[key]

    def allocation_target(self, policy_name, layer, current, affordable, spec):
        """Best exact target up to ``affordable``, cached by current capacity."""
        key = (policy_name, layer, current, spec.resident_bytes,
               spec.max_experts, spec.felt_miss_us)
        if key not in self.frontier_cache:
            maximum = spec.max_experts
            frontier = [None] * (maximum + 1)
            current_wait = (self.layer_misses(policy_name, layer, current)
                            * spec.felt_miss_us)
            best_key = None
            best_target = None
            for target in range(current + 1, maximum + 1):
                added_bytes = (target - current) * spec.resident_bytes
                target_wait = (self.layer_misses(policy_name, layer, target)
                               * spec.felt_miss_us)
                benefit = current_wait - target_wait
                candidate = (benefit / added_bytes, benefit, -added_bytes, target)
                if best_key is None or candidate > best_key:
                    best_key = candidate
                    best_target = target
                frontier[target] = best_target
            self.frontier_cache[key] = frontier
        return self.frontier_cache[key][affordable]


class ScaledReplayCache(LayerReplayCache):
    def __init__(self, base, specs):
        super().__init__(base.traces, specs, base.learned_counts, base=base)


def evaluate(
    traces,
    capacities,
    specs,
    policies,
    learned_counts=None,
    replay_cache=None,
) -> dict:
    replay_cache = replay_cache or LayerReplayCache(traces, specs, learned_counts)
    result = {}
    for policy_name in policies:
        aggregate = PolicyStats()
        per_trace = []
        for trace_index, trace in enumerate(traces):
            stats = replay_cache.trace_stats(
                policy_name, capacities, specs, trace_index)
            aggregate.add(stats)
            per_trace.append({
                "source": trace[0].source if trace else "",
                **stats.report(),
            })
        result[policy_name] = {"aggregate": aggregate.report(), "traces": per_trace}
    return result


def capacity_bytes(capacities, specs):
    return sum(capacities.get(layer, 0) * spec.resident_bytes
               for layer, spec in specs.items())


def scale_specs(specs, felt_scale, target_layer=None):
    if felt_scale <= 0 or not math.isfinite(felt_scale):
        raise ValueError("felt scale must be positive and finite")
    return {
        layer: LayerSpec(
            spec.resident_bytes,
            spec.read_bytes,
            (spec.felt_miss_us * felt_scale
             if target_layer is None or layer == target_layer else spec.felt_miss_us),
            spec.max_experts,
        )
        for layer, spec in specs.items()
    }


def trace_categories(traces, category_names=None):
    if category_names is not None:
        if len(category_names) != len(traces):
            raise ValueError("--eval-category requires one value per --eval trace")
        return list(category_names)
    categories = []
    for trace in traces:
        source = Path(trace[0].source).stem if trace and trace[0].source else "trace"
        match = re.match(r"^(.*?)(?:[_-]\d+)?$", source)
        categories.append(match.group(1) or source)
    return categories


def profile_costs(log_paths):
    """Extract per-layer felt costs from PROF logs when layer telemetry exists.

    Current GLM PROF output reports aggregate felt wait, so return the aggregate
    cost-per-miss fallback and leave per-layer calibration to a future layer
    telemetry field.
    """
    waits = []
    physical_misses = []
    requests_per_token = []
    complete = bool(log_paths)
    records = 0
    for path in log_paths or ():
        text = Path(path).read_text(encoding="utf-8")
        path_records = 0
        for line in text.splitlines():
            if "[PROF] expert I/O:" not in line:
                continue
            records += 1
            path_records += 1
            match = re.search(r"([0-9.]+)s read service / ([0-9.]+)s felt wait", line)
            miss_match = re.search(r"\([^)]*/\s*([0-9]+) loads?\)", line)
            request_match = re.search(r"\| ([0-9]+(?:\.[0-9]+)?) loads/token", line)
            if not match or not miss_match or not request_match:
                complete = False
                continue
            waits.append(float(match.group(2)))
            physical_misses.append(int(miss_match.group(1)))
            requests_per_token.append(float(request_match.group(1)))
        if not path_records:
            complete = False
    miss_count = sum(physical_misses) if complete and records else None
    return {
        "logs": [str(path) for path in log_paths or ()],
        "felt_wait_seconds": sum(waits),
        "physical_misses": miss_count,
        "requests_per_token": requests_per_token,
        "records": records,
        "complete": complete,
        "felt_us_per_physical_miss": (
            sum(waits) * 1e6 / miss_count
            if complete and miss_count else None),
        "scope": "aggregate; GLM PROF does not expose layer-level felt wait",
    }


def decision_gate(result, categories, minimum_gain=0.10, maximum_regression=0.03):
    """Compare every candidate with uniform LRU using equal-weight categories."""
    if not 0 <= minimum_gain < 1 or maximum_regression < 0:
        raise ValueError("invalid decision-gate thresholds")
    baseline_payload = result["results"].get("uniform", {}).get("lru")
    if not baseline_payload:
        raise ValueError("decision gate requires policy lru")
    if len(baseline_payload["traces"]) != len(categories):
        raise ValueError("category count does not match evaluation traces")

    baseline_by_category: dict[str, float] = defaultdict(float)
    category_trace_counts: Counter[str] = Counter(categories)
    for category, stats in zip(categories, baseline_payload["traces"]):
        baseline_by_category[category] += stats["felt_wait_us"]
    rows = []
    for allocation, policies in result["results"].items():
        for policy, payload in policies.items():
            if allocation == "uniform" and policy == "lru":
                continue
            candidate_by_category: dict[str, float] = defaultdict(float)
            for category, stats in zip(categories, payload["traces"]):
                candidate_by_category[category] += stats["felt_wait_us"]
            category_changes = {}
            for category in sorted(baseline_by_category):
                baseline = baseline_by_category[category]
                candidate = candidate_by_category[category]
                change = ((baseline - candidate) / baseline
                          if baseline else (0.0 if candidate == 0 else -math.inf))
                category_changes[category] = {
                    "baseline_felt_wait_us": baseline,
                    "candidate_felt_wait_us": candidate,
                    "gain": change,
                    "traces": category_trace_counts[category],
                }
            gains = [entry["gain"] for entry in category_changes.values()]
            mean_gain = statistics.fmean(gains)
            worst_gain = min(gains)
            aggregate_baseline = baseline_payload["aggregate"]["felt_wait_us"]
            aggregate_candidate = payload["aggregate"]["felt_wait_us"]
            aggregate_gain = ((aggregate_baseline - aggregate_candidate) / aggregate_baseline
                              if aggregate_baseline else 0.0)
            row = {
                "allocation": allocation,
                "policy": policy,
                "aggregate_gain": aggregate_gain,
                "category_mean_gain": mean_gain,
                "worst_category_gain": worst_gain,
                "categories": category_changes,
            }
            row["pass"] = mean_gain >= minimum_gain and worst_gain >= -maximum_regression
            rows.append(row)
    rows.sort(key=lambda row: (
        not row["pass"], -row["worst_category_gain"],
        -row["category_mean_gain"], row["allocation"], row["policy"]))
    return {
        "baseline": {"allocation": "uniform", "policy": "lru"},
        "minimum_category_mean_gain": minimum_gain,
        "maximum_category_regression": maximum_regression,
        "candidates": rows,
    }


def sensitivity_analysis(
    train,
    evaluation,
    specs,
    budget_bytes,
    policies,
    categories,
    felt_scales,
    minimum_gain=0.10,
    maximum_regression=0.03,
    sensitivity_layers=None,
    learned_counts=None,
    train_cache=None,
    evaluation_cache=None,
    baseline_result=None,
):
    learned_counts = learned_counts or training_counts(train)
    # Admission and hit/miss behavior is unchanged by a felt-cost scale within
    # one layer. Reuse the exact replay curves across perturbations and only
    # reweight their misses when optimizing dynamic capacity.
    train_cache = train_cache or LayerReplayCache(train, specs, learned_counts)
    evaluation_cache = evaluation_cache or LayerReplayCache(
        evaluation, specs, learned_counts)
    perturbations = [("baseline", None, 1.0)]
    perturbations.extend(
        (f"layer-{layer}-x{felt_scale:g}", layer, felt_scale)
        for layer in (sorted(specs) if sensitivity_layers is None
                      else sorted(set(sensitivity_layers)))
        if layer in specs
        for felt_scale in felt_scales
        if felt_scale != 1.0
    )
    scenarios = []
    for name, layer, felt_scale in perturbations:
        scenario_specs = scale_specs(specs, felt_scale, layer)
        if name == "baseline" and baseline_result is not None:
            result = baseline_result
        else:
            scenario_train_cache = ScaledReplayCache(train_cache, scenario_specs)
            scenario_evaluation_cache = ScaledReplayCache(evaluation_cache, scenario_specs)
            result = analyze(
                train, evaluation, scenario_specs, budget_bytes, policies,
                learned_counts, scenario_train_cache, scenario_evaluation_cache)
        scenarios.append({
            "name": name,
            "layer": layer,
            "felt_scale": felt_scale,
            "gate": decision_gate(
                result, categories, minimum_gain, maximum_regression),
        })
    candidates: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for scenario in scenarios:
        for candidate in scenario["gate"]["candidates"]:
            candidates[(candidate["allocation"], candidate["policy"])].append({
                "scenario": scenario["name"],
                "felt_scale": scenario["felt_scale"],
                "pass": candidate["pass"],
                "category_mean_gain": candidate["category_mean_gain"],
                "worst_category_gain": candidate["worst_category_gain"],
            })
    robust = []
    for (allocation, policy), outcomes in candidates.items():
        robust.append({
            "allocation": allocation,
            "policy": policy,
            "pass_all_scales": all(outcome["pass"] for outcome in outcomes),
            "outcomes": outcomes,
        })
    robust.sort(key=lambda row: (not row["pass_all_scales"], row["allocation"], row["policy"]))
    return {"scenarios": scenarios, "candidates": robust}


def _training_wait(traces, capacities, specs, policy_name, learned_counts):
    return sum(run_policy(policy_name, capacities, specs, trace, learned_counts).felt_wait_us
               for trace in traces)


def dynamic_capacities(
    traces,
    specs,
    budget_bytes,
    policy_name,
    learned_counts,
    replay_cache=None,
):
    """Greedily allocate slots using exact cold-trace replay for one policy."""
    if budget_bytes < 0:
        raise ValueError("budget must be non-negative")
    capacities = {layer: 0 for layer in specs}
    used = 0
    replay_cache = replay_cache or LayerReplayCache(traces, specs, learned_counts)
    layer_waits = {
        layer: replay_cache.layer_wait(policy_name, layer, 0, specs)
        for layer in specs
    }
    while True:
        best = None
        for layer, spec in specs.items():
            current = capacities[layer]
            affordable = min(
                spec.max_experts,
                current + (budget_bytes - used) // spec.resident_bytes,
            )
            if affordable <= current:
                continue
            target = replay_cache.allocation_target(
                policy_name, layer, current, affordable, spec)
            added_bytes = (target - current) * spec.resident_bytes
            wait = replay_cache.layer_wait(policy_name, layer, target, specs)
            benefit = layer_waits[layer] - wait
            candidate = (benefit / added_bytes, benefit, -added_bytes,
                          -layer, layer, target, wait, added_bytes)
            if best is None or candidate > best:
                best = candidate
        if best is None or best[1] <= 0:
            break
        (_ratio, _benefit, _neg_bytes, _neg_layer, layer, target,
         layer_wait, added_bytes) = best
        capacities[layer] = target
        layer_waits[layer] = layer_wait
        used += added_bytes
    return capacities


def analyze(
    train,
    evaluation,
    specs,
    budget_bytes,
    policies,
    learned_counts=None,
    train_cache=None,
    evaluation_cache=None,
):
    learned_counts = learned_counts or training_counts(train)
    train_cache = train_cache or LayerReplayCache(train, specs, learned_counts)
    evaluation_cache = evaluation_cache or LayerReplayCache(
        evaluation, specs, learned_counts)
    uniform = uniform_capacities(specs, budget_bytes)
    output = {
        "budget_bytes": budget_bytes,
        "specs": {str(k): asdict(v) for k, v in sorted(specs.items())},
        "allocations": {},
        "results": {},
    }
    output["allocations"]["uniform"] = {
        "capacities": {str(k): v for k, v in sorted(uniform.items())},
        "resident_bytes": capacity_bytes(uniform, specs),
    }
    output["allocations"]["uniform"]["unused_bytes"] = (
        budget_bytes - output["allocations"]["uniform"]["resident_bytes"])
    output["results"]["uniform"] = evaluate(
        evaluation, uniform, specs, policies, learned_counts, evaluation_cache)
    for policy in policies:
        capacities = dynamic_capacities(
            train, specs, budget_bytes, policy, learned_counts, train_cache)
        name = f"dynamic-{policy}"
        output["allocations"][name] = {
            "capacities": {str(k): v for k, v in sorted(capacities.items())},
            "resident_bytes": capacity_bytes(capacities, specs),
        }
        output["allocations"][name]["unused_bytes"] = (
            budget_bytes - output["allocations"][name]["resident_bytes"])
        output["results"][name] = evaluate(
            evaluation, capacities, specs, [policy], learned_counts, evaluation_cache)
    return output


def print_report(result):
    print(f"expert_budget={result['budget_bytes']} bytes")
    print("allocation policy hit_rate misses bytes_read felt_wait_ms evictions rejected seeded")
    for allocation, policies in result["results"].items():
        caps = result["allocations"][allocation]["capacities"]
        resident = result["allocations"][allocation]["resident_bytes"]
        unused = result["allocations"][allocation]["unused_bytes"]
        print(f"capacities[{allocation}]={caps} resident_bytes={resident} "
              f"unused_bytes={unused}")
        for policy, payload in policies.items():
            stats = payload["aggregate"]
            print(f"{allocation:18} {policy:11} {stats['hit_rate']:8.2%} "
                  f"{stats['misses']:7d} {stats['bytes_read']:12d} "
                  f"{stats['felt_wait_us']/1000:12.3f} "
                  f"{stats['evictions']:9d} {stats['rejected_admissions']:8d} "
                  f"{stats['seeded_objects']:6d}")


def print_gate(gate):
    print("decision_gate baseline=uniform/lru "
          f"min_mean_gain={gate['minimum_category_mean_gain']:.1%} "
          f"max_regression={gate['maximum_category_regression']:.1%}")
    print("allocation policy verdict category_mean worst_category aggregate")
    for candidate in gate["candidates"]:
        verdict = "PASS" if candidate["pass"] else "FAIL"
        print(f"{candidate['allocation']:18} {candidate['policy']:11} {verdict:7} "
              f"{candidate['category_mean_gain']:13.2%} "
              f"{candidate['worst_category_gain']:14.2%} "
              f"{candidate['aggregate_gain']:9.2%}")


def synthetic_trace(kind: str):
    specs = {
        0: LayerSpec(1_000_000, 1_000_000, 1000.0, 16),
        1: LayerSpec(1_000_000, 1_000_000, 4000.0, 16),
        2: LayerSpec(2_000_000, 2_000_000, 2000.0, 16),
    }
    budget = 8_000_000

    def events(layer, sequence):
        return [AccessEvent(layer, (expert,), (expert,), None, kind) for expert in sequence]

    hot0 = [0, 1, 0, 2, 0, 1, 0, 3] * 100
    hot1 = [0, 1, 0, 2, 0, 3, 0, 1, 4, 0, 2, 0] * 100
    burst2 = sum(([0, 1] * 8 + [cold] for cold in range(2, 12)), []) * 20
    train_trace = events(0, hot0) + events(1, hot1) + events(2, burst2)
    if kind == "stationary":
        eval_trace = events(0, hot0[:400]) + events(1, hot1[:600]) + events(2, burst2[:400])
    elif kind == "shifted":
        shifted1 = list(range(16)) * 40
        shifted0 = [7, 8] * 300
        eval_trace = events(0, shifted0) + events(1, shifted1) + events(2, burst2[:300])
    else:
        raise ValueError(kind)
    return [train_trace], [eval_trace], specs, budget


def synthetic_category_suite():
    """Two-category transfer check: one stationary and one shifted workload."""
    stationary_train, stationary_eval, specs, budget = synthetic_trace("stationary")
    _shifted_train, shifted_eval, _shifted_specs, _shifted_budget = synthetic_trace("shifted")
    return (
        stationary_train,
        [stationary_eval[0], shifted_eval[0]],
        specs,
        budget,
        ["stationary", "shifted"],
    )


def synthetic_budget_sweep(policies, budgets_mb=(2, 4, 6, 8, 10, 12, 16, 24, 32)):
    train, evaluation, specs, _budget, categories = synthetic_category_suite()
    rows = []
    for budget_mb in budgets_mb:
        budget_bytes = budget_mb * 1_000_000
        result = analyze(train, evaluation, specs, budget_bytes, policies)
        gate = decision_gate(result, categories)
        sensitivity = sensitivity_analysis(
            train, evaluation, specs, budget_bytes, policies, categories,
            [0.5, 1.0, 1.5])
        robust = {
            (row["allocation"], row["policy"]): row["pass_all_scales"]
            for row in sensitivity["candidates"]
        }
        for candidate in gate["candidates"]:
            if candidate["pass"]:
                rows.append({
                    "budget_mb": budget_mb,
                    "allocation": candidate["allocation"],
                    "policy": candidate["policy"],
                    "category_mean_gain": candidate["category_mean_gain"],
                    "worst_category_gain": candidate["worst_category_gain"],
                    "pass_sensitivity": robust[
                        (candidate["allocation"], candidate["policy"])],
                })
    return rows


def command_demo(args):
    for scenario in ("stationary", "shifted"):
        train, evaluation, specs, budget = synthetic_trace(scenario)
        print(f"\n=== {scenario} ===")
        result = analyze(train, evaluation, specs, budget, args.policies)
        print_report(result)
        if "lru" in args.policies:
            print_gate(decision_gate(result, [scenario]))
    if "lru" in args.policies:
        train, evaluation, specs, budget, categories = synthetic_category_suite()
        print("\n=== combined category-held-out gate ===")
        result = analyze(train, evaluation, specs, budget, args.policies)
        print_gate(decision_gate(result, categories))
        sensitivity = sensitivity_analysis(
            train, evaluation, specs, budget, args.policies, categories,
            [0.5, 1.0, 1.5])
        print("sensitivity " + " ".join(
            f"{row['allocation']}/{row['policy']}="
            f"{'PASS' if row['pass_all_scales'] else 'FAIL'}"
            for row in sensitivity["candidates"]))
        sweep = synthetic_budget_sweep(args.policies)
        print("\n=== synthetic budget operating regions ===")
        print("budget_mb allocation policy category_mean worst_category sensitivity")
        for row in sweep:
            print(f"{row['budget_mb']:9d} {row['allocation']:18} {row['policy']:11} "
                  f"{row['category_mean_gain']:13.2%} "
                  f"{row['worst_category_gain']:14.2%} "
                  f"{'PASS' if row['pass_sensitivity'] else 'FAIL'}")


def _real_inputs(args):
    train_paths = [path.resolve() for path in args.train]
    eval_paths = [path.resolve() for path in args.eval]
    if len(set(train_paths)) != len(train_paths):
        raise ValueError("duplicate path in --train")
    if len(set(eval_paths)) != len(eval_paths):
        raise ValueError("duplicate path in --eval")
    overlap = set(train_paths) & set(eval_paths)
    if overlap:
        raise ValueError(f"train/eval paths overlap: {sorted(map(str, overlap))[0]}")
    try:
        train_identities = [(path.stat().st_dev, path.stat().st_ino) for path in train_paths]
        eval_identities = [(path.stat().st_dev, path.stat().st_ino) for path in eval_paths]
    except OSError:
        train_identities = []
        eval_identities = []
    if len(set(train_identities)) != len(train_identities):
        raise ValueError("duplicate file in --train")
    if len(set(eval_identities)) != len(eval_identities):
        raise ValueError("duplicate file in --eval")
    if set(train_identities) & set(eval_identities):
        raise ValueError("train/eval paths refer to the same file")
    train = read_traces(train_paths)
    evaluation = read_traces(eval_paths)
    specs = load_specs(train + evaluation, args.manifest,
                       args.read_gbps, args.felt_fraction)
    return train, evaluation, specs


def command_run(args):
    train, evaluation, specs = _real_inputs(args)
    if args.max_training_events:
        train = [trace[:args.max_training_events] for trace in train]
        if any(not trace for trace in train):
            raise ValueError("--max-training-events removed every event from a trace")
    budget = int(args.expert_budget_gb * 1e9)
    learned_counts = training_counts(train)
    train_cache = LayerReplayCache(train, specs, learned_counts)
    evaluation_cache = LayerReplayCache(evaluation, specs, learned_counts)
    result = analyze(
        train, evaluation, specs, budget, args.policies, learned_counts,
        train_cache, evaluation_cache)
    print_report(result)
    categories = trace_categories(evaluation, args.eval_category)
    gate = decision_gate(result, categories, args.minimum_gain, args.maximum_regression)
    print_gate(gate)
    sensitivity = sensitivity_analysis(
        train, evaluation, specs, budget, args.policies, categories,
        args.felt_scales, args.minimum_gain, args.maximum_regression,
        args.sensitivity_layers, learned_counts, train_cache, evaluation_cache,
        result)
    costs = profile_costs(args.prof_log)
    print("sensitivity " + " ".join(
        f"{row['allocation']}/{row['policy']}="
        f"{'PASS' if row['pass_all_scales'] else 'FAIL'}"
        for row in sensitivity["candidates"]))
    if args.json_out:
        document = {
            "result": result,
            "categories": categories,
            "profile_costs": costs,
            "gate": gate,
            "sensitivity": sensitivity,
        }
        args.json_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def command_sweep(args):
    train, evaluation, specs = _real_inputs(args)
    if args.max_training_events:
        train = [trace[:args.max_training_events] for trace in train]
        if any(not trace for trace in train):
            raise ValueError("--max-training-events removed every event from a trace")
    categories = trace_categories(evaluation, args.eval_category)
    costs = profile_costs(args.prof_log)
    learned_counts = training_counts(train)
    train_cache = LayerReplayCache(train, specs, learned_counts)
    evaluation_cache = LayerReplayCache(evaluation, specs, learned_counts)
    results = []
    for budget_gb in args.expert_budgets:
        budget_bytes = int(budget_gb * 1e9)
        result = analyze(
            train, evaluation, specs, budget_bytes, args.policies,
            learned_counts, train_cache, evaluation_cache)
        results.append({
            "expert_budget_gb": budget_gb,
            "result": result,
            "profile_costs": costs,
            "gate": decision_gate(
                result, categories, args.minimum_gain, args.maximum_regression),
            "sensitivity": sensitivity_analysis(
                train, evaluation, specs, budget_bytes, args.policies,
                categories, args.felt_scales, args.minimum_gain,
                args.maximum_regression, args.sensitivity_layers,
                learned_counts, train_cache, evaluation_cache, result),
        })
    for entry in results:
        print(f"\n=== expert budget {entry['expert_budget_gb']:g} GB ===")
        print_report(entry["result"])
        print_gate(entry["gate"])
        print("sensitivity " + " ".join(
            f"{row['allocation']}/{row['policy']}="
            f"{'PASS' if row['pass_all_scales'] else 'FAIL'}"
            for row in entry["sensitivity"]["candidates"]))
    if args.json_out:
        args.json_out.write_text(json.dumps({"sweeps": results}, indent=2) + "\n",
                                 encoding="utf-8")


def add_real_trace_arguments(parser):
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--eval", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--read-gbps", type=float, default=3.0)
    parser.add_argument("--felt-fraction", type=float, default=1.0)
    parser.add_argument("--felt-scales", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--eval-category", action="append")
    parser.add_argument("--minimum-gain", type=float, default=0.10)
    parser.add_argument("--maximum-regression", type=float, default=0.03)
    parser.add_argument("--max-training-events", type=int, default=0,
                        help="limit events per training trace for a quick smoke run")
    parser.add_argument("--sensitivity-layer", dest="sensitivity_layers",
                        type=int, action="append",
                        help="limit cost sensitivity to selected layers")
    parser.add_argument("--prof-log", type=Path, action="append", default=[],
                        help="GLM PROF=1 log used for aggregate cost calibration")
    parser.add_argument("--policies", nargs="+", choices=tuple(POLICIES),
                        default=list(POLICIES))
    parser.add_argument("--json-out", type=Path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run deterministic stationary/shifted fixtures")
    demo.add_argument("--policies", nargs="+", choices=tuple(POLICIES),
                      default=list(POLICIES))
    demo.set_defaults(func=command_demo)

    run = sub.add_parser("run", help="simulate real GLM ROUTE_TRACE files")
    add_real_trace_arguments(run)
    run.add_argument("--expert-budget-gb", type=float, required=True)
    run.set_defaults(func=command_run)

    sweep = sub.add_parser("sweep", help="compare policies over expert RAM budgets")
    add_real_trace_arguments(sweep)
    sweep.add_argument("--expert-budgets", type=float, nargs="+", required=True)
    sweep.set_defaults(func=command_sweep)
    args = parser.parse_args()
    budgets = ([args.expert_budget_gb] if hasattr(args, "expert_budget_gb")
               else getattr(args, "expert_budgets", []))
    if any(value <= 0 or not math.isfinite(value) or value > 1e9 for value in budgets):
        parser.error("expert budget values must be positive, finite, and <= 1e9 GB")
    if hasattr(args, "felt_scales") and any(
            value <= 0 or not math.isfinite(value) for value in args.felt_scales):
        parser.error("--felt-scales values must be positive and finite")
    if hasattr(args, "minimum_gain") and not 0 <= args.minimum_gain < 1:
        parser.error("--minimum-gain must be in [0,1)")
    if hasattr(args, "maximum_regression") and args.maximum_regression < 0:
        parser.error("--maximum-regression must be non-negative")
    if hasattr(args, "max_training_events") and args.max_training_events < 0:
        parser.error("--max-training-events must be non-negative")
    if hasattr(args, "policies") and "lru" not in args.policies:
        parser.error("--policies must include lru as the decision-gate baseline")
    try:
        args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
