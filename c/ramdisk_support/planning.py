"""Pure profile, capacity, and placement planning primitives."""

from __future__ import print_function

import json
import os
import re

from .common import (
    PROFILE_LINE_RE,
    RamdiskError,
    _format_range_list,
    _parse_range_list,
)


def _load_profile(path, model):
    if not path:
        path = os.path.join(model["path"], ".coli_usage")
    if not os.path.isfile(path):
        raise RamdiskError(
            "partial staging requires .coli_usage or an explicit compatible --profile"
        )
    counts = {}
    fingerprint = None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            text = stream.read()
        if path.endswith(".json") or text.lstrip().startswith("{"):
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise RamdiskError("profile JSON must contain an object")
            fingerprint = payload.get("model_fingerprint")
            rows = payload.get("counts", [])
            if not isinstance(rows, list):
                raise RamdiskError("profile JSON counts must be a list")
            for row in rows:
                if isinstance(row, dict):
                    layer, expert, count = row.get("layer"), row.get("expert"), row.get("count")
                else:
                    layer, expert, count = row
                counts["%d:%d" % (int(layer), int(expert))] = int(count)
        else:
            for number, line in enumerate(text.splitlines(), 1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                match = PROFILE_LINE_RE.match(line)
                if not match:
                    raise RamdiskError("invalid profile line %d in %s" % (number, path))
                layer, expert, count = (int(value) for value in match.groups())
                counts["%d:%d" % (layer, expert)] = count
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, RamdiskError):
            raise
        raise RamdiskError("cannot parse profile %s: %s" % (path, exc))
    if fingerprint and fingerprint != model["fingerprint"]:
        raise RamdiskError("profile model fingerprint does not match the selected model")
    compatible = {key: count for key, count in counts.items() if key in model["experts"] and count > 0}
    if not compatible:
        raise RamdiskError("profile contains no experts compatible with this model")
    return os.path.realpath(path), compatible

def _select_partial(model, counts, budget_bytes):
    shard_sizes = {item["name"]: item["size_bytes"] for item in model["shards"]}
    # Experts commonly share the same one- or two-shard closure.  Grouping those
    # closures makes the greedy score both exact and cheap: each candidate gets
    # credit for every newly completed profiled expert, not just the expert that
    # happened to nominate the shard set.
    closure_groups = {}
    for key in sorted(counts):
        closure = frozenset(model["experts"][key]["shards"])
        group = closure_groups.setdefault(closure, {"keys": [], "benefit": 0})
        group["keys"].append(key)
        group["benefit"] += counts[key] * model["experts"][key]["tensor_bytes"]
    selected = set()
    covered_closures = set()
    while True:
        candidates = []
        for closure in sorted(closure_groups, key=lambda value: tuple(sorted(value))):
            if closure in covered_closures:
                continue
            trial = selected | set(closure)
            added = trial - selected
            cost = sum(shard_sizes[name] for name in added)
            if not added or sum(shard_sizes[name] for name in selected) + cost > budget_bytes:
                continue
            newly_covered = [
                other
                for other in closure_groups
                if other not in covered_closures and other.issubset(trial)
            ]
            benefit = sum(closure_groups[item]["benefit"] for item in newly_covered)
            ratio = float(benefit) / float(cost)
            # The sorted closure tuple is the final deterministic tie-breaker.
            candidates.append((ratio, benefit, -cost, tuple(sorted(closure)), trial))
        if not candidates:
            break
        _, _, _, _, selected = max(candidates)
        covered_closures = {
            closure for closure in closure_groups if closure.issubset(selected)
        }
    staged_experts = sorted(
        key
        for key, expert in model["experts"].items()
        if set(expert["shards"]).issubset(selected) and expert["direct_map_eligible"]
    )
    return sorted(selected), staged_experts

def _runtime_reserve(model, ctx, direct_experts, cache_cap=8, kv_slots=1):
    config = model.get("config") or {}
    layers = int(config.get("num_hidden_layers", 0) or 0)
    kv_lora = int(config.get("kv_lora_rank", 0) or 0)
    rope = int(config.get("qk_rope_head_dim", 0) or 0)
    index_dim = int(config.get("index_head_dim", 0) or 0)
    qk_nope = int(config.get("qk_nope_head_dim", 0) or 0)
    v_head = int(config.get("v_head_dim", 0) or 0)
    heads = int(config.get("num_attention_heads", 0) or 0)
    kv_bytes = (layers + 1) * max(1, ctx) * (kv_lora + rope) * 4 * kv_slots
    index_bytes = layers * max(1, ctx) * index_dim * 4 * kv_slots
    attention_scratch = max(1, ctx) * heads * (qk_nope + v_head) * 4
    dense = model["dense_tensor_bytes"]
    direct = set(direct_experts)
    fallback_by_layer = {}
    for key, expert in model["experts"].items():
        if key not in direct:
            fallback_by_layer.setdefault(expert["layer"], []).append(expert["tensor_bytes"])
    fallback_cache = sum(
        min(cache_cap, len(sizes)) * max(sizes)
        for sizes in fallback_by_layer.values()
        if sizes
    )
    max_expert = max((entry["tensor_bytes"] for entry in model["experts"].values()), default=0)
    working_set = min(64, max((len(sizes) for sizes in fallback_by_layer.values()), default=0)) * max_expert
    engine_overhead = max(int(1.2e9), dense // 100)
    return {
        "dense_bytes": dense,
        "kv_bytes": kv_bytes,
        "index_bytes": index_bytes,
        "attention_scratch_bytes": attention_scratch,
        "fallback_cache_bytes": fallback_cache,
        "working_set_bytes": working_set,
        "engine_overhead_bytes": engine_overhead,
    }

def _requested_ids(value, label, allowed, default):
    """Normalize an operator range list without ever widening its effective mask."""
    allowed = sorted(set(int(item) for item in allowed))
    if value is None or value == "":
        selected = sorted(set(int(item) for item in default))
    elif isinstance(value, str):
        if len(value) > 4096:
            raise RamdiskError("%s range list is unreasonably long" % label)
        for token in value.split(","):
            token = token.strip()
            match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
            if not match:
                raise RamdiskError("%s must be a CPU/NUMA range list such as 0-3,8" % label)
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if end < start:
                raise RamdiskError("%s contains a descending range" % label)
            if allowed and (start > allowed[-1] or end > allowed[-1]):
                raise RamdiskError("%s requests IDs outside the effective host mask" % label)
        selected = _parse_range_list(value)
    elif isinstance(value, (list, tuple, set)):
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        ):
            raise RamdiskError("%s must contain non-negative integer IDs" % label)
        selected = sorted(set(value))
    else:
        raise RamdiskError("%s must be a CPU/NUMA range list" % label)
    if not selected:
        raise RamdiskError("%s resolves to an empty effective mask" % label)
    outside = sorted(set(selected) - set(allowed))
    if outside:
        raise RamdiskError(
            "%s requests IDs outside the effective host mask: %s"
            % (label, _format_range_list(outside))
        )
    return selected

def _build_placement(args, hardware, topology):
    """Resolve selected memory nodes and whole-core CPU masks for one plan."""
    online_nodes = sorted(set(int(node) for node in hardware.get("online_nodes", [])))
    node_rows = {
        int(node["id"]): node
        for node in hardware.get("nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), int)
    }
    all_cpus = sorted(
        {
            int(cpu)
            for node in node_rows.values()
            for cpu in node.get("cpus", [])
            if isinstance(cpu, int) and not isinstance(cpu, bool) and cpu >= 0
        }
    )
    effective_nodes = sorted(
        set(int(node) for node in hardware.get("effective_nodes", online_nodes))
        & set(online_nodes)
    )
    effective_cpus = sorted(
        set(int(cpu) for cpu in hardware.get("effective_cpus", all_cpus))
        & set(all_cpus)
    )
    if not effective_nodes:
        raise RamdiskError("the effective cpuset exposes no NUMA memory nodes")
    if not effective_cpus:
        raise RamdiskError("the effective cpuset exposes no CPUs")

    default_nodes = [
        node
        for node in effective_nodes
        if set(node_rows.get(node, {}).get("cpus", [])) & set(effective_cpus)
    ] or effective_nodes
    memory_nodes = _requested_ids(
        getattr(args, "memory_nodes", None),
        "--memory-nodes",
        effective_nodes,
        default_nodes,
    )
    missing_rows = sorted(set(memory_nodes) - set(node_rows))
    if missing_rows:
        raise RamdiskError(
            "hardware discovery has no details for selected NUMA node(s): %s"
            % _format_range_list(missing_rows)
        )
    default_cpus = sorted(
        set(effective_cpus)
        & {
            int(cpu)
            for node in memory_nodes
            for cpu in node_rows.get(node, {}).get("cpus", [])
        }
    ) or effective_cpus
    cpus = _requested_ids(
        getattr(args, "cpu_list", None),
        "--cpu-list",
        effective_cpus,
        default_cpus,
    )
    memory_node_cpus = {
        int(cpu)
        for node in memory_nodes
        for cpu in node_rows.get(node, {}).get("cpus", [])
    }
    remote_cpus = sorted(set(cpus) - memory_node_cpus)
    if topology == "per-node" and remote_cpus:
        raise RamdiskError(
            "per-node --cpu-list includes CPUs outside the selected replica nodes: %s"
            % _format_range_list(remote_cpus)
        )

    raw_groups = hardware.get("core_groups") or [[cpu] for cpu in effective_cpus]
    core_groups = []
    covered = set()
    for raw_group in raw_groups:
        group = sorted(set(int(cpu) for cpu in raw_group) & set(effective_cpus))
        if group and not (set(group) & covered):
            core_groups.append(group)
            covered.update(group)
    core_groups.extend([[cpu] for cpu in effective_cpus if cpu not in covered])
    selected = set(cpus)
    split_groups = [
        group
        for group in core_groups
        if selected.intersection(group) and not set(group).issubset(selected)
    ]
    if split_groups:
        raise RamdiskError(
            "--cpu-list must select whole effective physical cores; split sibling group(s): %s"
            % ", ".join(_format_range_list(group) for group in split_groups)
        )

    engine_cpu_sets = []
    if topology == "interleaved":
        targets = [(None, cpus)]
    else:
        targets = [
            (
                node,
                sorted(set(cpus) & set(node_rows.get(node, {}).get("cpus", []))),
            )
            for node in memory_nodes
        ]
    for node, engine_cpus in targets:
        physical_cores = sum(
            1 for group in core_groups if set(group).issubset(set(engine_cpus))
        )
        engine_cpu_sets.append(
            {
                "node": node,
                "cpus": engine_cpus,
                "cpu_list": _format_range_list(engine_cpus),
                "physical_cores": physical_cores,
            }
        )
    return {
        "memory_nodes": memory_nodes,
        "memory_node_list": _format_range_list(memory_nodes),
        "cpus": cpus,
        "cpu_list": _format_range_list(cpus),
        "engine_cpu_sets": engine_cpu_sets,
        "effective_nodes": effective_nodes,
        "effective_node_list": _format_range_list(effective_nodes),
        "effective_cpus": effective_cpus,
        "effective_cpu_list": _format_range_list(effective_cpus),
        "remote_cpus": remote_cpus,
        "remote_cpu_list": _format_range_list(remote_cpus),
        "memory_policy": (
            "equal-interleave"
            if topology == "interleaved" and len(memory_nodes) > 1
            else "strict-bind"
        ),
        "dimm_control": "informational-only",
    }
