"""Pure profile, capacity, and placement planning primitives."""

from __future__ import print_function

import copy
import json
import math
import os
import re
import subprocess

from .common import (
    DEFAULT_MOUNT_ROOT,
    GIB,
    MANIFEST_VERSION,
    MIB,
    PLAN_SCHEMA,
    PROFILE_LINE_RE,
    RamdiskError,
    _format_range_list,
    _parse_range_list,
    _path_is_below,
    _path_without_symlinks,
    _utc_now,
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


def build_plan(
    args,
    hardware=None,
    model=None,
    *,
    discover_hardware,
    scan_model,
    load_profile=_load_profile,
    select_partial=_select_partial,
    runtime_reserve=_runtime_reserve,
    build_placement=_build_placement,
    reusable_empty_mountpoint,
    filesystem_for_path,
    state_root,
    manifest_path,
    benchmarks_path,
    current_euid,
    get_platform_ops,
):
    """Build a RAM-disk deployment plan using facade-supplied host services."""
    hardware = copy.deepcopy(hardware or discover_hardware())
    model = model or scan_model(args.model)
    mode = getattr(args, "mode", "full")
    topology = getattr(args, "topology", "interleaved")
    capacity_gb = getattr(args, "capacity_gb", None)
    if mode not in ("full", "partial") or topology not in ("interleaved", "per-node"):
        raise RamdiskError("invalid RAM-disk mode or topology")
    placement = build_placement(args, hardware, topology)
    if capacity_gb is not None and (
        isinstance(capacity_gb, bool)
        or not isinstance(capacity_gb, (int, float))
        or not math.isfinite(capacity_gb)
        or capacity_gb <= 0
    ):
        raise RamdiskError("--capacity-gb must be a finite positive number")
    raw_ctx = getattr(args, "ctx", 0)
    if isinstance(raw_ctx, bool) or not isinstance(raw_ctx, int) or raw_ctx < 0:
        raise RamdiskError("--ctx must be zero (default) or a positive integer")
    raw_parallel = getattr(args, "parallel", 2)
    if (
        isinstance(raw_parallel, bool)
        or not isinstance(raw_parallel, int)
        or not 1 <= raw_parallel <= 64
    ):
        raise RamdiskError("--parallel must be an integer between 1 and 64")
    capacity_bytes = int(capacity_gb * GIB) if capacity_gb is not None else model["total_shard_bytes"]
    profile_path = None
    counts = None
    if mode == "full":
        selected = list(model["shard_names"])
        direct_experts = sorted(
            key for key, entry in model["experts"].items() if entry["direct_map_eligible"]
        )
    else:
        if not capacity_gb or capacity_gb <= 0:
            raise RamdiskError("partial staging requires a positive --capacity-gb budget")
        profile_path, counts = load_profile(getattr(args, "profile", None), model)
        selected, direct_experts = select_partial(model, counts, capacity_bytes)
        if not selected:
            raise RamdiskError("no complete shard closure fits the partial staging budget")
    resident_experts = sorted(
        key
        for key, expert in model["experts"].items()
        if set(expert["shards"]).issubset(selected)
    )
    shard_sizes = {item["name"]: item["size_bytes"] for item in model["shards"]}
    staged_bytes = sum(shard_sizes[name] for name in selected)
    managed_ctx = int(raw_ctx or 4096)
    managed_cache_cap = 8
    managed_kv_slots = 1
    managed_reserve = runtime_reserve(
        model,
        managed_ctx,
        direct_experts,
        cache_cap=managed_cache_cap,
        kv_slots=managed_kv_slots,
    )
    # The benchmark contract includes SSD and tmpfs-through-slab baselines even
    # for a fully staged model. Those paths need the ordinary cap/LRU working
    # set in addition to the resident tmpfs copy, so admission uses the larger
    # of managed-direct and non-RAMMAP benchmark runtime projections.
    benchmark_reserve = runtime_reserve(
        model,
        managed_ctx,
        [],
        cache_cap=managed_cache_cap,
        kv_slots=managed_kv_slots,
    )
    managed_runtime_bytes = sum(managed_reserve.values())
    benchmark_runtime_bytes = sum(benchmark_reserve.values())
    runtime_bytes = max(managed_runtime_bytes, benchmark_runtime_bytes)
    memory = hardware["memory"]
    selected_nodes = list(placement["memory_nodes"])
    selected_node_rows = [
        node for node in hardware.get("nodes", []) if node.get("id") in selected_nodes
    ]
    selected_total = sum(
        int(node.get("memory_total_bytes", 0)) for node in selected_node_rows
    ) or int(memory["total_bytes"])
    selected_available = sum(
        int(node.get("memory_available_bytes", 0)) for node in selected_node_rows
    ) or int(memory["available_bytes"])
    cgroup_memory = hardware.get("cgroup_memory") or {}
    cgroup_available = cgroup_memory.get("available_bytes")
    if (
        not isinstance(cgroup_available, int)
        or isinstance(cgroup_available, bool)
        or cgroup_available < 0
    ):
        cgroup_available = None
    cgroup_high_available = cgroup_memory.get("high_available_bytes")
    if (
        not isinstance(cgroup_high_available, int)
        or isinstance(cgroup_high_available, bool)
        or cgroup_high_available < 0
    ):
        cgroup_high_available = None
    effective_available = (
        min(selected_available, cgroup_available)
        if cgroup_available is not None
        else selected_available
    )
    global_margin = max(selected_total // 10, 16 * GIB)
    page_tables = int(math.ceil(float(staged_bytes + runtime_bytes) / 512.0))
    required_global = staged_bytes + runtime_bytes + page_tables + global_margin
    blockers = []
    warnings = []
    if cgroup_memory.get("error"):
        blockers.append(
            "cannot validate cgroup memory headroom: %s"
            % cgroup_memory["error"]
        )
    if not hardware["linux"]:
        blockers.append("coli ramdisk is supported only on Linux")
    if not hardware["tmpfs"]["supported"]:
        blockers.append("tmpfs is not available in /proc/filesystems")
    allow_swappable = bool(getattr(args, "allow_swappable", False))
    if not hardware["tmpfs"]["noswap_supported"] and not allow_swappable:
        blockers.append("this kernel does not advertise tmpfs noswap; use --allow-swappable only if accepted")
    if hardware["swap"]["used_bytes"]:
        warnings.append("swap is already in use; managed commands never run swapoff")
    if topology == "per-node" and not hardware.get("numactl"):
        blockers.append("per-node topology requires numactl")
    if topology == "per-node":
        engine_cpu_sets = {
            item["node"]: item for item in placement["engine_cpu_sets"]
        }
        for node in selected_node_rows:
            if not node.get("cpus"):
                blockers.append(
                    "NUMA node %d has no online CPUs and cannot host a node-local engine"
                    % node["id"]
                )
            elif not engine_cpu_sets[node["id"]]["cpus"]:
                blockers.append(
                    "NUMA node %d has no selected whole-core CPUs for its replica"
                    % node["id"]
                )
    if capacity_bytes < staged_bytes:
        blockers.append("selected shard closures exceed the staging budget")
    if topology == "interleaved":
        if selected_available < required_global:
            blockers.append(
                "selected NUMA nodes would breach the runtime/OS reserve"
            )
        if placement["remote_cpus"]:
            warnings.append(
                "selected engine CPUs outside the memory-node mask will perform "
                "intentional remote NUMA access: %s" % placement["remote_cpu_list"]
            )
        if len(selected_nodes) > 1:
            warnings.append(
                "Linux interleave may fall back outside the selected nodes under severe "
                "memory pressure; Colibri reserves headroom and verifies initial page placement"
            )
    else:
        for node in selected_node_rows:
            margin = max(node["memory_total_bytes"] // 10, 8 * GIB)
            node_page_tables = int(math.ceil(float(staged_bytes + runtime_bytes) / 512.0))
            required = staged_bytes + runtime_bytes + node_page_tables + margin
            node["required_bytes"] = required
            node["reserve_bytes"] = margin
            if node["memory_available_bytes"] < required:
                blockers.append("NUMA node %d cannot hold its replica and reserve" % node["id"])
    if mode == "partial":
        total_count = sum(counts.values())
        covered_profile = [
            key
            for key in counts
            if set(model["experts"][key]["shards"]).issubset(selected)
        ]
        staged_count = sum(counts[key] for key in covered_profile)
        coverage = float(staged_count) / total_count if total_count else 0.0
        predicted_avoided = sum(
            counts[key] * model["experts"][key]["tensor_bytes"]
            for key in covered_profile
        )
        pin_selected = []
        pin_bytes = 0
        for key in sorted(counts, key=lambda item: (-counts[item], item)):
            expert_bytes = model["experts"][key]["tensor_bytes"]
            if pin_bytes + expert_bytes <= capacity_bytes:
                pin_selected.append(key)
                pin_bytes += expert_bytes
        pin_count = sum(counts[key] for key in pin_selected)
        pin_comparison = {
            "budget_bytes": capacity_bytes,
            "selected_experts": pin_selected,
            "resident_expert_bytes": pin_bytes,
            "coverage": float(pin_count) / total_count if total_count else 0.0,
        }
    else:
        covered_profile = []
        coverage = 1.0
        predicted_avoided = sum(
            model["experts"][key]["tensor_bytes"] for key in resident_experts
        )
        pin_comparison = None
    direct_bytes = sum(model["experts"][key]["tensor_bytes"] for key in direct_experts)
    resident_expert_bytes = sum(
        model["experts"][key]["tensor_bytes"] for key in resident_experts
    )
    efficiency = float(resident_expert_bytes) / staged_bytes if staged_bytes else 0.0
    mount_root = os.path.abspath(os.path.expanduser(getattr(args, "mount_root", DEFAULT_MOUNT_ROOT)))
    invoking_euid = current_euid()
    mount_root_preexisting = os.path.isdir(mount_root) and not os.path.islink(mount_root)
    forbidden_roots = {
        "/",
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/home",
        "/lib",
        "/lib64",
        "/mnt",
        "/opt",
        "/proc",
        "/root",
        "/run",
        "/sbin",
        "/srv",
        "/sys",
        "/tmp",
        "/usr",
        "/var",
        os.path.normpath(os.path.expanduser("~")),
    }
    if os.path.normpath(mount_root) in forbidden_roots:
        blockers.append("mount root is a protected broad directory")
    try:
        under_mnt = os.path.commonpath([mount_root, "/mnt"]) == "/mnt" and mount_root != "/mnt"
    except ValueError:
        under_mnt = False
    if not under_mnt:
        blockers.append("v1 managed mount roots must be below /mnt")
    if os.path.realpath(mount_root) != mount_root:
        blockers.append("mount root path must not traverse symbolic links")
    if os.path.lexists(mount_root):
        if os.path.islink(mount_root) or not os.path.isdir(mount_root):
            blockers.append("mount root exists but is not a real directory")
        elif os.stat(mount_root).st_mode & 0o022:
            blockers.append("existing mount root must not be group/world writable")
        elif invoking_euid != 0 and os.access(mount_root, os.W_OK):
            blockers.append("existing mount root must not be writable by the invoking user")
        elif topology == "interleaved":
            try:
                entries = os.listdir(mount_root)
                reusable = bool(entries) and all(
                    re.fullmatch(r"node\d+", name)
                    and reusable_empty_mountpoint(os.path.join(mount_root, name))
                    for name in entries
                )
                if entries and not reusable:
                    blockers.append("interleaved mount root must be absent or empty")
                elif reusable:
                    warnings.append(
                        "interleaved mount will temporarily cover verified empty node mountpoints from an earlier topology"
                    )
            except OSError as exc:
                blockers.append("cannot inspect mount root: %s" % exc)
    # Every existing parent below /mnt must be non-writable by this user, so a
    # second process cannot exchange a directory for a symlink between review
    # and the privileged mount(8) call. X-mount.mkdir creates absent parents.
    if under_mnt:
        parent = os.path.dirname(mount_root)
        while parent.startswith("/mnt"):
            if os.path.lexists(parent):
                if os.path.islink(parent) or not os.path.isdir(parent):
                    blockers.append("mount root has an unsafe parent: %s" % parent)
                    break
                if invoking_euid != 0 and os.access(parent, os.W_OK):
                    blockers.append("mount root parent is writable by the invoking user: %s" % parent)
                    break
            if parent == "/mnt":
                break
            parent = os.path.dirname(parent)
    model_path = os.path.normpath(model["path"])
    try:
        if os.path.commonpath([mount_root, model_path]) in (mount_root, model_path):
            blockers.append("mount root must not contain or be contained by the canonical model")
    except ValueError:
        blockers.append("mount root and model path are on incompatible path roots")
    nodes = selected_nodes
    requested_thp = getattr(args, "thp", "auto") or "auto"
    thp = (
        "within_size" if hardware["thp"]["within_size_supported"] else "advise"
    ) if requested_thp == "auto" else requested_thp
    if thp == "within_size" and not hardware["thp"]["within_size_supported"]:
        warnings.append("THP within_size is not advertised; mount will fall back to advise if rejected")
    if thp == "advise" and not hardware["thp"]["advise_supported"]:
        blockers.append("tmpfs THP advise mode is not available")
    replicas = [None] if topology == "interleaved" else nodes
    replica_count = len(replicas)
    mounts = []
    for node in replicas:
        path = mount_root if node is None else os.path.join(mount_root, "node%d" % node)
        # A contiguous range avoids an option-separator comma in mount(8)'s
        # ``-o`` string on the overwhelmingly common 0..N online-node layout.
        node_list = _format_range_list(nodes)
        if "," in node_list:
            node_list = node_list.replace(",", "\\,")
        # Prevent ordinal remapping while reviewed nodes remain allowed.
        # Without ``static``, Linux maps the policy's ordinal nodes into a new
        # cpuset; Start/Benchmark separately refuse every effective-mask drift.
        policy = (
            "interleave=static:" + node_list
            if node is None and len(nodes) > 1
            else "bind=static:%d" % (nodes[0] if node is None else node)
        )
        mounts.append(
            {
                "node": node,
                "path": path,
                "path_preexisting": os.path.isdir(path) and not os.path.islink(path),
                "policy": policy,
                "size_bytes": max(staged_bytes + max(64 * MIB, staged_bytes // 100), 64 * MIB),
            }
        )
        if os.path.lexists(path):
            if os.path.islink(path) or not os.path.isdir(path):
                blockers.append("managed mount path exists but is not a real directory: %s" % path)
            elif invoking_euid != 0 and os.access(path, os.W_OK):
                blockers.append("managed mount path is writable by the invoking user: %s" % path)
            else:
                try:
                    if os.listdir(path):
                        blockers.append("managed mount path is not empty: %s" % path)
                except OSError as exc:
                    blockers.append("cannot inspect managed mount path %s: %s" % (path, exc))
    durable_state = {}
    try:
        durable_state = {
            "root": state_root(),
            "manifest": manifest_path(),
            "benchmarks": benchmarks_path(),
        }
        for label, durable_path in durable_state.items():
            if not _path_without_symlinks(durable_path):
                blockers.append("durable %s path must not traverse symbolic links" % label)
            if _path_is_below(durable_path, mount_root, allow_equal=True):
                blockers.append("durable %s path must be outside every volatile mount" % label)
            if hardware["linux"] and get_platform_ops().is_linux:
                filesystem = filesystem_for_path(durable_path)
                if filesystem in ("tmpfs", "ramfs"):
                    blockers.append(
                        "durable %s path is on volatile %s; use an SSD-backed XDG state directory"
                        % (label, filesystem)
                    )
    except (RamdiskError, OSError, subprocess.SubprocessError) as exc:
        blockers.append(str(exc))
    total_staged_bytes = staged_bytes * replica_count
    total_runtime_bytes = runtime_bytes * replica_count
    total_page_table_bytes = page_tables * replica_count
    if topology == "per-node":
        total_os_margin = sum(
            int(node.get("reserve_bytes", 0)) for node in selected_node_rows
        )
        total_required = sum(
            int(node.get("required_bytes", 0)) for node in selected_node_rows
        )
        if selected_available < total_required:
            blockers.append("available memory cannot hold all per-node replicas and reserves")
    else:
        total_os_margin = global_margin
        total_required = required_global
    if cgroup_available is not None and cgroup_available < total_required:
        blockers.append(
            "cgroup memory hard-limit headroom cannot hold the staged copies, "
            "managed runtime, and reserve"
        )
    if (
        cgroup_high_available is not None
        and cgroup_high_available < total_required
    ):
        warnings.append(
            "cgroup memory.high headroom is below the projected deployment; "
            "staging or runtime may be heavily reclaimed/throttled"
        )
    return {
        "schema": PLAN_SCHEMA,
        "version": MANIFEST_VERSION,
        "created_at": _utc_now(),
        "mode": mode,
        "topology": topology,
        "mount_root": mount_root,
        "capacity_bytes": capacity_bytes,
        "model": {
            "path": model["path"],
            "fingerprint": model["fingerprint"],
            "fingerprint_algorithm": model["fingerprint_algorithm"],
            "shard_count": len(model["shards"]),
            "total_shard_bytes": model["total_shard_bytes"],
            "complete_experts": model["complete_experts"],
        },
        "profile": {
            "path": profile_path,
            "coverage": coverage,
            "staging_efficiency": efficiency,
            "predicted_expert_bytes_avoided": predicted_avoided,
            "predicted_expert_bytes_avoided_per_staged_byte": (
                float(predicted_avoided) / staged_bytes if staged_bytes else 0.0
            ),
            "covered_experts": covered_profile,
            "pin_comparison": pin_comparison,
        },
        "staging": {
            "selected_shards": selected,
            "linked_shards": sorted(set(model["shard_names"]) - set(selected)),
            "staged_bytes": staged_bytes,
            "staged_experts": resident_experts,
            "staged_expert_count": len(resident_experts),
            "staged_expert_bytes": resident_expert_bytes,
            "direct_mapped_experts": direct_experts,
            "direct_mapped_expert_count": len(direct_experts),
            "direct_mapped_bytes": direct_bytes,
            "replica_count": replica_count,
            "total_staged_bytes": total_staged_bytes,
        },
        "reserve": {
            "runtime": managed_reserve,
            "benchmark_runtime": benchmark_reserve,
            "managed_runtime_bytes": managed_runtime_bytes,
            "benchmark_runtime_bytes": benchmark_runtime_bytes,
            "runtime_bytes": runtime_bytes,
            "page_table_bytes": page_tables,
            "os_margin_bytes": global_margin,
            "required_global_bytes": required_global,
            "available_bytes": effective_available,
            "host_available_bytes": selected_available,
            "cgroup_available_bytes": cgroup_available,
            "cgroup_high_available_bytes": cgroup_high_available,
            "total_runtime_bytes": total_runtime_bytes,
            "total_page_table_bytes": total_page_table_bytes,
            "total_os_margin_bytes": total_os_margin,
            "total_required_bytes": total_required,
        },
        "placement": placement,
        "hardware": hardware,
        "mounts": mounts,
        "mount_root_preexisting": mount_root_preexisting,
        "mount_options": {
            "noswap": hardware["tmpfs"]["noswap_supported"],
            "allow_swappable": allow_swappable,
            "thp": thp,
            "fixed": ["noatime", "nodev", "nosuid", "noexec", "mode=0700"],
        },
        "prefault": int(
            getattr(args, "prefault", None)
            if getattr(args, "prefault", None) is not None
            else mode == "full"
        ),
        "parallel": raw_parallel,
        "managed_runtime": {
            "ctx": managed_ctx,
            "kv_slots": managed_kv_slots,
            "cache_cap": managed_cache_cap,
            "autopin": 0,
            "cap_raise": 0,
        },
        "blockers": sorted(set(blockers)),
        "warnings": warnings,
        "durable_state": durable_state,
        # Internal source identities are retained in a plan used by prepare but
        # omitted by the compact human renderer only.
        "source_shards": model["shards"],
    }
