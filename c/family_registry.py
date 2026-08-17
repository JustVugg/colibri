#!/usr/bin/env python3
"""Authoritative model-family registry for Colibri's Python control plane."""

from dataclasses import dataclass
import json
import re
from pathlib import Path


class RegistryError(ValueError):
    pass


class FamilyConfigError(ValueError):
    pass


class UnknownFamilyError(ValueError):
    pass


class PlannerUnsupportedError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FamilyCapabilities:
    tools: bool
    grammar_payload: bool
    audio_payload: bool
    thinking: bool


@dataclass(frozen=True, slots=True)
class FamilyLimits:
    default_context: int
    max_context: int
    default_max_output: int
    interactive_max_output: int
    max_kv_slots: int
    implicit_cap: int
    context_env: str


@dataclass(frozen=True, slots=True)
class PlannerGeometry:
    context_state_bytes: int
    fixed_state_bytes: int
    workspace_bytes: int
    configured_experts: int


@dataclass(frozen=True, slots=True)
class FamilyDescriptor:
    id: str
    model_types: tuple
    display_name: str
    display_scale: str
    engine_artifact: str
    engine_aliases: tuple
    engine_group: str
    internal_arch: str
    build_target: str
    process_names: tuple
    default_model_id: str
    cli_adapter: str
    gateway_adapter: str
    planner_id: str
    planner_geometry: object
    planner_unsupported_reason: str
    expert_inventory: object
    config_section: str
    limits: FamilyLimits
    capabilities: FamilyCapabilities


@dataclass(frozen=True, slots=True)
class ResolvedFamily:
    descriptor: FamilyDescriptor
    model_type: str
    config: dict
    family_config: dict
    model_dir: str


def _required_int(config, key, family, minimum=1):
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{family}: missing or invalid planning key {key!r}")
    return value


def _optional_int(config, key, default=0, minimum=0):
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid planning key {key!r}")
    return value


def _glm_geometry(config, context, _model_dir):
    layers = _required_int(config, "num_hidden_layers", "glm") + 1
    experts = _required_int(config, "n_routed_experts", "glm")
    kv_lora = _required_int(config, "kv_lora_rank", "glm")
    rope = _required_int(config, "qk_rope_head_dim", "glm")
    heads = _required_int(config, "num_attention_heads", "glm")
    nope = _required_int(config, "qk_nope_head_dim", "glm")
    value = _required_int(config, "v_head_dim", "glm")
    state = layers * context * (kv_lora + rope) * 4
    index_dim = _optional_int(config, "index_head_dim", 0)
    if index_dim and config.get("_colibri_indexer_present", False):
        kinds = config.get("indexer_types")
        if isinstance(kinds, list):
            active = sum(kind == "full" for kind in kinds[:layers - 1])
        else:
            frequency = max(1, _optional_int(config, "index_topk_freq", 1, 1))
            offset = _optional_int(config, "index_skip_topk_offset", 2)
            active = 0
            for layer in range(layers - 1):
                index_value = max(layer - offset + 1, 0)
                active += index_value % frequency == 0
        state += active * context * index_dim * 4
    workspace = context * heads * (nope + value) * 4
    return PlannerGeometry(state, 0, workspace, experts)


_GLM_EXPERT = re.compile(
    r"(?:^|\.)model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
)
_KIMI_EXPERT = re.compile(
    r"^(?:language_model\.)?model\.layers\.(\d+)\.block_sparse_moe\."
    r"experts\.(\d+)\."
)
_V4_EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.")
_INKLING_EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\."
    r"(?:gate_up_proj|down_proj)(?:\.|$)"
)


def _individual_expert_inventory(pattern):
    def inventory(name, size, _config):
        match = pattern.search(name)
        if match is None:
            return ()
        return ((int(match.group(1)), int(match.group(2)), size),)
    return inventory


def _inkling_expert_inventory(name, size, config):
    match = _INKLING_EXPERT.fullmatch(name)
    if match is None:
        return ()
    experts = _required_int(config, "n_routed_experts", "inkling")
    if size % experts:
        raise ValueError(f"inkling: fused expert tensor {name!r} is not divisible "
                         f"by {experts} experts")
    per_expert = size // experts
    layer = int(match.group(1))
    return tuple((layer, expert, per_expert) for expert in range(experts))


COMMON_CAP = FamilyCapabilities(False, False, False, True)

FAMILIES = (
    FamilyDescriptor(
        "glm", ("glm_moe_dsa", "glm5_moe", "glm"), "GLM-5.2", "744B",
        "colibri", ("glm",), "colibri-core", "glm", "colibri",
        ("colibri", "glm"), "glm-5.2-colibri", "glm", "glm", "glm_mla",
        _glm_geometry, "", _individual_expert_inventory(_GLM_EXPERT), "root",
        FamilyLimits(4096, 1048576, 1024, 16384, 16, 0, "CTX"),
        FamilyCapabilities(True, True, False, True)),
    FamilyDescriptor(
        "inkling", ("inkling_mm_model", "inkling"), "Inkling", "975B",
        "inkling", (), "inkling", "inkling", "inkling", ("inkling",),
        "inkling-colibri", "inkling", "inkling", "inkling_hybrid", None,
        "Inkling planning awaits a measured hybrid GQA/sconv runtime adapter",
        _inkling_expert_inventory, "text_config",
        FamilyLimits(8192, 1048576, 1024, 1024, 1, 8, "CTX_MAX"),
        FamilyCapabilities(False, False, True, True)),
    FamilyDescriptor(
        "kimi", ("kimi_k3",), "Kimi K3", "2.8T", "kimi_k3", (), "kimi_k3",
        "kimi_k3", "kimi_k3", ("kimi_k3",), "kimi-k3-colibri", "kimi", "kimi",
        "kimi_hybrid", None,
        "Kimi planning awaits measured KDA recurrent-state and native MXFP4 reserves",
        _individual_expert_inventory(_KIMI_EXPERT), "text_config",
        FamilyLimits(8192, 1048576, 1024, 1024, 1, 8, "K3_MAXT"), COMMON_CAP),
    FamilyDescriptor(
        "olmoe", ("olmoe",), "OLMoE", "7B", "olmoe", (), "olmoe", "olmoe",
        "olmoe", ("olmoe",), "olmoe-colibri", "olmoe", "olmoe", "olmoe_gqa",
        None, "OLMoE planning awaits a measured runtime/workspace reserve",
        _individual_expert_inventory(_GLM_EXPERT), "root",
        FamilyLimits(4096, 4096, 1024, 1024, 1, 8, "CTX"),
        FamilyCapabilities(False, False, False, False)),
    FamilyDescriptor(
        "deepseek_v4", ("deepseek_v4",), "DeepSeek V4 Flash", "284B",
        "deepseek_v4", (), "deepseek_v4", "deepseek_v4", "deepseek-v4",
        ("deepseek_v4",), "deepseek-v4-colibri", "deepseek_v4", "deepseek_v4",
        "deepseek_v4", None,
        "DeepSeek V4 uses its C resident-tier planner; Python parity is not yet proven",
        _individual_expert_inventory(_V4_EXPERT), "root",
        FamilyLimits(4096, 1048576, 1024, 16384, 1, 8, "CTX"),
        FamilyCapabilities(True, False, False, True)),
)


def _build_registry(families):
    by_id = {}
    by_type = {}
    identities = set()
    for family in families:
        if not re.fullmatch(r"[a-z0-9_]+", family.id) or family.id in by_id:
            raise RegistryError(f"invalid or duplicate family id: {family.id!r}")
        if (not family.model_types or
                (not callable(family.planner_geometry) and
                 not family.planner_unsupported_reason) or
                not callable(family.expert_inventory)):
            raise RegistryError(f"incomplete family descriptor: {family.id}")
        identity = (family.engine_artifact, family.internal_arch)
        if identity in identities:
            raise RegistryError(f"duplicate engine identity: {identity}")
        identities.add(identity)
        by_id[family.id] = family
        for model_type in family.model_types:
            normalized = _normalize_model_type(model_type)
            if normalized in by_type:
                raise RegistryError(f"duplicate model_type alias: {normalized}")
            by_type[normalized] = family
        if (family.limits.default_context < 1 or family.limits.max_context < family.limits.default_context or
                family.limits.default_max_output < 1 or family.limits.interactive_max_output < 1 or
                family.limits.max_kv_slots < 1 or family.limits.implicit_cap < 0):
            raise RegistryError(f"invalid limits for family: {family.id}")
    return by_id, by_type


def _normalize_model_type(model_type):
    if not isinstance(model_type, str) or not model_type.strip():
        raise FamilyConfigError("config.json has no non-empty string model_type")
    return model_type.strip().lower()


_BY_ID, _BY_TYPE = _build_registry(FAMILIES)


def all_families():
    return FAMILIES


def family_ids():
    return tuple(family.id for family in FAMILIES)


def family_by_id(family_id):
    try:
        return _BY_ID[family_id]
    except KeyError as error:
        raise UnknownFamilyError(f"unknown model family: {family_id}") from error


def family_for_config(config):
    if not isinstance(config, dict):
        raise FamilyConfigError("config.json is not a JSON object")
    model_type = _normalize_model_type(config.get("model_type"))
    try:
        return _BY_TYPE[model_type]
    except KeyError as error:
        raise UnknownFamilyError(f"unsupported model_type: {model_type}") from error


def resolve_model(model_dir):
    model = Path(model_dir).expanduser().resolve()
    path = model / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FamilyConfigError(f"cannot read config.json: {model}") from error
    except json.JSONDecodeError as error:
        raise FamilyConfigError(f"invalid config.json: {error}") from error
    family = family_for_config(config)
    family_config = config
    if family.config_section == "text_config":
        family_config = config.get("text_config", config)
        if not isinstance(family_config, dict):
            raise FamilyConfigError(f"{family.id}: text_config is not an object")
    return ResolvedFamily(family, _normalize_model_type(config.get("model_type")),
                          config, family_config, str(model))


def planner_geometry(resolved, context):
    if isinstance(context, bool) or not isinstance(context, int) or context < 1:
        raise ValueError("context must be a positive integer")
    if context > resolved.descriptor.limits.max_context:
        raise ValueError(f"{resolved.descriptor.id}: context {context} exceeds "
                         f"the registered maximum {resolved.descriptor.limits.max_context}")
    if resolved.descriptor.planner_geometry is None:
        raise PlannerUnsupportedError(
            f"{resolved.descriptor.display_name}: "
            f"{resolved.descriptor.planner_unsupported_reason}")
    geometry = resolved.descriptor.planner_geometry(
        resolved.family_config, context, resolved.model_dir)
    if not isinstance(geometry, PlannerGeometry) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (geometry.context_state_bytes, geometry.fixed_state_bytes,
                          geometry.workspace_bytes, geometry.configured_experts)):
        raise RegistryError(f"invalid planner geometry for {resolved.descriptor.id}")
    if geometry.configured_experts < 1:
        raise ValueError(f"{resolved.descriptor.id}: configured expert count is zero")
    return geometry


def expert_contributions(resolved, name, size):
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("tensor size must be a non-negative integer")
    contributions = resolved.descriptor.expert_inventory(
        name, size, resolved.family_config)
    for layer, expert, byte_count in contributions:
        if layer < 0 or expert < 0 or byte_count < 0:
            raise RegistryError(f"invalid expert inventory for {resolved.descriptor.id}")
    return contributions


def public_metadata(family):
    return {
        "id": family.id,
        "model_types": list(family.model_types),
        "display_name": family.display_name,
        "display_scale": family.display_scale,
        "engine_artifact": family.engine_artifact,
        "engine_aliases": list(family.engine_aliases),
        "engine_group": family.engine_group,
        "internal_arch": family.internal_arch,
        "build_target": family.build_target,
        "process_names": list(family.process_names),
        "default_model_id": family.default_model_id,
        "cli_adapter": family.cli_adapter,
        "gateway_adapter": family.gateway_adapter,
        "planner_id": family.planner_id,
        "limits": {
            "default_context": family.limits.default_context,
            "max_context": family.limits.max_context,
            "default_max_output": family.limits.default_max_output,
            "interactive_max_output": family.limits.interactive_max_output,
            "max_kv_slots": family.limits.max_kv_slots,
            "implicit_cap": family.limits.implicit_cap,
            "context_env": family.limits.context_env,
        },
        "capabilities": {
            "tools": family.capabilities.tools,
            "grammar_payload": family.capabilities.grammar_payload,
            "audio_payload": family.capabilities.audio_payload,
            "thinking": family.capabilities.thinking,
        },
    }
