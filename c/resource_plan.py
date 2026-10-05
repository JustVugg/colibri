#!/usr/bin/env python3
"""Hardware and model placement planning for colibri's disk/RAM/VRAM tiers."""

import collections
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from family_registry import (expert_contributions, planner_geometry,
                             fixed_resident_contribution, resident_contribution,
                             trunk_contribution,
                             resolve_model)


GB = 1_000_000_000
EXPERT_RE = re.compile(r"(?:model\.)?layers\.(\d+)\.(?:mlp|ffn)\.experts\.(\d+)\.")

# Cgroup and procfs inputs are tiny kernel-maintained text files. Bound every
# read anyway: the planner may run in a container with a synthetic or partially
# mounted cgroup tree, and admission must never turn an unexpected file into an
# unbounded read or an unbounded hierarchy walk.
_CGROUP_VALUE_MAX_BYTES = 128
_CGROUP_STAT_MAX_BYTES = 64 * 1024
_CGROUP_MEMBERSHIP_MAX_BYTES = 64 * 1024
_CGROUP_MOUNTINFO_MAX_BYTES = 4 * 1024 * 1024
_CGROUP_MAX_ANCESTORS = 64
_MEMINFO_MAX_BYTES = 256 * 1024
_CGROUP_UINT64_MAX = (1 << 64) - 1
_CGROUP_UINT64_DIGITS = len(str(_CGROUP_UINT64_MAX))
# cgroup v1 represents "unlimited" with a page-aligned value just below the
# signed 64-bit maximum (commonly 9223372036854771712). No consumer host can
# provide an exbibyte of usable RAM, so this conservative threshold also covers
# the other kernel sentinel variants without confusing a real finite limit.
_CGROUP_V1_UNLIMITED_MIN = 1 << 60


class CgroupError(ValueError):
    """A present cgroup or procfs input cannot be trusted for memory admission."""


class CgroupFormatError(CgroupError):
    """A present procfs or cgroup file does not have the kernel's grammar."""


class CgroupAccessError(CgroupError):
    """A present cgroup input cannot be read completely or mapped soundly."""


@dataclass(frozen=True)
class _CgroupMemberships:
    v2: tuple | None
    v1_memory: tuple | None


@dataclass(frozen=True)
class _CgroupMount:
    mount_id: int
    parent_id: int
    root: tuple
    mount_point: Path
    fs_type: str
    options: frozenset


# analyze_model() scans every shard header + regex-matches ~116k tensor names on the
# 372 GB model; it reruns on every `coli plan/doctor/tune/run --auto-tier`. Its output
# is a pure function of each shard's and config.json's (size, mtime), so cache it to a
# sidecar and self-invalidate on any change. Best-effort: any read/write failure falls
# straight back to a full recompute (see analyze_model). Sits alongside .coli_usage/.coli_ssd.
_ANALYSIS_CACHE_NAME = ".coli_analysis.json"
_ANALYSIS_CACHE_VERSION = 11


def _dense_in_ram(descriptor, on_disk_bytes):
    """I byte che i pesi densi occuperanno, non quelli che occupano sul disco.

    Coincidono per chi li carica come stanno; una famiglia che li riquantizza
    al caricamento lo dichiara nel registro. Un errore qui non e' cosmetico:
    e' la differenza fra dire a qualcuno che il modello ci sta e dirgli di no."""
    # GLM53's quantization is per matrix, with F32 vectors/vision and an int8
    # fallback for narrow rows. A whole-checkpoint ratio cannot represent it;
    # _glm53_dense_tensors prices its source inventory below, outside the cache.
    if descriptor.id == "glm53":
        return on_disk_bytes
    ratio = getattr(descriptor, "dense_load_ratio", None)
    if ratio is None:
        return on_disk_bytes
    try:
        return max(0, int(ratio(on_disk_bytes)))
    except Exception:
        return on_disk_bytes


# A token embedding table: "embed_tokens", DeepSeek's "embed", "tok_embeddings",
# "wte"; not a vision tower's patch or position embedding.
_EMBED_TABLE = re.compile(r"(^|\.)(embed_tokens|embed|tok_embeddings|wte|word_embeddings)\.weight$")


def _vk_flag(env, name):
    value = (env.get(name) or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return 1


# The families whose dense chain is on by default on an integrated GPU with the
# expert tier (coli_vk_chain_decide's measured `igpu`, docs/vulkan.md).
_VK_IGPU_CHAIN_ON = ("qwen36", "olmoe")
_VK_DENSE_HOST_FAMILIES = frozenset((
    "qwen36", "qwen38", "glm", "glm53", "inkling", "kimi", "mimo",
    "olmoe", "deepseek_v4", "deepseek_v41",
))


def _vk_device(env, vulkan):
    if vulkan is not None or not _vk_flag(env, "COLI_VULKAN"):
        return vulkan
    try:
        from setup_hw import best_vulkan_device, detect_vulkan
        return best_vulkan_device(detect_vulkan().get("devices", []))
    except Exception:
        return None


def _vk_chain_dense(family_id, env, vulkan):
    """(chain, dense): whether the engine runs its dense chain, and its per-matrix
    path, on the Vulkan device (coli_vk_chain_decide, coli_vk_dense_decide); both
    False when it does not open one."""
    if (family_id not in _VK_DENSE_HOST_FAMILIES or
            not _vk_flag(env, "COLI_VULKAN") or _vk_flag(env, "COLI_CUDA")):
        return False, False
    kind = (vulkan or {}).get("type")
    if kind not in ("discrete", "integrated", "cpu", "virtual", "other"):
        return False, False
    tier = _vk_flag(env, "COLI_VK_TIER") != 0
    chain, dense = _vk_flag(env, "COLI_VK_CHAIN"), _vk_flag(env, "COLI_VK_DENSE")
    if chain is None:
        chain = (kind not in ("integrated", "cpu") or
                 kind == "integrated" and tier and family_id in _VK_IGPU_CHAIN_ON)
    if dense is None:
        # colibri.c passes default=0 to coli_vk_dense_decide; its chain is
        # independent, but disabling that chain does not enable per-matrix GEMVs.
        dense = family_id != "glm" and (kind not in ("integrated", "cpu") or not tier)
    return bool(chain), bool(dense)


def _vk_dense_active(family_id, env, vulkan):
    return any(_vk_chain_dense(family_id, env, vulkan))


def _text_weight_name(name):
    return name.removeprefix("language_model.").replace("model.language_model.", "model.", 1)


_Q38_TRUNK_COMPONENTS = {
    "self_attn.q_proj": "attnq", "self_attn.k_proj": "attnk",
    "self_attn.v_proj": "attnv", "self_attn.o_proj": "attno",
    "self_attn.indexer.index_qk_proj": "qsaidx",
    "linear_attn.in_proj_qkv": "dnqkv", "linear_attn.in_proj_z": "dnz",
    "linear_attn.out_proj": "dnout", "mlp.gate": "router",
    "mlp.shared_expert.gate_proj": "shg", "mlp.shared_expert.up_proj": "shu",
    "mlp.shared_expert.down_proj": "shd",
}
for _block, _tag in (("attn", "hca"), ("mlp", "hcm")):
    for _part, _suffix in (("input_mix_weight_down", "d"),
                           ("input_mix_weight_up", "u"), ("block_inject_weight", "i")):
        _Q38_TRUNK_COMPONENTS[f"{_block}_hyper_connection.{_part}"] = _tag + _suffix


def _layer_component(name):
    match = re.fullmatch(r"model\.layers\.\d+\.(.+)\.weight", _text_weight_name(name))
    return match[1] if match else None


def _matrix_shape(tensor):
    shape = tensor.get("shape")
    element = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1}.get(tensor["dtype"])
    if (not element or not isinstance(shape, list) or len(shape) != 2 or
            any(type(n) is not int or n <= 0 for n in shape) or
            shape[0] * shape[1] * element != tensor["size"]):
        return None
    return shape


def _q38_cpu_dense_tensors(info, env):
    """Price q38_trunk_offer_all's CPU rows, independently of CUDA placement.

    The scan cache is environment independent. Selection happens here on each
    plan: the threshold includes a float scale per output row, the skip list is
    component based, and F16 checkpoints are loaded as F32 before quantization.
    """
    tensors = info.get("dense_tensors", [])
    if info["resolved_family"].descriptor.id != "qwen38" or env.get("Q38_TRUNK_CPU_INT8") == "0":
        return info
    threshold = re.match(r"\s*([+-]?\d+)", env.get("Q38_TRUNK_MIN_KB", "1024"))
    min_kb = int(threshold[1]) if threshold else 0   # C atol; negative casts to size_t
    skipped = set(env.get("Q38_TRUNK_SKIP", "").split(","))
    adjusted, delta = [], 0
    for tensor in tensors:
        shape = _matrix_shape(tensor)
        name = _text_weight_name(tensor["name"])
        tag = "lmhead" if name == "lm_head.weight" else _Q38_TRUNK_COMPONENTS.get(_layer_component(name))
        if shape and tag and tag not in skipped and min_kb >= 0:
            rows, columns = shape
            quantized = rows * columns + 4 * rows
            if quantized >= min_kb * 1024:
                delta += quantized - tensor["resident"]
                tensor = dict(tensor, resident=quantized)
        adjusted.append(tensor)
    return dict(info, dense_bytes=info["dense_bytes"] + delta, dense_tensors=adjusted)


_ATTENTION_PROJECTIONS = frozenset("self_attn." + p + "_proj" for p in ("q", "k", "v", "o"))
_MLP_PROJECTIONS = frozenset("mlp." + p + "_proj" for p in ("gate", "up", "down"))
_SHARED_PROJECTIONS = frozenset("mlp.shared_experts." + p + "_proj" for p in ("gate", "up", "down"))
_G53_MATRICES = (_ATTENTION_PROJECTIONS | _MLP_PROJECTIONS | _SHARED_PROJECTIONS |
                 frozenset("self_attn." + p + "_proj" for p in ("q_a", "q_b", "g_a", "g_b", "f_a", "f_b", "b")) |
                 {"self_attn.kv_a_proj_with_mqa", "self_attn.indexer.wq_b",
                  "self_attn.indexer.wk", "self_attn.indexer.weights_proj"})


def _glm53_dense_tensors(info, env):
    """Mirror glm53.c's quantize_loaded for known matrices, reserve F32 otherwise.

    Cached scans contain source geometry. Never cache GLM53_BITS or multiply
    vectors, embeddings and the vision tower by a quantized-matrix ratio. kv_b's
    absorbed halves and unrecognized components retain a conservative F32 budget.
    """
    if info["resolved_family"].descriptor.id != "glm53":
        return info
    setting = re.match(r"\s*([+-]?\d+)", env.get("GLM53_BITS", "4"))
    bits = int(setting[1]) if setting else 0
    if bits not in (4, 8, 32):
        raise ValueError("GLM53_BITS must be 4, 8 or 32")
    adjusted, total, embedding = [], 0, 0
    for tensor in info.get("dense_tensors", []):
        name = _text_weight_name(tensor["name"])
        size, dtype = tensor["size"], tensor["dtype"]
        element = {"BF16": 2, "F16": 2, "F32": 4}.get(dtype)
        resident = (size // element * 4) if element and size % element == 0 else size
        shape = _matrix_shape(tensor)
        known = (name == "lm_head.weight" or _layer_component(name) in _G53_MATRICES or
                 re.fullmatch(r"model\.layers\.\d+\.self_attn\.indexer\.index_kpool_compress_gate", name))
        if element and shape and known and bits != 32:
            rows, columns = shape
            if bits == 4 and columns % 64 == 0:
                resident = rows * (columns // 2) + rows * (columns // 64) * 4
            else:
                resident = rows * columns + rows * 4
        adjusted.append(dict(tensor, resident=resident))
        total += resident
        if _EMBED_TABLE.search(name):
            embedding += resident
    return dict(info, dense_bytes=total, embed_bytes=embedding, dense_tensors=adjusted)


def _vk_released_tensor_bytes(tensor, family, env):
    """A lower bound on the host allocation an engine's dho pass can free.

    Only audited matrix names and source formats get credit. Keep norms, vision,
    routers used by the CPU, unknown packed layouts and hardware-dependent BF16
    kernels in RAM. Upload refusal can still retain a copy at runtime.
    """
    shape = _matrix_shape(tensor)
    if not shape:
        return 0
    name, dtype = _text_weight_name(tensor["name"]), tensor["dtype"]
    part = _layer_component(name)
    rows, columns = shape
    count, resident = rows * columns, tensor["resident"]
    head = name == "lm_head.weight"
    floating = dtype in ("BF16", "F16", "F32")
    if family == "qwen38":
        known = (head or part in _Q38_TRUNK_COMPONENTS or
                 part in ("linear_attn.in_proj_a", "linear_attn.in_proj_b", "ple.key_proj", "ple.value_proj") or
                 name in ("model.hyper_connection_mixer.input_mix_weight_down.weight",
                          "model.hyper_connection_mixer.input_mix_weight_up.weight"))
        return resident if known and floating else 0
    if family == "qwen36":
        known = (head or part in _ATTENTION_PROJECTIONS or part in _MLP_PROJECTIONS or
                 part in ("mlp.gate", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj") or
                 part in {p.replace("shared_experts", "shared_expert") for p in _SHARED_PROJECTIONS})
        setting = re.match(r"\s*([+-]?\d+)", env.get("COLI_DENSE_BITS", "8"))
        bits = int(setting[1]) if setting else 8
        if not known or not floating or bits == 16:
            return 0
        # The checkpoint can be F32 while the engine holds int4/int8. Crediting
        # the checkpoint bytes would also erase RAM that was never duplicated.
        lower = count // 2 if bits == 4 else count
        return min(resident, lower)
    if family == "olmoe":
        return min(resident, count * 4) if floating and (head or part in _ATTENTION_PROJECTIONS or part == "mlp.gate") else 0
    if family == "glm53":
        # kv_b's absorbed halves and a tied head can remain CPU-readable.
        # Conservatively reserve those; every matrix listed here is in g53_dho_pass.
        if not floating or part not in _G53_MATRICES:
            return 0
        bits = env.get("GLM53_BITS", "4")
        lower = count * 4 if bits == "32" else count if bits == "8" else count // 2
        return min(resident, lower)
    if family == "kimi":
        if env.get("K3_MMAP") not in (None, "", "0") or not floating:
            return 0
        known = (head or part in _ATTENTION_PROJECTIONS or part in _MLP_PROJECTIONS or
                 part in ("self_attn.g_proj", "self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj") or
                 part in ("block_sparse_moe.routed_expert_down_proj", "block_sparse_moe.routed_expert_up_proj") or
                 part in {p.replace("mlp.", "block_sparse_moe.", 1) for p in _SHARED_PROJECTIONS})
        return min(resident, count // 2) if known else 0
    if family == "inkling":
        known = head or part in _ATTENTION_PROJECTIONS or part == "self_attn.r_proj" or part in _MLP_PROJECTIONS or part in _SHARED_PROJECTIONS
        # BF16 may use AVX512-BF16 on the CPU and be ineligible for Vulkan;
        # packed 2/3-bit or unknown group layouts may be ineligible as well.
        return resident if known and dtype == "F32" else 0
    if family == "mimo":
        known = head or part in ("self_attn.qkv_proj", "self_attn.o_proj") or part in _MLP_PROJECTIONS
        return min(resident, count) if known else 0
    if family == "deepseek_v4":
        # coli_v4_dense_device_only_tensor: layer FP8 matrices and the two
        # BF16 compressor projections. Head, mHC, routers and scales stay.
        known = re.fullmatch(r"layers\.\d+\.(?:attn|ffn)\..+\.weight", name)
        droppable = dtype == "F8_E4M3" or (dtype == "BF16" and name.endswith(("compressor.wkv.weight", "compressor.wgate.weight")))
        return resident if known and droppable else 0
    if family == "deepseek_v41":
        # v41_dho_bytes names the mapped trunk. Sidecar block scales stay.
        known = re.fullmatch(r"layers\.\d+\.(?:attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b|compressor\.(?:wkv|wgate)|indexer\.(?:wk|wq_b|weights_proj))|ffn\.shared_experts\.w[123]|engram\.wkv)\.weight", name)
        return resident if known and dtype in ("BF16", "F8_E4M3") else 0
    # GLM's CLI-selected QT format (including unsupported planar/IQ/E8/2-bit
    # forms) is not described by this scan. Do not guess a releasable allocation.
    return 0


def vk_dense_device_only(dense_bytes, family_id=None, env=None, vulkan=None):
    """(device_only, why): whether a Vulkan engine keeps its dense weights on the
    device only, with no host copy (docs/vulkan.md, "Dense weights on the device
    only"), as the engine's coli_vk_dense_host_decide() decides it: COLI_VULKAN=1,
    the dense part on the device (the chain or the per-matrix path), and
    COLI_VK_DENSE_HOST=0, or unset with an integrated GPU or a discrete one whose
    free memory, less 1 GiB, holds them. `vulkan` is the device setup_hw would
    report (probed when None and needed)."""
    env = os.environ if env is None else env
    host = _vk_flag(env, "COLI_VK_DENSE_HOST")
    if host is not None and host != 0:
        return False, None
    vulkan = _vk_device(env, vulkan)
    kind = (vulkan or {}).get("type")
    if not _vk_dense_active(family_id, env, vulkan):
        return False, None
    if host == 0:
        return True, "COLI_VK_DENSE_HOST=0"
    if kind == "integrated":
        return True, "an integrated GPU: its memory is the same RAM"
    if kind == "discrete":
        used = sum(heap.get("usage", 0) for heap in vulkan.get("heaps", [])
                   if heap.get("device_local"))
        free = ((vulkan.get("budget_bytes") or vulkan.get("device_local_bytes") or 0)
                - used - (1 << 30))
        if dense_bytes <= free:
            return True, "a discrete GPU with room for the dense weights"
    return False, None


# ---- the partial chain (docs/vulkan.md, "A partial chain") ------------------------------
# When the dense layers do not all fit the device, the chain takes the first N of them and
# the CPU the rest and the head; only the N layers' host copies can then be dropped. The
# engine decides N with vkc_fit (vk_chain.c) from what the device has free; this mirrors
# it, per family, from the scanned tensors and the config:
#   _VK_CHAIN_LAYOUT[family_id](info, env) -> VkChainLayout(layers, fixed, tail) or None
#     layers  each layer's device bytes as the engine's fit counts them (its matrices,
#             each range aligned, its state at its first size, its share of the
#             parameters), fixed  the engine's fixed bytes (the scratch of one prompt
#             chunk, without the pools' granularity), tail  what goes up only with every
#             layer (the head, matrices the per-matrix path uploads as it meets them).
# Families without an entry keep N = L (the plan as before).
VkChainLayout = collections.namedtuple("VkChainLayout", "layers fixed tail")
_VK_CHAIN_LAYOUT = {}
_VK_ALIGN = 256   # coli_vk_buffer_alignment: a storage buffer's alignment, at least 256


def _vk_up(value, align=_VK_ALIGN):
    return (value + align - 1) // align * align


def vk_tensor_bytes(fmt, columns, rows, group=0):
    """vkc_fit_tensor: a resident tensor's rows at their padded stride and its scales,
    each range aligned (fmt as the backend numbers them: 10 f32, 11 bf16, 12 fp8 in
    groups, 1 int8 rows, 2 int4 rows, 4 int4 in groups, 7 MXFP4, 13 int8 in groups)."""
    row = {10: 4 * columns, 11: 2 * columns, 14: 2 * columns, 12: columns, 13: columns, 1: columns,
           2: (columns + 1) // 2, 4: (columns + 1) // 2, 7: (columns + 1) // 2}[fmt]
    data = (row + 3) // 4 * 4 * rows
    if fmt in (10, 11, 14):
        scales = 1
    elif fmt in (4, 7, 12, 13):
        scales = rows * -(-columns // group)
    else:
        scales = rows
    return _vk_up(data or 4) + _vk_up(4 * scales)


def vk_buf_bytes(value):
    """vkc_fit_buf: a chain buffer (vkc_buf) of `value` bytes."""
    return _vk_up((max(value, 4) + 3) // 4 * 4)


def _vk_layer_index(name):
    match = re.match(r"(?:model\.)?layers\.(\d+)\.", _text_weight_name(name))
    return int(match[1]) if match else None


def _vk_cap_bytes(env):
    value = (env.get("COLI_VK_DEVICE_CAP_MB") or "").strip()
    try:
        return max(0, int(float(value) * 1048576)) if value else 0
    except ValueError:
        return 0


def _vk_block(cap, default):
    """coli_vk_block_bytes: a pool's block, smaller under COLI_VK_DEVICE_CAP_MB."""
    if not cap:
        return default
    block = 64 << 10
    while block < cap // 4096:
        block <<= 1
    return min(block, default)


def vk_fit_pools(cap=0):
    """vkc_fit_pools: a weight block, a block of each chain pool, the frames' staging."""
    return _vk_block(cap, 256 << 20) + 3 * _vk_block(cap, 64 << 20) + 4 * (4 << 20)


def _vk_free_bytes(env, vulkan):
    """coli_vk_free_bytes as a plan sees it: the cap (the process holds nothing yet),
    else the device's budget (or its device-local heap) less its heaps' usage."""
    cap = _vk_cap_bytes(env)
    if cap:
        return cap
    vulkan = vulkan or {}
    used = sum(heap.get("usage", 0) for heap in vulkan.get("heaps", []) if heap.get("device_local"))
    return max(0, (vulkan.get("budget_bytes") or vulkan.get("device_local_bytes") or 0) - used)


def vk_chain_fit(info, family_id, env=None, vulkan=None):
    """The engine's vkc_fit for this model and device: {"n", "L", "tail", "forced",
    "free", "reserve", "fixed", "layers"}, or None when the chain does not run or the
    family has no layout (N = L then, as before)."""
    env = os.environ if env is None else env
    vulkan = _vk_device(env, vulkan)
    chain, _ = _vk_chain_dense(family_id, env, vulkan)
    layout_of = _VK_CHAIN_LAYOUT.get(family_id)
    layout = layout_of(info, env, vulkan) if chain and layout_of else None
    if layout is None:
        return None
    layers = list(layout.layers)
    count = len(layers)
    free = _vk_free_bytes(env, vulkan)
    try:
        reserve = max(0, int(float(env.get("COLI_VK_TIER_RESERVE_GB") or 1.0) * (1 << 30)))
    except ValueError:
        reserve = 1 << 30
    fixed = layout.fixed + vk_fit_pools(_vk_cap_bytes(env))
    room = max(0, free - reserve)
    forced = (env.get("COLI_VK_CHAIN_LAYERS") or "").strip()
    if forced and forced != "auto":
        match = re.match(r"\s*([+-]?\d+)", forced)
        n = min(max(int(match[1]) if match else 0, 0), count)
        tail = n == count and count > 0
    elif fixed + sum(layers) <= room:
        n, tail = count, fixed + sum(layers) + layout.tail <= room
    else:
        n, used = 0, fixed
        while n < count and used + layers[n] <= room:
            used += layers[n]
            n += 1
        tail = False
    return {"n": n, "L": count, "tail": tail, "forced": bool(forced and forced != "auto"),
            "free": free, "reserve": reserve, "fixed": fixed, "layers": layers}


def _v4_chain_rows(env):
    """deepseek_v4_chain.h v4c_rows: the CPU's prefill block, COLI_VK_CHAIN_ROWS below it."""
    def number(name, default):
        match = re.match(r"\s*([+-]?\d+)", env.get(name) or "")
        return int(match[1]) if match else default
    block = number("V4_PREFILL_CHUNK", 128)
    if block < 1 or block > 128:
        block = 128
    value = env.get("COLI_VK_CHAIN_ROWS") or ""
    rows = number("COLI_VK_CHAIN_ROWS", block) if value.strip() and value.strip() != "auto" else block
    return min(max(rows, 1), block)


def _v4_chain_layout(info, env, vulkan):
    """deepseek_v4_chain.h: v4c_layer_bytes, v4c_fixed_bytes, v4c_tail_bytes."""
    c = info.get("config") or {}
    try:
        L = int(c["num_hidden_layers"])
        ratios = [int(r) for r in c["compress_ratios"][:L]]
        D, H, nh, hd = c["hidden_size"], c.get("hc_mult", 1), c["num_attention_heads"], c["head_dim"]
        rd, QL, og, ol = c["qk_rope_head_dim"], c["q_lora_rank"], c.get("o_groups", 1), c["o_lora_rank"]
        W, inter = c["sliding_window"], c["moe_intermediate_size"]
    except (KeyError, TypeError, ValueError):
        return None
    if len(ratios) < L or L < 1:
        return None
    IH = c.get("index_n_heads") or 0
    IH = IH if IH > 0 else 1
    ID = c.get("index_head_dim") or 0
    ID = ID if ID > 0 else 2
    topk = c.get("index_topk") or 0
    HD, nm, hr = H * D, (2 + H) * H, 2 * H + H * H
    rows = _v4_chain_rows(env)
    layers = [0] * L
    for tensor in info.get("dense_tensors", []):
        layer, shape = _vk_layer_index(tensor["name"]), tensor.get("shape")
        if layer is None or layer >= L or not isinstance(shape, list) or len(shape) != 2:
            continue
        suffix = re.sub(r"^(?:model\.)?layers\.\d+\.", "", _text_weight_name(tensor["name"]))
        out, columns, parts, fmt = shape[0], shape[1], 1, 0
        if tensor["dtype"] == "F8_E4M3":
            fmt = 12
            if suffix == "attn.wo_a.weight":
                parts, out = og, ol
        elif tensor["dtype"] == "BF16" and (suffix.endswith(("compressor.wkv.weight", "compressor.wgate.weight"))
                                            or suffix == "attn.indexer.weights_proj.weight"):
            fmt = 11
        elif tensor["dtype"] == "F32" and suffix in ("hc_attn_fn", "hc_ffn_fn"):
            fmt = 10
        if fmt:
            layers[layer] += parts * vk_tensor_bytes(fmt, columns, out, 128 if fmt == 12 else 0)
    for i, r in enumerate(ratios):
        crows, cproj = (8 if r == 4 else r), (2 if r == 4 else 1) * hd
        state = vk_buf_bytes((W + rows) * hd * 4)
        if r > 0:
            state += vk_buf_bytes(2 * crows * cproj * 4) + vk_buf_bytes(64 * hd * 4)
        if r == 4:
            state += vk_buf_bytes(2 * 8 * 2 * ID * 4) + vk_buf_bytes(64 * ID * 4)
        floats = 2 * D + QL + hd + nh + 2 * (3 + nm)
        if r > 0:
            floats += hd + r * cproj
        if r == 4:
            floats += ID + 8 * ID
        floats += min(rows, W) * hd
        if r > 0:
            floats += (rows // r) * hd + 2 * (crows if r == 4 else min(rows, r)) * cproj
        if r == 4:
            floats += (rows // 4) * ID + 2 * 8 * 2 * ID
        layers[i] += state + 4 * floats
    # v4c_scratch's counting pass at `rows` rows, the context of the window and them
    E = W + rows
    K = max([topk if r == 4 else E // r if r > 0 else 0 for r in ratios] + [0])
    LR = W + (K if K > 0 else 1)
    kinds = 1 + len({r for r in ratios if r > 0 and r != 4})
    rmax = max([1] + ratios)
    taps = 3 if int(c.get("num_nextn_predict_layers") or 0) >= 3 else 1
    r_ = rows
    counts = [r_ * HD, r_ * HD, r_ * nm, r_ * hr, r_ * hr, r_ * D, r_ * D, r_ * D, r_ * QL, r_ * QL, r_ * QL,
              r_ * nh * hd, r_ * hd, r_ * nh * hd, r_ * og * ol, r_ * D, r_ * 2 * hd, r_ * 2 * hd,
              r_ * 2 * ID, r_ * 2 * ID, r_ * IH * ID, r_ * IH, r_ * max(E // 4, 1), kinds * r_ * LR,
              (2 * r_ + rmax) * rd, r_ * D, r_ * inter, r_ * inter, r_ * inter, r_ * D, r_ * D, r_ * D,
              r_ * HD, taps * r_ * HD]
    fixed = sum(4 * (n if n else 1) for n in counts)
    _, dense = _vk_chain_dense("deepseek_v4", env, vulkan)
    tail = 0
    if dense:
        tail = (vk_tensor_bytes(11, D, int(c.get("vocab_size") or 0)) +
                L * vk_tensor_bytes(11, D, int(c.get("n_routed_experts") or 0)))
    return VkChainLayout(layers, fixed, tail)


_VK_CHAIN_LAYOUT["deepseek_v4"] = _v4_chain_layout


# ---- qwen36 (Qwen3.6, Qwen3-Coder, Qwen3.8-27B, Clef's backbone) and olmoe ------------
def _vk_chain_rows(env, default=256):
    """vkc_fit_rows: COLI_VK_CHAIN_ROWS when it is a number, else the engine's block."""
    value = (env.get("COLI_VK_CHAIN_ROWS") or "").strip()
    match = re.match(r"\s*([+-]?\d+)", value)
    if not value or value == "auto" or not match:
        return default
    return min(max(int(match[1]), 1), 65535)


def _vk_kv_block(env):
    match = re.match(r"\s*([+-]?\d+)", env.get("COLI_VK_KV_BLOCK") or "")
    return max(int(match[1]) if match else 64, 1)


def _q36_dense_fmt(env, tag, columns):
    """The format qwen36.c puts a dense matrix on the device in (load_tq, vk_qw_fmt): f32
    with COLI_DENSE_I8=0, f16 with COLI_DENSE_BITS=16, int4 in groups of 64 with
    COLI_DENSE_BITS=4 where COLI_DENSE_INT4 (unset: every tag) names its tag and the
    columns divide by 64, else int8 rows."""
    if (env.get("COLI_DENSE_I8") or "").strip() == "0":
        return 10
    match = re.match(r"\s*([+-]?\d+)", env.get("COLI_DENSE_BITS") or "")
    bits = int(match[1]) if match else 8
    if bits == 16:
        return 14
    if bits == 4 and columns % 64 == 0:
        tags = (env.get("COLI_DENSE_INT4") or "").strip()
        if not tags or tag in re.split(r"[ ,]+", tags):
            return 4
    return 1


# the matrices qwen36's chain multiplies by, and load_tq's tag for each (a dense model's
# MLP loads as the shared expert)
_Q36_CHAIN_TAGS = {"self_attn.q_proj": "attn", "self_attn.k_proj": "attn", "self_attn.v_proj": "attn",
                   "self_attn.o_proj": "attn", "linear_attn.in_proj_qkv": "dnproj",
                   "linear_attn.in_proj_z": "dnproj", "linear_attn.out_proj": "dnout", "mlp.gate": "router",
                   "mlp.shared_expert.gate_proj": "shexp", "mlp.shared_expert.up_proj": "shexp",
                   "mlp.shared_expert.down_proj": "shexp", "mlp.gate_proj": "shexp", "mlp.up_proj": "shexp",
                   "mlp.down_proj": "shexp"}


def _q36_chain_layout(info, env, vulkan):
    """qwen36_chain.h: q36c_fit_layer, q36c_start's fixed bytes and lm_head as the tail,
    from the scanned matrices' shapes and the scalars of qwen36_meta.json (what the engine
    reads; config.json's when the checkpoint has none)."""
    try:
        meta = json.loads((Path(info["path"]) / "qwen36_meta.json").read_text(encoding="utf-8"))
    except (KeyError, OSError, ValueError):
        meta = {}
    c = info.get("config") or {}
    c = c.get("text_config", c)
    try:
        L = int(meta.get("n_layers") or c["num_hidden_layers"])
        D = int(meta.get("hidden") or c["hidden_size"])
        hd = int(meta.get("head_dim") or c["head_dim"])
        kdim = int(meta.get("dn_kdim") or c.get("linear_key_head_dim") or 0)
        convk = int(meta.get("dn_convk") or c.get("linear_conv_kernel_dim") or 0)
    except (KeyError, TypeError, ValueError):
        return None
    types = meta.get("layer_types") or c.get("layer_types")
    if not isinstance(types, list):
        interval = int(c.get("full_attention_interval") or 4)
        types = ["full_attention" if i % interval == interval - 1 else "linear_attention" for i in range(L)]
    if L < 1 or len(types) < L:
        return None
    attn = [t == "full_attention" for t in types[:L]]
    rot = int(meta.get("rotary_dim") or 0)
    if rot <= 0:
        factor = float(meta.get("partial_rotary_factor") or c.get("partial_rotary_factor") or 0)
        rot = int(hd * factor + 0.5) if factor > 0 else hd
        rot = min(max(rot + rot % 2, 2), hd)
    # the shapes the engine loads, from the checkpoint
    rows_of, layers, norms = {}, [0] * L, [0] * L
    matrices = []
    for tensor in info.get("dense_tensors", []):
        name, shape = _text_weight_name(tensor["name"]), tensor.get("shape")
        layer, part = _vk_layer_index(name), _layer_component(name)
        if layer is not None and layer < L and part in ("self_attn.q_norm", "self_attn.k_norm"):
            norms[layer] += hd
            continue
        if not isinstance(shape, list) or len(shape) != 2:
            continue
        if name == "lm_head.weight" or (layer is not None and layer < L and (part in _Q36_CHAIN_TAGS or part in (
                "linear_attn.in_proj_b", "mlp.shared_expert_gate"))):
            rows_of.setdefault(name if layer is None else part, shape)
            matrices.append((layer, part if layer is not None else name, shape))
    def rows(part, default=0):
        return rows_of[part][0] if part in rows_of else default
    E = rows("mlp.gate")
    SI = rows("mlp.shared_expert.gate_proj") or rows("mlp.gate_proj")
    vh = rows("linear_attn.in_proj_b")
    conv, vd = rows("linear_attn.in_proj_qkv"), rows("linear_attn.in_proj_z")
    qo, kvo = rows("self_attn.q_proj"), rows("self_attn.k_proj")
    o_in = rows_of["self_attn.o_proj"][1] if "self_attn.o_proj" in rows_of else 0
    vocab = rows("lm_head.weight", int(c.get("vocab_size") or 0))
    vdim = vd // vh if vh else 0
    # COLI_VK_IMPORT (qwen36.c main, vk_qw_tensor): int8 and f16 rows read in place, only
    # their scales on the device; by default for a model without routed experts on a device
    # sharing the CPU's RAM, never while host copies are dropped off a non-integrated one
    kind = (vulkan or {}).get("type")
    flag = _vk_flag(env, "COLI_VK_IMPORT")
    imports = flag != 0 if flag is not None else E == 0 and kind in ("integrated", "cpu")
    if imports and kind != "integrated" and _vk_flag(env, "COLI_VK_DENSE_HOST") == 0:
        imports = False

    def placed(tag, shape):
        out, columns = shape
        fmt = _q36_dense_fmt(env, tag, columns)
        if imports and fmt in (1, 14):
            return _vk_up(4 * (out if fmt == 1 else 1))
        return vk_tensor_bytes(fmt, columns, out, 64 if fmt == 4 else 0)
    head = 0
    for layer, part, shape in matrices:
        if layer is None:
            head = placed("lmhead", shape)
        elif part in _Q36_CHAIN_TAGS:
            layers[layer] += placed(_Q36_CHAIN_TAGS[part], shape)
        elif part == "mlp.shared_expert_gate" and E and SI:
            layers[layer] += vk_tensor_bytes(10, D, 1)                 # the gate row, f32
    for i in range(L):
        floats = 2 * D
        if attn[i]:
            floats += norms[i]
            layers[i] += 2 * vk_buf_bytes(kvo * 3 * _vk_kv_block(env) * 4)   # the K/V mirror at the split's floor
        else:
            floats += conv * convk + 2 * vh + vdim
            layers[i] += vk_tensor_bytes(10, D, 2 * vh)                # the b|a rows, f32
            layers[i] += vk_buf_bytes(vh * kdim * vdim * 4) + vk_buf_bytes(conv * (convk - 1) * 4)
        layers[i] += 4 * floats                                        # its share of the parameters
    # q36c_bufs for the fit's rows, their read-back, the arena's final norm
    r, n_attn = _vk_chain_rows(env), sum(attn)
    counts = [r * D, r * D, r * D, r * qo, r * kvo, r * kvo, r * o_in, r * conv, r * vd, r * 2 * vh, r * conv,
              r * vd, r * D, r * max(E, 1), r * max(SI, 1), r * max(SI, 1), r * max(SI, 1), r * D, r, D, r * D,
              r * max(E, 1), max(n_attn, 1) * 2 * r * kvo, vocab, r * D, r * max(rot, 2), r * D]
    fixed = sum(vk_buf_bytes(4 * max(n, 1)) for n in counts) + vk_buf_bytes(4 * D) + vk_buf_bytes(4)
    return VkChainLayout(layers, fixed, head)


_VK_CHAIN_LAYOUT["qwen36"] = _q36_chain_layout


def _olmoe_chain_layout(info, env, vulkan):
    """olmoe_chain.h olc_fit_start: q, k, v, o and the router as f32 tensors, the K/V
    mirror at the split's floor, the parameters; one chunk's scratch; lm_head the tail."""
    c = info.get("config") or {}
    try:
        L, D, H = int(c["num_hidden_layers"]), int(c["hidden_size"]), int(c["num_attention_heads"])
        E, vocab = int(c["num_experts"]), int(c["vocab_size"])
    except (KeyError, TypeError, ValueError):
        return None
    if L < 1 or H < 1 or D % H:
        return None
    hd = D // H
    layer = (4 * vk_tensor_bytes(10, D, D) + vk_tensor_bytes(10, D, E) +
             2 * vk_buf_bytes(H * 3 * _vk_kv_block(env) * hd * 4) + 4 * 4 * D)
    r = _vk_chain_rows(env)
    counts = [r * D] * 8 + [r * E, D, r * D, r * E, 2 * r * D, vocab, r * D, r * hd, r * D]
    if (env.get("PILOT") or "").strip() not in ("", "0"):
        counts.append(r * D)   # PILOT's rows after attention
    fixed = sum(vk_buf_bytes(4 * max(n, 1)) for n in counts) + vk_buf_bytes(4 * D) + vk_buf_bytes(4)
    return VkChainLayout([layer] * L, fixed, vk_tensor_bytes(10, D, vocab))


_VK_CHAIN_LAYOUT["olmoe"] = _olmoe_chain_layout


_Q38_CHAIN_MATRICES = frozenset(_Q38_TRUNK_COMPONENTS) | {
    "linear_attn.in_proj_a", "linear_attn.in_proj_b", "ple.key_proj", "ple.value_proj"}


def _q38_env_int(env, name, default):
    """C's atol/atoi: the leading integer, 0 when there is none."""
    value = env.get(name)
    if value is None or not value.strip():
        return default
    match = re.match(r"\s*([+-]?\d+)", value)
    return int(match[1]) if match else 0


def _q38_vk_fmt(tensor, tag, env):
    """q38_vk_fmt: the format a resident matrix goes up in. The trunk's int8 rows (fmt 1)
    when q38_trunk_cpu_int8 quantized it (a trunk component of at least Q38_TRUNK_MIN_KB
    with its float scales, not in Q38_TRUNK_SKIP); else the rows as loaded: bf16 (fmt 11)
    from a BF16 tensor with Q38_NATIVE_BF16, f32 (fmt 10) otherwise."""
    rows, columns = tensor["shape"]
    if tag and env.get("Q38_TRUNK_CPU_INT8") != "0":
        min_kb = _q38_env_int(env, "Q38_TRUNK_MIN_KB", 1024)
        skipped = set((env.get("Q38_TRUNK_SKIP") or "").split(","))
        if min_kb >= 0 and rows * columns + 4 * rows >= min_kb * 1024 and tag not in skipped:
            return 1
    return 11 if tensor["dtype"] == "BF16" and env.get("Q38_NATIVE_BF16") != "0" else 10


def _q38_chain_layout(info, env, vulkan):
    """qwen38_chain.h: q38c_fit_layer, q38c_fit_fixed, q38c_fit_tail. None for a device
    whose memory the plan does not know (no budget, no heap size, no cap) unless N is
    forced: no prediction, the plan as before (every layer and every host copy)."""
    device = vulkan or {}
    known = _vk_cap_bytes(env) or device.get("budget_bytes") or device.get("device_local_bytes")
    if not known and (env.get("COLI_VK_CHAIN_LAYERS") or "").strip() in ("", "auto"):
        return None
    c = info.get("config") or {}
    c = c.get("text_config") or c
    try:
        L, H, vocab = int(c["num_hidden_layers"]), int(c["hidden_size"]), int(c["vocab_size"])
        kinds = [0 if t == "linear_attention" else 1 for t in c["layer_types"]][:L]
        QH, KVH, D = int(c["num_attention_heads"]), int(c["num_key_value_heads"]), int(c["head_dim"])
        IQ, ID = int(c["indexer_n_heads"]), int(c["indexer_head_dim"])
        budget, ratio = int(c["indexer_budget"]), int(c["indexer_compress_ratio"])
        VH, VD = int(c["linear_num_value_heads"]), int(c["linear_value_head_dim"])
        KH, KD, CK = int(c["linear_num_key_heads"]), int(c["linear_key_head_dim"]), int(c["linear_conv_kernel_dim"])
        E, SI = int(c["num_experts"]), int(c["shared_expert_intermediate_size"])
    except (KeyError, TypeError, ValueError):
        return None
    if L < 1 or len(kinds) != L:
        return None
    C, R = int(c.get("hc_count") or 4), int(c.get("hc_lowrank") or 320)
    W = C * H
    rope = c.get("rope_parameters") or {}
    partial = rope.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0))
    rotary = int(D * float(partial if partial is not None else 1.0))
    ngram, per_ngram = int(c.get("ngram_size") or 3), int(c.get("heads_per_ngram") or 8)
    ple_dim, ple_k = int(c.get("ple_embed_dim") or H), int(c.get("ple_conv_kernel_size") or 4)
    heads = (ngram - 1) * per_ngram
    Ep = max(heads * (ple_dim // heads if heads > 0 else 0), 1)
    ids = c.get("ple_layer_ids") or []
    ple = int(ids[0]) - 1 if ids else -1
    CD = 2 * KH * KD + VH * VD
    rows = _q38_chain_rows(env)
    rows0 = 3 * max(_q38_env_int(env, "COLI_VK_KV_BLOCK", 64), 1)
    kvo = KVH * D
    # each layer's matrices as they go up, and the shared expert's gate (fmt 10, H x 1)
    layers = [vk_tensor_bytes(10, H, 1) for _ in range(L)]
    for tensor in info.get("dense_tensors", []):
        layer, part, shape = _vk_layer_index(tensor["name"]), _layer_component(tensor["name"]), tensor.get("shape")
        if layer is None or layer >= L or part not in _Q38_CHAIN_MATRICES or not isinstance(shape, list) or len(shape) != 2:
            continue
        if part.startswith("ple.") and layer != ple:
            continue
        fmt = _q38_vk_fmt(tensor, _Q38_TRUNK_COMPONENTS.get(part), env)
        layers[layer] += vk_tensor_bytes(fmt, shape[1], shape[0])
    for i in range(L):
        floats = 2 * W
        if kinds[i]:
            floats += 2 * D + 2 * ID
            layers[i] += (2 * vk_buf_bytes(kvo * rows0 * 4) + vk_buf_bytes(ID * rows0 * 4) +
                          vk_buf_bytes(ID * (rows0 // max(ratio, 1) + 1) * 4) + rows * (2 * kvo + ID) * 4)
        else:
            floats += CD * CK + 2 * VH + VD
            layers[i] += 2 * vk_buf_bytes(VH * KD * VD * 4) + 2 * vk_buf_bytes(CD * (CK - 1) * 4)
        if i == ple:
            floats += 3 * W + W * ple_k
            layers[i] += vk_buf_bytes(2 * W * (ple_k - 1) * ngram * 4)
        layers[i] += 4 * floats
    # q38c_bufs' counting pass at `rows` rows, no context yet and no attention layer, the
    # streams' read-back and the final norm
    r, nb, rot = rows, 1, rotary if rotary > 0 else 2
    counts = [r * W, r * W, r * R, r * W, r * H, r * C, r * C, r * H, r * QH * 2 * D, r * kvo, r * kvo,
              r * (IQ + 1) * ID, r * QH * D, r * 2 * nb, r * (budget + ratio), r * CD, r * VH * VD, r * 2 * VH,
              r * CD, r * VH * VD, r * E, r * SI, r * SI, r * SI, r * H, r, r * H, r * W, r * H, r * W, r * W,
              r * H, r * E, r * (2 * kvo + ID), 2 * vocab, r * H, r * Ep, r * rot, nb * rot]
    fixed = sum(4 * (n if n else 1) for n in counts) + (rows + 1) * W * 4
    # the tail: the final mixer and lm_head; under Q38_MTP=1 the MTP head's matrices too
    # (the scan leaves its tensors out: its decoder layer has an attention layer's shapes,
    # its two projections are H x H and its mixer the final mixer's)
    tail, head_dtype = 0, "BF16"
    attn_parts = {}
    last_attn = max([i for i in range(L) if kinds[i]] + [-1])
    for tensor in info.get("dense_tensors", []):
        name, shape = _text_weight_name(tensor["name"]), tensor.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            continue
        if name == "lm_head.weight":
            head_dtype = tensor["dtype"]
            tail += vk_tensor_bytes(_q38_vk_fmt(tensor, "lmhead", env), shape[1], shape[0])
        elif name in ("model.hyper_connection_mixer.input_mix_weight_down.weight",
                      "model.hyper_connection_mixer.input_mix_weight_up.weight"):
            tail += vk_tensor_bytes(_q38_vk_fmt(tensor, None, env), shape[1], shape[0])
        elif _vk_layer_index(tensor["name"]) == last_attn and _layer_component(tensor["name"]) in _Q38_CHAIN_MATRICES:
            attn_parts[_layer_component(tensor["name"])] = tensor
    if env.get("Q38_MTP") == "1" and int(c.get("mtp_num_hidden_layers") or 0) == 1:
        for part, tensor in attn_parts.items():
            tail += vk_tensor_bytes(_q38_vk_fmt(tensor, _Q38_TRUNK_COMPONENTS.get(part), env),
                                    tensor["shape"][1], tensor["shape"][0])
        for tag, out, cols in (("mtpfce", H, H), ("mtpfch", H, H), ("mtpmixd", R, W), ("mtpmixu", W, R)):
            tensor = {"shape": [out, cols], "dtype": head_dtype}
            tail += vk_tensor_bytes(_q38_vk_fmt(tensor, tag, env), cols, out)
    return VkChainLayout(layers, fixed, tail)


def _q38_chain_rows(env):
    """vkc_fit_rows(256): COLI_VK_CHAIN_ROWS when it is a number, else 256."""
    value = (env.get("COLI_VK_CHAIN_ROWS") or "").strip()
    if not value or value == "auto":
        return 256
    return min(max(_q38_env_int(env, "COLI_VK_CHAIN_ROWS", 256), 1), 65535)


_VK_CHAIN_LAYOUT["qwen38"] = _q38_chain_layout


def _glm53_chain_layout(info, env, vulkan):
    """glm53_chain.h g53c_fit_plan: each layer's device bytes (the mHC mixes in f32, its
    matrices at GLM53_BITS as quantize_loaded leaves them, an int4 container's as it is,
    the absorbed kv_b halves, its share of the parameter arena, a KDA layer's state and
    window, an MLA layer's caches at their first 256 positions), the scratch of one prompt
    chunk at GLM53_MAXT and the head."""
    config = info.get("config") or {}
    c = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    try:
        L, D, V = int(c["num_hidden_layers"]), int(c["hidden_size"]), int(c["vocab_size"])
        H = int(c.get("hc_mult") or 1)
        linear = c["linear_attn_config"]
        kh, kd, ck = int(linear["num_heads"]), int(linear["head_dim"]), int(linear["short_conv_kernel_size"])
        nh, ql, kvl = int(c["num_attention_heads"]), int(c["q_lora_rank"]), int(c["kv_lora_rank"])
        qn, qr, vh = int(c["qk_nope_head_dim"]), int(c.get("qk_rope_head_dim") or 0), int(c["v_head_dim"])
        di, mi, ns = int(c["intermediate_size"]), int(c["moe_intermediate_size"]), int(c.get("n_shared_experts") or 1)
        kinds = list(c["layer_types"])[:L]
    except (KeyError, TypeError, ValueError):
        return None
    if len(kinds) < L or L < 1:
        return None
    IH, ID = int(c.get("index_n_heads") or 0), int(c.get("index_head_dim") or 0)
    pool, topk = int(c.get("index_kpool") or 1), int(c.get("index_topk") or 0)
    full = ["linear" not in str(k) for k in kinds]
    if c.get("first_k_dense_replace") is not None:
        first_dense = int(c["first_k_dense_replace"])
    else:
        mlp = c.get("mlp_layer_types") or []
        first_dense = next((i for i, k in enumerate(mlp) if "sparse" in str(k)), len(mlp))
    setting = re.match(r"\s*([+-]?\d+)", env.get("GLM53_BITS", "4"))
    bits = int(setting[1]) if setting else 4
    packed = {_text_weight_name(t["name"]) for t in info.get("dense_tensors", []) if t.get("dtype") in ("U8", "I8")}
    P, nm, HD = kh * kd, (2 + H) * H, H * D

    def mat(rows, columns, name=None):
        if name is not None and name in packed:
            return vk_tensor_bytes(4, columns, rows, 64)
        if bits == 32:
            return vk_tensor_bytes(10, columns, rows)
        if bits == 4 and columns % 64 == 0:
            return vk_tensor_bytes(4, columns, rows, 64)
        return vk_tensor_bytes(1, columns, rows)
    layers = []
    for i in range(L):
        p = f"model.layers.{i}."
        size = 2 * vk_tensor_bytes(10, HD, nm)
        if full[i]:
            a = p + "self_attn."
            size += (mat(ql, D, a + "q_a_proj.weight") + mat(nh * (qn + qr), ql, a + "q_b_proj.weight") +
                     mat(kvl + qr, D, a + "kv_a_proj_with_mqa.weight") + mat(nh * kvl, qn) + mat(nh * vh, kvl) +
                     mat(D, nh * vh, a + "o_proj.weight") + mat(IH * ID, ql, a + "indexer.wq_b.weight") +
                     mat(ID, D, a + "indexer.wk.weight") + mat(IH, D, a + "indexer.weights_proj.weight") +
                     mat(ID, D, a + "indexer.index_kpool_compress_gate"))
            floats = 2 * D + 2 * (3 + nm) + ql + kvl + 2 * ID + pool * ID
            size += (vk_buf_bytes(256 * kvl * 4) + 2 * vk_buf_bytes(256 * ID * 4) +
                     vk_buf_bytes((256 // pool + 1) * ID * 4))
        else:
            a = p + "self_attn."
            size += (mat(P, D, a + "q_proj.weight") + mat(P, D, a + "k_proj.weight") + mat(P, D, a + "v_proj.weight") +
                     mat(D, P, a + "o_proj.weight") + mat(kd, D, a + "g_a_proj.weight") + mat(P, kd, a + "g_b_proj.weight") +
                     mat(kd, D, a + "f_a_proj.weight") + mat(P, kd, a + "f_b_proj.weight") + mat(kh, D, a + "b_proj.weight"))
            floats = 2 * D + 2 * (3 + nm) + 3 * P * ck + kh + P + kd
            size += vk_buf_bytes(3 * P * ck * 4) + vk_buf_bytes(kh * kd * kd * 4)
        m = p + ("mlp." if i < first_dense else "mlp.shared_experts.")
        inner = di if i < first_dense else mi * ns
        size += mat(inner, D, m + "gate_proj.weight") + mat(inner, D, m + "up_proj.weight") + mat(D, inner, m + "down_proj.weight")
        layers.append(size + 4 * floats)
    # g53c_scratch's counting pass for one chunk (every layer's new rows down) and the MLA
    # scratch, at vkc_fit_rows(128) rows and a serve slot's context
    value = (env.get("COLI_VK_CHAIN_ROWS") or "").strip()
    match = re.match(r"([+-]?\d+)", value) if value and value != "auto" else None
    r = min(max(int(match[1]), 1), 65535) if match else 128
    match = re.match(r"\s*([+-]?\d+)", env.get("GLM53_MAXT") or "")
    ctx = max(int(match[1]) if match else 8192, 64)
    wide = max(di, mi)
    width = topk + pool - 1 if c.get("index_kpool_always_select_tail") else topk
    counts = [r * HD, r * HD, r * D, r * D, r * D, r * nm, r * (2 * H + H * H), r * wide, r * wide, r * wide, r * D,
              r * D, L * r * (kvl + 2 * ID), r * HD, r * D]
    if P > 0:
        counts += [3 * r * P, 3 * r * P, r * P, r * kh, r * P, r * kd, r * P]
    if any(full):
        counts += [r * ID, r * IH * ID, r * IH, r * ID, r * (ctx // pool + 1), r * (1 + width)]
        counts += [r * ql, r * nh * qn, r * kvl, r * nh * kvl, r * nh * kvl, r * nh * vh]
    fixed = sum(vk_buf_bytes(4 * (n if n else 1)) for n in counts) + vk_buf_bytes(4)
    tail = 0 if c.get("tie_word_embeddings") else mat(V, D, "lm_head.weight")
    return VkChainLayout(layers, fixed, tail)


_VK_CHAIN_LAYOUT["glm53"] = _glm53_chain_layout


def _analysis_signature(shards, config_path):
    parts = [f"v{_ANALYSIS_CACHE_VERSION}"]
    st = config_path.stat()
    parts.append(f"config:{st.st_size}:{st.st_mtime_ns}")
    for shard in shards:
        s = shard.stat()
        parts.append(f"{shard.name}:{s.st_size}:{s.st_mtime_ns}")
    return "|".join(parts)


def _tensor_sizes(path, with_shape=False):
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"short safetensors header: {path}")
        length = int.from_bytes(raw, "little")
        if length < 2 or length > file_size - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(stream.read(length))
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        if not 0 <= start <= end <= file_size - 8 - length:
            raise ValueError(f"invalid tensor offsets for {name}: {path}")
        dtype = meta.get("dtype")
        if not isinstance(dtype, str):
            raise ValueError(f"invalid tensor dtype for {name}: {path}")
        item = (name, end - start, dtype)
        yield (*item, meta.get("shape")) if with_shape else item


def analyze_model(model):
    resolved = resolve_model(model)
    model = Path(resolved.model_dir)
    config = resolved.config
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no safetensors shards: {model}")

    # Sidecar cache: return the stored analysis if every shard + config is unchanged.
    # resolve_model() already read config.json, so it exists by this point.
    signature = _analysis_signature(shards, model / "config.json")
    cache_path = model / _ANALYSIS_CACHE_NAME
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and cached.get("signature") == signature \
                and isinstance(cached.get("analysis"), dict):
            analysis = cached["analysis"]
            # JSON object keys are always strings: restore the int layer
            # indices so a cache hit is identical to a fresh scan.
            by_layer = analysis.get("expert_bytes_by_layer")
            if isinstance(by_layer, dict):
                analysis["expert_bytes_by_layer"] = {
                    int(layer): size for layer, size in by_layer.items()}
            # resolved_family is a live registry object and cannot be JSON'd.
            # The resolve_model() above recomputed it cheaply (config.json
            # only); the one scan-derived bit it carries for glm is the
            # indexer-presence flag, which the cache stores alongside.
            indexer = analysis.pop("_colibri_indexer_present", None)
            if indexer is not None and resolved.descriptor.id == "glm":
                family_cfg = dict(resolved.family_config,
                                  _colibri_indexer_present=indexer)
                resolved = type(resolved)(resolved.descriptor, resolved.model_type,
                                          resolved.config, family_cfg,
                                          resolved.model_dir)
            analysis["resolved_family"] = resolved
            return analysis
    except (OSError, ValueError):
        pass  # missing/corrupt/unreadable cache -> recompute

    dense_bytes = 0
    # The embedding tables among them: the engines gather their rows on the CPU, so
    # they keep a host copy when the rest of the dense weights live on a Vulkan
    # device only (vk_dense_device_only below).
    embed_bytes = 0
    # Model-owned allocations that are retained once, independently of the
    # number of per-layer cache slots. Qwen3.8's native FP8 path uses this for
    # its normalized scale bank; fallback accounting remains conservative.
    expert_fixed_bytes = 0
    # What the engine's GPU trunk offload would put in VRAM (int8), for the
    # families that have one; taken out of the VRAM budget before experts.
    trunk_int8_bytes = 0
    # Keep the source geometry, dtype and resident contribution for placement
    # decisions. These facts can be cached; environment-dependent credits cannot.
    dense_tensors = []
    expert_groups = {}
    tensor_names = set()
    for shard in shards:
        try:
            sizes = list(_tensor_sizes(shard, with_shape=True))
        except OSError as error:
            # Name the file. An OSError raised by read() on an already-open
            # stream carries no filename, so `coli doctor` reported bare
            # "[Errno 5] Input/output error" for a bad sector or a dropped
            # network mount — indistinguishable from a corrupt download, which
            # is what the reporter in #191 assumed and re-downloaded 372 GB to
            # rule out. Which shard failed is the whole diagnosis: one file is
            # storage, all of them is the mount.
            raise OSError(error.errno,
                          f"{error.strerror or error}: {shard}") from error
        for name, size, dtype, shape in sizes:
            tensor_names.add(name)
            contributions = expert_contributions(resolved, name, size, dtype)
            if contributions:
                for layer, expert, byte_count in contributions:
                    key = (layer, expert)
                    expert_groups[key] = expert_groups.get(key, 0) + byte_count
            else:
                fixed_bytes = fixed_resident_contribution(
                    resolved, name, size, dtype)
                if fixed_bytes:
                    expert_fixed_bytes += fixed_bytes
                else:
                    resident = resident_contribution(resolved, name, size, dtype)
                    dense_bytes += resident
                    if resident:
                        dense_tensors.append({"name": name, "size": size, "dtype": dtype,
                                              "shape": shape,
                                              "resident": _dense_in_ram(resolved.descriptor, resident)})
                    if _EMBED_TABLE.search(name):
                        embed_bytes += resident
                    trunk_int8_bytes += trunk_contribution(
                        resolved, name, size, dtype)

    layer_sizes = {}
    for (layer, _), size in expert_groups.items():
        layer_sizes.setdefault(layer, []).append(size)
    # Most families ship uniform experts and retain the historical median to
    # tolerate container padding. Qwen3.8 explicitly accepts heterogeneous
    # source dtypes; one LRU slot may hold any expert, so each layer must be
    # priced at its largest retained representation just like the C runtime.
    reducer = max if resolved.descriptor.id == "qwen38" else statistics.median
    per_layer = {layer: int(reducer(sizes)) for layer, sizes in layer_sizes.items()}
    per_cap_bytes = sum(per_layer.values())
    typical_expert_bytes = int(statistics.median(per_layer.values())) if per_layer else 0
    max_expert_bytes = max(per_layer.values(), default=0)
    model_bytes = sum(shard.stat().st_size for shard in shards)
    if resolved.descriptor.id == "glm":
        family_cfg = resolved.family_config
        layers = int(family_cfg.get("num_hidden_layers") or 0)
        kinds = family_cfg.get("indexer_types")
        if isinstance(kinds, list):
            required = [layer for layer, kind in enumerate(kinds[:layers])
                        if kind == "full"]
        else:
            frequency = max(1, int(family_cfg.get("index_topk_freq") or 1))
            offset = int(family_cfg.get("index_skip_topk_offset") or 2)
            required = [layer for layer in range(layers)
                        if max(layer - offset + 1, 0) % frequency == 0]
        indexer_present = bool(required and all(
            f"model.layers.{layer}.self_attn.indexer.wq_b.weight" in tensor_names
            for layer in required))
        family_cfg = dict(family_cfg, _colibri_indexer_present=indexer_present)
        resolved = type(resolved)(resolved.descriptor, resolved.model_type,
                                  resolved.config, family_cfg, resolved.model_dir)
    result = {
        "path": str(model),
        "shards": len(shards),
        "model_bytes": model_bytes,
        # Come pesano DAVVERO in RAM: un motore che riquantizza al caricamento
        # ne occupa meno di quanto ne occupino sul disco, e il pianificatore
        # serve a rispondere "ci sta?", non "quanto pesa il file".
        "dense_bytes": _dense_in_ram(resolved.descriptor, dense_bytes),
        "dense_disk_bytes": dense_bytes,
        "embed_bytes": _dense_in_ram(resolved.descriptor, embed_bytes),
        "expert_fixed_bytes": expert_fixed_bytes,
        "trunk_int8_bytes": trunk_int8_bytes,
        "dense_tensors": dense_tensors,
        "expert_bytes": sum(expert_groups.values()),
        "expert_count": len(expert_groups),
        "expert_layers": len(per_layer),
        "typical_expert_bytes": typical_expert_bytes,
        "max_expert_bytes": max_expert_bytes,
        "expert_bytes_by_layer": per_layer,
        "per_cap_bytes": per_cap_bytes,
        "config": config,
        "resolved_family": resolved,
    }
    # analyze_model has one stable, default-format answer regardless of the
    # caller's process environment. build_plan reapplies its requested settings.
    result = _glm53_dense_tensors(result, {})
    try:  # best-effort write; a read-only model dir must never break planning.
        # Atomic write (tmp file + os.replace): a concurrent `coli plan` on the
        # same model dir must never observe a half-written cache -- write_text()
        # alone can leave a truncated file for a racing reader's json.loads to
        # choke on, which just falls back to a full recompute (harmless but
        # defeats the cache for that call). Same directory so the replace stays
        # on one filesystem (os.replace requires that to be atomic).
        # resolved_family is a registry object, not JSON-serializable: persist
        # only the scan-derived indexer flag it carries; the hit path rebuilds
        # the object from a fresh (cheap) resolve_model().
        payload = dict(result)
        resolved_family = payload.pop("resolved_family")
        family_cfg = getattr(resolved_family, "family_config", None)
        if isinstance(family_cfg, dict) and "_colibri_indexer_present" in family_cfg:
            payload["_colibri_indexer_present"] = family_cfg["_colibri_indexer_present"]
        # The tmp name must be per-THREAD, not per-process: concurrent writers
        # in one process would otherwise write the same tmp file, and one
        # thread's os.replace renames it out from under the other. On Windows
        # the replace can also fail (PermissionError) while a concurrent
        # reader holds the destination open -- caught below, and the tmp must
        # not be left behind either way.
        tmp_path = cache_path.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp_path.write_text(
                json.dumps({"signature": signature, "analysis": payload}),
                encoding="utf-8")
            os.replace(tmp_path, cache_path)
        except (OSError, TypeError, ValueError):
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
    except (OSError, TypeError, ValueError):
        pass
    return result


QWEN38_INT4_SIDECAR = "experts-int4g64"


def qwen38_int4_sidecar(info):
    """Price Qwen3.8's routed experts as qwen38 will hold them.

    tools/convert_qwen38_experts_int4.py writes <model>/experts-int4g64/, and
    the engine loads every routed expert from it instead of the FP8 shards
    when its index says it is complete (Q38_EXPERT_INT4=0 keeps FP8). One of
    its records is 56% of an FP8 expert, so pricing the cache with FP8 bytes
    would hand the engine a cap for half the RAM it was given. Only the index
    is read here; the engine validates the files themselves and refuses a
    sidecar that disagrees with it. Returns the analysis unchanged when the
    sidecar does not apply, otherwise a copy priced with its records and
    marked `qwen38_int4_experts`: no FP8 scale bank is built either.
    """
    resolved = info["resolved_family"]
    if resolved.descriptor.id != "qwen38" or os.environ.get("Q38_EXPERT_INT4") == "0":
        return info
    try:
        index = json.loads((Path(info["path"]) / QWEN38_INT4_SIDECAR / "index.json")
                           .read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return info
    config = info["config"]
    text = config.get("text_config", config) if isinstance(config, dict) else {}
    want = {"format": "colibri.qwen38.experts-int4g64", "version": 1, "complete": True,
            "layers": text.get("num_hidden_layers"), "experts": text.get("num_experts"),
            "hidden_size": text.get("hidden_size"),
            "moe_intermediate_size": text.get("moe_intermediate_size")}
    record = index.get("record_bytes") if isinstance(index, dict) else None
    if (any(index.get(key) != value for key, value in want.items()) or
            isinstance(record, bool) or not isinstance(record, int) or record < 1):
        return info
    per_layer = {layer: record for layer in info["expert_bytes_by_layer"]}
    return dict(info,
                expert_bytes=record * want["experts"] * len(per_layer),
                typical_expert_bytes=record, max_expert_bytes=record,
                expert_bytes_by_layer=per_layer, per_cap_bytes=record * len(per_layer),
                expert_fixed_bytes=0, qwen38_int4_experts=True)


#: MEMORYSTATUSEX as Windows defines it, in order. Kept as data so a test can
#: pin the order without a Windows machine.
WINDOWS_MEMORYSTATUSEX_FIELDS = (
    ("dwLength", "c_ulong"), ("dwMemoryLoad", "c_ulong"),
    ("ullTotalPhys", "c_ulonglong"), ("ullAvailPhys", "c_ulonglong"),
    ("ullTotalPageFile", "c_ulonglong"), ("ullAvailPageFile", "c_ulonglong"),
    ("ullTotalVirtual", "c_ulonglong"), ("ullAvailVirtual", "c_ulonglong"),
    ("ullAvailExtendedVirtual", "c_ulonglong"),
)


def windows_available_bytes(avail_phys, avail_pagefile):
    """What a Windows process can still get from malloc: the smaller of free
    physical memory and the commit still grantable (RAM + page file, minus
    what every process has already committed).

    #1375: the planner budgeted 88 % of ullAvailPhys on a 128 GB machine and
    handed the resulting cap to the engine; the engine filled it slot by slot
    until Windows refused the next 14 MB. Physical memory was there. Commit
    was not, and nothing had looked at it."""
    if avail_pagefile and 0 < avail_pagefile < avail_phys:
        return avail_pagefile
    return avail_phys


def _read_bounded_bytes(path, max_bytes, description):
    """Return ``(present, raw)`` for one small kernel pseudo-file.

    ``present`` keeps an absent file (that controller or proc path is simply
    not mounted here) distinct from one that exists but cannot be trusted. Any
    other read failure, and any file larger than the kernel would ever write,
    is a typed refusal: it is never evidence that the process is unconstrained.
    """
    path = Path(path)
    try:
        with path.open("rb") as stream:
            raw = stream.read(max_bytes + 1)
    except (FileNotFoundError, NotADirectoryError):
        return (False, None)
    except OSError as error:
        raise CgroupAccessError(f"cannot read {description} {path}: {error}") from error
    if len(raw) > max_bytes:
        raise CgroupFormatError(f"oversized {description}: {path}")
    return (True, raw)


def _read_bounded_ascii(path, max_bytes, description):
    present, raw = _read_bounded_bytes(path, max_bytes, description)
    if not present:
        return (False, None)
    try:
        return (True, raw.decode("ascii"))
    except UnicodeDecodeError as error:
        raise CgroupFormatError(f"non-ASCII {description}: {path}") from error


def _proc_lines(text, description):
    """Split one procfs record file; an empty file or a blank record is malformed."""
    if not text:
        raise CgroupFormatError(f"empty {description}")
    newline = b"\n" if isinstance(text, bytes) else "\n"
    lines = (text[:-1] if text.endswith(newline) else text).split(newline)
    if any(not line for line in lines):
        raise CgroupFormatError(f"blank record in {description}")
    return lines


def _parse_bounded_id(field, description, *, allow_zero=True):
    """Parse one canonical decimal u64 procfs field after a lexical length bound."""
    if (not field or len(field) > _CGROUP_UINT64_DIGITS or not field.isdigit()
            or (len(field) > 1 and field.startswith(b"0"))):
        raise CgroupFormatError(f"malformed {description}: {field!r}")
    value = int(field)
    if value > _CGROUP_UINT64_MAX or (value == 0 and not allow_zero):
        raise CgroupFormatError(f"out-of-range {description}: {field!r}")
    return value


def _path_components(raw, description):
    """Split one absolute kernel path into components; "", "." and ".." are malformed."""
    if raw == b"/":
        return ()
    if not raw.startswith(b"/"):
        raise CgroupFormatError(f"{description} is not absolute: {raw!r}")
    components = raw[1:].split(b"/")
    if any(component in (b"", b".", b"..") for component in components):
        raise CgroupFormatError(f"unnormalized {description}: {raw!r}")
    if len(components) >= _CGROUP_MAX_ANCESTORS:
        raise CgroupFormatError(
            f"{description} nesting exceeds {_CGROUP_MAX_ANCESTORS - 1} levels")
    return tuple(os.fsdecode(component) for component in components)


# mountinfo(5) escapes exactly these four bytes as three-digit octal.
_PROC_PATH_ESCAPES = {b"011": b"\t", b"012": b"\n", b"040": b" ", b"134": b"\\"}


def _decode_proc_path(field, description):
    """Decode one escaped mountinfo path field into normalized components."""
    decoded = bytearray()
    index = 0
    while index < len(field):
        byte = field[index]
        if byte == ord("\\"):
            escape = field[index + 1:index + 4]
            if escape not in _PROC_PATH_ESCAPES:
                raise CgroupFormatError(f"malformed escape in {description}: {field!r}")
            decoded += _PROC_PATH_ESCAPES[escape]
            index += 4
            continue
        if byte in (0, ord(" "), ord("\t"), ord("\n")):
            raise CgroupFormatError(f"unescaped byte in {description}: {field!r}")
        decoded.append(byte)
        index += 1
    return _path_components(bytes(decoded), description)


def _decode_cgroup_membership_path(field, description):
    """Decode a raw /proc/self/cgroup path (the kernel does not escape it)."""
    if b"\0" in field:
        raise CgroupFormatError(f"NUL byte in {description}: {field!r}")
    return _path_components(field, description)


def _parse_cgroup_counter(text):
    """Parse one non-negative cgroup byte counter under an explicit u64 cap."""
    match = re.fullmatch(r"(\d+)\n?", text or "")
    if match is None or len(match.group(1)) > _CGROUP_UINT64_DIGITS:
        return None
    value = int(match.group(1))
    return value if value <= _CGROUP_UINT64_MAX else None


def _cgroup_inactive_file(directory, v1=False):
    """Return the inactive file cache charged to one cgroup, in bytes.

    ``memory.current`` (v1: ``usage_in_bytes``) charges clean page cache -- for
    this engine, the mmap'd shards a previous run left behind -- while the host
    half of the minimum, MemAvailable, counts such cache as available. Subtract
    ``inactive_file`` (v1: ``total_inactive_file``, the hierarchical figure that
    pairs with the hierarchical usage counter) the way Docker and cAdvisor
    derive a working set. Absent statistics, or a kernel without the key,
    subtract nothing; a present but malformed file is a refusal.
    """
    stat_path = Path(directory) / "memory.stat"
    present, text = _read_bounded_ascii(stat_path, _CGROUP_STAT_MAX_BYTES,
                                        "cgroup memory statistics")
    if not present:
        return 0
    key = "total_inactive_file" if v1 else "inactive_file"
    values = []
    for line in _proc_lines(text, "cgroup memory statistics"):
        match = re.fullmatch(r"([A-Za-z0-9_]+) (\d+)", line)
        if match is None:
            raise CgroupFormatError(f"malformed cgroup memory statistics: {stat_path}")
        if match.group(1) == key:
            values.append(match.group(2))
    if len(values) > 1:
        raise CgroupFormatError(f"duplicate {key} in cgroup memory statistics: {stat_path}")
    value = _parse_cgroup_counter(values[0]) if values else 0
    if value is None:
        raise CgroupFormatError(f"malformed {key} in cgroup memory statistics: {stat_path}")
    return value


def _cgroup_pair_remaining(directory, limit_name, current_name, v1=False):
    """Return ``(controls_present, finite_remaining_or_none)`` for one cgroup.

    Present controls with ``None`` remaining are the kernel's unlimited value
    (v2 ``max``; v1's sentinel just below 2**63). Both files must be present
    and well-formed together: an incomplete pair or a malformed counter is a
    refusal, never something a looser ancestor may mask. Headroom is measured
    against the working set, charged usage less reclaimable inactive file
    cache, so a previous run's cached model pages do not read as spent budget.
    """
    directory = Path(directory)
    limit_path = directory / limit_name
    current_path = directory / current_name
    limit_present, limit_text = _read_bounded_ascii(
        limit_path, _CGROUP_VALUE_MAX_BYTES, "cgroup memory limit")
    current_present, current_text = _read_bounded_ascii(
        current_path, _CGROUP_VALUE_MAX_BYTES, "cgroup memory usage")
    if not limit_present and not current_present:
        return (False, None)
    if not limit_present or not current_present:
        missing = current_path if limit_present else limit_path
        raise CgroupAccessError(f"incomplete cgroup memory control pair: {missing}")
    current = _parse_cgroup_counter(current_text)
    if current is None:
        raise CgroupFormatError(f"malformed cgroup memory usage: {current_path}")
    if not v1 and limit_text in ("max", "max\n"):
        return (True, None)
    limit = _parse_cgroup_counter(limit_text)
    if limit is None:
        raise CgroupFormatError(f"malformed cgroup memory limit: {limit_path}")
    if v1 and limit >= _CGROUP_V1_UNLIMITED_MIN:
        return (True, None)
    working_set = max(0, current - _cgroup_inactive_file(directory, v1=v1))
    return (True, max(0, limit - working_set))


def _read_cgroup_memberships(proc_cgroup_path):
    """Strictly parse v2 and v1-memory membership from ``/proc/self/cgroup``.

    ``None`` means the file is absent. Hybrid layouts are ordinary -- a
    ``0::/...`` v2 record beside v1 records for controllers this planner never
    consumes, such as ``1:net_cls:/`` from a VPN client -- and must not fail.
    """
    present, raw = _read_bounded_bytes(
        proc_cgroup_path, _CGROUP_MEMBERSHIP_MAX_BYTES, "cgroup membership file")
    if not present:
        return None
    v2_path = None
    v1_path = None
    hierarchy_ids = set()
    for line in _proc_lines(raw, "cgroup membership file"):
        fields = line.split(b":", 2)
        if len(fields) != 3:
            raise CgroupFormatError(f"malformed cgroup membership record: {line!r}")
        hierarchy, controllers, member_path = fields
        hierarchy_id = _parse_bounded_id(hierarchy, "cgroup hierarchy id")
        if hierarchy_id in hierarchy_ids:
            raise CgroupFormatError(f"duplicate cgroup hierarchy id: {hierarchy!r}")
        hierarchy_ids.add(hierarchy_id)
        controller_list = controllers.split(b",") if controllers else []
        if (any(re.fullmatch(rb"[A-Za-z0-9_.=-]+", item) is None for item in controller_list)
                or len(set(controller_list)) != len(controller_list)):
            raise CgroupFormatError(f"malformed cgroup controller list: {controllers!r}")
        member = _decode_cgroup_membership_path(member_path, "cgroup membership path")
        if hierarchy_id == 0:
            if controller_list or v2_path is not None:
                raise CgroupFormatError("ambiguous cgroup v2 membership")
            v2_path = member
        elif not controller_list:
            raise CgroupFormatError(f"v1 cgroup membership lacks controllers: {line!r}")
        elif b"memory" in controller_list:
            if v1_path is not None:
                raise CgroupFormatError("ambiguous cgroup v1 memory membership")
            v1_path = member
    return _CgroupMemberships(v2=v2_path, v1_memory=v1_path)


def _parse_mount_options(field, description):
    try:
        options = field.decode("ascii").split(",")
    except UnicodeDecodeError as error:
        raise CgroupFormatError(f"non-ASCII {description}: {field!r}") from error
    if any(not option for option in options):
        raise CgroupFormatError(f"malformed {description}: {field!r}")
    return frozenset(options)


def _read_cgroup_mounts(proc_mountinfo_path):
    """Return every cgroup/cgroup2 mount from a strict, bounded mountinfo read.

    ``None`` means the file is absent. Every record is held to mountinfo(5)'s
    grammar so a truncated or synthetic file is a refusal rather than a quiet
    "no cgroup mounts here"; paths are decoded only for cgroup records, so an
    unrelated mount with an unusual name cannot block discovery.
    """
    present, raw = _read_bounded_bytes(
        proc_mountinfo_path, _CGROUP_MOUNTINFO_MAX_BYTES, "cgroup mountinfo file")
    if not present:
        return None
    mounts = []
    mount_ids = set()
    for line in _proc_lines(raw, "cgroup mountinfo file"):
        fields = line.split(b" ")
        if any(not field for field in fields):
            raise CgroupFormatError(f"malformed mountinfo spacing: {line!r}")
        try:
            separator = fields.index(b"-", 6)
        except ValueError as error:
            raise CgroupFormatError(f"mountinfo record lacks separator: {line!r}") from error
        if len(fields) != separator + 4:
            raise CgroupFormatError(f"malformed mountinfo record: {line!r}")
        mount_id = _parse_bounded_id(fields[0], "mountinfo mount id", allow_zero=False)
        parent_id = _parse_bounded_id(fields[1], "mountinfo parent id")
        device = fields[2].split(b":")
        if len(device) != 2:
            raise CgroupFormatError(f"malformed mountinfo device: {line!r}")
        for number in device:
            _parse_bounded_id(number, "mountinfo device number")
        if mount_id in mount_ids:
            raise CgroupFormatError(f"duplicate mountinfo id: {fields[0]!r}")
        mount_ids.add(mount_id)
        fs_type = fields[separator + 1]
        if fs_type not in (b"cgroup", b"cgroup2"):
            continue
        if fields[3].split(b"/")[1:2] == [b".."]:
            # cgroup_namespaces(7): a cgroupfs mount inherited from outside the
            # process's cgroup namespace reports its root relative to that
            # namespace as leading ".." components. Membership is reported
            # relative to the namespace root, so no sound mapping exists.
            raise CgroupAccessError(
                f"unsupported cgroup mount root {fields[3]!r}: remount cgroupfs "
                "inside the cgroup namespace")
        root = _decode_proc_path(fields[3], "cgroup mount root")
        mount_point = Path("/").joinpath(
            *_decode_proc_path(fields[4], "cgroup mount point"))
        options = (_parse_mount_options(fields[5], "cgroup mount options")
                   | _parse_mount_options(fields[separator + 3], "cgroup super options"))
        mounts.append(_CgroupMount(mount_id=mount_id, parent_id=parent_id, root=root,
                                   mount_point=mount_point,
                                   fs_type=fs_type.decode("ascii"), options=options))
    return mounts


def _resolve_cgroup_hierarchy(mounts, member, *, v1=False):
    """Return the mounts exposing ``member``, shallowest root first.

    A mount's root is the cgroup its mount point shows; ``member`` maps into
    it by stripping that root and appending the rest to the mount point. Of
    several applicable mounts, the one with the shallowest root shows the
    most ancestors: a deeper bind mount can hide an ancestor's limit, so it
    must never be preferred. Every view is returned and the caller takes the
    minimum, which also makes two mounts of one root harmless.
    """
    if v1:
        applicable = [mount for mount in mounts
                      if mount.fs_type == "cgroup" and "memory" in mount.options]
    else:
        applicable = [mount for mount in mounts if mount.fs_type == "cgroup2"]
    views = [mount for mount in applicable if member[:len(mount.root)] == mount.root]
    return sorted(views, key=lambda mount: len(mount.root))


def _cgroup_hierarchy_remaining(views, member, limit_name, current_name, v1=False):
    """Return ``(controls_seen, tightest finite headroom)`` across every view.

    Each view is walked from the membership leaf up to and including its mount
    point, so an ancestor's finite limit bounds the result whenever any view
    exposes it. Missing directories simply contribute nothing.
    """
    seen = False
    remaining = None
    for mount in views:
        suffix = member[len(mount.root):]
        for depth in range(len(suffix), -1, -1):
            directory = mount.mount_point.joinpath(*suffix[:depth])
            present, candidate = _cgroup_pair_remaining(
                directory, limit_name, current_name, v1=v1)
            seen = seen or present
            if candidate is not None:
                remaining = candidate if remaining is None else min(remaining, candidate)
    return (seen, remaining)


_CGROUP_V2_CONTROLS = ("memory.max", "memory.current")
_CGROUP_V1_CONTROLS = ("memory.limit_in_bytes", "memory.usage_in_bytes")


def _fixed_root_view(root, v1=False):
    """A synthetic whole-hierarchy mount at ``root`` for kernels without mountinfo."""
    return _CgroupMount(mount_id=0, parent_id=0, root=(), mount_point=Path(root),
                        fs_type="cgroup" if v1 else "cgroup2",
                        options=frozenset({"memory"}) if v1 else frozenset())


def _cgroup_memory_remaining(cgroup_root=Path("/sys/fs/cgroup"),
                             proc_cgroup_path=Path("/proc/self/cgroup"),
                             proc_mountinfo_path=Path("/proc/self/mountinfo")):
    """Return this process's tightest finite cgroup memory headroom, or ``None``.

    Cgroup v2 is authoritative when its memory controls are visible; v1 is
    consulted through mounts whose options include the memory controller.
    ``None`` means no finite limit applies (or no cgroupfs is mounted here at
    all). Present but malformed, incomplete or unreadable inputs raise
    ``CgroupError`` instead of reading as unconstrained. ``cgroup_root`` is
    consulted only by the fixed-layout probe used when mountinfo is absent.
    """
    memberships = _read_cgroup_memberships(proc_cgroup_path)
    mounts = _read_cgroup_mounts(proc_mountinfo_path)
    v2_member = memberships.v2 if memberships else None
    v1_member = memberships.v1_memory if memberships else None
    if mounts is None:
        # No mountinfo: probe the conventional roots. The leaf-to-mount-point
        # walk also covers runtimes that mount the process's own subgroup at
        # the root while procfs still reports its host-side path.
        probes = ((cgroup_root, v2_member, _CGROUP_V2_CONTROLS, False),
                  (Path(cgroup_root) / "memory", v1_member, _CGROUP_V1_CONTROLS, True),
                  (cgroup_root, v1_member, _CGROUP_V1_CONTROLS, True))
        for root, member, controls, v1 in probes:
            seen, remaining = _cgroup_hierarchy_remaining(
                [_fixed_root_view(root, v1=v1)], member or (), *controls, v1=v1)
            if seen:
                return remaining
        return None

    has_v2_mount = any(mount.fs_type == "cgroup2" for mount in mounts)
    has_v1_memory_mount = any(mount.fs_type == "cgroup" and "memory" in mount.options
                              for mount in mounts)
    if memberships is None:
        if has_v2_mount or has_v1_memory_mount:
            raise CgroupAccessError(
                "cgroup membership is unavailable for a mounted memory hierarchy")
        return None
    if has_v2_mount and v2_member is None:
        raise CgroupAccessError("cgroup v2 mount has no corresponding membership record")
    if has_v1_memory_mount and v1_member is None:
        raise CgroupAccessError(
            "cgroup v1 memory mount has no corresponding membership record")

    if has_v2_mount:
        views = _resolve_cgroup_hierarchy(mounts, v2_member)
        if not views:
            raise CgroupAccessError(
                f"no visible cgroup2 mount exposes cgroup /{'/'.join(v2_member)}")
        seen, remaining = _cgroup_hierarchy_remaining(views, v2_member, *_CGROUP_V2_CONTROLS)
        if seen:
            return remaining
    if has_v1_memory_mount:
        views = _resolve_cgroup_hierarchy(mounts, v1_member, v1=True)
        if not views:
            raise CgroupAccessError(
                f"no visible cgroup v1 memory mount exposes cgroup /{'/'.join(v1_member)}")
        seen, remaining = _cgroup_hierarchy_remaining(
            views, v1_member, *_CGROUP_V1_CONTROLS, v1=True)
        if not seen:
            raise CgroupAccessError("cgroup v1 memory mount exposes no memory control files")
        return remaining
    return None


def _host_memory_available(meminfo_path):
    """Return strict host MemAvailable bytes, or ``None`` when it is unknown.

    Unknown covers an absent ``/proc/meminfo`` and a kernel that predates
    MemAvailable (3.14), exactly the cases the planner always fell back on. A
    present line that is duplicated, not ``<digits> kB``, or beyond a u64 byte
    counter is a refusal.
    """
    meminfo_path = Path(meminfo_path)
    present, text = _read_bounded_ascii(meminfo_path, _MEMINFO_MAX_BYTES,
                                        "memory information file")
    if not present:
        return None
    lines = [line for line in text.split("\n") if line.startswith("MemAvailable")]
    if not lines:
        return None
    match = (re.fullmatch(r"MemAvailable:[ \t]+(\d+)[ \t]+kB[ \t]*", lines[0])
             if len(lines) == 1 else None)
    if match is None:
        raise CgroupFormatError(f"MemAvailable is malformed: {meminfo_path}")
    kib = _parse_cgroup_counter(match.group(1))
    if kib is None or kib > _CGROUP_UINT64_MAX // 1024:
        raise CgroupFormatError(f"MemAvailable overflows a byte counter: {meminfo_path}")
    return kib * 1024


def memory_available(*, meminfo_path=Path("/proc/meminfo"),
                     cgroup_root=Path("/sys/fs/cgroup"),
                     proc_cgroup_path=Path("/proc/self/cgroup"),
                     proc_mountinfo_path=Path("/proc/self/mountinfo")):
    """Bytes this process may plan to allocate; ``None`` when nothing could measure it.

    On Linux this is ``min(host MemAvailable, remaining finite cgroup budget)``:
    a container started with a memory limit is killed at that limit however
    much the host has free, and the cgroup half is what lets the planner refuse
    an over-budget model before it allocates anything substantial. The result
    is a tri-state -- an int (including an authoritative 0 when the budget is
    already exhausted), ``None`` when no probe could measure anything so the
    planner keeps its historical fallback, or ``CgroupError`` when a present
    cgroup or procfs input is malformed, incomplete or unreadable. That last
    case is a refusal with a reason, never permission to assume the process is
    unconstrained, so it deliberately propagates.
    """
    # Linux (and MSYS2/Git-Bash CPython where /proc exists): MemAvailable.
    host_available = _host_memory_available(meminfo_path)
    cgroup_remaining = None
    if sys.platform.startswith("linux"):
        # Cgroups are a Linux kernel facility; nothing else has a hierarchy to
        # read. Probed even when the host figure is unknown, because a finite
        # limit is still a valid conservative bound when a restricted container
        # hides /proc/meminfo.
        cgroup_remaining = _cgroup_memory_remaining(
            cgroup_root=cgroup_root, proc_cgroup_path=proc_cgroup_path,
            proc_mountinfo_path=proc_mountinfo_path)
    if host_available is not None and cgroup_remaining is not None:
        return min(host_available, cgroup_remaining)
    if host_available is not None:
        return host_available
    if cgroup_remaining is not None:
        return cgroup_remaining
    # Windows native CPython: GlobalMemoryStatusEx -> ullAvailPhys.
    # Same definition the C engine uses (compat_meminfo in compat.h):
    # standby/free/zero pages, i.e. reclaimable without swapping.
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                # The real layout. The previous copy skipped the two PageFile
                # fields, which put ullTotalVirtual/ullAvailVirtual at the
                # wrong offsets; ullAvailPhys happened to be right, so nobody
                # noticed. The PageFile pair is the point now (#1375).
                _fields_ = [(name, getattr(ctypes, kind))
                            for name, kind in WINDOWS_MEMORYSTATUSEX_FIELDS]

            stat = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
            kernel32 = ctypes.windll.kernel32
            kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]
            kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)) and stat.ullAvailPhys:
                return windows_available_bytes(stat.ullAvailPhys, stat.ullAvailPageFile)
            # Fallback (e.g. sandboxed callers where GlobalMemoryStatusEx reports
            # nothing): total installed RAM in KB. Less precise than ullAvailPhys
            # — it ignores standby/reclaimable pages — but never returns 0 on a
            # real machine, which keeps the expert cache from being mis-sized.
            total_kb = ctypes.c_ulonglong(0)
            kernel32.GetPhysicallyInstalledSystemMemory.argtypes = [ctypes.c_void_p]
            kernel32.GetPhysicallyInstalledSystemMemory.restype = ctypes.c_int
            if kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(total_kb)):
                return total_kb.value * 1024
        except OSError:
            pass
    # macOS: no /proc and not win32. Sum the reclaimable pages reported by vm_stat
    # (free + inactive + speculative + purgeable) — the same "reclaimable without swapping"
    # definition the C engine's compat_meminfo uses. Fall back to total RAM (never 0 on a Mac).
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["vm_stat"], text=True, capture_output=True, timeout=5).stdout
            page_match = re.search(r"page size of (\d+) bytes", out)
            page = int(page_match.group(1)) if page_match else os.sysconf("SC_PAGE_SIZE")
            pages = 0
            for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                match = re.search(rf"{key}:\s+(\d+)\.", out)
                if match:
                    pages += int(match.group(1))
            if pages:
                return pages * page
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        try:
            total = subprocess.run(["sysctl", "-n", "hw.memsize"], text=True,
                                   capture_output=True, timeout=5).stdout.strip()
            if total:
                return int(total)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return None


def _host_unified_memory():
    """Whether the host itself exposes one CPU/GPU physical memory pool.

    This is intentionally separate from accelerator placement.  A CPU-only
    engine such as glm53 still runs on unified-memory Apple Silicon, but that
    fact must not fabricate a VRAM tier or make an unrelated GPU steer cache
    placement.
    """
    return sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64")


# Strict .coli_ssd grammar -- the byte-for-byte mirror of colibri.c's
# coli_ssd_cache_parse() (keep the two in lockstep; test_resource_plan.py and
# test_ssd_probe.c chew the same vector file, tests/fixtures/ssd_cache_vectors.txt):
#   v2:     b"v2 <gbs> <st_dev>"   single spaces, at most one trailing \n
#   legacy: b"<gbs>"               the pre-fix format, never trusted
# where <gbs> = digits["."digits] with 0 < gbs < 1000 and <st_dev> = 1..20
# digits fitting unsigned 64-bit; total length <= 64 bytes, no NULs, nothing
# else. float() permissiveness ("inf", "nan", "1e99", whitespace, signs) is
# deliberately out: it let corrupt caches surface as measurements, and "inf"
# reached doctor's JSON as the invalid literal Infinity.
_SSD_CACHE_V2 = re.compile(rb"\Av2 (\d+(?:\.\d+)?) (\d{1,20})\n?\Z")
_SSD_CACHE_LEGACY = re.compile(rb"\A(\d+(?:\.\d+)?)\n?\Z")


def parse_ssd_cache(data):
    """Classify raw .coli_ssd bytes under the strict grammar above. Returns
    ("v2", gbs, st_dev), ("legacy", gbs, None), or (None, None, None) for
    garbage. Classification only -- trust (the st_dev match) is the caller's."""
    if not data or len(data) > 64 or b"\x00" in data:
        return (None, None, None)
    match = _SSD_CACHE_V2.match(data)
    if match:
        gbs, dev = float(match.group(1)), int(match.group(2))
        if 0 < gbs < 1000 and dev <= 0xFFFFFFFFFFFFFFFF:
            return ("v2", gbs, dev)
        return (None, None, None)
    match = _SSD_CACHE_LEGACY.match(data)
    if match:
        gbs = float(match.group(1))
        if 0 < gbs < 1000:
            return ("legacy", gbs, None)
    return (None, None, None)


def ssd_probe_state(model_dir):
    """Classify the cached F_NOCACHE storage probe the C engine writes to
    <model>/.coli_ssd on its first Metal+darwin startup (colibri.c
    coli_ssd_probe_cached, issue #379). Read-only: never re-measures, never
    guesses -- mirrors S4's "read-and-display only" contract for `coli
    doctor`/`coli plan`. Returns (state, gbs):
      ("ok", gbs)        v2 cache recorded on THIS volume -- the one case the
                         engine itself would trust
      ("legacy", None)   pre-v2 bare number; the engine re-probes + upgrades
      ("foreign", None)  v2 from another volume (st_dev mismatch); re-probed
      ("garbage", None)  a file exists but fails the strict grammar
      ("absent", None)   no cache file at all
    The distinctions matter for wording (#386 r2, F10): "no cached probe yet"
    is a lie when a file exists. The read is bounded to 65 bytes (F13): the
    strict grammar caps a well-formed cache at 64, so byte 65 alone already
    convicts -- no reason to slurp an arbitrarily large impostor file."""
    try:
        with open(Path(model_dir) / ".coli_ssd", "rb") as fh:
            data = fh.read(65)
    except OSError:
        return ("absent", None)
    kind, gbs, dev = parse_ssd_cache(data)
    if kind == "v2":
        try:
            if dev == os.stat(model_dir).st_dev:
                return ("ok", gbs)
        except OSError:
            pass
        return ("foreign", None)
    if kind == "legacy":
        return ("legacy", None)
    return ("garbage", None)


def read_ssd_probe(model_dir):
    """The measured GB/s as a float when the engine itself would trust the
    cache (ssd_probe_state "ok"), else None."""
    return ssd_probe_state(model_dir)[1]


# What doctor/plan say for a cache that exists but is not trusted (#386 r2,
# F10): each state names what will actually happen, never "no cached probe
# yet" while a file sits right there.
SSD_PROBE_PENDING = {
    "legacy": "legacy cache pending engine upgrade; re-measured on the next Metal+darwin start",
    "foreign": "cache from another volume; the engine will re-probe here",
    "garbage": "unreadable cache; the engine will re-probe",
}


def discover_gpus():
    if sys.platform == "darwin":
        return _discover_metal_gpus()
    # NVIDIA first; if there are none (or no nvidia-smi), fall back to ROCm/HIP so
    # a working AMD engine isn't planned CPU-only and --gpu N stops failing (#662).
    devices = _discover_nvidia_gpus()
    if devices:
        return devices
    return _discover_amd_gpus()


def _discover_metal_gpus():
    """Return Apple Metal devices without pretending unified RAM is VRAM."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            text=True, capture_output=True, check=True, timeout=10)
        displays = json.loads(result.stdout).get("SPDisplaysDataType", [])
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return []
    devices = []
    for index, display in enumerate(displays):
        if not isinstance(display, dict):
            continue
        metal = display.get("spdisplays_mtlgpufamilysupport")
        if not metal:
            continue
        name = display.get("sppci_model") or display.get("_name")
        if not isinstance(name, str) or not name:
            continue
        # Apple Silicon has one unified pool. Its free capacity cannot be used
        # as an independent VRAM budget, so retain device identity only.
        devices.append({"index": index, "name": name,
                        "total_bytes": 0, "free_bytes": None,
                        "unified_memory": True, "backend": "metal"})
    return devices


def _discover_nvidia_gpus():
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    import csv
    for fields in csv.reader(result.stdout.splitlines()):
        fields = [f.strip() for f in fields]
        if len(fields) != 4:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        # Unified-memory chips (e.g. NVIDIA GB10 Grace Blackwell) have no
        # discrete VRAM pool, so nvidia-smi reports memory.total/memory.free
        # as "[N/A]" rather than a number. Fall back to system RAM figures
        # in that case instead of silently dropping the GPU from discovery.
        try:
            total, free = int(fields[2]), int(fields[3])
        except ValueError:
            try:
                meminfo = Path("/proc/meminfo").read_text()
                total = int(re.search(r"MemTotal:\s+(\d+)", meminfo).group(1)) // 1024
                free = (memory_available() or 0) // (1024 * 1024)
            except (OSError, AttributeError, CgroupError):
                total = free = 0
        name = fields[1]
        unified = any(token in name.lower() for token in ("gb10", "jetson", "grace blackwell"))
        devices.append({"index": index, "name": name,
                        "total_bytes": total * 1024 * 1024,
                        "free_bytes": free * 1024 * 1024,
                        "unified_memory": unified})
    return devices


_HIPINFO_UNITS = {"B": 1, "KB": 1024, "MB": 1024 ** 2,
                  "GB": 1024 ** 3, "TB": 1024 ** 4}


def _hipinfo_executable():
    """Locate hipInfo.exe, preferring the runtime Colibri will actually bind.

    rocm-smi does not exist on Windows -- neither the HIP SDK installer nor a
    source build ships it -- so the rocm-smi probe below finds nothing there and
    every Windows AMD host planned CPU-only. hipInfo.exe is what both shipped
    SDKs do provide, and it sits in the same directory as amdhip64_7.dll.

    Lookup order, and why:

    1. ``COLI_HIP_RUNTIME_DIR`` -- the directory the loader binds the HIP
       runtime from (docs/windows.md). hipInfo lives beside amdhip64_7.dll
       there, so its answer describes the runtime the engine will actually
       load.
    2. ``HIP_PATH``\\bin -- the SDK root the Windows HIP SDK installer sets, and
       the same variable c/Makefile derives HIP_SDK_ROOT from.
    3. ``PATH``.

    The order is the point on a host carrying more than one HIP install: a
    stale ambient HIP_PATH must not describe the hardware through a runtime the
    engine is not going to bind. No install location is hardcoded.
    """
    if sys.platform != "win32":
        return None
    candidates = []
    runtime_dir = os.environ.get("COLI_HIP_RUNTIME_DIR")
    if runtime_dir:
        candidates.append(Path(runtime_dir.strip('"')) / "hipInfo.exe")
    hip_path = os.environ.get("HIP_PATH")
    if hip_path:
        candidates.append(Path(hip_path.strip('"')) / "bin" / "hipInfo.exe")
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    found = shutil.which("hipInfo")
    return Path(found) if found else None


def _hipinfo_bytes(value):
    """``"89.39 GB"`` -> bytes, or None. hipInfo divides by 1024 (it prints a
    65536-byte shared block as ``64.00 KB``), so the units are binary."""
    match = re.match(r"([0-9.]+)\s*([KMGT]?B)\b", (value or "").strip())
    if not match:
        return None
    try:
        return int(float(match.group(1)) * _HIPINFO_UNITS[match.group(2)])
    except (ValueError, KeyError, OverflowError):
        return None


def _parse_hipinfo(text):
    """Devices from hipInfo output, one block per ``device#`` line.

    A block that does not carry both a name and a total is dropped rather than
    completed with zeros: a half-trusted device is worse than no device,
    because the zeros would read as measurements.

    ``memInfo.free`` is deliberately NOT carried into ``free_bytes``. hipInfo
    does report it, but on integrated hardware it has not been qualified as a
    budget. On the validated gfx1151 host, four controlled rebooted sessions
    varying the firmware shared-memory limit reported the same 76.79 GiB total
    at the ~6, ~32 and ~64 GB settings -- about 12.8x the configured limit at
    the minimum -- and 93.00 GiB only at the ~123 GB maximum, with
    Windows-visible memory unchanged throughout. What that figure permits, and
    at what cost to the host, is a later slice; until then the value is
    observed and discarded, and ``free_bytes`` stays None. See
    plans_placement().
    """
    blocks = []
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("device#"):
            index = stripped[len("device#"):].strip()
            current = {"index": int(index)} if index.isdigit() else None
            if current is not None:
                blocks.append(current)
            continue
        if current is None:
            continue
        key, sep, value = line.partition(":")
        if sep:
            current[key.strip()] = value.strip()
    devices = []
    for block in blocks:
        name = block.get("Name", "")
        total = _hipinfo_bytes(block.get("memInfo.total")
                               or block.get("totalGlobalMem"))
        if not name or not total:
            continue
        devices.append({"index": block["index"], "name": name,
                        "arch": block.get("gcnArchName", ""),
                        "total_bytes": total,
                        "free_bytes": None,
                        "unified_memory": block.get("isIntegrated") == "1"})
    return devices


def _discover_amd_gpus_windows():
    hipinfo = _hipinfo_executable()
    if hipinfo is None:
        return []
    try:
        result = subprocess.run([str(hipinfo)], text=True, capture_output=True,
                                check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    return _parse_hipinfo(result.stdout)


def _discover_amd_gpus():
    """ROCm/HIP discovery. Windows goes through hipInfo (see above); everywhere
    else through rocm-smi (#662), which is absent on non-AMD hosts so this
    returns [] there. rocm-smi --showmeminfo vram reports BYTES (unlike
    nvidia-smi's MiB), so no unit scaling. Column names drift across ROCm
    versions, so match them by substring rather than position. The rocm-smi
    branch remains VERIFY-on-AMD-hardware (labelled hardware-owner-needed) --
    authored without a ROCm host to test against."""
    if sys.platform == "win32":
        return _discover_amd_gpus_windows()
    command = ["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--csv"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    import csv
    rows = list(csv.DictReader(result.stdout.splitlines()))
    if not rows:
        return []

    def find_col(row, *needles):
        for key in row:
            low = (key or "").lower()
            if all(n in low for n in needles):
                return key
        return None

    devices = []
    for i, row in enumerate(rows):
        dev = (row.get("device") or "").strip()
        m = re.search(r"(\d+)", dev)
        index = int(m.group(1)) if m else i
        total_col = find_col(row, "vram", "total", "memory")
        used_col = find_col(row, "vram", "used")
        name_col = (find_col(row, "card", "series") or find_col(row, "card", "model")
                    or find_col(row, "product"))
        try:
            total = int((row.get(total_col) or "0").strip())
        except (ValueError, TypeError):
            total = 0
        try:
            used = int((row.get(used_col) or "0").strip())
        except (ValueError, TypeError):
            used = 0
        free = max(total - used, 0)
        name = (row.get(name_col) or "").strip() if name_col else ""
        devices.append({"index": index, "name": name or f"AMD GPU {index}",
                        "total_bytes": total, "free_bytes": free,
                        "unified_memory": False})
    return devices


def plans_placement(gpu):
    """Whether a discovered device's memory is qualified to drive placement.

    Discovery is a fact; a placement budget is a policy decision, and the two
    are not the same thing. ``free_bytes`` normally carries both, because a
    discrete card's free VRAM *is* the budget. It is ``None`` for a device that
    exists and is worth reporting but whose free memory has not been qualified
    as a Colibri budget -- today, a Windows AMD part found through hipInfo,
    where the GPU and the host draw on one physical pool and the relationship
    between the runtime's free figure and host-available memory has not been
    measured.

    ``None`` is deliberately distinct from ``0``. Zero is a measurement ("the
    card is full") and keeps every behaviour it has always had. ``None`` says
    "not measured in a way this planner may spend", which is a different claim
    and must not silently collapse into the other -- hence ``is not None``
    rather than a truthiness test.
    """
    return gpu.get("free_bytes") is not None


def _physical_cores_warn(message):
    """Visibility for a mis-detected core count: a silent "1" here becomes
    OMP_NUM_THREADS=1 and pins the whole run to a single core (#325). Emit on
    stderr so it surfaces in the [PLAN]/[OMP] stream without being swallowed."""
    print(f"[plan] warning: {message}", file=sys.stderr)


def physical_cpu_count():
    """Number of physical CPU cores (not SMT siblings).

    Per-expert matmul regions are tiny and back-to-back; two SMT siblings share
    one AVX-512 unit and contend, so logical (SMT) counts over-subscribe and
    hurt throughput. We want true physical cores. A silent 1 here propagates to
    OMP_NUM_THREADS=1 and pins the run to one core (#325), so every fallback
    must be visible, never just ``or 1``.
    """
    if sys.platform == "win32":
        # Contiamo i core fisici veri con GetLogicalProcessorInformationEx
        # (RelationProcessorCore). Le firme vanno dichiarate: su Python a 64 bit
        # una WinAPI non dichiarata ritorna c_int (32 bit) e riceve i puntatori
        # come c_int di default, quindi il probe puo' fallire silenziosamente.
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetLogicalProcessorInformationEx.argtypes = [
                ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            k32.GetLogicalProcessorInformationEx.restype = ctypes.c_int
            need = ctypes.c_ulong(0)
            k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(need))
            buf = (ctypes.c_char * need.value)()
            if k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(need)):
                raw, cores, off = bytes(buf), 0, 0
                while off + 8 <= need.value:
                    relationship = int.from_bytes(raw[off:off + 4], "little")
                    size = int.from_bytes(raw[off + 4:off + 8], "little")
                    if size <= 0:
                        break
                    if relationship == 0:  # RelationProcessorCore
                        cores += 1
                    off += size
                if cores:
                    return cores
            _physical_cores_warn("GetLogicalProcessorInformationEx returned no cores")
        except (OSError, ValueError, AttributeError) as error:
            _physical_cores_warn(f"Windows core probe failed: {error}")
    if sys.platform == "darwin":
        # Apple Silicon's performance cores pace barriered expert matmuls. The
        # physical count includes efficiency cores, which is not the useful
        # OpenMP team size for this workload.
        try:
            result = subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                                    text=True, capture_output=True, check=True, timeout=5)
            cores = int(result.stdout.strip())
            if cores > 0:
                return cores
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        try:
            result = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], text=True,
                                    capture_output=True, check=True, timeout=5)
            cores = int(result.stdout.strip())
            if cores > 0:
                return cores
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            _physical_cores_warn(f"sysctl core probe failed: {error}")
    try:
        # Ask lscpu for exactly core,socket and dedupe on (core, socket).
        # Counting un-deduplicated rows would return logical threads (SMT),
        # which was the original over-subscription bug. Empty fields ("-")
        # mark an offline core/socket and fail int() -> skipped.
        #
        # Column layout robustness: `lscpu -p=<list>` emits *exactly* the
        # requested columns (no CPU prefix), while bare `lscpu -p` prepends
        # CPU. We requested two columns, but take the LAST TWO fields so the
        # parser stays correct whether or not a CPU column is present
        # (JustVugg review: the previous fields[1]/fields[2] indexing assumed
        #  a 3-column layout and regressed 2-column output to the logical
        # count -- the opposite of the fix).
        result = subprocess.run(["lscpu", "-p=core,socket"], text=True,
                                capture_output=True, check=True, timeout=5)
        cores = set()
        for line in result.stdout.splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split(",")
            if len(fields) < 2:
                continue
            try:
                core, socket = int(fields[-2]), int(fields[-1])
            except ValueError:
                continue  # "-" for an offline core/socket
            cores.add((core, socket))
        if cores:
            return len(cores)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        _physical_cores_warn(f"lscpu core probe failed: {error}")
    logical = os.cpu_count()
    if not logical:
        _physical_cores_warn(
            "could not detect any CPU cores; falling back to 1. "
            "Set OMP_NUM_THREADS manually to fix single-core decode (#325).")
        return 1
    _physical_cores_warn(
        f"physical-core probes unavailable; using {logical} logical CPUs "
        f"(SMT may over-subscribe). Set OMP_NUM_THREADS to physical cores if slow.")
    return logical


def _resolve_physical_cores(physical_cpus):
    """Coerce the build_plan() physical-core argument to a sane positive int.

    A None/0/None-ish value reaching here means physical_cpu_count() already
    warned; clamp to 1 (so the engine always gets a positive team size) but keep
    that clamp visible rather than silently masking it as the old ``max(1, int())``
    did (#325)."""
    try:
        count = int(physical_cpus or 0)
    except (TypeError, ValueError):
        count = 0
    if count < 1:
        _physical_cores_warn(
            "physical core count resolved to 0; defaulting to 1. "
            "Set OMP_NUM_THREADS to fix single-core decode (#325).")
        return 1
    return count


def cpu_socket_count():
    """Return the number of physical CPU sockets visible to this process."""
    if not sys.platform.startswith("linux"):
        return 1
    try:
        result = subprocess.run(["lscpu", "-p=socket"], text=True,
                                capture_output=True, check=True, timeout=5)
        sockets = {int(line) for line in result.stdout.splitlines()
                   if line and not line.startswith("#")}
        if sockets:
            return len(sockets)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 1


def _auto_tune(bottleneck_class, projected_hit, gpus, cpu_sockets, plan_has_metal,
               engine_group=None):
    """Derive tuning knobs from the bottleneck classification."""
    tune = {}
    has_gpu = bool(gpus)
    n_gpu = len(gpus)

    # glm53 has its own loader/cache controls and does not consume the generic
    # DRAFT/PIPE/PIN/NUMA knobs below. Recommending them is worse than leaving
    # them unset because `coli tune` then reports changes the engine ignores.
    # The same holds for every engine but colibri.c: DRAFT, PIPE, COLI_CUDA_PIPE,
    # COLI_NUMA and PIN_GB have no reader anywhere else.
    if engine_group is not None and engine_group != "colibri-core":
        return tune

    # MTP: costs more than it saves when compute-bound (#389 measured 42% loss)
    # or streaming-bound (#467 measured 32% loss under CUDA at 85% hit).
    # EXCEPTION: an explicit COLI_CUDA_MTP=1 in the environment is a documented
    # opt-in to test speculation under CUDA (glm.c resolves DRAFT=-1 -> 3 only
    # when it sees the var). Exporting DRAFT=0 here preempted that auto path,
    # so the opt-in was silently inert on the Windows bare-run/auto-tier flows
    # (#467): respect it and let the engine's auto path take over. Unset still
    # gets DRAFT=0 -> MTP off, which is the measured-correct default.
    if os.environ.get("COLI_CUDA_MTP") == "1":
        pass  # explicit opt-in: leave DRAFT to the engine's auto resolution
    elif bottleneck_class == "compute":
        tune["DRAFT"] = {"value": "0",
                         "reason": "compute-bound: MTP batch overhead exceeds yield"}
    elif bottleneck_class == "disk" and projected_hit < 0.90:
        tune["DRAFT"] = {"value": "0",
                         "reason": "low hit rate: MTP widens expert union, adds disk reads"}
    # otherwise leave DRAFT unset (engine default: auto)

    # PIPE: resident pipeline mode depends on GPU count
    if has_gpu and n_gpu == 1:
        tune["COLI_CUDA_PIPE"] = {"value": "1",
                                  "reason": "single GPU: S=1 pipeline gate"}
    elif has_gpu and n_gpu > 1:
        tune["COLI_CUDA_PIPE"] = {"value": "2",
                                  "reason": "multi-GPU: residual stays on-device across layers"}
    elif not has_gpu and bottleneck_class == "disk":
        tune["PIPE"] = {"value": "1",
                        "reason": "overlap disk reads with resident expert compute"}

    # NUMA: selective interleave for GPU hosts, blanket hint for CPU-only
    if cpu_sockets > 1 and has_gpu:
        tune["COLI_NUMA"] = {"value": "1",
                             "reason": "multi-socket + GPU: interleave expert slabs, protect DMA buffers"}
    elif cpu_sockets > 1 and not has_gpu:
        tune["COLI_NUMA"] = {"value": "1",
                             "reason": "multi-socket CPU-only: interleave expert slabs across nodes"}
        tune["_numa_hint"] = "numactl --interleave=all may perform better on CPU-only hosts"

    # OMP: kill hot-thread spin when GPU/Metal owns the power budget
    if plan_has_metal:
        tune["COLI_NO_OMP_TUNE"] = {"value": "1",
                                    "reason": "Metal: OMP spin-wait steals GPU power budget"}

    # PIN: fully resident if RAM allows and no GPU tier competes
    if projected_hit >= 0.99 and not has_gpu:
        tune["PIN_GB"] = {"value": "all",
                          "reason": "enough RAM for full expert residency"}

    return tune


def _next_actions(bottleneck_class, projected_hit, probe_state, probe_gbs,
                  planning_gpus):
    """Turn the capacity diagnosis into bounded, evidence-producing work.

    These are deliberately not more automatic knobs.  The planner may spend
    measured capacity, but it must not invent bandwidth or claim that a larger
    tier will improve end-to-end decode without a run on this machine.
    """
    actions = []
    if bottleneck_class == "disk":
        if probe_gbs is None:
            actions.append({
                "id": "measure-storage",
                "priority": "required",
                "reason": "cold experts remain on disk and no trusted storage probe is available",
                "command": "make -C c iobench && ./c/iobench",
            })
        actions.append({
            "id": "measure-residency",
            "priority": "recommended",
            "reason": f"projected expert residency is {projected_hit:.0%}",
            "command": "coli tune --model <model>",
        })
        if len(planning_gpus) > 1:
            actions.append({
                "id": "profile-interconnect",
                "priority": "recommended",
                "reason": "multiple GPU tiers can move the bottleneck from storage to interconnect",
                "command": "PROF=1 coli run --auto-tier --model <model> <prompt>",
            })
    elif bottleneck_class == "mixed":
        actions.append({
            "id": "profile-overlap",
            "priority": "recommended",
            "reason": "CPU expert tail and GPU work must be timed separately before retuning placement",
            "command": "PROF=1 coli run --auto-tier --model <model> <prompt>",
        })
    elif bottleneck_class in ("compute", "memory"):
        actions.append({
            "id": "measure-kernels",
            "priority": "recommended",
            "reason": "weights are resident; kernel and memory-bandwidth timings decide the next optimization",
            "command": "PROF=1 coli run --auto-tier --model <model> <prompt>",
        })
    return actions


POLICIES = {
    "quality": {"preserve_quantization": True, "preserve_router": True},
    "balanced": {"preserve_quantization": True, "preserve_router": True},
    "experimental-fast": {"preserve_quantization": False, "preserve_router": False},
}


def _family_expert_cache_knob(family_id, cache_bytes):
    """Env var that actually sizes the expert LRU, when it is not RAM_GB.

    Kimi K3 reads K3_EXPERT_GB (default 8) and treats RAM_GB as a ceiling;
    main() never takes argv as a cap. GLM-5.3 reads GLM53_EXPERT_GB and never
    RAM_GB. Exporting only RAM_GB therefore left both engines on their own
    default cache under --auto-tier.
    """
    if (not isinstance(cache_bytes, int) or isinstance(cache_bytes, bool)
            or cache_bytes <= 0):
        return None
    if family_id == "kimi":
        return ("K3_EXPERT_GB",
                "Kimi K3 sizes the expert LRU from K3_EXPERT_GB; RAM_GB is only a ceiling")
    if family_id == "glm53":
        return ("GLM53_EXPERT_GB",
                "GLM-5.3 sizes the expert LRU from GLM53_EXPERT_GB, not RAM_GB")
    return None


def build_plan(model, ram_gb=0, context=4096, gpu_indices=None, vram_gb=0,
               available_memory=None, available_disk=None, gpus=None,
               policy="quality", physical_cpus=None, cpu_sockets=None,
               kv_slots=1, env=None, vulkan=None):
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    info = qwen38_int4_sidecar(analyze_model(model))
    env_now = os.environ if env is None else env
    info = _q38_cpu_dense_tensors(info, env_now)
    info = _glm53_dense_tensors(info, env_now)
    # Only the matrices a family's dho pass can release earn RAM credit. The
    # embedding, norms, CPU-only components and unrecognized formats stay reserved.
    vulkan = _vk_device(env_now, vulkan)
    family_id = info["resolved_family"].descriptor.id
    kept = min(info["dense_bytes"], info.get("embed_bytes", 0))
    device_dense_bytes = max(0, info["dense_bytes"] - kept)
    # A partial chain (vk_chain_fit): only the first N layers' tensors go to the device;
    # the CPU's layers and the head keep their host copies.
    chain_fit = vk_chain_fit(info, family_id, env_now, vulkan)
    partial = chain_fit is not None and (chain_fit["n"] < chain_fit["L"] or not chain_fit["tail"])

    def on_device(tensor):
        if not partial:
            return True
        layer = _vk_layer_index(tensor["name"])
        return layer is not None and layer < chain_fit["n"]
    tensors_on_device = [tensor for tensor in info.get("dense_tensors", []) if on_device(tensor)]
    if partial:
        released = sum(_vk_released_tensor_bytes(tensor, family_id, env_now) for tensor in tensors_on_device)
        vk_dense, vk_dense_why = ((False, None) if chain_fit["n"] == 0 else
                                  vk_dense_device_only(released, family_id, env_now, vulkan))
    else:
        vk_dense, vk_dense_why = vk_dense_device_only(
            device_dense_bytes, family_id, env_now, vulkan)
    dense_on_device = 0
    if vk_dense:
        dense_on_device = min(device_dense_bytes, sum(
            _vk_released_tensor_bytes(tensor, family_id, env_now)
            for tensor in tensors_on_device))
        info = dict(info, dense_bytes=info["dense_bytes"] - dense_on_device)
    # On an integrated/software device the remaining device copy still consumes
    # physical RAM. Keeping host copies costs two copies; dropping them saves one,
    # not both. Price the device copy even when COLI_VK_DENSE_HOST=1.
    device_copy = (min(device_dense_bytes, sum(tensor["resident"] for tensor in tensors_on_device))
                   if partial else device_dense_bytes)
    shared_dense_bytes = (device_copy
                          if (vulkan or {}).get("type") in ("integrated", "cpu")
                          and _vk_dense_active(family_id, env_now, vulkan) else 0)
    physical_cpus = physical_cpu_count() if physical_cpus is None else physical_cpus
    cpu_sockets = cpu_socket_count() if cpu_sockets is None else cpu_sockets
    resolved = info["resolved_family"]
    if info.get("qwen38_int4_experts"):
        # qwen38's CUDA tier streams FP8 experts only and stays off when the
        # int4-g64 sidecar is in use (trunk included): plan the CPU.
        if gpu_indices or vram_gb > 0:
            raise ValueError(
                "Qwen3.8 int4-g64 experts (experts-int4g64/) run on the CPU; "
                "Q38_EXPERT_INT4=0 gives the GPU tier the FP8 experts")
        gpus = []
    if not resolved.descriptor.supports_accelerator:
        if gpu_indices:
            raise ValueError(
                f"{resolved.descriptor.display_name} currently supports CPU only; "
                "GPU selection is unavailable")
        if vram_gb > 0:
            raise ValueError(
                f"{resolved.descriptor.display_name} currently supports CPU only; "
                "a VRAM budget is unavailable")
        # Do not let unrelated GPUs distort unified-memory/RAM/cache planning
        # for an engine that cannot execute any part of this model on them.
        gpus = []
    geometry = planner_geometry(resolved, context)
    if (isinstance(kv_slots, bool) or not isinstance(kv_slots, int) or
            not 1 <= kv_slots <= resolved.descriptor.limits.max_kv_slots):
        raise ValueError(f"{resolved.descriptor.id}: invalid KV slot count {kv_slots}")
    automatic_ram = ram_gb <= 0
    if available_memory is None:
        available_memory = memory_available()
    available_memory_known = available_memory is not None
    if not available_memory_known:
        # Nothing could measure host memory (no /proc, vm_stat failed). Report
        # the historical 0 and let the 8 GB compatibility fallback below size
        # the budget; doctor's "could not be measured" warning keys on it.
        available_memory = 0
    elif automatic_ram and available_memory <= 0:
        raise ValueError(
            "no available RAM remains for automatic placement "
            "(host/cgroup memory budget is exhausted)")
    if available_disk is None:
        try:
            usage = shutil.disk_usage(info["path"])
            available_disk = usage.free
        except OSError:
            available_disk = 500 * GB
    gpus = discover_gpus() if gpus is None else gpus
    if gpu_indices is not None:
        wanted = set(gpu_indices)
        gpus = [gpu for gpu in gpus if gpu["index"] in wanted]

    # Every discovered device is reported; only the ones whose free memory is a
    # qualified budget may steer placement. Keeping the two lists apart is what
    # stops "a GPU exists" from being read as "a GPU should be used" -- see
    # plans_placement().
    planning_gpus = [gpu for gpu in gpus if plans_placement(gpu)]

    placement_unified = any(gpu.get("unified_memory", False)
                            for gpu in planning_gpus)
    unified = placement_unified or _host_unified_memory() or bool(shared_dense_bytes)
    typical = info["typical_expert_bytes"]
    max_expert = info["max_expert_bytes"] or typical
    kv_bytes = (geometry.context_state_bytes + geometry.fixed_state_bytes) * kv_slots
    kv_buffer = geometry.workspace_bytes
    # expert_fixed_bytes is retained once per model (currently Qwen3.8's
    # normalized FP8 scale bank), unlike per_cap_bytes which is paid for
    # every cache slot. Include it in the resident runtime reservation so the
    # selected capacity cannot overrun the model's actual allocation.
    runtime_bytes = int(1.2 * GB + 2.5 * GB + 64 * max_expert +
                        info["expert_fixed_bytes"] + kv_bytes + kv_buffer + shared_dense_bytes)
    per_cap = info["per_cap_bytes"]
    configured_experts = geometry.configured_experts

    reserve = 2 * GB
    gpu_plan = []
    safe_vram = 0
    for gpu in gpus:
        usable = max(0, gpu["free_bytes"] - reserve) if plans_placement(gpu) else 0
        safe_vram += usable
        gpu_plan.append(dict(gpu, reserve_bytes=reserve, usable_bytes=usable))
    requested_vram = int(vram_gb * GB) if vram_gb > 0 else safe_vram
    # The engine's placer (qwen36_tier.c, COLI_PLACE=auto) takes the dense
    # trunk first -- it is read every token -- on the first planned device,
    # if it fits that device's allowance; the experts get what is left.
    trunk_bytes = int(info.get("trunk_int8_bytes", 0) or 0)
    trunk_placed = 0
    if trunk_bytes and gpu_plan and gpu_plan[0]["usable_bytes"] >= trunk_bytes \
            and requested_vram >= trunk_bytes:
        trunk_placed = trunk_bytes
    requested_vram = max(0, requested_vram - trunk_placed)
    safe_vram = max(0, safe_vram - trunk_placed)
    requested_vram_before_clamp = requested_vram
    unified_pool = max(0, available_memory - info["dense_bytes"] - runtime_bytes)
    if placement_unified:
        # Unified devices expose one physical pool to CUDA and the host. Do not
        # let an expert tier consume pages that the RAM tier also believes are
        # available. Dense/runtime reservations are shared exactly once below.
        requested_vram = min(requested_vram, unified_pool)
    vram_limit = unified_pool if placement_unified else safe_vram
    vram_budget = min(requested_vram, vram_limit, info["expert_bytes"])
    vram_experts = int(vram_budget // typical) if typical else 0
    hot_bytes = min(info["expert_bytes"], vram_experts * typical)
    warnings = []
    if placement_unified:
        requested_ram = int(ram_gb * GB) if ram_gb > 0 else int(available_memory * 0.88)
        requested_ram_experts = max(0, requested_ram - info["dense_bytes"] - runtime_bytes)
        ram_expert_bytes = min(requested_ram_experts,
                               max(0, unified_pool - vram_budget))
        # The fixed resident footprint can itself exceed a small requested or
        # cgroup-derived budget. Keep reporting the actual admission ceiling in
        # that case; inflating the tier to the fixed footprint would make an
        # exhausted shared-memory plan look admissible. The zero-cache warning
        # below still explains that the model cannot retain even one expert.
        ram_budget = min(requested_ram,
                         info["dense_bytes"] + runtime_bytes + ram_expert_bytes)
        if requested_ram_experts > ram_expert_bytes:
            warnings.append(
                f"RAM budget clamped from {format_bytes(requested_ram)} to "
                f"{format_bytes(ram_budget)} because the GPU shares physical memory")
    else:
        ram_budget = int(ram_gb * GB) if ram_gb > 0 else int(available_memory * 0.88)
    # Keep the legacy 8 GB fallback only when every host-memory probe failed.
    # A known small value -- especially finite cgroup headroom -- must retain
    # the 12% safety reserve and can never be inflated above what is available.
    # An explicit --ram is an intentional override, but its smaller value must
    # likewise not be silently raised behind the user's back.
    if automatic_ram and not available_memory_known and not placement_unified:
        ram_budget = 8 * GB
    cache_bytes = max(0, ram_budget - info["dense_bytes"] - runtime_bytes)
    cap = int(cache_bytes // per_cap) if per_cap else 0
    if configured_experts:
        cap = min(cap, configured_experts)
    if geometry.dense:
        # Nothing to cache: the engine still takes a cache size, and one slot is
        # the smallest it accepts; no slot is ever filled.
        cap = 1
    warm_bytes = min(max(0, info["expert_bytes"] - hot_bytes), cache_bytes)
    cold_bytes = max(0, info["expert_bytes"] - hot_bytes - warm_bytes)

    if geometry.dense:
        if info["dense_bytes"] + runtime_bytes > ram_budget:
            warnings.append(
                f"a dense model keeps every weight resident: {format_bytes(info['dense_bytes'])} "
                f"of weights (as stored; the engine quantizes them at load) do not fit the "
                f"RAM budget of {format_bytes(ram_budget)}")
    elif cap < 1:
        warnings.append("RAM budget cannot hold one expert slot per sparse layer")
    if gpu_indices is not None and len(gpus) != len(set(gpu_indices)):
        warnings.append("one or more requested GPUs were not detected")
    if planning_gpus and vram_budget < requested_vram_before_clamp:
        warnings.append("VRAM tier was clamped by free VRAM, shared memory, or model expert size")
    for gpu in gpus:
        if not plans_placement(gpu):
            warnings.append(
                f"GPU {gpu['index']} ({gpu['name']}) was detected but its free memory is "
                "not qualified as a placement budget on this platform; it is reported "
                "only and drives no automatic tier")
    if placement_unified:
        warnings.append(
            "GPU and RAM share one physical memory pool; budgets were jointly constrained")
    # The plan sizes the hot tier from *free* VRAM, so running it while an engine
    # instance already holds the GPUs silently produces a tiny tier and a
    # pessimistic hit rate that describe nothing. That is exactly when a user
    # reaches for `coli plan` -- before changing a live deployment -- so say so
    # rather than let the numbers be read as a capacity answer.
    if planning_gpus:
        gpu_total = sum(g["total_bytes"] for g in planning_gpus)
        gpu_free = sum(g["free_bytes"] for g in planning_gpus)
        if gpu_total and gpu_free < 0.75 * gpu_total:
            warnings.append(
                f"{format_bytes(gpu_total - gpu_free)} of VRAM is already in use "
                f"(only {format_bytes(gpu_free)} of {format_bytes(gpu_total)} free): "
                "this plan plans against the remainder. Stop the running engine "
                "for a representative plan.")
    if cold_bytes:
        warnings.append("cold expert misses may reach disk; normal decode speed depends on hit rate")

    total_expert = info["expert_bytes"]
    resident_expert = hot_bytes + warm_bytes
    projected_hit = resident_expert / total_expert if total_expert else 1.0

    if geometry.dense:
        bottleneck = ("GPU compute (dense model, weights in VRAM)" if trunk_placed else
                      "RAM bandwidth (dense model: every weight is read for every token)")
        bottleneck_class = "compute" if trunk_placed else "memory"
    elif cold_bytes:
        bottleneck = "disk expert misses"
        bottleneck_class = "disk"
    elif warm_bytes and planning_gpus:
        bottleneck = "CPU expert tail and GPU compute"
        bottleneck_class = "mixed"
    elif projected_hit >= 0.99:
        if planning_gpus:
            bottleneck = "GPU compute and interconnect"
        else:
            bottleneck = "CPU expert compute (fully resident)"
        bottleneck_class = "compute"
    else:
        bottleneck = "CPU expert compute and RAM bandwidth"
        bottleneck_class = "memory"

    tune = _auto_tune(bottleneck_class, projected_hit, planning_gpus, cpu_sockets,
                      plan_has_metal=False,
                      engine_group=resolved.descriptor.engine_group)
    # DRAFT/PIPE/PIN/NUMA stay colibri.c-only (#1585). These two are the
    # opposite leftover: the engines that size the expert LRU from their own
    # *EXPERT_GB variable, which RAM_GB does not set.
    knob = _family_expert_cache_knob(resolved.descriptor.id, cache_bytes)
    if knob:
        name, reason = knob
        tune[name] = {"value": f"{cache_bytes / GB:.3f}", "reason": reason}
    probe_state, probe_gbs = ssd_probe_state(info["path"])
    actions = _next_actions(bottleneck_class, projected_hit, probe_state,
                            probe_gbs, planning_gpus)

    return {
        "version": 2,
        "policy": {"name": policy, **POLICIES[policy],
                   "quality_preserving": policy != "experimental-fast"},
        "model": {**{key: value for key, value in info.items()
                     if key not in ("config", "resolved_family")},
                  "family_id": resolved.descriptor.id,
                  "model_type": resolved.model_type,
                  "configured_experts": configured_experts},
        "cpu": {"physical_cores": _resolve_physical_cores(physical_cpus),
                 "sockets": max(1, int(cpu_sockets)),
                 "thread_policy": "physical-cores"},
        "memory": {"unified": unified, "available_bytes": available_memory},
        "tiers": {
            "disk": {"role": "cold-backing", "model_bytes": info["model_bytes"],
                     "available_bytes": available_disk, "cold_expert_bytes": cold_bytes},
            "ram": {"role": "resident+warm-experts", "available_bytes": available_memory,
                    "budget_bytes": ram_budget, "dense_bytes": info["dense_bytes"],
                    "dense_on_device_bytes": dense_on_device,
                    "dense_on_device_reason": vk_dense_why if vk_dense else None,
                    "vk_chain_layers": ({"on_device": chain_fit["n"], "layers": chain_fit["L"],
                                         "tail": chain_fit["tail"], "forced": chain_fit["forced"]}
                                        if chain_fit is not None else None),
                    "shared_device_dense_bytes": shared_dense_bytes,
                    "runtime_bytes": runtime_bytes,
                    "expert_fixed_bytes": info["expert_fixed_bytes"],
                    "sequence_state_bytes": geometry.context_state_bytes,
                    "fixed_state_bytes": geometry.fixed_state_bytes,
                    "workspace_bytes": geometry.workspace_bytes,
                    "expert_cache_bytes": cache_bytes,
                    "warm_expert_bytes": warm_bytes, "cache_slots_per_layer": cap},
            "vram": {"role": "hot-experts", "devices": gpu_plan,
                     "budget_bytes": vram_budget, "hot_expert_bytes": hot_bytes,
                     "trunk_bytes": trunk_placed,
                     "expert_capacity": vram_experts, "requires_host_backing": False},
        },
        "expected_bottleneck": bottleneck,
        "bottleneck_class": bottleneck_class,
        "projected_hit_rate": round(projected_hit, 4),
        "tune": tune,
        "next_actions": actions,
        # Un motore solo-CPU non ha un tier VRAM: annunciarlo comunque fa
        # scrivere al piano una riga che nessuno puo' eseguire.
        "decisions": ([{"target": "VRAM", "reason": "dense trunk as int8 residents"}]
                      if trunk_placed else []) + (
            [{"target": "RAM", "reason": "dense model: every weight resident, read every token"}]
            if geometry.dense else
            ([{"target": "VRAM", "reason": "profile-ranked hot experts"}]
             if resolved.descriptor.supports_accelerator else []) + [
                {"target": "RAM", "reason": "warm experts execute on CPU without quality loss"},
                {"target": "Disk", "reason": "immutable recovery source for cold experts"},
            ]),
        "warnings": warnings,
        # #379: read-only surfacing of the cached Metal-cache storage probe, if
        # the engine has already measured this model dir. gbs is None unless
        # the engine itself would trust the cache; the state says WHY (#386 r2,
        # F10) -- never re-measured or guessed here.
        "ssd_probe_gbs": probe_gbs,
        "ssd_probe_state": probe_state,
    }


def environment_for_plan(plan, env=None, cuda_enabled=True):
    """Apply a plan without overriding explicit user environment settings."""
    result = dict(env or {})
    result.setdefault("COLI_POLICY", plan["policy"]["name"])
    result.setdefault("OMP_NUM_THREADS", str(plan["cpu"]["physical_cores"]))
    # NOTE: we intentionally do NOT set OMP_PROC_BIND / OMP_PLACES here.
    # The engine's own hot-thread tuning (glm.c main(), the COLI_OMP_TUNED
    # self-exec) sets OMP_PROC_BIND=close with overwrite=0 -- it prefers
    # packing the team onto adjacent cores for the tiny back-to-back per-expert
    # matmuls. Pre-setting OMP_PROC_BIND=spread here ran first and won (the
    # engine's overwrite=0 setenv could not override an already-set var), and
    # spread + OMP_PLACES=cores collapsed the team to one CPU on some libgomp /
    # multi-socket topologies (#325: --auto-tier pinned decode to 1 core on a
    # 64-core box even with OMP_NUM_THREADS=64). Leaving affinity to the engine
    # makes --auto-tier match the plain (working) path. A user who wants a
    # specific policy can still set OMP_PROC_BIND/OMP_PLACES in the environment
    # themselves -- setdefault above only covers OMP_NUM_THREADS.
    tune = plan.get("tune", {})
    for key, entry in tune.items():
        if key.startswith("_"):
            continue
        result.setdefault(key, entry["value"])
    if plan["policy"]["name"] == "balanced":
        result.setdefault("REPIN", "64")
    ram = plan["tiers"]["ram"]
    result.setdefault("RAM_GB", f"{ram['budget_bytes'] / GB:.3f}")
    knob = _family_expert_cache_knob(plan.get("model", {}).get("family_id"),
                                    ram.get("expert_cache_bytes"))
    if knob:
        result.setdefault(knob[0], f"{ram['expert_cache_bytes'] / GB:.3f}")
    planned_cap = ram.get("cache_slots_per_layer")
    if (plan.get("model", {}).get("family_id") == "qwen38" and
            (not isinstance(planned_cap, int) or
             isinstance(planned_cap, bool) or planned_cap < 1)):
        raise ValueError(
            "Qwen3.8 RAM budget cannot hold one expert slot per loaded layer")
    if plan.get("model", {}).get("family_id") == "qwen38":
        disabled = [name for name in ("Q38_NATIVE_FP8", "Q38_NATIVE_BF16")
                    if result.get(name) == "0"]
        if disabled:
            raise ValueError(
                "Qwen3.8 auto-tier requires native expert storage; unset " +
                ", ".join(disabled) +
                " or disable auto-tier and choose an explicit cache cap")
    if (isinstance(planned_cap, int) and not isinstance(planned_cap, bool) and
            planned_cap >= 1):
        # Private bridge for engines whose expert capacity is an argv value.
        # The gateway consumes and removes it before starting the engine.
        result.setdefault("COLI_PLAN_CAP", str(planned_cap))

    vram = plan["tiers"]["vram"]
    # Report every device, but only name the placement-qualified ones to the
    # engine: COLI_GPU/COLI_GPUS is an instruction, not an inventory.
    devices = [device["index"] for device in vram["devices"] if plans_placement(device)]
    if not cuda_enabled or not devices or vram["budget_bytes"] <= 0:
        return result
    if result.get("COLI_CUDA", "1") == "0":
        return result

    result.setdefault("COLI_CUDA", "1")
    if "COLI_GPU" not in result and "COLI_GPUS" not in result:
        key = "COLI_GPU" if len(devices) == 1 else "COLI_GPUS"
        result[key] = ",".join(map(str, devices))
    result.setdefault("CUDA_EXPERT_GB", f"{vram['budget_bytes'] / GB:.3f}")
    if result.get("PIN"):
        result.setdefault("PIN_GB", f"{vram['budget_bytes'] / GB:.3f}")
    return result


def format_bytes(value):
    return f"{value / GB:.1f} GB"


def format_plan(plan):
    model, tiers = plan["model"], plan["tiers"]
    policy=plan["policy"]
    lines = [f"policy {policy['name']} · quality-preserving {'yes' if policy['quality_preserving'] else 'no'}",
             f"model  {model['shards']} shards · {format_bytes(model['model_bytes'])}",
             f"disk   {format_bytes(tiers['disk']['cold_expert_bytes'])} cold experts · "
             f"{format_bytes(tiers['disk']['available_bytes'])} free",
             f"RAM    {format_bytes(tiers['ram']['budget_bytes'])} budget · "
             f"{format_bytes(tiers['ram']['dense_bytes'])} dense"
             + (f" (+{format_bytes(tiers['ram']['dense_on_device_bytes'])} on the Vulkan device only"
                + (f", the first {tiers['ram']['vk_chain_layers']['on_device']} of "
                   f"{tiers['ram']['vk_chain_layers']['layers']} layers"
                   if (tiers['ram'].get('vk_chain_layers') or {}).get('on_device', 0)
                   < (tiers['ram'].get('vk_chain_layers') or {}).get('layers', 0) else "") + ")"
                if tiers['ram'].get('dense_on_device_bytes') else "") + " · "
             f"{format_bytes(tiers['ram']['runtime_bytes'])} runtime · "
             f"{format_bytes(tiers['ram']['warm_expert_bytes'])} warm experts · "
             f"cap {tiers['ram']['cache_slots_per_layer']}/layer"]
    vram = tiers["vram"]
    if vram["devices"]:
        metal_identity = [gpu for gpu in vram["devices"]
                          if gpu.get("backend") == "metal"
                          and gpu.get("free_bytes") is None]
        if len(metal_identity) == len(vram["devices"]):
            names = ", ".join(f"{gpu['index']}:{gpu['name']}"
                              for gpu in metal_identity)
            lines.append(f"Metal  {names} · unified memory · no independent VRAM budget")
        else:
            names = ", ".join(
                f"{gpu['index']}:{gpu['name']}"
                + ("" if plans_placement(gpu) else " (identity only)")
                for gpu in vram["devices"])
            trunk = vram.get("trunk_bytes", 0)
            lines.append("VRAM   " + (f"{format_bytes(trunk)} int8 trunk + " if trunk else "") +
                         f"{format_bytes(vram['budget_bytes'])} hot tier · "
                         f"~{vram['expert_capacity']} experts · {names}")
    else:
        # Backend-neutral, matching the accelerator wording #903 settled on:
        # an AMD or Intel host that finds nothing is not "no NVIDIA device".
        lines.append("VRAM   no supported GPU detected · CPU path")
    if plan.get("ssd_probe_gbs") is not None:
        lines.append(f"ssd    {plan['ssd_probe_gbs']:.1f} GB/s F_NOCACHE (cached probe, #379)")
    elif plan.get("ssd_probe_state") in SSD_PROBE_PENDING:
        lines.append(f"ssd    {SSD_PROBE_PENDING[plan['ssd_probe_state']]}")
    lines.append(f"limit  {plan['expected_bottleneck']}")
    hit = plan.get("projected_hit_rate", 0)
    lines.append(f"hit    {hit:.0%} projected expert residency")
    tune = plan.get("tune", {})
    if tune:
        lines.append("")
        lines.append("auto-tune:")
        for key, entry in tune.items():
            if key.startswith("_"):
                continue
            lines.append(f"  {key}={entry['value']:12s} {entry['reason']}")
        hint = tune.get("_numa_hint")
        if hint:
            lines.append(f"  hint: {hint}")
    actions = plan.get("next_actions", [])
    if actions:
        lines.append("")
        lines.append("next actions:")
        for action in actions:
            lines.append(f"  [{action['priority']}] {action['id']}: {action['reason']}")
            lines.append(f"    {action['command']}")
    lines.extend(f"warn   {warning}" for warning in plan["warnings"])
    return "\n".join(lines)
