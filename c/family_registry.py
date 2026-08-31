#!/usr/bin/env python3
"""Authoritative model-family registry for Colibri's Python control plane."""

from dataclasses import dataclass
import json
import math
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
    resident_inventory: object = None
    # Quanto pesano in RAM i pesi densi rispetto a come stanno su disco.
    # Vale 1.0 per chi li carica cosi' come sono; un motore che li riquantizza
    # a load time pesa meno, e senza questo il pianificatore direbbe che il
    # modello non ci sta quando invece ci sta.
    dense_load_ratio: object = None      # callable(bytes_su_disco) -> bytes in RAM
    has_gateway_adapter: bool = False
    has_cli_adapter: bool = False
    tune_prompt_template: str = "{prompt}"
    supports_accelerator: bool = True
    # Optional model-owned allocations that are neither dense resident
    # tensors nor per-expert cache entries. Qwen3.8 uses this for the
    # normalized FP8 scale bank shared by every cache slot. Keeping this
    # separate from expert_inventory prevents a fixed allocation from
    # being multiplied by the cache capacity.
    fixed_resident_inventory: object = None


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


def _qwen36_geometry(config, context, _model_dir):
    """Hybrid: only the full_attention layers hold a KV cache; the linear
    (DeltaNet) layers carry a recurrent state whose size does not depend on the
    context at all. One scaled context term would over-promise on a model where
    30 of 40 layers never grow."""
    layers = _required_int(config, "num_hidden_layers", "qwen36")
    kinds = config.get("layer_types")
    if not isinstance(kinds, list) or len(kinds) != layers:
        raise ValueError("qwen36: missing or invalid planning key 'layer_types'")
    full = sum(kind == "full_attention" for kind in kinds)
    kv = (full * context * _required_int(config, "num_key_value_heads", "qwen36") *
          _required_int(config, "head_dim", "qwen36") * 2 * 4)
    key_heads = _required_int(config, "linear_num_key_heads", "qwen36")
    key_dim = _required_int(config, "linear_key_head_dim", "qwen36")
    value_heads = _required_int(config, "linear_num_value_heads", "qwen36")
    value_dim = _required_int(config, "linear_value_head_dim", "qwen36")
    conv_k = _required_int(config, "linear_conv_kernel_dim", "qwen36", 2)
    conv_dim = key_heads * key_dim * 2 + value_heads * value_dim
    fixed = (layers - full) * (value_heads * key_dim * value_dim +
                               conv_dim * (conv_k - 1)) * 4
    return PlannerGeometry(kv, fixed, 0, _required_int(config, "num_experts", "qwen36"))


_QWEN38_PREFILL_BATCH_ROWS = 32
_QWEN38_PREFILL_WORKSPACE_BYTES = 64 << 20


def _qwen38_bounded_prefill(rows, fixed_bytes, row_bytes):
    """Mirror q38_bounded_prefill_rows() without context-sized scratch."""
    selected = min(rows, _QWEN38_PREFILL_BATCH_ROWS)
    while selected > 1 and fixed_bytes + selected * row_bytes > \
            _QWEN38_PREFILL_WORKSPACE_BYTES:
        selected -= 1
    return selected


def _qwen38_geometry(config, context, _model_dir):
    """Qwen3.8-Flash-Next's hybrid state layout.

    Full-attention layers retain GQA K/V plus the raw QSA index key.  Linear
    (GDN) layers retain a recurrent matrix and a four-stream convolution ring;
    the single PLE layer adds its own convolution history and two int64 ngram
    history positions.  Hyper-connection streams also contribute to prefill
    scratch, so workspace is deliberately conservative rather than zero.
    """
    family = "qwen38"
    layers = _required_int(config, "num_hidden_layers", family)
    hidden = _required_int(config, "hidden_size", family)
    vocab = _required_int(config, "vocab_size", family)
    max_positions = _required_int(config, "max_position_embeddings", family)
    eos = _required_int(config, "eos_token_id", family, 0)
    if layers > 512 or hidden > 65536 or max_positions > 262144 or eos >= vocab:
        raise ValueError(f"{family}: model dimensions exceed the native engine limits")
    if context > max_positions:
        raise ValueError(f"{family}: context {context} exceeds max_position_embeddings "
                         f"{max_positions}")
    kinds = config.get("layer_types")
    if not isinstance(kinds, list) or len(kinds) != layers:
        raise ValueError(f"{family}: missing or invalid planning key 'layer_types'")
    # Upstream calls these ``full_attention``; early Qwen4-Exp exports and the
    # tiny oracle call the same QSA layer ``qwen_sparse_attention``.
    allowed_kinds = {"linear_attention", "full_attention", "qwen_sparse_attention"}
    unknown_kinds = sorted({kind for kind in kinds if kind not in allowed_kinds})
    if unknown_kinds:
        raise ValueError(f"{family}: unsupported layer types {unknown_kinds!r}")
    full = sum(kind in ("full_attention", "qwen_sparse_attention") for kind in kinds)
    linear = layers - full

    n_kv = _required_int(config, "num_key_value_heads", family)
    head_dim = _required_int(config, "head_dim", family)
    q_heads = _required_int(config, "num_attention_heads", family)
    if q_heads % n_kv:
        raise ValueError(f"{family}: attention heads are not divisible by KV heads")
    rope = config.get("rope_parameters")
    rope = rope if isinstance(rope, dict) else config
    partial = rope.get("partial_rotary_factor", config.get("partial_rotary_factor", 1.0))
    if (isinstance(partial, bool) or not isinstance(partial, (int, float)) or
            not math.isfinite(partial) or partial <= 0):
        raise ValueError(f"{family}: invalid partial_rotary_factor")
    rotary_dim = int(head_dim * partial)
    if rotary_dim < 1 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"{family}: invalid derived rotary dimension")
    index_kv = _required_int(config, "indexer_kv_heads", family)
    index_dim = _required_int(config, "indexer_head_dim", family)
    index_q = _required_int(config, "indexer_n_heads", family)
    index_budget = _required_int(config, "indexer_budget", family)
    index_ratio = _required_int(config, "indexer_compress_ratio", family)
    if index_kv != 1 or index_dim < rotary_dim or index_budget % index_ratio:
        raise ValueError(f"{family}: invalid sparse indexer geometry")
    # Serve allocates this complete QSA bank once before publishing READY and
    # never grows it in place. One context-sized bank is therefore both the
    # steady allocation and the allocation peak; no old/new growth pair exists.
    context_state = full * context * (2 * n_kv * head_dim + index_kv * index_dim) * 4

    value_heads = _required_int(config, "linear_num_value_heads", family)
    key_dim = _required_int(config, "linear_key_head_dim", family)
    value_dim = _required_int(config, "linear_value_head_dim", family)
    hc_count = _required_int(config, "hc_count", family)
    hc_rank = _required_int(config, "hc_lowrank", family)
    if hc_count > 16 or hc_rank > 65536:
        raise ValueError(f"{family}: invalid hyper-connection geometry")
    key_heads = _required_int(config, "linear_num_key_heads", family)
    gdn_conv = _required_int(config, "linear_conv_kernel_dim", family, 2) - 1
    if value_heads % key_heads or value_dim > 512:
        raise ValueError(f"{family}: invalid Gated DeltaNet geometry")
    gdn_conv_width = (2 * key_heads * key_dim + value_heads * value_dim)
    fixed = linear * (value_heads * key_dim * value_dim +
                      gdn_conv_width * gdn_conv) * 4

    ple_ids = config.get("ple_layer_ids", [])
    if (not isinstance(ple_ids, list) or len(ple_ids) != 1 or
            isinstance(ple_ids[0], bool) or not isinstance(ple_ids[0], int) or
            not 1 <= ple_ids[0] <= layers):
        raise ValueError(f"{family}: invalid planning key 'ple_layer_ids'")
    ple_conv = _required_int(config, "ple_conv_kernel_size", family, 2) - 1
    ngram = _required_int(config, "ngram_size", family, 2)
    heads_per_ngram = _optional_int(config, "heads_per_ngram", 8, 1)
    ngram_heads = (ngram - 1) * heads_per_ngram
    ple_dim = _required_int(config, "ple_embed_dim", family)
    ngram_parts = _optional_int(config, "split_ngram_parts", 1, 1)
    if (ngram != 3 or not 1 <= ngram_heads <= 64 or ple_dim % ngram_heads or
            ple_dim // ngram_heads > 512 or ngram_parts > 512):
        raise ValueError(f"{family}: invalid PLE geometry")
    # PLE's retained convolution ring is per hyper-connection stream: the
    # runtime allocates hc_count * hidden, not the projection embed dimension.
    hc_width = hc_count * hidden
    fixed += len(ple_ids) * (hc_width * ngram * ple_conv * 4 +
                             (ngram - 1) * 8)

    # The one-slot serve path keeps an exact prompt-prefix checkpoint. QSA rows
    # remain in their existing append-only cache, while every recurrent/PLE
    # payload is copied once so decode can mutate the live state without
    # damaging the reusable prompt. It also retains the prompt token IDs and
    # final prompt logits. The adapter/runtime base reserve covers allocator and
    # pointer metadata; these are the complete model-sized payloads.
    prefix_snapshot = fixed + vocab * 4
    fixed += prefix_snapshot
    context_state += context * 4

    # Match the peak buffers held concurrently by step(): the persistent
    # hyper/mixed/inject/block rows plus the largest layer-local row set. The
    # base planner reserve still covers allocator and non-scaling overhead.
    base_row = hc_width + 2 * hidden + hc_count
    gated_residual_row = base_row + 2 * hc_width + hc_rank
    qsa_row = (base_row + 2 * q_heads * head_dim + 2 * n_kv * head_dim +
               (index_q + index_kv) * index_dim + q_heads * head_dim)
    ple_row = base_row + hc_width

    experts = _required_int(config, "num_experts", family)
    topk = _required_int(config, "num_experts_per_tok", family)
    if experts > 1024 or topk > 256 or topk > experts:
        raise ValueError(f"{family}: invalid routed-expert geometry")
    intermediate = _required_int(config, "moe_intermediate_size", family)
    shared = _required_int(config, "shared_expert_intermediate_size", family)
    moe_fixed = experts + 3 * shared + 3 * intermediate + 2 * hidden
    gdn_fixed = (2 * gdn_conv_width + 3 * value_heads * value_dim +
                 2 * value_heads + 2 * value_heads * key_dim)
    ple_fixed = ple_dim + 5 * hc_width + hidden
    qsa_fixed = (index_q * index_dim + index_dim + index_budget + index_ratio - 1 +
                 max(2 * (context // index_ratio),
                     head_dim + index_budget + index_ratio - 1))
    workspace = (context * max(gated_residual_row, qsa_row, ple_row) +
                 max(moe_fixed, gdn_fixed, ple_fixed, qsa_fixed)) * 4

    # q38_moe_prefill() and q38_deltanet() batch only a bounded row chunk.
    # Account their exact simultaneous allocations beside the context-sized
    # outer hyper/mixed/inject/block rows. Q38RouteAssignment is one int plus
    # one float (8 bytes), and supported hosts use 64-bit pointers.
    moe_assignment_bytes = 2 * (hidden + intermediate) * 4 + 8 + 2 * 4
    moe_row_bytes = (experts + 3 * shared + hidden + 1) * 4 + \
        topk * moe_assignment_bytes
    moe_fixed_bytes = experts * (4 * 4 + 8) + 4
    moe_rows = _qwen38_bounded_prefill(
        context, moe_fixed_bytes, moe_row_bytes)
    moe_chunk_bytes = moe_fixed_bytes + moe_rows * moe_row_bytes

    delta_row_bytes = (gdn_conv_width + 2 * value_heads * value_dim +
                       2 * value_heads) * 4
    delta_fixed_bytes = (gdn_conv_width +
                         2 * value_heads * key_dim +
                         value_heads * value_dim) * 4
    delta_rows = _qwen38_bounded_prefill(
        context, delta_fixed_bytes, delta_row_bytes)
    delta_chunk_bytes = delta_fixed_bytes + delta_rows * delta_row_bytes
    prefill_workspace = context * base_row * 4 + max(
        moe_chunk_bytes, delta_chunk_bytes)
    workspace = max(workspace, prefill_workspace)
    return PlannerGeometry(context_state, fixed, workspace,
                           experts)


def _olmoe_geometry(config, context, _model_dir):
    """Conventional full multi-head attention: every layer holds a K and a V
    cache in fp32, and olmoe.c sizes both at num_attention_heads * head_dim, not
    num_key_value_heads (`c/olmoe.c:1019-1020`, with head_dim = hidden_size //
    num_attention_heads at `:327`). There is no recurrent or convolutional state,
    and the KV cache is the only buffer that grows with context, so fixed and
    workspace are zero -- the bounded per-forward scratch (attention scores,
    logits, expert temporaries) is already covered by the base runtime reserve."""
    layers = _required_int(config, "num_hidden_layers", "olmoe")
    hidden = _required_int(config, "hidden_size", "olmoe")
    heads = _required_int(config, "num_attention_heads", "olmoe")
    head_dim = hidden // heads
    if head_dim < 1:
        raise ValueError("olmoe: hidden_size is smaller than num_attention_heads")
    experts = _required_int(config, "num_experts", "olmoe")
    state = layers * context * heads * head_dim * 2 * 4
    return PlannerGeometry(state, 0, 0, experts)



def _kimi_geometry(config, context, _model_dir):
    """Kimi K3: hybrid -- 69 KDA (recurrent) + 24 gated MLA layers.

    The engine marks KDA layers via linear_attn_config.kda_layers (1-based
    indices, kimi_k3.c:552-555); every other layer is a gated MLA layer.
    Only the MLA layers keep a positional Lc/Rc cache (kimi_k3.c:1704-1706),
    so only they scale with context:

        MLA cache = n_mla * context * (kv_lora + qk_rope) * 4

    The KDA layers carry a per-head recurrent state that does NOT depend on
    context (kimi_k3.c:691): kda_heads * kda_hd * kda_hd floats per layer.
    That belongs in fixed_state_bytes, exactly like GLM's indexer state.

    Workspace mirrors the prefill temporary set (C == context):
      KDA (kimi_k3.c:896-898): q,k,v,gp,on,graw = 6*C*P, t1 = C*kda_hd,
                               braw = C*hidden, P = kda_heads*kda_hd
      MLA (kimi_k3.c:992-994): qa = C*q_lora, qv = C*H*qh, ckv = C*(kvl+qr),
                               gv = C*H*vh, ctx = C*H*vh
    The planner reserves the larger of the two (a single forward pass only
    touches one layer type at a time).
    """
    layers = _required_int(config, "num_hidden_layers", "kimi_k3")
    experts = _required_int(config, "num_experts", "kimi_k3")
    hidden = _required_int(config, "hidden_size", "kimi_k3")
    heads = _required_int(config, "num_attention_heads", "kimi_k3")
    q_lora = _required_int(config, "q_lora_rank", "kimi_k3")
    kv_lora = _required_int(config, "kv_lora_rank", "kimi_k3")
    qk_nope = _required_int(config, "qk_nope_head_dim", "kimi_k3")
    qk_rope = _required_int(config, "qk_rope_head_dim", "kimi_k3")
    v_head = _required_int(config, "v_head_dim", "kimi_k3")

    la = config.get("linear_attn_config")
    if not isinstance(la, dict):
        raise ValueError("kimi_k3: missing or invalid planning key 'linear_attn_config'")
    kda_heads = _required_int(la, "num_heads", "kimi_k3")
    kda_hd = _required_int(la, "head_dim", "kimi_k3")
    kda_list = la.get("kda_layers")
    if not isinstance(kda_list, list):
        raise ValueError("kimi_k3: missing or invalid planning key 'linear_attn_config.kda_layers'")
    n_kda = sum(1 for v in kda_list if isinstance(v, int) and 1 <= v <= layers)
    n_mla = layers - n_kda
    if n_mla < 0:
        n_mla = 0

    state = n_mla * context * (kv_lora + qk_rope) * 4
    fixed = n_kda * kda_heads * kda_hd * kda_hd * 4

    kda_proj = kda_heads * kda_hd
    ws_kda = context * (6 * kda_proj + kda_hd + hidden) * 4
    qh = qk_nope + qk_rope
    ws_mla = context * (q_lora + heads * qh + kv_lora + qk_rope +
                        2 * heads * v_head) * 4
    workspace = max(ws_kda, ws_mla)
    return PlannerGeometry(state, fixed, workspace, experts)



# Il default di GLM53_PREFILL_CHUNK in glm53.c. Se cambia li', cambia qui:
# e' una costante con due consumatori.
_GLM53_PREFILL_CHUNK = 128


def _glm53_dense_in_ram(on_disk_bytes):
    """I densi di GLM-5.3 non restano come sul disco.

    Il checkpoint li porta in BF16 e il motore li quantizza al caricamento
    secondo GLM53_BITS (default 4), che e' int4 con una scala ogni 64 colonne:
    0,5625 byte per parametro contro i 2 del file. Contarli come stanno su
    disco fa dire al pianificatore che manca RAM per un solo esperto per layer,
    quando ce ne stanno diciassette."""
    import os
    try:
        bits = int(os.environ.get("GLM53_BITS", "4"))
    except ValueError:
        bits = 4
    per_parameter = {4: 0.5625, 8: 1.0, 32: 4.0}.get(bits, 0.5625)
    return int(on_disk_bytes * per_parameter / 2.0)      # il file e' BF16


def _glm53_geometry(config, context, _model_dir):
    """GLM-5.3-Flash: 34 layer KDA ricorrenti piu' 11 layer MLA con DSA.

    Solo i layer DSA crescono col contesto, e crescono meno di quanto sembri:
    il motore assorbe kv_b_proj (glm53.c, absorb_kvb), quindi in cache finisce
    il latente da kv_lora e non chiavi e valori espansi. Con 64 teste a 256 la
    differenza e' fra 33 KB e 1,39 MB per token. Ai latenti si aggiungono le
    chiavi e i gate dell'indexer, due vettori da index_head_dim per posizione.

    I layer KDA portano uno stato ricorrente che NON dipende dal contesto, piu'
    la finestra della convoluzione corta: entrambi vanno nel fisso.

    Lo spazio di lavoro NON scala col contesto: il prefill va a pezzi da
    GLM53_PREFILL_CHUNK (glm53.c, forward_prefill), quindi i temporanei costano
    quanto un pezzo. Il costo che domina dentro al pezzo non e' l'attenzione ma
    i flussi delle hyper-connections, hc_mult copie del residuo per posizione,
    tenute due volte perche' il passaggio le scambia.
    """
    layers = _required_int(config, "num_hidden_layers", "glm5_next")
    experts = _required_int(config, "n_routed_experts", "glm5_next")
    hidden = _required_int(config, "hidden_size", "glm5_next")
    heads = _required_int(config, "num_attention_heads", "glm5_next")
    q_lora = _required_int(config, "q_lora_rank", "glm5_next")
    kv_lora = _required_int(config, "kv_lora_rank", "glm5_next")
    qk_nope = _required_int(config, "qk_nope_head_dim", "glm5_next")
    v_head = _required_int(config, "v_head_dim", "glm5_next")
    index_hd = _required_int(config, "index_head_dim", "glm5_next")
    hc_mult = _required_int(config, "hc_mult", "glm5_next")

    la = config.get("linear_attn_config")
    if not isinstance(la, dict):
        raise ValueError("glm5_next: missing or invalid planning key 'linear_attn_config'")
    kda_heads = _required_int(la, "num_heads", "glm5_next")
    kda_hd = _required_int(la, "head_dim", "glm5_next")
    kernel = _required_int(la, "short_conv_kernel_size", "glm5_next")

    # Quali layer sono ad attenzione piena: il checkpoint lo dice in due modi e
    # devono concordare, altrimenti non si sa quale credere.
    types = config.get("layer_types")
    full_list = la.get("full_attn_layers")
    if isinstance(types, list) and types:
        n_full = sum(1 for t in types if isinstance(t, str) and "linear" not in t)
    elif isinstance(full_list, list):
        n_full = len(full_list)
    else:
        raise ValueError("glm5_next: missing 'layer_types' and 'linear_attn_config.full_attn_layers'")
    n_kda = layers - n_full
    if n_kda < 0:
        n_kda = 0

    state = n_full * context * (kv_lora + 2 * index_hd) * 4
    kda_proj = kda_heads * kda_hd
    fixed = n_kda * (kda_heads * kda_hd * kda_hd + 3 * kda_proj * kernel) * 4

    # prefill: due banchi di flussi residui, piu' i temporanei del layer piu'
    # caro fra KDA e MLA (un passaggio ne tocca uno solo per volta)
    chunk = _GLM53_PREFILL_CHUNK
    ws_streams = 2 * chunk * hc_mult * hidden * 4
    ws_kda = chunk * (6 * kda_proj + kda_hd + hidden) * 4
    ws_mla = chunk * (q_lora + heads * (qk_nope + kv_lora) + 2 * heads * v_head) * 4
    workspace = ws_streams + max(ws_kda, ws_mla)
    return PlannerGeometry(state, fixed, workspace, experts)


def _inkling_local_layers(config, layers):
    """Which Inkling layers are sliding-window (1-based rule from inkling.c:557-568).

    Precedence, mirroring the engine:
      1. explicit ``layer_types`` array ("hybrid_sliding" marks local layers)
      2. ``local_layer_ids`` array of 0-based layer indices
      3. the default ``(i + 1) % 6 != 0`` rule (5 of every 6 layers sliding)
    """
    lt = config.get("layer_types")
    if isinstance(lt, list):
        if len(lt) != layers:
            raise ValueError("inkling: layer_types length must match num_hidden_layers")
        return [t == "hybrid_sliding" for t in lt]
    ll = config.get("local_layer_ids")
    if isinstance(ll, list):
        local = [False] * layers
        for v in ll:
            if isinstance(v, int) and 0 <= v < layers:
                local[v] = True
        return local
    return [(i + 1) % 6 != 0 for i in range(layers)]


def _inkling_geometry(config, context, _model_dir):
    """Inkling: hybrid GQA -- sliding-window layers (window ring) + global layers
    + short convolutions (sconv) on K/V/attn/mlp + optional DMel audio tower.

    Mirrors the engine's allocation in inkling.c:

    KV cache (kv_alloc, inkling.c:1687-1699): every layer keeps its own
    K/V pair sized from the layer's kv heads and head dim (L_KV/L_HD,
    inkling.c:80-82). Sliding layers are a ring of ``window`` rows
    (kv_ring_rows, inkling.c:1124-1125) -- at context > window that is a
    ~64x cut on 5-of-6 layers; global layers keep the full context.

        state[layer] = 2 * kv * rows * hd * 4,  rows = window | context

    Conv states (inkling.c:888-890, 1646): four depthwise short-conv states
    per layer -- cs[0]/cs[1] are kv-wide (kvdim), cs[2]/cs[3] are hidden-wide
    -- each holding ``conv_k - 1`` history rows. These do not scale with
    context, so they are fixed_state_bytes.

        fixed[layer] = (2 * kvdim + 2 * hidden) * (conv_k - 1) * 4

    Workspace (attention temporaries, inkling.c:1141-1147): q, k, vv, rr
    and ctx are per-batch S buffers; with S == context at prefill the peak
    is the maximum across layer types (global vs sliding differ).

        ws = S * (2 * qdim + 2 * kvdim + H * d_rel) * 4

    Audio tower (inkling.c:808-832): a DMel encoder embedding table
    [mel_bins * mel_vocab, hidden] plus one RMSNorm weight. It is a fixed
    resident allocation when ``audio_config`` is present, so it joins
    fixed_state_bytes.
    """
    layers = _required_int(config, "num_hidden_layers", "inkling")
    experts = _required_int(config, "n_routed_experts", "inkling")
    hidden = _required_int(config, "hidden_size", "inkling")
    heads = _required_int(config, "num_attention_heads", "inkling")
    n_kv = _required_int(config, "num_key_value_heads", "inkling")
    head_dim = _required_int(config, "head_dim", "inkling")
    swa_heads = _optional_int(config, "swa_num_attention_heads", heads, 1)
    swa_kv = _optional_int(config, "swa_num_key_value_heads", n_kv, 1)
    swa_hd = _optional_int(config, "swa_head_dim", head_dim, 1)
    window = _optional_int(config, "sliding_window_size", 512, 1)
    d_rel = _optional_int(config, "d_rel", 16, 1)
    conv_k = _optional_int(config, "sconv_kernel_size",
                           _optional_int(config, "conv_kernel_size", 4, 1), 1)

    local = _inkling_local_layers(config, layers)
    if len(local) != layers:
        raise ValueError("inkling: layer_types length must match num_hidden_layers")

    state = 0
    fixed = 0
    workspace = 0
    for i in range(layers):
        kv = swa_kv if local[i] else n_kv
        hd = swa_hd if local[i] else head_dim
        h = swa_heads if local[i] else heads
        rows = window if (local[i] and window > 0 and window < context) else context
        state += 2 * kv * rows * hd * 4
        kvdim = kv * hd
        fixed += (2 * kvdim + 2 * hidden) * (conv_k - 1) * 4
        qdim = h * hd
        ws = context * (2 * qdim + 2 * kvdim + h * d_rel) * 4
        workspace = max(workspace, ws)

    ac = config.get("audio_config")
    if isinstance(ac, dict):
        mel_bins = _optional_int(ac, "n_mel_bins", 80, 1)
        mel_vocab = _optional_int(ac, "mel_vocab_size", 16, 1)
        fixed += (mel_bins * mel_vocab * hidden + hidden) * 4

    return PlannerGeometry(state, fixed, workspace, experts)



def _dsv4_geometry(config, context, _model_dir):
    """DeepSeek V4: sliding-window ring + per-layer compressor/indexer states.

    Mirrors the engine's own ``context_bytes()`` (deepseek_v4.c:1271-1284)
    exactly -- this is the resident cache the C runtime sizes at load:

        base       = num_hidden_layers * sliding_window * head_dim * 4
        per layer: ratio = compress_ratios[layer]
                   compressed = ceil(context / ratio)
                   total += compressed * head_dim * 4
                   if ratio == 4:
                       total += compressed * index_head_dim * 4

    The window ring (``base``) does not depend on the context: it is a fixed
    circular buffer, so it belongs in fixed_state_bytes. The compressor and
    indexer states scale with ``context / ratio`` and belong in
    context_state_bytes.

    Workspace mirrors the batched-attention scratch set
    (deepseek_v4.c:2741-2750): qa (q_rank), q (q_width), kv (head_dim),
    attended (q_width), oa (oa_width), norm (max(q_rank, head_dim)) -- with
    q_width = heads * head_dim and oa_width = o_groups * o_lora_rank. With
    batch == context at prefill the peak is the sum of the batch-wide float
    buffers plus the single norm buffer.

    Experts: configured_experts = n_routed_experts.
    """
    layers = _required_int(config, "num_hidden_layers", "deepseek_v4")
    experts = _required_int(config, "n_routed_experts", "deepseek_v4")
    hidden = _required_int(config, "hidden_size", "deepseek_v4")
    heads = _required_int(config, "num_attention_heads", "deepseek_v4")
    head_dim = _required_int(config, "head_dim", "deepseek_v4")
    q_rank = _required_int(config, "q_lora_rank", "deepseek_v4")
    o_groups = _required_int(config, "o_groups", "deepseek_v4")
    o_rank = _required_int(config, "o_lora_rank", "deepseek_v4")
    window = _required_int(config, "sliding_window", "deepseek_v4")
    index_hd = _required_int(config, "index_head_dim", "deepseek_v4")

    ratios = config.get("compress_ratios")
    if not isinstance(ratios, list) or len(ratios) < layers:
        raise ValueError(
            "deepseek_v4: compress_ratios must be a list of at least "
            "num_hidden_layers entries")

    # Fixed window ring: n_layers * window * head_dim fp32.
    fixed = layers * window * head_dim * 4

    # Compressor + indexer states, scaling with context / ratio.
    state = 0
    for ratio in ratios[:layers]:
        if not isinstance(ratio, bool) and isinstance(ratio, int) and ratio > 0:
            compressed = (context + ratio - 1) // ratio
            state += compressed * head_dim * 4
            if ratio == 4:
                state += compressed * index_hd * 4
        elif ratio != 0:
            raise ValueError("deepseek_v4: compress_ratios entries must be "
                             "non-negative integers")

    # Batched-attention scratch (prefill, batch == context).
    q_width = heads * head_dim
    oa_width = o_groups * o_rank
    workspace = (context * (q_rank + 2 * q_width + head_dim + oa_width) +
                 max(q_rank, head_dim)) * 4

    return PlannerGeometry(state, fixed, workspace, experts)


_GLM_EXPERT = re.compile(
    r"(?:^|\.)model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
)
_KIMI_EXPERT = re.compile(
    r"^(?:language_model\.)?model\.layers\.(\d+)\.block_sparse_moe\."
    r"experts\.(\d+)\."
)
_V4_EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.")
_GLM53_EXPERT = re.compile(
    r"^model\.(?:language_model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\."
)
_INKLING_EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\."
    r"(?:gate_up_proj|down_proj)(?:\.|$)"
)
_QWEN38_EXPERT = re.compile(
    r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.mlp\.experts\."
    r"(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
_QWEN38_EXPERT_SCALE = re.compile(
    r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.mlp\.experts\."
    r"(\d+)\.(gate_proj|up_proj|down_proj)\.weight_scale_inv$"
)
_QWEN38_FUSED_EXPERT = re.compile(
    r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|down_proj)$"
)


def _individual_expert_inventory(pattern):
    def inventory(name, size, _config, _dtype=None):
        match = pattern.search(name)
        if match is None:
            return ()
        return ((int(match.group(1)), int(match.group(2)), size),)
    return inventory


def _inkling_expert_inventory(name, size, config, _dtype=None):
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


def _qwen38_expert_inventory(name, size, config, dtype=None):
    """Index per-expert FP8 and fused BF16 expert tensor layouts.

    Resource planning calls this once per tensor; summing the returned byte
    counts therefore combines gate/up and down tensors for each expert without
    multiplying either tensor by the expert count a second time. FP8 scale
    sidecars are validated here but accounted for by the fixed resident
    inventory because the native loader shares their normalized bank.
    """
    match = _QWEN38_EXPERT.fullmatch(name)
    if match:
        hidden = _required_int(config, "hidden_size", "qwen38")
        intermediate = _required_int(config, "moe_intermediate_size", "qwen38")
        elements = hidden * intermediate
        if size not in (elements, elements * 2, elements * 4):
            raise ValueError(f"qwen38: expert tensor {name!r} has unexpected size {size}")
        inferred = {elements: "F8_E4M3", elements * 2: "BF16",
                    elements * 4: "F32"}[size]
        dtype = inferred if dtype is None else dtype
        expected = {"F8_E4M3": elements, "F8_E4M3FN": elements,
                    "float8_e4m3fn": elements, "BF16": elements * 2,
                    "F16": elements * 2, "F32": elements * 4}.get(dtype)
        if expected != size:
            raise ValueError(f"qwen38: expert tensor {name!r} has unsupported "
                             f"dtype/size {dtype}/{size}")
        # Native FP8/BF16 retain their source representation. F16 is accepted
        # by the engine but expands to F32, so its two-byte file size is not a
        # safe cache budget.
        retained = size if dtype not in ("F16",) else elements * 4
        return ((int(match.group(1)), int(match.group(2)), retained),)
    match = _QWEN38_EXPERT_SCALE.fullmatch(name)
    if match:
        # The native loader normalizes all scales into one fixed per-layer
        # bank. They are deliberately excluded from per-slot accounting;
        # _qwen38_fixed_resident_inventory() accounts for that bank once.
        _qwen38_scale_bytes(name, size, config, dtype)
        return ()
    match = _QWEN38_FUSED_EXPERT.fullmatch(name)
    if not match:
        return ()
    experts = _required_int(config, "num_experts", "qwen38")
    hidden = _required_int(config, "hidden_size", "qwen38")
    intermediate = _required_int(config, "moe_intermediate_size", "qwen38")
    matrices = 2 if match.group(2) == "gate_up_proj" else 1
    elements = experts * matrices * hidden * intermediate
    if size not in (elements * 2, elements * 4):
        raise ValueError(f"qwen38: fused expert tensor {name!r} has unexpected size {size}")
    inferred = "BF16" if size == elements * 2 else "F32"
    dtype = inferred if dtype is None else dtype
    expected = {"BF16": elements * 2, "F16": elements * 2,
                "F32": elements * 4}.get(dtype)
    if expected != size:
        raise ValueError(f"qwen38: fused expert tensor {name!r} has unsupported "
                         f"dtype/size {dtype}/{size}")
    per_expert = (elements * 4 if dtype == "F16" else size) // experts
    layer = int(match.group(1))
    return tuple((layer, expert, per_expert) for expert in range(experts))


def _qwen38_scale_bytes(name, size, config, dtype=None):
    """Validate a Qwen3.8 FP8 scale sidecar and return its retained bytes.

    The source sidecar may be BF16 or F32, while the native engine always
    retains the decoded values as F32. This helper is shared by the expert
    inventory and fixed-resident inventory so validation and sizing cannot
    drift apart.
    """
    match = _QWEN38_EXPERT_SCALE.fullmatch(name)
    if match is None:
        return 0
    hidden = _required_int(config, "hidden_size", "qwen38")
    intermediate = _required_int(config, "moe_intermediate_size", "qwen38")
    projection = match.group(3)
    rows = intermediate if projection != "down_proj" else hidden
    cols = hidden if projection != "down_proj" else intermediate
    scale_count = math.ceil(rows / 128) * math.ceil(cols / 128)
    if dtype is None:
        if size == scale_count * 4:
            dtype = "F32"
        elif size == scale_count * 2:
            dtype = "BF16"
    source_bytes = {"BF16": 2, "F16": 2, "F32": 4}.get(dtype)
    if source_bytes is None or size != scale_count * source_bytes:
        raise ValueError(f"qwen38: expert scale {name!r} has unsupported "
                         f"dtype/size {dtype}/{size}")
    return scale_count * 4


def _qwen38_fixed_resident_inventory(name, size, config, dtype=None):
    """Return bytes for the native loader's fixed normalized scale bank.

    A sidecar contributes its normalized F32 footprint exactly once. For a
    native official checkpoint, summing all sidecars produces the complete
    bank (layers x experts x three projections), independent of cache slots.
    For a fallback layout the same charge is conservative: expanded loaders
    do not retain the bank, but reserving it cannot understate memory.
    """
    return _qwen38_scale_bytes(name, size, config, dtype)


_QWEN38_NATIVE_MATRIX_SUFFIXES = (
    "embed_tokens.weight",
    "input_mix_weight_down.weight",
    "input_mix_weight_up.weight",
    "block_inject_weight.weight",
    "mlp.gate.weight",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.indexer.index_qk_proj.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.out_proj.weight",
    "ple.key_proj.weight",
    "ple.value_proj.weight",
)


def _qwen38_resident_inventory(name, size, _config, dtype=None):
    """Resident bytes for tensors the native text engine actually loads.

    The official checkpoints store non-quantized text tensors as BF16.  Major
    rank-two matrices stay in that two-byte representation; the comparatively
    small norms, gates and convolution vectors still expand through st_read_f32.
    """
    if _QWEN38_EXPERT_SCALE.fullmatch(name):
        # Expert scales belong to the fixed normalized bank, not dense
        # residency. The dedicated inventory hook accounts them once.
        return 0
    if name.startswith("mtp.") or name.startswith("model.visual."):
        return 0
    if ".ple.ple_embedding.ngram_embedding." in name and name.endswith(".weight"):
        return 0  # 51B-parameter PLE table stays pageable in st.h.
    if name.endswith((".ple.ple_embedding.layer_multipliers",
                      ".ple.ple_embedding.ngram_heads_vocab_sizes",
                      ".ple.ple_embedding.ngram_heads_offsets",
                      ".ple.ple_embedding.ngram_embedding.weight_scale")):
        # These few values are read into fixed fields in Model, not retained as
        # separately allocated source tensors.
        return 0
    text_tensor = (name == "lm_head.weight" or
                   name.startswith("model.language_model.") or
                   (name.startswith("model.") and
                    not name.startswith("model.visual.")))
    if not text_tensor:
        return 0
    # Direct registry tests predate dtype-bearing planner entries and use BF16
    # sizes; real scans always supply the safetensors dtype.
    dtype = "BF16" if dtype is None else dtype
    element_bytes = {"BF16": 2, "F16": 2, "F32": 4}.get(dtype)
    if not element_bytes:
        raise ValueError(f"qwen38: resident tensor {name!r} has unsupported dtype {dtype}")
    elements = size // element_bytes
    if elements * element_bytes != size:
        raise ValueError(f"qwen38: resident tensor {name!r} has invalid byte size {size}")
    if (name == "lm_head.weight" or
            name.endswith(_QWEN38_NATIVE_MATRIX_SUFFIXES)) and dtype == "BF16":
        return size
    return elements * 4


COMMON_CAP = FamilyCapabilities(False, False, False, True)

FAMILIES = (
    FamilyDescriptor(
        id="glm53",
        # "glm5_next" e' il tipo della radice, che e' il wrapper vision; un
        # export di solo testo dichiara "glm5_next_text" al primo livello.
        model_types=("glm5_next", "glm5_next_text"),
        display_name="GLM-5.3-Flash",
        display_scale="321B",
        engine_artifact="glm53",
        engine_aliases=(),
        engine_group="glm53",
        internal_arch="glm53",
        build_target="glm53",
        process_names=("glm53",),
        default_model_id="glm-5.3-flash-colibri",
        cli_adapter="glm53",
        gateway_adapter="glm53",
        planner_id="glm53_hybrid",
        dense_load_ratio=_glm53_dense_in_ram,
        planner_geometry=_glm53_geometry,
        planner_unsupported_reason="",
        expert_inventory=_individual_expert_inventory(_GLM53_EXPERT),
        config_section="text_config",
        # implicit_cap 0, non 8: questo motore dimensiona la cache degli
        # esperti dalla RAM disponibile, quindi "nessuna scelta esplicita"
        # deve arrivargli come 0 e non come otto slot per layer.
        # L'interattivo era 1024, uguale al default non interattivo: l'unica
        # famiglia che ragiona a cui non era stato alzato. Su un modello che
        # riflette prima di rispondere quel tetto non e' una rete di sicurezza,
        # e' una ghigliottina che cade DENTRO al blocco di pensiero e chiude il
        # turno senza risposta (#1278). 16384 e' il valore che hanno gia' tutte
        # le famiglie con lo stesso contesto massimo di 1048576.
        limits=FamilyLimits(8192, 1048576, 1024, 16384, 1, 0, "GLM53_MAXT"),
        # tools=True: this family DOES render and parse tool calls. It used to
        # share COMMON_CAP, which says otherwise -- the flag is descriptive
        # (it only feeds the capability dict) so nothing broke, but a client
        # reading it programmatically was told the opposite of the truth.
        capabilities=FamilyCapabilities(True, False, False, True),
        has_gateway_adapter=True,
        has_cli_adapter=True,
        # Dal chat_template.jinja del checkpoint: nessun a capo, e <think>
        # subito dopo <|assistant|>, che e' quello che apre il ragionamento.
        tune_prompt_template="[gMASK]<sop><|user|>{prompt}<|assistant|><think>",
    ),
    FamilyDescriptor(
        id="glm",
        model_types=("glm_moe_dsa", "glm5_moe", "glm"),
        display_name="GLM-5.2",
        display_scale="744B",
        engine_artifact="colibri",
        engine_aliases=("glm",),
        engine_group="colibri-core",
        internal_arch="glm",
        build_target="colibri",
        process_names=("colibri", "glm"),
        default_model_id="glm-5.2-colibri",
        cli_adapter="glm",
        gateway_adapter="glm",
        planner_id="glm_mla",
        planner_geometry=_glm_geometry,
        planner_unsupported_reason="",
        expert_inventory=_individual_expert_inventory(_GLM_EXPERT),
        config_section="root",
        limits=FamilyLimits(4096, 1048576, 1024, 16384, 16, 0, "CTX"),
        capabilities=FamilyCapabilities(True, True, False, True),
        has_gateway_adapter=True,
        has_cli_adapter=True,
        tune_prompt_template="[gMASK]<sop><|user|>{prompt}<|assistant|><think></think>",
    ),
    FamilyDescriptor(
        id="inkling",
        model_types=("inkling_mm_model", "inkling"),
        display_name="Inkling",
        display_scale="975B",
        engine_artifact="inkling",
        engine_aliases=(),
        engine_group="inkling",
        internal_arch="inkling",
        build_target="inkling",
        process_names=("inkling",),
        default_model_id="inkling-colibri",
        cli_adapter="inkling",
        gateway_adapter="inkling",
        planner_id="inkling_hybrid",
        planner_geometry=_inkling_geometry,
        planner_unsupported_reason="",
        expert_inventory=_inkling_expert_inventory,
        config_section="text_config",
        # L'interattivo era 1024, uguale al default non interattivo: l'unica
        # famiglia che ragiona a cui non era stato alzato. Su un modello che
        # riflette prima di rispondere quel tetto non e' una rete di sicurezza,
        # e' una ghigliottina che cade DENTRO al blocco di pensiero e chiude il
        # turno senza risposta (#1278). 16384 e' il valore che hanno gia' tutte
        # le famiglie con lo stesso contesto massimo di 1048576.
        limits=FamilyLimits(8192, 1048576, 1024, 16384, 1, 8, "CTX_MAX"),
        capabilities=FamilyCapabilities(False, False, True, True),
        has_gateway_adapter=True,
        tune_prompt_template="<|user|>{prompt}<|assistant|>",
    ),
    FamilyDescriptor(
        id="kimi",
        # "kimi_linear" is Moonshot's model_type for the text model itself
        # (the real checkpoint's text_config carries it; the root "kimi_k3"
        # belongs to the vision wrapper). A text-only export -- the tiny
        # fixture included -- is therefore kimi_linear at top level.
        model_types=("kimi_k3", "kimi_linear"),
        display_name="Kimi K3",
        display_scale="2.8T",
        engine_artifact="kimi_k3",
        engine_aliases=(),
        engine_group="kimi_k3",
        internal_arch="kimi_k3",
        build_target="kimi_k3",
        process_names=("kimi_k3",),
        default_model_id="kimi-k3-colibri",
        cli_adapter="kimi",
        gateway_adapter="kimi",
        planner_id="kimi_hybrid",
        planner_geometry=_kimi_geometry,
        planner_unsupported_reason="",
        expert_inventory=_individual_expert_inventory(_KIMI_EXPERT),
        config_section="text_config",
        # L'interattivo era 1024, uguale al default non interattivo: l'unica
        # famiglia che ragiona a cui non era stato alzato. Su un modello che
        # riflette prima di rispondere quel tetto non e' una rete di sicurezza,
        # e' una ghigliottina che cade DENTRO al blocco di pensiero e chiude il
        # turno senza risposta (#1278). 16384 e' il valore che hanno gia' tutte
        # le famiglie con lo stesso contesto massimo di 1048576.
        limits=FamilyLimits(8192, 1048576, 1024, 16384, 1, 8, "K3_MAXT"),
        # tools=True: this family DOES render and parse tool calls. It used to
        # share COMMON_CAP, which says otherwise -- the flag is descriptive
        # (it only feeds the capability dict) so nothing broke, but a client
        # reading it programmatically was told the opposite of the truth.
        capabilities=FamilyCapabilities(True, False, False, True),
        has_gateway_adapter=True,
        tune_prompt_template="K3CHAT1\nM user {prompt_len}\n{prompt}G 0\n\n",
    ),
    FamilyDescriptor(
        id="olmoe",
        model_types=("olmoe",),
        display_name="OLMoE",
        display_scale="7B",
        engine_artifact="olmoe",
        engine_aliases=(),
        engine_group="olmoe",
        internal_arch="olmoe",
        build_target="olmoe",
        process_names=("olmoe",),
        default_model_id="olmoe-colibri",
        cli_adapter="olmoe",
        gateway_adapter="olmoe",
        planner_id="olmoe_gqa",
        planner_geometry=_olmoe_geometry,
        planner_unsupported_reason="",
        expert_inventory=_individual_expert_inventory(_GLM_EXPERT),
        config_section="root",
        limits=FamilyLimits(4096, 4096, 1024, 1024, 1, 8, "CTX"),
        capabilities=FamilyCapabilities(False, False, False, False),
        has_gateway_adapter=True,
        has_cli_adapter=True,
        tune_prompt_template="<|user|>\n{prompt}\n<|assistant|>\n",
    ),
    FamilyDescriptor(
        id="qwen36",
        model_types=("qwen3_5_moe", "qwen3_5_moe_text"),
        display_name="Qwen3.6-35B-A3B",
        display_scale="35B",
        engine_artifact="qwen36",
        engine_aliases=(),
        engine_group="qwen36",
        internal_arch="qwen36",
        build_target="qwen36",
        process_names=("qwen36",),
        default_model_id="qwen3.6-colibri",
        cli_adapter="qwen36",
        gateway_adapter="qwen36",
        planner_id="qwen36_hybrid",
        planner_geometry=_qwen36_geometry,
        planner_unsupported_reason="",
        expert_inventory=_individual_expert_inventory(_GLM_EXPERT),
        config_section="text_config",
        limits=FamilyLimits(8192, 262144, 1024, 8192, 1, 8, "Q36_MAXT"),
        capabilities=FamilyCapabilities(False, False, False, True),
        has_gateway_adapter=True,
        # coli run stays unwired on purpose: cmd_run dispatches per arch after
        # this gate, and without a qwen36 branch the engine would inherit GLM's
        # prompt template -- a wrong template does not fail loudly, it degrades
        # the answer. False gives the user "use coli chat or coli serve", which
        # is true and actionable; chat/serve/web all work through the gateway.
        has_cli_adapter=False,
        tune_prompt_template=(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ),
    FamilyDescriptor(
        id="qwen38",
        model_types=("qwen4_exp", "qwen4_exp_text"),
        display_name="Qwen3.8-Flash-Next",
        display_scale="176B",
        engine_artifact="qwen38",
        engine_aliases=(),
        engine_group="qwen38",
        internal_arch="qwen38",
        build_target="qwen38",
        process_names=("qwen38",),
        default_model_id="qwen3.8-flash-next-colibri",
        cli_adapter="qwen38",
        gateway_adapter="qwen38",
        planner_id="qwen38_hybrid",
        planner_geometry=_qwen38_geometry,
        planner_unsupported_reason="",
        expert_inventory=_qwen38_expert_inventory,
        resident_inventory=_qwen38_resident_inventory,
        fixed_resident_inventory=_qwen38_fixed_resident_inventory,
        config_section="text_config",
        limits=FamilyLimits(8192, 262144, 1024, 8192, 1, 1, "Q38_MAXT"),
        capabilities=FamilyCapabilities(True, False, False, True),
        has_gateway_adapter=True,
        # Like Qwen3.6, direct `coli run` is intentionally not exposed until
        # an engine-specific CLI prompt path exists; chat/serve use the gateway.
        has_cli_adapter=False,
        tune_prompt_template=(
            "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully "
            "through the task, validate key assumptions, consider plausible alternatives, "
            "and prioritize correctness, consistency, and clarity in the final answer."
            "<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"),
        supports_accelerator=False,
    ),
    FamilyDescriptor(
        id="deepseek_v4",
        model_types=("deepseek_v4",),
        display_name="DeepSeek V4 Flash",
        display_scale="284B",
        engine_artifact="deepseek_v4",
        engine_aliases=(),
        engine_group="deepseek_v4",
        internal_arch="deepseek_v4",
        build_target="deepseek-v4",
        process_names=("deepseek_v4",),
        default_model_id="deepseek-v4-colibri",
        cli_adapter="deepseek_v4",
        gateway_adapter="deepseek_v4",
        planner_id="deepseek_v4",
        planner_geometry=_dsv4_geometry,
        planner_unsupported_reason="",
        expert_inventory=_individual_expert_inventory(_V4_EXPERT),
        config_section="root",
        limits=FamilyLimits(4096, 1048576, 1024, 16384, 1, 8, "CTX"),
        capabilities=FamilyCapabilities(True, False, False, True),
        has_gateway_adapter=True,
        has_cli_adapter=True,
    ),
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
                 not callable(family.expert_inventory) or
                (family.resident_inventory is not None and
                 not callable(family.resident_inventory)) or
                (family.fixed_resident_inventory is not None and
                 not callable(family.fixed_resident_inventory)) or
                not isinstance(family.supports_accelerator, bool) or
                not isinstance(family.has_gateway_adapter, bool) or
                not isinstance(family.has_cli_adapter, bool) or
                not isinstance(family.tune_prompt_template, str) or
                "{prompt}" not in family.tune_prompt_template):
            raise RegistryError(f"incomplete family descriptor: {family.id}")
        try:
            family.tune_prompt_template.format(prompt="test", prompt_len=4)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            raise RegistryError(f"invalid tune prompt template: {family.id}") from error
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


def tuning_replay_prompt(family, prompt):
    if not isinstance(prompt, str):
        raise ValueError("tuning prompt must be a string")
    return family.tune_prompt_template.format(prompt=prompt, prompt_len=len(prompt))


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


def expert_contributions(resolved, name, size, dtype=None):
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("tensor size must be a non-negative integer")
    if dtype is not None and not isinstance(dtype, str):
        raise ValueError("tensor dtype must be a string")
    contributions = resolved.descriptor.expert_inventory(
        name, size, resolved.family_config, dtype)
    for layer, expert, byte_count in contributions:
        if layer < 0 or expert < 0 or byte_count < 0:
            raise RegistryError(f"invalid expert inventory for {resolved.descriptor.id}")
    return contributions


def resident_contribution(resolved, name, size, dtype=None):
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("tensor size must be a non-negative integer")
    inventory = resolved.descriptor.resident_inventory
    if dtype is not None and not isinstance(dtype, str):
        raise ValueError("tensor dtype must be a string")
    contribution = size if inventory is None else inventory(
        name, size, resolved.family_config, dtype)
    if isinstance(contribution, bool) or not isinstance(contribution, int) or contribution < 0:
        raise RegistryError(f"invalid resident inventory for {resolved.descriptor.id}")
    return contribution


def fixed_resident_contribution(resolved, name, size, dtype=None):
    """Return resident bytes that do not belong to dense or cache storage.

    This pool is model-owned and independent of cache capacity. Families
    without such an allocation return zero.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("tensor size must be a non-negative integer")
    if dtype is not None and not isinstance(dtype, str):
        raise ValueError("tensor dtype must be a string")
    inventory = resolved.descriptor.fixed_resident_inventory
    contribution = 0 if inventory is None else inventory(
        name, size, resolved.family_config, dtype)
    if isinstance(contribution, bool) or not isinstance(contribution, int) or contribution < 0:
        raise RegistryError(
            f"invalid fixed resident inventory for {resolved.descriptor.id}")
    return contribution


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
        "supports_accelerator": family.supports_accelerator,
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
