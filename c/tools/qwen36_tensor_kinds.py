"""Every tensor name a Qwen3.6/3.8 MoE checkpoint can carry, classified.

The converter used to select what it understood (``<prefix>layers.<i>.`` and
four globals) and let everything else fall through unmentioned. That skipped
the ``mtp.*`` head and the ``visual.*`` tower by accident, and it would have
copied an unfamiliar tensor inside a layer as f16 without a word. Both are
the failure GLM-5.3's converter was built to refuse: a container that loads
and is quietly missing, or quietly carrying, a tensor (#1045).

This module is the contract. ``classify`` names every tensor or raises; the
converter stops on the first name it cannot place. Torch-free on purpose so
the test can run wherever the registry tests run.

Derived from the safetensors indexes of Qwen/Qwen3.6-35B-A3B (1045 tensors,
prefix ``model.language_model.``, plus ``visual.*``) and
Qwen/Qwen3.8-2.4T-A95B (1609 tensors, prefix ``model.``), and from the
transformers-saved tiny fixture (per-expert layout). The test pins all three.
"""
import re

# Suffixes after "<prefix>layers.<i>." -- exact, no fragments, so a new
# tensor kind fails here instead of matching a substring of an old one.
LAYER_KINDS = frozenset((
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    # Gated DeltaNet (linear_attention layers)
    "linear_attn.A_log",
    "linear_attn.conv1d.weight",
    "linear_attn.dt_bias",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.norm.weight",
    "linear_attn.out_proj.weight",
    # Gated Attention (full_attention layers)
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    # MoE block: router, shared expert, its gate
    "mlp.gate.weight",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "mlp.shared_expert_gate.weight",
    # routed experts, fused layout (real checkpoints): one tensor per layer
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
))

# routed experts, per-expert layout (transformers save_pretrained on the
# text model, i.e. the tiny fixture): mlp.experts.<e>.<proj>.weight
_EXPERT_SEPARATE = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

GLOBAL_KINDS = frozenset(("embed_tokens.weight", "norm.weight", "lm_head.weight"))

# Whole subtrees the text engine does not implement. Skipped deliberately,
# counted, and reported -- never silently.
SKIP_PREFIXES = (
    ("mtp.", "mtp", "multi-token-prediction head (mtp_num_hidden_layers); "
                    "the engine predicts one token per step and never reads it"),
    ("model.visual.", "visual", "vision tower; the engine is text-only"),
    ("visual.", "visual", "vision tower; the engine is text-only"),
)

_LAYER = re.compile(r"^layers\.(\d+)\.(.+)$")


class UnknownTensor(KeyError):
    """A name the contract does not place. The converter must stop on it."""


def resolve_prefix(keys):
    """'model.language_model.' for VL checkpoints, else 'model.'."""
    for k in keys:
        if k.startswith("model.language_model.layers."):
            return "model.language_model."
    return "model."


def classify(name, prefix):
    """Return one of
         ("global", kind)            embed / final norm / lm_head
         ("layer", index, kind)      a tensor of transformer layer <index>
         ("skip", group)             mtp / visual, deliberately not converted
       or raise UnknownTensor.

    ``kind`` for a layer is the suffix after ``layers.<i>.``; for the
    per-expert layout it is normalised to ``mlp.experts.<e>.<proj>.weight``
    so the converter can key on it.
    """
    for skip_prefix, group, _why in SKIP_PREFIXES:
        if name.startswith(skip_prefix):
            return ("skip", group)
    if name == "lm_head.weight" or name == prefix + "lm_head.weight":
        return ("global", "lm_head.weight")
    if name.startswith(prefix):
        rest = name[len(prefix):]
        if rest in GLOBAL_KINDS:
            return ("global", rest)
        m = _LAYER.match(rest)
        if m:
            kind = m.group(2)
            if kind in LAYER_KINDS or _EXPERT_SEPARATE.match(kind):
                return ("layer", int(m.group(1)), kind)
    raise UnknownTensor(name)


def skip_reason(group):
    for _prefix, g, why in SKIP_PREFIXES:
        if g == group:
            return why
    return ""
