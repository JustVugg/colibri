#!/usr/bin/env python3
"""Reference logits of the Qwen3.8 MTP head on a tiny fixture, in torch.

The head is built from our own reading of its tensors, the same reading
qwen38_core.h documents next to q38_mtp_input (the MTP head):

* the model's four hyper-connection streams at position p -- the
  ``hidden_size * hc_count`` vector the final mixer reads, captured here as the
  input of ``model.hyper_connection_mixer`` -- and the embedding of the token
  at p+1 become four streams through ``pre_fc_norm_*`` and ``fc_*``, with
  ``pre_fc_norm_hidden`` grouped per stream (wiring b, the engine's default)
  or over the whole vector (wiring a; ``fuse`` below);
* those streams run through one decoder layer with the tensors under
  ``mtp.layers.0`` -- the transformers ``Qwen4ExpTextDecoderLayer``, the oracle
  the engine's own layers are checked against, built as an attention layer
  without PLE -- with RoPE on ``mtp.rope_theta`` and causal attention over the
  pairs;
* ``mtp.hyper_connection_mixer`` (a ``Qwen4ExpTextGatedResidual`` without
  inject) and the shared ``lm_head`` give the logits.

Row i of the result pairs the streams at position i with ``ids[i + 1]``: the
logits that predict ``ids[i + 2]``. Everything runs in float32 on the bytes on
disk: BF16 widens exactly, and the FP8 routed experts (the model's and the
head's) are dequantized the way qwen38's expanded path does it, e4m3 value
times the block's ``weight_scale_inv`` in f32, so the reference sees the
engine's weights and not the BF16 rounding the fixture generator kept in
memory.

    python tools/qwen38_mtp_ref.py --model ./qwen38_tiny_mtp --wiring b
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

FP8_BLOCK = 128


def _e4m3_table():
    """quant.h's E4M3 table (float8_e4m3fn: bias 7, subnormals, 0x7F/0xFF NaN)."""
    table = np.empty(256, np.float32)
    for byte in range(256):
        exponent, mantissa = (byte >> 3) & 0xF, byte & 7
        if exponent == 0xF and mantissa == 7:
            value = math.nan
        elif exponent == 0:
            value = mantissa / 8.0 * 2.0 ** -6
        else:
            value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        table[byte] = -value if byte & 0x80 else value
    return torch.from_numpy(table)


E4M3 = _e4m3_table()


def load_tensors(model_dir: Path):
    """Every tensor of the snapshot as float32; FP8 matrices dequantized with
    their block scales, the scale tensors themselves dropped."""
    from safetensors.torch import load_file
    raw = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        raw.update(load_file(str(path)))
    out = {}
    for name, tensor in raw.items():
        if name.endswith("_scale_inv"):
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            scale = raw[name + "_scale_inv"].float()
            rows, cols = tensor.shape
            expand = scale.repeat_interleave(FP8_BLOCK, 0)[:rows].repeat_interleave(FP8_BLOCK, 1)[:, :cols]
            out[name] = E4M3[tensor.view(torch.uint8).long()] * expand
        elif tensor.is_floating_point():
            out[name] = tensor.float()
        else:
            out[name] = tensor          # the PLE hash tables are int64
    return out


def _fuse_experts(weights, prefix, experts):
    """Per-expert gate/up/down -> transformers' fused gate_up_proj / down_proj."""
    gate_up = torch.stack([torch.cat([weights[f"{prefix}{e}.gate_proj.weight"],
                                      weights[f"{prefix}{e}.up_proj.weight"]], 0) for e in range(experts)])
    down = torch.stack([weights[f"{prefix}{e}.down_proj.weight"] for e in range(experts)])
    return gate_up, down


def load_model(model_dir: Path):
    """The fixture's model in float32 with the exact weights on disk, its
    config, and the f32 tensors."""
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config = Qwen4ExpTextConfig.from_pretrained(str(model_dir))
    if hasattr(config, "quantization_config"):
        del config.quantization_config     # the experts are dequantized here, not by transformers
    config._attn_implementation = "eager"
    model = Qwen4ExpForCausalLM(config).float().eval()
    weights = load_tensors(model_dir)
    state = model.state_dict()
    with torch.no_grad():
        for name, param in state.items():
            if name.endswith(".mlp.experts.gate_up_proj") or name.endswith(".mlp.experts.down_proj"):
                base = name[: name.rindex(".") + 1]
                gate_up, down = _fuse_experts(weights, base, config.num_experts)
                param.copy_(gate_up if name.endswith("gate_up_proj") else down)
            elif name in weights:
                param.copy_(weights[name])
            elif name.endswith(("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")):
                continue        # buffers the model computes itself
            else:
                shards = sorted(k for k in weights if k.startswith(name.replace(".weight", ".shard_")))
                if not shards:
                    raise KeyError(f"{name}: not in the snapshot")
                param.copy_(torch.cat([weights[k] for k in shards], 0))
    return model, config, cfg, weights


def model_streams(model, ids):
    """The four streams the final mixer reads, [len(ids), hc_count * hidden]."""
    captured = {}

    def hook(module, args):
        captured["streams"] = args[0].detach()

    handle = model.model.hyper_connection_mixer.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        handle.remove()
    return captured["streams"][0]


def rms0(x, weight, eps):
    """Qwen4-Exp's zero-centered RMSNorm over the last dimension: x / rms * (1 + w)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight)


def fuse(streams, embedded, weights, config, wiring):
    """The head's input streams [S, hc_count*H] from the model's streams at p
    [S, hc_count*H] and the embeddings of the tokens at p+1 [S, H]: fc_hidden
    on every normalized stream, fc_embedding of the normalized embedding added
    to all four."""
    C, H, eps = config.hc_count, config.hidden_size, config.rms_norm_eps
    w_hid = weights["mtp.pre_fc_norm_hidden.weight"]
    fe = rms0(embedded, weights["mtp.pre_fc_norm_embedding.weight"], eps) @ weights["mtp.fc_embedding.weight"].T
    fc_hidden = weights["mtp.fc_hidden.weight"]
    S = streams.shape[0]
    if wiring == "b":       # a norm per stream
        normed = rms0(streams.view(S, C, H), w_hid.view(C, H), eps)
    elif wiring == "a":     # one norm over the whole vector
        normed = rms0(streams, w_hid, eps).view(S, C, H)
    else:
        raise ValueError(f"wiring {wiring!r}: b or a")
    return (normed @ fc_hidden.T + fe[:, None, :]).reshape(S, C * H)


def mtp_layer(config, cfg, weights):
    """The head's decoder layer: transformers' layer, as one of the model's
    attention layers without PLE, holding the mtp.layers.0 tensors."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
    attn = [i for i, kind in enumerate(config.layer_types) if kind != "linear_attention"
            and (i + 1) not in (config.ple_layer_ids or [])]
    layer = Qwen4ExpTextDecoderLayer(config, attn[-1]).float().eval()
    state = layer.state_dict()
    with torch.no_grad():
        for name, param in state.items():
            if name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.down_proj"):
                gate_up, down = _fuse_experts(weights, "mtp.layers.0.mlp.experts.", config.num_experts)
                param.copy_(gate_up if name.endswith("gate_up_proj") else down)
            else:
                param.copy_(weights["mtp.layers.0." + name])
    return layer


def mtp_logits(model_dir, ids, wiring):
    """[len(ids) - 1, vocab]: row i is the head's logits for the pair
    (streams at i, ids[i + 1])."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextGatedResidual, Qwen4ExpTextRotaryEmbedding)
    model_dir = Path(model_dir)
    model, config, cfg, weights = load_model(model_dir)
    streams = model_streams(model, ids)[:-1]
    nxt = torch.tensor(ids[1:])
    embedded = weights["model.embed_tokens.weight"][nxt]
    hyper = fuse(streams, embedded, weights, config, wiring)
    S = hyper.shape[0]
    head_config = copy.deepcopy(config)
    theta = (cfg.get("mtp") or {}).get("rope_theta", config.rope_parameters["rope_theta"])
    head_config.rope_parameters = dict(config.rope_parameters, rope_theta=theta)
    rotary = Qwen4ExpTextRotaryEmbedding(config=head_config)
    positions = torch.arange(S).view(1, 1, -1).expand(3, 1, -1)
    position_embeddings = rotary(hyper[None], positions)
    mask = torch.full((1, 1, S, S), torch.finfo(torch.float32).min).triu(1)
    layer = mtp_layer(config, cfg, weights)
    mixer = Qwen4ExpTextGatedResidual(config, use_combine=False).float().eval()
    with torch.no_grad():
        for name, param in mixer.state_dict().items():
            param.copy_(weights["mtp.hyper_connection_mixer." + name])
        out = layer(hyper[None], position_embeddings=position_embeddings, attention_mask=mask,
                    conv_mask=None, past_key_values=None, ple_input_ids=None)
        logits = mixer(out)[0] @ weights["lm_head.weight"].T
    return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--wiring", default="b", choices="ba")
    parser.add_argument("--ids", help="comma-separated ids (default: ref.json full_ids)")
    args = parser.parse_args()
    ids = ([int(x) for x in args.ids.split(",")] if args.ids else
           json.loads((args.model / "ref.json").read_text(encoding="utf-8"))["full_ids"])
    logits = mtp_logits(args.model, ids, args.wiring)
    for i, row in enumerate(logits):
        print(f"row {i}: token {ids[i + 1]} -> draft {int(row.argmax())} (max {float(row.max()):.6f})")


if __name__ == "__main__":
    main()
