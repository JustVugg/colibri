#!/usr/bin/env python3
"""Float32 reference for Qwen3.6's checkpoint MTP head on converted tiny models.

Qwen3-Next/Qwen3.5 MTP combines normalized embed(token_{p+1}) and h_p
(embedding first), projects 2H -> H, runs one gated full-attention/MoE
layer, then mtp.norm and the shared lm_head. The decoder and zero-centered
RMSNorm here are Qwen's transformers implementation, not engine equations.

h_p is the backbone's output, AFTER its final norm (model(...).last_hidden_state),
as vLLM's Qwen3-Next MTP takes the target model's hidden states
(https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_next_mtp/);
mtp.pre_fc_norm_hidden normalizes it once more with the head's own weights. A deeper
speculative row takes the head's output after mtp.norm the same way. On the released
35B checkpoint this pairing drafts better than the residual before the norm (first
drafts accepted 85-89 % against 84-89 % on three prompts). The engine normalizes the
rows q36_mtp_feed receives with model.norm; --pre-norm here gives the other wiring,
which the harness checks the engine is NOT.

Row p pairs h_p with ids[p+1] and predicts ids[p+2]. RoPE positions 0..S-1
are equivalent to 1..S for text: all query/key relative offsets are unchanged.
There is no dummy cache row at position zero. All tensors are read from the
converted container: f16 widened exactly, merged expert integers multiplied
by their on-disk scales. This tests quantized fixtures without attributing
quantization error to the head's arithmetic.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_weights(path):
    path = Path(path)
    meta = json.loads((path / "qwen36_meta.json").read_text())
    raw = {}
    for shard in sorted(path.glob("*.safetensors")):
        raw.update(load_file(str(shard)))
    weights = {k: v.float() for k, v in raw.items() if not k.endswith((".merged_weight", ".qs"))}
    H, I = meta["hidden"], meta["moe_inter"]
    for name, tensor in raw.items():
        if not name.endswith(".merged_weight"):
            continue
        base = name.removesuffix("merged_weight")
        q = tensor.reshape(-1)
        n = H * I
        mixed = q.numel() == 2*n
        if q.numel() in (3*n//2, 2*n):
            packed = q[:n] if mixed else q
            pairs = torch.stack((packed & 15, (packed >> 4) & 15), -1).to(torch.int8).flatten()
            pairs = torch.where(pairs >= 8, pairs - 16, pairs)
            q = torch.cat((pairs, q[n:].view(torch.int8))) if mixed else pairs
        scales = raw[base + "qs"].flatten()
        off = at = 0
        for proj, rows, cols in (("gate", I, H), ("up", I, H), ("down", H, I)):
            gs = meta.get("expert_down_gs", 0) if mixed and proj == "down" else meta.get("expert_gs", 0)
            ng = (cols + gs - 1)//gs if gs else 1
            scale = scales[at:at + rows*ng].reshape(rows, ng)
            scale = scale.repeat_interleave(gs or cols, 1)[:, :cols]
            weights[base + proj + "_proj.weight"] = q[off:off + rows*cols].float().reshape(rows, cols) * scale
            off += rows*cols
            at += rows*ng
    return weights


def copy_weights(module, weights, prefix=""):
    with torch.no_grad():
        for name, param in module.state_dict().items():
            key = prefix + name
            if key.endswith(".experts.gate_up_proj"):
                base = key.removesuffix("gate_up_proj")
                value = torch.stack([torch.cat([weights[f"{base}{e}.gate_proj.weight"],
                                                weights[f"{base}{e}.up_proj.weight"]])
                                     for e in range(param.shape[0])])
            elif key.endswith(".experts.down_proj"):
                base = key.removesuffix("down_proj")
                value = torch.stack([weights[f"{base}{e}.down_proj.weight"] for e in range(param.shape[0])])
            else:
                value = weights[key]
            param.copy_(value)


def mtp_logits(path, ids, pre_norm=False):
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer, Qwen3_5MoeRMSNorm)
    config = Qwen3_5MoeTextConfig.from_pretrained(str(path))
    config._attn_implementation = "eager"
    weights = load_weights(path)
    model = Qwen3_5MoeForCausalLM(config).float().eval()
    copy_weights(model, weights)
    captured = []
    handle = model.model.norm.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    with torch.no_grad():
        out = model.model(torch.tensor([ids]), use_cache=False).last_hidden_state
    handle.remove()
    hidden = (captured[0] if pre_norm else out)[:, :-1]
    layer_idx = config.layer_types.index("full_attention")
    layer = Qwen3_5MoeDecoderLayer(config, layer_idx).float().eval()
    copy_weights(layer, weights, "mtp.layers.0.")
    norms = {}
    for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
        norms[name] = Qwen3_5MoeRMSNorm(config.hidden_size, config.rms_norm_eps).float()
        copy_weights(norms[name], weights, f"mtp.{name}.")
    with torch.no_grad():
        emb = weights["model.embed_tokens.weight"][torch.tensor([ids[1:]])]
        x = torch.cat((norms["pre_fc_norm_embedding"](emb), norms["pre_fc_norm_hidden"](hidden)), -1)
        x = x @ weights["mtp.fc.weight"].T
        S = x.shape[1]
        positions = torch.arange(S).view(1, 1, -1).expand(3, 1, -1)
        rope = model.model.rotary_emb(x, positions)
        mask = torch.full((1, 1, S, S), torch.finfo(torch.float32).min).triu(1)
        x = layer(x, position_embeddings=rope, attention_mask=mask, past_key_values=None)
        return (norms["norm"](x)[0] @ weights["lm_head.weight"].T).detach()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--ids", required=True, help="comma-separated token IDs")
    args = ap.parse_args()
    for row, logits in enumerate(mtp_logits(args.model, [int(t) for t in args.ids.split(',')])):
        print(row, int(logits.argmax()), float(logits.max()))


if __name__ == "__main__":
    main()
