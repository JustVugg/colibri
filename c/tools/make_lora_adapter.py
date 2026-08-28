"""Create a colibri-lora-v1 adapter directory for a Colibri snapshot.

Writes adapter.safetensors (F32 lora_A [rank,I] / lora_B [O,rank]) + adapter.json
with the base fingerprint computed from config.json — the same FNV-1a over the
same fields, in the same order, as cfg_fingerprint() in glm.c.

  python3 tools/make_lora_adapter.py --model ./glm_tiny --out /tmp/ad \
      --targets o_proj --layers 0,1,2,3,4 --rank 4 --alpha 8 --init zero|random

`--init zero` (B=0) must leave engine output bit-identical (identity test);
`--init random` must change logits (smoke test).
"""
import argparse, json, struct, os, sys

import numpy as np

FP_SEED = 1469598103934665603
FP_PRIME = 1099511628211
MASK = (1 << 64) - 1

def fp_mix(h, v):
    v &= MASK
    for i in range(8):
        h ^= (v >> (8 * i)) & 0xFF
        h = (h * FP_PRIME) & MASK
    return h

def fingerprint(cfg):
    h = FP_SEED
    for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
              "n_routed_experts", "q_lora_rank", "kv_lora_rank",
              "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "vocab_size"):
        h = fp_mix(h, int(cfg[k]))
    return h

# target -> (suffix, O, I) given the config
def target_dims(cfg, tgt):
    D = cfg["hidden_size"]; H = cfg["num_attention_heads"]
    qk = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    dims = {
        "o_proj":  ("self_attn.o_proj", D, H * cfg["v_head_dim"]),
        "q_a_proj": ("self_attn.q_a_proj", cfg["q_lora_rank"], D),
        "q_b_proj": ("self_attn.q_b_proj", H * qk, cfg["q_lora_rank"]),
        "kv_a_proj": ("self_attn.kv_a_proj_with_mqa", cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"], D),
        "kv_b_proj": ("self_attn.kv_b_proj", H * (cfg["qk_nope_head_dim"] + cfg["v_head_dim"]), cfg["kv_lora_rank"]),
    }
    return dims[tgt]

def save_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [off, off + arr.nbytes]}
        blobs.append(arr.tobytes()); off += arr.nbytes
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj)
        for b in blobs: f.write(b)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="snapshot dir with config.json")
ap.add_argument("--out", required=True, help="adapter output dir")
ap.add_argument("--targets", default="o_proj", help="comma list: o_proj,q_a_proj,...")
ap.add_argument("--layers", default="", help="comma list of layer indices (default: all)")
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=float, default=16.0)
ap.add_argument("--init", choices=["zero", "random"], default="zero",
                help="zero: A~N(0,.02),B=0 (standard LoRA init, exact no-op); random: both nonzero")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

cfg = json.load(open(os.path.join(args.model, "config.json")))
nl = cfg["num_hidden_layers"]
layers = [int(x) for x in args.layers.split(",") if x != ""] or list(range(nl))
bad = [l for l in layers if not 0 <= l < nl]
if bad:
    sys.exit(f"layers {bad} outside [0,{nl})")

rng = np.random.default_rng(args.seed)
tensors, targets = {}, []
for li in layers:
    for tgt in args.targets.split(","):
        suffix, O, I = target_dims(cfg, tgt)
        name = f"model.layers.{li}.{suffix}"
        A = rng.normal(0, 0.02, (args.rank, I))
        B = np.zeros((O, args.rank)) if args.init == "zero" else rng.normal(0, 0.2, (O, args.rank))
        tensors[name + ".lora_A.weight"] = A
        tensors[name + ".lora_B.weight"] = B
        targets.append(name)

os.makedirs(args.out, exist_ok=True)
save_safetensors(os.path.join(args.out, "adapter.safetensors"), tensors)
meta = {"format": "colibri-lora-v1",
        "base_model": cfg.get("model_type", "?"),
        "base_fingerprint": f"{fingerprint(cfg):016x}",
        "rank": args.rank, "alpha": args.alpha, "dtype": "F32",
        "tensor_name_map_version": 1, "targets": targets}
json.dump(meta, open(os.path.join(args.out, "adapter.json"), "w"), indent=1)
print(f"adapter: {args.out} | {len(targets)} target(s) rank={args.rank} "
      f"alpha={args.alpha} init={args.init} fp={meta['base_fingerprint']}")
