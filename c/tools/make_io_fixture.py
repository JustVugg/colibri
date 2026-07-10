#!/usr/bin/env python3
"""Synthetic disk-streamed GLM-MoE fixture, directly in the engine container.

Unlike make_glm_bench_model.py (which needs torch + transformers), this writes
the safetensors by hand with only numpy: random weights, real glm_moe_dsa
tensor layout, experts pre-quantized int8 with the three matrices CONTIGUOUS
in-file (so expert_load takes the single coalesced pread / O_DIRECT path).
~9 GB of routed experts vs ~200 MB resident: the engine must stream from disk,
which is exactly what I/O A/B tests (PILOT / PREFETCH / DIRECT / cache-cap)
need. Not a language model — tokens are noise; only the data flow is real.

usage: python3 tools/make_io_fixture.py <outdir>
then:  SNAP=<outdir> REF=<outdir>/ref_glm.json DRAFT=0 AUTOPIN=0 RAM_GB=4 ./glm 4 8
"""
import json, os, struct, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "fixture"
D, VOCAB, L, FIRST_DENSE = 1024, 4096, 12, 1
E, TOPK, MOE_I, DENSE_I, NSH = 64, 8, 4096, 2048, 1
QLORA, KVLORA, NOPE, ROPE, VH, H = 256, 128, 64, 32, 64, 8
rng = np.random.default_rng(1234)

os.makedirs(OUT, exist_ok=True)
cfg = dict(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
    n_routed_experts=E, num_experts_per_tok=TOPK, moe_intermediate_size=MOE_I,
    intermediate_size=DENSE_I, first_k_dense_replace=FIRST_DENSE,
    q_lora_rank=QLORA, kv_lora_rank=KVLORA, qk_nope_head_dim=NOPE,
    qk_rope_head_dim=ROPE, v_head_dim=VH, n_shared_experts=NSH,
    vocab_size=VOCAB, n_group=1, topk_group=1, norm_topk_prob=True,
    rms_norm_eps=1e-5, routed_scaling_factor=2.5,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    eos_token_id=VOCAB - 1, index_topk=0, index_n_heads=0, index_head_dim=0)
json.dump(cfg, open(f"{OUT}/config.json", "w"))

def wmat(o, i):   # resident f32 weight, sane scale
    return (rng.standard_normal((o, i), dtype=np.float32) / np.sqrt(i)).astype(np.float32)

class Shard:
    """streaming safetensors writer: declare tensors, then write data in order"""
    def __init__(s, path): s.path, s.meta, s.gen, s.off = path, {}, [], 0
    def add(s, name, arr_fn, shape, dtype):
        nb = int(np.prod(shape)) * (4 if dtype == "F32" else 1)
        s.meta[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [s.off, s.off + nb]}
        s.gen.append(arr_fn); s.off += nb
    def write(s):
        hdr = json.dumps(s.meta).encode()
        with open(s.path, "wb") as f:
            f.write(struct.pack("<Q", len(hdr))); f.write(hdr)
            for g in s.gen: f.write(g().tobytes())

def f32(sh, name, shape, fn=None):
    fn = fn or (lambda: wmat(*shape) if len(shape) == 2 else np.ones(shape, np.float32))
    sh.add(name, fn, shape, "F32")

# ---- shard 0: dense/resident ----
sh = Shard(f"{OUT}/out-00000.safetensors")
f32(sh, "model.embed_tokens.weight", (VOCAB, D))
f32(sh, "lm_head.weight", (VOCAB, D))
f32(sh, "model.norm.weight", (D,))
for i in range(L):
    P = f"model.layers.{i}."
    f32(sh, P + "input_layernorm.weight", (D,))
    f32(sh, P + "post_attention_layernorm.weight", (D,))
    f32(sh, P + "self_attn.q_a_proj.weight", (QLORA, D))
    f32(sh, P + "self_attn.q_a_layernorm.weight", (QLORA,))
    f32(sh, P + "self_attn.q_b_proj.weight", (H * (NOPE + ROPE), QLORA))
    f32(sh, P + "self_attn.kv_a_proj_with_mqa.weight", (KVLORA + ROPE, D))
    f32(sh, P + "self_attn.kv_a_layernorm.weight", (KVLORA,))
    f32(sh, P + "self_attn.kv_b_proj.weight", (H * (NOPE + VH), KVLORA))
    f32(sh, P + "self_attn.o_proj.weight", (D, H * VH))
    if i < FIRST_DENSE:
        f32(sh, P + "mlp.gate_proj.weight", (DENSE_I, D))
        f32(sh, P + "mlp.up_proj.weight", (DENSE_I, D))
        f32(sh, P + "mlp.down_proj.weight", (D, DENSE_I))
    else:
        f32(sh, P + "mlp.gate.weight", (E, D))
        f32(sh, P + "mlp.gate.e_score_correction_bias", (E,),
            lambda: np.linspace(-0.05, 0.05, E, dtype=np.float32))
        f32(sh, P + "mlp.shared_experts.gate_proj.weight", (MOE_I * NSH, D))
        f32(sh, P + "mlp.shared_experts.up_proj.weight", (MOE_I * NSH, D))
        f32(sh, P + "mlp.shared_experts.down_proj.weight", (D, MOE_I * NSH))
sh.write(); print("dense shard done")

# ---- expert shards: pre-quantized int8 container (name U8 + name.qs F32) ----
def qdata(nb): return lambda: np.frombuffer(os.urandom(nb), np.uint8)
def qscale(o): return lambda: np.full(o, 4e-4, np.float32)
si = 1
for i in range(FIRST_DENSE, L):
    sh = Shard(f"{OUT}/out-{si:05d}.safetensors"); si += 1
    for e in range(E):
        P = f"model.layers.{i}.mlp.experts.{e}."
        mats = [("gate_proj", (MOE_I, D)), ("up_proj", (MOE_I, D)),
                ("down_proj", (D, MOE_I))]
        for nm, (o, ii) in mats:   # weights contiguous (single coalesced pread)
            sh.add(P + nm + ".weight", qdata(o * ii), (o, ii), "U8")
        for nm, (o, ii) in mats:
            sh.add(P + nm + ".weight.qs", qscale(o), (o,), "F32")
    sh.write(); print(f"layer {i} experts done", flush=True)

# ref ids: arbitrary (fixture is not a language model)
ids = rng.integers(1, VOCAB - 2, 80).tolist()
json.dump({"prompt_ids": ids[:16], "full_ids": ids, "tf_pred": ids},
          open(f"{OUT}/ref_glm.json", "w"))
print("fixture complete")
