"""Costruisce un GLM-5.2 (glm_moe_dsa) MINUSCOLO a pesi random come ORACOLO.
Architettura vera (MLA + DSA indexer + router sigmoid/noaux_tc + shared expert),
dimensioni minuscole. Salva pesi+config in c/glm_tiny/ e un riferimento greedy in
c/ref_glm.json. seq corta (<= index_topk) cosi' il DSA seleziona tutte le key e
l'attenzione coincide con la MLA densa: il motore C puo' validare senza implementare
l'indexer sparso.

--fp8: salva i pesi come FP8 e4m3 + scale a blocchi 128x128 (layout del checkpoint reale
GLM-5.2-FP8) invece di bf16, cosi' convert_fp8_to_int4.py puo' esercitare il path FP8->int4
su un modello minuscolo. PRIMA di calcolare ref_glm.json fa il round-trip dei pesi per FP8
(quant->dequant, copy_ nel modello): cosi' il riferimento riflette ESATTAMENTE il modello
FP8 che il converter legge, non il modello bf16 a precisione piena. Default: bf16 (oracolo
originale invariato).
EN: --fp8 writes FP8 e4m3 + 128x128 block scale_inv (real GLM-5.2-FP8 layout) instead of bf16,
EN: so convert_fp8_to_int4.py can run its FP8->int4 path on a tiny model. ref_glm.json is
EN: computed AFTER the FP8 round-trip, so the reference matches exactly what the converter
EN: ingests. Default: bf16 (original oracle unchanged)."""
import json, sys, argparse
from pathlib import Path
import torch
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))   # importa glm_fp8_emit se lanciato da c/
from glm_fp8_emit import (fp8_block_quantize, fp8_block_dequantize, keep_f32,
                          save_fp8_safetensors, unfuse_experts)

ap = argparse.ArgumentParser()
ap.add_argument("--fp8", action="store_true",
                help="salva in FP8 e4m3 + 128x128 block scale_inv (layout GLM-5.2-FP8) e "
                     "calcola ref_glm.json sul modello dopo il round-trip FP8. "
                     "EN: write FP8 e4m3 + block scale_inv, ref computed on FP8-rounded model")
args = ap.parse_args()

torch.manual_seed(1234)

cfg = GlmMoeDsaConfig(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=64,          # MLP densa (primi 3 layer)
    moe_intermediate_size=32,      # expert
    num_hidden_layers=5,           # 3 densi + 2 sparse
    first_k_dense_replace=3,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_shared_experts=1,
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=24,
    qk_rope_head_dim=8,            # pari -> interleave ok; head_dim diventa 8
    v_head_dim=32,
    index_topk=4096,              # >> seq_len -> DSA seleziona tutto (no-op)
    index_head_dim=16,
    index_n_heads=2,
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    routed_scaling_factor=2.5,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    tie_word_embeddings=False,
    rms_norm_eps=1e-5,
    attention_bias=False,
    max_position_embeddings=4096,
)
cfg._attn_implementation = "eager"

model = GlmMoeDsaForCausalLM(cfg).eval()
# rende i pesi non banali (default init e' molto piccolo): scala router/bias per topk vario
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.05)
    # bias di correzione del router: valori distinti cosi' la selezione e' sensata
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.e_score_correction_bias.copy_(
                torch.linspace(-0.1, 0.1, cfg.n_routed_experts))

# --fp8: round-trip dei pesi quantizzabili per FP8 PRIMA di calcolare il riferimento,
# cosi' ref_glm.json riflette esattamente il modello FP8 che il converter leggera'.
# Norme/router/bias (keep_f32) restano a precisione piena. EN: --fp8: round-trip quantizable
# weights through FP8 before computing the reference, so ref_glm.json matches the FP8 model.
if args.fp8:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if keep_f32(n, p) or p.dim() != 2:
                continue
            q, s = fp8_block_quantize(p)
            p.copy_(fp8_block_dequantize(q, s))

print("=== state_dict tensors (names used by the C loader) ===")
for n, p in model.state_dict().items():
    print(f"  {n:60s} {tuple(p.shape)}")

prompt = [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]          # token id arbitrari, seq corta
ids = torch.tensor([prompt])
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=20, do_sample=False, use_cache=True)
full = out[0].tolist()
print("\nprompt:", prompt)
print("full  :", full)

# teacher-forcing: un singolo forward su tutta la sequenza -> argmax per posizione.
# Per il greedy vale tf_pred[i] == full[i+1] per i >= len(prompt)-1; serve a validare
# il PREFILL del motore C separandolo dal decode.
with torch.no_grad():
    lg = model(torch.tensor([full]), use_cache=False).logits[0]   # [seq, vocab]
tf_pred = lg.argmax(-1).tolist()
print("tf_pred:", tf_pred)

# Unfuse experts AFTER reference generation (model needs fused weights for
# forward/generate) but BEFORE saving — the real checkpoint and the converter
# + C engine all expect per-expert 2-D gate_proj/up_proj/down_proj tensors.
sd = model.state_dict()
unfuse_experts(sd)

if args.fp8:
    n_fp8, n_tot = save_fp8_safetensors(sd, "glm_tiny/model.safetensors")
    print(f"\nsaved FP8: {n_fp8} e4m3 tensors (+{n_tot - n_fp8} scale_inv sidecars / f32) "
          f"-> glm_tiny/model.safetensors")
else:
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in sd.items()}, "glm_tiny/model.safetensors")
json.dump(cfg.to_dict(), open("glm_tiny/config.json", "w"))
json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred}, open("ref_glm.json", "w"))
print("saved: glm_tiny/ (weights + config) and ref_glm.json"
      + (" [fp8]" if args.fp8 else ""))
