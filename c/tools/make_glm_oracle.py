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
EN: ingests. Default: bf16 (original oracle unchanged).

--fmt6 / --fmt4 (ticket #3): quantize ONLY the ROUTED experts to fmt=6 (E8/IQ3, the only
rotation-bearing format) or fmt=4 (grouped int4, the no-rotation control); shared/dense/attn
stay f32. This is the fixture that lets the token-exact parity gate (#7) exercise the E8
activation pre-rotation (E8_XE / e8_rot_rows) — the default f32 oracle never touches it.

The E8 super-block is 256 weights (98 bytes), so the routed-expert contraction dims must be
multiples of 256; these two flags therefore generate the model with hidden_size=256 and
moe_intermediate_size=256 (the f32 default keeps 128/32). Weights are packed from the
ORIGINAL weights and the ref is computed from the DEQUANTIZED weights (round-trip through the
same quantizer), so the engine's decode reproduces the reference token-exactly.

Regeneration (run from c/):
    python3 tools/make_glm_oracle.py --fmt6     # -> glm_tiny_fmt6/ (model.safetensors, config.json, ref_glm.json)
    python3 tools/make_glm_oracle.py --fmt4     # -> glm_tiny_fmt4/
    # verify: SNAP=./glm_tiny_fmt6 REF=./glm_tiny_fmt6/ref_glm.json TF=1 COLI_TEMP=0 ./colibri 64 16 16"""
import json, sys, argparse
from pathlib import Path

# --- Version gate (must run BEFORE the heavy `from transformers import ...`,
# which triggers transformers' lazy-loading and can in turn reset the in-memory
# __version__ attribute; importlib.metadata reads the installed package version
# directly and is immune to that). ---
#
# GLM-5.2's MLA attention uses interleaved (DeepSeek-style) RoPE, which is what
# the C engine implements. transformers < 5.11.0 applied split-half (Llama-style)
# RoPE in GlmMoeDsa* instead; an oracle built on those versions drifts and the
# engine then scores 25/32 instead of the documented 32/32 (issue #281). Weights
# come out identical across versions — only the forward pass differs — so there
# is no safe "partial" run: a too-old transformers silently produces an invalid
# ref_glm.json. Hard-fail rather than warn. EN: same.
import transformers
from importlib.metadata import version as _pkg_version, PackageNotFoundError

_MIN_TRANSFORMERS = (5, 11)
def _tf_version_tuple():
    try:
        v = _pkg_version("transformers")          # authoritative: installed dist metadata
    except PackageNotFoundError:
        v = getattr(transformers, "__version__", "0")   # fallback (editable/src installs)
    out = []
    for part in v.split(".")[:2]:                 # major.minor only
        out.append(int("".join(c for c in part if c.isdigit()) or "0"))
    while len(out) < 2:
        out.append(0)
    return tuple(out[:2])

_tf_ver = _tf_version_tuple()
if _tf_ver < _MIN_TRANSFORMERS:
    sys.exit(
        f"\nERROR: make_glm_oracle.py requires transformers >= "
        f"{'.'.join(map(str, _MIN_TRANSFORMERS))}.0 (found {_tf_ver[0]}.{_tf_ver[1]}). "
        f"GLM-5.2 MLA uses interleaved RoPE; older versions apply split-half RoPE "
        f"and silently produce an oracle the engine scores 25/32 against (issue #281). "
        f"Upgrade: pip install -U 'transformers>=5.11'\n"
    )

import torch
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))   # importa glm_fp8_emit se lanciato da c/
from glm_fp8_emit import (fp8_block_quantize, fp8_block_dequantize, keep_f32,
                          save_fp8_safetensors, unfuse_experts)

# Codec per le fixture quantizzate (fmt=6/fmt=4). Importati a livello di modulo: sono
# numpy-only (niente torch/transformers) e safetensors e' gia' richiesto da OGNI ramo di
# save (default bf16 incluso), quindi non toccano il path default dependency-free.
# EN: codecs for the quantized fixtures. Imported at module scope: numpy-only (no
# torch/transformers) and safetensors is already required by every save path, so they
# leave the dependency-free default path untouched.
import numpy as np
import iq3_pack
from convert_fp8_to_int4 import quant_int4_grouped
from convert_fmt4_to_fmt2 import dequant_fmt4
from safetensors.torch import save_file

FMT6 = 6                    # container E8/IQ3 con rotazione (98B per super-blocco di 256 pesi)
FMT4 = 4                    # int4 grouped, braccio di controllo senza rotazione
FMT6_SCALE_TAG = 6.0        # il singolo float del companion .qs che il motore legge come tag fmt=6
GROUPED_INT4_BITS = 4       # bit depth del braccio int4 grouped
GROUPED_INT4_GROUP = 64     # group size (input dim per scala) del braccio int4 grouped
# EN: FMT6 = rotation-bearing E8/IQ3 container; FMT4 = grouped int4 control; FMT6_SCALE_TAG
# EN: = single .qs float the engine reads to detect fmt=6; grouped int4 uses 4 bits / gs=64.

def quantize_routed(w, fmt):
    """Quantizza una matrice di pesi routed [O, I] nel container fmt=6 (E8/IQ3) o fmt=4
    (grouped int4). Ritorna (packed_or_q, scale_or_none): per fmt=6 i byte E8 impacchettati
    (scale None, il codec non ha scale esterne); per fmt=4 i nibble U8 appiattiti + le scale
    f32. UNICO encoder condiviso da round-trip e save, cosi' i due path non possono divergere.
    EN: quantize routed weight matrix w [O,I] to fmt=6 (E8/IQ3) or fmt=4 (grouped int4).
    Returns (packed_or_q, scale_or_none): fmt=6 -> packed E8 bytes (scale None); fmt=4 -> flat
    U8 nibbles + f32 scales. Single shared encoder for round-trip and save, so they can't drift."""
    if fmt == FMT6:
        return iq3_pack.encode(iq3_pack.rotate_rows(w)), None
    q, s = quant_int4_grouped(w, GROUPED_INT4_BITS, GROUPED_INT4_GROUP)
    return q, s

ap = argparse.ArgumentParser()
_quant = ap.add_mutually_exclusive_group()
_quant.add_argument("--fp8", action="store_true",
                help="salva in FP8 e4m3 + 128x128 block scale_inv (layout GLM-5.2-FP8) e "
                     "calcola ref_glm.json sul modello dopo il round-trip FP8. "
                     "EN: write FP8 e4m3 + block scale_inv, ref computed on FP8-rounded model")
_quant.add_argument("--fmt6", action="store_true",
                help="quantizza SOLO gli expert routed in fmt=6 (E8/IQ3, #452) e lascia il "
                     "resto (shared/dense/attn) f32; oracolo per esercitare la rotazione E8. "
                     "EN: quantize ONLY routed experts to fmt=6 (E8/IQ3); everything else f32")
_quant.add_argument("--fmt4", action="store_true",
                help="come --fmt6 ma fmt=4 (int4 grouped gs=64, nessuna rotazione): il "
                     "braccio di controllo senza rotazione. "
                     "EN: like --fmt6 but fmt=4 (grouped int4, no rotation) — the control arm")
args = ap.parse_args()

fmt = FMT6 if args.fmt6 else FMT4 if args.fmt4 else 0

torch.manual_seed(1234)

# E8/IQ3 (fmt=6) e il suo controllo int4-grouped (fmt=4) richiedono che le
# dimensioni di contrazione degli expert routed siano multiple di 256: il codec E8
# impacchetta 256 pesi per super-blocco (98 byte) e quant_e8 rifiuta qualsiasi altra
# I (iq3_pack.encode: `K % 256 == 0`). La config tiny di default (hidden=128,
# moe_inter=32) NON e' quantizzabile in E8, quindi le fixture quantizzate usano
# hidden=256 / moe_inter=256. Il default f32 (nessuna flag) resta INVARIATO a 128/32.
# EN: fmt=6's E8 super-block (256 weights/98B) forces routed-expert contraction dims
# to be multiples of 256; the quantized fixtures use hidden=256/moe_inter=256 while
# the f32 default is unchanged at 128/32.
if fmt:
    _HIDDEN, _MOE_INTER = 256, 256
else:
    _HIDDEN, _MOE_INTER = 128, 32

cfg = GlmMoeDsaConfig(
    vocab_size=256,
    hidden_size=_HIDDEN,
    intermediate_size=64,          # MLP densa (primi 3 layer)
    moe_intermediate_size=_MOE_INTER,   # expert
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

# --fmt6/--fmt4: round-trip degli expert routed FUSI (gate_up_proj [E,2M,I] e
# down_proj [E,I,M]) attraverso la quantizzazione PRIMA di calcolare il riferimento,
# cosi' ref_glm.json riflette ESATTAMENTE i pesi che il motore decodifichera' dal
# container quantizzato (stesso pattern del ramo --fp8). La quantizzazione opera per
# riga sull'ultima dim (= la dim di contrazione), quindi quantizzare il fuso 3-D
# equivale a quantizzare gli expert 2-D non fusi: le righe sono identiche.
# EN: round-trip the FUSED routed experts through the quantizer before computing the
# ref (same pattern as --fp8), so the ref matches exactly what the engine decodes.
# Quantization is per-row on the last (contraction) dim, so fused and unfused encodings
# are identical.
if fmt:
    # Cattura i pesi ORIGINALI (pre-round-trip) PRIMA di toccare il modello: servono
    # per il SAVE — i pesi impacchettati devono venire dai pesi originali, non da quelli
    # gia' dequantizzati (encode(decode(x)) != encode(x): il codec E8 non e' idempotente
    # byte-per-byte). state_dict() ritorna viste che aliasano i parametri, quindi clone
    # esplicito. EN: clone the original weights before the round-trip; the SAVE path must
    # pack the ORIGINAL weights (encode∘decode is not idempotent), and state_dict()
    # aliases the live parameters.
    sd_orig = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".mlp.experts." not in n or not n.endswith(("gate_up_proj", "down_proj")):
                continue
            shp = tuple(p.shape); I = shp[-1]
            w = p.detach().float().numpy().reshape(-1, I)
            packed, scale = quantize_routed(w, fmt)
            if fmt == FMT6:
                w_eff = iq3_pack.unrotate_rows(iq3_pack.decode(packed, I))
            else:
                w_eff = dequant_fmt4(packed.reshape(-1, (I + 1) // 2), scale, w.shape[0], I)
            p.copy_(torch.from_numpy(w_eff.reshape(shp)))

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

if fmt:
    # Salva SOLO gli expert routed quantizzati (fmt=6 o fmt=4 + .qs); shared/dense/attn/
    # norme restano f32. I pesi impacchettati provengono dai pesi ORIGINALI (sd_orig,
    # non fusi), non da quelli gia' dequantizzati, cosi' il motore decodifica ESATTAMENTE
    # i pesi che il round-trip sopra ha usato per il riferimento. EN: quantize ONLY the
    # routed experts (fmt=6 or fmt=4 + .qs); everything else stays f32. Packed weights
    # come from the ORIGINAL (unfused) weights, not the dequantized ones, so the engine
    # decodes exactly the weights the round-trip used for the reference.
    outdir = "glm_tiny_fmt6" if fmt == FMT6 else "glm_tiny_fmt4"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    sd = unfuse_experts(sd_orig)
    out = {}
    for name, t in sd.items():
        if ".mlp.experts." in name and name.endswith(
                (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")):
            w = t.detach().float().numpy()
            packed, scale = quantize_routed(w, fmt)
            if fmt == FMT6:
                out[name] = torch.from_numpy(np.ascontiguousarray(packed.reshape(-1)))
                out[name + ".qs"] = torch.tensor([FMT6_SCALE_TAG], dtype=torch.float32)
            else:
                out[name] = torch.from_numpy(packed)
                out[name + ".qs"] = torch.from_numpy(scale)
        else:
            out[name] = t.detach().contiguous()
    save_file(out, f"{outdir}/model.safetensors")
    json.dump(cfg.to_dict(), open(f"{outdir}/config.json", "w"))
    json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred},
              open(f"{outdir}/ref_glm.json", "w"))
    print(f"saved: {outdir}/ (routed experts fmt={fmt}, shared/dense/attn f32) + "
          f"{outdir}/ref_glm.json")
else:
    Path("glm_tiny").mkdir(parents=True, exist_ok=True)   # safetensors/json won't create the dir themselves
    if args.fp8:
        n_fp8, n_tot = save_fp8_safetensors(sd, "glm_tiny/model.safetensors")
        print(f"\nsaved FP8: {n_fp8} e4m3 tensors (+{n_tot - n_fp8} scale_inv sidecars / f32) "
              f"-> glm_tiny/model.safetensors")
    else:
        save_file({k: v.contiguous() for k, v in sd.items()}, "glm_tiny/model.safetensors")
    json.dump(cfg.to_dict(), open("glm_tiny/config.json", "w"))
    json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred}, open("ref_glm.json", "w"))
    print("saved: glm_tiny/ (weights + config) and ref_glm.json"
          + (" [fp8]" if args.fp8 else ""))
