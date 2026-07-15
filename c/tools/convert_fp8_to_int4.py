"""
Convertitore GLM-5.2-FP8 -> nostro container int4 (STADIO B).

Strategia DISK-SAFE (richiesta dell'utente): scarica UNO shard (~5 GB), lo converte in
int4, lo CANCELLA, passa al prossimo. Il disco non si riempie mai: picco = 1 shard + l'output
int4 che cresce fino a ~372 GB. Controllo di spazio che si ferma se manca margine.

Cosa fa per ogni tensore:
  - pesi FP8 (e4m3) con `*.weight_scale_inv`  -> dequant a blocchi 128x128 -> f32
  - pesi BF16 (norme/embed/lm_head/...)        -> f32
  poi:
  - attn/mlp/shared/expert/embed/lm_head -> QUANTIZZATO int4 (o int8) con la STESSA matematica
    del motore C (np.rint = lrintf, stesse soglie, stesso packing dei nibble) -> token identici
  - norme / router (mlp.gate.weight) / bias / e_score_correction_bias -> tenuti F32
  - indexer DSA / layer MTP (78) / shared_head / eh_proj / *norm dell'indexer -> SALTATI

Output: una dir di safetensors leggibile dal motore C (per ogni peso quantizzato: `nome` U8 =
dati impacchettati, `nome.qs` F32 = scale per riga).

USO:
  # test locale (oracolo tiny, niente download): converte una dir gia' presente
  python3 tools/convert_fp8_to_int4.py --indir glm_tiny --outdir glm_tiny_i4 --ebits 4 --io-bits 4
  # selftest del dequant fp8 (richiede torch)
  python3 tools/convert_fp8_to_int4.py --selftest
  # reale: scarica+converte+cancella shard per shard
  python3 tools/convert_fp8_to_int4.py --repo zai-org/GLM-5.2-FP8 --outdir /home/vincenzo/glm52_i4
"""
import os, sys, glob, json, shutil, argparse
import numpy as np

# ---------- quantizzazione: identica al C (glm.c) ----------
def quant_int8(w, bits):                       # w: [O,I] f32 -> (qbytes U8 [O*I], scale f32 [O])
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -qmax - 1, qmax).astype(np.int8)
    return q.reshape(-1).view(np.uint8).copy(), s[:, 0].astype(np.float32)

def quant_int4(w, bits):                        # -> (qbytes U8 [O*ceil(I/2)], scale f32 [O])
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -8, qmax).astype(np.int32)  # nibble [-8,7]
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    return out.reshape(-1), s[:, 0].astype(np.float32)

def quant_int4_grouped(w, bits, gs=128):
    """Group-scaled int4: one scale per group of `gs` elements along the input dim.
    Drastically reduces quantization error vs per-row scaling — matches the FP8
    source's 128x128 block-scale granularity. Output layout:
      qbytes: same packed nibble format as quant_int4
      scales: f32 [O * ngroups] where ngroups = ceil(I/gs), laid out as
              s[o * ngroups + g] = scale for row o, group g.
    The engine detects this format (fmt=4) by checking the .qs array size."""
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1
    ngroups = (I + gs - 1) // gs
    # pad I to a multiple of gs for clean reshape, then trim
    Ipad = ngroups * gs
    wpad = np.zeros((O, Ipad), np.float32)
    wpad[:, :I] = w
    wr = wpad.reshape(O, ngroups, gs)                     # [O, ngroups, gs]
    amax = np.abs(wr).max(axis=2, keepdims=True)          # [O, ngroups, 1]
    s = np.maximum(amax / qmax, 1e-8)                     # [O, ngroups, 1]
    q = np.clip(np.rint(wr / s), -8, qmax).astype(np.int32)  # [O, ngroups, gs]
    q = q.reshape(O, Ipad)[:, :I]                         # trim padding -> [O, I]
    # pack nibbles (identical to quant_int4)
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    # scales: flatten [O, ngroups] -> [O * ngroups]
    s_flat = s[:, :, 0].astype(np.float32).reshape(-1)
    return out.reshape(-1), s_flat

def quant_int2(w, bits):                        # -> (qbytes U8 [O*ceil(I/4)], scale f32 [O]); 4/byte
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1                 # bits=2 -> qmax=1, valori [-2,1]
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -2, qmax).astype(np.int32)
    rb = (I + 3) // 4
    out = np.zeros((O, rb), np.uint8)
    for k in range(4):                           # impacchetta 4 valori per byte (identico a pack_int2 in C)
        vk = q[:, k::4]
        out[:, :vk.shape[1]] |= ((vk + 2).astype(np.uint8) << (k * 2))
    return out.reshape(-1), s[:, 0].astype(np.float32)

# ---------- NVFP4 (modelopt) : LUT e2m1 ----------
# FP4 e2m1 = 1 sign + 2 exp + 1 mantissa. 16 codici, magnitudini {0,.5,1,1.5,2,3,4,6}.
# Bit 3 = segno. Ordine impacchettato (compressed_tensors/vLLM): nibble BASSO = elemento
# pari, nibble ALTO = elemento dispari. LUT verificata 1:1 con ml_dtypes.float4_e2m1fn.
# EN: FP4 e2m1 = 1 sign + 2 exp + 1 mantissa. 16 codes, magnitudes {0,.5,1,1.5,2,3,4,6}.
# EN: bit 3 = sign. Packed order (compressed_tensors/vLLM): LOW nibble = even element,
# EN: HIGH nibble = odd element. LUT verified 1:1 against ml_dtypes.float4_e2m1fn.
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# ---------- classificazione dei tensori ----------
def layer_idx(name):
    p = name.split(".")
    if len(p) > 2 and p[0] == "model" and p[1] == "layers":
        try: return int(p[2])
        except ValueError: return -1
    return -1

def classify(name, n_layers, keep_mtp=False, keep_idx=False):
    if name.endswith("_scale_inv"): return "consumed"   # FP8 base: gestito col suo peso
    # NVFP4 (modelopt): i sidecar delle scale sono consumati insieme al loro U8 .weight.
    # EN: NVFP4 (modelopt): scale sidecars are consumed together with their U8 .weight.
    if name.endswith((".weight_scale", ".weight_scale_2", ".input_scale")): return "consumed"
    li = layer_idx(name)
    if keep_idx:
        # modalita' --indexer: SOLO i pesi del DSA lightning indexer dei layer principali
        if li < 0 or li >= n_layers or "indexer" not in name: return "skip"
        if name.endswith("norm.weight"): return "f32"
        return "q"                                       # int8 consigliato (--ebits 8): pesi di scoring
    if keep_mtp:
        if li != n_layers: return "skip"                 # solo il layer MTP
        if "indexer" in name: return "skip"              # il DSA indexer resta un no-op
    else:
        if li >= n_layers: return "skip"                 # layer MTP (78)
        if any(k in name for k in ["indexer", "indexers_proj", "eh_proj",
                                    "enorm", "hnorm", "shared_head"]): return "skip"
    if name.endswith("e_score_correction_bias"): return "f32"
    if name.endswith("mlp.gate.weight"): return "f32"    # router (NON gate_proj)
    if name.endswith("norm.weight") or name == "model.norm.weight": return "f32"
    if name in ("model.embed_tokens.weight", "lm_head.weight"): return "io"
    if ".mlp.experts." in name and name.endswith(".weight"): return "x"  # expert ROUTED (streaming)
    # Split resident weights by type for mixed-precision control:
    #   "sh" = shared expert (fires on every token, highest sensitivity)
    #   "o"  = o_proj attention (reconstructs output, biggest attn tensor)
    #   "kvb" = kv_b_proj (reconstructs KV cache on every decode step)
    #   "attn" = other attention projections (q_a, q_b, kv_a)
    #   "dmlp" = dense MLP (first 3 layers)
    if "shared_experts" in name: return "sh"
    if name.endswith("o_proj.weight"): return "o"
    if name.endswith("kv_b_proj.weight"): return "kvb"
    if any(name.endswith(k) for k in ("q_a_proj.weight", "q_b_proj.weight",
                                       "kv_a_proj_with_mqa.weight")): return "attn"
    if any(name.endswith(k) for k in ("mlp.gate_proj.weight", "mlp.up_proj.weight",
                                       "mlp.down_proj.weight")): return "dmlp"
    if name.endswith(".weight"): return "q"              # fallback: other resident weights
    return "f32"

# ---------- dequant NVFP4 (modelopt) di UN tensore expert -> f32 [O,I] ----------
def dequant_nvfp4(f, name):
    """NVFP4 di NVIDIA modelopt (quant_algo=NVFP4, quant_method=modelopt).
      - `name`               U8   [O, I/2]  : due nibble e2m1 per byte lungo la dim di
                                              contrazione (input); pari=nibble basso, dispari=alto.
      - `name.weight_scale`  F8_E4M3 [O, I/16] : scala per-BLOCCO di 16 elementi (group_size=16),
                                              lungo la dim di input. Decodifica f8e4m3 -> f32.
      - `name.weight_scale_2` F32 []        : scala GLOBALE per-tensore, ~amax/(6*448) (piccola).
    Dequant (convenzione modelopt = MOLTIPLICA, NON dividere):
        W[o,i] = e2m1_lut[nibble] * f8_block_scale[o, i//16] * weight_scale_2
    FOOTGUN: llm-compressor/compressed-tensors memorizza il RECIPROCO (global grande) e DIVIDE;
    modelopt memorizza il valore piccolo e MOLTIPLICA. Questo checkpoint e' modelopt -> moltiplica.
    EN: NVIDIA modelopt NVFP4. LOW nibble=even elem, HIGH=odd. weight_scale = per-16-block FP8
    EN: (group_size 16) along the input dim; weight_scale_2 = per-tensor global FP32 (~amax/2688,
    EN: small). Dequant MULTIPLIES both scales. FOOTGUN: llm-compressor stores the reciprocal
    EN: (large global) and DIVIDES; modelopt stores the small value and MULTIPLIES."""
    import torch
    GS = 16                                                       # NVFP4: block scale ogni 16 elementi
    packed = f.get_tensor(name)                                    # uint8 [O, I/2]
    bscale = f.get_tensor(name + "_scale").to(torch.float32)        # [O, ceil(I/16)] da f8e4m3
    gscale = f.get_tensor(name + "_scale_2").to(torch.float32)      # scalare per-tensore
    O, Ih = packed.shape; I = Ih * 2
    # Convenzione: modelopt memorizza il global PICCOLO e MOLTIPLICA. Se e' >=1 e'
    # quasi certamente il reciproco di compressed-tensors (che DIVIDE) -> ci fermiamo
    # invece di corrompere silenziosamente ogni tensore. EN: guard modelopt-vs-CT.
    assert float(gscale) < 1.0, (
        f"{name}: weight_scale_2={float(gscale):.4g} >= 1 sembra il reciproco "
        "(compressed-tensors, che DIVIDE); questo path assume modelopt (MOLTIPLICA)")
    # Il layout deve essere lo scale per-blocco piatto di modelopt: una colonna ogni
    # 16 elementi di input (niente swizzle cutlass/TensorRT). Verifichiamo, non deduciamo:
    # dedurre gs = I // ncol misallinea in silenzio su layout paddati/swizzati.
    nb = (I + GS - 1) // GS
    assert bscale.shape[1] == nb, (
        f"{name}: weight_scale ha {bscale.shape[1]} colonne, attese {nb} = ceil({I}/{GS}); "
        "layout scale inatteso (swizzled/paddato?), rifiuto per non corrompere")
    lut = torch.tensor(_E2M1, dtype=torch.float32)
    nib = torch.empty((O, I), dtype=torch.long)
    nib[:, 0::2] = (packed & 0x0F).to(torch.long)                  # elemento pari = nibble basso
    nib[:, 1::2] = ((packed >> 4) & 0x0F).to(torch.long)           # elemento dispari = nibble alto
    w4 = lut[nib]                                                  # [O, I] valori e2m1
    sc = bscale.repeat_interleave(GS, dim=1)[:, :I]               # blocco parziale di coda: slice a I
    return (w4 * sc * gscale).numpy()

# ---------- dequant di un tensore (nvfp4 / fp8+scale a blocchi / bf16 / f32) ----------
def dequant(f, name, keys):
    import torch
    sl = f.get_slice(name); dt = sl.get_dtype()
    # NVFP4 (modelopt): pesi expert U8 con sidecar `.weight_scale`. In questo checkpoint gli
    # UNICI tensori U8 sono gli expert NVFP4, ma richiediamo comunque il sidecar (keys e'
    # obbligatorio: senza, un qualunque U8 verrebbe decodificato come NVFP4).
    # EN: NVFP4 expert weights are U8 with a `.weight_scale` sidecar; require the sidecar.
    if dt in ("U8", "uint8") and (name + "_scale") in keys:
        return dequant_nvfp4(f, name)
    if dt in ("F8_E4M3", "float8_e4m3fn"):
        w = f.get_tensor(name).to(torch.float32)
        sc = f.get_tensor(name + "_scale_inv").to(torch.float32)   # [ceil(O/128),ceil(I/128)]
        O, I = w.shape
        sc = sc.repeat_interleave(128, 0).repeat_interleave(128, 1)[:O, :I]
        return (w * sc).numpy()
    return f.get_tensor(name).to(torch.float32).numpy()

def convert_shard(path, out_dict, n_layers, ebits, io_bits, xbits,
                  keep_mtp=False, keep_idx=False, group_size=0, bits_map=None):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for name in f.keys():
            kind = classify(name, n_layers, keep_mtp, keep_idx)
            if kind in ("skip", "consumed"): continue
            w = dequant(f, name, keys)
            if kind == "f32":
                out_dict[name] = w.astype(np.float32)
            else:
                # Resolve bits for this tensor type: use bits_map override if provided,
                # otherwise fall back to the classic ebits/xbits/io_bits scheme.
                if bits_map and kind in bits_map:
                    bits = bits_map[kind]
                else:
                    bits = io_bits if kind == "io" else xbits if kind == "x" else ebits
                # Any unknown kind that fell through classify as "q"
                if bits_map and kind not in bits_map and kind not in ("io", "x", "sh", "o", "kvb", "attn", "dmlp"):
                    bits = ebits
                if w.ndim != 2:        # es. bias 1D non previsto come 'q' -> tienilo f32
                    out_dict[name] = w.astype(np.float32); continue
                if group_size > 0 and bits <= 4:
                    q, s = quant_int4_grouped(w, bits, group_size)
                else:
                    q, s = (quant_int2(w, bits) if bits <= 2 else
                            quant_int4(w, bits) if bits <= 4 else quant_int8(w, bits))
                out_dict[name] = q
                out_dict[name + ".qs"] = s

def free_gb(p): return shutil.disk_usage(p).free / 1e9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--indir", default=None)
    ap.add_argument("--outdir", required=False)
    ap.add_argument("--ebits", type=int, default=None)   # bit residenti (default 4; 8 per --mtp/--indexer)
    ap.add_argument("--io-bits", type=int, default=8)    # bit di embed/lm_head
    ap.add_argument("--xbits", type=int, default=None)   # bit degli expert ROUTED (streaming); default=ebits
    # Mixed-precision: per-tensor-type bit overrides. Default = ebits (all same).
    # Set these higher to protect sensitive tensors from quantization error.
    ap.add_argument("--shared-bits", type=int, default=None,
        help="bits for shared expert (fires on every token, highest sensitivity). Default=ebits")
    ap.add_argument("--o-bits", type=int, default=None,
        help="bits for o_proj attention (reconstructs output, biggest attn tensor). Default=ebits")
    ap.add_argument("--kvb-bits", type=int, default=None,
        help="bits for kv_b_proj (reconstructs KV cache on every decode). Default=ebits")
    ap.add_argument("--attn-bits", type=int, default=None,
        help="bits for other attention projections (q_a, q_b, kv_a). Default=ebits")
    ap.add_argument("--dmlp-bits", type=int, default=None,
        help="bits for dense MLP (first 3 layers). Default=ebits")
    ap.add_argument("--group-size", type=int, default=0,  # 0 = per-row (backward compat); 128 = group-scaled
        help="group size for int4 scales: 0=per-row (default), 128=one scale per 128 elements (much better quality)")
    ap.add_argument("--n-layers", type=int, default=78)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-nvfp4", action="store_true",
        help="unit-test del dequant NVFP4 (LUT e2m1 + round-trip), nessun download / no network")
    ap.add_argument("--mtp", action="store_true",
        help="download and convert ONLY the MTP head (model.layers.<n_layers>.*) -> out-mtp-*.safetensors")
    ap.add_argument("--indexer", action="store_true",
        help="extract ONLY the DSA lightning-indexer weights -> out-idx-*.safetensors. WARNING: "
             "indexer tensors are spread across nearly every shard, so this re-downloads the whole "
             "repository (~756 GB of traffic) to retain only a few GB. Resumable per shard. "
             "Recommended: --ebits 8.")
    a = ap.parse_args()
    if a.ebits is None:
        # testa MTP a int4 = acceptance ~0-4% (misurato, issue #8): il draft sbaglia sempre
        # e la speculazione non parte mai. A int8: 39-59%, 2.2-2.8 token/forward.
        a.ebits = 8 if (a.mtp or a.indexer) else 4
    if a.xbits is None: a.xbits = a.ebits

    # Build per-type bits map. If a type-specific arg is set, use it; otherwise the
    # converter falls back to ebits for that type.
    bits_map = {}
    if a.shared_bits is not None: bits_map["sh"] = a.shared_bits
    if a.o_bits is not None:      bits_map["o"] = a.o_bits
    if a.kvb_bits is not None:    bits_map["kvb"] = a.kvb_bits
    if a.attn_bits is not None:   bits_map["attn"] = a.attn_bits
    if a.dmlp_bits is not None:   bits_map["dmlp"] = a.dmlp_bits
    if bits_map:
        print(f"[MIXED] precision map: " + ", ".join(f"{k}={v}bit" for k,v in sorted(bits_map.items())))

    if a.selftest_nvfp4:
        import torch
        # 1) LUT e2m1: i 16 codici devono decodificare esattamente ai valori attesi.
        lut = torch.tensor(_E2M1, dtype=torch.float32)
        expect = [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,-0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0]
        assert lut.tolist() == expect, "LUT e2m1 errata"
        print("[nvfp4] LUT e2m1: 16/16 codici OK")
        # 2) round-trip: costruisco un tensore ai SOLI valori rappresentabili (scala nota per
        #    blocco+globale), impacchetto come modelopt, poi dequant deve tornare ESATTO.
        import numpy as np, io
        from safetensors.torch import save as st_save
        from safetensors import safe_open
        rng = np.random.default_rng(0); O, I, GS = 8, 64, 16
        codes = rng.integers(0, 16, size=(O, I)).astype(np.uint8)   # nibble e2m1 casuali
        w4 = np.array(_E2M1, np.float32)[codes]                      # [O,I]
        # scale per-blocco (rappresentabili in f8e4m3) + globale piccola (stile modelopt)
        blk = rng.choice([0.5,1.0,2.0,4.0,8.0], size=(O, I//GS)).astype(np.float32)
        gscale = np.float32(3.9e-5)
        W = w4 * np.repeat(blk, GS, axis=1) * gscale                 # riferimento esatto
        # impacchetto: pari->nibble basso, dispari->alto
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
        import ml_dtypes  # solo per il test: encode f8e4m3 delle scale di blocco
        tens = {name: torch.from_numpy(arr) for name, arr in {
            "w.weight": packed,
            "w.weight_scale": blk.astype(ml_dtypes.float8_e4m3fn).view(np.uint8),  # placeholder
        }.items()}
        # torch non ha un costruttore da bytes f8: passo via file safetensors scritto a mano.
        # piu' semplice: uso direttamente dequant_nvfp4 su un finto 'f' in-memory.
        class _F:
            def __init__(s, d): s.d = d
            def get_tensor(s, n): return s.d[n]
            def get_slice(s, n): return None
        blk_f8 = blk.astype(ml_dtypes.float8_e4m3fn)                 # quantizza le scale a f8
        f = _F({"w.weight": torch.from_numpy(packed),
                "w.weight_scale": torch.from_numpy(blk_f8.view(np.uint8)).view(torch.float8_e4m3fn),
                "w.weight_scale_2": torch.tensor(gscale)})
        got = dequant_nvfp4(f, "w.weight")
        # riferimento con scale gia' quantizzate a f8 (per confronto esatto)
        Wq = w4 * np.repeat(blk_f8.astype(np.float32), GS, axis=1) * gscale
        maxerr = float(np.abs(got - Wq).max())
        print(f"[nvfp4] round-trip encode->dequant: max abs err = {maxerr:.3e} "
              f"({'OK' if maxerr < 1e-9 else 'FAIL'})")
        assert maxerr < 1e-9
        # 3) requant colibri int4 su valori dequantati -> errore piccolo atteso
        q, s = quant_int4(got.astype(np.float32), 4)
        rb = (I + 1)//2; qb = q.reshape(O, rb)
        lo = (qb & 0x0F).astype(np.int32) - 8; hi = ((qb >> 4) & 0x0F).astype(np.int32) - 8
        deq = np.empty((O, I), np.float32); deq[:, 0::2] = lo; deq[:, 1::2] = hi[:, :I-I//2]
        deq = deq * s[:, None]
        rel = np.abs(deq - got).mean() / (np.abs(got).mean() + 1e-12)
        # Informativo, NON un test di uguaglianza: requantizzare int4 per-riga dati che
        # spaziano 16x per il block-scale costa ~0.17 di errore relativo di suo. La soglia
        # larga becca solo una corruzione grossolana, non e' un bound di precisione.
        # EN: informational — per-row int4 requant of 16x-block-range data inherently ~0.17.
        print(f"[nvfp4] dequant->colibri int4->dequant: errore rel medio = {rel:.4f} "
              f"(atteso ~0.17; {'OK' if rel < 0.30 else 'ANOMALO'})")
        assert rel < 0.30, f"requant rel err {rel:.3f} troppo alto: dequant probabilmente corrotto"
        print("[nvfp4] SELFTEST OK")
        return

    if a.selftest:
        import torch
        w = (torch.randn(256, 256) * 0.3)
        O, I = w.shape; bs = 128
        sc = torch.zeros(O // bs, I // bs)
        for bi in range(O // bs):
            for bj in range(I // bs):
                blk = w[bi*bs:(bi+1)*bs, bj*bs:(bj+1)*bs]
                sc[bi, bj] = blk.abs().max() / 448.0
        q = (w / sc.repeat_interleave(bs,0).repeat_interleave(bs,1)).to(torch.float8_e4m3fn)
        deq = (q.to(torch.float32) * sc.repeat_interleave(bs,0).repeat_interleave(bs,1))
        rel = (deq - w).abs().mean() / w.abs().mean()
        print(f"[selftest fp8 block-dequant] mean relative error = {rel:.4f}  "
              f"({'OK' if rel < 0.05 else 'HIGH'})")
        return

    os.makedirs(a.outdir, exist_ok=True)
    if a.indir:    # conversione locale (test)
        shards = sorted(glob.glob(os.path.join(a.indir, "*.safetensors")))
        from safetensors.numpy import save_file
        for i, sp in enumerate(shards):
            out = {}; convert_shard(sp, out, a.n_layers, a.ebits, a.io_bits, a.xbits, group_size=a.group_size, bits_map=bits_map)
            save_file(out, os.path.join(a.outdir, f"out-{i:05d}.safetensors"))
        # copia config + tokenizer
        for fn in ["config.json"]:
            src = os.path.join(a.indir, fn)
            if os.path.exists(src): shutil.copy(src, a.outdir)
        print(f"converted {len(shards)} shards -> {a.outdir}")
        return

    # reale: scarica shard per shard, converte, cancella
    # EN: real: download shard by shard, convert, delete
    #
    # ROBUSTEZZA RETE: timeout brevi sulle read cosi' un download appeso FALLISCE invece
    # di restare fermo per sempre. 8s, non 30: "timeout" = ZERO byte ricevuti in quella
    # finestra; su un transfer vivo i chunk arrivano di continuo, quindi 8s e' sicuro e
    # uno stallo costa 8s invece di 30.
    # EN: NETWORK ROBUSTNESS: short read timeouts so a hung download FAILS instead of
    # EN: sitting there forever. 8s, not 30: a "timeout" means ZERO bytes received in that
    # EN: window; a live transfer delivers chunks constantly, so 8s is safe and a stall
    # EN: costs 8s instead of 30.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "8")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")
    # log con timestamp: i messaggi "Trying to resume" di hf_hub diventano databili.
    # EN: timestamped logs: hf_hub's "Trying to resume" messages become datable.
    import logging
    logging.basicConfig(format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    # hf_xet si blocca quando la rete si riavvia (connessioni zombie senza timeout):
    # forza la via HTTP classica, che curl ha dimostrato funzionare. (misurato 2026-07-02)
    # EN: hf_xet hangs when the network restarts (zombie connections with no timeout):
    # EN: force the classic HTTP path, which curl proved works (measured 2026-07-02).
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # =0 per riabilitare xet / to re-enable xet
    from huggingface_hub import HfApi, hf_hub_download

    # lock anti-doppione: DUE convertitori sulla stessa outdir si corrompono a vicenda.
    # EN: anti-duplicate lock: TWO converters on the same outdir corrupt each other.
    # fcntl is Unix-only; on Windows use msvcrt or skip locking.
    lock = open(os.path.join(a.outdir, ".convert.lock"), "w")
    try:
        import fcntl
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("ERROR: another converter is already using this output directory. Exiting."); return
    except ImportError:
        try:
            import msvcrt
            try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                print("ERROR: another converter is already using this output directory. Exiting."); return
        except ImportError:
            pass  # no locking available — single-user converter, acceptable

    # dimensioni note dei file, riempite dopo repo_info: il downloader multi-stream le usa
    # per calcolare i confini dei segmenti e per sapere quando un file e' completo.
    # EN: known file sizes, filled after repo_info: the multi-stream downloader uses them
    # EN: to compute segment boundaries and to know when a file is complete.
    SIZES = {}

    def download_retry(repo, fn, dest, tries=999):
        """Downloader multi-stream con resume via Range. Apre N segmenti concorrenti
        (default 2, COLI_DL_STREAMS per cambiarli) e salva lo stato per-segmento in un
        sidecar .seg -> NESSUN byte perso comunque muoia la connessione. Un singolo stream
        HF e' limitato a ~2 MB/s (misurato); 2 stream ~ raddoppiano il throughput senza
        saturare una linea domestica. File piccoli, COLI_DL_STREAMS=1 o un vecchio .part
        legacy -> percorso a stream singolo (_download_single).
        EN: multi-stream Range-resume downloader. Opens N concurrent segments (default 2,
        EN: COLI_DL_STREAMS to change) and saves per-segment state in a .seg sidecar -> NO
        EN: byte is lost however the connection dies. A single HF stream is paced at
        EN: ~2 MB/s (measured); 2 streams roughly double throughput without saturating a
        EN: home line. Small files, COLI_DL_STREAMS=1 or a legacy .part -> single-stream
        EN: path (_download_single)."""
        import time as _t, threading, urllib.request, urllib.error
        url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
        out = os.path.join(dest, fn); part = out + ".part"; side = part + ".seg"
        os.makedirs(dest, exist_ok=True)
        expected = SIZES.get(fn)
        if os.path.exists(out) and (expected is None or os.path.getsize(out) == expected):
            return out
        NS = max(1, min(8, int(os.environ.get("COLI_DL_STREAMS", "2"))))
        # un .part senza sidecar l'ha scritto una versione precedente a stream singolo.
        # EN: a .part without a sidecar was written by an older single-stream version.
        legacy = os.path.exists(part) and not os.path.exists(side)
        if expected is None or expected < (256 << 20) or NS == 1 or legacy:
            return _download_single(url, fn, out, part, expected)
        # ---- multi-stream ----
        segs = [(expected * t // NS, expected * (t + 1) // NS) for t in range(NS)]
        done = [0] * NS
        # riprendi lo stato dei segmenti se il sidecar combacia (stesso N, stessa size).
        # EN: resume per-segment progress if the sidecar matches (same N, same size).
        if os.path.exists(side):
            try:
                st = json.loads(open(side).read())
                if st.get("n") == NS and st.get("size") == expected: done = st["done"]
            except Exception: pass
        if not os.path.exists(part):
            with open(part, "wb") as f: f.truncate(expected)   # file sparse / sparse file
        fd = os.open(part, os.O_WRONLY)
        t0 = _t.time(); nres = [0]; log_lock = threading.Lock(); stopfail = []
        def worker(t):
            s0, s1 = segs[t]
            while done[t] < s1 - s0 and not stopfail:
                pos = s0 + done[t]
                req = urllib.request.Request(url, headers={"User-Agent": "colibri-convert",
                                                           "Range": f"bytes={pos}-{s1-1}"})
                try:
                    with urllib.request.urlopen(req, timeout=8) as r:
                        if r.status != 206:               # Range ignorato: multi-stream impossibile
                            stopfail.append(t); return    # EN: Range ignored: multi-stream impossible
                        while done[t] < s1 - s0:
                            chunk = r.read(1 << 20)
                            if not chunk: break
                            rem = (s1 - s0) - done[t]     # mai oltre il segmento / never past the segment
                            if len(chunk) > rem: chunk = chunk[:rem]
                            os.pwrite(fd, chunk, s0 + done[t])
                            done[t] += len(chunk)
                except KeyboardInterrupt: raise
                except Exception as ex:
                    with log_lock:
                        nres[0] += 1
                        print(f"    [dl] s{t}: {type(ex).__name__} at {(s0+done[t])/1e9:.2f} GB: "
                              f"resuming (#{nres[0]})", flush=True)
                    _t.sleep(min(15, 1 + nres[0] // NS))
        th = [threading.Thread(target=worker, args=(t,), daemon=True) for t in range(NS)]
        for x in th: x.start()
        print(f"    [dl {_t.strftime('%H:%M:%S')}] connected: {NS} streams, "
              f"{sum(done)/1e9:.2f} of {expected/1e9:.2f} GB", flush=True)
        mark = sum(done); tmark = t0
        while any(x.is_alive() for x in th):
            _t.sleep(5)
            have = sum(done)
            tmpside = side + ".tmp"                       # checkpoint atomico / atomic checkpoint
            open(tmpside, "w").write(json.dumps({"n": NS, "size": expected, "done": list(done)}))
            os.replace(tmpside, side)
            now = _t.time()
            if now - tmark >= 30:
                print(f"    [dl {_t.strftime('%H:%M:%S')}] {have/1e9:5.2f} GB "
                      f"({(have-mark)/max(now-tmark,1e-9)/1e6:5.1f} MB/s, {NS} stream)", flush=True)
                mark = have; tmark = now
        os.close(fd)
        if stopfail:                                      # il server non onora il Range: fallback
            for f2 in (part, side):                       # EN: server won't honor Range: fall back
                if os.path.exists(f2): os.remove(f2)
            return _download_single(url, fn, out, part, expected)
        assert sum(done) == expected
        if os.path.exists(side): os.remove(side)
        os.replace(part, out)
        dt = max(_t.time() - t0, 1e-9)
        print(f"    [dl] {fn}: {expected/1e9:.2f} GB in {dt/60:.1f} min "
              f"({expected/dt/1e6:.1f} MB/s avg, {NS} streams, {nres[0]} resumes)", flush=True)
        return out

    def _download_single(url, fn, out, part, expected):
        """Percorso a stream singolo con resume via Range (file piccoli / .part legacy /
        COLI_DL_STREAMS=1). Un EOF corto ma pulito conta come ripresa; se non arriva
        NESSUN byte nuovo, backoff invece di girare a vuoto.
        EN: single-stream path with Range resume (small files / legacy .part /
        EN: COLI_DL_STREAMS=1). A clean short EOF counts as a resume; if NO new byte
        EN: arrives, back off instead of spinning."""
        import time as _t, urllib.request, urllib.error
        t0 = _t.time(); nres = 0; mark = 0; tmark = t0
        while True:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if expected is not None and have >= expected: break
            have0 = have
            req = urllib.request.Request(url, headers={"User-Agent": "colibri-convert"})
            if have: req.add_header("Range", f"bytes={have}-")
            try:
                with urllib.request.urlopen(req, timeout=8) as r:
                    if have and r.status == 200:          # server ha ignorato il Range: riparti pulito
                        have = 0                          # EN: server ignored Range: restart clean
                    if expected is None:
                        cl = r.headers.get("Content-Length")
                        if cl: expected = have + int(cl)
                    if have == 0 or nres:                 # segnale di vita subito / immediate sign of life
                        print(f"    [dl {_t.strftime('%H:%M:%S')}] connected"
                              f"{f' @ {have/1e9:.2f} GB' if have else ''}"
                              f"{f' of {expected/1e9:.2f} GB' if expected else ''}", flush=True)
                    with open(part, "ab" if have else "wb") as f:
                        if not have: f.truncate(0)
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk: break
                            f.write(chunk); have += len(chunk)
                            if have - mark >= 512 * 1024 * 1024 or _t.time() - tmark >= 30:
                                now = _t.time()
                                print(f"    [dl {_t.strftime('%H:%M:%S')}] {have/1e9:5.2f} GB "
                                      f"({(have-mark)/max(now-tmark,1e-9)/1e6:5.1f} MB/s)", flush=True)
                                mark = have; tmark = now
                if expected is None: break                # lunghezza ignota: passata singola / unknown length
                if have < expected:                       # EOF corto ma pulito: conta come ripresa
                    nres += 1                             # EN: clean short EOF: counts as a resume
                    if have == have0: _t.sleep(min(15, 1 + nres))   # zero progresso -> backoff / zero progress -> back off
            except KeyboardInterrupt: raise
            except urllib.error.HTTPError as ex:
                if ex.code == 416: break                  # gia' completo / already complete
                nres += 1
                print(f"    [dl] HTTP {ex.code} at {have/1e9:.2f} GB: resuming (#{nres})", flush=True)
                _t.sleep(min(15, 1 + nres))
            except Exception as ex:
                nres += 1
                print(f"    [dl] {type(ex).__name__} at {have/1e9:.2f} GB: resuming (#{nres})", flush=True)
                _t.sleep(min(15, 1 + nres))
        os.replace(part, out)
        dt = max(_t.time() - t0, 1e-9); sz = os.path.getsize(out)
        print(f"    [dl] {fn}: {sz/1e9:.2f} GB in {dt/60:.1f} min "
              f"({sz/dt/1e6:.1f} MB/s avg, {nres} resumes)", flush=True)
        return out

    from safetensors.numpy import save_file
    import time as _t
    info = None
    for att in range(10):
        try:
            info = HfApi().repo_info(a.repo, files_metadata=True)
            # dimensioni note dallo store: abilitano il download multi-stream a segmenti.
            # EN: sizes known from the store: enable segmented multi-stream download.
            SIZES.update({s.rfilename: s.size for s in info.siblings if s.size})
            break
        except KeyboardInterrupt: raise
        except Exception as ex:
            w = min(60, 5*(att+1)); print(f"repo_info failed ({type(ex).__name__}); retrying in {w}s", flush=True); _t.sleep(w)
    if info is None:
        print("ERROR: could not reach the repository after 10 retries. Check your network and repo name.", flush=True)
        return
    shards = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors"))
    if not shards:
        print("ERROR: no .safetensors shards found in this repository.", flush=True)
        return
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        try: shutil.copy(hf_hub_download(a.repo, fn, local_dir=a.outdir+"/_meta"), a.outdir)
        except Exception: pass
    tmp = os.path.join(a.outdir, "_inflight"); os.makedirs(tmp, exist_ok=True)
    if a.mtp:
        import urllib.request
        idx = json.loads(urllib.request.urlopen(
            f"https://huggingface.co/{a.repo}/resolve/main/model.safetensors.index.json", timeout=30).read())["weight_map"]
        pref = f"model.layers.{a.n_layers}."
        mtp_shards = sorted(set(v for k, v in idx.items() if k.startswith(pref)))
        print(f"[MTP] head at layer {a.n_layers}: {len(mtp_shards)} shards to process: {mtp_shards}")
        for i, sh in enumerate(mtp_shards):
            outp = os.path.join(a.outdir, f"out-mtp-{i:05d}.safetensors")
            if os.path.exists(outp): print(f"[MTP] {outp} already done"); continue
            print(f"[MTP {i+1}/{len(mtp_shards)}] downloading {sh}...", flush=True)
            p = download_retry(a.repo, sh, tmp)
            out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, keep_mtp=True, group_size=a.group_size, bits_map=bits_map)
            save_file(out, outp)
            os.remove(p)
            for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
                if os.path.isfile(blob): os.remove(blob)
            print(f"    -> {os.path.basename(outp)} ({os.path.getsize(outp)/1e9:.2f} GB, {len(out)} tensors)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True); print("[MTP] DONE."); return
    if a.indexer:
        import urllib.request
        idx = json.loads(urllib.request.urlopen(
            f"https://huggingface.co/{a.repo}/resolve/main/model.safetensors.index.json", timeout=30).read())["weight_map"]
        idx_shards = sorted(set(v for k, v in idx.items()
                                if "indexer" in k and 0 <= layer_idx(k) < a.n_layers))
        tot_gb = len(idx_shards) * 5.4
        print(f"[IDX] indexer weights across {len(idx_shards)} shards (~{tot_gb:.0f} GB total download, resumable)")
        for i, sh in enumerate(idx_shards):
            outp = os.path.join(a.outdir, f"out-idx-{i:05d}.safetensors")
            if os.path.exists(outp): continue             # gia' fatto -> ripartibile
            print(f"[IDX {i+1}/{len(idx_shards)}] downloading {sh}...", flush=True)
            p = download_retry(a.repo, sh, tmp)
            out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, keep_idx=True, group_size=a.group_size, bits_map=bits_map)
            if out: save_file(out, outp)
            os.remove(p)
            for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
                if os.path.isfile(blob): os.remove(blob)
            print(f"    -> {os.path.basename(outp)} ({len(out)} tensors)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True); print("[IDX] DONE."); return
    for i, sh in enumerate(shards):
        if free_gb(a.outdir) < a.min_free_gb:
            print(f"STOP: free space is below {a.min_free_gb} GB. Free space and rerun to resume."); break
        outp = os.path.join(a.outdir, f"out-{i:05d}.safetensors")
        if os.path.exists(outp): continue                 # gia' fatto -> ripartibile
        print(f"[{i+1}/{len(shards)}] downloading {sh} ({free_gb(a.outdir):.0f} GB free)...", flush=True)
        p = download_retry(a.repo, sh, tmp)
        out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, group_size=a.group_size, bits_map=bits_map)
        save_file(out, outp)
        os.remove(p)                                       # <-- cancella subito lo shard fp8
        for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
            if os.path.isfile(blob): os.remove(blob)
        print(f"    -> {os.path.basename(outp)} ({os.path.getsize(outp)/1e9:.2f} GB)", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("DONE." if i == len(shards)-1 else "INTERRUPTED (rerun to resume).")

if __name__ == "__main__":
    main()
