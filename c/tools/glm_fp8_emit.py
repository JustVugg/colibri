"""Helper: salva pesi in FP8 e4m3 + scale a blocchi 128x128, nello STESSO layout del
checkpoint reale GLM-5.2-FP8 che `convert_fp8_to_int4.py` legge.

Layout (deve combaciare col `dequant()` del converter, convert_fp8_to_int4.py:164-169):
  - `name`            F8_E4M3  [O, I]
  - `name_scale_inv`  F32      [ceil(O/128), ceil(I/128)]   (NOTA: '_scale_inv', underscore)
  dequant: W = q.float() * scale.repeat_interleave(128,0).repeat_interleave(128,1)[:O,:I]

Convenzione FBGEMM/TransformerEngine: scale = amax(blocco)/448 (448 = max e4m3),
si MEMORIZZA il valore e si MOLTIPLICA in dequant. Malgrado il nome "_scale_inv" il
checkpoint memorizza la scala (non il reciproco): e' un MOLTIPLIER.

EN: Helper that writes weights as FP8 e4m3 with 128x128 block scales, in the SAME layout
EN: as the real GLM-5.2-FP8 checkpoint that `convert_fp8_to_int4.py` reads.
EN: FBGEMM/TransformerEngine convention: scale = amax(block)/448, stored (not its
EN: reciprocal) and MULTIPLIED on dequant. Despite the name "_scale_inv" it is a multiplier.
"""
import torch

E4M3_MAX = 448.0   # max valore rappresentabile in float8_e4m3fn / max representable value
BLOCK = 128         # granularita' delle scale a blocchi del checkpoint FP8 / FP8 block scale granularity


def keep_f32(name, t):
    """Stesso set F32 di `classify()` in convert_fp8_to_int4.py (norme, router, bias 1-D).
    Tutti gli altri tensori 2-D vengono quantizzati FP8 (attn/mlp/shared/expert/embed/lm_head).
    EN: Same F32 set as the converter's classify(): norms, router, 1-D biases. All other 2-D
    EN: tensors are FP8-quantized (attn/mlp/shared/expert/embed/lm_head)."""
    if t.dim() < 2:
        return True                                  # bias 1-D, e_score_correction_bias
    if name.endswith("e_score_correction_bias"):
        return True
    if name.endswith("mlp.gate.weight"):
        return True                                  # router (NON gate_proj): tenuto F32 / kept F32
    if name.endswith("norm.weight") or name == "model.norm.weight":
        return True                                  # RMSNorm
    return False


def fp8_block_quantize(w):
    """w: [O,I] f32 -> (w_fp8 float8_e4m3fn [O,I], scale_inv f32 [ceil(O/128),ceil(I/128)]).
    Identica matematica al `--selftest` del converter (scale = amax(blocco)/448). Padda a
    multipli di 128 internamente (gli zeri non alzano l'amax) e fa slice al risultato.
    EN: same math as the converter's --selftest. Pads to 128 multiples internally (zeros do
    EN: not raise amax), slices the result back to [O,I]."""
    O, I = w.shape
    nbO, nbI = (O + BLOCK - 1) // BLOCK, (I + BLOCK - 1) // BLOCK
    Op, Ip = nbO * BLOCK, nbI * BLOCK
    wpad = torch.zeros(Op, Ip, dtype=torch.float32, device=w.device)
    wpad[:O, :I] = w
    wb = wpad.view(nbO, BLOCK, nbI, BLOCK)               # [nbO, BLOCK, nbI, BLOCK]
    amax = wb.abs().amax(dim=(1, 3))                      # [nbO, nbI]
    scale = amax / E4M3_MAX                               # FBGEMM/TE: memorizza la scala / store the scale
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # blocco tutto-zero -> no div0
    scale = scale.to(torch.float32)
    q = (wpad / scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)).clamp(-E4M3_MAX, E4M3_MAX)
    w_fp8 = q.to(torch.float8_e4m3fn)
    return w_fp8[:O, :I].contiguous(), scale.contiguous()


def fp8_block_dequantize(w_fp8, scale):
    """Esatto inverso di fp8_block_quantize, e identico al `dequant()` del converter.
    EN: exact inverse of fp8_block_quantize, identical to the converter's dequant()."""
    O, I = w_fp8.shape
    qf = w_fp8.to(torch.float32)
    return qf * scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[:O, :I]


def unfuse_experts(sd):
    """Split HF's fused 3-D `experts.gate_up_proj` [E, 2*M, I] into per-expert 2-D
    `experts.{e}.gate_proj` [M, I] + `experts.{e}.up_proj` [M, I], and
    `experts.down_proj` [E, I, M] -> `experts.{e}.down_proj` [M_out, I].

    The real GLM-5.2-FP8 checkpoint stores experts UNFUSED as per-expert 2-D tensors
    (gate_proj, up_proj, down_proj), each with its own _scale_inv. HF's
    GlmMoeDsaForCausalLM fuses gate+up into a single 3-D gate_up_proj for efficiency.
    The converter (classify + ndim!=2 guard) and the C engine both expect the unfused
    layout, so we split before saving.

    Idempotent: if experts are already unfused (no 3-D gate_up_proj), returns sd as-is.
    EN: split HF's fused 3-D expert weights into the per-expert 2-D layout that the real
    EN: checkpoint uses and the converter/engine expect. No-op if already unfused."""
    keys_to_remove = []
    new_entries = {}
    for name, t in sd.items():
        if not name.endswith(".mlp.experts.gate_up_proj"):
            continue
        # prefix = everything before ".mlp.experts.gate_up_proj"
        prefix = name[:-len(".mlp.experts.gate_up_proj")]
        E, twoM, I = t.shape          # [E, 2*intermediate, input]
        M = twoM // 2
        for e in range(E):
            new_entries[f"{prefix}.mlp.experts.{e}.gate_proj.weight"] = t[e, :M, :].contiguous()
            new_entries[f"{prefix}.mlp.experts.{e}.up_proj.weight"]   = t[e, M:, :].contiguous()
        keys_to_remove.append(name)
    # down_proj may be 3-D [E, I, M] in the fused form, or already per-expert
    for name, t in sd.items():
        if not name.endswith(".mlp.experts.down_proj") or t.dim() != 3:
            continue
        prefix = name[:-len(".mlp.experts.down_proj")]
        E = t.shape[0]
        for e in range(E):
            new_entries[f"{prefix}.mlp.experts.{e}.down_proj.weight"] = t[e].contiguous()
        keys_to_remove.append(name)
    for k in keys_to_remove:
        sd.pop(k, None)
    sd.update(new_entries)
    return sd


def state_dict_to_fp8(sd):
    """Converte uno state_dict HuggingFace nel layout FP8 del checkpoint reale:
    per ogni tensore quantizzabile 2-D scrive `{name}` (F8_E4M3) + `{name}_scale_inv` (F32);
    norme/router/bias e qualsiasi tensore NON 2-D (es. pesi MLA impaccati 3-D) restano nel
    dtype originale. Questo rispecchia il guard `w.ndim != 2 -> f32` del converter
    (convert_fp8_to_int4.py:184). EN: builds the real-checkpoint FP8 layout. Only exactly-2-D
    tensors are FP8-quantized; anything else (1-D, 3-D packed MLA weights, ...) is kept, exactly
    like the converter's `ndim != 2 -> f32` guard."""
    out = {}
    for name, t in sd.items():
        if keep_f32(name, t) or t.dim() != 2:
            out[name] = t                               # f32 / 1-D / 3-D+: tieni / keep
        else:
            w_fp8, scale = fp8_block_quantize(t.float())
            out[name] = w_fp8
            out[name + "_scale_inv"] = scale
    return out


def save_fp8_safetensors(sd, path):
    """Quantizza a blocchi FP8 e salva in un singolo safetensors leggibile dal converter
    via `--indir`. EN: block-quantize to FP8 and save a single safetensors for the converter."""
    from safetensors.torch import save_file
    out = state_dict_to_fp8(sd)
    save_file({k: v.contiguous() for k, v in out.items()}, str(path))
    n_fp8 = sum(1 for v in out.values() if v.dtype == torch.float8_e4m3fn)
    return n_fp8, len(out)
