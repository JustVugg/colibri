#!/usr/bin/env python3
"""Oracolo tiny di GLM-5.3-Flash completo: testo piu' immagine.

Gli altri due oracoli coprono le meta' separate - il testo da solo e la torre
vision da sola. Nessuno dei due tocca il punto in cui si incontrano, cioe' la
sostituzione degli embedding dell'immagine nelle posizioni dei token segnaposto.
E' esattamente il punto dove un motore puo' sembrare giusto e non esserlo:
prendere gli embedding nell'ordine sbagliato, o allinearli sulla posizione
sbagliata, produce comunque un modello che parla.

Il riferimento e' il `Glm5NextForConditionalGeneration` ufficiale, che innesta
l'immagine con `inputs_embeds.masked_scatter(input_ids == image_token_id,
image_embeds)`: gli embedding entrano in ordine, uno per segnaposto, e le
posizioni non si spostano.

Il fixture porta anche le patch grezze in un file .f32 separato, cosi' la CLI
del motore puo' rileggerle senza dipendere da safetensors.

USO:
  python3 tools/make_glm53_multimodal_tiny.py --output ~/glm53_mm_tiny
"""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

EXPECTED_TRANSFORMERS = "5.16.1"
SEED = 1253

# La torre esce a OUT_HIDDEN e il flusso testuale entra a HIDDEN: il merger e'
# il pezzo che li fa combaciare, quindi devono essere lo stesso numero.
HIDDEN = 128
# Il vocabolario deve contenere il tokenizzatore vero che il fixture porta con
# se': 256 byte piu' i marcatori di GLM. Il token immagine e' uno di quelli,
# non un numero scelto a caso, cosi' un prompt scritto a mano puo' nominarlo.
# Esattamente quanti id definisce il tokenizzatore: 256 byte piu' i
# marcatori. Un vocabolario piu' largo lascerebbe al modello id che nessuna
# stringa rappresenta, e la prima cosa che ci inciampa e' la decodifica.
VOCAB = None                  # risolto dal tokenizzatore in build()
IMAGE_TOKEN = None

VIS_DEPTH, VIS_HIDDEN, VIS_HEADS, VIS_INTER = 2, 64, 4, 128
PATCH, TEMPORAL, MERGE, IN_CHANNELS = 4, 2, 2, 3
GRID_H, GRID_W = 4, 4          # 4x4 patch, merge 2 -> 4 token immagine

TEXT_CONFIG = {
    "vocab_size": VOCAB,
    "hidden_size": HIDDEN,
    "intermediate_size": 256,
    "moe_intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "n_shared_experts": 1,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "kv_lora_rank": 64,
    "q_lora_rank": 128,
    "qk_rope_head_dim": 0,
    "qk_nope_head_dim": 32,
    "v_head_dim": 32,
    "max_position_embeddings": 128,
    "layer_types": ["linear_attention", "linear_attention",
                    "linear_attention", "deepseek_sparse_attention"],
    "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
    "indexer_types": ["full", "full", "full", "full"],
    "index_topk": 4,
    "index_kpool": 2,
    "index_head_dim": 32,
    "index_n_heads": 2,
    "linear_head_dim": 32,
    "linear_num_heads": 4,
    "linear_conv_kernel_dim": 4,
    "hc_mult": 2,
    "hc_sinkhorn_iters": 3,
    "hc_eps": 1e-06,
    "rms_norm_eps": 1e-05,
    "routed_scaling_factor": 2.5,
    "swiglu_limit": 10.0,
    "pad_token_id": None,          # il default della classe cade fuori da VOCAB
    "bos_token_id": 0,
    "eos_token_id": 1,
    "tie_word_embeddings": False,
    "linear_attn_config": {
        "num_heads": 4, "head_dim": 32, "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0, "kda_layers": [0, 1, 2],
    },
    "num_nextn_predict_layers": 0,
}

# 8 posizioni, 4 delle quali sono l'immagine, e non all'inizio: se il motore
# sbagliasse allineamento con un offset costante, un blocco in testa lo
# nasconderebbe.
PROMPT_TEXT_BEFORE = "gu"     # due byte prima dell'immagine
PROMPT_TEXT_AFTER = "xy"      # due byte dopo, per vedere se l'allineamento slitta
GREEDY_STEPS = 4


@contextmanager
def tie_watch(torch, length, select_k):
    """Segnala se l'indexer ha scelto i pool con un pareggio sul bordo.

    Intercetta le chiamate a topk sui punteggi dei pool per l'ultima posizione,
    che e' quella che decide il token successivo. Se il k-esimo punteggio e il
    (k+1)-esimo sono identici, quale pool entri non lo decide il modello."""
    tied = [False]
    original = torch.Tensor.topk

    def watched(self, k, *args, **kwargs):
        if (self.dim() == 3 and self.shape[0] == 1 and self.shape[1] == length
                and self.shape[2] <= 2 * select_k + 4):
            row = sorted((float(v) for v in self[0, length - 1]), reverse=True)
            if len(row) > select_k and row[select_k - 1] == row[select_k]:
                tied[0] = True
        return original(self, k, *args, **kwargs)

    torch.Tensor.topk = watched
    try:
        yield tied
    finally:
        torch.Tensor.topk = original


def to_checkpoint_names(tensors, torch):
    """state_dict di transformers -> nomi del checkpoint pubblicato.

    L'albero dei moduli e il checkpoint su Hugging Face non si chiamano allo
    stesso modo, e due tensori sono anche fusi: la conv corta tiene q, k e v in
    un unico banco depthwise, e gli esperti stanno impilati per layer. Il
    motore legge il checkpoint, quindi e' il fixture che deve adeguarsi."""
    import re
    renamed = {}
    for name, tensor in tensors.items():
        match = re.match(r"(model\.language_model\.layers\.\d+\.)(.*)", name)
        if not match:
            renamed[name] = tensor
            continue
        prefix, rest = match.group(1), match.group(2)
        if rest.startswith("attn_hc."):
            renamed[prefix + "hc_attn_" + rest.split(".", 1)[1]] = tensor
        elif rest.startswith("ffn_hc."):
            renamed[prefix + "hc_ffn_" + rest.split(".", 1)[1]] = tensor
        elif rest.startswith("self_attn.forget_gate."):
            renamed[prefix + "self_attn." + rest.split("forget_gate.", 1)[1]] = tensor
        elif rest == "self_attn.conv1d.weight":
            # [3*qkv, 1, K], nell'ordine in cui il modulo poi la rilegge:
            # torch.split(mixed_qkv, [qkv_dim]*3) da' query, key, value.
            third = tensor.shape[0] // 3
            for slot, part in enumerate(("q", "k", "v")):
                renamed[f"{prefix}self_attn.{part}_conv1d.weight"] = \
                    tensor[slot * third:(slot + 1) * third].contiguous()
        elif rest == "mlp.experts.gate_up_proj":
            # [esperti, 2*inter, hidden]; _apply_gate fa chunk(2, dim=-1)
            # sull'uscita, quindi la prima meta' delle righe e' il gate.
            inter = tensor.shape[1] // 2
            for expert in range(tensor.shape[0]):
                renamed[f"{prefix}mlp.experts.{expert}.gate_proj.weight"] = \
                    tensor[expert, :inter].contiguous()
                renamed[f"{prefix}mlp.experts.{expert}.up_proj.weight"] = \
                    tensor[expert, inter:].contiguous()
        elif rest == "mlp.experts.down_proj":
            # [esperti, hidden, inter], gia' nel verso [uscita, ingresso]
            for expert in range(tensor.shape[0]):
                renamed[f"{prefix}mlp.experts.{expert}.down_proj.weight"] = \
                    tensor[expert].contiguous()
        else:
            renamed[name] = tensor
    return renamed


def build(output: Path, seed: int) -> int:
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers.models.glm5_next.configuration_glm5_next import (
        Glm5NextConfig, Glm5NextTextConfig, Glm5NextVisionConfig)
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration)

    if transformers.__version__ != EXPECTED_TRANSFORMERS:
        print(f"expected Transformers {EXPECTED_TRANSFORMERS}, "
              f"found {transformers.__version__}")
        return 2

    output.mkdir(parents=True, exist_ok=True)
    # Il tokenizzatore del fixture, e da li' l'id del token immagine.
    from make_glm53_tokenizer import SPECIALS, build as build_tokenizer
    build_tokenizer(output / "tokenizer.json")
    image_token = 256 + SPECIALS.index("<|image|>")
    text_config = dict(TEXT_CONFIG, vocab_size=256 + len(SPECIALS))
    n_image = (GRID_H // MERGE) * (GRID_W // MERGE)
    prompt = ([ord(ch) for ch in PROMPT_TEXT_BEFORE]
              + [image_token] * n_image
              + [ord(ch) for ch in PROMPT_TEXT_AFTER])

    torch.manual_seed(seed)
    vision_config = Glm5NextVisionConfig(
        depth=VIS_DEPTH, hidden_size=VIS_HIDDEN, num_heads=VIS_HEADS,
        intermediate_size=VIS_INTER, patch_size=PATCH,
        temporal_patch_size=TEMPORAL, spatial_merge_size=MERGE,
        out_hidden_size=HIDDEN, projection_intermediate_size=VIS_INTER,
        in_channels=IN_CHANNELS, image_size=PATCH * GRID_H,
    )
    config = Glm5NextConfig(
        text_config=Glm5NextTextConfig(**text_config),
        vision_config=vision_config,
        image_token_id=image_token,
    )
    # Pesi dall'init nativo di transformers sotto seme fisso, non riempiti a
    # mano: un riempimento uniforme ignora il fan-in, il segnale si spegne
    # strato dopo strato e l'immagine finisce per spostare i logit di 1e-05.
    # Con l'init vero l'immagine cambia davvero la risposta, che e' la sola
    # condizione per cui questo fixture prova qualcosa. I pesi finiscono nel
    # model.safetensors, quindi il test non dipende dal fatto che una torch
    # diversa rigeneri gli stessi numeri: dipende solo da questo file.
    model = Glm5NextForConditionalGeneration(config).eval()

    n_patches = GRID_H * GRID_W
    width = IN_CHANNELS * TEMPORAL * PATCH * PATCH
    pixels = torch.linspace(-1.0, 1.0, n_patches * width,
                            dtype=torch.float32).view(n_patches, width)
    grid_thw = torch.tensor([[1, GRID_H, GRID_W]], dtype=torch.long)
    ids = torch.tensor([prompt], dtype=torch.long)

    expected_image_tokens = n_image
    placeholders = sum(1 for t in prompt if t == image_token)
    if placeholders != expected_image_tokens:
        print(f"il prompt ha {placeholders} segnaposto ma la griglia "
              f"{GRID_H}x{GRID_W} con merge {MERGE} ne produce "
              f"{expected_image_tokens}")
        return 2

    with torch.no_grad():
        logits = model(input_ids=ids, pixel_values=pixels,
                       image_grid_thw=grid_thw).logits[0].float()
        forcing = logits.argmax(-1).tolist()

        sequence = list(prompt)
        greedy = []
        exact_steps = GREEDY_STEPS
        select_k = TEXT_CONFIG["index_topk"] // TEXT_CONFIG["index_kpool"]
        for step in range(GREEDY_STEPS):
            with tie_watch(torch, len(sequence), select_k) as tied:
                out = model(input_ids=torch.tensor([sequence], dtype=torch.long),
                            pixel_values=pixels, image_grid_thw=grid_thw).logits[0]
            # Il punteggio dei pool passa per una ReLU, quindi ogni pool che non
            # piace a nessuna testa vale esattamente 0 e i pareggi sono comuni.
            # Su un pareggio al bordo della selezione, quale pool entri lo
            # decide l'ordine interno di torch.topk, che non e' specificato e
            # cambia fra backend: da li' in poi questa traccia non e' piu' un
            # bersaglio legittimo per nessuna implementazione, nemmeno per una
            # seconda esecuzione di torch su un'altra macchina.
            if tied[0] and exact_steps == GREEDY_STEPS:
                exact_steps = step
            nxt = int(out[-1].argmax())
            greedy.append(nxt)
            sequence.append(nxt)

        # Una traccia di SOLO TESTO, per i cancelli segment ed edge: quella
        # pila non ha la torre vision, quindi confrontarla con una risposta
        # che dipende dall'immagine sarebbe chiederle di indovinare.
        text_only = [ord(ch) for ch in PROMPT_TEXT_BEFORE + PROMPT_TEXT_AFTER]
        text_sequence = list(text_only)
        for _ in range(GREEDY_STEPS):
            step = model(input_ids=torch.tensor([text_sequence], dtype=torch.long)).logits[0]
            text_sequence.append(int(step[-1].argmax()))

        # Guardia: un'altra immagine deve dare un'altra risposta. Se non
        # cambia nulla, questo fixture non e' in grado di accorgersi di una
        # vision rotta e non va scritto, per quanto passi.
        other = model(input_ids=ids, pixel_values=-pixels,
                      image_grid_thw=grid_thw).logits[0].argmax(-1).tolist()

    if forcing == other:
        print("il fixture non distingue due immagini opposte: sarebbe un test "
              "del solo testo travestito da test vision")
        return 2
    if len(set(forcing + greedy)) < 3:
        print(f"uscita degenere ({sorted(set(forcing + greedy))}): un motore "
              "sbagliato passerebbe lo stesso")
        return 2

    tensors = to_checkpoint_names(
        {name: parameter.detach().contiguous().to(torch.float32)
         for name, parameter in model.state_dict().items()
         if parameter.dtype.is_floating_point}, torch)
    save_file(tensors, str(output / "model.safetensors"), metadata={"format": "pt"})
    (output / "patches.f32").write_bytes(pixels.contiguous().numpy().tobytes())

    payload = config.to_dict()
    payload["architectures"] = ["Glm5NextForConditionalGeneration"]
    (output / "config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    reference = {
        "schema_version": 1,
        "source": "transformers",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "seed": seed,
        "prompt": prompt,
        "image_token_id": image_token,
        "grid": [GRID_H, GRID_W],
        "patch_shape": list(pixels.shape),
        "image_tokens": expected_image_tokens,
        "teacher_forcing": forcing,
        "greedy": greedy,
        # Fin dove la traccia greedy e' un bersaglio legittimo: oltre, la
        # selezione dell'indexer e' passata da un pareggio e il seguito
        # dipende dall'ordine interno di torch.topk, non dal modello.
        "greedy_exact_steps": exact_steps,
        # Nomi che i test condivisi di segment/edge si aspettano, sulla
        # traccia senza immagine.
        "prompt_ids": text_only,
        "full_ids": text_sequence,
        "last_logits": logits[-1].tolist(),
    }
    (output / "ref.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    note = ("" if exact_steps == GREEDY_STEPS else
            f"; greedy confrontabile solo per {exact_steps} passi su "
            f"{GREEDY_STEPS} (pareggio nella selezione dei pool)")
    print(f"wrote {output} ({len(tensors)} tensors, {expected_image_tokens} "
          f"token immagine su {len(prompt)} posizioni{note})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=12,
                        help="quanti semi provare per trovarne uno dove la "
                             "traccia greedy resta confrontabile fino in fondo")
    arguments = parser.parse_args()

    # I pareggi nella selezione dei pool non sono un difetto da nascondere, ma
    # un fixture in cui arrivano al primo passo prova quasi niente. Si provano
    # piu' semi e si tiene il primo che resta confrontabile per intero; se non
    # ce n'e' nessuno si tiene il migliore, dicendo quale.
    import json as _json
    best_seed, best_steps = None, -1
    for offset in range(arguments.seeds):
        seed = SEED + offset
        code = build(arguments.output, seed)
        if code:
            continue
        steps = _json.loads((arguments.output / "ref.json").read_text())["greedy_exact_steps"]
        if steps > best_steps:
            best_seed, best_steps = seed, steps
        if steps == GREEDY_STEPS:
            return 0
    if best_seed is None:
        print(f"nessuno dei {arguments.seeds} semi ha prodotto un fixture valido")
        return 2
    print(f"nessun seme su {arguments.seeds} resta confrontabile per intero; "
          f"tengo {best_seed} con {best_steps} passi su {GREEDY_STEPS}")
    return build(arguments.output, best_seed)


if __name__ == "__main__":
    raise SystemExit(main())
