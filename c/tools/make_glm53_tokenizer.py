#!/usr/bin/env python3
"""Un tokenizer.json vero, piccolo abbastanza per le fixture di GLM-5.3.

Il fixture del testo porta un tokenizzatore segnaposto con una sola voce nel
vocabolario. Basta per gli oracoli, che parlano in id, e non basta per niente
altro: il protocollo di serve prende testo, lo tokenizza e risponde testo,
quindi con un vocabolario da una voce ogni prompt diventa vuoto e non si puo'
provare nulla di quello che gli sta intorno.

Qui si scrive un tokenizzatore a livello di byte: i 256 byte con la mappa
GPT-2 che tok.h costruisce in tk_build_bytemap, piu' i token speciali di GLM
come added_tokens atomici. Nessun merge, quindi ogni stringa si scompone nei
suoi byte: non e' efficiente e non deve esserlo, deve essere completo e
reversibile su qualunque input, compresi i marcatori di ruolo e i blocchi
degli strumenti.

USO:
  python3 tools/make_glm53_tokenizer.py --output tokenizer.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Gli stessi intervalli di tk_build_bytemap in tok.h: i byte stampabili si
# rappresentano da soli, gli altri traslano a partire da U+0100.
DIRECT = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))

# I marcatori di GLM-5.3 che il template e il parser degli strumenti usano
# davvero. Sono added_tokens con "special": true, cosi' tok.h li tratta come
# atomici e li mette nel suo insieme di speciali.
SPECIALS = (
    "<|endoftext|>", "[MASK]", "[gMASK]", "[sMASK]", "<sop>", "<eop>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>",
    "<|begin_of_image|>", "<|end_of_image|>", "<|image|>",
    "<think>", "</think>",
    "<tool_call>", "</tool_call>",
    "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
    "<tool_response>", "</tool_response>", "<tools>", "</tools>",
)


def byte_to_char():
    """byte -> carattere, la mappa di GPT-2 che tok.h ricostruisce da sola."""
    mapping, extra = {}, 0
    for value in range(256):
        if value in DIRECT:
            mapping[value] = chr(value)
        else:
            mapping[value] = chr(256 + extra)
            extra += 1
    return mapping


def build(output: Path) -> int:
    table = byte_to_char()
    vocab = {table[value]: value for value in range(256)}
    added = [
        {"id": 256 + position, "content": token, "special": True,
         "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": False}
        for position, token in enumerate(SPECIALS)
    ]
    payload = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                          "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                    "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": False, "byte_fallback": True,
                  "vocab": vocab, "merges": []},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    print(f"scritto {output}: {len(vocab)} byte + {len(added)} speciali "
          f"= {len(vocab) + len(added)} id (vocab_size minimo "
          f"{256 + len(SPECIALS)})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return build(parser.parse_args().output)


if __name__ == "__main__":
    raise SystemExit(main())
