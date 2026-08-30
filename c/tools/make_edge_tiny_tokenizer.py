#!/usr/bin/env python3
"""Write a deterministic byte-BPE tokenizer for generated Edge fixtures.

The real adapters always load the checkpoint's production tokenizer. Several
existing math-only tiny generators intentionally omit tokenizer.json; this
small fixture supplies an identity byte vocabulary so the Edge release gate
can exercise text sizing/round-trips without downloading a model tokenizer.
It is test data, never a tokenizer substitute for a real checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def byte_symbols() -> list[str]:
    direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    symbols: list[str] = []
    extra = 0
    for byte in range(256):
        if byte in direct:
            codepoint = byte
        else:
            codepoint = 256 + extra
            extra += 1
        symbols.append(chr(codepoint))
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--vocab-size", type=int, required=True)
    args = parser.parse_args()
    if args.vocab_size < 34:
        raise SystemExit("Edge fixture vocabulary must contain at least 34 IDs")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    symbols = byte_symbols()
    vocab = {symbols[token]: token for token in range(min(256, args.vocab_size))}
    # Math-only fixtures can have vocabularies larger than the 256 byte IDs.
    # Give every remaining row a unique decodable piece so a generated oracle
    # token never falls outside the fixture tokenizer's id table.
    for token in range(256, args.vocab_size):
        vocab[f"<|edge_fixture_{token}|>"] = token
    payload = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel", "add_prefix_space": False,
            "trim_offsets": True, "use_regex": True,
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel", "add_prefix_space": False,
            "trim_offsets": True, "use_regex": True,
        },
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None,
            "continuing_subword_prefix": "", "end_of_word_suffix": "",
            "fuse_unk": False, "byte_fallback": False,
            "ignore_merges": True, "vocab": vocab, "merges": [],
        },
    }
    output = args.model_dir / "tokenizer.json"
    output.write_text(json.dumps(payload, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"wrote Edge fixture tokenizer: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
