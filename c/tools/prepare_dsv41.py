#!/usr/bin/env python3
"""Write the engram sidecar a DeepSeek-V4.1-Flash container needs, once.

The engine cannot derive these three tables at load time and should not try:

  token_map    every token id mapped onto a smaller space where ids that normalize
               alike collapse together, so " The", "the" and "THE" hash the same.
               Needs the checkpoint's own tokenizer and its normalizer chain.
  primes       each (n-gram size, head) pair owns a disjoint prime-sized bucket range
               in its layer's table. Needs a primality test.
  multipliers  one odd multiplier per (layer, lookback), drawn from numpy's PCG64
               seeded with 10007 * layer_id -- reproducing that generator in C would
               be a liability, and it never changes for a given checkpoint.

All three come straight from the vendor's inference/engram.py; this script is that
code with its results written to disk instead of held in memory.

    python3 tools/prepare_dsv41.py --model ~/Models/DeepSeek-V4.1-Flash

The sidecar lands beside config.json as dsv41_engram.json (about 1 MB for the released
vocabulary) and the container is then complete: nothing else about this model needs
converting, because its dense weights already ship as fp8 and its experts as fp4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_compressed_token_map(tokenizer):
    """engram.py build_compressed_token_map, unchanged in substance.

    The returned size is not a detail: every hash multiplier is derived from it, so a
    map built with a different normalizer silently rehashes the whole table.
    """
    from tokenizers import Regex, normalizers

    sentinel = ""   # a private-use char, so a single-space token survives Strip()
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer if hasattr(tokenizer, "backend_tokenizer") else tokenizer
    size = backend.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = [0] * size
    for token_id in range(size):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)     # a partial UTF-8 byte token
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def layout(cfg: dict, compressed_vocab: int) -> dict:
    """engram.py EngramLayout.from_args + compute_hash_multipliers."""
    import numpy as np
    from sympy import isprime

    layer_ids = list(cfg["engram_layer_ids"])
    max_ngram = int(cfg["engram_max_ngram_size"])
    heads = int(cfg["engram_n_heads"])
    seen: set[int] = set()
    primes, offsets, rows = [], [], []
    for _ in layer_ids:
        per_layer, flat = [], []
        for _ in range(max_ngram - 1):
            row, candidate = [], int(cfg["engram_vocab_size"]) - 1
            for _ in range(heads):
                candidate += 1
                while not isprime(candidate) or candidate in seen:
                    candidate += 1
                seen.add(candidate)
                row.append(candidate)
            per_layer.append(row)
            flat.extend(row)
        primes.append(per_layer)
        offsets.append([int(v) for v in np.cumsum([0, *flat[:-1]])])
        rows.append(int(sum(flat)))
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab) // 2)
    multipliers = []
    for layer_id in layer_ids:
        rng = np.random.default_rng(10007 * layer_id)
        values = rng.integers(low=0, high=bound, size=(max_ngram,), dtype=np.int64)
        multipliers.append([str(int(v) * 2 + 1) for v in values])
    return dict(primes=primes, offsets=offsets, multipliers=multipliers, num_embeddings=rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, required=True, help="the container directory")
    ap.add_argument("--out", type=Path, default=None, help="defaults to <model>/dsv41_engram.json")
    args = ap.parse_args()

    config = json.loads((args.model / "config.json").read_text())
    cfg = config.get("text_config", config)
    if not cfg.get("engram_layer_ids"):
        print("this checkpoint declares no engram layers: nothing to prepare")
        return 0

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    token_map, compressed = build_compressed_token_map(tokenizer)
    declared = int(cfg.get("engram_compressed_vocab_size", compressed))
    if compressed != declared:
        raise SystemExit(
            f"the normalized vocabulary came out at {compressed} ids, the config declares "
            f"{declared}. Every hash multiplier is derived from that number, so the tables "
            f"this would write hash into different rows than the model was trained on. "
            f"Check the tokenizer version rather than overriding this.")

    tables = layout(cfg, compressed)
    for i, rows in enumerate(tables["num_embeddings"]):
        declared_rows = int(cfg["engram_num_embeddings"][i])
        if rows != declared_rows:
            raise SystemExit(f"table {i}: primes sum to {rows} rows, the config declares "
                             f"{declared_rows}")
    out = args.out or (args.model / "dsv41_engram.json")
    out.write_text(json.dumps({
        "max_ngram_size": int(cfg["engram_max_ngram_size"]),
        "n_heads": int(cfg["engram_n_heads"]),
        "head_dim": int(cfg["engram_head_dim"]),
        "pad_id": token_map[int(cfg.get("engram_pad_token_id", cfg.get("engram_pad_id", 2)))],
        "layer_ids": list(cfg["engram_layer_ids"]),
        "num_embeddings": tables["num_embeddings"],
        "primes": tables["primes"],
        "offsets": tables["offsets"],
        "multipliers": tables["multipliers"],
        "token_map": token_map,
    }))
    print(f"{out}  ({out.stat().st_size/1e6:.2f} MB)")
    print(f"  {len(cfg['engram_layer_ids'])} tables, {compressed} compressed ids, "
          f"{tables['num_embeddings']} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
