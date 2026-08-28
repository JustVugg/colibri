#!/usr/bin/env python3
"""Build a deterministic tiny HF OLMoE checkpoint for Colibri tests.

The result is a source checkpoint.  Convert it with
``tools/convert_olmoe_merged.py`` before running the C engine.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import OlmoeConfig, OlmoeForCausalLM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"output exists (use --force): {output}")
        shutil.rmtree(output)
    torch.manual_seed(20260825)
    torch.set_num_threads(1)
    config = OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        eos_token_id=None,
        pad_token_id=None,
        tie_word_embeddings=False,
    )
    model = OlmoeForCausalLM(config).eval().float()
    prompt = [3, 11, 29, 7, 41, 19]
    with torch.no_grad():
        generated = model.generate(
            torch.tensor([prompt]), max_new_tokens=8, do_sample=False,
            use_cache=True,
        )[0].tolist()
    output.mkdir(parents=True)
    model.save_pretrained(output, safe_serialization=True)
    (output / "ref_olmoe.json").write_text(
        json.dumps({"prompt_ids": prompt, "full_ids": generated}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote tiny OLMoE source checkpoint: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
