#!/usr/bin/env python3
"""Dependency-free contract checks for the generated GLM-5.3 tiny oracle."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def safetensors_header(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.fixture / "config.json").read_text())
    text = config["text_config"]
    assert config["model_type"] == "glm5_next"
    assert text["layer_types"] == ["linear_attention"] * 3 + ["deepseek_sparse_attention"]
    assert text["mlp_layer_types"] == ["dense"] * 3 + ["sparse"]
    assert text["linear_attn_config"]["kda_layers"] == [0, 1, 2]

    reference = json.loads((args.fixture / "ref.json").read_text())
    assert reference["source"] == "transformers"
    assert reference["transformers_version"] == "5.16.1"
    assert len(reference["prompt_ids"]) == len(reference["teacher_forcing_ids"])
    assert len(reference["last_logits_top_ids"]) == 8
    assert reference["cache_max_abs_delta"] <= 1e-5
    assert [state["kind"] for state in reference["cache_shapes"]] == [
        "kda", "kda", "kda", "dsa"
    ]

    header = safetensors_header(args.fixture / "model.safetensors")
    required = {
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.q_conv1d.weight",
        "model.language_model.layers.0.self_attn.k_conv1d.weight",
        "model.language_model.layers.0.self_attn.v_conv1d.weight",
        "model.language_model.layers.0.hc_attn_fn",
        "model.language_model.layers.3.self_attn.indexer.wk.weight",
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.3.mlp.experts.3.down_proj.weight",
        "model.language_model.layers.3.mlp.shared_experts.up_proj.weight",
        "lm_head.weight",
    }
    missing = required - header.keys()
    assert not missing, f"missing production-layout tensors: {sorted(missing)}"
    assert not any("experts.gate_up_proj" in name for name in header)
    assert not any("self_attn.conv1d.weight" in name for name in header)
    print("PASS GLM-5.3 tiny oracle contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
