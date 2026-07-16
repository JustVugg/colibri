"""Build a deterministic, medium-size GLM-MoE fixture for backend benchmarks.

This is not a useful language model. It preserves the real glm_moe_dsa data
flow while remaining small enough to generate locally and run repeated CPU/CUDA
A/B tests without downloading the 379 GB checkpoint.

With --fp8 the weights are written as FP8 e4m3 + 128x128 block scale_inv, in the
SAME layout as the real GLM-5.2-FP8 checkpoint, so convert_fp8_to_int4.py can
exercise its FP8->int4 dequant path on a local fixture (its dims are 128-friendly,
so this is also the right fixture for --group-size 128 testing):

  python tools/make_glm_bench_model.py --fp8 --output glm_bench_fp8
  python tools/convert_fp8_to_int4.py --indir glm_bench_fp8 --outdir glm_bench_i4 --ebits 4 --group-size 128
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))   # importa glm_fp8_emit se lanciato da c/
from glm_fp8_emit import save_fp8_safetensors, unfuse_experts


def build_config() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        vocab_size=8192,
        hidden_size=1024,
        intermediate_size=2048,
        moe_intermediate_size=512,
        num_hidden_layers=8,
        first_k_dense_replace=3,
        num_attention_heads=16,
        num_key_value_heads=16,
        n_routed_experts=32,
        num_experts_per_tok=8,
        n_shared_experts=1,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        index_topk=4096,
        index_head_dim=32,
        index_n_heads=4,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="glm_bench_medium")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fp8", action="store_true",
                        help="write weights as FP8 e4m3 + 128x128 block scale_inv (same layout as "
                             "GLM-5.2-FP8) instead of bf16, so convert_fp8_to_int4.py can dequant+requant")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = build_config()
    cfg._attn_implementation = "eager"
    model = GlmMoeDsaForCausalLM(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() >= 2:
                param.normal_(0, 0.02)
        for layer in model.model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.e_score_correction_bias.copy_(
                    torch.linspace(-0.1, 0.1, cfg.n_routed_experts)
                )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    params = sum(p.numel() for p in model.parameters())

    model.to(args.device)
    prompt = [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]
    ids = torch.tensor([prompt], device=args.device)
    with torch.inference_mode():
        full = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True)[0]
        logits = model(full.unsqueeze(0), use_cache=False).logits[0]

    # Unfuse experts AFTER reference generation (model needs fused weights for
    # forward/generate) but BEFORE saving — the real checkpoint and the converter
    # + C engine all expect per-expert 2-D gate_proj/up_proj/down_proj tensors.
    sd = model.state_dict()
    unfuse_experts(sd)

    if args.fp8:
        n_fp8, n_tot = save_fp8_safetensors(sd, output / "model.safetensors")
        # save_pretrained scrive config.json; nel path FP8 lo bypassiamo, quindi lo scriviamo
        # a mano (serve al converter e al motore C). EN: save_pretrained writes config.json;
        # the FP8 path bypasses it, so write it manually (converter + C engine need it).
        (output / "config.json").write_text(json.dumps(cfg.to_dict()))
        print(f"saved FP8: {n_fp8} e4m3 tensors (+{n_tot - n_fp8} scale_inv sidecars / f32) "
              f"-> {output / 'model.safetensors'}")
    else:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in sd.items()}, str(output / "model.safetensors"))
        (output / "config.json").write_text(json.dumps(cfg.to_dict()))

    ref = {
        "prompt_ids": prompt,
        "full_ids": full.cpu().tolist(),
        "tf_pred": logits.argmax(-1).cpu().tolist(),
    }
    (output / "ref_glm.json").write_text(json.dumps(ref))
    manifest = {
        "seed": args.seed,
        "parameters": params,
        "parameters_billions": round(params / 1e9, 4),
        "format": "fp8-e4m3-128" if args.fp8 else "bf16",
        "purpose": "backend benchmark fixture; random weights, not a language model",
    }
    (output / "bench_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
