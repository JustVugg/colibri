#!/usr/bin/env python3
"""Load DeepSeek V4 through vLLM and report its actual MoE sharding."""

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.model_executor.layers.fused_moe import routed_experts
from vllm.model_executor.layers.fused_moe.experts import deep_gemm_moe


_init = routed_experts.RoutedExperts.__init__


def _probe_init(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    parallel = self.moe_config.moe_parallel_config
    shapes = {
        name: tuple(getattr(self, name).shape)
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        )
        if hasattr(self, name)
    }
    print(
        "COLIBRI_MOE_PROBE",
        self.layer_name,
        f"tp={parallel.tp_size}:{parallel.tp_rank}",
        f"ep={parallel.ep_size}:{parallel.ep_rank}",
        f"use_ep={parallel.use_ep}",
        f"local_experts={self.local_num_experts}",
        f"intermediate={self.intermediate_size_per_partition}",
        shapes,
        flush=True,
    )


routed_experts.RoutedExperts.__init__ = _probe_init


_permute = deep_gemm_moe.deepgemm_moe_permute
_permute_calls = 0


def _probe_permute(*args, **kwargs):
    global _permute_calls
    result = _permute(*args, **kwargs)
    if _permute_calls < 4:
        aq_out, aq_scale_out, expert_ids, inv_perm, align_used = result
        aq = kwargs.get("aq", args[0] if args else None)
        topk_ids = kwargs.get("topk_ids", args[2] if len(args) > 2 else None)
        print(
            "COLIBRI_PERMUTE_PROBE",
            f"rank={aq_out.device.index}",
            f"aq={tuple(aq.shape)}",
            f"topk={tuple(topk_ids.shape)}",
            f"aq_out={tuple(aq_out.shape)}",
            f"scale={tuple(aq_scale_out.shape)}",
            f"experts={tuple(expert_ids.shape)}",
            f"inv_perm={tuple(inv_perm.shape)}",
            f"alignment={align_used}",
            flush=True,
        )
    _permute_calls += 1
    return result


deep_gemm_moe.deepgemm_moe_permute = _probe_permute


def main():
    llm = LLM(
        model="/model",
        tensor_parallel_size=2,
        pipeline_parallel_size=3,
        max_model_len=128,
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.97,
        max_num_seqs=1,
        trust_remote_code=True,
    )
    output = llm.generate(
        [TokensPrompt(prompt_token_ids=[1, 2])],
        SamplingParams(max_tokens=1, temperature=0),
    )
    print("COLIBRI_TOKEN_PROBE", output[0].outputs[0].token_ids, flush=True)


if __name__ == "__main__":
    main()
