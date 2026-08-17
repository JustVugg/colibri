#!/usr/bin/env python3
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main() -> None:
    llm = LLM(
        model="/model",
        tensor_parallel_size=2,
        pipeline_parallel_size=3,
        max_model_len=64,
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
        enforce_eager=True,
    )
    output = llm.generate(
        [TokensPrompt(prompt_token_ids=[1, 2])],
        SamplingParams(max_tokens=1, temperature=0),
    )
    print("ORACLE", output[0].outputs[0].token_ids)


if __name__ == "__main__":
    main()
