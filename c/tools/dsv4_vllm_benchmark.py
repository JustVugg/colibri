#!/usr/bin/env python3
import time
import os

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main() -> None:
    llm = LLM(
        model="/model",
        tensor_parallel_size=int(os.getenv("DSV4_TP", "2")),
        pipeline_parallel_size=int(os.getenv("DSV4_PP", "3")),
        max_model_len=128,
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.97,
        max_num_seqs=1,
        trust_remote_code=True,
    )
    prompt = TokensPrompt(prompt_token_ids=[1])
    llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0))
    repeats = int(os.getenv("DSV4_REPEATS", "1"))
    begin = time.perf_counter()
    made = 0
    for _ in range(repeats):
        output = llm.generate([prompt], SamplingParams(max_tokens=64, temperature=0))[0]
        made += len(output.outputs[0].token_ids)
    elapsed = time.perf_counter() - begin
    print(f"VLLM_BENCH tokens={made} elapsed={elapsed:.6f}s speed={made / elapsed:.4f} tok/s")
    print("VLLM_IDS", output.outputs[0].token_ids)


if __name__ == "__main__":
    main()
