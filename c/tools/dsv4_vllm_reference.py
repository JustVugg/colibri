#!/usr/bin/env python3
"""Pinned DeepSeek-V4 reference path used before native Colibri porting.

This intentionally delegates execution to the upstream vLLM implementation.
Do not add replacement kernels here: its purpose is to preserve the exact
known-good TP2/PP3 behavior and provide an oracle for the native backend.
"""

import argparse
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/model")
    parser.add_argument("--prompt-token", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=2,
        pipeline_parallel_size=3,
        max_model_len=128,
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.97,
        max_num_seqs=1,
        trust_remote_code=True,
    )
    prompt = TokensPrompt(prompt_token_ids=[args.prompt_token])
    params = SamplingParams(max_tokens=args.tokens, temperature=0)

    llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0))
    started = time.perf_counter()
    made = 0
    output = None
    for _ in range(args.repeats):
        output = llm.generate([prompt], params)[0]
        made += len(output.outputs[0].token_ids)
    elapsed = time.perf_counter() - started

    assert output is not None
    print(f"VLLM_REFERENCE tokens={made} elapsed={elapsed:.6f}s speed={made / elapsed:.4f} tok/s")
    print("VLLM_REFERENCE_IDS", output.outputs[0].token_ids)


if __name__ == "__main__":
    main()
