import os, sys, time
print("python", sys.version)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())

from airllm import AutoModel
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
TOKENIZER = MODEL

t0 = time.time()
print("loading model", MODEL, flush=True)
model = AutoModel.from_pretrained(MODEL, device="cpu", dtype=torch.bfloat16)
print("model loaded in %.1fs" % (time.time() - t0), flush=True)

tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

prompt = "You are a helpful coding assistant. Write a short Python function that sums even numbers in a list.\n\ndef sum_even(lst):"
ids = tok(prompt, return_tensors="pt")
print("prompt tokens:", ids["input_ids"].shape)

t0 = time.time()
with torch.inference_mode():
    out_ids = model.generate(
        ids["input_ids"],
        max_new_tokens=120,
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
        eos_token_id=tok.eos_token_id,
    )
print("generate took %.1fs" % (time.time() - t0))
print("TEXT OUTPUT:\n" + tok.decode(out_ids[0], skip_special_tokens=True))
