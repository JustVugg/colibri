"""Generate reference token IDs for the real OLMoE-1B-7B model.

Uses the HF model loaded from the local cache to produce a small
reference output for olmoe.exe validation. Saves to ref_olmoe_real.json.

Usage: python tools/make_olmoe_real_oracle.py
"""
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

try:
    import torch
    from transformers import AutoTokenizer, OlmoeForCausalLM
except ImportError as exc:
    sys.exit(f"Missing deps: {exc}. Run: pip install torch transformers")

MODEL_ID = "allenai/OLMoE-1B-7B-0125-Instruct"

OUT_JSON = Path(__file__).resolve().parent.parent / "ref_olmoe_real.json"

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 12

print(f"Loading tokenizer from {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Encoding prompt ...")
enc = tokenizer(PROMPT, return_tensors="pt")
prompt_ids = enc["input_ids"][0].tolist()
print(f"  Prompt IDs ({len(prompt_ids)}): {prompt_ids}")

print(f"Loading OLMoE model from {MODEL_ID} ...")
print("  (this will use ~14 GB RAM — please be patient)")
model = OlmoeForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
model.eval()
print("  Model loaded!")

print(f"Generating {MAX_NEW_TOKENS} tokens ...")
with torch.no_grad():
    out = model.generate(
        enc["input_ids"],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
    )

full_ids = out[0].tolist()
gen_ids = full_ids[len(prompt_ids):]

print(f"Prompt IDs  : {prompt_ids}")
print(f"Full IDs    : {full_ids}")
print(f"Generated   : {gen_ids}")
print(f"Text        : {tokenizer.decode(gen_ids, skip_special_tokens=True)!r}")

payload = {"prompt_ids": prompt_ids, "full_ids": full_ids}
OUT_JSON.write_text(json.dumps(payload, indent=2))
print(f"\nSaved reference to {OUT_JSON}")
