import os, sys
from huggingface_hub import hf_hub_download

DEST = r"D:\Colibri\models\Qwen3-Coder-30B-A3B"
os.makedirs(DEST, exist_ok=True)

# 1) Config + tokenizer from official repo (fast, few MB) for weight analysis
for f in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors.index.json"]:
    try:
        hf_hub_download(repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct", filename=f, local_dir=DEST)
        print(f"got {f}")
    except Exception as e:
        print(f"skip {f}: {e}")

# 2) GGUF Q4_K_M for local inference (~18GB, runs on 8GB VRAM + 32GB RAM via Ollama offload)
# NOTE: Qwen has no official GGUF repo. Community quant used: unsloth.
try:
    p = hf_hub_download(repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
                        filename="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", local_dir=DEST)
    print("GGUF Q4_K_M downloaded:", p)
except Exception as e:
    print(f"GGUF failed: {e}", file=sys.stderr)

print("Done. Files in", DEST)
