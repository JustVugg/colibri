import sys, os
sys.path.insert(0, "C:\\Users\\Liso1\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages")

from pathlib import Path
from huggingface_hub import hf_hub_download

DEST = Path("D:/Colibri/models/Qwen3-Coder-30B-A3B")
DEST.mkdir(parents=True, exist_ok=True)

# Official repo only has safetensors shards, no GGUF. Start with one shard to
# test the weight-space path cheaply. ~3.9 GB each, 16 shards total (~61 GB).
SHARDS_TO_DOWNLOAD = 1  # set higher once you know you want the full set

print("Repo: Qwen/Qwen3-Coder-30B-A3B-Instruct")
print("Dest:", DEST)
print("Downloading", SHARDS_TO_DOWNLOAD, "shard(s) first (16 total).")
print("Tip: full set is ~61 GB; one shard is enough to test the weight-space analyze/patch path.")

for i in range(1, SHARDS_TO_DOWNLOAD + 1):
    fname = f"model-{i:05d}-of-00016.safetensors"
    try:
        p = hf_hub_download(
            repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            filename=fname,
            local_dir=str(DEST),
            local_dir_use_symlinks=False,
        )
        print("OK:", fname, "->", p)
    except Exception as e:
        print("FAIL:", fname, repr(e)[:300])

print("Done downloading shards.")
if not list(DEST.glob("model-*-of-*.safetensors")):
    print("No shards present. Weight-space analyze/patch will skip until at least one shard is here.")
