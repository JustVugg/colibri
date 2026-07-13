"""
Download dei pesi reali di GLM-5.2 per il motore C — STADIO B.

Target: zai-org/GLM-5.2-FP8  (FP8 e4m3, 141 shard, ~756 GB) -> ENTRA nei 926 GB di ext4.
(La variante bf16 zai-org/GLM-5.2 e' 1.5 TB e NON entra.)

Il motore C leggera' questi safetensors in streaming e li (ri)quantizzera' a int4/int8.
NB: i pesi sono F8_E4M3 + tensori `*.weight_scale_inv` (blocchi 128x128). Il loader st.h
deve supportare fp8+block-scale prima di poterli usare (vedi memoria glm52-specs).

USO:
    python3 tools/download_glm52.py            # scarica tutto in /home/vincenzo/glm52  (ripartibile)
    python3 tools/download_glm52.py --check    # solo stima spazio e conteggio file, niente download

Lo scaricamento e' di centinaia di GB e ore: lancialo tu quando il resto e' pronto.
"""
import os, sys, shutil
from huggingface_hub import snapshot_download, HfApi

REPO = "zai-org/GLM-5.2-FP8"
DEST = os.environ.get("GLM_DIR", "/home/vincenzo/glm52")   # su ext4 (/dev/sdd), MAI su /mnt/c

def human(n): return f"{n/1e9:.0f} GB"

def check():
    info = HfApi().repo_info(REPO, files_metadata=True)
    tot = sum((s.size or 0) for s in info.siblings)
    sts = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    free = shutil.disk_usage(os.path.dirname(DEST) or "/").free
    print(f"repo: {REPO}")
    print(f"  total files: {len(info.siblings)} ({len(sts)} safetensors shards)")
    print(f"  total size: {human(tot)}")
    print(f"  free space in {DEST}: {human(free)}")
    print(f"  {'OK: enough space' if free > tot*1.05 else 'WARNING: not enough space'}")

def download():
    os.makedirs(DEST, exist_ok=True)
    free = shutil.disk_usage(DEST).free
    print(f"Downloading {REPO} -> {DEST}  (free: {human(free)})")
    # resume_download e' implicito; in caso di interruzione, rilancia e riprende.
    snapshot_download(
        repo_id=REPO,
        local_dir=DEST,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        max_workers=8,
    )
    print("DONE. Weights saved in:", DEST)

if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        check(); print("---"); download()
