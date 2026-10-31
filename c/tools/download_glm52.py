"""
Download dei pesi reali di GLM-5.2 per il motore C — STADIO B.

Target: zai-org/GLM-5.2-FP8  (FP8 e4m3, 141 shard, ~756 GB) -> ENTRA nei 926 GB di ext4.
(La variante bf16 zai-org/GLM-5.2 e' 1.5 TB e NON entra.)

Il motore C leggera' questi safetensors in streaming e li (ri)quantizzera' a int4/int8.
NB: i pesi sono F8_E4M3 + tensori `*.weight_scale_inv` (blocchi 128x128). Il loader st.h
deve supportare fp8+block-scale prima di poterli usare (vedi memoria glm52-specs).

USO:
    GLM_DIR=/path/to/glm52 python3 tools/download_glm52.py     # scarica tutto li' (ripartibile)
    python3 tools/download_glm52.py --check    # solo stima spazio e conteggio file, niente download

Lo scaricamento e' di centinaia di GB e ore: lancialo tu quando il resto e' pronto.
"""
import os, sys, shutil
from huggingface_hub import snapshot_download, HfApi

REPO = "zai-org/GLM-5.2-FP8"
# Pin the model revision for supply-chain integrity: set GLM_REVISION to a commit SHA
# so a mutated/compromised upstream can't silently swap the weights you fetch.
REVISION = os.environ.get("GLM_REVISION", "main")
# No default: this writes ~400 GB, so it must land where the user chose and never on a
# path inherited from whoever wrote the script (#724). Put it on a native filesystem —
# on WSL that means NOT /mnt/c.
DEST = os.environ.get("GLM_DIR")

def human(n): return f"{n/1e9:.0f} GB"

def check():
    info = HfApi().repo_info(REPO, revision=REVISION, files_metadata=True)
    tot = sum((s.size or 0) for s in info.siblings)
    sts = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    # A check must not create the destination. Probe its filesystem, or the
    # nearest existing ancestor when download() will create nested directories.
    disk_path = os.path.abspath(DEST)
    while not os.path.exists(disk_path):
        parent = os.path.dirname(disk_path)
        if parent == disk_path:
            break
        disk_path = parent
    free = shutil.disk_usage(disk_path).free
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
        revision=REVISION,
        local_dir=DEST,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        max_workers=8,
    )
    print("DONE. Weights saved in:", DEST)

if __name__ == "__main__":
    if not DEST:
        sys.exit("set GLM_DIR to the destination directory, e.g.\n"
                 "  GLM_DIR=/data/glm52 python3 tools/download_glm52.py")
    if "--check" in sys.argv:
        check()
    else:
        check(); print("---"); download()
