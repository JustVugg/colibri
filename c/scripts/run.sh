#!/usr/bin/env bash
# Pipeline GLM-5.2 (int4, streaming, 15 GB RAM) — tutto in WSL, modello su ext4.
#   uso da c/:  scripts/run.sh ["prompt"] [n_token]
# Fa: (1) attende lo spostamento su ext4, (2) riprende la conversione fino a completarla,
#     (3) compila il motore, (4) genera testo restando nel budget RAM.
set -euo pipefail

DIR="${COLI_MODEL:-/home/vincenzo/glm52_i4}"          # modello int4 su ext4 (NON /mnt/c!)
REPO="${COLI_REPO:-zai-org/GLM-5.2-FP8}"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAM_GB="${RAM_GB:-15}"
PROMPT="${1:-Ciao, chi sei?}"
NGEN="${2:-64}"

cd "$CODE"

# 0) sanity: il modello deve stare su ext4, non su 9p/Windows
case "$DIR" in /mnt/*) echo "ERROR: $DIR is on /mnt (9p/Windows). Move it to ext4."; exit 1;; esac

# 1) se un rsync di spostamento e' ancora vivo, aspettalo
while pgrep -f "rsync.*glm52_i4" >/dev/null 2>&1; do
    echo "[1/4] waiting for the move to ext4... ($(du -sh "$DIR" 2>/dev/null | cut -f1))"; sleep 20
done
echo "[1/4] move complete: $(du -sh "$DIR" | cut -f1), shards $(ls "$DIR"/*.safetensors 2>/dev/null | wc -l)"

# 2) riprende+completa la conversione (ripartibile: salta gli shard gia' fatti)
echo "[2/4] conversion (resumes where it stopped): output -> $DIR"
python3 tools/convert_fp8_to_int4.py --repo "$REPO" --outdir "$DIR" --ebits 4 --io-bits 8

# 3) il motore richiede tokenizer.json + config.json nella dir del modello
for f in config.json tokenizer.json; do
    [ -f "$DIR/$f" ] || { echo "ERROR: missing $DIR/$f"; exit 1; }
done
echo "[3/4] building the engine"; make -s glm

# 4) generazione reale, con auto-cap dal budget RAM e heartbeat RSS su stderr
echo "[4/4] generating (RAM_GB=$RAM_GB, NGEN=$NGEN)"; echo "------"
SNAP="$DIR" RAM_GB="$RAM_GB" PROMPT="$PROMPT" NGEN="$NGEN" ./glm 64
