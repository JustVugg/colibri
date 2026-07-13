#!/usr/bin/env bash
# Supervisore della conversione GLM-5.2 — a prova di rete WSL che si blocca.
#  - tiene SEMPRE vivo un (solo) convertitore
#  - se un download resta FERMO >180s (connessione zombie), lo ammazza e lo rilancia:
#    hf_hub riprende il .incomplete dal punto esatto, non si perde nulla
#  - esce da solo quando tutti i 141 shard sono fatti
# uso da c/:  nohup scripts/supervisor.sh > supervisor.log 2>&1 &
set -u
DIR="${COLI_MODEL:-/home/vincenzo/glm52_i4}"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOTAL="${TOTAL_SHARDS:-141}"
STALL_S=180          # secondi senza crescita del download -> riavvio
CONVLOG=/tmp/convert_supervised.log

exec 9>"$DIR/.supervisor.lock"
flock -n 9 || { echo "a supervisor is already running; exiting"; exit 1; }

log(){ echo "[$(date +%H:%M:%S)] $*"; }

start_conv(){
    cd "$CODE"
    nohup python3 tools/convert_fp8_to_int4.py --repo zai-org/GLM-5.2-FP8 \
        --outdir "$DIR" --ebits 4 --io-bits 8 >> "$CONVLOG" 2>&1 &
    log "converter started (PID $!)"
}

last_size=-1; stall=0
while :; do
    done_n=$(ls "$DIR"/out-*.safetensors 2>/dev/null | wc -l)
    if [ "$done_n" -ge "$TOTAL" ]; then log "DONE: $done_n/$TOTAL shards. Exiting."; pkill -f convert_fp8 2>/dev/null; exit 0; fi

    if ! pgrep -f convert_fp8 >/dev/null; then
        log "converter is not running ($done_n/$TOTAL): starting it"
        start_conv; last_size=-1; stall=0; sleep 20; continue
    fi

    inc=$(find "$DIR/_inflight" -name "*.incomplete" 2>/dev/null | head -1)
    if [ -n "$inc" ]; then
        size=$(stat -c%s "$inc" 2>/dev/null || echo 0)
        if [ "$size" = "$last_size" ]; then
            stall=$((stall+30))
            if [ "$stall" -ge "$STALL_S" ]; then
                log "download stalled for ${stall}s at $((size/1000000)) MB ($done_n/$TOTAL): restarting the converter"
                pkill -f convert_fp8; sleep 5
                start_conv; last_size=-1; stall=0
            fi
        else
            [ "$last_size" -ge 0 ] && [ "$stall" -ge 60 ] && log "download resumed ($((size/1000000)) MB)"
            last_size=$size; stall=0
        fi
    else
        last_size=-1; stall=0     # niente .incomplete = sta convertendo/salvando: tutto ok
    fi
    sleep 30
done
