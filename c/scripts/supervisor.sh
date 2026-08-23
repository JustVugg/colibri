#!/usr/bin/env bash
# Supervisore della conversione GLM-5.2 — a prova di rete WSL che si blocca.
#  - tiene SEMPRE vivo un (solo) convertitore
#  - se un download resta FERMO >180s (connessione zombie), lo ammazza e lo rilancia:
#    hf_hub riprende il .incomplete dal punto esatto, non si perde nulla
#  - esce da solo quando tutti i 141 shard sono fatti
# uso da c/:  nohup scripts/supervisor.sh > supervisor.log 2>&1 &
set -u
DIR="${COLI_MODEL:?set COLI_MODEL to the model directory}"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOTAL="${TOTAL_SHARDS:-141}"
STALL_S=180          # secondi senza crescita del download -> riavvio
CONVLOG=/tmp/convert_supervised.log

exec 9>"$DIR/.supervisor.lock"
flock -n 9 || { echo "a supervisor is already running; exiting"; exit 1; }

log(){ echo "[$(date +%H:%M:%S)] $*"; }

start_conv(){
    cd "$CODE" || { log "cannot enter $CODE"; return 1; }
    nohup python3 tools/convert_fp8_to_int4.py --repo zai-org/GLM-5.2-FP8 \
        --outdir "$DIR" --ebits 4 --io-bits 8 >> "$CONVLOG" 2>&1 &
    conv_pid=$!
    log "converter started (PID $!)"
}

# Match only a converter that owns this supervisor's output directory. A global
# pgrep/pkill also sees conversions for other models and can stop all of them
# when this one finishes or stalls.
find_conv_pid(){
    local cmdline pid arg i saw_script saw_outdir
    local -a argv
    for cmdline in /proc/[0-9]*/cmdline; do
        pid=${cmdline#/proc/}; pid=${pid%/cmdline}
        [ "$pid" = "$$" ] && continue
        [ -r "$cmdline" ] || continue
        argv=()
        while IFS= read -r -d '' arg; do argv+=("$arg"); done < "$cmdline"
        saw_script=0; saw_outdir=0
        for ((i=0; i<${#argv[@]}; i++)); do
            case "${argv[i]}" in */convert_fp8_to_int4.py|convert_fp8_to_int4.py) saw_script=1;; esac
            if [ "${argv[i]}" = "--outdir" ] && [ "${argv[i+1]:-}" = "$DIR" ]; then
                saw_outdir=1
            fi
        done
        if [ "$saw_script" -eq 1 ] && [ "$saw_outdir" -eq 1 ]; then
            echo "$pid"
            return
        fi
    done
}

last_size=-1; stall=0
while :; do
    done_n=$(ls "$DIR"/out-*.safetensors 2>/dev/null | wc -l)
    conv_pid=$(find_conv_pid)
    if [ "$done_n" -ge "$TOTAL" ]; then
        log "DONE: $done_n/$TOTAL shards. Exiting."
        [ -z "$conv_pid" ] || kill "$conv_pid" 2>/dev/null
        exit 0
    fi

    if [ -z "$conv_pid" ]; then
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
                kill "$conv_pid" 2>/dev/null; sleep 5
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
