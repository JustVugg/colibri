#!/usr/bin/env bash
# C20 validation launcher: the disk prompt cache under KV_SLOTS=2, interleaved
# A/B per the C20 scoping doc. Two arms, each = round 1 (one process) ->
# restart -> round 2 + corruption (fresh process):
#   cacheoff - COLI_PROMPT_CACHE=0 (every request a full cold prefill)
#   cacheon  - cache on, its own COLI_PROMPT_CACHE_DIR (the user's real
#              snapshot store beside the model container is not touched)
set -u
cd /Users/creemac/colibri/colibri || exit 1
W=researchLogs/logs/colibri-c12-workloads
KEY='-4Jc6egKcHmI-1okTj4_MUWOU4etmppT8LLG-wxzUBk'
MODEL=/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ2_XXS.colibri
PCDIR=$W/c20_pcache

launch() { # $1 = log name, $2.. = extra env
  local log="$1"; shift
  pkill -f "colibri.cli serve" 2>/dev/null; pkill -f "c/qwen36" 2>/dev/null; sleep 3
  env OMP_NUM_THREADS=6 COLI_PREFIX_LOG=1 COLI_PCACHE_LOG=1 COLI_KV_SLOTS=2 \
    COLI_PROMPT_CACHE_DIR=$PCDIR "$@" \
    PYTHONPATH=/Users/creemac/colibri/colibri \
    nohup /opt/homebrew/bin/python3 -m colibri.cli serve \
    --model "$MODEL" --host 0.0.0.0 --port 8001 --ctx 66000 \
    --api-key="$KEY" > "$W/$log" 2>&1 &
  sleep 20
}

rm -rf "$PCDIR"; mkdir -p "$PCDIR"
rm -f "$W"/phaseC20_mux_*_results.jsonl

launch serve-c20-cacheoff.log COLI_PROMPT_CACHE=0
echo "=== cacheoff round1 ==="
/opt/homebrew/bin/python3 "$W/c20_mux_ab.py" "$KEY" cacheoff 2>&1 | tee "$W/phaseC20_mux_cacheoff_results.jsonl"
echo "=== cacheoff round2 (fresh process) ==="
/opt/homebrew/bin/python3 "$W/c20_mux_ab.py" "$KEY" cacheoff round2 2>&1 | tee -a "$W/phaseC20_mux_cacheoff_results.jsonl"

launch serve-c20-cacheon.log
echo "=== cacheon round1 ==="
/opt/homebrew/bin/python3 "$W/c20_mux_ab.py" "$KEY" cacheon 2>&1 | tee "$W/phaseC20_mux_cacheon_results.jsonl"
echo "=== cacheon round2 (fresh process) ==="
/opt/homebrew/bin/python3 "$W/c20_mux_ab.py" "$KEY" cacheon round2 2>&1 | tee -a "$W/phaseC20_mux_cacheon_results.jsonl"

echo "=== C20 validation complete ==="
