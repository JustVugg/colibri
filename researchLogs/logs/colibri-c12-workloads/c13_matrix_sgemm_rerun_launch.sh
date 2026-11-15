#!/usr/bin/env bash
# C15 adoption follow-up: re-run the C13 matrix cold cells on the enabled path.
# The default build now ships Q36_ATTN_SGEMM enabled (no env needed); the two
# arms differ only in the speculation toggle, mirroring the 2026-11-12 matrix:
#   speon  - COLI_PROMPT_CACHE=0
#   speoff - COLI_PROMPT_CACHE=0 COLI_LOOKUP=0
# Each arm is a fresh restart; results land in NEW files (the original
# phaseM_cold_*_results.jsonl stay untouched as the scalar-path baselines).
set -u
cd /Users/creemac/colibri/colibri || exit 1
W=researchLogs/logs/colibri-c12-workloads
KEY='-4Jc6egKcHmI-1okTj4_MUWOU4etmppT8LLG-wxzUBk'
MODEL=/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ2_XXS.colibri

launch() { # $1 = log name, $2.. = extra env
  local log="$1"; shift
  pkill -f "colibri.cli serve" 2>/dev/null; pkill -f "c/qwen36" 2>/dev/null; sleep 3
  env OMP_NUM_THREADS=6 COLI_PREFIX_LOG=1 COLI_PCACHE_LOG=1 "$@" \
    PYTHONPATH=/Users/creemac/colibri/colibri \
    nohup /opt/homebrew/bin/python3 -m colibri.cli serve \
    --model "$MODEL" --host 0.0.0.0 --port 8001 --ctx 66000 \
    --api-key="$KEY" > "$W/$log" 2>&1 &
  sleep 20
}

launch serve-matrix-sgemm-speon.log COLI_PROMPT_CACHE=0
echo "=== speon colds (sgemm default build) ==="
/opt/homebrew/bin/python3 "$W/c13_matrix.py" "$KEY" cold 2>&1 | tee "$W/phaseM_cold_speon_results_sgemm.jsonl"

launch serve-matrix-sgemm-speoff.log COLI_PROMPT_CACHE=0 COLI_LOOKUP=0
echo "=== speoff colds (sgemm default build) ==="
/opt/homebrew/bin/python3 "$W/c13_matrix.py" "$KEY" cold 2>&1 | tee "$W/phaseM_cold_speoff_results_sgemm.jsonl"

echo "=== rerun complete ==="
