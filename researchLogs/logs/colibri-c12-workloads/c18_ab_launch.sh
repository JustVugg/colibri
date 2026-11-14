#!/bin/bash
# C18 A/B launcher: alternates baseline (Q36_EXPERT_DEC1 unset) and decode-once
# (Q36_EXPERT_DEC1=1) runs of the SAME committed engine binary across fresh
# server restarts (drift rule: paired alternating runs, cache off so every
# request is a full prefill of the same tagged prompts).
# Resumable: each run is self-contained; re-running redoes all four, tee
# overwrites the per-run result files.
set -u
cd /Users/creemac/colibri/colibri/researchLogs/logs/colibri-c12-workloads
KEY='-4Jc6egKcHmI-1okTj4_MUWOU4etmppT8LLG-wxzUBk'
MODEL=/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ2_XXS.colibri

launch () {  # $1 = extra env assignments, $2 = log name
  pkill -f "colibri.cli serve" 2>/dev/null; pkill -f "c/qwen36" 2>/dev/null; sleep 3
  env $1 OMP_NUM_THREADS=6 COLI_PROMPT_CACHE=0 COLI_PREFIX_LOG=1 \
    PYTHONPATH=/Users/creemac/colibri/colibri \
    nohup /opt/homebrew/bin/python3 -m colibri.cli serve --model "$MODEL" \
    --host 0.0.0.0 --port 8001 --ctx 66000 --api-key="$KEY" > "$2" 2>&1 &
  for i in $(seq 1 30); do
    sleep 5
    if curl -s -m 3 -H "Authorization: Bearer $KEY" http://localhost:8001/v1/models | grep -q qwen3.6-colibri; then return 0; fi
  done
  echo "server $2 did not come up"; return 1
}

for r in 1 2; do
  echo "=== base r$r ==="
  launch "" "serve-c18-ab-base-r$r.log" || exit 1
  python3 c18_ab.py "$KEY" "base_r$r" 2>&1 | tee "phaseC18_ab_base_r$r.jsonl"
  echo "=== dec1 r$r ==="
  launch "Q36_EXPERT_DEC1=1" "serve-c18-ab-dec1-r$r.log" || exit 1
  python3 c18_ab.py "$KEY" "dec1_r$r" 2>&1 | tee "phaseC18_ab_dec1_r$r.jsonl"
done
echo "=== A/B complete ==="
