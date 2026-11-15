#!/bin/bash
# C15 adoption sweep launcher: the pre-default-flip agreement sweep.
# 4 alternating fresh restarts of the SAME committed binary (base =
# Q36_ATTN_SGEMM unset, sgemm = =1), COLI_PROMPT_CACHE=0. Per round:
#   1. agent-ext full workload (c13_agent_ext.py: large stable system prompt +
#      tools + tool-result turns, engine-output replayed — the real-use shape)
#   2. 16K + 24K cold prompts (c15_sweep_colds.py — the long-prompt agreement
#      + timing arm, 2-4x more scored tokens than the C15 A/B covered)
# Correctness gate: sha AGREEMENT across base/sgemm for the same turn
# (tolerance contract), plus paired timing deltas. Resumable per round:
# each round's output file is self-contained.
set -u
cd /Users/creemac/colibri/colibri/researchLogs/logs/colibri-c12-workloads
KEY='-4Jc6egKcHmI-1okTj4_MUWOU4etmppT8LLG-wxzUBk'
MODEL=/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ2_XXS.colibri

launch () {  # $1 = sgemm env value ("" for base or "1"), $2 = log name
  pkill -f "colibri.cli serve" 2>/dev/null; pkill -f "c/qwen36" 2>/dev/null; sleep 3
  if [ -n "$1" ]; then SG="Q36_ATTN_SGEMM=$1"; else SG=""; fi
  env OMP_NUM_THREADS=6 COLI_PROMPT_CACHE=0 $SG \
    PYTHONPATH=/Users/creemac/colibri/colibri \
    nohup /opt/homebrew/bin/python3 -m colibri.cli serve --model "$MODEL" \
    --host 0.0.0.0 --port 8001 --ctx 66000 --api-key="$KEY" > "$2" 2>&1 &
  for i in $(seq 1 30); do
    sleep 5
    if curl -s -m 3 -H "Authorization: Bearer $KEY" http://localhost:8001/v1/models | grep -q qwen3.6-colibri; then return 0; fi
  done
  echo "server $2 did not come up"; return 1
}

run_round () {  # $1 = kind (base|sgemm), $2 = round (1|2)
  local kind=$1 r=$2
  echo "=== $kind r$r: agent-ext ==="
  if [ "$kind" = base ]; then launch "" "serve-c15-sweep-base-r$r.log" || return 1
  else launch 1 "serve-c15-sweep-sgemm-r$r.log" || return 1; fi
  /opt/homebrew/bin/python3 c13_agent_ext.py "$KEY" full 2>&1 | tee "phaseC15_sweep_${kind}_r${r}.jsonl"
  echo "=== $kind r$r: 16K/24K colds ==="
  /opt/homebrew/bin/python3 c15_sweep_colds.py "$KEY" "${kind}_r${r}" 2>&1 | tee -a "phaseC15_sweep_${kind}_r${r}.jsonl"
}

run_round base 1    || exit 1
run_round sgemm 1   || exit 1
run_round base 2    || exit 1
run_round sgemm 2   || exit 1
echo "=== sweep complete ==="
