#!/bin/bash
# C14 pressure-arm launcher: interleaved arms of the 4-turn agent-session
# workload across fresh server restarts of the SAME committed binary.
#   idle0  - no hog, Q36_HYBRID_GB=0 (default)
#   idle2  - no hog, Q36_HYBRID_GB=2  (hybrid hot-expert copies on)
#   press0 - hog running, Q36_HYBRID_GB=0
#   press2 - hog running, Q36_HYBRID_GB=2
# Drift rule: the same arm order runs in both rounds (r1 then r2), so any
# machine-degradation trend hits each arm pair identically. Decode tok/s per
# turn is the metric; vm_stat/swap snapshots bracket every run.
set -u
cd /Users/creemac/colibri/colibri/researchLogs/logs/colibri-c12-workloads
KEY='-4Jc6egKcHmI-1okTj4_MUWOU4etmppT8LLG-wxzUBk'
MODEL=/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ2_XXS.colibri
HOG_GB="${HOG_GB:-5}"

# build the hog once
clang -O2 c14_hog.c -o /tmp/c14_hog || exit 1

launch () {  # $1 = extra env, $2 = log name
  pkill -f "colibri.cli serve" 2>/dev/null; pkill -f "c/qwen36" 2>/dev/null; pkill -f c14_hog 2>/dev/null; sleep 3
  env $1 OMP_NUM_THREADS=6 COLI_PREFIX_LOG=1 COLI_PCACHE_LOG=1 \
    PYTHONPATH=/Users/creemac/colibri/colibri \
    nohup /opt/homebrew/bin/python3 -m colibri.cli serve --model "$MODEL" \
    --host 0.0.0.0 --port 8001 --ctx 66000 --api-key="$KEY" > "$2" 2>&1 &
  for i in $(seq 1 30); do
    sleep 5
    if curl -s -m 3 -H "Authorization: Bearer $KEY" http://localhost:8001/v1/models | grep -q qwen3.6-colibri; then return 0; fi
  done
  echo "server $2 did not come up"; return 1
}

snapshot () {  # $1 = label: memory-pressure snapshot into the run's log
  { echo "== $1 $(date +%H:%M:%S)"; vm_stat | sed -n '1,12p'; sysctl vm.swapusage; } >> phaseC14_pressure_vmstat.log
}

arm () {  # $1 = arm name, $2 = extra env ("" or Q36_HYBRID_GB=2), $3 = hog on?
  snapshot "$1-start"
  launch "$2" "serve-c14-$1.log" || exit 1
  if [ "$3" = hog ]; then nohup /tmp/c14_hog "$HOG_GB" > /dev/null 2>&1 & sleep 4; fi
  /opt/homebrew/bin/python3 c14_pressure.py "$KEY" "$1" 2>&1 | tee "phaseC14_pressure_$1.jsonl"
  pkill -f c14_hog 2>/dev/null
  snapshot "$1-end"
}

for r in 1 2; do
  echo "=== round $r ==="
  arm "idle0_r$r"  ""            ""
  arm "idle2_r$r"  "Q36_HYBRID_GB=2" ""
  arm "press0_r$r" ""            hog
  arm "press2_r$r" "Q36_HYBRID_GB=2" hog
done
echo "=== pressure arms complete ==="
