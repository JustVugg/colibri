#!/usr/bin/env bash
# C22 real-model A/B launcher: the fused S==1 IQ2_S/IQ3_S kernels (Q36_EXPERT_FUSE
# default ON) vs the generic unpack path (=0), same UD-IQ3_XXS container both arms.
# 4 alternating fresh restarts (fuse1_r1 -> fuse0_r1 -> fuse1_r2 -> fuse0_r2), cache
# off, each round = the C16 driver's 1K/4K/12K calibrated colds + quality probe.
# Gates: output shas identical across arms (the kernels are bit-exact); decode rate
# is the perf read (C21 measured 0.74x IQ2 at the 12K anchor on the generic path).
set -u
cd /Users/creemac/colibri/colibri || exit 1
W=researchLogs/logs/colibri-c12-workloads
KEY='-4Jc6egKcHmI-1okTj4_MUWOU4etmppT8LLG-wxzUBk'
IQ3=/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-UD-IQ3_XXS/Qwen3.6-35B-A3B-UD-IQ3_XXS.colibri

launch() { # $1 = log name
  pkill -f "colibri.cli serve" 2>/dev/null; pkill -f "c/qwen36" 2>/dev/null; sleep 3
  OMP_NUM_THREADS=6 COLI_PROMPT_CACHE=0 $2 \
    PYTHONPATH=/Users/creemac/colibri/colibri \
    nohup /opt/homebrew/bin/python3 -m colibri.cli serve \
    --model "$IQ3" --host 0.0.0.0 --port 8001 --ctx 66000 \
    --api-key="$KEY" > "$W/$1" 2>&1 &
  sleep 20
  grep -m1 "RSS after load" "$W/$1" || true
}

round() { # $1 = arm tag, $2 = extra env (empty = default)
  launch "serve-c22-$1.log" "$2"
  echo "=== $1 ==="
  /opt/homebrew/bin/python3 "$W/c16_q4k_ab.py" "$KEY" 2>&1 | tee "$W/phaseC22_fuse_$1.jsonl"
}

round fuse1_r1 ""
round fuse0_r1 "Q36_EXPERT_FUSE=0"
round fuse1_r2 ""
round fuse0_r2 "Q36_EXPERT_FUSE=0"

echo "=== C22 A/B complete ==="
