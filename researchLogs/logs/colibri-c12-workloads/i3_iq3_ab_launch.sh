#!/usr/bin/env bash
# IQ3 eval launcher: UD-IQ3_XXS vs IQ2_XXS quantization comparison, interleaved A/B.
# 4 alternating fresh restarts (iq2_r1 -> iq3_r1 -> iq2_r2 -> iq3_r2), cache off,
# each round = 1K/4K/12K colds + the quality probe (re-uses the C16 driver).
# The server log per round carries the memory evidence (RSS after load).
# File facts (from phaseI3_*_tensors.txt): IQ2 arm = IQ2_XXS gate/up + IQ3_XXS down
# (fused decode-once covered); IQ3 arm = IQ2_S gate/up + IQ3_S down (generic path).
set -u
cd /Users/creemac/colibri/colibri || exit 1
W=researchLogs/logs/colibri-c12-workloads
KEY='-4Jc6egKcHmI-1okTj4_MUWOU4etmppT8LLG-wxzUBk'
IQ2=/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ2_XXS.colibri
IQ3=/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-UD-IQ3_XXS/Qwen3.6-35B-A3B-UD-IQ3_XXS.colibri

launch() { # $1 = log name, $2 = container
  pkill -f "colibri.cli serve" 2>/dev/null; pkill -f "c/qwen36" 2>/dev/null; sleep 3
  OMP_NUM_THREADS=6 COLI_PROMPT_CACHE=0 \
    PYTHONPATH=/Users/creemac/colibri/colibri \
    nohup /opt/homebrew/bin/python3 -m colibri.cli serve \
    --model "$2" --host 0.0.0.0 --port 8001 --ctx 66000 \
    --api-key="$KEY" > "$W/$1" 2>&1 &
  sleep 20
  grep -m1 "RSS after load" "$W/$1" || true
}

round() { # $1 = arm tag, $2 = container
  launch "serve-i3-$1.log" "$2"
  echo "=== $1 ==="
  /opt/homebrew/bin/python3 "$W/c16_q4k_ab.py" "$KEY" 2>&1 | tee "$W/phaseI3_iq3_$1.jsonl"
}

round iq2_r1 "$IQ2"
round iq3_r1 "$IQ3"
round iq2_r2 "$IQ2"
round iq3_r2 "$IQ3"

echo "=== IQ3 comparison complete ==="
