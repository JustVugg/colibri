#!/usr/bin/env bash
# C23 three-container graded batch (the user's plan step 3): the committed C23
# eval set on IQ2_XXS / UD-IQ3_XXS / Q4_K_M, interleaved rounds on the C24 build.
# SEQUENTIAL, one container resident at a time: each round = pkill -> fresh
# launch with ONE model -> run the set -> pkill. Never two models in RAM.
# Order iq2_r1 -> iq3_r1 -> q4km_r1 -> iq2_r2 -> iq3_r2 -> q4km_r2 keeps the
# three containers inside the same drift window (the C13 paired-reads rule).
# Cache ON (the set exercises the real reuse path deliberately); no env
# overrides -- every flag is the C24 default build (Q36_EXPERT_FUSE default ON).
set -u
cd /Users/creemac/colibri/colibri || exit 1
W=researchLogs/logs/colibri-c12-workloads
KEY='-4Jc6egKcHmI-1okTj4_MUWOU4etmppT8LLG-wxzUBk'
IQ2=/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ2_XXS.colibri
IQ3=/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-UD-IQ3_XXS/Qwen3.6-35B-A3B-UD-IQ3_XXS.colibri
Q4KM=/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-UD-Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.colibri

launch() { # $1 = log name, $2 = container path
  pkill -f "colibri.cli serve" 2>/dev/null; pkill -f "c/qwen36" 2>/dev/null; sleep 3
  OMP_NUM_THREADS=6 COLI_PROMPT_CACHE=1 \
    PYTHONPATH=/Users/creemac/colibri/colibri \
    nohup /opt/homebrew/bin/python3 -m colibri.cli serve \
    --model "$2" --host 0.0.0.0 --port 8001 --ctx 66000 \
    --api-key="$KEY" > "$W/$1" 2>&1 &
  sleep 20
  grep -m1 "RSS after load" "$W/$1" || true
}

round() { # $1 = round tag (jsonl suffix), $2 = log name, $3 = container path
  echo "=== $1 ==="
  launch "$2" "$3"
  /opt/homebrew/bin/python3 "$W/c23_eval.py" "$1" --key "$KEY" 2>&1 | tail -30
}

round iq2_r1  serve-c23tri-iq2_r1.log  "$IQ2"
round iq3_r1  serve-c23tri-iq3_r1.log  "$IQ3"
round q4km_r1 serve-c23tri-q4km_r1.log "$Q4KM"
round iq2_r2  serve-c23tri-iq2_r2.log  "$IQ2"
round iq3_r2  serve-c23tri-iq3_r2.log  "$IQ3"
round q4km_r2 serve-c23tri-q4km_r2.log "$Q4KM"

echo "=== C23 three-container batch complete ==="
