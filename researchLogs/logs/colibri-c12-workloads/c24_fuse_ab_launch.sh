#!/usr/bin/env bash
# C24 real-model A/B launcher: the fused S==1 Q4_K/Q5_K kernels (Q36_EXPERT_FUSE
# default ON) vs the generic unpack path (=0), same Q4_K_M container both arms.
# 4 alternating fresh restarts (fuse1_r1 -> fuse0_r1 -> fuse1_r2 -> fuse0_r2),
# cache off, each round = the C16 driver's 1K/4K/12K calibrated colds + probe.
# Gates: output shas identical across arms (the kernels are bit-exact); decode
# rate is the perf read (C16 measured 0.72x IQ2 at the 12K anchor on the
# generic path; targets ~0.9x+ with the fused kernels).
# NOTE: run only after the Q4_K_M container is re-imported (it was cleaned from
# disk after C16): unsloth/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf (22,663,387,424 B),
# link-mode import via c/tools/gguf_import.py into the path below.
set -u
cd /Users/creemac/colibri/colibri || exit 1
W=researchLogs/logs/colibri-c12-workloads
KEY='-4Jc6egKcHmI-1okTj4_MUWOU4etmppT8LLG-wxzUBk'
Q4KM=/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-UD-Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.colibri

launch() { # $1 = log name, $2 = arm: "fuse1" (default) or "fuse0" (Q36_EXPERT_FUSE=0)
  pkill -f "colibri.cli serve" 2>/dev/null; pkill -f "c/qwen36" 2>/dev/null; sleep 3
  if [ "$2" = fuse0 ]; then
    OMP_NUM_THREADS=6 COLI_PROMPT_CACHE=0 Q36_EXPERT_FUSE=0 \
      PYTHONPATH=/Users/creemac/colibri/colibri \
      nohup /opt/homebrew/bin/python3 -m colibri.cli serve \
      --model "$Q4KM" --host 0.0.0.0 --port 8001 --ctx 66000 \
      --api-key="$KEY" > "$W/$1" 2>&1 &
  else
    OMP_NUM_THREADS=6 COLI_PROMPT_CACHE=0 \
      PYTHONPATH=/Users/creemac/colibri/colibri \
      nohup /opt/homebrew/bin/python3 -m colibri.cli serve \
      --model "$Q4KM" --host 0.0.0.0 --port 8001 --ctx 66000 \
      --api-key="$KEY" > "$W/$1" 2>&1 &
  fi
  sleep 20
  grep -m1 "RSS after load" "$W/$1" || true
}

round() { # $1 = arm tag, $2 = arm
  launch "serve-c24-$1.log" "$2"
  echo "=== $1 ==="
  /opt/homebrew/bin/python3 "$W/c16_q4k_ab.py" "$KEY" 2>&1 | tee "$W/phaseC24_fuse_$1.jsonl"
}

round fuse1_r1 fuse1
round fuse0_r1 fuse0
round fuse1_r2 fuse1
round fuse0_r2 fuse0

echo "=== C24 A/B complete ==="
