#!/usr/bin/env bash
# Four alternating MTP-off/on rounds on the selected IQ2_XXS, then restore
# the canonical IQ2 service. Cache is disabled for a clean, fixed decode read.
set -u
cd /Users/creemac/colibri/colibri || exit 1
W=researchLogs/logs/colibri-c12-workloads
KEY='-4Jc6egKcHmI-1okTj4_MUWOU4etmppT8LLG-wxzUBk'
MODEL=/Users/creemac/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ2_XXS.colibri
LOG_TIMINGS=/Users/creemac/colibri/colibri/researchLogs/logs/colibri-c12-workloads/timings-requests.jsonl

stop_server() {
  pkill -f "colibri.cli serve" 2>/dev/null || true
  pkill -f "c/qwen36" 2>/dev/null || true
  sleep 3
}
launch_arm() { # log tag, MTP 0|1
  local log="$1" mtp="$2"
  stop_server
  if [ "$mtp" = 1 ]; then
    OMP_NUM_THREADS=6 Q36_MTP=1 Q36_MTP_DRAFTS=1 COLI_LOOKUP=0 \
      COLI_PROMPT_CACHE=0 COLI_PREFIX_LOG=1 COLI_PCACHE_LOG=1 COLI_LOG_TIMINGS="$LOG_TIMINGS" \
      PYTHONPATH=/Users/creemac/colibri/colibri \
      nohup /opt/homebrew/bin/python3 -m colibri.cli serve \
      --model "$MODEL" --host 0.0.0.0 --port 8001 --ctx 66000 --api-key="$KEY" \
      > "$W/$log" 2>&1 &
  else
    OMP_NUM_THREADS=6 Q36_MTP=0 COLI_LOOKUP=0 COLI_PROMPT_CACHE=0 \
      COLI_PREFIX_LOG=1 COLI_PCACHE_LOG=1 COLI_LOG_TIMINGS="$LOG_TIMINGS" \
      PYTHONPATH=/Users/creemac/colibri/colibri \
      nohup /opt/homebrew/bin/python3 -m colibri.cli serve \
      --model "$MODEL" --host 0.0.0.0 --port 8001 --ctx 66000 --api-key="$KEY" \
      > "$W/$log" 2>&1 &
  fi
  for _ in $(seq 1 90); do
    if curl -fsS --max-time 2 http://127.0.0.1:8001/health >/dev/null 2>&1; then
      grep -m1 "RSS after load" "$W/$log" || true
      return 0
    fi
    sleep 2
  done
  echo "server failed to become healthy: $log" >&2
  tail -50 "$W/$log" >&2
  return 1
}
restore() {
  echo "Restoring canonical IQ2_XXS service"
  stop_server
  OMP_NUM_THREADS=6 Q36_MTP=0 COLI_LOOKUP=1 COLI_PROMPT_CACHE=1 \
    COLI_PREFIX_LOG=1 COLI_PCACHE_LOG=1 COLI_LOG_TIMINGS="$LOG_TIMINGS" \
    PYTHONPATH=/Users/creemac/colibri/colibri \
    nohup /opt/homebrew/bin/python3 -m colibri.cli serve \
    --model "$MODEL" --host 0.0.0.0 --port 8001 --ctx 66000 --api-key="$KEY" \
    > "$W/serve-restored-c25-mtp.log" 2>&1 &
  for _ in $(seq 1 90); do
    if curl -fsS --max-time 2 http://127.0.0.1:8001/health >/dev/null 2>&1; then
      grep -m1 "RSS after load" "$W/serve-restored-c25-mtp.log" || true
      return 0
    fi
    sleep 2
  done
  echo "canonical IQ2 service failed to restore" >&2
  tail -50 "$W/serve-restored-c25-mtp.log" >&2
  return 1
}
trap restore EXIT INT TERM

for spec in off1:0 on1:1 on2:1 off2:0; do
  tag=${spec%%:*}; mtp=${spec##*:}
  echo "=== $tag MTP=$mtp ==="
  launch_arm "serve-c25-mtp-$tag.log" "$mtp" || exit 1
  /opt/homebrew/bin/python3 "$W/c25_mtp_probe.py" "$tag" "$mtp" || exit 1
  tail -8 "$W/serve-c25-mtp-$tag.log"
done
echo "C25 MTP probe arms complete"
