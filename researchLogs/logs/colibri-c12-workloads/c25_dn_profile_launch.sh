#!/usr/bin/env bash
# Cold long-prefill CPU sample of the DeltaNet path. The canonical server is
# stopped before profiling and restored (cache on) on every exit.
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
launch_server() { # log, cache 0|1
  local log="$1" cache="$2"
  stop_server
  if [ "$cache" = 0 ]; then
    OMP_NUM_THREADS=6 Q36_MTP=0 COLI_LOOKUP=0 COLI_PROMPT_CACHE=0 COLI_TIMERS=1 \
      COLI_PREFIX_LOG=1 COLI_PCACHE_LOG=1 COLI_LOG_TIMINGS="$LOG_TIMINGS" \
      PYTHONPATH=/Users/creemac/colibri/colibri \
      nohup /opt/homebrew/bin/python3 -m colibri.cli serve \
      --model "$MODEL" --host 0.0.0.0 --port 8001 --ctx 66000 --api-key="$KEY" \
      > "$W/$log" 2>&1 &
  else
    OMP_NUM_THREADS=6 Q36_MTP=0 COLI_LOOKUP=1 COLI_PROMPT_CACHE=1 COLI_TIMERS=0 \
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
  launch_server serve-restored-c25-dn.log 1
}
trap restore EXIT INT TERM

launch_server serve-c25-dn-profile.log 0 || exit 1
/opt/homebrew/bin/python3 "$W/c25_dn_profile.py" > "$W/c25_dn_profile_request.out" 2>&1 &
request_pid=$!
sleep 3
engine_pid=$(pgrep -x qwen36 | head -1)
if [ -z "$engine_pid" ]; then
  echo "qwen36 engine process not found" >&2
  wait "$request_pid" || true
  exit 1
fi
echo "sampling qwen36 pid=$engine_pid while the cold prefill is active"
sample "$engine_pid" 30 1 -fullPaths -file "$W/phaseC25_dn_profile.sample.txt"
wait "$request_pid"
cat "$W/c25_dn_profile_request.out"
echo "DeltaNet CPU profile sample complete"
