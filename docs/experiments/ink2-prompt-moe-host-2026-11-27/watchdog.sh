#!/usr/bin/env bash
# watchdog.sh <engine_pid> <log>: SIGKILL the engine on a lost device or a hot CPU (Tctl >= 85 C)
pid=$1; log=$2
while kill -0 $pid 2>/dev/null; do
  if grep -qE 'device is lost|device was lost|stopped answering' "$log" 2>/dev/null; then echo "$(date +%T) device lost: killing $pid"; kill -9 $pid; exit; fi
  t=$(sensors 2>/dev/null | awk '/Tctl/{gsub(/[+°C]/,"",$2); print int($2)}')
  if [ -n "$t" ] && [ "$t" -ge 85 ]; then echo "$(date +%T) Tctl $t C: killing $pid"; kill -9 $pid; exit; fi
  sleep 1
done
