#!/usr/bin/env bash
E=$KIT
for r in 1 2 3; do for n in 8000 16000; do for tile in 1 0; do
  [ $r = 1 ] && [ $n = 8000 ] && [ $tile = 1 ] && continue   # done already
  t=$(sensors | awk '/Tctl/{gsub(/[+°C]/,"",$2); print int($2)}'); [ "$t" -ge 80 ] && { echo "Tctl $t C: stopping"; exit 1; }
  bash $E/tile/prof.sh $n $tile $r 2>&1 | tail -1
done; done; done
echo "done $(date +%T)"
