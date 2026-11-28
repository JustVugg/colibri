#!/usr/bin/env bash
# ab.sh: dev against the hand-off change, the 8k prompt alone (max_tokens 1), Q3 and Q4, three interleaved rounds
E=$KIT; DEV=$DEV; FIX=$FIX
for r in 1 2 3; do for q in q3 q4; do for arm in dev fix; do
  t=$(sensors | awk '/Tctl/{gsub(/[+°C]/,"",$2); print int($2)}'); [ "$t" -ge 80 ] && { echo "Tctl $t C: stopping"; exit 1; }
  T=$DEV; [ $arm = fix ] && T=$FIX; H=$E/step1/hist; [ $q = q4 ] && H=$E/q4/hist; tag=hab-$q-$arm-r$r; port=$((18500 + RANDOM % 90))
  pid=$(TREE=$T HIST=$H MDIR=$MODEL $E/step1/coli_serve.sh 24576 $port $tag COLI_PREFIX_STORE=0 COLI_VK_STAGED=1) || { echo "$tag: server failed"; continue; }
  $E/step1/watchdog.sh $pid $E/logs/coli-$tag.log > $E/q4/$tag.watchdog 2>&1 & w=$!
  model=$(curl -s http://127.0.0.1:$port/v1/models | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
  curl -s -m 600 http://127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":8}" -o /dev/null
  res=$(python3 -I $E/step1/nbench.py run http://127.0.0.1:$port "$model" $E/step1/prompts/v0-8000.txt 0 $E/q4/$tag.json 1 | python3 -I -c 'import json,sys; r=json.loads(sys.stdin.read()); print(r["prompt_tokens"], r["prefill_s"], r["prompt_tok_s"])')
  (cd $T/c && timeout 90 python3 ./coli stop --port $port | tail -1) >/dev/null; sleep 3; kill $w 2>/dev/null; kill -0 $pid 2>/dev/null && kill -9 $pid
  echo "$tag $res"
done; done; done
echo "done $(date +%T)"
