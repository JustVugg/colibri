#!/usr/bin/env bash
# needle.sh <n> <tile>: this branch, the full needle run (thinking on), watchdog on
E=$KIT; T=$TREE
n=$1; tile=$2; port=$((18800 + RANDOM % 90)); tag=needle-tile$tile-$n
pid=$(TREE=$T $E/step1/coli_serve.sh 73728 $port $tag COLI_PREFIX_STORE=0 COLI_VK_STAGED=1 COLI_VK_RELATTN_TILE=$tile) || { echo "$tag: server failed"; exit 1; }
$E/step1/watchdog.sh $pid $E/logs/coli-$tag.log > $E/tile/$tag.watchdog 2>&1 & w=$!
model=$(curl -s http://127.0.0.1:$port/v1/models | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
curl -s -m 600 http://127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":8}" -o /dev/null
NB_EXTRA='{"enable_thinking": true}' python3 -I $E/step1/nbench.py run http://127.0.0.1:$port "$model" $E/step1/prompts/v0-$n.txt 0 $E/tile/$tag.json 6000 \
  | python3 -I -c 'import json,sys; r=json.loads(sys.stdin.read()); print(r["prompt_tokens"], "tok | prefill", r["prefill_s"], "s,", r["prompt_tok_s"], "tok/s | decode", r["decode_tok_s"], "tok/s over", r["completion_tokens"], "| needle", r["score"], "/ 5")'
(cd $T/c && timeout 90 python3 ./coli stop --port $port | tail -1) >/dev/null; sleep 3; kill $w 2>/dev/null; kill -0 $pid 2>/dev/null && kill -9 $pid
[ -s $E/tile/$tag.watchdog ] && cat $E/tile/$tag.watchdog
true
