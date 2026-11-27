#!/usr/bin/env bash
# prof.sh <n> <tile> <round>: one fresh server of this branch, the prompt only, COLI_VK_CHAIN_PROF=1
E=$KIT; T=$TREE
n=$1; tile=$2; r=$3; port=$((18900 + RANDOM % 90)); tag=tile$tile-$n-r$r
pid=$(TREE=$T $E/step1/coli_serve.sh 24576 $port $tag COLI_PREFIX_STORE=0 COLI_VK_STAGED=1 COLI_VK_CHAIN_PROF=1 COLI_VK_RELATTN_TILE=$tile) || { echo "$tag: server failed"; exit 1; }
$E/step1/watchdog.sh $pid $E/logs/coli-$tag.log & w=$!
model=$(curl -s http://127.0.0.1:$port/v1/models | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
curl -s -m 600 http://127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":8}" -o /dev/null
res=$(python3 -I $E/step1/nbench.py run http://127.0.0.1:$port "$model" $E/step1/prompts/v0-$n.txt 0 $E/tile/$tag.json 8 | python3 -I -c 'import json,sys; r=json.loads(sys.stdin.read()); print(r["prompt_tokens"], "tok, prefill", r["prefill_s"], "s,", r["prompt_tok_s"], "tok/s")')
(cd $T/c && timeout 90 python3 ./coli stop --port $port | tail -1) >/dev/null; sleep 3; kill $w 2>/dev/null; kill -0 $pid 2>/dev/null && kill -9 $pid
prof=$(sed 's/\x1b\[[0-9;]*m//g' $E/logs/coli-$tag.log | grep 'chain profile' | tail -1 | grep -oE '^[^|]*device time|attention [0-9.]+ ms \([0-9]+\)' | tr '\n' ' ')
echo "$tag: $res | $prof"
