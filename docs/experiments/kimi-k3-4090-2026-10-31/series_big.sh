#!/bin/bash
# A ~6,000-token prompt in two steps of 4,096 against one of 8,192: the logits of all positions,
# then a timed run of each (one token generated)
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
export K3_RUN_PROMPT="$(cat "$SP/long6k_prompt.txt")"
for ch in 4096 8192; do
    bash "$SP/k3run.sh" "big_${ch}_logits" 0 K3_CHUNK=$ch K3_MAXT=8192 K3_LOGITS="$SP/ref/big6k_$ch.f32" > /dev/null
    printf "chunk %s logits run: " "$ch"; grep -h -E 'prompt: [0-9]+ tokens' "$SP/runs/big_${ch}_logits.err" | sed 's/.*prompt: //'
    grep -h -E 'prefill done|the prompt:|cache 619|rc=' "$SP/runs/big_${ch}_logits.err" | cut -c1-260
done
if cmp -s "$SP/ref/big6k_4096.f32" "$SP/ref/big6k_8192.f32"; then echo "one step of 8,192: all positions' logits identical to two steps of 4,096"
else echo "LOGITS DIFFER between the step sizes"; cmp "$SP/ref/big6k_4096.f32" "$SP/ref/big6k_8192.f32" | head -2; fi
for ch in 4096 8192; do
    bash "$SP/k3run.sh" "big_${ch}_time" 1 K3_CHUNK=$ch K3_MAXT=8192 > /dev/null
    printf "chunk %s timed: " "$ch"; grep -h -E 'prefill done' "$SP/runs/big_${ch}_time.err" | sed 's/.*prefill done in //'
    grep -h -E 'the prompt:|a decode token' "$SP/runs/big_${ch}_time.err" | cut -c1-260
done
