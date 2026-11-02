#!/bin/bash
# The exact dense kernel (k3_dense_b) on the real model: the logits of every position of the
# 2,637-token prompt (against long_ref) and of the 5,862-token one in a step of 8,192 (against
# the dump of two steps of 4,096), then A/B timings with K3_GPU_DENSE_B=0 (k3_dense)
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
P2637="$(cat "$SP/long_prompt.txt")"; P5862="$(cat "$SP/long6k_prompt.txt")"
K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" kb_2637_logits 0 K3_CHUNK=4096 K3_LOGITS="$SP/ref/long_kb.f32" > /dev/null
if cmp -s "$SP/ref/long_kb.f32" "$SP/ref/long_ref.f32"; then echo "2637: all positions' logits identical to long_ref"; else echo "2637: LOGITS DIFFER"; fi
K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" kb_5862_logits 0 K3_CHUNK=8192 K3_MAXT=8192 K3_LOGITS="$SP/ref/big6k_kb.f32" > /dev/null
if cmp -s "$SP/ref/big6k_kb.f32" "$SP/ref/big6k_4096.f32"; then echo "5862: all positions' logits identical to two steps of 4,096 with k3_dense"; else echo "5862: LOGITS DIFFER"; fi
for f in kb_2637_logits kb_5862_logits; do printf "%s: " $f; grep -h "prefill done" "$SP/runs/$f.err" | sed 's/.*prefill done in //'; done
for round in 1 2; do
    for v in kb old; do
        if [ $v = old ]; then d=0; else d=1; fi
        K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" "kb5862_${v}_$round" 1 K3_CHUNK=8192 K3_MAXT=8192 K3_GPU_DENSE_B=$d > /dev/null
        printf "5862 %s %s: " "$v" "$round"; grep -h -E 'prefill done' "$SP/runs/kb5862_${v}_$round.err" | sed 's/.*prefill done in //'
        grep -h -E 'the prompt:' "$SP/runs/kb5862_${v}_$round.err" | sed 's/.*the prompt: /  /'
    done
done
for round in 1 2; do
    for v in kb old; do
        if [ $v = old ]; then d=0; else d=1; fi
        K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" "kb2637_${v}_$round" 1 K3_CHUNK=4096 K3_GPU_DENSE_B=$d > /dev/null
        printf "2637 %s %s: " "$v" "$round"; grep -h -E 'prefill done' "$SP/runs/kb2637_${v}_$round.err" | sed 's/.*prefill done in //'
        grep -h -E 'the prompt:' "$SP/runs/kb2637_${v}_$round.err" | sed 's/.*the prompt: /  /'
    done
done
