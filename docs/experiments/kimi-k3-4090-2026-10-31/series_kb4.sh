#!/bin/bash
# The final build (k3_dense_b, the GPU's calls 1,024 rows at a time, MLA scores within 128 MB) at
# the default 2 GB reserve and the launcher's 8,192 step: the 2,637-token prompt's logits at every
# position against long_ref, two timed runs of it, one of the 5,862-token prompt
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
P2637="$(cat "$SP/long_prompt.txt")"; P5862="$(cat "$SP/long6k_prompt.txt")"
K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" kb4_2637_logits 0 K3_CHUNK=8192 K3_MAXT=8192 K3_LOGITS="$SP/ref/long_kb4.f32" > /dev/null
if cmp -s "$SP/ref/long_kb4.f32" "$SP/ref/long_ref.f32"; then echo "2637: all positions' logits identical to long_ref"; else echo "2637: LOGITS DIFFER"; fi
for i in 1 2; do
    K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" "kb4_2637_$i" 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null
    printf "2637 %s: " $i; grep -h -E 'prefill done' "$SP/runs/kb4_2637_$i.err" | sed 's/.*prefill done in //'
    grep -h -E 'the prompt:|the dense matrices the GPU does not|dense \(prompt\)' "$SP/runs/kb4_2637_$i.err"
done
K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" kb4_5862 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null
printf "5862: "; grep -h -E 'prefill done' "$SP/runs/kb4_5862.err" | sed 's/.*prefill done in //'
grep -h -E 'the prompt:|the dense matrices the GPU does not' "$SP/runs/kb4_5862.err"
