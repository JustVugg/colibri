#!/bin/bash
# The KDA layer on the GPU (k3cu_kda) at the launcher's 8,192 step: the 2,637-token prompt's logits at
# every position against long_ref, then the new build and the one before (pre_kda/) alternated on the
# 2,637-token prompt twice each, then the 5,862-token prompt once each.
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
P2637="$(cat "$SP/long_prompt.txt")"; P5862="$(cat "$SP/long6k_prompt.txt")"
OLD="$SP/pre_kda/kimi_k3.exe"
show(){   # name
    printf "%s: " "$1"; grep -h -E 'prefill done' "$SP/runs/$1.err" | sed 's/.*prefill done in //'
    grep -h -E 'the prompt:|KDA on the CPU|the KDA layers|the dense matrices the GPU does not|dense \(prompt\)' "$SP/runs/$1.err"
}
K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" kda_2637_logits 0 K3_CHUNK=8192 K3_MAXT=8192 K3_LOGITS="$SP/ref/long_kda.f32" > /dev/null
if cmp -s "$SP/ref/long_kda.f32" "$SP/ref/long_ref.f32"; then echo "2637: all positions' logits identical to long_ref"; else echo "2637: LOGITS DIFFER"; fi
show kda_2637_logits
for i in 1 2; do
    K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" "kda_2637_new$i" 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "kda_2637_new$i"
    K3_RUN_PROMPT="$P2637" K3_EXE="$OLD" bash "$SP/k3run.sh" "kda_2637_old$i" 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "kda_2637_old$i"
done
K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" kda_5862_new 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show kda_5862_new
K3_RUN_PROMPT="$P5862" K3_EXE="$OLD" bash "$SP/k3run.sh" kda_5862_old 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show kda_5862_old
K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" kda_5862_logits 0 K3_CHUNK=8192 K3_MAXT=8192 K3_LOGITS="$SP/ref/big6k_kda.f32" > /dev/null
if cmp -s "$SP/ref/big6k_kda.f32" "$SP/ref/big6k_4096.f32"; then echo "5862: all positions' logits identical to big6k_4096"; else echo "5862: LOGITS DIFFER"; fi
show kda_5862_logits
