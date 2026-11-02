#!/bin/bash
# The MoE step's tail (k3_down's rows whole, the inputs' quantization, the weighted sums, the latent
# norm and the shared experts' add over every worker) at the launcher's 8,192 step: the logits of
# every position of both long prompts against their references, then the new build and fdcce1bc
# (pre_moe/) alternated, twice on each prompt.
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
P2637="$(cat "$SP/long_prompt.txt")"; P5862="$(cat "$SP/long6k_prompt.txt")"
OLD="$SP/pre_moe/kimi_k3.exe"
show(){   # name
    printf "%s: " "$1"; grep -h -E 'prefill done' "$SP/runs/$1.err" | sed 's/.*prefill done in //'
    grep -h -E 'the prompt:|copying their bytes|the shared experts:' "$SP/runs/$1.err"
}
K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" moe_2637_logits 0 K3_CHUNK=8192 K3_MAXT=8192 K3_LOGITS="$SP/ref/long_moe.f32" > /dev/null
if cmp -s "$SP/ref/long_moe.f32" "$SP/ref/long_ref.f32"; then echo "2637: all positions' logits identical to long_ref"; else echo "2637: LOGITS DIFFER"; fi
K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" moe_5862_logits 0 K3_CHUNK=8192 K3_MAXT=8192 K3_LOGITS="$SP/ref/big6k_moe.f32" > /dev/null
if cmp -s "$SP/ref/big6k_moe.f32" "$SP/ref/big6k_4096.f32"; then echo "5862: all positions' logits identical to big6k_4096"; else echo "5862: LOGITS DIFFER"; fi
rm -f "$SP/ref/long_moe.f32" "$SP/ref/big6k_moe.f32"
for i in 1 2; do
    K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" "moe_5862_new$i" 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "moe_5862_new$i"
    K3_RUN_PROMPT="$P5862" K3_EXE="$OLD" bash "$SP/k3run.sh" "moe_5862_old$i" 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "moe_5862_old$i"
done
for i in 1 2; do
    K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" "moe_2637_new$i" 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "moe_2637_new$i"
    K3_RUN_PROMPT="$P2637" K3_EXE="$OLD" bash "$SP/k3run.sh" "moe_2637_old$i" 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "moe_2637_old$i"
done
