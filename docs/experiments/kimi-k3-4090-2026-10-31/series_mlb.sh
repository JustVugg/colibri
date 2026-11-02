#!/bin/bash
# The MLA core's whole step in one call (chunks of up to 256 rows, the copies under the kernels) and exp_f in
# float pairs in every kernel, at the launcher's 8,192 step: the logits of every position of both long prompts
# against their references, then the new build and 04623a27 (pre_mla/) alternated: twice at 5,862 tokens,
# once at 2,637.
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
P2637="$(cat "$SP/long_prompt.txt")"; P5862="$(cat "$SP/long6k_prompt.txt")"
OLD="$SP/pre_mla/kimi_k3.exe"
show(){   # name
    printf "%s: " "$1"; grep -h -E 'prefill done' "$SP/runs/$1.err" | sed 's/.*prefill done in //'
    grep -h -E 'the prompt:|gated MLA|the KDA layers|the shared experts:' "$SP/runs/$1.err"
}
K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" mlb_5862_logits 0 K3_CHUNK=8192 K3_MAXT=8192 K3_LOGITS="$SP/ref/big6k_mlb.f32" > /dev/null
if cmp -s "$SP/ref/big6k_mlb.f32" "$SP/ref/big6k_4096.f32"; then echo "5862: all positions' logits identical to big6k_4096"; else echo "5862: LOGITS DIFFER"; fi
show mlb_5862_logits
K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" mlb_2637_logits 0 K3_CHUNK=8192 K3_MAXT=8192 K3_LOGITS="$SP/ref/long_mlb.f32" > /dev/null
if cmp -s "$SP/ref/long_mlb.f32" "$SP/ref/long_ref.f32"; then echo "2637: all positions' logits identical to long_ref"; else echo "2637: LOGITS DIFFER"; fi
show mlb_2637_logits
rm -f "$SP/ref/long_mlb.f32" "$SP/ref/big6k_mlb.f32"
for i in 1 2; do
    K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" "mlb_5862_new$i" 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "mlb_5862_new$i"
    K3_RUN_PROMPT="$P5862" K3_EXE="$OLD" bash "$SP/k3run.sh" "mlb_5862_old$i" 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "mlb_5862_old$i"
done
K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" mlb_2637_new1 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show mlb_2637_new1
K3_RUN_PROMPT="$P2637" K3_EXE="$OLD" bash "$SP/k3run.sh" mlb_2637_old1 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show mlb_2637_old1
