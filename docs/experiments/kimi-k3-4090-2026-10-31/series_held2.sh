#!/bin/bash
# The final kernels (k3_mv4 a row at a time with four outputs a task, k3_mv8 through the warp's shared memory):
# the 35-token chat prompt one token a step (every position through a decode step's kernels) with the matrices
# on the card against the CPU alone (logits of every position, then 45 tokens of text); the 2,637-token prompt's
# logits again; then decode and the 5,862-token prompt against 9372715b (pre_held/), alternated.
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
P2637="$(cat "$SP/long_prompt.txt")"; P5862="$(cat "$SP/long6k_prompt.txt")"
OLD="$SP/pre_held/kimi_k3.exe"
show(){   # name
    printf "%s: " "$1"; grep -h -E 'prefill done' "$SP/runs/$1.err" | sed 's/.*prefill done in //'
    grep -h -E 'the prompt:|gated MLA layers|the KDA layers|one by one|dense \((prompt|run)\)|a decode token|decode [0-9]+ tokens' "$SP/runs/$1.err" | cut -c1-260
}
same(){   # a b what
    if [ -s "$SP/ref/$1" ] && cmp -s "$SP/ref/$1" "$SP/ref/$2"; then echo "$3: every position's logits identical ($(stat -c %s "$SP/ref/$1") bytes)"
    else echo "$3: LOGITS DIFFER ($(cmp "$SP/ref/$1" "$SP/ref/$2" 2>&1 | head -1))"; fi
}
bash "$SP/k3run.sh" h2_short_held1 48 K3_CHUNK=1 K3_MAXT=8192 K3_LOGITS="$SP/ref/short_held1.f32" > /dev/null; show h2_short_held1
bash "$SP/k3run.sh" h2_short_cpu 48 K3_CHUNK=8192 K3_MAXT=8192 K3_CUDA_DENSE=0 K3_GPU_EXPERTS=0 K3_LOGITS="$SP/ref/short_cpu.f32" > /dev/null; show h2_short_cpu
same short_held1.f32 short_cpu.f32 "35 tokens one a step on the card vs the CPU alone"
txt(){ sed 's/TUNE.*//' "$SP/runs/$1.out"; }
if cmp -s <(txt h2_short_held1) <(txt h2_short_cpu); then echo "48 generated tokens: the same text"; else echo "48 generated tokens: TEXT DIFFERS"; fi
K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" h2_2637_logits 0 K3_CHUNK=8192 K3_MAXT=8192 K3_LOGITS="$SP/ref/long_held2.f32" > /dev/null; show h2_2637_logits
same long_held2.f32 long_cpu.f32 "2637, the card's matrices vs the CPU alone"
for i in 1 2; do
    bash "$SP/k3run.sh" "h2_dec_new$i" 48 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "h2_dec_new$i"
    K3_EXE="$OLD" bash "$SP/k3run.sh" "h2_dec_old$i" 48 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "h2_dec_old$i"
done
K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" h2_5862_new1 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show h2_5862_new1
K3_RUN_PROMPT="$P5862" K3_EXE="$OLD" bash "$SP/k3run.sh" h2_5862_old1 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show h2_5862_old1
for f in h2_dec_new1 h2_dec_old1 h2_dec_new2 h2_dec_old2; do cmp -s <(txt $f) <(txt h2_short_cpu) && echo "$f: the CPU's text" || echo "$f: text differs from the CPU's"; done
