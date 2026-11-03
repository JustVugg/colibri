#!/bin/bash
# The tile kernels with the weights decoded once a block into shared memory (k3_dense_s4, k3_dense_o8):
# both long prompts' logits at every position against the CPU alone (long_cpu, big6k_cpu); then the
# 5,862-token prompt and 48 tokens of decode against 254338ab (pre_tiles/), alternated.
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
P2637="$(cat "$SP/long_prompt.txt")"; P5862="$(cat "$SP/long6k_prompt.txt")"
OLD="$SP/pre_tiles/kimi_k3.exe"
show(){   # name
    printf "%s: " "$1"; grep -h -E 'prefill done' "$SP/runs/$1.err" | sed 's/.*prefill done in //'
    grep -h -E 'the prompt:|gated MLA layers|the KDA layers|shared experts:|does not hold|one by one|dense \((prompt|run)\)|a decode token|decode [0-9]+ tokens|rc=' "$SP/runs/$1.err" | cut -c1-260
}
same(){   # a b what
    if [ -s "$SP/ref/$1" ] && cmp -s "$SP/ref/$1" "$SP/ref/$2"; then echo "$3: every position's logits identical ($(stat -c %s "$SP/ref/$1") bytes)"
    else echo "$3: LOGITS DIFFER ($(cmp "$SP/ref/$1" "$SP/ref/$2" 2>&1 | head -1))"; fi
}
txt(){ sed 's/TUNE.*//' "$SP/runs/$1.out"; }
K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" tl_2637_logits 0 K3_CHUNK=8192 K3_MAXT=8192 K3_LOGITS="$SP/ref/long_tiles.f32" > /dev/null; show tl_2637_logits
same long_tiles.f32 long_cpu.f32 "2637, the new tile kernels vs the CPU alone"
K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" tl_5862_logits 0 K3_CHUNK=8192 K3_MAXT=8192 K3_LOGITS="$SP/ref/big6k_tiles.f32" > /dev/null; show tl_5862_logits
same big6k_tiles.f32 big6k_cpu.f32 "5862, the new tile kernels vs the CPU alone"
for i in 1 2; do
    K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" "tl_5862_new$i" 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "tl_5862_new$i"
    K3_RUN_PROMPT="$P5862" K3_EXE="$OLD" bash "$SP/k3run.sh" "tl_5862_old$i" 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "tl_5862_old$i"
done
bash "$SP/k3run.sh" tl_dec_new1 48 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show tl_dec_new1
K3_EXE="$OLD" bash "$SP/k3run.sh" tl_dec_old1 48 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show tl_dec_old1
for f in tl_dec_new1 tl_dec_old1; do cmp -s <(txt $f) <(txt h2_short_cpu) && echo "$f: the CPU's text" || echo "$f: text differs from the CPU's"; done
