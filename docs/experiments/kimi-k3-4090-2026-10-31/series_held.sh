#!/bin/bash
# The card's matrices in the CPU's arithmetic (k3cu_matmul_dev; k3_dense_b, k3_dense_b8 and the decode kernels on
# coli_cuda.dll's copies), at the launcher's 8,192 step: the logits of every position of both long prompts with the
# matrices on the card, with none there (every other GPU path on), and the 2,637-token one on the CPU alone; then
# the new build and 9372715b (pre_held/) alternated: twice at 5,862 tokens, once at 2,637, twice 48 tokens of decode;
# last the 5,862-token prompt on the CPU alone.
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
P2637="$(cat "$SP/long_prompt.txt")"; P5862="$(cat "$SP/long6k_prompt.txt")"
OLD="$SP/pre_held/kimi_k3.exe"
show(){   # name
    printf "%s: " "$1"; grep -h -E 'prefill done' "$SP/runs/$1.err" | sed 's/.*prefill done in //'
    grep -h -E 'the prompt:|gated MLA layers|the KDA layers|one by one|dense \((prompt|run)\)|a decode token|decode [0-9]+ tokens|K3-CUDA\] dense:' "$SP/runs/$1.err" | cut -c1-260
}
same(){   # a b what
    if [ -s "$SP/ref/$1" ] && cmp -s "$SP/ref/$1" "$SP/ref/$2"; then echo "$3: every position's logits identical ($(stat -c %s "$SP/ref/$1") bytes)"
    else echo "$3: LOGITS DIFFER ($(cmp "$SP/ref/$1" "$SP/ref/$2" 2>&1 | head -1))"; fi
}
K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" hl_5862_logits 0 K3_CHUNK=8192 K3_MAXT=8192 K3_LOGITS="$SP/ref/big6k_held.f32" > /dev/null; show hl_5862_logits
K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" hl_2637_logits 0 K3_CHUNK=8192 K3_MAXT=8192 K3_LOGITS="$SP/ref/long_held.f32" > /dev/null; show hl_2637_logits
K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" hl_2637_none 0 K3_CHUNK=8192 K3_MAXT=8192 K3_CUDA_DENSE=0 K3_LOGITS="$SP/ref/long_none.f32" > /dev/null; show hl_2637_none
same long_held.f32 long_none.f32 "2637, the card's matrices vs none there"
K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" hl_5862_none 0 K3_CHUNK=8192 K3_MAXT=8192 K3_CUDA_DENSE=0 K3_LOGITS="$SP/ref/big6k_none.f32" > /dev/null; show hl_5862_none
same big6k_held.f32 big6k_none.f32 "5862, the card's matrices vs none there"
K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" hl_2637_cpu 0 K3_CHUNK=8192 K3_MAXT=8192 K3_CUDA_DENSE=0 K3_GPU_EXPERTS=0 K3_LOGITS="$SP/ref/long_cpu.f32" > /dev/null; show hl_2637_cpu
same long_held.f32 long_cpu.f32 "2637, the card's matrices vs the CPU alone"
for i in 1 2; do
    K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" "hl_5862_new$i" 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "hl_5862_new$i"
    K3_RUN_PROMPT="$P5862" K3_EXE="$OLD" bash "$SP/k3run.sh" "hl_5862_old$i" 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "hl_5862_old$i"
done
K3_RUN_PROMPT="$P2637" bash "$SP/k3run.sh" hl_2637_new1 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show hl_2637_new1
K3_RUN_PROMPT="$P2637" K3_EXE="$OLD" bash "$SP/k3run.sh" hl_2637_old1 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show hl_2637_old1
for i in 1 2; do
    bash "$SP/k3run.sh" "hl_dec_new$i" 48 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "hl_dec_new$i"
    K3_EXE="$OLD" bash "$SP/k3run.sh" "hl_dec_old$i" 48 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "hl_dec_old$i"
done
K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" hl_5862_cpu 0 K3_CHUNK=8192 K3_MAXT=8192 K3_CUDA_DENSE=0 K3_GPU_EXPERTS=0 K3_LOGITS="$SP/ref/big6k_cpu.f32" > /dev/null; show hl_5862_cpu
same big6k_held.f32 big6k_cpu.f32 "5862, the card's matrices vs the CPU alone"
