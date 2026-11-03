#!/bin/bash
# KDA's pieces of 1,024 rows (K3_GPU_KDA_ROWS) against the default 512 with the new tile kernels: the 5,862-token
# prompt alternated twice, then its logits with 1,024-row pieces against the CPU alone.
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
P5862="$(cat "$SP/long6k_prompt.txt")"
show(){   # name
    printf "%s: " "$1"; grep -h -E 'prefill done' "$SP/runs/$1.err" | sed 's/.*prefill done in //'
    grep -h -E 'the prompt:|gated MLA layers|the KDA layers|shared experts:|does not hold|rc=' "$SP/runs/$1.err" | cut -c1-260
    awk -F';' 'NR>1 && $0 ~ /[0-9]/ {for(i=1;i<=NF;i++) if(h[i]=="gpu_mem_mb" && $i+0>m) m=$i+0} NR==1 {for(i=1;i<=NF;i++) h[i]=$i} END {if(m) print "peak GPU memory " m " MiB"}' "$SP/runs/$1.csv"
}
same(){   # a b what
    if [ -s "$SP/ref/$1" ] && cmp -s "$SP/ref/$1" "$SP/ref/$2"; then echo "$3: every position's logits identical ($(stat -c %s "$SP/ref/$1") bytes)"
    else echo "$3: LOGITS DIFFER ($(cmp "$SP/ref/$1" "$SP/ref/$2" 2>&1 | head -1))"; fi
}
for i in 1 2; do
    K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" "kr_5862_1024_$i" 1 K3_CHUNK=8192 K3_MAXT=8192 K3_GPU_KDA_ROWS=1024 > /dev/null; show "kr_5862_1024_$i"
    K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" "kr_5862_512_$i" 1 K3_CHUNK=8192 K3_MAXT=8192 > /dev/null; show "kr_5862_512_$i"
done
K3_RUN_PROMPT="$P5862" bash "$SP/k3run.sh" kr_5862_logits 0 K3_CHUNK=8192 K3_MAXT=8192 K3_GPU_KDA_ROWS=1024 K3_LOGITS="$SP/ref/big6k_kr1024.f32" > /dev/null; show kr_5862_logits
same big6k_kr1024.f32 big6k_cpu.f32 "5862, KDA pieces of 1,024 rows vs the CPU alone"
