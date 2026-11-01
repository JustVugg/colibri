#!/bin/bash
# A/B of decode: a token's routed experts on the CPU alone (K3_GPU_EXPERT_ROWS at the default 32)
# against the CPU and the GPU together (K3_GPU_EXPERT_ROWS=1), the default 35-token prompt and 48
# tokens, alternating, twice; the texts compared
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
for round in 1 2; do
    for v in cpu gpu; do
        if [ $v = gpu ]; then er=1; else er=32; fi
        bash "$SP/k3run.sh" "dec_${v}_$round" 48 K3_GPU_EXPERT_ROWS=$er > /dev/null
        printf "%s %s: " "$v" "$round"
        grep -h -E '^\[K3\] decode [0-9]+ tokens' "$SP/runs/dec_${v}_$round.err" | sed 's/.*decode //'
        grep -h -E 'a decode token' "$SP/runs/dec_${v}_$round.err" | sed 's/.*a decode token: /  /'
    done
done
for f in dec_gpu_1 dec_cpu_2 dec_gpu_2; do
    if cmp -s <(sed 's/TUNE.*//' "$SP/runs/dec_cpu_1.out") <(sed 's/TUNE.*//' "$SP/runs/$f.out"); then echo "$f: the same text as dec_cpu_1"
    else echo "$f: TEXT DIFFERS from dec_cpu_1"; fi
done
