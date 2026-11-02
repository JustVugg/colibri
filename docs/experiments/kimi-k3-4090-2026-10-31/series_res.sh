#!/bin/bash
# The VRAM reserve beside the resident matrices: 2 GB (19.5 -> 20 GiB resident) against 3 GB
# (19.5 GiB): decode (48 tokens, alternating, twice), then the 5,862-token prompt at 3 GB
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
for round in 1 2; do
    for r in 2 3; do
        bash "$SP/k3run.sh" "decr${r}_$round" 48 K3_CHUNK=8192 K3_MAXT=8192 K3_CUDA_RESERVE_GB=$r > /dev/null
        printf "reserve %s GB, %s: " "$r" "$round"
        grep -h -o 'dense: [0-9]* of 929 matrices on GPU 0, [0-9.]* of' "$SP/runs/decr${r}_$round.err" | tr '\n' ' '
        grep -h -E '^\[K3\] decode [0-9]+ tokens' "$SP/runs/decr${r}_$round.err" | sed 's/.*decode //'
    done
done
for f in decr3_1 decr2_2 decr3_2; do
    if cmp -s <(sed 's/TUNE.*//' "$SP/runs/decr2_1.out") <(sed 's/TUNE.*//' "$SP/runs/$f.out"); then echo "$f: the same text"; else echo "$f: TEXT DIFFERS"; fi
done
K3_RUN_PROMPT="$(cat "$SP/long6k_prompt.txt")" bash "$SP/k3run.sh" kb5862_res3 1 K3_CHUNK=8192 K3_MAXT=8192 K3_CUDA_RESERVE_GB=3 > /dev/null
printf "5862 at 3 GB: "; grep -h -E 'prefill done' "$SP/runs/kb5862_res3.err" | sed 's/.*prefill done in //'
grep -h -E 'the prompt:|the dense matrices the GPU does not' "$SP/runs/kb5862_res3.err"
K3_RUN_PROMPT="$(cat "$SP/long_prompt.txt")" bash "$SP/k3run.sh" kb2637_res3b 1 K3_CHUNK=8192 K3_MAXT=8192 K3_CUDA_RESERVE_GB=3 > /dev/null
printf "2637 at 3 GB, the launcher's step: "; grep -h -E 'prefill done' "$SP/runs/kb2637_res3b.err" | sed 's/.*prefill done in //'
grep -h -E 'the prompt:' "$SP/runs/kb2637_res3b.err"
