#!/bin/bash
# Decode with the RAM a step of 4,096 reserves for activations against a step of 8,192's (the
# expert cache a few slots a layer smaller): the default 35-token prompt and 48 tokens,
# alternating, twice; the texts compared
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
for round in 1 2; do
    for ch in 4096 8192; do
        bash "$SP/k3run.sh" "decc_${ch}_$round" 48 K3_CHUNK=$ch K3_MAXT=8192 > /dev/null
        printf "chunk %s, %s: " "$ch" "$round"
        grep -h -o 'cache 619->[0-9]*/layer' "$SP/runs/decc_${ch}_$round.err" | head -1 | tr '\n' ' '
        grep -h -E '^\[K3\] decode [0-9]+ tokens' "$SP/runs/decc_${ch}_$round.err" | sed 's/.*decode //'
    done
done
for f in decc_8192_1 decc_4096_2 decc_8192_2; do
    if cmp -s <(sed 's/TUNE.*//' "$SP/runs/decc_4096_1.out") <(sed 's/TUNE.*//' "$SP/runs/$f.out"); then echo "$f: the same text"
    else echo "$f: TEXT DIFFERS"; fi
done
