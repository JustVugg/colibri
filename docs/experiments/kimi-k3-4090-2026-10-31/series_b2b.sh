#!/bin/bash
# A/B of the expert uploads: dll_sync (each copy waited for) against dll_b2b (back to back), twice
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
C=/c/colibri/dev/.claude/worktrees/kimi-k3-main/c
export K3_RUN_PROMPT="$(cat "$SP/long_prompt.txt")"
for round in 1 2; do
    for v in sync b2b; do
        cp "$SP/dll_$v/coli_cuda_k3.dll" "$C/coli_cuda_k3.dll"
        bash "$SP/k3run.sh" "ab_${v}_$round" 1 K3_CHUNK=4096 > /dev/null
        printf "%s %s: " "$v" "$round"
        grep -h -E 'prefill done' "$SP/runs/ab_${v}_$round.err" | sed 's/.*prefill done in //'
        grep -h -E 'the prompt:' "$SP/runs/ab_${v}_$round.err" | sed 's/.*MoE/  MoE/'
    done
done
cp "$SP/dll_b2b/coli_cuda_k3.dll" "$C/coli_cuda_k3.dll"
