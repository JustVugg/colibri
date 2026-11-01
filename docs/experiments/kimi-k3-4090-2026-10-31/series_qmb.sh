#!/bin/bash
# A/B of the resident matrices' kernel: quant_matmul_blocked (COLI_CUDA_QMB unset) against
# quant_matmul_tiled (COLI_CUDA_QMB=0), the same coli_cuda.dll, alternating, twice
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
export K3_RUN_PROMPT="$(cat "$SP/long_prompt.txt")"
for round in 1 2; do
    for v in qmb tiled; do
        if [ $v = tiled ]; then q=0; else q=1; fi
        bash "$SP/k3run.sh" "ab3_${v}_$round" 1 K3_CHUNK=4096 COLI_CUDA_QMB=$q > /dev/null
        printf "%s %s: " "$v" "$round"
        grep -h -E 'prefill done' "$SP/runs/ab3_${v}_$round.err" | sed 's/.*prefill done in //'
        grep -h -E 'the prompt:' "$SP/runs/ab3_${v}_$round.err" | sed 's/.*the prompt: /  /'
        grep -h -E 'dense \(prompt\)' "$SP/runs/ab3_${v}_$round.err"
    done
done
