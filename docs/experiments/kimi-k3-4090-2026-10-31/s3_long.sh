#!/bin/bash
# s3_long.sh TAG -- the 2,637-token prompt twice with the current c/ binaries: the logits of
# all positions against ref/long_ref.f32, then a timed run (one token generated)
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
tag=$1; shift
export K3_RUN_PROMPT="$(cat "$SP/long_prompt.txt")"
bash "$SP/k3run.sh" "s3_${tag}_logits" 0 K3_CHUNK=4096 K3_LOGITS="$SP/ref/long_${tag}.f32" "$@" > /dev/null
if cmp -s "$SP/ref/long_${tag}.f32" "$SP/ref/long_ref.f32"; then echo "$tag: all 2,637 positions' logits identical to long_ref"
else echo "$tag: LOGITS DIFFER from long_ref"; cmp "$SP/ref/long_${tag}.f32" "$SP/ref/long_ref.f32" | head -2; fi
bash "$SP/k3run.sh" "s3_${tag}_time" 1 K3_CHUNK=4096 "$@" > /dev/null
for r in logits time; do
    printf "%s %s: " "$tag" "$r"
    grep -h -E 'prefill done' "$SP/runs/s3_${tag}_$r.err" | sed 's/.*prefill done in //'
    grep -h -E 'the prompt:|dense \(prompt\)|the dense matrices the GPU does not hold' "$SP/runs/s3_${tag}_$r.err"
done
