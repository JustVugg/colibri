#!/bin/bash
# series_pair.sh FIRST LAST -- the parent's sequence around hl_2637_none: the 2,637-token prompt's logits with the
# default profile (the card's matrices; its logits file left in place while the next run starts, as there), then
# at once the same prompt with every matrix the host's (K3_CUDA_DENSE=0, K3_CHECK on, hashes dumped); odd pairs
# with one reader a drive in the second run. Every logits file against the CPU-only reference.
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev--claude-worktrees-blissful-tharp-db22d9/ba71ed45-e940-4b2e-9bd9-30cf3f1f6dfd/scratchpad"
REF="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad/ref/long_cpu.f32"
first=$1; last=$2
P2637="$(cat "$SP/long_prompt.txt")"
verdict(){   # name
    if cmp -s "$REF" "$SP/ref/$1.f32"; then res="the CPU's bytes"; rm -f "$SP/ref/$1.f32"
    else res="LOGITS DIFFER ($(cmp "$REF" "$SP/ref/$1.f32" 2>&1 | head -1))"; fi
    echo "$1: $res | $(grep -h 'K3-CHECK\] the prompt' "$SP/runs/$1.err") | $(grep -h 'routed experts on the GPU' "$SP/runs/$1.err" | sed 's/.*experts: //' | cut -c1-80) | $(grep -h 'prefill done' "$SP/runs/$1.err" | sed 's/.*done in //') | $(grep -o 'waiting for reads [0-9.]*' "$SP/runs/$1.err")" >> "$SP/series_ck.txt"
}
for i in $(seq "$first" "$last"); do
    while tasklist 2>/dev/null | grep -qi -E "^(kimi_k3|mimo|colibri|glm)[^ ]*\.exe"; do sleep 20; done
    d="pd_$i"; n="pn_$i"; extra=""
    [ $(( i % 2 )) -eq 1 ] && { extra="K3_IO_PER_DRIVE=1"; n="pn_${i}_r1"; }
    K3_RUN_PROMPT="$P2637" K3_EXE=./kimi_k3.exe bash "$SP/k3run_wt.sh" "$d" 0 K3_CHUNK=8192 K3_MAXT=8192 \
        K3_LOGITS="$SP/ref/$d.f32" > /dev/null
    K3_RUN_PROMPT="$P2637" K3_EXE=./kimi_k3.exe bash "$SP/k3run_wt.sh" "$n" 0 K3_CHUNK=8192 K3_MAXT=8192 K3_CUDA_DENSE=0 \
        K3_CHECK=1 K3_CHECK_DUMP="$SP/runs/$n.ck" K3_LOGITS="$SP/ref/$n.f32" $extra > /dev/null
    verdict "$d"; verdict "$n"
done
echo "series pair $first..$last done" >> "$SP/series_ck.txt"
