#!/bin/bash
# series_mix.sh FIRST LAST -- the hl_2637_none configuration (K3_CHECK on, hashes dumped, logits against the
# CPU-only reference) alternating with timing variants that leave every number the same: a 3 GB read-ahead pool
# (most experts misses: the layer cache's slots reused within every step), CPU waves of one expert, reads
# delayed 1.5 ms. Waits while another kimi_k3 / mimo process runs.
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev--claude-worktrees-blissful-tharp-db22d9/ba71ed45-e940-4b2e-9bd9-30cf3f1f6dfd/scratchpad"
REF="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad/ref/long_cpu.f32"
first=$1; last=$2
P2637="$(cat "$SP/long_prompt.txt")"
variants=("" "K3_READAHEAD_GB=3" "" "K3_GPU_CPU_WAVE=1" "" "K3_IO_DELAY_US=1500")
tags=("as" "ra3" "as" "wave1" "as" "iod")
for i in $(seq "$first" "$last"); do
    while tasklist 2>/dev/null | grep -qi -E "^(kimi_k3|mimo|colibri|glm)[^ ]*\.exe"; do sleep 20; done
    k=$(( i % 6 )); v=${variants[$k]}; name="mx_${i}_${tags[$k]}"
    K3_RUN_PROMPT="$P2637" K3_EXE=./kimi_k3_ck.exe bash "$SP/k3run_wt.sh" "$name" 0 K3_CHUNK=8192 K3_MAXT=8192 K3_CUDA_DENSE=0 \
        K3_CHECK=1 K3_CHECK_DUMP="$SP/runs/$name.ck" K3_LOGITS="$SP/ref/$name.f32" $v > /dev/null
    if cmp -s "$REF" "$SP/ref/$name.f32"; then res="the CPU's bytes"; rm -f "$SP/ref/$name.f32"
    else res="LOGITS DIFFER ($(cmp "$REF" "$SP/ref/$name.f32" 2>&1 | head -1))"; fi
    echo "$name: $res | $(grep -h 'K3-CHECK\] the prompt' "$SP/runs/$name.err") | $(grep -h 'routed experts on the GPU' "$SP/runs/$name.err" | sed 's/.*experts: //' | cut -c1-80) | $(grep -h 'prefill done' "$SP/runs/$name.err" | sed 's/.*done in //')" >> "$SP/series_ck.txt"
done
echo "series mix $first..$last done" >> "$SP/series_ck.txt"
