#!/bin/bash
# series_ck.sh PREFIX FIRST LAST [VAR=value ...] -- the 2,637-token prompt's logits at the launcher's 8,192 step with
# every dense matrix the host's (K3_CUDA_DENSE=0: the configuration of hl_2637_none), K3_CHECK on and its hashes
# dumped; each run's logits against the CPU-only reference (kept only when they differ). Waits while another
# kimi_k3 / mimo process runs (a shared box).
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev--claude-worktrees-blissful-tharp-db22d9/ba71ed45-e940-4b2e-9bd9-30cf3f1f6dfd/scratchpad"
REF="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad/ref/long_cpu.f32"
pre=$1; first=$2; last=$3; shift 3
P2637="$(cat "$SP/long_prompt.txt")"
for i in $(seq "$first" "$last"); do
    while tasklist 2>/dev/null | grep -qi -E "^(kimi_k3|mimo|colibri|glm)[^ ]*\.exe"; do sleep 20; done
    name="${pre}_$i"
    K3_RUN_PROMPT="$P2637" bash "$SP/k3run_wt.sh" "$name" 0 K3_CHUNK=8192 K3_MAXT=8192 K3_CUDA_DENSE=0 K3_CHECK=1 \
        K3_CHECK_DUMP="$SP/runs/$name.ck" K3_LOGITS="$SP/ref/$name.f32" "$@" > /dev/null
    if cmp -s "$REF" "$SP/ref/$name.f32"; then res="the CPU's bytes"; rm -f "$SP/ref/$name.f32"
    else res="LOGITS DIFFER ($(cmp "$REF" "$SP/ref/$name.f32" 2>&1 | head -1))"; fi
    echo "$name: $res | $(grep -h 'K3-CHECK\] the prompt' "$SP/runs/$name.err") | $(grep -h 'routed experts on the GPU' "$SP/runs/$name.err" | sed 's/.*experts: //' | cut -c1-80) | $(grep -h 'prefill done' "$SP/runs/$name.err" | sed 's/.*done in //')" >> "$SP/series_ck.txt"
done
echo "series $pre $first..$last done" >> "$SP/series_ck.txt"
