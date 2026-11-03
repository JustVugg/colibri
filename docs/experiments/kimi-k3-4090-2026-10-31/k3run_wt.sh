#!/bin/bash
# k3run.sh NAME NGEN [VAR=value ...] -- one real-model run of kimi_k3 with the 4090 profile (the
# parent session's runner, this worktree's build): stdout/stderr and a 2 s monitor CSV under
# runs/NAME.*; extra VAR=value override the profile.
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev--claude-worktrees-blissful-tharp-db22d9/ba71ed45-e940-4b2e-9bd9-30cf3f1f6dfd/scratchpad"
name=$1; ngen=$2; shift 2
cd /c/colibri/dev/.claude/worktrees/blissful-tharp-db22d9/c || exit 1
export K3_DIRS="D:\\AI\\Models\\Kimi-K3" K3_CUDA=0 K3_CUDA_DENSE=1 K3_CUDA_DENSE_GB=20 K3_CUDA_RESERVE_GB=2 \
       K3_CUDA_RELEASE_HOST=1 K3_EXPERT_GB=1000 RAM_GB=174 K3_MAXT=4096 K3_CHUNK=512 OMP_NUM_THREADS=16 \
       AUTOPIN=0 USAGE_SAVE=0 COLI_VULKAN=0 K3_STAT_EVERY=8
for kv in "$@"; do export "$kv"; done
prompt=${K3_RUN_PROMPT:-"What is the capital of France? Answer in one short sentence."}
rm -f "$SP/runs/$name.stop"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$SP/monitor.ps1" -Out "$SP/runs/$name.csv" \
    -StopFile "$SP/runs/$name.stop" -Seconds 2 &
t0=$(date +%s)
"${K3_EXE:-./kimi_k3.exe}" "C:\\AI\\Kimi-K3" --chat "$prompt" --ngen "$ngen" > "$SP/runs/$name.out" 2> "$SP/runs/$name.err"
rc=$?
echo "rc=$rc wall=$(( $(date +%s) - t0 ))s" >> "$SP/runs/$name.err"
touch "$SP/runs/$name.stop"; wait
echo "$name done rc=$rc"
