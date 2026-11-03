#!/bin/bash
# hang_check.sh NAME EXE [VAR=value ...] -- the 2,637-token prompt (K3_CUDA_DENSE=0, reads delayed 1.5 ms) with a
# watchdog: still running after 600 s -> every thread's stack (gdb) into runs/NAME.bt, then killed.
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev--claude-worktrees-blissful-tharp-db22d9/ba71ed45-e940-4b2e-9bd9-30cf3f1f6dfd/scratchpad"
REF="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad/ref/long_cpu.f32"
export PATH="/c/Users/Cyber/AppData/Local/Nuitka/Nuitka/Cache/downloads/gcc/x86_64/14.2.0posix-19.1.1-12.0.0-msvcrt-r2/mingw64/bin:$PATH"
name=$1; exe=$2; shift 2
img=$(basename "$exe")
while tasklist 2>/dev/null | grep -qi -E "^(kimi_k3|mimo|colibri|glm)[^ ]*\.exe"; do sleep 20; done
P2637="$(cat "$SP/long_prompt.txt")"
K3_RUN_PROMPT="$P2637" K3_EXE="$exe" bash "$SP/k3run_wt.sh" "$name" 0 K3_CHUNK=8192 K3_MAXT=8192 K3_CUDA_DENSE=0 \
    K3_IO_DELAY_US=1500 K3_LOGITS="$SP/ref/$name.f32" "$@" > /dev/null &
t0=$(date +%s); hung=0
sleep 30
while tasklist 2>/dev/null | grep -qi "^$img"; do
    if [ $(( $(date +%s) - t0 )) -gt 600 ]; then
        pid=$(tasklist 2>/dev/null | grep -i "^$img" | awk '{print $2}' | head -1)
        timeout 300 gdb -p "$pid" -batch -ex "set pagination off" -ex "thread apply all bt 12" > "$SP/runs/$name.bt" 2>&1
        taskkill //F //PID "$pid" > /dev/null 2>&1; hung=1; break
    fi
    sleep 10
done
wait
if [ $hung = 1 ]; then res="HUNG after 600 s ($(grep -c 'k3_io_stop' "$SP/runs/$name.bt") threads in k3_io_stop)"
elif cmp -s "$REF" "$SP/ref/$name.f32"; then res="the CPU's bytes"; rm -f "$SP/ref/$name.f32"
else res="LOGITS DIFFER ($(cmp "$REF" "$SP/ref/$name.f32" 2>&1 | head -1))"; fi
echo "$name: $res | $(grep -h 'K3-CHECK\] the prompt' "$SP/runs/$name.err") | $(grep -h 'prefill done' "$SP/runs/$name.err" | sed 's/.*done in //') | $(grep -o 'waiting for reads [0-9.]*' "$SP/runs/$name.err") | $(grep -h 'overflow' "$SP/runs/$name.err")" >> "$SP/series_ck.txt"
tail -1 "$SP/series_ck.txt"
