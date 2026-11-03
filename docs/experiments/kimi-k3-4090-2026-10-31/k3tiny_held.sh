#!/bin/bash
# k3tiny_held.sh TAG EXE -- kimi_k3_tiny, 6,000 ids at steps of 2,048, 4,096 and 8,192: the logits of every
# position with every matrix on the card (K3_CUDA_DENSE=1, the GPU forced on every path), with none there,
# and on the CPU alone (no GPU path at all); then 40 tokens of decode the same three ways. Every pair must be
# the same bytes: the card's matrices in the CPU's arithmetic.
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
tag=$1; exe=$2
cd /c/colibri/dev/.claude/worktrees/kimi-k3-main/c || exit 1
V=$(python -c "import json;c=json.load(open('kimi_k3_tiny/config.json'));print(c.get('text_config',c)['vocab_size'])")
IDS=$(python -c "print(' '.join(str((5+7*i+(i*i)%13)%($V-10)+3) for i in range(6000)))")
IDS3=$(python -c "print(' '.join(str((5+7*i)%($V-10)+3) for i in range(300)))")
bad=0
run(){   # name chunk extra-env... -> ref/big_TAG_name_chunk.f32
    local name=$1 ch=$2; shift 2
    env "$@" K3_CHUNK=$ch K3_MAXT=8192 K3_LOGITS="$SP/ref/big_${tag}_${name}_$ch.f32" AUTOPIN=0 USAGE_SAVE=0 \
        K3_EXPERT_GB=0.00003 "$exe" kimi_k3_tiny --ids "$IDS" --ngen 0 >/dev/null 2>"$SP/ref/err_big_${tag}_${name}_$ch.txt"
}
for ch in 2048 4096 8192; do
    run held $ch K3_GPU_MIN_ROWS=1 K3_CUDA_DENSE=1
    run none $ch K3_GPU_MIN_ROWS=1 K3_CUDA_DENSE=0
    run cpu $ch K3_GPU_EXPERTS=0 K3_CUDA_DENSE=0
    h="$SP/ref/big_${tag}_held_$ch.f32"; n="$SP/ref/big_${tag}_none_$ch.f32"; c="$SP/ref/big_${tag}_cpu_$ch.f32"
    sz=$(stat -c %s "$h" 2>/dev/null)
    if [ -n "$sz" ] && [ "$sz" -gt 0 ] && cmp -s "$h" "$n" && cmp -s "$h" "$c"; then
        echo "step $ch: held = none = CPU ($sz bytes); $(grep -h 'K3-CUDA\] dense:' "$SP/ref/err_big_${tag}_held_$ch.txt" | cut -c1-140)"
    else echo "step $ch: LOGITS DIFFER (held $sz bytes; held/none $(cmp "$h" "$n" 2>&1 | head -1); held/cpu $(cmp "$h" "$c" 2>&1 | head -1))"; bad=1; fi
done
for v in held:K3_CUDA_DENSE=1 cpu:K3_GPU_EXPERTS=0; do
    env ${v#*:} AUTOPIN=0 USAGE_SAVE=0 K3_EXPERT_GB=0.00003 "$exe" kimi_k3_tiny --ids "$IDS3" --ngen 40 2>/dev/null \
        | sed 's/TUNE.*//' > "$SP/ref/gen_${tag}_${v%%:*}.txt"
done
if cmp -s "$SP/ref/gen_${tag}_held.txt" "$SP/ref/gen_${tag}_cpu.txt" && [ -s "$SP/ref/gen_${tag}_held.txt" ]; then
    echo "decode 40 tokens: held = CPU"
else echo "decode 40 tokens: DIFFER"; bad=1; fi
exit $bad
