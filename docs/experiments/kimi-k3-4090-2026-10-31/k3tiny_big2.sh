#!/bin/bash
# k3tiny_big2.sh TAG EXE -- kimi_k3_tiny, 6,000 ids: the logits of all positions in steps of 2,048, of
# 4,096 and in one step of 8,192 (the GPU forced on every path) into ref/big_TAG_STEP.f32
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
tag=$1; exe=$2
cd /c/colibri/dev/.claude/worktrees/kimi-k3-main/c || exit 1
V=$(python -c "import json;c=json.load(open('kimi_k3_tiny/config.json'));print(c.get('text_config',c)['vocab_size'])")
IDS=$(python -c "print(' '.join(str((5+7*i+(i*i)%13)%($V-10)+3) for i in range(6000)))")
for ch in 2048 4096 8192; do
    K3_CHUNK=$ch K3_MAXT=8192 K3_GPU_MIN_ROWS=1 K3_CUDA_DENSE=1 K3_LOGITS="$SP/ref/big_${tag}_$ch.f32" AUTOPIN=0 USAGE_SAVE=0 \
        K3_EXPERT_GB=0.00003 "$exe" kimi_k3_tiny --ids "$IDS" --ngen 0 >/dev/null 2>"$SP/ref/err_big_${tag}_$ch.txt"
    echo "$tag chunk $ch: rc=$? $(stat -c %s "$SP/ref/big_${tag}_$ch.f32" 2>/dev/null) bytes; $(grep -h 'prefill done\|gated MLA' "$SP/ref/err_big_${tag}_$ch.txt" | tr '\n' ' ' | cut -c1-300)"
done
