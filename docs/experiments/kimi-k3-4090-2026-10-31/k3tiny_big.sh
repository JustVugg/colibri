#!/bin/bash
# k3tiny_big.sh -- kimi_k3_tiny, 6,000 ids: the logits of all positions in steps of 2,048, of
# 4,096 and in one step of 8,192 (past K3_GPU_ROWS: the GPU calls in pieces), every GPU path on
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
export PATH="/c/Users/Cyber/AppData/Local/Nuitka/Nuitka/Cache/downloads/gcc/x86_64/14.2.0posix-19.1.1-12.0.0-msvcrt-r2/mingw64/bin:$PATH"
cd /c/colibri/dev/.claude/worktrees/kimi-k3-main/c || exit 1
V=$(python -c "import json;c=json.load(open('kimi_k3_tiny/config.json'));print(c.get('text_config',c)['vocab_size'])")
IDS=$(python -c "print(' '.join(str((5+7*i+(i*i)%13)%($V-10)+3) for i in range(6000)))")
for ch in 2048 4096 8192; do
    K3_CHUNK=$ch K3_MAXT=8192 K3_GPU_MIN_ROWS=1 K3_CUDA_DENSE=1 K3_LOGITS="$SP/ref/big_$ch.f32" AUTOPIN=0 USAGE_SAVE=0 \
        K3_EXPERT_GB=0.00003 ./kimi_k3.exe kimi_k3_tiny --ids "$IDS" --ngen 0 >/dev/null 2>"$SP/ref/err_big_$ch.txt"
    echo "chunk $ch: rc=$? $(stat -c %s "$SP/ref/big_$ch.f32" 2>/dev/null) bytes; $(grep -h 'prefill done\|K3-CUDA\] dense:\|K3-CUDA\] dense (prompt)\|gated MLA\|routers:' "$SP/ref/err_big_$ch.txt" | tr '\n' ' ' | cut -c1-400)"
done
for ch in 4096 8192; do
    if cmp -s "$SP/ref/big_2048.f32" "$SP/ref/big_$ch.f32"; then echo "chunk $ch: logits identical to chunk 2048"; else echo "chunk $ch: LOGITS DIFFER"; fi
done
