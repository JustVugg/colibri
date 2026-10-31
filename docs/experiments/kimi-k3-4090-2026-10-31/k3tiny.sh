#!/bin/bash
# k3tiny.sh TAG [EXE] -- kimi_k3_tiny: 300 ids' logits at steps of 1/32/300/2048 against
# ref/lo_old_32.f32 (the HEAD build before the long-prompt work), a 40-token decode against
# ref/gen_old.txt, then the engine's tiny tests.
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
tag=$1; exe=${2:-./kimi_k3.exe}
export PATH="/c/Users/Cyber/AppData/Local/Nuitka/Nuitka/Cache/downloads/gcc/x86_64/14.2.0posix-19.1.1-12.0.0-msvcrt-r2/mingw64/bin:$PATH"
cd /c/colibri/dev/.claude/worktrees/kimi-k3-main/c || exit 1
V=$(python -c "import json;c=json.load(open('kimi_k3_tiny/config.json'));print(c.get('text_config',c)['vocab_size'])")
IDS=$(python -c "print(' '.join(str((5+7*i)%($V-10)+3) for i in range(300)))")
bad=0
for ch in 1 32 300 2048; do
    K3_CHUNK=$ch K3_LOGITS="$SP/ref/lo_${tag}_$ch.f32" AUTOPIN=0 USAGE_SAVE=0 K3_EXPERT_GB=0.00003 \
        "$exe" kimi_k3_tiny --ids "$IDS" --ngen 0 >/dev/null 2>"$SP/ref/err_${tag}_$ch.txt"
    if cmp -s "$SP/ref/lo_${tag}_$ch.f32" "$SP/ref/lo_old_32.f32"; then echo "chunk $ch: logits identical to HEAD"
    else echo "chunk $ch: LOGITS DIFFER"; bad=1; fi
done
AUTOPIN=0 USAGE_SAVE=0 K3_EXPERT_GB=0.00003 "$exe" kimi_k3_tiny --ids "$IDS" --ngen 40 2>/dev/null > "$SP/ref/gen_${tag}.txt"
if cmp -s <(sed 's/TUNE.*//' "$SP/ref/gen_${tag}.txt") <(sed 's/TUNE.*//' "$SP/ref/gen_old.txt"); then echo "decode 40 tokens: identical"
else echo "decode 40 tokens: DIFFER"; bad=1; fi
for t in test_kimi_k3_tiny test_kimi_k3_ckpt test_kimi_k3_dashboard; do
    [ -f tests/$t.py ] && python tests/$t.py --binary "$exe" --fixture kimi_k3_tiny 2>&1 | tail -1
done
exit $bad
