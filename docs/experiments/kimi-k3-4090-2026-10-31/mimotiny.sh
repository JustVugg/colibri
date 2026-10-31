#!/bin/bash
# mimotiny.sh TAG EXE [REF] -- the MiMo tiny fixture (scratchpad mimo_tiny): 120 ids' logits at
# steps of 1 and 40 into mimo/lo_TAG_*.f32, compared with mimo_head/lo_old_* (the build before the
# K3 header work) or with mimo/lo_REF_*, then MiMo's tiny harness on EXE.
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
tag=$1; exe=$2; ref=$3
export PATH="/c/Users/Cyber/AppData/Local/Nuitka/Nuitka/Cache/downloads/gcc/x86_64/14.2.0posix-19.1.1-12.0.0-msvcrt-r2/mingw64/bin:$PATH"
cd /c/colibri/dev/.claude/worktrees/kimi-k3-main/c || exit 1
mkdir -p "$SP/mimo"
V=$(python -c "import json;c=json.load(open('$SP/mimo_tiny/config.json'));print(c.get('text_config',c).get('vocab_size',c.get('vocab_size')))")
IDS=$(python -c "print(' '.join(str((5+7*i)%($V-10)+3) for i in range(120)))")
bad=0
for ch in 1 40; do
    MIMO_CHUNK=$ch COLI_CHUNK=$ch MIMO_LOGITS="$SP/mimo/lo_${tag}_$ch.f32" AUTOPIN=0 USAGE_SAVE=0 \
        "$exe" "$SP/mimo_tiny" --ids "$IDS" --ngen 0 >/dev/null 2>"$SP/mimo/err_${tag}_$ch.txt" || { echo "step $ch failed"; tail -3 "$SP/mimo/err_${tag}_$ch.txt"; bad=1; }
    if [ -n "$ref" ]; then r="$SP/mimo/lo_${ref}_$ch.f32"; else r="$SP/mimo_head/lo_old_$ch.f32"; fi
    if cmp -s "$SP/mimo/lo_${tag}_$ch.f32" "$r"; then echo "MiMo tiny, step $ch: logits identical ($(stat -c %s "$r") bytes)"
    else echo "MiMo tiny, step $ch: LOGITS DIFFER from $r"; bad=1; fi
done
python tests/mimo_tiny_harness.py --binary "$exe" --fixture "$SP/mimo_tiny" 2>&1 | tail -2
exit $bad
