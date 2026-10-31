#!/bin/bash
# suite2.sh TAG -- in the existing worktree wt_TAG: every engine, then test-c with -k (every C
# test, not only up to the first failure) and test-python; logs in suite2_TAG.*
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
tag=$1
export PATH="/c/Users/Cyber/AppData/Local/Nuitka/Nuitka/Cache/downloads/gcc/x86_64/14.2.0posix-19.1.1-12.0.0-msvcrt-r2/mingw64/bin:$PATH"
cd "$SP/wt_$tag/c" || exit 1
echo "== $(git log --oneline -1) $(date +%T)" > "$SP/suite2_$tag.log"
mingw32-make -k -j16 ARCH=native VK=0 kimi_k3 mimo qwen36 qwen38 olmoe glm53 inkling colibri deepseek_v41 > "$SP/suite2_$tag.build" 2>&1
echo "engines rc=$? $(date +%T)" >> "$SP/suite2_$tag.log"
mingw32-make -k -j16 ARCH=native VK=0 test-c > "$SP/suite2_$tag.c" 2>&1
echo "test-c rc=$? $(date +%T)" >> "$SP/suite2_$tag.log"
mingw32-make ARCH=native VK=0 test-python > "$SP/suite2_$tag.py" 2>&1
echo "test-python rc=$? $(date +%T)" >> "$SP/suite2_$tag.log"
echo done >> "$SP/suite2_$tag.log"
