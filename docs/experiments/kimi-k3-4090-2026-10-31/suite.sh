#!/bin/bash
# suite.sh TAG COMMIT -- a detached worktree of COMMIT under the scratchpad, every engine built
# (no CUDA DLL: the C tests link without it), then test-c and test-python; logs in suite_TAG.*
SP="C:/Users/Cyber/AppData/Local/Temp/claude/C--colibri-dev/3e28662a-f5fb-47b0-b9b2-11d1218b91f7/scratchpad"
tag=$1; commit=$2
export PATH="/c/Users/Cyber/AppData/Local/Nuitka/Nuitka/Cache/downloads/gcc/x86_64/14.2.0posix-19.1.1-12.0.0-msvcrt-r2/mingw64/bin:$PATH"
WT="$SP/wt_$tag"
git -C /c/colibri/dev/.claude/worktrees/kimi-k3-main worktree add -f --detach "$WT" "$commit" > "$SP/suite_$tag.log" 2>&1
cd "$WT/c" || exit 1
echo "== build $(date +%T)" >> "$SP/suite_$tag.log"
mingw32-make -j16 ARCH=native VK=0 >> "$SP/suite_$tag.build" 2>&1
echo "build rc=$? $(date +%T)" >> "$SP/suite_$tag.log"
mingw32-make -j16 ARCH=native VK=0 test-c > "$SP/suite_$tag.c" 2>&1
echo "test-c rc=$? $(date +%T)" >> "$SP/suite_$tag.log"
mingw32-make ARCH=native VK=0 test-python > "$SP/suite_$tag.py" 2>&1
echo "test-python rc=$? $(date +%T)" >> "$SP/suite_$tag.log"
echo done >> "$SP/suite_$tag.log"
