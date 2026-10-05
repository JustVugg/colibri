#!/usr/bin/env bash
# Every engine's Vulkan path against its own CPU run, on Lavapipe (Mesa's software
# Vulkan), one family per call so CI can run them side by side:
#
#   bash tests/vulkan_engines.sh qwen | qwen-sanitize | inkling-olmoe | inkling-olmoe-sanitize
#   bash tests/vulkan_engines.sh mimo-qwenimage | kimi | kimi-mimo-sanitize | deepseek | deepseek-sanitize
#   bash tests/vulkan_engines.sh glm | glm-sanitize   # GLM-5.2 (colibri) and GLM-5.3 Flash (glm53)
#   bash tests/vulkan_engines.sh shader    # the qmatmul formats, the expert batch and the tier, no engine
#   bash tests/vulkan_engines.sh qwen-chain | qwen-chain-sanitize   # the dense chain (COLI_VK_CHAIN=1)
#   bash tests/vulkan_engines.sh qwen-spec | qwen-spec-sanitize     # MTP and prompt-lookup verifies (docs/speculative.md)
#   bash tests/vulkan_engines.sh mimo-chain | mimo-chain-sanitize   # MiMo-V2.6's dense chain
#   bash tests/vulkan_engines.sh inkling-olmoe-chain | inkling-olmoe-chain-sanitize
#   bash tests/vulkan_engines.sh glm-chain | glm-chain-sanitize     # the same for colibri and glm53
#   bash tests/vulkan_engines.sh kimi-chain | kimi-chain-sanitize   # the same for Kimi K3
#   bash tests/vulkan_engines.sh deepseek-chain | deepseek-chain-sanitize   # deepseek_v41 and deepseek_v4
#   bash tests/vulkan_engines.sh prefill-qwen | prefill-qwen-sanitize | prefill-inkling-olmoe | prefill-mimo-kimi
#   bash tests/vulkan_engines.sh prefill-glm | prefill-deepseek   # big prompt chunks and expert streaming
#   bash tests/vulkan_engines.sh kv-split | kv-split-sanitize   # the KV cache split between the device and
#                                                  # the host past the device's budget, every chain engine
#   bash tests/vulkan_engines.sh kv-split-deepseek | kv-split-deepseek-sanitize   # the same for DeepSeek's
#   bash tests/vulkan_engines.sh staged    # staged uploads: the same bits as mapped memory, the decision
#   bash tests/vulkan_engines.sh <family>-staged   # a family with COLI_VK_STAGED=1 (qwen-staged: and
#                                                  # an engine under an emulated small window)
#   bash tests/vulkan_engines.sh staged-faults | staged-faults-sanitize   # staged uploads failing
#   bash tests/vulkan_engines.sh dense-only-<group> | dense-only-<group>-sanitize   # the dense weights
#        on the device only (COLI_VK_DENSE_HOST=0), tests/vulkan_dense_only_<group>.sh
#   bash tests/vulkan_engines.sh partial-<group> | partial-<group>-sanitize   # the partial chain:
#        the first N layers on the device (COLI_VK_CHAIN_LAYERS, COLI_VK_DEVICE_CAP_MB),
#        tests/vulkan_partial_<group>.sh
#   bash tests/vulkan_engines.sh decide | decide-sanitize   # Laya, GLiNER2.5-Decide and Clef's DECIDE
#   bash tests/vulkan_engines.sh dev2 | dev2-deepseek-kimi-mimo | dev2-sanitize   # the expert tier on
#        two devices (COLI_VK_DEV2), every MoE engine, tests/vulkan_dev2.sh
#
# Needs libvulkan-dev, glslc and mesa-vulkan-drivers, plus the Python packages of
# the family's tiny fixtures (see the vulkan-engines job in .github/workflows/ci.yml).
#
# Lavapipe is a CPU rasteriser: nothing here says anything about speed. What it does
# prove is that every resident format an engine uploads reaches the shader and comes
# back as the CPU computes it. Each configuration is gated on two things:
#   - the Vulkan run gives the tokens of the CPU run with the same snapshot and
#     settings (and, where the engine has one, passes its own oracle);
#   - its "[VK] <engine>: N matmuls on the GPU" line has N > 0, because a hook that
#     declines every matrix would otherwise pass the first gate trivially. A
#     configuration of the routed-expert tier with the dense trunk on the CPU (the
#     default on Lavapipe while the tier is on, see dense_where) gates on its count of
#     routed experts the device served instead.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}
export COLI_NO_OMP_TUNE=1
PY=${PY:-python3}
# Speculative decoding is on by default (MTP and prompt lookup, docs/speculative.md): its
# verifies change how many rows a forward dumps and when (the gate decides from measured
# times), which the families' row-by-row logits comparisons do not expect. They run with it
# off; the cases that test it turn it on themselves (Q38_MTP=1, COLI_LOOKUP=1, DRAFT=n).
export COLI_LOOKUP=${COLI_LOOKUP:-0} Q38_MTP=${Q38_MTP:-0}

fail() { echo "FAIL: $*"; exit 1; }

# Every sanitized run below sets ASAN_OPTIONS with detect_stack_use_after_return=0:
# ASan's fake stack frames are only 32-byte aligned, and on an AVX-512 runner
# (-march=native) GCC keeps 64-byte aligned locals there and stores them with
# vmovdqa64, which faults (c/Makefile, ASAN_ENV, has the details).

# vk_count <engine> <log>: N from the last "[VK] <engine>: N matmuls on the GPU" line
vk_count() {
  local n
  n=$(sed -n "s/^\[VK\] $1: \([0-9][0-9]*\) matmuls on the GPU.*/\1/p" "$2" | tail -1)
  echo "${n:-0}"
}
need_gpu() {  # <engine> <log> <tag>
  [ "$(vk_count "$1" "$2")" -gt 0 ] || { cat "$2"; fail "$3: no matmul ran on the device"; }
}
same_tokens() {  # <cpu log> <vk log> <tag>: the engines' "C engine" token lines
  grep -a '^C engine' "$1" > cpu.tok || true
  grep -a '^C engine' "$2" > vk.tok || true
  { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; fail "$3: Vulkan tokens differ from the CPU"; }
}

# The shader itself: every weight format against a CPU reference, before any engine;
# then the expert batch and the weight pool in the same harness, and the routed-expert
# tier (vk_tier.c) on a synthetic model in every source format.
shader_formats() {
  make tests/test_vk_tier VK=1   # every shader too: the harness's expert batch needs them
  cc -O2 -pthread -DVK_TEST backend_vulkan.c -o vk_test -lvulkan -lm
  COLI_VK_TEST_MATMUL_ONLY=1 ./vk_test shaders/qmatmul.spv | tee vk_test.log
  tail -1 vk_test.log | grep -qx PASS || fail "qmatmul format cases"
  ./tests/test_vk_tier shaders/qmatmul.spv | tee vk_tier.log
  tail -1 vk_tier.log | grep -qx PASS || fail "routed-expert tier"
}

# Staged uploads (docs/vulkan.md, "Memory placement without Resizable BAR"): resident
# data copied from a host staging buffer into device-local memory the host does not map,
# as a discrete card without Resizable BAR needs, against the mapped path on this device:
#   - the harness's results (every format, the tiled GEMMs, the expert batch) bit for bit
#     the same mapped (COLI_VK_STAGED=0), staged (=1, twice: no run-to-run difference) and
#     under an emulated 246 MB host-visible window (COLI_VK_HOST_VISIBLE_CAP_MB=246 and
#     nothing else), which must choose staging on its own;
#   - the routed-expert tier (test_vk_tier) and the chain's ops (test_vk_chain) staged;
#   - each staged run ends with no resident data in host memory ("[VK] memory at exit").
staged_check() {  # <err log> <tag>: staged, and nothing resident left in host memory
  grep -q '^\[VK\] memory: staged uploads' "$1" || { cat "$1"; fail "$2: not staged"; }
  grep -q 'resident data in host memory: 0.0 MiB' "$1" || { grep '\[VK\] memory' "$1"; fail "$2: resident data in host memory"; }
}
family_staged() {
  make tests/test_vk_tier tests/test_vk_chain VK=1   # the shaders too
  cc -O2 -pthread -DVK_TEST backend_vulkan.c -o vk_test -lvulkan -lm
  local m e d d0=""
  for m in mapped staged staged-again staged-noimport window; do
    case $m in
      mapped) e=COLI_VK_STAGED=0 ;;
      staged|staged-again) e=COLI_VK_STAGED=1 ;;
      staged-noimport) e="COLI_VK_STAGED=1 COLI_VK_UP_IMPORT=0" ;;   # every copy through the staging buffer
      window) e=COLI_VK_HOST_VISIBLE_CAP_MB=246 ;;
    esac
    env -u COLI_VK_STAGED -u COLI_VK_HOST_VISIBLE_CAP_MB $e COLI_VK_TEST_MATMUL_ONLY=1 ./vk_test shaders/qmatmul.spv > vk_test.log 2> vk_test.err
    tail -1 vk_test.log | grep -qx PASS || { cat vk_test.log vk_test.err; fail "staged harness, $m"; }
    if [ $m = mapped ]; then grep -qx 'memory: mapped' vk_test.log || { cat vk_test.err; fail "staged harness: COLI_VK_STAGED=0 staged"; }
    else staged_check vk_test.err "staged harness, $m"; fi
    d=$(sed -n 's/^outputs digest //p' vk_test.log)
    [ -n "$d" ] || fail "staged harness, $m: no digest"
    [ -n "$d0" ] || d0=$d
    [ "$d" = "$d0" ] || fail "staged harness, $m: the results differ from the mapped run's ($d against $d0)"
    echo "OK staged harness $m: $(grep '^memory:' vk_test.log), results digest $d"
  done
  COLI_VK_STAGED=1 ./tests/test_vk_tier shaders/qmatmul.spv > vk_tier.log 2> vk_tier.err
  tail -1 vk_tier.log | grep -qx PASS || { cat vk_tier.log vk_tier.err; fail "staged routed-expert tier"; }
  staged_check vk_tier.err "staged routed-expert tier"
  echo "OK staged routed-expert tier: $(grep -o 'resident data in host memory: .*' vk_tier.err)"
  COLI_VK_STAGED=1 COLI_VK_UP_IMPORT=0 ./tests/test_vk_tier shaders/qmatmul.spv > vk_tier.log 2> vk_tier.err
  tail -1 vk_tier.log | grep -qx PASS || { cat vk_tier.log vk_tier.err; fail "staged routed-expert tier, no imports"; }
  echo "OK staged routed-expert tier, every copy staged (COLI_VK_UP_IMPORT=0)"
  COLI_VK_STAGED=1 ./tests/test_vk_chain shaders/qmatmul.spv > vk_chain.log 2> vk_chain.err
  tail -1 vk_chain.log | grep -qx PASS || { cat vk_chain.log vk_chain.err; fail "staged chain ops"; }
  staged_check vk_chain.err "staged chain ops"
  echo "OK staged chain ops: $(grep -o 'resident data in host memory: .*' vk_chain.err)"
  staged_import
}
# imported_copies <log>: the copies the exit report says came straight from host memory
imported_copies() { sed -n 's/.* MiB of them straight from host memory in \([0-9]*\) copies.*/\1/p' "$1" | tail -1; }
# Staged uploads straight from host memory (up_import, VK_EXT_external_memory_host): the
# tier's experts from the host image they are converted in, the trunk's rows from the
# weights. The same tokens and logits as every copy through the staging buffer
# (COLI_VK_UP_IMPORT=0); the exit report counts the imported copies; an import the driver
# refuses (COLI_VK_STAGED_FAULT=import) is staged instead.
staged_import() {
  make qwen36 VK=1
  [ -f qwen36_tiny/ref_full.json ] && [ -d qwen36_tiny_c ] || {
    $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
    $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8; }
  local i n f
  SNAP=qwen36_tiny_c COLI_DENSE_I8=0 ./qwen36 8 8 qwen36_tiny/ref_full.json > cpu.log 2>&1 || true
  COLI_VULKAN=1 COLI_VK_STAGED=1 SNAP=qwen36_tiny_c ./qwen36 1 8 qwen36_tiny/ref_full.json > imp.log 2>&1 || true
  grep -q '^\[VK\] ready' imp.log || { cat imp.log; fail "staged imports: the probe did not open the device"; }
  if ! grep -q '^\[VK\] staged uploads: copied straight from host memory' imp.log; then
    echo "OK staged imports: this driver has no VK_EXT_external_memory_host, every copy is staged"; return 0
  fi
  for i in 1 0; do
    rm -f imp.usage
    env COLI_USAGE=imp.usage COLI_VULKAN=1 COLI_VK_STAGED=1 COLI_VK_UP_IMPORT=$i COLI_VK_DENSE=1 COLI_VK_TIER_SYNC=1 \
      COLI_DENSE_I8=0 DUMP=imp$i.f32 SNAP=qwen36_tiny_c ./qwen36 8 8 qwen36_tiny/ref_full.json > imp$i.log 2>&1 || true
    same_tokens cpu.log imp$i.log "staged imports COLI_VK_UP_IMPORT=$i"
    staged_check imp$i.log "staged imports COLI_VK_UP_IMPORT=$i"
  done
  cmp -s imp1.f32 imp0.f32 || fail "staged imports: the logits differ from every copy staged"
  n=$(imported_copies imp1.log); [ "${n:-0}" -gt 0 ] || { grep '\[VK\] memory' imp1.log; fail "staged imports: nothing imported"; }
  [ "$(imported_copies imp0.log)" = 0 ] || { grep '\[VK\] memory' imp0.log; fail "staged imports: COLI_VK_UP_IMPORT=0 imported"; }
  echo "OK staged imports: tokens = CPU, logits = every copy staged, $n copies straight from host memory"
  rm -f imp.usage
  env COLI_USAGE=imp.usage COLI_VULKAN=1 COLI_VK_STAGED=1 COLI_VK_STAGED_FAULT=import:1 COLI_VK_DENSE=1 COLI_VK_TIER_SYNC=1 \
    COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 qwen36_tiny/ref_full.json > impf.log 2>&1 || true
  grep -q "COLI_VK_STAGED_FAULT: import #1 fails" impf.log || { cat impf.log; fail "staged imports: the import fault never fired"; }
  same_tokens cpu.log impf.log "staged imports, a refused import"
  f=$(sed -n 's/.* copies (\([0-9]*\) not imported).*/\1/p' impf.log | tail -1)
  [ "${f:-0}" -ge 1 ] || { grep '\[VK\] memory' impf.log; fail "staged imports: the refused import is not counted"; }
  echo "OK staged imports, a refused import: staged instead, tokens = CPU"
  rm -f imp.usage imp0.f32 imp1.f32
}
# An engine under the emulated window, COLI_VK_STAGED unset: it stages on its own, the
# tier with the trunk on the device and the chain give the CPU's tokens (and logits),
# and nothing resident ends in host memory.
staged_window() {
  make qwen36 VK=1
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8
  tier_gate qwen36 "window qwen36 tier and trunk" COLI_VK_HOST_VISIBLE_CAP_MB=246 COLI_VK_DENSE=1 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 qwen36_tiny/ref_full.json
  staged_check vk.log "window qwen36 tier and trunk"
  chain_gate qwen36 "window qwen36 chain" 1 COLI_VK_HOST_VISIBLE_CAP_MB=246 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 qwen36_tiny/ref_full.json
  staged_check vk.log "window qwen36 chain"
  echo "OK window qwen36: staged on its own, $(grep -o 'resident data in host memory: .*' vk.log)"
}

# Staged uploads failing (COLI_VK_STAGED_FAULT=<point>[:n], the n-th time) at every point
# they can: the uploader's staging buffer (stage), the KV mirror's (pwstage), a weight
# pool's device-local block (block), a KV mirror or norm-weight buffer (kvbuf), a command
# buffer's begin or end (record), a submit, a fence wait (the device is then lost), a tier
# expert's commit after its first matrix (commit). qwen36 with its trunk's matrices on the
# device one by one, its tier awaited (COLI_VK_TIER_SYNC=1) and with the uploader thread
# free after a warm start, and its chain; colibri's attention core for the KV mirror. Every
# run: no crash and no sanitizer report, the fault fired, the CPU's tokens. A matrix that
# failed stays on the CPU (one fewer resident); an expert whose commit failed stays on the
# CPU and the tier's budget stands; a lost device takes everything to the CPU, the chain
# rebuilding its state there. SAN=1 (staged-faults-sanitize): a sanitized build.
fault_run() {  # <tag> <fault> <cpu tokens> <env...> -- <command...>
  local tag=$1 fault=$2 ref=$3 rc=0; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  if [ -f hist.usage ]; then cp hist.usage run.usage; else rm -f run.usage; fi
  env "${envs[@]}" COLI_USAGE=run.usage COLI_VULKAN=1 COLI_VK_STAGED=1 ${fault:+COLI_VK_STAGED_FAULT=$fault} "$@" > vk.log 2>&1 || rc=$?
  if [ $rc -ge 128 ] || grep -qE "ERROR: AddressSanitizer|runtime error:" vk.log; then cat vk.log; fail "$tag: crashed (exit $rc)"; fi
  [ -z "$fault" ] || grep -q "COLI_VK_STAGED_FAULT: ${fault%%:*} " vk.log || { cat vk.log; fail "$tag: the fault never fired"; }
  grep -a 'C engine' vk.log > vk.tok || true
  { [ -s "$ref" ] && cmp -s "$ref" vk.tok; } || { cat "$ref" vk.tok; fail "$tag: tokens differ from the CPU's"; }
}
fault_num() {  # <sed expression> <log>: the first number it extracts, 0 without one
  local n; n=$(sed -n "$1" "$2" | tail -1); echo "${n:-0}"
}
family_staged_faults() {
  if [ "${SAN:-0}" = 1 ]; then
    make clean >/dev/null 2>&1 || true
    make qwen36 colibri VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
    export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  else
    make qwen36 colibri VK=1
  fi
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8
  mkdir -p glm_fp8 && (cd glm_fp8 && $PY ../tools/make_glm_oracle.py --fp8 > /dev/null)
  $PY tools/convert_fp8_to_int4.py --indir glm_fp8/glm_tiny --outdir glm_tiny_i4 --ebits 4 --io-bits 4 \
    --n-layers 5 --min-free-gb 0 > /dev/null
  cp glm_fp8/ref_glm.json glm_tiny_i4/
  local q=(COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 qwen36_tiny/ref_full.json)
  local g=(IDOT=0 SNAP=glm_tiny_i4 REF=glm_tiny_i4/ref_glm.json ./colibri 2 4 4)
  env "${q[@]}" > cpu.log 2>&1 || true; grep -a 'C engine' cpu.log > q.tok || true
  env "${g[@]}" > cpu.log 2>&1 || true; grep -a 'C engine' cpu.log > g.tok || true
  local f n S
  rm -f hist.usage
  # the trunk's matrices one by one (no tier, no chain)
  local D=(COLI_VK_TIER=0 COLI_VK_CHAIN=0 COLI_VK_DENSE=1)
  fault_run "faults: qwen36 matrices, none" "" q.tok "${D[@]}" -- "${q[@]}"
  n=$(fault_num 's/.*matmuls on the GPU (\([0-9]*\) matrices.*/\1/p' vk.log)
  for f in stage:1 block:1 record:4 submit:4 wait:1 wait:4; do
    fault_run "faults: qwen36 matrices, $f" $f q.tok "${D[@]}" -- "${q[@]}"
    case $f in block:1|record:4|submit:4)
      [ "$(fault_num 's/.*matmuls on the GPU (\([0-9]*\) matrices.*/\1/p' vk.log)" = $((n - 1)) ] || {
        grep '\[VK\]' vk.log; fail "faults: qwen36 matrices, $f: not exactly one matrix left on the CPU"; } ;;
      wait:*) grep -q 'the device is lost' vk.log || { cat vk.log; fail "faults: qwen36 matrices, $f: the device was not lost"; } ;;
    esac
    echo "OK faults: qwen36 matrices, $f: tokens = CPU, $(grep -ao 'staged upload failed[^:]*: [^)]*): .*' vk.log | head -1 | sed 's/.*): //')"
  done
  # the tier awaited: the trunk on the CPU, every upload at the next step
  local T=(COLI_VK_CHAIN=0 COLI_VK_DENSE=0 COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0)
  fault_run "faults: qwen36 tier, none" "" q.tok "${T[@]}" -- "${q[@]}"
  local budget; budget=$(fault_num 's/.*resident [0-9]* (budget \([0-9]*\),.*/\1/p' vk.log)
  cp run.usage hist.usage.next
  for f in stage:1 block:1 record:3 record:4 submit:2 wait:2 commit:1 commit:3; do
    fault_run "faults: qwen36 tier, $f" $f q.tok "${T[@]}" -- "${q[@]}"
    case $f in commit:*|record:*|submit:*)
      [ "$(fault_num 's/.*resident [0-9]* (budget \([0-9]*\),.*/\1/p' vk.log)" = "$budget" ] || {
        grep '\[VK\] tier' vk.log; fail "faults: qwen36 tier, $f: the budget shrank"; } ;;
    esac
    echo "OK faults: qwen36 tier, $f: tokens = CPU, $(grep -ao 'resident [0-9]* (budget [0-9]*' vk.log | tail -1), $(grep -ao 'failed [0-9]* |' vk.log | tail -1 | tr -d '|')"
  done
  # the tier's uploader thread free: a warm start from a history (vkt_put, any thread),
  # then promotions as experts pass
  mv hist.usage.next hist.usage
  T=(COLI_VK_CHAIN=0 COLI_VK_DENSE=0)
  for f in stage:1 block:1 record:3 submit:2 wait:3 commit:1 commit:5; do
    fault_run "faults: qwen36 tier, uploader thread, $f" $f q.tok "${T[@]}" -- "${q[@]}"
    echo "OK faults: qwen36 tier, uploader thread, $f: tokens = CPU, $(grep -ao 'resident [0-9]* (budget [0-9]*' vk.log | tail -1)"
  done
  rm -f hist.usage
  # the chain: a failure before it starts declines it (or leaves one matrix behind), a
  # lost device in the middle has the CPU rebuild the state
  local C=(COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1)
  fault_run "faults: qwen36 chain, none" "" q.tok "${C[@]}" -- "${q[@]}"
  S=$(fault_num 's/.*copies (\([0-9]*\) submits).*/\1/p' vk.log)
  for f in stage:1 block:1 record:4 submit:4 commit:1 wait:4 wait:$((S - 2)); do
    fault_run "faults: qwen36 chain, $f" $f q.tok "${C[@]}" -- "${q[@]}"
    [ $f != wait:$((S - 2)) ] || grep -q 'rebuilding the state of [1-9]' vk.log || { cat vk.log; fail "faults: qwen36 chain, $f: no state rebuilt on the CPU"; }
    echo "OK faults: qwen36 chain, $f: tokens = CPU, $(grep -ao 'qwen36 chain: [0-9]* forwards\|rebuilding the state of [0-9]* positions' vk.log | tr '\n' ' ')"
  done
  # colibri's attention core: the KV mirror's buffers and its staging
  local A=(COLI_VK_DENSE=1 COLI_VK_ATTN=1 COLI_VK_TIER_SYNC=1)
  for f in pwstage:1 kvbuf:1 kvbuf:3 wait:3; do
    fault_run "faults: colibri attention, $f" $f g.tok "${A[@]}" -- "${g[@]}"
    echo "OK faults: colibri attention, $f: tokens = CPU"
  done
  if [ "${SAN:-0}" = 1 ]; then make clean >/dev/null 2>&1 || true; fi
}

# tier_count <engine> <log>: N from the last "[VK] tier <engine> run: device N of M" line;
# tier_evictions <engine> <log>: the evictions of that line; tier_failed: its failed
# uploads (an eviction case must have none: a promotion that displaces a resident
# while a batch is in flight waits for the join to free it, even with
# COLI_VK_TIER_SYNC=1, instead of failing and shrinking the budget)
tier_count() {
  local n
  n=$(sed -n "s/^\[VK\] tier $1 run: device \([0-9][0-9]*\) of .*/\1/p" "$2" | tail -1)
  echo "${n:-0}"
}
tier_evictions() {
  local n
  n=$(sed -n "s/^\[VK\] tier $1 run: .* evictions \([0-9][0-9]*\),.*/\1/p" "$2" | tail -1)
  echo "${n:-0}"
}
tier_failed() {
  local n
  n=$(sed -n "s/^\[VK\] tier $1 run: .* failed \([0-9][0-9]*\) |.*/\1/p" "$2" | tail -1)
  echo "${n:-0}"
}

# dense_where <engine> <log> <tag> <env...>: where the run put the dense trunk, checked
# against what it asked for. "[VK] <engine>: device ready, dense matrices on the
# device|CPU" must agree with the matmul count; COLI_VK_DENSE=1 must put the trunk on
# the device; with COLI_VK_DENSE unset and the tier on, Lavapipe (a CPU device, its
# memory the CPU's RAM) must keep it on the CPU, as an integrated GPU does.
dense_where() {
  local eng=$1 log=$2 tag=$3 where; shift 3
  if grep -qa "^\[VK\] $eng: device ready, dense matrices on the device" "$log"; then
    # inkling places none when the host keeps them by design (bf16 weights under the
    # AVX512-BF16 dot, which rounds the activations as the shader does not): its
    # placement line then says "0 resident matrices go to the GPU", and the count is
    # not required, as in family_inkling_olmoe's own bf16 check.
    grep -qa "^\[VK\] $eng: 0 resident matrices go to the GPU" "$log" || need_gpu "$eng" "$log" "$tag"
    where=device
  else
    [ "$(vk_count "$eng" "$log")" = 0 ] || { cat "$log"; fail "$tag: dense matmuls ran on the device with the trunk on the CPU"; }
    where=CPU
  fi
  case " $* " in
    *" COLI_VK_DENSE=1 "*) [ $where = device ] || { cat "$log"; fail "$tag: COLI_VK_DENSE=1 left the trunk on the CPU"; } ;;
    *" COLI_VK_DENSE="*) ;;
    *) if grep -qa '^\[VK\] ready: llvmpipe' "$log" && grep -qa "^\[VK\] tier $eng: on" "$log"; then
         [ $where = CPU ] || { cat "$log"; fail "$tag: Lavapipe with the tier on kept the trunk on the device"; }
       fi ;;
  esac
  echo $where
}

# tier_gate <engine> <tag> <env...> -- <argv...>
# The routed-expert tier (vk_tier.c) against the CPU: the same tokens as the CPU run
# with the same settings, the device served some of the routed experts, and the dense
# trunk ran where it was asked to (dense_where). With EVICT=1 the run must also have
# evicted (its budget is set below the hot set). COLI_USAGE points at a fresh file: no
# warm start from an earlier run's history. COLI_VK_TIER_SYNC=1: a fixture's whole run
# can end before the uploader thread is first scheduled (it did in 5 of 40 runs with
# the trunk on the CPU), so the gate waits for each staged upload at the next step.
tier_gate() {
  local eng=$1 tag=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f tier.usage
  env "${envs[@]}" ./"$eng" "$@" > cpu.log 2>&1 || true
  env "${envs[@]}" COLI_USAGE=tier.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 ./"$eng" "$@" > vk.log 2>&1 || true
  same_tokens cpu.log vk.log "$tag"
  [ "$(tier_count "$eng" vk.log)" -gt 0 ] || { cat vk.log; fail "$tag: no routed expert ran on the device"; }
  if [ "${EVICT:-0}" = 1 ]; then
    [ "$(tier_evictions "$eng" vk.log)" -gt 0 ] || { grep '\[VK\] tier' vk.log; fail "$tag: the budget forced no eviction"; }
    [ "$(tier_failed "$eng" vk.log)" = 0 ] || { grep '\[VK\] tier' vk.log; fail "$tag: an upload failed (the budget shrank)"; }
  fi
  local where; where=$(dense_where "$eng" vk.log "$tag" "${envs[@]}") || { echo "$where"; exit 1; }
  echo "OK $tag: tokens = CPU, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' vk.log | tail -1), $(grep -a -o 'evictions [0-9]*' vk.log | tail -1), trunk on the $where"
}

# vk_gate <engine> <placed-regex> <tag> <env...> -- <argv...>
# CPU arm and Vulkan arm of one configuration; both must exit 0 (the engine's own
# oracle), give the same tokens, run matmuls on the device, and report the expected
# placement ("placed int8 a, int4 b, f32 c" / "int8 a, bf16 b, f32 c").
vk_gate() {
  local eng=$1 placed=$2 tag=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  env "${envs[@]}" ./"$eng" "$@" > cpu.log 2>&1 || { cat cpu.log; fail "$tag: CPU run"; }
  env "${envs[@]}" COLI_VULKAN=1 ./"$eng" "$@" > vk.log 2>&1 || { cat vk.log; fail "$tag: Vulkan run misses the oracle"; }
  same_tokens cpu.log vk.log "$tag"
  need_gpu "$eng" vk.log "$tag"
  grep -qE "\[VK\] $eng: .*placed .*$placed" vk.log || { grep '\[VK\]' vk.log; fail "$tag: expected placement '$placed'"; }
  echo "OK $tag: $(grep -a -o 'Matching tokens: [0-9/]*' vk.log), tokens = CPU, $(grep -o '[0-9]* matmuls on the GPU.*' vk.log | tail -1)"
}

family_qwen() {
  make qwen36 qwen38 VK=1
  # qwen36: the hybrid, Qwen3-Coder (qwen3_moe), the 27B dense and the 2.4T geometry
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/make_qwen36_tiny.py --geometry qwen3-coder-30b --out qwen3_coder_tiny --ref-mode full --emit-ref qwen3_coder_tiny/ref_full.json
  $PY tools/make_qwen36_tiny.py --geometry qwen38-27b-dense --out qwen38_27b_tiny --ref-mode full --emit-ref qwen38_27b_tiny/ref_full.json
  $PY tools/make_qwen36_tiny.py --geometry qwen38-2p4t --seed 3 --out qwen38_2p4t_tiny --ref-mode full --emit-ref qwen38_2p4t_tiny/ref_full.json
  local fx cap caps
  # These arms test the dense trunk's formats: COLI_VK_DENSE=1, because on Lavapipe (a
  # CPU device sharing the CPU's RAM) the trunk otherwise stays on the CPU while the
  # expert tier is on; the tier runs beside it.
  for fx in qwen36_tiny qwen3_coder_tiny qwen38_27b_tiny qwen38_2p4t_tiny; do
    $PY tools/convert_qwen36.py --model $fx --out ${fx}_c --ebits 8
    # cap=1 evicts on every routed expert; on the 92-layer 2.4T geometry that costs
    # ~90 s on Lavapipe for nothing the device path adds, so that one runs at cap=8.
    caps="1 8"; [ $fx = qwen38_2p4t_tiny ] && caps=8
    for cap in $caps; do
      # the CPU job's own configuration: f32 dense weights (fmt 10), token-exact
      # against transformers and equal to the CPU run
      vk_gate qwen36 'f32 [1-9]' "qwen36 $fx f32 cap=$cap" COLI_VK_DENSE=1 COLI_DENSE_I8=0 SNAP=${fx}_c -- $cap 8 $fx/ref_full.json
    done
    # int8 dense rows (fmt 1). int8 weights alone miss the torch oracle on some
    # fixtures, CPU or GPU alike, so this arm gates on the CPU's tokens only, with
    # COLI_DENSE_IDOT=0 giving the CPU the shader's f32 activations.
    COLI_DENSE_IDOT=0 SNAP=${fx}_c ./qwen36 8 8 $fx/ref_full.json > cpu.log 2>&1 || true
    COLI_DENSE_IDOT=0 COLI_VK_DENSE=1 COLI_VULKAN=1 SNAP=${fx}_c ./qwen36 8 8 $fx/ref_full.json > vk.log 2>&1 || true
    same_tokens cpu.log vk.log "qwen36 $fx int8"
    grep -qE '\[VK\] qwen36: [1-9][0-9]* matmuls on the GPU.*placed int8 [1-9]' vk.log || { cat vk.log; fail "qwen36 $fx int8: nothing placed"; }
    echo "OK qwen36 $fx int8: tokens = CPU, $(grep -o '[0-9]* matmuls on the GPU.*' vk.log | tail -1)"
  done

  # The routed-expert tier on every expert container qwen36 reads (COLI_VULKAN=1 turns
  # it on; the runs above already had it): int8 per row (fmt 1), the shared kernel's
  # planar int4-g64 (fmt 4), int4 per row from the unpacked slots (fmt 2), int8 gs64
  # (fmt 13) and the mixed int4 gate/up + int8 down container, at cap=1 with the trunk
  # where the default puts it (on the CPU here) and at cap=8 with the trunk on the
  # device (COLI_VK_DENSE=1); the tier alone (COLI_VK_DENSE=0); and a budget of two
  # experts, which must evict as the routing moves.
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny64 --ref-mode full --inter 64 --emit-ref qwen36_tiny64/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_c --ebits 4 --gs 64
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_g8 --ebits 8 --gs 64
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_d8 --ebits 4 --gs 64 --down-bits 8
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_i4r --ebits 4
  local D
  for cap in 1 8; do
    D=; [ $cap = 8 ] && D=COLI_VK_DENSE=1
    tier_gate qwen36 "qwen36 tier int8 cap=$cap" $D COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- $cap 8 qwen36_tiny/ref_full.json
    tier_gate qwen36 "qwen36 tier int4-g64 planar cap=$cap" $D COLI_DENSE_I8=0 SNAP=qwen36_tiny64_c -- $cap 4 qwen36_tiny64/ref_full.json
    tier_gate qwen36 "qwen36 tier int4 per row cap=$cap" $D COLI_DENSE_I8=0 SNAP=qwen36_tiny_i4r -- $cap 4 qwen36_tiny/ref_full.json
    tier_gate qwen36 "qwen36 tier int8 gs64 cap=$cap" $D COLI_DENSE_I8=0 SNAP=qwen36_tiny64_g8 -- $cap 8 qwen36_tiny64/ref_full.json
    tier_gate qwen36 "qwen36 tier mixed int4/int8 cap=$cap" $D COLI_DENSE_I8=0 SNAP=qwen36_tiny64_d8 -- $cap 4 qwen36_tiny64/ref_full.json
  done
  tier_gate qwen36 "qwen36 tier alone (COLI_VK_DENSE=0)" COLI_VK_DENSE=0 COLI_DENSE_I8=0 SNAP=qwen36_tiny64_c -- 8 4 qwen36_tiny64/ref_full.json
  EVICT=1 tier_gate qwen36 "qwen36 tier, a budget of two experts" COLI_VK_TIER_GB=0.00002 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 qwen36_tiny/ref_full.json
  # COLI_VK_TIER=0: the dense trunk alone, as before the tier; with no tier the default
  # puts the trunk on the device, Lavapipe included
  COLI_VK_TIER=0 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 qwen36_tiny/ref_full.json > cpu.log 2>&1 || true
  COLI_VK_TIER=0 COLI_VULKAN=1 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 qwen36_tiny/ref_full.json > vk.log 2>&1 || true
  same_tokens cpu.log vk.log "qwen36 COLI_VK_TIER=0"
  ! grep -q '^\[VK\] tier' vk.log || { cat vk.log; fail "qwen36 COLI_VK_TIER=0: the tier started"; }
  need_gpu qwen36 vk.log "qwen36 COLI_VK_TIER=0"
  echo "OK qwen36 COLI_VK_TIER=0: tokens = CPU, no tier, $(grep -o '[0-9]* matmuls on the GPU' vk.log | tail -1)"

  # qwen38: one fixture, every resident format, with and without prefill batching
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny
  local batch O
  for batch in 0 1; do
    O="OMP_NUM_THREADS=2 SNAP=qwen38_tiny Q38_PREFILL_BATCH=$batch COLI_VK_DENSE=1"   # the trunk's formats, see qwen36 above
    # the default: every fixture matrix is under 1 MiB, so the trunk stays BF16 (fmt 11)
    vk_gate qwen38 'bf16 [1-9]' "qwen38 bf16 batch=$batch" $O -- 1 8 qwen38_tiny/ref.json
    # Q38_TRUNK_MIN_KB=0: the int8 trunk (fmt 1); what stays outside it is BF16
    vk_gate qwen38 'int8 [1-9].*bf16 [1-9]' "qwen38 int8 batch=$batch" $O Q38_TRUNK_MIN_KB=0 -- 1 8 qwen38_tiny/ref.json
    # Q38_NATIVE_BF16=0 expands the rows to f32 at load (fmt 10)
    vk_gate qwen38 'f32 [1-9]' "qwen38 f32 batch=$batch" $O Q38_TRUNK_CPU_INT8=0 Q38_NATIVE_BF16=0 -- 1 8 qwen38_tiny/ref.json
  done

  # The routed-expert tier on qwen38's three expert forms: BF16 (fmt 11), the release's
  # FP8 with 128x128 block scales (fmt 12, gs 128), the experts-int4g64 sidecar's planar
  # int4 (fmt 4); decode one row at a time and prefill batched; at cap=1 with the trunk
  # where the default puts it (the CPU here), at cap=4 with the trunk on the device;
  # with the MTP head drafting (its verify rows take the device's per-row route; the
  # batched one with the trunk on the device); the tier alone; and a budget of two
  # experts, which must evict.
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_fp8 --fp8-experts
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_int4 --fp8-experts --int4-experts --expert-gain 3
  local fx ref
  for fx in qwen38_tiny qwen38_tiny_fp8 qwen38_tiny_int4; do
    ref=$fx/ref.json; [ $fx = qwen38_tiny_int4 ] && ref=$fx/ref_int4.json
    for batch in 0 1; do for cap in 1 4; do
      D=; [ $cap = 4 ] && D=COLI_VK_DENSE=1
      tier_gate qwen38 "qwen38 tier $fx batch=$batch cap=$cap" OMP_NUM_THREADS=2 $D Q38_PREFILL_BATCH=$batch SNAP=$fx -- $cap 8 $ref
    done; done
  done
  tier_gate qwen38 "qwen38 tier alone (COLI_VK_DENSE=0)" OMP_NUM_THREADS=2 COLI_VK_DENSE=0 SNAP=qwen38_tiny_int4 -- 4 8 qwen38_tiny_int4/ref_int4.json
  EVICT=1 tier_gate qwen38 "qwen38 tier, a budget of two experts" OMP_NUM_THREADS=2 Q38_PREFILL_BATCH=0 COLI_VK_TIER_GB=0.0000065 SNAP=qwen38_tiny_fp8 -- 1 8 qwen38_tiny_fp8/ref.json
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_fp8_mtp --fp8-experts --mtp
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_int4_mtp --fp8-experts --int4-experts --expert-gain 3 --mtp
  for fx in qwen38_tiny_fp8_mtp qwen38_tiny_int4_mtp; do
    for batch in 0 1; do
      D=; [ $batch = 1 ] && D=COLI_VK_DENSE=1
      tier_gate qwen38 "qwen38 tier MTP $fx batch=$batch" OMP_NUM_THREADS=2 $D Q38_MTP=1 Q38_PREFILL_BATCH=$batch SNAP=$fx -- 2 8 $fx/ref.json
    done
  done
}

# The routed-expert tier under ASan and UBSan: a sanitized VK=1 build of both qwen
# engines, the tier's configurations on Lavapipe (formats, eviction, PILOT's worker
# against the tier, MTP, the tier alone, the trunk on the device or on the CPU). Memory safety is the gate, not the tokens
# (a sanitized build vectorizes differently); each run must still put experts on the
# device, or the tier was never exercised.
family_qwen_sanitize() {
  make clean >/dev/null 2>&1 || true
  make qwen36 qwen38 VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny64 --ref-mode full --inter 64 --emit-ref qwen36_tiny64/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_c --ebits 4 --gs 64
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_g8 --ebits 8 --gs 64
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_fp8 --fp8-experts
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_int4 --fp8-experts --int4-experts --expert-gain 3
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_int4_mtp --fp8-experts --int4-experts --expert-gain 3 --mtp
  san() {  # <engine> <tag> <env and argv...>
    local eng=$1 tag=$2; shift 2
    rm -f tier.usage
    env ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2 \
      COLI_USAGE=tier.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 "$@" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    [ "$(tier_count "$eng" san.log)" -gt 0 ] || { cat san.log; fail "$tag: no routed expert ran on the device"; }
    echo "OK $tag: sanitizers clean, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' san.log | tail -1), $(grep -a -o 'dense matrices on the [a-zA-Z]*' san.log | head -1)"
  }
  local cap D   # cap=8 and batch=1 with the trunk on the device, the others where the default puts it
  for cap in 1 8; do
    D=; [ $cap = 8 ] && D=COLI_VK_DENSE=1
    san qwen36 "asan qwen36 int8 PILOT cap=$cap" $D COLI_DENSE_I8=0 PILOT=1 WIDE=2 SNAP=qwen36_tiny_c ./qwen36 $cap 8 qwen36_tiny/ref_full.json
    san qwen36 "asan qwen36 int4-g64 PILOT cap=$cap" $D COLI_DENSE_I8=0 PILOT=1 WIDE=2 SNAP=qwen36_tiny64_c ./qwen36 $cap 4 qwen36_tiny64/ref_full.json
    san qwen36 "asan qwen36 int8 gs64 cap=$cap" $D COLI_DENSE_I8=0 SNAP=qwen36_tiny64_g8 ./qwen36 $cap 8 qwen36_tiny64/ref_full.json
  done
  san qwen36 "asan qwen36 eviction" COLI_VK_TIER_GB=0.00002 COLI_DENSE_I8=0 PILOT=1 SNAP=qwen36_tiny_c ./qwen36 8 8 qwen36_tiny/ref_full.json
  san qwen36 "asan qwen36 tier alone" COLI_VK_DENSE=0 SNAP=qwen36_tiny64_c ./qwen36 8 4 qwen36_tiny64/ref_full.json
  local b
  for b in 0 1; do
    D=; [ $b = 1 ] && D=COLI_VK_DENSE=1
    san qwen38 "asan qwen38 fp8 batch=$b" $D Q38_PREFILL_BATCH=$b SNAP=qwen38_tiny_fp8 ./qwen38 1 8 qwen38_tiny_fp8/ref.json
    san qwen38 "asan qwen38 int4 batch=$b" $D Q38_PREFILL_BATCH=$b SNAP=qwen38_tiny_int4 ./qwen38 1 8 qwen38_tiny_int4/ref_int4.json
  done
  san qwen38 "asan qwen38 eviction" Q38_PREFILL_BATCH=0 COLI_VK_TIER_GB=0.0000065 SNAP=qwen38_tiny_fp8 ./qwen38 1 8 qwen38_tiny_fp8/ref.json
  san qwen38 "asan qwen38 MTP" COLI_VK_DENSE=1 Q38_MTP=1 Q38_PREFILL_BATCH=0 SNAP=qwen38_tiny_int4_mtp ./qwen38 2 8 qwen38_tiny_int4_mtp/ref.json
  san qwen38 "asan qwen38 tier alone" COLI_VK_DENSE=0 SNAP=qwen38_tiny_int4 ./qwen38 4 8 qwen38_tiny_int4/ref_int4.json
  make clean >/dev/null 2>&1 || true
}

family_inkling_olmoe() {
  make inkling olmoe VK=1
  local cap
  # inkling, f32 fixture (fmt 10): the oracle and the CPU's ids, at three caps. These
  # arms and the next two test the dense matrices' formats: COLI_VK_DENSE=1, because on
  # Lavapipe (a CPU device sharing the CPU's RAM) the trunk otherwise stays on the CPU
  # while the expert tier is on; the tier runs beside it.
  $PY tools/make_tiny_inkling.py tiny_inkling
  for cap in 1 2 8; do
    SNAP=tiny_inkling ./inkling $cap 0 tiny_inkling/ref_inkling.json > cpu.log 2>&1
    COLI_VK_DENSE=1 COLI_VULKAN=1 SNAP=tiny_inkling ./inkling $cap 0 tiny_inkling/ref_inkling.json > vk.log 2>&1
    grep -qE 'Matching tokens: ([0-9]+)/\1$' vk.log || { cat vk.log; fail "inkling f32 cap=$cap: oracle"; }
    same_tokens cpu.log vk.log "inkling f32 cap=$cap"
    need_gpu inkling vk.log "inkling f32 cap=$cap"
    echo "OK inkling f32 cap=$cap: $(grep -o 'Matching tokens: [0-9/]*' vk.log), $(vk_count inkling vk.log) matmuls on the GPU"
  done

  # the same weights stored as bf16 (fmt 11). A build with the AVX512-BF16 dot rounds
  # activations to bf16 on the CPU and keeps those matrices there ("0 bf16" in the
  # placement line), so the count is only required when the line places any.
  mkdir -p tiny_inkling_bf16
  cp tiny_inkling/config.json tiny_inkling/generation_config.json tiny_inkling_bf16/
  $PY - <<'EOF'
import json, struct, numpy as np
src, dst = "tiny_inkling/model.safetensors", "tiny_inkling_bf16/model.safetensors"
raw = open(src, "rb").read(); n = struct.unpack("<Q", raw[:8])[0]; hdr = json.loads(raw[8:8 + n])
out, blobs, off = {}, [], 0
for k, v in hdr.items():
    if k == "__metadata__": out[k] = v; continue
    a = np.frombuffer(raw[8 + n + v["data_offsets"][0]: 8 + n + v["data_offsets"][1]], np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    b = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()
    out[k] = {"dtype": "BF16", "shape": v["shape"], "data_offsets": [off, off + len(b)]}
    blobs.append(b); off += len(b)
h = json.dumps(out).encode(); h += b" " * (-len(h) % 8)
with open(dst, "wb") as f:
    f.write(struct.pack("<Q", len(h))); f.write(h); [f.write(b) for b in blobs]
EOF
  SNAP=tiny_inkling_bf16 ./inkling 8 0 tiny_inkling/ref_inkling.json > cpu.log 2>&1 || true
  COLI_VK_DENSE=1 COLI_VULKAN=1 SNAP=tiny_inkling_bf16 ./inkling 8 0 tiny_inkling/ref_inkling.json > vk.log 2>&1 || true
  same_tokens cpu.log vk.log "inkling bf16"
  grep -q ', 0 bf16)' vk.log || need_gpu inkling vk.log "inkling bf16"
  echo "OK inkling bf16: tokens = CPU, $(vk_count inkling vk.log) matmuls on the GPU"

  # the dense-int4g64 container (int8 fmt 1, int4-g64 fmt 4, f32 fmt 10 side by side)
  rm -rf tiny_inkling_q && cp -r tiny_inkling tiny_inkling_q
  $PY - tools/convert_inkling_dense_int4.py tiny_inkling_q <<'EOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("conv", sys.argv[1])
conv = importlib.util.module_from_spec(spec); spec.loader.exec_module(conv)
conv.MIN_ELEMS = 0
conv.ATTN_BITS = 4
base = conv.classify
conv.classify = lambda n, s, d: "int8" if n.endswith(".mlp.down_proj.weight") else base(n, s, d)
sys.argv = ["convert_inkling_dense_int4.py", "--dir", sys.argv[2]]
conv.main()
EOF
  mkdir -p tiny_inkling_q/dense-int4g64
  mv tiny_inkling_q/dense-int4g64.safetensors tiny_inkling_q/dense-int4g64/dense.safetensors
  SNAP=tiny_inkling_q ./inkling 8 0 tiny_inkling/ref_inkling.json > cpu.log 2>&1 || true
  COLI_VK_DENSE=1 COLI_VULKAN=1 SNAP=tiny_inkling_q ./inkling 8 0 tiny_inkling/ref_inkling.json > vk.log 2>&1 || true
  same_tokens cpu.log vk.log "inkling int4-g64 container"
  need_gpu inkling vk.log "inkling int4-g64 container"
  echo "OK inkling int4-g64 container: tokens = CPU, $(vk_count inkling vk.log) matmuls on the GPU"

  # olmoe: f32 residents (fmt 10), the oracle and the CPU's ids (the trunk on the
  # device, as above)
  $PY tools/make_olmoe_tiny.py --output olmoe_tiny
  $PY tools/convert_olmoe_merged.py --model olmoe_tiny --out olmoe_tiny_c
  SNAP=olmoe_tiny_c ./olmoe 8 8 olmoe_tiny/ref_olmoe.json > cpu.log 2>&1
  COLI_VK_DENSE=1 COLI_VULKAN=1 SNAP=olmoe_tiny_c ./olmoe 8 8 olmoe_tiny/ref_olmoe.json > vk.log 2>&1
  grep -qE 'Matching tokens: ([0-9]+)/\1$' vk.log || { cat vk.log; fail "olmoe: oracle"; }
  same_tokens cpu.log vk.log "olmoe"
  need_gpu olmoe vk.log "olmoe"
  echo "OK olmoe: $(grep -o 'Matching tokens: [0-9/]*' vk.log), $(vk_count olmoe vk.log) matmuls on the GPU"

  inkling_olmoe_tier_fixtures
  # The routed-expert tier (COLI_VULKAN=1 turns it on; the runs above already had it).
  # inkling: experts in f32 (fmt 10) under the f32, bf16 and dense-int4g64 snapshots,
  # the int4 and int8 expert containers (fmt 2 and fmt 1: gate and up in one fused
  # tensor, up I rows in), the runtime int4 and int8 quantization (fmt 2 from int8 rows,
  # fmt 1), at cap=1 with the trunk where the default puts it (on the CPU here) and at
  # cap=8 with the trunk on the device (COLI_VK_DENSE=1); TOPP's trimmed ranks; the tier
  # alone (COLI_VK_DENSE=0); and a budget of two experts, which must evict.
  local fx D bits R=tiny_inkling/ref_inkling.json OR=olmoe_tiny/ref_olmoe.json
  for fx in tiny_inkling tiny_inkling_bf16 tiny_inkling_q tiny_inkling_x-i4 tiny_inkling_x-i8; do
    for cap in 1 8; do
      D=; [ $cap = 8 ] && D=COLI_VK_DENSE=1
      tier_gate inkling "inkling tier $fx cap=$cap" $D SNAP=$fx -- $cap 0 $R
    done
  done
  for bits in 4 8; do tier_gate inkling "inkling tier runtime int$bits" SNAP=tiny_inkling -- 2 $bits $R; done
  tier_gate inkling "inkling tier TOPP" TOPP=0.3 SNAP=tiny_inkling_x-i4 -- 2 0 $R
  tier_gate inkling "inkling tier alone (COLI_VK_DENSE=0)" COLI_VK_DENSE=0 SNAP=tiny_inkling_x-i4 -- 8 0 $R
  EVICT=1 tier_gate inkling "inkling tier, a budget of two experts" COLI_VK_TIER_GB=0.00006 SNAP=tiny_inkling -- 8 0 $R
  # olmoe: int8 rows (fmt 1), the same placements, PILOT's prefetch worker beside the
  # tier, the tier alone and a budget of three experts, which must evict.
  for cap in 1 8; do
    D=; [ $cap = 8 ] && D=COLI_VK_DENSE=1
    tier_gate olmoe "olmoe tier cap=$cap" $D SNAP=olmoe_tiny_c -- $cap 8 $OR
  done
  tier_gate olmoe "olmoe tier PILOT" PILOT=1 WIDE=2 SNAP=olmoe_tiny_c -- 2 8 $OR
  tier_gate olmoe "olmoe tier alone (COLI_VK_DENSE=0)" COLI_VK_DENSE=0 SNAP=olmoe_tiny_c -- 8 8 $OR
  EVICT=1 tier_gate olmoe "olmoe tier, a budget of three experts" COLI_VK_TIER_GB=0.000025 SNAP=olmoe_tiny_c -- 8 8 $OR

  # The warm start: a CPU run writes the history (inkling's PIN=<file>, olmoe's
  # COLI_USAGE), and the tier's run fills the device from it before the first token.
  rm -f tier.hist
  # Streaming can report its first device batch after the prompt has been printed.
  # Keep diagnostics separate so the exact text comparison measures stdout alone.
  PIN=tier.hist SNAP=tiny_inkling ./inkling 8 -p "The capital of France is" -n 8 > cpu.log 2> cpu.err
  PIN=tier.hist COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 SNAP=tiny_inkling ./inkling 8 -p "The capital of France is" -n 8 > vk.log 2> vk.err
  $PY - cpu.log vk.log <<'PY' || { cat cpu.log vk.log vk.err; fail "inkling tier warm start: the text differs from the CPU's"; }
import re, sys
def text(p):
    m = re.search(rb"\[\d+ prompt tokens\](.*?)\n\[prefill", open(p, "rb").read(), re.S)
    return m.group(1) if m else None
a, b = text(sys.argv[1]), text(sys.argv[2])
sys.exit(0 if a is not None and a == b else 1)
PY
  grep -qa '^\[VK\] tier inkling: warm start, [1-9]' vk.err || { cat vk.err; fail "inkling tier: no warm start"; }
  [ "$(tier_count inkling vk.err)" -gt 0 ] || { cat vk.err; fail "inkling tier warm start: no routed expert ran on the device"; }
  echo "OK inkling tier warm start: text = CPU, $(grep -a -o 'warm start, [0-9]* experts' vk.err), $(grep -a -o 'device [0-9]* of [0-9]* routed experts' vk.err | tail -1)"
  rm -f tier.hist
  COLI_USAGE=tier.hist SNAP=olmoe_tiny_c ./olmoe 8 8 $OR > cpu.log 2>&1
  COLI_USAGE=tier.hist COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 SNAP=olmoe_tiny_c ./olmoe 8 8 $OR > vk.log 2>&1
  same_tokens cpu.log vk.log "olmoe tier warm start"
  grep -qa '^\[VK\] tier olmoe: warm start, [1-9]' vk.log || { cat vk.log; fail "olmoe tier: no warm start"; }
  [ "$(tier_count olmoe vk.log)" -gt 0 ] || { cat vk.log; fail "olmoe tier warm start: no routed expert ran on the device"; }
  echo "OK olmoe tier warm start: tokens = CPU, $(grep -a -o 'warm start, [0-9]* experts' vk.log), $(grep -a -o 'device [0-9]* of [0-9]* routed experts' vk.log | tail -1)"

  # The serve protocol with the tier on: KV prefix reuse token-identical to a cold
  # engine, HITS and EMAP (tier 2 on the device) after every turn, Brio's snapshot
  # scoring; every engine after the first warm-starts from the history the one before
  # it saved.
  COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 INKLING_TINY=tiny_inkling \
    $PY -m unittest tests.test_inkling_prefix_serve tests.test_inkling_dashboard_hits
  COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 OLMOE_TINY=olmoe_tiny_c \
    $PY -m unittest tests.test_olmoe_prefix_serve tests.test_olmoe_dashboard_hits tests.test_brio_serve
}

# The fixtures the expert tier's arms add to the family's own: inkling's int4 and int8
# expert containers through tools/convert_inkling_int4.py's round trip (its fake
# vendor checkpoint knows the embedding norm by the name transformers used before
# 5.18), a tokenizer for inkling's -p mode and serve, and olmoe's for serve.
inkling_olmoe_tier_fixtures() {
  $PY - <<'PY'
import importlib.util, os, shutil
from safetensors.torch import load_file, save_file
spec = importlib.util.spec_from_file_location("conv", "tools/convert_inkling_int4.py")
conv = importlib.util.module_from_spec(spec); spec.loader.exec_module(conv)
t = load_file("tiny_inkling/model.safetensors")
if "model.embed_tokens.embed_norm.weight" in t:
    t["model.embed_norm.weight"] = t.pop("model.embed_tokens.embed_norm.weight")
for d in ("tiny_inkling_hf", "tiny_inkling_x-tml", "tiny_inkling_x-pass", "tiny_inkling_x-i4", "tiny_inkling_x-i8"):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs("tiny_inkling_hf")
save_file(t, "tiny_inkling_hf/model.safetensors")
shutil.copy("tiny_inkling/config.json", "tiny_inkling_hf/")
conv.selftest_e2e("tiny_inkling_hf", "tiny_inkling_x")          # tiny_inkling_x-i4: int4 experts
conv.convert_dir("tiny_inkling_x-tml", "tiny_inkling_x-i8", 8)   # int8 experts
PY
  $PY -c 'import sys; from pathlib import Path; sys.path.insert(0, "."); from tests.test_inkling_prefix_serve import ensure_tokenizer; ensure_tokenizer(Path("tiny_inkling"))'
  $PY tools/make_edge_tiny_tokenizer.py --vocab-size 128 olmoe_tiny_c
}

# inkling's and olmoe's expert tier under ASan and UBSan: a sanitized VK=1 build, the
# tier's configurations on Lavapipe (the expert formats, eviction, TOPP, PILOT's worker
# beside the tier, the tier alone, the trunk on the device or on the CPU, the warm
# start) and a serve session of each engine, twice, the second warm-started. Memory
# safety is the gate, not the tokens (a sanitized build vectorizes differently); each
# run must still put experts on the device, or the tier was never exercised.
family_inkling_olmoe_sanitize() {
  make clean >/dev/null 2>&1 || true
  make inkling olmoe VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  $PY tools/make_tiny_inkling.py tiny_inkling
  $PY tools/make_olmoe_tiny.py --output olmoe_tiny --force
  $PY tools/convert_olmoe_merged.py --model olmoe_tiny --out olmoe_tiny_c
  inkling_olmoe_tier_fixtures
  san_io() {  # <engine> <tag> <env and argv...>; KEEP=1 keeps the history of the run before
    local eng=$1 tag=$2; shift 2
    [ "${KEEP:-0}" = 1 ] || rm -f tier.usage
    env ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2 \
      COLI_USAGE=tier.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 "$@" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    [ "$(tier_count "$eng" san.log)" -gt 0 ] || { cat san.log; fail "$tag: no routed expert ran on the device"; }
    echo "OK $tag: sanitizers clean, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' san.log | tail -1), $(grep -a -o 'evictions [0-9]*' san.log | tail -1)$(grep -a -o 'warm start, [0-9]* experts' san.log | sed 's/^/, /')"
  }
  local fx cap D R=tiny_inkling/ref_inkling.json OR=olmoe_tiny/ref_olmoe.json
  for fx in tiny_inkling tiny_inkling_x-i4 tiny_inkling_x-i8; do
    for cap in 1 8; do
      D=; [ $cap = 8 ] && D=COLI_VK_DENSE=1
      san_io inkling "asan inkling $fx cap=$cap" $D SNAP=$fx ./inkling $cap 0 $R
    done
  done
  san_io inkling "asan inkling runtime int4" SNAP=tiny_inkling ./inkling 2 4 $R
  san_io inkling "asan inkling TOPP" TOPP=0.3 SNAP=tiny_inkling_x-i4 ./inkling 2 0 $R
  san_io inkling "asan inkling eviction" COLI_VK_TIER_GB=0.00006 SNAP=tiny_inkling ./inkling 8 0 $R
  san_io inkling "asan inkling tier alone" COLI_VK_DENSE=0 SNAP=tiny_inkling_x-i4 ./inkling 8 0 $R
  san_io inkling "asan inkling -p, history written" PIN=tier.usage SNAP=tiny_inkling ./inkling 8 -p "The capital of France is" -n 8
  KEEP=1 san_io inkling "asan inkling -p, warm start" PIN=tier.usage SNAP=tiny_inkling ./inkling 8 -p "The capital of France is" -n 8
  grep -qa 'tier inkling: warm start, [1-9]' san.log || { cat san.log; fail "asan inkling: no warm start"; }
  for cap in 1 8; do
    D=; [ $cap = 8 ] && D=COLI_VK_DENSE=1
    san_io olmoe "asan olmoe PILOT cap=$cap" $D PILOT=1 WIDE=2 SNAP=olmoe_tiny_c ./olmoe $cap 8 $OR
  done
  san_io olmoe "asan olmoe eviction" COLI_VK_TIER_GB=0.000025 PILOT=1 SNAP=olmoe_tiny_c ./olmoe 8 8 $OR
  san_io olmoe "asan olmoe tier alone" COLI_VK_DENSE=0 SNAP=olmoe_tiny_c ./olmoe 8 8 $OR
  KEEP=1 san_io olmoe "asan olmoe warm start" PILOT=1 SNAP=olmoe_tiny_c ./olmoe 2 8 $OR
  grep -qa 'tier olmoe: warm start, [1-9]' san.log || { cat san.log; fail "asan olmoe: no warm start"; }
  # serve: two turns, then a second engine warm-started from the history the first saved
  local eng
  for eng in inkling olmoe; do
    rm -f tier.usage
    $PY - $eng <<'PY' || fail "asan $eng serve"
import os, subprocess, sys, threading
eng = sys.argv[1]
snap, argv = ("tiny_inkling", ["8"]) if eng == "inkling" else ("olmoe_tiny_c", ["4", "8"])
env = dict(os.environ, SNAP=snap, SERVE="1", PIN="tier.usage", COLI_USAGE="tier.usage", COLI_VULKAN="1",
           COLI_VK_TIER_SYNC="1", OMP_NUM_THREADS="2", ASAN_OPTIONS="detect_leaks=0:detect_stack_use_after_return=0",
           UBSAN_OPTIONS="print_stacktrace=1")
for run in range(2):
    p = subprocess.Popen(["./" + eng] + argv, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, bufsize=0)
    err = []
    t = threading.Thread(target=lambda: err.extend(iter(p.stderr.readline, b"")), daemon=True)
    t.start()
    while b"READY" not in p.stdout.readline():
        pass
    for rid, prompt in (("1", b"The capital of France is"), ("2", b"The capital of France is, and Spain")):
        p.stdin.write(f"SUBMIT {rid} 0 {len(prompt)} 6 0 1\n".encode() + prompt + b"\n")
        p.stdin.flush()
        while True:
            line = p.stdout.readline()
            if not line or line.split(b" ", 1)[0] in (b"DONE", b"ERROR"):
                break
            if line.startswith(b"DATA "):
                p.stdout.read(int(line.split()[2])); p.stdout.readline()
    p.stdin.close(); p.wait(timeout=300); t.join(10)
    log = b"".join(err).decode(errors="replace")
    if "ERROR: AddressSanitizer" in log or "runtime error:" in log:
        print(log); sys.exit(1)
    turns = [l for l in log.splitlines() if l.startswith(f"[VK] tier {eng} turn: device ")]
    if len(turns) != 2 or turns[-1].split()[5] == "0" or (run == 1 and "warm start" not in log):
        print(log); sys.exit(1)
    print(f"OK asan {eng} serve, engine {run + 1}: sanitizers clean, " + " ".join(turns[-1].split()[4:9]) +
          (", warm-started" if run else ""))
PY
  done
  make clean >/dev/null 2>&1 || true
}

family_mimo_qwenimage() {
  make mimo qwenimage VK=1
  # mimo: Xiaomi's vendor oracle on the GPU (its engine_env is the f32 dense
  # configuration; its variants include the native FP8/BF16 one and the BF16 vision
  # tower): the trunk on the device with the experts on the CPU (COLI_VK_TIER=0), the
  # routed experts on the shared tier (MXFP4, fmt 7) with the trunk where the default
  # puts it (the CPU on Lavapipe), and both on the device.
  $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force --vision
  COLI_VULKAN=1 COLI_VK_TIER=0 $PY tests/mimo_tiny_harness.py --binary ./mimo --fixture ./mimo_tiny
  COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 $PY tests/mimo_tiny_harness.py --binary ./mimo --fixture ./mimo_tiny
  COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 COLI_VK_DENSE=1 $PY tests/mimo_tiny_harness.py --binary ./mimo --fixture ./mimo_tiny
  # MIMO_DENSE_BITS 0 (native fp8/bf16: fmt 12, 11), 8 (fmt 1) and 32 (fmt 10):
  # the CPU's tokens for every case of ref.json and for the picture, with the experts
  # on the CPU (MIMO_VK_EXPERTS=0, the trunk on the device), on the tier with the trunk
  # on the CPU (the tier must have served some), and on the tier with the trunk on the
  # device (both).
  ids() { $PY -c "import json,sys;r=json.load(open('mimo_tiny/ref.json'));c=r['image'] if sys.argv[1]=='image' else r['cases'][sys.argv[1]];print(' '.join(map(str,c['prompt_ids'])))" "$1"; }
  local grid bits c x extra
  grid=$($PY -c "import json;i=json.load(open('mimo_tiny/ref.json'))['image'];print(i['grid_h'],i['grid_w'])")
  for bits in 0 8 32; do
    for c in short window long image; do
      extra=(); [ "$c" = image ] && extra=(--image mimo_tiny/patches.f32 --grid $grid)
      MIMO_DENSE_BITS=$bits COLI_TEMP=0 ./mimo mimo_tiny --ids "$(ids $c)" --ngen 6 "${extra[@]}" > mimo-cpu.txt 2>/dev/null
      for x in MIMO_VK_EXPERTS=0 COLI_VK_DENSE=0 COLI_VK_DENSE=1; do
        env COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 $x MIMO_DENSE_BITS=$bits COLI_TEMP=0 \
          ./mimo mimo_tiny --ids "$(ids $c)" --ngen 6 "${extra[@]}" > mimo-vk.txt 2> mimo-vk.err
        cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-vk.err; fail "mimo bits=$bits $c $x differs from the CPU"; }
        [ $x = COLI_VK_DENSE=0 ] || need_gpu mimo mimo-vk.err "mimo bits=$bits $c $x"
        if [ $x = MIMO_VK_EXPERTS=0 ]; then
          ! grep -q '^\[VK\] tier' mimo-vk.err || { cat mimo-vk.err; fail "mimo bits=$bits $c: MIMO_VK_EXPERTS=0 started the tier"; }
        else
          [ "$(tier_count mimo mimo-vk.err)" -gt 0 ] || { cat mimo-vk.err; fail "mimo bits=$bits $c $x: no routed expert ran on the device"; }
        fi
      done
    done
    echo "OK mimo MIMO_DENSE_BITS=$bits: the CPU's tokens, text and image, experts on the CPU and on the tier, trunk on the device and on the CPU"
  done
  # MIMO_VK_EXPERTS=N sizes the tier at N experts: at 2 it must evict as the routing
  # moves, and still give the CPU's tokens
  MIMO_DENSE_BITS=32 COLI_TEMP=0 ./mimo mimo_tiny --ids "$(ids long)" --ngen 6 > mimo-cpu.txt 2>/dev/null
  COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 MIMO_VK_EXPERTS=2 MIMO_DENSE_BITS=32 COLI_TEMP=0 \
    ./mimo mimo_tiny --ids "$(ids long)" --ngen 6 > mimo-vk.txt 2> mimo-vk.err
  cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-vk.err; fail "mimo MIMO_VK_EXPERTS=2 differs from the CPU"; }
  grep -q 'budget [0-9.]* KiB = 2 experts' mimo-vk.err || { grep '\[VK\]' mimo-vk.err; fail "mimo MIMO_VK_EXPERTS=2: not a budget of two experts"; }
  [ "$(tier_count mimo mimo-vk.err)" -gt 0 ] && [ "$(tier_evictions mimo mimo-vk.err)" -gt 0 ] || { grep '\[VK\] tier' mimo-vk.err; fail "mimo MIMO_VK_EXPERTS=2: the tier served nothing or never evicted"; }
  grep -q ' failed 0 ' mimo-vk.err || { grep '\[VK\] tier' mimo-vk.err; fail "mimo MIMO_VK_EXPERTS=2: an upload failed"; }
  echo "OK mimo, a budget of two experts: tokens = CPU, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' mimo-vk.err | tail -1), $(grep -a -o 'evictions [0-9]*' mimo-vk.err | tail -1)"

  # qwenimage: at 8, 16 and 32 bits (fmt 1, 11, 10) every oracle stage of the
  # Lavapipe run against the CPU run with the same bits and f32 activations. int8 is
  # outside the oracle's own tolerance on both sides, so the reports are compared
  # with each other, not with the reference; at 16 bits the Vulkan run must also pass.
  $PY tools/make_qwenimage_tiny.py qwenimage_tiny
  for bits in 8 16 32; do
    COLI_IMG_BITS=$bits COLI_IMG_ACT8=0 ./qwenimage --model qwenimage_tiny --ref qwenimage_tiny/ref > qi-cpu.log 2>&1 || true
    local rc=0
    COLI_VULKAN=1 COLI_IMG_BITS=$bits COLI_IMG_ACT8=0 ./qwenimage --model qwenimage_tiny --ref qwenimage_tiny/ref > qi-vk.log 2>&1 || rc=$?
    [ $bits != 16 ] || [ $rc = 0 ] || { cat qi-vk.log; fail "qwenimage bf16 oracle on Vulkan"; }
    need_gpu qwenimage qi-vk.log "qwenimage oracle bits=$bits"
    BITS=$bits $PY - <<'PY'
import os, re
def stages(p):
    return {k.strip(): float(r) for k, r in
            re.findall(r"\[oracle\] (.+?)\s+n=\d+\s+max\|err\| \S+\s+rel (\S+)", open(p).read())}
c, v = stages("qi-cpu.log"), stages("qi-vk.log")
assert len(c) >= 10 and c.keys() == v.keys(), (c, v)
for k in c:   # relative errors against the reference, CPU vs GPU: equal up to summation order
    assert abs(c[k] - v[k]) <= 0.25 * c[k] + 2e-7, (k, c[k], v[k])
print(f"OK qwenimage oracle: {len(c)} stages of the Lavapipe run match the CPU run at {os.environ['BITS']} bits")
PY
  done
  # a generated picture at the default int8 weights, Lavapipe against the CPU
  COLI_IMG_BITS=8 COLI_IMG_ACT8=0 ./qwenimage --model qwenimage_tiny --prompt "a red fox in the snow" \
    --width 256 --height 256 --steps 2 --seed 1 --out qi-cpu.png
  COLI_VULKAN=1 COLI_IMG_BITS=8 COLI_IMG_ACT8=0 ./qwenimage --model qwenimage_tiny --prompt "a red fox in the snow" \
    --width 256 --height 256 --steps 2 --seed 1 --out qi-vk.png 2> qi-vk-gen.err
  need_gpu qwenimage qi-vk-gen.err "qwenimage picture"
  $PY - <<'PY'
import sys; sys.path.insert(0, ".")
import image_engine as e
_, _, _, a = e.decode_png(open("qi-cpu.png", "rb").read())
_, _, _, b = e.decode_png(open("qi-vk.png", "rb").read())
d = [abs(x - y) for x, y in zip(a, b)]
assert len(a) == len(b) and max(d) <= 2 and sum(t > 0 for t in d) <= len(d) // 1000, (max(d), sum(t > 0 for t in d))
print(f"OK qwenimage picture: {sum(t > 0 for t in d)} of {len(d)} bytes differ from the CPU's, max {max(d)}")
PY
  # the serve protocol with the device on
  COLI_VULKAN=1 QWENIMAGE_TINY=qwenimage_tiny $PY -m unittest tests.test_qwenimage_engine_serve
}

family_deepseek() {
  make deepseek_v41 VK=1
  # deepseek_v41: fp8 dense in 32x32 ue8m0 tiles (fmt 12, gs 32, the tile scale
  # repeated over its rows) and bf16 (fmt 11). The engine exits non-zero on any
  # token mismatch with the reference; the CPU run must print the same stream.
  # These arms test the dense trunk: COLI_VK_DENSE=1, because on Lavapipe the trunk
  # otherwise stays on the CPU while the expert tier is on; the tier runs beside it.
  $PY tools/make_dsv41_tiny.py --out dsv41_tiny --emit-ref dsv41_tiny/ref.json
  $PY tools/make_dsv41_tiny.py --out dsv41_long --emit-ref dsv41_long/ref.json --prompt-len 40 --max-new 6
  v41() {  # <tag> <env and argv...>
    local tag=$1; shift
    env "$@" > v41-cpu.txt 2> v41-cpu.err || { cat v41-cpu.err; fail "deepseek_v41 $tag: CPU run"; }
    env COLI_VULKAN=1 "$@" > v41-vk.txt 2> v41-vk.err || { cat v41-vk.err; fail "deepseek_v41 $tag: Vulkan run misses the oracle"; }
    cmp -s v41-cpu.txt v41-vk.txt || { diff v41-cpu.txt v41-vk.txt | head; fail "deepseek_v41 $tag: Vulkan output differs from the CPU"; }
    need_gpu deepseek_v41 v41-vk.err "deepseek_v41 $tag"
    echo "OK deepseek_v41 $tag: output = CPU, $(vk_count deepseek_v41 v41-vk.err) matmuls on the GPU, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' v41-vk.err | tail -1)"
  }
  local cap force
  for cap in 1 2 8; do v41 "cap=$cap" COLI_VK_DENSE=1 SNAP=dsv41_tiny ./deepseek_v41 $cap dsv41_tiny/ref.json; done
  for cap in 2 8; do v41 "40-token prompt cap=$cap" COLI_VK_DENSE=1 SNAP=dsv41_long ./deepseek_v41 $cap dsv41_long/ref.json; done
  for force in 1 2 3 4 5; do
    v41 "DSpark spec=$force" COLI_VK_DENSE=1 SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=$force ./deepseek_v41 8 dsv41_tiny/ref.json
  done

  # The routed-expert tier (vk_tier.c) on deepseek_v41's experts, fp4 with a ue8m0
  # scale per 32 (fmt 7), the clamped SwiGLU; the trunk where the default puts it (the
  # CPU here). Each run must pass the oracle, print the CPU run's stream and have
  # served routed experts from the device; with EVICT=1 the budget (below the hot set)
  # must also have evicted. COLI_USAGE points at a fresh history, COLI_VK_TIER_SYNC=1
  # awaits each staged upload at the next step (as in tier_gate).
  v41_tier() {  # <tag> <env and argv...>
    local tag=$1; shift
    rm -f tier.usage
    env "$@" > v41-cpu.txt 2> v41-cpu.err || { cat v41-cpu.err; fail "deepseek_v41 tier $tag: CPU run"; }
    env COLI_USAGE=tier.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 "$@" > v41-vk.txt 2> v41-vk.err ||
      { cat v41-vk.err; fail "deepseek_v41 tier $tag: Vulkan run misses the oracle"; }
    cmp -s v41-cpu.txt v41-vk.txt || { diff v41-cpu.txt v41-vk.txt | head; fail "deepseek_v41 tier $tag: Vulkan output differs from the CPU"; }
    [ "$(tier_count deepseek_v41 v41-vk.err)" -gt 0 ] || { cat v41-vk.err; fail "deepseek_v41 tier $tag: no routed expert ran on the device"; }
    if [ "${EVICT:-0}" = 1 ]; then
      [ "$(tier_evictions deepseek_v41 v41-vk.err)" -gt 0 ] || { grep '\[VK\] tier' v41-vk.err; fail "deepseek_v41 tier $tag: the budget forced no eviction"; }
      [ "$(tier_failed deepseek_v41 v41-vk.err)" = 0 ] || { grep '\[VK\] tier' v41-vk.err; fail "deepseek_v41 tier $tag: an upload failed (the budget shrank)"; }
    fi
    echo "OK deepseek_v41 tier $tag: output = CPU, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' v41-vk.err | tail -1), $(grep -a -o 'evictions [0-9]*' v41-vk.err | tail -1)"
  }
  for cap in 1 2 8; do v41_tier "cap=$cap" SNAP=dsv41_tiny ./deepseek_v41 $cap dsv41_tiny/ref.json; done
  for cap in 2 8; do v41_tier "40-token prompt cap=$cap" SNAP=dsv41_long ./deepseek_v41 $cap dsv41_long/ref.json; done
  # DSpark: the drafts stay on the CPU, the verify rows of the backbone take the device
  for force in 1 2 3 4 5; do
    v41_tier "DSpark spec=$force" SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=$force ./deepseek_v41 1 dsv41_tiny/ref.json
  done
  EVICT=1 v41_tier "a budget of three experts" COLI_VK_TIER_GB=0.00005 SNAP=dsv41_long ./deepseek_v41 8 dsv41_long/ref.json
  v41_tier "trunk on the device too" COLI_VK_DENSE=1 SNAP=dsv41_long ./deepseek_v41 2 dsv41_long/ref.json

  # deepseek_v4: fp8 128x128 blocks (fmt 12, gs 128) and the bf16 router, compressors
  # and head (fmt 11). The GPU gets the activations after the CPU's own E4M3 rounding,
  # so the two runs do the same arithmetic. The tiny check builds the VK=1 binary and
  # keeps passing with the device open (the expert tier on, the trunk on the CPU);
  # its --oracle path reloads the dense weights every forward and so stays on the CPU,
  # which is why the device is checked on the session path below: ids and
  # teacher-forced predictions equal to the CPU's and to the reference's greedy stream.
  COLI_VULKAN=1 make deepseek-v4-tiny-check VK=1
  local prompt
  prompt=$($PY -c 'import json; c=json.load(open("deepseek_v4_tiny/ref.json"))["cases"]["long"]; print("".join("<t%03d>" % t for t in c["prompt_ids"]))')
  ./deepseek_v4 ./deepseek_v4_tiny "$prompt" --raw-prompt --max-tokens 4 --record-oracle v4-cpu.json > /dev/null
  COLI_VK_DENSE=1 COLI_VULKAN=1 ./deepseek_v4 ./deepseek_v4_tiny "$prompt" --raw-prompt --max-tokens 4 --record-oracle v4-vk.json > /dev/null 2> v4-vk.err
  $PY - <<'PY' || fail "deepseek_v4: the Vulkan session differs from the CPU's"
import json, sys
a, b = json.load(open("v4-cpu.json")), json.load(open("v4-vk.json"))
ref = json.load(open("deepseek_v4_tiny/ref.json"))["cases"]["long"]["greedy_full_ids"]
ok = a["full_ids"] == b["full_ids"] == ref and a["tf_pred"] == b["tf_pred"]
print("OK deepseek_v4 session: ids = CPU = reference" if ok else ("CPU", a, "VK", b, "ref", ref))
sys.exit(0 if ok else 1)
PY
  need_gpu deepseek_v4 v4-vk.err "deepseek_v4 session"
  echo "OK deepseek_v4: $(vk_count deepseek_v4 v4-vk.err) matmuls on the GPU, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' v4-vk.err | tail -1)"

  # The routed-expert tier on deepseek_v4's fp4 experts (fmt 7) with its own
  # activation (VKT_ACT_SWIGLU_V4: the CPU kernel's bf16 and E4M3 roundings and the
  # route weight before down, on the device). Every oracle case against the CPU run:
  # ids equal to the CPU's and the reference's, teacher-forced predictions equal,
  # routed experts served by the device. The history is the store's own .coli_usage
  # in the fixture, removed before each run unless WARM=1 (the tier then warm-starts
  # from the CPU run's): the runs use a copy of the fixture, whose committed history
  # stays as it is. An 8-expert variant of the fixture has room for pinned hot
  # experts, which the store keeps in its rows16 layout: the tier gets them unpacked.
  rm -rf deepseek_v4_tiny_t && cp -r deepseek_v4_tiny deepseek_v4_tiny_t
  $PY - tools/make_deepseek_v4_tiny.py deepseek_v4_tiny_e8 <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
gen.EXPERTS = 8          # 8 routed experts a layer: two cache slots a layer for pins
sys.argv = ["make_deepseek_v4_tiny.py", "--output", sys.argv[2], "--force"]
gen.main()
PY
  v4_tier() {  # <tag> <fixture> <case> <env...>
    local tag=$1 fx=$2 c=$3; shift 3
    local p mt
    p=$($PY -c 'import json,sys; c=json.load(open(sys.argv[1]+"/ref.json"))["cases"][sys.argv[2]]; print("".join("<t%03d>" % t for t in c["prompt_ids"]))' $fx $c)
    mt=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]+"/ref.json"))["cases"][sys.argv[2]]["max_new_tokens"])' $fx $c)
    rm -f $fx/.coli_usage
    env "$@" ./deepseek_v4 ./$fx "$p" --raw-prompt --max-tokens $mt --record-oracle v4-cpu.json > /dev/null 2> v4-cpu.err ||
      { cat v4-cpu.err; fail "deepseek_v4 tier $tag: CPU run"; }
    [ "${WARM:-0}" = 1 ] || rm -f $fx/.coli_usage
    env "$@" COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 ./deepseek_v4 ./$fx "$p" --raw-prompt --max-tokens $mt --record-oracle v4-vk.json > /dev/null 2> v4-vk.err ||
      { cat v4-vk.err; fail "deepseek_v4 tier $tag: Vulkan run"; }
    rm -f $fx/.coli_usage
    $PY - $fx $c <<'PY' || fail "deepseek_v4 tier $tag: the Vulkan session differs from the CPU's"
import json, sys
a, b = json.load(open("v4-cpu.json")), json.load(open("v4-vk.json"))
ref = json.load(open(sys.argv[1] + "/ref.json"))["cases"][sys.argv[2]]["greedy_full_ids"]
ok = a["full_ids"] == b["full_ids"] == ref and a["tf_pred"] == b["tf_pred"]
if not ok: print("CPU", a, "VK", b, "ref", ref)
sys.exit(0 if ok else 1)
PY
    [ "$(tier_count deepseek_v4 v4-vk.err)" -gt 0 ] || { cat v4-vk.err; fail "deepseek_v4 tier $tag: no routed expert ran on the device"; }
    if [ "${EVICT:-0}" = 1 ]; then
      [ "$(tier_evictions deepseek_v4 v4-vk.err)" -gt 0 ] || { grep '\[VK\] tier' v4-vk.err; fail "deepseek_v4 tier $tag: the budget forced no eviction"; }
      [ "$(tier_failed deepseek_v4 v4-vk.err)" = 0 ] || { grep '\[VK\] tier' v4-vk.err; fail "deepseek_v4 tier $tag: an upload failed (the budget shrank)"; }
    fi
    if [ "${WARM:-0}" = 1 ]; then
      grep -q '^\[VK\] tier deepseek_v4: warm start' v4-vk.err || { grep '\[VK\] tier' v4-vk.err; fail "deepseek_v4 tier $tag: no warm start"; }
    fi
    echo "OK deepseek_v4 tier $tag ($c): ids = CPU = reference, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' v4-vk.err | tail -1), $(grep -a -o 'evictions [0-9]*' v4-vk.err | tail -1), $(grep -a -o 'packed_slots=[0-9]*' v4-vk.err) rows16"
  }
  for c in short compressed long; do
    v4_tier "4 experts" deepseek_v4_tiny_t $c
    v4_tier "8 experts, pinned rows16" deepseek_v4_tiny_e8 $c
  done
  EVICT=1 v4_tier "a budget of three experts" deepseek_v4_tiny_e8 long COLI_VK_TIER_GB=0.00009
  v4_tier "trunk on the device too" deepseek_v4_tiny_t long COLI_VK_DENSE=1
  v4_tier "the GEMM route from 2 rows" deepseek_v4_tiny_t long COLI_VK_TIER_GEMM_ROWS=2
  WARM=1 v4_tier "warm start from the CPU run's history" deepseek_v4_tiny_e8 long
  # A served prompt with every routed expert on the device (warm start from the CPU
  # run's history): its per-position logprob echoes against the CPU's. Measured on
  # Lavapipe they are the same bytes; the gate allows 1e-3, for an exp() that rounds
  # one hidden value to the other bf16 neighbour on another driver.
  $PY - deepseek_v4_tiny_e8 <<'PY' || fail "deepseek_v4 tier: served logprobs differ from the CPU's"
import json, os, sys
from pathlib import Path
sys.path.insert(0, "tests")
import test_deepseek_v4_brio as b
fx = Path(sys.argv[1])
case = json.load(open(fx / "ref.json"))["cases"]["long"]
def echoes(vk):
    os.environ.pop("COLI_VULKAN", None)
    if vk: os.environ.update(COLI_VULKAN="1", COLI_VK_TIER_SYNC="1")
    else:
        try: os.remove(fx / ".coli_usage")
        except FileNotFoundError: pass
    s = b.Serve(Path("deepseek_v4").resolve(), fx)
    try: return s.submit(b.token_prompt(case["prompt_ids"]), 0, logprobs=5).echoes
    finally:
        s.close()
        tier = [l for l in s.process.stderr.read().decode(errors="replace").splitlines() if "tier deepseek_v4 turn:" in l]
        if vk: print("  ", tier[-1][:90] if tier else "no tier line")
cpu, dev = echoes(False), echoes(True)
os.environ.pop("COLI_VULKAN", None)
same = sum(cpu[p] == dev.get(p) for p in cpu)
worst = max(abs(cpu[p]["lp"] - dev[p]["lp"]) for p in cpu) if len(cpu) == len(dev) else 1.0
print(f"OK deepseek_v4 tier served: {same} of {len(cpu)} positions' logprob echoes identical to the CPU's, worst |delta| {worst:.2e}")
sys.exit(0 if len(cpu) == len(dev) and worst <= 1e-3 else 1)
PY
  rm -rf deepseek_v4_tiny_t deepseek_v4_tiny_e8
}

# The routed-expert tier of both deepseek engines under ASan and UBSan: sanitized VK=1
# builds, the tier's configurations on Lavapipe (decode and prefill, eviction, DSpark,
# pinned rows16 experts, the warm start, the trunk on the device). Memory safety is
# the gate; each run must still put experts on the device.
family_deepseek_sanitize() {
  local SAN="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  make clean >/dev/null 2>&1 || true
  make deepseek_v41 VK=1 EXTRA_CFLAGS="$SAN"
  make deepseek-v4 VK=1 LTO=0 EXTRA_CFLAGS="$SAN" EXTRA_LDFLAGS="$SAN"
  $PY tools/make_dsv41_tiny.py --out dsv41_tiny --emit-ref dsv41_tiny/ref.json
  $PY tools/make_dsv41_tiny.py --out dsv41_long --emit-ref dsv41_long/ref.json --prompt-len 40 --max-new 6
  $PY tools/make_deepseek_v4_tiny.py --output deepseek_v4_tiny_t --force
  $PY - tools/make_deepseek_v4_tiny.py deepseek_v4_tiny_e8 <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
gen.EXPERTS = 8
sys.argv = ["make_deepseek_v4_tiny.py", "--output", sys.argv[2], "--force"]
gen.main()
PY
  san() {  # <engine> <tag> <env and argv...>
    local eng=$1 tag=$2; shift 2
    rm -f tier.usage deepseek_v4_tiny_t/.coli_usage deepseek_v4_tiny_e8/.coli_usage
    env ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2 \
      COLI_USAGE=tier.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 "$@" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    [ "$(tier_count "$eng" san.log)" -gt 0 ] || { cat san.log; fail "$tag: no routed expert ran on the device"; }
    echo "OK $tag: sanitizers clean, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' san.log | tail -1), $(grep -a -o 'evictions [0-9]*' san.log | tail -1)"
  }
  local cap force p
  for cap in 1 8; do san deepseek_v41 "asan deepseek_v41 cap=$cap" SNAP=dsv41_long ./deepseek_v41 $cap dsv41_long/ref.json; done
  for force in 2 4; do san deepseek_v41 "asan deepseek_v41 DSpark spec=$force" SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=$force ./deepseek_v41 1 dsv41_tiny/ref.json; done
  san deepseek_v41 "asan deepseek_v41 eviction" COLI_VK_TIER_GB=0.00005 SNAP=dsv41_long ./deepseek_v41 8 dsv41_long/ref.json
  san deepseek_v41 "asan deepseek_v41 trunk on the device" COLI_VK_DENSE=1 SNAP=dsv41_tiny ./deepseek_v41 2 dsv41_tiny/ref.json
  p=$($PY -c 'import json; c=json.load(open("deepseek_v4_tiny_t/ref.json"))["cases"]["long"]; print("".join("<t%03d>" % t for t in c["prompt_ids"]))')
  san deepseek_v4 "asan deepseek_v4" ./deepseek_v4 ./deepseek_v4_tiny_t "$p" --raw-prompt --max-tokens 4 --record-oracle san.json
  san deepseek_v4 "asan deepseek_v4 pinned rows16" ./deepseek_v4 ./deepseek_v4_tiny_e8 "$p" --raw-prompt --max-tokens 4 --record-oracle san.json
  san deepseek_v4 "asan deepseek_v4 eviction" COLI_VK_TIER_GB=0.00009 ./deepseek_v4 ./deepseek_v4_tiny_e8 "$p" --raw-prompt --max-tokens 4 --record-oracle san.json
  san deepseek_v4 "asan deepseek_v4 trunk on the device" COLI_VK_DENSE=1 ./deepseek_v4 ./deepseek_v4_tiny_t "$p" --raw-prompt --max-tokens 4
  # the warm start: a run that leaves its history, then one that starts from it
  ./deepseek_v4 ./deepseek_v4_tiny_e8 "$p" --raw-prompt --max-tokens 4 > /dev/null 2>&1 || true
  env ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 \
    ./deepseek_v4 ./deepseek_v4_tiny_e8 "$p" --raw-prompt --max-tokens 4 > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan deepseek_v4 warm start: sanitizer diagnostic"; fi
  grep -q '^\[VK\] tier deepseek_v4: warm start' san.log || { cat san.log; fail "asan deepseek_v4 warm start: no warm start"; }
  echo "OK asan deepseek_v4 warm start: sanitizers clean, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' san.log | tail -1)"
  rm -rf deepseek_v4_tiny_t deepseek_v4_tiny_e8 san.json
  make clean >/dev/null 2>&1 || true
  make deepseek-v4-clean >/dev/null 2>&1 || true
}

# Kimi K3: the routed experts on the shared tier (MXFP4 with ue8m0 scales, fmt 7,
# SiTU-GLU in the latent space), against Moonshot's vendor oracle and the CPU run.
# K3_IDOT=0 everywhere: the CPU's default int8-activation expert kernel is an
# approximation the device does not make (the oracle's engine_env sets it too).
family_kimi() {
  make kimi_k3 VK=1
  $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  # Moonshot's oracle (greedy, teacher forcing at every position, determinism, the
  # bite) with the tier on, the shared experts where the default puts them (the CPU
  # on Lavapipe) and on the device
  COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 \
    $PY tests/test_kimi_k3_tiny.py --binary ./kimi_k3 --fixture ./kimi_k3_tiny
  COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 COLI_VK_DENSE=1 \
    $PY tests/test_kimi_k3_tiny.py --binary ./kimi_k3 --fixture ./kimi_k3_tiny
  k3ids() { $PY -c "import json,sys;print(' '.join(map(str,json.load(open('kimi_k3_tiny/ref.json'))['cases'][sys.argv[1]]['prompt_ids'])))" "$1"; }
  # k3_gate <tag> <env...>: the tokens of the CPU run, the tier served some experts,
  # the shared experts where they were asked to be (dense_where); EVICT=1: it evicted
  k3_gate() {
    local tag=$1 c; shift
    for c in short chunk long; do
      rm -f k3.usage
      env "$@" COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 ./kimi_k3 kimi_k3_tiny --ids "$(k3ids $c)" --ngen 8 2>/dev/null | sed 's/ *TUNE.*//' > cpu.tok
      env "$@" COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 \
        ./kimi_k3 kimi_k3_tiny --ids "$(k3ids $c)" --ngen 8 2> vk.log | sed 's/ *TUNE.*//' > vk.tok
      { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok vk.log; fail "$tag $c: Vulkan tokens differ from the CPU"; }
      [ "$(tier_count kimi_k3 vk.log)" -gt 0 ] || { cat vk.log; fail "$tag $c: no routed expert ran on the device"; }
      if [ "${EVICT:-0}" = 1 ]; then
        [ "$(tier_evictions kimi_k3 vk.log)" -gt 0 ] || { grep '\[VK\] tier' vk.log; fail "$tag $c: the budget forced no eviction"; }
        grep -q ' failed 0 ' vk.log || { grep '\[VK\] tier' vk.log; fail "$tag $c: an upload failed"; }
      fi
      local where; where=$(dense_where kimi_k3 vk.log "$tag $c" "$@") || { echo "$where"; exit 1; }
      echo "OK $tag $c: tokens = CPU, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' vk.log | tail -1), $(grep -a -o 'evictions [0-9]*' vk.log | tail -1), shared experts on the $where"
    done
  }
  local O="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0" b
  k3_gate "kimi_k3 tier f32" $O
  k3_gate "kimi_k3 tier f32, shared experts on the device" $O COLI_VK_DENSE=1
  for b in 8 4; do   # the shared experts as int8 rows (fmt 1) and int4-g64 (fmt 4) on the device
    k3_gate "kimi_k3 tier K3_BITS=$b, shared experts on the device" K3_BITS=$b K3_MLA_BITS=$b K3_HEAD_BITS=$b K3_IDOT=0 COLI_TEMP=0 COLI_VK_DENSE=1
  done
  k3_gate "kimi_k3 tier, prefill one token at a time" $O K3_CHUNK=1
  k3_gate "kimi_k3 tier, loads not pipelined" $O K3_PIPE=0
  EVICT=1 k3_gate "kimi_k3 tier, a budget of two experts" $O COLI_VK_TIER_GB=0.000005
  # the old switches: K3_VK=1 opens the device as COLI_VULKAN=1 does and K3_VK_GB caps
  # the tier; K3_VK=0 keeps it closed whatever COLI_VULKAN says
  rm -f k3.usage
  env $O COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 ./kimi_k3 kimi_k3_tiny --ids "$(k3ids long)" --ngen 8 2>/dev/null | sed 's/ *TUNE.*//' > cpu.tok
  env $O COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 K3_VK=1 K3_VK_GB=0.000005 \
    ./kimi_k3 kimi_k3_tiny --ids "$(k3ids long)" --ngen 8 2> vk.log | sed 's/ *TUNE.*//' > vk.tok
  cmp -s cpu.tok vk.tok || { cat vk.log; fail "kimi_k3 K3_VK=1: tokens differ from the CPU"; }
  grep -q 'K3_VK=1 read as COLI_VULKAN=1' vk.log && grep -q 'budget [0-9.]* KiB = 2 experts' vk.log &&
    [ "$(tier_count kimi_k3 vk.log)" -gt 0 ] || { cat vk.log; fail "kimi_k3 K3_VK=1 K3_VK_GB: not the tier it asked for"; }
  env $O COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 K3_VK=0 COLI_VULKAN=1 \
    ./kimi_k3 kimi_k3_tiny --ids "$(k3ids long)" --ngen 8 2> vk.log | sed 's/ *TUNE.*//' > vk.tok
  cmp -s cpu.tok vk.tok && ! grep -q '^\[VK\]' vk.log || { cat vk.log; fail "kimi_k3 K3_VK=0: the device opened"; }
  echo "OK kimi_k3 K3_VK=1 / K3_VK_GB / K3_VK=0: the shared tier's switches"
  # a warm start from the history of the run before: the tier starts full, serves
  # every routed expert of the same prompt, and the tokens stay the CPU's
  rm -f k3.usage
  env $O COLI_USAGE=$PWD/k3.usage COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 ./kimi_k3 kimi_k3_tiny --ids "$(k3ids long)" --ngen 8 > /dev/null 2>&1
  env $O COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 \
    ./kimi_k3 kimi_k3_tiny --ids "$(k3ids long)" --ngen 8 2> vk.log | sed 's/ *TUNE.*//' > vk.tok
  cmp -s cpu.tok vk.tok || { cat vk.log; fail "kimi_k3 warm start: tokens differ from the CPU"; }
  grep -q 'tier kimi_k3: warm start, [1-9]' vk.log || { cat vk.log; fail "kimi_k3: no warm start from the history"; }
  echo "OK kimi_k3 warm start: tokens = CPU, $(grep -a -o 'warm start, [0-9]* experts' vk.log), $(grep -a -o 'device [0-9]* of [0-9]* routed experts' vk.log | tail -1)"
  # COLI_VK_TIER=0: the shared experts alone on the device, as before the tier
  env $O COLI_USAGE=$PWD/k3.usage USAGE_SAVE=0 COLI_VK_TIER=0 COLI_VULKAN=1 \
    ./kimi_k3 kimi_k3_tiny --ids "$(k3ids long)" --ngen 8 2> vk.log | sed 's/ *TUNE.*//' > vk.tok
  cmp -s cpu.tok vk.tok || { cat vk.log; fail "kimi_k3 COLI_VK_TIER=0: tokens differ from the CPU"; }
  ! grep -q '^\[VK\] tier' vk.log || { cat vk.log; fail "kimi_k3 COLI_VK_TIER=0: the tier started"; }
  need_gpu kimi_k3 vk.log "kimi_k3 COLI_VK_TIER=0"
  echo "OK kimi_k3 COLI_VK_TIER=0: tokens = CPU, no tier, $(vk_count kimi_k3 vk.log) matmuls on the GPU"
}

# The Kimi K3 and MiMo expert tiers under ASan and UBSan, as qwen-sanitize does for
# the qwen engines: memory safety is the gate, and each run must still put routed
# experts on the device.
family_kimi_mimo_sanitize() {
  make clean >/dev/null 2>&1 || true
  make kimi_k3 mimo VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force --vision
  ksan() {  # <engine> <tag> <env and argv...>
    local eng=$1 tag=$2; shift 2
    env ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2 \
      COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 "$@" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    [ "$(tier_count "$eng" san.log)" -gt 0 ] || { cat san.log; fail "$tag: no routed expert ran on the device"; }
    echo "OK $tag: sanitizers clean, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' san.log | tail -1), $(grep -a -o 'evictions [0-9]*' san.log | tail -1)"
  }
  local O="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0" K3IDS MIDS IMG GRID
  K3IDS=$($PY -c "import json;print(' '.join(map(str,json.load(open('kimi_k3_tiny/ref.json'))['cases']['long']['prompt_ids'])))")
  MIDS=$($PY -c "import json;print(' '.join(map(str,json.load(open('mimo_tiny/ref.json'))['cases']['long']['prompt_ids'])))")
  IMG=$($PY -c "import json;print(' '.join(map(str,json.load(open('mimo_tiny/ref.json'))['image']['prompt_ids'])))")
  GRID=$($PY -c "import json;i=json.load(open('mimo_tiny/ref.json'))['image'];print(i['grid_h'],i['grid_w'])")
  rm -f k3.usage
  ksan kimi_k3 "asan kimi_k3 tier" $O COLI_USAGE=$PWD/k3.usage ./kimi_k3 kimi_k3_tiny --ids "$K3IDS" --ngen 8
  ksan kimi_k3 "asan kimi_k3 warm start, shared experts on the device" $O COLI_USAGE=$PWD/k3.usage COLI_VK_DENSE=1 ./kimi_k3 kimi_k3_tiny --ids "$K3IDS" --ngen 8
  ksan kimi_k3 "asan kimi_k3 eviction, int8 shared experts" K3_BITS=8 K3_IDOT=0 COLI_USAGE=$PWD/k3e.usage USAGE_SAVE=0 COLI_VK_TIER_GB=0.000005 COLI_VK_DENSE=1 ./kimi_k3 kimi_k3_tiny --ids "$K3IDS" --ngen 8
  ksan kimi_k3 "asan kimi_k3 prefill one token at a time, no pipeline" $O COLI_USAGE=$PWD/k3e.usage USAGE_SAVE=0 K3_CHUNK=1 K3_PIPE=0 ./kimi_k3 kimi_k3_tiny --ids "$K3IDS" --ngen 8
  ksan mimo "asan mimo tier" MIMO_DENSE_BITS=32 COLI_TEMP=0 ./mimo mimo_tiny --ids "$MIDS" --ngen 6
  ksan mimo "asan mimo eviction (MIMO_VK_EXPERTS=2)" MIMO_DENSE_BITS=32 COLI_TEMP=0 MIMO_VK_EXPERTS=2 ./mimo mimo_tiny --ids "$MIDS" --ngen 6
  ksan mimo "asan mimo picture, native dense on the device" MIMO_DENSE_BITS=0 COLI_TEMP=0 COLI_VK_DENSE=1 ./mimo mimo_tiny --ids "$IMG" --ngen 6 --image mimo_tiny/patches.f32 --grid $GRID
  ksan mimo "asan mimo prefill blocks of 3, cache 4" MIMO_DENSE_BITS=32 COLI_TEMP=0 MIMO_CHUNK=3 MIMO_CAP=4 ./mimo mimo_tiny --ids "$MIDS" --ngen 6
  make clean >/dev/null 2>&1 || true
}

# MiMo-V2.6's dense chain (mimo_chain.h, COLI_VK_CHAIN=1): every layer's attention with
# its sliding window (a ring of 8 rows on the fixture, which every case outgrows), the
# sink logits, partial RoPE with two thetas and the value scale on the device, the
# routed experts on the host. Gates, per configuration (mimo_chain_gate): the CPU run's
# generated tokens; the teacher-forced logits of every position (MIMO_LOGITS) within
# 1e-4 of the largest one; a "[VK] mimo chain: N forwards" line with N > 0.
mimo_ids() {  # <case|image> <field>: a field of ref.json's case
  $PY -c "import json,sys;r=json.load(open('mimo_tiny/ref.json'));c=r['image'] if sys.argv[1]=='image' else r['cases'][sys.argv[1]];print(' '.join(map(str,c[sys.argv[2]])))" "$1" "$2"
}
mimo_chain_gate() {  # <tag> <case> <env...>   (CHAINMODE: COLI_VK_CHAIN's value, default 1)
  local tag=$1 c=$2 extra=() fw lg; shift 2
  [ "$c" = image ] && extra=(--image mimo_tiny/patches.f32 --grid $MGRID)
  env "$@" COLI_TEMP=0 ./mimo mimo_tiny --ids "$(mimo_ids $c prompt_ids)" --ngen 6 "${extra[@]}" > mimo-cpu.txt 2>/dev/null
  env "$@" COLI_TEMP=0 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} COLI_VK_TIER_SYNC=1 \
    ./mimo mimo_tiny --ids "$(mimo_ids $c prompt_ids)" --ngen 6 "${extra[@]}" > mimo-vk.txt 2> mimo-vk.err
  [ -s mimo-cpu.txt ] && cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-cpu.txt mimo-vk.txt mimo-vk.err; fail "$tag $c: the chain's tokens differ from the CPU"; }
  fw=$(chain_count mimo mimo-vk.err)
  [ "$fw" -gt 0 ] || { cat mimo-vk.err; fail "$tag $c: the chain never ran"; }
  rm -f cpu.f32 vk.f32
  env "$@" MIMO_LOGITS=cpu.f32 ./mimo mimo_tiny --ids "$(mimo_ids $c greedy_full_ids)" --ngen 0 "${extra[@]}" > /dev/null 2>&1
  env "$@" MIMO_LOGITS=vk.f32 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} COLI_VK_TIER_SYNC=1 \
    ./mimo mimo_tiny --ids "$(mimo_ids $c greedy_full_ids)" --ngen 0 "${extra[@]}" > /dev/null 2> mimo-vk.err
  lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag $c: logits"; }
  echo "OK $tag $c: tokens = CPU, $lg, $fw chain forwards"
}
# mimo_lost_gate <tag> <case> <back> <env...>: the device "lost" (COLI_VK_CHAIN_FAULT) at
# the chain frame <back> frames before the end of the same run without a fault (the
# frames a device spends setting up differ: on Lavapipe its memory needs no zeroing
# frames). The host's caches hold whole steps only, so the CPU runs the rest of the
# step from where they end, with nothing to rebuild (a picture included): the CPU
# run's tokens all the same.
mimo_lost_gate() {
  local tag=$1 c=$2 back=$3 extra=() frames; shift 3
  [ "$c" = image ] && extra=(--image mimo_tiny/patches.f32 --grid $MGRID)
  env "$@" COLI_TEMP=0 ./mimo mimo_tiny --ids "$(mimo_ids $c prompt_ids)" --ngen 6 "${extra[@]}" > mimo-cpu.txt 2>/dev/null
  env "$@" COLI_TEMP=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 \
    ./mimo mimo_tiny --ids "$(mimo_ids $c prompt_ids)" --ngen 6 "${extra[@]}" > /dev/null 2> mimo-vk.err
  frames=$(sed -n 's/^\[VK\] mimo chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p' mimo-vk.err | tail -1)
  [ "${frames:-0}" -gt "$back" ] || { cat mimo-vk.err; fail "$tag: the run had no frame $back before its end"; }
  env "$@" COLI_TEMP=0 COLI_VK_CHAIN_FAULT=$((frames - back)) COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 \
    ./mimo mimo_tiny --ids "$(mimo_ids $c prompt_ids)" --ngen 6 "${extra[@]}" > mimo-vk.txt 2> mimo-vk.err
  [ -s mimo-cpu.txt ] && cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-cpu.txt mimo-vk.txt mimo-vk.err; fail "$tag: tokens differ from the CPU"; }
  grep -q "the device was lost at position [0-9]" mimo-vk.err || { cat mimo-vk.err; fail "$tag: the device was never lost"; }
  echo "OK $tag: tokens = CPU, $(grep -o 'the device was lost at position [0-9]*' mimo-vk.err)"
}
mimo_served_fixture() {  # the fixture plus a byte tokenizer, for the serve tests
  rm -rf mimo_tiny_served
  $PY -c "import sys, shutil; sys.path.insert(0, 'tests'); import mimo_serve_fixture as f; shutil.copytree(f.served_fixture(), 'mimo_tiny_served')"
}

family_mimo_chain() {
  make mimo tests/test_vk_chain VK=1
  ./tests/test_vk_chain shaders/qmatmul.spv | tee vk_chain.log
  tail -1 vk_chain.log | grep -qx PASS || fail "the chain's ops"
  $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force --vision
  MGRID=$($PY -c "import json;i=json.load(open('mimo_tiny/ref.json'))['image'];print(i['grid_h'],i['grid_w'])")
  # Xiaomi's vendor oracle with the chain on: greedy and teacher-forced in every case,
  # the native FP8/BF16 trunk, prefill in blocks of 3 and of 1, the picture
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 $PY tests/mimo_tiny_harness.py --binary ./mimo --fixture ./mimo_tiny
  # every dense form (MIMO_DENSE_BITS 32: f32, fmt 10; 0: the release's FP8 and BF16, fmt
  # 12 and 11; 8: int8 rows, fmt 1), the prompt in one block (the tiled GEMM, the window
  # gathered beside the block) and one token at a time (the ring read in place), text
  # and picture, the routed experts on the tier
  local bits ch c k
  for bits in 32 0 8; do for ch in 64 1; do for c in short window long image; do
    mimo_chain_gate "chain mimo bits=$bits MIMO_CHUNK=$ch" $c MIMO_DENSE_BITS=$bits MIMO_CHUNK=$ch
  done; done; done
  for c in short long; do
    mimo_chain_gate "chain mimo blocks of 3" $c MIMO_DENSE_BITS=32 MIMO_CHUNK=3
    mimo_chain_gate "chain mimo tier off" $c MIMO_DENSE_BITS=0 COLI_VK_TIER=0
    mimo_chain_gate "chain mimo experts on the CPU (MIMO_VK_EXPERTS=0)" $c MIMO_DENSE_BITS=32 MIMO_VK_EXPERTS=0
    mimo_chain_gate "chain mimo a block in chunks of 3 rows" $c MIMO_DENSE_BITS=32 COLI_VK_CHAIN_ROWS=3
    mimo_chain_gate "chain mimo tiled GEMM from S=2" $c MIMO_DENSE_BITS=0 COLI_VK_GEMM_MIN_S=2
    mimo_chain_gate "chain mimo the per-row GEMV" $c MIMO_DENSE_BITS=32 COLI_VK_CHAIN_GEMV=0
    mimo_chain_gate "chain mimo eviction (MIMO_VK_EXPERTS=2)" $c MIMO_DENSE_BITS=32 MIMO_VK_EXPERTS=2
  done
  # the picture's tower on the device too (COLI_VK_DENSE=1: the per-matrix path for the
  # tower, the chain for the layers)
  mimo_chain_gate "chain mimo tower on the device" image MIMO_DENSE_BITS=0 COLI_VK_DENSE=1
  # The window boundary, bit for bit: with every matrix on the per-row GEMV and the tier
  # off the chain's logits are the same bytes whatever the blocks -- one row reading the
  # ring in place, blocks of 3, of the window's 8 and of 9 rows gathering it, the whole
  # prompt -- so a row's window is the same rows in the same order on every path.
  # Keep the per-row attention too: blocked attention has its own reduction order;
  # the default blocked path's numerical gates above remain enabled.
  for c in window long image; do
    local x=(); [ $c = image ] && x=(--image mimo_tiny/patches.f32 --grid $MGRID)
    for ch in 64 1 3 8 9; do
      COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER=0 COLI_VK_GEMM_MIN_S=0 COLI_VK_ATTN_BLOCK=0 MIMO_DENSE_BITS=32 MIMO_CHUNK=$ch MIMO_LOGITS=blk$ch.f32 \
        ./mimo mimo_tiny --ids "$(mimo_ids $c greedy_full_ids)" --ngen 0 "${x[@]}" > /dev/null 2> mimo-vk.err
      [ "$(chain_count mimo mimo-vk.err)" -gt 0 ] || { cat mimo-vk.err; fail "chain mimo blocks $ch $c: the chain never ran"; }
      [ $ch = 64 ] || cmp -s blk64.f32 blk$ch.f32 || fail "chain mimo $c: blocks of $ch are not the bytes of one block"
    done
    echo "OK chain mimo window boundary $c: blocks of 64, 1, 3, 8, 9 give the same logits bytes"
  done
  # COLI_VK_CHAIN=2: prompts on the device, decode on the CPU, the caches moving between them
  for c in short long image; do CHAINMODE=2 mimo_chain_gate "chain mimo prompts only" $c MIMO_DENSE_BITS=32; done
  # the device lost (frames counted back from the end of the run: 5 decode steps of 6
  # frames follow the prompt): inside the last decode step; inside the first block of
  # a prompt split in chunks of 3 rows (the CPU takes the step's remaining rows); inside
  # a picture's prompt; inside a prompt fed one token at a time
  mimo_lost_gate "chain mimo device lost mid-decode" long 3 MIMO_DENSE_BITS=32
  mimo_lost_gate "chain mimo device lost inside a chunked prompt" long 100 MIMO_DENSE_BITS=32 COLI_VK_CHAIN_ROWS=3
  mimo_lost_gate "chain mimo device lost inside a picture's prompt" image 33 MIMO_DENSE_BITS=0
  mimo_lost_gate "chain mimo device lost, one token at a time" short 50 MIMO_DENSE_BITS=32 MIMO_CHUNK=1
  # the prefix-reuse contract and Brio photos (the rings restored into the device's
  # mirrors), compared bit for bit with a cold engine: the per-row GEMV and no tier, so
  # a resumed prompt's rows get the bits of the same rows computed cold
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER=0 COLI_VK_GEMM_MIN_S=0 COLI_VK_ATTN_BLOCK=0 $PY -m unittest tests.test_mimo_prefix_serve
  # a serve session against the CPU's, frame for frame, in both modes and with the
  # device lost mid-session. The fixture's logits run to 300 (its head is scaled for wide
  # greedy margins), so its logprobs compare within 1e-3: about 3e-6 of the largest logit
  mimo_served_fixture
  for k in 1 2; do CHAIN_SERVE_TOL=1e-3 COLI_VK_CHAIN=$k $PY tests/vulkan_chain_serve.py ./mimo mimo_tiny_served OMP_NUM_THREADS=2; done
  CHAIN_SERVE_TOL=1e-3 $PY tests/vulkan_chain_serve.py ./mimo mimo_tiny_served OMP_NUM_THREADS=2 COLI_VK_CHAIN_FAULT=60
}

# MiMo's chain under ASan and UBSan: memory safety is the gate (a sanitized build
# vectorizes differently, so tokens are not compared); each run must have run the chain.
family_mimo_chain_sanitize() {
  make clean >/dev/null 2>&1 || true
  make mimo tests/test_vk_chain VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  ./tests/test_vk_chain shaders/qmatmul.spv > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log || ! tail -1 san.log | grep -qx PASS; then cat san.log; fail "asan: the chain's ops"; fi
  echo "OK asan: the chain's ops"
  $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force --vision
  MGRID=$($PY -c "import json;i=json.load(open('mimo_tiny/ref.json'))['image'];print(i['grid_h'],i['grid_w'])")
  msan() {  # <tag> <case> <env...>
    local tag=$1 c=$2 x=(); shift 2
    [ "$c" = image ] && x=(--image mimo_tiny/patches.f32 --grid $MGRID)
    env OMP_NUM_THREADS=2 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_TEMP=0 "$@" \
      ./mimo mimo_tiny --ids "$(mimo_ids $c prompt_ids)" --ngen 6 "${x[@]}" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    [ "$(chain_count mimo san.log)" -gt 0 ] || { cat san.log; fail "$tag: the chain never ran"; }
    echo "OK $tag: sanitizers clean, $(chain_count mimo san.log) chain forwards"
  }
  msan "asan chain mimo f32, the tier" long MIMO_DENSE_BITS=32
  msan "asan chain mimo native FP8/BF16, picture, tower on the device" image MIMO_DENSE_BITS=0 COLI_VK_DENSE=1
  msan "asan chain mimo blocks of 3 in chunks of 2 rows" long MIMO_DENSE_BITS=32 MIMO_CHUNK=3 COLI_VK_CHAIN_ROWS=2
  msan "asan chain mimo one token at a time, int8" window MIMO_DENSE_BITS=8 MIMO_CHUNK=1
  msan "asan chain mimo experts on the CPU" short MIMO_DENSE_BITS=32 MIMO_VK_EXPERTS=0
  # the device lost inside a prompt split in chunks of 3 rows: the frame 100 before the
  # end of the same run without a fault (a device's setup frames differ)
  msan "asan chain mimo frames to the end" long MIMO_DENSE_BITS=32 COLI_VK_CHAIN_ROWS=3
  local frames
  frames=$(sed -n 's/^\[VK\] mimo chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p' san.log | tail -1)
  env OMP_NUM_THREADS=2 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_TEMP=0 MIMO_DENSE_BITS=32 COLI_VK_CHAIN_ROWS=3 \
    COLI_VK_CHAIN_FAULT=$((frames - 100)) ./mimo mimo_tiny --ids "$(mimo_ids long prompt_ids)" --ngen 6 > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan chain mimo device lost: sanitizer diagnostic"; fi
  grep -q "the device was lost at position [1-9]" san.log || { cat san.log; fail "asan chain mimo device lost: not lost inside the prompt"; }
  echo "OK asan chain mimo device lost: sanitizers clean, $(grep -o 'the device was lost at position [0-9]*' san.log)"
  mimo_served_fixture
  CHAIN_SERVE_TOL=1e-3 $PY tests/vulkan_chain_serve.py ./mimo mimo_tiny_served OMP_NUM_THREADS=2 > san.log 2>&1 || { cat san.log; fail "asan chain mimo serve"; }
  echo "OK asan chain mimo serve: $(tail -1 san.log)"
  make clean >/dev/null 2>&1 || true
}

# GLM-5.2 (colibri) and GLM-5.3 Flash (glm53) on the routed-expert tier: the fixtures.
# colibri: the bf16 oracle, whose experts the loader quantizes to the bits asked for
# (16: f32, fmt 10 on the device; 8: int8, fmt 1; 4: int4 per row, fmt 2; 3: int3-g64,
# fmt 5), the int4-g64 and E8/IQ3 containers of the parity gate, and the FP8 oracle
# converted to int4-g64 (its 32-wide down rows stay per row), int4 per row, int3-g64 and
# int4-g64 with an int3-g64 down. glm53: the int4-gs64 streaming container.
glm_fixtures() {
  $PY tools/make_glm_oracle.py > /dev/null
  $PY tools/make_glm_oracle.py --fmt4 > /dev/null
  $PY tools/make_glm_oracle.py --fmt6 > /dev/null
  mkdir -p glm_fp8 && (cd glm_fp8 && $PY ../tools/make_glm_oracle.py --fp8 > /dev/null)
  local v
  for v in "i4:" "i4r:--group-size 0" "i3:--xbits 3" "d3:--down-bits 3"; do
    # shellcheck disable=SC2086
    $PY tools/convert_fp8_to_int4.py --indir glm_fp8/glm_tiny --outdir glm_tiny_${v%%:*} \
      --ebits 4 --io-bits 4 --n-layers 5 --min-free-gb 0 ${v#*:} > /dev/null
    cp glm_fp8/ref_glm.json glm_tiny_${v%%:*}/
  done
  $PY tools/make_glm53_tiny.py --output glm53_tiny --force > /dev/null
  $PY tools/make_glm53_streaming_pair.py --fixture glm53_tiny --output glm53_stream > /dev/null
}

# glm_tier <tag> <snap> <ref> <env...> -- <argv...>
# colibri with the tier against its CPU run: the same greedy tokens (or teacher-forced
# predictions, TF=1, mismatches included) and a "[VK] tier colibri run: device N" with
# N > 0; EVICT=1: evictions too. The history the tier warms from is <snap>/.coli_usage:
# none by default (the tier fills as experts pass by), HIST=1 writes one first from a
# greedy run (the warm start, and the device in a teacher-forced prefill). DEV2=1: the
# second device's registry served experts ("+ N vk" in the hit-rate line).
glm_tier() {
  local tag=$1 snap=$2 ref=$3; shift 3
  local envs=() pre=() e; while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  for e in "${envs[@]}"; do [ "${e%%=*}" = TF ] || pre+=("$e"); done
  rm -f "$snap/.coli_usage"
  [ "${HIST:-0}" = 1 ] && { env "${pre[@]}" SNAP=$snap REF=$ref STATS=$snap/.coli_usage ./colibri "$@" > /dev/null 2>&1 || true; }
  env "${envs[@]}" SNAP=$snap REF=$ref ./colibri "$@" > cpu.log 2>&1 || true
  env "${envs[@]}" SNAP=$snap REF=$ref COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 ./colibri "$@" > vk.log 2>&1 || true
  rm -f "$snap/.coli_usage"
  grep -aE '^GLM C engine|^PREFILL|^\[ORACLE\] mismatch' cpu.log | sed 's/ | [0-9.]* pos\/s//' > cpu.tok || true
  grep -aE '^GLM C engine|^PREFILL|^\[ORACLE\] mismatch' vk.log | sed 's/ | [0-9.]* pos\/s//' > vk.tok || true
  { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; tail -20 vk.log; fail "$tag: the tier's tokens differ from the CPU"; }
  if [ "${DEV2:-0}" = 1 ]; then
    grep -qaE 'lru \+ [1-9][0-9]* vk /' vk.log || { cat vk.log; fail "$tag: the second device served no expert"; }
  else
    [ "$(tier_count colibri vk.log)" -gt 0 ] || { cat vk.log; fail "$tag: no routed expert ran on the device"; }
  fi
  if [ "${EVICT:-0}" = 1 ]; then
    [ "$(tier_evictions colibri vk.log)" -gt 0 ] || { grep '\[VK\] tier' vk.log; fail "$tag: the budget forced no eviction"; }
    [ "$(tier_failed colibri vk.log)" = 0 ] || { grep '\[VK\] tier' vk.log; fail "$tag: an upload failed (the budget shrank)"; }
  fi
  echo "OK $tag: tokens = CPU, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' vk.log | tail -1)$(grep -a -o ' + [0-9]* vk' vk.log | tail -1), $(grep -a -o 'evictions [0-9]*' vk.log | tail -1), $(grep -a -o '(fmt [^)]*)' vk.log | head -1)"
}

# g53_tier <tag> <model dir> <env...> -- <argv...>: glm53, the same gate on its
# teacher_forcing and greedy lines; the history is a fresh COLI_USAGE file (HIST=1: one
# greedy run writes it first).
g53_tier() {
  local tag=$1 model=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f g53.usage
  [ "${HIST:-0}" = 1 ] && { env "${envs[@]}" COLI_USAGE=g53.usage ./glm53 --model $model "$@" > /dev/null 2>&1 || true; }
  env "${envs[@]}" COLI_USAGE=g53.usage USAGE_SAVE=0 ./glm53 --model $model "$@" > cpu.log 2>&1 || true
  env "${envs[@]}" COLI_USAGE=g53.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 ./glm53 --model $model "$@" > vk.log 2>&1 || true
  rm -f g53.usage
  grep -aE '^teacher_forcing|^greedy' cpu.log > cpu.tok || true
  grep -aE '^teacher_forcing|^greedy' vk.log > vk.tok || true
  { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; tail -20 vk.log; fail "$tag: the tier's tokens differ from the CPU"; }
  [ "$(tier_count glm53 vk.log)" -gt 0 ] || { cat vk.log; fail "$tag: no routed expert ran on the device"; }
  if [ "${EVICT:-0}" = 1 ]; then
    [ "$(tier_evictions glm53 vk.log)" -gt 0 ] || { grep '\[VK\] tier' vk.log; fail "$tag: the budget forced no eviction"; }
    [ "$(tier_failed glm53 vk.log)" = 0 ] || { grep '\[VK\] tier' vk.log; fail "$tag: an upload failed (the budget shrank)"; }
  fi
  local where; where=$(dense_where glm53 vk.log "$tag" "${envs[@]}") || { echo "$where"; exit 1; }
  echo "OK $tag: tokens = CPU, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' vk.log | tail -1), $(grep -a -o 'evictions [0-9]*' vk.log | tail -1), trunk on the $where"
}

# GLM-5.2 and GLM-5.3 Flash on the routed-expert tier, in every expert format the two
# engines read, warm and cold, decode and teacher-forced prefill, under eviction (a
# budget of two experts, by COLI_VK_TIER_GB and by the deprecated COLI_VK_EXPERTS), with
# the trunk and the attention core on the device, beside COLI_VK_DEV2's second
# (logical) device, and with the tier off. IDOT=0 gives the CPU's int8 and int4-per-row
# experts the f32 activations the device uses (the CPU default rounds them to int8 on
# some ISAs), so the two sides differ by summation order only.
family_glm() {
  make colibri glm53 VK=1
  glm_fixtures
  export OMP_NUM_THREADS=2 CAP_RAISE=0
  local cap b
  # the oracle with f32 experts (fmt 10): decode cold at a one-slot and a full cache,
  # the teacher-forced prefill from a warm start
  for cap in 1 64; do glm_tier "colibri f32 experts cap=$cap" glm_tiny ref_glm.json -- $cap 16 16; done
  HIST=1 glm_tier "colibri f32 experts, prefill from a warm start" glm_tiny ref_glm.json TF=1 -- 64 16 16
  # quantized at load: int8 (fmt 1), int4 per row (fmt 2), int3-g64 (fmt 5)
  for b in 8 4 3; do
    glm_tier "colibri ${b}-bit experts cap=2" glm_tiny ref_glm.json IDOT=0 -- 2 $b 4
    HIST=1 glm_tier "colibri ${b}-bit experts, prefill warm" glm_tiny ref_glm.json IDOT=0 TF=1 -- 2 $b 4
  done
  # the containers: int4-g64 gate/up/down, int4-g64 with a per-row down, int4 per row,
  # int3-g64, int4-g64 with an int3-g64 down
  HIST=1 glm_tier "colibri int4-g64 container" glm_tiny_fmt4 glm_tiny_fmt4/ref_glm.json -- 2 16 16
  HIST=1 glm_tier "colibri int4-g64 container, prefill" glm_tiny_fmt4 glm_tiny_fmt4/ref_glm.json TF=1 -- 2 16 16
  local fx
  for fx in i4 i4r i3 d3; do
    HIST=1 glm_tier "colibri $fx container cap=1" glm_tiny_$fx glm_tiny_$fx/ref_glm.json IDOT=0 -- 1 4 4
    HIST=1 glm_tier "colibri $fx container, prefill" glm_tiny_$fx glm_tiny_$fx/ref_glm.json IDOT=0 TF=1 -- 2 4 4
  done
  # PIPE's asynchronous loads and the PILOT prefetcher beside the tier
  glm_tier "colibri PIPE=1" glm_tiny_i4 glm_tiny_i4/ref_glm.json IDOT=0 PIPE=1 -- 1 4 4
  glm_tier "colibri PILOT=1" glm_tiny_i4 glm_tiny_i4/ref_glm.json IDOT=0 PILOT=1 -- 2 4 4
  # the trunk and the MLA attention core on the device too
  HIST=1 glm_tier "colibri tier + COLI_VK_DENSE=1 COLI_VK_ATTN=1" glm_tiny_i4 glm_tiny_i4/ref_glm.json IDOT=0 COLI_VK_DENSE=1 COLI_VK_ATTN=1 -- 2 4 4
  # a budget of two experts must evict as the routing moves: COLI_VK_TIER_GB, and the
  # deprecated COLI_VK_EXPERTS mapped onto the tier as a cap
  EVICT=1 glm_tier "colibri, a budget of two experts" glm_tiny ref_glm.json COLI_VK_TIER_GB=0.0001 -- 64 16 16
  EVICT=1 glm_tier "colibri, COLI_VK_EXPERTS=2" glm_tiny ref_glm.json COLI_VK_EXPERTS=2 -- 64 16 16
  # COLI_VK_DEV2: a second logical device on Lavapipe holds what the capped tier does not
  HIST=1 glm_tier "colibri tier + COLI_VK_DEV2" glm_tiny_i4r glm_tiny_i4r/ref_glm.json IDOT=0 COLI_VK_EXPERTS=4 COLI_VK_DEV2=0 -- 64 4 4
  HIST=1 DEV2=1 glm_tier "colibri COLI_VK_DEV2 alone (COLI_VK_TIER=0)" glm_tiny_i4r glm_tiny_i4r/ref_glm.json IDOT=0 COLI_VK_TIER=0 COLI_VK_DEV2=0 -- 64 4 4
  # E8/IQ3 (fmt 6) has no device form: the tier declines, the CPU computes them
  SNAP=glm_tiny_fmt6 REF=glm_tiny_fmt6/ref_glm.json ./colibri 2 16 16 > cpu.log 2>&1 || true
  SNAP=glm_tiny_fmt6 REF=glm_tiny_fmt6/ref_glm.json COLI_VULKAN=1 ./colibri 2 16 16 > vk.log 2>&1 || true
  cmp -s <(grep -a '^GLM C engine' cpu.log) <(grep -a '^GLM C engine' vk.log) || { cat vk.log; fail "colibri fmt 6: tokens differ"; }
  grep -qa 'tier colibri: experts in fmt 6/6/6' vk.log || { cat vk.log; fail "colibri fmt 6: the tier did not decline"; }
  echo "OK colibri E8/IQ3 (fmt 6): the tier declines, tokens = CPU"
  # COLI_VK_TIER=0: no tier, the trunk and the attention core on the device as before
  SNAP=glm_tiny REF=ref_glm.json COLI_VK_DENSE=1 COLI_VK_ATTN=1 ./colibri 2 16 16 > cpu.log 2>&1 || true
  SNAP=glm_tiny REF=ref_glm.json COLI_VK_DENSE=1 COLI_VK_ATTN=1 COLI_VK_TIER=0 COLI_VULKAN=1 ./colibri 2 16 16 > vk.log 2>&1 || true
  cmp -s <(grep -a '^GLM C engine' cpu.log) <(grep -a '^GLM C engine' vk.log) || { cat vk.log; fail "colibri COLI_VK_TIER=0: tokens differ"; }
  ! grep -qa '^\[VK\] tier colibri' vk.log || { cat vk.log; fail "colibri COLI_VK_TIER=0: the tier started"; }
  echo "OK colibri COLI_VK_TIER=0: tokens = CPU, no tier"

  # glm53: its int4-gs64 streaming container (swiglu_limit 10), dense matrices f32 and int4
  local ids
  ids=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
  g53_tier "glm53 decode, cold" glm53_stream-i4 GLM53_BITS=32 -- --ids 5,7,9,11,13,17,19,23 --greedy 8
  HIST=1 g53_tier "glm53 100-token prompt, warm (two blocks of rows)" glm53_stream-i4 GLM53_BITS=32 -- --ids $ids --greedy 4
  HIST=1 g53_tier "glm53 prefill chunks of 7, one cache slot" glm53_stream-i4 GLM53_BITS=4 GLM53_PREFILL_CHUNK=7 GLM53_EXPERT_GB=0.000001 -- --ids $ids --greedy 4
  HIST=1 g53_tier "glm53 trunk on the device (COLI_VK_DENSE=1)" glm53_stream-i4 GLM53_BITS=4 COLI_VK_DENSE=1 -- --ids $ids --greedy 4
  EVICT=1 g53_tier "glm53, a budget of two experts" glm53_stream-i4 GLM53_BITS=32 COLI_VK_TIER_GB=0.00006 -- --ids $ids --greedy 6
  # COLI_VK_TIER=0: the trunk alone, on the device by default, Lavapipe included
  GLM53_BITS=4 COLI_USAGE=g53.usage USAGE_SAVE=0 ./glm53 --model glm53_stream-i4 --ids 5,7,9,11 --greedy 4 > cpu.log 2>&1 || true
  GLM53_BITS=4 COLI_USAGE=g53.usage USAGE_SAVE=0 COLI_VK_TIER=0 COLI_VULKAN=1 ./glm53 --model glm53_stream-i4 --ids 5,7,9,11 --greedy 4 > vk.log 2>&1 || true
  cmp -s <(grep -aE '^teacher_forcing|^greedy' cpu.log) <(grep -aE '^teacher_forcing|^greedy' vk.log) || { cat vk.log; fail "glm53 COLI_VK_TIER=0: tokens differ"; }
  ! grep -qa '^\[VK\] tier glm53' vk.log || { cat vk.log; fail "glm53 COLI_VK_TIER=0: the tier started"; }
  need_gpu glm53 vk.log "glm53 COLI_VK_TIER=0"
  echo "OK glm53 COLI_VK_TIER=0: tokens = CPU, no tier, $(vk_count glm53 vk.log) matmuls on the GPU"
  unset OMP_NUM_THREADS CAP_RAISE
}

# The same engines under ASan and UBSan with the tier on: memory safety is the gate, and
# each run must still put experts on the device.
family_glm_sanitize() {
  make clean >/dev/null 2>&1 || true
  make colibri glm53 VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  glm_fixtures
  export OMP_NUM_THREADS=2 CAP_RAISE=0
  gsan() {  # <engine> <tag> <history: colibri snap, or - for glm53's COLI_USAGE> <env and argv...>
    local eng=$1 tag=$2 snap=$3; shift 3
    rm -f g53.usage; [ "$snap" = - ] || rm -f "$snap/.coli_usage"
    if [ "${HIST:-0}" = 1 ]; then   # a history from a greedy run (TF=1 writes none)
      local a=() x; for x in "$@"; do [ "$x" = TF=1 ] || a+=("$x"); done
      if [ "$snap" = - ]; then env COLI_USAGE=g53.usage "${a[@]}" > /dev/null 2>&1 || true
      else env STATS=$snap/.coli_usage "${a[@]}" > /dev/null 2>&1 || true; fi
    fi
    env ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 COLI_USAGE=g53.usage \
      COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 "$@" > san.log 2>&1 || true
    rm -f g53.usage; [ "$snap" = - ] || rm -f "$snap/.coli_usage"
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    [ "$(tier_count $eng san.log)" -gt 0 ] || { cat san.log; fail "$tag: no routed expert ran on the device"; }
    echo "OK $tag: sanitizers clean, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' san.log | tail -1), $(grep -a -o 'evictions [0-9]*' san.log | tail -1)"
  }
  gsan colibri "asan colibri f32 cap=1" glm_tiny SNAP=glm_tiny REF=ref_glm.json ./colibri 1 16 16
  HIST=1 gsan colibri "asan colibri prefill, warm" glm_tiny SNAP=glm_tiny REF=ref_glm.json TF=1 ./colibri 64 16 16
  gsan colibri "asan colibri eviction" glm_tiny SNAP=glm_tiny REF=ref_glm.json COLI_VK_EXPERTS=2 ./colibri 64 16 16
  HIST=1 gsan colibri "asan colibri int4-g64, trunk + attention on the device" glm_tiny_i4 SNAP=glm_tiny_i4 REF=glm_tiny_i4/ref_glm.json COLI_VK_DENSE=1 COLI_VK_ATTN=1 ./colibri 2 4 4
  HIST=1 gsan colibri "asan colibri int3-g64 prefill" glm_tiny_i3 SNAP=glm_tiny_i3 REF=glm_tiny_i3/ref_glm.json TF=1 ./colibri 2 4 4
  HIST=1 gsan colibri "asan colibri int3-g64 down, cap=1" glm_tiny_d3 SNAP=glm_tiny_d3 REF=glm_tiny_d3/ref_glm.json ./colibri 1 4 4
  HIST=1 gsan colibri "asan colibri tier + COLI_VK_DEV2" glm_tiny_i4r SNAP=glm_tiny_i4r REF=glm_tiny_i4r/ref_glm.json COLI_VK_EXPERTS=4 COLI_VK_DEV2=0 ./colibri 64 4 4
  gsan colibri "asan colibri int4 at load, PIPE=1" glm_tiny SNAP=glm_tiny REF=ref_glm.json PIPE=1 ./colibri 1 4 4
  gsan colibri "asan colibri int8 at load, PILOT=1" glm_tiny SNAP=glm_tiny REF=ref_glm.json PILOT=1 ./colibri 2 8 8
  local ids
  ids=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
  gsan glm53 "asan glm53 decode" - GLM53_BITS=32 ./glm53 --model glm53_stream-i4 --ids 5,7,9,11,13,17,19,23 --greedy 20
  HIST=1 gsan glm53 "asan glm53 100-token prompt, warm" - GLM53_BITS=32 ./glm53 --model glm53_stream-i4 --ids $ids --greedy 4
  HIST=1 gsan glm53 "asan glm53 one slot, trunk on the device" - GLM53_BITS=4 GLM53_EXPERT_GB=0.000001 COLI_VK_DENSE=1 ./glm53 --model glm53_stream-i4 --ids $ids --greedy 4
  gsan glm53 "asan glm53 eviction" - GLM53_BITS=32 COLI_VK_TIER_GB=0.00006 ./glm53 --model glm53_stream-i4 --ids $ids --greedy 6
  unset OMP_NUM_THREADS CAP_RAISE
  make clean >/dev/null 2>&1 || true
}

# The dense chain (vk_chain.c, COLI_VK_CHAIN=1): every layer recorded into one
# submission, the residual stream, the KV mirrors and the recurrent state on the device.
# Gates, per configuration: the CPU run's tokens (and its oracle, where the CPU passes
# it); the last logits within 1e-4 of the largest one where both runs multiply f32
# activations (tol 1); a "[VK] <engine> chain: N forwards" line with N > 0. Beside them:
# the chain's ops against CPU references (tests/test_vk_chain), qwen38's oracle targets
# with the chain on, the prefix-reuse contract, and serve sessions (pins, the prompt
# cache, the prefill read-out, MTP drafts) frame for frame against the CPU.
chain_count() {
  local n
  n=$(sed -n "s/^\[VK\] $1 chain: \([0-9][0-9]*\) forwards.*/\1/p" "$2" | tail -1)
  echo "${n:-0}"
}
logits_close() {  # <cpu.f32> <vk.f32>: max |diff| within 1e-4 of the largest |logit|
  $PY - "$1" "$2" <<'PY'
import array, sys
a = array.array("f", open(sys.argv[1], "rb").read()); b = array.array("f", open(sys.argv[2], "rb").read())
d = max(abs(x - y) for x, y in zip(a, b)) if a and len(a) == len(b) else float("inf")
m = max(abs(x) for x in a) if a else 0.0
print(f"max |logit diff| {d:.2e} of {m:.2e}")
sys.exit(0 if d <= 1e-4 * m else 1)
PY
}
chain_gate() {  # <engine> <tag> <tol 0|1> <env...> -- <argv...>   (CPUENV: the CPU arm's own settings)
  local eng=$1 tag=$2 tol=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage cpu.f32 vk.f32
  local rc_cpu=0
  env "${envs[@]}" ${CPUENV:-} DUMP=cpu.f32 ./"$eng" "$@" > cpu.log 2>&1 || rc_cpu=$?
  env "${envs[@]}" DUMP=vk.f32 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} \
    ./"$eng" "$@" > vk.log 2>&1 || { [ $rc_cpu != 0 ] || { cat vk.log; fail "$tag: the chain misses the oracle the CPU passes"; }; }
  same_tokens cpu.log vk.log "$tag"
  [ "$(chain_count "$eng" vk.log)" -gt 0 ] || { cat vk.log; fail "$tag: the chain never ran"; }
  local lg=""
  if [ "$tol" = 1 ]; then lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag: logits"; }; lg=", $lg"; fi
  echo "OK $tag: tokens = CPU$lg, $(chain_count "$eng" vk.log) chain forwards"
}

# lost_gate <engine> <tag> <frame> <env...> -- <argv...>: the device "lost" at the given
# chain frame (COLI_VK_CHAIN_FAULT); the engine rebuilds the recurrent state on the CPU
# from the prefix record and continues there: the CPU run's tokens all the same.
lost_gate() {
  local eng=$1 tag=$2 k=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage
  env "${envs[@]}" ./"$eng" "$@" > cpu.log 2>&1 || true
  env "${envs[@]}" COLI_VK_CHAIN_FAULT=$k COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 \
    ./"$eng" "$@" > vk.log 2>&1 || true
  same_tokens cpu.log vk.log "$tag"
  grep -q "rebuilding the state of [1-9]" vk.log || { cat vk.log; fail "$tag: no state was rebuilt"; }
  echo "OK $tag: tokens = CPU, $(grep -o 'rebuilding the state of [0-9]* positions' vk.log)"
}

family_qwen_chain() {
  make qwen36 qwen38 tests/test_vk_chain VK=1
  ./tests/test_vk_chain shaders/qmatmul.spv | tee vk_chain.log
  tail -1 vk_chain.log | grep -qx PASS || fail "the chain's ops"
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/make_qwen36_tiny.py --geometry qwen3-coder-30b --out qwen3_coder_tiny --ref-mode full --emit-ref qwen3_coder_tiny/ref_full.json
  $PY tools/make_qwen36_tiny.py --geometry qwen38-27b-dense --out qwen38_27b_tiny --ref-mode full --emit-ref qwen38_27b_tiny/ref_full.json
  $PY tools/make_qwen36_tiny.py --geometry qwen38-2p4t --seed 3 --out qwen38_2p4t_tiny --ref-mode full --emit-ref qwen38_2p4t_tiny/ref_full.json
  local fx cap caps
  # qwen36: the hybrid, Qwen3-Coder (attention only, no gate, no shared expert), the
  # 27B dense geometry (no routed experts: no host step at all) and the 2.4T one
  for fx in qwen36_tiny qwen3_coder_tiny qwen38_27b_tiny qwen38_2p4t_tiny; do
    $PY tools/convert_qwen36.py --model $fx --out ${fx}_c --ebits 8
    caps="1 8"; [ $fx = qwen38_2p4t_tiny ] && caps=8
    for cap in $caps; do
      chain_gate qwen36 "chain qwen36 $fx f32 cap=$cap" 1 COLI_DENSE_I8=0 SNAP=${fx}_c -- $cap 8 $fx/ref_full.json
    done
    # int8 dense rows (fmt 1): the CPU arm with f32 activations (COLI_DENSE_IDOT=0), as the device
    CPUENV=COLI_DENSE_IDOT=0 chain_gate qwen36 "chain qwen36 $fx int8" 1 SNAP=${fx}_c -- 8 8 $fx/ref_full.json
  done
  # the expert containers the tier reads, the tier off, prefill in chunks of 3 rows,
  # the tiled GEMM inside the chain from S = 2
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny64 --ref-mode full --inter 64 --emit-ref qwen36_tiny64/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_c --ebits 4 --gs 64
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_d8 --ebits 4 --gs 64 --down-bits 8
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_i4r --ebits 4
  chain_gate qwen36 "chain qwen36 int4-g64 planar" 1 QWEN_EXPERT_ACT=f32 COLI_DENSE_I8=0 SNAP=qwen36_tiny64_c -- 8 4 qwen36_tiny64/ref_full.json
  chain_gate qwen36 "chain qwen36 int4-g64 int8 activations" 0 COLI_DENSE_I8=0 SNAP=qwen36_tiny64_c -- 1 4 qwen36_tiny64/ref_full.json
  chain_gate qwen36 "chain qwen36 mixed int4/int8" 1 COLI_DENSE_I8=0 SNAP=qwen36_tiny64_d8 -- 8 4 qwen36_tiny64/ref_full.json
  chain_gate qwen36 "chain qwen36 int4 per row" 1 COLI_DENSE_I8=0 SNAP=qwen36_tiny_i4r -- 8 4 qwen36_tiny/ref_full.json
  chain_gate qwen36 "chain qwen36 tier off" 1 COLI_VK_TIER=0 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 qwen36_tiny/ref_full.json
  chain_gate qwen36 "chain qwen36 prefill in chunks of 3" 1 COLI_VK_CHAIN_ROWS=3 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 qwen36_tiny/ref_full.json
  chain_gate qwen36 "chain qwen36 tiled GEMM" 1 COLI_VK_GEMM_MIN_S=2 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 qwen36_tiny/ref_full.json
  chain_gate qwen36 "chain qwen36 the per-row GEMV" 1 COLI_VK_CHAIN_GEMV=0 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 qwen36_tiny/ref_full.json
  # an image: the M-RoPE positions reach the chain as the host's cos/sin table,
  # token-exact against transformers, on the CLI and through the IMAGE frame
  $PY tools/make_qwen36_vl_tiny.py --out qwen38_27b_vl_tiny
  $PY tools/convert_qwen36.py --model qwen38_27b_vl_tiny --out qwen38_27b_vl_tiny_c --ebits 8
  chain_gate qwen36 "chain qwen36 vision" 1 COLI_DENSE_I8=0 SNAP=qwen38_27b_vl_tiny_c -- 8 8 qwen38_27b_vl_tiny/ref.json
  QWEN36_VL_TINY=qwen38_27b_vl_tiny_c QWEN36_VL_REF=qwen38_27b_vl_tiny/ref.json COLI_VULKAN=1 COLI_VK_CHAIN=1 \
    COLI_USAGE=$PWD/chain.usage $PY -m unittest tests.test_qwen36_vision_serve
  # the device lost mid-decode: the state rebuilt on the CPU, the run finishes there
  lost_late qwen36 "chain qwen36 device lost" 20 rebuild COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 qwen36_tiny/ref_full.json
  # the prefix-reuse contract with the state on the device, and a serve session
  QWEN36_TINY=$PWD/qwen36_tiny_c COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_USAGE=$PWD/chain.usage $PY tests/test_qwen36_prefix_serve.py
  $PY tests/vulkan_chain_serve.py ./qwen36 qwen36_tiny_c COLI_DENSE_I8=0
  # COLI_VK_CHAIN=2, prompts on the device and decode on the CPU: the state crosses
  # between them at every turn's first decode step and next prompt
  CHAINMODE=2 chain_gate qwen36 "chain qwen36 prompts only" 1 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 qwen36_tiny/ref_full.json
  COLI_VK_CHAIN=2 $PY tests/vulkan_chain_serve.py ./qwen36 qwen36_tiny_c COLI_DENSE_I8=0

  # qwen38: every resident format, prefill batching, every expert form, MTP
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_fp8 --fp8-experts
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_int4 --fp8-experts --int4-experts --expert-gain 3
  local trunk batch bf16 fx ref f
  for batch in 0 1; do for bf16 in 0 1; do
    chain_gate qwen38 "chain qwen38 bf16=$bf16 batch=$batch" 1 OMP_NUM_THREADS=2 Q38_PREFILL_BATCH=$batch Q38_NATIVE_BF16=$bf16 SNAP=qwen38_tiny -- 4 8 qwen38_tiny/ref.json
  done; done
  # the int8 trunk (fmt 1): the CPU's integer kernel rounds activations, so tokens only
  chain_gate qwen38 "chain qwen38 int8 trunk" 0 OMP_NUM_THREADS=2 Q38_TRUNK_MIN_KB=0 SNAP=qwen38_tiny -- 4 8 qwen38_tiny/ref.json
  for fx in qwen38_tiny_fp8 qwen38_tiny_int4; do
    ref=$fx/ref.json; [ $fx = qwen38_tiny_int4 ] && ref=$fx/ref_int4.json
    for batch in 0 1; do chain_gate qwen38 "chain qwen38 $fx batch=$batch" 1 OMP_NUM_THREADS=2 Q38_PREFILL_BATCH=$batch SNAP=$fx -- 2 8 $ref; done
  done
  chain_gate qwen38 "chain qwen38 tier off" 1 OMP_NUM_THREADS=2 COLI_VK_TIER=0 SNAP=qwen38_tiny -- 4 8 qwen38_tiny/ref.json
  chain_gate qwen38 "chain qwen38 prefill in chunks of 3" 1 OMP_NUM_THREADS=2 COLI_VK_CHAIN_ROWS=3 SNAP=qwen38_tiny -- 4 8 qwen38_tiny/ref.json
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_mtp --mtp
  $PY tools/make_edge_tiny_tokenizer.py --vocab-size 64 ./qwen38_tiny_mtp   # serve speaks text
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_fp8_mtp --fp8-experts --mtp
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_int4_mtp --fp8-experts --int4-experts --expert-gain 3 --mtp
  # MTP verify (S = 2) with the device's snapshot and rollback: drafts as they come,
  # every draft rejected (a rollback per token), every one accepted, alternating
  for fx in qwen38_tiny_mtp qwen38_tiny_fp8_mtp qwen38_tiny_int4_mtp; do for f in "" reject accept mixed; do
    chain_gate qwen38 "chain qwen38 MTP $fx ${f:-drafting}" 1 OMP_NUM_THREADS=2 Q38_MTP=1 Q38_MTP_FORCE=$f SNAP=$fx -- 2 8 $fx/ref.json
  done; done
  # the device lost mid-decode, once between steps and once inside an MTP verify
  lost_late qwen38 "chain qwen38 device lost" 10 rebuild OMP_NUM_THREADS=2 SNAP=qwen38_tiny -- 4 8 qwen38_tiny/ref.json
  lost_late qwen38 "chain qwen38 device lost in a verify" 15 rebuild OMP_NUM_THREADS=2 Q38_MTP=1 Q38_MTP_FORCE=mixed SNAP=qwen38_tiny_mtp -- 2 8 qwen38_tiny_mtp/ref.json
  # the oracle targets with the chain on (COLI_VK_TIER_BALANCE=0: the int4 target wants
  # the same last logits from every expert path, and the balancer moves experts between
  # the device and the CPU by measured times); the MTP harness, whose prompt-cache and pin
  # checks compare a warm session's draft logits bit for bit with a fresh one's, with the
  # tier off: the tier's residency differs between two sessions, with or without the chain
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 \
    make qwen38-tiny-check qwen38-tiny-fp8-check qwen38-tiny-int4-check VK=1
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER=0 make qwen38-tiny-mtp-check VK=1
  $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp OMP_NUM_THREADS=2 Q38_MTP=1
  $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp OMP_NUM_THREADS=2
  CHAINMODE=2 chain_gate qwen38 "chain qwen38 prompts only, MTP" 1 OMP_NUM_THREADS=2 Q38_MTP=1 Q38_MTP_FORCE=mixed SNAP=qwen38_tiny_mtp -- 2 8 qwen38_tiny_mtp/ref.json
  COLI_VK_CHAIN=2 $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp OMP_NUM_THREADS=2 Q38_MTP=1
}

# The chain under ASan and UBSan: memory safety is the gate (a sanitized build
# vectorizes differently, so tokens are not compared); each run must have run the chain.
family_qwen_chain_sanitize() {
  make clean >/dev/null 2>&1 || true
  make qwen36 qwen38 tests/test_vk_chain VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  ./tests/test_vk_chain shaders/qmatmul.spv > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log || ! tail -1 san.log | grep -qx PASS; then cat san.log; fail "asan: the chain's ops"; fi
  echo "OK asan: the chain's ops"
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8
  $PY tools/make_qwen36_tiny.py --geometry qwen38-27b-dense --out qwen38_27b_tiny --ref-mode full --emit-ref qwen38_27b_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen38_27b_tiny --out qwen38_27b_tiny_c --ebits 8
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny64 --ref-mode full --inter 64 --emit-ref qwen36_tiny64/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_c --ebits 4 --gs 64
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_int4_mtp --fp8-experts --int4-experts --expert-gain 3 --mtp
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_mtp --mtp
  $PY tools/make_edge_tiny_tokenizer.py --vocab-size 64 ./qwen38_tiny_mtp   # serve speaks text
  csan() {  # <engine> <tag> <env and argv...>
    local eng=$1 tag=$2; shift 2
    rm -f chain.usage
    env OMP_NUM_THREADS=2 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 "$@" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    [ "$(chain_count "$eng" san.log)" -gt 0 ] || { cat san.log; fail "$tag: the chain never ran"; }
    echo "OK $tag: sanitizers clean, $(chain_count "$eng" san.log) chain forwards"
  }
  csan qwen36 "asan chain qwen36 int8 experts" COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 1 8 qwen36_tiny/ref_full.json
  csan qwen36 "asan chain qwen36 int8 dense, chunks of 3" COLI_VK_CHAIN_ROWS=3 SNAP=qwen36_tiny_c ./qwen36 8 8 qwen36_tiny/ref_full.json
  csan qwen36 "asan chain qwen36 int4-g64 planar" COLI_DENSE_I8=0 SNAP=qwen36_tiny64_c ./qwen36 8 4 qwen36_tiny64/ref_full.json
  csan qwen36 "asan chain qwen36 dense model" COLI_DENSE_I8=0 SNAP=qwen38_27b_tiny_c ./qwen36 8 8 qwen38_27b_tiny/ref_full.json
  csan qwen38 "asan chain qwen38 batch=0" Q38_PREFILL_BATCH=0 SNAP=qwen38_tiny ./qwen38 4 8 qwen38_tiny/ref.json
  csan qwen38 "asan chain qwen38 int8 trunk, chunks of 3" Q38_TRUNK_MIN_KB=0 COLI_VK_CHAIN_ROWS=3 SNAP=qwen38_tiny ./qwen38 4 8 qwen38_tiny/ref.json
  csan qwen38 "asan chain qwen38 MTP reject" Q38_MTP=1 Q38_MTP_FORCE=reject SNAP=qwen38_tiny_int4_mtp ./qwen38 2 8 qwen38_tiny_int4_mtp/ref.json
  csan qwen38 "asan chain qwen38 MTP mixed" Q38_MTP=1 Q38_MTP_FORCE=mixed SNAP=qwen38_tiny_int4_mtp ./qwen38 2 8 qwen38_tiny_int4_mtp/ref.json
  $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp Q38_MTP=1 > san.log 2>&1 || { cat san.log; fail "asan chain qwen38 serve"; }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan chain qwen38 serve: sanitizer diagnostic"; fi
  echo "OK asan chain qwen38 serve: $(tail -1 san.log)"
  make clean >/dev/null 2>&1 || true
}

# Speculative decoding with deeper verifies (docs/speculative.md): the MTP head's drafts
# up to 3 per verify (qwen38, Q38_MTP_DRAFTS) and prompt-lookup drafts up to 5 (qwen38
# and qwen36, COLI_LOOKUP=1), a verify of up to 6 rows with the recurrent state copied
# after each row; and colibri's n-gram drafts from the same lookup and gate. tests/spec_drafts_harness.py (make spec-drafts-check) compares every
# run with drafts against the same run without, tokens and last logits byte for byte:
#   - on the CPU (this VK=1 build with COLI_VULKAN unset);
#   - in the dense chain, the tier off (with it on, the tier's residency follows the
#     routed rows, the rejected drafts' included, and an expert on the device sums in
#     another order than on the CPU, so the last bits depend on the history); and the
#     chain in chunks of 3 rows (a verify of 6 rows split over two chunks), the
#     per-matrix path (a verify's rows one GEMV at a time) and prompts only
#     (COLI_VK_CHAIN=2: verifies on the per-matrix path);
#   - against the CPU with the tier on (chain_gate: the CPU's tokens, logits within
#     tolerance), and the device lost inside a deep verify (rebuilt on the CPU).
spec_fixtures() {
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_mtp --mtp
  $PY tools/make_edge_tiny_tokenizer.py --vocab-size 64 ./qwen38_tiny_mtp   # serve speaks text
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_int4_mtp --fp8-experts --int4-experts --expert-gain 3 --mtp
  make qwen36-tiny-v128-generate
}
family_qwen_spec() {
  make qwen36 qwen38 tests/test_spec_draft VK=1
  make spec-drafts-check VK=1
  local chain="--env COLI_VULKAN=1 --env COLI_VK_CHAIN=1 --env COLI_VK_TIER=0 --env COLI_VK_TIER_SYNC=1"
  make spec-drafts-run VK=1 SPEC_ENV="$chain"
  local h="$PY tests/spec_drafts_harness.py" e
  for e in "--env COLI_VULKAN=1 --env COLI_VK_CHAIN=1 --env COLI_VK_TIER=0 --env COLI_VK_CHAIN_ROWS=3" \
           "--env COLI_VULKAN=1 --env COLI_VK_CHAIN=0 --env COLI_VK_DENSE=1 --env COLI_VK_TIER=0" \
           "--env COLI_VULKAN=1 --env COLI_VK_CHAIN=2 --env COLI_VK_TIER=0"; do
    $h --engine ./qwen38 --kind qwen38 --fixture ./qwen38_tiny_mtp --mtp --quick $e
    $h --engine ./qwen36 --kind qwen36 --fixture ./qwen36_tiny_v128_c --ref ./qwen36_tiny_v128/ref_full.json \
      --cap 8 --n-new 64 --quick --env COLI_DENSE_I8=0 $e
  done
  # A six-row verify crosses several one-row KV chunks. Rollback must lower both
  # the mirror and split-cache validity, including rejected rows already uploaded.
  # Keep the same attention kernel in plain/speculative arms for byte comparisons.
  e="$chain --env COLI_VK_KV_DEVICE_ROWS=8 --env COLI_VK_KV_BLOCK=2 --env COLI_VK_CHAIN_ROWS=1 --env COLI_VK_ATTN_BLOCK=0"
  $h --engine ./qwen38 --kind qwen38 --fixture ./qwen38_tiny_mtp --mtp --serve --quick --require-kv-split $e
  $h --engine ./qwen36 --kind qwen36 --fixture ./qwen36_tiny_v128_c --ref ./qwen36_tiny_v128/ref_full.json \
    --cap 8 --n-new 64 --serve --quick --require-kv-split --env COLI_DENSE_I8=0 $e
  # A completed MTP chunk has already returned its hidden streams. A loss in
  # the next chunk must restart the CPU replay from the original embeddings.
  $h --engine ./qwen38 --kind qwen38 --fixture ./qwen38_tiny_mtp --mtp --n-new 32 \
    --device-loss --require-kv-split --env COLI_VK_DENSE_HOST=0 $e
  local f
  for f in cycle row2 row3; do
    chain_gate qwen38 "spec chain qwen38 MTP depth 3 $f, tier on" 1 OMP_NUM_THREADS=2 Q38_MTP=1 Q38_MTP_DRAFTS=3 \
      Q38_MTP_FORCE=$f SNAP=qwen38_tiny_mtp -- 2 8 qwen38_tiny_mtp/ref.json
    chain_gate qwen36 "spec chain qwen36 lookup $f, tier on" 1 COLI_DENSE_I8=0 COLI_LOOKUP=1 COLI_SPEC_GATE=0 \
      COLI_LOOKUP_FORCE=$f SNAP=qwen36_tiny_v128_c -- 8 8 qwen36_tiny_v128/ref_full.json
  done
  chain_gate qwen38 "spec chain qwen38 lookup row4, int4 experts, tier on" 1 OMP_NUM_THREADS=2 Q38_EXPERT_INT4=1 \
    COLI_VK_TIER_BALANCE=0 COLI_LOOKUP=1 COLI_SPEC_GATE=0 COLI_LOOKUP_FORCE=row4 SNAP=qwen38_tiny_int4_mtp -- 2 8 \
    qwen38_tiny_int4_mtp/ref_int4.json
  # colibri (GLM-5.2): its n-gram drafts from spec_draft.h under COLI_LOOKUP=1, on the
  # CPU and in the chain, against the oracle (every token) and against DRAFT=0
  make colibri VK=1
  $PY tools/make_glm_oracle.py > /dev/null
  local v base
  base=$(env OMP_NUM_THREADS=2 SNAP=glm_tiny REF=ref_glm.json DRAFT=0 ./colibri 64 16 16 2>/dev/null | grep -a '^GLM C engine')
  for v in "" "COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER=0" "COLI_VULKAN=1 COLI_VK_CHAIN=1"; do
    for e in "COLI_SPEC_GATE=0" "COLI_SPEC_GATE=0 DRAFT=2" ""; do
      # shellcheck disable=SC2086
      env OMP_NUM_THREADS=2 SNAP=glm_tiny REF=ref_glm.json COLI_USAGE=chain.usage COLI_LOOKUP=1 $e $v ./colibri 64 16 16 > vk.log 2>&1 \
        || { cat vk.log; fail "spec colibri lookup $e $v: misses the oracle"; }
      [ "$(grep -a '^GLM C engine' vk.log)" = "$base" ] || { cat vk.log; fail "spec colibri lookup $e $v: tokens differ from DRAFT=0"; }
      grep -q "colibri lookup\] acceptance" vk.log || { cat vk.log; fail "spec colibri lookup $e $v: no lookup line"; }
      echo "OK spec colibri lookup ${e:-gated} ${v:-CPU}: oracle and DRAFT=0's tokens, $(grep -o 'acceptance [^|]*' vk.log | head -1)"
    done
  done
  lost_late qwen38 "spec chain qwen38 device lost mid-decode, verifies of 4 rows" 15 rebuild OMP_NUM_THREADS=2 Q38_MTP=1 \
    Q38_MTP_DRAFTS=3 Q38_MTP_FORCE=cycle SNAP=qwen38_tiny_mtp -- 2 8 qwen38_tiny_mtp/ref.json
  lost_late qwen36 "spec chain qwen36 device lost mid-decode, lookup verifies" 20 rebuild COLI_DENSE_I8=0 COLI_LOOKUP=1 \
    COLI_SPEC_GATE=0 COLI_LOOKUP_FORCE=cycle SNAP=qwen36_tiny_v128_c -- 8 8 qwen36_tiny_v128/ref_full.json
}
# The same under ASan and UBSan: every run of the harness's quick subset (depth 3
# rejected at rows 1 and 2 and cycling, lookup cycling and at row 3, a serve session
# per source) on the CPU and in the chain, sanitizer-clean and still equal to its own
# plain decoding.
family_qwen_spec_sanitize() {
  make clean >/dev/null 2>&1 || true
  make qwen36 qwen38 tests/test_spec_draft tests/test_qwen38_spec_alloc VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  ./tests/test_spec_draft > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log || ! tail -1 san.log | grep -q "behave as"; then cat san.log; fail "asan: spec_draft.h"; fi
  echo "OK asan: spec_draft.h"
  ./tests/test_qwen38_spec_alloc > san.log 2>&1 || { cat san.log; fail "asan: speculative allocation"; }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan: speculative allocation"; fi
  echo "OK asan: speculative allocation"
  spec_fixtures
  local h="$PY tests/spec_drafts_harness.py --sanitize --quick" e
  for e in "" "--env COLI_VULKAN=1 --env COLI_VK_CHAIN=1 --env COLI_VK_TIER=0" \
           "--require-kv-split --env COLI_VULKAN=1 --env COLI_VK_CHAIN=1 --env COLI_VK_TIER=0 --env COLI_VK_KV_DEVICE_ROWS=8 --env COLI_VK_KV_BLOCK=2 --env COLI_VK_CHAIN_ROWS=1 --env COLI_VK_ATTN_BLOCK=0"; do
    $h --engine ./qwen38 --kind qwen38 --fixture ./qwen38_tiny_mtp --mtp --serve $e
    $h --engine ./qwen38 --kind qwen38 --fixture ./qwen38_tiny_int4_mtp --ref ./qwen38_tiny_int4_mtp/ref_int4.json \
      --mtp --env Q38_EXPERT_INT4=1 --env Q38_TRUNK_MIN_KB=0 $e
    $h --engine ./qwen36 --kind qwen36 --fixture ./qwen36_tiny_v128_c --ref ./qwen36_tiny_v128/ref_full.json \
      --cap 8 --n-new 64 --serve --env COLI_DENSE_I8=0 $e
  done
  $h --engine ./qwen38 --kind qwen38 --fixture ./qwen38_tiny_mtp --mtp --n-new 32 --device-loss \
    --require-kv-split --env COLI_VULKAN=1 --env COLI_VK_CHAIN=1 --env COLI_VK_TIER=0 --env COLI_VK_DENSE_HOST=0 \
    --env COLI_VK_KV_DEVICE_ROWS=8 --env COLI_VK_KV_BLOCK=2 --env COLI_VK_CHAIN_ROWS=1 --env COLI_VK_ATTN_BLOCK=0
  make clean >/dev/null 2>&1 || true
}

# inkling's and olmoe's fixtures for their dense chain: the family inkling-olmoe's (the
# bf16 copy of the f32 snapshot, the dense-int4g64 container, the expert containers and
# the tokenizers), and Inkling at its real width (--wide: D = 6144, head dim 128).
inkling_olmoe_chain_fixtures() {
  $PY tools/make_tiny_inkling.py tiny_inkling
  $PY tools/make_tiny_inkling.py tiny_inkling_wide --wide
  mkdir -p tiny_inkling_bf16
  cp tiny_inkling/config.json tiny_inkling/generation_config.json tiny_inkling_bf16/
  $PY - <<'EOF'
import json, struct, numpy as np
src, dst = "tiny_inkling/model.safetensors", "tiny_inkling_bf16/model.safetensors"
raw = open(src, "rb").read(); n = struct.unpack("<Q", raw[:8])[0]; hdr = json.loads(raw[8:8 + n])
out, blobs, off = {}, [], 0
for k, v in hdr.items():
    if k == "__metadata__": out[k] = v; continue
    a = np.frombuffer(raw[8 + n + v["data_offsets"][0]: 8 + n + v["data_offsets"][1]], np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    b = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()
    out[k] = {"dtype": "BF16", "shape": v["shape"], "data_offsets": [off, off + len(b)]}
    blobs.append(b); off += len(b)
h = json.dumps(out).encode(); h += b" " * (-len(h) % 8)
with open(dst, "wb") as f:
    f.write(struct.pack("<Q", len(h))); f.write(h); [f.write(b) for b in blobs]
EOF
  rm -rf tiny_inkling_q && cp -r tiny_inkling tiny_inkling_q
  $PY - tools/convert_inkling_dense_int4.py tiny_inkling_q <<'EOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("conv", sys.argv[1])
conv = importlib.util.module_from_spec(spec); spec.loader.exec_module(conv)
conv.MIN_ELEMS = 0
conv.ATTN_BITS = 4
base = conv.classify
conv.classify = lambda n, s, d: "int8" if n.endswith(".mlp.down_proj.weight") else base(n, s, d)
sys.argv = ["convert_inkling_dense_int4.py", "--dir", sys.argv[2]]
conv.main()
EOF
  mkdir -p tiny_inkling_q/dense-int4g64
  mv tiny_inkling_q/dense-int4g64.safetensors tiny_inkling_q/dense-int4g64/dense.safetensors
  $PY tools/make_olmoe_tiny.py --output olmoe_tiny --force
  $PY tools/convert_olmoe_merged.py --model olmoe_tiny --out olmoe_tiny_c
  inkling_olmoe_tier_fixtures
}

# chain_gate for a bf16 snapshot: where this CPU's bf16 dot rounds the activations
# (AVX512-BF16), the per-matrix path keeps bf16 on the CPU and so does the chain, which
# must then decline cleanly ("bf16 stays on the CPU") with the CPU's tokens.
chain_gate_bf16() {  # <engine> <tag> <env...> -- <argv...>
  local eng=$1 tag=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  env "${envs[@]}" COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 ./"$eng" "$@" > vk.log 2>&1 || true
  if grep -qa 'bf16 stays on the CPU' vk.log; then
    env "${envs[@]}" ./"$eng" "$@" > cpu.log 2>&1 || true
    same_tokens cpu.log vk.log "$tag"
    echo "OK $tag: tokens = CPU, the chain declined (this CPU's bf16 dot keeps bf16 on the CPU)"
  else
    chain_gate "$eng" "$tag" 1 "${envs[@]}" -- "$@"
  fi
}

# lost_gate for olmoe: attention only, so nothing to rebuild; the CPU redoes the step
# the device was lost in from its first position and runs from there.
lost_gate_redo() {  # <engine> <tag> <frame> <env...> -- <argv...>
  local eng=$1 tag=$2 k=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage
  env "${envs[@]}" ./"$eng" "$@" > cpu.log 2>&1 || true
  env "${envs[@]}" COLI_VK_CHAIN_FAULT=$k COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 \
    ./"$eng" "$@" > vk.log 2>&1 || true
  same_tokens cpu.log vk.log "$tag"
  grep -q "the CPU redoes the step from position [1-9]" vk.log || { cat vk.log; fail "$tag: no step was redone"; }
  echo "OK $tag: tokens = CPU, $(grep -o 'redoes the step from position [0-9]*' vk.log)"
}

# The device lost `back` frames before the end of a run (a probe run counts the frames,
# so the fault lands mid-decode on any device, whatever its setup submits): lost_gate
# (the state rebuilt on the CPU) or lost_gate_redo (the step redone there).
lost_late() {  # <engine> <tag> <frames back> rebuild|redo <env...> -- <argv...>
  local eng=$1 tag=$2 back=$3 kind=$4; shift 4
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  env "${envs[@]}" COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 ./"$eng" "$@" > vk.log 2>&1 || true
  local f; f=$(sed -n "s/^\[VK\] $eng chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p" vk.log | tail -1)
  { [ -n "$f" ] && [ "$f" -gt "$back" ]; } || { cat vk.log; fail "$tag: the probe run reports no frames"; }
  if [ "$kind" = rebuild ]; then lost_gate "$eng" "$tag" $((f - back)) "${envs[@]}" -- "$@"
  else lost_gate_redo "$eng" "$tag" $((f - back)) "${envs[@]}" -- "$@"; fi
}

# The dense chain (COLI_VK_CHAIN=1) of inkling and olmoe on Lavapipe, each configuration
# against the CPU run's tokens and logits (chain_gate: every forward's logits within 1e-4
# of the largest): inkling's resident forms (f32, bf16, the dense-int4g64 container's
# int8 and int4-g64), its expert containers and runtime quantizations, TOPP, the tier
# off, the trunk's device copies shared with the per-matrix path (COLI_VK_DENSE=1),
# prefill in chunks of 3, the tiled GEMM and the per-row GEMV inside the chain, its real
# width (D = 6144, beside the tier's expert batch at that width); olmoe's caps, PILOT's
# prefetch from the chain's residual rows (cap 1: the slot the forward pass reads is
# never the prefetcher's victim), an eviction budget and the same chain settings; both
# with prompts only (COLI_VK_CHAIN=2), the device lost mid-run, and the serve, prefix
# and dashboard tests with the chain on.
family_inkling_olmoe_chain() {
  export OMP_NUM_THREADS=2   # tiny models: a wide team only waits for itself (and for Lavapipe's)
  make inkling olmoe tests/test_vk_chain VK=1
  ./tests/test_vk_chain shaders/qmatmul.spv | tee vk_chain.log
  tail -1 vk_chain.log | grep -qx PASS || fail "the chain's ops"
  inkling_olmoe_chain_fixtures
  local fx cap bits R=tiny_inkling/ref_inkling.json OR=olmoe_tiny/ref_olmoe.json
  for fx in tiny_inkling tiny_inkling_q tiny_inkling_x-i4 tiny_inkling_x-i8; do
    for cap in 1 8; do chain_gate inkling "chain inkling $fx cap=$cap" 1 SNAP=$fx -- $cap 0 $R; done
  done
  chain_gate_bf16 inkling "chain inkling bf16" SNAP=tiny_inkling_bf16 -- 8 0 $R
  for bits in 4 8; do chain_gate inkling "chain inkling runtime int$bits" 1 SNAP=tiny_inkling -- 2 $bits $R; done
  chain_gate inkling "chain inkling TOPP" 1 TOPP=0.3 SNAP=tiny_inkling_x-i4 -- 2 0 $R
  chain_gate inkling "chain inkling tier off" 1 COLI_VK_TIER=0 SNAP=tiny_inkling -- 8 0 $R
  chain_gate inkling "chain inkling trunk shared with the per-matrix path" 1 COLI_VK_DENSE=1 SNAP=tiny_inkling_q -- 8 0 $R
  chain_gate inkling "chain inkling prefill in chunks of 3" 1 COLI_VK_CHAIN_ROWS=3 SNAP=tiny_inkling -- 8 0 $R
  chain_gate inkling "chain inkling tiled GEMM" 1 COLI_VK_GEMM_MIN_S=2 SNAP=tiny_inkling -- 8 0 $R
  chain_gate inkling "chain inkling the per-row GEMV" 1 COLI_VK_CHAIN_GEMV=0 SNAP=tiny_inkling -- 8 0 $R
  CHAINMODE=2 chain_gate inkling "chain inkling prompts only" 1 SNAP=tiny_inkling -- 8 0 $R
  # D = 6144: the CPU's tokens through the chain, and through the tier's expert batch
  # (its gate/up shader stages x up to 6144 floats) with the chain off
  chain_gate inkling "chain inkling D=6144" 1 SNAP=tiny_inkling_wide -- 8 0 tiny_inkling_wide/ref_inkling.json
  tier_gate inkling "inkling tier D=6144" COLI_VK_CHAIN=0 SNAP=tiny_inkling_wide -- 8 0 tiny_inkling_wide/ref_inkling.json
  # the device lost mid-decode, in the last step but one: in a shared-expert frame
  # nobody waits for (20 frames before the end) and at a layer's router (19): the
  # state rebuilt on the CPU from the prefix record, the run finished there
  lost_late inkling "chain inkling device lost in a shared-expert frame" 20 rebuild SNAP=tiny_inkling -- 8 0 $R
  lost_late inkling "chain inkling device lost at a router" 19 rebuild SNAP=tiny_inkling -- 8 0 $R

  for cap in 1 8; do chain_gate olmoe "chain olmoe cap=$cap" 1 SNAP=olmoe_tiny_c -- $cap 8 $OR; done
  chain_gate olmoe "chain olmoe 4-bit experts" 1 SNAP=olmoe_tiny_c -- 8 4 $OR
  for cap in 1 2; do chain_gate olmoe "chain olmoe PILOT cap=$cap" 1 PILOT=1 WIDE=2 SNAP=olmoe_tiny_c -- $cap 8 $OR; done
  chain_gate olmoe "chain olmoe PILOT=3" 1 PILOT=3 SNAP=olmoe_tiny_c -- 8 8 $OR
  chain_gate olmoe "chain olmoe tier off" 1 COLI_VK_TIER=0 SNAP=olmoe_tiny_c -- 8 8 $OR
  chain_gate olmoe "chain olmoe trunk shared with the per-matrix path" 1 COLI_VK_DENSE=1 SNAP=olmoe_tiny_c -- 8 8 $OR
  chain_gate olmoe "chain olmoe a budget of three experts" 1 COLI_VK_TIER_GB=0.000025 SNAP=olmoe_tiny_c -- 8 8 $OR
  chain_gate olmoe "chain olmoe prefill in chunks of 3" 1 COLI_VK_CHAIN_ROWS=3 SNAP=olmoe_tiny_c -- 8 8 $OR
  chain_gate olmoe "chain olmoe tiled GEMM" 1 COLI_VK_GEMM_MIN_S=2 SNAP=olmoe_tiny_c -- 8 8 $OR
  chain_gate olmoe "chain olmoe the per-row GEMV" 1 COLI_VK_CHAIN_GEMV=0 SNAP=olmoe_tiny_c -- 8 8 $OR
  CHAINMODE=2 chain_gate olmoe "chain olmoe prompts only" 1 SNAP=olmoe_tiny_c -- 8 8 $OR
  lost_late olmoe "chain olmoe device lost mid-decode" 7 redo SNAP=olmoe_tiny_c -- 8 8 $OR

  # serve: a session of pins, prompt-cache extensions, a divergent prompt and logprobs
  # against the CPU's, frame by frame (and with prompts only); the prefix-reuse and
  # dashboard contracts with the chain on
  $PY tests/vulkan_chain_serve.py ./inkling tiny_inkling INK_PREFIX_LOG=1
  COLI_VK_CHAIN=2 $PY tests/vulkan_chain_serve.py ./inkling tiny_inkling INK_PREFIX_LOG=1
  $PY tests/vulkan_chain_serve.py ./olmoe olmoe_tiny_c
  $PY tests/vulkan_chain_serve.py ./olmoe olmoe_tiny_c PILOT=1 WIDE=2
  COLI_VK_CHAIN=2 $PY tests/vulkan_chain_serve.py ./olmoe olmoe_tiny_c
  # (these compare a warm session's logprobs with a cold one's to the last printed
  # digit: no history file and COLI_VK_TIER_BALANCE=0, so both sessions split their
  # experts between the device and the CPU alike. Use per-row attention on both:
  # a cold prefill and a short suffix must share a reduction order for exact bits.)
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VK_ATTN_BLOCK=0 INKLING_TINY=tiny_inkling \
    $PY -m unittest tests.test_inkling_prefix_serve tests.test_inkling_dashboard_hits
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VK_ATTN_BLOCK=0 OLMOE_TINY=olmoe_tiny_c \
    $PY -m unittest tests.test_olmoe_prefix_serve tests.test_olmoe_dashboard_hits tests.test_brio_serve
}

# inkling's and olmoe's chain under ASan and UBSan: memory safety is the gate (a
# sanitized build vectorizes differently, so tokens are not compared); each run must
# have run the chain. detect_stack_use_after_return=0: ASan's fake stack does not keep
# the 64-byte alignment an AVX-512 build gives its locals.
family_inkling_olmoe_chain_sanitize() {
  make clean >/dev/null 2>&1 || true
  make inkling olmoe tests/test_vk_chain VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  ./tests/test_vk_chain shaders/qmatmul.spv > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log || ! tail -1 san.log | grep -qx PASS; then cat san.log; fail "asan: the chain's ops"; fi
  echo "OK asan: the chain's ops"
  inkling_olmoe_chain_fixtures
  csan() {  # <engine> <tag> <env and argv...>
    local eng=$1 tag=$2; shift 2
    rm -f chain.usage
    env OMP_NUM_THREADS=2 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} "$@" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    [ "$(chain_count "$eng" san.log)" -gt 0 ] || { cat san.log; fail "$tag: the chain never ran"; }
    echo "OK $tag: sanitizers clean, $(chain_count "$eng" san.log) chain forwards"
  }
  local R=tiny_inkling/ref_inkling.json OR=olmoe_tiny/ref_olmoe.json
  csan inkling "asan chain inkling f32 cap=1" SNAP=tiny_inkling ./inkling 1 0 $R
  csan inkling "asan chain inkling dense-int4g64, the trunk shared" COLI_VK_DENSE=1 SNAP=tiny_inkling_q ./inkling 8 0 $R
  csan inkling "asan chain inkling TOPP, chunks of 3" TOPP=0.3 COLI_VK_CHAIN_ROWS=3 SNAP=tiny_inkling_x-i4 ./inkling 2 0 $R
  csan inkling "asan chain inkling D=6144" SNAP=tiny_inkling_wide ./inkling 8 0 tiny_inkling_wide/ref_inkling.json
  CHAINMODE=2 csan inkling "asan chain inkling prompts only" SNAP=tiny_inkling ./inkling 8 0 $R
  csan inkling "asan chain inkling device lost" COLI_VK_CHAIN_FAULT=60 SNAP=tiny_inkling ./inkling 8 0 $R
  csan olmoe "asan chain olmoe PILOT cap=1" PILOT=1 WIDE=2 SNAP=olmoe_tiny_c ./olmoe 1 8 $OR
  csan olmoe "asan chain olmoe eviction, chunks of 3" COLI_VK_TIER_GB=0.000025 COLI_VK_CHAIN_ROWS=3 SNAP=olmoe_tiny_c ./olmoe 8 8 $OR
  csan olmoe "asan chain olmoe device lost" COLI_VK_CHAIN_FAULT=20 SNAP=olmoe_tiny_c ./olmoe 8 8 $OR
  local eng
  for eng in inkling olmoe; do
    local snap=tiny_inkling; [ $eng = olmoe ] && snap=olmoe_tiny_c
    $PY tests/vulkan_chain_serve.py ./$eng $snap OMP_NUM_THREADS=2 INK_PREFIX_LOG=1 > san.log 2>&1 || { cat san.log; fail "asan chain $eng serve"; }
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan chain $eng serve: sanitizer diagnostic"; fi
    echo "OK asan chain $eng serve: $(tail -1 san.log)"
  done
  make clean >/dev/null 2>&1 || true
}

# The dense chain of the MLA engines: GLM-5.2 (colibri) and GLM-5.3 Flash (glm53).
# mla_gate <engine> <tag> <tol 0|1> <env...> -- <argv...>: the CPU run's token lines
# (colibri: the generated tokens, the teacher-forced predictions and the oracle's
# mismatches; glm53: teacher_forcing and greedy), every logits row within 1e-4 of the
# largest |logit| (tol 1; DUMP= writes them), and a "[VK] <engine> chain: N forwards"
# line with N > 0. FAULT_BACK=k: the device is lost k frames before the end of the same
# run without a fault (COLI_VK_CHAIN_FAULT counted from that run's frames, so the loss
# lands in the same forward on every device, whatever frames its setup took), and the
# run must say so (and, for glm53 with REBUILD=1, rebuild the KDA state of some positions).
mla_toks() {  # <engine> <log>
  if [ "$1" = colibri ]; then grep -aE '^GLM C engine|^PREFILL|^\[ORACLE\] mismatch' "$2" | sed 's/ | [0-9.]* pos\/s//'
  else grep -aE '^teacher_forcing|^greedy' "$2"; fi
}
mla_gate() {
  local eng=$1 tag=$2 tol=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage cpu.f32 vk.f32
  env "${envs[@]}" COLI_USAGE=chain.usage USAGE_SAVE=0 DUMP=cpu.f32 ./"$eng" "$@" > cpu.log 2>&1 || true
  if [ -n "${FAULT_BACK:-}" ]; then
    rm -f chain.usage
    env "${envs[@]}" COLI_USAGE=chain.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 \
      ./"$eng" "$@" > vk.log 2>&1 || true
    local frames; frames=$(sed -n "s/^\[VK\] $eng chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p" vk.log | tail -1)
    [ -n "$frames" ] && [ "$frames" -gt "$FAULT_BACK" ] || { cat vk.log; fail "$tag: no fault-free run to count frames from"; }
    envs+=("COLI_VK_CHAIN_FAULT=$((frames - FAULT_BACK + 1))")
  fi
  rm -f chain.usage
  env "${envs[@]}" COLI_USAGE=chain.usage USAGE_SAVE=0 DUMP=vk.f32 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 \
    COLI_VK_CHAIN=${CHAINMODE:-1} ./"$eng" "$@" > vk.log 2>&1 || true
  mla_toks "$eng" cpu.log > cpu.tok; mla_toks "$eng" vk.log > vk.tok
  { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; tail -20 vk.log; fail "$tag: the chain's tokens differ from the CPU"; }
  if [ -n "${FAULT_BACK:-}" ]; then
    grep -q "$eng chain: the device was lost" vk.log || { cat vk.log; fail "$tag: no loss was handled"; }
    if [ "${REBUILD:-0}" = 1 ]; then grep -q "rebuilding the state of [1-9]" vk.log || { cat vk.log; fail "$tag: no state was rebuilt"; }; fi
  else
    [ "$(chain_count "$eng" vk.log)" -gt 0 ] || { cat vk.log; fail "$tag: the chain never ran"; }
  fi
  local lg=""
  if [ "$tol" = 1 ]; then lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag: logits"; }; lg=", $lg"; fi
  echo "OK $tag: tokens = CPU$lg, $(chain_count "$eng" vk.log) chain forwards$(grep -o 'rebuilding the state of [0-9]* positions' vk.log | sed 's/^/, /')"
}
glm_chain_fixtures() {
  glm_fixtures
  $PY tools/make_glm_mtp_tiny.py --src glm_tiny --out glm_tiny_mtp > /dev/null && cp ref_glm.json glm_tiny_mtp/
  rm -rf glm_tiny_serve && cp -r glm_tiny glm_tiny_serve
  $PY tools/make_edge_tiny_tokenizer.py --vocab-size 256 ./glm_tiny_serve > /dev/null   # serve speaks text
  rm -rf glm53_serve && cp -r glm53_stream-i4 glm53_serve
  $PY tools/make_edge_tiny_tokenizer.py --vocab-size 128 ./glm53_serve > /dev/null
  $PY tools/make_glm53_multimodal_tiny.py --output glm53_mm_tiny > /dev/null
  rm -rf glm53_lim0 && cp -r glm53_stream-i4 glm53_lim0   # swiglu_limit 0: the tier declines, the chain does not
  $PY - <<'PY'
import json
p = "glm53_lim0/config.json"; c = json.load(open(p))
(c["text_config"] if "text_config" in c else c)["swiglu_limit"] = 0.0
json.dump(c, open(p, "w"))
PY
}

# Every configuration the chain takes: colibri in every expert format (IDOT=0 where the
# CPU would round activations to int8: the device matches IDOT=0, as for the tier),
# decode and teacher-forced prefill in chunks, the DSA indexer's selection active
# (DSA_TOPK=4 on the 12-token prompt, and DSA_FORCE), n-gram and MTP drafts accepted and
# rejected (the MTP fixture's head is the last layer's copy), the tier off, the per-matrix
# trunk and attention core beside it, prompts only, a device lost mid-decode and in a
# prompt, serve sessions (pins, the prompt cache, two KV slots, the prefill read-out);
# glm53 with f32, int8 and int4 trunks, streamed and resident experts, an image, prefill
# chunks and chain chunks, swiglu_limit 0, a lost device whose KDA state is rebuilt on
# the CPU, serve sessions, and its pin-branch harness with the chain on.
family_glm_chain() {
  make colibri glm53 tests/test_vk_chain VK=1
  ./tests/test_vk_chain shaders/qmatmul.spv | tee vk_chain.log
  tail -1 vk_chain.log | grep -qx PASS || fail "the chain's ops"
  glm_chain_fixtures
  export OMP_NUM_THREADS=2 CAP_RAISE=0
  local cap b fx d
  for cap in 1 64; do mla_gate colibri "chain colibri f32 cap=$cap" 1 SNAP=glm_tiny REF=ref_glm.json -- $cap 16 16; done
  mla_gate colibri "chain colibri f32 prefill" 1 SNAP=glm_tiny REF=ref_glm.json TF=1 -- 64 16 16
  mla_gate colibri "chain colibri prefill in chunks of 3" 1 SNAP=glm_tiny REF=ref_glm.json TF=1 COLI_VK_CHAIN_ROWS=3 -- 64 16 16
  for b in 8 4 3; do
    mla_gate colibri "chain colibri ${b}-bit trunk and experts" 1 SNAP=glm_tiny REF=ref_glm.json IDOT=0 -- 2 $b $b
  done
  mla_gate colibri "chain colibri int4-g64 experts" 1 SNAP=glm_tiny_fmt4 REF=glm_tiny_fmt4/ref_glm.json -- 2 16 16
  mla_gate colibri "chain colibri E8/IQ3 experts on the CPU" 1 SNAP=glm_tiny_fmt6 REF=glm_tiny_fmt6/ref_glm.json -- 2 16 16
  for fx in i4 i4r i3 d3; do
    mla_gate colibri "chain colibri $fx container" 1 SNAP=glm_tiny_$fx REF=glm_tiny_$fx/ref_glm.json IDOT=0 -- 1 4 4
  done
  mla_gate colibri "chain colibri i4 container, prefill" 1 SNAP=glm_tiny_i4 REF=glm_tiny_i4/ref_glm.json IDOT=0 TF=1 -- 2 4 4
  # the DSA indexer's selection on the device: index_topk 4 against a 32-token context
  mla_gate colibri "chain colibri DSA top-4" 1 SNAP=glm_tiny REF=ref_glm.json DSA_TOPK=4 -- 64 16 16
  mla_gate colibri "chain colibri DSA top-4, prefill in chunks of 5" 1 SNAP=glm_tiny REF=ref_glm.json DSA_TOPK=4 TF=1 COLI_VK_CHAIN_ROWS=5 -- 64 16 16
  mla_gate colibri "chain colibri DSA_FORCE" 1 SNAP=glm_tiny REF=ref_glm.json DSA_FORCE=1 -- 64 16 16
  # drafts: n-gram, and the MTP head at depths 1 to 3 (accepted and rejected)
  mla_gate colibri "chain colibri n-gram drafts" 1 SNAP=glm_tiny REF=ref_glm.json DRAFT=3 -- 64 16 16
  for d in 1 2 3; do mla_gate colibri "chain colibri MTP depth $d" 1 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json DRAFT=$d -- 64 16 16; done
  mla_gate colibri "chain colibri MTP with DSA top-4" 1 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json DSA_TOPK=4 -- 64 16 16
  mla_gate colibri "chain colibri MTP, 4-bit" 1 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json IDOT=0 DRAFT=2 -- 2 4 4
  # around the chain: the tier off, the per-matrix trunk and core, the GEMM and GEMV choices
  mla_gate colibri "chain colibri tier off" 1 SNAP=glm_tiny REF=ref_glm.json COLI_VK_TIER=0 -- 64 16 16
  mla_gate colibri "chain colibri beside COLI_VK_DENSE=1 COLI_VK_ATTN=1" 1 SNAP=glm_tiny_i4 REF=glm_tiny_i4/ref_glm.json IDOT=0 COLI_VK_DENSE=1 COLI_VK_ATTN=1 -- 2 4 4
  # COLI_VK_DEV2 beside the chain: a second logical device holds what the capped tier does not
  SNAP=glm_tiny_i4r REF=glm_tiny_i4r/ref_glm.json IDOT=0 STATS=glm_tiny_i4r/.coli_usage ./colibri 64 4 4 > /dev/null 2>&1 || true
  mla_gate colibri "chain colibri beside COLI_VK_DEV2" 1 SNAP=glm_tiny_i4r REF=glm_tiny_i4r/ref_glm.json IDOT=0 COLI_VK_EXPERTS=4 COLI_VK_DEV2=0 -- 64 4 4
  grep -qaE 'lru \+ [1-9][0-9]* vk /' vk.log || { cat vk.log; fail "chain colibri beside COLI_VK_DEV2: no expert on the devices"; }
  rm -f glm_tiny_i4r/.coli_usage
  mla_gate colibri "chain colibri tiled GEMM from 2 rows" 1 SNAP=glm_tiny REF=ref_glm.json TF=1 COLI_VK_GEMM_MIN_S=2 -- 64 16 16
  mla_gate colibri "chain colibri the per-row GEMV" 1 SNAP=glm_tiny REF=ref_glm.json COLI_VK_CHAIN_GEMV=0 -- 64 16 16
  CHAINMODE=2 mla_gate colibri "chain colibri prompts only, drafts" 1 SNAP=glm_tiny REF=ref_glm.json DRAFT=3 -- 64 16 16
  # the device lost: mid-decode, in the prompt (the teacher-forced pass is one forward),
  # between MTP drafts
  FAULT_BACK=12 mla_gate colibri "chain colibri device lost mid-decode" 1 SNAP=glm_tiny REF=ref_glm.json -- 64 16 16
  FAULT_BACK=3 mla_gate colibri "chain colibri device lost in the prompt" 1 SNAP=glm_tiny REF=ref_glm.json TF=1 -- 64 16 16
  FAULT_BACK=12 mla_gate colibri "chain colibri device lost with MTP" 1 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json -- 64 16 16
  # serve sessions frame for frame: pins, the prompt cache, the prefill read-out, two KV slots
  CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0
  CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 DSA_TOPK=4 DRAFT=3
  CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 KV_SLOTS=2
  COLI_VK_CHAIN=2 CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 DSA_TOPK=4

  # glm53
  local ids bits
  ids=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
  mla_gate glm53 "chain glm53 decode" 1 GLM53_BITS=32 -- --model glm53_stream-i4 --ids 5,7,9,11,13,17,19,23 --greedy 8
  for bits in 32 8 4; do
    mla_gate glm53 "chain glm53 100-token prompt, ${bits}-bit trunk" 1 GLM53_BITS=$bits -- --model glm53_stream-i4 --ids $ids --greedy 4
  done
  mla_gate glm53 "chain glm53 prefill chunks of 7, one cache slot" 1 GLM53_BITS=4 GLM53_PREFILL_CHUNK=7 GLM53_EXPERT_GB=0.000001 -- --model glm53_stream-i4 --ids $ids --greedy 4
  mla_gate glm53 "chain glm53 chain chunks of 3" 1 GLM53_BITS=32 COLI_VK_CHAIN_ROWS=3 -- --model glm53_stream-i4 --ids $ids --greedy 4
  mla_gate glm53 "chain glm53 resident experts" 1 GLM53_BITS=32 -- --model glm53_tiny --ids $ids --greedy 6
  mla_gate glm53 "chain glm53 tier off" 1 GLM53_BITS=32 COLI_VK_TIER=0 -- --model glm53_stream-i4 --ids $ids --greedy 6
  mla_gate glm53 "chain glm53 an image" 1 GLM53_BITS=32 -- --model glm53_mm_tiny --ids 103,117,268,268,268,268,120,121 --patches glm53_mm_tiny/patches.f32 --grid 4x4 --greedy 4
  mla_gate glm53 "chain glm53 swiglu_limit 0" 1 GLM53_BITS=32 -- --model glm53_lim0 --ids $ids --greedy 4
  grep -qa 'tier glm53: swiglu_limit is 0' vk.log || { cat vk.log; fail "chain glm53 swiglu_limit 0: the tier did not decline"; }
  CHAINMODE=2 mla_gate glm53 "chain glm53 prompts only" 1 GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 -- --model glm53_stream-i4 --ids $ids --greedy 6
  FAULT_BACK=7 REBUILD=1 mla_gate glm53 "chain glm53 device lost, the KDA state rebuilt" 1 GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 -- --model glm53_stream-i4 --ids $ids --greedy 6
  FAULT_BACK=5 REBUILD=1 mla_gate glm53 "chain glm53 device lost after an image" 1 GLM53_BITS=32 -- --model glm53_mm_tiny --ids 103,117,268,268,268,268,120,121 --patches glm53_mm_tiny/patches.f32 --grid 4x4 --greedy 4
  CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_serve GLM53_BITS=32
  CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_serve GLM53_BITS=4 COLI_VK_CHAIN_ROWS=3
  CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_serve GLM53_BITS=32 KV_SLOTS=2
  COLI_VK_CHAIN=2 CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_serve GLM53_BITS=32
  # a pin restored over rows another branch rewrote: the KDA state goes up from the pin,
  # the MLA rows' watermark comes down
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_USAGE=$PWD/chain.usage COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 $PY tests/glm53_pin_branch_harness.py --binary ./glm53 --fixture glm53_mm_tiny
  unset OMP_NUM_THREADS CAP_RAISE
}

# The same chains under ASan and UBSan: memory safety is the gate; each run must have run
# the chain (or handled the loss). detect_stack_use_after_return=0 as in every sanitized
# family (ASan's fake stack breaks 64-byte aligned AVX-512 locals).
family_glm_chain_sanitize() {
  make clean >/dev/null 2>&1 || true
  make colibri glm53 tests/test_vk_chain VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  ./tests/test_vk_chain shaders/qmatmul.spv > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log || ! tail -1 san.log | grep -qx PASS; then cat san.log; fail "asan: the chain's ops"; fi
  echo "OK asan: the chain's ops"
  glm_chain_fixtures
  export OMP_NUM_THREADS=2 CAP_RAISE=0
  msan() {  # <engine> <tag> <env and argv...>
    local eng=$1 tag=$2; shift 2
    rm -f chain.usage
    env COLI_USAGE=chain.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 "$@" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    grep -q "$eng chain: \([1-9][0-9]* forwards\|the device was lost\)" san.log || { cat san.log; fail "$tag: the chain never ran"; }
    echo "OK $tag: sanitizers clean, $(chain_count "$eng" san.log) chain forwards"
  }
  local ids
  ids=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
  msan colibri "asan chain colibri f32 decode" SNAP=glm_tiny REF=ref_glm.json ./colibri 1 16 16
  msan colibri "asan chain colibri DSA top-4 prefill, chunks of 3" SNAP=glm_tiny REF=ref_glm.json TF=1 DSA_TOPK=4 COLI_VK_CHAIN_ROWS=3 ./colibri 64 16 16
  msan colibri "asan chain colibri i4 container" SNAP=glm_tiny_i4 REF=glm_tiny_i4/ref_glm.json IDOT=0 ./colibri 2 4 4
  msan colibri "asan chain colibri MTP depth 2" SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json DRAFT=2 ./colibri 64 16 16
  msan colibri "asan chain colibri device lost" SNAP=glm_tiny REF=ref_glm.json COLI_VK_CHAIN_FAULT=30 ./colibri 64 16 16
  msan glm53 "asan chain glm53 decode" GLM53_BITS=32 ./glm53 --model glm53_stream-i4 --ids 5,7,9,11,13,17,19,23 --greedy 8
  msan glm53 "asan chain glm53 int4 trunk, chain chunks of 3" GLM53_BITS=4 COLI_VK_CHAIN_ROWS=3 ./glm53 --model glm53_stream-i4 --ids $ids --greedy 4
  msan glm53 "asan chain glm53 an image" GLM53_BITS=32 ./glm53 --model glm53_mm_tiny --ids 103,117,268,268,268,268,120,121 --patches glm53_mm_tiny/patches.f32 --grid 4x4 --greedy 4
  msan glm53 "asan chain glm53 device lost, rebuilt" GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 COLI_VK_CHAIN_FAULT=20 ./glm53 --model glm53_stream-i4 --ids $ids --greedy 6
  local args
  for args in "./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 KV_SLOTS=2" "./glm53 glm53_serve GLM53_BITS=32 KV_SLOTS=2"; do
    # shellcheck disable=SC2086
    CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=$([ "${args%% *}" = ./colibri ] && echo colibri || echo numeric) \
      $PY tests/vulkan_chain_serve.py $args > san.log 2>&1 || { cat san.log; fail "asan chain serve $args"; }
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan chain serve $args: sanitizer diagnostic"; fi
    echo "OK asan chain serve ${args%% *}: $(tail -1 san.log)"
  done
  unset OMP_NUM_THREADS CAP_RAISE
  make clean >/dev/null 2>&1 || true
}

# Kimi K3's dense chain (kimi_k3_chain.h). The tolerance: the tiny fixture amplifies
# rounding at a few positions. The CPU against itself, with only its RMSNorm's sum taken
# in float instead of double, moves the logits by up to 1.4e-4 of the largest one on the
# f32 trunk and 1.0e-3 on the 8-bit one, and the served logprobs by up to 4.1e-3 (int4
# trunk), at the positions where the chain moves them most (measured on Lavapipe: 1.8e-4
# and 4.5e-4; logprobs 6.1e-3): every logits row is held within 2e-3 of the largest
# |logit|, the logprobs of the serve sessions within 2e-2, the tokens exactly. With the CPU's int8 expert activations
# (K3_IDOT=1) a rounding that flips an int8 step moves the logits further (8e-3, the
# tokens unchanged): that configuration gates on its tokens (TOKENS=1).
k3c_ids() {   # case "ids": the prompt K3C_IDS holds (a prompt past the oracle's)
  if [ "$1" = ids ] && [ -n "${K3C_IDS:-}" ]; then echo "$K3C_IDS"; return; fi
  $PY -c "import json,sys;print(' '.join(map(str,json.load(open('kimi_k3_tiny/ref.json'))['cases'][sys.argv[1]]['prompt_ids'])))" "$1"
}
k3c_close() {  # <cpu.f32> <vk.f32>: max |diff| within 2e-3 of the largest |logit|
  $PY - "$1" "$2" <<'PY'
import array, sys
a = array.array("f", open(sys.argv[1], "rb").read()); b = array.array("f", open(sys.argv[2], "rb").read())
d = max(abs(x - y) for x, y in zip(a, b)) if a and len(a) == len(b) else float("inf")
m = max(abs(x) for x in a) if a else 0.0
print(f"max |logit diff| {d / m if m else d:.1e} of the largest")
sys.exit(0 if d <= 2e-3 * m else 1)
PY
}
# k3c_gate <tag> <env...>: for each of the oracle's prompts (K3C_CASES, default all
# three), the CPU run's tokens, every logits row (K3_VAL_LOGITS: every prefill row and
# every decode step) within k3c_close (TOKENS=1: reported, not gated), and the chain
# ran. EVICT=1: the tier evicted.
# FAULT_BACK=k: the device is lost k frames before the end of the same run without a
# fault (counted from that run's frames, so on any device the loss lands in the same
# forward), and the run must say so; REBUILD=1: and rebuild the KDA state of some
# positions on the CPU.
k3c_gate() {
  local tag=$1 c frames fault; shift
  for c in ${K3C_CASES:-short chunk long}; do
    rm -f k3c.usage cpu.f32 vk.f32
    env "$@" COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 K3_VAL_LOGITS=cpu.f32 ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids $c)" --ngen 8 \
      2> cpu.log | sed 's/ *TUNE.*//' > cpu.tok
    fault=""
    if [ -n "${FAULT_BACK:-}" ]; then
      env "$@" COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 \
        ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids $c)" --ngen 8 2> vk.log > /dev/null
      frames=$(sed -n 's/^\[VK\] kimi_k3 chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p' vk.log | tail -1)
      [ -n "$frames" ] && [ "$frames" -gt "$FAULT_BACK" ] || { cat vk.log; fail "$tag $c: no fault-free run to count frames from"; }
      fault="COLI_VK_CHAIN_FAULT=$((frames - FAULT_BACK + 1))"
    fi
    env "$@" $fault COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 K3_VAL_LOGITS=vk.f32 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 \
      COLI_VK_CHAIN=${CHAINMODE:-1} ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids $c)" --ngen 8 2> vk.log | sed 's/ *TUNE.*//' > vk.tok
    { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; tail -20 vk.log; fail "$tag $c: the chain's tokens differ from the CPU"; }
    if [ -n "$fault" ]; then
      grep -q "kimi_k3 chain: the device was lost" vk.log || { cat vk.log; fail "$tag $c: no loss was handled"; }
      if [ "${REBUILD:-0}" = 1 ]; then grep -q "rebuilding the state of [1-9]" vk.log || { cat vk.log; fail "$tag $c: no state was rebuilt"; }; fi
    else
      [ "$(chain_count kimi_k3 vk.log)" -gt 0 ] || { cat vk.log; fail "$tag $c: the chain never ran"; }
    fi
    if [ "${EVICT:-0}" = 1 ]; then
      [ "$(tier_evictions kimi_k3 vk.log)" -gt 0 ] || { grep '\[VK\] tier' vk.log; fail "$tag $c: the budget forced no eviction"; }
      grep -q ' failed 0 ' vk.log || { grep '\[VK\] tier' vk.log; fail "$tag $c: an upload failed"; }
    fi
    local lg; lg=$(k3c_close cpu.f32 vk.f32) || { echo "$lg"; [ "${TOKENS:-0}" = 1 ] || fail "$tag $c: logits"; }
    echo "OK $tag $c: tokens = CPU, $lg, $(chain_count kimi_k3 vk.log) chain forwards$(grep -o 'rebuilding the state of [0-9]* positions\|the host.s state is current' vk.log | sed 's/^/, /')"
  done
}
k3c_serve_fixture() {   # the tiny fixture with the tokenizer the serve tests speak through
  rm -rf kimi_k3_serve && mkdir kimi_k3_serve
  cp kimi_k3_tiny/config.json kimi_k3_tiny/model.safetensors kimi_k3_serve/
  cp tests/tok_kimi_tiny.json kimi_k3_serve/tokenizer.json
}

# Every configuration the chain takes: Moonshot's oracle with the chain on (the tier on,
# off, the shared experts on the device), every dense format (f32, int8 rows, int4-g64,
# mixed), the CPU's int8 expert activations with the tier off, prefill a token at a
# time and in chain chunks of 3, the tiled GEMM and the per-row GEMV, K3_TOPP, a tier
# budget that evicts, prompts only (the KDA state crossing between the device and the
# CPU at every decode), a device lost mid-decode (the KDA state rebuilt on the CPU), in
# the first prompt chunk (the host's state current) and in a later one (rebuilt); serve
# sessions frame for frame (pins, the prompt cache, prefix reuse, a prompt that diverges
# and one that starts over, the prefill read-out), with recurrent-state checkpoints
# (COLI_K3_CKPT: photos taken from the device's state and restored to it), in chain
# chunks, on prompts only, with a device lost mid-session; the checkpoint and dashboard
# harnesses with the chain on.
family_kimi_chain() {
  make kimi_k3 tests/test_vk_chain VK=1
  ./tests/test_vk_chain shaders/qmatmul.spv | tee vk_chain.log
  tail -1 vk_chain.log | grep -qx PASS || fail "the chain's ops"
  $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  k3c_serve_fixture
  export OMP_NUM_THREADS=2
  local e b out frames
  for e in COLI_VK_TIER=1 COLI_VK_TIER=0 COLI_VK_DENSE=1; do
    env $e COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 \
      $PY tests/test_kimi_k3_tiny.py --binary ./kimi_k3 --fixture ./kimi_k3_tiny
    echo "OK chain kimi_k3 vendor oracle ($e)"
  done
  local O="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0"
  k3c_gate "chain kimi_k3 f32" $O
  k3c_gate "chain kimi_k3 f32, tier off" $O COLI_VK_TIER=0
  k3c_gate "chain kimi_k3 f32, shared experts by COLI_VK_DENSE=1" $O COLI_VK_DENSE=1
  for b in 8 4; do   # int8 rows (fmt 1); int4-g64 (fmt 4) where a row is whole groups, int8 elsewhere
    k3c_gate "chain kimi_k3 ${b}-bit trunk" K3_BITS=$b K3_MLA_BITS=$b K3_HEAD_BITS=$b K3_IDOT=0 COLI_TEMP=0
  done
  k3c_gate "chain kimi_k3 int4 KDA and MoE, int8 MLA and head" K3_BITS=4 K3_IDOT=0 COLI_TEMP=0
  TOKENS=1 k3c_gate "chain kimi_k3 int8 expert activations, tier off" K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 COLI_TEMP=0 COLI_VK_TIER=0
  k3c_gate "chain kimi_k3 prefill one token at a time" $O K3_CHUNK=1
  k3c_gate "chain kimi_k3 chain chunks of 3" $O COLI_VK_CHAIN_ROWS=3
  k3c_gate "chain kimi_k3 tiled GEMM from 2 rows" $O COLI_VK_GEMM_MIN_S=2
  k3c_gate "chain kimi_k3 the per-row GEMV" $O COLI_VK_CHAIN_GEMV=0
  k3c_gate "chain kimi_k3 K3_TOPP=0.6" $O K3_TOPP=0.6
  EVICT=1 k3c_gate "chain kimi_k3 a tier budget of two experts" $O COLI_VK_TIER_GB=0.000005
  CHAINMODE=2 k3c_gate "chain kimi_k3 prompts only" $O
  # the device lost: 9 frames a forward on this fixture (two per sparse layer and the
  # head); the long prompt is three forwards (32, 32 and 8 rows), then 7 decode steps
  FAULT_BACK=3 REBUILD=1 k3c_gate "chain kimi_k3 device lost mid-decode" $O
  K3C_CASES=long FAULT_BACK=86 k3c_gate "chain kimi_k3 device lost in the first prompt chunk" $O K3_CHUNK=32
  K3C_CASES=long FAULT_BACK=77 REBUILD=1 k3c_gate "chain kimi_k3 device lost in a later prompt chunk" $O K3_CHUNK=32
  # in chain chunks of 5 rows (seven chunks a 32-row forward), the loss in the third
  # chunk of the second forward: the state its first chunks advanced is not used
  K3C_CASES=long FAULT_BACK=124 REBUILD=1 k3c_gate "chain kimi_k3 device lost inside a chunked forward" $O COLI_VK_CHAIN_ROWS=5
  # serve sessions frame for frame, the KDA state across turns
  local S="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 K3_PREFIX_LOG=1 USAGE_SAVE=0"
  export CHAIN_SERVE_TOL=2e-2
  rm -rf k3c_photos && mkdir k3c_photos
  # shellcheck disable=SC2086
  {
    $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $S
    out=$($PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $S COLI_K3_CKPT=4); echo "$out"
    $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $S COLI_K3_CKPT=2 COLI_K3_CKPT_DIR=$PWD/k3c_photos
    $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve K3_BITS=4 K3_IDOT=0 K3_PREFIX_LOG=1 USAGE_SAVE=0 COLI_VK_CHAIN_ROWS=3
    COLI_VK_CHAIN=2 $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $S COLI_K3_CKPT=4
    # a device lost mid-session, 134 frames before the end of the same session without a
    # fault (on Lavapipe frame 200 of 334: after a photo brought the state to the host,
    # the state of the 3 positions since then is rebuilt from it)
    frames=$(echo "$out" | sed -n 's/.*kimi_k3 chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p')
    [ -n "$frames" ] && [ "$frames" -gt 134 ] || fail "chain kimi_k3 serve: no fault-free session to count frames from"
    CHAIN_SERVE_EXPECT='kimi_k3 chain: the device was lost' \
      $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $S COLI_K3_CKPT=4 COLI_VK_CHAIN_FAULT=$((frames - 134))
  }
  unset CHAIN_SERVE_TOL
  # recurrent-state checkpoints and the dashboard's lines with the chain on
  COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 \
    $PY tests/test_kimi_k3_ckpt.py --binary ./kimi_k3 --fixture ./kimi_k3_tiny
  COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 \
    $PY tests/test_kimi_k3_dashboard.py --binary ./kimi_k3 --fixture ./kimi_k3_tiny
  unset OMP_NUM_THREADS
}

# The same chain under ASan and UBSan: memory safety is the gate; each run must have
# run the chain (or handled the loss).
family_kimi_chain_sanitize() {
  make clean >/dev/null 2>&1 || true
  make kimi_k3 tests/test_vk_chain VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  ./tests/test_vk_chain shaders/qmatmul.spv > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log || ! tail -1 san.log | grep -qx PASS; then cat san.log; fail "asan: the chain's ops"; fi
  echo "OK asan: the chain's ops"
  $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  k3c_serve_fixture
  export OMP_NUM_THREADS=2
  k3san() {  # <tag> <env and argv...>
    local tag=$1; shift
    rm -f k3c.usage
    env COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 "$@" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    grep -q "kimi_k3 chain: \([1-9][0-9]* forwards\|the device was lost\)" san.log || { cat san.log; fail "$tag: the chain never ran"; }
    echo "OK $tag: sanitizers clean, $(chain_count kimi_k3 san.log) chain forwards"
  }
  local O="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0" ids frames
  ids=$(k3c_ids long)
  # shellcheck disable=SC2086
  {
    k3san "asan chain kimi_k3 f32" env $O ./kimi_k3 kimi_k3_tiny --ids "$ids" --ngen 8
    frames=$(sed -n 's/^\[VK\] kimi_k3 chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p' san.log | tail -1)
    k3san "asan chain kimi_k3 int4 trunk, chain chunks of 3" env K3_BITS=4 K3_IDOT=0 COLI_VK_CHAIN_ROWS=3 ./kimi_k3 kimi_k3_tiny --ids "$ids" --ngen 8
    k3san "asan chain kimi_k3 tier off, every logits row" env $O COLI_VK_TIER=0 K3_VAL_LOGITS=san.f32 ./kimi_k3 kimi_k3_tiny --ids "$ids" --ngen 8
    k3san "asan chain kimi_k3 prompts only" env $O COLI_VK_CHAIN=2 ./kimi_k3 kimi_k3_tiny --ids "$ids" --ngen 8
    # the device lost mid-decode: 3 frames before the end of the run above (a fixed frame
    # number can land in the buffers' setup on a device whose memory is not mapped)
    k3san "asan chain kimi_k3 device lost, rebuilt" env $O COLI_VK_CHAIN_FAULT=$((frames - 2)) ./kimi_k3 kimi_k3_tiny --ids "$ids" --ngen 8
    grep -q "rebuilding the state of [1-9]" san.log || { cat san.log; fail "asan chain kimi_k3 device lost: no state was rebuilt"; }
  }
  CHAIN_SERVE_TOL=2e-2 $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 \
    K3_PREFIX_LOG=1 USAGE_SAVE=0 COLI_K3_CKPT=4 > san.log 2>&1 || { cat san.log; fail "asan chain kimi_k3 serve"; }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan chain kimi_k3 serve: sanitizer diagnostic"; fi
  echo "OK asan chain kimi_k3 serve: $(tail -1 san.log)"
  unset OMP_NUM_THREADS
  make clean >/dev/null 2>&1 || true
}

# DeepSeek V4.1 Flash (deepseek_v41) and DeepSeek V4 (deepseek_v4) on the dense chain
# (COLI_VK_CHAIN=1). v41_gate runs the CPU and then the chain with the same settings:
# the same exit code (the engine fails on a token off its reference), the same printed
# token stream, every logits row (DUMP) within 1e-4 of the largest, the chain ran (or,
# with FAULT_BACK=k, the device was lost k frames before the end of a fault-free run and
# the CPU took over).
v41_gate() {  # <tag> <env...> -- <argv...>
  local tag=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local rc_cpu=0 rc_vk=0
  rm -f chain.usage cpu.f32 vk.f32
  env "${envs[@]}" DUMP=cpu.f32 ./deepseek_v41 "$@" > cpu.txt 2> cpu.log || rc_cpu=$?
  if [ -n "${FAULT_BACK:-}" ]; then
    env "${envs[@]}" COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 \
      ./deepseek_v41 "$@" > /dev/null 2> vk.log || true
    local frames; frames=$(sed -n 's/^\[VK\] deepseek_v41 chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p' vk.log | tail -1)
    [ -n "$frames" ] && [ "$frames" -gt "$FAULT_BACK" ] || { cat vk.log; fail "$tag: no fault-free run to count frames from"; }
    envs+=("COLI_VK_CHAIN_FAULT=$((frames - FAULT_BACK + 1))")
    rm -f chain.usage
  fi
  env "${envs[@]}" DUMP=vk.f32 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 \
    COLI_VK_CHAIN=${CHAINMODE:-1} ./deepseek_v41 "$@" > vk.txt 2> vk.log || rc_vk=$?
  [ "$rc_cpu" = "$rc_vk" ] || { tail -20 vk.log; fail "$tag: exit $rc_vk, the CPU's $rc_cpu"; }
  { [ -s cpu.txt ] && cmp -s cpu.txt vk.txt; } || { cat cpu.txt vk.txt; tail -20 vk.log; fail "$tag: the chain's tokens differ from the CPU"; }
  if [ -n "${FAULT_BACK:-}" ]; then
    grep -q "deepseek_v41 chain: the device was lost" vk.log || { cat vk.log; fail "$tag: no loss was handled"; }
  else
    [ "$(chain_count deepseek_v41 vk.log)" -gt 0 ] || { cat vk.log; fail "$tag: the chain never ran"; }
  fi
  local lg; lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag: logits"; }
  echo "OK $tag: tokens = CPU (exit $rc_cpu), $lg, $(chain_count deepseek_v41 vk.log) chain forwards$(grep -q 'the device was lost' vk.log && echo ', the device lost and the CPU on')"
}
v41_chain_fixtures() {
  $PY tools/make_dsv41_tiny.py --out dsv41_tiny --emit-ref dsv41_tiny/ref.json > /dev/null
  $PY tools/make_dsv41_tiny.py --out dsv41_long --emit-ref dsv41_long/ref.json --prompt-len 40 --max-new 6 > /dev/null
}

# DeepSeek V4 (deepseek_v4) on the dense chain: the deepseek-chain family's V4 half. The
# functions use $PY, fail and chain_count from tests/vulkan_engines.sh and run from c/.
#
# v4_chain_gate <tag> <fixture> <case> <env...>: the CPU run (no COLI_VULKAN) and the
# chain run of one oracle case of the fixture (or ids:a,b,...:n, those ids and n new
# tokens), with the same settings and --record-oracle: the generated ids equal to the
# CPU's (and to the reference's greedy stream), every logits row (DUMP) within
# V4_CHAIN_TOL (5e-1) of its largest |logit|, the same draft attempts and acceptances as
# the CPU (V4_DRAFT), the chain ran, and, on the chain throughout, each decode row equal
# bit for bit to the teacher-forced row at its position. The teacher-forced argmaxes that
# moved are counted, not gated (the logits bound already covers them). FAULT_BACK=k: the
# device is lost k frames before the end of a fault-free run (counted without
# --record-oracle) and the CPU takes over; CHAINMODE=2: prompts only. Each run starts
# from no expert history (the fixture's .coli_usage removed), the tier's uploads awaited.
# Why 0.5: DeepSeek V4 rounds to bf16 after nearly every step and the input of every fp8
# matrix to E4M3 per block; a value the device's sums put on the other side of a bf16
# boundary can move a whole E4M3 block by a step and the indexer's top-k to another row.
# Measured worst on Lavapipe and a Radeon 780M: 0.32 (deepseek_v4_tiny_g2, long). The
# check that no driver blurs is v4_chain_same (and the decode-row check above).
v4_chain_prompt() {  # <fixture> <case>: the case's prompt as the tiny vocabulary's text (ids:a,b,...: those ids)
  $PY -c 'import json,sys; c=sys.argv[2]; ids=[int(t) for t in c[4:].split(":")[0].split(",")] if c.startswith("ids:") else json.load(open(sys.argv[1]+"/ref.json"))["cases"][c]["prompt_ids"]; print("".join("<t%03d>" % t for t in ids))' "$1" "$2"
}
v4_chain_logits() {  # <cpu.f32> <vk.f32> <vocab>: the worst row's |diff| over its largest |logit|
  $PY - "$1" "$2" "$3" "${V4_CHAIN_TOL:-5e-1}" <<'PY'
import array, sys
a = array.array("f", open(sys.argv[1], "rb").read()); b = array.array("f", open(sys.argv[2], "rb").read())
V, tol = int(sys.argv[3]), float(sys.argv[4])
if not a or len(a) != len(b): print(f"logits: {len(a)} and {len(b)} values"); sys.exit(1)
worst, same = 0.0, 0
for r in range(len(a) // V):
    ra, rb = a[r * V:(r + 1) * V], b[r * V:(r + 1) * V]
    d = max(abs(x - y) for x, y in zip(ra, rb)); m = max(abs(x) for x in ra)
    same += d == 0
    worst = max(worst, d / m if m else d)
print(f"logits: {same} of {len(a) // V} rows identical, worst row {worst:.2e} of its largest")
sys.exit(0 if worst <= tol else 1)
PY
}
# v4_chain_same <tag> <fixture> <case> <A> <B> <env...>: the chain twice, with the settings
# A and then B (space-separated KEY=VALUE lists) on top of env: the same ids and every
# logits row the same bits. The chain's arithmetic does not depend on how the rows are
# cut (every matrix takes the per-row GEMV, every other op is per row or in the CPU's
# row order) nor on which side of the tier computed an expert, so chunks of 1 or 3, the
# tier off or a draft's rows cut differently give the default run's bits; a state the
# device kept stale across forwards (a rejected draft longer than a chunk) shows here
# even where its logits stay within the CPU's tolerance.
v4_chain_same() {
  local tag=$1 fx=$2 c=$3 A=$4 B=$5; shift 5
  local p mt side
  p=$(v4_chain_prompt "$fx" "$c")
  mt=$($PY -c 'import json,sys; c=sys.argv[2]; print(c.split(":")[2] if c.startswith("ids:") else json.load(open(sys.argv[1]+"/ref.json"))["cases"][c]["max_new_tokens"])' "$fx" "$c")
  for side in a b; do
    rm -f "$fx/.coli_usage" "v4-$side.f32"
    # shellcheck disable=SC2046
    env "$@" $([ $side = a ] && echo "$A" || echo "$B") DUMP=v4-$side.f32 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 \
      ./deepseek_v4 "./$fx" "$p" --raw-prompt --max-tokens "$mt" --record-oracle v4-$side.json > /dev/null 2> v4-$side.err ||
      { tail -20 v4-$side.err; fail "$tag: run $side"; }
    [ "$(chain_count deepseek_v4 v4-$side.err)" -gt 0 ] || { cat v4-$side.err; fail "$tag: the chain never ran ($side)"; }
  done
  rm -f "$fx/.coli_usage"
  cmp -s v4-a.f32 v4-b.f32 && $PY -c 'import json,sys; sys.exit(json.load(open("v4-a.json"))["full_ids"] != json.load(open("v4-b.json"))["full_ids"])' ||
    { v4_chain_logits v4-a.f32 v4-b.f32 "$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]+"/config.json"))["vocab_size"])' "$fx")"; fail "$tag: $A and $B give different bits"; }
  echo "OK $tag (${c%%:*}): $A and $B: the same ids and the same logits bit for bit ($(($(wc -c < v4-a.f32) / 4)) values), $(chain_count deepseek_v4 v4-b.err) chain forwards"
}
v4_chain_gate() {
  local tag=$1 fx=$2 c=$3; shift 3
  local p mt lg frames fault=()
  p=$(v4_chain_prompt "$fx" "$c")
  mt=$($PY -c 'import json,sys; c=sys.argv[2]; print(c.split(":")[2] if c.startswith("ids:") else json.load(open(sys.argv[1]+"/ref.json"))["cases"][c]["max_new_tokens"])' "$fx" "$c")
  rm -f "$fx/.coli_usage" cpu.f32 vk.f32 v4-cpu.json v4-vk.json
  env "$@" DUMP=cpu.f32 ./deepseek_v4 "./$fx" "$p" --raw-prompt --max-tokens "$mt" --record-oracle v4-cpu.json > /dev/null 2> v4-cpu.err ||
    { cat v4-cpu.err; fail "$tag: CPU run"; }
  if [ -n "${FAULT_BACK:-}" ]; then
    rm -f "$fx/.coli_usage"
    env "$@" COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 ./deepseek_v4 "./$fx" "$p" --raw-prompt --max-tokens "$mt" > /dev/null 2> v4-vk.err || true
    frames=$(sed -n 's/^\[VK\] deepseek_v4 chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p' v4-vk.err | tail -1)
    [ -n "$frames" ] && [ "$frames" -gt "$FAULT_BACK" ] || { cat v4-vk.err; fail "$tag: no fault-free run to count frames from"; }
    fault=("COLI_VK_CHAIN_FAULT=$((frames - FAULT_BACK + 1))")
  fi
  rm -f "$fx/.coli_usage"
  env "$@" "${fault[@]}" DUMP=vk.f32 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} \
    ./deepseek_v4 "./$fx" "$p" --raw-prompt --max-tokens "$mt" --record-oracle v4-vk.json > /dev/null 2> v4-vk.err ||
    { tail -20 v4-vk.err; fail "$tag: chain run"; }
  rm -f "$fx/.coli_usage"
  $PY - "$fx" "$c" <<'PY' || { tail -20 v4-vk.err; fail "$tag: the chain's session differs from the CPU's"; }
import json, sys
a, b = json.load(open("v4-cpu.json")), json.load(open("v4-vk.json"))
ref = a["full_ids"] if sys.argv[2].startswith("ids:") else json.load(open(sys.argv[1] + "/ref.json"))["cases"][sys.argv[2]]["greedy_full_ids"]
ok = a["full_ids"] == b["full_ids"] == ref and len(a["tf_pred"]) == len(b["tf_pred"])
if not ok: print("CPU", a["full_ids"], "chain", b["full_ids"], "ref", ref)
open("v4-tf.txt", "w").write(str(sum(x != y for x, y in zip(a["tf_pred"], b["tf_pred"]))))
sys.exit(0 if ok else 1)
PY
  if [ "${EVICT:-0}" = 1 ]; then   # the budget forced evictions
    [ "$(tier_evictions deepseek_v4 v4-vk.err)" -gt 0 ] || { grep '\[VK\] tier' v4-vk.err; fail "$tag: the budget forced no eviction"; }
  fi
  # drafts (V4_DRAFT): the same attempts, drafted and accepted tokens as the CPU's
  [ "$(grep -ao 'v4_dspark attempts=[0-9]* drafted=[0-9]* accepted=[0-9]*' v4-cpu.err)" = \
    "$(grep -ao 'v4_dspark attempts=[0-9]* drafted=[0-9]* accepted=[0-9]*' v4-vk.err)" ] || { grep -a v4_dspark v4-cpu.err v4-vk.err; fail "$tag: the drafts went otherwise"; }
  if [ -n "${FAULT_BACK:-}" ]; then
    grep -q "deepseek_v4 chain: the device was lost" v4-vk.err || { cat v4-vk.err; fail "$tag: no loss was handled"; }
  else
    [ "$(chain_count deepseek_v4 v4-vk.err)" -gt 0 ] || { cat v4-vk.err; fail "$tag: the chain never ran"; }
  fi
  lg=$(v4_chain_logits cpu.f32 vk.f32 "$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]+"/config.json"))["vocab_size"])' "$fx")") ||
    { echo "$lg"; fail "$tag: logits"; }
  # an oracle case on the chain throughout: its decode rows are the teacher-forced
  # forward's rows at the same positions, bit for bit (as on the CPU)
  if [ "${c#ids:}" = "$c" ] && [ "${CHAINMODE:-1}" = 1 ] && [ -z "${FAULT_BACK:-}" ] && [ "${*#*V4_DRAFT}" = "$*" ]; then
    $PY - vk.f32 "$fx" "$c" <<'PY' || fail "$tag: a decode row differs from the teacher-forced row at its position"
import array, json, sys
v = array.array("f", open(sys.argv[1], "rb").read())
V = json.load(open(sys.argv[2] + "/config.json"))["vocab_size"]
case = json.load(open(sys.argv[2] + "/ref.json"))["cases"][sys.argv[3]]
P, mt = len(case["prompt_ids"]), case["max_new_tokens"]
row = lambda r: v[r * V:(r + 1) * V]
for j in range(mt):            # session rows: the prompt's last position, then each decode step
    if row(j) != row(mt + P - 1 + j): sys.exit(1)
PY
    lg="$lg, decode rows = teacher-forced rows"
  fi
  echo "OK $tag (${c%%:*}): ids = CPU$(case $c in ids:*) ;; *) echo ' = reference';; esac), $(cat v4-tf.txt) teacher-forced argmaxes moved, $lg, $(chain_count deepseek_v4 v4-vk.err) chain forwards$(grep -q 'the device was lost' v4-vk.err && echo ', the device lost and the CPU on')$(grep -ao 'v4_dspark attempts=[0-9]* drafted=[0-9]* accepted=[0-9]*' v4-vk.err | tail -1 | sed 's/^/, /')$([ "${EVICT:-0}" = 1 ] && grep -ao 'evictions [0-9]*' v4-vk.err | tail -1 | sed 's/^/, /')"
}
# v4_chain_fixtures: the 4-expert fixture (a copy), the 8-expert one (pinned rows16
# experts), and three more geometries: 8 heads in 2 output groups (wo_a per group), a
# window of 4 with ratios 4 and 2 (the window and the rings roll over more often), and
# DeepSeek V4's indexer of 64 heads of 128 (the scores past the shared staging).
v4_chain_fixtures() {
  $PY tools/make_deepseek_v4_tiny.py --output deepseek_v4_tiny_t --force > /dev/null
  local spec
  for spec in "deepseek_v4_tiny_e8 EXPERTS=8" "deepseek_v4_tiny_g2 HEADS=8 O_GROUPS=2" \
              "deepseek_v4_tiny_w4 SLIDING=4 COMPRESS_RATIOS=[0,4,2] HCA=2" "deepseek_v4_tiny_ix INDEX_HEADS=64 INDEX_DIM=128"; do
    # shellcheck disable=SC2086
    $PY - tools/make_deepseek_v4_tiny.py $spec <<'PY' > /dev/null
import importlib.util, sys
spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
for kv in sys.argv[3:]:
    k, v = kv.split("=", 1)
    if k != "HCA": setattr(gen, k, eval(v)); continue
    hf = gen.make_hf_config                      # the heavily compressed layers' ratio
    def patched(C, hf=hf, r=int(v)):
        cfg = hf(C); cfg.compress_rates["heavily_compressed_attention"] = r; return cfg
    gen.make_hf_config = patched
sys.argv = ["make_deepseek_v4_tiny.py", "--output", sys.argv[2], "--force"]
gen.main()
PY
  done
}
# v4_chain_serve: the brio served prompt's per-position logprob echoes with the chain
# against the CPU's (pinned, then a prompt that extends the pin, then max_tokens 0),
# and the prefix-reuse contract (tests/test_deepseek_v4_prefix.py) with the chain on.
v4_chain_serve() {  # <fixture>
  $PY - "$1" <<'PY' || fail "deepseek_v4 chain served: the chain's echoes differ from the CPU's"
import json, os, sys
from pathlib import Path
sys.path.insert(0, "tests")
import test_deepseek_v4_brio as b
fx = Path(sys.argv[1])
case = json.load(open(fx / "ref.json"))["cases"]["long"]
ids = case["prompt_ids"]
def turns(chain):
    for k in ("COLI_VULKAN", "COLI_VK_CHAIN", "COLI_VK_TIER_SYNC"): os.environ.pop(k, None)
    try: os.remove(fx / ".coli_usage")
    except FileNotFoundError: pass
    if chain: os.environ.update(COLI_VULKAN="1", COLI_VK_CHAIN=os.environ.get("V4_CHAIN_MODE", "1"), COLI_VK_TIER_SYNC="1")
    s = b.Serve(Path("deepseek_v4").resolve(), fx)
    out = []
    try:
        out.append(s.submit(b.token_prompt(ids[:40]), 3, logprobs=4, pin=True))
        out.append(s.submit(b.token_prompt(ids[:40] + ids[40:60]), 4, logprobs=4))
        out.append(s.submit(b.token_prompt(ids), 0, logprobs=5))
        out.append(s.submit(b.token_prompt(ids[:30]), 4))
    finally:
        s.close()
        err = s.process.stderr.read().decode(errors="replace")
    return out, err
cpu, err_cpu = turns(False)
dev, err = turns(True)
for text in (err_cpu, err):   # a sanitized build reports into the engine's stderr
    if "ERROR: AddressSanitizer" in text or "runtime error:" in text: print(text[-4000:]); sys.exit("sanitizer diagnostic")
if "deepseek_v4 chain:" not in err or " forwards" not in err:
    print(err[-3000:]); sys.exit("the chain never ran")
worst, n, same = 0.0, 0, 0
for x, y in zip(cpu, dev):
    if [d[0] for d in x.data] != [d[0] for d in y.data] or sorted(x.echoes) != sorted(y.echoes) or x.reuse != y.reuse:
        print("CPU", x.data, x.echoes.keys(), x.reuse, "chain", y.data, y.echoes.keys(), y.reuse); sys.exit(1)
    for p in x.echoes:
        n += 1; same += x.echoes[p] == y.echoes[p]
        if x.echoes[p]["token"] != y.echoes[p]["token"]: sys.exit(1)
        worst = max(worst, abs(x.echoes[p]["lp"] - y.echoes[p]["lp"]))
fw = [l for l in err.splitlines() if "deepseek_v4 chain:" in l and "forwards" in l][-1]
print(f"OK deepseek_v4 chain served: the pin, its extension, a read-only prompt and a shorter one: texts and reuse = CPU, "
      f"{same} of {n} logprob echoes identical, worst |delta| {worst:.2e}; {fw.split('] ', 1)[1][:48]}")
sys.exit(0 if worst <= 0.5 else 1)
PY
}
# v4_chain_san <tag> <env and argv...>: a sanitized build's chain run (FAULT_BACK=k: the
# loss k frames before the end of a fault-free run): no sanitizer diagnostic, and the
# chain ran (or handled the loss).
v4_chain_san() {
  local tag=$1; shift
  local fault=() frames
  if [ -n "${FAULT_BACK:-}" ]; then
    rm -f deepseek_v4_tiny_*/.coli_usage
    env COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 "$@" > san.log 2>&1 || true
    frames=$(sed -n 's/^\[VK\] deepseek_v4 chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p' san.log | tail -1)
    [ -n "$frames" ] && [ "$frames" -gt "$FAULT_BACK" ] || { cat san.log; fail "$tag: no fault-free run to count frames from"; }
    fault=("COLI_VK_CHAIN_FAULT=$((frames - FAULT_BACK + 1))")
  fi
  rm -f deepseek_v4_tiny_*/.coli_usage
  env COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 "${fault[@]}" "$@" > san.log 2>&1 || true
  rm -f deepseek_v4_tiny_*/.coli_usage
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
  grep -q "deepseek_v4 chain: \([1-9][0-9]* forwards\|the device was lost\)" san.log || { cat san.log; fail "$tag: the chain never ran"; }
  echo "OK $tag: sanitizers clean, $(chain_count deepseek_v4 san.log) chain forwards$(grep -q 'the device was lost' san.log && echo ', the device lost')"
}

# The V4 half of the family: the engine, its fixtures, then every configuration: 4 and 8
# experts (pinned rows16), the three oracle cases (the long one rolls the window of 8 and
# both compressors over many times), 2 output groups, a window of 4 with ratios 4 and 2,
# V4's indexer of 64 heads of 128, eviction, the tier off, the per-matrix trunk beside,
# prefill and chain chunks, the per-row GEMV, V4_IDX_IDENTITY, prompts only, n-gram
# drafts accepted and rejected (V4_MTP on a one-layer MTP checkpoint runs target-only),
# the chain against itself, a device lost mid-decode, in the prompt and between drafts,
# served turns (a pin, its extension, a read-only prompt), and the engine's own serve
# tests with the chain on.
v4_chain_family() {
  make deepseek-v4 VK=1
  v4_chain_fixtures
  local c fx
  for c in short compressed long; do
    v4_chain_gate "chain deepseek_v4 4 experts" deepseek_v4_tiny_t $c
    v4_chain_gate "chain deepseek_v4 8 experts, pinned rows16" deepseek_v4_tiny_e8 $c
  done
  for c in short long; do
    v4_chain_gate "chain deepseek_v4 2 output groups" deepseek_v4_tiny_g2 $c
    v4_chain_gate "chain deepseek_v4 a window of 4, ratios 4 and 2" deepseek_v4_tiny_w4 $c
    v4_chain_gate "chain deepseek_v4 an indexer of 64 heads of 128" deepseek_v4_tiny_ix $c
  done
  EVICT=1 v4_chain_gate "chain deepseek_v4 a budget of three experts" deepseek_v4_tiny_e8 long COLI_VK_TIER_GB=0.00009
  v4_chain_gate "chain deepseek_v4 tier off" deepseek_v4_tiny_t long COLI_VK_TIER=0 COLI_VK_DENSE=0
  v4_chain_gate "chain deepseek_v4 beside the per-matrix trunk" deepseek_v4_tiny_t long COLI_VK_DENSE=1
  v4_chain_gate "chain deepseek_v4 prefill chunks of 7" deepseek_v4_tiny_t long V4_PREFILL_CHUNK=7
  v4_chain_gate "chain deepseek_v4 chain chunks of 3" deepseek_v4_tiny_e8 long COLI_VK_CHAIN_ROWS=3
  v4_chain_gate "chain deepseek_v4 chunks of 5, a window of 4" deepseek_v4_tiny_w4 long COLI_VK_CHAIN_ROWS=5
  v4_chain_gate "chain deepseek_v4 the per-row GEMV" deepseek_v4_tiny_t long COLI_VK_CHAIN_GEMV=0
  v4_chain_gate "chain deepseek_v4 V4_IDX_IDENTITY=1" deepseek_v4_tiny_t long V4_IDX_IDENTITY=1
  CHAINMODE=2 v4_chain_gate "chain deepseek_v4 prompts only" deepseek_v4_tiny_t long
  # n-gram drafts on prompts whose tails repeat: accepted and rejected (A), every one
  # rejected (R), a mix over three attempts (M)
  local A=ids:20,21,22,23,24,25,26,27,28,29,20,21,22:12 R=ids:20,21,22,23,50,51,52,20,21,22:12
  local M=ids:30,31,32,33,34,35,60,61,30,31,32,33,34,35,36,37,38,39,40,41,30,31,32:12
  v4_chain_gate "chain deepseek_v4 n-gram drafts accepted and rejected" deepseek_v4_tiny_t $A V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  v4_chain_gate "chain deepseek_v4 an n-gram draft rejected" deepseek_v4_tiny_t $R V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  v4_chain_gate "chain deepseek_v4 n-gram drafts, a window of 4" deepseek_v4_tiny_w4 $M V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  v4_chain_gate "chain deepseek_v4 n-gram drafts, 8 experts" deepseek_v4_tiny_e8 $M V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  CHAINMODE=2 v4_chain_gate "chain deepseek_v4 prompts only, drafts" deepseek_v4_tiny_w4 $M V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  v4_chain_gate "chain deepseek_v4 V4_MTP=1 (one MTP layer: target only)" deepseek_v4_tiny_t long V4_MTP=1 V4_DRAFT=3
  # the chain against itself
  for fx in deepseek_v4_tiny_t deepseek_v4_tiny_w4 deepseek_v4_tiny_g2; do
    v4_chain_same "chain deepseek_v4 chunks, $fx" $fx long "X=1" "COLI_VK_CHAIN_ROWS=1"
    v4_chain_same "chain deepseek_v4 prefill chunks, $fx" $fx long "X=1" "V4_PREFILL_CHUNK=3"
  done
  v4_chain_same "chain deepseek_v4 the tier off" deepseek_v4_tiny_e8 long "X=1" "COLI_VK_TIER=0 COLI_VK_DENSE=0"
  v4_chain_same "chain deepseek_v4 a draft rejected, chunks of 2" deepseek_v4_tiny_w4 $R "X=1" "COLI_VK_CHAIN_ROWS=2" V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  v4_chain_same "chain deepseek_v4 drafts, chunks of 1" deepseek_v4_tiny_w4 $M "X=1" "COLI_VK_CHAIN_ROWS=1" V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  v4_chain_same "chain deepseek_v4 drafts, chunks of 3, 8 experts" deepseek_v4_tiny_e8 $A "X=1" "COLI_VK_CHAIN_ROWS=3" V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  FAULT_BACK=3 v4_chain_gate "chain deepseek_v4 device lost mid-decode" deepseek_v4_tiny_t long
  FAULT_BACK=40 v4_chain_gate "chain deepseek_v4 device lost in the prompt" deepseek_v4_tiny_t long COLI_VK_CHAIN_ROWS=7
  FAULT_BACK=10 v4_chain_gate "chain deepseek_v4 device lost between drafts" deepseek_v4_tiny_w4 $M V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  v4_chain_serve deepseek_v4_tiny_t
  V4_CHAIN_MODE=2 v4_chain_serve deepseek_v4_tiny_t
  local t
  for t in test_deepseek_v4_prefix test_deepseek_v4_brio test_deepseek_v4_tiny; do
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_USAGE=$PWD/chain.usage \
    $PY tests/$t.py --binary "$PWD/deepseek_v4" --fixture "$PWD/deepseek_v4_tiny_t" > v4-test.log 2>&1 ||
    { cat v4-test.log; fail "deepseek_v4 $t with the chain"; }
  echo "OK deepseek_v4 $t with the chain: $(tail -1 v4-test.log)"
  done
  rm -rf deepseek_v4_tiny_t deepseek_v4_tiny_e8 deepseek_v4_tiny_g2 deepseek_v4_tiny_w4 deepseek_v4_tiny_ix chain.usage
  rm -f cpu.f32 vk.f32 v4-*.f32 v4-*.json v4-*.err v4-tf.txt v4-test.log
}

# The same under ASan and UBSan (LTO off, as family_deepseek_sanitize builds it): no
# diagnostic, and each run ran the chain (or handled the loss); the serve tests with
# UBSan halting on its first report.
v4_chain_sanitize_runs() {
  local p t
  p=$(v4_chain_prompt deepseek_v4_tiny_t long)
  mk() { v4_chain_prompt x "ids:$1"; }
  v4_chain_san "asan chain deepseek_v4 long, 4 experts" ./deepseek_v4 ./deepseek_v4_tiny_t "$p" --raw-prompt --max-tokens 4 --record-oracle san.json
  v4_chain_san "asan chain deepseek_v4 8 experts, pinned rows16, eviction" COLI_VK_TIER_GB=0.00009 ./deepseek_v4 ./deepseek_v4_tiny_e8 "$p" --raw-prompt --max-tokens 4 --record-oracle san.json
  v4_chain_san "asan chain deepseek_v4 chunks of 3, a window of 4" COLI_VK_CHAIN_ROWS=3 ./deepseek_v4 ./deepseek_v4_tiny_w4 "$(v4_chain_prompt deepseek_v4_tiny_w4 long)" --raw-prompt --max-tokens 4 --record-oracle san.json
  v4_chain_san "asan chain deepseek_v4 2 output groups" ./deepseek_v4 ./deepseek_v4_tiny_g2 "$(v4_chain_prompt deepseek_v4_tiny_g2 long)" --raw-prompt --max-tokens 4
  v4_chain_san "asan chain deepseek_v4 an indexer of 64 heads of 128" ./deepseek_v4 ./deepseek_v4_tiny_ix "$(v4_chain_prompt deepseek_v4_tiny_ix long)" --raw-prompt --max-tokens 4
  v4_chain_san "asan chain deepseek_v4 drafts rejected, chunks of 2" V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1 COLI_VK_CHAIN_ROWS=2 ./deepseek_v4 ./deepseek_v4_tiny_w4 "$(mk 20,21,22,23,50,51,52,20,21,22)" --raw-prompt --max-tokens 12
  v4_chain_san "asan chain deepseek_v4 drafts" V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1 ./deepseek_v4 ./deepseek_v4_tiny_w4 "$(mk 30,31,32,33,34,35,60,61,30,31,32,33,34,35,36,37,38,39,40,41,30,31,32)" --raw-prompt --max-tokens 12
  FAULT_BACK=10 v4_chain_san "asan chain deepseek_v4 device lost" ./deepseek_v4 ./deepseek_v4_tiny_t "$p" --raw-prompt --max-tokens 4
  v4_chain_serve deepseek_v4_tiny_t > san.log 2>&1 || { cat san.log; fail "asan chain deepseek_v4 served"; }
  echo "OK asan chain deepseek_v4 served: $(tail -1 san.log | cut -c1-120)"
  for t in test_deepseek_v4_prefix test_deepseek_v4_brio; do
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 $PY tests/$t.py --binary $PWD/deepseek_v4 --fixture $PWD/deepseek_v4_tiny_t > san.log 2>&1 || { cat san.log; fail "asan $t with the chain"; }
    echo "OK asan $t with the chain: $(tail -1 san.log)"
  done
}
v4_chain_family_sanitize() {
  local SAN="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  make deepseek-v4-clean >/dev/null 2>&1 || true
  make deepseek-v4 VK=1 LTO=0 EXTRA_CFLAGS="$SAN" EXTRA_LDFLAGS="$SAN"
  v4_chain_fixtures
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  v4_chain_sanitize_runs
  rm -rf deepseek_v4_tiny_t deepseek_v4_tiny_e8 deepseek_v4_tiny_g2 deepseek_v4_tiny_w4 deepseek_v4_tiny_ix san.json san.log
  make deepseek-v4-clean >/dev/null 2>&1 || true
}

# deepseek_v41 in every configuration the chain takes: the expert cache from one slot
# to all (cap 1, 2, 8), the 8-token and the 40-token prompt (the window ring of 8 and
# the ratio-2 groups roll over many times; the candidate blocks and the published index
# keys of the ratio-1 group), DSpark drafts accepted (1, 3) and rejected (2, 4, 5:
# undo rows on the device's copies), prompts in chunks of 3 and 7 (the device's ring
# wraps), prompts only (decode on the CPU between them), the tier off, the per-matrix
# trunk beside, the tiled GEMM from two rows, the per-row GEMV, V41_INDEX_OWNER, a
# device lost mid-decode, in a prompt and between drafts; serve sessions frame for
# frame (pins, the prompt cache, the prefill read-out, DSpark, prompts only), images
# on the wire, and the engine's own serve tests with the chain on. Then deepseek_v4
# (v4_chain_family above).
family_deepseek_chain() {
  make deepseek_v41 tests/test_vk_chain VK=1
  ./tests/test_vk_chain shaders/qmatmul.spv | tee vk_chain.log
  tail -1 vk_chain.log | grep -qx PASS || fail "the chain's ops"
  v41_chain_fixtures
  export OMP_NUM_THREADS=2
  local cap f
  for cap in 1 2 8; do v41_gate "chain deepseek_v41 cap=$cap" SNAP=dsv41_tiny -- $cap dsv41_tiny/ref.json; done
  for cap in 2 8; do v41_gate "chain deepseek_v41 40-token prompt cap=$cap" SNAP=dsv41_long -- $cap dsv41_long/ref.json; done
  for f in 1 2 3 4 5; do v41_gate "chain deepseek_v41 DSpark spec=$f" SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=$f -- 8 dsv41_tiny/ref.json; done
  v41_gate "chain deepseek_v41 DSpark spec=5, one cache slot" SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=5 -- 1 dsv41_tiny/ref.json
  v41_gate "chain deepseek_v41 prompt in chunks of 3" SNAP=dsv41_long COLI_VK_CHAIN_ROWS=3 -- 8 dsv41_long/ref.json
  v41_gate "chain deepseek_v41 prompt in chunks of 7" SNAP=dsv41_long COLI_VK_CHAIN_ROWS=7 -- 2 dsv41_long/ref.json
  CHAINMODE=2 v41_gate "chain deepseek_v41 prompts only" SNAP=dsv41_long -- 8 dsv41_long/ref.json
  CHAINMODE=2 v41_gate "chain deepseek_v41 prompts only, DSpark spec=2" SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=2 -- 8 dsv41_tiny/ref.json
  v41_gate "chain deepseek_v41 tier off" SNAP=dsv41_long COLI_VK_TIER=0 -- 8 dsv41_long/ref.json
  v41_gate "chain deepseek_v41 beside the per-matrix trunk" SNAP=dsv41_long COLI_VK_DENSE=1 -- 8 dsv41_long/ref.json
  v41_gate "chain deepseek_v41 tiled GEMM from 2 rows" SNAP=dsv41_long COLI_VK_GEMM_MIN_S=2 -- 8 dsv41_long/ref.json
  v41_gate "chain deepseek_v41 the per-row GEMV" SNAP=dsv41_tiny COLI_VK_CHAIN_GEMV=0 -- 8 dsv41_tiny/ref.json
  v41_gate "chain deepseek_v41 V41_INDEX_OWNER=1" SNAP=dsv41_long V41_INDEX_OWNER=1 -- 8 dsv41_long/ref.json
  FAULT_BACK=5 v41_gate "chain deepseek_v41 device lost mid-decode" SNAP=dsv41_long -- 8 dsv41_long/ref.json
  FAULT_BACK=100 v41_gate "chain deepseek_v41 device lost in the prompt" SNAP=dsv41_long COLI_VK_CHAIN_ROWS=7 -- 8 dsv41_long/ref.json
  FAULT_BACK=8 v41_gate "chain deepseek_v41 device lost between drafts" SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=5 -- 8 dsv41_tiny/ref.json
  $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=0
  $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=1
  COLI_VK_CHAIN=2 $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=1 COLI_VK_CHAIN_ROWS=3
  $PY tests/vulkan_chain_v41_image.py ./deepseek_v41 dsv41_tiny V41_DSPARK=0
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_USAGE=$PWD/chain.usage \
    $PY -m unittest tests.test_dsv41_prefix_serve tests.test_dsv41_dspark_serve
  rm -f chain.usage chain-serve.usage chain-image.usage
  v4_chain_family
  unset OMP_NUM_THREADS
}

# The same chain under ASan and UBSan: memory safety is the gate; each run must have run
# the chain (or handled the loss), with the ASAN_OPTIONS of every sanitized family.
family_deepseek_chain_sanitize() {
  local SAN="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  make clean >/dev/null 2>&1 || true
  make deepseek_v41 tests/test_vk_chain VK=1 EXTRA_CFLAGS="$SAN"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  ./tests/test_vk_chain shaders/qmatmul.spv > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log || ! tail -1 san.log | grep -qx PASS; then cat san.log; fail "asan: the chain's ops"; fi
  echo "OK asan: the chain's ops"
  v41_chain_fixtures
  export OMP_NUM_THREADS=2
  dsan() {  # <tag> <env and argv...>   (FAULT_BACK=k: the loss k frames before the end of a fault-free run)
    local tag=$1; shift
    local fault=()
    if [ -n "${FAULT_BACK:-}" ]; then
      rm -f chain.usage
      env COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 "$@" > san.log 2>&1 || true
      local frames; frames=$(sed -n 's/^\[VK\] deepseek_v41 chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p' san.log | tail -1)
      [ -n "$frames" ] && [ "$frames" -gt "$FAULT_BACK" ] || { cat san.log; fail "$tag: no fault-free run to count frames from"; }
      fault=("COLI_VK_CHAIN_FAULT=$((frames - FAULT_BACK + 1))")
    fi
    rm -f chain.usage
    env COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 "${fault[@]}" "$@" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    grep -q "deepseek_v41 chain: \([1-9][0-9]* forwards\|the device was lost\)" san.log || { cat san.log; fail "$tag: the chain never ran"; }
    echo "OK $tag: sanitizers clean, $(chain_count deepseek_v41 san.log) chain forwards"
  }
  dsan "asan chain deepseek_v41 40-token prompt" SNAP=dsv41_long ./deepseek_v41 8 dsv41_long/ref.json
  dsan "asan chain deepseek_v41 DSpark spec=5, one cache slot" SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=5 ./deepseek_v41 1 dsv41_tiny/ref.json
  dsan "asan chain deepseek_v41 prompt in chunks of 3" SNAP=dsv41_long COLI_VK_CHAIN_ROWS=3 ./deepseek_v41 2 dsv41_long/ref.json
  FAULT_BACK=20 dsan "asan chain deepseek_v41 device lost" SNAP=dsv41_long ./deepseek_v41 8 dsv41_long/ref.json
  $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=1 > san.log 2>&1 || { cat san.log; fail "asan chain deepseek_v41 serve"; }
  echo "OK asan chain serve deepseek_v41: $(tail -1 san.log)"
  $PY tests/vulkan_chain_v41_image.py ./deepseek_v41 dsv41_tiny V41_DSPARK=0 > san.log 2>&1 || { cat san.log; fail "asan chain deepseek_v41 images"; }
  echo "OK asan chain images deepseek_v41: $(tail -1 san.log)"
  rm -f chain.usage chain-serve.usage chain-image.usage
  v4_chain_family_sanitize
  unset OMP_NUM_THREADS
  make clean >/dev/null 2>&1 || true
}

# ---- the dense weights on the device only (COLI_VK_DENSE_HOST=0; docs/vulkan.md,
# "Dense weights on the device only"): one file per engine group,
# tests/vulkan_dense_only_<group>.sh, each defining dho_family_<group> and
# dho_family_<group>_sanitize (run as dense-only-<group> and dense-only-<group>-sanitize).
#
# dho_dropped <log>: matrices the engine placed on the device only (its startup line)
dho_dropped() {
  local n; n=$(sed -n 's/^\[VK\] [a-z0-9_]*: \([0-9][0-9]*\) dense matrices on the device only.*/\1/p' "$1" | tail -1)
  echo "${n:-0}"
}
# dho_reloaded <log>: host copies read back from disk for the CPU (the exit line)
dho_reloaded() {
  local n; n=$(sed -n 's/^\[VK\] [a-z0-9_]*: dense weights at exit: .*), \([0-9][0-9]*\) read back from disk.*/\1/p' "$1" | tail -1)
  echo "${n:--1}"
}
# dho_rss <log>: the resident set the exit line reports, in GiB
dho_rss() { sed -n 's/^\[VK\] [a-z0-9_]*: dense weights at exit: .*; RSS \([0-9.]*\) GiB$/\1/p' "$1" | tail -1; }
# dho_gate <engine> <tag> <tol 0|1|2> <env...> -- <argv...>: one configuration three
# times, on the CPU, on the device keeping its host copies (COLI_VK_DENSE_HOST=1) and on
# the device only (=0), the tier deterministic (COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0)
# and each run from a fresh history. Gates: the CPU's tokens; matrices dropped and none read
# back; with the chain asked for (CHAINMODE, default 1; 0 = the per-matrix path alone with
# COLI_VK_DENSE=1), chain forwards; tol 2: the logits the bytes of the host-copy run (what
# runs on the device is the same), 1: within 1e-4 of the CPU's largest logit (logits_close),
# 0: tokens only. CPUENV: the CPU arm's own settings; DUMPENV: the variable the engine
# writes its logits to (default DUMP).
dho_gate() {
  local eng=$1 tag=$2 tol=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local dv=${DUMPENV:-DUMP} mode=${CHAINMODE:-1}
  local vk=(COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VULKAN=1 COLI_VK_CHAIN=$mode)
  [ "$mode" = 0 ] && vk+=(COLI_VK_DENSE=1)
  rm -f chain.usage cpu.f32 vk.f32 host.f32
  env "${envs[@]}" ${CPUENV:-} $dv=cpu.f32 ./"$eng" "$@" > cpu.log 2>&1 || true
  env "${envs[@]}" "${vk[@]}" COLI_USAGE=chain.usage COLI_VK_DENSE_HOST=1 $dv=host.f32 ./"$eng" "$@" > host.log 2>&1 || true
  rm -f chain.usage
  env "${envs[@]}" "${vk[@]}" COLI_USAGE=chain.usage COLI_VK_DENSE_HOST=0 $dv=vk.f32 ./"$eng" "$@" > vk.log 2>&1 || true
  rm -f chain.usage
  same_tokens cpu.log vk.log "$tag"
  same_tokens cpu.log host.log "$tag (host copies)"
  local n; n=$(dho_dropped vk.log)
  [ "$n" -gt 0 ] || { cat vk.log; fail "$tag: no dense matrix went to the device only"; }
  [ "$(dho_reloaded vk.log)" = 0 ] || { grep '^\[VK\]' vk.log; fail "$tag: a matrix was read back from disk without a lost device"; }
  if [ "$mode" != 0 ]; then [ "$(chain_count "$eng" vk.log)" -gt 0 ] || { cat vk.log; fail "$tag: the chain did not stay on"; }; fi
  local lg=""
  case $tol in
    2) cmp -s host.f32 vk.f32 || fail "$tag: the logits differ from the run that keeps its host copies"; lg=", logits = the host-copy run's" ;;
    1) lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag: logits"; }; lg=", $lg" ;;
  esac
  echo "OK $tag: tokens = CPU$lg, $n matrices on the device only, $(chain_count "$eng" vk.log) chain forwards"
}
# dho_lost <engine> <tag> <frames back> <env...> -- <argv...>: device only, the device lost
# `back` frames before the end of a run (COLI_VK_CHAIN_FAULT, counted on a probe run): the
# CPU's tokens, and the matrices the CPU then needed read back from disk.
dho_lost() {
  local eng=$1 tag=$2 back=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local vk=(COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0)
  rm -f chain.usage
  env "${envs[@]}" ./"$eng" "$@" > cpu.log 2>&1 || true
  env "${envs[@]}" "${vk[@]}" COLI_USAGE=chain.usage ./"$eng" "$@" > vk.log 2>&1 || true
  local f; f=$(sed -n "s/^\[VK\] $eng chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p" vk.log | tail -1)
  { [ -n "$f" ] && [ "$f" -gt "$back" ]; } || { cat vk.log; fail "$tag: the probe run reports no frames"; }
  rm -f chain.usage
  env "${envs[@]}" "${vk[@]}" COLI_VK_CHAIN_FAULT=$((f - back)) COLI_USAGE=chain.usage ./"$eng" "$@" > vk.log 2>&1 || true
  rm -f chain.usage
  same_tokens cpu.log vk.log "$tag"
  local r; r=$(dho_reloaded vk.log)
  [ "$r" -gt 0 ] || { grep '^\[VK\]' vk.log; fail "$tag: nothing was read back for the CPU"; }
  echo "OK $tag: tokens = CPU after the loss, $r matrices read back from disk"
}
# dho_rss_gate <engine> <tag> <MiB at least> <env...> -- <argv...>: the resident set at
# exit falls by most of what was dropped (the device copies are host memory on a CPU
# device, the host copies are not there twice): RSS(host copies) - RSS(device only) is at
# least the given MiB.
dho_rss_gate() {
  local eng=$1 tag=$2 mib=$3; shift 3
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local vk=(COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1})
  rm -f chain.usage
  env "${envs[@]}" "${vk[@]}" COLI_USAGE=chain.usage COLI_VK_DENSE_HOST=1 ./"$eng" "$@" > host.log 2>&1 || true
  rm -f chain.usage
  env "${envs[@]}" "${vk[@]}" COLI_USAGE=chain.usage COLI_VK_DENSE_HOST=0 ./"$eng" "$@" > vk.log 2>&1 || true
  rm -f chain.usage
  local a b; a=$(dho_rss host.log); b=$(dho_rss vk.log)
  { [ -n "$a" ] && [ -n "$b" ]; } || { grep '^\[VK\]' host.log vk.log; fail "$tag: no exit line with the resident set"; }
  $PY - "$a" "$b" "$mib" <<'PY' || { grep '^\[VK\].*dense' host.log vk.log; fail "$tag: the resident set did not fall"; }
import sys
a, b, mib = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
print(f"RSS at exit {a:.2f} GiB with the host copies, {b:.2f} GiB on the device only")
sys.exit(0 if (a - b) * 1024 >= mib else 1)
PY
  echo "OK $tag: $(grep -o 'given back.*' vk.log | head -1)"
}
for f in tests/vulkan_dense_only_*.sh; do [ -e "$f" ] && . "$f"; done

# ---- the partial chain (docs/vulkan.md, "A partial chain"): the first N layers on the
# device, the rest and the head on the CPU. One file per engine group,
# tests/vulkan_partial_<group>.sh, each defining ptl_family_<group> and
# ptl_family_<group>_sanitize (run as partial-<group> and partial-<group>-sanitize);
# tests/vulkan_partial_deepseek.sh is the template. The helpers read the engine's lines:
#   [VK] <engine> chain fit: free F B, reserve R B, fixed X B (...), tail T B, layers b0 .. B, matrices m0 .. B
#   [VK] <engine> chain: N of L layers on the device ...        (the last one is the final N)
#   [VK] <engine> chain: N of L layers placed: M B of matrices on the device (the fit counted M' B ...), ...
#
# ptl_n <engine> <log>: the final N (-1 without a line)
ptl_n() {
  local n; n=$(sed -n "s/^\[VK\] $1 chain: \([0-9][0-9]*\) of [0-9][0-9]* layers on the device.*/\1/p" "$2" | tail -1)
  echo "${n:--1}"
}
# ptl_L <engine> <log>: L, the model's layers, from the same line
ptl_L() {
  local n; n=$(sed -n "s/^\[VK\] $1 chain: [0-9][0-9]* of \([0-9][0-9]*\) layers on the device.*/\1/p" "$2" | tail -1)
  echo "${n:--1}"
}
# ptl_calc <what> <engine> <log> [k]: arithmetic on the fit line (a probe run's for "cap")
#   predict  N as vkc_fit's rule gives it from the line's numbers
#   cap      the COLI_VK_DEVICE_CAP_MB (MiB, a fraction) under which exactly k layers fit:
#            the bytes the probe held at the fit (its cap PTL_PROBE_CAP_MB less its free),
#            the reserve, the engine's fixed bytes, the pools' granularity at the new cap,
#            k layers, and half of layer k (k < L) or the tail too (k = L). Probe with the
#            same settings and a cap of 256 or less: up to 256 MiB the pools' blocks (and so
#            what is held at the fit) do not depend on the cap
#   fault    the COLI_VK_STAGED_FAULT count to aim inside layer k's setup: the second time
#            the point is reached within it (the placed line's marks; a first one there may
#            be a fresh block's zero fill, which only skips the fill)
#   matrices M and M' of the placed line, "M M'"
ptl_calc() {
  $PY - "$@" <<'PY'
import os, re, sys
what, eng, log = sys.argv[1], sys.argv[2], sys.argv[3]
k = int(sys.argv[4]) if len(sys.argv) > 4 else 0
lines = open(log, errors="replace").read().splitlines()
fit = [l for l in lines if l.startswith(f"[VK] {eng} chain fit: ")]
if what in ("predict", "cap"):
    if not fit: sys.exit("no fit line")
    f = fit[-1]
    num = lambda key: int(re.search(key + r" (\d+) B", f)[1])
    free, res, fixed, tail = num("free"), num("reserve"), num("fixed"), num("tail")
    layers = [int(x) for x in re.search(r"layers((?: \d+)*) B", f)[1].split()]
    L, room = len(layers), max(0, free - res)
    if what == "predict":
        if fixed + sum(layers) <= room: print(L); sys.exit(0)
        n, acc = 0, fixed
        while n < L and acc + layers[n] <= room: acc += layers[n]; n += 1
        print(n); sys.exit(0)
    # the pools' part of fixed depends on the cap (coli_vk_block_bytes): recompute it
    def blk(cap, d):
        b = 64 << 10
        while b < cap // 4096: b <<= 1
        return min(b, d)
    pools = lambda cap: blk(cap, 256 << 20) + 3 * blk(cap, 64 << 20) + 4 * (4 << 20)
    engine = int(re.search(r"the engine's (\d+) B", f)[1])
    held = int(float(os.environ["PTL_PROBE_CAP_MB"]) * 1048576) - free
    cap = held + free
    for _ in range(4):
        cap = held + res + engine + pools(cap) + sum(layers[:k]) + (layers[k] // 2 if k < L else tail + 4096)
    print(f"{cap / 1048576:.6f}")
elif what == "fault":
    p = [l for l in lines if re.match(rf"\[VK\] {eng} chain: \d+ of \d+ layers placed", l)]
    m = re.search(r"reached ([\d,]+) times", p[-1] if p else "")
    if not m: sys.exit("no marks on the placed line")
    c = [int(x) for x in m[1].split(",")]
    lo = c[k - 1] if k else 0
    if c[k] - lo < 2: sys.exit(f"layer {k} reaches the point {c[k] - lo} times")
    print(lo + 2)
elif what == "matrices":
    p = [l for l in lines if re.match(rf"\[VK\] {eng} chain: \d+ of \d+ layers placed", l)]
    m = re.search(r"placed: (\d+) B of matrices on the device \(the fit counted (\d+) B", p[-1] if p else "")
    if not m: sys.exit("no placed line")
    print(m[1], m[2])
PY
}
# ptl_check_n <engine> <log> <want> <tag>: the final N is want, and the line says L
ptl_check_n() {
  local n; n=$(ptl_n "$1" "$2")
  [ "$n" = "$3" ] || { grep -E "^\[VK\] $1 chain" "$2"; fail "$4: N is $n, not $3"; }
}
# ptl_check_placed <engine> <log> <tag>: the matrices on the device after setup are the N
# layers' as the fit counted them, nothing of a layer that did not complete
ptl_check_placed() {
  local mm; mm=$(ptl_calc matrices "$1" "$2") || { grep -E "^\[VK\] $1 chain" "$2"; fail "$3: no placed line"; }
  set -- $mm "$3"
  [ "$1" = "$2" ] || fail "$3: $1 B of matrices on the device after setup, the N layers' are $2 B"
}
for f in tests/vulkan_partial_*.sh; do [ -e "$f" ] && . "$f"; done
. tests/vulkan_dev2.sh   # the expert tier on two devices
# ---- big prompt chunks and expert streaming (docs/vulkan.md, "Big prompt chunks and
# expert streaming") ----
# Every engine with the chain, on a prompt longer than its usual block, against its CPU
# run with the same snapshot and settings:
#   default   the chunk from the budget (the whole prompt in one chunk here) and the cold
#             experts the rule streams: the CPU's tokens and logits, the chunk's line,
#             and one tier step of the whole prompt;
#   forced    a small chunk cap and a forced streaming threshold (PF_FORCE: chunks of
#             PF_ROWS rows, a tier of a few experts, cold experts with 3 rows or more
#             streamed through four slots in sub-batches of 24 rows, so experts are cut
#             into parts): the CPU's tokens and logits, streamed experts counted, and the
#             later chunks' prefetch from the earlier ones' routing (PF_PREFETCH=1);
#   staged    the forced configuration with COLI_VK_STAGED=1 (the staging slots filled
#             through host images): the same, nothing resident left in host memory;
#   off       COLI_VK_CHAIN_ROWS=512 COLI_VK_TIER_STREAM=0 COLI_VK_ATTN_BLOCK=0, the
#             build before's behaviour: the CPU's tokens, nothing streamed.
# <family>-sanitize runs the forced configurations under ASan and UBSan.
PF_FORCE="COLI_VK_TIER_STREAM_ROWS=3 COLI_VK_TIER_STREAM_SLOTS=4 COLI_VK_TIER_STREAM_HALF=24"
PF_OFF="COLI_VK_CHAIN_ROWS=512 COLI_VK_TIER_STREAM=0 COLI_VK_ATTN_BLOCK=0"
stream_count() {   # <engine> <log>: cold experts streamed in the run (the last report)
  local n; n=$(sed -n "s/^\[VK\] tier $1 run: .* stream: [0-9]* steps, \([0-9]*\) cold experts.*/\1/p" "$2" | tail -1)
  echo "${n:-0}"
}
prefetch_count() {
  local n; n=$(sed -n "s/^\[VK\] tier $1 run: .* prefetched \([0-9]*\) (\([0-9]*\) used).*/\2/p" "$2" | tail -1)
  echo "${n:-0}"
}
# pf_check <engine> <log> <tag> <kind> [rows]: what the run must show besides the CPU's tokens
pf_check() {
  local eng=$1 log=$2 tag=$3 kind=$4 rows=${5:-}
  case $kind in
    default)
      grep -q "^\[VK\] $eng chain: prompt chunks of up to 8192 rows" "$log" || { grep '\[VK\]' "$log"; fail "$tag: the chunk is not the budget's"; }
      if [ -n "$rows" ]; then grep -q "^\[VK\] tier $eng stream: a step of $rows rows" "$log" ||
        { grep '\[VK\] tier' "$log"; fail "$tag: no tier step of the whole prompt ($rows rows)"; }; fi ;;
    forced|staged)
      [ "$(stream_count "$eng" "$log")" -gt 0 ] || { grep '\[VK\] tier' "$log"; fail "$tag: nothing streamed"; }
      if [ "${PF_PREFETCH:-0}" = 1 ]; then [ "$(prefetch_count "$eng" "$log")" -gt 0 ] ||
        { grep '\[VK\] tier' "$log"; fail "$tag: no prefetched expert was used"; }; fi
      if [ $kind = staged ]; then staged_check "$log" "$tag"; fi ;;
    off)
      ! grep -q "^\[VK\] tier $eng stream:" "$log" || { grep '\[VK\] tier' "$log"; fail "$tag: streamed with the feature off"; } ;;
  esac
  echo "   $tag: $(grep -o 'stream: [0-9]* steps, [0-9]* cold experts[^|]*' "$log" | tail -1 | cut -c1-160)"
}
# pf_runs <gate> <engine> <tag> <log> <rows> <budget env> <env...>: the four runs above.
# <gate> is a command line (eval'd: its quotes hold) taking <tag> <tol> <env...> after it.
pf_runs() {
  local g=$1 eng=$2 tag=$3 log=$4 rows=$5 bud=$6; shift 6
  local rw="COLI_VK_CHAIN_ROWS=${PF_ROWS:-64}"
  eval "$g"' "pf $tag default" 1 "$@"'; pf_check "$eng" "$log" "pf $tag default" default "$rows"
  eval "$g"' "pf $tag forced" 1 $PF_FORCE $bud $rw "$@"'; pf_check "$eng" "$log" "pf $tag forced" forced
  eval "$g"' "pf $tag staged" 1 $PF_FORCE $bud $rw COLI_VK_STAGED=1 "$@"'; pf_check "$eng" "$log" "pf $tag staged" staged
  eval "$g"' "pf $tag off" 1 $PF_OFF "$@"'; pf_check "$eng" "$log" "pf $tag off" off
}
# the engines' gates in pf_runs' calling convention: <fixed args...> <tag> <tol> <env...>
pf_chain() {  # <engine> <argv as one string> <tag> <tol> <env...>
  local eng=$1 argv=$2 tag=$3 tol=$4; shift 4
  # shellcheck disable=SC2086
  chain_gate "$eng" "$tag" "$tol" "$@" -- $argv
}
pf_mla() {  # <engine> <argv as one string> <tag> <tol> <env...>
  local eng=$1 argv=$2 tag=$3 tol=$4; shift 4
  # shellcheck disable=SC2086
  mla_gate "$eng" "$tag" "$tol" "$@" -- $argv
}
pf_ids() { $PY -c "print(','.join(str($2 + (i * 37) % $3) for i in range($1)))"; }
# a long prompt's ref.json for the engines whose oracle reads one: the ids and a few more
pf_ref() {  # <out> <n> <first id> <span>
  $PY - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
out, n, lo, span = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
p = [lo + (i * 37) % span for i in range(n)]
full = p + [lo] * 8
json.dump({"prompt_ids": p, "full_ids": full, "tf_pred": full[1:] + [lo]}, open(out, "w"))
PY
}
# MiMo on a prompt of ids (its cases are at most 72 tokens): tokens with --ngen 6, every
# position's logits with --ngen 0, against the CPU
pf_mimo() {  # <ids> <tag> <tol> <env...>
  local ids=$1 tag=$2; shift 3
  env "$@" COLI_TEMP=0 ./mimo mimo_tiny --ids "$ids" --ngen 6 > mimo-cpu.txt 2>/dev/null
  env "$@" COLI_TEMP=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 ./mimo mimo_tiny --ids "$ids" --ngen 6 > mimo-vk.txt 2> mimo-vk.err
  [ -s mimo-cpu.txt ] && cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-cpu.txt mimo-vk.txt mimo-vk.err; fail "$tag: tokens differ from the CPU"; }
  rm -f cpu.f32 vk.f32
  env "$@" MIMO_LOGITS=cpu.f32 ./mimo mimo_tiny --ids "$ids" --ngen 0 > /dev/null 2>&1
  env "$@" MIMO_LOGITS=vk.f32 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 ./mimo mimo_tiny --ids "$ids" --ngen 0 > /dev/null 2> mimo-vk2.err
  local lg; lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag: logits"; }
  echo "OK $tag: tokens = CPU, $lg, $(chain_count mimo mimo-vk.err) chain forwards"
}
pf_k3() {  # <ids> <tag> <tol> <env...>: k3c_gate on one prompt of ids
  local ids=$1 tag=$2; shift 3
  K3C_IDS="$ids" K3C_CASES=ids k3c_gate "$tag" "$@"
}
pf_v41() {  # <snap> <tag> <tol> <env...>
  local snap=$1 tag=$2; shift 3
  v41_gate "$tag" SNAP="$snap" "$@" -- 2 "$snap"/ref.json
}
pf_v4() {  # <fixture> <case> <tag> <tol> <env...>
  local fx=$1 c=$2 tag=$3; shift 4
  v4_chain_gate "$tag" "$fx" "$c" "$@"
}

family_prefill_qwen() {
  make qwen36 qwen38 tests/test_vk_chain tests/test_vk_tier VK=1
  ./tests/test_vk_tier shaders/qmatmul.spv | tee vk_tier.log
  tail -1 vk_tier.log | grep -qx PASS || fail "the tier (its streaming section)"
  ./tests/test_vk_chain shaders/qmatmul.spv | tee vk_chain.log
  tail -1 vk_chain.log | grep -qx PASS || fail "the chain's ops (the blocked attention)"
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny64 --ref-mode full --inter 64 --emit-ref qwen36_tiny64/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny64 --out qwen36_tiny64_c --ebits 4 --gs 64
  pf_ref qwen36_tiny/ref_long.json 300 1 250
  local ids; ids=$(pf_ids 110 1 60)
  $PY tools/make_qwen38_tiny.py --out qwen38_long_int4 --prompt-ids "$ids" --fp8-experts --int4-experts --expert-gain 3
  $PY tools/make_qwen38_tiny.py --out qwen38_long --prompt-ids "$ids"
  export OMP_NUM_THREADS=2
  pf_runs "pf_chain qwen36 '8 8 qwen36_tiny/ref_long.json'" qwen36 "qwen36 int8 experts" vk.log 300 COLI_VK_TIER_GB=0.00015 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c
  pf_runs "pf_chain qwen36 '2 4 qwen36_tiny/ref_long.json'" qwen36 "qwen36 planar int4-g64, cap 2" vk.log 300 COLI_VK_TIER_GB=0.0003 QWEN_EXPERT_ACT=f32 COLI_DENSE_I8=0 SNAP=qwen36_tiny64_c
  pf_runs "pf_chain qwen38 '4 8 qwen38_long/ref.json'" qwen38 "qwen38 bf16" vk.log 110 COLI_VK_TIER_GB=0.00005 SNAP=qwen38_long
  pf_runs "pf_chain qwen38 '2 8 qwen38_long_int4/ref_int4.json'" qwen38 "qwen38 int4 sidecar" vk.log 110 COLI_VK_TIER_GB=0.00005 SNAP=qwen38_long_int4
  PF_PREFETCH=1 chain_gate qwen36 "pf qwen36 prefetch across chunks" 1 $PF_FORCE COLI_VK_TIER_GB=0.00015 COLI_VK_CHAIN_ROWS=100 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 qwen36_tiny/ref_long.json
  PF_PREFETCH=1 pf_check qwen36 vk.log "pf qwen36 prefetch across chunks" forced
  PF_PREFETCH=1 chain_gate qwen38 "pf qwen38 prefetch across chunks" 1 $PF_FORCE COLI_VK_TIER_GB=0.00005 COLI_VK_CHAIN_ROWS=40 SNAP=qwen38_long_int4 -- 2 8 qwen38_long_int4/ref_int4.json
  PF_PREFETCH=1 pf_check qwen38 vk.log "pf qwen38 prefetch across chunks" forced
  # the CPU's prefill batching off (Q38_PREFILL_BATCH=0): the chain's MoE step still whole
  chain_gate qwen38 "pf qwen38 batch=0" 1 Q38_PREFILL_BATCH=0 COLI_VK_TIER_GB=0.00005 COLI_VK_TIER_STREAM_ROWS=3 SNAP=qwen38_long -- 4 8 qwen38_long/ref.json
  # the device lost while a chunk's streamed sub-batches are in flight: the state rebuilt
  # on the CPU, the run finishes there
  lost_late qwen36 "pf qwen36 device lost in a streamed prompt" 60 rebuild $PF_FORCE COLI_VK_TIER_GB=0.00015 COLI_VK_CHAIN_ROWS=64 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 qwen36_tiny/ref_long.json
  unset OMP_NUM_THREADS
}
family_prefill_qwen_sanitize() {
  make clean >/dev/null 2>&1 || true
  make qwen36 qwen38 tests/test_vk_tier VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  ./tests/test_vk_tier shaders/qmatmul.spv > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log || ! tail -1 san.log | grep -qx PASS; then cat san.log; fail "asan: the tier"; fi
  echo "OK asan: the tier, its streaming section included"
  COLI_VK_STAGED=1 ./tests/test_vk_tier shaders/qmatmul.spv > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log || ! tail -1 san.log | grep -qx PASS; then cat san.log; fail "asan: the tier staged"; fi
  echo "OK asan: the tier staged"
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8
  pf_ref qwen36_tiny/ref_long.json 300 1 250
  local ids; ids=$(pf_ids 110 1 60)
  $PY tools/make_qwen38_tiny.py --out qwen38_long_int4 --prompt-ids "$ids" --fp8-experts --int4-experts --expert-gain 3
  export OMP_NUM_THREADS=2
  pfsan qwen36 "asan pf qwen36 forced" COLI_VK_TIER_GB=0.00015 COLI_VK_CHAIN_ROWS=64 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 qwen36_tiny/ref_long.json
  pfsan qwen36 "asan pf qwen36 forced, staged" COLI_VK_STAGED=1 COLI_VK_TIER_GB=0.00015 COLI_VK_CHAIN_ROWS=64 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c ./qwen36 8 8 qwen36_tiny/ref_long.json
  pfsan qwen38 "asan pf qwen38 forced" COLI_VK_TIER_GB=0.00005 COLI_VK_CHAIN_ROWS=40 SNAP=qwen38_long_int4 ./qwen38 2 8 qwen38_long_int4/ref_int4.json
  unset OMP_NUM_THREADS
  make clean >/dev/null 2>&1 || true
}
# pfsan <engine> <tag> <env and argv...>: the forced configuration under the sanitizers
pfsan() {
  local eng=$1 tag=$2; shift 2
  rm -f chain.usage
  # shellcheck disable=SC2086
  env $PF_FORCE COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 "$@" > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
  [ "$(chain_count "$eng" san.log)" -gt 0 ] || { cat san.log; fail "$tag: the chain never ran"; }
  [ "$(stream_count "$eng" san.log)" -gt 0 ] || { grep '\[VK\]' san.log; fail "$tag: nothing streamed"; }
  echo "OK $tag: sanitizers clean, $(chain_count "$eng" san.log) chain forwards, $(stream_count "$eng" san.log) experts streamed"
}

family_prefill_inkling_olmoe() {
  export OMP_NUM_THREADS=2
  make inkling olmoe VK=1
  $PY tools/make_tiny_inkling.py tiny_inkling
  $PY tools/make_olmoe_tiny.py --output olmoe_tiny --force
  $PY tools/convert_olmoe_merged.py --model olmoe_tiny --out olmoe_tiny_c
  pf_ref tiny_inkling/ref_long.json 200 1 250
  pf_ref olmoe_tiny/ref_long.json 200 1 120
  pf_runs "pf_chain inkling '8 0 tiny_inkling/ref_long.json'" inkling "inkling f32" vk.log 200 COLI_VK_TIER_GB=0.0001 SNAP=tiny_inkling
  pf_runs "pf_chain inkling '2 8 tiny_inkling/ref_long.json'" inkling "inkling runtime int8, cap 2" vk.log 200 COLI_VK_TIER_GB=0.0001 SNAP=tiny_inkling
  pf_runs "pf_chain olmoe '8 8 olmoe_tiny/ref_long.json'" olmoe "olmoe" vk.log 200 COLI_VK_TIER_GB=0.0001 SNAP=olmoe_tiny_c
  pf_runs "pf_chain olmoe '1 4 olmoe_tiny/ref_long.json'" olmoe "olmoe 4-bit experts, cap 1" vk.log 200 COLI_VK_TIER_GB=0.0001 SNAP=olmoe_tiny_c
  PF_PREFETCH=1 chain_gate olmoe "pf olmoe prefetch across chunks" 1 $PF_FORCE COLI_VK_TIER_GB=0.0001 COLI_VK_CHAIN_ROWS=64 SNAP=olmoe_tiny_c -- 8 8 olmoe_tiny/ref_long.json
  PF_PREFETCH=1 pf_check olmoe vk.log "pf olmoe prefetch across chunks" forced
  chain_gate olmoe "pf olmoe PILOT" 1 PILOT=1 WIDE=2 $PF_FORCE COLI_VK_TIER_GB=0.0001 SNAP=olmoe_tiny_c -- 2 8 olmoe_tiny/ref_long.json
  pf_check olmoe vk.log "pf olmoe PILOT" forced
  # sanitizers
  make clean >/dev/null 2>&1 || true
  make inkling olmoe VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  pfsan inkling "asan pf inkling forced" COLI_VK_TIER_GB=0.0001 COLI_VK_CHAIN_ROWS=64 SNAP=tiny_inkling ./inkling 2 8 tiny_inkling/ref_long.json
  pfsan olmoe "asan pf olmoe forced, staged" COLI_VK_STAGED=1 COLI_VK_TIER_GB=0.0001 COLI_VK_CHAIN_ROWS=64 SNAP=olmoe_tiny_c ./olmoe 1 8 olmoe_tiny/ref_long.json
  unset OMP_NUM_THREADS
  make clean >/dev/null 2>&1 || true
}

family_prefill_mimo_kimi() {
  export OMP_NUM_THREADS=2
  make mimo kimi_k3 VK=1
  $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force --vision
  $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  local mids kids
  mids=$(pf_ids 200 1 300 | tr ',' ' '); kids=$(pf_ids 200 1 300 | tr ',' ' ')
  pf_runs "pf_mimo '$mids'" mimo "mimo f32" mimo-vk.err 200 MIMO_VK_EXPERTS=6 MIMO_DENSE_BITS=32
  pf_runs "pf_mimo '$mids'" mimo "mimo fp8/bf16" mimo-vk.err 200 MIMO_VK_EXPERTS=6 MIMO_DENSE_BITS=0
  local O="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0"
  # shellcheck disable=SC2086
  pf_runs "pf_k3 '$kids'" kimi_k3 "kimi_k3 f32" vk.log 200 COLI_VK_TIER_GB=0.0005 $O
  PF_PREFETCH=1 pf_mimo "$mids" "pf mimo prefetch across chunks" 1 $PF_FORCE MIMO_VK_EXPERTS=6 MIMO_DENSE_BITS=32 COLI_VK_CHAIN_ROWS=50
  PF_PREFETCH=1 pf_check mimo mimo-vk.err "pf mimo prefetch across chunks" forced
  # shellcheck disable=SC2086
  PF_PREFETCH=1 pf_k3 "$kids" "pf kimi_k3 prefetch across chunks" 1 $PF_FORCE COLI_VK_TIER_GB=0.0005 COLI_VK_CHAIN_ROWS=50 $O
  PF_PREFETCH=1 pf_check kimi_k3 vk.log "pf kimi_k3 prefetch across chunks" forced
  make clean >/dev/null 2>&1 || true
  make mimo kimi_k3 VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  pfsan mimo "asan pf mimo forced" MIMO_VK_EXPERTS=6 MIMO_DENSE_BITS=32 COLI_VK_CHAIN_ROWS=50 COLI_TEMP=0 ./mimo mimo_tiny --ids "$mids" --ngen 4
  # shellcheck disable=SC2086
  pfsan kimi_k3 "asan pf kimi_k3 forced, staged" COLI_VK_STAGED=1 COLI_VK_TIER_GB=0.0005 COLI_VK_CHAIN_ROWS=50 $O ./kimi_k3 kimi_k3_tiny --ids "$kids" --ngen 4
  unset OMP_NUM_THREADS
  make clean >/dev/null 2>&1 || true
}

family_prefill_glm() {
  export OMP_NUM_THREADS=2 CAP_RAISE=0
  make colibri glm53 VK=1
  glm_fixtures
  pf_ref glm_tiny/ref_long.json 200 1 250
  local ids; ids=$(pf_ids 200 2 120)
  pf_runs "pf_mla colibri '64 16 16'" colibri "colibri f32, teacher-forced" vk.log 208 COLI_VK_TIER_GB=0.0003 SNAP=glm_tiny REF=glm_tiny/ref_long.json TF=1
  pf_runs "pf_mla colibri '2 16 16'" colibri "colibri int4-g64 experts" vk.log 200 COLI_VK_TIER_GB=0.0012 SNAP=glm_tiny_fmt4 REF=glm_tiny/ref_long.json
  pf_runs "pf_mla glm53 '--model glm53_stream-i4 --ids $ids --greedy 4'" glm53 "glm53 f32 trunk" vk.log 200 COLI_VK_TIER_GB=0.0002 GLM53_BITS=32
  pf_runs "pf_mla glm53 '--model glm53_stream-i4 --ids $ids --greedy 4'" glm53 "glm53 int4 trunk" vk.log 200 COLI_VK_TIER_GB=0.0002 GLM53_BITS=4
  PF_PREFETCH=1 mla_gate colibri "pf colibri prefetch across chunks" 1 $PF_FORCE COLI_VK_TIER_GB=0.0003 COLI_VK_CHAIN_ROWS=64 SNAP=glm_tiny REF=glm_tiny/ref_long.json TF=1 -- 2 16 16
  PF_PREFETCH=1 pf_check colibri vk.log "pf colibri prefetch across chunks" forced
  PF_PREFETCH=1 mla_gate glm53 "pf glm53 prefetch across chunks" 1 $PF_FORCE COLI_VK_TIER_GB=0.0002 COLI_VK_CHAIN_ROWS=64 GLM53_BITS=4 GLM53_PREFILL_CHUNK=200 -- --model glm53_stream-i4 --ids "$ids" --greedy 4
  PF_PREFETCH=1 pf_check glm53 vk.log "pf glm53 prefetch across chunks" forced
  make clean >/dev/null 2>&1 || true
  make colibri glm53 VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  pfsan colibri "asan pf colibri forced" COLI_VK_TIER_GB=0.0003 COLI_VK_CHAIN_ROWS=64 SNAP=glm_tiny REF=glm_tiny/ref_long.json TF=1 ./colibri 2 16 16
  pfsan glm53 "asan pf glm53 forced, staged" COLI_VK_STAGED=1 COLI_VK_TIER_GB=0.0002 COLI_VK_CHAIN_ROWS=64 GLM53_BITS=4 ./glm53 --model glm53_stream-i4 --ids "$ids" --greedy 4
  unset OMP_NUM_THREADS CAP_RAISE
  make clean >/dev/null 2>&1 || true
}

family_prefill_deepseek() {
  export OMP_NUM_THREADS=2
  make deepseek_v41 VK=1
  make deepseek-v4 VK=1
  $PY tools/make_dsv41_tiny.py --out dsv41_vlong --emit-ref dsv41_vlong/ref.json --prompt-len 160 --max-new 6 > /dev/null
  pf_v4_fixture
  local ids; ids=$(pf_ids 300 3 120)
  pf_runs "pf_v41 dsv41_vlong" deepseek_v41 "deepseek_v41" vk.log 160 COLI_VK_TIER_GB=0.0005
  pf_runs "pf_v4 deepseek_v4_tiny_p ids:$ids:4" deepseek_v4 "deepseek_v4" v4-vk.err 300 COLI_VK_TIER_GB=0.0005
  PF_PREFETCH=1 v41_gate "pf deepseek_v41 prefetch across chunks" SNAP=dsv41_vlong $PF_FORCE COLI_VK_TIER_GB=0.0005 COLI_VK_CHAIN_ROWS=64 -- 2 dsv41_vlong/ref.json
  PF_PREFETCH=1 pf_check deepseek_v41 vk.log "pf deepseek_v41 prefetch across chunks" forced
  PF_PREFETCH=1 v4_chain_gate "pf deepseek_v4 prefetch across chunks" deepseek_v4_tiny_p "ids:$ids:4" $PF_FORCE COLI_VK_TIER_GB=0.0005 COLI_VK_CHAIN_ROWS=100
  PF_PREFETCH=1 pf_check deepseek_v4 v4-vk.err "pf deepseek_v4 prefetch across chunks" forced
  make clean >/dev/null 2>&1 || true
  make deepseek-v4-clean >/dev/null 2>&1 || true
  local SAN="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  make deepseek_v41 VK=1 EXTRA_CFLAGS="$SAN"
  make deepseek-v4 VK=1 LTO=0 EXTRA_CFLAGS="$SAN" EXTRA_LDFLAGS="$SAN"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  pfsan deepseek_v41 "asan pf deepseek_v41 forced" COLI_VK_TIER_GB=0.0005 COLI_VK_CHAIN_ROWS=64 SNAP=dsv41_vlong ./deepseek_v41 2 dsv41_vlong/ref.json
  pfsan deepseek_v4 "asan pf deepseek_v4 forced, staged" COLI_VK_STAGED=1 COLI_VK_TIER_GB=0.0005 COLI_VK_CHAIN_ROWS=100 ./deepseek_v4 ./deepseek_v4_tiny_p \
    "$(v4_chain_prompt deepseek_v4_tiny_p "ids:$ids:4")" --raw-prompt --max-tokens 4
  unset OMP_NUM_THREADS
  make clean >/dev/null 2>&1 || true
  make deepseek-v4-clean >/dev/null 2>&1 || true
}
# DeepSeek V4's tiny model at 512 positions, 8 experts: room for a prompt past its 128-row block
pf_v4_fixture() {
  $PY - tools/make_deepseek_v4_tiny.py <<'PY' > /dev/null
import importlib.util, sys
spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
gen.EXPERTS = 8; gen.MAX_POSITIONS = 512
sys.argv = ["make_deepseek_v4_tiny.py", "--output", "deepseek_v4_tiny_p", "--force"]
gen.main()
PY
}
# ---- the KV cache split between the device and the host (vk_kvsplit.h) ----------------
# Past the device's budget each chain engine keeps a few blocks of every attention
# layer's cache on the device and attends over the rest on the CPU, the two parts merged
# through their softmax statistics. The tiny fixtures emulate a small device with
# COLI_VK_KV_DEVICE_ROWS (rows a layer keeps) and COLI_VK_KV_BLOCK (positions a block),
# over prompts and generations long enough that most steps have a host part.
# kv_hostparts <engine> <log>: M from "[VK] <engine> chain: KV split: N layer steps, M with a host part"
kv_hostparts() {
  local n
  n=$(sed -n "s/^\[VK\] $1 chain: KV split: [0-9]* layer steps, \([0-9]*\) with a host part.*/\1/p" "$2" | tail -1)
  echo "${n:-0}"
}
# kv_gate <engine> <tag> <tol 0|1> <rows> <block> <env...> -- <argv...>: chain_gate (the
# CPU's tokens, logits within 1e-4 of the largest with tol 1) with the split on, which
# must have run steps with a host part.
kv_gate() {
  local eng=$1 tag=$2 tol=$3 rows=$4 blk=$5; shift 5
  chain_gate "$eng" "$tag" "$tol" COLI_VK_KV_DEVICE_ROWS=$rows COLI_VK_KV_BLOCK=$blk "$@"
  [ "$(kv_hostparts "$eng" vk.log)" -gt 0 ] || { cat vk.log; fail "$tag: the split never ran a host part"; }
  echo "   $tag: $(grep -a "KV split:" vk.log | tail -1 | sed 's/.*KV split: //')"
}
# kv_san <engine> <tag> <env and argv...>: a sanitized run under the split (the build is
# the caller's): no diagnostic, and steps with a host part
kv_san() {
  local eng=$1 tag=$2; shift 2
  rm -f chain.usage
  env OMP_NUM_THREADS=2 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 "$@" > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
  [ "$(kv_hostparts "$eng" san.log)" -gt 0 ] || { cat san.log; fail "$tag: the split never ran a host part"; }
  echo "OK $tag: sanitizers clean, $(kv_hostparts "$eng" san.log) layer steps with a host part"
}

# qwen36: the hybrid and the dense geometry, a 40-token prompt and 40 new tokens
kv_qwen36_fixtures() {
  local P; P=$($PY -c "print(','.join(str(3 + i * 7 % 60) for i in range(40)))")
  $PY tools/make_qwen36_tiny.py --out qwen36_kv --ref-mode full --prompt-ids "$P" --max-new 40 --emit-ref qwen36_kv/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_kv --out qwen36_kv_c --ebits 8
  $PY tools/make_qwen36_tiny.py --geometry qwen38-27b-dense --out qwen38_27b_kv --ref-mode full --prompt-ids "$P" --max-new 40 \
    --emit-ref qwen38_27b_kv/ref_full.json
  $PY tools/convert_qwen36.py --model qwen38_27b_kv --out qwen38_27b_kv_c --ebits 8
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8
}
kv_split_qwen36() {
  kv_qwen36_fixtures
  local R=qwen36_kv/ref_full.json
  kv_gate qwen36 "kv qwen36 16 rows on the device" 1 16 4 COLI_DENSE_I8=0 SNAP=qwen36_kv_c -- 8 8 $R
  kv_gate qwen36 "kv qwen36 two blocks on the device" 1 8 4 COLI_DENSE_I8=0 SNAP=qwen36_kv_c -- 8 8 $R
  kv_gate qwen36 "kv qwen36 prefill in chunks of 3" 1 16 4 COLI_VK_CHAIN_ROWS=3 COLI_DENSE_I8=0 SNAP=qwen36_kv_c -- 8 8 $R
  kv_gate qwen36 "kv qwen36 cap 1, blocks of 8" 1 24 8 COLI_DENSE_I8=0 SNAP=qwen36_kv_c -- 1 8 $R
  kv_gate qwen36 "kv qwen36 tier off, tiled GEMM" 1 24 4 COLI_VK_TIER=0 COLI_VK_GEMM_MIN_S=2 COLI_DENSE_I8=0 SNAP=qwen36_kv_c -- 8 8 $R
  CPUENV=COLI_DENSE_IDOT=0 kv_gate qwen36 "kv qwen36 int8 trunk" 1 16 4 SNAP=qwen36_kv_c -- 8 8 $R
  kv_gate qwen36 "kv qwen36 dense model" 1 16 4 COLI_DENSE_I8=0 SNAP=qwen38_27b_kv_c -- 8 8 qwen38_27b_kv/ref_full.json
  CHAINMODE=2 kv_gate qwen36 "kv qwen36 prompts only" 1 16 4 COLI_DENSE_I8=0 SNAP=qwen36_kv_c -- 8 8 $R
  # COLI_VK_KV_SPLIT=0: the whole mirrors, as before the split
  chain_gate qwen36 "kv qwen36 split off" 1 COLI_VK_KV_SPLIT=0 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 COLI_DENSE_I8=0 SNAP=qwen36_kv_c -- 8 8 $R
  if grep -qa 'KV cache split' vk.log; then cat vk.log; fail "kv qwen36 split off: the split ran"; fi
  # the device lost mid-decode with the split on: the DeltaNet state rebuilt on the CPU
  lost_late qwen36 "kv qwen36 device lost" 20 rebuild COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 COLI_DENSE_I8=0 SNAP=qwen36_kv_c -- 8 8 $R
  # serve sessions (pins, prompt-cache extensions, a divergent prompt, logprobs) and the
  # prefix-reuse contract across the split
  QWEN36_TINY=$PWD/qwen36_tiny_c COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 \
    COLI_USAGE=$PWD/chain.usage $PY tests/test_qwen36_prefix_serve.py
  CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' \
    $PY tests/vulkan_chain_serve.py ./qwen36 qwen36_tiny_c COLI_DENSE_I8=0 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4
}
kv_split_qwen36_sanitize() {
  kv_qwen36_fixtures
  kv_san qwen36 "asan kv qwen36" COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 COLI_DENSE_I8=0 SNAP=qwen36_kv_c ./qwen36 8 8 qwen36_kv/ref_full.json
  kv_san qwen36 "asan kv qwen36 chunks of 3" COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 COLI_VK_CHAIN_ROWS=3 SNAP=qwen36_kv_c \
    ./qwen36 8 8 qwen36_kv/ref_full.json
  kv_san qwen36 "asan kv qwen36 dense model" COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 COLI_DENSE_I8=0 SNAP=qwen38_27b_kv_c \
    ./qwen36 8 8 qwen38_27b_kv/ref_full.json
  $PY -c "import sys; sys.path.insert(0, 'tests'); from prefix_serve_harness import ensure_byte_tokenizer as t; from pathlib import Path; t(Path('qwen36_tiny_c'))"
  CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' \
    $PY tests/vulkan_chain_serve.py ./qwen36 qwen36_tiny_c COLI_DENSE_I8=0 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 > san.log 2>&1 || { cat san.log; fail "asan kv qwen36 serve"; }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan kv qwen36 serve: sanitizer diagnostic"; fi
  echo "OK asan kv qwen36 serve: $(tail -1 san.log)"
}

# qwen38: a 40-token prompt and 40 new tokens (the generation the transformers reference
# and the CPU agree on), with and without the MTP head; the serve fixture with a tokenizer
kv_qwen38_fixtures() {
  local P; P=$($PY -c "print(','.join(str(3 + (i * 9 + i * i * 2) % 60) for i in range(40)))")
  $PY tools/make_qwen38_tiny.py --out qwen38_kv --prompt-ids "$P" --max-new 40
  $PY tools/make_qwen38_tiny.py --out qwen38_kv_mtp --prompt-ids "$P" --max-new 40 --mtp
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_mtp --mtp
  $PY tools/make_edge_tiny_tokenizer.py --vocab-size 64 ./qwen38_tiny_mtp   # serve speaks text
}
kv_split_qwen38() {
  kv_qwen38_fixtures
  local R=qwen38_kv/ref.json M=qwen38_kv_mtp/ref.json batch f
  for batch in 0 1; do
    kv_gate qwen38 "kv qwen38 16 rows on the device, batch=$batch" 1 16 4 Q38_PREFILL_BATCH=$batch SNAP=qwen38_kv -- 4 8 $R
  done
  kv_gate qwen38 "kv qwen38 two blocks on the device" 1 8 4 SNAP=qwen38_kv -- 4 8 $R
  kv_gate qwen38 "kv qwen38 prefill in chunks of 3" 1 16 4 COLI_VK_CHAIN_ROWS=3 SNAP=qwen38_kv -- 4 8 $R
  # QSA's lists with blocks pinned by their reads (steps of 4 rows leave the window three
  # of six slots): the device's part and the host's take the listed positions
  COLI_VK_KV_PIN=1 kv_gate qwen38 "kv qwen38 blocks pinned by QSA's reads" 1 24 4 COLI_VK_CHAIN_ROWS=4 SNAP=qwen38_kv -- 4 8 $R
  grep -qa 'KV split: .* [1-9][0-9]* blocks pinned by reads' vk.log || { cat vk.log; fail "kv qwen38: no block was pinned"; }
  kv_gate qwen38 "kv qwen38 int8 trunk" 0 16 4 Q38_TRUNK_MIN_KB=0 SNAP=qwen38_kv -- 4 8 $R
  kv_gate qwen38 "kv qwen38 tier off" 1 16 4 COLI_VK_TIER=0 SNAP=qwen38_kv -- 4 8 $R
  # MTP verify (S = 2) and its rollback across the split: drafts as they come, every
  # draft rejected, every one accepted, alternating
  for f in "" reject accept mixed; do
    kv_gate qwen38 "kv qwen38 MTP ${f:-drafting}" 1 16 4 Q38_MTP=1 Q38_MTP_FORCE=$f SNAP=qwen38_kv_mtp -- 2 8 $M
  done
  CHAINMODE=2 kv_gate qwen38 "kv qwen38 prompts only" 1 16 4 SNAP=qwen38_kv -- 4 8 $R
  CHAINMODE=2 kv_gate qwen38 "kv qwen38 prompts only, MTP" 1 16 4 Q38_MTP=1 Q38_MTP_FORCE=mixed SNAP=qwen38_kv_mtp -- 2 8 $M
  chain_gate qwen38 "kv qwen38 split off" 1 COLI_VK_KV_SPLIT=0 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 SNAP=qwen38_kv -- 4 8 $R
  if grep -qa 'KV cache split' vk.log; then cat vk.log; fail "kv qwen38 split off: the split ran"; fi
  # the device lost with the split on, between steps and inside an MTP verify: the
  # DeltaNet, conv and PLE states rebuilt on the CPU from the prefix record
  lost_late qwen38 "kv qwen38 device lost" 10 rebuild COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 SNAP=qwen38_kv -- 4 8 $R
  lost_late qwen38 "kv qwen38 device lost in a verify" 15 rebuild COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 Q38_MTP=1 \
    Q38_MTP_FORCE=mixed SNAP=qwen38_kv_mtp -- 2 8 $M
  # serve sessions (pins, prompt-cache extensions, a divergent prompt, logprobs) with
  # and without MTP, and prompts only
  local E='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part'
  CHAIN_SERVE_EXPECT="$E" $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp Q38_MTP=1 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4
  CHAIN_SERVE_EXPECT="$E" $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4
  CHAIN_SERVE_EXPECT="$E" COLI_VK_CHAIN=2 $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp Q38_MTP=1 \
    COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4
}
kv_split_qwen38_sanitize() {
  kv_qwen38_fixtures
  kv_san qwen38 "asan kv qwen38" COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 SNAP=qwen38_kv ./qwen38 4 8 qwen38_kv/ref.json
  COLI_VK_KV_PIN=1 kv_san qwen38 "asan kv qwen38 pinned blocks, chunks of 4" COLI_VK_KV_DEVICE_ROWS=24 COLI_VK_KV_BLOCK=4 COLI_VK_CHAIN_ROWS=4 \
    SNAP=qwen38_kv ./qwen38 4 8 qwen38_kv/ref.json
  kv_san qwen38 "asan kv qwen38 MTP mixed" COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 Q38_MTP=1 Q38_MTP_FORCE=mixed \
    SNAP=qwen38_kv_mtp ./qwen38 2 8 qwen38_kv_mtp/ref.json
  CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' \
    $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp Q38_MTP=1 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 \
    > san.log 2>&1 || { cat san.log; fail "asan kv qwen38 serve"; }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan kv qwen38 serve: sanitizer diagnostic"; fi
  echo "OK asan kv qwen38 serve: $(tail -1 san.log)"
}

# olmoe: the tiny checkpoint and, from it, a 40-token prompt's 40-token generation by
# transformers (the oracle the engine's run checks)
kv_olmoe_fixtures() {
  $PY tools/make_olmoe_tiny.py --output olmoe_tiny --force
  $PY tools/convert_olmoe_merged.py --model olmoe_tiny --out olmoe_tiny_c
  $PY tools/make_edge_tiny_tokenizer.py --vocab-size 128 olmoe_tiny_c
  $PY - <<'PY'
import json, torch
from transformers import OlmoeForCausalLM
torch.set_num_threads(1)
m = OlmoeForCausalLM.from_pretrained("olmoe_tiny").eval().float()
prompt = [(17 + i * 21 + (i * i) % 13) % 125 + 2 for i in range(40)]
with torch.no_grad():
    g = m.generate(torch.tensor([prompt]), max_new_tokens=40, do_sample=False, use_cache=True)[0].tolist()
json.dump({"prompt_ids": prompt, "full_ids": g}, open("olmoe_tiny/ref_olmoe_long.json", "w"))
PY
}
kv_split_olmoe() {
  kv_olmoe_fixtures
  local R=olmoe_tiny/ref_olmoe_long.json
  kv_gate olmoe "kv olmoe 16 rows on the device" 1 16 4 SNAP=olmoe_tiny_c -- 8 8 $R
  kv_gate olmoe "kv olmoe two blocks on the device" 1 8 4 SNAP=olmoe_tiny_c -- 8 8 $R
  kv_gate olmoe "kv olmoe prefill in chunks of 3" 1 16 4 COLI_VK_CHAIN_ROWS=3 SNAP=olmoe_tiny_c -- 8 8 $R
  kv_gate olmoe "kv olmoe cap 1, blocks of 8" 1 24 8 SNAP=olmoe_tiny_c -- 1 8 $R
  kv_gate olmoe "kv olmoe 4-bit experts" 1 16 4 SNAP=olmoe_tiny_c -- 8 4 $R
  kv_gate olmoe "kv olmoe PILOT cap 2" 1 16 4 PILOT=1 WIDE=2 SNAP=olmoe_tiny_c -- 2 8 $R
  kv_gate olmoe "kv olmoe tier off, tiled GEMM" 1 24 4 COLI_VK_TIER=0 COLI_VK_GEMM_MIN_S=2 SNAP=olmoe_tiny_c -- 8 8 $R
  CHAINMODE=2 kv_gate olmoe "kv olmoe prompts only" 1 16 4 SNAP=olmoe_tiny_c -- 8 8 $R
  chain_gate olmoe "kv olmoe split off" 1 COLI_VK_KV_SPLIT=0 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 SNAP=olmoe_tiny_c -- 8 8 $R
  if grep -qa 'KV cache split' vk.log; then cat vk.log; fail "kv olmoe split off: the split ran"; fi
  # the device lost mid-decode with the split on: the CPU redoes the step (nothing to rebuild)
  lost_late olmoe "kv olmoe device lost" 7 redo COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 SNAP=olmoe_tiny_c -- 8 8 $R
  # serve sessions and the prefix-reuse and dashboard contracts across the split
  CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' \
    $PY tests/vulkan_chain_serve.py ./olmoe olmoe_tiny_c COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4
  CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' \
    $PY tests/vulkan_chain_serve.py ./olmoe olmoe_tiny_c PILOT=1 WIDE=2 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 \
    OLMOE_TINY=olmoe_tiny_c $PY -m unittest tests.test_olmoe_prefix_serve tests.test_olmoe_dashboard_hits
}
kv_split_olmoe_sanitize() {
  kv_olmoe_fixtures
  local R=olmoe_tiny/ref_olmoe_long.json
  kv_san olmoe "asan kv olmoe" COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 SNAP=olmoe_tiny_c ./olmoe 8 8 $R
  kv_san olmoe "asan kv olmoe chunks of 3, PILOT" COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 COLI_VK_CHAIN_ROWS=3 PILOT=1 WIDE=2 \
    SNAP=olmoe_tiny_c ./olmoe 2 8 $R
  CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' \
    $PY tests/vulkan_chain_serve.py ./olmoe olmoe_tiny_c COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 > san.log 2>&1 || { cat san.log; fail "asan kv olmoe serve"; }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan kv olmoe serve: sanitizer diagnostic"; fi
  echo "OK asan kv olmoe serve: $(tail -1 san.log)"
}

# inkling: the tiny model with a 40-token prompt and 40 new tokens (its global layer
# attends over all 80 positions, the bias bank reaching back 32), its expert containers,
# the dense-int4g64 container and the model at D = 6144
kv_inkling_fixtures() {
  $PY tools/make_tiny_inkling.py tiny_inkling
  $PY - tiny_inkling <<'PY'
import json, sys, torch
from transformers import InklingForCausalLM
d = sys.argv[1]
m = InklingForCausalLM.from_pretrained(d).eval().float()
prompt = [(3 + i * 7) % 250 for i in range(40)]
with torch.no_grad():
    full = m.generate(torch.tensor([prompt]), max_new_tokens=40, do_sample=False, use_cache=True)[0].tolist()
    tf = m(torch.tensor([full])).logits[0].argmax(-1).tolist()
json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf}, open(f"{d}/ref_long.json", "w"))
PY
  $PY - <<'PY'
import importlib.util, os, shutil
from safetensors.torch import load_file, save_file
spec = importlib.util.spec_from_file_location("conv", "tools/convert_inkling_int4.py")
conv = importlib.util.module_from_spec(spec); spec.loader.exec_module(conv)
t = load_file("tiny_inkling/model.safetensors")
if "model.embed_tokens.embed_norm.weight" in t:
    t["model.embed_norm.weight"] = t.pop("model.embed_tokens.embed_norm.weight")
for d in ("tiny_inkling_hf", "tiny_inkling_x-tml", "tiny_inkling_x-pass", "tiny_inkling_x-i4", "tiny_inkling_x-i8"):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs("tiny_inkling_hf")
save_file(t, "tiny_inkling_hf/model.safetensors")
shutil.copy("tiny_inkling/config.json", "tiny_inkling_hf/")
conv.selftest_e2e("tiny_inkling_hf", "tiny_inkling_x")          # tiny_inkling_x-i4: int4 experts
conv.convert_dir("tiny_inkling_x-tml", "tiny_inkling_x-i8", 8)   # int8 experts
PY
  rm -rf tiny_inkling_q && cp -r tiny_inkling tiny_inkling_q
  $PY - tools/convert_inkling_dense_int4.py tiny_inkling_q <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("conv", sys.argv[1])
conv = importlib.util.module_from_spec(spec); spec.loader.exec_module(conv)
conv.MIN_ELEMS = 0
conv.ATTN_BITS = 4
base = conv.classify
conv.classify = lambda n, s, d: "int8" if n.endswith(".mlp.down_proj.weight") else base(n, s, d)
sys.argv = ["convert_inkling_dense_int4.py", "--dir", sys.argv[2]]
conv.main()
PY
  mkdir -p tiny_inkling_q/dense-int4g64
  mv tiny_inkling_q/dense-int4g64.safetensors tiny_inkling_q/dense-int4g64/dense.safetensors
  $PY tools/make_tiny_inkling.py tiny_inkling_wide --wide
  $PY -c 'import sys; from pathlib import Path; sys.path.insert(0, "."); from tests.test_inkling_prefix_serve import ensure_tokenizer; ensure_tokenizer(Path("tiny_inkling"))'
}
kv_split_inkling() {
  kv_inkling_fixtures
  local R=tiny_inkling/ref_long.json fx
  kv_gate inkling "kv inkling 16 rows on the device" 1 16 4 SNAP=tiny_inkling -- 8 0 $R
  kv_gate inkling "kv inkling two blocks on the device" 1 8 4 SNAP=tiny_inkling -- 8 0 $R
  kv_gate inkling "kv inkling cap 1, blocks of 8" 1 24 8 SNAP=tiny_inkling -- 1 0 $R
  for fx in tiny_inkling_q tiny_inkling_x-i4 tiny_inkling_x-i8; do
    kv_gate inkling "kv inkling $fx" 1 16 4 SNAP=$fx -- 8 0 $R
  done
  kv_gate inkling "kv inkling runtime int4" 1 16 4 SNAP=tiny_inkling -- 2 4 $R
  kv_gate inkling "kv inkling TOPP" 1 16 4 TOPP=0.3 SNAP=tiny_inkling_x-i4 -- 2 0 $R
  kv_gate inkling "kv inkling tier off" 1 16 4 COLI_VK_TIER=0 SNAP=tiny_inkling -- 8 0 $R
  kv_gate inkling "kv inkling prefill in chunks of 3" 1 16 4 COLI_VK_CHAIN_ROWS=3 SNAP=tiny_inkling -- 8 0 $R
  kv_gate inkling "kv inkling tiled GEMM" 1 24 4 COLI_VK_GEMM_MIN_S=2 SNAP=tiny_inkling -- 8 0 $R
  CHAINMODE=2 kv_gate inkling "kv inkling prompts only" 1 16 4 SNAP=tiny_inkling -- 8 0 $R
  kv_gate inkling "kv inkling D=6144" 1 8 4 SNAP=tiny_inkling_wide -- 8 0 tiny_inkling_wide/ref_inkling.json
  # COLI_VK_KV_SPLIT=0: the whole mirrors, as before the split
  chain_gate inkling "kv inkling split off" 1 COLI_VK_KV_SPLIT=0 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 SNAP=tiny_inkling -- 8 0 $R
  if grep -qa 'KV cache split' vk.log; then cat vk.log; fail "kv inkling split off: the split ran"; fi
  # the device lost mid-decode with the split on: K/V and convolution states rebuilt on the CPU
  lost_late inkling "kv inkling device lost" 20 rebuild COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 SNAP=tiny_inkling -- 8 0 $R
  # serve sessions (pins, prompt-cache extensions, a divergent prompt, logprobs), prompts
  # only, and the prefix-reuse and dashboard contracts, across the split
  CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' \
    $PY tests/vulkan_chain_serve.py ./inkling tiny_inkling INK_PREFIX_LOG=1 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4
  COLI_VK_CHAIN=2 CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' \
    $PY tests/vulkan_chain_serve.py ./inkling tiny_inkling INK_PREFIX_LOG=1 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 \
    INKLING_TINY=tiny_inkling $PY -m unittest tests.test_inkling_prefix_serve tests.test_inkling_dashboard_hits
}
kv_split_inkling_sanitize() {
  kv_inkling_fixtures
  local R=tiny_inkling/ref_long.json
  kv_san inkling "asan kv inkling" COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 SNAP=tiny_inkling ./inkling 8 0 $R
  kv_san inkling "asan kv inkling chunks of 3" COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 COLI_VK_CHAIN_ROWS=3 SNAP=tiny_inkling \
    ./inkling 8 0 $R
  kv_san inkling "asan kv inkling int4 experts" COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 SNAP=tiny_inkling_x-i4 ./inkling 2 0 $R
  CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' \
    $PY tests/vulkan_chain_serve.py ./inkling tiny_inkling INK_PREFIX_LOG=1 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 \
    > san.log 2>&1 || { cat san.log; fail "asan kv inkling serve"; }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan kv inkling serve: sanitizer diagnostic"; fi
  echo "OK asan kv inkling serve: $(tail -1 san.log)"
}

# mimo: its full-attention layers split (the windowed ones keep their rings); the tiny
# fixture's long case (a 72-token prompt) and the picture's, each with 40 new tokens
kv_mimo_fixtures() {
  $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force --vision
  MGRID=$($PY -c "import json;i=json.load(open('mimo_tiny/ref.json'))['image'];print(i['grid_h'],i['grid_w'])")
  mimo_served_fixture
}
# mimo_kv_gate <tag> <case> <rows> <block> <env...>: the case's prompt and 40 new tokens,
# the CPU's tokens, then every position's logits over the CPU's whole sequence within
# 1e-4 of the largest (CHAINMODE: COLI_VK_CHAIN's value); steps with a host part
mimo_kv_gate() {
  local tag=$1 c=$2 rows=$3 blk=$4 extra=() P lg; shift 4
  [ "$c" = image ] && extra=(--image mimo_tiny/patches.f32 --grid $MGRID)
  P=$(mimo_ids $c prompt_ids)
  local kv=(COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} COLI_VK_TIER_SYNC=1 COLI_VK_KV_DEVICE_ROWS=$rows COLI_VK_KV_BLOCK=$blk)
  env "$@" COLI_TEMP=0 ./mimo mimo_tiny --ids "$P" --ngen 40 "${extra[@]}" > mimo-cpu.txt 2>/dev/null
  env "$@" "${kv[@]}" COLI_TEMP=0 ./mimo mimo_tiny --ids "$P" --ngen 40 "${extra[@]}" > mimo-vk.txt 2> mimo-vk.err
  [ -s mimo-cpu.txt ] && cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-cpu.txt mimo-vk.txt mimo-vk.err; fail "$tag $c: the chain's tokens differ from the CPU"; }
  [ "$(kv_hostparts mimo mimo-vk.err)" -gt 0 ] || { cat mimo-vk.err; fail "$tag $c: the split never ran a host part"; }
  local hp; hp=$(grep -a "KV split:" mimo-vk.err | tail -1 | sed 's/.*KV split: //')
  rm -f cpu.f32 vk.f32
  env "$@" MIMO_LOGITS=cpu.f32 ./mimo mimo_tiny --ids "$P $(cat mimo-cpu.txt)" --ngen 0 "${extra[@]}" > /dev/null 2>&1
  env "$@" "${kv[@]}" MIMO_LOGITS=vk.f32 ./mimo mimo_tiny --ids "$P $(cat mimo-cpu.txt)" --ngen 0 "${extra[@]}" > /dev/null 2> mimo-vk.err
  lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag $c: logits"; }
  echo "OK $tag $c: tokens = CPU, $lg; $hp"
}
kv_split_mimo() {
  kv_mimo_fixtures
  local bits c
  for bits in 32 0 8; do mimo_kv_gate "kv mimo bits=$bits, 16 rows on the device" long 16 4 MIMO_DENSE_BITS=$bits; done
  mimo_kv_gate "kv mimo two blocks on the device" long 8 4 MIMO_DENSE_BITS=32
  mimo_kv_gate "kv mimo a picture" image 16 4 MIMO_DENSE_BITS=0
  mimo_kv_gate "kv mimo one token at a time" long 16 4 MIMO_DENSE_BITS=32 MIMO_CHUNK=1
  mimo_kv_gate "kv mimo blocks of 3" long 16 4 MIMO_DENSE_BITS=32 MIMO_CHUNK=3
  mimo_kv_gate "kv mimo chunks of 3 rows" long 24 8 MIMO_DENSE_BITS=32 COLI_VK_CHAIN_ROWS=3
  mimo_kv_gate "kv mimo tier off, tiled GEMM" long 16 4 MIMO_DENSE_BITS=0 COLI_VK_TIER=0 COLI_VK_GEMM_MIN_S=2
  CHAINMODE=2 mimo_kv_gate "kv mimo prompts only" long 16 4 MIMO_DENSE_BITS=32
  # COLI_VK_KV_SPLIT=0: the whole mirrors, as before the split
  mimo_chain_gate "kv mimo split off" long MIMO_DENSE_BITS=32 COLI_VK_KV_SPLIT=0 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4
  if grep -qa 'KV cache split' mimo-vk.err; then cat mimo-vk.err; fail "kv mimo split off: the split ran"; fi
  # the device lost with the split on (mid-decode; inside a prompt in chunks of 3): the
  # host's caches hold whole steps, the CPU runs on from where they end
  mimo_lost_gate "kv mimo device lost mid-decode" long 3 MIMO_DENSE_BITS=32 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4
  mimo_lost_gate "kv mimo device lost inside a chunked prompt" long 100 MIMO_DENSE_BITS=32 COLI_VK_CHAIN_ROWS=3 \
    COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4
  # serve sessions (photos restored, prompt-cache extensions, a divergent prompt,
  # logprobs) frame for frame within the family's 1e-3, both modes and a lost device
  local k
  for k in 1 2; do
    CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' CHAIN_SERVE_TOL=1e-3 COLI_VK_CHAIN=$k \
      $PY tests/vulkan_chain_serve.py ./mimo mimo_tiny_served OMP_NUM_THREADS=2 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4
  done
  CHAIN_SERVE_TOL=1e-3 $PY tests/vulkan_chain_serve.py ./mimo mimo_tiny_served OMP_NUM_THREADS=2 COLI_VK_CHAIN_FAULT=60 \
    COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4
}
kv_split_mimo_sanitize() {
  kv_mimo_fixtures
  local P; P=$(mimo_ids long prompt_ids)
  kv_san mimo "asan kv mimo" COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 MIMO_DENSE_BITS=32 COLI_TEMP=0 ./mimo mimo_tiny --ids "$P" --ngen 20
  kv_san mimo "asan kv mimo blocks of 3, the FP8 trunk" COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 MIMO_DENSE_BITS=0 MIMO_CHUNK=3 \
    COLI_TEMP=0 ./mimo mimo_tiny --ids "$P" --ngen 10
  CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' CHAIN_SERVE_TOL=1e-3 \
    $PY tests/vulkan_chain_serve.py ./mimo mimo_tiny_served OMP_NUM_THREADS=2 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 \
    > san.log 2>&1 || { cat san.log; fail "asan kv mimo serve"; }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan kv mimo serve: sanitizer diagnostic"; fi
  echo "OK asan kv mimo serve: $(tail -1 san.log)"
}

# kv_mla_gate <engine> <tag> <tol 0|1> <rows> <block> <env...> -- <argv...>: mla_gate (the
# CPU's tokens, logits within 1e-4 with tol 1; FAULT_BACK, REBUILD as there) with the
# split on, which must have run steps with a host part (PINNED=1: and pinned blocks)
kv_mla_gate() {
  local eng=$1 tag=$2 tol=$3 rows=$4 blk=$5; shift 5
  mla_gate "$eng" "$tag" "$tol" COLI_VK_KV_DEVICE_ROWS=$rows COLI_VK_KV_BLOCK=$blk "$@"
  [ "$(kv_hostparts "$eng" vk.log)" -gt 0 ] || { cat vk.log; fail "$tag: the split never ran a host part"; }
  if [ "${PINNED:-0}" = 1 ]; then grep -qa 'KV split: .* [1-9][0-9]* blocks pinned by reads' vk.log || { cat vk.log; fail "$tag: no block was pinned"; }; fi
  echo "   $tag: $(grep -a "KV split:" vk.log | tail -1 | sed 's/.*KV split: //')"
}
# colibri (GLM-5.2): the 32-token oracle over 8 or 16 rows on the device; DSA's lists
kv_split_colibri() {
  [ -f glm_tiny_serve/tokenizer.json ] || glm_chain_fixtures
  export CAP_RAISE=0
  local G="SNAP=glm_tiny REF=ref_glm.json"
  kv_mla_gate colibri "kv colibri decode, 8 rows on the device" 1 8 4 $G -- 64 16 16
  kv_mla_gate colibri "kv colibri cap 1, blocks of 2" 1 8 2 $G -- 1 16 16
  kv_mla_gate colibri "kv colibri prefill in chunks of 3" 1 8 4 $G TF=1 COLI_VK_CHAIN_ROWS=3 -- 64 16 16
  kv_mla_gate colibri "kv colibri 4-bit trunk and experts" 1 8 4 $G IDOT=0 -- 2 4 4
  kv_mla_gate colibri "kv colibri i4 container" 1 8 4 SNAP=glm_tiny_i4 REF=glm_tiny_i4/ref_glm.json IDOT=0 -- 1 4 4
  # DSA's lists: the device skips the positions it does not hold, the host takes them,
  # and the blocks the lists read most are pinned (chunks of 3 leave room for them)
  COLI_VK_KV_PIN=1 PINNED=1 kv_mla_gate colibri "kv colibri DSA top-4, pinned blocks" 1 16 4 $G DSA_TOPK=4 COLI_VK_CHAIN_ROWS=3 -- 64 16 16
  kv_mla_gate colibri "kv colibri DSA top-4, prefill in chunks of 5" 1 8 4 $G DSA_TOPK=4 TF=1 COLI_VK_CHAIN_ROWS=5 -- 64 16 16
  kv_mla_gate colibri "kv colibri DSA_FORCE" 1 8 4 $G DSA_FORCE=1 -- 64 16 16
  # drafts: the verify rows over the split, rejected rows rewritten
  kv_mla_gate colibri "kv colibri n-gram drafts" 1 8 4 $G DRAFT=3 -- 64 16 16
  kv_mla_gate colibri "kv colibri MTP depth 2" 1 8 4 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json DRAFT=2 -- 64 16 16
  kv_mla_gate colibri "kv colibri tier off" 1 8 4 $G COLI_VK_TIER=0 -- 64 16 16
  # prompts only: the prompt and the drafts' verify rows on the device, decode on the CPU
  CHAINMODE=2 kv_mla_gate colibri "kv colibri prompts only, drafts" 1 8 4 $G DRAFT=3 -- 64 16 16
  # COLI_VK_KV_SPLIT=0: the whole mirrors, as before the split
  mla_gate colibri "kv colibri split off" 1 COLI_VK_KV_SPLIT=0 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 $G -- 64 16 16
  if grep -qa 'KV cache split' vk.log; then cat vk.log; fail "kv colibri split off: the split ran"; fi
  FAULT_BACK=12 kv_mla_gate colibri "kv colibri device lost mid-decode" 1 8 4 $G -- 64 16 16
  # serve sessions frame for frame: pins, the prompt cache, the prefill read-out, two KV
  # slots taking the device in turn, DSA's lists with drafts
  local E='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' K="COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4"
  # shellcheck disable=SC2086
  {
    CHAIN_SERVE_EXPECT="$E" CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 $K
    CHAIN_SERVE_EXPECT="$E" CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 DSA_TOPK=4 DRAFT=3 $K
    CHAIN_SERVE_EXPECT="$E" CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=colibri $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 KV_SLOTS=2 $K
  }
  unset CAP_RAISE
}
kv_split_colibri_sanitize() {
  [ -f glm_tiny_serve/tokenizer.json ] || glm_chain_fixtures
  export CAP_RAISE=0
  local K="COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 USAGE_SAVE=0"
  # shellcheck disable=SC2086
  {
    kv_san colibri "asan kv colibri decode" $K SNAP=glm_tiny REF=ref_glm.json ./colibri 64 16 16
    COLI_VK_KV_PIN=1 kv_san colibri "asan kv colibri DSA top-4, pinned, chunks of 3" COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 USAGE_SAVE=0 SNAP=glm_tiny \
      REF=ref_glm.json DSA_TOPK=4 COLI_VK_CHAIN_ROWS=3 ./colibri 64 16 16
    kv_san colibri "asan kv colibri MTP depth 2" $K SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json DRAFT=2 ./colibri 64 16 16
    CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=colibri \
      $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 KV_SLOTS=2 $K > san.log 2>&1 \
      || { cat san.log; fail "asan kv colibri serve"; }
  }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan kv colibri serve: sanitizer diagnostic"; fi
  echo "OK asan kv colibri serve: $(tail -1 san.log)"
  unset CAP_RAISE
}

# glm53 (GLM-5.3 Flash): its MLA layer's latent rows split, the k-pooled lists through
# the split, the KDA layers untouched; the 100-token prompt and 8-token decodes
kv_split_glm53() {
  [ -f glm53_serve/tokenizer.json ] || glm_chain_fixtures
  local ids bits M="--model glm53_stream-i4"
  ids=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
  kv_mla_gate glm53 "kv glm53 decode, 8 rows on the device" 1 8 4 GLM53_BITS=32 -- $M --ids 5,7,9,11,13,17,19,23 --greedy 8
  for bits in 32 8 4; do
    kv_mla_gate glm53 "kv glm53 100-token prompt, ${bits}-bit trunk" 1 16 4 GLM53_BITS=$bits -- $M --ids $ids --greedy 8
  done
  COLI_VK_KV_PIN=1 PINNED=1 kv_mla_gate glm53 "kv glm53 chain chunks of 3, pinned blocks" 1 16 4 GLM53_BITS=32 COLI_VK_CHAIN_ROWS=3 -- $M --ids $ids --greedy 8
  kv_mla_gate glm53 "kv glm53 prefill chunks of 7, one cache slot" 1 16 4 GLM53_BITS=4 GLM53_PREFILL_CHUNK=7 GLM53_EXPERT_GB=0.000001 \
    -- $M --ids $ids --greedy 4
  kv_mla_gate glm53 "kv glm53 blocks of 2" 1 6 2 GLM53_BITS=32 -- $M --ids $ids --greedy 6
  kv_mla_gate glm53 "kv glm53 resident experts" 1 16 4 GLM53_BITS=32 -- --model glm53_tiny --ids $ids --greedy 6
  kv_mla_gate glm53 "kv glm53 tier off" 1 16 4 GLM53_BITS=32 COLI_VK_TIER=0 -- $M --ids $ids --greedy 6
  kv_mla_gate glm53 "kv glm53 an image" 1 4 2 GLM53_BITS=32 -- --model glm53_mm_tiny --ids 103,117,268,268,268,268,120,121 \
    --patches glm53_mm_tiny/patches.f32 --grid 4x4 --greedy 4
  CHAINMODE=2 kv_mla_gate glm53 "kv glm53 prompts only" 1 16 4 GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 -- $M --ids $ids --greedy 6
  # COLI_VK_KV_SPLIT=0: the whole mirrors, as before the split
  mla_gate glm53 "kv glm53 split off" 1 COLI_VK_KV_SPLIT=0 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 GLM53_BITS=32 -- $M --ids $ids --greedy 4
  if grep -qa 'KV cache split' vk.log; then cat vk.log; fail "kv glm53 split off: the split ran"; fi
  FAULT_BACK=7 REBUILD=1 kv_mla_gate glm53 "kv glm53 device lost, the KDA state rebuilt" 1 16 4 GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 \
    -- $M --ids $ids --greedy 6
  # serve sessions frame for frame (pins, the prompt cache, read-outs), two sessions
  # taking the device in turn; a pin restored over rows another branch rewrote
  local E='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' K="COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4"
  # shellcheck disable=SC2086
  {
    CHAIN_SERVE_EXPECT="$E" CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_serve GLM53_BITS=32 $K
    CHAIN_SERVE_EXPECT="$E" CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_serve GLM53_BITS=4 COLI_VK_CHAIN_ROWS=3 $K
    CHAIN_SERVE_EXPECT="$E" CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=numeric $PY tests/vulkan_chain_serve.py ./glm53 glm53_serve GLM53_BITS=32 KV_SLOTS=2 $K
    env $K COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_USAGE=$PWD/chain.usage COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 $PY tests/glm53_pin_branch_harness.py --binary ./glm53 --fixture glm53_mm_tiny
  }
}
kv_split_glm53_sanitize() {
  [ -f glm53_serve/tokenizer.json ] || glm_chain_fixtures
  local ids K="COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 USAGE_SAVE=0"
  ids=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
  # shellcheck disable=SC2086
  {
    kv_san glm53 "asan kv glm53 100-token prompt" $K GLM53_BITS=32 ./glm53 --model glm53_stream-i4 --ids $ids --greedy 6
    COLI_VK_KV_PIN=1 kv_san glm53 "asan kv glm53 int4 trunk, chunks of 3, pinned" $K GLM53_BITS=4 COLI_VK_CHAIN_ROWS=3 ./glm53 --model glm53_stream-i4 --ids $ids --greedy 6
    kv_san glm53 "asan kv glm53 device lost, rebuilt" $K GLM53_BITS=32 GLM53_PREFILL_CHUNK=16 COLI_VK_CHAIN_FAULT=20 ./glm53 \
      --model glm53_stream-i4 --ids $ids --greedy 6
    CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=numeric \
      $PY tests/vulkan_chain_serve.py ./glm53 glm53_serve GLM53_BITS=32 KV_SLOTS=2 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 > san.log 2>&1 \
      || { cat san.log; fail "asan kv glm53 serve"; }
  }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan kv glm53 serve: sanitizer diagnostic"; fi
  echo "OK asan kv glm53 serve: $(tail -1 san.log)"
}

# kv_k3c_gate <tag> <rows> <block> <env...>: k3c_gate (kimi's three cases, logits within
# 2e-3 of the largest) with the split on; the last case run (long, 72 prompt tokens)
# must have had steps with a host part
kv_k3c_gate() {
  local tag=$1 rows=$2 blk=$3; shift 3
  k3c_gate "$tag" "$@" COLI_VK_KV_DEVICE_ROWS=$rows COLI_VK_KV_BLOCK=$blk
  [ "$(kv_hostparts kimi_k3 vk.log)" -gt 0 ] || { cat vk.log; fail "$tag: the split never ran a host part"; }
  echo "   $tag: $(grep -a "KV split:" vk.log | tail -1 | sed 's/.*KV split: //')"
}
# kimi_k3: its gated MLA layers (NoPE) split, the KDA layers untouched
kv_split_kimi() {
  [ -f kimi_k3_tiny/model.safetensors ] || $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  k3c_serve_fixture
  local O="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0" b
  # shellcheck disable=SC2086
  {
    kv_k3c_gate "kv kimi_k3 f32, 16 rows on the device" 16 4 $O
    kv_k3c_gate "kv kimi_k3 f32, blocks of 2" 8 2 $O
    kv_k3c_gate "kv kimi_k3 f32, tier off" 16 4 $O COLI_VK_TIER=0
    for b in 8 4; do kv_k3c_gate "kv kimi_k3 ${b}-bit trunk" 16 4 K3_BITS=$b K3_MLA_BITS=$b K3_HEAD_BITS=$b K3_IDOT=0 COLI_TEMP=0; done
    kv_k3c_gate "kv kimi_k3 prefill one token at a time" 16 4 $O K3_CHUNK=1
    kv_k3c_gate "kv kimi_k3 chain chunks of 3" 16 4 $O COLI_VK_CHAIN_ROWS=3
    CHAINMODE=2 kv_k3c_gate "kv kimi_k3 prompts only" 16 4 $O
    # COLI_VK_KV_SPLIT=0: the whole mirrors, as before the split
    K3C_CASES=long k3c_gate "kv kimi_k3 split off" $O COLI_VK_KV_SPLIT=0 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4
    if grep -qa 'KV cache split' vk.log; then cat vk.log; fail "kv kimi_k3 split off: the split ran"; fi
    # the device lost: mid-decode (the KDA state rebuilt), and inside the long prompt
    FAULT_BACK=3 REBUILD=1 kv_k3c_gate "kv kimi_k3 device lost mid-decode" 16 4 $O
    K3C_CASES=long FAULT_BACK=150 kv_k3c_gate "kv kimi_k3 device lost in the prompt" 16 4 $O
  }
  # serve sessions frame for frame with recurrent-state checkpoints, and the checkpoint
  # tests, across the split
  local S="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 K3_PREFIX_LOG=1 USAGE_SAVE=0" K="COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4"
  local E='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part'
  # shellcheck disable=SC2086
  {
    CHAIN_SERVE_TOL=2e-2 CHAIN_SERVE_EXPECT="$E" $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $S $K
    CHAIN_SERVE_TOL=2e-2 CHAIN_SERVE_EXPECT="$E" $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $S COLI_K3_CKPT=4 $K
    CHAIN_SERVE_TOL=2e-2 CHAIN_SERVE_EXPECT="$E" $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve K3_BITS=4 K3_IDOT=0 K3_PREFIX_LOG=1 \
      USAGE_SAVE=0 COLI_VK_CHAIN_ROWS=3 $K
    env $K COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 \
      $PY tests/test_kimi_k3_ckpt.py --binary ./kimi_k3 --fixture ./kimi_k3_tiny
  }
}
kv_split_kimi_sanitize() {
  [ -f kimi_k3_tiny/model.safetensors ] || $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  k3c_serve_fixture
  local O="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0 USAGE_SAVE=0" K="COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4" ids
  ids=$(k3c_ids long)
  # shellcheck disable=SC2086
  {
    kv_san kimi_k3 "asan kv kimi_k3 f32" $K $O ./kimi_k3 kimi_k3_tiny --ids "$ids" --ngen 8
    kv_san kimi_k3 "asan kv kimi_k3 int4 trunk, chunks of 3" $K K3_BITS=4 K3_IDOT=0 USAGE_SAVE=0 COLI_VK_CHAIN_ROWS=3 ./kimi_k3 kimi_k3_tiny \
      --ids "$ids" --ngen 8
    kv_san kimi_k3 "asan kv kimi_k3 prompts only" $K $O COLI_VK_CHAIN=2 ./kimi_k3 kimi_k3_tiny --ids "$ids" --ngen 8
    CHAIN_SERVE_TOL=2e-2 CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' \
      $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 K3_PREFIX_LOG=1 \
      USAGE_SAVE=0 COLI_K3_CKPT=4 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 > san.log 2>&1 || { cat san.log; fail "asan kv kimi_k3 serve"; }
  }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan kv kimi_k3 serve: sanitizer diagnostic"; fi
  echo "OK asan kv kimi_k3 serve: $(tail -1 san.log)"
}

# deepseek_v41: the compressed rows of its kv_source layers split (ratios 2 and 1 in the
# fixtures; the window ring and the index keys stay whole on the device). kv_v41_gate:
# v41_gate (the CPU's tokens, every forward's logits within 1e-4 of the largest) under an
# emulated small device, which must have run attention steps with a host part.
kv_v41_gate() {  # <tag> <rows> <block> <env...> -- <argv...>
  local tag=$1 rows=$2 blk=$3; shift 3
  v41_gate "$tag" COLI_VK_KV_DEVICE_ROWS=$rows COLI_VK_KV_BLOCK=$blk "$@"
  if [ -z "${FAULT_BACK:-}" ]; then
    [ "$(kv_hostparts deepseek_v41 vk.log)" -gt 0 ] || { cat vk.log; fail "$tag: the split never ran a host part"; }
  fi
  echo "   $tag: $(grep -a "KV split:" vk.log | tail -1 | sed 's/.*KV split: //')"
}
kv_split_deepseek_v41() {
  v41_chain_fixtures
  local cap f K=(COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2)
  for cap in 2 8; do kv_v41_gate "kv deepseek_v41 40-token prompt cap=$cap" 16 4 SNAP=dsv41_long -- $cap dsv41_long/ref.json; done
  kv_v41_gate "kv deepseek_v41 four blocks of 2" 8 2 SNAP=dsv41_long -- 8 dsv41_long/ref.json
  COLI_VK_KV_PIN=1 kv_v41_gate "kv deepseek_v41 eight blocks of 4, four pinned" 32 4 SNAP=dsv41_long -- 8 dsv41_long/ref.json
  kv_v41_gate "kv deepseek_v41 12-token prompt" 8 2 SNAP=dsv41_tiny -- 8 dsv41_tiny/ref.json
  # DSpark: a verify is one chunk, and a chunk's new compressed rows fit the window (four
  # blocks of 4 take a verify of 4 rows, eight blocks one of 12)
  for f in 1 3; do kv_v41_gate "kv deepseek_v41 DSpark spec=$f" 16 4 SNAP=dsv41_long V41_DSPARK=1 V41_SPEC_FORCE=$f -- 8 dsv41_long/ref.json; done
  kv_v41_gate "kv deepseek_v41 DSpark spec=5" 32 4 SNAP=dsv41_long V41_DSPARK=1 V41_SPEC_FORCE=5 -- 8 dsv41_long/ref.json
  kv_v41_gate "kv deepseek_v41 prompt in chunks of 3" 16 4 SNAP=dsv41_long COLI_VK_CHAIN_ROWS=3 -- 8 dsv41_long/ref.json
  CHAINMODE=2 kv_v41_gate "kv deepseek_v41 prompts only" 8 2 SNAP=dsv41_long -- 8 dsv41_long/ref.json
  kv_v41_gate "kv deepseek_v41 tier off" 16 4 SNAP=dsv41_long COLI_VK_TIER=0 -- 8 dsv41_long/ref.json
  kv_v41_gate "kv deepseek_v41 V41_INDEX_OWNER=1" 16 4 SNAP=dsv41_long V41_INDEX_OWNER=1 -- 8 dsv41_long/ref.json
  # COLI_VK_KV_SPLIT=0: the whole mirrors, as before the split
  v41_gate "kv deepseek_v41 split off" SNAP=dsv41_long COLI_VK_KV_SPLIT=0 "${K[@]}" -- 8 dsv41_long/ref.json
  if grep -qa 'KV cache split' vk.log; then cat vk.log; fail "kv deepseek_v41 split off: the split ran"; fi
  FAULT_BACK=5 kv_v41_gate "kv deepseek_v41 device lost mid-decode" 8 2 SNAP=dsv41_long -- 8 dsv41_long/ref.json
  FAULT_BACK=60 kv_v41_gate "kv deepseek_v41 device lost in the prompt" 8 2 SNAP=dsv41_long -- 8 dsv41_long/ref.json
  CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' \
    $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=0 "${K[@]}"
  CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' \
    $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=1 "${K[@]}"
  env "${K[@]}" COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_USAGE=$PWD/chain.usage \
    COLI_DSV41_FIXTURE=$PWD/dsv41_tiny $PY -m unittest tests.test_dsv41_prefix_serve tests.test_dsv41_dspark_serve
  rm -f chain.usage chain-serve.usage
}
kv_split_deepseek_v41_sanitize() {
  v41_chain_fixtures
  local K=(COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2)
  kv_san deepseek_v41 "asan kv deepseek_v41 40-token prompt" "${K[@]}" SNAP=dsv41_long ./deepseek_v41 8 dsv41_long/ref.json
  COLI_VK_KV_PIN=1 kv_san deepseek_v41 "asan kv deepseek_v41 pinned blocks, chunks of 3" COLI_VK_KV_DEVICE_ROWS=32 COLI_VK_KV_BLOCK=4 \
    COLI_VK_CHAIN_ROWS=3 SNAP=dsv41_long ./deepseek_v41 2 dsv41_long/ref.json
  kv_san deepseek_v41 "asan kv deepseek_v41 DSpark spec=5" COLI_VK_KV_DEVICE_ROWS=32 COLI_VK_KV_BLOCK=4 SNAP=dsv41_long \
    V41_DSPARK=1 V41_SPEC_FORCE=5 ./deepseek_v41 8 dsv41_long/ref.json
  CHAIN_SERVE_EXPECT='KV split: [0-9]+ layer steps, [1-9][0-9]* with a host part' \
    $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=1 "${K[@]}" > san.log 2>&1 || { cat san.log; fail "asan kv deepseek_v41 serve"; }
  echo "OK asan kv deepseek_v41 serve: $(tail -1 san.log)"
  rm -f chain.usage chain-serve.usage
}

# DeepSeek's split keeps sparse attention in its original order on the device:
# only selected missing compressed rows are staged from RAM. The exact chunk and
# decode-versus-teacher-forced checks therefore apply unchanged, including V4 bf16.
kv_v4_gate() {  # <tag> <rows> <block> <fixture> <case> <env...>
  local tag=$1 rows=$2 blk=$3 fx=$4 c=$5; shift 5
  v4_chain_gate "$tag" "$fx" "$c" COLI_VK_KV_DEVICE_ROWS=$rows COLI_VK_KV_BLOCK=$blk "$@"
  if [ -z "${FAULT_BACK:-}" ]; then
    [ "$(kv_hostparts deepseek_v4 v4-vk.err)" -gt 0 ] || { cat v4-vk.err; fail "$tag: the split never ran a host part"; }
  fi
  echo "   $tag: $(grep -a "KV split:" v4-vk.err | tail -1 | sed 's/.*KV split: //')"
}
kv_split_deepseek_v4() {
  v4_chain_fixtures
  local c fx
  # the compressed case makes 6 rows at ratio 4 (four blocks of 1 on the device: a split
  # holds three blocks at least), the long one 19 (four blocks of 2)
  for fx in deepseek_v4_tiny_t deepseek_v4_tiny_e8; do
    kv_v4_gate "kv deepseek_v4 $fx compressed" 4 1 $fx compressed
    kv_v4_gate "kv deepseek_v4 $fx long" 8 2 $fx long
  done
  COLI_VK_KV_PIN=1 kv_v4_gate "kv deepseek_v4 eight blocks of 2, four pinned" 16 2 deepseek_v4_tiny_t long
  kv_v4_gate "kv deepseek_v4 2 output groups" 8 2 deepseek_v4_tiny_g2 long
  kv_v4_gate "kv deepseek_v4 a window of 4, ratios 4 and 2" 8 2 deepseek_v4_tiny_w4 long
  kv_v4_gate "kv deepseek_v4 an indexer of 64 heads of 128" 8 2 deepseek_v4_tiny_ix long
  kv_v4_gate "kv deepseek_v4 tier off" 8 2 deepseek_v4_tiny_t long COLI_VK_TIER=0 COLI_VK_DENSE=0
  kv_v4_gate "kv deepseek_v4 chain chunks of 3" 16 2 deepseek_v4_tiny_e8 long COLI_VK_CHAIN_ROWS=3
  kv_v4_gate "kv deepseek_v4 prefill chunks of 7" 16 2 deepseek_v4_tiny_t long V4_PREFILL_CHUNK=7
  CHAINMODE=2 kv_v4_gate "kv deepseek_v4 prompts only" 8 2 deepseek_v4_tiny_t long
  # n-gram drafts (V4_DRAFT): accepted and rejected, every one rejected, a mix
  local A=ids:20,21,22,23,24,25,26,27,28,29,20,21,22:12 R=ids:20,21,22,23,50,51,52,20,21,22:12
  local M=ids:30,31,32,33,34,35,60,61,30,31,32,33,34,35,36,37,38,39,40,41,30,31,32:12
  kv_v4_gate "kv deepseek_v4 n-gram drafts accepted and rejected" 4 1 deepseek_v4_tiny_t $A V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  kv_v4_gate "kv deepseek_v4 an n-gram draft rejected" 4 1 deepseek_v4_tiny_t $R V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  kv_v4_gate "kv deepseek_v4 n-gram drafts, a window of 4" 4 2 deepseek_v4_tiny_w4 $M V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  # The same model arithmetic with full mirrors, a split, different chunking and
  # read-based pins. These comparisons are bit-exact and have no logit tolerance.
  local K=(COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2)
  v4_chain_same "kv deepseek_v4 split = full mirror" deepseek_v4_tiny_t long "COLI_VK_KV_SPLIT=0" "COLI_VK_KV_SPLIT=1" "${K[@]}"
  v4_chain_same "kv deepseek_v4 chunks 1 = 3" deepseek_v4_tiny_t long "COLI_VK_CHAIN_ROWS=1" "COLI_VK_CHAIN_ROWS=3" "${K[@]}"
  v4_chain_same "kv deepseek_v4 pinned residency" deepseek_v4_tiny_e8 long "COLI_VK_KV_PIN=0" "COLI_VK_KV_PIN=1" "${K[@]}"
  v4_chain_same "kv deepseek_v4 draft chunks" deepseek_v4_tiny_w4 $M "COLI_VK_CHAIN_ROWS=1" "COLI_VK_CHAIN_ROWS=3" "${K[@]}" V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  # COLI_VK_KV_SPLIT=0: the whole mirrors, as before the split
  v4_chain_gate "kv deepseek_v4 split off" deepseek_v4_tiny_t long COLI_VK_KV_SPLIT=0 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2
  if grep -qa 'KV cache split' v4-vk.err; then cat v4-vk.err; fail "kv deepseek_v4 split off: the split ran"; fi
  FAULT_BACK=3 kv_v4_gate "kv deepseek_v4 device lost mid-decode" 8 2 deepseek_v4_tiny_t long
  FAULT_BACK=40 kv_v4_gate "kv deepseek_v4 device lost in the prompt" 8 2 deepseek_v4_tiny_t long
  # serve: a pin, its extension, a read-only prompt, a shorter one (texts and reuse =
  # CPU, logprobs within 0.5)
  COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2 v4_chain_serve deepseek_v4_tiny_t
  rm -rf deepseek_v4_tiny_t deepseek_v4_tiny_e8 deepseek_v4_tiny_g2 deepseek_v4_tiny_w4 deepseek_v4_tiny_ix chain.usage
  rm -f cpu.f32 vk.f32 v4-*.f32 v4-*.json v4-*.err v4-tf.txt
}
kv_split_deepseek_v4_sanitize() {
  # the V4 sanitized build as its chain family makes it (no LTO, the sanitizers at the link)
  local SAN="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  make deepseek-v4-clean >/dev/null 2>&1 || true
  make deepseek-v4 VK=1 LTO=0 EXTRA_CFLAGS="$SAN" EXTRA_LDFLAGS="$SAN"
  v4_chain_fixtures
  local p K=(COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2)
  p=$(v4_chain_prompt deepseek_v4_tiny_t long)
  kv_v4_san() {  # <tag> <env and argv...>: v4_chain_san, and steps with a host part
    v4_chain_san "$@"
    [ -n "${FAULT_BACK:-}" ] || [ "$(kv_hostparts deepseek_v4 san.log)" -gt 0 ] || { cat san.log; fail "$1: the split never ran a host part"; }
  }
  kv_v4_san "asan kv deepseek_v4 long, 4 experts" "${K[@]}" ./deepseek_v4 ./deepseek_v4_tiny_t "$p" --raw-prompt --max-tokens 4
  kv_v4_san "asan kv deepseek_v4 chunks of 3, a window of 4" "${K[@]}" COLI_VK_CHAIN_ROWS=3 ./deepseek_v4 ./deepseek_v4_tiny_w4 \
    "$(v4_chain_prompt deepseek_v4_tiny_w4 long)" --raw-prompt --max-tokens 4
  COLI_VK_KV_PIN=1 kv_v4_san "asan kv deepseek_v4 pinned blocks, 64 index heads" COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=2 ./deepseek_v4 \
    ./deepseek_v4_tiny_ix "$(v4_chain_prompt deepseek_v4_tiny_ix long)" --raw-prompt --max-tokens 4
  kv_v4_san "asan kv deepseek_v4 drafts" COLI_VK_KV_DEVICE_ROWS=4 COLI_VK_KV_BLOCK=1 V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1 ./deepseek_v4 \
    ./deepseek_v4_tiny_w4 "$(v4_chain_prompt x ids:30,31,32,33,34,35,60,61,30,31,32,33,34,35,36,37,38,39,40,41,30,31,32)" --raw-prompt --max-tokens 12
  FAULT_BACK=10 kv_v4_san "asan kv deepseek_v4 device lost" "${K[@]}" ./deepseek_v4 ./deepseek_v4_tiny_t "$p" --raw-prompt --max-tokens 4
  COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2 v4_chain_serve deepseek_v4_tiny_t > san.log 2>&1 || { cat san.log; fail "asan kv deepseek_v4 served"; }
  echo "OK asan kv deepseek_v4 served: $(tail -1 san.log | cut -c1-120)"
  rm -rf deepseek_v4_tiny_t deepseek_v4_tiny_e8 deepseek_v4_tiny_g2 deepseek_v4_tiny_w4 deepseek_v4_tiny_ix san.json san.log
  make deepseek-v4-clean >/dev/null 2>&1 || true
}

# kv-split: every chain engine but DeepSeek's (whose fixtures want another transformers)
family_kv_split() {
  export OMP_NUM_THREADS=2
  make qwen36 qwen38 olmoe inkling mimo colibri glm53 kimi_k3 tests/test_vk_chain VK=1
  ./tests/test_vk_chain shaders/qmatmul.spv | tee vk_chain.log
  tail -1 vk_chain.log | grep -qx PASS || fail "the chain's ops"
  grep -a 'kvs ' vk_chain.log
  kv_split_qwen36; kv_split_qwen38; kv_split_olmoe; kv_split_inkling; kv_split_mimo
  kv_split_colibri; kv_split_glm53; kv_split_kimi
  unset OMP_NUM_THREADS
}
family_kv_split_sanitize() {
  make clean >/dev/null 2>&1 || true
  make qwen36 qwen38 olmoe inkling mimo colibri glm53 kimi_k3 tests/test_vk_chain VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  ./tests/test_vk_chain shaders/qmatmul.spv > san.log 2>&1 || true
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log || ! tail -1 san.log | grep -qx PASS; then cat san.log; fail "asan: the chain's ops"; fi
  echo "OK asan: the chain's ops, the split KV cache's among them"
  kv_split_qwen36_sanitize; kv_split_qwen38_sanitize; kv_split_olmoe_sanitize; kv_split_inkling_sanitize
  kv_split_mimo_sanitize; kv_split_colibri_sanitize; kv_split_glm53_sanitize; kv_split_kimi_sanitize
  make clean >/dev/null 2>&1 || true
}
family_kv_split_deepseek() {
  export OMP_NUM_THREADS=2
  make deepseek_v41 deepseek-v4 VK=1
  kv_split_deepseek_v41; kv_split_deepseek_v4
  unset OMP_NUM_THREADS
}
family_kv_split_deepseek_sanitize() {
  make clean >/dev/null 2>&1 || true
  # deepseek_v4's sanitized build is its own (no LTO, the sanitizers at the link too):
  # kv_split_deepseek_v4_sanitize makes it, as v4_chain_family_sanitize does
  make deepseek_v41 VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  kv_split_deepseek_v41_sanitize; kv_split_deepseek_v4_sanitize
  make clean >/dev/null 2>&1 || true
}

# The decision engines (docs/vulkan.md, "The decision engines"): laya and gliner_decide
# with their forward on the device (COLI_VK_CHAIN=1, chain_enc.comp) and matrix by matrix
# (COLI_VK_CHAIN=0), Clef's DECIDE on qwen36's per-matrix path and its chain, in f32, int8
# and f16 (fmt 14), the rows read in place and copied. Each configuration against the CPU
# (tests/vulkan_decide_compare.py: every case of the fixture, the same decisions, the
# probabilities within 1e-5 and the logits within 1e-4 of the largest, matmuls on the
# device); the tiny oracles through the device; the device opened with every path off,
# byte for byte the CPU's; a device lost mid-run (the rest of the forwards on the CPU);
# the serve tests and TypeSafe's SDKs (where installed) through the device, checked by
# the engines' own lines. SAN=1 (decide-sanitize): a sanitized build, the comparisons
# and the serve tests, UBSan halting on its first report.
decide_vk_lines() {  # <log> <tag>: every decision engine's forwards ran on the device
  local e
  for e in laya gliner_decide; do
    grep -qE "^\[VK\] $e chain: [1-9][0-9]* forwards on the device" "$1" || { tail -50 "$1"; fail "$2: $e never ran on the device"; }
  done
  grep -qE "^\[VK\] qwen36 chain: [1-9][0-9]* forwards" "$1" || { tail -50 "$1"; fail "$2: Clef never ran on the device"; }
}
family_decide() {
  local SANF=""
  if [ -n "${SAN:-}" ]; then
    SANF="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
    make clean >/dev/null 2>&1 || true
    export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1
  fi
  make laya gliner_decide qwen36 tests/test_qwen36_weight_alloc VK=1 EXTRA_CFLAGS="$SANF"
  COLI_VULKAN=0 ./tests/test_qwen36_weight_alloc
  COLI_VULKAN=1 ./tests/test_qwen36_weight_alloc
  make laya-tiny-generate gliner-decide-tiny-generate clef-tiny-generate
  export OMP_NUM_THREADS=2
  local c e
  local -a cmp=(
    "laya COLI_VK_CHAIN=1" "laya COLI_VK_CHAIN=0" "gliner COLI_VK_CHAIN=1" "gliner COLI_VK_CHAIN=0"
    "laya COLI_VK_CHAIN=1 COLI_VK_STAGED=1" "gliner COLI_VK_CHAIN=1 COLI_VK_STAGED=1"
    "laya COLI_VK_CHAIN=1 COLI_VK_CHAIN_FAULT=10" "gliner COLI_VK_CHAIN=1 COLI_VK_CHAIN_FAULT=10"
    "clef" "clef COLI_VK_CHAIN=1"
    "clef COLI_VK_CHAIN=1 COLI_VK_CHAIN_ROWS=1 COLI_VK_CHAIN_FAULT=10 COLI_VK_DENSE_HOST=0"
    "clef COLI_DENSE_I8=1 COLI_DENSE_BITS=16 COLI_VK_CHAIN=1 COLI_VK_CHAIN_ROWS=1 COLI_VK_CHAIN_FAULT=10 COLI_VK_DENSE_HOST=0 COLI_VK_IMPORT=0"
    "clef COLI_DENSE_I8=1 COLI_DENSE_IDOT=0 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0"
    "clef COLI_DENSE_I8=1 COLI_DENSE_BITS=16 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0"
    "clef COLI_DENSE_I8=1 COLI_DENSE_IDOT=0" "clef COLI_DENSE_I8=1 COLI_DENSE_IDOT=0 COLI_VK_CHAIN=1"
    "clef COLI_DENSE_I8=1 COLI_DENSE_IDOT=0 COLI_VK_IMPORT=0"
    "clef COLI_DENSE_I8=1 COLI_DENSE_BITS=16" "clef COLI_DENSE_I8=1 COLI_DENSE_BITS=16 COLI_VK_CHAIN=1"
    "clef COLI_DENSE_I8=1 COLI_DENSE_BITS=16 COLI_VK_IMPORT=0")
  for c in "${cmp[@]}"; do
    # shellcheck disable=SC2086
    $PY tests/vulkan_decide_compare.py $c > cmp.log 2>&1 || { cat cmp.log; fail "decide: $c"; }
    tail -1 cmp.log
  done
  for e in laya gliner clef; do
    $PY tests/vulkan_decide_compare.py --off $e COLI_VK_CHAIN=0 COLI_VK_DENSE=0 > cmp.log 2>&1 || { cat cmp.log; fail "decide: $e with every path off"; }
    tail -1 cmp.log
  done
  [ -n "${SAN:-}" ] || for c in 0 1; do
    COLI_VULKAN=1 COLI_VK_CHAIN=$c LAYA_TINY_REQUIRED=1 GLINER_DECIDE_TINY_REQUIRED=1 CLEF_TINY_REQUIRED=1 \
      $PY -m unittest tests.test_laya_tiny tests.test_gliner_decide_tiny tests.test_clef_tiny > oracle.log 2>&1 \
      || { cat oracle.log; fail "decide: the tiny oracles through the device, COLI_VK_CHAIN=$c"; }
    echo "OK decide: the tiny oracles (Laya, GLiNER2.5-Decide, Clef) through the device, COLI_VK_CHAIN=$c"
  done
  COLI_VULKAN=1 COLI_VK_CHAIN=1 LAYA_TINY_REQUIRED=1 GLINER_DECIDE_TINY_REQUIRED=1 CLEF_TINY_REQUIRED=1 \
    $PY -m unittest tests.test_decision_serve > serve.log 2>&1 || { tail -80 serve.log; fail "decide: the serve tests through the device"; }
  decide_vk_lines serve.log "decide serve"
  echo "OK decide: the serve tests (tests.test_decision_serve) through the device: $(tail -1 serve.log)"
  if [ -z "${SAN:-}" ] && $PY -c "import typesafe_sdk" 2>/dev/null && [ -d "${JEV_TS_SDK_DIR:-tools/jev-sdk}/node_modules" ]; then
    make mimo VK=1 mimo-tiny-generate
    rm -f sdk-serve.log
    COLI_VULKAN=1 COLI_VK_CHAIN=1 JEV_SDK_REQUIRED=1 JEV_SDK_SERVE_LOG=sdk-serve.log \
      $PY -m unittest tests.test_jev_sdk > sdk.log 2>&1 || { tail -80 sdk.log; fail "decide: TypeSafe's SDKs through the device"; }
    decide_vk_lines sdk-serve.log "decide SDKs"
    echo "OK decide: TypeSafe's Python and TypeScript SDKs (tests.test_jev_sdk) through the device"
  elif [ -z "${SAN:-}" ]; then
    echo "SKIP decide: TypeSafe's SDKs are not installed (tools/requirements-jev-sdk.txt, npm ci in tools/jev-sdk)"
  fi
  rm -f cmp.log oracle.log serve.log sdk.log sdk-serve.log
  unset OMP_NUM_THREADS
  [ -z "${SAN:-}" ] || make clean >/dev/null 2>&1 || true
}

case "${1:-}" in
  shader)         shader_formats ;;
  qwen)           family_qwen ;;
  qwen-sanitize)  family_qwen_sanitize ;;
  inkling-olmoe)  family_inkling_olmoe ;;
  inkling-olmoe-sanitize) family_inkling_olmoe_sanitize ;;
  mimo-qwenimage) family_mimo_qwenimage ;;
  deepseek)       family_deepseek ;;
  deepseek-sanitize) family_deepseek_sanitize ;;
  kimi)           family_kimi ;;
  kimi-mimo-sanitize) family_kimi_mimo_sanitize ;;
  mimo-chain)     family_mimo_chain ;;
  mimo-chain-sanitize) family_mimo_chain_sanitize ;;
  glm)            family_glm ;;
  glm-sanitize)   family_glm_sanitize ;;
  qwen-chain)     family_qwen_chain ;;
  qwen-chain-sanitize) family_qwen_chain_sanitize ;;
  qwen-spec)      family_qwen_spec ;;
  qwen-spec-sanitize) family_qwen_spec_sanitize ;;
  inkling-olmoe-chain) family_inkling_olmoe_chain ;;
  inkling-olmoe-chain-sanitize) family_inkling_olmoe_chain_sanitize ;;
  glm-chain)      family_glm_chain ;;
  glm-chain-sanitize) family_glm_chain_sanitize ;;
  kimi-chain)     family_kimi_chain ;;
  kimi-chain-sanitize) family_kimi_chain_sanitize ;;
  deepseek-chain) family_deepseek_chain ;;
  deepseek-chain-sanitize) family_deepseek_chain_sanitize ;;
  prefill-qwen)   family_prefill_qwen ;;
  prefill-qwen-sanitize) family_prefill_qwen_sanitize ;;
  prefill-inkling-olmoe) family_prefill_inkling_olmoe ;;
  prefill-mimo-kimi) family_prefill_mimo_kimi ;;
  prefill-glm)    family_prefill_glm ;;
  prefill-deepseek) family_prefill_deepseek ;;
  kv-split)       family_kv_split ;;
  kv-split-sanitize) family_kv_split_sanitize ;;
  kv-split-deepseek) family_kv_split_deepseek ;;
  kv-split-deepseek-sanitize) family_kv_split_deepseek_sanitize ;;
  staged)         family_staged ;;
  *-staged)       # COLI_VK_STAGED unset for the window's run, then the whole family staged
                  if [ "$1" = qwen-staged ]; then env -u COLI_VK_STAGED bash tests/vulkan_engines.sh staged-window; fi
                  COLI_VK_STAGED=1 bash tests/vulkan_engines.sh "${1%-staged}" ;;
  staged-window)  staged_window ;;
  staged-faults)  family_staged_faults ;;
  staged-faults-sanitize) SAN=1 family_staged_faults ;;
  decide)         family_decide ;;
  dev2)           family_dev2 ;;
  dev2-deepseek-kimi-mimo) family_dev2_deepseek_kimi_mimo ;;
  dev2-sanitize)  family_dev2_sanitize ;;
  decide-sanitize) SAN=1 family_decide ;;
  dense-only-*)   g=${1#dense-only-}; fn=dho_family_${g//-/_}
                  declare -F "$fn" >/dev/null || { echo "no dense-only group ${g}" >&2; exit 2; }
                  "$fn" ;;
  partial-*)      g=${1#partial-}; fn=ptl_family_${g//-/_}
                  declare -F "$fn" >/dev/null || { echo "no partial-chain group ${g}" >&2; exit 2; }
                  "$fn" ;;
  *) echo "usage: $0 decide|decide-sanitize|staged|<family>-staged|shader|qwen|qwen-sanitize|inkling-olmoe|inkling-olmoe-sanitize|mimo-qwenimage|deepseek|deepseek-sanitize|kimi|kimi-mimo-sanitize|glm|glm-sanitize|qwen-chain|qwen-chain-sanitize|qwen-spec|qwen-spec-sanitize|mimo-chain|mimo-chain-sanitize|inkling-olmoe-chain|inkling-olmoe-chain-sanitize|glm-chain|glm-chain-sanitize|kimi-chain|kimi-chain-sanitize|deepseek-chain|deepseek-chain-sanitize|dense-only-<group>[-sanitize]|partial-<group>[-sanitize]|prefill-qwen|prefill-qwen-sanitize|prefill-inkling-olmoe|prefill-mimo-kimi|prefill-glm|prefill-deepseek|kv-split|kv-split-sanitize|kv-split-deepseek|kv-split-deepseek-sanitize|dev2|dev2-deepseek-kimi-mimo|dev2-sanitize" >&2; exit 2 ;;
esac
