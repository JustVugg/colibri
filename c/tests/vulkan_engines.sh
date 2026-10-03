#!/usr/bin/env bash
# Every engine's Vulkan path against its own CPU run, on Lavapipe (Mesa's software
# Vulkan), one family per call so CI can run them side by side:
#
#   bash tests/vulkan_engines.sh qwen | qwen-sanitize | inkling-olmoe | inkling-olmoe-sanitize
#   bash tests/vulkan_engines.sh mimo-qwenimage | kimi | kimi-mimo-sanitize | deepseek | deepseek-sanitize
#   bash tests/vulkan_engines.sh glm | glm-sanitize   # GLM-5.2 (colibri) and GLM-5.3 Flash (glm53)
#   bash tests/vulkan_engines.sh shader    # the qmatmul formats, the expert batch and the tier, no engine
#   bash tests/vulkan_engines.sh qwen-chain | qwen-chain-sanitize   # the dense chain (COLI_VK_CHAIN=1)
#   bash tests/vulkan_engines.sh mimo-chain | mimo-chain-sanitize   # MiMo-V2.6's dense chain
#   bash tests/vulkan_engines.sh inkling-olmoe-chain | inkling-olmoe-chain-sanitize
#   bash tests/vulkan_engines.sh glm-chain | glm-chain-sanitize     # the same for colibri and glm53
#
# Needs libvulkan-dev, glslc and mesa-vulkan-drivers, plus the Python packages of
# the family's tiny fixtures (see the vulkan-engines job in .github/workflows/ci.yml).
#
# Lavapipe is a CPU rasteriser: nothing here says anything about speed. What it does
# prove is that every resident format an engine uploads reaches the shader and comes
# back as the CPU computes it. Each configuration is gated on two things:
#   - the Vulkan run gives the tokens of the CPU run with the same snapshot and
#     settings (and, where the engine has one, passes its own oracle);
#   - its per-matrix and dense-chain matmul counts sum to N > 0, because a hook that
#     declines every matrix would otherwise pass the first gate trivially. A
#     configuration of the routed-expert tier with the dense trunk on the CPU (the
#     default on Lavapipe while the tier is on, see dense_where) gates on its count of
#     routed experts the device served instead.
set -euo pipefail
cd "$(dirname "$0")/.."
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json}
export COLI_NO_OMP_TUNE=1
PY=${PY:-python3}

fail() { echo "FAIL: $*"; exit 1; }

# Every sanitized run below sets ASAN_OPTIONS with detect_stack_use_after_return=0:
# ASan's fake stack frames are only 32-byte aligned, and on an AVX-512 runner
# (-march=native) GCC keeps 64-byte aligned locals there and stores them with
# vmovdqa64, which faults (c/Makefile, ASAN_ENV, has the details).

# vk_count <engine> <log>: the latest per-matrix and dense-chain matmul counts.
# The counters are independent and cumulative; sum their last reports, not turns.
vk_count() {
  local n chain
  n=$(sed -n "s/^\[VK\] $1: \([0-9][0-9]*\) matmuls on the GPU.*/\1/p" "$2" | tail -1)
  chain=$(sed -n "s/^\[VK\] $1 chain: .* ops, \([0-9][0-9]*\) matmuls,.*/\1/p" "$2" | tail -1)
  echo "$(( ${n:-0} + ${chain:-0} ))"
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
  make tests/test_vk_tier tests/test_glm53_vk_f32 VK=1   # every shader too: the harness's expert batch needs them
  cc -O2 -pthread -DVK_TEST backend_vulkan.c -o vk_test -lvulkan -lm
  COLI_VK_TEST_MATMUL_ONLY=1 ./vk_test shaders/qmatmul.spv | tee vk_test.log
  tail -1 vk_test.log | grep -qx PASS || fail "qmatmul format cases"
  ./tests/test_vk_tier shaders/qmatmul.spv | tee vk_tier.log
  tail -1 vk_tier.log | grep -qx PASS || fail "routed-expert tier"
  ./tests/test_glm53_vk_f32
  COLI_VK_GEMM_MIN_S=0 ./tests/test_glm53_vk_f32
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
  PIN=tier.hist SNAP=tiny_inkling ./inkling 8 -p "The capital of France is" -n 8 > cpu.log 2>&1
  PIN=tier.hist COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 SNAP=tiny_inkling ./inkling 8 -p "The capital of France is" -n 8 > vk.log 2>&1
  $PY - cpu.log vk.log <<'PY' || { cat vk.log; fail "inkling tier warm start: the text differs from the CPU's"; }
import re, sys
def text(p):
    m = re.search(rb"\[\d+ prompt tokens\](.*?)\n\[prefill", open(p, "rb").read(), re.S)
    return m.group(1) if m else None
a, b = text(sys.argv[1]), text(sys.argv[2])
sys.exit(0 if a is not None and a == b else 1)
PY
  grep -qa '^\[VK\] tier inkling: warm start, [1-9]' vk.log || { cat vk.log; fail "inkling tier: no warm start"; }
  [ "$(tier_count inkling vk.log)" -gt 0 ] || { cat vk.log; fail "inkling tier warm start: no routed expert ran on the device"; }
  echo "OK inkling tier warm start: text = CPU, $(grep -a -o 'warm start, [0-9]* experts' vk.log), $(grep -a -o 'device [0-9]* of [0-9]* routed experts' vk.log | tail -1)"
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
  for c in window long image; do
    local x=(); [ $c = image ] && x=(--image mimo_tiny/patches.f32 --grid $MGRID)
    for ch in 64 1 3 8 9; do
      COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER=0 COLI_VK_GEMM_MIN_S=0 MIMO_DENSE_BITS=32 MIMO_CHUNK=$ch MIMO_LOGITS=blk$ch.f32 \
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
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER=0 COLI_VK_GEMM_MIN_S=0 $PY -m unittest tests.test_mimo_prefix_serve
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
  lost_gate qwen36 "chain qwen36 device lost" 40 COLI_DENSE_I8=0 SNAP=qwen36_tiny_c -- 8 8 qwen36_tiny/ref_full.json
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
  lost_gate qwen38 "chain qwen38 device lost" 20 OMP_NUM_THREADS=2 SNAP=qwen38_tiny -- 4 8 qwen38_tiny/ref.json
  lost_gate qwen38 "chain qwen38 device lost in a verify" 45 OMP_NUM_THREADS=2 Q38_MTP=1 Q38_MTP_FORCE=mixed SNAP=qwen38_tiny_mtp -- 2 8 qwen38_tiny_mtp/ref.json
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
  # experts between the device and the CPU alike)
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 INKLING_TINY=tiny_inkling \
    $PY -m unittest tests.test_inkling_prefix_serve tests.test_inkling_dashboard_hits
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 OLMOE_TINY=olmoe_tiny_c \
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
  # A pin restored over rows another branch rewrote: the KDA state goes up from the
  # pin, the MLA rows' watermark comes down. This check compares printed logprobs
  # exactly, so keep the per-row GEMV for every prompt length (as for MiMo's exact
  # prefix check). A cold prompt and a short resumed tail can otherwise straddle
  # the GEMM threshold, including at the now-device-resident f32 lm_head, and differ
  # in the last printed digit from reduction order alone. The gates above exercise
  # the default GEMM/GEMV policy against CPU tokens and the numerical tolerance.
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_GEMM_MIN_S=0 COLI_USAGE=$PWD/chain.usage \
    $PY tests/glm53_pin_branch_harness.py --binary ./glm53 --fixture glm53_mm_tiny
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
  inkling-olmoe-chain) family_inkling_olmoe_chain ;;
  inkling-olmoe-chain-sanitize) family_inkling_olmoe_chain_sanitize ;;
  glm-chain)      family_glm_chain ;;
  glm-chain-sanitize) family_glm_chain_sanitize ;;
  *) echo "usage: $0 shader|qwen|qwen-sanitize|inkling-olmoe|inkling-olmoe-sanitize|mimo-qwenimage|deepseek|deepseek-sanitize|kimi|kimi-mimo-sanitize|glm|glm-sanitize|qwen-chain|qwen-chain-sanitize|mimo-chain|mimo-chain-sanitize|inkling-olmoe-chain|inkling-olmoe-chain-sanitize|glm-chain|glm-chain-sanitize" >&2; exit 2 ;;
esac
