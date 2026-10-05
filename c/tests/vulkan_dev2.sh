# Sourced by tests/vulkan_engines.sh: the routed-expert tier on two devices (COLI_VK_DEV2,
# docs/vulkan.md, "A second device") on every MoE engine. Lavapipe is opened twice
# (COLI_VK_DEV2=0: a second logical device on the same physical one, the test mode), and
# the primary device's tier is held to a few experts (COLI_VK_TIER_GB, or the engine's
# own count), so the second device holds the rest.
#
#   dev2                       qwen36, qwen38, olmoe, inkling, colibri (GLM-5.2), glm53
#   dev2-deepseek-kimi-mimo    deepseek_v41, deepseek_v4, kimi_k3, mimo
#   dev2-sanitize              the shared code under ASan and UBSan: olmoe, deepseek_v4,
#                              kimi_k3, mimo
#
# Every engine runs four cases against its own CPU run (d2_case):
#   decode   the CPU's tokens; the tier took the second device and that device served
#            rows (d2_check);
#   big      a warm start, then prompt steps as big steps: a Vulkan run writes the
#            history, the next one fills both devices from it before the prompt (the
#            hottest on the primary device), and with COLI_VK_TIER_GEMM_ROWS=2 (from two
#            rows a step streams; the primary device's budget four times larger, so it
#            has room for streaming slots) the second device's batch ran beside the
#            primary device's sub-batches (MiMo keeps no history: its prompt goes in
#            blocks of four rows instead, MIMO_CHUNK=4, and what the first blocks
#            promoted to the second device meets the next ones there);
#   evict    the second device full too (COLI_VK_EXPERTS2=2, or D2_E2): evictions, no
#            failed upload;
#   fault    the second device's second batch fails (COLI_VK_DEV2_FAULT=2): the CPU's
#            tokens, the second device stopped and its experts went back to the CPU,
#            the primary device still served;
#   excl     exclusive RAM/VRAM (COLI_VK_TIER_EXCLUSIVE, on by default) with a RAM expert
#            cache of a slot or two (D2_SMALL=1 in the runners) and the second device held
#            to a few experts, so the CPU still computes after the prompt: the CPU's tokens,
#            and the RAM cache gave up first slots whose expert the devices hold;
#   noexcl   the same with COLI_VK_TIER_EXCLUSIVE=0: the CPU's tokens, nothing given up.
# The other runs start without history (a fresh usage file): the experts reach the
# devices as the routing passes them. In the runners D2_HIST=1 keeps the history there
# is, D2_KEEP=1 has the run write one and leaves it for the next.

# d2_cap <usual> <small>: a runner's RAM expert cache, a slot or two in the excl cases
d2_cap() { [ "${D2_SMALL:-0}" = 1 ] && echo "$2" || echo "$1"; }
# d2_gb <GiB>: the primary device's budget for this case (D2_X times it: the big case)
d2_gb() { awk -v g="$1" -v x="${D2_X:-1}" 'BEGIN { printf "%.9f", g * x }'; }

# d2_line <engine> <log>: the second device's part of the last "[VK] tier <engine> run:" line
d2_line() { sed -n "s/^\[VK\] tier $1 run: .* | second device \(.*\)/\1/p" "$2" | tail -1; }
# d2_check <engine> <log> <tag>: the tier took the second device, which served rows
d2_check() {
  local eng=$1 log=$2 tag=$3 l n
  grep -qa "^\[VK\] tier $eng: second device .*, budget .* = [1-9][0-9]* experts" "$log" ||
    { grep -a '\[VK\]' "$log"; fail "$tag: the tier did not take the second device"; }
  l=$(d2_line "$eng" "$log")
  n=$(echo "$l" | sed -n 's/.*), \([0-9][0-9]*\) rows in .*/\1/p')
  [ "${n:-0}" -gt 0 ] || { grep -a '\[VK\] tier' "$log"; fail "$tag: the second device served no row"; }
}
# d2_case <engine> <runner> <tag> <decode|big|evict|fault> <env...>
# <runner> <tokens out> <log> <env...> runs the engine once and writes what its CPU and
# Vulkan runs must share (tokens, ids) to <tokens out>.
d2_case() {
  local eng=$1 run=$2 tag=$3 kind=$4; shift 4
  local x=()
  local D2_X=1 D2_SMALL=0
  case $kind in
    big)   x=(COLI_VK_TIER_GEMM_ROWS=2); D2_X=4 ;;
    evict) x=(COLI_VK_EXPERTS2=${D2_E2:-2}) ;;
    fault) x=(COLI_VK_DEV2_FAULT=2) ;;
    # the devices short of the experts too (COLI_VK_EXPERTS2): with room for all of them,
    # everything routed after the prompt is on a device and the CPU reads nothing more
    excl)   D2_SMALL=1; x=(COLI_VK_EXPERTS2=${D2_E2:-8}) ;;
    noexcl) D2_SMALL=1; x=(COLI_VK_EXPERTS2=${D2_E2:-8} COLI_VK_TIER_EXCLUSIVE=0) ;;
  esac
  $run cpu.tok cpu.log "$@"
  local D2_HIST=0 D2_KEEP=0
  if [ $kind = big ]; then   # the history the warm start fills both devices from
    D2_KEEP=1 $run hist.tok hist.log "$@" "${x[@]}" COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 COLI_VK_DEV2=0
    D2_HIST=1
  fi
  $run vk.tok vk.log "$@" "${x[@]}" COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 COLI_VK_DEV2=0
  { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; tail -30 vk.log; fail "$tag $kind: tokens differ from the CPU"; }
  [ "$(tier_count "$eng" vk.log)" -gt 0 ] || { cat vk.log; fail "$tag $kind: no routed expert ran on the devices"; }
  local l; l=$(d2_line "$eng" vk.log)
  case $kind in
    decode) d2_check "$eng" vk.log "$tag $kind" ;;
    big)
      d2_check "$eng" vk.log "$tag $kind"
      [ "$eng" = mimo ] || grep -qaE "^\[VK\] tier $eng: warm start, [1-9]" vk.log || { grep -a '\[VK\]' vk.log; fail "$tag big: no warm start"; }
      echo "$l" | grep -qE '\(([1-9][0-9]*) beside prompt steps\)' ||
        { grep -a '\[VK\] tier' vk.log; fail "$tag big: no batch of the second device beside a prompt step"; } ;;
    evict)
      d2_check "$eng" vk.log "$tag $kind"
      echo "$l" | grep -qE ": resident [0-${D2_E2:-2}] \\(budget ${D2_E2:-2}," || { grep -a '\[VK\] tier' vk.log; fail "$tag evict: the second device is not held to ${D2_E2:-2} experts"; }
      [ "${D2_NOEVICT:-0}" = 1 ] || [ "$(tier_evictions "$eng" vk.log)" -gt 0 ] || { grep -a '\[VK\] tier' vk.log; fail "$tag evict: no eviction"; }
      [ "$(tier_failed "$eng" vk.log)" = 0 ] || { grep -a '\[VK\] tier' vk.log; fail "$tag evict: an upload failed"; } ;;
    excl)
      d2_check "$eng" vk.log "$tag $kind"
      local g; g=$(sed -n "s/^\[VK\] tier $eng run: .* | exclusive: \([0-9]*\) RAM copies.*/\1/p" vk.log | tail -1)
      [ "${D2_NOEVICT:-0}" = 1 ] || [ "${g:-0}" -gt 0 ] || { grep -a '\[VK\] tier' vk.log; fail "$tag excl: the RAM cache gave up no copy of a device expert"; }
      l="$l; $g RAM copies of device experts given up first" ;;
    noexcl)
      ! grep -qa "^\[VK\] tier $eng run: .* | exclusive:" vk.log || { grep -a '\[VK\] tier' vk.log; fail "$tag noexcl: COLI_VK_TIER_EXCLUSIVE=0 gave copies up"; } ;;
    fault)
      grep -qa "^\[VK\] tier $eng: a batch on the second device failed, its experts go back to the CPU" vk.log &&
        echo "$l" | grep -q '(stopped)$' || { grep -a '\[VK\]' vk.log; fail "$tag fault: the second device did not stop"; } ;;
  esac
  echo "OK $tag $kind: tokens = CPU, $(grep -a -o 'device [0-9]* of [0-9]* routed experts' vk.log | tail -1); second device $l"
}
# ---- runners: <tokens out> <log> <env...> ---------------------------------------------
# d2_fresh <file>: a run without history (unless D2_HIST=1)
d2_fresh() { [ "${D2_HIST:-0}" = 1 ] || rm -f "$1"; }
# d2_done <file>: the history goes after the run (unless D2_KEEP=1)
d2_done() { [ "${D2_KEEP:-0}" = 1 ] || rm -f "$1"; }
# d2_save: USAGE_SAVE for engines that save only when asked (D2_KEEP=1)
d2_save() { [ "${D2_KEEP:-0}" = 1 ] && echo 1 || echo 0; }

d2_qwen36() {
  local tok=$1 log=$2; shift 2; d2_fresh d2.usage
  env COLI_DENSE_I8=0 SNAP=qwen36_tiny_c COLI_USAGE=d2.usage COLI_VK_TIER_GB=$(d2_gb 0.00002) "$@" \
    ./qwen36 $(d2_cap 8 2) 8 qwen36_tiny/ref_full.json > "$log" 2>&1 || true
  grep -a '^C engine' "$log" > "$tok" || true
}
d2_qwen38() {
  local tok=$1 log=$2; shift 2; d2_fresh d2.usage
  # the excl cases prefill a token at a time: batched, every expert reaches the devices
  # within the prompt and the decode reads none into RAM, so nothing is evicted
  env OMP_NUM_THREADS=2 Q38_PREFILL_BATCH=$(d2_cap 1 0) SNAP=qwen38_tiny_fp8 COLI_USAGE=d2.usage COLI_VK_TIER_GB=$(d2_gb 0.0000065) "$@" \
    ./qwen38 1 8 qwen38_tiny_fp8/ref.json > "$log" 2>&1 || true
  grep -a '^C engine' "$log" > "$tok" || true
}
d2_olmoe() {
  local tok=$1 log=$2; shift 2; d2_fresh d2.usage
  env SNAP=olmoe_tiny_c COLI_USAGE=d2.usage COLI_VK_TIER_GB=$(d2_gb 0.000025) "$@" \
    ./olmoe $(d2_cap 8 2) 8 olmoe_tiny/ref_olmoe.json > "$log" 2>&1 || true
  grep -a '^C engine' "$log" > "$tok" || true
}
d2_inkling() {   # the ref.json oracle reads no history: the big case generates text with PIN=<history>
  local tok=$1 log=$2; shift 2
  if [ "${D2_X:-1}" = 1 ]; then
    d2_fresh d2.usage
    env SNAP=tiny_inkling COLI_USAGE=d2.usage COLI_VK_TIER_GB=$(d2_gb 0.00006) "$@" \
      ./inkling $(d2_cap 8 2) 0 tiny_inkling/ref_inkling.json > "$log" 2>&1 || true
    grep -a '^C engine' "$log" > "$tok" || true
    return
  fi
  d2_fresh d2.hist
  env SNAP=tiny_inkling PIN=d2.hist COLI_VK_TIER_GB=$(d2_gb 0.00006) "$@" \
    ./inkling 8 -p "The capital of France is" -n 8 > d2-ink.out 2> "$log" || true
  d2_done d2.hist
  $PY - d2-ink.out > "$tok" <<'PY'
import re, sys
m = re.search(rb"\[\d+ prompt tokens\](.*?)\n\[prefill", open(sys.argv[1], "rb").read(), re.S)
sys.stdout.write(repr(m.group(1)) + "\n" if m else "")   # no text: an empty file, which the gate refuses
PY
}
d2_colibri() {   # the primary device held at two experts by the engine's own cap
  local tok=$1 log=$2 st=(); shift 2; d2_fresh glm_tiny_i4r/.coli_usage
  [ "${D2_KEEP:-0}" = 1 ] && st=(STATS=glm_tiny_i4r/.coli_usage)
  [ "${D2_SMALL:-0}" = 1 ] && st+=(CAP_RAISE=0)   # the excl cases: the cache stays at the slots asked for
  env SNAP=glm_tiny_i4r REF=glm_tiny_i4r/ref_glm.json IDOT=0 COLI_VK_EXPERTS=2 "${st[@]}" "$@" ./colibri $(d2_cap 64 2) 4 4 > "$log" 2>&1 || true
  d2_done glm_tiny_i4r/.coli_usage
  grep -aE '^GLM C engine|^PREFILL|^\[ORACLE\] mismatch' "$log" | sed 's/ | [0-9.]* pos\/s//' > "$tok" || true
}
d2_glm53() {   # its fixture routes four experts in all: the big case's budget twice, not four times
  local tok=$1 log=$2; shift 2; d2_fresh d2.usage
  local D2_X=$(( ${D2_X:-1} > 2 ? 2 : ${D2_X:-1} ))
  local sm=(); [ "${D2_SMALL:-0}" = 1 ] && sm=(GLM53_EXPERT_GB=0.000001)   # one cache slot a layer
  env GLM53_BITS=32 COLI_USAGE=d2.usage USAGE_SAVE=$(d2_save) COLI_VK_TIER_GB=$(d2_gb 0.00006) "${sm[@]}" "$@" \
    ./glm53 --model glm53_stream-i4 --ids "$D2_G53_IDS" --greedy 6 > "$log" 2>&1 || true
  grep -aE '^teacher_forcing|^greedy' "$log" > "$tok" || true
}
d2_v41() {   # the engine exits non-zero when it misses its reference
  local tok=$1 log=$2; shift 2; d2_fresh d2.usage
  env SNAP=dsv41_long COLI_USAGE=d2.usage COLI_VK_TIER_GB=$(d2_gb 0.00005) "$@" ./deepseek_v41 $(d2_cap 8 2) dsv41_long/ref.json > "$tok" 2> "$log" ||
    { cat "$log"; fail "deepseek_v41 $*: misses its reference"; }
}
d2_v4() {   # ids and teacher-forced predictions of the long case; equal to the reference's
  # the excl cases: 16 experts a layer in a RAM cache of 6 (COLI_V4_EXPERT_SLOTS), so it evicts
  local tok=$1 log=$2 fx=deepseek_v4_tiny_e8 p=$D2_V4_PROMPT mt=$D2_V4_MT sm=(); shift 2
  [ "${D2_SMALL:-0}" = 1 ] && { fx=deepseek_v4_tiny_e16; p=$D2_V4_PROMPT16; mt=$D2_V4_MT16; sm=(COLI_V4_EXPERT_SLOTS=6); }
  d2_fresh $fx/.coli_usage
  env COLI_VK_TIER_GB=$(d2_gb 0.00009) "${sm[@]}" "$@" ./deepseek_v4 ./$fx "$p" --raw-prompt --max-tokens "$mt" \
    --record-oracle d2-v4.json > /dev/null 2> "$log" || { cat "$log"; fail "deepseek_v4 $*: run"; }
  d2_done $fx/.coli_usage
  $PY - $fx > "$tok" <<'PY'
import json, sys
a = json.load(open("d2-v4.json"))
ref = json.load(open(sys.argv[1] + "/ref.json"))["cases"]["long"]["greedy_full_ids"]
print(a["full_ids"], a["tf_pred"], a["full_ids"] == ref)
PY
}
d2_kimi() {
  local tok=$1 log=$2; shift 2; d2_fresh k3.usage
  env K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0 COLI_USAGE=$PWD/k3.usage USAGE_SAVE=$(d2_save) \
    COLI_VK_TIER_GB=$(d2_gb 0.000005) $([ "${D2_SMALL:-0}" = 1 ] && echo K3_EXPERT_GB=0.000001) "$@" ./kimi_k3 kimi_k3_tiny --ids "$D2_K3_IDS" --ngen 8 2> "$log" | sed 's/ *TUNE.*//' > "$tok" || true
}
d2_mimo() {   # the primary device held at two experts (eight in the big case) by the engine's own count
  local tok=$1 log=$2 ch=(); shift 2; d2_fresh d2.usage
  [ "${D2_X:-1}" = 1 ] || ch=(MIMO_CHUNK=4)   # the big case: the prompt in blocks
  [ "${D2_SMALL:-0}" = 1 ] && ch+=(MIMO_CAP=8)   # the excl cases: a small RAM cache
  env MIMO_DENSE_BITS=32 COLI_TEMP=0 COLI_USAGE=d2.usage MIMO_VK_EXPERTS=$((2 * ${D2_X:-1})) "${ch[@]}" "$@" \
    ./mimo mimo_tiny --ids "$D2_MIMO_IDS" --ngen 6 > "$tok" 2> "$log" || true
}

# ---- fixtures ---------------------------------------------------------------------------
d2_fixtures_main() {
  rm -rf olmoe_tiny olmoe_tiny_c
  $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
  $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8
  $PY tools/make_qwen38_tiny.py --out qwen38_tiny_fp8 --fp8-experts
  $PY tools/make_olmoe_tiny.py --output olmoe_tiny
  $PY tools/convert_olmoe_merged.py --model olmoe_tiny --out olmoe_tiny_c
  $PY tools/make_tiny_inkling.py tiny_inkling
  $PY -c 'import sys; from pathlib import Path; sys.path.insert(0, "."); from tests.test_inkling_prefix_serve import ensure_tokenizer; ensure_tokenizer(Path("tiny_inkling"))'
  mkdir -p glm_fp8 && (cd glm_fp8 && $PY ../tools/make_glm_oracle.py --fp8 > /dev/null)
  $PY tools/convert_fp8_to_int4.py --indir glm_fp8/glm_tiny --outdir glm_tiny_i4r \
    --ebits 4 --io-bits 4 --n-layers 5 --min-free-gb 0 --group-size 0 > /dev/null
  cp glm_fp8/ref_glm.json glm_tiny_i4r/
  $PY tools/make_glm53_tiny.py --output glm53_tiny --force > /dev/null
  $PY tools/make_glm53_streaming_pair.py --fixture glm53_tiny --output glm53_stream > /dev/null
  D2_G53_IDS=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
}
d2_fixtures_dkm() {
  $PY tools/make_dsv41_tiny.py --out dsv41_long --emit-ref dsv41_long/ref.json --prompt-len 40 --max-new 6
  local n
  for n in 8 16; do
  $PY - tools/make_deepseek_v4_tiny.py deepseek_v4_tiny_e$n $n <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
gen.EXPERTS = int(sys.argv[3])   # 8 routed experts a layer: room for two devices' worth; 16 for the excl cases
sys.argv = ["make_deepseek_v4_tiny.py", "--output", sys.argv[2], "--force"]
gen.main()
PY
  done
  D2_V4_PROMPT=$($PY -c 'import json; c=json.load(open("deepseek_v4_tiny_e8/ref.json"))["cases"]["long"]; print("".join("<t%03d>" % t for t in c["prompt_ids"]))')
  D2_V4_MT=$($PY -c 'import json; print(json.load(open("deepseek_v4_tiny_e8/ref.json"))["cases"]["long"]["max_new_tokens"])')
  D2_V4_PROMPT16=$($PY -c 'import json; c=json.load(open("deepseek_v4_tiny_e16/ref.json"))["cases"]["long"]; print("".join("<t%03d>" % t for t in c["prompt_ids"]))')
  D2_V4_MT16=$($PY -c 'import json; print(json.load(open("deepseek_v4_tiny_e16/ref.json"))["cases"]["long"]["max_new_tokens"])')
  $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  D2_K3_IDS=$($PY -c "import json;print(' '.join(map(str,json.load(open('kimi_k3_tiny/ref.json'))['cases']['long']['prompt_ids'])))")
  $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force
  D2_MIMO_IDS=$($PY -c "import json;print(' '.join(map(str,json.load(open('mimo_tiny/ref.json'))['cases']['long']['prompt_ids'])))")
}

family_dev2() {
  make qwen36 qwen38 olmoe inkling colibri glm53 VK=1
  d2_fixtures_main
  local k
  for k in decode big evict fault excl noexcl; do
    d2_case qwen36 d2_qwen36 "dev2 qwen36" $k
    d2_case qwen38 d2_qwen38 "dev2 qwen38" $k
    d2_case olmoe d2_olmoe "dev2 olmoe" $k
    d2_case inkling d2_inkling "dev2 inkling" $k
    d2_case colibri d2_colibri "dev2 colibri" $k
    # glm53's fixture routes four experts in all, and in six tokens none gets the heat to
    # displace another (LFRU's margin): its evict case checks the bound and the uploads.
    # Three on the devices leave one for the CPU, which its one RAM slot keeps: its excl
    # case has nothing to give up and checks the tokens and the second device
    D2_E2=1 D2_NOEVICT=1 d2_case glm53 d2_glm53 "dev2 glm53" $k
  done
  # (colibri's own registry on the second device, with the tier off, is the glm family's)
  # without COLI_VK_DEV2 nothing of the second device appears
  d2_olmoe vk.tok vk.log COLI_VULKAN=1 COLI_VK_TIER_SYNC=1
  ! grep -qa 'second device\|dev2' vk.log || { grep -a '\[VK\]' vk.log; fail "dev2: a second device without COLI_VK_DEV2"; }
  echo "OK dev2: without COLI_VK_DEV2 the tier keeps one device"
}

family_dev2_deepseek_kimi_mimo() {
  make deepseek_v41 kimi_k3 mimo VK=1
  make deepseek-v4 VK=1
  d2_fixtures_dkm
  local k
  for k in decode big evict fault excl noexcl; do
    d2_case deepseek_v41 d2_v41 "dev2 deepseek_v41" $k
    d2_case deepseek_v4 d2_v4 "dev2 deepseek_v4" $k
    grep -q ' True$' vk.tok || { cat vk.tok; fail "dev2 deepseek_v4 $k: ids differ from the reference"; }
    d2_case kimi_k3 d2_kimi "dev2 kimi_k3" $k
    d2_case mimo d2_mimo "dev2 mimo" $k
  done
}

# Memory safety of the two devices' paths: a sanitized build, the four cases' settings,
# no diagnostic; each run must still have used the second device (or, with the fault,
# stopped it).
family_dev2_sanitize() {
  local SAN="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  make clean >/dev/null 2>&1 || true
  make deepseek-v4-clean >/dev/null 2>&1 || true
  make olmoe kimi_k3 mimo VK=1 EXTRA_CFLAGS="$SAN"
  make deepseek-v4 VK=1 LTO=0 EXTRA_CFLAGS="$SAN" EXTRA_LDFLAGS="$SAN"
  rm -rf olmoe_tiny olmoe_tiny_c
  $PY tools/make_olmoe_tiny.py --output olmoe_tiny
  $PY tools/convert_olmoe_merged.py --model olmoe_tiny --out olmoe_tiny_c
  d2_fixtures_dkm
  local eng run k x
  for k in decode big evict fault excl; do
    for eng in olmoe:d2_olmoe deepseek_v4:d2_v4 kimi_k3:d2_kimi mimo:d2_mimo; do
      run=${eng#*:}; eng=${eng%%:*}
      local D2_X=1 D2_SMALL=0
      case $k in big) x=COLI_VK_TIER_GEMM_ROWS=2; D2_X=4 ;; evict) x=COLI_VK_EXPERTS2=2 ;; fault) x=COLI_VK_DEV2_FAULT=2 ;;
                  excl) x=COLI_VK_EXPERTS2=8; D2_SMALL=1 ;; *) x=D2=1 ;; esac
      local A=(ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
               OMP_NUM_THREADS=2 $x COLI_VULKAN=1 COLI_VK_TIER_SYNC=1 COLI_VK_DEV2=0)
      local D2_HIST=0
      [ $k = big ] && { D2_KEEP=1 $run san.tok san.log "${A[@]}"; D2_HIST=1; }   # the warm start's history
      $run san.tok san.log "${A[@]}"
      if grep -qaE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan dev2 $eng $k: sanitizer diagnostic"; fi
      if [ $k = excl ]; then
        grep -qaE "^\[VK\] tier $eng run: .* \| exclusive: [1-9]" san.log || { grep -a '\[VK\] tier' san.log; fail "asan dev2 $eng excl: no RAM copy given up"; }
      elif [ $k = fault ]; then
        grep -qa "^\[VK\] tier $eng: a batch on the second device failed" san.log || { grep -a '\[VK\]' san.log; fail "asan dev2 $eng fault: the second device did not stop"; }
      else
        d2_check "$eng" san.log "asan dev2 $eng $k"
      fi
      echo "OK asan dev2 $eng $k: sanitizers clean; second device $(d2_line "$eng" san.log)"
    done
  done
  make clean >/dev/null 2>&1 || true
}
