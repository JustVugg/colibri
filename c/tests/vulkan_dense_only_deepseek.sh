# Sourced by tests/vulkan_engines.sh: DeepSeek V4 (deepseek_v4) and DeepSeek V4.1 Flash
# (deepseek_v41) with their dense weights on the device only (COLI_VK_DENSE_HOST=0,
# docs/vulkan.md "Dense weights on the device only"), on the deepseek-chain family's
# fixtures and gates (v41_gate, v4_chain_gate, v4_chain_same: the CPU's tokens, the
# reference's, the logits within each engine's bound, the chain against itself), plus:
#   - the matrices went to the device only and none came back from disk on a healthy run;
#   - the same bits as the run that keeps its host copies (the device holds the same
#     tensors either way);
#   - a lost device: the CPU's tokens, the matrices read back from disk (deepseek_v4: its
#     dense layers read per forward from there on, as a low-memory plan does);
#   - deepseek_v4 under a RAM limit (--memory-gb) where the plan streams the dense layers
#     with the host copies and the chain stays off; on the device only the plan keeps them
#     there and the chain runs; and one where the experts' cache gets the dense layers' RAM;
#   - staged uploads, serve sessions, prompts only, the per-matrix path alone.

dho_v41_check() {  # <tag> <log> <lost 0|1>
  local n r; n=$(dho_dropped "$2"); r=$(dho_reloaded "$2")
  [ "$n" -gt 0 ] || { cat "$2"; fail "$1: no matrix went to the device only"; }
  if [ "$3" = 1 ]; then [ "$r" -gt 0 ] || { grep '^\[VK\]' "$2"; fail "$1: nothing was read back after the loss"; }
  else [ "$r" = 0 ] || { grep '^\[VK\]' "$2"; fail "$1: $r matrices read back from disk on a healthy run"; }; fi
  echo "   $n matrices on the device only, $r read back"
}
# dho_v41 <tag> <env...> -- <argv...>: v41_gate with the host copies dropped, then the
# same run keeping them (COLI_VK_DENSE_HOST=1): the same logits bit for bit (the tier's
# balancer off in both: it moves experts between the device and the CPU by measured
# times, and V4.1's two sides sum an expert in different orders). Not with prompts only
# (CHAINMODE=2): there the run with host copies decodes its trunk on the CPU (this
# device's default with the tier on), the run without them on the device.
dho_v41() {
  local tag=$1; shift
  # The production streaming threshold is timed. Fix it when comparing bits so
  # host-copy allocation timing cannot move experts between CPU and GPU arithmetic.
  local COLI_VK_TIER_STREAM_ROWS=16
  export COLI_VK_TIER_STREAM_ROWS
  v41_gate "$tag" COLI_VK_DENSE_HOST=0 COLI_VK_TIER_BALANCE=0 "$@"
  dho_v41_check "$tag" vk.log "$([ -n "${FAULT_BACK:-}" ] && echo 1 || echo 0)"
  if [ -z "${FAULT_BACK:-}" ] && [ "${CHAINMODE:-1}" = 1 ]; then
    local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    cp vk.f32 dho.f32
    rm -f chain.usage host.f32
    env "${envs[@]}" COLI_VK_DENSE_HOST=1 COLI_VK_TIER_BALANCE=0 DUMP=host.f32 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 \
      COLI_VK_CHAIN=${CHAINMODE:-1} ./deepseek_v41 "$@" > /dev/null 2> host.log || true
    cmp -s dho.f32 host.f32 || fail "$tag: the logits differ from the run that keeps its host copies"
    echo "   logits = the host-copy run's, bit for bit"
  fi
}
# the per-matrix path alone (no chain), the device's matrices still its own
dho_v41_matrix() {  # <tag> <env...> -- <argv...>
  local tag=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage cpu.f32 vk.f32
  local rc_cpu=0 rc_vk=0
  env "${envs[@]}" DUMP=cpu.f32 ./deepseek_v41 "$@" > cpu.txt 2> cpu.log || rc_cpu=$?
  env "${envs[@]}" COLI_VK_DENSE_HOST=0 COLI_VK_CHAIN=0 COLI_VK_DENSE=1 DUMP=vk.f32 COLI_USAGE=chain.usage \
    COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 ./deepseek_v41 "$@" > vk.txt 2> vk.log || rc_vk=$?
  [ "$rc_cpu" = "$rc_vk" ] && cmp -s cpu.txt vk.txt || { cat cpu.txt vk.txt; tail -20 vk.log; fail "$tag: tokens"; }
  local lg; lg=$(logits_close cpu.f32 vk.f32) || { echo "$lg"; fail "$tag: logits"; }
  dho_v41_check "$tag" vk.log 0 > /dev/null
  echo "OK $tag: tokens = CPU, $lg, $(vk_count deepseek_v41 vk.log) matmuls on the device, $(dho_dropped vk.log) matrices on it only"
}

dho_v4_check() {  # <tag> <log> <lost 0|1>
  local n r; n=$(dho_dropped "$2"); r=$(dho_reloaded "$2")
  [ "$n" -gt 0 ] || { cat "$2"; fail "$1: no matrix went to the device only"; }
  grep -q '^ram_tiers .*dense=device(' "$2" || { grep ram_tiers "$2"; fail "$1: the plan did not count the dense layers on the device"; }
  if [ "$3" = 1 ]; then
    [ "$r" -gt 0 ] || { grep '^\[VK\]' "$2"; fail "$1: nothing was read back after the loss"; }
    grep -q 'the device that held the dense layers is gone' "$2" || { grep '^\[VK\]' "$2"; fail "$1: the per-forward reads did not take over"; }
  else [ "$r" = 0 ] || { grep '^\[VK\]' "$2"; fail "$1: $r matrices read back from disk on a healthy run"; }; fi
  echo "   $n matrices on the device only, $r read back, $(grep -o 'dense=device([^)]*)' "$2" | head -1)"
}
dho_v4() {  # <tag> <fixture> <case> <env...>: v4_chain_gate on the device only
  local tag=$1 fx=$2 c=$3; shift 3
  v4_chain_gate "$tag" "$fx" "$c" COLI_VK_DENSE_HOST=0 "$@"
  dho_v4_check "$tag" v4-vk.err "$([ -n "${FAULT_BACK:-}" ] && echo 1 || echo 0)"
}
# the per-matrix path alone on the device only: ids = CPU = reference, logits within V4's bound
dho_v4_matrix() {  # <tag> <fixture> <case> <env...>
  local tag=$1 fx=$2 c=$3; shift 3
  local p mt
  p=$(v4_chain_prompt "$fx" "$c")
  mt=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]+"/ref.json"))["cases"][sys.argv[2]]["max_new_tokens"])' "$fx" "$c")
  rm -f "$fx/.coli_usage" cpu.f32 vk.f32
  env "$@" DUMP=cpu.f32 ./deepseek_v4 "./$fx" "$p" --raw-prompt --max-tokens "$mt" --record-oracle v4-cpu.json > /dev/null 2> v4-cpu.err || { cat v4-cpu.err; fail "$tag: CPU"; }
  rm -f "$fx/.coli_usage"
  env "$@" COLI_VK_DENSE_HOST=0 COLI_VK_CHAIN=0 COLI_VK_DENSE=1 DUMP=vk.f32 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 \
    ./deepseek_v4 "./$fx" "$p" --raw-prompt --max-tokens "$mt" --record-oracle v4-vk.json > /dev/null 2> v4-vk.err || { tail -20 v4-vk.err; fail "$tag: device"; }
  rm -f "$fx/.coli_usage"
  $PY -c 'import json,sys; a=json.load(open("v4-cpu.json")); b=json.load(open("v4-vk.json")); r=json.load(open(sys.argv[1]+"/ref.json"))["cases"][sys.argv[2]]["greedy_full_ids"]; sys.exit(not (a["full_ids"]==b["full_ids"]==r))' "$fx" "$c" ||
    fail "$tag: the ids differ from the CPU's"
  local lg; lg=$(v4_chain_logits cpu.f32 vk.f32 "$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]+"/config.json"))["vocab_size"])' "$fx")") || { echo "$lg"; fail "$tag: logits"; }
  [ "$(vk_count deepseek_v4 v4-vk.err)" -gt 0 ] || { cat v4-vk.err; fail "$tag: no matmul on the device"; }
  dho_v4_check "$tag" v4-vk.err 0 > /dev/null
  echo "OK $tag (${c}): ids = CPU = reference, $lg, $(vk_count deepseek_v4 v4-vk.err) matmuls on the device, $(dho_dropped v4-vk.err) matrices on it only"
}
# The RAM limit of the user's case (#1852), on the 8-expert fixture: at LOW GiB the plan
# with the host copies streams the dense layers (read per forward, the chain declines),
# on the device only it keeps them there and the chain runs; at MID GiB the host copies
# fit and take RAM the experts' cache gets on the device only (one slot more a layer).
# The two limits are this fixture's plan arithmetic (found by bisection on --memory-gb;
# the plan is exact integer arithmetic on the configured limit, no machine reading).
DHO_V4_LOW=0.500917917
DHO_V4_MID=0.501555663
dho_v4_lowram() {  # <tag> <GiB> <expect: stream|slots>
  local tag=$1 gib=$2 expect=$3 fx=deepseek_v4_tiny_e8 c=long
  local p mt side
  p=$(v4_chain_prompt "$fx" "$c")
  mt=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]+"/ref.json"))["cases"][sys.argv[2]]["max_new_tokens"])' "$fx" "$c")
  for side in cpu host dev; do
    rm -f "$fx/.coli_usage" "lr-$side.f32" "lr-$side.json"
    local e=()
    case $side in
      host) e=(COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_VK_DENSE_HOST=1) ;;
      dev) e=(COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_VK_DENSE_HOST=0) ;;
    esac
    env "${e[@]}" DUMP=lr-$side.f32 ./deepseek_v4 "./$fx" "$p" --raw-prompt --max-tokens "$mt" --memory-gb "$gib" \
      --record-oracle lr-$side.json > /dev/null 2> lr-$side.err || { tail -20 lr-$side.err; fail "$tag: $side run"; }
  done
  rm -f "$fx/.coli_usage"
  $PY -c 'import json,sys; r=json.load(open(sys.argv[1]+"/ref.json"))["cases"][sys.argv[2]]["greedy_full_ids"]; ids=[json.load(open("lr-%s.json"%s))["full_ids"] for s in ("cpu","host","dev")]; sys.exit(not all(i==r for i in ids))' "$fx" "$c" ||
    fail "$tag: the ids differ from the CPU's"
  local V; V=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]+"/config.json"))["vocab_size"])' "$fx")
  local lg; lg=$(v4_chain_logits lr-cpu.f32 lr-dev.f32 "$V") || { echo "$lg"; fail "$tag: logits"; }
  local sh sd; sh=$(sed -n 's/^ram_tiers .*target_slots=\([0-9]*\).*/\1/p' lr-host.err); sd=$(sed -n 's/^ram_tiers .*target_slots=\([0-9]*\).*/\1/p' lr-dev.err)
  grep -q '^ram_tiers .*dense=device(' lr-dev.err || { grep ram_tiers lr-dev.err; fail "$tag: the plan did not keep the dense layers on the device"; }
  [ "$(chain_count deepseek_v4 lr-dev.err)" -gt 0 ] || { grep '\[VK\]' lr-dev.err; fail "$tag: the chain did not run on the device only"; }
  if [ "$expect" = stream ]; then
    grep -q '^ram_tiers .*dense=streamed' lr-host.err || { grep ram_tiers lr-host.err; fail "$tag: the host-copy plan did not stream at this limit"; }
    grep -q 'deepseek_v4 chain: the dense layers are not resident' lr-host.err || { grep '\[VK\]' lr-host.err; fail "$tag: the host-copy chain did not decline"; }
    echo "OK $tag: --memory-gb $gib: host copies: dense=streamed, the chain declined, $sh slots; device only: $(grep -o 'dense=device([^)]*)' lr-dev.err), $(chain_count deepseek_v4 lr-dev.err) chain forwards, $sd slots; ids = CPU = reference, $lg"
  else
    grep -q '^ram_tiers .*dense=resident' lr-host.err || { grep ram_tiers lr-host.err; fail "$tag: the host copies did not fit at this limit"; }
    [ "$sd" -gt "$sh" ] || fail "$tag: the experts' cache got no more slots on the device only ($sh, $sd)"
    echo "OK $tag: --memory-gb $gib: host copies resident, $sh slots a layer; device only $sd slots a layer; ids = CPU = reference, $lg"
  fi
}
# A serve session's engine on the device only: the mode is on at READY (the frames
# themselves are v4_chain_serve's, below, which runs in the same environment).
dho_v4_serve_mode() {  # <fixture>
  rm -f "$1/.coli_usage"
  SERVE=1 SNAP=$1 CTX=128 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0 ./deepseek_v4 < /dev/null > /dev/null 2> sv.err || true
  rm -f "$1/.coli_usage"
  [ "$(dho_dropped sv.err)" -gt 0 ] && grep -q '^ram_tiers .*dense=device(' sv.err || { cat sv.err; fail "deepseek_v4 serve: not on the device only"; }
  echo "OK deepseek_v4 serve on the device only: $(dho_dropped sv.err) matrices, $(grep -o 'dense=device([^)]*)' sv.err)"
}

dho_family_deepseek() {
  export OMP_NUM_THREADS=2
  make deepseek_v41 VK=1
  make deepseek-v4 VK=1
  # ---- deepseek_v41
  v41_chain_fixtures
  local cap f
  for cap in 1 8; do dho_v41 "dense-only deepseek_v41 cap=$cap" SNAP=dsv41_tiny -- $cap dsv41_tiny/ref.json; done
  dho_v41 "dense-only deepseek_v41 40-token prompt" SNAP=dsv41_long -- 8 dsv41_long/ref.json
  for f in 2 5; do dho_v41 "dense-only deepseek_v41 DSpark spec=$f" SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=$f -- 8 dsv41_tiny/ref.json; done
  dho_v41 "dense-only deepseek_v41 prompt in chunks of 3" SNAP=dsv41_long COLI_VK_CHAIN_ROWS=3 -- 8 dsv41_long/ref.json
  dho_v41 "dense-only deepseek_v41 tier off" SNAP=dsv41_long COLI_VK_TIER=0 -- 8 dsv41_long/ref.json
  CHAINMODE=2 dho_v41 "dense-only deepseek_v41 prompts only" SNAP=dsv41_long -- 8 dsv41_long/ref.json
  COLI_VK_STAGED=1 dho_v41 "dense-only deepseek_v41 staged" SNAP=dsv41_long -- 8 dsv41_long/ref.json
  dho_v41_matrix "dense-only deepseek_v41 the per-matrix path alone" SNAP=dsv41_long -- 8 dsv41_long/ref.json
  FAULT_BACK=5 dho_v41 "dense-only deepseek_v41 device lost mid-decode" SNAP=dsv41_long -- 8 dsv41_long/ref.json
  FAULT_BACK=100 dho_v41 "dense-only deepseek_v41 device lost in the prompt" SNAP=dsv41_long COLI_VK_CHAIN_ROWS=7 -- 8 dsv41_long/ref.json
  CHAIN_SERVE_EXPECT='dense matrices on the device only' COLI_VK_DENSE_HOST=0 \
    $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=1
  # unset on a CPU device the host copies stay (the auto rule)
  rm -f chain.usage
  env COLI_USAGE=chain.usage COLI_VULKAN=1 COLI_VK_CHAIN=1 SNAP=dsv41_tiny ./deepseek_v41 8 dsv41_tiny/ref.json > /dev/null 2> vk.log || true
  grep -q 'deepseek_v41: dense weights on the device and in host RAM (a CPU device' vk.log && [ "$(dho_dropped vk.log)" = 0 ] ||
    { grep '^\[VK\]' vk.log; fail "deepseek_v41: the auto rule dropped the host copies on a CPU device"; }
  echo "OK dense-only deepseek_v41 auto on a CPU device: host copies kept"
  rm -f chain.usage chain-serve.usage dho.f32 host.f32 host.log
  # ---- deepseek_v4
  v4_chain_fixtures
  local c fx
  for c in short compressed long; do
    dho_v4 "dense-only deepseek_v4 4 experts" deepseek_v4_tiny_t $c
    dho_v4 "dense-only deepseek_v4 8 experts, pinned rows16" deepseek_v4_tiny_e8 $c
  done
  dho_v4 "dense-only deepseek_v4 2 output groups (wo_a per group)" deepseek_v4_tiny_g2 long
  dho_v4 "dense-only deepseek_v4 a window of 4, ratios 4 and 2" deepseek_v4_tiny_w4 long
  dho_v4 "dense-only deepseek_v4 an indexer of 64 heads of 128" deepseek_v4_tiny_ix long
  dho_v4 "dense-only deepseek_v4 tier off" deepseek_v4_tiny_t long COLI_VK_TIER=0
  dho_v4 "dense-only deepseek_v4 chain chunks of 3" deepseek_v4_tiny_e8 long COLI_VK_CHAIN_ROWS=3
  CHAINMODE=2 dho_v4 "dense-only deepseek_v4 prompts only" deepseek_v4_tiny_t long
  COLI_VK_STAGED=1 dho_v4 "dense-only deepseek_v4 staged" deepseek_v4_tiny_t long
  COLI_VK_STAGED=1 dho_v4 "dense-only deepseek_v4 staged, 2 output groups" deepseek_v4_tiny_g2 long
  local M=ids:30,31,32,33,34,35,60,61,30,31,32,33,34,35,36,37,38,39,40,41,30,31,32:12
  dho_v4 "dense-only deepseek_v4 n-gram drafts, a window of 4" deepseek_v4_tiny_w4 $M V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  dho_v4_matrix "dense-only deepseek_v4 the per-matrix path alone" deepseek_v4_tiny_t long
  dho_v4_matrix "dense-only deepseek_v4 the per-matrix path alone, 2 output groups" deepseek_v4_tiny_g2 long
  # the same bits as the run that keeps its host copies
  for fx in deepseek_v4_tiny_t deepseek_v4_tiny_g2; do
    v4_chain_same "dense-only deepseek_v4 host copies against device only, $fx" $fx long "COLI_VK_DENSE_HOST=1" "COLI_VK_DENSE_HOST=0"
  done
  FAULT_BACK=3 dho_v4 "dense-only deepseek_v4 device lost mid-decode" deepseek_v4_tiny_t long
  FAULT_BACK=40 dho_v4 "dense-only deepseek_v4 device lost in the prompt" deepseek_v4_tiny_t long COLI_VK_CHAIN_ROWS=7
  dho_v4_lowram "dense-only deepseek_v4 a low-RAM plan (#1852)" $DHO_V4_LOW stream
  dho_v4_lowram "dense-only deepseek_v4 the dense layers' RAM to the experts" $DHO_V4_MID slots
  dho_v4_serve_mode deepseek_v4_tiny_t
  COLI_VK_DENSE_HOST=0 v4_chain_serve deepseek_v4_tiny_t
  local t
  for t in test_deepseek_v4_prefix test_deepseek_v4_brio; do
    COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0 COLI_VK_TIER_SYNC=1 COLI_USAGE=$PWD/chain.usage \
      $PY tests/$t.py --binary "$PWD/deepseek_v4" --fixture "$PWD/deepseek_v4_tiny_t" > v4-test.log 2>&1 ||
      { cat v4-test.log; fail "deepseek_v4 $t on the device only"; }
    echo "OK deepseek_v4 $t on the device only: $(tail -1 v4-test.log)"
  done
  # unset on a CPU device the host copies stay
  rm -f deepseek_v4_tiny_t/.coli_usage
  COLI_VULKAN=1 COLI_VK_CHAIN=1 ./deepseek_v4 ./deepseek_v4_tiny_t "$(v4_chain_prompt deepseek_v4_tiny_t short)" --raw-prompt --max-tokens 2 > /dev/null 2> vk.log || true
  grep -q 'deepseek_v4: dense weights on the device and in host RAM (a CPU device' vk.log && [ "$(dho_dropped vk.log)" = 0 ] &&
    grep -q '^ram_tiers .*dense=resident' vk.log || { grep -E '^\[VK\]|ram_tiers' vk.log; fail "deepseek_v4: the auto rule dropped the host copies on a CPU device"; }
  echo "OK dense-only deepseek_v4 auto on a CPU device: host copies kept"
  rm -rf deepseek_v4_tiny_t deepseek_v4_tiny_e8 deepseek_v4_tiny_g2 deepseek_v4_tiny_w4 deepseek_v4_tiny_ix chain.usage
  rm -f cpu.f32 vk.f32 v4-*.f32 v4-*.json v4-*.err v4-tf.txt v4-test.log lr-*.f32 lr-*.json lr-*.err sv.err
  unset OMP_NUM_THREADS
}

# The same under ASan and UBSan (LTO off for deepseek_v4, as its chain family builds it):
# no diagnostic, the matrices on the device only, and after a loss the read-back.
dho_family_deepseek_sanitize() {
  local SAN="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  make clean >/dev/null 2>&1 || true
  make deepseek-v4-clean >/dev/null 2>&1 || true
  make deepseek_v41 VK=1 EXTRA_CFLAGS="$SAN"
  make deepseek-v4 VK=1 LTO=0 EXTRA_CFLAGS="$SAN" EXTRA_LDFLAGS="$SAN"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2
  v41_chain_fixtures
  dsan41() {  # <tag> <lost 0|1> <env and argv...>
    local tag=$1 lost=$2; shift 2
    rm -f chain.usage
    env COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0 "$@" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
    dho_v41_check "$tag" san.log "$lost" > /dev/null
    echo "OK $tag: sanitizers clean, $(dho_dropped san.log) matrices on the device only, $(dho_reloaded san.log) read back"
  }
  dsan41 "asan dense-only deepseek_v41 40-token prompt" 0 SNAP=dsv41_long ./deepseek_v41 8 dsv41_long/ref.json
  dsan41 "asan dense-only deepseek_v41 DSpark spec=5" 0 SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=5 ./deepseek_v41 1 dsv41_tiny/ref.json
  dsan41 "asan dense-only deepseek_v41 prompts only" 0 COLI_VK_CHAIN=2 SNAP=dsv41_long ./deepseek_v41 8 dsv41_long/ref.json
  dsan41 "asan dense-only deepseek_v41 device lost" 1 COLI_VK_CHAIN_FAULT=40 SNAP=dsv41_long ./deepseek_v41 8 dsv41_long/ref.json
  CHAIN_SERVE_EXPECT='dense matrices on the device only' COLI_VK_DENSE_HOST=0 \
    $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=1 > san.log 2>&1 || { cat san.log; fail "asan dense-only deepseek_v41 serve"; }
  echo "OK asan dense-only deepseek_v41 serve: $(tail -1 san.log | cut -c1-120)"
  v4_chain_fixtures
  local p; p=$(v4_chain_prompt deepseek_v4_tiny_t long)
  v4_chain_san "asan dense-only deepseek_v4 long" COLI_VK_DENSE_HOST=0 ./deepseek_v4 ./deepseek_v4_tiny_t "$p" --raw-prompt --max-tokens 4 --record-oracle san.json
  dho_v4_check "asan dense-only deepseek_v4 long" san.log 0
  v4_chain_san "asan dense-only deepseek_v4 2 output groups, prompts only" COLI_VK_DENSE_HOST=0 COLI_VK_CHAIN=2 ./deepseek_v4 ./deepseek_v4_tiny_g2 "$(v4_chain_prompt deepseek_v4_tiny_g2 long)" --raw-prompt --max-tokens 4
  dho_v4_check "asan dense-only deepseek_v4 2 output groups, prompts only" san.log 0
  FAULT_BACK=10 v4_chain_san "asan dense-only deepseek_v4 device lost" COLI_VK_DENSE_HOST=0 ./deepseek_v4 ./deepseek_v4_tiny_t "$p" --raw-prompt --max-tokens 4
  dho_v4_check "asan dense-only deepseek_v4 device lost" san.log 1
  v4_chain_san "asan dense-only deepseek_v4 a low-RAM plan" COLI_VK_DENSE_HOST=0 ./deepseek_v4 ./deepseek_v4_tiny_e8 "$(v4_chain_prompt deepseek_v4_tiny_e8 long)" --raw-prompt --max-tokens 4 --memory-gb $DHO_V4_LOW
  dho_v4_check "asan dense-only deepseek_v4 a low-RAM plan" san.log 0
  COLI_VK_DENSE_HOST=0 v4_chain_serve deepseek_v4_tiny_t > san.log 2>&1 || { cat san.log; fail "asan dense-only deepseek_v4 served"; }
  echo "OK asan dense-only deepseek_v4 served: $(tail -1 san.log | cut -c1-120)"
  rm -rf deepseek_v4_tiny_t deepseek_v4_tiny_e8 deepseek_v4_tiny_g2 deepseek_v4_tiny_w4 deepseek_v4_tiny_ix san.json san.log chain.usage chain-serve.usage
  unset OMP_NUM_THREADS
  make deepseek-v4-clean >/dev/null 2>&1 || true
  make clean >/dev/null 2>&1 || true
}
