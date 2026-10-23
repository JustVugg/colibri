# Sourced by tests/vulkan_engines.sh: inkling, MiMo and Kimi K3 with their dense matrices
# on the device only (COLI_VK_DENSE_HOST=0, docs/vulkan.md "Dense weights on the device
# only"), run as `bash tests/vulkan_engines.sh dense-only-inkling-mimo-kimi` (and
# `...-sanitize`). Each configuration against the CPU's tokens and against the run that
# keeps its host copies (the same device copies: the logits are its bytes); a lost device
# reads the matrices back from disk and the CPU's tokens follow; serve sessions frame for
# frame; staged uploads; the resident set and the expert cache where the engine sizes it.

# mimo_dho_gate <tag> <case> <env...>: MiMo's CLI (mimo_tiny, --ids, its text on stdout),
# on the CPU, keeping the host copies and on the device only; the text the CPU's, the
# logits of the greedy stream (MIMO_LOGITS) the host-copy run's bytes, or with prompts
# only (CHAINMODE=2: decode on the per-matrix path, the device's on the device only and
# the CPU's with the host copies on this device) within 1e-4 of the CPU's.
mimo_dho_gate() {
  local tag=$1 c=$2 extra=() h n; shift 2
  [ "$c" = image ] && extra=(--image mimo_tiny/patches.f32 --grid $MGRID)
  local vk=(COLI_TEMP=0 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1} COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VK_TIER_STREAM_ROWS=16)
  # A dropped trunk also enables per-matrix GPU work, including the vision tower.
  # Keep that arithmetic identical in both arms of this byte-exact comparison.
  if [ "${CHAINMODE:-1}" = 0 ] || [ "$c" = image ]; then vk+=(COLI_VK_DENSE=1); fi
  env "$@" COLI_TEMP=0 ./mimo mimo_tiny --ids "$(mimo_ids $c prompt_ids)" --ngen 6 "${extra[@]}" > mimo-cpu.txt 2>/dev/null
  for h in 1 0; do
    env "$@" "${vk[@]}" COLI_VK_DENSE_HOST=$h ./mimo mimo_tiny --ids "$(mimo_ids $c prompt_ids)" --ngen 6 "${extra[@]}" \
      > mimo-vk$h.txt 2> mimo-vk$h.err
    { [ -s mimo-cpu.txt ] && cmp -s mimo-cpu.txt mimo-vk$h.txt; } || { cat mimo-cpu.txt mimo-vk$h.txt mimo-vk$h.err; fail "$tag $c: tokens differ from the CPU (COLI_VK_DENSE_HOST=$h)"; }
    rm -f host$h.f32
    env "$@" "${vk[@]}" COLI_VK_DENSE_HOST=$h MIMO_LOGITS=host$h.f32 \
      ./mimo mimo_tiny --ids "$(mimo_ids $c greedy_full_ids)" --ngen 0 "${extra[@]}" > /dev/null 2>> mimo-vk$h.err
  done
  local lg="logits = the host-copy run's"
  if [ "${CHAINMODE:-1}" = 2 ]; then
    rm -f cpu.f32
    env "$@" MIMO_LOGITS=cpu.f32 ./mimo mimo_tiny --ids "$(mimo_ids $c greedy_full_ids)" --ngen 0 "${extra[@]}" > /dev/null 2>&1
    lg=$(logits_close cpu.f32 host0.f32) || { echo "$lg"; fail "$tag $c: logits"; }
  else cmp -s host1.f32 host0.f32 || fail "$tag $c: the logits differ from the run that keeps its host copies"; fi
  n=$(dho_dropped mimo-vk0.err)
  [ "$n" -gt 0 ] || { cat mimo-vk0.err; fail "$tag $c: no dense matrix went to the device only"; }
  [ "$(dho_reloaded mimo-vk0.err)" = 0 ] || { grep '^\[VK\]' mimo-vk0.err; fail "$tag $c: a matrix was read back without a lost device"; }
  if [ "${CHAINMODE:-1}" != 0 ]; then [ "$(chain_count mimo mimo-vk0.err)" -gt 0 ] || { cat mimo-vk0.err; fail "$tag $c: the chain did not stay on"; }; fi
  echo "OK $tag $c: tokens = CPU, $lg, $n matrices on the device only"
}
# mimo_dho_lost <tag> <case> <back> <env...>: the device lost <back> frames before the end
mimo_dho_lost() {
  local tag=$1 c=$2 back=$3 extra=() frames r; shift 3
  [ "$c" = image ] && extra=(--image mimo_tiny/patches.f32 --grid $MGRID)
  local vk=(COLI_TEMP=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_VK_DENSE_HOST=0)
  env "$@" COLI_TEMP=0 ./mimo mimo_tiny --ids "$(mimo_ids $c prompt_ids)" --ngen 6 "${extra[@]}" > mimo-cpu.txt 2>/dev/null
  env "$@" "${vk[@]}" ./mimo mimo_tiny --ids "$(mimo_ids $c prompt_ids)" --ngen 6 "${extra[@]}" > /dev/null 2> mimo-vk.err
  frames=$(sed -n 's/^\[VK\] mimo chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p' mimo-vk.err | tail -1)
  [ "${frames:-0}" -gt "$back" ] || { cat mimo-vk.err; fail "$tag: the run had no frame $back before its end"; }
  env "$@" "${vk[@]}" COLI_VK_CHAIN_FAULT=$((frames - back)) ./mimo mimo_tiny --ids "$(mimo_ids $c prompt_ids)" --ngen 6 "${extra[@]}" \
    > mimo-vk.txt 2> mimo-vk.err
  { [ -s mimo-cpu.txt ] && cmp -s mimo-cpu.txt mimo-vk.txt; } || { cat mimo-cpu.txt mimo-vk.txt mimo-vk.err; fail "$tag: tokens differ from the CPU"; }
  grep -q "the device was lost at position [0-9]" mimo-vk.err || { cat mimo-vk.err; fail "$tag: the device was never lost"; }
  r=$(dho_reloaded mimo-vk.err)
  [ "$r" -gt 0 ] || { grep '^\[VK\]' mimo-vk.err; fail "$tag: nothing was read back for the CPU"; }
  echo "OK $tag: tokens = CPU after the loss, $r matrices read back from disk"
}

# k3_dho_gate <tag> <env...>: Kimi K3's CLI on the fixture's cases (K3C_CASES), on the CPU,
# keeping the host copies and on the device only: the CPU's tokens, the logits within the
# chain family's bound of the CPU's (k3c_close); the decode logits are the host-copy
# run's bytes. The earlier prompt rows are echoed by the CPU head with host copies,
# and by the device head without them: those retain the CPU-reference numeric gate.
# The last prompt row and every generated step use the chain head in both runs.
# RELOADS=n: the
# matrices a run without a lost device must read back (the CPU's MLA path reads kv_b's rows
# where the chain does not run: prompts only, the per-matrix path); BYTES=0: the logits held
# to the CPU's bound only (a forward the chain does not run multiplies on the device here
# and on the CPU with the host copies kept); FAULT_BACK=k: the device lost k frames before
# the end, the matrices read back and the state rebuilt.
k3_dho_gate() {
  local tag=$1 c frames fault h n r; shift
  local vk=(COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VK_TIER_STREAM_ROWS=16 COLI_VULKAN=1 COLI_VK_CHAIN=${CHAINMODE:-1})
  [ "${CHAINMODE:-1}" = 0 ] && vk+=(COLI_VK_DENSE=1)
  for c in ${K3C_CASES:-short long}; do
    rm -f k3c.usage cpu.f32 host1.f32 host0.f32
    env "$@" COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 K3_VAL_LOGITS=cpu.f32 ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids $c)" --ngen 8 \
      2> cpu.log | sed 's/ *TUNE.*//' > cpu.tok
    fault=""
    if [ -n "${FAULT_BACK:-}" ]; then
      env "$@" "${vk[@]}" COLI_VK_DENSE_HOST=0 COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 \
        ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids $c)" --ngen 8 2> vk.log > /dev/null
      frames=$(sed -n 's/^\[VK\] kimi_k3 chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p' vk.log | tail -1)
      [ -n "$frames" ] && [ "$frames" -gt "$FAULT_BACK" ] || { cat vk.log; fail "$tag $c: no fault-free run to count frames from"; }
      fault="COLI_VK_CHAIN_FAULT=$((frames - FAULT_BACK + 1))"
    fi
    for h in 1 0; do
      [ -n "$fault" ] && [ $h = 1 ] && continue
      rm -f k3c.usage
      env "$@" "${vk[@]}" $fault COLI_VK_DENSE_HOST=$h COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 K3_VAL_LOGITS=host$h.f32 \
        ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids $c)" --ngen 8 2> vk$h.log | sed 's/ *TUNE.*//' > vk$h.tok
      { [ -s cpu.tok ] && cmp -s cpu.tok vk$h.tok; } || { cat cpu.tok vk$h.tok; tail -20 vk$h.log; fail "$tag $c: tokens differ from the CPU (COLI_VK_DENSE_HOST=$h)"; }
    done
    n=$(dho_dropped vk0.log); r=$(dho_reloaded vk0.log)
    [ "$n" -gt 0 ] || { cat vk0.log; fail "$tag $c: no dense matrix went to the device only"; }
    local lg; lg=$(k3c_close cpu.f32 host0.f32) || { echo "$lg"; fail "$tag $c: logits"; }
    if [ -n "$fault" ]; then
      grep -q "kimi_k3 chain: the device was lost" vk0.log || { cat vk0.log; fail "$tag $c: no loss was handled"; }
      [ "$r" -gt 0 ] || { grep '^\[VK\]' vk0.log; fail "$tag $c: nothing was read back for the CPU"; }
      echo "OK $tag $c: tokens = CPU after the loss, $lg, $r matrices read back from disk"
      continue
    fi
    [ "$r" = "${RELOADS:-0}" ] || { grep '^\[VK\]' vk0.log; fail "$tag $c: $r matrices read back from disk, expected ${RELOADS:-0}"; }
    if [ "${BYTES:-1}" = 1 ]; then
      $PY - <<'PY' || fail "$tag $c: decode logits differ from the run that keeps its host copies"
import json
from pathlib import Path
vocab = json.loads(Path("kimi_k3_tiny/config.json").read_text())["vocab_size"]
steps = len(Path("vk0.tok").read_text().split())
kept, dropped = (Path(f"host{h}.f32").read_bytes() for h in (1, 0))
assert steps > 0 and len(kept) == len(dropped) and len(kept) % (4 * vocab) == 0
size = steps * vocab * 4
assert len(kept) >= size and kept[-size:] == dropped[-size:]
PY
      lg="$lg, decode logits = the host-copy run's"
    fi
    if [ "${CHAIN_EXPECT:-${CHAINMODE:-1}}" != 0 ]; then
      [ "$(chain_count kimi_k3 vk0.log)" -gt 0 ] || { cat vk0.log; fail "$tag $c: the chain did not stay on"; }
    else
      [ "$(chain_count kimi_k3 vk0.log)" = 0 ] || fail "$tag $c: the chain ran despite the CPU fallback"
      need_gpu kimi_k3 vk0.log "$tag $c per-matrix fallback"
    fi
    echo "OK $tag $c: tokens = CPU, $lg, $n matrices on the device only, $r read back"
  done
}
# The expert cache Kimi K3 sizes from RAM_GB counts the dropped host copies out: a budget
# set (from a probe run's own figures) between the plan with the host copies (cap 1) and
# the plan without them, and the cap the run takes is the larger.
k3_dho_cap() {
  local O="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0" line ram a b off
  rm -f k3c.usage
  # shellcheck disable=SC2086
  env $O COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0 \
    ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids short)" --ngen 2 > /dev/null 2> vk.log
  line=$(grep '^\[K3\] dense weights on the device only:' vk.log) || { cat vk.log; fail "kimi_k3 cap: no plan line"; }
  ram=$($PY - "$line" <<'PY'
import re, sys
d, r, s = (float(x) for x in re.search(r"only: ([0-9.]+) GB of host copies out of the resident ([0-9.]+) GB \(reserve ([0-9.]+) GB\)", sys.argv[1]).groups())
print(f"{r + s - d / 2:.6f}")
PY
)
  rm -f k3c.usage
  # shellcheck disable=SC2086
  env $O RAM_GB=$ram COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0 \
    ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids short)" --ngen 2 > /dev/null 2> vk.log
  line=$(grep '^\[K3\] dense weights on the device only:' vk.log) || { cat vk.log; fail "kimi_k3 cap: no plan line under RAM_GB=$ram"; }
  a=$(echo "$line" | sed -n 's/.*expert cache \([0-9]*\)\/layer instead of \([0-9]*\)$/\1/p')
  b=$(echo "$line" | sed -n 's/.*expert cache \([0-9]*\)\/layer instead of \([0-9]*\)$/\2/p')
  { [ -n "$a" ] && [ -n "$b" ] && [ "$a" -gt "$b" ]; } || { echo "$line"; fail "kimi_k3 cap: the cache did not grow with the host copies out"; }
  grep -q "expert cache $a/layer" vk.log || { grep '\[K3\]' vk.log; fail "kimi_k3 cap: the run did not take the larger cache"; }
  rm -f k3c.usage
  # shellcheck disable=SC2086
  env $O RAM_GB=$ram COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=1 \
    ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids short)" --ngen 2 > /dev/null 2> vk.log
  off=$(sed -n 's/.*expert cache \([0-9]*\)\/layer (.*/\1/p' vk.log | tail -1)
  [ "$off" = "$b" ] || { grep '\[K3\]' vk.log; fail "kimi_k3 cap: with the host copies kept the cache is $off, the plan said $b"; }
  rm -f k3c.usage
  echo "OK dense-only kimi_k3 expert cache under RAM_GB=$ram: $a/layer on the device only, $b/layer with the host copies"
}

# dho_gate for a bf16 snapshot. Where this CPU's bf16 dot rounds the activations
# (AVX512-BF16, as on part of the CI's runners), inkling keeps its bf16 matrices on the
# CPU on purpose ("bf16 stays on the CPU"): nothing goes to the device, so nothing is
# dropped, and the tokens must be the CPU's. Elsewhere the full dho_gate applies.
dho_gate_bf16() {  # <engine> <tag> <env...> -- <argv...>
  local eng=$1 tag=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage
  env "${envs[@]}" COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 \
    COLI_USAGE=chain.usage COLI_VK_DENSE_HOST=0 ./"$eng" "$@" > vk.log 2>&1 || true
  rm -f chain.usage
  if grep -qa 'bf16 stays on the CPU' vk.log; then
    env "${envs[@]}" ./"$eng" "$@" > cpu.log 2>&1 || true
    same_tokens cpu.log vk.log "$tag"
    [ "$(dho_dropped vk.log)" = 0 ] || { grep '^\[VK\]' vk.log; fail "$tag: bf16 stayed on the CPU, yet host copies were dropped"; }
    [ "$(dho_reloaded vk.log)" = 0 ] || { grep '^\[VK\]' vk.log; fail "$tag: a matrix was read back from disk"; }
    echo "OK $tag: tokens = CPU, nothing on the device only (this CPU's bf16 dot keeps bf16 on the CPU)"
  else
    dho_gate "$eng" "$tag" 2 "${envs[@]}" -- "$@"
  fi
}

dho_family_inkling_mimo_kimi() {
  export OMP_NUM_THREADS=2
  make inkling mimo kimi_k3 VK=1
  # ---- inkling: f32, the dense-int4g64 container (int8 and int4-g64), bf16, the expert
  # containers and runtime quantization, the tier off, chunks, prompts only, the
  # per-matrix path alone, D = 6144, staged, a lost device, a serve session
  inkling_olmoe_chain_fixtures
  local fx bits c R=tiny_inkling/ref_inkling.json W=tiny_inkling_wide/ref_inkling.json
  for fx in tiny_inkling tiny_inkling_q tiny_inkling_x-i4 tiny_inkling_x-i8; do
    dho_gate inkling "dense-only inkling $fx" 2 SNAP=$fx -- 8 0 $R
  done
  dho_gate_bf16 inkling "dense-only inkling tiny_inkling_bf16" SNAP=tiny_inkling_bf16 -- 8 0 $R
  dho_gate inkling "dense-only inkling runtime int8 experts" 2 SNAP=tiny_inkling -- 2 8 $R
  dho_gate inkling "dense-only inkling tier off" 2 COLI_VK_TIER=0 SNAP=tiny_inkling_q -- 8 0 $R
  dho_gate inkling "dense-only inkling prefill in chunks of 3" 2 COLI_VK_CHAIN_ROWS=3 SNAP=tiny_inkling -- 8 0 $R
  # prompts only: decode on the per-matrix path, the device's here and the CPU's with the
  # host copies kept on this device, so its logits are held to the CPU's within 1e-4
  CHAINMODE=2 dho_gate inkling "dense-only inkling prompts only" 1 SNAP=tiny_inkling -- 8 0 $R
  CHAINMODE=0 dho_gate inkling "dense-only inkling per-matrix path" 2 SNAP=tiny_inkling_q -- 8 0 $R
  dho_gate inkling "dense-only inkling D=6144" 2 SNAP=tiny_inkling_wide -- 8 0 $W
  dho_gate inkling "dense-only inkling staged" 2 COLI_VK_STAGED=1 SNAP=tiny_inkling_q -- 8 0 $R
  dho_lost inkling "dense-only inkling device lost" 20 SNAP=tiny_inkling -- 8 0 $R
  dho_lost inkling "dense-only inkling device lost, int4-g64 trunk" 19 SNAP=tiny_inkling_q -- 8 0 $R
  CHAIN_SERVE_EXPECT='dense matrices on the device only' COLI_VK_DENSE_HOST=0 \
    $PY tests/vulkan_chain_serve.py ./inkling tiny_inkling INK_PREFIX_LOG=1
  # the resident set at exit: the host copies of D = 6144's trunk are gone
  dho_rss_gate inkling "dense-only inkling D=6144 resident set" 80 SNAP=tiny_inkling_wide -- 8 0 $W
  # the auto cap measures RAM after the drop (here a CPU device: the device copy holds it)
  rm -f chain.usage
  env COLI_USAGE=chain.usage COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0 SNAP=tiny_inkling_wide ./inkling 0 0 $W > vk.log 2>&1 || true
  grep -q '^\[cap auto\] dense weights on the device only: RAM measured after 0\.[0-9]*[1-9][0-9]* GB of host copies went' vk.log ||
    { grep 'cap auto\|^\[VK\]' vk.log; fail "dense-only inkling auto cap: not measured after the drop"; }
  echo "OK dense-only inkling auto cap: $(grep -o 'RAM measured after.*' vk.log)"
  rm -f chain.usage

  # ---- mimo: every dense form, text and picture, the tower on the device, the tier off,
  # chunks, prompts only, the per-matrix path alone, staged, the vendor oracle, lost
  # devices, a serve session
  $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force --vision
  MGRID=$($PY -c "import json;i=json.load(open('mimo_tiny/ref.json'))['image'];print(i['grid_h'],i['grid_w'])")
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_VK_DENSE_HOST=0 $PY tests/mimo_tiny_harness.py --binary ./mimo --fixture ./mimo_tiny
  echo "OK dense-only mimo vendor oracle"
  for bits in 32 0 8; do for c in short long image; do
    mimo_dho_gate "dense-only mimo bits=$bits" $c MIMO_DENSE_BITS=$bits
  done; done
  mimo_dho_gate "dense-only mimo tower on the device" image MIMO_DENSE_BITS=0 COLI_VK_DENSE=1
  mimo_dho_gate "dense-only mimo tier off" long MIMO_DENSE_BITS=0 COLI_VK_TIER=0
  mimo_dho_gate "dense-only mimo chunks of 3 rows" long MIMO_DENSE_BITS=32 COLI_VK_CHAIN_ROWS=3
  mimo_dho_gate "dense-only mimo staged" image MIMO_DENSE_BITS=0 COLI_VK_STAGED=1
  CHAINMODE=2 mimo_dho_gate "dense-only mimo prompts only" long MIMO_DENSE_BITS=0
  CHAINMODE=0 mimo_dho_gate "dense-only mimo per-matrix path" image MIMO_DENSE_BITS=0
  mimo_dho_lost "dense-only mimo device lost mid-decode" long 3 MIMO_DENSE_BITS=0
  mimo_dho_lost "dense-only mimo device lost in a picture's prompt" image 33 MIMO_DENSE_BITS=8
  mimo_served_fixture
  CHAIN_SERVE_TOL=1e-3 CHAIN_SERVE_EXPECT='dense matrices on the device only' COLI_VK_DENSE_HOST=0 \
    $PY tests/vulkan_chain_serve.py ./mimo mimo_tiny_served OMP_NUM_THREADS=2

  # ---- kimi_k3: Moonshot's oracle, every dense format, the tier off, chunks, prompts
  # only and the per-matrix path alone (kv_b read back for the CPU's MLA path), a trace
  # (the chain declines), staged, lost devices, serve sessions, the expert cache
  $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  k3c_serve_fixture
  COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_VK_DENSE_HOST=0 \
    $PY tests/test_kimi_k3_tiny.py --binary ./kimi_k3 --fixture ./kimi_k3_tiny
  echo "OK dense-only kimi_k3 vendor oracle"
  local O="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0" b
  # shellcheck disable=SC2086
  {
    k3_dho_gate "dense-only kimi_k3 f32" $O
    for b in 8 4; do k3_dho_gate "dense-only kimi_k3 ${b}-bit trunk" K3_BITS=$b K3_MLA_BITS=$b K3_HEAD_BITS=$b K3_IDOT=0 COLI_TEMP=0; done
    k3_dho_gate "dense-only kimi_k3 int4 KDA and MoE, int8 MLA and head" K3_BITS=4 K3_IDOT=0 COLI_TEMP=0
    k3_dho_gate "dense-only kimi_k3 tier off" $O COLI_VK_TIER=0
    k3_dho_gate "dense-only kimi_k3 chain chunks of 3" $O COLI_VK_CHAIN_ROWS=3
    k3_dho_gate "dense-only kimi_k3 staged" K3_BITS=4 K3_IDOT=0 COLI_TEMP=0 COLI_VK_STAGED=1
    CHAINMODE=2 RELOADS=4 BYTES=0 k3_dho_gate "dense-only kimi_k3 prompts only" $O
    CHAINMODE=0 RELOADS=4 BYTES=0 k3_dho_gate "dense-only kimi_k3 per-matrix path" $O
    CHAIN_EXPECT=0 RELOADS=4 BYTES=0 k3_dho_gate "dense-only kimi_k3 K3_TRACE (the chain declines)" $O K3_TRACE=/dev/null
    FAULT_BACK=3 k3_dho_gate "dense-only kimi_k3 device lost mid-decode" $O
    K3C_CASES=long FAULT_BACK=77 k3_dho_gate "dense-only kimi_k3 device lost in a later prompt chunk" K3_BITS=4 K3_IDOT=0 COLI_TEMP=0 COLI_VK_CHAIN_ROWS=7
  }
  local S="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 K3_PREFIX_LOG=1 USAGE_SAVE=0"
  # shellcheck disable=SC2086
  {
    CHAIN_SERVE_TOL=2e-2 CHAIN_SERVE_EXPECT='dense matrices on the device only' COLI_VK_DENSE_HOST=0 \
      $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $S COLI_K3_CKPT=4
    CHAIN_SERVE_TOL=2e-2 CHAIN_SERVE_EXPECT='dense matrices on the device only' COLI_VK_DENSE_HOST=0 COLI_VK_CHAIN=2 \
      $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $S
  }
  k3_dho_cap
  rm -f k3c.usage chain.usage chain-serve.usage
  unset OMP_NUM_THREADS
}

# The same under ASan and UBSan: memory safety is the gate, each run with its host copies
# dropped (and, where the device is lost, read back).
dho_family_inkling_mimo_kimi_sanitize() {
  make clean >/dev/null 2>&1 || true
  make inkling mimo kimi_k3 VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 OMP_NUM_THREADS=2
  $PY tools/make_tiny_inkling.py tiny_inkling
  $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force --vision
  MGRID=$($PY -c "import json;i=json.load(open('mimo_tiny/ref.json'))['image'];print(i['grid_h'],i['grid_w'])")
  $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  k3c_serve_fixture
  dsan_run() {  # <tag> <lost 0|1> <env and argv...>
    local tag=$1 lost=$2; shift 2
    rm -f chain.usage
    env COLI_USAGE=chain.usage USAGE_SAVE=0 COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DENSE_HOST=0 "$@" > san.log 2>&1 || true
    if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan $tag: sanitizer diagnostic"; fi
    [ "$(dho_dropped san.log)" -gt 0 ] || { cat san.log; fail "asan $tag: nothing on the device only"; }
    if [ "$lost" = 1 ]; then
      grep -q 'queue submit (COLI_VK_CHAIN_FAULT)' san.log || { cat san.log; fail "asan $tag: device loss was not injected"; }
      [ "$(dho_reloaded san.log)" -gt 0 ] || { cat san.log; fail "asan $tag: nothing read back"; }
    fi
    echo "OK asan $tag: sanitizers clean, $(dho_dropped san.log) matrices on the device only, $(dho_reloaded san.log) read back"
  }
  local R=tiny_inkling/ref_inkling.json
  dsan_run "dense-only inkling f32" 0 SNAP=tiny_inkling ./inkling 8 0 $R
  dsan_run "dense-only inkling per-matrix path" 0 COLI_VK_CHAIN=0 COLI_VK_DENSE=1 SNAP=tiny_inkling ./inkling 8 0 $R
  dsan_run "dense-only inkling device lost" 1 COLI_VK_CHAIN_FAULT=150 SNAP=tiny_inkling ./inkling 8 0 $R
  dsan_run "dense-only mimo picture" 0 MIMO_DENSE_BITS=0 COLI_VK_DENSE=1 ./mimo mimo_tiny --ids "$(mimo_ids image prompt_ids)" --ngen 4 --image mimo_tiny/patches.f32 --grid $MGRID
  dsan_run "dense-only mimo device lost" 1 MIMO_DENSE_BITS=0 COLI_VK_CHAIN_FAULT=20 ./mimo mimo_tiny --ids "$(mimo_ids long prompt_ids)" --ngen 4
  local O=(K3_BITS=4 K3_IDOT=0 COLI_TEMP=0) frames
  dsan_run "dense-only kimi_k3 int4" 0 "${O[@]}" ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids long)" --ngen 4
  # Chunk sizing changes the submission count. Inject during the last decode of
  # this exact configuration instead of assuming it reaches a fixed frame.
  frames=$(sed -n 's/^\[VK\] kimi_k3 chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p' san.log | tail -1)
  [ "${frames:-0}" -gt 3 ] || { cat san.log; fail "asan dense-only kimi_k3: too few frames for device-loss test"; }
  dsan_run "dense-only kimi_k3 prompts only" 0 "${O[@]}" COLI_VK_CHAIN=2 ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids long)" --ngen 4
  dsan_run "dense-only kimi_k3 device lost" 1 "${O[@]}" COLI_VK_CHAIN_FAULT=$((frames - 2)) ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids long)" --ngen 4
  CHAIN_SERVE_TOL=2e-2 CHAIN_SERVE_EXPECT='dense matrices on the device only' COLI_VK_DENSE_HOST=0 \
    $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 USAGE_SAVE=0 > san.log 2>&1 ||
    { cat san.log; fail "asan dense-only kimi_k3 serve"; }
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "asan dense-only kimi_k3 serve: sanitizer diagnostic"; fi
  echo "OK asan dense-only kimi_k3 serve: $(tail -1 san.log)"
  rm -f chain.usage chain-serve.usage
  unset OMP_NUM_THREADS
  make clean >/dev/null 2>&1 || true
}
