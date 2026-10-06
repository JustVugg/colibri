# Sourced by tests/vulkan_engines.sh: the dense chain's layers on two devices (docs/vulkan.md,
# "Layers on two devices"). Lavapipe is opened twice (COLI_VK_DEV2=0: a second logical device
# on the same physical one, the test mode); COLI_VK_CHAIN_LAYERS puts the first layers on
# the primary device, COLI_VK_CHAIN_LAYERS2 the next ones on the second, the CPU the rest.
#
#   layers-dev2                     qwen36, qwen38, olmoe, mimo, inkling against their CPU runs
#   layers-dev2-mla                 GLM-5.2 (colibri), GLM-5.3, Kimi K3
#   layers-dev2-deepseek            DeepSeek V4.1 and V4
#   layers-dev2[-mla|-deepseek]-sanitize   their paths under ASan and UBSan
#   layers-dev2-<engine>            one engine's gates (the engine built)
#
# Each gate (ld2_gate; ld2_mla for the MLA engines) is chain_gate's or mla_gate's (the
# CPU's tokens, logits within 1e-4 of the largest) with the split forced, and the second
# device's chain must have run with neither device lost.

# ld2_gate <engine> <tag> <n0> <n1> <env...> -- <argv...>
ld2_gate() {
  local eng=$1 tag=$2 n0=$3 n1=$4; shift 4
  chain_gate "$eng" "$tag" 1 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=$n0 COLI_VK_CHAIN_LAYERS2=$n1 COLI_VK_TIER_BALANCE=0 "$@"
  ld2_ran "$eng" "$tag" vk.log
}
# ld2_mla <engine> <tag> <n0> <n1> <env...> -- <argv...>
ld2_mla() {
  local eng=$1 tag=$2 n0=$3 n1=$4; shift 4
  mla_gate "$eng" "$tag" 1 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=$n0 COLI_VK_CHAIN_LAYERS2=$n1 COLI_VK_TIER_BALANCE=0 "$@"
  ld2_ran "$eng" "$tag" vk.log
}
# ld2_ran <engine> <tag> <log>: the second device's chain ran, no device was lost
ld2_ran() {
  local f; f=$(ld2_forwards "$1" "$3")
  [ "$f" -gt 0 ] || { cat "$3"; fail "$2: the second device's chain never ran"; }
  ! grep -qa 'the device is lost\|the device was lost' "$3" || { grep -a 'lost' "$3"; fail "$2: a device was lost"; }
  echo "   $2: the second device's chain ran $f forwards"
}
# ld2_forwards <engine> <log>: N from "[VK] <engine> dev2 chain: N forwards"
ld2_forwards() {
  local n; n=$(sed -n "s/^\[VK\] $1 dev2 chain: \([0-9]*\) forwards.*/\1/p" "$2" | tail -1)
  echo "${n:-0}"
}
# ld2_lost <engine> <tag> <n0> <n1> <frame> <env...> -- <argv...>: the second device lost at
# its given frame (COLI_VK_CHAIN_FAULT2): the CPU's tokens; at its first frame (the setup's
# parameter upload) LD2_EXPECT=setup: its layers stay on the CPU, nothing to rebuild; past
# it, the state rebuilt on the CPU (LD2_EXPECT=redo, an engine without recurrent state:
# the CPU redoes the step)
ld2_lost() {
  local eng=$1 tag=$2 n0=$3 n1=$4 k=$5; shift 5
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -f chain.usage
  env "${envs[@]}" ./"$eng" "$@" > cpu.log 2>&1 || true
  env "${envs[@]}" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=$n0 COLI_VK_CHAIN_LAYERS2=$n1 COLI_VK_CHAIN_FAULT2=$k \
    COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VK_TIER_BALANCE=0 COLI_VULKAN=1 COLI_VK_CHAIN=1 ./"$eng" "$@" > vk.log 2>&1 || true
  case $eng in
    colibri|glm53) mla_toks "$eng" cpu.log > cpu.tok; mla_toks "$eng" vk.log > vk.tok
             { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; tail -20 vk.log; fail "$tag: the tokens differ from the CPU"; } ;;
    *) same_tokens cpu.log vk.log "$tag" ;;
  esac
  grep -q "COLI_VK_CHAIN_FAULT2" vk.log || { cat vk.log; fail "$tag: the second device's fault never fired"; }
  if [ "${LD2_EXPECT:-}" = setup ]; then
    grep -q "^\[VK\] $eng dev2 chain: 0 of [0-9]* layers on the device: the chain stays off (layer 0 did not reach the device: the device was lost" vk.log ||
      { cat vk.log; fail "$tag: the second device's layers did not go to the CPU"; }
    echo "OK $tag: tokens = CPU, the second device's layers on the CPU from the start"
    return 0
  fi
  if [ "${LD2_EXPECT:-}" = again ]; then   # the KV cache is the host's: the CPU runs the forward again
    grep -q "^\[VK\] $eng dev2 chain: the device was lost; the CPU runs this forward again" vk.log ||
      { cat vk.log; fail "$tag: the CPU did not take the forward over"; }
    echo "OK $tag: tokens = CPU, the CPU ran the forward again"
    return 0
  fi
  if [ "${LD2_EXPECT:-}" = redo ]; then
    grep -q "the CPU redoes the step" vk.log || { cat vk.log; fail "$tag: the CPU did not redo the step"; }
    echo "OK $tag: tokens = CPU, the CPU redid the step"
    return 0
  fi
  grep -q "rebuilding the state of [1-9]" vk.log || { cat vk.log; fail "$tag: no state was rebuilt"; }
  echo "OK $tag: tokens = CPU, $(grep -o 'rebuilding the state of [0-9]* positions' vk.log | head -1)"
}

ld2_qwen36() {
  [ -d qwen36_tiny_c ] || { $PY tools/make_qwen36_tiny.py --out qwen36_tiny --ref-mode full --emit-ref qwen36_tiny/ref_full.json
                            $PY tools/convert_qwen36.py --model qwen36_tiny --out qwen36_tiny_c --ebits 8; }
  [ -d qwen36_kv_c ] || kv_qwen36_fixtures
  local R=qwen36_tiny/ref_full.json T="SNAP=qwen36_tiny_c COLI_DENSE_I8=0"
  ld2_gate qwen36 "ld2 qwen36 3 + 5, the head on the second device" 3 5 $T -- 8 8 $R
  ld2_gate qwen36 "ld2 qwen36 2 + 3, the CPU the rest and the head" 2 3 $T -- 8 8 $R
  ld2_gate qwen36 "ld2 qwen36 1 + 7, prompt in chunks of 3" 1 7 $T COLI_VK_CHAIN_ROWS=3 -- 8 8 $R
  ld2_gate qwen36 "ld2 qwen36 cap 1" 4 4 $T -- 1 8 $R
  CPUENV=COLI_DENSE_IDOT=0 ld2_gate qwen36 "ld2 qwen36 int8 trunk" 4 4 SNAP=qwen36_tiny_c -- 8 8 $R
  ld2_gate qwen36 "ld2 qwen36 experts on both devices too" 3 5 $T COLI_VK_TIER_GB=0.00002 -- 8 8 $R
  CHAINMODE=2 ld2_gate qwen36 "ld2 qwen36 prompts only" 3 5 $T -- 8 8 $R
  ld2_gate qwen36 "ld2 qwen36 dense model" 2 4 SNAP=qwen38_27b_kv_c COLI_DENSE_I8=0 -- 8 8 qwen38_27b_kv/ref_full.json
  # the KV cache split on both devices' attention layers
  ld2_gate qwen36 "ld2 qwen36 KV split on both" 4 4 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 SNAP=qwen36_kv_c COLI_DENSE_I8=0 \
    -- 8 8 qwen36_kv/ref_full.json
  [ "$(kv_hostparts qwen36 vk.log)" -gt 0 ] && [ "$(kv_hostparts "qwen36 dev2" vk.log)" -gt 0 ] ||
    { cat vk.log; fail "ld2 qwen36 KV split on both: a device's split ran no host part"; }
  # with COLI_VK_KV_COLD=device the primary attends its host part on the device; the
  # second device's stays on the CPU (the import is the primary's)
  ld2_gate qwen36 "ld2 qwen36 KV split on both, the host's part on the device" 4 4 COLI_VK_KV_COLD=device COLI_VK_KV_DEVICE_ROWS=16 \
    COLI_VK_KV_BLOCK=4 SNAP=qwen36_kv_c COLI_DENSE_I8=0 -- 8 8 qwen36_kv/ref_full.json
  [ "$(kv_devcold qwen36 vk.log)" -gt 0 ] || { cat vk.log; fail "ld2 qwen36 COLI_VK_KV_COLD=device: the primary's host part never ran on the device"; }
  { grep -q "^\[VK\] qwen36 dev2 chain: no shadow for layer" vk.log && [ "$(kv_hostparts "qwen36 dev2" vk.log)" -gt 0 ]; } ||
    { cat vk.log; fail "ld2 qwen36 COLI_VK_KV_COLD=device: the second device's host part did not stay on the CPU"; }
  # prompt-lookup drafts verified across both devices (rollbacks of both chains' copies)
  local f
  for f in accept mixed cycle row1 row3; do
    ld2_gate qwen36 "ld2 qwen36 lookup $f" 3 5 COLI_DENSE_I8=0 COLI_LOOKUP=1 COLI_SPEC_GATE=0 COLI_LOOKUP_FORCE=$f \
      SNAP=qwen36_kv_c -- 8 8 qwen36_kv/ref_full.json
  done
  # the second device lost: at its first frame, and mid-decode
  LD2_EXPECT=setup ld2_lost qwen36 "ld2 qwen36 second device lost at its setup" 3 5 1 $T -- 8 8 $R
  ld2_lost qwen36 "ld2 qwen36 second device lost mid-decode" 3 5 40 SNAP=qwen36_kv_c COLI_DENSE_I8=0 -- 8 8 qwen36_kv/ref_full.json
  # the prefix-reuse contract with the state on both devices, and serve sessions (pins,
  # prompt-cache extensions, a divergent prompt, logprobs)
  QWEN36_TINY=$PWD/qwen36_tiny_c COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_LAYERS2=5 \
    COLI_USAGE=$PWD/chain.usage $PY tests/test_qwen36_prefix_serve.py
  CHAIN_SERVE_EXPECT='qwen36 dev2 chain: [1-9][0-9]* forwards' \
    $PY tests/vulkan_chain_serve.py ./qwen36 qwen36_tiny_c COLI_DENSE_I8=0 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_LAYERS2=5
}

ld2_qwen38() {
  [ -d qwen38_tiny ] || $PY tools/make_qwen38_tiny.py --out qwen38_tiny
  [ -d qwen38_tiny_fp8 ] || $PY tools/make_qwen38_tiny.py --out qwen38_tiny_fp8 --fp8-experts
  [ -d qwen38_kv_mtp ] || kv_qwen38_fixtures
  local R=qwen38_tiny/ref.json T="OMP_NUM_THREADS=2 SNAP=qwen38_tiny" f b
  for b in 0 1; do
    ld2_gate qwen38 "ld2 qwen38 2 + 2, the head on the second device, batch=$b" 2 2 $T Q38_PREFILL_BATCH=$b -- 4 8 $R
  done
  ld2_gate qwen38 "ld2 qwen38 1 + 2, the CPU the rest and the head" 1 2 $T -- 4 8 $R
  ld2_gate qwen38 "ld2 qwen38 1 + 3, prompt in chunks of 3" 1 3 $T COLI_VK_CHAIN_ROWS=3 -- 4 8 $R
  ld2_gate qwen38 "ld2 qwen38 bf16 rows" 2 2 $T Q38_NATIVE_BF16=1 -- 4 8 $R
  ld2_gate qwen38 "ld2 qwen38 fp8 experts" 2 2 OMP_NUM_THREADS=2 SNAP=qwen38_tiny_fp8 -- 2 8 qwen38_tiny_fp8/ref.json
  ld2_gate qwen38 "ld2 qwen38 experts on both devices too" 2 2 $T COLI_VK_TIER_GB=0.00002 -- 4 8 $R
  CHAINMODE=2 ld2_gate qwen38 "ld2 qwen38 prompts only" 2 2 $T -- 4 8 $R
  # MTP drafts verified across both devices (both chains' copies rolled back)
  for f in "" accept reject mixed; do
    ld2_gate qwen38 "ld2 qwen38 MTP ${f:-drafting}" 2 2 OMP_NUM_THREADS=2 Q38_MTP=1 Q38_MTP_FORCE=$f SNAP=qwen38_tiny_mtp -- 2 8 qwen38_tiny_mtp/ref.json
  done
  # the KV split on both devices' QSA layers (layers 1 and 3), with MTP
  ld2_gate qwen38 "ld2 qwen38 KV split on both" 2 2 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 SNAP=qwen38_kv -- 4 8 qwen38_kv/ref.json
  [ "$(kv_hostparts qwen38 vk.log)" -gt 0 ] && [ "$(kv_hostparts "qwen38 dev2" vk.log)" -gt 0 ] ||
    { cat vk.log; fail "ld2 qwen38 KV split on both: a device's split ran no host part"; }
  ld2_gate qwen38 "ld2 qwen38 KV split on both, MTP mixed" 2 2 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 Q38_MTP=1 \
    Q38_MTP_FORCE=mixed SNAP=qwen38_kv_mtp -- 4 8 qwen38_kv_mtp/ref.json
  # the second device lost: at its setup, mid-decode, and in a verify
  LD2_EXPECT=setup ld2_lost qwen38 "ld2 qwen38 second device lost at its setup" 2 2 1 $T -- 4 8 $R
  ld2_lost qwen38 "ld2 qwen38 second device lost mid-decode" 2 2 20 $T -- 4 8 $R
  ld2_lost qwen38 "ld2 qwen38 second device lost in a verify" 2 2 15 OMP_NUM_THREADS=2 Q38_MTP=1 Q38_MTP_FORCE=mixed \
    SNAP=qwen38_tiny_mtp -- 2 8 qwen38_tiny_mtp/ref.json
  # serve sessions with MTP and without
  CHAIN_SERVE_EXPECT='qwen38 dev2 chain: [1-9][0-9]* forwards' \
    $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp Q38_MTP=1 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=2
  CHAIN_SERVE_EXPECT='qwen38 dev2 chain: [1-9][0-9]* forwards' \
    $PY tests/vulkan_chain_serve.py ./qwen38 qwen38_tiny_mtp COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=2
}

ld2_olmoe() {
  [ -d olmoe_tiny_c ] || { $PY tools/make_olmoe_tiny.py --output olmoe_tiny; $PY tools/convert_olmoe_merged.py --model olmoe_tiny --out olmoe_tiny_c; }
  [ -f olmoe_tiny/ref_olmoe_long.json ] || kv_olmoe_fixtures
  local OR=olmoe_tiny/ref_olmoe.json T=SNAP=olmoe_tiny_c cap
  ld2_gate olmoe "ld2 olmoe 2 + 2, the head on the second device" 2 2 $T -- 8 8 $OR
  ld2_gate olmoe "ld2 olmoe 1 + 2, the CPU the rest and the head" 1 2 $T -- 8 8 $OR
  ld2_gate olmoe "ld2 olmoe 1 + 3, prompt in chunks of 3" 1 3 $T COLI_VK_CHAIN_ROWS=3 -- 8 8 $OR
  for cap in 1 2; do ld2_gate olmoe "ld2 olmoe PILOT cap=$cap" 2 2 PILOT=1 WIDE=2 $T -- $cap 8 $OR; done
  ld2_gate olmoe "ld2 olmoe 4-bit experts" 2 2 $T -- 8 4 $OR
  ld2_gate olmoe "ld2 olmoe experts on both devices too" 2 2 $T COLI_VK_TIER_GB=0.000025 -- 8 8 $OR
  CHAINMODE=2 ld2_gate olmoe "ld2 olmoe prompts only" 2 2 $T -- 8 8 $OR
  ld2_gate olmoe "ld2 olmoe KV split on both" 2 2 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 $T -- 8 8 olmoe_tiny/ref_olmoe_long.json
  [ "$(kv_hostparts olmoe vk.log)" -gt 0 ] && [ "$(kv_hostparts "olmoe dev2" vk.log)" -gt 0 ] ||
    { cat vk.log; fail "ld2 olmoe KV split on both: a device's split ran no host part"; }
  LD2_EXPECT=setup ld2_lost olmoe "ld2 olmoe second device lost at its setup" 2 2 1 $T -- 8 8 $OR
  LD2_EXPECT=redo ld2_lost olmoe "ld2 olmoe second device lost mid-decode" 2 2 8 $T -- 8 8 $OR
  CHAIN_SERVE_EXPECT='olmoe dev2 chain: [1-9][0-9]* forwards' \
    $PY tests/vulkan_chain_serve.py ./olmoe olmoe_tiny_c COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=2
  CHAIN_SERVE_EXPECT='olmoe dev2 chain: [1-9][0-9]* forwards' \
    $PY tests/vulkan_chain_serve.py ./olmoe olmoe_tiny_c PILOT=1 WIDE=2 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=2
}

# mimo: its own gate (mimo_chain_gate: the vendor fixture's cases, tokens and logits), with
# the split forced; the second device's chain must have run in the logits run
ld2_mimo_gate() {   # <tag> <case> <n0> <n1> <env...>
  local tag=$1 c=$2 n0=$3 n1=$4; shift 4
  mimo_chain_gate "$tag" "$c" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=$n0 COLI_VK_CHAIN_LAYERS2=$n1 COLI_VK_TIER_BALANCE=0 "$@"
  ld2_ran mimo "$tag $c" mimo-vk.err
}
ld2_mimo() {
  $PY tools/make_mimo_tiny.py --output ./mimo_tiny --force --vision
  MGRID=$($PY -c "import json;i=json.load(open('mimo_tiny/ref.json'))['image'];print(i['grid_h'],i['grid_w'])")
  local c
  for c in $($PY -c "import json;print(' '.join(json.load(open('mimo_tiny/ref.json'))['cases']))"); do
    ld2_mimo_gate "ld2 mimo 3 + 3, the head on the second device" $c 3 3 MIMO_DENSE_BITS=32
    ld2_mimo_gate "ld2 mimo 1 + 2, the CPU the rest and the head" $c 1 2 MIMO_DENSE_BITS=32
  done
  ld2_mimo_gate "ld2 mimo 2 + 4, blocks of 3" image 2 4 MIMO_DENSE_BITS=32 MIMO_CHUNK=3
  ld2_mimo_gate "ld2 mimo 1 + 5, the release's FP8 and BF16" image 1 5 MIMO_DENSE_BITS=0
  ld2_mimo_gate "ld2 mimo experts on both devices too" image 3 3 MIMO_DENSE_BITS=32 MIMO_VK_EXPERTS=2 COLI_VK_TIER_GB=0.00002
  ld2_mimo_gate "ld2 mimo KV split on both" image 3 3 MIMO_DENSE_BITS=32 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2
  CHAINMODE=2 ld2_mimo_gate "ld2 mimo prompts only" image 3 3 MIMO_DENSE_BITS=32
  # the second device lost at its third frame: the host's caches hold whole steps, the CPU
  # runs the rest from where they end
  local x=(--image mimo_tiny/patches.f32 --grid $MGRID) ids
  ids=$(mimo_ids image prompt_ids)
  COLI_TEMP=0 MIMO_DENSE_BITS=32 ./mimo mimo_tiny --ids "$ids" --ngen 6 "${x[@]}" > mimo-cpu.txt 2>/dev/null
  COLI_TEMP=0 MIMO_DENSE_BITS=32 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_TIER_SYNC=1 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=3 \
    COLI_VK_CHAIN_LAYERS2=3 COLI_VK_CHAIN_FAULT2=3 ./mimo mimo_tiny --ids "$ids" --ngen 6 "${x[@]}" > mimo-vk.txt 2> mimo-vk.err
  cmp -s mimo-cpu.txt mimo-vk.txt || { cat mimo-cpu.txt mimo-vk.txt mimo-vk.err; fail "ld2 mimo second device lost: tokens differ from the CPU"; }
  grep -q "COLI_VK_CHAIN_FAULT2" mimo-vk.err && grep -q "the device was lost at position" mimo-vk.err ||
    { cat mimo-vk.err; fail "ld2 mimo second device lost: the loss was not taken over"; }
  echo "OK ld2 mimo second device lost: tokens = CPU, $(grep -o 'the device was lost at position [0-9]*' mimo-vk.err)"
}

ld2_inkling() {
  [ -f tiny_inkling/ref_long.json ] || kv_inkling_fixtures
  local R=tiny_inkling/ref_inkling.json T=SNAP=tiny_inkling b
  ld2_gate inkling "ld2 inkling 3 + 5, lm_head on the second device" 3 5 $T -- 8 0 $R
  ld2_gate inkling "ld2 inkling 2 + 2, the CPU the rest and lm_head" 2 2 $T -- 8 0 $R
  ld2_gate inkling "ld2 inkling 1 + 7, prompt in chunks of 3" 1 7 $T COLI_VK_CHAIN_ROWS=3 -- 8 0 $R
  for b in 4 8; do ld2_gate inkling "ld2 inkling runtime int$b" 3 5 $T -- 2 $b $R; done
  ld2_gate inkling "ld2 inkling experts on both devices too" 3 5 $T COLI_VK_TIER_GB=0.00002 -- 8 0 $R
  CHAINMODE=2 ld2_gate inkling "ld2 inkling prompts only" 3 5 $T -- 8 0 $R
  # the global layer's KV split (the fixture has one global layer, 5): on the second
  # device, then on the first
  ld2_gate inkling "ld2 inkling KV split on the second device" 4 4 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 $T -- 8 0 tiny_inkling/ref_long.json
  [ "$(kv_hostparts "inkling dev2" vk.log)" -gt 0 ] || { cat vk.log; fail "ld2 inkling KV split on the second device: no host part"; }
  ld2_gate inkling "ld2 inkling KV split on the first device" 6 2 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 $T -- 8 0 tiny_inkling/ref_long.json
  [ "$(kv_hostparts inkling vk.log)" -gt 0 ] || { cat vk.log; fail "ld2 inkling KV split on the first device: no host part"; }
  LD2_EXPECT=setup ld2_lost inkling "ld2 inkling second device lost at its setup" 3 5 1 $T -- 8 0 $R
  ld2_lost inkling "ld2 inkling second device lost mid-decode" 3 5 30 $T -- 8 0 $R
  CHAIN_SERVE_EXPECT='inkling dev2 chain: [1-9][0-9]* forwards' \
    $PY tests/vulkan_chain_serve.py ./inkling tiny_inkling INK_PREFIX_LOG=1 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_LAYERS2=5
}

# colibri (GLM-5.2): the head stays on the host. glm_tiny_shx shares indexers (full,
# shared, full, shared, shared): with DSA top-4 the selection of the layer before the
# second device's first one crosses to it.
ld2_colibri() {
  [ -f glm_tiny_serve/tokenizer.json ] || glm_chain_fixtures
  [ -d glm_tiny_shx ] || ptl_glm_shx glm_tiny glm_tiny_shx
  export CAP_RAISE=0
  local G="SNAP=glm_tiny REF=ref_glm.json USAGE_SAVE=0" X="SNAP=glm_tiny_shx REF=ref_glm.json USAGE_SAVE=0" s
  ld2_mla colibri "ld2 colibri 2 + 3, every layer on the devices" 2 3 $G -- 64 16 16
  ld2_mla colibri "ld2 colibri 1 + 2, the CPU the rest" 1 2 $G -- 64 16 16
  ld2_mla colibri "ld2 colibri cap 1" 3 2 $G -- 1 16 16
  ld2_mla colibri "ld2 colibri prefill in chunks of 3" 2 3 $G TF=1 COLI_VK_CHAIN_ROWS=3 -- 64 16 16
  ld2_mla colibri "ld2 colibri 4-bit trunk and experts" 2 3 $G IDOT=0 -- 2 4 4
  ld2_mla colibri "ld2 colibri i4 container" 2 3 SNAP=glm_tiny_i4 REF=glm_tiny_i4/ref_glm.json USAGE_SAVE=0 IDOT=0 -- 1 4 4
  ld2_mla colibri "ld2 colibri DSA top-4" 2 3 $G DSA_TOPK=4 -- 64 16 16
  ld2_mla colibri "ld2 colibri DSA_FORCE" 2 3 $G DSA_FORCE=1 -- 64 16 16
  for s in "1 2" "3 2"; do
    ld2_mla colibri "ld2 colibri DSA top-4, a shared indexer first on the second device, ${s/ / + }" ${s% *} ${s#* } $X DSA_TOPK=4 -- 64 16 16
    ld2_mla colibri "ld2 colibri DSA top-4, a shared indexer first on the second device, ${s/ / + }, prefill in chunks of 5" \
      ${s% *} ${s#* } $X DSA_TOPK=4 TF=1 COLI_VK_CHAIN_ROWS=5 -- 64 16 16
  done
  ld2_mla colibri "ld2 colibri n-gram drafts" 2 3 $G DRAFT=3 -- 64 16 16
  ld2_mla colibri "ld2 colibri MTP depth 2" 2 3 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json USAGE_SAVE=0 DRAFT=2 -- 64 16 16
  ld2_mla colibri "ld2 colibri MTP with DSA top-4" 1 2 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json USAGE_SAVE=0 DSA_TOPK=4 -- 64 16 16
  ld2_mla colibri "ld2 colibri experts on both devices too" 2 3 $G COLI_VK_TIER_GB=0.00002 -- 64 16 16
  CHAINMODE=2 ld2_mla colibri "ld2 colibri prompts only, drafts" 2 3 $G DRAFT=3 -- 64 16 16
  ld2_mla colibri "ld2 colibri KV split on both" 2 3 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4 $G DSA_TOPK=4 -- 64 16 16
  [ "$(kv_hostparts colibri vk.log)" -gt 0 ] && [ "$(kv_hostparts "colibri dev2" vk.log)" -gt 0 ] ||
    { cat vk.log; fail "ld2 colibri KV split on both: a device's split ran no host part"; }
  # the second device lost: at its setup, in the prompt (one forward), mid-decode
  LD2_EXPECT=setup ld2_lost colibri "ld2 colibri second device lost at its setup" 2 2 1 $G -- 64 16 16
  LD2_EXPECT=again ld2_lost colibri "ld2 colibri second device lost in the prompt" 2 2 2 $G TF=1 -- 64 16 16
  LD2_EXPECT=again ld2_lost colibri "ld2 colibri second device lost mid-decode" 2 2 10 $G -- 64 16 16
  LD2_EXPECT=again ld2_lost colibri "ld2 colibri second device lost with MTP" 2 2 10 SNAP=glm_tiny_mtp REF=glm_tiny_mtp/ref_glm.json \
    USAGE_SAVE=0 DRAFT=2 -- 64 16 16
  # serve sessions: pins, the prompt cache, the prefill read-out, two KV slots
  CHAIN_SERVE_EXPECT='colibri dev2 chain: [1-9][0-9]* forwards' CHAIN_SERVE_DIALECT=colibri \
    $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=3
  CHAIN_SERVE_EXPECT='colibri dev2 chain: [1-9][0-9]* forwards' CHAIN_SERVE_DIALECT=colibri \
    $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 DSA_TOPK=4 DRAFT=3 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=2
  CHAIN_SERVE_EXPECT='colibri dev2 chain: [1-9][0-9]* forwards' CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=colibri \
    $PY tests/vulkan_chain_serve.py ./colibri glm_tiny_serve SERVE_BATCH=1 IDOT=0 KV_SLOTS=2 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=2
}

# glm53 (GLM-5.3 Flash): the six-layer fixture, KDA and MLA alternating (the KDA state on
# both devices); the head stays on the host
ld2_glm53() {
  [ -d glm53_mm_tiny ] || glm_chain_fixtures
  [ -d glm53_l6s-i4 ] || { ptl_g53_l6 glm53_l6; $PY tools/make_glm53_streaming_pair.py --fixture glm53_l6 --output glm53_l6s > /dev/null; }
  [ -f glm53_l6_serve/tokenizer.json ] || { rm -rf glm53_l6_serve && cp -r glm53_l6s-i4 glm53_l6_serve
                                            $PY tools/make_edge_tiny_tokenizer.py --vocab-size 128 ./glm53_l6_serve > /dev/null; }
  local ids F="--model glm53_l6s-i4" G="GLM53_BITS=32 USAGE_SAVE=0" s
  local M="--model glm53_mm_tiny --ids 103,117,268,268,268,268,120,121 --patches glm53_mm_tiny/patches.f32 --grid 4x4"
  ids=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
  for s in "2 2" "3 3" "1 5" "4 1"; do
    ld2_mla glm53 "ld2 glm53 ${s/ / + }" ${s% *} ${s#* } $G -- $F --ids $ids --greedy 6
  done
  ld2_mla glm53 "ld2 glm53 decode" 3 3 $G -- $F --ids 5,7,9,11,13,17,19,23 --greedy 8
  ld2_mla glm53 "ld2 glm53 4-bit trunk" 2 3 GLM53_BITS=4 USAGE_SAVE=0 -- $F --ids $ids --greedy 4
  ld2_mla glm53 "ld2 glm53 chain chunks of 3" 2 2 $G COLI_VK_CHAIN_ROWS=3 -- $F --ids $ids --greedy 4
  ld2_mla glm53 "ld2 glm53 prefill chunks of 7, one cache slot" 3 2 GLM53_BITS=4 USAGE_SAVE=0 GLM53_PREFILL_CHUNK=7 \
    GLM53_EXPERT_GB=0.000001 -- $F --ids $ids --greedy 4
  ld2_mla glm53 "ld2 glm53 experts on both devices too" 3 3 $G COLI_VK_TIER_GB=0.0002 -- $F --ids $ids --greedy 6
  ld2_mla glm53 "ld2 glm53 an image" 2 1 $G -- $M --greedy 4
  CHAINMODE=2 ld2_mla glm53 "ld2 glm53 prompts only" 3 3 $G GLM53_PREFILL_CHUNK=16 -- $F --ids $ids --greedy 6
  ld2_mla glm53 "ld2 glm53 KV split on both" 2 2 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 $G -- $F --ids $ids --greedy 8
  [ "$(kv_hostparts glm53 vk.log)" -gt 0 ] && [ "$(kv_hostparts "glm53 dev2" vk.log)" -gt 0 ] ||
    { cat vk.log; fail "ld2 glm53 KV split on both: a device's split ran no host part"; }
  # the second device lost: at its setup, mid-prompt and mid-decode (both devices' KDA
  # state rebuilt from the primary's record), after an image
  LD2_EXPECT=setup ld2_lost glm53 "ld2 glm53 second device lost at its setup" 3 3 1 $G -- $F --ids $ids --greedy 6
  ld2_lost glm53 "ld2 glm53 second device lost in a prompt chunk" 3 3 12 $G GLM53_PREFILL_CHUNK=16 -- $F --ids $ids --greedy 6
  ld2_lost glm53 "ld2 glm53 second device lost mid-decode" 3 3 40 $G -- $F --ids $ids --greedy 6
  ld2_lost glm53 "ld2 glm53 second device lost after an image" 2 1 6 $G -- $M --greedy 4
  # a pin's branch over the image fixture, its KDA state on both devices (layer 2 on the second)
  rm -f chain.usage
  COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_GEMM_MIN_S=0 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=1 \
    COLI_USAGE=$PWD/chain.usage COLI_VK_TIER=0 \
    $PY tests/glm53_pin_branch_harness.py --binary ./glm53 --fixture glm53_mm_tiny --tol 1e-5
  rm -f chain.usage
  # serve sessions: pins, the prompt cache, two KV slots
  CHAIN_SERVE_EXPECT='glm53 dev2 chain: [1-9][0-9]* forwards' CHAIN_SERVE_DIALECT=numeric \
    $PY tests/vulkan_chain_serve.py ./glm53 glm53_l6_serve GLM53_BITS=32 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_LAYERS2=3
  CHAIN_SERVE_EXPECT='glm53 dev2 chain: [1-9][0-9]* forwards' CHAIN_SERVE_DIALECT=numeric \
    $PY tests/vulkan_chain_serve.py ./glm53 glm53_l6_serve GLM53_BITS=4 COLI_VK_CHAIN_ROWS=3 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=2
  CHAIN_SERVE_EXPECT='glm53 dev2 chain: [1-9][0-9]* forwards' CHAIN_SERVE_SLOTS=2 CHAIN_SERVE_DIALECT=numeric \
    $PY tests/vulkan_chain_serve.py ./glm53 glm53_l6_serve GLM53_BITS=32 KV_SLOTS=2 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_LAYERS2=3
}

# kimi_k3: k3c_gate (the CPU's tokens, every logits row within 2e-3 of the largest) for
# each of the oracle's prompts, with the split forced; the second device's chain ran
ld2_k3_gate() {   # <tag> <n0> <n1> <env...>
  local tag=$1 n0=$2 n1=$3 c; shift 3
  for c in ${K3C_CASES:-short chunk long}; do
    K3C_CASES=$c k3c_gate "$tag" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=$n0 COLI_VK_CHAIN_LAYERS2=$n1 COLI_VK_TIER_BALANCE=0 "$@"
    ld2_ran kimi_k3 "$tag $c" vk.log
  done
}
# ld2_k3_lost <tag> <n0> <n1> <back> <env...>: the long prompt, the second device lost
# <back> of its frames before the end of the same run without a fault (back = 0: at its
# first frame, its setup); the CPU's tokens, and LD2_EXPECT=setup (its layers on the CPU
# from the start), rebuild (both devices' KDA state rebuilt) or current (the host's state
# current: nothing to rebuild)
ld2_k3_lost() {
  local tag=$1 n0=$2 n1=$3 back=$4 frames k ids; shift 4
  local D2=(COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=$n0 COLI_VK_CHAIN_LAYERS2=$n1 COLI_VK_TIER_BALANCE=0 COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0)
  ids=$(k3c_ids long)
  rm -f k3c.usage
  env "$@" COLI_USAGE=$PWD/k3c.usage USAGE_SAVE=0 ./kimi_k3 kimi_k3_tiny --ids "$ids" --ngen 8 2> cpu.log | sed 's/ *TUNE.*//' > cpu.tok
  k=1
  if [ "$back" -gt 0 ]; then
    env "$@" "${D2[@]}" COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 ./kimi_k3 kimi_k3_tiny --ids "$ids" --ngen 8 2> vk.log > /dev/null
    frames=$(sed -n 's/^\[VK\] kimi_k3 dev2 chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p' vk.log | tail -1)
    [ -n "$frames" ] && [ "$frames" -gt "$back" ] || { cat vk.log; fail "$tag: no fault-free run to count frames from"; }
    k=$((frames - back + 1))
  fi
  env "$@" "${D2[@]}" COLI_VK_CHAIN_FAULT2=$k COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 \
    ./kimi_k3 kimi_k3_tiny --ids "$ids" --ngen 8 2> vk.log | sed 's/ *TUNE.*//' > vk.tok
  { [ -s cpu.tok ] && cmp -s cpu.tok vk.tok; } || { cat cpu.tok vk.tok; tail -20 vk.log; fail "$tag: the tokens differ from the CPU"; }
  grep -q "COLI_VK_CHAIN_FAULT2" vk.log || { cat vk.log; fail "$tag: the second device's fault never fired"; }
  case ${LD2_EXPECT:-rebuild} in
    setup)   grep -q "^\[VK\] kimi_k3 dev2 chain: 0 of [0-9]* layers on the device: the chain stays off (layer 0 did not reach the device: the device was lost" vk.log ||
               { cat vk.log; fail "$tag: the second device's layers did not go to the CPU"; } ;;
    rebuild) grep -q "kimi_k3 dev2 chain: the device was lost; rebuilding the state of [1-9]" vk.log || { cat vk.log; fail "$tag: no state was rebuilt"; } ;;
    current) grep -q "kimi_k3 dev2 chain: the device was lost; the host's state is current" vk.log || { cat vk.log; fail "$tag: the loss was not taken over"; } ;;
  esac
  echo "OK $tag: tokens = CPU (fault at the second device's frame $k), $(grep -o 'rebuilding the state of [0-9]* positions\|the host.s state is current\|the chain stays off' vk.log | head -1)"
}
ld2_kimi_k3() {
  $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  k3c_serve_fixture
  local O="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0" s b
  for s in "2 4" "1 2" "3 1" "2 2"; do ld2_k3_gate "ld2 kimi_k3 f32 ${s/ / + }" ${s% *} ${s#* } $O; done
  for b in 8 4; do
    ld2_k3_gate "ld2 kimi_k3 ${b}-bit trunk" 2 4 K3_BITS=$b K3_MLA_BITS=$b K3_HEAD_BITS=$b K3_IDOT=0 COLI_TEMP=0
  done
  ld2_k3_gate "ld2 kimi_k3 prefill one token at a time" 1 3 $O K3_CHUNK=1
  ld2_k3_gate "ld2 kimi_k3 chain chunks of 3" 2 4 $O COLI_VK_CHAIN_ROWS=3
  ld2_k3_gate "ld2 kimi_k3 tier off" 2 4 $O COLI_VK_TIER=0
  ld2_k3_gate "ld2 kimi_k3 shared experts by COLI_VK_DENSE=1" 2 4 $O COLI_VK_DENSE=1
  ld2_k3_gate "ld2 kimi_k3 K3_TOPP=0.6" 2 2 $O K3_TOPP=0.6
  CHAINMODE=2 ld2_k3_gate "ld2 kimi_k3 prompts only" 2 4 $O
  K3C_CASES=long ld2_k3_gate "ld2 kimi_k3 KV split on both" 3 3 $O COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4
  [ "$(kv_hostparts kimi_k3 vk.log)" -gt 0 ] && [ "$(kv_hostparts "kimi_k3 dev2" vk.log)" -gt 0 ] ||
    { cat vk.log; fail "ld2 kimi_k3 KV split on both: a device's split ran no host part"; }
  # the second device lost (a setup frame, then 9 frames a forward; the long prompt in
  # forwards of 32, 32 and 8 rows, then 7 decode steps): at its setup, in the last decode
  # step, in the second prompt forward, in the third chain chunk of the second prompt
  # forward in chunks of 5 (the state its first chunks advanced not used); both devices'
  # KDA state rebuilt from the token ids
  LD2_EXPECT=setup ld2_k3_lost "ld2 kimi_k3 second device lost at its setup" 2 4 0 $O K3_CHUNK=32
  ld2_k3_lost "ld2 kimi_k3 second device lost mid-decode" 2 4 3 $O K3_CHUNK=32
  ld2_k3_lost "ld2 kimi_k3 second device lost in a later prompt chunk" 2 4 77 $O K3_CHUNK=32
  ld2_k3_lost "ld2 kimi_k3 second device lost inside a chunked forward" 2 4 122 $O K3_CHUNK=32 COLI_VK_CHAIN_ROWS=5
  # serve sessions frame for frame, the KDA state across turns on both devices: photos
  # taken from both and restored to both
  local SV="K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 K3_PREFIX_LOG=1 USAGE_SAVE=0 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=4"
  export CHAIN_SERVE_TOL=2e-2 CHAIN_SERVE_EXPECT='kimi_k3 dev2 chain: [1-9][0-9]* forwards'
  # shellcheck disable=SC2086
  {
    $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $SV
    $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $SV COLI_K3_CKPT=4
    $PY tests/vulkan_chain_serve.py ./kimi_k3 kimi_k3_serve $SV COLI_K3_CKPT=2 COLI_VK_CHAIN_ROWS=3
  }
  unset CHAIN_SERVE_TOL CHAIN_SERVE_EXPECT
}

# deepseek_v41: v41_gate (the CPU's tokens and exit, every forward's logits within 1e-4 of
# the largest) with the split forced at a layer the second device may start at (the
# fixture's 1, 3 and 5: its first compressed layer owns its caches and runs its indexer);
# the second device's chain ran
ld2_v41_gate() {   # <tag> <n0> <n1> <env...> -- <argv...>
  local tag=$1 n0=$2 n1=$3; shift 3
  v41_gate "$tag" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=$n0 COLI_VK_CHAIN_LAYERS2=$n1 COLI_VK_TIER_BALANCE=0 "$@"
  ld2_ran deepseek_v41 "$tag" vk.log
}
# ld2_v41_lost <tag> <n0> <n1> <back> <env...> -- <argv...>: the second device lost <back>
# of its frames before the end of the same run without a fault (0: at its first frame,
# its setup); the CPU's tokens, and its layers on the CPU from the start (back 0) or from
# the forward it was lost in
ld2_v41_lost() {
  local tag=$1 n0=$2 n1=$3 back=$4 frames k=1; shift 4
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local D2=(COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=$n0 COLI_VK_CHAIN_LAYERS2=$n1 COLI_VK_TIER_BALANCE=0 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1)
  rm -f chain.usage
  env "${envs[@]}" ./deepseek_v41 "$@" > cpu.txt 2> cpu.log || true
  if [ "$back" -gt 0 ]; then
    env "${envs[@]}" "${D2[@]}" ./deepseek_v41 "$@" > /dev/null 2> vk.log || true
    frames=$(sed -n 's/^\[VK\] deepseek_v41 dev2 chain: [0-9]* forwards, \([0-9]*\) frames.*/\1/p' vk.log | tail -1)
    [ -n "$frames" ] && [ "$frames" -gt "$back" ] || { cat vk.log; fail "$tag: no fault-free run to count frames from"; }
    k=$((frames - back + 1))
  fi
  rm -f chain.usage
  env "${envs[@]}" "${D2[@]}" COLI_VK_CHAIN_FAULT2=$k ./deepseek_v41 "$@" > vk.txt 2> vk.log || true
  { [ -s cpu.txt ] && cmp -s cpu.txt vk.txt; } || { cat cpu.txt vk.txt; tail -20 vk.log; fail "$tag: the tokens differ from the CPU"; }
  grep -q "COLI_VK_CHAIN_FAULT2" vk.log || { cat vk.log; fail "$tag: the second device's fault never fired"; }
  if [ "$back" = 0 ]; then
    grep -q "^\[VK\] deepseek_v41 dev2 chain: 0 of [0-9]* layers on the device: the chain stays off (layer 0 did not reach the device: the device was lost" vk.log ||
      { cat vk.log; fail "$tag: the second device's layers did not go to the CPU"; }
  else
    grep -q "deepseek_v41 dev2 chain: the device was lost; the CPU runs its layers" vk.log || { cat vk.log; fail "$tag: the loss was not taken over"; }
  fi
  rm -f chain.usage
  echo "OK $tag: tokens = CPU (fault at the second device's frame $k)"
}
ld2_deepseek_v41() {
  v41_chain_fixtures
  local s f cap T=(SNAP=dsv41_long) A=(-- 8 dsv41_long/ref.json)
  for s in "1 2" "1 5" "3 3" "5 1"; do ld2_v41_gate "ld2 deepseek_v41 ${s/ / + }" ${s% *} ${s#* } "${T[@]}" "${A[@]}"; done
  for cap in 1 2; do ld2_v41_gate "ld2 deepseek_v41 cap=$cap" 1 4 SNAP=dsv41_tiny -- $cap dsv41_tiny/ref.json; done
  # a layer the second device may not start at: forced there, it stays off; the fit's
  # choice comes down to the last layer it may start at (a device for 2 layers: 1)
  v41_gate "ld2 deepseek_v41 forced at layer 2" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=2 "${T[@]}" "${A[@]}"
  grep -q "layer 2 reads what the layers before it make in a forward" vk.log && [ "$(ld2_forwards deepseek_v41 vk.log)" = 0 ] ||
    { cat vk.log; fail "ld2 deepseek_v41 forced at layer 2: the second device did not stay off"; }
  ptl_v41_probe ptl-probe.log COLI_VK_DEVICE_CAP_MB=256 COLI_VK_TIER_RESERVE_GB=0.04 "${T[@]}" "${A[@]}"
  cap=$(PTL_PROBE_CAP_MB=256 ptl_calc cap deepseek_v41 ptl-probe.log 2) || fail "ld2 deepseek_v41 the fit: no cap from the probe"
  v41_gate "ld2 deepseek_v41 the fit at layer 2 (cap $cap MiB)" COLI_VK_DEV2=0 COLI_VK_DEVICE_CAP_MB=$cap COLI_VK_TIER_RESERVE_GB=0.04 \
    COLI_VK_TIER_BALANCE=0 "${T[@]}" "${A[@]}"
  grep -q "deepseek_v41 chain: 1 of 6 layers on the device: the second device's layers start at layer 1" vk.log ||
    { grep -a '^\[VK\]' vk.log; fail "ld2 deepseek_v41 the fit at layer 2: it did not come down to layer 1"; }
  ld2_ran deepseek_v41 "ld2 deepseek_v41 the fit at layer 2" vk.log
  # DSpark drafts accepted (1, 3) and rejected (2, 4, 5), the targets on the second device
  for f in 1 2 3 4 5; do ld2_v41_gate "ld2 deepseek_v41 DSpark spec=$f" 1 5 SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=$f -- 8 dsv41_tiny/ref.json; done
  ld2_v41_gate "ld2 deepseek_v41 DSpark spec=5, one cache slot" 3 3 SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=5 -- 1 dsv41_tiny/ref.json
  ld2_v41_gate "ld2 deepseek_v41 prompt in chunks of 3" 1 5 "${T[@]}" COLI_VK_CHAIN_ROWS=3 "${A[@]}"
  ld2_v41_gate "ld2 deepseek_v41 prompt in chunks of 7" 3 3 "${T[@]}" COLI_VK_CHAIN_ROWS=7 -- 2 dsv41_long/ref.json
  CHAINMODE=2 ld2_v41_gate "ld2 deepseek_v41 prompts only" 1 5 "${T[@]}" "${A[@]}"
  ld2_v41_gate "ld2 deepseek_v41 tier off" 3 3 "${T[@]}" COLI_VK_TIER=0 "${A[@]}"
  ld2_v41_gate "ld2 deepseek_v41 beside the per-matrix trunk" 3 3 "${T[@]}" COLI_VK_DENSE=1 "${A[@]}"
  ld2_v41_gate "ld2 deepseek_v41 V41_INDEX_OWNER=1" 1 5 "${T[@]}" V41_INDEX_OWNER=1 "${A[@]}"
  ld2_v41_gate "ld2 deepseek_v41 experts on both devices too" 3 3 "${T[@]}" COLI_VK_TIER_GB=0.00002 "${A[@]}"
  # the compressed rows split on both devices' kv_source layers (1 on the first, 3 on the second)
  ld2_v41_gate "ld2 deepseek_v41 KV split on both" 3 3 COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2 "${T[@]}" "${A[@]}"
  [ "$(kv_hostparts deepseek_v41 vk.log)" -gt 0 ] && [ "$(kv_hostparts "deepseek_v41 dev2" vk.log)" -gt 0 ] ||
    { cat vk.log; fail "ld2 deepseek_v41 KV split on both: a device's split ran no host part"; }
  # the second device lost: at its setup, mid-decode, in the prompt, between drafts
  ld2_v41_lost "ld2 deepseek_v41 second device lost at its setup" 1 5 0 "${T[@]}" "${A[@]}"
  ld2_v41_lost "ld2 deepseek_v41 second device lost mid-decode" 1 5 5 "${T[@]}" "${A[@]}"
  ld2_v41_lost "ld2 deepseek_v41 second device lost in the prompt" 3 3 60 "${T[@]}" COLI_VK_CHAIN_ROWS=7 "${A[@]}"
  ld2_v41_lost "ld2 deepseek_v41 second device lost between drafts" 1 5 8 SNAP=dsv41_tiny V41_DSPARK=1 V41_SPEC_FORCE=5 -- 8 dsv41_tiny/ref.json
  # serve sessions frame for frame, DSpark, prompts only; images on the wire
  export CHAIN_SERVE_EXPECT='deepseek_v41 dev2 chain: [1-9][0-9]* forwards'
  $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=0 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=5
  $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=1 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_LAYERS2=3
  COLI_VK_CHAIN=2 $PY tests/vulkan_chain_serve.py ./deepseek_v41 dsv41_tiny V41_DSPARK=1 COLI_VK_CHAIN_ROWS=3 COLI_VK_DEV2=0 \
    COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=5
  unset CHAIN_SERVE_EXPECT
  $PY tests/vulkan_chain_v41_image.py ./deepseek_v41 dsv41_tiny V41_DSPARK=0 COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=5
  rm -f chain.usage chain-serve.usage chain-image.usage ptl-probe.log
}

# deepseek_v4: v4_chain_gate (the CPU's and the reference's ids, the teacher-forced rows,
# the drafts, every logits row within V4_CHAIN_TOL) with the split forced; the second
# device's chain ran (FAULT2_BACK=k: lost k of its frames before the end of a fault-free
# run, 0 at its setup: the CPU runs its layers)
ld2_v4_gate() {   # <tag> <n0> <n1> <fixture> <case> <env...>
  local tag=$1 n0=$2 n1=$3 fx=$4 c=$5; shift 5
  v4_chain_gate "$tag" "$fx" "$c" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=$n0 COLI_VK_CHAIN_LAYERS2=$n1 COLI_VK_TIER_BALANCE=0 "$@"
  [ -n "${FAULT2_BACK:-}" ] || ld2_ran deepseek_v4 "$tag" v4-vk.err
}
ld2_deepseek_v4() {
  v4_chain_fixtures
  local c fx A=ids:20,21,22,23,24,25,26,27,28,29,20,21,22:12 R=ids:20,21,22,23,50,51,52,20,21,22:12
  local M=ids:30,31,32,33,34,35,60,61,30,31,32,33,34,35,36,37,38,39,40,41,30,31,32:12
  for c in short compressed long; do
    ld2_v4_gate "ld2 deepseek_v4 1 + 2" 1 2 deepseek_v4_tiny_t $c
    ld2_v4_gate "ld2 deepseek_v4 2 + 1" 2 1 deepseek_v4_tiny_t $c
  done
  ld2_v4_gate "ld2 deepseek_v4 8 experts" 1 2 deepseek_v4_tiny_e8 long
  ld2_v4_gate "ld2 deepseek_v4 2 output groups" 1 2 deepseek_v4_tiny_g2 long
  ld2_v4_gate "ld2 deepseek_v4 a window of 4, ratios 4 and 2" 1 2 deepseek_v4_tiny_w4 long
  ld2_v4_gate "ld2 deepseek_v4 an indexer of 64 heads of 128" 1 2 deepseek_v4_tiny_ix long
  ld2_v4_gate "ld2 deepseek_v4 experts on both devices too" 1 2 deepseek_v4_tiny_e8 long COLI_VK_TIER_GB=0.00009
  ld2_v4_gate "ld2 deepseek_v4 tier off" 1 2 deepseek_v4_tiny_t long COLI_VK_TIER=0 COLI_VK_DENSE=0
  ld2_v4_gate "ld2 deepseek_v4 beside the per-matrix trunk" 1 2 deepseek_v4_tiny_t long COLI_VK_DENSE=1
  ld2_v4_gate "ld2 deepseek_v4 prefill chunks of 7" 2 1 deepseek_v4_tiny_t long V4_PREFILL_CHUNK=7
  ld2_v4_gate "ld2 deepseek_v4 chain chunks of 3" 1 2 deepseek_v4_tiny_e8 long COLI_VK_CHAIN_ROWS=3
  ld2_v4_gate "ld2 deepseek_v4 V4_IDX_IDENTITY=1" 1 2 deepseek_v4_tiny_t long V4_IDX_IDENTITY=1
  CHAINMODE=2 ld2_v4_gate "ld2 deepseek_v4 prompts only" 1 2 deepseek_v4_tiny_t long
  ld2_v4_gate "ld2 deepseek_v4 n-gram drafts accepted and rejected" 1 2 deepseek_v4_tiny_t $A V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  ld2_v4_gate "ld2 deepseek_v4 an n-gram draft rejected" 2 1 deepseek_v4_tiny_t $R V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  ld2_v4_gate "ld2 deepseek_v4 n-gram drafts, a window of 4" 1 2 deepseek_v4_tiny_w4 $M V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  ld2_v4_gate "ld2 deepseek_v4 V4_MTP=1" 1 2 deepseek_v4_tiny_t long V4_MTP=1 V4_DRAFT=3
  # the compressed rows split on both devices (layer 1 ratio 4 on the first, layer 2 ratio 8 on the second)
  ld2_v4_gate "ld2 deepseek_v4 KV split on both" 2 1 deepseek_v4_tiny_t long COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2
  [ "$(kv_hostparts deepseek_v4 v4-vk.err)" -gt 0 ] && [ "$(kv_hostparts "deepseek_v4 dev2" v4-vk.err)" -gt 0 ] ||
    { cat v4-vk.err; fail "ld2 deepseek_v4 KV split on both: a device's split ran no host part"; }
  # the second device lost: at its setup, mid-decode, in the prompt, between drafts
  FAULT2_BACK=0 ld2_v4_gate "ld2 deepseek_v4 second device lost at its setup" 1 2 deepseek_v4_tiny_t long
  FAULT2_BACK=3 ld2_v4_gate "ld2 deepseek_v4 second device lost mid-decode" 1 2 deepseek_v4_tiny_t long
  FAULT2_BACK=30 ld2_v4_gate "ld2 deepseek_v4 second device lost in the prompt" 1 2 deepseek_v4_tiny_t long COLI_VK_CHAIN_ROWS=7
  FAULT2_BACK=10 ld2_v4_gate "ld2 deepseek_v4 second device lost between drafts" 1 2 deepseek_v4_tiny_w4 $M V4_DRAFT=4 V4_NGRAM_PARTIAL_KEEP=1
  # served: the pin, its extension, a read-only prompt and a shorter one
  COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=2 V4_SERVE_EXPECT='deepseek_v4 dev2 chain: [1-9][0-9]* forwards' \
    v4_chain_serve deepseek_v4_tiny_t
  rm -rf deepseek_v4_tiny_t deepseek_v4_tiny_e8 deepseek_v4_tiny_g2 deepseek_v4_tiny_w4 deepseek_v4_tiny_ix chain.usage
  rm -f cpu.f32 vk.f32 v4-*.f32 v4-*.json v4-*.err v4-tf.txt
}

family_layers_dev2() {
  export OMP_NUM_THREADS=2
  make qwen36 qwen38 olmoe mimo inkling tests/test_vk_chain VK=1
  COLI_VK_DEV2=0 ./tests/test_vk_chain shaders/qmatmul.spv | tee vk_chain.log
  tail -1 vk_chain.log | grep -qx PASS || fail "the chain's ops on two devices"
  ld2_qwen36
  ld2_qwen38
  ld2_olmoe
  ld2_mimo
  ld2_inkling
  unset OMP_NUM_THREADS
}
family_layers_dev2_mla() {
  export OMP_NUM_THREADS=2
  make colibri glm53 kimi_k3 VK=1
  ld2_colibri
  ld2_glm53
  ld2_kimi_k3
  unset OMP_NUM_THREADS CAP_RAISE
}
family_layers_dev2_deepseek() {
  export OMP_NUM_THREADS=2
  make deepseek_v41 VK=1
  make deepseek-v4 VK=1
  ld2_deepseek_v41
  ld2_deepseek_v4
  unset OMP_NUM_THREADS
}
family_layers_dev2_sanitize() {
  make clean >/dev/null 2>&1 || true
  make qwen36 qwen38 olmoe mimo inkling VK=1 EXTRA_CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  [ -d qwen36_kv_c ] || kv_qwen36_fixtures
  kv_san qwen36 "asan ld2 qwen36 KV split on both" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=4 COLI_VK_CHAIN_LAYERS2=4 \
    COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 COLI_DENSE_I8=0 SNAP=qwen36_kv_c ./qwen36 8 8 qwen36_kv/ref_full.json
  [ "$(ld2_forwards qwen36 san.log)" -gt 0 ] || { cat san.log; fail "asan ld2 qwen36: the second device's chain never ran"; }
  [ -d qwen38_kv_mtp ] || kv_qwen38_fixtures
  kv_san qwen38 "asan ld2 qwen38 KV split on both, MTP mixed" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=2 \
    COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 Q38_MTP=1 Q38_MTP_FORCE=mixed SNAP=qwen38_kv_mtp ./qwen38 4 8 qwen38_kv_mtp/ref.json
  [ "$(ld2_forwards qwen38 san.log)" -gt 0 ] || { cat san.log; fail "asan ld2 qwen38: the second device's chain never ran"; }
  [ -f olmoe_tiny/ref_olmoe_long.json ] || kv_olmoe_fixtures
  kv_san olmoe "asan ld2 olmoe KV split on both, PILOT" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=2 \
    COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 PILOT=1 WIDE=2 SNAP=olmoe_tiny_c ./olmoe 2 8 olmoe_tiny/ref_olmoe_long.json
  [ "$(ld2_forwards olmoe san.log)" -gt 0 ] || { cat san.log; fail "asan ld2 olmoe: the second device's chain never ran"; }
  [ -f tiny_inkling/ref_long.json ] || kv_inkling_fixtures
  # the fixture's one global layer (5) on the first device, where kv_san counts the host parts
  kv_san inkling "asan ld2 inkling KV split on the first device, layers 6 and 7 on the second" COLI_VK_DEV2=0 COLI_VK_CHAIN_LAYERS=6 COLI_VK_CHAIN_LAYERS2=2 \
    COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 SNAP=tiny_inkling ./inkling 8 0 tiny_inkling/ref_long.json
  [ "$(ld2_forwards inkling san.log)" -gt 0 ] || { cat san.log; fail "asan ld2 inkling: the second device's chain never ran"; }
  make clean >/dev/null 2>&1 || true
}
# ld2_san <engine> <tag> <env and argv...>: a sanitized build's run with the split: no
# sanitizer diagnostic, and the second device's chain ran (LD2_SAN_LOST=1: its loss was
# taken over)
ld2_san() {
  local eng=$1 tag=$2; shift 2
  rm -f chain.usage
  env OMP_NUM_THREADS=2 COLI_USAGE=chain.usage COLI_VK_TIER_SYNC=1 COLI_VULKAN=1 COLI_VK_CHAIN=1 COLI_VK_DEV2=0 COLI_VK_TIER_BALANCE=0 \
    "$@" > san.log 2>&1 || true
  rm -f chain.usage
  if grep -qE "ERROR: AddressSanitizer|runtime error:" san.log; then cat san.log; fail "$tag: sanitizer diagnostic"; fi
  if [ "${LD2_SAN_LOST:-0}" = 1 ]; then
    grep -q "COLI_VK_CHAIN_FAULT2" san.log && grep -q "the device was lost" san.log || { cat san.log; fail "$tag: the loss was not taken over"; }
    echo "OK $tag: sanitizers clean, the second device's loss taken over"
  else
    [ "$(ld2_forwards "$eng" san.log)" -gt 0 ] || { cat san.log; fail "$tag: the second device's chain never ran"; }
    echo "OK $tag: sanitizers clean, the second device's chain ran $(ld2_forwards "$eng" san.log) forwards"
  fi
}
family_layers_dev2_mla_sanitize() {
  local SAN="-fsanitize=address,undefined -fno-omit-frame-pointer -g" ids
  make clean >/dev/null 2>&1 || true
  make colibri glm53 kimi_k3 VK=1 EXTRA_CFLAGS="$SAN"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1 CAP_RAISE=0
  [ -f glm_tiny_serve/tokenizer.json ] || glm_chain_fixtures
  [ -d glm_tiny_shx ] || ptl_glm_shx glm_tiny glm_tiny_shx
  [ -d glm53_l6s-i4 ] || { ptl_g53_l6 glm53_l6; $PY tools/make_glm53_streaming_pair.py --fixture glm53_l6 --output glm53_l6s > /dev/null; }
  $PY tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
  ids=$($PY -c "print(','.join(str((i*37+5)%120+2) for i in range(100)))")
  local G=(SNAP=glm_tiny REF=ref_glm.json USAGE_SAVE=0) K=(COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=4)
  ld2_san colibri "asan ld2 colibri KV split on both, DSA top-4" COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=3 "${K[@]}" "${G[@]}" DSA_TOPK=4 \
    ./colibri 64 16 16
  ld2_san colibri "asan ld2 colibri a shared indexer first on the second device, chunks of 5" COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=2 \
    SNAP=glm_tiny_shx REF=ref_glm.json USAGE_SAVE=0 DSA_TOPK=4 TF=1 COLI_VK_CHAIN_ROWS=5 ./colibri 64 16 16
  LD2_SAN_LOST=1 ld2_san colibri "asan ld2 colibri second device lost mid-decode" COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=2 \
    COLI_VK_CHAIN_FAULT2=10 "${G[@]}" ./colibri 64 16 16
  ld2_san glm53 "asan ld2 glm53 KV split on both" COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=2 COLI_VK_KV_DEVICE_ROWS=16 COLI_VK_KV_BLOCK=4 \
    GLM53_BITS=32 USAGE_SAVE=0 ./glm53 --model glm53_l6s-i4 --ids $ids --greedy 8
  LD2_SAN_LOST=1 ld2_san glm53 "asan ld2 glm53 second device lost, the KDA state rebuilt" COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_LAYERS2=3 \
    COLI_VK_CHAIN_FAULT2=40 GLM53_BITS=32 USAGE_SAVE=0 ./glm53 --model glm53_l6s-i4 --ids $ids --greedy 6
  local O=(K3_BITS=32 K3_MLA_BITS=32 K3_HEAD_BITS=32 K3_IDOT=0 COLI_TEMP=0 USAGE_SAVE=0)
  ld2_san kimi_k3 "asan ld2 kimi_k3 KV split on both" COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_LAYERS2=3 COLI_VK_KV_DEVICE_ROWS=16 \
    COLI_VK_KV_BLOCK=4 "${O[@]}" ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids long)" --ngen 8
  LD2_SAN_LOST=1 ld2_san kimi_k3 "asan ld2 kimi_k3 second device lost in a prompt forward" COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=4 \
    COLI_VK_CHAIN_FAULT2=15 K3_CHUNK=32 "${O[@]}" ./kimi_k3 kimi_k3_tiny --ids "$(k3c_ids long)" --ngen 8
  unset CAP_RAISE
  make clean >/dev/null 2>&1 || true
}
family_layers_dev2_deepseek_sanitize() {
  local SAN="-fsanitize=address,undefined -fno-omit-frame-pointer -g" p
  make clean >/dev/null 2>&1 || true
  make deepseek_v41 VK=1 EXTRA_CFLAGS="$SAN"
  export ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=0 UBSAN_OPTIONS=print_stacktrace=1
  v41_chain_fixtures
  ld2_san deepseek_v41 "asan ld2 deepseek_v41 DSpark spec=5" COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=5 SNAP=dsv41_tiny V41_DSPARK=1 \
    V41_SPEC_FORCE=5 ./deepseek_v41 8 dsv41_tiny/ref.json
  ld2_san deepseek_v41 "asan ld2 deepseek_v41 KV split on both, chunks of 3" COLI_VK_CHAIN_LAYERS=3 COLI_VK_CHAIN_LAYERS2=3 \
    COLI_VK_KV_DEVICE_ROWS=8 COLI_VK_KV_BLOCK=2 COLI_VK_CHAIN_ROWS=3 SNAP=dsv41_long ./deepseek_v41 8 dsv41_long/ref.json
  LD2_SAN_LOST=1 ld2_san deepseek_v41 "asan ld2 deepseek_v41 second device lost" COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=5 \
    COLI_VK_CHAIN_FAULT2=40 SNAP=dsv41_long ./deepseek_v41 8 dsv41_long/ref.json
  make clean >/dev/null 2>&1 || true
  # the V4 sanitized build as its chain family makes it (no LTO, the sanitizers at the link)
  make deepseek-v4-clean >/dev/null 2>&1 || true
  make deepseek-v4 VK=1 LTO=0 EXTRA_CFLAGS="$SAN" EXTRA_LDFLAGS="$SAN"
  v4_chain_fixtures
  p=$(v4_chain_prompt deepseek_v4_tiny_t long)
  ld2_san deepseek_v4 "asan ld2 deepseek_v4 KV split on both" COLI_VK_CHAIN_LAYERS=2 COLI_VK_CHAIN_LAYERS2=1 COLI_VK_KV_DEVICE_ROWS=8 \
    COLI_VK_KV_BLOCK=2 ./deepseek_v4 ./deepseek_v4_tiny_t "$p" --raw-prompt --max-tokens 4
  ld2_san deepseek_v4 "asan ld2 deepseek_v4 drafts, a window of 4" COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=2 V4_DRAFT=4 \
    V4_NGRAM_PARTIAL_KEEP=1 ./deepseek_v4 ./deepseek_v4_tiny_w4 \
    "$(v4_chain_prompt x ids:30,31,32,33,34,35,60,61,30,31,32,33,34,35,36,37,38,39,40,41,30,31,32)" --raw-prompt --max-tokens 12
  LD2_SAN_LOST=1 ld2_san deepseek_v4 "asan ld2 deepseek_v4 second device lost" COLI_VK_CHAIN_LAYERS=1 COLI_VK_CHAIN_LAYERS2=2 \
    COLI_VK_CHAIN_FAULT2=8 ./deepseek_v4 ./deepseek_v4_tiny_t "$p" --raw-prompt --max-tokens 4
  rm -rf deepseek_v4_tiny_t deepseek_v4_tiny_e8 deepseek_v4_tiny_g2 deepseek_v4_tiny_w4 deepseek_v4_tiny_ix san.log
  make deepseek-v4-clean >/dev/null 2>&1 || true
}
